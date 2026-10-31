"""One-layer GRU with shared cached/full-prefix/TBPTT equations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gf_common.domain import ContractError
from gf_models.config import GRUModelConfig
from gf_models.encoder import feature_names
from gf_models.features import critic_indices
from gf_models.sequence import (
    MAX_HISTORY,
    SEQUENCE_RECIPE,
    inspect_artifact,
    load_artifact,
    save_artifact,
)
from gf_models.torch_runtime import torch_runtime

__all__ = [
    'BACKEND',
    'MAX_HISTORY',
    'SEQUENCE_RECIPE',
    'HiddenCache',
    'inspect_artifact',
    'load_artifact',
    'make_network',
    'model_storage_bytes',
    'save_artifact',
]

BACKEND = {'gru': 'torch-nn-gru', 'cudnn_enabled': False}


@dataclass
class HiddenCache:
    hidden: Any
    position: int = 0

    def reset(self) -> None:
        self.hidden.zero_()
        self.position = 0


def model_storage_bytes(model: GRUModelConfig) -> int:
    """Closed parameter shapes, available before Torch or cache allocation."""
    state = len(critic_indices(model.encoder))
    actor = len(feature_names(model.encoder))
    floats = (128 * state + 128 + 2 * (3 * 128 * 128 + 3 * 128)
              + 128 * (128 + actor) + 128 + 128 + 1
              + 128 * (128 + state) + 128 + 128 + 1)
    return floats * 4


def make_network(model: GRUModelConfig, seed: int) -> Any:
    torch = torch_runtime('cpu')
    torch.backends.cudnn.enabled = False
    original_rng = torch.get_rng_state()
    torch.random.default_generator.manual_seed(seed)
    nn = torch.nn
    width = len(critic_indices(model.encoder))
    base: Any = nn.Module

    class ActorCritic(base):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(width, 128)
            self.recurrent = nn.GRU(128, 128, num_layers=1, batch_first=True, dropout=0, bidirectional=False)
            self.actor = nn.Sequential(nn.Linear(128 + len(feature_names(model.encoder)), 128), nn.ReLU(), nn.Linear(128, 1))
            self.cache_storage_bytes = 128 * 4
            self.critic = nn.Sequential(nn.Linear(128 + width, 128), nn.ReLU(), nn.Linear(128, 1))
            for layer, gain in ((self.projection, math.sqrt(2)), (self.actor[0], math.sqrt(2)),
                                (self.actor[2], .01), (self.critic[0], math.sqrt(2)), (self.critic[2], 1.)):
                nn.init.orthogonal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)
            for name, parameter in self.recurrent.named_parameters():
                if name.startswith('weight'):
                    for gate in parameter.chunk(3, 0):
                        nn.init.orthogonal_(gate, 1.)
                else:
                    nn.init.zeros_(parameter)

        def encode(self, tokens: Any, hidden: Any = None) -> tuple[Any, Any]:
            return self.recurrent(self.projection(tokens).relu(), hidden)

        def create_cache(self) -> HiddenCache:
            return HiddenCache(torch.zeros((1, 1, 128), dtype=torch.float32, device=self.projection.weight.device))

        def step(self, state: Any, cache: HiddenCache) -> Any:
            if cache.position >= MAX_HISTORY:
                raise ContractError('GRU own-decision history resource limit exceeded')
            with torch.no_grad():
                output, hidden = self.encode(state[:, None], cache.hidden)
                cache.hidden.copy_(hidden)
                cache.position += 1
                return output[:, 0]

        def heads(self, context: Any, state: Any, actor: Any, mask: Any) -> tuple[Any, Any]:
            expanded = context[:, None, :].expand(-1, actor.shape[1], -1)
            logits = self.actor(torch.cat((expanded, actor), -1)).squeeze(-1).masked_fill(~mask, -torch.inf)
            value = self.critic(torch.cat((context, state), -1)).squeeze(-1)
            return logits.log_softmax(-1), value

        def forward(self, tokens: Any, actor: Any, mask: Any, hidden: Any = None) -> tuple[Any, Any]:
            output, _ = self.encode(tokens, hidden)
            return self.heads(output[:, -1], tokens[:, -1], actor, mask)

        def split(self, tokens: Any, actor: Any, mask: Any, prefix_length: int) -> tuple[Any, Any]:
            hidden = None
            if prefix_length:
                with torch.no_grad():
                    _, hidden = self.encode(tokens[:, :prefix_length])
                hidden = hidden.detach()
            return self(tokens[:, prefix_length:], actor, mask, hidden)

    try:
        return ActorCritic()
    finally:
        torch.set_rng_state(original_rng)
