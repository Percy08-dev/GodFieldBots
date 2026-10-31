"""CPU categorical Transformer policy with fixed-capacity private KV arrays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gf_common.domain import CandidateSet, ContractError, DecisionInput, PolicyDecision
from gf_models.config import transformer_model_config
from gf_models.features import encode_decision
from gf_models.sequence import MAX_HISTORY, load_artifact
from gf_models.transformer.model import cache_storage_bytes, model_storage_bytes


class TransformerPolicy:
    def __init__(self, checkpoint: str, expected_manifest_digest: str) -> None:
        import torch
        self.torch = torch
        config = transformer_model_config(json.loads((Path(checkpoint) / 'inference.json').read_text())['model'])
        self.cache_bytes = cache_storage_bytes()
        self.reserved_bytes = model_storage_bytes(config) + self.cache_bytes
        if self.reserved_bytes > config.inference.cache_budget_bytes:
            raise ContractError('Transformer participant cache/model budget exceeded')
        self.network, self.manifest = load_artifact(Path(checkpoint), expected_digest=expected_manifest_digest)
        self.config = transformer_model_config(self.manifest['model'])
        self.identity = expected_manifest_digest
        self.rng: Any = torch.Generator(device='cpu')
        self.cache = self.network.create_cache()

    def reset(self, sampling_seed: int) -> None:
        self.rng.manual_seed(sampling_seed)
        self.cache.reset()

    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        if self.cache.position >= MAX_HISTORY:
            raise ContractError('Transformer own-decision history resource limit exceeded')
        torch = self.torch
        actor, state = encode_decision(decision_input, candidates.actions, self.config.encoder)
        with torch.no_grad():
            token = torch.from_numpy(state)[None]
            context = self.network.step(token, self.cache)
            logs, value = self.network.heads(context, token, torch.from_numpy(actor)[None],
                                             torch.ones((1, len(actor)), dtype=torch.bool))
            index = int(torch.multinomial(logs.exp()[0], 1, generator=self.rng).item())
        return PolicyDecision(index, 'transformer-ppo-v1-' + self.identity, float(logs[0, index]), float(value[0]))
