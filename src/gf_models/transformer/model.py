"""Two-layer pre-norm Transformer using the standard deterministic SDPA math path."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from gf_common.domain import ContractError
from gf_models.config import TransformerModelConfig
from gf_models.encoder import feature_names
from gf_models.features import critic_indices
from gf_models.sequence import MAX_HISTORY
from gf_models.torch_runtime import torch_runtime


@dataclass
class KVCache:
    keys: Any
    values: Any
    position: int = 0

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.position = 0


def cache_storage_bytes() -> int:
    return 2 * 2 * MAX_HISTORY * 128 * 4


def model_storage_bytes(model: TransformerModelConfig) -> int:
    state, actor = len(critic_indices(model.encoder)), len(feature_names(model.encoder))
    layer = 4 * (128 * 128 + 128) + 4 * 128 + 256 * 128 + 256 + 128 * 256 + 128
    floats = (128 * state + 128 + 2 * layer + 128 * (128 + actor) + 257
              + 128 * (128 + state) + 257 + MAX_HISTORY * 128)
    return floats * 4


def _positions() -> Any:
    # Calculate once on CPU in FP32; the same bytes move with the model.
    positions = np.arange(MAX_HISTORY, dtype=np.float32)[:, None]
    frequencies = np.exp(np.arange(0, 128, 2, dtype=np.float32) * np.float32(-math.log(10000.) / 128))
    angles = positions * frequencies
    result = np.empty((MAX_HISTORY, 128), dtype=np.float32)
    result[:, 0::2], result[:, 1::2] = np.sin(angles), np.cos(angles)
    return result


def make_network(model: TransformerModelConfig, seed: int) -> Any:
    torch = torch_runtime('cpu')
    original_rng = torch.get_rng_state()
    torch.random.default_generator.manual_seed(seed)
    nn = torch.nn
    width = len(critic_indices(model.encoder))
    base: Any = nn.Module

    def linear(layer: Any, gain: float) -> None:
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)

    class Layer(base):
        def __init__(self) -> None:
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
            self.query, self.key, self.value, self.output = (nn.Linear(128, 128) for _ in range(4))
            self.feedforward1, self.feedforward2 = nn.Linear(128, 256), nn.Linear(256, 128)
            for layer in (self.query, self.key, self.value, self.output):
                linear(layer, 1.)
            linear(self.feedforward1, math.sqrt(2))
            linear(self.feedforward2, 1.)
            for layer in (self.norm1, self.norm2):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        def projections(self, x: Any) -> tuple[Any, Any, Any]:
            normalized = self.norm1(x)
            return tuple(layer(normalized).reshape(x.shape[0], x.shape[1], 4, 32).transpose(1, 2)
                         for layer in (self.query, self.key, self.value))

        def attend(self, query: Any, key: Any, value: Any, allowed: Any = None) -> Any:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=allowed,
                                                                    dropout_p=0., is_causal=False)
            return output.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[2], 128)

        def finish(self, x: Any, attention: Any) -> Any:
            x = x + self.output(attention)
            return x + self.feedforward2(self.feedforward1(self.norm2(x)).relu())

        def forward(self, x: Any, past: Any, offset: int, valid: Any) -> tuple[Any, tuple[Any, Any]]:
            query, key, value = self.projections(x)
            if past is not None:
                key, value = torch.cat((past[0], key), 2), torch.cat((past[1], value), 2)
            # Explicit absolute offset causal and key padding masks. All queries
            # only see prefix/current keys; future and padded keys are excluded.
            query_positions = offset + torch.arange(x.shape[1], device=x.device)
            key_positions = torch.arange(key.shape[2], device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            allowed = allowed[None, None] & valid[:, None, None, :]
            output = self.finish(x, self.attend(query, key, value, allowed))
            output = output.masked_fill(~valid[:, offset:, None], 0)
            return output, (key, value)

        def step(self, x: Any, keys: Any, values: Any, position: int) -> Any:
            query, key, value = self.projections(x)
            keys[:, :, position:position + 1].copy_(key)
            values[:, :, position:position + 1].copy_(value)
            # Query length is one, and slices contain exactly past/current keys.
            # is_causal=False inside attend avoids the wrong offset-zero mask.
            return self.finish(x, self.attend(query, keys[:, :, :position + 1], values[:, :, :position + 1]))

    class ActorCritic(base):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(width, 128)
            self.layers = nn.ModuleList([Layer(), Layer()])
            self.actor = nn.Sequential(nn.Linear(128 + len(feature_names(model.encoder)), 128), nn.ReLU(), nn.Linear(128, 1))
            self.critic = nn.Sequential(nn.Linear(128 + width, 128), nn.ReLU(), nn.Linear(128, 1))
            self.register_buffer('positions', torch.from_numpy(_positions()), persistent=False)
            self.cache_storage_bytes = cache_storage_bytes()
            for layer, gain in ((self.projection, math.sqrt(2)), (self.actor[0], math.sqrt(2)),
                                (self.actor[2], .01), (self.critic[0], math.sqrt(2)), (self.critic[2], 1.)):
                linear(layer, gain)

        def encode(self, tokens: Any, past: Any = None, offset: int = 0, valid: Any = None) -> tuple[Any, Any]:
            length = tokens.shape[1]
            if not 0 < length <= MAX_HISTORY or offset < 0 or offset + length > MAX_HISTORY:
                raise ContractError('Transformer own-decision history resource limit exceeded')
            if valid is None:
                valid = torch.ones((tokens.shape[0], offset + length), device=tokens.device, dtype=torch.bool)
            x = self.projection(tokens).relu() + self.positions[offset:offset + length][None]
            caches = []
            for index, layer in enumerate(self.layers):
                x, cache = layer(x, None if past is None else past[index], offset, valid)
                caches.append(cache)
            return x, tuple(caches)

        def heads(self, context: Any, state: Any, actor: Any, mask: Any) -> tuple[Any, Any]:
            expanded = context[:, None].expand(-1, actor.shape[1], -1)
            logits = self.actor(torch.cat((expanded, actor), -1)).squeeze(-1).masked_fill(~mask, -torch.inf)
            return logits.log_softmax(-1), self.critic(torch.cat((context, state), -1)).squeeze(-1)

        def forward(self, tokens: Any, actor: Any, mask: Any, past: Any = None,
                    offset: int = 0, valid: Any = None) -> tuple[Any, Any]:
            output, _ = self.encode(tokens, past, offset, valid)
            if valid is None:
                return self.heads(output[:, -1], tokens[:, -1], actor, mask)
            positions = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1)
            endpoint = positions.masked_fill(~valid[:, offset:], -1).max(-1).values
            if torch.any(endpoint < 0):
                raise ContractError('Transformer endpoint must contain a visible decision token')
            rows = torch.arange(tokens.shape[0], device=tokens.device)
            return self.heads(output[rows, endpoint], tokens[rows, endpoint], actor, mask)

        def split(self, tokens: Any, actor: Any, mask: Any, prefix_length: int) -> tuple[Any, Any]:
            past = None
            if prefix_length:
                with torch.no_grad():
                    _, past = self.encode(tokens[:, :prefix_length])
                past = tuple((key.detach(), value.detach()) for key, value in past)
            return self(tokens[:, prefix_length:], actor, mask, past, prefix_length)

        def create_cache(self) -> KVCache:
            shape = (2, 1, 4, MAX_HISTORY, 32)
            return KVCache(torch.zeros(shape, dtype=torch.float32, device=self.projection.weight.device),
                           torch.zeros(shape, dtype=torch.float32, device=self.projection.weight.device))

        def step(self, state: Any, cache: KVCache) -> Any:
            if cache.position >= MAX_HISTORY:
                raise ContractError('Transformer own-decision history resource limit exceeded')
            with torch.no_grad():
                x = self.projection(state[:, None]).relu() + self.positions[cache.position:cache.position + 1][None]
                for index, layer in enumerate(self.layers):
                    x = layer.step(x, cache.keys[index], cache.values[index], cache.position)
                cache.position += 1
                return x[:, 0]

    try:
        return ActorCritic()
    finally:
        torch.set_rng_state(original_rng)
