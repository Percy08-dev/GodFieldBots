"""Participant-local hidden cache and categorical CPU inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gf_common.domain import CandidateSet, ContractError, DecisionInput, PolicyDecision
from gf_models.config import gru_model_config
from gf_models.features import encode_decision
from gf_models.gru.model import MAX_HISTORY, load_artifact, model_storage_bytes


class GRUPolicy:
    def __init__(self, checkpoint: str, expected_manifest_digest: str) -> None:
        import torch
        self.torch = torch
        # Read only the closed small recipe to admit all model/cache arrays
        # before the artifact loader constructs tensors or unpacks weight arrays.
        config = gru_model_config(json.loads((Path(checkpoint) / 'inference.json').read_text())['model'])
        self.cache_bytes = 128 * 4
        self.reserved_bytes = model_storage_bytes(config) + self.cache_bytes
        if self.reserved_bytes > config.inference.cache_budget_bytes:
            raise ContractError('GRU participant cache/model budget exceeded')
        self.network, self.manifest = load_artifact(Path(checkpoint), expected_digest=expected_manifest_digest)
        self.config = gru_model_config(self.manifest['model'])
        self.identity = expected_manifest_digest
        self.rng: Any = torch.Generator(device='cpu')
        # Fixed storage, no per-decision history concatenation or hidden growth.
        self.cache = self.network.create_cache()
        self.hidden = self.cache.hidden
        self.position = 0

    def reset(self, sampling_seed: int) -> None:
        self.rng.manual_seed(sampling_seed)
        self.cache.reset()
        self.position = 0

    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        if self.position >= MAX_HISTORY:
            raise ContractError('GRU own-decision history resource limit exceeded')
        torch = self.torch
        actor, state = encode_decision(decision_input, candidates.actions, self.config.encoder)
        with torch.no_grad():
            token = torch.from_numpy(state)[None]
            context = self.network.step(token, self.cache)
            logs, value = self.network.heads(context, token, torch.from_numpy(actor)[None],
                                             torch.ones((1, len(actor)), dtype=torch.bool))
            index = int(torch.multinomial(logs.exp()[0], 1, generator=self.rng).item())
            self.position = self.cache.position
        return PolicyDecision(index, 'gru-ppo-v1-' + self.identity, float(logs[0, index]), float(value[0]))
