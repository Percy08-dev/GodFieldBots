"""CPU categorical inference from the participant's allowed decision input."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gf_common.domain import CandidateSet, DecisionInput, PolicyDecision
from gf_models.config import mlp_model_config
from gf_models.features import encode_decision
from gf_models.mlp.model import load_artifact


class MLPPolicy:
    def __init__(self, checkpoint: str, expected_manifest_digest: str) -> None:
        import torch
        self.torch = torch
        self.network, self.manifest = load_artifact(Path(checkpoint), expected_digest=expected_manifest_digest)
        self.config = mlp_model_config(self.manifest['model'])
        self.identity = expected_manifest_digest
        self.rng: Any = torch.Generator(device='cpu')

    def reset(self, sampling_seed: int) -> None:
        self.rng.manual_seed(sampling_seed)

    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        torch = self.torch
        actor, state = encode_decision(decision_input, candidates.actions, self.config.encoder)
        with torch.no_grad():
            logprobs, value = self.network(torch.from_numpy(actor).unsqueeze(0), torch.from_numpy(state).unsqueeze(0),
                                          torch.ones((1, len(actor)), dtype=torch.bool))
            index = int(torch.multinomial(logprobs.exp()[0], 1, generator=self.rng).item())
        return PolicyDecision(index, 'mlp-ppo-v1-' + self.identity, float(logprobs[0, index]), float(value[0]))
