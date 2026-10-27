"""Deterministic candidate-content inference with exact artifact identity."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gf_common.domain import CandidateSet, ContractError, DecisionInput, PolicyDecision
from gf_models.lightgbm.config import encoder_config
from gf_models.lightgbm.encoder import Array, encode
from gf_models.lightgbm.model import load_artifact


class LightGBMPolicy:
    def __init__(self, checkpoint: str, expected_manifest_digest: str, *, allow_toy: bool = False) -> None:
        self.booster, self.manifest = load_artifact(Path(checkpoint), expected_digest=expected_manifest_digest, allow_toy=allow_toy)
        self.encoder = encoder_config(self.manifest["encoder"])
        self.version = "lightgbm-q-v1:" + expected_manifest_digest

    def reset(self, sampling_seed: int) -> None:
        pass  # Deterministic argmax; no artificial behavior logprob/state value.

    def scores(self, value: DecisionInput, candidates: CandidateSet) -> Array:
        if candidates.config_hash != self.manifest["action_config_hash"] or candidates.generator_version != self.manifest["generator_version"] or candidates.scorer_version != self.manifest["scorer_version"]:
            raise ContractError("candidate/checkpoint semantics mismatch")
        if not candidates.actions:
            raise ContractError("cannot score empty candidates")
        features = np.stack([encode(value, action, self.encoder) for action in candidates.actions])
        scores = np.asarray(self.booster.predict(features, num_threads=1), dtype=np.float64)
        if scores.shape != (len(candidates.actions),) or not np.isfinite(scores).all():
            raise ContractError("native scores must be finite and exact-length")
        return scores

    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        return PolicyDecision(int(np.argmax(self.scores(decision_input, candidates))), self.version)


def manifest_digest(checkpoint: str) -> str:
    return hashlib.sha256((Path(checkpoint) / "inference.json").read_bytes()).hexdigest()
