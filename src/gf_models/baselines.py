"""Participant-local baseline policies; no transport or experiment configuration."""
from __future__ import annotations

import math
import random

from gf_common.actions import content_score
from gf_common.domain import (
    CandidateSet,
    ContractError,
    DecisionInput,
    Json,
    Policy,
    PolicyDecision,
    validated_policy_options,
)


class RandomPolicy:
    def __init__(self) -> None:
        self.rng = random.Random()

    def reset(self, sampling_seed: int) -> None:
        self.rng.seed(sampling_seed)

    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        return PolicyDecision(self.rng.randrange(len(candidates.actions)), "random-v1", -math.log(len(candidates.actions)))


class HeuristicPolicy(RandomPolicy):
    def select(self, decision_input: DecisionInput, candidates: CandidateSet) -> PolicyDecision:
        scores = [content_score(decision_input, a) for a in candidates.actions]
        best = max(scores)
        ties = [i for i, score in enumerate(scores) if score == best]
        return PolicyDecision(self.rng.choice(ties), "heuristic-v1", -math.log(len(ties)), best)


def make_policy(model_id: str, options: Json | None = None) -> Policy:
    if model_id == "mlp":
        from gf_models.mlp.policy import MLPPolicy
        options = validated_policy_options(options)
        return MLPPolicy(options["checkpoint"], options["expected_manifest_digest"])
    if model_id == "lightgbm":
        from gf_models.lightgbm.policy import LightGBMPolicy
        options = validated_policy_options(options)
        return LightGBMPolicy(options["checkpoint"], options["expected_manifest_digest"], allow_toy=options.get("allow_toy", False))
    if options is not None:
        raise ContractError("baseline cannot accept policy options")
    if model_id == "random":
        return RandomPolicy()
    if model_id == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"unsupported model: {model_id}")
