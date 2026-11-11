"""Strict pure HPO scoring of independently verified, closed input projections.

These records and the resulting objective confer no Source, evaluation or formal
qualification authority. The concrete evaluator must verify the actual originals,
creation manifest, final checkpoint and cleanup before supplying these inputs.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from gf_common.domain import ContractError, canonical, digest, integer
from gf_common.validation import result as validate_result


def _text(value: Any) -> None:
    if type(value) is not str or not value:
        raise ContractError("nonempty evaluation identity required")


def _sha(value: Any) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("evaluation SHA256 required")


@dataclass(frozen=True)
class GameIdentity:
    """Expected identity from an independently verified post-creation manifest."""

    game_seed: str
    learner_seat: int
    episode_id: str
    game_id: str


@dataclass(frozen=True)
class RunIdentity:
    training_seed: int
    run_id: str
    checkpoint_sha256: str
    source_receipt_sha256: str
    games: tuple[GameIdentity, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    training_seeds: tuple[int, ...]
    game_seeds: tuple[str, ...]
    decision_budget: int
    conditions_sha256: str
    runs: tuple[RunIdentity, ...]


@dataclass(frozen=True)
class GameProjection:
    """Own projection envelope; result_json freezes the actual result DTO only."""

    identity: GameIdentity
    result_json: str
    decisions: int = 0
    end_kind: str = "natural"
    transport_ok: bool = True
    save_ok: bool = True
    cleanup_ok: bool = True
    elapsed_seconds: float = 0.

    @classmethod
    def from_result(cls, identity: GameIdentity, *, result: Any, decisions: int = 0) -> GameProjection:
        validate_result(result)
        if result is None:
            raise ContractError("natural evaluation requires an actual terminal result")
        return cls(identity, canonical(result), decisions)


@dataclass(frozen=True)
class SeedEvaluationProjection:
    run: RunIdentity
    used_decisions: int
    conditions_sha256: str
    games: tuple[GameProjection, ...]
    completed_normally: bool = True
    source_ok: bool = True
    save_ok: bool = True
    cleanup_ok: bool = True


@dataclass(frozen=True)
class EvaluationObjective:
    value: float
    seed_scores: tuple[tuple[int, float], ...]
    projection_sha256: str
    label: str = "exploratory-hpo-validation"


def _tuple(value: Any) -> None:
    if type(value) is not tuple or not value:
        raise ContractError("nonempty immutable evaluation tuple required")


def _identity(identity: GameIdentity) -> None:
    if type(identity) is not GameIdentity:
        raise ContractError("typed game identity required")
    for value in (identity.game_seed, identity.episode_id, identity.game_id):
        _text(value)
    integer(identity.learner_seat, 1, 2)


def _plan(plan: EvaluationPlan) -> dict[int, RunIdentity]:
    if type(plan) is not EvaluationPlan:
        raise ContractError("typed evaluation plan required")
    integer(plan.decision_budget, 1)
    _sha(plan.conditions_sha256)
    _tuple(plan.training_seeds)
    _tuple(plan.game_seeds)
    _tuple(plan.runs)
    for seed in plan.training_seeds:
        integer(seed, 0, 2**63 - 1)
    for game_seed in plan.game_seeds:
        _text(game_seed)
    if len(set(plan.training_seeds)) != len(plan.training_seeds) or len(set(plan.game_seeds)) != len(plan.game_seeds):
        raise ContractError("duplicate evaluation seed")
    runs: dict[int, RunIdentity] = {}
    ids: set[str] = set()
    episodes: set[str] = set()
    run_ids: set[str] = set()
    panel = {(seed, seat) for seed in plan.game_seeds for seat in (1, 2)}
    for run in plan.runs:
        if type(run) is not RunIdentity:
            raise ContractError("typed run identity required")
        integer(run.training_seed, 0, 2**63 - 1)
        _text(run.run_id)
        _sha(run.checkpoint_sha256)
        _sha(run.source_receipt_sha256)
        _tuple(run.games)
        if run.training_seed in runs or run.run_id in run_ids:
            raise ContractError("duplicate evaluation run")
        pairs = set()
        for game in run.games:
            _identity(game)
            key = (game.game_seed, game.learner_seat)
            if key in pairs or game.game_id in ids or game.episode_id in episodes:
                raise ContractError("duplicate game/episode/seat identity")
            pairs.add(key)
            ids.add(game.game_id)
            episodes.add(game.episode_id)
        if pairs != panel:
            raise ContractError("expected games do not match common two-seat panel")
        runs[run.training_seed] = run
        run_ids.add(run.run_id)
    if set(runs) != set(plan.training_seeds):
        raise ContractError("expected runs do not match complete training-seed group")
    return runs


def _healthy(*values: bool) -> None:
    if any(value is not True for value in values):
        raise ContractError("failed or unconfirmed evaluation evidence has no objective")


def aggregate_evaluation(plan: EvaluationPlan, projections: tuple[SeedEvaluationProjection, ...]) -> EvaluationObjective:
    """Score every planned result or raise; structural validation is not authority."""
    runs = _plan(plan)
    _tuple(projections)
    seen: dict[int, SeedEvaluationProjection] = {}
    means = {}
    for projection in projections:
        if type(projection) is not SeedEvaluationProjection or type(projection.run) is not RunIdentity:
            raise ContractError("typed seed projection required")
        seed = integer(projection.run.training_seed, 0, 2**63 - 1)
        if seed in seen or seed not in runs or projection.run != runs[seed]:
            raise ContractError("unexpected, duplicated or mismatched run identity")
        integer(projection.used_decisions, 0)
        if projection.used_decisions != plan.decision_budget or projection.conditions_sha256 != plan.conditions_sha256:
            raise ContractError("evaluation must use exact final-D checkpoint and conditions")
        _healthy(projection.completed_normally, projection.source_ok, projection.save_ok, projection.cleanup_ok)
        _tuple(projection.games)
        expected = {game: None for game in runs[seed].games}
        scores: dict[tuple[str, int], float] = {}
        for game in projection.games:
            if type(game) is not GameProjection:
                raise ContractError("typed terminal game projection required")
            _identity(game.identity)
            if game.identity not in expected:
                raise ContractError("unexpected game identity")
            key = (game.identity.game_seed, game.identity.learner_seat)
            if key in scores:
                raise ContractError("duplicate game result")
            integer(game.decisions, 0)
            if type(game.elapsed_seconds) not in (int, float) or not math.isfinite(game.elapsed_seconds) or game.elapsed_seconds < 0:
                raise ContractError("nonfinite or negative evaluation duration")
            _healthy(game.transport_ok, game.save_ok, game.cleanup_ok)
            if game.end_kind != "natural" or type(game.result_json) is not str:
                raise ContractError("only natural terminal results can score")
            try:
                result = json.loads(game.result_json)
                validate_result(result)
                if result is None or canonical(result) != game.result_json:
                    raise ContractError("actual canonical terminal result required")
            except (ValueError, TypeError) as error:
                raise ContractError("invalid terminal result projection") from error
            scores[key] = .5 if result == "draw" else float(result["win"]["winner"] == game.identity.learner_seat)
        if len(scores) != len(expected):
            raise ContractError("missing planned evaluation games")
        # Keep the declared hierarchy explicit even though the balanced panel
        # also permits a flat mean. No failure changes any denominator.
        pair_means = [(scores[(game_seed, 1)] + scores[(game_seed, 2)]) / 2 for game_seed in plan.game_seeds]
        means[seed] = math.fsum(pair_means) / len(pair_means)
        seen[seed] = projection
    if set(seen) != set(runs):
        raise ContractError("missing training seed evaluation")
    seed_scores = tuple((seed, means[seed]) for seed in plan.training_seeds)
    value = math.fsum(score for _, score in seed_scores) / len(seed_scores)
    receipt = digest({"plan": asdict(plan), "projections": [asdict(seen[seed]) for seed in plan.training_seeds]})
    return EvaluationObjective(value, seed_scores, receipt)
