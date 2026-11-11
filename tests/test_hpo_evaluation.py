"""Strict structural projections cannot turn failed/missing games into scores."""
from dataclasses import replace

import pytest

from gf_common.domain import ContractError
from gf_evaluation.hpo import (
    EvaluationPlan,
    GameIdentity,
    GameProjection,
    RunIdentity,
    SeedEvaluationProjection,
    aggregate_evaluation,
)


def panel():
    runs = tuple(RunIdentity(seed, f"run-{seed}", str(seed % 10) * 64, "a" * 64,
        tuple(GameIdentity(game, seat, f"episode-{seed}-{game}-{seat}",
                           f"game-{seed}-{game}-{seat}")
              for game in ("alpha", "beta") for seat in (1, 2))) for seed in (11, 22))
    plan = EvaluationPlan((11, 22), ("alpha", "beta"), 100, "b" * 64, runs)
    # Seed11: win,draw,loss,win => .625. Seed22: all loss => 0. Mean .3125.
    outcomes = ({"win": {"winner": 1}}, "draw", {"win": {"winner": 2}},
                {"win": {"winner": 2}})
    projections = tuple(SeedEvaluationProjection(run, 100, "b" * 64,
        tuple(GameProjection.from_result(identity, result=(outcomes[i] if run.training_seed == 11
             else {"win": {"winner": 3 - identity.learner_seat}}), decisions=0)
              for i, identity in enumerate(run.games))) for run in runs)
    return plan, projections


def test_complete_panel_scores_actual_dto_from_learner_perspective_and_keeps_zero_decisions():
    plan, projections = panel()
    objective = aggregate_evaluation(plan, projections)
    assert objective.value == .3125
    assert objective.seed_scores == ((11, .625), (22, 0.))
    assert objective.label == "exploratory-hpo-validation"


def test_result_is_frozen_at_projection_boundary():
    plan, projections = panel()
    original = {"win": {"winner": 1}}
    game = GameProjection.from_result(projections[0].games[0].identity, result=original)
    original["win"]["winner"] = 2
    projections = (replace(projections[0], games=(game, *projections[0].games[1:])), projections[1])
    assert aggregate_evaluation(plan, projections).value == .3125


@pytest.mark.parametrize("mutation", ["missing-seed", "extra-seed", "duplicate-seed", "missing-game",
    "extra-game", "duplicate-game", "wrong-gameid", "wrong-episode", "wrong-seat", "wrong-gameseed",
    "wrong-run", "wrong-checkpoint", "wrong-source", "wrong-final-d", "wrong-conditions",
    "aborted", "truncated", "transport", "save", "cleanup", "source", "run-failed", "run-save", "run-cleanup", "nan", "infinity"])
def test_incomplete_or_failed_evidence_has_no_objective(mutation):
    plan, projections = panel()
    first, second = projections
    game = first.games[0]
    if mutation == "missing-seed":
        projections = (first,)
    elif mutation == "extra-seed":
        projections += (replace(first, run=replace(first.run, training_seed=33)),)
    elif mutation == "duplicate-seed":
        projections = (first, first)
    else:
        if mutation == "missing-game":
            first = replace(first, games=first.games[1:])
        elif mutation in ("extra-game", "duplicate-game"):
            first = replace(first, games=(*first.games, game))
        elif mutation in ("wrong-gameid", "wrong-episode", "wrong-seat", "wrong-gameseed"):
            key, value = {"wrong-gameid": ("game_id", "substituted"),
                "wrong-episode": ("episode_id", "substituted"), "wrong-seat": ("learner_seat", 2),
                "wrong-gameseed": ("game_seed", "other")}[mutation]
            game = replace(game, identity=replace(game.identity, **{key: value}))
            first = replace(first, games=(game, *first.games[1:]))
        elif mutation in ("wrong-run", "wrong-checkpoint", "wrong-source"):
            key = {"wrong-run": "run_id", "wrong-checkpoint": "checkpoint_sha256",
                   "wrong-source": "source_receipt_sha256"}[mutation]
            first = replace(first, run=replace(first.run, **{key: "c" * 64}))
        elif mutation == "wrong-final-d":
            first = replace(first, used_decisions=99)
        elif mutation == "wrong-conditions":
            first = replace(first, conditions_sha256="c" * 64)
        elif mutation in ("source", "run-failed", "run-save", "run-cleanup"):
            key = {"source": "source_ok", "run-failed": "completed_normally", "run-save": "save_ok", "run-cleanup": "cleanup_ok"}[mutation]
            first = replace(first, **{key: False})
        else:
            fields = ({"end_kind": mutation} if mutation in ("aborted", "truncated") else
                {mutation + "_ok": False} if mutation in ("transport", "save", "cleanup") else
                {"elapsed_seconds": float("nan") if mutation == "nan" else float("inf")})
            first = replace(first, games=(replace(game, **fields), *first.games[1:]))
        projections = (first, second)
    with pytest.raises(ContractError):
        aggregate_evaluation(plan, projections)


@pytest.mark.parametrize("result", [None, {"win": {"winner": 0}}, {"win": {"winner": True}},
    {"win": {"winner": 1, "hidden": 4}}, {"draw": True}, float("nan")])
def test_invalid_terminal_result_never_scores(result):
    plan, projections = panel()
    with pytest.raises((ContractError, ValueError)):
        game = GameProjection.from_result(projections[0].games[0].identity, result=result)
        aggregate_evaluation(plan, (replace(projections[0], games=(game, *projections[0].games[1:])), projections[1]))


@pytest.mark.parametrize("field,value", [("training_seeds", (11, 11)), ("training_seeds", (True, 22)),
    ("game_seeds", ("alpha", "alpha")), ("game_seeds", ()), ("decision_budget", True),
    ("decision_budget", 0), ("conditions_sha256", "short")])
def test_invalid_plan_rejects(field, value):
    plan, projections = panel()
    with pytest.raises(ContractError):
        aggregate_evaluation(replace(plan, **{field: value}), projections)


def test_cross_seed_gameid_collision_rejects_and_projection_digest_binds_results():
    plan, projections = panel()
    baseline = aggregate_evaluation(plan, projections)
    game = projections[0].games[0]
    changed = GameProjection.from_result(game.identity, result="draw")
    updated = (replace(projections[0], games=(changed, *projections[0].games[1:])), projections[1])
    assert aggregate_evaluation(plan, updated).projection_sha256 != baseline.projection_sha256
    duplicate = replace(projections[1].games[0].identity, game_id=game.identity.game_id)
    second_run = replace(plan.runs[1], games=(duplicate, *plan.runs[1].games[1:]))
    with pytest.raises(ContractError):
        aggregate_evaluation(replace(plan, runs=(plan.runs[0], second_run)), projections)


def test_projection_order_does_not_change_equal_seed_or_game_weighting():
    plan, projections = panel()
    reordered = tuple(replace(projection, games=tuple(reversed(projection.games)))
                      for projection in reversed(projections))
    assert aggregate_evaluation(plan, reordered).value == .3125


def test_expected_manifest_must_cover_full_declared_training_seed_group():
    plan, projections = panel()
    with pytest.raises(ContractError):
        aggregate_evaluation(replace(plan, runs=plan.runs[:1]), projections)
