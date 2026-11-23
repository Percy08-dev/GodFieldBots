"""Pure declarations for optional HPO checkpoint evaluation."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, Literal

from gf_common.domain import ContractError, canonical, closed

_DEFAULT_UPDATES = (0, 100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000)
_OPTIONAL_FIELDS = {"updates", "retain_updates", "final_test_game_seeds"}
_PANEL_DIGEST = "evaluation_game_seeds_sha256"


@dataclass(frozen=True)
class HPOCheckpointEvaluationConfig:
    wall_seconds: float
    updates: tuple[int, ...]
    retain_updates: tuple[int, ...]
    final_test_game_seeds: tuple[str, ...]
    enabled: Literal[True] = field(default=True, init=False)


def game_seeds(value: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(value) not in (list, tuple) or (not value and not allow_empty):
        raise ContractError("explicit evaluation GameSeed sequence required")
    result: set[str] = set()
    for seed in value:
        if type(seed) is not str:
            raise ContractError("evaluation GameSeed must be a canonical UUID")
        try:
            parsed = uuid.UUID(seed)
        except ValueError as exc:
            raise ContractError("evaluation GameSeed must be a canonical UUID") from exc
        if str(parsed) != seed:
            raise ContractError("evaluation GameSeed must be a canonical UUID")
        if seed in result:
            raise ContractError("duplicate experiment seed")
        result.add(seed)
    return tuple(value)


def _updates(value: Any, *, name: str, cap: int) -> tuple[int, ...]:
    if (type(value) not in (list, tuple)
            or any(type(update) is not int or not 0 <= update <= cap for update in value)
            or any(left >= right for left, right in pairwise(value))):
        raise ContractError(f"checkpoint_evaluation.{name} must contain increasing integers in 0..{cap}")
    return tuple(value)


def checkpoint_evaluation_config(value: Any, *, schema_version: int, max_updates: int,
                                 run_wall_seconds: float,
                                 evaluation_game_seeds: tuple[str, ...]) -> HPOCheckpointEvaluationConfig | None:
    """Normalize the optional campaign table and check any configured seed split."""
    if type(value) is not dict or "enabled" not in value:
        raise ContractError("checkpoint_evaluation table with explicit enabled required")
    enabled = value["enabled"]
    if type(enabled) is not bool:
        raise ContractError("checkpoint_evaluation.enabled must be a boolean")
    if not enabled:
        if set(value) != {"enabled"}:
            raise ContractError("disabled checkpoint_evaluation accepts only enabled")
        return None
    if schema_version not in (4, 5, 6):
        raise ContractError("checkpoint_evaluation requires HPO campaign schema 4, 5 or 6")
    if set(value) - {"enabled", "wall_seconds"} - _OPTIONAL_FIELDS:
        raise ContractError("unknown checkpoint_evaluation setting")
    wall: Any = value.get("wall_seconds")
    if type(wall) not in (int, float) or not 0 < wall <= run_wall_seconds:
        raise ContractError("checkpoint_evaluation.wall_seconds must be positive and within run_wall_seconds")
    updates = (_updates(value["updates"], name="updates", cap=max_updates) if "updates" in value
        else tuple(update for update in _DEFAULT_UPDATES if update <= max_updates))
    if not updates:
        raise ContractError("checkpoint_evaluation.updates must contain at least one checkpoint")
    retained = _updates(value.get("retain_updates", ()), name="retain_updates", cap=max_updates)
    holdout = game_seeds(value.get("final_test_game_seeds", ()), allow_empty=True)
    if not set(holdout).isdisjoint(evaluation_game_seeds):
        raise ContractError("checkpoint_evaluation final-test seeds must be disjoint from evaluation_game_seeds")
    return HPOCheckpointEvaluationConfig(float(wall), updates, retained, holdout)


def require_checkpoint_evaluation_room(config: HPOCheckpointEvaluationConfig, *, finalization_seconds: float) -> None:
    if config.wall_seconds <= finalization_seconds:
        raise ContractError("checkpoint_evaluation.wall_seconds must exceed the run finalization reserve")


def persisted_checkpoint_evaluation(value: Any, *, schema_version: int, max_updates: int,
                                    run_wall_seconds: float, finalization_seconds: float) -> HPOCheckpointEvaluationConfig:
    """Read an active effective-run block, which binds the campaign panel by digest."""
    raw = closed(value, {"enabled", "wall_seconds", _PANEL_DIGEST} | _OPTIONAL_FIELDS)
    panel = raw[_PANEL_DIGEST]
    if type(panel) is not str or re.fullmatch(r"[0-9a-f]{64}", panel) is None:
        raise ContractError("checkpoint_evaluation requires the original evaluation_game_seeds_sha256")
    options = {key: item for key, item in raw.items() if key != _PANEL_DIGEST}
    config = checkpoint_evaluation_config(options, schema_version=schema_version, max_updates=max_updates,
                                          run_wall_seconds=run_wall_seconds, evaluation_game_seeds=())
    if config is None or canonical(asdict(config)) != canonical(options):
        raise ContractError("active normalized checkpoint_evaluation required")
    require_checkpoint_evaluation_room(config, finalization_seconds=finalization_seconds)
    return config
