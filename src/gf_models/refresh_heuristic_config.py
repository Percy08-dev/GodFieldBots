"""Distinct configured baseline identity; safe without optional learning extras."""

from __future__ import annotations

from dataclasses import dataclass

from gf_common.domain import ContractError, Json, closed
from gf_common.refresh_heuristic import RefreshOptions


@dataclass(frozen=True)
class RefreshHeuristicConfig:
    model_id: str = "heuristic_refresh"
    version: str = "1"
    attack_threshold: int = 12

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or self.model_id != "heuristic_refresh":
            raise ContractError("closed refresh model identity required")
        RefreshOptions(self.version, self.attack_threshold)


def parse_refresh_model(value: Json) -> RefreshHeuristicConfig:
    return RefreshHeuristicConfig(**closed(value, set(RefreshHeuristicConfig.__dataclass_fields__)))
