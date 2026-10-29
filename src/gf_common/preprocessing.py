"""Closed, stdlib-only resource settings shared by CLI and preparation."""
from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from gf_common.domain import ContractError, Json, closed, integer


@dataclass(frozen=True)
class PreparationConfig:
    memory_budget_mib: int = 8192
    cache_budget_mib: int = 512
    chunk_target_mib: int = 64
    max_source_file_mib: int = 64
    max_record_mib: int = 8
    max_in_flight: int = 1
    workers: int = 1
    backend: str = "python"
    scratch_budget_mib: int = 8192
    wall_seconds: float = 7200
    cleanup_seconds: float = 5


def preparation_config(value: Json) -> PreparationConfig:
    defaults = asdict(PreparationConfig())
    closed(value, set(), set(defaults))
    values = {**defaults, **value}
    for key in defaults:
        if key in ("backend", "wall_seconds", "cleanup_seconds"):
            continue
        integer(values[key], 0 if key == "cache_budget_mib" else 1)
    for key in ("wall_seconds", "cleanup_seconds"):
        number = values[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ContractError(f"preprocessing.{key}: positive finite number required")
    if values["backend"] not in ("python", "rust"):
        raise ContractError("unsupported preprocessing backend")
    if values["backend"] == "python" and (values["workers"] != 1 or values["max_in_flight"] != 1):
        raise ContractError("Python reference preparation is serial")
    if values["backend"] == "rust" and values["workers"] not in (1, 2, 4, 8):
        raise ContractError("native preprocessing supports 1, 2, 4 or 8 workers")
    if (values["cache_budget_mib"] > values["memory_budget_mib"]
            or values["chunk_target_mib"] > values["memory_budget_mib"]
            or values["max_record_mib"] > values["max_source_file_mib"]
            or values["cleanup_seconds"] >= values["wall_seconds"]):
        raise ContractError("inconsistent preprocessing limits")
    return PreparationConfig(**values)


def load_preprocessing_config(path: str | Path | None) -> PreparationConfig:
    """Load an independent preset without loading server/model configuration."""
    if path is None:
        return PreparationConfig()
    data = tomllib.loads(Path(path).read_text())
    closed(data, {"schema_version", "kind", "preprocessing"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["kind"] != "preprocessing":
        raise ContractError("unsupported preprocessing preset")
    return preparation_config(data["preprocessing"])
