"""Closed TOML config, fieldwise precedence, origin-relative paths and provenance."""
from __future__ import annotations

import hashlib
import math
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_common.domain import (
    SIMULATOR_REVISION,
    ActionConfig,
    ContractError,
    Json,
    clone,
    closed,
    digest,
    integer,
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "random"
    version: str = "1"


@dataclass(frozen=True)
class RunConfig:
    name: str = "baseline"
    split: str = "validation"
    output_dir: str = "runs"
    seeds: tuple[str, ...] = ("00000000-0000-0000-0000-000000000001",)
    sampling_seeds: tuple[int, ...] = (0, 1, 2)
    pairs_per_seed: int = 1
    concurrency: int = 2
    wall_seconds: float = 7200
    admission_seconds: float = 5400
    execution_seconds: float = 7140
    finalization_seconds: float = 60
    report_reserve_seconds: float = 1
    max_decisions: int = 20000


@dataclass(frozen=True)
class ServerConfig:
    binary: str = "../GodFieldSimurator/target/debug/gf-test-server"
    config: str = "configs/test-server.toml"
    external_url: str | None = None
    simulator_revision: str = SIMULATOR_REVISION
    source_dir: str | None = None
    fixed_seed_experiment: bool = False
    expected_fingerprint: str | None = None
    startup_seconds: float = 15
    request_seconds: float = 10
    worker_seconds: float = 10
    shutdown_seconds: float = 30
    cleanup_reserve_seconds: float = 2
    transport_retries: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    kind: str = "experiment"
    run: RunConfig = field(default_factory=RunConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opponent: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


@dataclass(frozen=True)
class LoadedConfig:
    config: ExperimentConfig
    effective: Json
    sha256: str
    inputs: tuple[Json, ...]


def _merge(base: Json, override: Json) -> Json:
    result = clone(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else clone(value)
    return result


_TABLES: dict[str, type[Any]] = {"run": RunConfig, "server": ServerConfig, "actions": ActionConfig,
                               "model": ModelConfig, "opponent": ModelConfig, "evaluation": EvaluationConfig}
_PATHS = {("run", "output_dir"), ("server", "binary"), ("server", "config"), ("server", "source_dir")}


def _validate_table(name: str, values: Json) -> None:
    defaults = asdict(_TABLES[name]())
    closed(values, set(), set(defaults))
    for key, value in values.items():
        default = defaults[key]
        if default is None:
            if value is not None and not isinstance(value, str):
                raise ContractError(f"{name}.{key}: expected string")
        elif type(default) is bool:
            if type(value) is not bool:
                raise ContractError(f"{name}.{key}: expected bool")
        elif isinstance(default, tuple):
            if not isinstance(value, (tuple, list)) or not value:
                raise ContractError(f"{name}.{key}: nonempty array required")
            for item in value:
                if type(item) is not type(default[0]):
                    raise ContractError(f"{name}.{key}: invalid array element")
        elif type(default) is int and not key.endswith("seconds"):
            integer(value, 0)
        elif isinstance(default, (int, float)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ContractError(f"{name}.{key}: positive finite number required")
        elif type(value) is not str or not value:
            raise ContractError(f"{name}.{key}: nonempty string required")


def load_config(path: str | Path, overrides: Json | None = None) -> LoadedConfig:
    inputs: list[Json] = []
    stack: set[Path] = set()

    def read(source: Path, expected_kind: str | None = None) -> Json:
        source = source.resolve()
        if source in stack:
            raise ContractError("configuration reference cycle")
        stack.add(source)
        raw_bytes = source.read_bytes()
        data = tomllib.loads(raw_bytes.decode())
        closed(data, {"schema_version", "kind"}, set(_TABLES) | {"references", "training"})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ContractError("unsupported config schema_version")
        if data["kind"] not in ("experiment", "actions", "model") or expected_kind is not None and data["kind"] != expected_kind:
            raise ContractError("unsupported/mismatched preset kind")
        if "training" in data:
            raise ContractError("training adapter not implemented")
        allowed = set(_TABLES) if data["kind"] == "experiment" else {data["kind"]}
        if set(data) - allowed - {"schema_version", "kind", "references"}:
            raise ContractError("preset contains unrelated tables")
        refs = data.get("references", {})
        closed(refs, set(), {"actions", "model", "opponent", "training"})
        if data["kind"] != "experiment" and refs:
            raise ContractError("only experiments may reference presets")
        if "training" in refs:
            raise ContractError("training adapter not implemented")
        merged: Json = {}
        for key, ref in refs.items():
            if not isinstance(ref, str) or not ref:
                raise ContractError("reference path must be a string")
            inherited = read(source.parent / ref, "model" if key == "opponent" else key)
            merged[key] = inherited["model" if key == "opponent" else key]
        for name in _TABLES:
            if name in data:
                _validate_table(name, data[name])
                table = clone(data[name])
                for table_name, key in _PATHS:
                    if name == table_name and key in table and table[key] is not None:
                        table[key] = str((source.parent / table[key]).resolve())
                merged[name] = _merge(merged.get(name, {}), table)
        inputs.append({"path": str(source), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "kind": data["kind"]})
        stack.remove(source)
        return merged

    source = Path(path).resolve()
    data = read(source, "experiment")
    base = asdict(ExperimentConfig())
    for table, key in _PATHS:
        if base[table][key] is not None:
            base[table][key] = str((source.parent / base[table][key]).resolve())
    if overrides:
        closed(overrides, set(), set(_TABLES))
        for name, values in overrides.items():
            _validate_table(name, values)
            for table, key in _PATHS:
                if name == table and key in values and values[key] is not None:
                    values[key] = str(Path(values[key]).resolve())
        data = _merge(data, overrides)
    effective = _merge(base, data)
    for name in _TABLES:
        _validate_table(name, effective[name])
    run = effective["run"]
    if run["split"] not in ("train", "validation", "test"):
        raise ContractError("invalid split")
    if not run["name"].strip() or any(c in run["name"] for c in "/\\"):
        raise ContractError("invalid run name")
    for seed in run["seeds"]:
        try:
            if str(uuid.UUID(seed)) != seed:
                raise ValueError
        except ValueError as exc:
            raise ContractError("seeds must be canonical UUIDs") from exc
    for key in ("pairs_per_seed", "concurrency", "max_decisions"):
        integer(run[key], 1, 2**32 - 1)
    for seed in run["sampling_seeds"]:
        integer(seed)
    if run["admission_seconds"] > run["execution_seconds"] or run["execution_seconds"] + run["finalization_seconds"] > run["wall_seconds"]:
        raise ContractError("run deadlines inconsistent")
    for name in ("model", "opponent"):
        if effective[name]["model_id"] not in ("random", "heuristic") or effective[name]["version"] != "1":
            raise ContractError("unsupported model/version")
    actions = effective["actions"]
    if actions["scorer"] != "public_heuristic_v1":
        raise ContractError("unsupported search scorer")
    if actions["strategy"] not in ("beam", "enumerate"):
        raise ContractError("unsupported candidate strategy")
    for key in ("beam_width", "max_candidates", "max_expansions"):
        integer(actions[key], 1)
    integer(actions["max_sequence_length"])
    if set(actions["exchange_allocations"]) - {"balanced", "hp", "mp", "money"} or set(actions["targets"]) - {"self", "opponent"}:
        raise ContractError("unknown allocation or target")
    server = effective["server"]
    if run["report_reserve_seconds"] + server["cleanup_reserve_seconds"] >= run["finalization_seconds"]:
        raise ContractError("cleanup/report reserves must fit inside finalization allowance")
    integer(server["transport_retries"], 0, 2)
    if server["simulator_revision"] != SIMULATOR_REVISION:
        raise ContractError("unsupported simulator contract revision")
    if not server["fixed_seed_experiment"]:
        raise ContractError("repeated server requires explicit fixed_seed_experiment")
    if server["external_url"] is not None:
        from urllib.parse import urlsplit
        parts = urlsplit(server["external_url"])
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ContractError("invalid external URL")
        if len(set(run["seeds"])) != 1 or not server["expected_fingerprint"]:
            raise ContractError("external server requires one seed and explicit expected fingerprint")
    if server["expected_fingerprint"] is not None:
        import hashlib as _hashlib
        expected = "v1-" + _hashlib.sha256(b"godfield-test-server-seed-fingerprint-v1" + uuid.UUID(run["seeds"][0]).bytes).hexdigest()
        if server["expected_fingerprint"] != expected or len(set(run["seeds"])) != 1:
            raise ContractError("expected fingerprint/seed mismatch")
    for key in ("bootstrap_samples",):
        integer(effective["evaluation"][key], 1)
    converted = {}
    for name, cls in _TABLES.items():
        values = effective[name]
        defaults = asdict(cls())
        values = {k: tuple(v) if isinstance(defaults[k], tuple) else v for k, v in values.items()}
        converted[name] = cls(**values)
    config = ExperimentConfig(**converted)
    return LoadedConfig(config, clone(asdict(config)), digest(asdict(config)), tuple(inputs))
