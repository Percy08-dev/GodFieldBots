"""Closed TOML config, fieldwise precedence, origin-relative paths and provenance."""
from __future__ import annotations

import hashlib
import math
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_common.action_contract import engineering_contract, parse_action_contract
from gf_common.card_instance import (
    CARD_INSTANCE_SIMULATOR_REVISION,
    CORRECTED_SIMULATOR_REVISION,
    ActionConfigV3,
    action_config_v3,
    card_instance_contract,
    parse_card_instance_contract,
)
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
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.structured import ActionConfigV2, action_config_v2
from gf_common.structured_layout import layout_for_contract
from gf_models.config import (
    GRUModelConfig,
    MLPModelConfig,
    ModelConfig,
    PPOTrainingConfig,
    SequencePPOTrainingConfig,
    TrainingConfig,
    TransformerModelConfig,
    model_config,
    training_config,
    validate_capacity,
)
from gf_models.structured_binding import (
    ConfiguredNeuralModelV2,
    ConfiguredNeuralModelV3,
    model_defaults_v2,
    model_defaults_v3,
)
from gf_models.structured_tree_config import ConfiguredTreeModelV2


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
class RunConfigV2(RunConfig):
    global_cuda_budget_bytes: int = 12884901888
    max_participant_workers: int = 4


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
class ServerConfigV2(ServerConfig):
    simulator_revision: str = CARD_INSTANCE_SIMULATOR_REVISION
    gameplay_api: str = "v2"


@dataclass(frozen=True)
class CorrectedServerConfigV2(ServerConfigV2):
    binary: str = "../GodFieldSimurator/target/release/gf-test-server"
    simulator_revision: str = CORRECTED_SIMULATOR_REVISION


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    kind: str = "experiment"
    run: RunConfig | RunConfigV2 = field(default_factory=RunConfig)
    server: ServerConfig | ServerConfigV2 = field(default_factory=ServerConfig)
    actions: ActionConfig | ActionConfigV2 | ActionConfigV3 = field(default_factory=ActionConfig)
    model: ModelConfig | MLPModelConfig | GRUModelConfig | TransformerModelConfig | ConfiguredNeuralModelV2 | ConfiguredNeuralModelV3 | ConfiguredTreeModelV2 = field(default_factory=ModelConfig)
    opponent: ModelConfig | MLPModelConfig | GRUModelConfig | TransformerModelConfig | ConfiguredNeuralModelV2 | ConfiguredNeuralModelV3 | ConfiguredTreeModelV2 = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig | PPOTrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreparationConfig = field(default_factory=PreparationConfig)


@dataclass(frozen=True)
class LoadedConfig:
    config: ExperimentConfig
    effective: Json
    sha256: str
    inputs: tuple[Json, ...]


def _merge(base: Json, override: Json) -> Json:
    try:
        result = clone(base)
        for key, value in override.items():
            result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else clone(value)
        return result
    except ContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractError("configuration contains invalid/nonfinite values") from exc


_TABLES: dict[str, type[Any]] = {"run": RunConfig, "server": ServerConfig, "actions": ActionConfig,
                               "model": ModelConfig, "opponent": ModelConfig, "evaluation": EvaluationConfig, "training": TrainingConfig, "preprocessing": PreparationConfig}
_PATHS = {("run", "output_dir"), ("server", "binary"), ("server", "config"), ("server", "source_dir"), ("model", "checkpoint"), ("opponent", "checkpoint"), ("training", "pilot_run"), ("training", "dataset_manifest")}


def _validate_table(name: str, values: Json, *, context: Json | None = None) -> None:
    context = {} if context is None else context
    values = _merge(context.get(name, {}), values)
    def check(values: Json, defaults: Json) -> None:
        closed(values, set(), set(defaults))
        for key, value in values.items():
            default = defaults[key]
            if isinstance(default, dict):
                check(value, default)
            elif default is None:
                if value is not None:
                    if name == "training" and key == "accepted_decision_target":
                        integer(value, 1)
                    elif name in ("model", "opponent") and key == "cuda_visible_devices" and type(value) is str:
                        pass
                    elif not isinstance(value, str) or not value:
                        raise ContractError(f"{name}.{key}: expected nonempty string")
            elif type(default) is bool:
                if type(value) is not bool:
                    raise ContractError(f"{name}.{key}: expected bool")
            elif isinstance(default, tuple):
                if not isinstance(value, (tuple, list)) or not value and key not in ("train_runs", "validation_runs"):
                    raise ContractError(f"{name}.{key}: array required")
                item_type = type(default[0]) if default else str
                if any(type(item) is not item_type for item in value):
                    raise ContractError(f"{name}.{key}: invalid array element")
            elif type(default) is int and not key.endswith("seconds"):
                integer(value, 0)
            elif isinstance(default, (int, float)):
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value == 0 and key != "max_overshoot":
                    raise ContractError(f"{name}.{key}: positive finite number required")
            elif type(value) is not str or not value:
                raise ContractError(f"{name}.{key}: nonempty string required")
    cls = TransformerModelConfig if name in ("model", "opponent") and values.get("model_id") == "transformer" else GRUModelConfig if name in ("model", "opponent") and values.get("model_id") == "gru" else SequencePPOTrainingConfig if name == "training" and "sequence_length" in values else MLPModelConfig if name in ("model", "opponent") and values.get("model_id") == "mlp" else PPOTrainingConfig if name == "training" and values.get("algorithm") == "ppo" else _TABLES[name]
    # Partial model presets may specify closed structural keys before dispatch.
    defaults = asdict(cls())
    structured_model = name in ("model", "opponent") and values.get("version") in ("2", "3")
    if structured_model:
        if values.get("version") == "3":
            if values.get("model_id", "mlp") not in ("mlp", "gru", "transformer"):
                raise ContractError("model3 currently supports neural Config only; Tree3 is not implemented")
            defaults = asdict(model_defaults_v3(values.get("model_id", "mlp")))
        else:
            defaults = asdict(ConfiguredTreeModelV2() if values.get("model_id") == "lightgbm" else model_defaults_v2(values.get("model_id", "mlp")))
    elif name == "actions" and values.get("version") == "2":
        defaults = asdict(ActionConfigV2(action_contract=engineering_contract()))
    elif name == "actions" and values.get("version") == "3":
        defaults = asdict(ActionConfigV3(action_contract=card_instance_contract()))
    elif name == "server" and values.get("gameplay_api") == "v2":
        defaults = asdict(CorrectedServerConfigV2() if values.get("simulator_revision") == CORRECTED_SIMULATOR_REVISION else ServerConfigV2())
    elif name == "run" and context.get("actions", {}).get("version") in ("2", "3"):
        defaults = asdict(RunConfigV2())
    if name in ("model", "opponent") and "model_id" not in values and not structured_model:
        defaults["inference"] = asdict(MLPModelConfig().inference)
        defaults["network"].update(asdict(MLPModelConfig().network))
        defaults["network"].update(asdict(GRUModelConfig().network))
        defaults["network"].update(asdict(TransformerModelConfig().network))
        defaults["inference"].update(asdict(GRUModelConfig().inference))
    if name == "training" and "algorithm" not in values:
        defaults.update(asdict(SequencePPOTrainingConfig()))
    check(values, defaults)


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
        if data["kind"] not in ("experiment", "actions", "model", "training", "preprocessing") or expected_kind is not None and data["kind"] != expected_kind:
            raise ContractError("unsupported/mismatched preset kind")
        if data["kind"] == "preprocessing" and "preprocessing" not in data:
            raise ContractError("preprocessing preset requires its settings table")
        allowed = set(_TABLES) if data["kind"] == "experiment" else {data["kind"]} | ({"training"} if data["kind"] == "model" else set())
        if set(data) - allowed - {"schema_version", "kind", "references"}:
            raise ContractError("preset contains unrelated tables")
        refs = data.get("references", {})
        closed(refs, set(), {"actions", "model", "opponent", "training", "preprocessing"})
        if data["kind"] != "experiment" and refs:
            raise ContractError("only experiments may reference presets")
        merged: Json = {}
        for key in ("actions", "training", "model", "opponent", "preprocessing"):
            if key not in refs:
                continue
            ref = refs[key]
            if not isinstance(ref, str) or not ref:
                raise ContractError("reference path must be a string")
            try:
                inherited = read(source.parent / ref, "model" if key == "opponent" else key)
            except OSError as exc:
                raise ContractError("referenced configuration unavailable") from exc
            merged[key] = inherited["model" if key == "opponent" else key]
            if key == "model" and "training" in inherited:
                merged["training"] = _merge(merged.get("training", {}), inherited["training"])
        for name in _TABLES:
            if name in data:
                table_context = _merge(merged, {"actions": data.get("actions", merged.get("actions", {}))})
                _validate_table(name, data[name], context=table_context)
                table = clone(data[name])
                for table_name, key in _PATHS:
                    if name == table_name and key in table and table[key] is not None:
                        table[key] = str((source.parent / table[key]).resolve())
                if name == "training":
                    for key in ("train_runs", "validation_runs"):
                        if key in table:
                            table[key] = [str((source.parent / item).resolve()) for item in table[key]]
                merged[name] = _merge(merged.get(name, {}), table)
        inputs.append({"path": str(source), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "kind": data["kind"]})
        stack.remove(source)
        return merged

    source = Path(path).resolve()
    data = read(source, "experiment")
    base = asdict(ExperimentConfig())
    for table, key in _PATHS:
        if key in base[table] and base[table][key] is not None:
            base[table][key] = str((source.parent / base[table][key]).resolve())
    if overrides:
        closed(overrides, set(), set(_TABLES))
        validation_context = _merge(data, {"actions": _merge(data.get("actions", {}), overrides.get("actions", {}))})
        for name, values in overrides.items():
            _validate_table(name, values, context=validation_context)
            for table, key in _PATHS:
                if name == table and key in values and values[key] is not None:
                    values[key] = str(Path(values[key]).resolve())
            if name == "training":
                for key in ("train_runs", "validation_runs"):
                    if key in values:
                        values[key] = [str(Path(item).resolve()) for item in values[key]]
        data = _merge(data, overrides)
    structured_actions = data.get("actions", {}).get("version") == "2"
    card_instance_actions = data.get("actions", {}).get("version") == "3"
    card_instance_server = data.get("server", {}).get("gameplay_api") == "v2"
    card_server_cls: type[ServerConfigV2] = ServerConfigV2
    card_revision = CARD_INSTANCE_SIMULATOR_REVISION
    if card_instance_actions != card_instance_server:
        raise ContractError("actions3 requires an explicit Gameplay v2 server and conversely")
    if card_instance_actions:
        if "action_contract" not in data["actions"]:
            raise ContractError("raw actions3 Config must explicitly declare an action_contract")
        selected_contract = parse_card_instance_contract(data["actions"]["action_contract"])
        card_revision = selected_contract.simulator_revision
        if data["server"].get("simulator_revision") != card_revision:
            raise ContractError("Gameplay v2 requires the explicit pinned simulator revision")
        if card_revision == CORRECTED_SIMULATOR_REVISION:
            card_server_cls = CorrectedServerConfigV2
        base["actions"] = asdict(ActionConfigV3(action_contract=selected_contract))
        base["server"] = asdict(card_server_cls())
        base["run"] = asdict(RunConfigV2())
        for table, key in _PATHS:
            if table in ("server", "run") and base[table].get(key) is not None:
                base[table][key] = str((source.parent / base[table][key]).resolve())
    if structured_actions:
        # Resolved raw inputs must explicitly declare/inherit the descriptor.
        # Current defaults must never upgrade a pre-adoption v2 Config.
        if "action_contract" not in data["actions"]:
            raise ContractError("raw actions2 Config must explicitly declare an action_contract")
        parse_action_contract(data["actions"]["action_contract"])
        base["actions"] = asdict(ActionConfigV2(action_contract=engineering_contract()))
        base["run"] = asdict(RunConfigV2())
    if (any(data.get(name, {}).get("version") == "3" for name in ("model", "opponent"))
            and (not card_instance_actions or card_revision != CORRECTED_SIMULATOR_REVISION)):
        raise ContractError("model3 experiment Config requires the corrected card-instance contract")
    for name in ("model", "opponent"):
        if data.get(name, {}).get("version") == "3":
            base[name] = asdict(model_defaults_v3(data[name].get("model_id", "mlp")))
            continue
        if data.get(name, {}).get("version") == "2":
            base[name] = asdict(ConfiguredTreeModelV2() if data[name].get("model_id") == "lightgbm" else model_defaults_v2(data[name].get("model_id", "mlp")))
            continue
        if data.get(name, {}).get("model_id") == "transformer":
            base[name] = asdict(TransformerModelConfig())
        if data.get(name, {}).get("model_id") == "gru":
            base[name] = asdict(GRUModelConfig())
        if data.get(name, {}).get("model_id") == "mlp":
            base[name] = asdict(MLPModelConfig())
    if data.get("training", {}).get("algorithm") == "ppo":
        base["training"] = asdict(SequencePPOTrainingConfig() if data.get("model", {}).get("model_id") in ("gru", "transformer") and data.get("model", {}).get("version", "1") == "1" else PPOTrainingConfig())
    effective = _merge(base, data)
    for name in _TABLES:
        _validate_table(name, effective[name], context=effective)
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
    if structured_actions or card_instance_actions:
        integer(run["concurrency"], 1, 2)
        integer(run["max_participant_workers"], 1, 4)
        integer(run["global_cuda_budget_bytes"], 1, 12884901888)
        if run["max_participant_workers"] < 2 * run["concurrency"]:
            raise ContractError("participant capacity must cover both players of every concurrent game")
    models = {name: model_config(effective[name], layout=layout_for_contract(selected_contract))
              if effective[name].get("version") == "3" else model_config(effective[name])
              for name in ("model", "opponent")}
    training = training_config(effective["training"])
    preprocessing = preparation_config(effective["preprocessing"])
    if ("training" in data or "training" in data.get("references", {})) and models["model"].model_id not in ("lightgbm", "mlp", "gru", "transformer"):
        raise ContractError("training/model mismatch")
    if (isinstance(training, PPOTrainingConfig)) != (models["model"].model_id in ("mlp", "gru", "transformer")) and "training" in data:
        raise ContractError("training algorithm/model mismatch")
    legacy_sequence = models["model"].model_id in ("gru", "transformer") and models["model"].version == "1"
    if isinstance(training, SequencePPOTrainingConfig) != legacy_sequence and isinstance(training, PPOTrainingConfig):
        raise ContractError("sequence training/model mismatch")
    for model in models.values():
        validate_capacity(model, effective["actions"]["max_sequence_length"])
    if card_instance_actions:
        actions_v3 = action_config_v3(effective["actions"])
        if preprocessing.backend != "python" or preprocessing.reader != "legacy":
            raise ContractError("actions3 preprocessing is not implemented; only inert Python/legacy settings are supported")
        if any(not isinstance(model, ConfiguredNeuralModelV3)
               and (type(model) is not ModelConfig or model.model_id not in ("random", "heuristic"))
               for model in models.values()):
            raise ContractError("actions3 requires model3 neural or baseline Config; legacy neural/tree artifacts are incompatible")
        if any(isinstance(model, ConfiguredNeuralModelV3)
               and actions_v3.max_candidates > model.inference.max_candidates for model in models.values()):
            raise ContractError("action presentation limit exceeds configured model3 admission")
    elif structured_actions:
        actions_v2 = action_config_v2(effective["actions"])
        if preprocessing.reader != "structured-v1":
            raise ContractError("actions2 requires explicit preprocessing.reader='structured-v1'")
        if any(not isinstance(model, (ConfiguredNeuralModelV2, ConfiguredTreeModelV2)) and model.model_id not in ("random", "heuristic") for model in models.values()):
            raise ContractError("actions2 cannot consume legacy neural/tree model artifacts")
        if any(isinstance(model, (ConfiguredNeuralModelV2, ConfiguredTreeModelV2)) and actions_v2.max_candidates > model.inference.max_candidates for model in models.values()):
            raise ContractError("action presentation limit exceeds configured model admission")
    else:
        if preprocessing.reader == "structured-v1" or any(isinstance(model, (ConfiguredNeuralModelV2, ConfiguredTreeModelV2)) for model in models.values()):
            raise ContractError("structured model/reader requires actions2")
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
    if server["simulator_revision"] != (card_revision if card_instance_actions else SIMULATOR_REVISION):
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
    converted: dict[str, Any] = {}
    for name, cls in _TABLES.items():
        if name in models:
            converted[name] = models[name]
            continue
        if name == "training":
            converted[name] = training
            continue
        if name == "preprocessing":
            converted[name] = preprocessing
            continue
        if name == "actions" and structured_actions:
            converted[name] = actions_v2
            continue
        if name == "actions" and card_instance_actions:
            converted[name] = actions_v3
            continue
        if name == "server" and card_instance_server:
            cls = card_server_cls
        if name == "run" and (structured_actions or card_instance_actions):
            cls = RunConfigV2
        values = effective[name]
        defaults = asdict(cls())
        values = {k: tuple(v) if isinstance(defaults[k], tuple) else v for k, v in values.items()}
        converted[name] = cls(**values)
    config = ExperimentConfig(**converted)
    return LoadedConfig(config, clone(asdict(config)), digest(asdict(config)), tuple(inputs))
