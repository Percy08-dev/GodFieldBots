"""Explicit HPO TOML loading and pure joint operating admission.

This module performs no allocation, Simulator activity, directory creation,
filesystem-capacity validation or optional learning-library imports.
"""
from __future__ import annotations

import hashlib
import re
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gf_cli.config import LoadedConfig, load_config
from gf_common.card_instance import (
    CORRECTED_SIMULATOR_REVISION,
    ActionConfigV3,
    corrected_card_instance_contract,
)
from gf_common.domain import ContractError, Json, clone, closed, digest, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, StructuredLayout
from gf_models.hpo_config import model_config_v4, model_defaults_v4
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_models.structured_config import InferenceBindingV2, ModelConfigV4, resource_estimate
from gf_training.hpo_config import (
    HPOTrainingConfig,
    SearchSpace,
    effective_settings,
    finite_number,
    hpo_training_config,
    normalize_training_for_model,
    search_space,
)

GIB = 1024**3


@dataclass(frozen=True)
class HPOOperatingConfig:
    managed_ram_bytes: int = 24 * GIB
    managed_cuda_bytes: int = 12 * GIB
    game_concurrency: int = 2
    max_participant_workers: int = 4
    trial_concurrency: int = 1
    source_budget_bytes: int = 2 * GIB
    numeric_budget_bytes: int = 512 * 1024**2
    save_budget_bytes: int = 512 * 1024**2
    collector_budget_bytes: int = 256 * 1024**2
    parent_budget_bytes: int = 256 * 1024**2
    max_updates: int = 1000
    inference_latest: int = 2
    inference_periodic_decisions: int = 10000
    pool_every_decisions: int = 10000
    opponent_pool_size: int = 5
    resume_latest: int = 2


@dataclass(frozen=True)
class HPOStorageConfig:
    parent_root: str = "runs/hpo"
    numeric_root: str = "runs/hpo/numeric"
    optuna_path: str = "runs/hpo/optuna.sqlite3"
    originals_root: str | None = None
    windows_c_root: str = "/mnt/c"
    destination_capacity_bytes: int = 10 * GIB
    admission_reservation_bytes: int = GIB
    windows_c_min_free_bytes: int = GIB
    ext4_control_required: bool = True


@dataclass(frozen=True)
class HPOCampaignConfig:
    schema_version: int
    kind: str
    base_experiment: str
    base_config_sha256: str
    model_id: str
    stage: str
    trial_count: int
    decision_budget: int
    learning_seeds: tuple[int, ...]
    evaluation_game_seeds: tuple[str, ...]
    sampler_seed: int
    startup_trials: int
    session_seconds: float
    run_wall_seconds: float
    space: SearchSpace
    model: ModelConfigV4
    training: HPOTrainingConfig
    inference: InferenceBindingV2
    operating: HPOOperatingConfig = field(default_factory=HPOOperatingConfig)
    storage: HPOStorageConfig = field(default_factory=HPOStorageConfig)

    @property
    def study_identity(self) -> str:
        """Bind scientific settings, including the fixed opponent curriculum."""
        settings = effective_settings(self.model, self.training, {}, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
        return digest({name: getattr(self, name) for name in (
            "schema_version", "kind", "base_config_sha256", "model_id", "stage", "trial_count", "decision_budget",
            "learning_seeds", "evaluation_game_seeds", "sampler_seed", "startup_trials", "run_wall_seconds")}
            | {"space": asdict(self.space), "control": settings.identity,
               "curriculum": {"pool_every_decisions": self.operating.pool_every_decisions,
                              "opponent_pool_size": self.operating.opponent_pool_size}})


@dataclass(frozen=True)
class LoadedHPOConfig:
    config: HPOCampaignConfig
    base: LoadedConfig
    effective: Json
    sha256: str
    inputs: tuple[Json, ...]


def _merge(base: Json, override: Any) -> Json:
    if type(override) is not dict or set(override) - set(base):
        raise ContractError("unknown HPO config setting")
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(base[key], value) if type(base[key]) is dict else value
    return result


def _operating(value: Any) -> HPOOperatingConfig:
    raw = _merge(asdict(HPOOperatingConfig()), value)
    for item in raw.values():
        integer(item, 1, 2**63 - 1)
    integer(raw["game_concurrency"], 1, 2)
    integer(raw["max_participant_workers"], 1, 4)
    integer(raw["trial_concurrency"], 1, 1)
    integer(raw["resume_latest"], 2, 1024)
    if raw["max_participant_workers"] < 2 * raw["game_concurrency"]:
        raise ContractError("both simultaneous participant roles must be admitted")
    return HPOOperatingConfig(**raw)


def _path(value: Any, origin: Path) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ContractError("nonempty HPO destination path required")
    return str((origin / value).resolve())


def _storage(value: Any, origin: Path) -> HPOStorageConfig:
    raw = _merge(asdict(HPOStorageConfig()), value)
    for name in ("parent_root", "numeric_root", "optuna_path", "windows_c_root"):
        raw[name] = _path(raw[name], origin)
    if raw["originals_root"] is not None:
        raw["originals_root"] = _path(raw["originals_root"], origin)
    for name in ("destination_capacity_bytes", "admission_reservation_bytes", "windows_c_min_free_bytes"):
        integer(raw[name], 1, 2**63 - 1)
    if raw["ext4_control_required"] is not True:
        raise ContractError("HPO parent/numeric/Optuna controls require ext4")
    if raw["admission_reservation_bytes"] > raw["destination_capacity_bytes"]:
        raise ContractError("storage admission reservation exceeds destination capacity")
    return HPOStorageConfig(**raw)


def _seeds(value: Any, *, games: bool) -> tuple[Any, ...]:
    if type(value) not in (list, tuple) or not value:
        raise ContractError("nonempty explicit experiment seeds required")
    result: list[Any] = []
    for seed in value:
        if games:
            if type(seed) is not str:
                raise ContractError("evaluation GameSeed must be a canonical UUID")
            try:
                parsed = uuid.UUID(seed)
            except ValueError as exc:
                raise ContractError("evaluation GameSeed must be a canonical UUID") from exc
            if str(parsed) != seed:
                raise ContractError("evaluation GameSeed must be a canonical UUID")
        else:
            integer(seed, 0, 2**31 - 1)
        if seed in result:
            raise ContractError("duplicate experiment seed")
        result.append(seed)
    return tuple(result)


def campaign_config(value: Json, *, origin: Path) -> LoadedHPOConfig:
    required = {"schema_version", "kind", "base_experiment", "model_id", "stage", "trial_count",
                "decision_budget", "learning_seeds", "evaluation_game_seeds", "sampler_seed",
                "startup_trials", "session_seconds", "run_wall_seconds", "space"}
    closed(value, required, {"model", "training", "inference", "operating", "storage"})
    integer(value["schema_version"], 1, 1)
    if type(value["kind"]) is not str or value["kind"] != "ppo-hpo":
        raise ContractError("explicit ppo-hpo campaign kind required")
    model = model_defaults_v4(value["model_id"])
    if type(value["stage"]) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value["stage"]) is None:
        raise ContractError("bounded explicit campaign stage required")
    integer(value["trial_count"], 1, 2**31 - 1)
    integer(value["decision_budget"], 1, 2**63 - 1)
    integer(value["sampler_seed"], 0, 2**32 - 1)
    integer(value["startup_trials"], 0, 2**31 - 1)
    learning = _seeds(value["learning_seeds"], games=False)
    games = _seeds(value["evaluation_game_seeds"], games=True)
    session = finite_number(value["session_seconds"], 0, 1e12, strict_low=True)
    wall = finite_number(value["run_wall_seconds"], 0, 1e12, strict_low=True)
    model = model_config_v4(_merge(asdict(model), value.get("model", {})), layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if model.model_id != value["model_id"]:
        raise ContractError("campaign/model kind mismatch")
    recipe = normalize_training_for_model(model.model_id,
        hpo_training_config(_merge(asdict(HPOTrainingConfig()), value.get("training", {}))))
    space = search_space(value["space"], model_id=model.model_id, model=model, training=recipe)
    inference = InferenceBindingV2(**_merge(asdict(InferenceBindingV2(device="cuda:0")), value.get("inference", {})))
    if inference.device != "cuda:0":
        raise ContractError("HPO neural inference requires GPU")
    operating = _operating(value.get("operating", {}))
    storage = _storage(value.get("storage", {}), origin)
    base_path = _path(value["base_experiment"], origin)
    # Only after all caller experiment fields pass do we read the legacy base.
    base = load_config(base_path)
    if (type(base.config.actions) is not ActionConfigV3
            or base.config.actions.action_contract != corrected_card_instance_contract()
            or type(base.config.model) is not ConfiguredNeuralModelV3
            or base.config.model.model_id != model.model_id
            or base.config.server.simulator_revision != CORRECTED_SIMULATOR_REVISION
            or base.config.run.split != "train"):
        raise ContractError("HPO requires a matching corrected model3 training experiment base")
    if (base.config.actions.max_sequence_length > model.max_positions
            or base.config.actions.max_candidates > inference.max_candidates):
        raise ContractError("HPO model/actions capacity mismatch")
    config = HPOCampaignConfig(**{name: value[name] for name in (
        "schema_version", "kind", "model_id", "stage", "trial_count", "decision_budget", "sampler_seed", "startup_trials")},
        base_experiment=base_path, base_config_sha256=base.sha256, learning_seeds=learning, evaluation_game_seeds=games,
        session_seconds=session, run_wall_seconds=wall, space=space, model=model, training=recipe,
        inference=inference, operating=operating, storage=storage)
    effective = clone(asdict(config))
    return LoadedHPOConfig(config, base, effective, digest({"campaign": effective, "base": base.sha256}), base.inputs)


def joint_resource_admission(config: HPOCampaignConfig, *, layout: StructuredLayout) -> Json:
    """Worst simultaneous neural roles plus explicit non-model phase budgets.

    Reuses the existing structured endpoint/process envelope; this is a managed
    conservative reservation, not measured RSS or proof of physical availability.
    Later runtime owners must retain the SAME reservation until actual reap.
    """
    if type(config) is not HPOCampaignConfig:
        raise ContractError("closed HPO campaign required")
    limits = _operating(asdict(config.operating))
    recipe = normalize_training_for_model(config.model_id, config.training)
    model = model_config_v4(asdict(config.model), layout=layout)
    learner = resource_estimate(model, config.inference, layout=layout,
                                suffix_decisions=recipe.tbptt_length)["total_bytes"]
    worker = resource_estimate(model, config.inference, layout=layout, suffix_decisions=1)["total_bytes"]
    workers = 2 * limits.game_concurrency
    worker_total = workers * worker
    ram_phases = {
        "collection": limits.parent_budget_bytes + learner + worker_total + limits.collector_budget_bytes,
        "update": limits.parent_budget_bytes + learner + limits.source_budget_bytes + limits.numeric_budget_bytes,
        "save": limits.parent_budget_bytes + learner + limits.save_budget_bytes,
    }
    peak_ram, peak_cuda = max(ram_phases.values()), learner + worker_total
    if peak_ram > limits.managed_ram_bytes:
        raise ContractError("joint HPO RAM demand exceeds managed budget")
    if peak_cuda > limits.managed_cuda_bytes:
        raise ContractError("joint HPO CUDA demand exceeds managed budget")
    return {"recipe": "hpo-joint-process-envelope-v1", "workers": workers,
        "learner_bytes": learner, "worker_bytes": worker_total, "ram_phases": ram_phases,
        "peak_ram_bytes": peak_ram, "peak_cuda_bytes": peak_cuda,
        "managed_ram_bytes": limits.managed_ram_bytes, "managed_cuda_bytes": limits.managed_cuda_bytes}


def load_hpo_config(path: str | Path) -> LoadedHPOConfig:
    source = Path(path).resolve()
    raw = source.read_bytes()
    try:
        value = tomllib.loads(raw.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ContractError("invalid HPO TOML") from exc
    loaded = campaign_config(value, origin=source.parent)
    joint_resource_admission(loaded.config, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    receipt = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "kind": "ppo-hpo"}
    return LoadedHPOConfig(loaded.config, loaded.base, loaded.effective, loaded.sha256, (*loaded.inputs, receipt))
