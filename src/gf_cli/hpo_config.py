"""Explicit HPO TOML loading and pure joint operating admission.

This module performs no allocation, Simulator activity, directory creation,
filesystem-capacity validation or optional learning-library imports.
"""
from __future__ import annotations

import hashlib
import re
import tomllib
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from gf_cli.config import (
    GroupPackExperimentConfig,
    GroupPackStorageConfig,
    LoadedConfig,
    RunConfigV2,
    RustPlayerPublicHpoRunConfigV2,
    load_config,
)
from gf_common.card_instance import (
    CORRECTED_SIMULATOR_REVISION,
    SHARED_SIMULATOR_REVISION,
    ActionConfigV3,
    is_corrected_contract,
    shared_card_instance_contract,
)
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.hpo_resident import resident_runtime
from gf_common.rust_player import player_runtime
from gf_common.shared_inference import shared_runtime
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, StructuredLayout
from gf_common.training_opponents import parse_training_opponents
from gf_models.config import ModelConfig
from gf_models.hpo_config import model_config_v4, model_defaults_v4
from gf_models.refresh_heuristic_config import RefreshHeuristicConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3, ConfiguredNeuralModelV4
from gf_models.structured_config import InferenceBindingV2, ModelConfigV4
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
    collector_budget_bytes: int = 2 * GIB
    parent_budget_bytes: int = 256 * 1024**2
    max_updates: int = 1000
    inference_latest: int = 2
    inference_periodic_decisions: int = 10000
    pool_every_decisions: int = 10000
    opponent_pool_size: int = 5
    resume_latest: int = 2


@dataclass(frozen=True, kw_only=True)
class RustPlayerPublicOperatingConfig(HPOOperatingConfig):
    post_host_seconds: float


@dataclass(frozen=True)
class HPOStorageConfig:
    parent_root: str = "runs/hpo"
    numeric_root: str = "runs/hpo"
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


@dataclass(frozen=True, kw_only=True)
class RefreshOpponentsHPOCampaignConfig(HPOCampaignConfig):
    training_opponents: Json

    @property
    def study_identity(self) -> str:
        return digest({'recipe': 'hpo-mixed-refresh-study-v1', 'base': super().study_identity,
            'training_opponents': parse_training_opponents(self.training_opponents)})


@dataclass(frozen=True, kw_only=True)
class ResidentHPOCampaignConfig(HPOCampaignConfig):
    runtime: Json

    @property
    def study_identity(self) -> str:
        return digest({'recipe': 'hpo-resident-study-v1', 'base': super().study_identity,
            'runtime': resident_runtime(self.runtime)})


@dataclass(frozen=True, kw_only=True)
class ResidentRefreshHPOCampaignConfig(ResidentHPOCampaignConfig):
    training_opponents: Json

    @property
    def study_identity(self) -> str:
        return digest({'recipe': 'hpo-mixed-refresh-study-v1', 'base': super().study_identity,
            'training_opponents': parse_training_opponents(self.training_opponents)})


@dataclass(frozen=True, kw_only=True)
class OwnedHPOCampaignConfig(ResidentHPOCampaignConfig):
    origin: str
    encoder: str

    @property
    def study_identity(self) -> str:
        return digest({'recipe':'hpo-owned-study-v1','base':super().study_identity,
            'origin':self.origin,'encoder':self.encoder})


@dataclass(frozen=True, kw_only=True)
class OwnedRefreshHPOCampaignConfig(OwnedHPOCampaignConfig):
    training_opponents: Json

    @property
    def study_identity(self) -> str:
        return digest({'recipe':'hpo-mixed-refresh-study-v1','base':super().study_identity,
            'training_opponents':parse_training_opponents(self.training_opponents)})


@dataclass(frozen=True, kw_only=True)
class SharedHPOCampaignConfig(OwnedHPOCampaignConfig):
    @property
    def study_identity(self) -> str:
        return digest({'recipe':'hpo-shared-study-v1','base':super(ResidentHPOCampaignConfig,self).study_identity,
            'runtime':shared_runtime(self.runtime,game_concurrency=self.operating.game_concurrency),
            'origin':self.origin,'encoder':self.encoder})


@dataclass(frozen=True, kw_only=True)
class SharedRefreshHPOCampaignConfig(SharedHPOCampaignConfig):
    training_opponents: Json

    @property
    def study_identity(self) -> str:
        return digest({'base':super().study_identity,'training_opponents':self.training_opponents})


@dataclass(frozen=True, kw_only=True)
class RustPlayerHPOCampaignConfig(HPOCampaignConfig):
    runtime: Json
    origin: str
    encoder: str

    @property
    def study_identity(self) -> str:
        return digest({'recipe': 'hpo-rust-player-study-v1', 'base': super().study_identity,
            'runtime': player_runtime(self.runtime, game_concurrency=self.operating.game_concurrency),
            'origin': self.origin, 'encoder': self.encoder})


@dataclass(frozen=True, kw_only=True)
class RustPlayerRefreshHPOCampaignConfig(RustPlayerHPOCampaignConfig):
    training_opponents: Json

    @property
    def study_identity(self) -> str:
        return digest({'base': super().study_identity,
            'training_opponents': parse_training_opponents(self.training_opponents)})


@dataclass(frozen=True)
class LoadedHPOConfig:
    config: HPOCampaignConfig
    base: LoadedConfig
    effective: Json
    sha256: str
    inputs: tuple[Json, ...]


@dataclass(frozen=True)
class HPOExperimentConfig(GroupPackExperimentConfig):
    model: ConfiguredNeuralModelV4 = field(default_factory=ConfiguredNeuralModelV4)
    hpo_spec: Json = field(default_factory=dict)


def materialize_hpo_run(loaded_campaign: LoadedHPOConfig, settings: Any, *, learning_seed: int,
                        run_id: str) -> LoadedConfig:
    """Pure checked per-seed composition; filesystem/GPU work is caller-owned."""
    from gf_common.preprocessing import preparation_config
    from gf_models.config import PPOTrainingConfig
    from gf_training.hpo_artifacts import save_estimate
    from gf_training.hpo_config import EffectiveSettings
    if type(loaded_campaign) is not LoadedHPOConfig or type(settings) is not EffectiveSettings:
        raise ContractError('explicit loaded campaign and normalized effective settings required')
    campaign=loaded_campaign.config
    if settings.model.model_id!=campaign.model_id:
        raise ContractError('effective model settings belong to a different HPO model study')
    if learning_seed not in campaign.learning_seeds or type(learning_seed) is not int:
        raise ContractError('run learning seed is outside declared campaign')
    try:
        if str(uuid.UUID(run_id))!=run_id: raise ValueError('noncanonical UUID')
    except (ValueError,TypeError,AttributeError) as error:
        raise ContractError('predeclared canonical run UUID required') from error
    effective=effective_settings(settings.model,settings.training,{},layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if effective.identity!=settings.identity or canonical(asdict(effective.training))!=canonical(asdict(settings.training)):
        raise ContractError('HPO materialization requires canonical effective training')
    operating=_operating(asdict(campaign.operating), rust_player=isinstance(campaign, RustPlayerHPOCampaignConfig))
    storage=campaign.storage
    if storage.numeric_root!=storage.parent_root:
        raise ContractError('HPO currently requires numeric_root==parent_root for colocated per-run numeric storage')
    if save_estimate(effective.model)['additional_bytes']>operating.save_budget_bytes:
        raise ContractError('HPO save envelope insufficient before GPU admission')
    proposal=replace(campaign,model=effective.model,training=effective.training)
    rust_player = isinstance(campaign, RustPlayerHPOCampaignConfig)
    if not rust_player:
        joint_resource_admission(proposal,layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    # Rust configuration grants no host/resource authority. Its actual control
    # owner must admit the one-host lifetimes before any numerical allocation.
    base=loaded_campaign.base.config
    if type(base.model) is not ConfiguredNeuralModelV3 or not isinstance(base.run,RunConfigV2) or type(base.training) is not PPOTrainingConfig:
        raise ContractError('HPO materialization requires the original corrected PPO base')
    pack=base.storage if isinstance(base,GroupPackExperimentConfig) else GroupPackStorageConfig()
    pack=replace(pack,db_root=storage.originals_root or pack.db_root)
    if pack.budget_mib*1024**2>operating.collector_budget_bytes:
        raise ContractError('actual packed collector RAM exceeds declared HPO collector envelope')
    wall=campaign.run_wall_seconds
    final=base.run.finalization_seconds
    if not wall>final>0 or not 0<base.run.admission_seconds<base.run.execution_seconds:
        raise ContractError('HPO base/request deadlines inconsistent')
    if base.training.cleanup_seconds+base.run.report_reserve_seconds>final:
        raise ContractError('HPO original learner cleanup/report reserves exceed finalization')
    execute=wall-final
    base_run = base.run
    if isinstance(campaign, RustPlayerHPOCampaignConfig) and hasattr(base_run, 'player_runtime'):
        if player_runtime(base_run.player_runtime, game_concurrency=base_run.concurrency) != campaign.runtime:
            raise ContractError('HPO6 base player_runtime conflicts with authoritative campaign runtime')
        base_run = RunConfigV2(**{name: getattr(base_run, name) for name in RunConfigV2.__dataclass_fields__})
    run=replace(base_run,name=run_id,output_dir=storage.parent_root,wall_seconds=wall,
        execution_seconds=execute,admission_seconds=execute*base.run.admission_seconds/base.run.execution_seconds,
        concurrency=operating.game_concurrency,global_cuda_budget_bytes=operating.managed_cuda_bytes,
        max_participant_workers=operating.max_participant_workers)
    if isinstance(operating, RustPlayerPublicOperatingConfig):
        from gf_cli.config import _validate_post_host
        run = RustPlayerPublicHpoRunConfigV2(**asdict(run), post_host_seconds=operating.post_host_seconds)
        _validate_post_host(asdict(run))
    source_mib=min(base.preprocessing.memory_budget_mib,operating.source_budget_bytes//1024**2)
    _validate_hpo_source_codec(source_mib)
    preprocessing=preparation_config(asdict(replace(base.preprocessing,memory_budget_mib=source_mib)))
    training=replace(base.training,device='cuda:0',seeds=(learning_seed,),max_updates=operating.max_updates,
        max_owned_bytes=storage.destination_capacity_bytes,admission_reservation_bytes=storage.admission_reservation_bytes)
    if type(training) is not PPOTrainingConfig:
        raise ContractError('ordinary runtime PPO controls required')
    model=ConfiguredNeuralModelV4(**{name:getattr(effective.model,name) for name in effective.model.__dataclass_fields__},
        inference=campaign.inference,cuda_visible_devices=base.model.cuda_visible_devices)
    spec={'schema_version':1,'kind':'ppo-hpo-run','decision_budget':campaign.decision_budget,
        'training_recipe':asdict(effective.training),'effective_settings_sha256':effective.identity,
        'base_config_sha256':loaded_campaign.base.sha256,'campaign_config_sha256':loaded_campaign.sha256,
        'study_identity':campaign.study_identity,'operating':asdict(operating),'storage':asdict(storage),
        'selection_version':'hpo-endpoint-selection-v1','deadline_version':'hpo-deadline-v1',
        'deadline_inputs':{'run_wall_seconds':wall,'base_wall_seconds':base.run.wall_seconds,
            'base_admission_seconds':base.run.admission_seconds,'base_execution_seconds':base.run.execution_seconds,
            'base_finalization_seconds':final}}
    if isinstance(campaign, (RefreshOpponentsHPOCampaignConfig, ResidentRefreshHPOCampaignConfig, OwnedRefreshHPOCampaignConfig,SharedRefreshHPOCampaignConfig,RustPlayerRefreshHPOCampaignConfig)):
        _mixed_collection_games(campaign)
        spec.update(schema_version=2, training_opponents=parse_training_opponents(campaign.training_opponents))
    if isinstance(campaign, ResidentHPOCampaignConfig):
        spec.update(schema_version=3, runtime=(shared_runtime(campaign.runtime,game_concurrency=operating.game_concurrency) if isinstance(campaign,SharedHPOCampaignConfig) else resident_runtime(campaign.runtime)))
    if isinstance(campaign, OwnedHPOCampaignConfig):
        spec.update(schema_version=5 if isinstance(campaign,SharedHPOCampaignConfig) else 4, origin=campaign.origin, encoder=campaign.encoder)
        preprocessing=preparation_config(asdict(replace(preprocessing,backend='python')))
    if isinstance(campaign, RustPlayerHPOCampaignConfig):
        spec.update(schema_version=6, runtime=player_runtime(campaign.runtime, game_concurrency=operating.game_concurrency),
            origin=campaign.origin, encoder=campaign.encoder)
    config=HPOExperimentConfig(run=run,server=base.server,actions=base.actions,model=model,opponent=base.opponent,
        evaluation=base.evaluation,training=training,preprocessing=preprocessing,storage=pack,hpo_spec=spec)
    raw=clone(asdict(config))
    return LoadedConfig(config,raw,digest(raw),loaded_campaign.inputs)


def _merge(base: Json, override: Any) -> Json:
    if type(override) is not dict or set(override) - set(base):
        raise ContractError("unknown HPO config setting")
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(base[key], value) if type(base[key]) is dict else value
    return result


def _validate_hpo_source_codec(source_mib: int) -> None:
    if source_mib > 12 * 1024:
        raise ContractError('HPO current original snapshot codec supports at most12GiB; select a narrower base preprocessing envelope')


def normalized_rust_player_hpo(effective: Json, *, inputs: tuple[Json, ...]) -> LoadedConfig:
    """Reconstruct complete spec6 settings without a synthetic base/campaign."""
    from gf_cli.config import (
        CorrectedServerConfigV2,
        EvaluationConfig,
        _validate_evaluation_controls,
        _validate_run_controls,
        _validate_rust_player_profile,
        _validate_server_controls,
        _validate_table,
    )
    from gf_common.card_instance import action_config_v3
    from gf_common.preprocessing import preparation_config
    from gf_models.config import model_config
    from gf_models.structured_binding import configured_model_v4
    from gf_training.hpo_lineage import _run_projection, _runtime_training, rust_player_hpo_spec

    closed(effective, {"schema_version", "kind", "run", "server", "actions", "model",
        "opponent", "evaluation", "training", "preprocessing", "storage", "hpo_spec"})
    if effective["schema_version"] != 1 or effective["kind"] != "experiment":
        raise ContractError("complete materialized HPO experiment required")
    spec = rust_player_hpo_spec(effective)
    _operating(spec["operating"], rust_player=True)
    storage = closed(spec["storage"], set(HPOStorageConfig.__dataclass_fields__))
    for name in ("parent_root", "numeric_root", "optuna_path", "windows_c_root", "originals_root"):
        if storage[name] is not None and not Path(storage[name]).is_absolute():
            raise ContractError("normalized HPO paths must remain absolute")
    actions = action_config_v3(effective["actions"])
    model = configured_model_v4(effective["model"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if model.model_id != "mlp" or model.checkpoint is not None:
        raise ContractError("Rust player requires fresh configured Model4 MLP")
    training = _runtime_training(effective["training"], spec)
    _run_projection(effective, spec, training)
    for name in ("run", "server", "evaluation", "opponent"):
        _validate_table(name, effective[name], context=effective)
    timed = 'post_host_seconds' in spec['operating']
    run_cls = RustPlayerPublicHpoRunConfigV2 if timed else RunConfigV2
    run = closed(effective["run"], set(run_cls.__dataclass_fields__))
    run = {**run, "seeds": tuple(run["seeds"]), "sampling_seeds": tuple(run["sampling_seeds"])}
    settings = effective_settings(model.architecture(layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        hpo_training_config(spec["training_recipe"]), {},
        layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if settings.identity != spec["effective_settings_sha256"]:
        raise ContractError("normalized HPO effective model/training identity differs")
    opponent = (model_config(effective["opponent"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
        if effective["opponent"].get("version") == "3"
            or effective["opponent"].get("model_id") == "heuristic_refresh"
        else model_config(effective["opponent"]))
    preprocessing = preparation_config(effective["preprocessing"])
    _validate_hpo_source_codec(preprocessing.memory_budget_mib)
    _validate_evaluation_controls(effective["evaluation"])
    _validate_run_controls(effective["run"], hpo=True)
    _validate_server_controls(effective["server"], effective["run"],
        revision=actions.action_contract.simulator_revision)
    _validate_rust_player_profile(run=effective["run"], server=effective["server"],
        actions=actions, model=model, opponent=opponent, training=training,
        preprocessing=preprocessing, hpo=True)
    try:
        if str(uuid.UUID(run["name"])) != run["name"]:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as error:
        raise ContractError("materialized HPO run requires its canonical run UUID") from error
    if (storage["numeric_root"] != storage["parent_root"]
            or run["output_dir"] != storage["parent_root"]
            or storage["originals_root"] is not None
                and effective["storage"]["db_root"] != storage["originals_root"]):
        raise ContractError("materialized HPO storage/run projection differs from its spec")
    if (effective["storage"]["budget_mib"] * 1024**2 > spec["operating"]["collector_budget_bytes"]
            or preprocessing.memory_budget_mib * 1024**2 > spec["operating"]["source_budget_bytes"]):
        raise ContractError("materialized HPO collector/Source projection exceeds its spec")
    from gf_training.hpo_artifacts import save_estimate
    if save_estimate(settings.model)["additional_bytes"] > spec["operating"]["save_budget_bytes"]:
        raise ContractError("materialized HPO save envelope insufficient")
    config = HPOExperimentConfig(run=run_cls(**run),
        server=CorrectedServerConfigV2(**effective["server"]), actions=actions, model=model,
        opponent=opponent,
        evaluation=EvaluationConfig(**effective["evaluation"]), training=training,
        preprocessing=preprocessing,
        storage=GroupPackStorageConfig(**effective["storage"]), hpo_spec=clone(spec))
    return LoadedConfig(config, clone(asdict(config)), digest(asdict(config)), inputs)


def _operating(value: Any, *, rust_player: bool = False) -> HPOOperatingConfig:
    post = 'post_host_seconds' in value if type(value) is dict else False
    if post and rust_player:
        p = finite_number(value['post_host_seconds'], 0, 1e12, strict_low=True)
        value = {key: item for key, item in value.items() if key != 'post_host_seconds'}
    raw = _merge(asdict(HPOOperatingConfig()), value)
    for item in raw.values():
        integer(item, 1, 2**63 - 1)
    integer(raw["game_concurrency"], 1, 2**63 - 1)
    integer(raw["max_participant_workers"], 1, 2**63 - 1)
    integer(raw["trial_concurrency"], 1, 1)
    integer(raw["resume_latest"], 2, 1024)
    if raw["max_participant_workers"] < 2 * raw["game_concurrency"]:
        if rust_player:
            # This capacity belongs only to the unchanged fresh evaluation path.
            raise ContractError("fresh HPO evaluation capacity must cover both participant roles at operating.game_concurrency")
        raise ContractError("both simultaneous participant roles must be admitted")
    return RustPlayerPublicOperatingConfig(**raw, post_host_seconds=p) if post and rust_player else HPOOperatingConfig(**raw)


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


def campaign_config(value: Json, *, origin: Path, _pinned_base: LoadedConfig | None = None) -> LoadedHPOConfig:
    required = {"schema_version", "kind", "base_experiment", "model_id", "stage", "trial_count",
                "decision_budget", "learning_seeds", "evaluation_game_seeds", "sampler_seed",
                "startup_trials", "session_seconds", "run_wall_seconds", "space"}
    rust_player = type(value.get("schema_version")) is int and value["schema_version"] == 6
    owned = type(value.get("schema_version")) is int and value["schema_version"] in (4,5,6)
    resident = type(value.get("schema_version")) is int and value["schema_version"] in (3,4,5)
    mixed = value.get("schema_version") == 2 or (resident or rust_player) and "training_opponents" in value
    closed(value, required | ({"training_opponents"} if mixed else set()) | ({"runtime"} if resident or rust_player else set()) | ({"origin","encoder"} if owned else set()),
        {"model", "training", "inference", "operating", "storage"})
    integer(value["schema_version"], 1, 6)
    if owned and (value["origin"] != "producer-owned-v1" or value["encoder"] != "canonical-python"):
        raise ContractError("owned HPO requires explicit producer origin and canonical Python encoder")
    shared = value["schema_version"] == 5
    if rust_player and value['model_id'] != 'mlp':
        raise ContractError('Rust player HPO6 supports Model4 MLP only')
    runtime: Json | None
    if rust_player:
        operating = _operating(value.get('operating', {}), rust_player=True)
        runtime = player_runtime(value['runtime'], game_concurrency=operating.game_concurrency)
    else:
        runtime = (shared_runtime(value["runtime"],game_concurrency=value.get("operating",{}).get("game_concurrency",2)) if shared else resident_runtime(value["runtime"])) if resident else None
    descriptor = parse_training_opponents(value["training_opponents"]) if mixed else None
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
    if not rust_player:
        operating = _operating(value.get("operating", {}))
    storage = _storage(value.get("storage", {}), origin)
    base_path = _path(value["base_experiment"], origin)
    # Only after all caller experiment fields pass do we read the legacy base.
    if _pinned_base is None:
        base = load_config(base_path)
    else:
        if (not rust_player or type(_pinned_base) is not LoadedConfig
                or canonical(asdict(_pinned_base.config)) != canonical(_pinned_base.effective)
                or digest(_pinned_base.effective) != _pinned_base.sha256):
            raise ContractError('explicit unchanged pinned native campaign base required')
        base = _pinned_base
    if hasattr(base.config.run,'inference_service'):
        raise ContractError('HPO base must not declare ordinary run.inference_service; campaign runtime is authoritative')
    if hasattr(base.config.run, 'player_runtime'):
        if not rust_player:
            raise ContractError('legacy HPO base must not declare run.player_runtime')
        if base.config.run.player_runtime != runtime:
            raise ContractError('HPO6 base player_runtime conflicts with authoritative campaign runtime')
        if (hasattr(base.config.run, 'post_host_seconds')
                and base.config.run.post_host_seconds != getattr(operating, 'post_host_seconds', None)):
            raise ContractError('HPO6 base post_host_seconds conflicts with authoritative campaign operating value')
    if (type(base.config.actions) is not ActionConfigV3
            or not is_corrected_contract(base.config.actions.action_contract)
            or type(base.config.model) is not ConfiguredNeuralModelV3
            or base.config.model.model_id != model.model_id
            or base.config.server.simulator_revision not in (CORRECTED_SIMULATOR_REVISION,SHARED_SIMULATOR_REVISION)
            or base.config.run.split != "train"):
        raise ContractError("HPO requires a matching corrected model3 training experiment base")
    if (shared or rust_player) and (base.config.actions.action_contract != shared_card_instance_contract() or model.model_id != "mlp" or base.config.preprocessing.backend != "python"):
        raise ContractError("shared HPO requires new-profile Python MLP base")
    if rust_player:
        opponent = base.config.opponent
        if not ((type(opponent) is ConfiguredNeuralModelV3 and opponent.model_id == 'mlp')
                or (type(opponent) is ModelConfig and opponent.model_id in ('random', 'heuristic'))
                or type(opponent) is RefreshHeuristicConfig):
            raise ContractError('Rust player HPO6 base opponent must be Model3 MLP, Random, Heuristic or Refresh')
    if (base.config.actions.max_sequence_length > model.max_positions
            or base.config.actions.max_candidates > inference.max_candidates):
        raise ContractError("HPO model/actions capacity mismatch")
    cls: Any = ((RustPlayerRefreshHPOCampaignConfig if mixed else RustPlayerHPOCampaignConfig) if rust_player else (SharedRefreshHPOCampaignConfig if mixed else SharedHPOCampaignConfig) if shared else (OwnedRefreshHPOCampaignConfig if mixed else OwnedHPOCampaignConfig) if owned else
        (ResidentRefreshHPOCampaignConfig if mixed else ResidentHPOCampaignConfig) if resident else
        (RefreshOpponentsHPOCampaignConfig if mixed else HPOCampaignConfig))
    config: HPOCampaignConfig = cls(**{name: value[name] for name in (
        "schema_version", "kind", "model_id", "stage", "trial_count", "decision_budget", "sampler_seed", "startup_trials")},
        base_experiment=base_path, base_config_sha256=base.sha256, learning_seeds=learning, evaluation_game_seeds=games,
        session_seconds=session, run_wall_seconds=wall, space=space, model=model, training=recipe,
        inference=inference, operating=operating, storage=storage,
        **({'training_opponents': descriptor} if mixed else {}), **({'runtime': runtime} if resident or rust_player else {}),
        **({'origin':value['origin'],'encoder':value['encoder']} if owned else {}))
    if mixed:
        _mixed_collection_games(config)
    if hasattr(base.config, 'training_opponents'):
        raise ContractError('HPO requires an ordinary base without training_opponents; declare it in campaign2')
    effective = clone(asdict(config))
    return LoadedHPOConfig(config, base, effective, digest({"campaign": effective, "base": base.sha256}), base.inputs)


def joint_resource_admission(config: HPOCampaignConfig, *, layout: StructuredLayout) -> Json:
    """Worst simultaneous neural roles plus explicit non-model phase budgets.

    Reuses the existing structured endpoint/process envelope; this is a managed
    conservative reservation, not measured RSS or proof of physical availability.
    Later runtime owners must retain the SAME reservation until actual reap.
    """
    if isinstance(config, RustPlayerHPOCampaignConfig):
        raise ContractError('Rust player training admission requires authenticated HostControl; legacy process admission is unavailable')
    if type(config) not in (HPOCampaignConfig, RefreshOpponentsHPOCampaignConfig, ResidentHPOCampaignConfig, ResidentRefreshHPOCampaignConfig, OwnedHPOCampaignConfig, OwnedRefreshHPOCampaignConfig, SharedHPOCampaignConfig, SharedRefreshHPOCampaignConfig):
        raise ContractError("closed HPO campaign required")
    expected_schema = 5 if isinstance(config, SharedHPOCampaignConfig) else 4 if isinstance(config, OwnedHPOCampaignConfig) else 3 if isinstance(config, ResidentHPOCampaignConfig) else 2 if isinstance(config, RefreshOpponentsHPOCampaignConfig) else 1
    integer(config.schema_version, expected_schema, expected_schema)
    if isinstance(config, (RefreshOpponentsHPOCampaignConfig, ResidentRefreshHPOCampaignConfig, OwnedRefreshHPOCampaignConfig)):
        parse_training_opponents(config.training_opponents)
        _mixed_collection_games(config)
    limits = _operating(asdict(config.operating))
    recipe = normalize_training_for_model(config.model_id, config.training)
    model = model_config_v4(asdict(config.model), layout=layout)
    from gf_models.structured_runtime import hpo_resident_training_admission, hpo_training_admission
    if isinstance(config,SharedHPOCampaignConfig):
        from gf_models.shared_admission import shared_training_admission
        return shared_training_admission(model,config.inference,recipe.tbptt_length,asdict(limits),config.runtime,layout=layout)
    if isinstance(config, ResidentHPOCampaignConfig):
        return hpo_resident_training_admission(model,config.inference,recipe.tbptt_length,asdict(limits),
            resident_runtime(config.runtime),layout=layout,owned_cuda=config.schema_version==4)
    return hpo_training_admission(model,config.inference,recipe.tbptt_length,asdict(limits),layout=layout)


def load_hpo_config(path: str | Path) -> LoadedHPOConfig:
    source = Path(path).resolve()
    raw = source.read_bytes()
    try:
        value = tomllib.loads(raw.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ContractError("invalid HPO TOML") from exc
    loaded = campaign_config(value, origin=source.parent)
    if not isinstance(loaded.config, RustPlayerHPOCampaignConfig):
        joint_resource_admission(loaded.config, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    receipt = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "kind": "ppo-hpo"}
    return LoadedHPOConfig(loaded.config, loaded.base, loaded.effective, loaded.sha256, (*loaded.inputs, receipt))


def _mixed_collection_games(config: HPOCampaignConfig) -> None:
    if config.training.collection_games != 16:
        raise ContractError('mixed-refresh-16-v1 requires collection_games=16')
    for item in config.space.parameters:
        if item.path == 'training.collection_games':
            values = item.choices if item.kind == 'categorical' else (item.value,) if item.kind == 'fixed' else (item.low, item.high)
            if any(type(value) is not int or value != 16 for value in values):
                raise ContractError('mixed-refresh-16-v1 requires every collection_games search value=16')
    for coupled in config.space.coupled:
        if 'training.collection_games' in coupled.paths:
            index = coupled.paths.index('training.collection_games')
            if any(type(row[index]) is not int or row[index] != 16 for row in coupled.choices):
                raise ContractError('mixed-refresh-16-v1 requires every coupled collection_games=16')


def require_hpo_runtime_execution(config: HPOCampaignConfig | HPOExperimentConfig) -> None:
    """Validate the explicit HPO execution runtime before any external work."""
    if isinstance(config, RustPlayerHPOCampaignConfig) or isinstance(config, HPOExperimentConfig) and config.hpo_spec.get('schema_version') == 6:
        raise ContractError('Rust player selection requires an authenticated Rust-supervised entry')
    from gf_common.hpo_resident import require_hpo_runtime_execution as require_execution
    require_execution(asdict(config) if isinstance(config, HPOCampaignConfig) else config.hpo_spec)
