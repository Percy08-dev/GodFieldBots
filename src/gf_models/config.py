"""Stdlib-only model and fit configuration, with closed versioned recipes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer
from gf_models.structured_binding import ConfiguredNeuralModelV2, configured_model_v2


@dataclass(frozen=True)
class EncoderConfig:
    history: str = "none"
    max_elements: int = 3
    hash_buckets: int = 32
    history_window: int = 128


@dataclass(frozen=True)
class NetworkConfig:
    num_leaves: int = 31


@dataclass(frozen=True)
class InitializationConfig:
    recipe: str = "seed-derived-v1"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "random"
    version: str = "1"
    checkpoint: str | None = None
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "lightgbm_mc"
    train_runs: tuple[str, ...] = ()
    validation_runs: tuple[str, ...] = ()
    pilot_run: str | None = None
    accepted_decision_target: int | None = None
    max_overshoot: float = 0.05
    learning_rate: float = 0.05
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    min_data_in_leaf: int = 20
    num_threads: int = 1
    seeds: tuple[int, ...] = (0, 1, 2)
    wall_seconds: float = 7200
    cleanup_seconds: float = 2
    max_log_bytes: int = 1048576
    cost_mode: str = "inclusive_generation"
    dataset_manifest: str | None = None


def encoder_config(value: Json) -> EncoderConfig:
    closed(value, set(asdict(EncoderConfig())))
    config = EncoderConfig(**value)
    if config.history not in ("none", "public_aggregate"):
        raise ContractError("unsupported encoder history")
    integer(config.max_elements, 2, 3)
    integer(config.hash_buckets, 2, 4096)
    integer(config.history_window, 1, 100000)
    return config


def model_config(value: Json) -> ModelConfig | MLPModelConfig | SequenceModelConfig | ConfiguredNeuralModelV2:
    if value.get("version") == "2":
        return configured_model_v2(value)
    if value.get("model_id") == "transformer":
        return transformer_model_config(value)
    if value.get("model_id") == "gru":
        return gru_model_config(value)
    if value.get("model_id") == "mlp":
        return mlp_model_config(value)
    closed(value, set(asdict(ModelConfig())))
    encoder = encoder_config(value["encoder"])
    closed(value["network"], {"num_leaves"})
    integer(value["network"]["num_leaves"], 2, 65536)
    closed(value["initialization"], {"recipe"})
    if value["initialization"]["recipe"] != "seed-derived-v1":
        raise ContractError("unsupported initialization recipe")
    if value["model_id"] not in ("random", "heuristic", "lightgbm") or value["version"] != "1":
        raise ContractError("unsupported model/version")
    checkpoint = value["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ContractError("invalid checkpoint path")
    if value["model_id"] != "lightgbm" and checkpoint is not None:
        raise ContractError("baseline cannot bind a checkpoint")
    return ModelConfig(value["model_id"], value["version"], checkpoint, encoder,
                       NetworkConfig(**value["network"]), InitializationConfig(**value["initialization"]))


def training_config(value: Json) -> TrainingConfig | PPOTrainingConfig:
    if value.get("algorithm") == "ppo":
        return sequence_ppo_training_config(value) if "sequence_length" in value else ppo_training_config(value)
    closed(value, set(asdict(TrainingConfig())))
    if value["cost_mode"] not in ("inclusive_generation", "fixed_data"):
        raise ContractError("unsupported training cost mode")
    if value["dataset_manifest"] is not None and (not isinstance(value["dataset_manifest"], str) or not value["dataset_manifest"]):
        raise ContractError("invalid dataset manifest")
    if value["algorithm"] != "lightgbm_mc":
        raise ContractError("unsupported training algorithm")
    for key in ("train_runs", "validation_runs"):
        if not isinstance(value[key], (list, tuple)) or any(not isinstance(p, str) or not p for p in value[key]):
            raise ContractError("training source paths required")
    if value["pilot_run"] is not None and (not isinstance(value["pilot_run"], str) or not value["pilot_run"]):
        raise ContractError("invalid pilot path")
    if value["accepted_decision_target"] is not None:
        integer(value["accepted_decision_target"], 1)
    for key in ("max_overshoot", "learning_rate", "wall_seconds", "cleanup_seconds"):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or key != "max_overshoot" and number == 0:
            raise ContractError("invalid training numeric setting")
    if value["max_overshoot"] > 1 or value["learning_rate"] > 1 or value["cleanup_seconds"] >= value["wall_seconds"]:
        raise ContractError("invalid training range/budget")
    for key in ("num_boost_round", "early_stopping_rounds", "min_data_in_leaf", "num_threads", "max_log_bytes"):
        integer(value[key], 1, 2**31 - 1)
    if not isinstance(value["seeds"], (list, tuple)) or not value["seeds"] or len(set(value["seeds"])) != len(value["seeds"]):
        raise ContractError("distinct training seeds required")
    for seed in value["seeds"]:
        integer(seed, 0, 2**31 - 1)
    converted: dict[str, Any] = dict(value)
    for key in ("train_runs", "validation_runs", "seeds"):
        converted[key] = tuple(converted[key])
    return TrainingConfig(**converted)


def validate_capacity(model: ModelConfig | MLPModelConfig | SequenceModelConfig | ConfiguredNeuralModelV2, max_sequence_length: int) -> None:
    if isinstance(model, ConfiguredNeuralModelV2):
        if not 1 <= max_sequence_length <= model.max_positions:
            raise ContractError("generator exceeds structured model position capacity")
        return
    if model.model_id in ("lightgbm", "mlp", "gru", "transformer") and (not max_sequence_length or max_sequence_length > model.encoder.max_elements):
        raise ContractError("generator exceeds encoder element capacity")


@dataclass(frozen=True)
class MLPNetworkConfig:
    hidden_sizes: tuple[int, int] = (128, 128)


@dataclass(frozen=True)
class MLPInitializationConfig:
    recipe: str = "orthogonal-v1"


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "cpu"


@dataclass(frozen=True)
class MLPModelConfig:
    model_id: str = "mlp"
    version: str = "1"
    checkpoint: str | None = None
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(history="public_aggregate"))
    network: MLPNetworkConfig = field(default_factory=MLPNetworkConfig)
    initialization: MLPInitializationConfig = field(default_factory=MLPInitializationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass(frozen=True)
class PPOTrainingConfig:
    algorithm: str = "ppo"
    device: str = "cuda:0"
    seeds: tuple[int, ...] = (17,)
    max_updates: int = 1000
    max_owned_bytes: int = 10737418240
    admission_reservation_bytes: int = 1073741824
    cleanup_seconds: float = 30
    max_log_bytes: int = 1048576


def mlp_model_config(value: Json) -> MLPModelConfig:
    closed(value, set(asdict(MLPModelConfig())))
    encoder = encoder_config(value["encoder"])
    if encoder != EncoderConfig(history="public_aggregate"):
        raise ContractError("MLP v1 requires the pinned public aggregate encoder")
    closed(value["network"], {"hidden_sizes"})
    if value["network"]["hidden_sizes"] not in ([128, 128], (128, 128)):
        raise ContractError("MLP v1 requires two 128-wide layers")
    closed(value["initialization"], {"recipe"})
    closed(value["inference"], {"device"})
    if value["initialization"]["recipe"] != "orthogonal-v1" or value["inference"]["device"] != "cpu":
        raise ContractError("unsupported MLP initialization/inference device")
    if value["model_id"] != "mlp" or value["version"] != "1":
        raise ContractError("unsupported MLP model/version")
    checkpoint = value["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ContractError("invalid MLP checkpoint")
    return MLPModelConfig(checkpoint=checkpoint, encoder=encoder)


def ppo_training_config(value: Json) -> PPOTrainingConfig:
    closed(value, set(asdict(PPOTrainingConfig())))
    if value["algorithm"] != "ppo" or value["device"] not in ("cpu", "cuda:0"):
        raise ContractError("unsupported PPO algorithm/device")
    if not isinstance(value["seeds"], (list, tuple)) or len(value["seeds"]) != 1:
        raise ContractError("PPO requires exactly one learning seed")
    integer(value["seeds"][0], 0, 2**31 - 1)
    integer(value["max_updates"], 1, 1000)
    integer(value["max_owned_bytes"], 1, 10737418240)
    integer(value["admission_reservation_bytes"], 1, 1073741824)
    if value["admission_reservation_bytes"] > value["max_owned_bytes"]:
        raise ContractError("PPO reservation exceeds capacity")
    integer(value["max_log_bytes"], 1, 2**31 - 1)
    if type(value["cleanup_seconds"]) not in (int, float) or not math.isfinite(value["cleanup_seconds"]) or not 0 < value["cleanup_seconds"] <= 60:
        raise ContractError("invalid PPO cleanup budget")
    return PPOTrainingConfig(**{**value, "seeds": tuple(value["seeds"])})


@dataclass(frozen=True)
class GRUNetworkConfig:
    hidden_size: int = 128
    num_layers: int = 1


@dataclass(frozen=True)
class GRUInitializationConfig:
    recipe: str = "orthogonal-gru-v1"


@dataclass(frozen=True)
class SequenceInferenceConfig:
    device: str = "cpu"
    cache_budget_bytes: int = 134217728


@dataclass(frozen=True)
class GRUModelConfig:
    model_id: str = "gru"
    version: str = "1"
    checkpoint: str | None = None
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(history="public_aggregate"))
    network: GRUNetworkConfig = field(default_factory=GRUNetworkConfig)
    initialization: GRUInitializationConfig = field(default_factory=GRUInitializationConfig)
    inference: SequenceInferenceConfig = field(default_factory=SequenceInferenceConfig)


@dataclass(frozen=True)
class SequencePPOTrainingConfig(PPOTrainingConfig):
    sequence_length: int = 32


def gru_model_config(value: Json) -> GRUModelConfig:
    closed(value, set(asdict(GRUModelConfig())))
    encoder = encoder_config(value['encoder'])
    if encoder != EncoderConfig(history='public_aggregate'):
        raise ContractError('GRU v1 requires the pinned public aggregate encoder')
    closed(value['network'], {'hidden_size', 'num_layers'})
    integer(value['network']['hidden_size'], 128, 128)
    integer(value['network']['num_layers'], 1, 1)
    closed(value['initialization'], {'recipe'})
    closed(value['inference'], {'device', 'cache_budget_bytes'})
    integer(value['inference']['cache_budget_bytes'], 1, 2**31 - 1)
    if (value['initialization']['recipe'] != 'orthogonal-gru-v1'
            or value['inference']['device'] != 'cpu' or value['model_id'] != 'gru' or value['version'] != '1'):
        raise ContractError('unsupported GRU recipe/version/device')
    checkpoint = value['checkpoint']
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ContractError('invalid GRU checkpoint')
    return GRUModelConfig(checkpoint=checkpoint, encoder=encoder,
                          inference=SequenceInferenceConfig(**value['inference']))


def sequence_ppo_training_config(value: Json) -> SequencePPOTrainingConfig:
    closed(value, set(asdict(SequencePPOTrainingConfig())))
    integer(value['sequence_length'], 1, 1024)
    base = ppo_training_config({key: item for key, item in value.items() if key != 'sequence_length'})
    return SequencePPOTrainingConfig(**asdict(base), sequence_length=value['sequence_length'])


@dataclass(frozen=True)
class TransformerNetworkConfig:
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_size: int = 256


@dataclass(frozen=True)
class TransformerInitializationConfig:
    recipe: str = 'orthogonal-transformer-v1'


@dataclass(frozen=True)
class TransformerModelConfig:
    model_id: str = 'transformer'
    version: str = '1'
    checkpoint: str | None = None
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(history='public_aggregate'))
    network: TransformerNetworkConfig = field(default_factory=TransformerNetworkConfig)
    initialization: TransformerInitializationConfig = field(default_factory=TransformerInitializationConfig)
    inference: SequenceInferenceConfig = field(default_factory=SequenceInferenceConfig)


def transformer_model_config(value: Json) -> TransformerModelConfig:
    closed(value, set(asdict(TransformerModelConfig())))
    encoder = encoder_config(value['encoder'])
    if encoder != EncoderConfig(history='public_aggregate'):
        raise ContractError('Transformer v1 requires the pinned public aggregate encoder')
    closed(value['network'], set(asdict(TransformerNetworkConfig())))
    for key, expected in asdict(TransformerNetworkConfig()).items():
        integer(value['network'][key], expected, expected)
    closed(value['initialization'], {'recipe'})
    closed(value['inference'], {'device', 'cache_budget_bytes'})
    integer(value['inference']['cache_budget_bytes'], 1, 2**31 - 1)
    if (value['initialization']['recipe'] != 'orthogonal-transformer-v1' or value['inference']['device'] != 'cpu'
            or value['model_id'] != 'transformer' or value['version'] != '1'):
        raise ContractError('unsupported Transformer recipe/version/device')
    checkpoint = value['checkpoint']
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ContractError('invalid Transformer checkpoint')
    return TransformerModelConfig(checkpoint=checkpoint, encoder=encoder,
                                  inference=SequenceInferenceConfig(**value['inference']))


SequenceModelConfig = GRUModelConfig | TransformerModelConfig
