"""Stdlib-only model and fit configuration, with closed versioned recipes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer


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


def encoder_config(value: Json) -> EncoderConfig:
    closed(value, set(asdict(EncoderConfig())))
    config = EncoderConfig(**value)
    if config.history not in ("none", "public_aggregate"):
        raise ContractError("unsupported encoder history")
    integer(config.max_elements, 2, 3)
    integer(config.hash_buckets, 2, 4096)
    integer(config.history_window, 1, 100000)
    return config


def model_config(value: Json) -> ModelConfig:
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


def training_config(value: Json) -> TrainingConfig:
    closed(value, set(asdict(TrainingConfig())))
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


def validate_capacity(model: ModelConfig, max_sequence_length: int) -> None:
    if model.model_id == "lightgbm" and (not max_sequence_length or max_sequence_length > model.encoder.max_elements):
        raise ContractError("generator exceeds encoder element capacity")
