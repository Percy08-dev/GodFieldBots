"""Native numeric regression and closed inference artifacts; no training sources."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.actions import GENERATOR_VERSION, SCORER_VERSION
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    ContractError,
    Json,
    canonical,
    closed,
    digest,
    integer,
)
from gf_models.lightgbm.config import ModelConfig, TrainingConfig, encoder_config, model_config
from gf_models.lightgbm.encoder import ENCODER_VERSION, Array, feature_names, signature

ADAPTER_VERSION = "native-lightgbm-q-v1"
NATIVE_VERSION = "4.7.0"
NUMPY_VERSION = "2.5.4"
REWARD_VERSION = "natural-terminal-v1"


def runtime_metadata() -> Json:
    """Check installed identities without importing or executing native code."""
    versions: Json = {"lightgbm": importlib.metadata.version("lightgbm"), "numpy": np.__version__,
                "python": platform.python_version_tuple()[0:2], "gil_enabled": sys._is_gil_enabled()}
    if versions["lightgbm"] != NATIVE_VERSION or versions["numpy"] != NUMPY_VERSION or tuple(versions["python"]) != ("3", "14") or not versions["gil_enabled"]:
        raise ContractError("unverified native runtime")
    return versions


def runtime() -> tuple[Any, Json]:
    versions = runtime_metadata()
    lgb = importlib.import_module("lightgbm")
    if lgb.__version__ != versions["lightgbm"]:
        raise ContractError("native module/distribution identity mismatch")
    return lgb, versions


def native_parameters(model: ModelConfig, training: TrainingConfig, seed: int) -> Json:
    integer(seed, 0, 2**31 - 1)
    derived = int(digest({"recipe": model.initialization.recipe, "model_seed": seed})[:8], 16) % (2**31 - 1)
    return {"objective": "regression", "metric": "l2", "learning_rate": training.learning_rate,
            "num_leaves": model.network.num_leaves, "min_data_in_leaf": training.min_data_in_leaf,
            "num_threads": training.num_threads, "device_type": "cpu", "verbosity": -1,
            "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
            "seed": derived, "data_random_seed": derived, "feature_fraction_seed": derived, "bagging_seed": derived}


def validate_parameters(params: Json) -> None:
    expected = native_parameters(ModelConfig(model_id="lightgbm"), TrainingConfig(), 0)
    closed(params, set(expected))
    for key in ("objective", "metric", "device_type", "verbosity", "deterministic", "force_col_wise", "feature_pre_filter"):
        if type(params[key]) is not type(expected[key]) or params[key] != expected[key]:
            raise ContractError("unsupported native parameter")
    for key in ("num_leaves", "min_data_in_leaf", "num_threads"):
        integer(params[key], 1, 2**31 - 1)
    for key in ("seed", "data_random_seed", "feature_fraction_seed", "bagging_seed"):
        integer(params[key], 0, 2**31 - 1)
    if type(params["learning_rate"]) not in (int, float) or not math.isfinite(params["learning_rate"]) or not 0 < params["learning_rate"] <= 1:
        raise ContractError("invalid native learning rate")


def numeric_matrix(value: Any, width: int | None = None) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or width is not None and result.shape[1] != width or not np.isfinite(result).all():
        raise ContractError("finite nonempty exact-width matrix required")
    return result


def fit_numeric(x: Array, y: Array, validation_x: Array, validation_y: Array, *, params: Json,
                rounds: int, early_stopping: int, names: tuple[str, ...]) -> tuple[Any, Json]:
    validate_parameters(params)
    integer(rounds, 1)
    integer(early_stopping, 1)
    x = numeric_matrix(x, len(names))
    validation_x = numeric_matrix(validation_x, len(names))
    for features, labels in ((x, y), (validation_x, validation_y)):
        if labels.shape != (len(features),) or not np.isfinite(labels).all():
            raise ContractError("finite exact-length labels required")
    lgb, _ = runtime()
    train = lgb.Dataset(x, label=y, feature_name=list(names), free_raw_data=True)
    validation = lgb.Dataset(validation_x, label=validation_y, reference=train, feature_name=list(names), free_raw_data=True)
    booster = lgb.train(params, train, num_boost_round=rounds, valid_sets=[validation], valid_names=["validation"],
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
    scores = np.asarray(booster.predict(validation_x), dtype=np.float64)
    if scores.shape != validation_y.shape or not np.isfinite(scores).all():
        raise ContractError("invalid native fitted scores")
    metrics = {"validation_l2": float(np.mean((scores - validation_y) ** 2)), "best_iteration": booster.best_iteration}
    if not math.isfinite(metrics["validation_l2"]):
        raise ContractError("nonfinite fit metrics")
    return booster, metrics


def save_artifact(booster: Any, directory: Path, model: ModelConfig, action_config_hash: str, *,
                  purpose: str, ready: bool = False) -> Json:
    if purpose not in ("toy", "trained") or type(ready) is not bool:
        raise ContractError("invalid artifact purpose/readiness")
    _, versions = runtime()
    directory.mkdir(parents=True, exist_ok=False)
    booster.save_model(str(directory / "model.txt"))
    manifest = {"schema_version": 1, "adapter_version": ADAPTER_VERSION, "model_id": "lightgbm", "model_version": "1",
                "model_sha256": hashlib.sha256((directory / "model.txt").read_bytes()).hexdigest(),
                "encoder_version": ENCODER_VERSION, "encoder": asdict(model.encoder), "encoder_signature": signature(model.encoder),
                "feature_names": feature_names(model.encoder), "network": asdict(model.network),
                "initialization": asdict(model.initialization), "runtime": versions,
                "policy_input_version": POLICY_INPUT_VERSION, "catalog": CATALOG_META,
                "reward_version": REWARD_VERSION, "end_projection_version": END_PROJECTION_VERSION,
                "generator_version": GENERATOR_VERSION, "scorer_version": SCORER_VERSION,
                "action_config_hash": action_config_hash, "purpose": purpose, "ready": ready}
    (directory / "inference.json").write_text(canonical(manifest) + "\n", encoding="utf-8")
    return manifest


def _artifact_hash(path: Path, deadline: float) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if time.monotonic() >= deadline:
                raise TimeoutError("artifact inspection deadline")
            value.update(chunk)
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact inspection deadline")
    return value.hexdigest()


def inspect_artifact(directory: Path, *, expected_digest: str | None = None, action_config_hash: str | None = None,
                     model: ModelConfig | None = None, allow_toy: bool = False, allow_unready: bool = False,
                     deadline: float = float("inf")) -> Json:
    try:
        encoded = (directory / "inference.json").read_bytes()
        if expected_digest is not None and hashlib.sha256(encoded).hexdigest() != expected_digest:
            raise ContractError("inference manifest digest mismatch")
        manifest = json.loads(encoded)
        required = {"schema_version", "adapter_version", "model_id", "model_version", "model_sha256", "encoder_version",
                    "encoder", "encoder_signature", "feature_names", "network", "initialization", "runtime", "policy_input_version",
                    "catalog", "reward_version", "end_projection_version", "generator_version", "scorer_version",
                    "action_config_hash", "purpose", "ready"}
        closed(manifest, required)
        constants = {"schema_version": 1, "adapter_version": ADAPTER_VERSION, "model_id": "lightgbm", "model_version": "1",
                     "encoder_version": ENCODER_VERSION, "policy_input_version": POLICY_INPUT_VERSION, "catalog": CATALOG_META,
                     "reward_version": REWARD_VERSION, "end_projection_version": END_PROJECTION_VERSION,
                     "generator_version": GENERATOR_VERSION, "scorer_version": SCORER_VERSION}
        if any(type(manifest[k]) is not type(v) or manifest[k] != v for k, v in constants.items()):
            raise ContractError("unsupported inference semantics")
        config = encoder_config(manifest["encoder"])
        model_config({"model_id": "lightgbm", "version": "1", "checkpoint": None, "encoder": manifest["encoder"],
                      "network": manifest["network"], "initialization": manifest["initialization"]})
        if manifest["encoder_signature"] != signature(config) or manifest["feature_names"] != list(feature_names(config)):
            raise ContractError("encoder signature mismatch")
        if model is not None and any(manifest[key] != asdict(getattr(model, key)) for key in ("encoder", "network", "initialization")):
            raise ContractError("checkpoint/model configuration mismatch")
        if not isinstance(manifest["action_config_hash"], str) or len(manifest["action_config_hash"]) != 64:
            raise ContractError("invalid artifact action hash")
        if action_config_hash is not None and manifest["action_config_hash"] != action_config_hash:
            raise ContractError("checkpoint/action configuration mismatch")
        if type(manifest["ready"]) is not bool or not manifest["ready"] and not allow_unready:
            raise ContractError("checkpoint is unready")
        if manifest["purpose"] not in ("toy", "trained") or manifest["purpose"] != "trained" and not allow_toy:
            raise ContractError("formal inference rejects toy checkpoint")
        versions = runtime_metadata()
        if canonical(manifest["runtime"]) != canonical(versions):
            raise ContractError("checkpoint runtime mismatch")
        if _artifact_hash(directory / "model.txt", deadline) != manifest["model_sha256"]:
            raise ContractError("native model digest mismatch")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid inference artifact") from exc


def load_artifact(directory: Path, **options: Any) -> tuple[Any, Json]:
    manifest = inspect_artifact(directory, **options)
    lgb, _ = runtime()
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    if booster.num_feature() != len(manifest["feature_names"]) or booster.feature_name() != manifest["feature_names"]:
        raise ContractError("native model feature signature mismatch")
    return booster, manifest
