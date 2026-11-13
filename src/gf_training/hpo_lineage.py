"""Exact HPO structural records; no original Source or experiment authority.

The caller's complete-original Source must independently validate Collection6,
selection and context costs. Even a complete structural chain is not a qualified
training run, a live projection, a formal parent or an evaluation receipt.
"""
from __future__ import annotations

import os
import re
import stat
import uuid
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES, ActionConfigV3, action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.group_pack_game import checked_storage_binding
from gf_common.preprocessing import preparation_config
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.validation import result as validate_result
from gf_models.config import PPOTrainingConfig
from gf_models.structured_artifacts import adam_recipe
from gf_models.structured_binding import (
    child_cuda_visibility,
    configured_model_v4,
    inference_model,
    input_binding_game_v3,
    policy_label_game_v4,
    safe_options_game_v4,
)
from gf_models.structured_config import InferenceBindingV2, ModelConfigV4
from gf_models.structured_sealed import (
    SEMANTIC_KEYS_V3,
    basename,
    checksum,
    inspect_retained_game_artifact_v4,
    inspect_retained_game_inference_v4,
    pruning_basename,
)
from gf_training.hpo_config import effective_settings, finite_number, hpo_training_config
from gf_training.lineage import (
    RunOwnership,
    checkpoint_path,
    generation_path,
    require_hpo_run_owner,
)
from gf_training.structured_lineage import (
    GAME_EXTRA,
    GEN_FIELDS,
    RUN_OPTIONAL,
    RUN_REQUIRED,
    UPDATE_FIELDS,
    _artifact_directories,
    _ChainScan,
    _filemap,
    _stat_identity,
    _write,
)
from gf_training.structured_lineage import (
    _relative as _owned_relative,
)

SELECTION_VERSION = "hpo-endpoint-selection-v1"
HPO_NATIVE_CAPABILITY = "model4;collection6;proof3;numeric5;selection1;provider1"
REFERENCE_DIRECTORY = 'evaluation-reference-receipts'
REFERENCE_FILES = {'acquire.intent.json', 'acquire.receipt.json', 'release.intent.json', 'release.receipt.json'}
HPO_RUN_FIELDS = RUN_REQUIRED | GAME_EXTRA | {
    "run_variant", "hpo_spec_sha256", "collected_decisions", "used_decisions", "stop_reason"}
_OPERATING = {"managed_ram_bytes", "managed_cuda_bytes", "game_concurrency", "max_participant_workers",
    "trial_concurrency", "source_budget_bytes", "numeric_budget_bytes", "save_budget_bytes",
    "collector_budget_bytes", "parent_budget_bytes", "max_updates", "inference_latest",
    "inference_periodic_decisions", "pool_every_decisions", "opponent_pool_size", "resume_latest"}
_STORAGE = {"parent_root", "numeric_root", "optuna_path", "originals_root", "windows_c_root",
    "destination_capacity_bytes", "admission_reservation_bytes", "windows_c_min_free_bytes", "ext4_control_required"}
_RUN = {"name", "split", "output_dir", "seeds", "sampling_seeds", "pairs_per_seed", "concurrency",
    "wall_seconds", "admission_seconds", "execution_seconds", "finalization_seconds",
    "report_reserve_seconds", "max_decisions", "global_cuda_budget_bytes", "max_participant_workers"}
_CLASSIFICATION = {"action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature"}
HPO_GENERATION_FIELDS = GEN_FIELDS | GAME_EXTRA | {"hpo_spec_sha256", "collected_decisions", "used_decisions"}
HPO_UPDATE_FIELDS = UPDATE_FIELDS | GAME_EXTRA | {"hpo_spec_sha256", "collected_decisions",
    "used_decisions", "overshoot_decisions", "expected_optimizer_step_delta"}
_SEMANTICS = SEMANTIC_KEYS_V3 - {"reward_policy_version"}


def _number(value: Any, *, positive: bool = False) -> float:
    return finite_number(value, 0, 1e20, strict_low=positive)


def _same(left: Any, right: Any, message: str) -> None:
    if canonical(left) != canonical(right):
        raise ContractError(message)


def _path(value: Any) -> None:
    if type(value) is not str or not value or "\x00" in value or not Path(value).is_absolute():
        raise ContractError("HPO persisted destination requires an absolute path")


def hpo_spec(state: Json) -> Json:
    """Parse the sole immutable spec; this grants no live or original authority."""
    config = state.get("effective_config")
    if type(config) is not dict:
        raise ContractError("HPO effective config required")
    spec = closed(config.get("hpo_spec"), {"schema_version", "kind", "decision_budget", "training_recipe",
        "effective_settings_sha256", "base_config_sha256", "campaign_config_sha256", "study_identity",
        "operating", "storage", "selection_version", "deadline_version", "deadline_inputs"})
    integer(spec["schema_version"], 1, 1)
    if (spec["kind"] != "ppo-hpo-run" or spec["selection_version"] != SELECTION_VERSION
            or spec["deadline_version"] != "hpo-deadline-v1"):
        raise ContractError("unsupported explicit HPO spec version")
    integer(spec["decision_budget"], 1, 2**63 - 1)
    hpo_training_config(spec["training_recipe"])
    for name in ("effective_settings_sha256", "base_config_sha256", "campaign_config_sha256", "study_identity"):
        checksum(spec[name])
    operating = closed(spec["operating"], _OPERATING)
    for value in operating.values():
        integer(value, 1, 2**63 - 1)
    integer(operating["game_concurrency"], 1, 2)
    integer(operating["max_participant_workers"], 1, 4)
    integer(operating["trial_concurrency"], 1, 1)
    integer(operating["resume_latest"], 2, 1024)
    if operating["max_participant_workers"] < 2 * operating["game_concurrency"]:
        raise ContractError("HPO simultaneous player roles exceed worker projection")
    storage = closed(spec["storage"], _STORAGE)
    for name in ("parent_root", "numeric_root", "optuna_path", "windows_c_root"):
        _path(storage[name])
    if storage["originals_root"] is not None:
        _path(storage["originals_root"])
    for name in ("destination_capacity_bytes", "admission_reservation_bytes", "windows_c_min_free_bytes"):
        integer(storage[name], 1, 2**63 - 1)
    if (storage["ext4_control_required"] is not True
            or storage["admission_reservation_bytes"] > storage["destination_capacity_bytes"]):
        raise ContractError("HPO storage projection inconsistent")
    timers = closed(spec["deadline_inputs"], {"run_wall_seconds", "base_wall_seconds", "base_admission_seconds",
        "base_execution_seconds", "base_finalization_seconds"})
    for value in timers.values():
        _number(value, positive=True)
    if (not timers["base_admission_seconds"] < timers["base_execution_seconds"] < timers["base_wall_seconds"]
            or timers["run_wall_seconds"] <= timers["base_finalization_seconds"]
            or timers["base_execution_seconds"] + timers["base_finalization_seconds"] > timers["base_wall_seconds"] + .001):
        raise ContractError("HPO original base deadline inputs inconsistent")
    if checksum(state.get("hpo_spec_sha256")) != digest(spec):
        raise ContractError("HPO immutable spec hash mismatch")
    return spec


def _runtime_training(value: Json, spec: Json) -> PPOTrainingConfig:
    closed(value, set(PPOTrainingConfig.__dataclass_fields__))
    if value["algorithm"] != "ppo" or value["device"] not in ("cpu", "cuda:0"):
        raise ContractError("HPO explicit runtime PPO device required")
    if type(value["seeds"]) not in (tuple, list) or len(value["seeds"]) != 1:
        raise ContractError("HPO runtime requires one learning seed")
    integer(value["seeds"][0], 0, 2**31 - 1)
    for name in ("max_updates", "max_owned_bytes", "admission_reservation_bytes"):
        integer(value[name], 1, 2**63 - 1)
    integer(value["max_log_bytes"], 1, 2**31 - 1)
    finite_number(value["cleanup_seconds"], 0, 60, strict_low=True)
    if (value["max_updates"] != spec["operating"]["max_updates"]
            or value["max_owned_bytes"] != spec["storage"]["destination_capacity_bytes"]
            or value["admission_reservation_bytes"] != spec["storage"]["admission_reservation_bytes"]):
        raise ContractError("HPO runtime training projection differs from sole spec")
    return PPOTrainingConfig(**{**value, "seeds": tuple(value["seeds"])})


def _run_projection(config: Json, spec: Json, training: PPOTrainingConfig) -> tuple[float,float,float]:
    run = closed(config["run"], _RUN)
    if run["split"] != "train":
        raise ContractError("HPO requires training split")
    operating = spec['operating']
    for field, source in (("concurrency", "game_concurrency"), ("global_cuda_budget_bytes", "managed_cuda_bytes"),
            ("max_participant_workers", "max_participant_workers")):
        integer(run[field], 1, 2**63 - 1)
        if run[field] != operating[source]:
            raise ContractError("HPO run concurrency/resource projection differs from spec")
    inputs = spec["deadline_inputs"]
    wall = inputs["run_wall_seconds"]
    final = inputs["base_finalization_seconds"]
    execute = wall - final
    admission = execute * (inputs["base_admission_seconds"] / inputs["base_execution_seconds"])
    expected = {"wall_seconds": wall, "execution_seconds": execute,
        "admission_seconds": admission, "finalization_seconds": final}
    for key, value in expected.items():
        if abs(_number(run[key], positive=True) - value) > .001:
            raise ContractError("HPO runtime deadline projection mismatch")
    report = _number(run["report_reserve_seconds"], positive=True)
    if training.cleanup_seconds + report > final:
        raise ContractError("HPO original cleanup/report reserves exceed finalization")
    return float(wall),float(execute),float(admission)


def _deadlines(state: Json, spec: Json, training: PPOTrainingConfig) -> None:
    wall,execute,admission=_run_projection(state['effective_config'],spec,training)
    created = _number(state["created_unix"], positive=True)
    last = _number(state["last_unix"], positive=True)
    if not created <= last <= created + wall:
        raise ContractError("HPO original clock provenance mismatch")
    for field, duration in (("wall_deadline_unix", wall), ("execution_deadline_unix", execute),
            ("admission_deadline_unix", admission)):
        if abs((_number(state[field], positive=True) - created) - duration) > .001:
            raise ContractError("HPO original absolute deadline was rewritten")


def hpo_state(state: Json) -> tuple[ActionConfigV3, ModelConfigV4, PPOTrainingConfig]:
    """Validate exact healthy Run5 metadata, including honest CPU component data.

    Production CUDA admission/cleanup and original evidence remain caller-owned.
    """
    packed = "storage_binding" in state
    closed(state, HPO_RUN_FIELDS | ({"storage_binding"} if packed else set()), RUN_OPTIONAL)
    integer(state["schema_version"], 5, 5)
    if (state["algorithm"] != "ppo" or state["run_variant"] != "hpo-ppo-v1"
            or state["model_version"] != "4" or state["input_recipe"] != LAYOUT.input_recipe
            or state["status"] not in ("running", "paused", "incomplete", "complete")):
        raise ContractError("healthy exact HPO Run5 required")
    if any(state.get(key) for key in ("failure", "quarantine", "cleanup_failure", "budget_overrun_seconds")):
        raise ContractError("HPO parent contains failure evidence")
    if type(state["run_id"]) is not str or not state["run_id"] or type(state["config_inputs"]) not in (list, tuple):
        raise ContractError("HPO run/config input identity invalid")
    config = closed(state["effective_config"], {"schema_version", "kind", "run", "server", "actions", "model",
        "opponent", "evaluation", "training", "preprocessing", "hpo_spec"}, {"storage"})
    integer(config["schema_version"], 1, 1)
    if config["kind"] != "experiment" or checksum(state["config_sha256"]) != digest(config):
        raise ContractError("HPO complete effective Config hash mismatch")
    if packed != ("storage" in config):
        raise ContractError("HPO packed storage must be an explicit closed variant")
    if packed:
        checked_storage_binding(state["storage_binding"])
        storage = closed(config["storage"], {"format", "db_root", "ram_root", "budget_mib", "queue_mib", "max_pack_mib"})
        from gf_common.group_pack import MIB, StorageSettings
        _path(storage["db_root"])
        _path(storage["ram_root"])
        if storage["format"] != "sqlite-group-pack-v1":
            raise ContractError("HPO original packed storage format mismatch")
        for key in ("budget_mib", "queue_mib", "max_pack_mib"):
            integer(storage[key], 1, 2**63 - 1)
        StorageSettings(Path(storage["db_root"]), Path(storage["ram_root"]), storage["budget_mib"] * MIB,
            storage["queue_mib"] * MIB, storage["max_pack_mib"] * MIB)
    spec = hpo_spec(state)
    originals_root = spec["storage"]["originals_root"]
    if originals_root is not None and (not packed or config["storage"]["db_root"] != originals_root):
        raise ContractError("HPO original storage projection differs from sole spec")
    configured = configured_model_v4(config["model"], layout=LAYOUT)
    if configured.checkpoint is not None or configured.model_id != state["model_id"]:
        raise ContractError("HPO run requires matching fresh configured Model4")
    model = configured.architecture(layout=LAYOUT)
    training = _runtime_training(config["training"], spec)
    recipe = hpo_training_config(spec["training_recipe"])
    if effective_settings(model, recipe, {}, layout=LAYOUT).identity != spec["effective_settings_sha256"]:
        raise ContractError("HPO effective settings identity mismatch")
    actions = action_config_v3(config["actions"])
    if actions.max_sequence_length > model.max_positions or actions.max_candidates > configured.inference.max_candidates:
        raise ContractError("HPO model/action/inference capacity mismatch")
    classification = input_binding_game_v3(actions.action_contract, card_instance_rules())
    for key, value in classification.items():
        _same(state[key], value, "HPO run public input classification mismatch")
    integer(state["seed"], 0, 2**31 - 1)
    if state["device"] != training.device or training.seeds != (state["seed"],):
        raise ContractError("HPO seed/device differs from runtime projection")
    settings = preparation_config(config["preprocessing"])
    if (settings.reader != "participant-structured-v2"
            or settings.memory_budget_mib * 1024**2 > spec["operating"]["source_budget_bytes"]):
        raise ContractError("HPO source reader/memory projection mismatch")
    _deadlines(state, spec, training)
    execution = closed(state["execution"], {"learner", "inference"})
    for role, device in (("learner", training.device), ("inference", configured.inference.device)):
        declared = closed(execution[role], {"binding", "cuda_visible_devices"})
        expected = replace(configured.inference, device=device)
        _same(declared["binding"], asdict(expected), "HPO execution binding mismatch")
        if declared["cuda_visible_devices"] != child_cuda_visibility(device, configured.cuda_visible_devices):
            raise ContractError("HPO execution CUDA visibility mismatch")
    identity = state["identity"]
    if type(identity) is not dict or not identity:
        raise ContractError("HPO recorded source/runtime identity required")
    integration = closed(identity.get("integration"), {"binary_sha256", "server_config_sha256",
        "declared_simulator_revision", "source_revision", "source_spec_sha256"}, {"binary_build_revision"})
    from gf_common.card_instance_runtime import APPROVED_BINARY_SHA256
    if type(config["server"]) is not dict or config["server"].get("external_url") is not None:
        raise ContractError("HPO requires recorded locally owned Simulator runtime")
    if (integration.get("binary_sha256") != APPROVED_BINARY_SHA256
            or integration.get("source_revision") != actions.action_contract.simulator_revision
            or integration.get("declared_simulator_revision") != actions.action_contract.simulator_revision
            or config["server"].get("simulator_revision") != actions.action_contract.simulator_revision):
        raise ContractError("HPO recorded Simulator identity mismatch")
    checksum(integration.get("server_config_sha256"))
    _same(integration.get("source_spec_sha256"), dict(CORRECTED_REVIEWED_SPEC_HASHES), "HPO recorded spec hashes mismatch")
    preprocessing = identity.get("preprocessing", {})
    if settings.backend == "python":
        _same(preprocessing, {"backend": "python", "reader_contract": "participant-structured-v2",
            "implementation": "python-reference-player-card-refs-v2", **classification}, "HPO recorded Python reader mismatch")
    else:
        from gf_training.native_backend import recorded_card_instance_preprocessing
        recorded_card_instance_preprocessing(preprocessing, binding=classification)
        if preprocessing["binary"]["build"].get("card_instance_hpo") != HPO_NATIVE_CAPABILITY:
            raise ContractError("recorded native reader lacks explicit HPO capability")
    expected_resume = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
        "identity": identity, "execution": execution, "classification": classification,
        "storage_binding": state.get("storage_binding")})
    if checksum(state["resume_identity"]) != expected_resume:
        raise ContractError("HPO domain-separated resume identity mismatch")
    integer(state["updates"], 0, training.max_updates)
    used = integer(state["used_decisions"], 0, spec["decision_budget"])
    integer(state["collected_decisions"], used, 2**63 - 1)
    integer(state["accepted_decisions"], used, used)
    integer(state["optimizer_steps"], 0, 2**31 - 1)
    for key in ("collection_seconds", "update_seconds"):
        _number(state[key])
    status, reason = state["status"], state["stop_reason"]
    if state["completed_updates"] is not None:
        integer(state["completed_updates"], 0, training.max_updates)
    if status == "paused" and not state["updates"]:
        raise ContractError("HPO pause requires a committed update boundary")
    if ((status == "running" and reason is not None) or (status == "paused" and reason != "pause")
            or (status == "incomplete" and (reason not in ("update_cap", "time_cap", "storage_cap", "empty_collection") or used == spec["decision_budget"]))
            or (status == "complete" and (used != spec["decision_budget"] or reason is not None
                or state["updates"] == 0 or state["completed_updates"] != state["updates"]))):
        raise ContractError("HPO status/decision budget/completion mismatch")
    if status != "complete" and state["completed_updates"] is not None:
        raise ContractError("HPO unfinished run claims completed updates")
    if type(state["invocations"]) is not list:
        raise ContractError("HPO invocation history required")
    if status != "running":
        if abs(_number(state.get("elapsed_seconds")) - (state["last_unix"] - state["created_unix"])) > .001:
            raise ContractError("HPO elapsed time excludes original paused interval")
        integer(state.get("artifact_bytes"), 0, training.max_owned_bytes)
        if not state["invocations"]:
            raise ContractError("HPO closed run requires learner cleanup history")
    for invocation in state["invocations"]:
        closed(invocation, {"mode", "elapsed_seconds", "child_health"})
        if invocation["mode"] not in ("fresh", "resume"):
            raise ContractError("HPO invocation mode invalid")
        _number(invocation["elapsed_seconds"])
        health = closed(invocation["child_health"], {"exit_code", "stderr_bytes"})
        integer(health["exit_code"], 0, 0)
        integer(health["stderr_bytes"], 0)
    hardware = closed(state.get("hardware"), {"device", "name", "cuda_runtime", "device_count", "cuda_driver_api_version", "exact_driver_build"})
    if hardware["device"] != training.device or type(hardware["name"]) is not str or not hardware["name"] or type(hardware["exact_driver_build"]) is not str or not hardware["exact_driver_build"]:
        raise ContractError("HPO recorded hardware identity mismatch")
    integer(hardware["device_count"], 0, 1024)
    if training.device == "cpu":
        if hardware["device_count"] != 0 or hardware["cuda_driver_api_version"] is not None or hardware["name"] != "cpu":
            raise ContractError("HPO CPU observation has inconsistent hardware")
    elif not hardware["device_count"] or type(hardware["cuda_runtime"]) is not str or not hardware["cuda_runtime"]:
        raise ContractError("HPO CUDA hardware evidence missing")
    else:
        integer(hardware["cuda_driver_api_version"], 1)
    return actions, model, training


def _rng(value: Json) -> None:
    raw = closed(value, {"bit_generator", "state", "has_uint32", "uinteger"})
    nested = closed(raw["state"], {"state", "inc"})
    if raw["bit_generator"] != "PCG64":
        raise ContractError("HPO schedule requires PCG64")
    integer(nested["state"], 0, 2**128 - 1)
    integer(nested["inc"], 1, 2**128 - 1)
    integer(raw["has_uint32"], 0, 1)
    integer(raw["uinteger"], 0, 2**32 - 1)
    if nested["inc"] % 2 != 1:
        raise ContractError("HPO PCG64 increment must be odd")


def _generation(record: Json, state: Json, number: int, parent: str | None,
                previous: Json | None, pool: list[int]) -> list[int]:
    closed(record, HPO_GENERATION_FIELDS)
    integer(record["schema_version"], 4, 4)
    integer(record["generation"], number, number)
    for key in ("run_id", "model_id", "model_version", "input_recipe", "hpo_spec_sha256", *_CLASSIFICATION):
        _same(record[key], state[key], "HPO generation identity/classification mismatch")
    if record["parent_sha256"] != parent:
        raise ContractError("HPO generation parent hash mismatch")
    integer(record["used_decisions"], 0, state["used_decisions"])
    integer(record["accepted_decisions"], record["used_decisions"], record["used_decisions"])
    integer(record["collected_decisions"], record["used_decisions"], state["collected_decisions"])
    integer(record["optimizer_steps"], 0, state["optimizer_steps"])
    if not number:
        if any(record[key] != 0 for key in ("used_decisions", "collected_decisions", "optimizer_steps")) or record["update_sha256"] is not None:
            raise ContractError("HPO G0 must be an untrained fresh initialization")
        pool = [0]
    else:
        checksum(record["update_sha256"])
        assert previous is not None
        if record["used_decisions"] <= previous["used_decisions"]:
            raise ContractError("HPO committed update must advance unique decision count")
        operating = hpo_spec(state)["operating"]
        interval = operating["pool_every_decisions"]
        if record["used_decisions"] // interval > previous["used_decisions"] // interval:
            pool = (pool + [number])[-operating["opponent_pool_size"]:]
    controls = closed(record["parent_controls"], {"schedule_opponent_rng", "snapshot_pool", "collection_position"})
    integer(controls["collection_position"], number, number)
    if type(controls["snapshot_pool"]) is not list:
        raise ContractError("HPO snapshot pool must be an ordered unique generation list")
    for item in controls["snapshot_pool"]:
        integer(item, 0, number)
    if controls["snapshot_pool"] != pool:
        raise ContractError("HPO decision-threshold snapshot pool replay mismatch")
    _rng(controls["schedule_opponent_rng"])
    files = _filemap(record["artifact_files"], {"inference.json", "resume.json"})
    if (checksum(record["inference_sha256"]) != files["inference.json"]
            or checksum(record["resume_sha256"]) != files["resume.json"]):
        raise ContractError("HPO generation immutable artifact map mismatch")
    # These are the existing bundle1 names. A receipt cannot adopt foreign files.
    for name in files:
        if name not in ("inference.json", "resume.json") and re.fullmatch(r"(?:weights|resume)-[0-9]{6}\.(?:json|npz)", name) is None:
            raise ContractError("HPO artifact map contains a foreign basename")
    for prefix in ("weights", "resume"):
        shards = sorted(name for name in files if name.startswith(prefix + "-") and name.endswith(".npz"))
        if not shards or shards != [f"{prefix}-{index:06}.npz" for index in range(len(shards))]:
            raise ContractError("HPO immutable artifact shard inventory invalid")
        if any(name.replace(".npz", ".json") not in files for name in shards):
            raise ContractError("HPO immutable artifact sidecar missing")
        if len([name for name in files if name.startswith(prefix + "-")]) != 2 * len(shards):
            raise ContractError("HPO immutable artifact sidecar inventory invalid")
    return pool


def _identity(value: Any) -> None:
    # EXACT existing structured_lineage._stat_identity: dev,ino,size,mtime,ctime.
    if type(value) is not list or len(value) != 5:
        raise ContractError("HPO deletion requires the existing full stat identity")
    for item in value:
        integer(item, 0)


def _receipt(value: Json, expected: str) -> str:
    item = closed(value, {"sha256", "status", "deletion_intent", "deletion_receipt"})
    if checksum(item["sha256"]) != expected or item["status"] not in ("retained", "deleting", "deleted", "failed"):
        raise ContractError("HPO per-file registry hash/status invalid")
    status = item["status"]
    if status == "retained":
        if item["deletion_intent"] is not None or item["deletion_receipt"] is not None:
            raise ContractError("retained HPO file has deletion evidence")
        return status
    intent = closed(item["deletion_intent"], {"operation_id", "staging_basename", "sha256", "original_identity"})
    if type(intent["operation_id"]) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", intent["operation_id"]) is None:
        raise ContractError("HPO deletion operation identity invalid")
    pruning_basename(intent["staging_basename"])
    if checksum(intent["sha256"]) != expected:
        raise ContractError("HPO deletion intent checksum mismatch")
    _identity(intent["original_identity"])
    if status == "deleted":
        receipt = closed(item["deletion_receipt"], {"operation_id", "sha256", "original_identity", "completed_unix"})
        for key in ("operation_id", "sha256", "original_identity"):
            _same(receipt[key], intent[key], "HPO per-file deletion receipt does not match intent")
        _number(receipt["completed_unix"], positive=True)
    elif item["deletion_receipt"] is not None:
        raise ContractError("uncertain HPO deletion cannot claim a successful receipt")
    return status


def _registry(registry: Json, records: list[Json], state: Json, scan: _ChainScan,
              actions: ActionConfigV3, model: ModelConfigV4) -> None:
    closed(registry, {"schema_version", "generations", "evaluation_references"})
    integer(registry["schema_version"], 4, 4)
    generations = registry["generations"]
    if type(generations) is not dict or set(generations) != {str(index) for index in range(len(records))}:
        raise ContractError("HPO per-file registry generation inventory mismatch")
    spec = hpo_spec(state)
    count = len(records)
    refs = registry["evaluation_references"]
    if type(refs) is not dict:
        raise ContractError("HPO evaluation reference map required")
    scan.charge(512 * len(refs), temporary=True)
    for token, value in refs.items():
        if type(token) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", token) is None:
            raise ContractError("HPO bounded evaluation reference token invalid")
        closed(value, {"generation", "inference_sha256"})
        index = integer(value["generation"], 0, count - 1)
        if checksum(value["inference_sha256"]) != records[index]["inference_sha256"]:
            raise ContractError("HPO evaluation reference inference mismatch")
    protected = protected_hpo_files(state, records, registry, reference_journal(scan, state, records, registry))
    recipe = hpo_training_config(spec["training_recipe"])
    optimizer = adam_recipe(recipe.adam_learning_rate, recipe.adam_beta1, recipe.adam_beta2, recipe.adam_epsilon)
    for index, record in enumerate(records):
        entry = closed(generations[str(index)], {"files"})
        files = entry["files"]
        originals = record["artifact_files"]
        if type(files) is not dict or set(files) != set(originals):
            raise ContractError("HPO registry file inventory differs from immutable generation")
        retained = {name for name, value in files.items() if _receipt(value, originals[name]) == "retained"}
        if any(value["status"] in ("deleting", "failed") for value in files.values()):
            raise ContractError("HPO uncertain deletion cannot form a healthy structural chain")
        inference_files = {name for name in originals if name == "inference.json" or name.startswith("weights-")}
        resume_files = set(originals)
        if not protected[index] <= retained:
            raise ContractError("HPO protected file/dependency closure missing")
        if "resume.json" in retained and not resume_files <= retained:
            raise ContractError("HPO retained resume lost inference/weights/state dependencies")
        if "inference.json" in retained and not inference_files <= retained:
            raise ContractError("HPO retained inference lost weights dependencies")
        directory = checkpoint_path(scan.root, index)
        scan.inventory(directory, retained)
        for name in retained:
            with scan.file(directory / basename(name), originals[name]):
                pass
        if "resume.json" in retained:
            artifact = inspect_retained_game_artifact_v4(directory, expected_inference_digest=record["inference_sha256"],
                actions=actions, rules=card_instance_rules(), training_recipe=asdict(recipe), optimizer_recipe=optimizer, scan=scan)
            _same(artifact.files, originals, "HPO saved full artifact map mismatch")
            assert artifact.resume is not None
            resume = artifact.resume
            _same(resume["model"], asdict(model), "HPO retained full model/resources differ from run")
            _same(resume["binding"], state["execution"]["learner"]["binding"], "HPO retained learner binding mismatch")
            if (resume["optimizer_steps"] != record["optimizer_steps"] or resume["resume_identity"] != state["resume_identity"]
                    or resume["cuda_rng_count"] != (0 if state["device"] == "cpu" else state["hardware"]["device_count"])):
                raise ContractError("HPO retained optimizer/resume/RNG identity mismatch")
        elif "inference.json" in retained:
            artifact = inspect_retained_game_inference_v4(directory, expected_inference_digest=record["inference_sha256"],
                actions=actions, rules=card_instance_rules(), scan=scan)
            _same(artifact.files, {name: originals[name] for name in inference_files}, "HPO inference file map mismatch")
        else:
            continue
        _same(artifact.inference["model"], inference_model(model), "HPO resource-free inference architecture mismatch")


def protected_hpo_files(state: Json, records: list[Json], registry: Json,
                        pending_bindings: list[Json] | None = None) -> dict[int, set[str]]:
    """One pure protection calculation shared by the reader and owned retirement."""
    op = hpo_spec(state)['operating']
    count = len(records)
    resumes = {0, *range(max(0, count-op['resume_latest']), count)}
    inference = {0, *resumes, *range(max(0, count-op['inference_latest']), count),
        *records[-1]['parent_controls']['snapshot_pool']}
    interval = op['inference_periodic_decisions']
    inference.update(g for g in range(1, count)
        if records[g]['used_decisions']//interval > records[g-1]['used_decisions']//interval)
    inference.update(row['generation'] for row in registry['evaluation_references'].values())
    inference.update(row['generation'] for row in pending_bindings or ())
    return {g: set(record['artifact_files']) if g in resumes else
        {name for name in record['artifact_files'] if name == 'inference.json' or name.startswith('weights-')}
        if g in inference else set() for g, record in enumerate(records)}


def sync_hpo_ancestry(path: Path, *, ownership: RunOwnership) -> None:
    """Reconcile directory durability on every attempt, including existing paths."""
    require_hpo_run_owner(ownership.directory,ownership)
    if not path.is_relative_to(ownership.directory):
        raise ContractError('HPO synchronization target is outside the owned parent')
    identities = []
    for directory in (path, *path.parents):
        require_hpo_run_owner(ownership.directory,ownership)
        before = directory.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise ContractError('HPO control ancestry must be nonsymlink directories')
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            held = os.fstat(fd)
            if (held.st_dev, held.st_ino) != (before.st_dev, before.st_ino):
                raise ContractError('HPO control ancestry replaced before sync')
            os.fsync(fd)
        finally:
            os.close(fd)
        identities.append((directory, before.st_dev, before.st_ino))
    for directory, device, inode in identities:
        current = directory.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (device, inode):
            raise ContractError('HPO control ancestry changed during sync')


def write_hpo_control(path: Path, value: Json, *, scan: _ChainScan, immutable: bool = False,
                        ownership: RunOwnership | None = None) -> str:
    owner=require_hpo_run_owner(scan.root,ownership)
    try:
        sync_hpo_ancestry(path.parent,ownership=owner)
        if path.exists() or path.is_symlink():
            if immutable or path not in scan.files:
                raise ContractError('HPO publication refuses foreign/preexisting control entry')
            with scan.file(path,scan.verified_digest(path)): pass
        require_hpo_run_owner(scan.root,owner)
        result = _write(path, value, scan=scan, immutable=immutable)
        sync_hpo_ancestry(path.parent,ownership=owner)
        scan.check()
        return result
    except BaseException as error:
        error.__dict__['_hpo_storage_uncertain']=str(path)
        raise


def checked_reference_binding(value: Json, state: Json, records: list[Json], scan: _ChainScan) -> Json:
    closed(value, {'run_id', 'token', 'generation', 'inference_sha256', 'generation_sha256'})
    if (value['run_id'] != state['run_id'] or type(value['token']) is not str
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value['token']) is None):
        raise ContractError('HPO reference run/token binding invalid')
    generation = integer(value['generation'], 0, len(records)-1)
    if (checksum(value['inference_sha256']) != records[generation]['inference_sha256']
            or checksum(value['generation_sha256']) != scan.verified_digest(generation_path(scan.root, generation))):
        raise ContractError('HPO reference immutable generation binding mismatch')
    return value


def reference_journal(scan: _ChainScan, state: Json, records: list[Json], registry: Json) -> list[Json]:
    """Validate only this bounded write-ahead journal; pending pins stay protected."""
    root = scan.root/REFERENCE_DIRECTORY
    if not root.exists() and not root.is_symlink():
        scan._hpo_reference_snapshot = (None, ())
        return []
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ContractError('HPO reference control namespace must be a directory')
    names = []
    with os.scandir(root) as entries:
        for entry in entries:
            scan.charge(1024+len(entry.name)*4)
            if (not entry.is_dir(follow_symlinks=False)
                    or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', entry.name) is None):
                raise ContractError('foreign HPO reference namespace entry')
            names.append(entry.name)
    pending = []
    for token in sorted(names):
        directory = root/token
        files = set()
        with os.scandir(directory) as entries:
            for entry in entries:
                scan.charge(512)
                if entry.name not in REFERENCE_FILES or not entry.is_file(follow_symlinks=False):
                    raise ContractError('foreign HPO reference transaction file')
                files.add(entry.name)
        if 'acquire.intent.json' not in files or ('release.intent.json' in files and 'acquire.receipt.json' not in files):
            raise ContractError('HPO reference transaction has no original acquire intent/receipt')
        scan.inventory(directory, files)
        binding = None
        for phase in ('acquire', 'release'):
            name = phase+'.intent.json'
            if name not in files:
                if phase+'.receipt.json' in files:
                    raise ContractError('HPO reference completion lacks its intent')
                continue
            intent = closed(scan.object(directory/name), {'schema_version','kind','operation_id','binding'}
                | ({'acquire_receipt_sha256'} if phase=='release' else set()))
            integer(intent['schema_version'],1,1)
            if intent['kind'] != 'hpo-evaluation-reference-'+phase+'-intent':
                raise ContractError('HPO reference intent kind mismatch')
            try:
                if str(uuid.UUID(intent['operation_id'])) != intent['operation_id']:
                    raise ValueError('noncanonical UUID')
            except (ValueError, TypeError, AttributeError) as error:
                raise ContractError('HPO reference operation UUID required') from error
            checked_reference_binding(intent['binding'], state, records, scan)
            if intent['binding']['token'] != token:
                raise ContractError('HPO reference path/token mismatch')
            if binding is None:
                binding = intent['binding']
            _same(intent['binding'], binding, 'HPO reference transaction binding changed')
            if phase=='release' and checksum(intent['acquire_receipt_sha256']) != scan.verified_digest(directory/'acquire.receipt.json'):
                raise ContractError('HPO reference release changed acquire receipt')
            receipt_name = phase+'.receipt.json'
            if receipt_name in files:
                receipt = closed(scan.object(directory/receipt_name), {'schema_version','kind','binding',
                    'intent_sha256','registry_sha256','confirmed_unix'})
                integer(receipt['schema_version'],1,1)
                if (receipt['kind'] != 'hpo-evaluation-reference-'+phase+'-receipt'
                        or checksum(receipt['intent_sha256']) != scan.verified_digest(directory/name)):
                    raise ContractError('HPO reference completion/intent link mismatch')
                _same(receipt['binding'],binding,'HPO reference completion binding mismatch')
                checksum(receipt['registry_sha256'])  # Historical fact, not perpetual registry equality.
                _number(receipt['confirmed_unix'],positive=True)
        assert binding is not None
        expected = {'generation':binding['generation'],'inference_sha256':binding['inference_sha256']}
        actual = registry['evaluation_references'].get(token)
        if actual is not None:
            _same(actual,expected,'HPO active reference differs from original intent')
        released = 'release.receipt.json' in files
        if released and actual is not None:
            raise ContractError('released HPO reference token became active again')
        if 'acquire.receipt.json' in files and 'release.intent.json' not in files and actual is None:
            raise ContractError('acquired HPO reference vanished without release intent')
        if not released:
            pending.append(binding)
    if _stat_identity(root.lstat()) != _stat_identity(before):
        raise ContractError('HPO reference namespace changed during inspection')
    scan._hpo_reference_snapshot = (_stat_identity(before), tuple(sorted(names)))
    return pending


def _selection(value: Json, state: Json, previous: Json, batch: int) -> Json:
    selection = closed(value, {"context", "derived_seed", "episode_counts", "collected_decisions",
        "used_decisions", "indices", "indices_sha256"})
    spec = hpo_spec(state)
    recipe = hpo_training_config(spec["training_recipe"])
    context = closed(selection["context"], {"selection_version", "decision_budget", "used_before",
        "remaining", "learning_seed", "batch"})
    for key in ("decision_budget", "used_before", "remaining", "learning_seed", "batch"):
        integer(context[key], 0, 2**63 - 1)
    _same(context, {"selection_version": spec["selection_version"], "decision_budget": spec["decision_budget"],
        "used_before": previous["used_decisions"], "remaining": spec["decision_budget"] - previous["used_decisions"],
        "learning_seed": state["seed"], "batch": batch}, "HPO selection context differs from original parent position")
    seed = int(digest({"recipe": "hpo-ppo-selection-seed-v1", "context": context})[:32], 16)
    integer(selection["derived_seed"], seed, seed)
    counts = selection["episode_counts"]
    if type(counts) is not list or len(counts) != recipe.collection_games:
        raise ContractError("HPO selection must retain every complete learner episode count")
    for item in counts:
        integer(item, 0, 1024)
    collected = integer(selection["collected_decisions"], 0, recipe.collection_games * 1024)
    used = integer(selection["used_decisions"], 1, collected)
    if collected != sum(counts) or used != min(context["remaining"], collected):
        raise ContractError("HPO selected/collected/remaining arithmetic mismatch")
    indices = selection["indices"]
    if type(indices) is not list or len(indices) != used:
        raise ContractError("HPO selected endpoint indices count mismatch")
    last = -1
    for index in indices:
        integer(index, last + 1, collected - 1)
        last = index
    if checksum(selection["indices_sha256"]) != digest(indices):
        raise ContractError("HPO sorted selected endpoint indices hash mismatch")
    # This checks persisted arithmetic, not RNG/original authority. Source3
    # independently recreates the subset from complete original participant logs.
    return selection


def _proof(value: Json, state: Json, previous: Json, batch: int, delta: int) -> Json:
    packed = "storage_binding" in state
    fields = {"schema_version", "kind", "qualified", "parent_qualification", "manifest_sha256",
        "source_set_sha256", "files", "games", "accepted_decisions", "closure_counts", "missing_replays",
        "selection", "expected_optimizer_step_delta"} | _CLASSIFICATION
    proof = closed(value, fields | ({"storage", "custody"} if packed else set()))
    integer(proof["schema_version"], 3, 3)
    if (proof["kind"] != ("card-instance-hpo-packed-original-source-proof" if packed else "card-instance-hpo-original-source-proof")
            or proof["qualified"] is not False or proof["parent_qualification"] != "not_performed"):
        raise ContractError("HPO proof must preserve explicit unqualified Source authority")
    for key in _CLASSIFICATION:
        _same(proof[key], state[key], "HPO proof public input identity mismatch")
    selection = _selection(proof["selection"], state, previous, batch)
    integer(proof["games"], len(selection["episode_counts"]), len(selection["episode_counts"]))
    integer(proof["accepted_decisions"], selection["collected_decisions"])
    integer(proof["expected_optimizer_step_delta"], delta, delta)
    checksum(proof["manifest_sha256"])
    if type(proof["files"]) is not dict or checksum(proof["source_set_sha256"]) != digest(proof["files"]):
        raise ContractError("HPO original declared file set hash mismatch")
    if packed:
        # Reuse the unchanged physical custody/member validator only. HPO's
        # version/kind/selection authority is validated above, never inferred.
        from gf_common.group_pack_game import PROOF_KIND, checked_packed_proof
        physical = {key: item for key, item in proof.items() if key not in ("selection", "expected_optimizer_step_delta")}
        checked_packed_proof({**physical, "schema_version": 2, "kind": PROOF_KIND})
    else:
        for name, entry in proof["files"].items():
            _relative(Path("/"), name)
            closed(entry, {"sha256", "bytes", "identity"})
            checksum(entry["sha256"])
            integer(entry["bytes"], 0)
            _identity(entry["identity"])
            if entry["bytes"] != entry["identity"][2]:
                raise ContractError("HPO original byte/stat identity mismatch")
    closures = closed(proof["closure_counts"], {"natural"})
    integer(closures["natural"], proof["games"], proof["games"])
    if proof["missing_replays"] != []:
        raise ContractError("HPO update cannot use a missing original replay")
    return selection


def _relative(directory: Path, value: Any) -> Path:
    if type(value) is not str or not value or Path(value).is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ContractError("HPO owned relative source path required")
    if "\\" in value or "\x00" in value or any(not part for part in value.split("/")):
        raise ContractError("HPO noncanonical relative source path")
    return directory / value


def _policies(state: Json, records: list[Json], directory: Path, model: ModelConfigV4) -> tuple[Json, str]:
    execution = state["execution"]["inference"]
    binding = InferenceBindingV2(**execution["binding"])
    actions = action_config_v3(state["effective_config"]["actions"])
    classification = input_binding_game_v3(actions.action_contract, card_instance_rules())
    registry: Json = {"random-card-instance-v2": {"model_id": "random", "policy_options": None},
        "heuristic-card-instance-v2": {"model_id": "heuristic", "policy_options": None}}
    learner_label = ""
    indices = [records[-1]["generation"], *records[-1]["parent_controls"]["snapshot_pool"]]
    for index in indices:
        sha = records[index]["inference_sha256"]
        label = policy_label_game_v4(model, binding, sha, action_contract=actions.action_contract,
            rules=card_instance_rules(), action_config_hash=digest(state["effective_config"]["actions"]),
            cuda_visible_devices=execution["cuda_visible_devices"])
        if not learner_label:
            learner_label = label
        options = safe_options_game_v4({"version": "4", "checkpoint": str(checkpoint_path(directory, index)),
            "expected_manifest_digest": sha, "model": asdict(model), "inference": asdict(binding),
            "cuda_visible_devices": execution["cuda_visible_devices"],
            "action_config_hash": digest(state["effective_config"]["actions"]), **classification},
            actions=actions, rules=card_instance_rules())
        registry[label] = {"model_id": model.model_id, "policy_options": options}
    return registry, learner_label


def _update(directory: Path, state: Json, records: list[Json], current: Json, scan: _ChainScan,
            model: ModelConfigV4, *, value: Json | None = None) -> None:
    previous = records[-1]
    generation = current["generation"]
    packed = "storage_binding" in state
    update = closed(scan.object(directory / "updates" / f"{generation:06}.json", current["update_sha256"]) if value is None else value,
        HPO_UPDATE_FIELDS | ({"storage_binding"} if packed else set()))
    integer(update["schema_version"], 5, 5)
    integer(update["generation"], generation, generation)
    for key in ("run_id", "model_id", "model_version", "input_recipe", "hpo_spec_sha256", *_CLASSIFICATION):
        _same(update[key], state[key], "HPO update identity/classification mismatch")
    if packed:
        _same(update["storage_binding"], state["storage_binding"], "HPO update storage binding mismatch")
    batch = f"batch-{generation - 1:06}"
    if (update["status"] != "complete" or update["parent_sha256"] != current["parent_sha256"]
            or update["batch_id"] != batch or update["collection"] != "collections/" + batch
            or update["numeric"] != "numeric/" + batch or update["health"] != {}):
        raise ContractError("HPO update links/batch/health mismatch")
    spec = hpo_spec(state)
    recipe = hpo_training_config(spec["training_recipe"])
    used = integer(update["used_decisions"], 1, spec["decision_budget"])
    collected = integer(update["collected_decisions"], used, recipe.collection_games * 1024)
    integer(update["accepted_decisions"], used, used)
    integer(update["overshoot_decisions"], collected - used, collected - used)
    delta = recipe.epochs * ((used + recipe.minibatch_size - 1) // recipe.minibatch_size)
    integer(update["expected_optimizer_step_delta"], delta, delta)
    for key, amount in (("used_decisions", used), ("collected_decisions", collected), ("optimizer_steps", delta)):
        if current[key] != previous[key] + amount:
            raise ContractError("HPO update/generation cumulative counter or optimizer delta mismatch")
    for key in ("planned_games", "actual_games"):
        integer(update[key], recipe.collection_games, recipe.collection_games)
    seats = update["learner_seats"]
    if type(seats) is not dict or len(seats) != recipe.collection_games:
        raise ContractError("HPO update must preserve every original learner seat")
    for episode, seat in seats.items():
        basename(episode)
        integer(seat, 1, 2)
    selection = _proof(update["source_proof"], state, previous, generation - 1, delta)
    if (selection["used_decisions"] != used or selection["collected_decisions"] != collected
            or checksum(update["collection_manifest_sha256"]) != update["source_proof"]["manifest_sha256"]):
        raise ContractError("HPO update/proof count or original manifest hash mismatch")
    numeric = _owned_relative(directory, update["numeric"])
    files = _filemap(update["numeric_files"], {"rollout.json"})
    if files["rollout.json"] != checksum(update["numeric_sha256"]):
        raise ContractError("HPO numeric manifest hash differs from file map")
    manifest = closed(scan.object(numeric / "rollout.json", update["numeric_sha256"]),
        {"schema_version", "kind", "architecture_identity", "action_config_hash", "policy_version", "decision_count",
            "episodes", "data", "training_recipe", "selection", "expected_optimizer_step_delta"} | _CLASSIFICATION | _SEMANTICS)
    integer(manifest["schema_version"], 5, 5)
    from gf_models.structured_config import architecture_identity
    if (manifest["kind"] != "structured-rollout" or manifest["architecture_identity"] != architecture_identity(model, layout=LAYOUT)
            or manifest["action_config_hash"] != digest(state["effective_config"]["actions"])):
        raise ContractError("HPO Numeric5 model/action identity mismatch")
    for key in _CLASSIFICATION:
        _same(manifest[key], state[key], "HPO numeric classification mismatch")
    bootstrap = scan.object(checkpoint_path(directory, 0) / "inference.json", records[0]["inference_sha256"])
    for key in _SEMANTICS:
        _same(manifest[key], bootstrap[key], "HPO numeric/inference5 semantic mismatch")
    _same(manifest["selection"], selection, "HPO numeric/proof selection mismatch")
    _same(manifest["training_recipe"], spec["training_recipe"], "HPO numeric recipe mismatch")
    integer(manifest["expected_optimizer_step_delta"], delta, delta)
    integer(manifest["decision_count"], used, used)
    episodes = manifest["episodes"]
    if type(episodes) is not list or len(episodes) != recipe.collection_games:
        raise ContractError("HPO numeric complete episode population missing")
    owners = set()
    for episode, count in zip(episodes, selection["episode_counts"], strict=True):
        if type(episode) is not dict or type(episode.get("frames")) is not list or type(episode.get("selections")) is not list:
            raise ContractError("HPO numeric complete episode frames/selections required")
        integer(episode.get("actor"), 1, 2)
        validate_result(episode.get("outcome"))
        owner = (episode["actor"], episode.get("episode"))
        if (type(owner[1]) is not str or not owner[1] or owner in owners or len(episode["frames"]) != count
                or len(episode["selections"]) != count or episode.get("end_kind") != "natural"
                or episode.get("outcome") is None
                or episode.get("reason") is not None or episode.get("has_targets") is not bool(count)):
            raise ContractError("HPO numeric original episode closure/count/ownership mismatch")
        owners.add(owner)
        for key in _CLASSIFICATION:
            _same(episode.get(key), state[key], "HPO numeric episode classification mismatch")
    if type(manifest["data"]) is not dict or manifest["data"].get("prefix") != "rollout":
        raise ContractError("HPO Numeric5 physical bundle prefix mismatch")
    _same(files, {"rollout.json": update["numeric_sha256"], **manifest["data"]["files"]}, "HPO numeric exact file map mismatch")
    scan.inventory(numeric, set(files))
    scan.bundle(numeric, manifest["data"])
    policies, label = _policies(state, records, directory, model)
    _same(update["policy_bindings"], policies, "HPO update frozen policy registry mismatch")
    if manifest["policy_version"] != label:
        raise ContractError("HPO numeric learner policy identity mismatch")
    metrics = closed(update["metrics"], {"policy_loss", "value_mse", "entropy", "approx_kl", "total_loss",
        "optimizer_steps", "optimizer_step_delta", "collected_decisions", "used_decisions", "parity", "tbptt_length",
        "epochs", "endpoint_evaluations", "context_decisions", "suffix_decisions", "update_seconds"},
        {"elapsed_seconds", "save_seconds", "peak_cpu_memory_bytes"})
    for key in ("policy_loss", "value_mse", "entropy", "approx_kl", "total_loss"):
        finite_number(metrics[key], -1e20, 1e20)
    values = {"optimizer_steps": current["optimizer_steps"], "optimizer_step_delta": delta,
        "collected_decisions": collected, "used_decisions": used, "tbptt_length": recipe.tbptt_length,
        "epochs": recipe.epochs, "endpoint_evaluations": recipe.epochs * used}
    for key, expected_value in values.items():
        integer(metrics[key], expected_value, expected_value)
    integer(metrics["context_decisions"], recipe.epochs * used, recipe.epochs * used * 1024)
    integer(metrics["suffix_decisions"], recipe.epochs * used,
        min(metrics["context_decisions"], recipe.epochs * used * recipe.tbptt_length))
    parity = closed(metrics["parity"], {"atol", "rtol", "max_abs_error", "decisions", "checks"})
    integer(parity["decisions"], collected, collected)
    if parity["atol"] != 1e-5 or parity["rtol"] != 1e-5 or parity["checks"] != ["saved", "cpu-cache", "cpu-full", "learner-full", "learner-split"]:
        raise ContractError("HPO whole-original parity recipe mismatch")
    _number(parity["max_abs_error"])
    for key in ("collection_seconds", "update_seconds"):
        _number(update[key])
    for key in ("update_seconds", "elapsed_seconds", "save_seconds"):
        if key in metrics:
            _number(metrics[key])
    if "peak_cpu_memory_bytes" in metrics:
        integer(metrics["peak_cpu_memory_bytes"], 0)


def commit_hpo_generation(directory: Path,state: Json,*,generation: int,update: Json | None,
                           controls: Json,artifact_files: Json,max_owned_bytes: int,deadline: float,
                           projection: Any = None,ownership: RunOwnership | None = None) -> Json:
    """Append verified state under the original lock and actual Source capability."""
    import hashlib

    owner=require_hpo_run_owner(directory,ownership)
    directory=owner.directory
    actions,model,_=hpo_state(state)
    if state['status']!='running': raise ContractError('HPO commit requires running parent')
    spec=hpo_spec(state)
    recipe=hpo_training_config(spec['training_recipe'])
    with ExitStack() as scope:
        if generation:
            from gf_training.card_instance_source import IssuedGameProjection
            if type(projection) is not IssuedGameProjection or projection.source.online is None:
                raise ContractError('HPO append requires actual Source-issued completed projection')
            require_hpo_run_owner(directory,projection.source.online.ownership)
            projection.require(projection.source.online.process,after=True)
            projection.retire_packed_decoded()
            scan=projection.source.controls
            records=[scan.object(generation_path(directory,number)) for number in range(generation)]
            durable=scan.object(directory/'run.json')
            mutable={'accepted_decisions','used_decisions','collected_decisions','optimizer_steps',
                'collection_seconds','update_seconds','last_unix','hardware'}
            for key in set(state)-mutable:
                _same(durable.get(key),state[key],'HPO append changed durable frozen parent state')
            if generation!=state['updates']+1 or update is None or state['generation_sha256']!=scan.verified_digest(generation_path(directory,generation-1)):
                raise ContractError('HPO append must follow exactly one committed generation')
            registry=scan.object(directory/'artifact-registry.json')
            parent=state['generation_sha256']
        else:
            if projection is not None or update is not None or state['updates']!=0 or state['generation_sha256'] is not None:
                raise ContractError('HPO G0 must be a fresh untrained initialization')
            scan=scope.enter_context(_ChainScan(directory=directory,max_owned_bytes=max_owned_bytes,deadline=deadline))
            records=[]
            durable=scan.object(directory/'run.json')
            if durable['config_sha256']!=state['config_sha256'] or durable['run_id']!=state['run_id']:
                raise ContractError('HPO G0 differs from original parent declaration')
            registry=scan.object(directory/'artifact-registry.json')
            _same(registry,{'schema_version':4,'generations':{},'evaluation_references':{}},'HPO G0 registry is not fresh')
            parent=None
        files=_filemap(artifact_files,{'inference.json','resume.json'})
        if scan.artifact_snapshot is not None and scan.artifact_snapshot[:2] != _stat_identity((directory/'artifacts').lstat())[:2]:
            raise ContractError('HPO update replaced the original artifacts parent')
        scan.artifact_snapshot=None
        _artifact_directories(directory,generation,scan)
        # The parent child adapter already checked this pair within its explicit
        # save envelope. Commit independently rechecks the actual same bytes.
        artifact=inspect_retained_game_artifact_v4(checkpoint_path(directory,generation),
            expected_inference_digest=files['inference.json'],actions=actions,rules=card_instance_rules(),
            training_recipe=asdict(recipe),optimizer_recipe=adam_recipe(recipe.adam_learning_rate,
                recipe.adam_beta1,recipe.adam_beta2,recipe.adam_epsilon),scan=scan)
        _same(artifact.files,files,'HPO commit actual full saved files differ')
        assert artifact.resume is not None
        if artifact.resume['resume_identity']!=state['resume_identity'] or artifact.resume['optimizer_steps']!=state['optimizer_steps']:
            raise ContractError('HPO commit actual resume/counter mismatch')
        _same(artifact.resume['model'],asdict(model),'HPO commit actual model/resources mismatch')
        _same(artifact.resume['binding'],state['execution']['learner']['binding'],'HPO commit execution mismatch')
        update_sha=None if update is None else hashlib.sha256((canonical(update)+'\n').encode()).hexdigest()
        record={'schema_version':4,'run_id':state['run_id'],'generation':generation,'parent_sha256':parent,
            **{key:state[key] for key in ('model_id','model_version','input_recipe','hpo_spec_sha256',*_CLASSIFICATION)},
            'inference_sha256':files['inference.json'],'resume_sha256':files['resume.json'],'artifact_files':dict(files),
            'accepted_decisions':state['used_decisions'],'used_decisions':state['used_decisions'],
            'collected_decisions':state['collected_decisions'],'optimizer_steps':state['optimizer_steps'],
            'update_sha256':update_sha,'parent_controls':controls}
        _generation(record,state,generation,parent,records[-1] if records else None,
            records[-1]['parent_controls']['snapshot_pool'] if records else [0])
        if update is not None: _update(directory,state,records,record,scan,model,value=update)
        scan.finalize()
        if projection is not None: projection.prepare_commit()
        if update is not None and write_hpo_control(directory/'updates'/f'{generation:06}.json',update,scan=scan,immutable=True,ownership=owner)!=update_sha:
            raise ContractError('HPO update serialization changed during commit')
        sha=write_hpo_control(generation_path(directory,generation),record,scan=scan,immutable=True,ownership=owner)
        registry['generations'][str(generation)]={'files':{name:{'sha256':value,'status':'retained',
            'deletion_intent':None,'deletion_receipt':None} for name,value in files.items()}}
        write_hpo_control(directory/'artifact-registry.json',registry,scan=scan,ownership=owner)
        write_hpo_control(directory/'latest.json',{'generation':generation,'sha256':sha},scan=scan,ownership=owner)
        state.update(updates=generation,generation_sha256=sha)
        write_hpo_control(directory/'run.json',state,scan=scan,ownership=owner)
        # Own immutable appends update exactly the two known control inventories.
        scan.inventories[directory/'generations']=tuple(f'{n:06}.json' for n in range(generation+1))
        scan.inventories[directory/'updates']=tuple(f'{n:06}.json' for n in range(1,generation+1))
        scan.finalize()
        return record


def validate_hpo_chain(directory: Path, *, require_complete: bool = False, max_owned_bytes: int,
                       deadline: float, extra_owned_bytes: int = 0,
                       scan: _ChainScan | None = None) -> tuple[Json, list[Json]]:
    """Read exact structural HPO evidence using the caller's SAME scan, if given.

    Does not qualify CPU observations for production, resolve packed collections,
    replay original journals, issue updates or establish evaluation authority.
    """
    directory = directory.resolve()
    integer(extra_owned_bytes, 0, max_owned_bytes - 1)
    with ExitStack() as scope:
        if scan is None:
            scan = scope.enter_context(_ChainScan(directory=directory,
                max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline))
        elif (type(scan) is not _ChainScan or scan.root != directory or scan._closed
                or scan.limit > max_owned_bytes - extra_owned_bytes):
            raise ContractError("HPO chain requires SAME admitted open parent scan")
        scan.deadline = min(scan.deadline, deadline)
        state = scan.object(directory / "run.json")
        actions, model, _training = hpo_state(state)
        if require_complete and state["status"] != "complete":
            raise ContractError("complete exactly-D structural HPO chain required")
        count = state["updates"]
        _artifact_directories(directory, count, scan)
        scan.inventory(directory / "generations", {f"{number:06}.json" for number in range(count + 1)})
        scan.inventory(directory / "updates", {f"{number:06}.json" for number in range(1, count + 1)})
        latest = closed(scan.object(directory / "latest.json"), {"generation", "sha256"})
        integer(latest["generation"], count, count)
        if checksum(latest["sha256"]) != checksum(state["generation_sha256"]):
            raise ContractError("HPO latest/run hash mismatch")
        parent = None
        records: list[Json] = []
        pool = [0]
        for number in range(count + 1):
            path = generation_path(directory, number)
            record = scan.object(path)
            pool = _generation(record, state, number, parent, records[-1] if records else None, pool)
            if number:
                _update(directory, state, records, record, scan, model)
            records.append(record)
            parent = scan.verified_digest(path)
        if parent != state["generation_sha256"]:
            raise ContractError("HPO final generation hash differs from latest/run")
        for key in ("used_decisions", "collected_decisions", "accepted_decisions", "optimizer_steps"):
            if records[-1][key] != state[key]:
                raise ContractError("HPO cumulative generation/run counter mismatch")
        _registry(scan.object(directory / "artifact-registry.json"), records, state, scan, actions, model)
        scan.finalize()
        return state, records
