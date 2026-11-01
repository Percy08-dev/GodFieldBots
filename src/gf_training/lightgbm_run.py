"""One owned native subprocess per training seed, with original charged budgets."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gf_common.domain import ActionConfig, ContractError, Json, canonical, closed, digest, integer
from gf_models.lightgbm.config import (
    ModelConfig,
    TrainingConfig,
    model_config,
    training_config,
    validate_capacity,
)
from gf_models.lightgbm.encoder import Array, encode, feature_names
from gf_models.lightgbm.model import (
    inspect_artifact,
    native_parameters,
    numeric_matrix,
)
from gf_training.gates import feasibility
from gf_training.journal import atomic_json
from gf_training.native_preparation import NativePreparedInputs
from gf_training.preparation_numeric import PreparedInputs
from gf_training.q_data import (
    check_deadline,
    labeled_samples,
    load_run,
    provenance,
    validate_bundles,
)

if TYPE_CHECKING:
    from gf_cli.config import LoadedConfig


def _sha256(path: Path, deadline: float) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            check_deadline(deadline)
            value.update(chunk)
    check_deadline(deadline)
    return value.hexdigest()


async def _drain(stream: asyncio.StreamReader, path: Path, limit: int, health: Json) -> None:
    count = 0
    with path.open("xb") as output:
        while chunk := await stream.read(65536):
            remaining = max(0, limit - count)
            output.write(chunk[:remaining])
            count += len(chunk)
            if count > limit:
                health["log_limit_exceeded"] = True
    health[path.name + "_bytes"] = count


async def _stop(process: asyncio.subprocess.Process, drainers: list[asyncio.Task[None]], seconds: float, health: Json) -> None:
    started = time.monotonic()
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), max(0, seconds))
        except TimeoutError:
            health["emergency_kill"] = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), 1)
    try:
        await asyncio.wait_for(asyncio.gather(*drainers), max(.05, seconds - (time.monotonic() - started)))
    except TimeoutError:
        health["drain_failure"] = "pipe EOF deadline"
        for task in drainers:
            task.cancel()
        await asyncio.gather(*drainers, return_exceptions=True)
    health["exit_code"] = process.returncode
    if time.monotonic() - started > seconds:
        health["cleanup_overrun_seconds"] = time.monotonic() - started - seconds


async def _run_numeric(directory: Path, x: Array | None, y: Array | None, validation_x: Array | None, validation_y: Array | None, *,
                      model: ModelConfig, actions: ActionConfig, training: TrainingConfig, seed: int,
                      purpose: str = "toy", charged_seconds: float = 0, provenance_data: Json | None = None,
                      command: tuple[str, ...] | None = None, fit_code_provenance: Json | None = None,
                      fit_code_measurement_seconds: float = 0, prepared_inputs: PreparedInputs | NativePreparedInputs | None = None) -> Json:
    """Low-level synthetic component runner; real CLI uses verified source proof."""
    training_config(asdict(training))
    model_config(asdict(model))
    validate_capacity(model, actions.max_sequence_length)
    integer(seed, 0, 2**31 - 1)
    if purpose not in ("toy", "trained") or purpose == "trained" and (provenance_data is None or provenance_data.get("real_gate_passed") is not True):
        raise ContractError("real training requires verified gate provenance")
    if type(charged_seconds) not in (int, float) or not math.isfinite(charged_seconds) or charged_seconds < 0:
        raise ContractError("invalid charged collection/ingestion cost")
    started = time.monotonic()
    deadline = started + training.wall_seconds - charged_seconds
    cleanup_seconds = training.cleanup_seconds
    if prepared_inputs is not None:
        deadline = min(deadline, prepared_inputs.session.wall_deadline)
        cleanup_seconds = max(cleanup_seconds, prepared_inputs.session.config.cleanup_seconds)
        if prepared_inputs.config != model.encoder or any(value is not None for value in (x, y, validation_x, validation_y)):
            raise ContractError("prepared numeric ownership/config mismatch")
        sample_count = prepared_inputs.shapes['x'][0]
        validation_sample_count = prepared_inputs.shapes['validation_x'][0]
    else:
        width = len(feature_names(model.encoder))
        x, validation_x = numeric_matrix(x, width), numeric_matrix(validation_x, width)
        if y is None or validation_y is None:
            raise ContractError("invalid fit labels")
        for features, labels in ((x, y), (validation_x, validation_y)):
            if labels.shape != (len(features),) or not np.isfinite(labels).all():
                raise ContractError("invalid fit labels")
        sample_count, validation_sample_count = len(y), len(validation_y)
    check_deadline(deadline - cleanup_seconds)
    directory.mkdir(parents=True, exist_ok=False)
    manifest: Json = {"schema_version": 2 if training.cost_mode == "fixed_data" else 1, "cost_mode": training.cost_mode, "algorithm": "lightgbm_mc", "status": "running", "purpose": purpose,
                      "seed": seed, "charged_prior_seconds": charged_seconds, "wall_seconds": training.wall_seconds,
                      "fit_code_provenance": fit_code_provenance if fit_code_provenance is not None else {
                          "verification": "unknown", "revision": None, "dirty": None, "installed_build_revision": None},
                      "fit_code_measurement_seconds": fit_code_measurement_seconds,
                      "source_provenance": provenance_data, "health": {}, "model": asdict(model), "training": asdict(training),
                      "action_config_hash": digest(asdict(actions)), "sample_count": sample_count, "validation_sample_count": validation_sample_count,
                      "label_recipe": "behavior-selected-natural-return-gamma1-v1", "checkpoint": None}
    process = None
    drainers: list[asyncio.Task[None]] = []
    original_error: BaseException | None = None
    ready_inference: Json | None = None
    try:
        atomic_json(directory / "training.json", manifest)
        check_deadline(deadline - cleanup_seconds)
        if prepared_inputs is None:
            assert x is not None and y is not None and validation_x is not None and validation_y is not None
            np.savez(directory / "arrays.npz", x=x, y=y, validation_x=validation_x, validation_y=validation_y)
        elif isinstance(prepared_inputs, NativePreparedInputs):
            await prepared_inputs.awrite_numeric(directory / "arrays.npz")
        else:
            prepared_inputs.write_numeric(directory / "arrays.npz")
        request: Json = {"model": {**asdict(model), "checkpoint": None}, "params": native_parameters(model, training, seed),
                   "rounds": training.num_boost_round, "early_stopping": training.early_stopping_rounds,
                   "purpose": purpose, "action_config_hash": digest(asdict(actions))}
        atomic_json(directory / "request.json", request)
        manifest["input_hashes"] = {"request_sha256": _sha256(directory / "request.json", deadline),
                                    "arrays_sha256": _sha256(directory / "arrays.npz", deadline)}
        atomic_json(directory / "input-hashes.json", manifest["input_hashes"])
        atomic_json(directory / "training.json", manifest)
        check_deadline(deadline - cleanup_seconds)
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        creation = asyncio.create_task(asyncio.create_subprocess_exec(
            *(command or (sys.executable, "-E", "-m", "gf_training.lightgbm_worker", str(directory.resolve()))),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=environment))
        launch_cancel: asyncio.CancelledError | None = None
        while not creation.done():
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError as exc:
                launch_cancel = launch_cancel or exc
        process = creation.result()
        assert process.stdout is not None and process.stderr is not None
        drainers = [asyncio.create_task(_drain(process.stdout, directory / "stdout.log", training.max_log_bytes, manifest["health"])),
                    asyncio.create_task(_drain(process.stderr, directory / "stderr.log", training.max_log_bytes, manifest["health"]))]
        if launch_cancel is not None:
            raise launch_cancel
        async with asyncio.timeout_at(deadline - cleanup_seconds):
            while process.returncode is None:
                if manifest["health"].get("log_limit_exceeded"):
                    raise RuntimeError("native log limit exceeded")
                await asyncio.sleep(min(.01, max(0, deadline - cleanup_seconds - time.monotonic())))
            await asyncio.gather(*drainers)
        if process.returncode != 0 or manifest["health"].get("log_limit_exceeded"):
            raise RuntimeError("native fit failed")
        proof = json.loads((directory / "child-result.json").read_text())
        closed(proof, {"metrics", "input_hashes", "validated_model_sha256", "validated_inference_sha256"})
        if proof["input_hashes"] != manifest["input_hashes"] or proof["validated_model_sha256"] != _sha256(directory / "staging" / "model.txt", deadline) or proof["validated_inference_sha256"] != _sha256(directory / "staging" / "inference.json", deadline):
            raise ContractError("native validation proof mismatch")
        metrics = closed(proof["metrics"], {"validation_l2", "best_iteration"})
        if type(metrics["validation_l2"]) not in (int, float) or not math.isfinite(metrics["validation_l2"]) or metrics["validation_l2"] < 0:
            raise ContractError("invalid native fit metrics")
        integer(metrics["best_iteration"], 1)
        check_deadline(deadline)
        inference = inspect_artifact(directory / "staging", model=model, action_config_hash=request["action_config_hash"], allow_toy=True, allow_unready=True, deadline=deadline)
        if inference["ready"] is not False:
            raise ContractError("native child must leave staging unready")
        ready_inference = {**inference, "ready": True}
        manifest.update(metrics=metrics, child_validation=proof,
                        inference_manifest_sha256=hashlib.sha256((canonical(ready_inference) + "\n").encode()).hexdigest())
    except BaseException as exc:  # noqa: BLE001 - owned child must always be reaped
        original_error = exc
        manifest.update(status="interrupted" if isinstance(exc, asyncio.CancelledError) else "failed", failure=type(exc).__name__, message=str(exc))
    finally:
        if process is not None:
            cleanup = asyncio.create_task(_stop(process, drainers, min(cleanup_seconds, max(0, deadline - time.monotonic())), manifest["health"]))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    original_error = original_error or exc
                    manifest.update(status="interrupted", failure="CancelledError")
            try:
                cleanup.result()
            except BaseException as exc:  # noqa: BLE001 - owned child must always be reaped
                manifest["health"]["cleanup_failure"] = str(exc)
                original_error = original_error or exc
    x = y = validation_x = validation_y = None
    if prepared_inputs is not None:
        try:
            if original_error is None and not any(manifest["health"].get(key) for key in
                                                  ("emergency_kill", "cleanup_failure", "cleanup_overrun_seconds", "drain_failure", "log_limit_exceeded")):
                if isinstance(prepared_inputs, NativePreparedInputs):
                    await prepared_inputs.afinalize()
                    await asyncio.sleep(0)  # Cancellation queued with completion forbids Ready.
                else:
                    prepared_inputs.finalize()
            else:
                if isinstance(prepared_inputs, NativePreparedInputs):
                    await prepared_inputs.aabort()
                else:
                    prepared_inputs.abort()
        except BaseException as exc:  # noqa: BLE001 - finalizer failure forbids publication
            original_error = original_error or exc
            manifest["health"]["preparation_cleanup_failure"] = str(exc)
            if isinstance(prepared_inputs, NativePreparedInputs):
                interrupted = isinstance(original_error, asyncio.CancelledError)
                manifest.update(status="interrupted" if interrupted else "failed",
                                failure=type(original_error).__name__, quarantine=True)
            else:
                manifest.update(status="failed", failure=type(exc).__name__, quarantine=True)
    def measured() -> None:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["charged_total_seconds"] = charged_seconds + manifest["elapsed_seconds"]
        manifest["timing_scope"] = "pre-write snapshot; excludes the write persisting this snapshot and later corrections"
    bad_health = any(manifest["health"].get(key) for key in ("emergency_kill", "cleanup_failure", "cleanup_overrun_seconds", "drain_failure", "log_limit_exceeded"))
    if time.monotonic() >= deadline or bad_health:
        manifest.update(status="failed", failure="budget_or_cleanup_failure", quarantine=True)
    if original_error is None and manifest["status"] == "running" and ready_inference is not None:
        try:
            check_deadline(deadline)
            # Child/drainer cleanup is complete. Moving an unready directory
            # publishes no usable checkpoint; readiness is the last promotion.
            (directory / "staging").rename(directory / "checkpoint")
            manifest.update(status="complete", checkpoint="checkpoint")
            measured()
            atomic_json(directory / "training.json", manifest)
            check_deadline(deadline)
            atomic_json(directory / "checkpoint" / "inference.json", ready_inference)
            check_deadline(deadline)
        except BaseException as exc:  # noqa: BLE001 - publication failure must invalidate both stores
            original_error = exc
            manifest.update(status="failed", failure=type(exc).__name__, quarantine=True)
    elif manifest["status"] == "running":
        manifest.update(status="failed", failure="missing_validated_stage", quarantine=True)
    if manifest["status"] != "complete":
        # Independent stores: either durable failure blocks formal qualification.
        # Both failing remains explicitly quarantined under filesystem debt.
        measured()
        try:
            atomic_json(directory / "training.json", manifest)
        except OSError as exc:
            manifest.setdefault("correction_save_failures", []).append(str(exc))
        for path in (directory / "checkpoint" / "inference.json", directory / "staging" / "inference.json"):
            try:
                if path.exists():
                    artifact = json.loads(path.read_text())
                    artifact["ready"] = False
                    atomic_json(path, artifact)
            except (OSError, ValueError) as exc:
                manifest.setdefault("invalidation_failures", []).append(str(exc))
        measured()
        try:
            atomic_json(directory / "training.json", manifest)
        except OSError as exc:
            manifest.setdefault("correction_save_failures", []).append(str(exc))
    # Return/raise timing includes all persistence and correction attempts. A
    # persisted record truthfully declares its earlier snapshot scope.
    measured()
    manifest["timing_scope"] = "returned measurement after all persistence and correction attempts"
    if original_error is not None:
        raise original_error
    if manifest["status"] != "complete":
        raise RuntimeError("training failed/quarantined")
    return manifest


async def run_numeric(directory: Path, x: Array, y: Array, validation_x: Array, validation_y: Array, *,
                      model: ModelConfig, actions: ActionConfig, training: TrainingConfig, seed: int,
                      charged_seconds: float = 0, provenance_data: Json | None = None,
                      command: tuple[str, ...] | None = None) -> Json:
    """Explicitly synthetic component runner; it can only produce toy artifacts."""
    return await _run_numeric(directory, x, y, validation_x, validation_y, model=model, actions=actions,
                              training=training, seed=seed, purpose="toy", charged_seconds=charged_seconds,
                              provenance_data=provenance_data, command=command)


async def train_experiments(loaded: LoadedConfig, mode: str = "fresh", *,
                            code_provenance_provider: Callable[[float], Json] | None = None) -> Json:
    from gf_common.action_contract import require_qualified_contract
    from gf_common.structured import ActionConfigV2
    if isinstance(loaded.config.actions, ActionConfigV2):
        require_qualified_contract(loaded.config.actions.action_contract)
    if mode != "fresh":
        raise ContractError("resume is unsupported; no work performed")
    config = loaded.config
    if config.model.model_id != "lightgbm" or config.model.checkpoint is not None:
        raise ContractError("fresh lightgbm model configuration required")
    training = config.training
    if not isinstance(training, TrainingConfig) or not isinstance(config.model, ModelConfig):
        raise ContractError("LightGBM training requires closed LightGBM recipe")
    model, actions = config.model, config.actions
    if not isinstance(actions, ActionConfig):
        raise ContractError("legacy LightGBM training requires actions1")
    if training.cost_mode == "fixed_data":
        return await train_fixed_data(loaded, code_provenance_provider=code_provenance_provider)
    if config.preprocessing.backend != 'python':
        raise ContractError('legacy inclusive-generation training requires explicit Python preparation')
    if training.pilot_run is None:
        raise ContractError("clean original pilot evidence required before fitting")
    gate_started = time.monotonic()
    ingestion_deadline = gate_started + training.wall_seconds - training.cleanup_seconds
    pilot, proof = feasibility(Path(training.pilot_run), deadline=ingestion_deadline)
    gate_seconds = time.monotonic() - gate_started
    if training.accepted_decision_target is None or not training.train_runs or not training.validation_runs:
        raise ContractError("explicit decision target and complete train/validation sources required")
    ingest_started = time.monotonic()
    if len(set(training.train_runs)) != len(training.train_runs) or len(set(training.validation_runs)) != len(training.validation_runs):
        raise ContractError("duplicate training source reference")
    train = [load_run(Path(path), "train", deadline=ingestion_deadline) for path in training.train_runs]
    validation = [load_run(Path(path), "validation", deadline=ingestion_deadline) for path in training.validation_runs]
    if any(bundle.manifest["schema_version"] != 3 for bundle in (*train, *validation)):
        raise ContractError("inclusive-generation LightGBM accepts only legacy schema-3 sources; PPO sources require fixed_data whole-parent validation")
    collection = {bundle.directory: bundle for bundle in (*train, *validation)}
    all_bundles = {**collection, pilot.directory: pilot}
    rows, evidence = validate_bundles(list(all_bundles.values()), deadline=ingestion_deadline)
    if proof["action_config_hash"] != digest(asdict(config.actions)) or any(source["header"]["action_config_hash"] != proof["action_config_hash"] for source in evidence["sources"]):
        raise ContractError("training/pilot/search conditions mismatch")
    train_ids = {bundle.manifest["run_id"] for bundle in train}
    validation_ids = {bundle.manifest["run_id"] for bundle in validation}
    train_samples = [sample for sample in labeled_samples(rows, evidence, "train", deadline=ingestion_deadline) if sample.identity[0] in train_ids]
    validation_samples = [sample for sample in labeled_samples(rows, evidence, "validation", deadline=ingestion_deadline) if sample.identity[0] in validation_ids]
    target = training.accepted_decision_target
    if not target <= len(train_samples) <= target * (1 + training.max_overshoot):
        raise ContractError("accepted-decision count outside explicit target interval")
    if not validation_samples:
        raise ContractError("validation samples required")
    def features(samples: list[Any]) -> Array:
        vectors = []
        for sample in samples:
            check_deadline(ingestion_deadline)
            vectors.append(encode(sample.value, sample.action, model.encoder))
        check_deadline(ingestion_deadline)
        return np.stack(vectors)
    x = features(train_samples)
    validation_x = features(validation_samples)
    y = np.asarray([sample.target for sample in train_samples], dtype=np.float64)
    validation_y = np.asarray([sample.target for sample in validation_samples], dtype=np.float64)
    code_started = time.monotonic()
    check_deadline(ingestion_deadline)
    fit_code = code_provenance_provider(ingestion_deadline) if code_provenance_provider is not None else None
    check_deadline(ingestion_deadline)
    code_seconds = time.monotonic() - code_started
    historical = sum(bundle.collection_seconds for bundle in collection.values())
    collection_provenance = []
    for bundle in collection.values():
        check_deadline(ingestion_deadline)
        collection_provenance.append(provenance(bundle, deadline=ingestion_deadline))
    pilot_provenance = provenance(pilot, deadline=ingestion_deadline)
    source = {"real_gate_passed": True, "gate": proof, "shared_feasibility_seconds": gate_seconds,
              "historical_collection_seconds": historical,
              "collection": collection_provenance, "pilot": pilot_provenance,
              "source_projection": evidence, "effective_config_sha256": loaded.sha256}
    # Freeze the shared preparation charge only after every source identity has
    # been built/hashed under the original preparation deadline. Each model seed
    # still receives its own remaining budget, rather than a prior seed's clock.
    check_deadline(ingestion_deadline)
    ingest_seconds = time.monotonic() - ingest_started
    source["ingestion_feature_seconds"] = ingest_seconds
    charged = historical + ingest_seconds
    if charged + training.cleanup_seconds >= training.wall_seconds:
        raise ContractError("historical collection/ingestion exhausted original training budget")
    check_deadline(ingestion_deadline)
    results = []
    for seed in training.seeds:
        path = Path(config.run.output_dir) / f"{config.run.name}-training-{seed}-{uuid.uuid4().hex[:12]}"
        result = await _run_numeric(path, x, y, validation_x, validation_y, model=model, actions=actions,
                                   training=training, seed=seed, purpose="trained", charged_seconds=charged, provenance_data=source,
                                   fit_code_provenance=fit_code, fit_code_measurement_seconds=code_seconds)
        results.append({"directory": str(path), "status": result["status"], "seed": seed})
    return {"training_runs": results, "shared_feasibility_seconds": gate_seconds}


async def train_fixed_data(loaded: LoadedConfig, *, code_provenance_provider: Callable[[float], Json] | None = None) -> Json:
    from gf_common.action_contract import require_qualified_contract
    from gf_common.structured import ActionConfigV2
    if isinstance(loaded.config.actions, ActionConfigV2):
        require_qualified_contract(loaded.config.actions.action_contract)
    from gf_training.fixed_data import prepare_fixed_inputs
    config = loaded.config
    actions = config.actions
    if not isinstance(actions, ActionConfig):
        raise ContractError("legacy fixed-data LightGBM requires actions1")
    training, model = config.training, config.model
    if not isinstance(training, TrainingConfig) or not isinstance(model, ModelConfig) or model.model_id != "lightgbm" or model.checkpoint is not None or training.cost_mode != "fixed_data" or training.dataset_manifest is None:
        raise ContractError("fixed-data LightGBM requires an explicit frozen dataset manifest")
    if training.train_runs or training.validation_runs or training.pilot_run is not None or training.accepted_decision_target is not None:
        raise ContractError("fixed-data mode cannot mix inclusive-generation source controls")
    results = []
    shared_generation = None
    for seed in training.seeds:
        started = time.monotonic()
        wall_deadline = started + training.wall_seconds
        cleanup_seconds = max(training.cleanup_seconds, config.preprocessing.cleanup_seconds)
        deadline = wall_deadline - cleanup_seconds
        async with AsyncExitStack() as scope:
            prepared: PreparedInputs | NativePreparedInputs
            if config.preprocessing.backend == 'rust':
                native_prepared = await NativePreparedInputs.create_async(Path(training.dataset_manifest), model.encoder,
                                                                         config.preprocessing, deadline=deadline,
                                                                         wall_deadline=wall_deadline)
                prepared = await scope.enter_async_context(native_prepared)
            else:
                python_prepared = prepare_fixed_inputs(Path(training.dataset_manifest), model.encoder,
                                                       preprocessing=config.preprocessing,
                                                       deadline=deadline, wall_deadline=wall_deadline)
                if not isinstance(python_prepared, PreparedInputs):
                    raise ContractError('explicit Python fixed-data preparation required')
                prepared = scope.enter_context(python_prepared)
            dataset = prepared.dataset
            if dataset["action_config_hash"] != digest(asdict(config.actions)):
                raise ContractError("fixed dataset/search conditions mismatch")
            code_started = time.monotonic()
            fit_code = code_provenance_provider(prepared.session.deadline) if code_provenance_provider is not None else None
            check_deadline(prepared.session.deadline)
            code_seconds = time.monotonic() - code_started
            if isinstance(prepared, NativePreparedInputs):
                manifest_hash = await prepared.session.ahash_file(Path(training.dataset_manifest))
            else:
                manifest_hash = prepared.session.hash_file(Path(training.dataset_manifest))
            source = {"real_gate_passed": True, "fixed_dataset_content_sha256": dataset["content_sha256"],
                      "dataset_manifest_sha256": manifest_hash,
                      "shared_generation_seconds": dataset["shared_generation_seconds"],
                      "shared_generation_cost_key": dataset["shared_generation_cost_key"],
                      "shared_generation_cost_scope": "deduplicated whole PPO parents and final validation, reported separately, excluded from this fit budget",
                      "comparison_claim": "fixed selected source set; no equal-total-cost PPO comparison",
                      "preprocessing": prepared.diagnostics}
            prep_seconds = time.monotonic() - started
            path = Path(config.run.output_dir) / f"{config.run.name}-fixed-training-{seed}-{uuid.uuid4().hex[:12]}"
            result = await _run_numeric(path, None, None, None, None, model=model, actions=actions,
                                        training=training, seed=seed, purpose="trained", charged_seconds=prep_seconds,
                                        provenance_data=source, fit_code_provenance=fit_code,
                                        fit_code_measurement_seconds=code_seconds, prepared_inputs=prepared)
            results.append({"directory": str(path), "seed": seed, "status": result["status"]})
            shared_generation = dataset["shared_generation_seconds"]
    return {"training_runs": results, "shared_generation_seconds": shared_generation,
            "shared_generation_cost_key": dataset["shared_generation_cost_key"],
            "generation_cost_accounting": "reported once across all fits, excluded from per-fit original 7200s"}
