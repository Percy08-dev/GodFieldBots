"""Corrected game composition; runtime evidence stays outside participants."""
from __future__ import annotations

import asyncio
import hashlib
import stat
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_cli.config import LoadedConfig, RunConfigV2
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_validation import reconcile_replay, replay_closure
from gf_common.domain import (
    ContractError,
    EpisodeSpec,
    GroupResult,
    Json,
    RuntimePolicyBinding,
    canonical,
    digest,
)
from gf_common.schedule import schedule, serialize_schedule
from gf_evaluation.evaluation import report
from gf_models.config import ModelConfig, PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.environment import EnvironmentSettings
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import (
    CollectionPublication,
    collection_metadata,
    ppo_collection_metadata,
)
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal
from gf_training.collection import CollectionPlan, FinalizedCollection


def _check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("corrected collection deadline")


def _require_actions(loaded: LoadedConfig) -> None:
    config = loaded.config
    if (type(config.actions) is not ActionConfigV3
            or config.actions.action_contract != corrected_card_instance_contract()
            or config.actions.max_sequence_length > 18 or not isinstance(config.run, RunConfigV2)):
        raise ContractError("corrected baseline collection requires the corrected contract and real18card limits")


def _baseline_labels(loaded: LoadedConfig, *, artifact_inputs: Any = None) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    _require_actions(loaded)
    config = loaded.config
    labels = []
    bindings = {}
    for model in (config.model, config.opponent):
        if type(model) is ConfiguredNeuralModelV3 and artifact_inputs is not None:
            label, binding = artifact_inputs.binding(model)
            labels.append(label)
            bindings[label] = binding
            continue
        if (type(model) is not ModelConfig or model.model_id not in ("random", "heuristic")
                or asdict(model) != asdict(ModelConfig(model_id=model.model_id))):
            raise ContractError("corrected collection currently supports canonical Random/Heuristic only")
        label = model.model_id + "-card-instance-v2"
        labels.append(label)
        bindings[label] = RuntimePolicyBinding(model.model_id)
    return bindings, (labels[0], labels[1])


def _inventory(directory: Path, deadline: float) -> list[Json]:
    result = []
    for path in sorted(directory.rglob("*")):
        _check(deadline)
        before = path.lstat()
        if stat.S_ISDIR(before.st_mode):
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("corrected collection contains non-regular evidence")
        value = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024**2):
                _check(deadline)
                value.update(chunk)
        after = path.stat()
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if fields(before) != fields(after):
            raise ContractError("collection evidence changed during final inventory")
        result.append({"path": path.relative_to(directory).as_posix(), "sha256": value.hexdigest(), "bytes": before.st_size})
    return result


def _validate_recorded_episodes(directory: Path, episodes: list[Any], actions: ActionConfigV3,
                                run_id: str, deadline: float,
                                expected: dict[str, tuple[EpisodeSpec, Json]],
                                neural_metadata: dict[str, Json] | None = None) -> None:
    import json

    for episode in episodes:
        _check(deadline)
        spec, runtime = expected[episode.episode_id]
        closures = []
        for viewer in (1, 2):
            folder = directory / episode.episode_id
            journal = read_card_instance_journal(folder / f"player-{viewer}.jsonl", actions=actions, deadline=deadline)
            if (journal.header["run_id"] != run_id or journal.header["game_id"] != episode.game_id
                    or journal.header["episode_id"] != episode.episode_id or journal.header["viewer"] != viewer
                    or journal.header["policy"] != episode.policies[viewer - 1]
                    or journal.header["policy"] != spec.policies[viewer - 1]
                    or journal.header["sampling_seed"] != spec.sampling_seeds[viewer - 1]
                    or canonical(journal.header["runtime_evidence"]) != canonical(runtime)
                    or journal.end["end_kind"] != episode.end_kind or journal.end["outcome"] != episode.outcome):
                raise ContractError("persisted corrected journal/episode identity mismatch")
            if (neural_metadata is not None and journal.header["policy"] in neural_metadata
                    and canonical(journal.header.get("neural")) != canonical(neural_metadata[journal.header["policy"]])):
                raise ContractError("persisted neural journal differs from its retained artifact input")
            accepted = [journal.decisions[index].command["action"] for index in journal.accepted]
            value = json.loads((folder / f"replay-player-{viewer}.json").read_bytes())
            reconcile_replay(value, viewer, accepted, episode.outcome)
            closures.append(replay_closure(value, episode.game_id, viewer))
        if closures[0] != closures[1]:
            raise ContractError("persisted corrected participant Replay closures disagree")


async def run_baseline(loaded: LoadedConfig, command: str) -> tuple[Path, Json]:
    # Validate all unsupported combinations before filesystem/process ownership.
    if command not in ("smoke", "collect", "evaluate"):
        raise ContractError("corrected baseline integration supports smoke/collect/evaluate only")
    bindings, labels = _baseline_labels(loaded)
    return await _run_corrected(loaded, command, bindings, labels)


async def run_inference(loaded: LoadedConfig, command: str) -> tuple[Path, Json]:
    if command not in ("smoke", "collect", "evaluate"):
        raise ContractError("corrected neural inference supports smoke/collect/evaluate only")
    _require_actions(loaded)
    config = loaded.config
    assert type(config.actions) is ActionConfigV3 and isinstance(config.run, RunConfigV2)
    neural = tuple(model for model in (config.model, config.opponent)
                   if type(model) is ConfiguredNeuralModelV3)
    if not neural or any(model.checkpoint is None for model in neural):
        raise ContractError("corrected neural inference requires an explicit game checkpoint")
    for model in (config.model, config.opponent):
        if type(model) is ConfiguredNeuralModelV3:
            continue
        if (type(model) is not ModelConfig or model.model_id not in ("random", "heuristic")
                or asdict(model) != asdict(ModelConfig(model_id=model.model_id))):
            raise ContractError("corrected neural inference requires model3 or canonical baselines")
    if config.server.external_url is not None or config.server.source_dir is None:
        raise ContractError("corrected neural inference requires locally managed verified source/binary evidence")
    started = time.monotonic()
    work_deadline = started + config.run.wall_seconds - config.run.report_reserve_seconds
    preflight_deadline = min(started + config.run.execution_seconds,
                             work_deadline - config.server.cleanup_reserve_seconds)
    evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
                                          deadline=preflight_deadline)
    evidence.preflight()
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_game_workers_v3
    rules = card_instance_rules()
    owner = RetainedGameArtifacts(neural, config.actions, rules=rules,
        max_owned_bytes=min(model.resources.max_owned_bytes for model in neural),
        preflight_deadline=preflight_deadline, finalization_deadline=work_deadline)
    try:
        bindings, labels = _baseline_labels(loaded, artifact_inputs=owner)
        admission = admit_game_workers_v3(config.actions, (bindings[labels[0]], bindings[labels[1]]),
            rules=rules, game_concurrency=config.run.concurrency,
            global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers)
        return await _run_corrected(loaded, command, bindings, labels, started=started,
                                    artifact_inputs=owner, runtime_evidence=evidence,
                                    resource_admission=admission)
    finally:
        owner.release()  # no filesystem I/O or reportable operation after healthy publication.


async def _run_corrected(loaded: LoadedConfig, command: str,
                         bindings: dict[str, RuntimePolicyBinding], labels: tuple[str, ...], *,
                         started: float | None = None, artifact_inputs: Any = None,
                         runtime_evidence: CardInstanceRuntimeEvidence | None = None,
                         resource_admission: Json | None = None,
                         collection: CollectionPlan | None = None,
                         owned_ledger: ReservationLedger | None = None) -> tuple[Path, Json]:
    config = loaded.config
    assert isinstance(config.actions, ActionConfigV3)
    if config.server.external_url is not None or config.server.source_dir is None:
        raise ContractError("corrected baseline collection requires locally managed verified source/binary evidence")
    started = time.monotonic() if started is None else started
    wall_deadline = started + config.run.wall_seconds if collection is None else collection.wall_deadline
    work_deadline = wall_deadline - config.run.report_reserve_seconds
    replay_deadline = work_deadline - config.server.cleanup_reserve_seconds
    execution_deadline = started + config.run.execution_seconds if collection is None else collection.execution_deadline
    preflight_deadline = min(execution_deadline, replay_deadline)
    evidence = runtime_evidence or CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), deadline=preflight_deadline)
    evidence.preflight()
    planned = (schedule(config.run.seeds, config.run.sampling_seeds, config.run.pairs_per_seed,
                        labels[0], labels[1], deadline=preflight_deadline)
               if collection is None else collection.groups)
    if collection is not None:
        assert owned_ledger is not None
        _validate_online_collection(loaded, collection, owned_ledger, preflight_deadline)
    run_id = str(uuid.uuid4())
    directory = (Path(config.run.output_dir) / (config.run.name + "-" + run_id)
                 if collection is None else collection.output)
    directory.mkdir(parents=True, exist_ok=False)
    settings = EnvironmentSettings(binary=config.server.binary, server_config=config.server.config,
        output_dir=directory, run_id=run_id, actions=config.actions,
        fixed_seed_experiment=config.server.fixed_seed_experiment,
        startup_seconds=config.server.startup_seconds, request_seconds=config.server.request_seconds,
        worker_seconds=config.server.worker_seconds, shutdown_seconds=config.server.shutdown_seconds,
        transport_retries=config.server.transport_retries, max_decisions=config.run.max_decisions,
        admission_deadline=started + config.run.admission_seconds if collection is None else execution_deadline,
        execution_deadline=execution_deadline,
        wall_deadline=wall_deadline, work_deadline=work_deadline, replay_deadline=replay_deadline,
        policy_bindings=bindings)
    assert isinstance(config.run, RunConfigV2)
    ledger = (ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
              if owned_ledger is None else owned_ledger)
    extra: dict[str, Any] = {} if artifact_inputs is None else {"artifact_inputs": artifact_inputs}
    semaphore = asyncio.Semaphore(config.run.concurrency)
    tasks: list[asyncio.Task[GroupResult]] = []
    admission_closed = False
    diagnostics: Json = {}

    async def group_run(group: Any) -> GroupResult:
        nonlocal admission_closed
        async with semaphore:
            if admission_closed:
                return GroupResult([], {"failure": "admission_closed"})
            try:
                return await environment.run_group(group)
            except BaseException:
                admission_closed = True
                raise

    try:
        environment = CardInstanceSimulatorEnvironment(settings, CardInstanceJournal, ledger=ledger,
                                                      runtime_evidence=evidence, **extra)
        tasks = [asyncio.create_task(group_run(group)) for group in planned]
        groups = await asyncio.gather(*tasks)
        episodes = [episode for group in groups for episode in group.episodes]
        summary = report(episodes, planned, bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed, wall_seconds=time.monotonic() - started,
            groups_health=[group.health for group in groups], deadline=work_deadline)
        # Every runtime clone checks its executed process and finalizes after reap.
        evidence.deadline = work_deadline
        evidence.preflight()
        _check(work_deadline)
        if any(group.health.get("runtime_evidence_failure")
               or "runtime_evidence" not in group.health for group in groups):
            raise ContractError("corrected runtime evidence failed or is missing after group cleanup")
        if collection is None:
            if ledger.workers or ledger.learner is not None or ledger.updating or ledger.cuda_bytes:
                raise ContractError("corrected collection has unreaped resource ownership before publication")
        else:
            _validate_online_collection(loaded, collection, ledger, work_deadline)
            if ledger.workers or ledger.updating:
                raise ContractError("PPO collection has unreaped participant/update ownership")
        expected = {spec.episode_id: (spec, group.health["runtime_evidence"])
                    for plan, group in zip(planned, groups, strict=True) for spec in plan.episodes}
        neural_metadata = ({label: artifact_inputs.journal_metadata(label) for label, binding in bindings.items()
                            if binding.model_id in ("mlp", "gru", "transformer")}
                           if artifact_inputs is not None else None)
        _validate_recorded_episodes(directory, episodes, config.actions, run_id, work_deadline, expected,
                                   neural_metadata=neural_metadata)
        neural_inputs = artifact_inputs.finalize(deadline=work_deadline) if artifact_inputs is not None else None
        inventory = _inventory(directory, work_deadline)
        from gf_cli import _code_provenance
        producer = _code_provenance(work_deadline)
        receipts = [group.health["runtime_evidence"] for group in groups]
        metadata_options: dict[str, Any] = ({} if neural_inputs is None else
            {"policy_bindings": bindings, "neural_inputs": neural_inputs})
        metadata_producer = collection_metadata if collection is None else ppo_collection_metadata
        if collection is not None:
            metadata_options["linkage"] = collection.linkage
        manifest = metadata_producer(actions=config.actions, config=loaded.effective,
            runtime_evidence=receipts, schedule=serialize_schedule(planned, 4), inventory=inventory,
            policies=list(labels), bots_revision=producer["revision"], **metadata_options)
        if resource_admission is not None:
            manifest["resource_admission"] = resource_admission
        manifest.update(kind="card-instance-collection", run_id=run_id, command=command,
            status="complete" if summary["comparison_valid"] else "incomplete", schedule_schema_version=4,
            code=producer, inputs=loaded.inputs, groups=[group.health for group in groups],
            episodes=[asdict(episode) for episode in episodes])
        summary.update(healthy=summary["comparison_valid"], artifact_bytes=sum(row["bytes"] for row in inventory))
        summary["wall_seconds"] = time.monotonic() - started
        if not summary["healthy"]:
            # Diagnostic outputs have no healthy manifest/report completion names.
            _write_diagnostic(directory, "incomplete.json", {"manifest": manifest, "report": summary})
            return directory, summary
        manifest["report_sha256"] = hashlib.sha256((canonical(summary) + "\n").encode()).hexdigest()
        CollectionPublication(directory, work_deadline=work_deadline, cleanup_deadline=wall_deadline).commit(summary, manifest)
        return directory, summary  # no reportable checks after publication owner releases.
    except BaseException as error:
        from gf_cli import _join_owned_groups
        await _join_owned_groups(tasks, wall_deadline, diagnostics)
        try:
            _write_diagnostic(directory, "failure.json", {"error": type(error).__name__, "message": str(error), **diagnostics})
        except OSError:
            pass
        raise


def _write_diagnostic(directory: Path, name: str, value: Json) -> None:
    with (directory / name).open("x", encoding="utf-8") as output:
        output.write(canonical(value) + "\n")


def _validate_online_collection(loaded: LoadedConfig, plan: CollectionPlan,
                                 ledger: ReservationLedger, deadline: float) -> Any:
    from gf_training.card_instance_source import validate_owned_collection
    if not isinstance(loaded.config.training, PPOTrainingConfig):
        raise ContractError("online card-instance collection requires PPO training")
    online = validate_owned_collection(plan, ledger,
        max_owned_bytes=min(loaded.config.training.max_owned_bytes,
                            loaded.config.preprocessing.memory_budget_mib * 1024**2),
        deadline=deadline)
    effective = asdict(loaded.config)
    if (loaded.sha256 != online.state["config_sha256"] or digest(effective) != loaded.sha256
            or canonical(loaded.effective) != canonical(effective)
            or canonical(effective) != canonical(online.state["effective_config"])
            or canonical(list(loaded.inputs)) != canonical(online.state["config_inputs"])):
        raise ContractError("online collection caller differs from SAME parent Config/input origins")
    return online


async def collect_ppo(loaded: LoadedConfig, plan: CollectionPlan,
                      ledger: ReservationLedger) -> FinalizedCollection:
    """Same live parent's frozen plan; metadata cannot mint admission."""
    _require_actions(loaded)
    config = loaded.config
    assert type(config.actions) is ActionConfigV3 and isinstance(config.run, RunConfigV2)
    if (type(config.model) is not ConfiguredNeuralModelV3
            or not isinstance(config.training, PPOTrainingConfig)
            or config.run.split != "train" or config.preprocessing.backend not in ("python", "rust")
            or config.preprocessing.reader != "participant-structured-v2"):
        raise ContractError("online card-instance collector requires explicit model3/participant-structured-v2 PPO")
    online = _validate_online_collection(loaded, plan, ledger, plan.execution_deadline)
    if config.preprocessing.backend == "rust":
        from gf_training.native_backend import card_instance_reader_settings
        card_instance_reader_settings(config.preprocessing)
    if config.server.external_url is not None or config.server.source_dir is None:
        raise ContractError("PPO collector requires locally managed verified runtime")
    started = time.monotonic() if plan.started_monotonic is None else plan.started_monotonic
    work_deadline = plan.wall_deadline - config.run.report_reserve_seconds
    preflight_deadline = min(plan.execution_deadline, work_deadline - config.server.cleanup_reserve_seconds)
    evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
                                          deadline=preflight_deadline)
    evidence.preflight()
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_game_workers_v3
    rules = card_instance_rules()
    owner = RetainedGameArtifacts.for_registry(plan.bindings, config.actions, rules=rules,
        max_owned_bytes=min(config.training.max_owned_bytes, config.model.resources.max_owned_bytes,
                            config.preprocessing.memory_budget_mib * 1024**2),
        preflight_deadline=preflight_deadline, finalization_deadline=work_deadline)
    try:
        admissions = [admit_game_workers_v3(config.actions,
            (plan.bindings[spec.policies[0]], plan.bindings[spec.policies[1]]), rules=rules,
            game_concurrency=config.run.concurrency, global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers, learner_cuda_bytes=ledger.cuda_bytes)
            for group in plan.groups for spec in group.episodes]
        if not admissions:
            raise ContractError("empty frozen PPO collection schedule")
        # Keep the validated borrowed live context reachable until finalization.
        assert online is not None
        directory, summary = await _run_corrected(loaded, "collect", plan.bindings,
            tuple(plan.bindings), started=started, artifact_inputs=owner, runtime_evidence=evidence,
            resource_admission=max(admissions, key=lambda value: value["peak_cuda_bytes"]),
            collection=plan, owned_ledger=ledger)
        return FinalizedCollection(directory, summary)
    finally:
        owner.release()
