"""Corrected baseline composition; runtime evidence stays outside participants."""
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
)
from gf_common.schedule import schedule, serialize_schedule
from gf_evaluation.evaluation import report
from gf_models.config import ModelConfig
from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.environment import EnvironmentSettings
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import CollectionPublication, collection_metadata
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal


def _check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("corrected collection deadline")


def _baseline_labels(loaded: LoadedConfig) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    config = loaded.config
    if (type(config.actions) is not ActionConfigV3
            or config.actions.action_contract != corrected_card_instance_contract()
            or config.actions.max_sequence_length > 18 or not isinstance(config.run, RunConfigV2)):
        raise ContractError("corrected baseline collection requires the corrected contract and real18card limits")
    labels = []
    bindings = {}
    for model in (config.model, config.opponent):
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
                                expected: dict[str, tuple[EpisodeSpec, Json]]) -> None:
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
    config = loaded.config
    assert isinstance(config.actions, ActionConfigV3)
    if config.server.external_url is not None or config.server.source_dir is None:
        raise ContractError("corrected baseline collection requires locally managed verified source/binary evidence")
    started = time.monotonic()
    wall_deadline = started + config.run.wall_seconds
    work_deadline = wall_deadline - config.run.report_reserve_seconds
    replay_deadline = work_deadline - config.server.cleanup_reserve_seconds
    execution_deadline = started + config.run.execution_seconds
    preflight_deadline = min(execution_deadline, replay_deadline)
    evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), deadline=preflight_deadline)
    evidence.preflight()
    planned = schedule(config.run.seeds, config.run.sampling_seeds, config.run.pairs_per_seed,
                       labels[0], labels[1], deadline=preflight_deadline)
    run_id = str(uuid.uuid4())
    directory = Path(config.run.output_dir) / (config.run.name + "-" + run_id)
    directory.mkdir(parents=True, exist_ok=False)
    settings = EnvironmentSettings(binary=config.server.binary, server_config=config.server.config,
        output_dir=directory, run_id=run_id, actions=config.actions,
        fixed_seed_experiment=config.server.fixed_seed_experiment,
        startup_seconds=config.server.startup_seconds, request_seconds=config.server.request_seconds,
        worker_seconds=config.server.worker_seconds, shutdown_seconds=config.server.shutdown_seconds,
        transport_retries=config.server.transport_retries, max_decisions=config.run.max_decisions,
        admission_deadline=started + config.run.admission_seconds, execution_deadline=execution_deadline,
        wall_deadline=wall_deadline, work_deadline=work_deadline, replay_deadline=replay_deadline,
        policy_bindings=bindings)
    assert isinstance(config.run, RunConfigV2)
    ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
    environment = CardInstanceSimulatorEnvironment(settings, CardInstanceJournal, ledger=ledger, runtime_evidence=evidence)
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
        if ledger.workers or ledger.learner is not None or ledger.updating or ledger.cuda_bytes:
            raise ContractError("corrected collection has unreaped resource ownership before publication")
        expected = {spec.episode_id: (spec, group.health["runtime_evidence"])
                    for plan, group in zip(planned, groups, strict=True) for spec in plan.episodes}
        _validate_recorded_episodes(directory, episodes, config.actions, run_id, work_deadline, expected)
        inventory = _inventory(directory, work_deadline)
        from gf_cli import _code_provenance
        producer = _code_provenance(work_deadline)
        receipts = [group.health["runtime_evidence"] for group in groups]
        manifest = collection_metadata(actions=config.actions, config=loaded.effective,
            runtime_evidence=receipts, schedule=serialize_schedule(planned, 4), inventory=inventory,
            policies=list(labels), bots_revision=producer["revision"])
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
