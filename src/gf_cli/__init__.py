"""Composition root and command line. Optional ML libraries are never eager imports."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from gf_cli.config import LoadedConfig, load_config
from gf_common.action_contract import require_qualified_contract
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    ActionConfig,
    ContractError,
    GroupResult,
    Json,
    canonical,
    digest,
)
from gf_common.preprocessing import PreparationConfig
from gf_common.schedule import serialize_schedule
from gf_common.structured import ActionConfigV2
from gf_evaluation.evaluation import report, schedule
from gf_models.baselines import make_policy
from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment
from gf_simulator_client.process import server_effective
from gf_training.collection import CollectionPlan, FinalizedCollection
from gf_training.journal import (
    FEATURE_VERSION,
    JOURNAL_SCHEMA,
    REWARD_VERSION,
    ParticipantJournal,
    atomic_json,
)


def _check_budget(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("run budget exhausted")


def _git(directory: Path, deadline: float = float("inf")) -> Json:
    _check_budget(deadline)
    try:
        revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True, timeout=min(5, max(0.001, deadline - time.monotonic())) ).stdout.strip()
        _check_budget(deadline)
        dirty = bool(subprocess.run(["git", "-C", str(directory), "status", "--porcelain"],
                                   check=True, capture_output=True, text=True, timeout=min(5, max(0.001, deadline - time.monotonic())) ).stdout)
        _check_budget(deadline)
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        _check_budget(deadline)
        return {"revision": None, "dirty": None}



def _code_provenance(deadline: float = float("inf")) -> Json:
    _check_budget(deadline)
    source = Path(__file__).resolve()
    root = source.parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src/gf_cli/__init__.py").resolve() == source:
        return {**_git(root, deadline), "installed_build_revision": None}
    return {"revision": None, "dirty": None, "installed_build_revision": None}


def _fit_code_provenance(deadline: float, *, preprocessing_backend: str = "python",
                         preprocessing_config: PreparationConfig | None = None) -> Json:
    """Current fit identity, measured only in the parent's preparation budget."""
    selected_preprocessing = preprocessing_config or PreparationConfig(backend=preprocessing_backend)
    if selected_preprocessing.backend != preprocessing_backend:
        raise ValueError("preprocessing backend/config mismatch")
    if selected_preprocessing.backend == "python" and selected_preprocessing.reader != "legacy":
        raise ValueError("typed preprocessing reader requires the Rust backend")
    import importlib.metadata
    import importlib.util
    provenance = _code_provenance(deadline)
    try:
        provenance["package_version"] = importlib.metadata.version("godfieldbots")
    except importlib.metadata.PackageNotFoundError:
        provenance["package_version"] = None
    # The fingerprint names its exact scope; it does not claim a wheel build
    # revision or equate a dirty Git HEAD with uncommitted fit code bytes.
    from gf_common import preprocessing, validation
    from gf_models import config as common_config
    from gf_models import encoder as common_encoder
    from gf_models.lightgbm import config, encoder, model
    from gf_training import (
        fixed_data,
        journal,
        lightgbm_run,
        lineage,
        preparation,
        preparation_index,
        preparation_numeric,
        q_data,
    )
    provenance["fit_module_sha256"] = {module.__name__: _file_hash(module.__file__, deadline) if module.__file__ is not None else None
                                        for module in (config, encoder, common_config, common_encoder, model, lightgbm_run,
                                                       preprocessing, validation, fixed_data, journal, lineage, preparation,
                                                       preparation_index, preparation_numeric, q_data)}
    worker_spec = importlib.util.find_spec("gf_training.lightgbm_worker")
    worker_file = worker_spec.origin if worker_spec is not None else None
    provenance["fit_module_sha256"]["gf_training.lightgbm_worker"] = _file_hash(worker_file, deadline) if worker_file is not None else None
    if preprocessing_backend == "rust":
        from gf_training.native_backend import binary_identity, load_native, reader_settings
        native = load_native()
        reader_settings(selected_preprocessing, native)
        provenance["native_preprocessing"] = binary_identity(native)
        provenance["reader_contract"] = selected_preprocessing.reader
        for name in ("native_backend", "native_authorization", "native_execution", "native_learning",
                     "native_preparation", "native_index", "native_ppo"):
            spec = importlib.util.find_spec("gf_training." + name)
            if spec is None or spec.origin is None:
                raise ValueError("native preprocessing implementation is incomplete")
            provenance["fit_module_sha256"]["gf_training." + name] = _file_hash(spec.origin, deadline)
    elif preprocessing_backend != "python":
        raise ValueError("unknown preprocessing backend")
    _check_budget(deadline)
    return provenance


def _file_hash(path: str | Path, deadline: float = float("inf")) -> str:
    _check_budget(deadline)
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            _check_budget(deadline)
            chunk = source.read(1024 * 1024)
            _check_budget(deadline)
            if not chunk:
                break
            value.update(chunk)
    _check_budget(deadline)
    return value.hexdigest()


def _minimal_report(episodes: list[Any], planned_games: int, started: float, reason: str,
                    groups: list[Any]) -> Json:
    # Small diagnostic fallback: no bootstrap, NumPy batches, artifact walk or
    # repeated candidate/history serialization after the computation budget ends.
    wdl = {"wins": 0, "draws": 0, "losses": 0}
    failures: dict[str, int] = {reason: 1}
    for episode in episodes:
        if episode.end_kind == "natural":
            seat = 1 if episode.episode_id.endswith("seat0") else 2
            if episode.outcome == "draw":
                wdl["draws"] += 1
            elif isinstance(episode.outcome, dict) and "win" in episode.outcome:
                wdl["wins" if episode.outcome["win"]["winner"] == seat else "losses"] += 1
        if episode.failure:
            failures[episode.failure] = failures.get(episode.failure, 0) + 1
    return {"schema_version": 1, "report_status": "minimal", "planned_games": planned_games,
            "recorded_games": len(episodes), "natural_denominator": sum(wdl.values()), **wdl,
            "comparison_valid": False, "failures": failures, "wall_seconds": time.monotonic() - started,
            "group_health": [group.health for group in groups], "paired_score": None,
            "seed_cluster_ci95": None, "artifact_bytes": None, "peak_memory_bytes": None}


def _result_provenance(episode: Any) -> Json:
    from gf_simulator_client.structured_environment import StructuredEpisodeResult
    result = {key: getattr(episode, key) for key in ("episode_id", "pair_id", "seed", "policies", "game_id",
            "outcome", "end_kind", "simulator_abort_reason", "failure", "health", "replay_reconciled", "first_player", "decisions", "elapsed_seconds")}

    if isinstance(episode, StructuredEpisodeResult):
        result.update(action_contract=asdict(episode.action_contract), purpose=episode.purpose,
                      synthetic_capacity=episode.synthetic_capacity)
    return result


async def _join_owned_groups(tasks: list[asyncio.Task[Any]], wall_deadline: float, diagnostics: Json) -> list[GroupResult]:
    """ADR0004: retain ownership through repeated cancellation and OS overruns."""
    if tasks and all(task.done() for task in tasks):
        # A later statistics/preflight error has no resource cleanup to overrun.
        return [task.result() for task in tasks if not task.cancelled() and task.exception() is None
                and isinstance(task.result(), GroupResult)]
    for task in tasks:
        if not task.done():
            task.cancel()
    if not tasks:
        return []
    drain = asyncio.gather(*tasks, return_exceptions=True)
    started = time.monotonic()
    threshold = max(started, wall_deadline) + 1.6
    detect_threshold = True
    if started >= wall_deadline:
        diagnostics["task_cleanup_emergency"] = "exception detected after wall deadline; emergency reap only"
    while True:
        try:
            if detect_threshold:
                async with asyncio.timeout_at(threshold):
                    completed = await asyncio.shield(drain)
            else:
                # The resource owners retain their own bounded request/reap
                # limits. A delayed OS does not authorize orphaning them.
                completed = await asyncio.shield(drain)
            diagnostics["task_cleanup_elapsed_seconds"] = time.monotonic() - started
            results = [value for value in completed if isinstance(value, GroupResult)]
            if diagnostics.get("task_cleanup_failure") or diagnostics.get("task_cleanup_emergency"):
                for group in results:
                    group.health["cleanup_failure"] = diagnostics.get("task_cleanup_failure", diagnostics.get("task_cleanup_emergency"))
            return results
        except asyncio.CancelledError:
            continue
        except TimeoutError:
            diagnostics["task_cleanup_failure"] = "owned group cleanup exceeded wall and emergency reap allowance; joining finalizers"
            detect_threshold = False


async def run_experiment(loaded: LoadedConfig, command: str, *, collection: CollectionPlan | None = None) -> tuple[Path, Json]:
    config = loaded.config
    if isinstance(config.actions, ActionConfigV2):
        require_qualified_contract(config.actions.action_contract)
    started = collection.started_monotonic if collection is not None and collection.started_monotonic is not None else time.monotonic()
    from gf_cli.bindings import bindings, native_preflight
    wall_deadline = collection.wall_deadline if collection is not None else started + min(config.run.wall_seconds, config.run.execution_seconds + config.run.finalization_seconds)
    work_deadline = wall_deadline - config.run.report_reserve_seconds
    replay_deadline = work_deadline - config.server.cleanup_reserve_seconds
    execution_deadline = collection.execution_deadline if collection is not None else started + config.run.execution_seconds
    preflight_deadline = min(execution_deadline, replay_deadline)
    structured = isinstance(config.actions, ActionConfigV2)
    ledger = None
    structured_preprocessing = None
    if structured:
        from gf_cli.config import RunConfigV2
        from gf_cli.structured import structured_backend, structured_bindings
        from gf_simulator_client.structured_resources import ReservationLedger
        structured_preprocessing = structured_backend(config.preprocessing)
        assert isinstance(config.run, RunConfigV2)
        ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
    if collection is None and structured:
        registry, labels = structured_bindings(config.model, config.opponent)
    elif collection is None:
        from gf_models.structured_binding import ConfiguredNeuralModelV2
        assert isinstance(config.actions, ActionConfig)
        assert not isinstance(config.model, ConfiguredNeuralModelV2)
        assert not isinstance(config.opponent, ConfiguredNeuralModelV2)
        registry, labels = bindings(config.model, config.opponent, config.actions, deadline=preflight_deadline)
    else:
        registry, labels = collection.bindings, ("owned", "owned")
    if structured:
        # Generation2 owned/formal neural contexts are qualified by the next
        # integration stage, never by accepting bare component artifacts here.
        if any(binding.model_id not in ("random", "heuristic") or binding.policy_options is not None for binding in registry.values()):
            raise ContractError("structured neural collection requires Generation2 qualification")
        if any(label != binding.model_id + "-structured-v1" for label, binding in registry.items()):
            raise ContractError("structured baseline registry label/role mismatch")
        if collection is not None and any(label not in registry for group in collection.groups for spec in group.episodes for label in spec.policies):
            raise ContractError("structured collection schedule lacks an owned policy binding")
    else:
        assert isinstance(config.actions, ActionConfig)
        await native_preflight(registry, config.actions, deadline=preflight_deadline, worker_seconds=config.server.worker_seconds)
    run_id = str(uuid.uuid4())
    directory = collection.output if collection is not None else Path(config.run.output_dir) / (config.run.name + "-" + run_id[:12])
    directory.mkdir(parents=True, exist_ok=False)
    count = len(config.run.sampling_seeds) * config.run.pairs_per_seed * 2
    planned_games = sum(len(group.episodes) for group in collection.groups) if collection is not None else count * len(config.run.seeds)
    planned: tuple[Any, ...] = ()
    groups: list[Any] = []
    scheduled_tasks: list[asyncio.Task[Any]] = []
    admission_closed = False
    group_failure: BaseException | None = None
    episodes: list[Any] = []
    manifest: Json = {
        "schema_version": 4 if collection is not None else 3, "run_id": run_id, "command": command, "status": "running",
        "policy_bindings": {label: asdict(binding) for label, binding in registry.items()},
        "effective_config": loaded.effective, "config_sha256": loaded.sha256, "config_inputs": loaded.inputs,
        "code": None, "simulator": {"revision": None, "dirty": None, "verification": "unverified", "spec_sha256": None},
        "simulator_contract_revision": SIMULATOR_REVISION,
        "policy_input_version": POLICY_INPUT_VERSION, "reviewed_simulator_spec_sha256": REVIEWED_SPEC_HASHES, "binary_sha256": None, "binary_build_revision": None,
        "server_config_sha256": None, "effective_server_config": None,
        "catalog": CATALOG_META, "observation_schema_version": "1", "action_schema_version": 1,
        "journal_schema_version": JOURNAL_SCHEMA, "feature_version": FEATURE_VERSION,
        "reward_policy_version": REWARD_VERSION, "end_projection_version": END_PROJECTION_VERSION, "python": sys.version,
        "gil_enabled": sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else None,
        "schedule": [], "sampling_seed_derivation": "sha256(configured sampling_seed, repeat index, policy_stream); no GameSeed",
        "split": config.run.split, "learning_gate": "not_run",
        "budget": {"report_reserve_seconds": config.run.report_reserve_seconds,
                   "cleanup_reserve_seconds": config.server.cleanup_reserve_seconds},
        "reproducibility_limits": ["prebuilt binary build revision unverified", "seed alone does not guarantee future model training determinism",
                                   "synchronous local filesystem operations cannot provide a hard realtime wall guarantee"],
    }
    if structured:
        from gf_models.structured_binding import input_binding
        assert isinstance(config.actions, ActionConfigV2)
        manifest.update(**input_binding(config.actions.action_contract, "component", False), journal_schema_version=5, feature_version="participant-structured-v1",
                        input_recipe="participant-structured-v1", participant_runtime="python-reference-v2",
                        preprocessing_implementation=structured_preprocessing)
    if collection is not None:
        manifest.update(collection_kind=collection.kind, ppo=collection.linkage,
                        sampling_seed_derivation={
                            "ppo_batch": "sha256(canonical JSON: recipe=ppo-policy-stream-v1, seed=learning_seed, batch=batch_index, pair=pair_index, role=learner|opponent) first 16 hex digits as uint64; role stream follows seat swaps; no GameSeed",
                            "ppo_validation": "sha256(canonical JSON: recipe=ppo-final-validation-stream-v1, learning_seed=learning_seed, pair=pair_index, role=learner|opponent) first 16 hex digits as uint64; role stream follows seat swaps; no GameSeed",
                        }[collection.kind])
    summary: Json
    failure_reason: str | None = None
    try:
        _check_budget(preflight_deadline)
        atomic_json(directory / "manifest.json", manifest)
        _check_budget(preflight_deadline)
        manifest["effective_server_config"] = server_effective(config.server.config, max(len(group.episodes) for group in collection.groups) if collection is not None else count if config.server.external_url is None else None)
        _check_budget(preflight_deadline)
        if config.server.external_url is None:
            manifest["binary_sha256"] = _file_hash(config.server.binary, preflight_deadline)
        if config.server.source_dir is not None:
            simulator = _git(Path(config.server.source_dir), preflight_deadline)
            if simulator["revision"] != config.server.simulator_revision:
                raise ValueError("simulator source revision does not match accepted contract pin")
            simulator["verification"] = "source_checkout_verified_binary_build_unverified"
            simulator["spec_sha256"] = {name: _file_hash(str(Path(config.server.source_dir) / name), preflight_deadline)
                                         for name in REVIEWED_SPEC_HASHES}
            manifest["simulator"] = simulator
        manifest["code"] = _code_provenance(preflight_deadline)
        manifest["server_config_sha256"] = _file_hash(config.server.config, preflight_deadline)
        planned = collection.groups if collection is not None else schedule(config.run.seeds, config.run.sampling_seeds, config.run.pairs_per_seed,
                           labels[0], labels[1], deadline=preflight_deadline)
        manifest["schedule"] = serialize_schedule(planned, manifest["schema_version"])
        _check_budget(preflight_deadline)
        atomic_json(directory / "manifest.json", manifest)
        _check_budget(preflight_deadline)
        settings = EnvironmentSettings(
            policy_bindings=registry,
            binary=config.server.binary, server_config=config.server.config, output_dir=directory, run_id=run_id,
            actions=config.actions, fixed_seed_experiment=config.server.fixed_seed_experiment,
            expected_fingerprint=config.server.expected_fingerprint, external_url=config.server.external_url,
            startup_seconds=config.server.startup_seconds, request_seconds=config.server.request_seconds,
            worker_seconds=config.server.worker_seconds, shutdown_seconds=config.server.shutdown_seconds,
            transport_retries=config.server.transport_retries, max_decisions=config.run.max_decisions,
            admission_deadline=execution_deadline if collection is not None else started + config.run.admission_seconds, execution_deadline=execution_deadline,
            wall_deadline=wall_deadline, work_deadline=work_deadline, replay_deadline=replay_deadline,
        )
        environment: SimulatorEnvironment
        if structured:
            from gf_cli.structured import structured_environment
            assert ledger is not None
            environment = structured_environment(settings, ledger)
        else:
            environment = SimulatorEnvironment(settings, make_policy, ParticipantJournal)
        semaphore = asyncio.Semaphore(config.run.concurrency)
        async def group_run(group: Any) -> Any:
            nonlocal admission_closed, group_failure
            async with semaphore:
                if admission_closed:
                    return GroupResult([], {"failure": "admission_closed"})
                try:
                    return await environment.run_group(group)
                except BaseException as child_error:
                    # Close admission before releasing the semaphore. A queued
                    # task can wake before gather delivers this exception to
                    # the coordinator, but must not enter the environment.
                    admission_closed = True
                    if group_failure is None:
                        group_failure = child_error
                    raise
        scheduled_tasks = [asyncio.create_task(group_run(group)) for group in planned]
        groups = await asyncio.gather(*scheduled_tasks)
        episodes = [episode for group in groups for episode in group.episodes]
        _check_budget(wall_deadline)
        summary = report(episodes, planned, bootstrap_samples=config.evaluation.bootstrap_samples,
                         bootstrap_seed=config.evaluation.bootstrap_seed, wall_seconds=time.monotonic() - started,
                         groups_health=[group.health for group in groups], deadline=wall_deadline)
        _check_budget(wall_deadline)
        artifact_bytes = 0
        for path in directory.rglob("*"):
            _check_budget(wall_deadline)
            if path.is_file():
                artifact_bytes += path.stat().st_size
        _check_budget(wall_deadline)
        summary["artifact_bytes"] = artifact_bytes
        summary["benchmark"] = {key: summary[key] for key in ("decisions", "decisions_per_second", "candidate_count_mean", "search_attempts_mean", "search_latency_ms_p50", "search_latency_ms_p95", "wall_seconds", "peak_memory_bytes", "artifact_bytes")}
    except BaseException as error:
        groups = await _join_owned_groups(scheduled_tasks, wall_deadline, manifest)
        episodes = [episode for group in groups for episode in group.episodes]
        if isinstance(error, TimeoutError) and group_failure is None:
            # Only coordinator preflight/statistics budget exhaustion uses the
            # minimal fallback. A child TimeoutError is an original group
            # exception even if it has no remaining peers.
            failure_reason = "budget_exhausted"
            summary = _minimal_report(episodes, planned_games, started, failure_reason, groups)
        else:
            manifest["status"] = "interrupted"
            manifest["groups"] = [{**group.health, "episode_ids": [e.episode_id for e in group.episodes]} for group in groups]
            manifest["episodes"] = [_result_provenance(episode) for episode in episodes]
            if not manifest["episodes"]:
                manifest["episodes"] = [json.loads(path.read_text()) for path in directory.glob("*/result.json")]
            manifest["elapsed_seconds"] = time.monotonic() - started
            try:
                atomic_json(directory / "manifest.json", manifest)
            except OSError:
                # Persistence debt cannot replace the original exception.
                pass
            raise
    manifest["groups"] = [{**group.health, "episode_ids": [e.episode_id for e in group.episodes]} for group in groups]
    manifest["episodes"] = [_result_provenance(episode) for episode in episodes]
    persistence_started = time.monotonic()
    try:
        _check_budget(wall_deadline)
        atomic_json(directory / "report.json", summary)
        _check_budget(wall_deadline)
        manifest["report_sha256"] = _file_hash(directory / "report.json", wall_deadline)
        manifest["status"] = "complete" if summary["comparison_valid"] else "incomplete"
        manifest["elapsed_seconds"] = time.monotonic() - started
        _check_budget(wall_deadline)
        atomic_json(directory / "manifest.json", manifest)
        # Success is decided only after the final write returns.
        _check_budget(wall_deadline)
    except (TimeoutError, OSError) as exc:
        failure_reason = "budget_exhausted" if isinstance(exc, TimeoutError) else "save_failure"
        summary = _minimal_report(episodes, planned_games, started, failure_reason, groups)
        manifest["status"] = "incomplete"
        manifest["finalization_failure"] = failure_reason
        manifest["budget_overrun_seconds"] = max(0.0, time.monotonic() - wall_deadline)
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["persistence_duration_seconds"] = time.monotonic() - persistence_started
        manifest["persistence_measurement"] = "includes failed/delayed write, before best-effort correction"
        # Minimal durable correction is best effort after an uninterruptible OS
        # write/mandatory reap overrun. It can never claim successful completion.
        correction_errors: dict[str, str] = {}
        for name, artifact in (("report.json", summary), ("manifest.json", manifest)):
            try:
                atomic_json(directory / name, artifact)
            except OSError as save_error:
                correction_errors[name] = str(save_error)
                # Each path gets its own attempt. If both fail, only the caller
                # can know the durable files may still contain stale success.
                summary["correction_save_failures"] = dict(correction_errors)
                manifest["correction_save_failures"] = dict(correction_errors)
        if correction_errors:
            summary["correction_save_failure"] = canonical(correction_errors)
    summary["wall_seconds"] = time.monotonic() - started
    return directory, summary


async def collect_ppo(loaded: LoadedConfig, plan: CollectionPlan) -> FinalizedCollection:
    directory, summary = await run_experiment(loaded, "collect", collection=plan)
    if summary.get("comparison_valid") is not True:
        raise RuntimeError("PPO batch collection did not complete naturally and healthily")
    return FinalizedCollection(directory, summary)


def main(argv: list[str] | None = None) -> int:
    command_started = time.monotonic()
    parser = argparse.ArgumentParser(prog="gf-bots", description="Public-information GodField baselines")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "collect", "evaluate", "benchmark", "validate", "train", "validate-policy"):
        child = subcommands.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        if command == "train":
            child.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
            child.add_argument("--run-dir", type=Path)
            child.add_argument("--pause-after-update", type=int)
        if command == "validate-policy":
            child.add_argument("--parent-run", required=True, type=Path)
            child.add_argument("--output", required=True, type=Path)
        child.add_argument("--set", action="append", default=[], metavar="TABLE.FIELD=TOML_VALUE")
    export = subcommands.add_parser("export")
    export.add_argument("journals", nargs="+", type=Path)
    export.add_argument("--output", required=True, type=Path)
    dataset = subcommands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    build = dataset_commands.add_parser("build")
    build.add_argument("--ppo-runs", nargs=3, required=True, type=Path)
    build.add_argument("--validation-runs", nargs=3, required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    inspect = dataset_commands.add_parser("inspect")
    inspect.add_argument("manifest", type=Path)
    index = dataset_commands.add_parser("index")
    index.add_argument("manifest", type=Path)
    index.add_argument("--output", required=True, type=Path,
                       help="fresh Parquet path; output and .json sidecar must not exist")
    for dataset_parser in (build, inspect, index):
        dataset_parser.add_argument("--preprocessing-config", type=Path)
        dataset_parser.add_argument("--address-space-limit-mib", type=int,
                                    help="optional process virtual-address limit, not an RSS quota")
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            from gf_cli.resource_limits import dataset_address_limit
            from gf_common.preprocessing import load_preprocessing_config
            preprocessing = load_preprocessing_config(args.preprocessing_config)
            with dataset_address_limit(args.address_space_limit_mib):
                from gf_training.fixed_data import (
                    build_dataset,
                    index_dataset,
                    inspect_dataset_summary,
                )
                deadline = command_started + preprocessing.wall_seconds
                if args.dataset_command == "build":
                    result = build_dataset(tuple(args.ppo_runs), tuple(args.validation_runs), args.output,
                                           deadline=deadline, preprocessing=preprocessing)
                elif args.dataset_command == "index":
                    result = index_dataset(args.manifest, args.output, deadline=deadline,
                                           preprocessing=preprocessing)
                else:
                    result = inspect_dataset_summary(args.manifest, deadline=deadline,
                                                     preprocessing=preprocessing)
                print(canonical(result))
                return 0
        if args.command == "export":
            from gf_training.journal import export_parquet
            print(canonical(export_parquet(args.journals, args.output)))
            return 0
        import tomllib
        overrides: Json = {}
        for assignment in args.set:
            name, separator, raw_value = assignment.partition("=")
            if not separator or "." not in name:
                raise ValueError("--set expects TABLE.FIELD=TOML_VALUE")
            parts = name.split(".")
            current = overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = tomllib.loads("value=" + raw_value)["value"]
        if args.command == "train" and args.mode != "fresh" and args.run_dir is None:
            raise ValueError("resume is unsupported without --run-dir; no work performed")
        loaded = load_config(args.config, overrides)
        if args.command == "validate":
            server_effective(loaded.config.server.config, len(loaded.config.run.sampling_seeds) * loaded.config.run.pairs_per_seed * 2)
            print(canonical({"valid": True, "config_sha256": loaded.sha256, "effective_config": loaded.effective,
                             "inputs": loaded.inputs}))
            return 0
        if isinstance(loaded.config.actions, ActionConfigV2):
            require_qualified_contract(loaded.config.actions.action_contract)
        if args.command == "validate-policy":
            from gf_training.fixed_data import validate_policy
            config = replace(loaded.config, run=replace(loaded.config.run, split="validation"))
            validation_loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), loaded.inputs)
            async def collect_validation(plan: CollectionPlan) -> FinalizedCollection:
                return await collect_ppo(validation_loaded, plan)
            print(canonical(asyncio.run(validate_policy(validation_loaded, args.parent_run, args.output, collect_validation))))
            return 0
        if args.command == "train":
            if loaded.config.model.model_id in ("mlp", "gru", "transformer"):
                from gf_training.ppo_run import train_ppo
                if args.run_dir is None:
                    raise ValueError("PPO train requires --run-dir")
                async def collect_batch(plan: CollectionPlan) -> FinalizedCollection:
                    return await collect_ppo(loaded, plan)
                result = asyncio.run(train_ppo(loaded, collect_batch, args.run_dir, mode=args.mode, pause_after_update=args.pause_after_update))
            else:
                from gf_training.lightgbm_run import train_experiments
                if args.run_dir is not None or args.pause_after_update is not None:
                    raise ValueError("LightGBM has no PPO run/pause controls")
                def fit_identity(deadline: float) -> Json:
                    return _fit_code_provenance(deadline, preprocessing_backend=loaded.config.preprocessing.backend,
                                                preprocessing_config=loaded.config.preprocessing)
                result = asyncio.run(train_experiments(loaded, mode=args.mode, code_provenance_provider=fit_identity))
            print(canonical(result))
            return 0
        directory, summary = asyncio.run(run_experiment(loaded, args.command))
        print(canonical({"run_dir": str(directory), **{key: summary[key] for key in ("planned_games", "natural_denominator", "wins", "draws", "losses", "comparison_valid", "failures", "wall_seconds")}}))
        return 0 if summary["comparison_valid"] else 1
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
