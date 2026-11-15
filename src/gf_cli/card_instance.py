"""Corrected game composition; runtime evidence stays outside participants."""
from __future__ import annotations

import asyncio
import hashlib
import math
import stat
import time
import uuid
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gf_cli.config import (
    CorrectedServerConfigV2,
    GroupPackExperimentConfig,
    LoadedConfig,
    RunConfigV2,
    require_storage_command,
)
from gf_common.artifacts import ArtifactReader
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_validation import reconcile_replay, replay_closure
from gf_common.domain import (
    ContractError,
    EpisodeSpec,
    GroupResult,
    Json,
    RuntimePolicyBinding,
    canonical,
    clone,
    digest,
    integer,
)
from gf_common.schedule import schedule, serialize_schedule
from gf_evaluation.evaluation import report
from gf_models.config import ModelConfig, PPOTrainingConfig, TrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_models.structured_tree_config import ConfiguredTreeModelV3, TreeEncoderConfigV3
from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.environment import EnvironmentSettings
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import (
    CollectionPublication,
    collection_metadata,
    final_validation_collection_metadata,
    ppo_collection_metadata,
)
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal
from gf_training.collection import CollectionPlan, FinalizedCollection, PackedRunResult

if TYPE_CHECKING:
    from gf_cli.hpo_evaluation import HpoEvaluationPlan, _HpoEvaluationCleanupObservation
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_training.card_instance_final_validation import GameFinalValidationPlan
    from gf_training.hpo_gpu_lease import GpuJob


def _check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("corrected collection deadline")


def _require_actions(loaded: LoadedConfig) -> None:
    config = loaded.config
    if (type(config.actions) is not ActionConfigV3
            or config.actions.action_contract != corrected_card_instance_contract()
            or config.actions.max_sequence_length > 18 or not isinstance(config.run, RunConfigV2)):
        raise ContractError("corrected baseline collection requires the corrected contract and real18card limits")


def _require_refresh_native(loaded: LoadedConfig) -> None:
    if (loaded.config.preprocessing.backend == "rust"
            and any(model.model_id == "heuristic_refresh" for model in (loaded.config.model, loaded.config.opponent))):
        from gf_training.native_backend import require_native_refresh_baseline
        require_native_refresh_baseline()


def _baseline_labels(loaded: LoadedConfig, *, artifact_inputs: Any = None) -> tuple[dict[str, RuntimePolicyBinding], tuple[str, str]]:
    _require_actions(loaded)
    config = loaded.config
    _require_refresh_native(loaded)
    assert type(config.actions) is ActionConfigV3
    labels = []
    bindings = {}
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
    from gf_models.refresh_heuristic_config import RefreshHeuristicConfig
    for model in (config.model, config.opponent):
        if type(model) is RefreshHeuristicConfig:
            options = RefreshOptions(model.version, model.attack_threshold)
            label = refresh_policy_label(config.actions, card_instance_rules(), options)
            labels.append(label)
            bindings[label] = RuntimePolicyBinding(model.model_id, asdict(options))
            continue
        if type(model) in (ConfiguredNeuralModelV3, ConfiguredTreeModelV3) and artifact_inputs is not None:
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
                                neural_metadata: dict[str, Json] | None = None,
                                tree_metadata: dict[str, Json] | None = None,
                                artifacts: ArtifactReader | None = None) -> None:
    import json

    for episode in episodes:
        _check(deadline)
        spec, runtime = expected[episode.episode_id]
        closures = []
        for viewer in (1, 2):
            folder = directory / episode.episode_id
            journal = read_card_instance_journal(folder / f"player-{viewer}.jsonl", actions=actions, deadline=deadline,
                **({} if artifacts is None else {"artifacts": artifacts, "artifact_prefix": episode.episode_id}))
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
            if (tree_metadata is not None and journal.header["policy"] in tree_metadata
                    and canonical(journal.header.get("tree")) != canonical(tree_metadata[journal.header["policy"]])):
                raise ContractError("persisted Tree3 journal differs from its retained actual pair")
            accepted = [journal.decisions[index].command["action"] for index in journal.accepted]
            replay_name = episode.episode_id + f"/replay-player-{viewer}.json"
            if artifacts is not None:
                from gf_training.group_pack_memory import MemoryArtifactSnapshot
                if type(artifacts) is not MemoryArtifactSnapshot:
                    raise ContractError("runtime replay validation requires immutable original snapshot")
                raw = artifacts.read(replay_name)
                if len(raw) > 64 * 1024**2:
                    raise ContractError("runtime Replay original bytes exceed file bound")
                artifacts.admit_validation(32 * len(raw) + 65536)
            else:
                raw = (folder / f"replay-player-{viewer}.json").read_bytes()
            value = json.loads(raw)
            reconcile_replay(value, viewer, accepted, episode.outcome)
            closures.append(replay_closure(value, episode.game_id, viewer))
        if closures[0] != closures[1]:
            raise ContractError("persisted corrected participant Replay closures disagree")


async def run_baseline(loaded: LoadedConfig, command: str) -> tuple[Path, Json] | PackedRunResult:
    require_storage_command(loaded.config, command)
    # Validate all unsupported combinations before filesystem/process ownership.
    if command not in ("smoke", "collect", "evaluate"):
        raise ContractError("corrected baseline integration supports smoke/collect/evaluate only")
    bindings, labels = _baseline_labels(loaded)
    return await _run_corrected(loaded, command, bindings, labels)


async def run_inference(loaded: LoadedConfig, command: str) -> tuple[Path, Json] | PackedRunResult:
    require_storage_command(loaded.config, command)
    if command not in ("smoke", "collect", "evaluate"):
        raise ContractError("corrected neural inference supports smoke/collect/evaluate only")
    _require_actions(loaded)
    _require_refresh_native(loaded)
    config = loaded.config
    assert type(config.actions) is ActionConfigV3 and isinstance(config.run, RunConfigV2)
    neural = tuple(cast(ConfiguredNeuralModelV3 | ConfiguredTreeModelV3, model)
                   for model in (config.model, config.opponent)
                   if type(model) in (ConfiguredNeuralModelV3, ConfiguredTreeModelV3))
    if not neural or any(model.checkpoint is None for model in neural):
        raise ContractError("corrected neural inference requires an explicit game checkpoint")
    from gf_models.refresh_heuristic_config import RefreshHeuristicConfig
    for model in (config.model, config.opponent):
        if type(model) is RefreshHeuristicConfig:
            continue
        if type(model) in (ConfiguredNeuralModelV3, ConfiguredTreeModelV3):
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
    tree_runtime = any(type(model) is ConfiguredTreeModelV3 for model in neural)
    evidence = None
    if not tree_runtime:
        evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
                                              deadline=preflight_deadline)
        evidence.preflight()
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_game_workers_v3
    rules = card_instance_rules()
    from gf_models.card_instance_game_tree_runtime import RetainedGamePolicyArtifacts
    owner: RetainedGamePolicyArtifacts | RetainedGameArtifacts
    ownership_options: Json = {"max_owned_bytes": min(model.resources.max_owned_bytes for model in neural),
        "preflight_deadline": preflight_deadline, "finalization_deadline": work_deadline}
    if tree_runtime:
        owner = RetainedGamePolicyArtifacts(neural, config.actions, rules=rules, **ownership_options)
    else:
        owner = RetainedGameArtifacts(tuple(cast(ConfiguredNeuralModelV3, model) for model in neural),
            config.actions, rules=rules, **ownership_options)
    try:
        if evidence is None:
            evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir),
                                                  deadline=preflight_deadline)
            evidence.preflight()
        bindings, labels = _baseline_labels(loaded, artifact_inputs=owner)
        admission = admit_game_workers_v3(config.actions, (bindings[labels[0]], bindings[labels[1]]),
            rules=rules, game_concurrency=config.run.concurrency,
            global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers)
        result = await _run_corrected(loaded, command, bindings, labels, started=started,
                                      artifact_inputs=owner, runtime_evidence=evidence,
                                      resource_admission=admission)
    except BaseException as error:
        try:
            owner.release()
        except BaseException as cleanup:  # noqa: BLE001 - preserve initiating failure with reachable pair custody.
            vars(error)["_game_artifact_cleanup_owner"] = owner
            error.add_note("retained game pair cleanup: " + str(cleanup))
        raise
    # Confirmed collection ownership has already retired before publication;
    # this final artifact release is memory-only.
    owner.release()
    return result


async def _run_corrected(loaded: LoadedConfig, command: str,
                         bindings: dict[str, RuntimePolicyBinding], labels: tuple[str, ...], *,
                         started: float | None = None, artifact_inputs: Any = None,
                         runtime_evidence: CardInstanceRuntimeEvidence | None = None,
                         resource_admission: Json | None = None,
                         collection: CollectionPlan | GameFinalValidationPlan | HpoEvaluationPlan | None = None,
                         owned_ledger: ReservationLedger | None = None,
                         _final_validation_context: Any = None,
                         _hpo_evaluation_job: Any = None,
                         _hpo_evaluation_cleanup: Any = None,
                         resident_pool: Any = None) -> tuple[Path, Any] | PackedRunResult:
    config = loaded.config
    from gf_cli.hpo_config import HPOExperimentConfig
    from gf_cli.hpo_evaluation import HpoEvaluationPlan, ResidentHpoEvaluationPlan
    eval_plan=cast(HpoEvaluationPlan,collection) if type(collection) in (HpoEvaluationPlan, ResidentHpoEvaluationPlan) else None
    observation = _hpo_evaluation_cleanup
    hpo_training=type(config) is HPOExperimentConfig and type(collection) is CollectionPlan
    if eval_plan is not None:
        observation.require_identity(eval_plan, _hpo_evaluation_job, artifact_inputs)
        if observation._core_phase != 'entered' or observation._run_phase != 'requested':
            raise ContractError('original evaluation collector invocation required')
        observation._run_phase = 'entered'
        eval_plan.require(loaded,gpu_job=_hpo_evaluation_job)
        if command!='evaluate' or owned_ledger is not _hpo_evaluation_job.ledger or _final_validation_context is not None:
            raise ContractError('evaluation requires its actual job ledger and distinct role')
        eval_plan.require(loaded,gpu_job=_hpo_evaluation_job)
    elif _hpo_evaluation_job is not None:
        raise ContractError('evaluation job cannot authorize a training/final collection')
    assert isinstance(config.actions, ActionConfigV3)
    require_storage_command(config, command, standalone=collection is None and owned_ledger is None)
    if config.server.external_url is not None or config.server.source_dir is None:
        raise ContractError("corrected baseline collection requires locally managed verified source/binary evidence")
    started = time.monotonic() if started is None else started
    wall_deadline = started + config.run.wall_seconds if collection is None else collection.wall_deadline
    work_deadline = wall_deadline - config.run.report_reserve_seconds
    if eval_plan is not None: work_deadline=eval_plan.work_deadline
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
        if eval_plan is not None:
            eval_plan.require(loaded,gpu_job=_hpo_evaluation_job)
        elif _final_validation_context is None:
            _validate_online_collection(loaded, cast(CollectionPlan, collection), owned_ledger, preflight_deadline, resident_phase=None if resident_pool is None else resident_pool.admission)
        else:
            _validate_final_collection(loaded, collection, owned_ledger, _final_validation_context, preflight_deadline)
    run_id = eval_plan.linkage['evaluation_id'] if eval_plan is not None else str(uuid.uuid4())
    directory = (Path(config.run.output_dir) / (config.run.name + "-" + run_id)
                 if collection is None else collection.output)
    directory.mkdir(parents=True, exist_ok=False)
    settings = EnvironmentSettings(binary=config.server.binary, server_config=config.server.config,
        output_dir=directory, run_id=run_id, actions=config.actions,
        fixed_seed_experiment=config.server.fixed_seed_experiment,
        startup_seconds=config.server.startup_seconds, request_seconds=config.server.request_seconds,
        worker_seconds=config.server.worker_seconds, shutdown_seconds=config.server.shutdown_seconds,
        transport_retries=config.server.transport_retries, max_decisions=config.run.max_decisions,
        admission_deadline=(eval_plan.admission_deadline if eval_plan is not None else
                            cast("GameFinalValidationPlan", collection).admission_deadline if _final_validation_context is not None
                            else started + config.run.admission_seconds if collection is None else execution_deadline),
        execution_deadline=execution_deadline,
        wall_deadline=wall_deadline, work_deadline=work_deadline, replay_deadline=replay_deadline,
        policy_bindings=bindings)
    assert isinstance(config.run, RunConfigV2)
    ledger = (ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
              if owned_ledger is None else owned_ledger)
    extra: dict[str, Any] = {} if artifact_inputs is None else {"artifact_inputs": artifact_inputs}
    if resident_pool is not None:
        extra["resident_pool"] = resident_pool
    semaphore = asyncio.Semaphore(config.run.concurrency)
    tasks: list[asyncio.Task[GroupResult]] = []
    admission_closed = False
    diagnostics: Json = {}
    packed = None
    packed_publication = None
    if observation is not None:
        observation._tasks = tasks
    if isinstance(config, GroupPackExperimentConfig):
        from gf_training.group_pack_collection import GroupPackCollection
        owned_directory = None
        if collection is not None:
            if owned_ledger is None:
                raise ContractError('packed owned collection requires its SAME actual ledger')
            if eval_plan is not None:
                eval_plan.require(loaded,gpu_job=_hpo_evaluation_job)
                owned_directory=Path(config.storage.db_root)/run_id
            elif _final_validation_context is None:
                online = _validate_online_collection(loaded, cast(CollectionPlan, collection), owned_ledger, preflight_deadline, resident_phase=None if resident_pool is None else resident_pool.admission)
                owned_directory = Path(config.storage.db_root) / online.state['run_id'] / 'collections' / collection.linkage['batch_id']
            else:
                _validate_final_collection(loaded, collection, owned_ledger, _final_validation_context, preflight_deadline)
        if observation is not None:
            observation._pack_phase = 'creating'
            packed = GroupPackCollection.__new__(GroupPackCollection)
            observation._pack = packed
            try:
                GroupPackCollection.__init__(packed, config.storage.settings(), run_id, deadline=work_deadline,
                    owned_directory=owned_directory)
            except BaseException as error:
                vars(error)['hpo_partial_pack_owner'] = packed
                try:
                    await packed.close()
                except BaseException as cleanup:  # noqa: BLE001 -- preserve partial original owner and primary.
                    error.add_note('partial packed construction cleanup: ' + str(cleanup))
                raise
        else:
            packed = GroupPackCollection(config.storage.settings(), run_id, deadline=work_deadline,
                                         owned_directory=owned_directory)
        if observation is not None:
            observation._pack, observation._pack_phase = packed, 'owned'

    async def group_run(group: Any) -> GroupResult:
        nonlocal admission_closed
        async with semaphore:
            if admission_closed:
                return GroupResult([], {"failure": "admission_closed"})
            try:
                if packed is None:
                    return await environment.run_group(group)
                from gf_training.group_pack_memory import MemoryCardInstanceJournal
                if observation is not None:
                    observation._creating_environment(group)
                memory_group = packed.new_group(group.episodes[0].episode_id)
                task = asyncio.current_task()
                assert task is not None
                packed.track_group_task(task)
                if observation is not None:
                    own_environment = CardInstanceSimulatorEnvironment.__new__(CardInstanceSimulatorEnvironment)
                    memory_group.runtime_owner = own_environment
                    observation._partial_environment(group, own_environment)
                    CardInstanceSimulatorEnvironment.__init__(own_environment, settings,
                        partial(MemoryCardInstanceJournal, artifact_sink=memory_group.sink),
                        ledger=ledger, runtime_evidence=evidence, artifact_sink=memory_group.sink,
                        server_output_root=memory_group.server_root, **extra)
                else:
                    own_environment = CardInstanceSimulatorEnvironment(settings,
                        partial(MemoryCardInstanceJournal, artifact_sink=memory_group.sink),
                        ledger=ledger, runtime_evidence=evidence, artifact_sink=memory_group.sink,
                        server_output_root=memory_group.server_root, **extra)
                memory_group.runtime_owner = own_environment
                if observation is not None:
                    observation._environment(group, own_environment)
                result = await own_environment.run_group(group)
                if observation is not None:
                    observation._group_result(group, result)
                if not own_environment.storage_cleanup_complete(result):
                    from gf_training.group_pack_resources import CleanupError
                    raise CleanupError("Simulator/participant reap and drain are not confirmed", packed)
                expected = {spec.episode_id: (spec, result.health.get("runtime_evidence", {}))
                            for spec in group.episodes}
                neural_metadata = ({label: artifact_inputs.journal_metadata(label)
                    for label, binding in bindings.items()
                    if binding.model_id in ("mlp", "gru", "transformer")}
                    if artifact_inputs is not None else None)
                def validate(snapshot: Any) -> None:
                    healthy = [episode for episode in result.episodes
                               if episode.end_kind == "natural" and episode.failure is None
                               and episode.replay_reconciled and not any(episode.health.values())]
                    _validate_recorded_episodes(directory, healthy, cast(ActionConfigV3, config.actions),
                        run_id, work_deadline, expected, neural_metadata=neural_metadata,
                        artifacts=snapshot)
                await packed.archive(memory_group, group.episodes[0].episode_id, result,
                                     validate=validate)
                return result
            except BaseException:
                admission_closed = True
                raise

    try:
        if packed is not None:
            await packed.start()
        environment = CardInstanceSimulatorEnvironment(settings, CardInstanceJournal, ledger=ledger,
                                                      runtime_evidence=evidence, **extra)
        if observation is not None:
            observation._task_phase = 'creating'
        for group in planned:
            coroutine = group_run(group)
            try:
                task = asyncio.create_task(coroutine)
            except BaseException as creation_error:
                # A custom factory may have transferred the coroutine before
                # raising. Retain it without closing possibly scheduled work;
                # no never-started proof follows from an absent returned Task.
                if observation is not None:
                    observation._task_creation_failure = (coroutine, creation_error)
                    observation._task_phase = 'uncertain'
                    vars(creation_error)['hpo_task_creation_coroutine'] = coroutine
                else:
                    coroutine.close()
                raise
            tasks.append(task)
            if observation is not None:
                observation._register_task(group, task)
        if observation is not None:
            observation._task_phase = 'settled'
        groups = await asyncio.gather(*tasks)
        episodes = [episode for group in groups for episode in group.episodes]
        if resident_pool is not None:
            await resident_pool.retire_unused()
            resident_pool.retire_collection().require(ledger)
        if packed is not None:
            from gf_training.group_pack_resources import json_bound
            packed.reserve_metadata(32 * json_bound(loaded.effective) + 65536)
            packed.freeze_storage_metrics()
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
            if eval_plan is not None:
                _require_hpo_evaluation_retirement(eval_plan,_hpo_evaluation_job,ledger,resident_pool=resident_pool)
            elif _final_validation_context is None:
                _validate_online_collection(loaded, cast(CollectionPlan, collection), ledger, work_deadline,
                    resident_phase=None if resident_pool is None else resident_pool.authority)
            else:
                _validate_final_collection(loaded, collection, ledger, _final_validation_context, work_deadline)
            if resident_pool is not None:
                resident_pool.authority.require(ledger)
            elif ledger.workers or ledger.updating:
                raise ContractError("PPO collection has unreaped participant/update ownership")
        expected = {spec.episode_id: (spec, group.health["runtime_evidence"])
                    for plan, group in zip(planned, groups, strict=True) for spec in plan.episodes}
        neural_metadata = ({label: artifact_inputs.journal_metadata(label) for label, binding in bindings.items()
                            if binding.model_id in ("mlp", "gru", "transformer")}
                           if artifact_inputs is not None else None)
        tree_metadata = ({label: artifact_inputs.journal_metadata(label) for label, binding in bindings.items()
                          if binding.model_id == "lightgbm"} if artifact_inputs is not None else None)
        if packed is None:
            _validate_recorded_episodes(directory, episodes, config.actions, run_id, work_deadline, expected,
                                       neural_metadata=neural_metadata, tree_metadata=tree_metadata)
        from gf_models.card_instance_game_tree_runtime import RetainedGamePolicyArtifacts
        typed_inputs = artifact_inputs.finalize_inputs(deadline=work_deadline) if type(artifact_inputs) is RetainedGamePolicyArtifacts else None
        neural_inputs = (artifact_inputs.finalize(deadline=work_deadline)
            if artifact_inputs is not None and typed_inputs is None else None)
        inventory = (_inventory(directory, work_deadline) if packed is None
                     else sorted(packed.inventory, key=lambda row: row["path"]))
        from gf_cli import _code_provenance
        producer = _code_provenance(work_deadline)
        receipts = [group.health["runtime_evidence"] for group in groups]
        configured_baseline = any(binding.model_id == "heuristic_refresh" for binding in bindings.values())
        metadata_options: dict[str, Any] = ({"policy_bindings": bindings, **typed_inputs} if typed_inputs is not None else
            {"policy_bindings": bindings} if neural_inputs is None and configured_baseline else
            {} if neural_inputs is None else {"policy_bindings": bindings, "neural_inputs": neural_inputs})
        metadata_producer: Any = collection_metadata if collection is None else ppo_collection_metadata
        if hpo_training and resident_pool is not None:
            metadata_options["logical_retirement"] = resident_pool.authority.evidence
        if hpo_training:
            from gf_training.hpo_collection import hpo_collection_metadata
            metadata_producer=hpo_collection_metadata
        if type(eval_plan) is ResidentHpoEvaluationPlan:
            metadata_options.update(execution_runtime=eval_plan.execution_runtime,
                logical_retirement=resident_pool.authority.evidence)
        if eval_plan is not None:
            from gf_training.card_instance_collection import hpo_evaluation_collection_metadata
            metadata_producer=hpo_evaluation_collection_metadata
        if _final_validation_context is not None:
            summary.update(healthy=summary["comparison_valid"], artifact_bytes=sum(row["bytes"] for row in inventory))
            summary["wall_seconds"] = time.monotonic() - started
            metadata_producer = final_validation_collection_metadata
            metadata_options.update(config_projection=cast("GameFinalValidationPlan", collection).config_projection,
                validation_invocation=cast("GameFinalValidationPlan", collection).validation_invocation, elapsed_seconds=time.monotonic() - started)
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
            episodes=(clone([asdict(episode) for episode in episodes]) if eval_plan is not None
                      else [asdict(episode) for episode in episodes]))
        if _final_validation_context is None:
            summary.update(healthy=summary["comparison_valid"], artifact_bytes=sum(row["bytes"] for row in inventory))
            summary["wall_seconds"] = time.monotonic() - started
        if not summary["healthy"]:
            # Diagnostic outputs have no healthy manifest/report completion names.
            if packed is None:
                _write_diagnostic(directory, "incomplete.json", {"manifest": manifest, "report": summary})
            else:
                from gf_training.group_pack_publication import GroupPackCollectionPublication
                if observation is not None:
                    observation._publication_phase = 'creating'
                    packed_publication = GroupPackCollectionPublication.__new__(GroupPackCollectionPublication)
                    observation._publication = packed_publication
                    GroupPackCollectionPublication.__init__(packed_publication, packed, directory)
                else:
                    packed_publication = GroupPackCollectionPublication(packed, directory)
                if observation is not None:
                    observation._publication, observation._publication_phase = packed_publication, 'owned'
                return await packed_publication.diagnostic(summary, manifest)
            return directory, summary
        manifest["report_sha256"] = hashlib.sha256((canonical(summary) + "\n").encode()).hexdigest()
        if packed is not None:
            from gf_training.group_pack_publication import GroupPackCollectionPublication
            if observation is not None:
                observation._publication_phase = 'creating'
                packed_publication = GroupPackCollectionPublication.__new__(GroupPackCollectionPublication)
                observation._publication = packed_publication
                GroupPackCollectionPublication.__init__(packed_publication, packed, directory)
            else:
                packed_publication = GroupPackCollectionPublication(packed, directory)
            if observation is not None:
                observation._publication, observation._publication_phase = packed_publication, 'owned'
            await packed_publication.prepare(summary, manifest)
        if observation is not None:
            observation._composite_phase = 'creating'
            publication = CollectionPublication.__new__(CollectionPublication)
            observation._composite = publication
            CollectionPublication.__init__(publication, directory, work_deadline=work_deadline,
                cleanup_deadline=wall_deadline, group_pack=packed_publication)
            observation._composite_phase = 'owned'
        else:
            publication = CollectionPublication(directory, work_deadline=work_deadline,
                cleanup_deadline=wall_deadline, group_pack=packed_publication)
        if _final_validation_context is not None:
            issued = _final_validation_context.begin_collection(publication, summary, manifest)
            return directory, issued  # Actual pending owner; canonical manifest remains absent.
        if type(artifact_inputs) is RetainedGamePolicyArtifacts or packed is not None and artifact_inputs is not None:
            # Bound integrity check over the admitted actual owner; the callable
            # conveys no Source, fit or paired-artifact authority of its own.
            def retained_input_guard() -> None:
                if type(artifact_inputs) is RetainedGamePolicyArtifacts:
                    artifact_inputs.guard(deadline=work_deadline)
                else:
                    cast(Any, artifact_inputs).finalize(deadline=work_deadline)
            published_result = publication.commit(summary, manifest, retained_input_guard=retained_input_guard)
        else:
            published_result = publication.commit(summary, manifest)
        if packed is not None:
            return cast(PackedRunResult, published_result)
        return directory, summary  # no reportable checks after publication owner releases.
    except BaseException as error:
        if _final_validation_context is not None:
            _final_validation_context._record_failure(error)  # Invalidate FIRST, before reap/diagnostics.
        from gf_cli import _join_owned_groups
        try:
            await _join_owned_groups(tasks, wall_deadline, diagnostics)
        except BaseException as cleanup:
            if observation is None:
                raise
            vars(error)['hpo_group_cleanup_tasks'] = tasks
            error.add_note('owned group cleanup: ' + str(cleanup))
        if packed is not None:
            storage_cleanup: BaseException | None = None
            if observation is not None and packed_publication is not None:
                try:
                    if not packed_publication.committed:
                        packed_publication.revoke()
                except BaseException as cleanup:  # noqa: BLE001 -- independent Pack close still required.
                    storage_cleanup = cleanup
                    vars(error)['hpo_publication_cleanup_owner'] = packed_publication
            try:
                if not packed._closed and all(group.archived for group in packed._groups):
                    pending = await packed.prepare_diagnostic("failure.json",
                        {"error": type(error).__name__, "message": str(error), **diagnostics})
                    pending.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve initiating error and concrete cleanup owner.
                storage_cleanup = cleanup
            try:
                await packed.close()
            except BaseException as cleanup:  # noqa: BLE001 -- independent Writer/thread retirement is mandatory.
                storage_cleanup = storage_cleanup or cleanup
            if storage_cleanup is not None:
                vars(error)["cleanup_owner"] = packed
                error.add_note("packed collection cleanup: " + str(storage_cleanup))
            raise
        try:
            _write_diagnostic(directory, "failure.json", {"error": type(error).__name__, "message": str(error), **diagnostics})
        except OSError:
            pass
        raise


def _write_diagnostic(directory: Path, name: str, value: Json) -> None:
    with (directory / name).open("x", encoding="utf-8") as output:
        output.write(canonical(value) + "\n")


def _validate_online_collection(loaded: LoadedConfig, plan: CollectionPlan,
                                 ledger: ReservationLedger, deadline: float, *, resident_phase: Any = None) -> Any:
    from gf_training.card_instance_source import validate_owned_collection
    if not isinstance(loaded.config.training, PPOTrainingConfig):
        raise ContractError("online card-instance collection requires PPO training")
    online = validate_owned_collection(plan, ledger,
        max_owned_bytes=min(loaded.config.training.max_owned_bytes,
                            loaded.config.preprocessing.memory_budget_mib * 1024**2),
        deadline=deadline, **({} if resident_phase is None else {'resident_phase':resident_phase}))
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
    require_storage_command(loaded.config, "collect", standalone=False)
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
        result = await _run_corrected(loaded, "collect", plan.bindings,
            tuple(plan.bindings), started=started, artifact_inputs=owner, runtime_evidence=evidence,
            resource_admission=max(admissions, key=lambda value: value["peak_cuda_bytes"]),
            collection=plan, owned_ledger=ledger)
        if type(result) is PackedRunResult:
            from gf_training.collection import PackedFinalizedCollection
            return PackedFinalizedCollection(result.directory, result.report, result.locator,
                cast(str, result.envelope_sha256), result.storage_metrics)
        directory, summary = cast(tuple[Path, Json], result)
        return FinalizedCollection(directory, summary)
    finally:
        owner.release()


async def collect_hpo_ppo(loaded: LoadedConfig,plan: CollectionPlan,ledger: ReservationLedger) -> FinalizedCollection:
    """Known HPO-only composition of the same corrected collection owner loop."""
    from gf_cli.hpo_config import HPOExperimentConfig
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_hpo_game_workers
    from gf_training.collection import PackedFinalizedCollection
    config=loaded.config
    if type(config) is not HPOExperimentConfig:
        raise ContractError('concrete packed HPO runtime config required')
    if type(config.actions) is not ActionConfigV3 or not isinstance(config.run,RunConfigV2) or config.server.source_dir is None:
        raise ContractError('HPO collector requires owned corrected runtime/actions/run settings')
    require_storage_command(config,'collect',standalone=False)
    _require_actions(loaded)
    custody = getattr(ledger, '_resident_pool', None)
    resident_phase = None if custody is None else custody._admission
    online=_validate_online_collection(loaded,plan,ledger,plan.execution_deadline,resident_phase=resident_phase)
    resident = online.resident_runtime
    pool = None if resident is None else resident.pool
    from gf_training.hpo_source import require_hpo_production
    require_hpo_production(online.state)
    operating=config.hpo_spec['operating']
    if config.storage.budget_mib*1024**2>operating['collector_budget_bytes']:
        raise ContractError('actual packed collector exceeds HPO collection envelope')
    started=time.monotonic() if plan.started_monotonic is None else plan.started_monotonic
    work_deadline=plan.wall_deadline-config.run.report_reserve_seconds
    preflight_deadline=min(plan.execution_deadline,work_deadline-config.server.cleanup_reserve_seconds)
    evidence=CardInstanceRuntimeEvidence(Path(config.server.binary),Path(config.server.source_dir),deadline=preflight_deadline)
    evidence.preflight()
    # The retained registry shares the parent phase envelope with its already
    # admitted immutable config/plan graphs; collection bytes use Pack's account.
    parent_limit=hpo_artifact_budget(loaded,plan.bindings)
    owner=RetainedGameArtifacts.for_hpo_registry(plan.bindings,config.actions,
        pool_size=operating['opponent_pool_size'],rules=card_instance_rules(),
        max_owned_bytes=min(parent_limit,config.model.resources.max_owned_bytes),
        preflight_deadline=preflight_deadline,finalization_deadline=work_deadline)
    try:
        admissions=[admit_hpo_game_workers(config.actions,(plan.bindings[spec.policies[0]],plan.bindings[spec.policies[1]]),
            rules=card_instance_rules(),game_concurrency=config.run.concurrency,global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers,learner_cuda_bytes=ledger.learner.resident_bytes if pool is not None and ledger.learner is not None else ledger.cuda_bytes)
            for group in plan.groups for spec in group.episodes]
        if not admissions: raise ContractError('HPO collection cannot have an empty schedule')
        result=await _run_corrected(loaded,'collect',plan.bindings,tuple(plan.bindings),started=started,
            artifact_inputs=owner,runtime_evidence=evidence,
            resource_admission=max(admissions,key=lambda row:row['peak_cuda_bytes']),collection=plan,owned_ledger=ledger,**({} if pool is None else {'resident_pool':pool}))
        if type(result) is not PackedRunResult: raise ContractError('HPO collector did not return its actual packed publication')
        return PackedFinalizedCollection(result.directory,result.report,result.locator,cast(str,result.envelope_sha256),result.storage_metrics)
    finally:
        owner.release()


def hpo_artifact_budget(loaded: LoadedConfig,bindings: dict[str,RuntimePolicyBinding]) -> int:
    """Additional retained-custody allowance after the parent config/registry graphs."""
    from gf_cli.hpo_config import HPOExperimentConfig
    if type(loaded.config) is not HPOExperimentConfig:
        raise ContractError('concrete HPO runtime configuration required')
    reserved=32*(len(canonical(loaded.effective).encode())
        +sum(len(canonical(asdict(value)).encode()) for value in bindings.values()))+131072
    result=loaded.config.hpo_spec['operating']['parent_budget_bytes']-reserved
    if result<=0: raise ContractError('HPO parent metadata envelope exhausted before artifact custody')
    return result


def _require_hpo_evaluation_retirement(plan: HpoEvaluationPlan,gpu_job: GpuJob,
                                        ledger: ReservationLedger, *, resident_pool: Any = None) -> None:
    """Post-admission cleanup checks never reopen or extend the admission window."""
    from gf_cli.hpo_evaluation import HpoEvaluationPlan, ResidentHpoEvaluationPlan
    from gf_training.hpo_gpu_lease import GpuJob
    if type(plan) not in (HpoEvaluationPlan, ResidentHpoEvaluationPlan) or type(gpu_job) is not GpuJob:
        raise ContractError('exact held evaluation plan/job required')
    gpu_job._require_open()
    link=plan.linkage
    expected={'campaign_id':link['campaign_id'],'operation_id':link['evaluation_id'],'kind':'evaluation',
        'run_id':link['left_parent']['run_id'],'evaluation_id':link['evaluation_id']}
    if gpu_job.ticket!=link['gpu_ticket'] or gpu_job._admission['context']!=expected or ledger is not gpu_job.ledger:
        raise ContractError('evaluation cleanup differs from SAME job/context/ledger')
    if resident_pool is not None:
        resident_pool.authority.require(ledger)
        resident_pool.custody.evaluation_scope.require(gpu_job,plan)
    elif ledger.workers or ledger.learner is not None or ledger.updating or ledger.cuda_bytes:
        raise ContractError('evaluation still has live participant/learner ownership')
    plan.__post_init__()
    _check(min(plan.execution_deadline,plan.work_deadline,plan.wall_deadline))


async def collect_hpo_evaluation(loaded: LoadedConfig, plan: HpoEvaluationPlan, *, gpu_job: GpuJob,
                               artifact_inputs: RetainedGameArtifacts,
                               _cleanup_observation: _HpoEvaluationCleanupObservation) -> PackedRunResult:
    from gf_cli.hpo_evaluation import _HpoEvaluationCleanupObservation
    if type(_cleanup_observation) is not _HpoEvaluationCleanupObservation:
        raise ContractError('actual invocation-bound evaluation cleanup observation required')
    observation = _cleanup_observation
    observation._enter(plan, gpu_job, artifact_inputs)
    try:
        result = await _collect_hpo_evaluation(loaded, plan, gpu_job=gpu_job,
            artifact_inputs=artifact_inputs, _cleanup_observation=observation)
    except BaseException as error:
        observation._core_error, observation._core_phase = error, 'failed'
        raise
    observation._result, observation._core_phase = result, 'returned'
    return result


def _require_hpo_evaluation_policy(binding: RuntimePolicyBinding) -> None:
    options=binding.policy_options
    if options is None or options['version']!='4' or options['inference']['device']!='cuda:0':
        raise ContractError('HPO evaluation requires planned model4 CUDA neural inference')


async def _collect_hpo_evaluation(loaded: LoadedConfig,plan: HpoEvaluationPlan,*,gpu_job: GpuJob,
                                   artifact_inputs: RetainedGameArtifacts,
                                   _cleanup_observation: _HpoEvaluationCleanupObservation) -> PackedRunResult:
    """Borrow the outer evaluator's exact custody owner through the one collector."""
    from gf_cli.hpo_config import HPOExperimentConfig
    from gf_cli.hpo_evaluation import HpoEvaluationPlan, ResidentHpoEvaluationPlan
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_hpo_game_workers
    if type(plan) not in (HpoEvaluationPlan, ResidentHpoEvaluationPlan) or type(artifact_inputs) is not RetainedGameArtifacts:
        raise ContractError('evaluation requires its exact plan and actual retained artifact owner')
    plan.require(loaded,gpu_job=gpu_job)
    config=loaded.config
    if (type(config) is not HPOExperimentConfig or type(config.actions) is not ActionConfigV3
            or not isinstance(config.run,RunConfigV2) or config.server.source_dir is None):
        raise ContractError('evaluation requires concrete corrected packed HPO validation runtime')
    if canonical(loaded.effective)!=canonical(asdict(config)) or loaded.sha256!=digest(loaded.effective):
        raise ContractError('evaluation caller changed its effective configuration')
    require_storage_command(config,'evaluate',standalone=False)
    _require_actions(loaded)
    op=config.hpo_spec['operating']
    ledger=gpu_job.ledger
    if (ledger.global_cuda_bytes!=config.run.global_cuda_budget_bytes or ledger.max_workers!=config.run.max_participant_workers
            or ledger.global_cuda_bytes!=op['managed_cuda_bytes'] or ledger.max_workers!=op['max_participant_workers']
            or config.storage.budget_mib*1024**2>op['collector_budget_bytes']):
        raise ContractError('evaluation actual resource/storage limits differ from sole HPO spec')
    bindings=plan.bindings
    neural={name for name,value in bindings.items() if value.policy_options is not None}
    for value in bindings.values():
        if value.policy_options is not None:
            artifact_inputs.require_options(value.model_id,value.policy_options)
            _require_hpo_evaluation_policy(value)
    owned=artifact_inputs.finalize(deadline=min(plan.execution_deadline,plan.work_deadline))
    if set(owned['policies'])!=neural:
        raise ContractError('evaluation borrowed owner differs from exact frozen role bindings')
    if artifact_inputs._scan.peak>hpo_artifact_budget(loaded,bindings):
        raise ContractError('evaluation artifact custody exceeds additional parent envelope')
    evidence=CardInstanceRuntimeEvidence(Path(config.server.binary),Path(config.server.source_dir),
        deadline=min(plan.execution_deadline,plan.work_deadline-config.server.cleanup_reserve_seconds))
    evidence.preflight()
    admissions=[admit_hpo_game_workers(config.actions,(bindings[spec.policies[0]],bindings[spec.policies[1]]),
        rules=card_instance_rules(),game_concurrency=config.run.concurrency,global_cuda_bytes=ledger.global_cuda_bytes,
        max_workers=ledger.max_workers,learner_cuda_bytes=0) for group in plan.groups for spec in group.episodes]
    if not admissions: raise ContractError('empty HPO evaluation panel')
    resident_runtime = None
    if type(plan) is ResidentHpoEvaluationPlan:
        from gf_training.hpo_resident_runtime import ResidentEvaluationRuntime
        resident_runtime = ResidentEvaluationRuntime(gpu_job,plan,artifact_inputs,config)
        _cleanup_observation._resident_runtime = resident_runtime
    try:
        _cleanup_observation._run_phase = 'requested'
        result=await _run_corrected(loaded,'evaluate',bindings,tuple(bindings),started=plan.started_monotonic,
            artifact_inputs=artifact_inputs,runtime_evidence=evidence,resource_admission=max(admissions,key=lambda row:row['peak_cuda_bytes']),
            collection=plan,owned_ledger=ledger,_hpo_evaluation_job=gpu_job,
            _hpo_evaluation_cleanup=_cleanup_observation,
            **({} if resident_runtime is None else {'resident_pool':resident_runtime.pool}))
    finally:
        if resident_runtime is not None:
            await resident_runtime.close()
    _require_hpo_evaluation_retirement(plan,gpu_job,ledger)
    if (type(result) is not PackedRunResult or result.directory!=plan.output or result.report.get('healthy') is not True
            or result.report.get('comparison_valid') is not True or result.storage_metrics.get('publication_scope')!='published'):
        raise ContractError('evaluation collector did not complete its actual healthy packed publication')
    # The outer owner remains live for journal/Replay/Source verification and is
    # neither released nor replaced here. A packed result alone grants no score.
    artifact_inputs.finalize(deadline=min(plan.execution_deadline,plan.work_deadline))
    return result



def _validate_final_collection(loaded: LoadedConfig, plan: Any, ledger: ReservationLedger,
                               context: Any, deadline: float) -> None:
    from gf_training.card_instance_final_validation import (
        GameFinalValidationContext,
        GameFinalValidationPlan,
    )
    if type(context) is not GameFinalValidationContext or type(plan) is not GameFinalValidationPlan:
        raise ContractError('concrete actor-only final validation context/plan required')
    context.validate_collection(plan, ledger, loaded, deadline=deadline)


async def collect_owned_final_validation(context: Any) -> Any:
    """Normal corrected Environment under the SAME final-only actor context."""
    from gf_training.card_instance_final_validation import (
        GameFinalValidationContext,
        require_actor_backend,
    )
    if type(context) is not GameFinalValidationContext:
        raise ContractError('concrete actor-only final validation context required')
    loaded, plan, ledger = context.projected, context.plan, context.ledger
    require_storage_command(loaded.config, "validate-formal-policy", standalone=False)
    _validate_final_collection(loaded, plan, ledger, context, plan.execution_deadline)
    require_actor_backend(loaded)
    config = loaded.config
    work_deadline = plan.wall_deadline - config.run.report_reserve_seconds
    preflight_deadline = min(plan.execution_deadline, work_deadline - config.server.cleanup_reserve_seconds)
    evidence = CardInstanceRuntimeEvidence(Path(config.server.binary), Path(config.server.source_dir), deadline=preflight_deadline)
    evidence.preflight()
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_runtime import admit_game_workers_v3
    rules = card_instance_rules()
    owner = RetainedGameArtifacts.for_registry(plan.bindings, config.actions, rules=rules,
        max_owned_bytes=context.artifact_budget, preflight_deadline=preflight_deadline,
        finalization_deadline=work_deadline)
    try:
        admissions = [admit_game_workers_v3(config.actions,
            (plan.bindings[spec.policies[0]], plan.bindings[spec.policies[1]]), rules=rules,
            game_concurrency=config.run.concurrency, global_cuda_bytes=config.run.global_cuda_budget_bytes,
            max_workers=config.run.max_participant_workers, learner_cuda_bytes=0)
            for group in plan.groups for spec in group.episodes]
        directory, issued = cast(tuple[Path, Any], await _run_corrected(loaded, 'collect', plan.bindings, tuple(plan.bindings),
            started=plan.started_monotonic, artifact_inputs=owner, runtime_evidence=evidence,
            resource_admission=max(admissions, key=lambda value: value['peak_cuda_bytes']), collection=plan,
            owned_ledger=ledger, _final_validation_context=context))
        from gf_training.card_instance_final_validation import _IssuedFinalValidationCollection
        if type(issued) is not _IssuedFinalValidationCollection or issued.directory != directory:
            raise ContractError('collector did not retain the actual pending publication')
        return issued
    finally:
        owner.release()  # In-memory only; parent retains final artifact originals.


async def collect_formal_final_validation(loaded: LoadedConfig, parent: Path, output: Path,
                                          *, deadline: float | None = None) -> Json:
    """Explicit formal-only CLI/library composition; no legacy validation route."""
    require_storage_command(loaded.config, "validate-formal-policy", standalone=False)
    from gf_training.card_instance_final_validation import GameFinalValidationContext, _loaded
    from gf_training.card_instance_formal import FormalGameParent
    _loaded(loaded)  # Includes early Rust64 refusal, before ownership/runtime/output/model.
    deadline = time.monotonic() + loaded.config.run.wall_seconds if deadline is None else deadline
    if (type(deadline) not in (float, int) or not math.isfinite(deadline)
            or deadline <= time.monotonic() + loaded.config.run.finalization_seconds):
        raise ContractError('caller final validation duration cannot preserve finalization reserve')
    config = loaded.config
    assert isinstance(config.training, PPOTrainingConfig) and type(config.model) is ConfiguredNeuralModelV3
    assert isinstance(config.run, RunConfigV2)
    limit = min(config.training.max_owned_bytes, config.model.resources.max_owned_bytes,
        config.preprocessing.memory_budget_mib * 1024**2)
    original = FormalGameParent(parent, max_owned_bytes=limit, deadline=deadline)
    original.__enter__()
    context = None
    try:
        ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
        context = GameFinalValidationContext(loaded, original, output, ledger, deadline=deadline)
        await context.__aenter__()
        result = await collect_owned_final_validation(context)
        source = context.accept_collection(result)
        facts = source.facts
        await context._retire_for_owned_parent(original)
    except BaseException as error:
        if context is not None:
            context._record_failure(error)  # Revoke pending/complete manifest FIRST.
            try:
                await context._retire_for_owned_parent(original)
            except BaseException as cleanup:  # noqa: BLE001 - actual Source/actor owners stay reachable
                context._record_failure(cleanup)
            if not context.loan_cleanup_complete:
                vars(error)['_game_final_validation_cleanup_owner'] = context
                vars(error)['_game_final_validation_parent_owner'] = original
                raise
        elif getattr(error, '_game_final_validation_cleanup_parent', None) is original:
            vars(error)['_game_final_validation_parent_owner'] = original
            raise
        try:
            original.__exit__(type(error), error, error.__traceback__)
        except BaseException as cleanup:  # noqa: BLE001 - parent owns native/lock retry, preserve original
            vars(error)['_game_final_validation_parent_owner'] = original
            if context is not None:
                context._record_failure(cleanup)
        if context is not None:
            if original.cleanup_complete:
                try:
                    context._finish_after_owned_parent(original)
                except BaseException as cleanup:  # noqa: BLE001 - initiating error remains primary
                    context._record_failure(cleanup)
            else:
                vars(error)['_game_final_validation_cleanup_owner'] = context
                vars(error)['_game_final_validation_parent_owner'] = original
        raise
    else:
        try:
            original.__exit__(None, None, None)  # OWNED parent release precedes canonical exposure.
        except BaseException as error:
            context._record_failure(error)
            if original.cleanup_complete:
                try:
                    context._finish_after_owned_parent(original)
                except BaseException as cleanup:  # noqa: BLE001 - keep original release failure
                    context._record_failure(cleanup)
            else:
                vars(error)['_game_final_validation_cleanup_owner'] = context
                vars(error)['_game_final_validation_parent_owner'] = original
            raise
        public_facts = context._finish_after_owned_parent(original)
        if public_facts is not None:
            facts = public_facts
    return facts


def _game_tree_data_config(loaded: LoadedConfig, started: float, *, fit: bool = False
                           ) -> tuple[ConfiguredTreeModelV3, TrainingConfig, float]:
    """Pure current Config admission; neither metadata nor flags qualify Source."""
    require_storage_command(loaded.config, "game-dataset", standalone=False)
    from gf_common.card_instance_game_tree import MAX_OWNED_BYTES
    _require_actions(loaded)
    config = loaded.config
    if (type(config.model) is not ConfiguredTreeModelV3
            or type(config.training) is not TrainingConfig
            or type(config.server) is not CorrectedServerConfigV2
            or config.preprocessing.reader != "participant-structured-v2"):
        raise ContractError("game Tree data commands require corrected Tree3/actions3/server/participant reader and LightGBM training Config")
    if config.preprocessing.memory_budget_mib*1024**2 < MAX_OWNED_BYTES:
        raise ContractError("game Tree data commands retain the full joint8GiB allowance")
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ContractError("finite original command start required")
    seconds = config.preprocessing.wall_seconds
    if fit:
        if config.model.checkpoint is not None or config.training.wall_seconds > 7200:
            raise ContractError("game Tree fitting requires a fresh checkpoint=None and training wall<=7200")
        integer(config.model.resources.max_owned_bytes, 1, MAX_OWNED_BYTES)
        seconds = min(seconds, config.training.wall_seconds)
    deadline = started + seconds
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ContractError("original game Tree command deadline exhausted or nonfinite")
    return config.model, config.training, deadline


def _fresh_game_tree_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ContractError("game Tree output must be a fresh path")


def build_game_dataset(loaded: LoadedConfig, parents: tuple[Path, ...], validations: tuple[Path, ...],
                       output: Path, *, started: float) -> Json:
    model, _, deadline = _game_tree_data_config(loaded, started)
    if len(parents) != 3 or len(validations) != 3:
        raise ContractError("game Dataset requires all three ordered parents and validations")
    if any(not path.is_dir() for path in (*parents, *validations)):
        raise ContractError("all original game parent and validation directories must exist")
    _fresh_game_tree_output(output)
    from gf_common.card_instance_game_tree import MAX_OWNED_BYTES
    from gf_training.card_instance_game_tree_data import build_game_tree_dataset
    return build_game_tree_dataset(parents, validations, output, loaded.config.preprocessing,
        encoder=cast(TreeEncoderConfigV3, model.encoder), max_owned_bytes=MAX_OWNED_BYTES, deadline=deadline)


def inspect_game_dataset(loaded: LoadedConfig, manifest: Path, expected_digest: str, *, started: float) -> Json:
    _, _, deadline = _game_tree_data_config(loaded, started)
    from gf_models.structured_sealed import checksum
    checksum(expected_digest)
    if not manifest.is_file():
        raise ContractError("original game Dataset manifest must exist")
    from gf_common.card_instance_game_tree import MAX_OWNED_BYTES
    from gf_training.card_instance_game_tree_data import inspect_game_tree_dataset
    return inspect_game_tree_dataset(manifest, expected_digest=expected_digest,
        settings=loaded.config.preprocessing, max_owned_bytes=MAX_OWNED_BYTES, deadline=deadline)


async def fit_game_tree(loaded: LoadedConfig, manifest: Path, expected_digest: str, output: Path,
                        seed: int, *, started: float) -> Json:
    model, training, deadline = _game_tree_data_config(loaded, started, fit=True)
    from gf_models.structured_sealed import checksum
    checksum(expected_digest)
    integer(seed, 0, 2**31-1)
    if not manifest.is_file():
        raise ContractError("original game Dataset manifest must exist")
    _fresh_game_tree_output(output)
    from gf_common.card_instance_game_tree import MAX_OWNED_BYTES
    from gf_training.structured_tree_fit import fit_game_dataset
    return await fit_game_dataset(manifest, expected_digest=expected_digest, directory=output,
        model=model, settings=loaded.config.preprocessing, training=training, seed=seed,
        deadline=deadline, max_owned_bytes=MAX_OWNED_BYTES)
