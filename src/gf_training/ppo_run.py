"""Parent-only bounded PPO orchestration, commit ordering and pause/resume."""
from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import (
    REVIEWED_SPEC_HASHES,
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    digest,
)
from gf_models.arrays import sha256
from gf_models.config import (
    GRUModelConfig,
    MLPModelConfig,
    PPOTrainingConfig,
    SequencePPOTrainingConfig,
    TransformerModelConfig,
)
from gf_models.torch_runtime import NUMERICS, runtime_metadata
from gf_training.collection import (
    CollectionPlan,
    FinalizedCollection,
    OnlinePPOContext,
    training_groups,
)
from gf_training.journal import atomic_json
from gf_training.lineage import (
    checkpoint_path,
    commit_generation,
    owned_binding,
    read_json,
    retain,
    run_lock,
    validate_chain,
)
from gf_training.native_ppo import preprocessing_identity, stage_rollout
from gf_training.ppo import ingest_collection, write_rollout

Collector = Callable[[CollectionPlan], Awaitable[FinalizedCollection]]


def source_identity(deadline: float = float("inf")) -> Json:
    if time.monotonic() >= deadline:
        raise TimeoutError('PPO source identity deadline')
    package_names = ('gf_common', 'gf_models', 'gf_training', 'gf_cli',
                     'gf_evaluation', 'gf_simulator_client', 'godfieldbots')
    package_roots = {}
    for name in package_names:
        if time.monotonic() >= deadline:
            raise TimeoutError('PPO source discovery deadline')
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).name != '__init__.py':
            raise ContractError('owned Bots package source is unavailable')
        package_roots[name] = Path(spec.origin).resolve().parent
    module_path = Path(__file__).resolve()
    root = module_path.parent.parent.parent
    checkout = ((root / '.git').exists()
                and (root / 'pyproject.toml').is_file()
                and module_path == (root / 'src/gf_training/ppo_run.py').resolve()
                and all(path == (root / 'src' / name).resolve() for name, path in package_roots.items()))
    revision, dirty = None, None
    if checkout:
        try:
            revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=max(.001, min(5., deadline - time.monotonic())), check=True).stdout.strip()
            if time.monotonic() >= deadline:
                raise TimeoutError('source Git provenance deadline')
            dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, timeout=max(.001, min(5., deadline - time.monotonic())), check=True).stdout)
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
    hashes = {}
    for name, package_root in sorted(package_roots.items()):
        for path in sorted(package_root.rglob('*.py')):
            if time.monotonic() >= deadline:
                raise TimeoutError('PPO source hashing deadline')
            hashes[f'{name}/{path.relative_to(package_root).as_posix()}'] = sha256(path, deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('PPO source identity deadline')
    native_distributions = {}
    for distribution in importlib.metadata.distributions():
        if time.monotonic() >= deadline:
            raise TimeoutError('native distribution identity deadline')
        name = str(distribution.metadata.get('Name', '')).lower().replace('_', '-')
        if name.startswith('nvidia-') or name in {'cuda-bindings', 'cuda-toolkit', 'triton'}:
            native_distributions[name] = distribution.version
    return {'code': {'revision': revision, 'dirty': dirty, 'installed_build_revision': None}, 'module_sha256': hashes,
            'cuda_distributions': dict(sorted(native_distributions.items())),
            'runtime': runtime_metadata(), 'numerics': NUMERICS,
            'package': importlib.metadata.version('godfieldbots')}


def integration_identity(config: Any, deadline: float) -> Json:
    result: Json = {'binary_sha256': sha256(Path(config.server.binary), deadline),
                    'server_config_sha256': sha256(Path(config.server.config), deadline),
                    'declared_simulator_revision': config.server.simulator_revision,
                    'binary_build_revision': None, 'source_revision': None, 'source_spec_sha256': None}
    if config.server.source_dir is not None:
        source = Path(config.server.source_dir)
        if time.monotonic() >= deadline:
            raise TimeoutError('Simulator source provenance deadline')
        actual = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                                text=True, timeout=max(.001, min(5., deadline - time.monotonic()))).stdout.strip()
        if actual != config.server.simulator_revision:
            raise ContractError('Simulator checkout revision does not match accepted pin')
        hashes = {name: sha256(source / name, deadline) for name in REVIEWED_SPEC_HASHES}
        if hashes != REVIEWED_SPEC_HASHES:
            raise ContractError('Simulator normative source hashes differ from accepted contract')
        result.update(source_revision=actual, source_spec_sha256=hashes)
    return result


def owned_bytes(directory: Path, deadline: float = float("inf")) -> int:
    count = 0
    for path in directory.rglob('*'):
        if time.monotonic() >= deadline:
            raise TimeoutError('owned storage measurement deadline')
        if path.is_symlink():
            raise ContractError('symlink in owned PPO storage')
        if path.is_file():
            count += path.stat().st_size
    return count


class LearningProcess:
    def __init__(self, directory: Path, max_log_bytes: int) -> None:
        self.directory, self.limit = directory, max_log_bytes
        self.process: Any = None
        self.drainer: asyncio.Task[None] | None = None
        self.health: Json = {}
        self.close_task: asyncio.Task[None] | None = None
        self.creation_task: asyncio.Task[Any] | None = None

    def _environment(self) -> dict[str, str]:
        return {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

    def _process_created(self) -> None:
        pass

    def _close_request(self) -> Json:
        return {'command': 'close'}

    def _validate_close(self, result: Json) -> None:
        if result != {'closed': True}:
            raise ContractError('invalid learner close acknowledgement')

    async def start(self) -> None:
        environment = self._environment()
        creation = asyncio.create_task(asyncio.create_subprocess_exec(sys.executable, '-E', '-m', 'gf_training.ppo_worker',
                                      stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                      env=environment, limit=1024 * 1024))
        self.creation_task = creation
        cancelled = None
        while not creation.done():
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError as error:
                cancelled = cancelled or error
        self.process = creation.result()
        self._process_created()
        self.drainer = asyncio.create_task(self._drain())
        if cancelled:
            raise cancelled

    async def _drain(self) -> None:
        count = 0
        with (self.directory / f'learner-{uuid.uuid4().hex[:12]}.log').open('x') as output:
            while chunk := await self.process.stderr.read(65536):
                output.write(chunk[:max(0, self.limit - count)].decode('utf-8', errors='replace'))
                count += len(chunk)
                if count > self.limit:
                    self.health['log_limit_exceeded'] = True
        self.health['stderr_bytes'] = count

    async def request(self, request: Json, deadline: float) -> Json:
        if time.monotonic() >= deadline or self.health.get('log_limit_exceeded'):
            raise TimeoutError('PPO learner request deadline/log limit')
        async with asyncio.timeout_at(deadline):
            self.process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
        if not line or self.health.get('log_limit_exceeded'):
            raise RuntimeError('PPO numerical child failed; inspect bounded learner log')
        result = json.loads(line)
        if not isinstance(result, dict):
            raise ContractError('malformed numerical acknowledgement')
        return result

    async def _close(self, deadline: float) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.returncode is None:
                result = await self.request(self._close_request(), deadline)
                self._validate_close(result)
            async with asyncio.timeout_at(deadline):
                await process.wait()
                if self.drainer:
                    await self.drainer
            if process.returncode != 0 or self.health.get('log_limit_exceeded'):
                raise RuntimeError('learner close/health failure')
        except BaseException as error:
            self.health['cleanup_failure'] = str(error)
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), .5)
                except TimeoutError:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), 1.)
                    except TimeoutError:
                        # Record bounded reap failure; ownership survives delayed OS exit.
                        self.health['reap_failure'] = 'mandatory final join after kill grace'
                        await process.wait()
            if self.drainer:
                try:
                    await asyncio.wait_for(asyncio.shield(self.drainer), .5)
                except TimeoutError:
                    self.health['drain_failure'] = 'pipe EOF grace expired'
                    self.drainer.cancel()
                    await asyncio.gather(self.drainer, return_exceptions=True)
            raise
        finally:
            if process.stdin is not None:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.stdin.wait_closed(), .5)
                except (TimeoutError, BrokenPipeError, ConnectionResetError):
                    self.health['stdin_close_failure'] = True
            self.health['exit_code'] = process.returncode
            if time.monotonic() >= deadline:
                self.health['cleanup_overrun'] = True

    async def close(self, deadline: float) -> None:
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close(deadline))
        cancellation = None
        while not self.close_task.done():
            try:
                await asyncio.shield(self.close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:  # noqa: BLE001 - task.result preserves cleanup error
                break
        self.close_task.result()
        if cancellation:
            raise cancellation


def _controls(rng: Any, pool: list[int], collection_position: int) -> Json:
    return {'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': pool, 'collection_position': collection_position}


async def train_ppo(loaded: Any, collector: Collector, directory: Path, *, mode: str = 'fresh', pause_after_update: int | None = None) -> Json:
    from gf_common.action_contract import require_qualified_contract
    from gf_common.structured import ActionConfigV2
    if isinstance(loaded.config.actions, ActionConfigV2):
        require_qualified_contract(loaded.config.actions.action_contract)
    return await _run_ppo(loaded, collector, directory, mode=mode, pause_after_update=pause_after_update)


async def _run_ppo(loaded: Any, collector: Any, directory: Path, *, mode: str = 'fresh',
                   pause_after_update: int | None = None, _structured_component: bool = False,
                   _structured_ops: Any = None, _game_ops: Any = None) -> Json:
    operation_started = time.monotonic()
    operation_unix = time.time()
    config = loaded.config
    ops: Any = None
    game = _game_ops is not None
    if game:
        from gf_training import structured_ppo
        from gf_training.card_instance_ppo import _GameOperations
        if type(_game_ops) is not _GameOperations or _structured_component or _structured_ops is not None:
            raise ContractError("closed corrected game operations required")
        ops = _game_ops
    if _structured_ops is not None and not _structured_component:
        raise ContractError('private native operations require structured component composition')
    if _structured_component:
        from gf_training import structured_ppo
        if _structured_ops is not None:
            if type(_structured_ops) is not structured_ppo._NativeComponentOps:
                raise ContractError('concrete native structured operations required')
            if config.preprocessing.backend != 'rust' or config.preprocessing.reader != 'structured-v1':
                raise ContractError('native component operations require explicit rust/structured-v1')
        if _structured_ops is None and config.preprocessing.backend == 'rust':
            structured_ppo.validate_config(config)
            from gf_training.native_backend import load_native, reader_settings
            selected_native = load_native()
            reader_settings(config.preprocessing, selected_native)  # Real accepted marker/signature only.
            ops = structured_ppo._NativeComponentOps(selected_native)
            ops._implementation = 'native-structured-v2'
        else:
            ops = _structured_ops if _structured_ops is not None else structured_ppo
    formal = ops.formal if game else None
    if formal is not None:
        formal.preflight(loaded)
    model, training = ops.validate_config(config) if ops else config.model, config.training
    if not ops and (not isinstance(model, (MLPModelConfig, GRUModelConfig, TransformerModelConfig)) or not isinstance(training, PPOTrainingConfig) or model.checkpoint is not None or config.run.split != 'train'):
        raise ContractError('PPO requires fresh supported model recipe and train split')
    if not ops and isinstance(model, (GRUModelConfig, TransformerModelConfig)) != isinstance(training, SequencePPOTrainingConfig):
        raise ContractError("PPO model/sequence training mismatch")
    if config.server.external_url is not None:
        raise ContractError('PPO requires owned distinct-seed Simulator groups')
    if not (config.run.wall_seconds <= 7200 and config.run.admission_seconds <= 5400 and config.run.execution_seconds <= 7140 and config.run.finalization_seconds >= 60):
        raise ContractError('PPO original deadline caps exceeded')
    if mode not in ('fresh', 'resume'):
        raise ContractError('unsupported PPO mode')
    if mode == 'fresh' and pause_after_update is not None and (type(pause_after_update) is not int or not 0 < pause_after_update <= training.max_updates):
        raise ContractError('invalid absolute pause target')
    preprocessing_options: Json = {'model_id': model.model_id} if isinstance(model, (GRUModelConfig, TransformerModelConfig)) else {}
    preprocessing = ((ops.preprocessing_identity(config, deadline=operation_started + config.run.execution_seconds) if ops else preprocessing_identity(config.preprocessing, deadline=operation_started + config.run.execution_seconds, **preprocessing_options))
                     if mode == 'fresh' else None)
    directory = directory.resolve()
    if game and mode == "fresh":
        ops.preflight(config, deadline=operation_started + config.run.execution_seconds)
    with run_lock(directory, create=mode == 'fresh') as ownership:
        initial_scan = None
        if ops and mode == 'resume':
            initial_state, initial_scan = ops.read_control(directory / 'run.json', config,
                deadline=operation_started + config.run.execution_seconds)
            identity_deadline = operation_started + ops.saved_deadline(initial_state) - operation_unix
            del initial_state
        else:
            identity_deadline = operation_started + config.run.execution_seconds if mode == 'fresh' else operation_started + read_json(directory / 'run.json')['execution_deadline_unix'] - operation_unix
        if preprocessing is None:
            preprocessing = (ops.preprocessing_identity(config, deadline=identity_deadline) if ops else preprocessing_identity(config.preprocessing, deadline=identity_deadline, **preprocessing_options))
        identity = source_identity(identity_deadline)
        identity["integration"] = ops.integration_identity(config, identity_deadline) if game else integration_identity(config, identity_deadline)
        identity["configuration_inputs_sha256"] = digest(loaded.inputs)
        identity["preprocessing"] = preprocessing
        if ops:
            ops.decorate_identity(identity)
        if formal is not None:
            formal.attach_identity(directory, identity, ops.execution(config), mode=mode, deadline=identity_deadline, ownership=ownership)
        invocation_started = operation_started
        now = operation_unix
        if mode == 'fresh':
            for name in ('generations', 'updates', 'artifacts', 'collections', 'numeric'):
                (directory / name).mkdir()
            state = {'schema_version': 1, 'algorithm': 'ppo', 'run_id': str(uuid.uuid4()), 'status': 'running',
                     'config_sha256': loaded.sha256, 'effective_config': loaded.effective, 'config_inputs': list(loaded.inputs), 'identity': identity,
                     'seed': training.seeds[0], 'device': training.device, 'created_unix': now, 'last_unix': now,
                     'wall_deadline_unix': now + config.run.wall_seconds,
                     'admission_deadline_unix': now + config.run.admission_seconds,
                     'execution_deadline_unix': now + config.run.execution_seconds,
                     'updates': 0, 'accepted_decisions': 0, 'optimizer_steps': 0, 'generation_sha256': None,
                     'collection_seconds': 0., 'update_seconds': 0., 'invocations': [], 'completed_updates': None}
            if ops:
                ops.initialize_state(state, config)
            if ops:
                ops.persist_control(directory / 'artifact-registry.json', {'schema_version': 3 if game else 2, 'generations': {}}, config, deadline=identity_deadline)
                ops.persist_control(directory / 'run.json', state, config, deadline=identity_deadline)
            else:
                atomic_json(directory / 'artifact-registry.json', {})
                atomic_json(directory / 'run.json', state)
            rng = np.random.Generator(np.random.PCG64(training.seeds[0]))
            pool = [0]
        else:
            state, records = (ops.validate_parent(directory, config, identity_deadline,
                extra_owned_bytes=initial_scan.retained if initial_scan is not None else 0)
                if ops else validate_chain(directory, require_complete=False, deadline=identity_deadline))
            if initial_scan is not None:
                initial_scan.finalize()
            if state['status'] != 'paused' or state['updates'] == 0:
                raise ContractError('resume requires an explicitly paused committed PPO run')
            if state['config_sha256'] != loaded.sha256 or state['identity'] != identity or state['device'] != training.device:
                raise ContractError('resume code/runtime/config/device mismatch')
            if ops and canonical(state['execution']) != canonical(ops.execution(config)):
                raise ContractError('resume execution binding differs from current Config')
            if now < state['last_unix'] or now >= state['wall_deadline_unix'] or now >= state['execution_deadline_unix']:
                raise ContractError('resume clock rollback or original deadline expired')
            controls = records[-1]['parent_controls']
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = controls['schedule_opponent_rng']
            pool = controls['snapshot_pool']
            if controls['collection_position'] != state['updates'] or pool != [number for number in range(state['updates'] + 1) if number % 10 == 0][-5:]:
                raise ContractError('resume collection position/pool mismatch')
        if pause_after_update is not None and (type(pause_after_update) is not int or pause_after_update <= state['updates'] or pause_after_update > training.max_updates):
            raise ContractError('pause target must exceed current absolute committed-update count')
        remaining = state['wall_deadline_unix'] - now
        wall = ((invocation_started + state['wall_deadline_unix']) - now) if formal is not None else invocation_started + remaining
        execution = invocation_started + state['execution_deadline_unix'] - now
        admission = invocation_started + state['admission_deadline_unix'] - now
        # Even initialization and source-identity work consume the original clock.
        state['status'] = 'running'
        if ops:
            ops.persist_control(directory / 'run.json', state, config, deadline=execution, original=initial_scan)
            initial_scan = None
        else:
            atomic_json(directory / 'run.json', state)
        ledger = None
        if game:
            ops.bind_run(ownership, state)
        if ops:
            child, ledger = ops.make_child(directory, config, state, deadline=execution)
        else:
            child = LearningProcess(directory, training.max_log_bytes)
        error: BaseException | None = None
        final_status = 'complete'
        try:
            await child.start()
            request = {'command': 'initialize', 'model': asdict(model), 'seed': training.seeds[0], 'device': training.device,
                       'output': str(checkpoint_path(directory, 0)), 'action_hash': digest(asdict(config.actions))}
            if isinstance(training, SequencePPOTrainingConfig):
                request['sequence_length'] = training.sequence_length
            if mode == 'resume':
                request['restore'] = str(checkpoint_path(directory, state['updates']))
            initial = (await child.initialize(training.seeds[0], checkpoint_path(directory, 0),
                restore=checkpoint_path(directory, state['updates']) if mode == 'resume' else None, deadline=execution)
                if ops else await child.request(request, execution))
            if mode == 'fresh':
                state['hardware'] = initial['hardware']
                if ops:
                    ops.commit(directory, state, config, generation=0, update=None, controls=_controls(rng, pool, 0), acknowledgement=initial, deadline=execution)
                else:
                    commit_generation(directory, state, generation=0, update=None, controls=_controls(rng, pool, 0), deadline=execution)
            elif initial['hardware'] != state['hardware']:
                raise ContractError('resume hardware identity mismatch')
            def check_clock() -> None:
                if time.time() < operation_unix + (time.monotonic() - operation_started) - .1:
                    raise ContractError('wall clock rollback during PPO invocation')
            while True:
                check_clock()
                usage = owned_bytes(directory, execution)
                if formal is not None:
                    observed = time.monotonic()
                    if formal.observe_stop(state, operation_unix, invocation_started, observed, usage):
                        break
                else:
                    if usage > training.max_owned_bytes or time.monotonic() >= execution:
                        raise RuntimeError('PPO actual execution/storage limit exceeded')
                    if state['updates'] >= training.max_updates or time.monotonic() >= admission or usage + training.admission_reservation_bytes > training.max_owned_bytes:
                        if not state['updates']:
                            raise ContractError('zero-update PPO run ineligible')
                        break
                with ExitStack() as generation_scope:
                    generation = state['updates']
                    registry = ({'random-card-instance-v2': RuntimePolicyBinding('random'), 'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic')} if game else
                                {'random-structured-v1': RuntimePolicyBinding('random'), 'heuristic-structured-v1': RuntimePolicyBinding('heuristic')}
                                if ops else {'random': RuntimePolicyBinding('random'), 'heuristic': RuntimePolicyBinding('heuristic')})
                    learner_label, learner_binding = ops.binding(directory, generation, config, execution) if ops else owned_binding(directory, generation)
                    registry[learner_label] = learner_binding
                    snapshots = []
                    for snapshot in pool:
                        label, binding = ops.binding(directory, snapshot, config, execution) if ops else owned_binding(directory, snapshot)
                        registry[label] = binding
                        snapshots.append(label)
                    groups = (training_groups(training.seeds[0], generation, learner_label, tuple(snapshots), rng, actions_version=3 if game else 2)
                              if ops else training_groups(training.seeds[0], generation, learner_label, tuple(snapshots), rng))
                    batch_id = f'batch-{generation:06}'
                    linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(directory), 'batch_id': batch_id,
                               'collection_generation': generation,
                               'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
                    if game:
                        linkage = ops.linkage(state, directory, generation, groups, registry)
                    collection_started = time.monotonic()
                    plan = CollectionPlan(groups, registry, directory / 'collections' / batch_id, linkage, execution, wall)
                    if ops:
                        assert ledger is not None
                        ops.validate_owned_collection(plan, ledger, config=config, deadline=execution)
                        collection = await collector(plan, ledger)
                        if ledger.workers or ledger.updating:
                            raise ContractError('structured collection returned unreaped participants or active update')
                    else:
                        collection = await collector(plan)
                    if collection.directory.resolve() != (directory / 'collections' / batch_id).resolve() or collection.report.get('comparison_valid') is not True:
                        raise ContractError('collection callback did not finalize requested original batch')
                    check_clock()
                    from gf_common.schedule import serialize_schedule
                    collection_scan = None
                    if ops:
                        collection_manifest, collection_scan = ops.read_control(collection.directory / 'manifest.json', config, deadline=execution)
                    else:
                        collection_manifest = read_json(collection.directory / 'manifest.json')
                    expected_bindings = {key: asdict(value) for key, value in registry.items()}
                    bindings_match = (canonical(collection_manifest.get('policy_bindings')) == canonical(expected_bindings)
                                      if ops else collection_manifest.get('policy_bindings') == expected_bindings)
                    if collection_manifest.get('schedule') != serialize_schedule(groups, 4) or not bindings_match:
                        raise ContractError('collection callback changed frozen schedule/policies')
                    collection_seconds = time.monotonic() - collection_started
                    numeric = directory / 'numeric' / batch_id
                    numeric_files = None
                    with ExitStack() as source_scope:
                        if ops:
                            assert collection_scan is not None
                            projection_options: dict[str, Any] = {
                                'expected_manifest_sha256': collection_scan.files[collection.directory / 'manifest.json'][1],
                                'extra_owned_bytes': collection_scan.retained,
                            }
                            if type(ops) is structured_ppo._NativeComponentOps:
                                # Actual remaining owned storage bounds the private
                                # spool. The collection-admission reserve is separate.
                                projection_options['scratch_allowance_bytes'] = max(
                                    0, training.max_owned_bytes - owned_bytes(directory, execution))
                            projected = (generation_scope if game else source_scope).enter_context(
                                ops.project(plan, numeric, config, learner_label, generation, execution,
                                    **projection_options))
                            if game:
                                decision_count, proof, numeric_files, issued_projection = projected
                            else:
                                decision_count, proof, numeric_files = projected
                        else:
                            online_context = OnlinePPOContext(directory, state["run_id"], batch_id, generation, ownership.token)
                            if config.preprocessing.backend == 'rust':
                                decision_count, proof = await stage_rollout(collection.directory, numeric, model, linkage,
                                                                     config.preprocessing, deadline=execution,
                                                                     online_context=online_context,
                                                                     expected_identity=preprocessing, wall_deadline=wall)
                            else:
                                decisions, proof = ingest_collection(collection.directory, model, linkage, deadline=execution,
                                                                     online_context=online_context)
                                if isinstance(model, (GRUModelConfig, TransformerModelConfig)):
                                    from gf_training.sequence_rollout import (
                                        write_rollout as write_sequence,
                                    )
                                    write_sequence(numeric, decisions, model, deadline=execution)
                                else:
                                    write_rollout(numeric, decisions, deadline=execution)
                                decision_count = len(decisions)
                                del decisions
                        numeric_hash = sha256(numeric / 'rollout.json', execution)
                        if owned_bytes(directory, execution) > training.max_owned_bytes or time.monotonic() >= execution:
                            raise RuntimeError('PPO collection/encoding exceeded execution/storage limits')
                        next_generation = generation + 1
                        await asyncio.sleep(0)  # Serve pending cancellation before optimizer side effects.
                        update_started = time.monotonic()
                        if ops:
                            acknowledgement = (await child.update(issued_projection, checkpoint_path(directory, next_generation), deadline=execution) if game else
                                await child.update(numeric, numeric_hash, learner_label,
                                checkpoint_path(directory, next_generation), deadline=execution))
                            metrics = {**acknowledgement['metrics'], **acknowledgement['timings']}
                        else:
                            metrics = await child.request({'command': 'update', 'batch': str(numeric), 'batch_hash': numeric_hash,
                                                           'output': str(checkpoint_path(directory, next_generation)), 'action_hash': digest(asdict(config.actions))}, execution)
                        check_clock()
                        update_seconds = time.monotonic() - update_started
                        if time.monotonic() >= execution or owned_bytes(directory, execution) > training.max_owned_bytes:
                            raise RuntimeError('PPO update/save execution/storage overrun')
                    if collection_scan is not None:
                        collection_scan.finalize()
                    state['accepted_decisions'] += decision_count
                    state['optimizer_steps'] = metrics['optimizer_steps']
                    state['collection_seconds'] += collection_seconds
                    state['update_seconds'] += update_seconds
                    if next_generation % 10 == 0:
                        pool = (pool + [next_generation])[-5:]
                    update = {'schema_version': 1, 'status': 'complete', 'run_id': state['run_id'], 'generation': next_generation,
                              'parent_sha256': state['generation_sha256'], 'batch_id': batch_id,
                              'collection': str(collection.directory.relative_to(directory)), 'collection_manifest_sha256': proof['manifest_sha256'],
                              'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': numeric_hash,
                              'accepted_decisions': decision_count, 'source_proof': proof, 'metrics': metrics,
                              'planned_games': 16, 'actual_games': proof['games'], 'health': {},
                              'policy_bindings': {key: asdict(value) for key, value in registry.items()},
                              'learner_seats': linkage['learner_seats'], 'collection_seconds': collection_seconds, 'update_seconds': update_seconds}
                    if ops:
                        if game:
                            ops.update_record(update, state, numeric_files)
                        else:
                            update.update(schema_version=2, model_id=state['model_id'], model_version='2', input_recipe=state['input_recipe'],
                                          action_contract=state['action_contract'], purpose='component', synthetic_capacity=False,
                                          numeric_files=numeric_files)
                        ops.commit(directory, state, config, generation=next_generation, update=update,
                            controls=_controls(rng, pool, next_generation), acknowledgement=acknowledgement, deadline=execution)
                        ops.retain(directory, next_generation, config, execution)
                    else:
                        commit_generation(directory, state, generation=next_generation, update=update, controls=_controls(rng, pool, next_generation), deadline=execution)
                        retain(directory, next_generation, deadline=execution)
                check_clock()
                if time.monotonic() >= execution or owned_bytes(directory, execution) > training.max_owned_bytes:
                    raise RuntimeError('PPO commit/retention execution/storage overrun')
                state['last_unix'] = time.time()
                if pause_after_update == next_generation:
                    final_status = 'paused'
                    break
        except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
            error = failure
        finally:
            try:
                await child.close(min(wall, time.monotonic() + training.cleanup_seconds))
            except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
                error = error or failure
                state['cleanup_failure'] = str(failure)
        if set(child.health) - {'exit_code', 'stderr_bytes'} or child.health.get('exit_code') != 0:
            error = error or RuntimeError('PPO learner cleanup health failed')
        if time.time() < operation_unix + (time.monotonic() - operation_started) - .1:
            error = error or ContractError('wall clock rollback during PPO cleanup')
        state['last_unix'] = time.time()
        state['invocations'].append({'mode': mode, 'elapsed_seconds': time.monotonic() - invocation_started, 'child_health': child.health})
        state['elapsed_seconds'] = state['last_unix'] - state['created_unix']
        try:
            storage_bytes = owned_bytes(directory, wall)
        except TimeoutError as failure:
            storage_bytes = None
            error = error or failure
        state['artifact_bytes'] = storage_bytes
        state['peak_coordinator_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        state['coordinator_memory_scope'] = 'process lifetime peak RSS, distinct from learner child metrics'
        state['timing_scope'] = 'pre-final-write snapshot; original wall includes paused time'
        if error is not None or time.monotonic() >= wall or storage_bytes is None or storage_bytes > training.max_owned_bytes:
            state.update(status='failed', failure=type(error).__name__ if error else 'LimitOverrun', quarantine=True)
        else:
            state['status'] = final_status
            state['completed_updates'] = state['updates'] if final_status == 'complete' else None
        try:
            if ops:
                ops.persist_control(directory / 'run.json', state, config, deadline=wall)
            else:
                atomic_json(directory / 'run.json', state)
            if time.monotonic() >= wall or owned_bytes(directory, wall) > training.max_owned_bytes:
                if formal is not None:
                    raise RuntimeError('formal post-persistence wall/storage overrun')
                state.update(status='failed', failure='FinalPersistenceOverrun', quarantine=True)
                if ops:
                    ops.persist_control(directory / 'run.json', state, config, deadline=wall)
                else:
                    atomic_json(directory / 'run.json', state)
        except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
            state.update(status='failed', failure=type(failure).__name__, quarantine=True)
            if ops:
                try:
                    ops.persist_control(directory / 'run.json', state, config, deadline=wall)
                except BaseException as correction:  # noqa: BLE001 - initiating failure survives best-effort correction
                    failure.add_note('structured state correction failed: ' + type(correction).__name__)
            else:
                try:
                    atomic_json(directory / 'run.json', state)
                except OSError:
                    pass
            error = error or failure
        if error is not None:
            raise error
        if state['status'] == 'failed':
            raise RuntimeError('PPO run failed original limits')
        return {'run_dir': str(directory), 'status': state['status'], 'updates': state['updates'], 'accepted_decisions': state['accepted_decisions'],
                'elapsed_seconds': time.time() - state['created_unix'], 'generation_sha256': state['generation_sha256']}
