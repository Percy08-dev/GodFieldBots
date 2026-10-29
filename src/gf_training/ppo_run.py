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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import REVIEWED_SPEC_HASHES, ContractError, Json, RuntimePolicyBinding, digest
from gf_models.config import MLPModelConfig, PPOTrainingConfig
from gf_models.mlp.arrays import sha256
from gf_models.mlp.model import NUMERICS, runtime_metadata
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

    async def start(self) -> None:
        environment = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        creation = asyncio.create_task(asyncio.create_subprocess_exec(sys.executable, '-E', '-m', 'gf_training.ppo_worker',
                                      stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                      env=environment, limit=1024 * 1024))
        cancelled = None
        while not creation.done():
            try:
                await asyncio.shield(creation)
            except asyncio.CancelledError as error:
                cancelled = cancelled or error
        self.process = creation.result()
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
                result = await self.request({'command': 'close'}, deadline)
                if result != {'closed': True}:
                    raise ContractError('invalid learner close acknowledgement')
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
    operation_started = time.monotonic()
    operation_unix = time.time()
    config = loaded.config
    model, training = config.model, config.training
    if not isinstance(model, MLPModelConfig) or not isinstance(training, PPOTrainingConfig) or model.checkpoint is not None or config.run.split != 'train':
        raise ContractError('PPO requires fresh MLP recipe and train split')
    if config.server.external_url is not None:
        raise ContractError('PPO requires owned distinct-seed Simulator groups')
    if not (config.run.wall_seconds <= 7200 and config.run.admission_seconds <= 5400 and config.run.execution_seconds <= 7140 and config.run.finalization_seconds >= 60):
        raise ContractError('PPO original deadline caps exceeded')
    if mode not in ('fresh', 'resume'):
        raise ContractError('unsupported PPO mode')
    if mode == 'fresh' and pause_after_update is not None and (type(pause_after_update) is not int or not 0 < pause_after_update <= training.max_updates):
        raise ContractError('invalid absolute pause target')
    preprocessing = (preprocessing_identity(config.preprocessing, deadline=operation_started + config.run.execution_seconds)
                     if mode == 'fresh' else None)
    directory = directory.resolve()
    with run_lock(directory, create=mode == 'fresh') as ownership:
        identity_deadline = operation_started + config.run.execution_seconds if mode == 'fresh' else operation_started + read_json(directory / 'run.json')['execution_deadline_unix'] - operation_unix
        if preprocessing is None:
            preprocessing = preprocessing_identity(config.preprocessing, deadline=identity_deadline)
        identity = source_identity(identity_deadline)
        identity["integration"] = integration_identity(config, identity_deadline)
        identity["configuration_inputs_sha256"] = digest(loaded.inputs)
        identity["preprocessing"] = preprocessing
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
            atomic_json(directory / 'artifact-registry.json', {})
            atomic_json(directory / 'run.json', state)
            rng = np.random.Generator(np.random.PCG64(training.seeds[0]))
            pool = [0]
        else:
            state, records = validate_chain(directory, require_complete=False, deadline=identity_deadline)
            if state['status'] != 'paused' or state['updates'] == 0:
                raise ContractError('resume requires an explicitly paused committed PPO run')
            if state['config_sha256'] != loaded.sha256 or state['identity'] != identity or state['device'] != training.device:
                raise ContractError('resume code/runtime/config/device mismatch')
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
        wall = invocation_started + remaining
        execution = invocation_started + state['execution_deadline_unix'] - now
        admission = invocation_started + state['admission_deadline_unix'] - now
        # Even initialization and source-identity work consume the original clock.
        state['status'] = 'running'
        atomic_json(directory / 'run.json', state)
        child = LearningProcess(directory, training.max_log_bytes)
        error: BaseException | None = None
        final_status = 'complete'
        try:
            await child.start()
            request = {'command': 'initialize', 'model': asdict(model), 'seed': training.seeds[0], 'device': training.device,
                       'output': str(checkpoint_path(directory, 0)), 'action_hash': digest(asdict(config.actions))}
            if mode == 'resume':
                request['restore'] = str(checkpoint_path(directory, state['updates']))
            initial = await child.request(request, execution)
            if mode == 'fresh':
                state['hardware'] = initial['hardware']
                commit_generation(directory, state, generation=0, update=None, controls=_controls(rng, pool, 0), deadline=execution)
            elif initial['hardware'] != state['hardware']:
                raise ContractError('resume hardware identity mismatch')
            def check_clock() -> None:
                if time.time() < operation_unix + (time.monotonic() - operation_started) - .1:
                    raise ContractError('wall clock rollback during PPO invocation')
            while True:
                check_clock()
                usage = owned_bytes(directory, execution)
                if usage > training.max_owned_bytes or time.monotonic() >= execution:
                    raise RuntimeError('PPO actual execution/storage limit exceeded')
                if state['updates'] >= training.max_updates or time.monotonic() >= admission or usage + training.admission_reservation_bytes > training.max_owned_bytes:
                    if not state['updates']:
                        raise ContractError('zero-update PPO run ineligible')
                    break
                generation = state['updates']
                registry = {'random': RuntimePolicyBinding('random'), 'heuristic': RuntimePolicyBinding('heuristic')}
                learner_label, learner_binding = owned_binding(directory, generation)
                registry[learner_label] = learner_binding
                snapshots = []
                for snapshot in pool:
                    label, binding = owned_binding(directory, snapshot)
                    registry[label] = binding
                    snapshots.append(label)
                groups = training_groups(training.seeds[0], generation, learner_label, tuple(snapshots), rng)
                batch_id = f'batch-{generation:06}'
                linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(directory), 'batch_id': batch_id,
                           'collection_generation': generation,
                           'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
                collection_started = time.monotonic()
                collection = await collector(CollectionPlan(groups, registry, directory / 'collections' / batch_id, linkage, execution, wall))
                if collection.directory.resolve() != (directory / 'collections' / batch_id).resolve() or collection.report.get('comparison_valid') is not True:
                    raise ContractError('collection callback did not finalize requested original batch')
                check_clock()
                from gf_common.schedule import serialize_schedule
                collection_manifest = read_json(collection.directory / 'manifest.json')
                if collection_manifest.get('schedule') != serialize_schedule(groups, 4) or collection_manifest.get('policy_bindings') != {key: asdict(value) for key, value in registry.items()}:
                    raise ContractError('collection callback changed frozen schedule/policies')
                collection_seconds = time.monotonic() - collection_started
                numeric = directory / 'numeric' / batch_id
                online_context = OnlinePPOContext(directory, state["run_id"], batch_id, generation, ownership.token)
                if config.preprocessing.backend == 'rust':
                    decision_count, proof = await stage_rollout(collection.directory, numeric, model, linkage,
                                                         config.preprocessing, deadline=execution,
                                                         online_context=online_context,
                                                         expected_identity=preprocessing, wall_deadline=wall)
                else:
                    decisions, proof = ingest_collection(collection.directory, model, linkage, deadline=execution,
                                                         online_context=online_context)
                    write_rollout(numeric, decisions, deadline=execution)
                    decision_count = len(decisions)
                    del decisions
                numeric_hash = sha256(numeric / 'rollout.json', execution)
                if owned_bytes(directory, execution) > training.max_owned_bytes or time.monotonic() >= execution:
                    raise RuntimeError('PPO collection/encoding exceeded execution/storage limits')
                next_generation = generation + 1
                await asyncio.sleep(0)  # Serve pending cancellation before optimizer side effects.
                update_started = time.monotonic()
                metrics = await child.request({'command': 'update', 'batch': str(numeric), 'batch_hash': numeric_hash,
                                               'output': str(checkpoint_path(directory, next_generation)), 'action_hash': digest(asdict(config.actions))}, execution)
                check_clock()
                update_seconds = time.monotonic() - update_started
                if time.monotonic() >= execution or owned_bytes(directory, execution) > training.max_owned_bytes:
                    raise RuntimeError('PPO update/save execution/storage overrun')
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
            atomic_json(directory / 'run.json', state)
            if time.monotonic() >= wall or owned_bytes(directory, wall) > training.max_owned_bytes:
                state.update(status='failed', failure='FinalPersistenceOverrun', quarantine=True)
                atomic_json(directory / 'run.json', state)
        except BaseException as failure:  # noqa: BLE001 - persist failure after owned cleanup
            state.update(status='failed', failure=type(failure).__name__, quarantine=True)
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
