"""Predetermined public HTTP components; no executable Simulator or engine copy."""
from __future__ import annotations

import asyncio
import itertools
import json
import stat
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from gf_cli.config import LoadedConfig, RunConfigV2, load_config
from gf_common.action_contract import engineering_contract
from gf_common.domain import digest
from gf_common.preprocessing import PreparationConfig
from gf_common.structured import ActionConfigV2
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import model_defaults_v2
from gf_simulator_client.http import AttestedClient, seed_fingerprint
from gf_simulator_client.process import server_effective
from gf_training.collection import FinalizedCollection
from tests.episode_support import PublicScript
from tests.ppo_support import synthetic_server_identity


def component_inventory(parent):
    """Fingerprint the original reference, including directory membership."""
    from gf_models.arrays import sha256
    assert parent.is_dir()
    inventory = []
    for path in sorted(parent.rglob('*')):
        mode = path.lstat().st_mode
        assert stat.S_ISDIR(mode) or stat.S_ISREG(mode)
        inventory.append((str(path.relative_to(parent)), None if stat.S_ISDIR(mode) else sha256(path)))
    return tuple(inventory)


@dataclass(frozen=True)
class CompletedComponentExecution:
    """Only completed artifacts and closed diagnostic owners; never live work."""

    loaded: LoadedConfig
    parent: Path
    result: dict[str, Any]
    scripts: tuple[Any, ...]
    workers: tuple[Any, ...]
    ledgers: tuple[Any, ...]
    inventory: tuple[tuple[str, str | None], ...]

    def assert_closed(self):
        assert len(self.workers) == 64 and len(self.scripts) == 16 and self.ledgers
        assert all(worker.closed and worker.process.poll() is not None
                   and (worker.io_thread is None or not worker.io_thread.is_alive())
                   and worker.lease.released for worker in self.workers)
        assert all(script.deleted for script in self.scripts)
        assert all(ledger is self.ledgers[0] and ledger.learner is None
                   and not ledger.workers and not ledger.updating and ledger.cuda_bytes == 0
                   for ledger in self.ledgers)

    def assert_unchanged(self):
        assert component_inventory(self.parent) == self.inventory

    def assert_default_cpu_reference(self, *, kind, learning_device, inference_device):
        assert (kind, learning_device, inference_device) == ('mlp', 'cpu', 'cpu'), 'reference requires default CPU MLP'
        assert self.loaded.config.model == model_defaults_v2('mlp'), 'reference model must be default CPU MLP'
        assert self.loaded.config.training == PPOTrainingConfig(seeds=(17,), device='cpu', max_updates=2), 'reference training configuration differs'
        assert self.loaded.effective == asdict(self.loaded.config), 'reference configuration differs'
        assert self.loaded.sha256 == digest(self.loaded.effective), 'reference configuration digest differs'
        assert self.result['status'] == 'complete' and self.result['updates'] == 2 and self.result['accepted_decisions'] == 64
        self.assert_closed()
        self.assert_unchanged()
        state = json.loads((self.parent / 'run.json').read_text())
        assert digest(state['effective_config']) == self.loaded.sha256, 'recorded reference configuration differs'
        assert state['config_sha256'] == self.loaded.sha256, 'recorded reference configuration digest differs'
        assert state['status'] == 'complete' and state['updates'] == 2


def configuration(tmp_path, *, kind='mlp', learning_device='cpu', inference_device='cpu', max_updates=2, hand=None):
    tmp_path.mkdir(exist_ok=True)
    loaded = load_config('configs/experiments/ppo-cpu.toml', {'server': synthetic_server_identity(tmp_path)})
    model = model_defaults_v2(kind)
    model = replace(model, inference=replace(model.inference, device=inference_device))
    if hand is not None:
        model = replace(model, hand=replace(model.hand, kind=hand))
    config = replace(loaded.config, actions=ActionConfigV2(action_contract=engineering_contract()), model=model,
        run=RunConfigV2(split='train'), training=PPOTrainingConfig(seeds=(17,), device=learning_device, max_updates=max_updates),
        preprocessing=PreparationConfig(backend='python', reader='structured-v1', memory_budget_mib=2048,
                                       cache_budget_mib=512, chunk_target_mib=16))
    return LoadedConfig(config, asdict(config), digest(asdict(config)), ())


async def _parallel_ready_actors(workers, result):
    """Own both same-game initializers until every original call has finished."""
    from gf_simulator_client.structured_environment import StructuredEpisodeResult
    from gf_simulator_client.structured_worker import StructuredParticipantWorker

    assert isinstance(result, StructuredEpisodeResult)
    entries = tuple(workers.items())
    assert len(entries) <= 2
    assert all(isinstance(worker, StructuredParticipantWorker) for _, worker in entries)
    tasks = []
    creation_error = None
    try:
        for _, worker in entries:
            coroutine = worker.initialize()
            try:
                task = asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
            tasks.append(task)
    except BaseException as error:  # noqa: BLE001 - re-raised after owned tasks drain.
        creation_error = error
    joined = asyncio.gather(*tasks, return_exceptions=True)
    try:
        if creation_error is not None:
            raise creation_error
        outcomes = await asyncio.shield(joined)
    except BaseException:
        # A second caller cancellation must not interrupt a child's abort cleanup.
        for task in tasks:
            if not task.done():
                task.cancel()
        while not joined.done():
            try:
                await asyncio.shield(joined)
            except asyncio.CancelledError:
                pass
        joined.result()
        raise
    for (viewer, worker), outcome in zip(entries, outcomes, strict=True):
        if not isinstance(outcome, BaseException):
            assert worker.startup_elapsed_ms is not None
            result.model_startup_ms[viewer] = worker.startup_elapsed_ms
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


def component_http(monkeypatch, loaded, *, mode='natural', source_ops=None, validation_context=None, outcome='draw', actors=(2,1,2,1), pool_cpu=False, parallel_ready=False, preload_cpu=False):
    import gf_cli
    from gf_models.structured_baselines import make_structured_policy
    from gf_simulator_client import structured_environment
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    from gf_training.structured_journal import StructuredJournal
    from gf_training.structured_ppo import _NativeComponentOps
    from gf_training.structured_tree_validation import FinalValidationContext
    from tests.cpu_preload import (
        CURRENT_LAUNCH,
        CURRENT_TEMPLATE,
        CpuPreloadMixin,
        CpuTemplate,
        LaunchContext,
        eligible_configuration,
        supported,
    )
    template = CURRENT_TEMPLATE.get() if preload_cpu else None
    native_preload = (type(source_ops) is _NativeComponentOps and validation_context is None and not pool_cpu
        and loaded.config.preprocessing.backend == 'rust' and loaded.config.preprocessing.reader == 'structured-v1'
        and type(loaded.config.actions) is ActionConfigV2
        and loaded.config.actions.action_contract == engineering_contract() and loaded.config.run.split == 'train')
    if preload_cpu and (not supported() or not isinstance(template, CpuTemplate)
            or (source_ops is not None and not native_preload)
            or (validation_context is not None and (type(validation_context) is not FinalValidationContext
                or loaded is not validation_context.projected or loaded.config.run.split != 'validation' or pool_cpu))
            or loaded.config.training.device != 'cpu' or loaded.config.model.inference.device != 'cpu'
            or not (eligible_configuration(loaded) or native_preload)):
        raise ValueError('pytest CPU preload requires a supported bound owner and ordinary CPU engineering training, exact native training or final validation')
    if template is not None and template.failure is not None:
        raise template.failure
    if preload_cpu and (template.state not in ('UNSTARTED', 'RUNNING')
            or getattr(loaded.config.model, 'model_id', None) not in ('mlp', 'gru', 'transformer')):
        raise ValueError('pytest CPU preload requires a live supported owner and neural CPU model')
    if parallel_ready and (source_ops is not None or validation_context is not None
            or loaded.config.training.device != 'cpu' or loaded.config.model.inference.device != 'cpu'):
        raise ValueError('pytest parallel readiness requires ordinary CPU engineering training')
    # Only this test-owned subclass uses private bodies and fixed I/O doubles.
    # Public/direct Environment guards remain entirely unpatched.
    by_url, scripts, workers, seen_ledgers = {}, [], [], []
    serial = itertools.count()
    class Server:
        def __init__(self, binary, config, seed, count, directory, *args, **kwargs):
            self.expected = server_effective(config, count)
            self.script = PublicScript(mode=mode,outcome=outcome,actors=actors)
            self.script.info.update(self.expected, seed_fingerprint=seed_fingerprint(seed))
            self.url = f'http://component-{next(serial)}'
            by_url[self.url] = self.script
            scripts.append(self.script)
        async def start(self):
            return {'base_url': self.url, 'test_server_info': self.script.info}
        async def stop(self, timeout):
            return {'exit_code': 0, 'shutdown': 'stdin_eof', 'log_error': None}
    def client(url, **kwargs):
        return AttestedClient(url, transport=httpx.MockTransport(by_url[url].handle), **kwargs)
    if pool_cpu:
        from tests.pooled_structured_worker import (
            CURRENT_POOL,
            CURRENT_VIEWER,
            CpuParticipantPool,
            PooledStructuredParticipantWorker,
        )
        if (source_ops is not None or validation_context is not None or loaded.config.training.device != 'cpu'
                or loaded.config.model.inference.device != 'cpu'):
            raise ValueError('pytest CPU pool requires ordinary CPU engineering training')
    original_worker = PooledStructuredParticipantWorker if pool_cpu else StructuredParticipantWorker
    if preload_cpu and not pool_cpu:
        class PreloadedWorker(CpuPreloadMixin, StructuredParticipantWorker):
            pass
        original_worker = PreloadedWorker
    class Worker(original_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workers.append(self)
    class FakeEnvironment(structured_environment.StructuredSimulatorEnvironment):
        def __init__(self, settings, ledger):
            if validation_context is None:
                assert ledger.learner is not None and ledger.learner._cleanup_owner.process.returncode is None
            else:
                assert ledger is validation_context.ledger and ledger.learner is None and not ledger.updating
            seen_ledgers.append(ledger)
            self._initialize_structured(settings, make_structured_policy, StructuredJournal, ledger=ledger)
        def _create_actor(self, directory, spec, viewer, game, journals, actors, contexts, result):
            launch_token = CURRENT_LAUNCH.set(LaunchContext(template, self.settings.execution_deadline,
                min(self.settings.wall_deadline, self.settings.work_deadline), viewer)) if preload_cpu else None
            token = CURRENT_VIEWER.set(viewer) if pool_cpu else None
            try:
                return super()._create_actor(directory, spec, viewer, game, journals, actors, contexts, result)
            finally:
                if token is not None:
                    CURRENT_VIEWER.reset(token)
                if launch_token is not None:
                    CURRENT_LAUNCH.reset(launch_token)
        def _new_server(self, *args, **kwargs):
            return Server(*args, **kwargs)
        def _new_client(self, *args, **kwargs):
            return client(*args, **kwargs)
        async def _ready_actors(self, workers, result):
            if parallel_ready:
                return await _parallel_ready_actors(workers, result)
            return await super()._ready_actors(workers, result)
        async def run_group(self, group):
            return await self._run_group_body(group)
        async def _episode(self, client, definitions, group, spec):
            return await self._episode_body(client, definitions, group, spec)
    monkeypatch.setattr(structured_environment, 'StructuredParticipantWorker', Worker)
    async def collector(plan, ledger):
        before = time.monotonic()
        pool = CpuParticipantPool(ledger, cleanup_deadline=plan.wall_deadline - loaded.config.run.report_reserve_seconds) if pool_cpu else None
        token = CURRENT_POOL.set(pool) if pool_cpu else None
        cleanup_deadline = (plan.wall_deadline - loaded.config.run.report_reserve_seconds
                            if pool is not None or template is not None else float('inf'))
        finalized = None
        def validate_preloaded_collection(*, before_warmup=False):
            if template is not None and validation_context is not None:
                if (loaded is not validation_context.projected or plan is not validation_context.plan
                        or plan.kind != 'ppo_validation' or ledger is not validation_context.ledger):
                    raise ValueError('pytest CPU preload requires exact final validation LoadedConfig/plan/SAME ledger')
                validation_context.validate_collection(plan, ledger, loaded.config, deadline=cleanup_deadline)
            elif template is not None and source_ops is not None:
                source_ops.validate_config(loaded.config)
                source_ops.require_live_owner(ledger)
                source_ops.validate_owned_collection(plan, ledger, config=loaded.config,
                    deadline=min(plan.execution_deadline, cleanup_deadline) if before_warmup else cleanup_deadline,
                    loaded=loaded)
        async def finalizer():
            nonlocal finalized
            if finalized is None:
                candidate = await pool.close() if pool is not None else {'mode': 'pytest-fresh-forked-cpu-v1'}
                if pool is None:
                    fresh = [worker for worker in workers if worker.lease.ledger is ledger]
                    if ledger.workers or any(not worker.closed or not worker.lease.released
                            or worker.process is None or worker.process.poll() is None
                            or (worker.io_thread is not None and worker.io_thread.is_alive())
                            or any(stream is not None and not stream.closed
                                   for stream in (worker.process.stdin, worker.process.stdout))
                            for worker in fresh):
                        raise RuntimeError('pytest fresh-forked SAME ledger/process/stream/thread retirement unconfirmed')
                if template is not None:
                    candidate['cpu_preload'] = template.health(cleanup_deadline)
                if pool is None:
                    candidate['closed'] = True
                validate_preloaded_collection()
                finalized = candidate
            return finalized
        try:
            if template is not None:
                validate_preloaded_collection(before_warmup=True)
                template.start(plan.execution_deadline)
            directory, report = await gf_cli._run_experiment(loaded, 'collect', collection=plan, ledger=ledger,
                _structured_factory=FakeEnvironment, _structured_source_ops=source_ops,
                _structured_validation_context=validation_context,
                _pytest_worker_finalizer=finalizer if pool is not None or template is not None else None)
        finally:
            pending = sys.exception()
            try:
                try:
                    if pool is not None:
                        await pool.close()
                finally:
                    if template is not None and template.failure is not None:
                        await template.close(deadline=cleanup_deadline)
            except BaseException as cleanup_error:
                if pending is None:
                    raise
                pending.add_note(f'pytest pool finalization: {type(cleanup_error).__name__}: {cleanup_error}')
            finally:
                if token is not None:
                    CURRENT_POOL.reset(token)
        if pool is not None:
            receipt = json.loads((directory / 'manifest.json').read_text()).get('pytest_worker_lifecycle')
            # Minimal/failure paths may have no receipt; they cannot become a healthy Source.
            if report.get('comparison_valid'):
                assert receipt == await pool.close()
                assert receipt['closed'] and receipt['resets'] == receipt['logical_participants']
                assert receipt['physical_workers'] <= 4 and not ledger.workers
        assert time.monotonic() >= before and not ledger.workers
        assert (ledger.learner is not None) == (validation_context is None)
        return FinalizedCollection(directory, report)
    return collector, scripts, workers, seen_ledgers


def clone_component_parent(original, target):
    """Reseal a TEST-ONLY copy's paths and actual source-file fingerprints.

    The copied records remain engineering components. This is not a migration
    or qualification API, and no production source reader trusts this helper.
    """
    import json
    import shutil

    from gf_common.structured import action_config_v2
    from gf_models.arrays import sha256
    from gf_training.journal import atomic_json
    from gf_training.structured_source import inspect_component_collection
    shutil.copytree(original, target)
    state = json.loads((target / 'run.json').read_text())
    actions = action_config_v2(state['effective_config']['actions'])
    settings = PreparationConfig(**state['effective_config']['preprocessing'])
    def paths(value):
        if isinstance(value, str) and value.startswith(str(original) + '/'):
            return str(target) + value[len(str(original)):]
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        return value
    parent = None
    for number in range(state['updates'] + 1):
        generation = target / 'generations' / f'{number:06}.json'
        record = json.loads(generation.read_text())
        record['parent_sha256'] = parent
        if number:
            update_path = target / 'updates' / f'{number:06}.json'
            update = paths(json.loads(update_path.read_text()))
            collection = target / update['collection']
            manifest = paths(json.loads((collection / 'manifest.json').read_text()))
            manifest['ppo']['parent_run_path'] = str(target)
            atomic_json(collection / 'manifest.json', manifest)
            source = inspect_component_collection(collection, actions=actions, settings=settings, expected_split='train')
            update.update(parent_sha256=parent, collection_manifest_sha256=sha256(collection / 'manifest.json'), source_proof=source.proof)
            atomic_json(update_path, update)
            record['update_sha256'] = sha256(update_path)
        atomic_json(generation, record)
        parent = sha256(generation)
    state['generation_sha256'] = parent
    atomic_json(target / 'run.json', state)
    atomic_json(target / 'latest.json', {'generation': state['updates'], 'sha256': parent})
    return target
