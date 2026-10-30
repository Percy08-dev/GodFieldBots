"""Bounded async native fixed-data writing and toy-fit readiness ownership."""
import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ActionConfig, ContractError
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig, ModelConfig, TrainingConfig
from gf_models.encoder import feature_names
from gf_training.native_preparation import NativePreparedInputs, _deadlines, _owner
from tests.test_fixed_ppo_data import frozen_sources as frozen_sources  # noqa: PLC0414


def readonly(value, dtype):
    result = np.asarray(value, dtype=dtype)
    result.flags.writeable = False
    return result


def fake_preparation(monkeypatch, count=1000):
    config = EncoderConfig(hash_buckets=2)
    width, decisions = len(feature_names(config)), count * 2
    batch = SimpleNamespace(
        features=readonly(np.ones((decisions, width)), np.float64),
        state_features=readonly(np.zeros((decisions, width)), np.float64),
        selected=readonly(np.zeros(decisions), np.int64),
        candidate_counts=readonly(np.ones(decisions), np.int64),
        episode_offsets=readonly([0, 0, count, decisions], np.int64),
        candidate_offsets=readonly(np.arange(decisions + 1), np.int64),
        old_logprob=readonly(np.zeros(decisions), np.float64),
        old_value=readonly(np.zeros(decisions), np.float64),
        old_logprob_present=readonly(np.zeros(decisions), np.int64),
        old_value_present=readonly(np.zeros(decisions), np.int64),
        winner=readonly([0, 1, 2], np.int64), learner_viewer=readonly([1, 1, 2], np.int64),
        split=readonly([0, 0, 1], np.int64),
        terminal_rewards=readonly([[0, 0], [1, -1], [-1, 1]], np.float64),
        feature_layout='selected',
    )
    owners = []
    class Lease:
        released = False
        def release(self):
            self.released = True
    class Owner:
        def __init__(self):
            self.diagnostics = {}
            self.consumed = self.closed = False
            self.lease = Lease()
            owners.append(self)
        async def anext_batch(self, *_):
            if self.consumed:
                return None
            self.consumed = True
            return batch
        def proof(self):
            return {'structural_valid': True, 'sample_identity_sha256': '0' * 64,
                    'dataset_manifest': {'train_decisions': count, 'validation_decisions': count}}
        async def areserve_staging(self, *_):
            return self.lease
        def check(self):
            if self.closed:
                raise ContractError('closed fake owner')
        async def afinalize(self):
            self.closed = True
        async def aabort(self):
            self.closed = True
    async def create(plan, settings, *, deadline, wall_deadline):
        owner = Owner()
        owner.config, owner.deadline, owner.wall_deadline = settings, deadline, wall_deadline
        return owner
    monkeypatch.setattr('gf_training.native_preparation.NativePreparation.create_async', create)
    return config, owners


def test_sync_explicit_fit_deadline_preserves_original_wall(monkeypatch):
    settings = PreparationConfig(backend='rust', wall_seconds=10, cleanup_seconds=2)
    monkeypatch.setattr('gf_training.native_preparation.time.monotonic', lambda: 100.)
    monkeypatch.setattr('gf_training.native_preparation.NativePreparation',
                        lambda plan, config, **values: SimpleNamespace(config=config, **values))
    owner = _owner({}, settings, 7295., 7300.)
    assert owner.wall_deadline == 7300.
    assert owner.deadline == 7295.
    assert _deadlines(settings, float('inf'), 7300.) == (7298., 7300.)
    assert _deadlines(settings, float('inf'), None) == (108., 110.)


def test_async_explicit_fit_deadline_preserves_original_wall_and_cleanup(tmp_path, monkeypatch):
    config, owners = fake_preparation(monkeypatch, count=1)
    settings = PreparationConfig(backend='rust', wall_seconds=10, cleanup_seconds=2,
                                 memory_budget_mib=128, cache_budget_mib=16, chunk_target_mib=1)
    now = [100.]
    monkeypatch.setattr('gf_training.native_preparation.time.monotonic', lambda: now[0])

    async def run():
        prepared = await NativePreparedInputs.create_async(tmp_path / 'unused.json', config, settings,
                                                           deadline=7295., wall_deadline=7300.)
        assert owners[-1].deadline == 7295.
        assert owners[-1].wall_deadline == 7300.
        now[0] = 7299.  # Beyond the preprocessing window, inside original fit cleanup.
        await prepared.afinalize()
        assert owners[-1].closed

    asyncio.run(run())


def test_async_large_episode_writes_are_bounded_and_npz_bytes_stay_identical(tmp_path, monkeypatch):
    config, owners = fake_preparation(monkeypatch)
    writes = []
    class Guard:
        def __init__(self, wrapped, window):
            self.wrapped, self.window = wrapped, window
        def write(self, value):
            assert len(value) <= self.window
            writes.append(len(value))
            return self.wrapped.write(value)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
    class Guarded(NativePreparedInputs):
        def _open_writers(self):
            super()._open_writers()
            self._outputs = {name: Guard(output, self._window) for name, output in self._outputs.items()}
    settings = PreparationConfig(backend='rust', chunk_target_mib=1, memory_budget_mib=128, cache_budget_mib=16)
    async def run():
        ticks = []
        async def heartbeat():
            while not owners or not owners[0].closed:
                ticks.append(True)
                await asyncio.sleep(0)
        ticking = asyncio.create_task(heartbeat())
        async with await Guarded.create_async(tmp_path / 'unused', config, settings,
                                               deadline=time.monotonic() + 10) as prepared:
            wall = prepared.diagnostics['native_prepared_inputs_wall']
            assert wall['success'] is True
            assert wall['parent_nanoseconds'] == sum(wall['partition_nanoseconds'].values())
            assert wall['residual_nanoseconds'] == 0
            construction_parent = wall['parent_nanoseconds']
            prepared.write_numeric(tmp_path / 'sync.npz')
            await prepared.awrite_numeric(tmp_path / 'async.npz')
            assert len(ticks) >= 4
        await ticking
        assert prepared._directory is None and prepared._closed
        assert prepared.diagnostics['native_prepared_inputs_wall']['parent_nanoseconds'] == construction_parent
        assert owners[0].closed and owners[0].lease.released
    asyncio.run(run())
    assert len(writes) > 4
    assert max(writes) == 1024 * 1024
    assert hashlib.sha256((tmp_path / 'sync.npz').read_bytes()).digest() == hashlib.sha256((tmp_path / 'async.npz').read_bytes()).digest()
    with np.load(tmp_path / 'async.npz', allow_pickle=False) as arrays:
        assert arrays['x'].shape == (1000, len(feature_names(config)))
        np.testing.assert_array_equal(arrays['y'], np.ones(1000))
        np.testing.assert_array_equal(arrays['validation_y'], np.ones(1000))


def test_cancel_during_bounded_episode_copy_closes_owner_and_staging(tmp_path, monkeypatch):
    config, owners = fake_preparation(monkeypatch)
    temporary = []
    class Watched(NativePreparedInputs):
        def _open_writers(self):
            super()._open_writers()
            temporary.append(self._directory)
    settings = PreparationConfig(backend='rust', chunk_target_mib=1, memory_budget_mib=128, cache_budget_mib=16)
    async def run():
        task = asyncio.create_task(Watched.create_async(tmp_path / 'unused', config, settings,
                                                        deadline=time.monotonic() + 10))
        await asyncio.sleep(0)
        task.cancel('copy cancellation')
        with pytest.raises(asyncio.CancelledError, match='copy cancellation'):
            await task
    asyncio.run(run())
    assert owners[0].closed and owners[0].lease.released
    assert temporary and all(not path.exists() for path in temporary)



def test_npz_copy_cancellation_precedes_child_and_ready(tmp_path, monkeypatch):
    from gf_training import lightgbm_run as runner
    config, owners = fake_preparation(monkeypatch)
    settings = PreparationConfig(backend='rust', chunk_target_mib=1, memory_budget_mib=128, cache_budget_mib=16)
    model = ModelConfig(model_id='lightgbm', encoder=config)
    training = TrainingConfig(num_boost_round=2, early_stopping_rounds=1, min_data_in_leaf=1,
                              wall_seconds=20, cleanup_seconds=1, cost_mode='fixed_data')
    async def forbidden(*args, **kwargs):
        pytest.fail('cancelled numeric copy must not launch a learner')
    monkeypatch.setattr(runner.asyncio, 'create_subprocess_exec', forbidden)
    directory = tmp_path / 'cancelled-fit'
    async def run():
        prepared = await NativePreparedInputs.create_async(tmp_path / 'unused', config, settings,
                                                           deadline=time.monotonic() + 20)
        task = asyncio.create_task(runner._run_numeric(directory, None, None, None, None,
                                                        model=model, actions=ActionConfig(), training=training,
                                                        seed=17, purpose='toy', prepared_inputs=prepared))
        await asyncio.sleep(0)
        task.cancel('NPZ cancellation')
        with pytest.raises(asyncio.CancelledError, match='NPZ cancellation'):
            await task
        assert prepared._closed and prepared._directory is None
    asyncio.run(run())
    assert owners[0].closed and owners[0].lease.released
    assert json.loads((directory / 'training.json').read_text())['status'] == 'interrupted'
    assert not (directory / 'checkpoint').exists()

def test_real_native_toy_fit_source_and_cleanup_gates(tmp_path, frozen_sources, monkeypatch):
    pytest.importorskip('gf_native')
    pytest.importorskip('lightgbm')
    from gf_training import lightgbm_run as runner
    from gf_training.fixed_data import build_dataset
    from gf_training.lineage import _OWNED_LOCKS
    parents, validations = frozen_sources
    dataset = tmp_path / 'dataset.json'
    build_dataset(parents, validations, dataset)
    model = ModelConfig(model_id='lightgbm', encoder=EncoderConfig(hash_buckets=2))
    training = TrainingConfig(num_boost_round=2, early_stopping_rounds=1, min_data_in_leaf=1,
                              wall_seconds=20, cleanup_seconds=1, cost_mode='fixed_data')
    settings = PreparationConfig(backend='rust', memory_budget_mib=128, cache_budget_mib=16,
                                 chunk_target_mib=1, wall_seconds=20, cleanup_seconds=5)
    source = next(parents[0].glob('collections/*/*/player-1.jsonl'))
    original_source = source.read_bytes()
    stop = runner._stop
    async def run(mode):
        stopped = []
        directory = tmp_path / f'toy-{mode}'
        wall = time.monotonic() + 20
        prepared = await NativePreparedInputs.create_async(dataset, model.encoder, settings,
                                                           deadline=wall - 5, wall_deadline=wall)
        assert prepared.session.wall_deadline == wall
        assert prepared.session.deadline <= wall - 5
        async def stopped_first(*args, **kwargs):
            await stop(*args, **kwargs)
            stopped.append(True)
            assert args[0].returncode == 0
            if mode == 'source':
                source.write_bytes(original_source + b' ')
            elif mode == 'deadline':
                prepared.session.wall_deadline = time.monotonic() - 1
        monkeypatch.setattr(runner, '_stop', stopped_first)
        finalize = prepared.afinalize
        async def checked_finalize():
            assert stopped == [True]
            assert not (directory / 'checkpoint').exists()
            assert json.loads((directory / 'staging' / 'inference.json').read_text())['ready'] is False
            if mode == 'cancel':
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel, 'cleanup cancellation')
            await finalize()
        monkeypatch.setattr(prepared, 'afinalize', checked_finalize)
        try:
            async with prepared:
                operation = runner._run_numeric(directory, None, None, None, None, model=model,
                                                 actions=ActionConfig(), training=training, seed=17,
                                                 purpose='toy', prepared_inputs=prepared)
                if mode == 'normal':
                    result = await operation
                    assert result['status'] == 'complete'
                    assert json.loads((directory / 'checkpoint' / 'inference.json').read_text())['ready'] is True
                else:
                    with pytest.raises((ContractError, TimeoutError, asyncio.CancelledError)):
                        await operation
                    manifest = json.loads((directory / 'training.json').read_text())
                    assert manifest['status'] in ('failed', 'interrupted')
                    for path in directory.glob('*/inference.json'):
                        assert json.loads(path.read_text())['ready'] is False
        finally:
            source.write_bytes(original_source)
        assert prepared._closed and prepared.session._closed
        assert stopped == [True]
        assert not any(parent.resolve() in _OWNED_LOCKS for parent in parents)
        assert not [thread for thread in threading.enumerate() if thread.name.startswith('gf-native-ffi')]
    for mode in ('normal', 'source', 'cancel', 'deadline'):
        asyncio.run(run(mode))


def test_final_source_checks_use_original_cleanup_wall_not_work_clock(monkeypatch):
    from gf_training.native_authorization import NativePreparation
    from tests.test_native_execution import fake_native
    fake = fake_native(monkeypatch, None)
    async def run():
        wall = time.monotonic() + 20
        owner = await NativePreparation.create_async({'kind': 'offline'}, PreparationConfig(backend='rust'),
                                                     deadline=wall - 5, wall_deadline=wall)
        owner.deadline = time.monotonic() - 1
        await owner.afinalize()  # Work ended; the original cleanup reserve remains.
        assert owner._closed and fake.sessions[-1].closed
        owner = await NativePreparation.create_async({'kind': 'offline'}, PreparationConfig(backend='rust'),
                                                     deadline=wall - 5, wall_deadline=wall)
        owner.wall_deadline = time.monotonic() - 1
        with pytest.raises(TimeoutError):
            await owner.afinalize()
        assert owner._closed and fake.sessions[-1].closed
        assert not [thread for thread in threading.enumerate() if thread.name.startswith('gf-native-ffi')]
    asyncio.run(run())
