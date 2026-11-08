"""Real wire equivalence, reset boundaries and physical ownership for pytest pool."""
from __future__ import annotations

import asyncio
import copy
import io
import json
import subprocess
import sys
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, canonical
from gf_common.structured import EventBank
from gf_common.structured_wire import delta_dict, frame_dict
from gf_models.structured_baselines import make_structured_policy
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from tests.pooled_structured_bootstrap import PolicyCache, envelope
from tests.pooled_structured_worker import (
    CURRENT_POOL,
    CURRENT_VIEWER,
    CpuParticipantPool,
    PooledStructuredParticipantWorker,
)
from tests.structured_worker_fixtures import bad_hash, slow
from tests.test_structured_protocol import action_settings, neural_component, own_frame


def logical(pool, *, viewer=1, seed=17, kind='random', options=None, factory=make_structured_policy):
    pool_token, viewer_token = CURRENT_POOL.set(pool), CURRENT_VIEWER.set(viewer)
    try:
        return PooledStructuredParticipantWorker(factory, kind, seed, action_settings(), ledger=pool.ledger, policy_options=options)
    finally:
        CURRENT_VIEWER.reset(viewer_token)
        CURRENT_POOL.reset(pool_token)


def semantic_wire(value):
    value = copy.deepcopy(value)
    value.pop('model_load_ms', None)
    value.pop('timings', None)
    if 'generated' in value:
        for key in ('count_ms', 'search_ms', 'selection_ms'):
            value['generated']['report'].pop(key)
    return canonical(value)


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_full_original_wire_exact_fresh_pool_seeds_A_B_A_checkpoint_A(make_input, tmp_path, kind):
    pytest.importorskip('torch')
    learner, options = neural_component(tmp_path, kind)
    # A distinct artifact replaces the single cache; returning to A reloads A.
    with learner.torch.no_grad():
        next(learner.network.parameters()).add_(.125)
    identity = learner.save(tmp_path / 'other-checkpoint')
    other = {**options, 'checkpoint': str(tmp_path / 'other-checkpoint'),
             'expected_manifest_digest': identity['inference_digest']}
    assert other['expected_manifest_digest'] != options['expected_manifest_digest']
    async def run():
        pool, fresh_ledger = CpuParticipantPool(ReservationLedger()), ReservationLedger()
        answers = []
        try:
            for episode, (seed, binding) in enumerate(((17, options), (29, options), (17, options), (31, other), (17, options))):
                pooled = logical(pool, seed=seed, kind=kind, options=binding)
                fresh = StructuredParticipantWorker(make_structured_policy, kind, seed, action_settings(), ledger=fresh_ledger, policy_options=binding)
                fresh_wire, pooled_wire = [], []
                original_fresh, original_pooled = fresh._roundtrip, pooled._roundtrip
                async def record_fresh(payload, original=original_fresh, wire=fresh_wire):
                    reply = await original(payload)
                    wire.append((semantic_wire(payload), semantic_wire(reply)))
                    return reply
                async def record_pool(payload, original=original_pooled, wire=pooled_wire):
                    reply = await original(payload)
                    wire.append((semantic_wire(payload), semantic_wire(reply)))
                    return reply
                fresh._roundtrip, pooled._roundtrip = record_fresh, record_pool
                bank = EventBank(1, 'same-own-episode')
                selections = []
                try:
                    for revision in (0, 2, 5):
                        frame, delta = own_frame(make_input(hand=('bronze-club', 'steel-armor'), learned=('fireball', 'aura')),
                            bank, revision, [{'type': 'action_accepted', 'player': 1}])
                        a, b = await fresh.select_frame(frame, delta), await pooled.select_frame(frame, delta)
                        assert a[1] == b[1]  # exact logprob/value/index, no relaxed tolerance.
                        assert a[0].table == b[0].table
                        selections.append(asdict(b[1]))
                    assert fresh_wire == pooled_wire
                    answers.append(selections)
                finally:
                    await fresh.close()
                    await pooled.close()
                assert not fresh_ledger.workers
                assert not pooled.lease.released and pooled.process.poll() is None
                assert pooled.lease._cleanup_owner is pooled.owner and not pooled.lease.decisions
                assert pooled.owner.loan is None and not pooled.owner.io_thread.is_alive()
                assert len(pool.ledger.workers) == 1 and pool.ledger.serial == 1
            assert answers[0] == answers[2] == answers[4]
        finally:
            receipt = await pool.close()
        assert receipt['physical_workers'] == 1 and receipt['logical_participants'] == receipt['resets'] == 5
        assert receipt['model_loads'] == 3 and receipt['model_reuses'] == 2
        assert all(owner.process.poll() is not None and owner.lease.released for owner in pool.owners)
    asyncio.run(run())
    assert learner.model.model_id == kind


def test_fixed_viewer_partition_capacity_real_leases_and_reset(make_input):
    async def run():
        ledger, workers = ReservationLedger(), []
        pool = CpuParticipantPool(ledger)
        try:
            for viewer in (2, 1, 2, 1):
                worker = logical(pool, viewer=viewer)
                workers.append(worker)
                await worker.initialize()
            assert len(ledger.workers) == 4 and ledger.serial == 4
            assert [worker.owner.viewer for worker in workers] == [2, 1, 2, 1]
            assert all(worker.lease._cleanup_owner is worker.owner for worker in workers)
            with pytest.raises(ContractError, match='slot'):
                logical(pool, viewer=1)
            with pytest.raises(ContractError, match='slot'):
                logical(pool, viewer=2)
            for worker in workers:
                await worker.close()
            assert len(ledger.workers) == 4 and all(owner.process.poll() is None for owner in pool.owners)
            reused = logical(pool, viewer=1)
            assert reused.owner is workers[1].owner and reused._parent_bank is reused._last_frame is None
            await reused.initialize()
            await reused.close()
        finally:
            receipt = await pool.close()
        assert receipt['physical_workers'] == 4 and receipt['logical_participants'] == receipt['resets'] == 5
        assert receipt['model_loads'] == 4 and receipt['model_reuses'] == 1
        assert not ledger.workers and all(owner.lease.released for owner in pool.owners)
    asyncio.run(run())


def test_no_viewer_context_rejects_before_reservation():
    pool = CpuParticipantPool(ReservationLedger())
    token = CURRENT_POOL.set(pool)
    try:
        with pytest.raises(ContractError):
            PooledStructuredParticipantWorker(make_structured_policy, 'random', 0, action_settings(), ledger=pool.ledger)
    finally:
        CURRENT_POOL.reset(token)
    assert not pool.ledger.workers and not pool.owners


@pytest.mark.parametrize('fault', ['opposite_viewer', 'epoch', 'bad_selection', 'timeout', 'cancel'])
def test_failed_request_poisons_reaps_and_never_reloans(make_input, fault):
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool, factory=bad_hash if fault == 'bad_selection' else slow if fault in ('timeout', 'cancel') else make_structured_policy)
        try:
            await worker.initialize()
            frame, delta = own_frame(make_input(), EventBank(1, 'own'), 0)
            if fault == 'epoch':
                original = worker.owner._read_reply
                def stale(source):
                    result = original(source)
                    result['epoch'] -= 1
                    return result
                worker.owner._read_reply = stale
            if fault == 'opposite_viewer':
                payload = {'protocol_version': 2, 'kind': 'select', 'frame': frame_dict(frame), 'delta': delta_dict(delta)}
                payload['frame']['actor'] = payload['delta']['actor'] = 2
                operation = worker._roundtrip(payload)
            else:
                worker.timeout = .05 if fault == 'timeout' else 10
                operation = worker.select_frame(frame, delta)
            if fault == 'cancel':
                task = asyncio.create_task(operation)
                await asyncio.sleep(.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises((ContractError, RuntimeError, TimeoutError, EOFError)):
                    await operation
            assert worker.owner.poisoned and worker.process.poll() is not None
            assert worker.lease.released and not pool.ledger.workers
            assert worker.owner.io_thread is None or not worker.owner.io_thread.is_alive()
        finally:
            await worker.close()
            with pytest.raises(RuntimeError, match='poisoned'):
                await pool.close()
        task = pool._close_task
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        assert pool._close_task is task
    asyncio.run(run())


@pytest.mark.parametrize('ending', ['eof', 'truncated', 'epoch', 'viewer', 'select_error'])
def test_child_never_acknowledges_unclean_original_main_return(ending):
    async def initial_packet():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        initial = worker.initial
        await worker.close()  # No Begin: uncertain loan is physically retired.
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        return initial
    initial = asyncio.run(initial_packet())
    packets = io.BytesIO()
    write_frame(packets, envelope('begin', 1, 1, payload=initial))
    if ending == 'truncated':
        packets.write(b'\0\0\0')
    elif ending in ('epoch', 'viewer'):
        write_frame(packets, envelope('end', 2 if ending == 'viewer' else 1, 2 if ending == 'epoch' else 1))
    elif ending == 'select_error':
        write_frame(packets, envelope('select', 1, 1, payload={'frame': {'actor': 1}, 'delta': {'actor': 1}}))
        write_frame(packets, envelope('end', 1, 1))
    result = subprocess.run([sys.executable, '-E', '-m', 'tests.pooled_structured_bootstrap', '1'],
        input=packets.getvalue(), capture_output=True, check=False, timeout=10)
    replies = io.BytesIO(result.stdout)
    messages = []
    while replies.tell() < len(result.stdout):
        messages.append(read_frame(replies))
    assert messages and messages[0]['payload']['kind'] == 'ready'
    assert all(message['operation'] != 'reset' for message in messages)
    assert result.returncode != 0


def test_reset_verifier_rejects_broken_baseline_rng():
    from gf_common.action_contract import engineering_contract
    from gf_models.structured_baselines import StructuredBaseline
    cache = PolicyCache()
    cache.policy = StructuredBaseline('random', action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    cache.policy.rng.seed(31)
    with pytest.raises(ContractError, match='RNG'):
        cache.verify(0)
    cache.reset(0)


def test_reused_checkpoint_revalidates_original_weight_hash(tmp_path):
    pytest.importorskip('torch')
    _, options = neural_component(tmp_path, 'mlp')
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool, kind='mlp', options=options)
        await worker.initialize()
        await worker.close()
        manifest = json.loads((tmp_path / 'component-checkpoint/inference.json').read_text())
        # Corrupt the actual file named by ordinary artifact metadata.
        weights = tmp_path / 'component-checkpoint' / manifest['weights']['shards'][0]['file']
        weights.write_bytes(weights.read_bytes() + b'corrupt')
        next_worker = logical(pool, kind='mlp', options=options)
        with pytest.raises(ContractError, match='worker error'):
            await next_worker.initialize()
        assert next_worker.owner is worker.owner and next_worker.owner.poisoned
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        assert not pool.ledger.workers
    asyncio.run(run())


def test_idle_unexpected_exit_is_not_healthy_collection_close():
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        await worker.initialize()
        await worker.close()
        worker.process.kill()
        while worker.process.poll() is None:
            await asyncio.sleep(.01)
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        assert worker.owner.poisoned and worker.lease.released and not pool.ledger.workers
    asyncio.run(run())


def test_end_reset_ack_epoch_failure_reaps_and_cannot_publish():
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        await worker.initialize()
        original = worker.owner._read_reply
        def stale(source):
            result = original(source)
            result['epoch'] += 1
            return result
        worker.owner._read_reply = stale
        with pytest.raises(ContractError, match='epoch'):
            await worker.close()
        first = worker._close_task
        with pytest.raises(ContractError, match='epoch'):
            await worker.close()
        assert worker._close_task is first and worker.owner.poisoned
        assert worker.lease.released and worker.process.poll() is not None
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
    asyncio.run(run())


def test_repeated_pool_close_cancellation_joins_same_first_deadline(monkeypatch):
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        await worker.initialize()
        await worker.close()
        original = pool._close_owned
        entered, release, starts = asyncio.Event(), asyncio.Event(), []
        async def delayed(started):
            starts.append(started)
            entered.set()
            await release.wait()
            return await original(started)
        monkeypatch.setattr(pool, '_close_owned', delayed)
        waiter = asyncio.create_task(pool.close())
        await entered.wait()
        shared = pool._close_task
        waiter.cancel()
        await asyncio.sleep(0)
        waiter.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        receipt = await pool.close()
        assert pool._close_task is shared and len(starts) == 1
        assert receipt['closed'] and not pool.ledger.workers and worker.lease.released
    asyncio.run(run())


@pytest.mark.parametrize('fault', ['new_factory', 'stale_cache'])
def test_constructor_and_begin_errors_keep_physical_cleanup_owner(tmp_path, fault):
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        assert worker.lease._cleanup_owner is worker.owner
        if fault == 'new_factory':
            worker.initial['factory']['name'] = 'missing_factory'
        else:
            worker.initial['caps']['max_episode_events'] += 1
        with pytest.raises((EOFError, ContractError)):
            await worker.initialize()
        assert worker.owner.closed and worker.owner.poisoned and worker.lease.released
        assert worker.process.poll() is not None
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
    asyncio.run(run())


@pytest.mark.parametrize('case', ['normal', 'close_failure', 'minimal_report', 'false_receipt', 'live_lease_receipt'])
def test_owned_collector_closes_before_report_update_and_publication(tmp_path, monkeypatch, case):
    pytest.importorskip('torch')
    import gf_cli
    from gf_training.structured_ppo import _train_component_ppo
    from tests.structured_ppo_support import component_http, configuration
    loaded = configuration(tmp_path)
    pools, reports, updates = [], [], []
    original_close = CpuParticipantPool._close_owned
    async def finish(pool, started):
        pools.append(pool)
        receipt = await original_close(pool, started)
        assert all(owner.process.poll() is not None and owner.lease.released for owner in pool.owners)
        if case == 'close_failure':
            raise RuntimeError('injected pytest finalizer failure')
        return receipt
    monkeypatch.setattr(CpuParticipantPool, '_close_owned', finish)
    original_report = gf_cli.report
    def report(*args, **kwargs):
        assert case not in ('close_failure', 'false_receipt', 'live_lease_receipt')
        assert pools and all(pool._close_task.done() and not pool.ledger.workers for pool in pools)
        reports.append(1)
        if case == 'minimal_report':
            raise TimeoutError('injected postcollection report deadline')
        return original_report(*args, **kwargs)
    monkeypatch.setattr(gf_cli, 'report', report)
    if case in ('false_receipt', 'live_lease_receipt'):
        original_run = gf_cli._run_experiment
        async def invalid_receipt(*args, **kwargs):
            finalizer = kwargs['_pytest_worker_finalizer']
            async def invalid():
                if case == 'false_receipt':
                    receipt = await finalizer()
                    return {**receipt, 'closed': False}
                assert kwargs['ledger'].workers
                return {'closed': True}
            kwargs['_pytest_worker_finalizer'] = invalid
            return await original_run(*args, **kwargs)
        monkeypatch.setattr(gf_cli, '_run_experiment', invalid_receipt)
    original_update = ReservationLedger.begin_update
    def update(ledger, temporary_bytes):
        assert pools and all(pool._close_task.done() and not pool.ledger.workers for pool in pools)
        updates.append(1)
        return original_update(ledger, temporary_bytes)
    monkeypatch.setattr(ReservationLedger, 'begin_update', update)
    collector, scripts, workers, ledgers = component_http(monkeypatch, loaded, pool_cpu=True)
    parent = tmp_path / 'parent'
    if case == 'normal':
        result = asyncio.run(_train_component_ppo(loaded, collector, parent))
        assert result['status'] == 'complete' and result['updates'] == 2 and result['accepted_decisions'] == 64
        assert len(workers) == 64 and len(scripts) == 16 and len(pools) == len(reports) == len(updates) == 2
        assert len({id(worker.owner) for worker in workers}) == 8  # 64 logical participants, 8 real processes over 2 pools.
    else:
        with pytest.raises((RuntimeError, ContractError), match='finalizer|comparison|collection|eligible|source'):
            asyncio.run(_train_component_ppo(loaded, collector, parent))
        assert not updates and len(pools) == 1
        state = json.loads((parent / 'run.json').read_text())
        assert state['status'] == 'failed' and state['updates'] == 0
    for collection in (parent / 'collections').iterdir():
        manifest = json.loads((collection / 'manifest.json').read_text())
        assert len(manifest['episodes']) == 16
        assert all(episode['end_kind'] == 'natural' for episode in manifest['episodes'])
        if case in ('close_failure', 'false_receipt', 'live_lease_receipt'):
            assert manifest['status'] == 'interrupted' and not (collection / 'report.json').exists()
            assert 'pytest_worker_lifecycle' not in manifest and not reports
        else:
            receipt = manifest['pytest_worker_lifecycle']
            assert receipt['closed'] and receipt['physical_workers'] == 4
            assert receipt['logical_participants'] == receipt['resets'] == 32
            if case == 'minimal_report':
                assert manifest['status'] == 'incomplete'
                assert json.loads((collection / 'report.json').read_text())['report_status'] == 'minimal'
    assert all(worker.process.poll() is not None and worker.lease.released for worker in workers)
    assert all(not ledger.workers and ledger.learner is None for ledger in ledgers)
    assert all(pool._close_task.done() for pool in pools)


@pytest.mark.parametrize('private', [False, True])
def test_finalizer_rejects_unowned_or_public_collection_before_callback(tmp_path, private):
    import gf_cli
    from tests.structured_ppo_support import configuration
    loaded = configuration(tmp_path)
    calls = []
    async def finish():
        calls.append(1)
        return {}
    with pytest.raises(ContractError, match='pytest finalizer'):
        asyncio.run(gf_cli._run_experiment(loaded, 'collect', _pytest_worker_finalizer=finish,
            _structured_factory=object() if private else None))
    assert not calls


def test_neural_reset_verifier_checks_actual_cache_rng_and_timing(tmp_path):
    pytest.importorskip('torch')
    from pathlib import Path

    from gf_common.action_contract import engineering_contract
    from gf_common.domain import digest
    from gf_models.structured_policy import StructuredPolicy
    learner, options = neural_component(tmp_path, 'gru')
    cache = PolicyCache()
    cache.policy = StructuredPolicy.from_artifact(Path(options['checkpoint']), options['expected_manifest_digest'],
        learner.model, learner.binding, digest(asdict(action_settings())),
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    cache.reset(0)
    for field, value in (('position', 1), ('owner', (1, 'old-own-episode')), ('revision', 0), ('event_end', 1)):
        setattr(cache.policy.cache, field, value)
        with pytest.raises(ContractError, match='reset'):
            cache.verify(0)
        cache.reset(0)
    for field in ('keys', 'values', 'past'):
        setattr(cache.policy.cache, field, learner.torch.ones(2))
        with pytest.raises(ContractError, match='storage'):
            cache.verify(0)
        cache.reset(0)
        assert not bool(getattr(cache.policy.cache, field).count_nonzero())
    cache.policy.last_timing = {'policy_ms': 1.}
    with pytest.raises(ContractError, match='reset'):
        cache.verify(0)
    cache.reset(0)
    cache.policy.rng.manual_seed(31)
    with pytest.raises(ContractError, match='reset'):
        cache.verify(0)
    cache.reset(0)


@pytest.mark.parametrize('forbidden', ['native', 'validation', 'validation_kind', 'split', 'command'])
def test_finalizer_rejects_outside_private_engineering_train(tmp_path, forbidden):
    import time
    from dataclasses import replace

    import gf_cli
    from gf_training.collection import CollectionPlan
    from tests.structured_ppo_support import configuration
    loaded = configuration(tmp_path)
    config = loaded.config
    if forbidden == 'split':
        config = replace(config, run=replace(config.run, split='validation'))
    loaded = replace(loaded, config=config)
    plan = CollectionPlan((), {}, tmp_path / 'not-created', {}, time.monotonic() + 30, time.monotonic() + 60,
                          kind='ppo_validation' if forbidden == 'validation_kind' else 'ppo_batch',
                          started_monotonic=time.monotonic())
    kwargs = {'_structured_source_ops': object()} if forbidden == 'native' else {'_structured_validation_context': object()} if forbidden == 'validation' else {}
    calls = []
    async def finish():
        calls.append(1)
        return {}
    with pytest.raises(ContractError, match='pytest finalizer'):
        asyncio.run(gf_cli._run_experiment(loaded, 'evaluate' if forbidden == 'command' else 'collect',
            collection=plan, ledger=ReservationLedger(), _structured_factory=object(), _pytest_worker_finalizer=finish, **kwargs))
    assert not calls


@pytest.mark.parametrize('failure', ['deadline', 'health'])
def test_cleanup_deadline_and_owner_health_prevent_healthy_receipt(failure):
    import time
    async def run():
        deadline = time.monotonic() - 1 if failure == 'deadline' else time.monotonic() + 60
        pool = CpuParticipantPool(ReservationLedger(), cleanup_deadline=deadline)
        worker = logical(pool)
        await worker.initialize()
        if failure == 'deadline':
            with pytest.raises(TimeoutError, match='logical close deadline expired before reset'):
                await worker.close()
        else:
            await worker.close()
            worker.owner.health['cleanup_failure'] = 'injected physical close debt'
        with pytest.raises(RuntimeError, match='expired|unhealthy'):
            await pool.close()
        first = pool._close_task
        with pytest.raises(RuntimeError, match='expired|unhealthy'):
            await pool.close()
        assert pool._close_task is first and pool.cleanup_deadline == deadline
        assert worker.process.poll() is not None and worker.lease.released and not pool.ledger.workers
        if failure == 'deadline':
            assert worker.owner.health.get('emergency_grace')
    asyncio.run(run())


def test_queue_reply_does_not_allow_handoff_before_actual_thread_retirement(monkeypatch):
    import threading

    from gf_simulator_client import structured_worker
    original_thread = threading.Thread
    class DelayedThread(original_thread):
        def __init__(self, *args, **kwargs):
            self.release = threading.Event()
            self.joins = 0
            super().__init__(*args, **kwargs)
        def run(self):
            super().run()  # The actual exchange has published its queue result.
            self.release.wait(timeout=5)
        def join(self, timeout=None):
            self.joins += 1
            if self.joins > 1:
                self.release.set()  # Physical cleanup can now confirm real exit.
            return super().join(timeout)
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        monkeypatch.setattr(structured_worker.threading, 'Thread', DelayedThread)
        with pytest.raises(RuntimeError, match='thread retirement'):
            await worker.initialize()
        thread = worker.owner.io_thread
        assert thread.joins >= 2 and not thread.is_alive()
        assert worker.owner.poisoned and worker.process.poll() is not None and worker.lease.released
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        assert not pool.ledger.workers
    asyncio.run(run())


def broken_end_reset(model_id, options, action_config_hash, **classification):
    """Importable fault policy; its third reset leaves a defective sampling RNG."""
    from gf_models.structured_baselines import StructuredBaseline
    class Policy(StructuredBaseline):
        calls = 0
        def reset(self, seed):
            self.calls += 1
            return super().reset(31 if self.calls >= 3 else seed)
    return Policy(model_id, **classification)


def test_real_child_broken_end_rng_reset_never_acknowledges():
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool, factory=broken_end_reset)
        await worker.initialize()
        with pytest.raises(EOFError):
            await worker.close()
        assert worker.owner.poisoned and worker.owner.resets == 0
        assert worker.process.poll() is not None and worker.lease.released
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
    asyncio.run(run())


def test_failed_spawn_retains_unspawned_physical_cleanup_owner(monkeypatch):
    from gf_simulator_client import structured_worker
    from gf_simulator_client.structured_resources import WorkerReservation
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        original = WorkerReservation.cancel_before_spawn
        attempts = []
        def first_cancel_fails(lease):
            attempts.append(lease)
            if len(attempts) == 1:
                raise RuntimeError('injected first reservation retirement failure')
            return original(lease)
        def cannot_spawn(*args, **kwargs):
            raise OSError('injected Popen failure')
        monkeypatch.setattr(WorkerReservation, 'cancel_before_spawn', first_cancel_fails)
        monkeypatch.setattr(structured_worker.subprocess, 'Popen', cannot_spawn)
        with pytest.raises(OSError, match='Popen') as failure:
            logical(pool)
        owner = pool.owners[0]
        assert owner.process is None and not owner.lease.released and owner.poisoned
        assert failure.value._structured_startup_owner is owner
        assert owner.lease._cleanup_owner is owner and pool.ledger.workers[owner.lease.key] is owner.lease
        with pytest.raises(RuntimeError, match='poisoned'):
            await pool.close()
        assert owner.closed and owner.lease.released and not pool.ledger.workers
        assert attempts == [owner.lease, owner.lease]
    asyncio.run(run())


def test_default_helper_stays_fresh_after_explicit_pool_helper(tmp_path, monkeypatch):
    from gf_simulator_client import structured_environment
    from tests.structured_ppo_support import component_http, configuration
    loaded = configuration(tmp_path)
    component_http(monkeypatch, loaded, pool_cpu=True)
    assert issubclass(structured_environment.StructuredParticipantWorker, PooledStructuredParticipantWorker)
    component_http(monkeypatch, loaded)  # Same patch lifetime; explicit pool does not infect the default.
    assert issubclass(structured_environment.StructuredParticipantWorker, StructuredParticipantWorker)
    assert not issubclass(structured_environment.StructuredParticipantWorker, PooledStructuredParticipantWorker)


@pytest.mark.parametrize('learner_device,actor_device', [('cuda:0', 'cpu'), ('cuda:0', 'cuda:0'), ('cpu', 'cuda:0')])
def test_cpu_module_wrapper_leaves_cuda_helpers_fresh(tmp_path, monkeypatch, learner_device, actor_device):
    from gf_simulator_client import structured_environment
    from tests.structured_ppo_support import configuration
    from tests.test_structured_ppo import component_http
    loaded = configuration(tmp_path, learning_device=learner_device, inference_device=actor_device)
    component_http(monkeypatch, loaded)
    assert not issubclass(structured_environment.StructuredParticipantWorker, PooledStructuredParticipantWorker)


def test_retired_logical_handle_cannot_mutate_or_poison_next_loan(make_input):
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        retired = logical(pool)
        first_frame, first_delta = own_frame(make_input(), EventBank(1, 'retired-own-episode'), 0)
        await retired.select_frame(first_frame, first_delta)
        await retired.close()
        current = logical(pool, seed=29)
        assert current.owner is retired.owner
        owner, lease = current.owner, current.lease
        retired_bank, retired_frame, retired_search = retired._parent_bank, retired._last_frame, retired._search_state
        boundary = retired_bank.boundary
        lease_serial, epoch, process = lease.serial, owner.epoch, owner.process
        try:
            for operation in (retired.initialize(), retired.select_frame(first_frame, first_delta), retired._roundtrip(retired.initial)):
                with pytest.raises(RuntimeError, match='no current live loan'):
                    await operation
                assert owner.loan is current and owner.epoch == epoch and owner.process is process
                assert not owner.poisoned and not owner.closed and process.poll() is None
                assert not lease.released and lease.serial == lease_serial and not lease.decisions
                assert lease._cleanup_owner is owner and pool.ledger.workers == {lease.key: lease}
                assert current._parent_bank is current._last_frame is None and not current.initialized
                assert retired._parent_bank is retired_bank and retired_bank.boundary == boundary
                assert retired._last_frame is retired_frame and retired._search_state == retired_search
            frame, delta = own_frame(make_input(), EventBank(1, 'current-own-episode'), 0)
            generated, selected = await current.select_frame(frame, delta)
            selected.validate(generated.table, frame)
            assert owner.loan is current and not owner.poisoned and not lease.decisions
            assert lease.serial == lease_serial + 1
        finally:
            await current.close()
            receipt = await pool.close()
        assert receipt['logical_participants'] == receipt['resets'] == 2 and receipt['physical_workers'] == 1
    asyncio.run(run())


def test_reset_protocol_error_survives_secondary_physical_cleanup_failure(monkeypatch):
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        await worker.initialize()
        owner = worker.owner
        original_read, original_close = owner._read_reply, owner._close_owned
        secondary = RuntimeError('injected secondary physical cleanup failure')
        def malformed_reset(source):
            response = original_read(source)
            response['epoch'] += 1
            return response
        async def cleanup_failure(timeout, started):
            await original_close(timeout, started)
            raise secondary
        monkeypatch.setattr(owner, '_read_reply', malformed_reset)
        monkeypatch.setattr(owner, '_close_owned', cleanup_failure)
        with pytest.raises(ContractError, match='epoch') as failure:
            await worker.close()
        primary = failure.value
        assert any('secondary physical cleanup failure' in note for note in primary.__notes__)
        assert worker.health['abort_cleanup_error'] == 'RuntimeError'
        assert worker._close_task.exception() is primary
        with pytest.raises(ContractError, match='epoch') as repeated:
            await worker.close()
        assert repeated.value is primary
        assert owner.poisoned and owner.process.poll() is not None and owner.lease.released
        assert owner.io_thread is None or not owner.io_thread.is_alive()
        assert not pool.ledger.workers and owner.loan is None
        with pytest.raises(RuntimeError, match='secondary physical cleanup') as final:
            await pool.close()
        assert final.value is secondary
    asyncio.run(run())


@pytest.mark.parametrize('secondary_cleanup_failure', [False, True])
def test_short_reset_thread_join_overrun_cannot_return_healthy_loan(monkeypatch, secondary_cleanup_failure):
    import threading
    import time

    from gf_simulator_client import structured_worker
    original_thread = threading.Thread
    class ShortDelayedThread(original_thread):
        def run(self):
            super().run()  # ACK is already in the parent's result queue.
            time.sleep(.05)  # Within the 100 ms join allowance, beyond this 20 ms close.
    async def run():
        pool = CpuParticipantPool(ReservationLedger())
        worker = logical(pool)
        await worker.initialize()
        owner = worker.owner
        monkeypatch.setattr(structured_worker.threading, 'Thread', ShortDelayedThread)
        if secondary_cleanup_failure:
            original_close = owner._close_owned
            async def cleanup_fails(timeout, started):
                await original_close(timeout, started)
                raise RuntimeError('injected cleanup failure after short reset overrun')
            monkeypatch.setattr(owner, '_close_owned', cleanup_fails)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match='original logical close deadline') as failure:
            await worker.close(timeout=.02)
        assert time.monotonic() - started > .02
        if secondary_cleanup_failure:
            assert any('cleanup failure after short reset overrun' in note for note in failure.value.__notes__)
        first = worker._close_task
        with pytest.raises(TimeoutError, match='original logical close deadline') as repeated:
            await worker.close(timeout=10)
        assert worker._close_task is first and repeated.value is failure.value
        assert owner.poisoned and owner.resets == 0 and owner.loan is None
        assert owner.io_thread is not None and not owner.io_thread.is_alive()
        assert owner.process.poll() is not None and owner.lease.released and not pool.ledger.workers
        with pytest.raises(RuntimeError, match='poisoned|cleanup failure after short reset overrun'):
            await pool.close()
    asyncio.run(run())
