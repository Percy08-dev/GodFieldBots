"""Cheap admission/ownership guards; real numerical-launch faults are opt-in."""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.ppo_run import LearningProcess
from gf_training.structured_learning_process import StructuredLearningProcess
from tests.numeric_preload import NumericTemplate, _Record, activate, supported_numeric
from tests.numeric_preload_bootstrap import AdmissionInput

requires_numeric = pytest.mark.skipif(
    os.environ.get('GFB_RUN_CPU_PRELOAD_SAFETY') != '1' or not supported_numeric(),
    reason='set GFB_RUN_CPU_PRELOAD_SAFETY=1; pinned CPython/Linux/Torch required',
)


def test_unstarted_numeric_import_support_and_close_never_import_torch(tmp_path):
    source = """
import asyncio,sys,time
from tests.numeric_preload import NumericTemplate,supported_numeric
from pathlib import Path
assert 'torch' not in sys.modules
service=NumericTemplate(Path(sys.argv[1]))
supported_numeric()
assert 'torch' not in sys.modules
asyncio.run(service.close(deadline=time.monotonic()+1))
assert service.process is None and service.state=='CLOSED'
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, '-E', '-c', source, str(tmp_path)], check=True, timeout=10)


@pytest.mark.parametrize('mutation', ['legacy', 'cuda', 'visibility', 'bool'])
def test_admission_rejects_before_original_numeric_main(mutation):
    packet = {'protocol_version': 3, 'identity': {'execution': {'binding': {'device': 'cpu'}, 'cuda_visible_devices': ''}}}
    if mutation == 'legacy':
        packet['protocol_version'] = 1
    elif mutation == 'bool':
        packet['protocol_version'] = True
    elif mutation == 'cuda':
        packet['identity']['execution']['binding']['device'] = 'cuda:0'
    else:
        packet['identity']['execution']['cuda_visible_devices'] = '0'
    with pytest.raises(ValueError, match='explicit CPU'):
        next(AdmissionInput(io.StringIO(json.dumps(packet) + '\n')))


def test_production_visibility_guard_has_no_template_escape(monkeypatch):
    from gf_training.structured_numeric_worker import _select_visibility
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='2.14.1+cu132'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'unchanged')
    monkeypatch.setattr(sys, '_gfb_numeric_template_origin', (os.getppid(),), raising=False)
    with pytest.raises(ContractError, match='Torch imported before visibility'):
        _select_visibility({'protocol_version': 3}, SimpleNamespace(device='cpu'), '')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'unchanged'


def test_terminal_returncode_and_repeated_wait_block_canonical_update_before_write(tmp_path):
    service = NumericTemplate(tmp_path)
    owner = StructuredLearningProcess.__new__(StructuredLearningProcess)
    LearningProcess.__init__(owner, tmp_path, 1024)
    owner.protocol, owner.actions, owner.identity = 3, None, {}
    owner.lease = SimpleNamespace(released=False)
    record = _Record(service, 1, owner, time.monotonic() + 1)
    owner.process = record.facade
    writes = []
    owner.process.stdin = SimpleNamespace(write=writes.append)
    failure = service.poison(RuntimeError('retained numeric status loss'))
    async def run():
        for _ in range(2):
            with pytest.raises(RuntimeError) as result:
                await asyncio.wait_for(record.facade.wait(), .1)
            assert result.value is failure
        with pytest.raises(RuntimeError) as result:
            await owner.update(tmp_path, 'a' * 64, 'policy', tmp_path / 'out', deadline=time.monotonic() + 1)
        assert result.value is failure and not writes
        record.status = 23
        assert record.facade.returncode == 23 and await record.facade.wait() == 23
    asyncio.run(run())


@pytest.mark.parametrize('registered', [False, True])
@pytest.mark.parametrize('drainer_failure', ['none', 'task', 'coroutine'])
def test_canonical_partial_adoption_notifies_and_submits_drainer_at_most_once(tmp_path, monkeypatch, registered, drainer_failure):
    owner = StructuredLearningProcess.__new__(StructuredLearningProcess)
    LearningProcess.__init__(owner, tmp_path, 1024)
    owner.ledger = ReservationLedger()
    owner.lease = owner.ledger.reserve_learner(0, owner)
    owner.lease.creation_pending = True
    notifications, submissions, coroutines = [], [], []
    def notify():
        notifications.append(owner.process)
        if registered:
            owner.lease.spawned(owner.process)
        raise MemoryError('callback after possible registration')
    owner._process_created = notify
    original = asyncio.create_task
    def submit(coroutine, **kwargs):
        submissions.append(coroutine)
        if drainer_failure == 'task':
            coroutines.append(coroutine)
            raise RuntimeError('drainer submission failed')
        return original(coroutine, **kwargs)
    if drainer_failure == 'coroutine':
        def fail_coroutine():
            raise RuntimeError('drainer coroutine construction failed')
        monkeypatch.setattr(owner, '_drain', fail_coroutine)
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_eof()
        process = SimpleNamespace(pid=123, returncode=None, stderr=reader)
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, 'create_task', submit)
            with pytest.raises(MemoryError, match='callback'):
                owner._adopt_process(process)
            owner._adopt_process(process)
            assert owner.process is process and len(notifications) == 1
            assert len(submissions) == (0 if drainer_failure == 'coroutine' else 1)
            with pytest.raises(ContractError, match='replace'):
                owner._adopt_process(SimpleNamespace(pid=456))
        if owner.drainer is not None:
            await owner.drainer
        assert all(coroutine.cr_frame is None for coroutine in coroutines)
    asyncio.run(run())


def test_numeric_warmup_refuses_running_loop_and_activation_restores_hooks(tmp_path):
    service = NumericTemplate(tmp_path)
    originals = (LearningProcess._create_process, StructuredLearningProcess._settle_creation, StructuredLearningProcess.close)
    with activate(service):
        assert LearningProcess._create_process is not originals[0]
    assert originals == (LearningProcess._create_process, StructuredLearningProcess._settle_creation, StructuredLearningProcess.close)
    async def run():
        with pytest.raises(RuntimeError, match='precede'):
            service.start(time.monotonic() + 1)
        await service.close(deadline=time.monotonic() + 1)
    asyncio.run(run())


def _child(tmp_path, service):
    from tests.test_structured_learning import learner_for
    from tests.test_structured_numeric_worker import process_for
    return process_for(tmp_path, learner_for('mlp', synthetic_capacity=False), ReservationLedger())


async def _shutdown(service):
    if service.failure is None:
        await service.close(deadline=time.monotonic() + 2)
    else:
        failure = service.failure
        with pytest.raises(BaseException) as result:
            await service.close(deadline=time.monotonic() + 2)
        assert result.value is failure
        assert service._leader_reaped and service.containment['physical_group_gone']


@requires_numeric
@pytest.mark.parametrize(('stage', 'index'), [('pipe', 1), ('pipe', 2), ('pipe', 3), ('file', 1), ('file', 2), ('file', 3), ('write', 1), ('read', 1), ('read', 2)])
@pytest.mark.parametrize('cancel', [False, True])
def test_actual_successive_preparation_failures_retire_every_resource_before_nochild(tmp_path, monkeypatch, stage, index, cancel):
    before = len(list(Path('/proc/self/fd').iterdir()))
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    failure = asyncio.CancelledError if cancel else MemoryError
    calls = 0
    async def run():
        nonlocal calls
        loop = asyncio.get_running_loop()
        target, name = ((os, 'pipe') if stage == 'pipe' else (os, 'fdopen') if stage == 'file'
            else (loop, 'connect_write_pipe') if stage == 'write' else (loop, 'connect_read_pipe'))
        original = getattr(target, name)
        if stage in ('pipe', 'file'):
            def failed(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == index:
                    raise failure('injected local preparation failure')
                return original(*args, **kwargs)
        else:
            async def failed(*args, **kwargs):
                nonlocal calls
                result = await original(*args, **kwargs)
                calls += 1
                if calls == index:
                    raise failure('injected after connection_made')
                return result
        with monkeypatch.context() as patch, activate(service):
            patch.setattr(target, name, failed)
            with pytest.raises(failure):
                await owner.start()
        assert owner.process is None and owner.lease.released and owner.ledger.learner is None
        assert service.launches == 0 and service.quiescent()
        assert len(asyncio.all_tasks()) == 1
        await _shutdown(service)
    asyncio.run(run())
    assert len(list(Path('/proc/self/fd').iterdir())) == before


@requires_numeric
@pytest.mark.parametrize('fault', ['no_fork', 'lost_ack', 'delayed_ack', 'lost_retire_ack'])
def test_actual_nochild_ack_resolution_retirement_retry_and_loop_rebinding(tmp_path, fault):
    before = len(list(Path('/proc/self/fd').iterdir()))
    service = NumericTemplate(tmp_path / 'template', fault=fault)
    service.start(time.monotonic() + 20)
    pids = []
    for index in range(2):
        owner = _child(tmp_path / f'logs{index}', service)
        async def run(owner=owner, index=index):
            with activate(service):
                if fault == 'no_fork':
                    with pytest.raises(RuntimeError, match='refused'):
                        await owner.start()
                    assert owner.process is None and owner.lease.released
                else:
                    await owner.start()
                    pids.append(owner.process.pid)
                    await owner.initialize(17, tmp_path / f'g{index}', restore=None, deadline=time.monotonic() + 10)
                    await owner.close(time.monotonic() + 2)
                    assert owner.process.returncode == 0 and owner.lease.released
                assert service.quiescent() and len(asyncio.all_tasks()) == 1
        asyncio.run(run())
    if pids:
        assert len(set(pids)) == 2
    asyncio.run(_shutdown(service))
    assert len(list(Path('/proc/self/fd').iterdir())) == before


@requires_numeric
@pytest.mark.parametrize('fault', ['after_fork_before_registration', 'after_registration_before_ack', 'after_waitpid_before_status'])
def test_actual_unknown_creation_or_status_stays_held_after_containment(tmp_path, monkeypatch, fault):
    before = len(list(Path('/proc/self/fd').iterdir()))
    service = NumericTemplate(tmp_path / 'template', fault=fault)
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    async def run():
        with activate(service):
            try:
                await owner.start()
                await owner.process.wait()
            except BaseException:  # noqa: BLE001, S110 - exact failure is retained below
                pass
            failure = service.failure
            assert failure is not None
            if owner.process is not None:
                writes = []
                monkeypatch.setattr(owner.process.stdin, 'write', writes.append)
                with pytest.raises(type(failure)):
                    await owner.update(tmp_path, 'a' * 64, 'policy', tmp_path / 'forbidden', deadline=time.monotonic() + 1)
                assert not writes
            for _ in range(2):
                with pytest.raises(type(failure)):
                    if owner.process is not None:
                        await asyncio.wait_for(owner.process.wait(), .1)
                    else:
                        raise failure
                with pytest.raises(type(failure)):
                    await owner.close(time.monotonic() + 1)
            assert owner._numeric_launch.status is None and not owner.lease.released
            assert owner.lease._cleanup_owner is owner
            await _shutdown(service)
            assert owner._numeric_launch.status is None and not owner.lease.released
    asyncio.run(run())
    assert len(list(Path('/proc/self/fd').iterdir())) == before


@requires_numeric
@pytest.mark.parametrize('late', [False, True])
def test_actual_after_send_repeated_cancellation_or_late_pid_uses_canonical_adoption(tmp_path, monkeypatch, late):
    service = NumericTemplate(tmp_path / 'template', fault='lost_ack' if not late else '')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    notifications = []
    original_notify = owner._process_created
    def notify():
        notifications.append(owner.process.pid)
        original_notify()
    monkeypatch.setattr(owner, '_process_created', notify)
    if late:
        original_dispatch = service._dispatch
        def dispatch(value, descriptors):
            original_dispatch(value, descriptors)
            if value.get('op') == 'launched':
                owner._numeric_launch.start_deadline = time.monotonic() - 1
                # The packet is valid, but the original execution bound elapsed
                # before creation can return. Preserve its actual PID for cleanup.
                raise service.poison(TimeoutError('injected late actual PID receipt'))
        monkeypatch.setattr(service, '_dispatch', dispatch)
    async def run():
        with activate(service):
            starting = asyncio.create_task(owner.start())
            if not late:
                while not hasattr(owner, '_numeric_launch') or owner._numeric_launch.state != 'SENT_PENDING':
                    await asyncio.sleep(.001)
                starting.cancel('original cancellation')
                await asyncio.sleep(.005)
                starting.cancel('repeated cancellation')
            with pytest.raises(TimeoutError if late else asyncio.CancelledError):
                await starting
            assert len(notifications) == 1 and owner.process is owner._numeric_launch.facade
            if late:
                assert owner._numeric_launch.pid is not None and not owner.lease.released
            else:
                assert owner.lease.released and owner.process.returncode is not None
            await _shutdown(service)
    asyncio.run(run())


@requires_numeric

def test_actual_backpressure_stderr_limit_and_real_transport_retirement(tmp_path):
    before = len(list(Path('/proc/self/fd').iterdir()))
    service = NumericTemplate(tmp_path / 'template', fault='numeric_io_witness')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    async def run():
        ticks = []
        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(.001)
        timer = asyncio.create_task(ticker())
        with activate(service):
            await owner.start()
            owner.process.stdin.write(b'p' * (8 * 1024**2))
            draining = asyncio.create_task(owner.process.stdin.drain())
            await asyncio.sleep(.03)
            assert not draining.done()
            before_ticks = len(ticks)
            async with asyncio.timeout(10):
                await draining
                assert json.loads(await owner.process.stdout.readline()) == {'transport_witness': True}
                assert await owner.process.wait() == 0
                await owner.drainer
            assert len(ticks) > before_ticks + 20
            assert owner.health['stderr_bytes'] == 2 * 1024**2 and owner.health['log_limit_exceeded']
            with pytest.raises(RuntimeError, match='health failure'):
                await owner.close(time.monotonic() + 2)
            assert owner.lease.released and owner._numeric_launch.local_retired
            assert not service.records
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)
        assert sum(path.stat().st_size for path in (tmp_path / 'logs').glob('*.log')) == 1024**2
        # A health failure is never silently reset to permit a new numeric child.
        assert service.failure is not None
        await _shutdown(service)
    asyncio.run(run())
    assert len(list(Path('/proc/self/fd').iterdir())) == before


@requires_numeric
@pytest.mark.parametrize('failure_point', ['notification', 'drainer'])
def test_actual_partial_adoption_failure_keeps_process_and_attempts_each_stage_once(tmp_path, monkeypatch, failure_point):
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    calls, submitted = [], []
    original_notify = owner._process_created
    original_task = asyncio.create_task
    def notify():
        calls.append(owner.process.pid)
        original_notify()
        if failure_point == 'notification':
            raise MemoryError('partial notification')
    def task(coroutine, **kwargs):
        if coroutine.cr_code.co_name == '_drain':
            submitted.append(coroutine)
            if failure_point == 'drainer':
                raise MemoryError('partial drainer submission')
        return original_task(coroutine, **kwargs)
    monkeypatch.setattr(owner, '_process_created', notify)
    async def run():
        with activate(service), monkeypatch.context() as patch:
            patch.setattr(asyncio, 'create_task', task)
            with pytest.raises(MemoryError, match='partial'):
                await owner.start()
            assert owner.process is owner._numeric_launch.facade
            assert len(calls) == len(submitted) == 1
            assert owner.lease.released and owner.process.returncode is not None
            assert submitted[0].cr_frame is None
            with pytest.raises(RuntimeError):
                await owner.close(time.monotonic() + 20)
            assert len(calls) == len(submitted) == 1
        await _shutdown(service)
    asyncio.run(run())


@requires_numeric
def test_actual_known_status_remains_readable_after_later_broker_death(tmp_path):
    import signal
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    async def run():
        with activate(service):
            await owner.start()
            await owner.initialize(17, tmp_path / 'g0', restore=None, deadline=time.monotonic() + 10)
            await owner.close(time.monotonic() + 2)
            assert owner.process.returncode == 0
        os.kill(service.process.pid, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while os.waitid(os.P_PID, service.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:  # noqa: ASYNC222 - WNOHANG never blocks.
            assert time.monotonic() < deadline
            await asyncio.sleep(.001)
        with pytest.raises(RuntimeError):
            service.require_healthy()
        assert owner.process.returncode == 0 and await owner.process.wait() == 0
        await _shutdown(service)
    asyncio.run(run())


@requires_numeric
@pytest.mark.parametrize('retire_failure', [False, True])
def test_actual_concurrent_cancelled_close_retains_first_retirement_deadline_and_error(tmp_path, monkeypatch, retire_failure):
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        original_retire = service._retire_ticket
        failure = RuntimeError('after actual retirement send')
        async def retire(record, deadline):
            entered.set()
            await release.wait()
            if retire_failure:
                service._send('retire', token=record.token)
                raise failure
            return await original_retire(record, deadline)
        monkeypatch.setattr(service, '_retire_ticket', retire)
        with activate(service):
            await owner.start()
            await owner.initialize(17, tmp_path / 'g0', restore=None, deadline=time.monotonic() + 10)
            first_deadline = time.monotonic() + 2
            first = asyncio.create_task(owner.close(first_deadline))
            await entered.wait()
            retained = owner._numeric_launch.close_task
            second = asyncio.create_task(owner.close(time.monotonic() + 100))
            await asyncio.sleep(0)
            first.cancel('first caller cancellation')
            second.cancel('second caller cancellation')
            await asyncio.sleep(0)
            first.cancel('repeated first caller cancellation')
            assert owner._numeric_launch.close_task is retained
            assert owner._numeric_launch.close_deadline == first_deadline
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            if retire_failure:
                assert results == [failure, failure]
                with pytest.raises(RuntimeError) as again:
                    await owner.close(time.monotonic() + 100)
                assert again.value is failure
                assert owner._numeric_launch.close_task is retained
                assert owner.process.returncode == 0 and owner.lease.released
                with pytest.raises(RuntimeError):
                    await service.health_async(time.monotonic() + 1)
            else:
                assert all(isinstance(result, asyncio.CancelledError) for result in results)
                assert service.quiescent() and owner._numeric_launch.retired
                await owner.close(time.monotonic() + 100)
        await _shutdown(service)
    asyncio.run(run())


@pytest.mark.parametrize('cancelled', [False, True])
def test_finished_creation_without_authoritative_outcome_never_cancels_pending_lease(tmp_path, cancelled):
    owner = StructuredLearningProcess.__new__(StructuredLearningProcess)
    LearningProcess.__init__(owner, tmp_path, 1024)
    owner.ledger = ReservationLedger()
    owner.lease = owner.ledger.reserve_learner(0, owner)
    owner.lease.creation_pending = True
    async def run():
        owner._startup_boundary = asyncio.Event()
        owner._startup_boundary.set()
        async def unresolved():
            if cancelled:
                raise asyncio.CancelledError('external creation status unknown')
            raise RuntimeError('external send may have happened')
        owner.creation_task = asyncio.create_task(unresolved())
        await asyncio.gather(owner.creation_task, return_exceptions=True)
        with pytest.raises(ContractError, match='authoritative settlement'):
            await owner._settle_creation(time.monotonic() + 1)
        assert owner.lease.creation_pending and not owner.lease.released
        assert owner.lease._cleanup_owner is owner
        with pytest.raises(ContractError):
            owner.lease.cancel_before_spawn()
    asyncio.run(run())


@requires_numeric
def test_actual_execution_deadline_is_distinct_from_cleanup_and_has_bounded_resolve(tmp_path):
    service = NumericTemplate(tmp_path / 'template', fault='late_ack', start_allowance=.03)
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    owner.cleanup_deadline = time.monotonic() + 30
    async def run():
        started = time.monotonic()
        with activate(service):
            with pytest.raises(TimeoutError):
                await owner.start()
            assert time.monotonic() - started < 2.5
            record = owner._numeric_launch
            assert record.start_deadline < owner.cleanup_deadline - 20
            assert len(service.expected) <= 2 and record.resolve_seq is None
            assert record.status is None and not owner.lease.released
            assert owner.lease._cleanup_owner is owner
            await _shutdown(service)
    asyncio.run(run())


def test_auxiliary_close_gate_rejects_before_loop_binding_or_preparation(tmp_path, monkeypatch):
    service = NumericTemplate(tmp_path)
    service._close_task = object()
    owner = SimpleNamespace(_creation_no_child=False)
    def unexpected():
        raise AssertionError('shutdown rejection must precede loop binding')
    monkeypatch.setattr(service, 'attach', unexpected)
    async def run():
        with pytest.raises(RuntimeError, match='shutdown has begun'):
            await service.create(owner)
        assert owner._creation_no_child and not service.records and service.launches == 0
    asyncio.run(run())


@requires_numeric
def test_actual_memory_observation_cannot_extend_absolute_start_deadline(tmp_path, monkeypatch):
    from tests import numeric_preload
    service = NumericTemplate(tmp_path / 'template', start_allowance=.3)
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    original_memory, original_notify = numeric_preload.memory, owner._process_created
    notifications = []
    def memory(pid):
        result = original_memory(pid)
        time.sleep(max(0, owner._numeric_launch.start_deadline - time.monotonic()) + .02)
        return result
    def notify():
        notifications.append(owner.process.pid)
        original_notify()
    monkeypatch.setattr(numeric_preload, 'memory', memory)
    monkeypatch.setattr(owner, '_process_created', notify)
    async def run():
        with activate(service):
            with pytest.raises(TimeoutError) as failed:
                await owner.start()
            record = owner._numeric_launch
            assert record.pid is not None and owner.process is record.facade
            assert notifications == [record.pid]
            assert record.start_deadline < time.monotonic()
            assert not owner.lease.released and owner.lease._cleanup_owner is owner
            assert service._close_task.done() and service._leader_reaped
            assert service.containment['physical_group_gone'] and service.loop is None
            assert service.failure is failed.value
            await _shutdown(service)
    asyncio.run(run())


@requires_numeric
def test_actual_create_while_auxiliary_close_awaits_ack_does_not_send_or_poison_close(tmp_path, monkeypatch):
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    async def run():
        close_sent = asyncio.Event()
        holding = True
        original_readable, original_send = service._readable, service._send
        operations = []
        def readable():
            if not holding:
                original_readable()
        def send(op, **kwargs):
            result = original_send(op, **kwargs)
            operations.append(op)
            if op == 'close':
                close_sent.set()
            return result
        monkeypatch.setattr(service, '_readable', readable)
        monkeypatch.setattr(service, '_send', send)
        closing = asyncio.create_task(service.close(deadline=time.monotonic() + 2))
        await close_sent.wait()
        assert service.state == 'RUNNING' and service._close_task is not None
        with activate(service), pytest.raises(RuntimeError, match='shutdown has begun'):
            await owner.start()
        assert owner.process is None and owner.lease.released and owner.ledger.learner is None
        assert service.launches == 0 and not service.records and operations == ['close']
        holding = False
        await closing
        assert service.state == 'CLOSED' and service.process.returncode == 0
        assert service.failure is None and service._leader_reaped
    asyncio.run(run())


@requires_numeric
@pytest.mark.parametrize('failure_point', ['local', 'drainer', 'ticket'])
def test_actual_retirement_error_joins_terminal_cleanup_before_owner_close_returns(tmp_path, monkeypatch, failure_point):
    service = NumericTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    error = RuntimeError('injected ' + failure_point + ' retirement error')
    async def run():
        with activate(service):
            await owner.start()
            await owner.initialize(17, tmp_path / 'g0', restore=None, deadline=time.monotonic() + 10)
            if failure_point == 'drainer':
                original = owner.drainer
                async def drainer():
                    await original
                    raise error
                owner.drainer = asyncio.create_task(drainer())
            elif failure_point == 'local':
                original = service._local_close
                async def local(record, deadline):
                    await original(record, deadline)
                    raise error
                monkeypatch.setattr(service, '_local_close', local)
            else:
                async def ticket(record, deadline):
                    service._send('retire', token=record.token)
                    raise error
                monkeypatch.setattr(service, '_retire_ticket', ticket)
            deadline = time.monotonic() + 2
            with pytest.raises(RuntimeError) as failed:
                await owner.close(deadline)
            assert failed.value is error and service.failure is error
            assert service._close_task.done() and service._leader_reaped
            assert service.containment['physical_group_gone'] and service.loop is None
            first_deadline, retained = service._close_deadline, service._close_task
            assert first_deadline <= owner._numeric_launch.close_deadline == deadline
            with pytest.raises(RuntimeError) as again:
                await owner.close(time.monotonic() + 100)
            assert again.value is error and service._close_task is retained
            assert service._close_deadline == first_deadline
            await _shutdown(service)
    asyncio.run(run())


@requires_numeric
@pytest.mark.parametrize('fault', ['stalled_health', 'control-failure'])
def test_actual_failed_health_is_terminal_contained_and_future_waiters_do_not_block(tmp_path, monkeypatch, fault):
    service = NumericTemplate(tmp_path / 'template', fault='stalled_health' if fault == 'stalled_health' else '')
    service.start(time.monotonic() + 20)
    async def run():
        service.attach()
        # A pending real-owner-style wait shares the same failure notification;
        # no fabricated child/PID or status is introduced by this waiter witness.
        waiting = asyncio.create_task(service.changed())
        await asyncio.sleep(0)
        error = RuntimeError('injected health control send failure')
        if fault == 'control-failure':
            original_send = service._send
            def send(op, **kwargs):
                if op == 'health':
                    raise error
                return original_send(op, **kwargs)
            monkeypatch.setattr(service, '_send', send)
        deadline = time.monotonic() + .05
        with pytest.raises(TimeoutError if fault == 'stalled_health' else RuntimeError) as failed:
            await service.health_async(deadline)
        failure = service.failure
        assert failure is failed.value
        if fault == 'control-failure':
            assert failure is error
        assert service._close_task.done() and service._leader_reaped
        assert service.containment['physical_group_gone'] and service.loop is None
        await waiting  # Consume the current notification before testing future waits.
        for _ in range(2):
            with pytest.raises(type(failure)) as future:
                await asyncio.wait_for(service.changed(), .1)
            assert future.value is failure
        assert service._close_deadline <= deadline
        await _shutdown(service)
    asyncio.run(run())


@requires_numeric
def test_actual_unknown_status_plus_local_cleanup_error_keeps_first_failure_and_lease(tmp_path, monkeypatch):
    service = NumericTemplate(tmp_path / 'template', fault='after_registration_before_ack')
    service.start(time.monotonic() + 20)
    owner = _child(tmp_path / 'logs', service)
    original = service._local_close
    secondary = MemoryError('secondary local retirement failure')
    async def local(record, deadline):
        await original(record, deadline)
        raise secondary
    monkeypatch.setattr(service, '_local_close', local)
    async def run():
        with activate(service):
            with pytest.raises((EOFError, RuntimeError)) as failed:
                await owner.start()
            assert service.failure is failed.value and service.failure is not secondary
            assert not owner.lease.released and owner.lease._cleanup_owner is owner
            assert owner._numeric_launch.status is None
            assert service._close_task.done() and service._leader_reaped
            assert service.containment['physical_group_gone'] and service.loop is None
            assert any('MemoryError' in note for note in service.failure.__notes__)
            await _shutdown(service)
    asyncio.run(run())
