"""Owned native FFI cancellation joins work, including construction and races."""
import asyncio
import json
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from gf_common.preprocessing import PreparationConfig
from gf_training.native_authorization import NativePreparation
from gf_training.native_execution import NativeExecution


class Cancel:
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.event = threading.Event()
    def cancel(self):
        self.event.set()


def owned_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith('gf-native-ffi')]


async def wait_event(event):
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(.001)


def test_sync_interrupt_signals_handle_joins_and_allows_cleanup(monkeypatch):
    cancel = Cancel()
    entered, ended = threading.Event(), threading.Event()
    calls = []
    def operation():
        entered.set()
        cancel.event.wait(5)
        ended.set()
        return 7
    with pytest.raises(KeyboardInterrupt), NativeExecution(cancel) as execution:
        submit = execution._submit
        def interrupted(function, *args):
            future = submit(function, *args)
            original = future.result
            def result(timeout=None):
                if not calls:
                    calls.append('interrupt')
                    raise KeyboardInterrupt('original interrupt')
                return original(timeout=timeout)
            monkeypatch.setattr(future, 'result', result)
            return future
        monkeypatch.setattr(execution, '_submit', interrupted)
        execution.call(operation)
    assert entered.is_set() and ended.is_set() and cancel.event.is_set()
    assert not owned_threads()


def test_async_repeated_cancellation_waits_for_owned_work_and_success_race_loses():
    async def run():
        cancel = Cancel()
        entered, release, ended = threading.Event(), threading.Event(), threading.Event()
        def operation():
            entered.set()
            release.wait(5)
            ended.set()
            return 'successful result must be discarded'
        async with NativeExecution(cancel) as execution:
            task = asyncio.create_task(execution.acall(operation))
            await wait_event(entered)
            task.cancel('original cancellation')
            await wait_event(cancel.event)
            for _ in range(5):
                task.cancel('repeated cancellation')
                await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match='original cancellation'):
                await task
            assert ended.is_set()
            # Cleanup remains usable after cancellation on the same FFI thread.
            assert await execution.acall(lambda: 11) == 11
        assert not owned_threads()
    asyncio.run(run())


def test_completed_success_cannot_override_pending_cancellation():
    async def run():
        cancel = Cancel()
        loop = asyncio.get_running_loop()
        async with NativeExecution(cancel) as execution:
            def operation():
                loop.call_soon_threadsafe(task.cancel, 'race cancellation')
                return 123
            task = asyncio.create_task(execution.acall(operation))
            with pytest.raises(asyncio.CancelledError, match='race cancellation'):
                await task
        assert cancel.event.is_set()
        assert not owned_threads()
    asyncio.run(run())


@pytest.mark.parametrize('asynchronous', [False, True])
def test_completed_futures_do_not_retain_exported_batch_leases(asynchronous):
    class BatchLease:
        pass
    previous = None
    def next_batch():
        if previous is not None:
            assert previous() is None  # Admission max_in_flight=1; no GC workaround.
        return BatchLease()
    async def run_async():
        nonlocal previous
        async with NativeExecution(Cancel()) as execution:
            first = await execution.acall(next_batch)
            previous = weakref.ref(first)
            del first
            second = await execution.acall(next_batch)
            del second
    if asynchronous:
        asyncio.run(run_async())
    else:
        with NativeExecution(Cancel()) as execution:
            first = execution.call(next_batch)
            previous = weakref.ref(first)
            del first
            second = execution.call(next_batch)
            del second
    assert not owned_threads()


def fake_native(monkeypatch, blocked, *, success_on_cancel=False):
    import gf_training.native_authorization as module
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    sessions, handles, worker_threads, authorities = [], [], [], []
    main = threading.get_ident()
    class Handle(Cancel):
        def __init__(self, deadline):
            super().__init__(deadline)
            handles.append(self)
    def phase(name, handle):
        worker_threads.append(threading.get_ident())
        assert threading.get_ident() != main
        if name == blocked:
            entered.set()
            release.wait(5)
            ended.set()
            if handle.event.is_set() and not success_on_cancel:
                raise RuntimeError('fake native cancelled')
    class Session:
        def __init__(self, plan, settings, deadline, wall, handle):
            self.handle, self.closed = handle, False
            sessions.append(self)
            phase('construct', handle)
        def control_summary(self):
            phase('summary', self.handle)
            return json.dumps({'parents': [], 'collections': [], 'control_fingerprint': 'fake',
                               'authorization_wire_bound_bytes': 65536}).encode()
        def authorize(self, authorization):
            phase('authorize', self.handle)
        def next_batch(self, retained, layout):
            phase('next', self.handle)
        def finalize(self):
            phase('finalize', self.handle)
            self.closed = True
            return b'{}'
        def close(self):
            phase('close', self.handle)
            self.closed = True
        def peak_bytes(self):
            phase('peak', self.handle)
            return 0
    def controls(summary, check=None):
        authorities.append(threading.get_ident())
        assert threading.get_ident() == main
        return iter(())
    monkeypatch.setattr(module, 'load_native', lambda: SimpleNamespace(CancelHandle=Handle, NativeSession=Session))
    monkeypatch.setattr(module, 'binary_identity', lambda native: {'test': 'fake'})
    monkeypatch.setattr(module, '_expected_collection_items', controls)
    return SimpleNamespace(entered=entered, release=release, ended=ended, sessions=sessions,
                           handles=handles, worker_threads=worker_threads, authorities=authorities)


@pytest.mark.parametrize('phase', ['construct', 'summary', 'authorize', 'next', 'finalize'])
def test_async_preparation_cancellation_joins_each_lifecycle_phase(monkeypatch, phase):
    fake = fake_native(monkeypatch, phase)
    async def run():
        async def lifecycle():
            owner = await NativePreparation.create_async({'kind': 'offline'}, PreparationConfig(backend='rust'),
                                                         deadline=time.monotonic() + 10)
            async with owner:
                await owner.anext_batch(0, (0, 0, 0, 0))
        task = asyncio.create_task(lifecycle())
        await wait_event(fake.entered)
        task.cancel('lifecycle cancellation')
        await wait_event(fake.handles[0].event)
        for _ in range(3):
            task.cancel('repeat')
            await asyncio.sleep(0)
        assert not task.done()
        fake.release.set()
        with pytest.raises(asyncio.CancelledError, match='lifecycle cancellation'):
            await task
        assert fake.ended.is_set()
        if phase != 'construct':
            assert fake.sessions[0].closed
        assert len(set(fake.worker_threads)) == 1
        assert not owned_threads()
    asyncio.run(run())


def test_constructor_success_cancel_race_captures_handle_for_explicit_close(monkeypatch):
    fake = fake_native(monkeypatch, 'construct', success_on_cancel=True)
    async def run():
        task = asyncio.create_task(NativePreparation.create_async({'kind': 'offline'}, PreparationConfig(backend='rust'),
                                                                  deadline=time.monotonic() + 10))
        await wait_event(fake.entered)
        task.cancel('constructor cancellation')
        await wait_event(fake.handles[0].event)
        fake.release.set()
        with pytest.raises(asyncio.CancelledError, match='constructor cancellation'):
            await task
        assert fake.sessions[0].closed
        assert not fake.authorities  # No authorization after the cancelled constructor.
        assert not owned_threads()
    asyncio.run(run())


def test_real_native_precreated_handle_cancels_constructor_work(tmp_path):
    native = pytest.importorskip('gf_native')
    from gf_training.native_backend import reader_settings
    settings = PreparationConfig(backend='rust', memory_budget_mib=256, cache_budget_mib=16)
    # Source parsing is deliberately long; cancellation must be observed before
    # semantic parent validation. No simulator, source-run copy or policy is used.
    parent = tmp_path / 'parent'
    parent.mkdir()
    for directory in ('generations', 'updates', 'artifacts'):
        (parent / directory).mkdir()
    (parent / 'run.json').write_text('{"padding":[' + ','.join(['0'] * 300000) + ']}')
    deadline = time.monotonic() + 10
    handle = native.CancelHandle(deadline)
    plan = json.dumps({'kind': 'online', 'parent_path': str(parent), 'collection_path': str(parent / 'missing'),
                       'encoder': {'history': 'none', 'max_elements': 3, 'hash_buckets': 32, 'history_window': 128}}).encode()
    async def run():
        async with NativeExecution(handle) as execution:
            task = asyncio.create_task(execution.acall(native.NativeSession, plan,
                                                       json.dumps(reader_settings(settings, native)).encode(), deadline, deadline, handle))
            await asyncio.sleep(.001)
            task.cancel('real native cancellation')
            with pytest.raises(asyncio.CancelledError, match='real native cancellation'):
                await task
        assert not owned_threads()
    asyncio.run(run())


def test_worker_failure_after_repeated_cancellation_preserves_original_and_retrieves_error():
    async def run():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        cancel, entered, release = Cancel(), threading.Event(), threading.Event()
        def failed():
            entered.set()
            release.wait(5)
            raise RuntimeError('native failure after cancellation')
        async with NativeExecution(cancel) as execution:
            task = asyncio.create_task(execution.acall(failed))
            await wait_event(entered)
            task.cancel('first cancellation')
            await wait_event(cancel.event)
            task.cancel('second cancellation')
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError, match='first cancellation') as caught:
                await task
            assert any('native failure after cancellation' in note for note in caught.value.__notes__)
        await asyncio.sleep(0)
        assert not errors
        assert not owned_threads()
    asyncio.run(run())


@pytest.mark.parametrize('asynchronous', [False, True])
def test_native_deadline_exception_is_not_a_future_polling_timeout(asynchronous):
    error = TimeoutError('original native deadline')
    def expired():
        raise error
    async def run():
        async with NativeExecution(Cancel()) as execution:
            with pytest.raises(TimeoutError) as caught:
                await execution.acall(expired)
            assert caught.value is error
    if asynchronous:
        asyncio.run(run())
    else:
        with NativeExecution(Cancel()) as execution:
            with pytest.raises(TimeoutError) as caught:
                execution.call(expired)
            assert caught.value is error
    assert not owned_threads()
