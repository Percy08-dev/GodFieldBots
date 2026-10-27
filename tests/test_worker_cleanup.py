"""Actual resistant workers must remain owned through repeated cancellation."""
import asyncio
import sys
import time
from pathlib import Path

import pytest

from gf_cli.bindings import native_preflight
from gf_common.domain import ActionConfig, RuntimePolicyBinding
from gf_simulator_client.workers import ParticipantWorker
from tests.worker_fixtures import resistant_ready_factory, resistant_reset_factory

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="uses /proc to prove actual TERM resistance")


def assert_reaped(worker):
    assert worker.process.poll() is not None
    assert worker.process.stdin.closed and worker.process.stdout.closed
    assert worker.io_thread is None or not worker.io_thread.is_alive()
    assert worker._close_task.done()


async def await_term_resistant(worker):
    """Observe OS signal disposition before a test starts aborting initialization."""
    if sys.platform != "linux":
        pytest.skip("bounded /proc signal-disposition verification requires Linux")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = Path(f"/proc/{worker.process.pid}/status").read_text()
        ignored = next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:"))
        if int(ignored, 16) & (1 << (15 - 1)):
            return
        await asyncio.sleep(.01)
    pytest.fail("worker never installed TERM resistance")


def hold_owned_cleanup(worker):
    entered, release = asyncio.Event(), asyncio.Event()
    original = worker._close_owned
    calls = []
    async def held(timeout, started):
        calls.append((timeout, started))
        # The production-owned shared task is held after a real TERM has been
        # sent. The first deadline remains anchored before this barrier.
        worker.process.terminate()
        entered.set()
        await release.wait()
        await original(timeout, started)
    worker._close_owned = held
    return entered, release, calls


def test_preflight_close_cancel_joins_resistant_worker_and_later_close_is_idempotent(monkeypatch):
    import gf_simulator_client.workers as module
    async def run():
        instances, close_cancellations = [], []
        def factory(*args, **kwargs):
            worker = ParticipantWorker(resistant_ready_factory, "resistant", 0, ActionConfig(), 2)
            worker.barriers = hold_owned_cleanup(worker)
            original_close = worker.close
            async def close(timeout=2):
                try:
                    await original_close(timeout)
                except asyncio.CancelledError as exc:
                    close_cancellations.append(exc)
                    raise
            worker.close = close
            instances.append(worker)
            return worker
        monkeypatch.setattr(module, "ParticipantWorker", factory)
        registry = {"test": RuntimePolicyBinding("lightgbm")}
        task = asyncio.create_task(native_preflight(registry, ActionConfig(), deadline=time.monotonic() + 4, worker_seconds=2))
        while not instances:
            await asyncio.sleep(.01)
        worker = instances[0]
        entered, release, calls = worker.barriers
        joiner = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await await_term_resistant(worker)
            first_close_task = worker._close_task
            task.cancel("first close cancellation")
            await asyncio.sleep(.01)
            task.cancel("repeated close cancellation")
            joiner = asyncio.create_task(worker.close(timeout=100))
            await asyncio.sleep(.02)
            assert not task.done() and not joiner.done()
            assert worker.process.poll() is None and not worker.process.stdout.closed
            assert worker._close_task is first_close_task and len(calls) == 1
            release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value is close_cancellations[0]
            assert raised.value.args == ("first close cancellation",)
            await joiner
            assert_reaped(worker)
            await worker.close(timeout=100)
            assert worker._close_task is first_close_task and len(calls) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel("test finalizer")
            await asyncio.gather(task, return_exceptions=True)
            await worker.close(timeout=0)
            if joiner is not None:
                await asyncio.gather(joiner, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "cancel", "error"])
def test_initialize_abort_retains_original_exception_through_repeated_cleanup_cancels(failure):
    async def run():
        worker = ParticipantWorker(resistant_reset_factory if failure != "error" else resistant_ready_factory,
                                   "resistant", 0, ActionConfig(), .5 if failure == "timeout" else 3)
        originals = []
        abort = worker._abort
        async def observed_abort(original):
            originals.append(original)
            await abort(original)
        worker._abort = observed_abort
        if failure == "error":
            await worker.initialize()
        entered, release, calls = hold_owned_cleanup(worker)
        task = None
        try:
            if failure == "error":
                sentinel = RuntimeError("original native initialization error")
                task = asyncio.create_task(worker._abort(sentinel))
            else:
                task = asyncio.create_task(worker.initialize())
                await await_term_resistant(worker)
                if failure == "cancel":
                    task.cancel("original initialize cancellation")
            await asyncio.wait_for(entered.wait(), 2)
            first_close_task = worker._close_task
            task.cancel("secondary cleanup cancellation one")
            await asyncio.sleep(.01)
            task.cancel("secondary cleanup cancellation two")
            await asyncio.sleep(.02)
            assert not task.done() and worker.process.poll() is None
            assert worker.closed and not worker.process.stdin.closed and not worker.process.stdout.closed
            assert len(originals) == len(calls) == 1 and calls[0][0] == 0
            release.set()
            expected = TimeoutError if failure == "timeout" else asyncio.CancelledError if failure == "cancel" else RuntimeError
            with pytest.raises(expected) as raised:
                await task
            assert raised.value is originals[0]
            if failure == "cancel":
                assert raised.value.args == ("original initialize cancellation",)
            assert_reaped(worker)
            await worker.close(timeout=99)
            assert worker._close_task is first_close_task and len(calls) == 1
        finally:
            release.set()
            if task is not None:
                if not task.done():
                    task.cancel("test finalizer")
                await asyncio.gather(task, return_exceptions=True)
            await worker.close(timeout=0)
    asyncio.run(run())


def test_close_retains_original_cancellation_when_joined_cleanup_task_fails(monkeypatch):
    async def run():
        worker = ParticipantWorker(resistant_ready_factory, "resistant", 0, ActionConfig(), 2)
        await worker.initialize()
        entered, release, _ = hold_owned_cleanup(worker)
        held_cleanup = worker._close_owned
        cleanup_error = RuntimeError("cleanup failed after actual reap")
        async def failing_cleanup(timeout, started):
            await held_cleanup(timeout, started)
            raise cleanup_error
        worker._close_owned = failing_cleanup
        shield = asyncio.shield
        cancellations = []
        async def observed_shield(task):
            try:
                return await shield(task)
            except asyncio.CancelledError as exc:
                cancellations.append(exc)
                raise
        monkeypatch.setattr(asyncio, "shield", observed_shield)
        task = asyncio.create_task(worker.close(timeout=2))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel("original caller cancellation")
            await asyncio.sleep(.01)
            assert not task.done() and worker.process.poll() is None
            release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value is cancellations[0]
            assert raised.value.args == ("original caller cancellation",)
            assert raised.value.__cause__ is cleanup_error
            assert worker.health["cleanup_failure"] == str(cleanup_error)
            assert_reaped(worker)
            with pytest.raises(RuntimeError) as repeated:
                await worker.close(timeout=100)
            assert repeated.value is cleanup_error
        finally:
            release.set()
            if not task.done():
                task.cancel("test finalizer")
            await asyncio.gather(task, return_exceptions=True)
            try:
                await worker.close(timeout=0)
            except RuntimeError:
                pass
    asyncio.run(run())
