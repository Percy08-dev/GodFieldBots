"""Original-deadline server cleanup using real kernel children; no Source proof."""

import asyncio
import sys
import time
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_simulator_client.process import ManagedServer

CONFIG = Path(__file__).parents[1] / "configs/test-server.toml"


def owner(tmp_path, *, deadline, strict=True, shutdown=.01):
    return ManagedServer(sys.executable, str(CONFIG),
        "00000000-0000-0000-0000-000000000017", 1, tmp_path,
        shutdown_seconds=shutdown, cleanup_deadline=deadline, strict_cleanup_deadline=strict)


async def child(server, *, ignore_term=False):
    program = ("import signal,sys,time;"
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_term else "")
        + "print('ready',flush=True);"
        + ("time.sleep(60)" if ignore_term else "sys.stdin.buffer.read()"))
    server.process = await asyncio.create_subprocess_exec(sys.executable, "-c", program,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    assert await server.process.stdout.readline() == b"ready\n"


@pytest.mark.parametrize("deadline", [float("inf"), float("nan"), True])
def test_strict_cleanup_requires_finite_original_deadline(tmp_path, deadline):
    with pytest.raises(ContractError, match="original"):
        owner(tmp_path, deadline=deadline)


def test_actual_normal_strict_cleanup_has_positive_original_status(tmp_path):
    async def run():
        server = owner(tmp_path, deadline=time.monotonic() + 2, shutdown=.5)
        await child(server)
        health = await server.stop()
        assert server.stopped and health == {
            "exit_code": 0, "shutdown": "stdin_eof", "log_error": None}
    asyncio.run(run())


def test_spent_strict_cleanup_kills_without_fresh_reap_window(tmp_path):
    async def run():
        server = owner(tmp_path, deadline=time.monotonic() + .1)
        await child(server, ignore_term=True)
        await asyncio.sleep(max(0, server.cleanup_deadline - time.monotonic()) + .01)
        started = time.monotonic()
        health = await server.stop()
        assert time.monotonic() - started < .05
        assert not server.stopped and health["exit_code"] is None
        assert "reap_failure" in health
        # TEST ONLY cleanup of the actual retained original; no production grace.
        await asyncio.wait_for(server.process.wait(), 2)
        assert server.process.returncode == -9
    asyncio.run(run())


def test_old_default_spent_cleanup_keeps_existing_mandatory_reap(tmp_path):
    async def run():
        server = owner(tmp_path, deadline=time.monotonic() + .1, strict=False)
        await child(server, ignore_term=True)
        await asyncio.sleep(max(0, server.cleanup_deadline - time.monotonic()) + .01)
        health = await server.stop()
        assert server.stopped and health["exit_code"] == -9
    asyncio.run(run())


def test_cancelled_strict_stop_retains_original_and_requests_owned_kill(tmp_path):
    async def run():
        server = owner(tmp_path, deadline=time.monotonic() + 2, shutdown=1)
        await child(server, ignore_term=True)
        operation = asyncio.create_task(server.stop())
        await asyncio.sleep(.01)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await operation
        assert caught.value.cleanup_owner is server and not server.stopped
        await asyncio.wait_for(server.process.wait(), 2)  # TEST ONLY cleanup
        assert server.process.returncode == -9
    asyncio.run(run())


def test_real_startup_failure_uses_same_strict_cleanup_owner(tmp_path):
    script = tmp_path / "invalid-server"
    script.write_text(f"#!{sys.executable}\nimport sys\nprint('{{}}',flush=True)\nsys.stdin.buffer.read()\n")
    script.chmod(0o700)
    async def run():
        server = ManagedServer(str(script), str(CONFIG),
            "00000000-0000-0000-0000-000000000017", 1, tmp_path / "server",
            startup_seconds=.05, cleanup_deadline=time.monotonic() + 1,
            strict_cleanup_deadline=True)
        with pytest.raises(TimeoutError):
            await server.start()
        assert server.stopped and server.health["exit_code"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("disposition", ["timeout", "cancel", "error", "eof"])
def test_fix1_strict_real_drain_failure_is_sticky_after_actual_settlement(tmp_path, disposition):
    async def run():
        server = owner(tmp_path, deadline=time.monotonic() + .05)
        server.process = await asyncio.create_subprocess_exec("/bin/true")
        await server.process.wait()
        stream = asyncio.StreamReader()
        drainer = asyncio.create_task(server._drain(stream, tmp_path / "actual-drain.log", False))
        server.drainers = [drainer]
        await asyncio.sleep(0)
        if disposition == "cancel":
            drainer.cancel()
            await asyncio.gather(drainer, return_exceptions=True)
        elif disposition == "error":
            stream.set_exception(OSError("actual reader error"))
        elif disposition == "eof":
            stream.feed_eof()
        health = await server.stop()
        assert drainer.done() and server.stopped
        if disposition == "eof":
            assert health["log_error"] is None and not drainer.cancelled()
        else:
            assert health["log_error"] is not None
            failure = health["log_error"]
            assert (await server.stop())["log_error"] == failure
    asyncio.run(run())
