"""Run-owned subprocess lifecycle, effective TestServer config and bounded cleanup."""
from __future__ import annotations

import asyncio
import json
import math
import time
import tomllib
from pathlib import Path

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_simulator_client.http import EFFECTIVE_KEYS, seed_fingerprint, validate_info


def server_effective(path: str | Path, count: int | None) -> Json:
    data = tomllib.loads(Path(path).read_text())
    closed(data, {"test_server", "lifecycle"})
    test = closed(data["test_server"], {"seed_use_limit", "server_idle_timeout_secs", "graceful_shutdown_timeout_secs"})
    lifecycle = closed(data["lifecycle"], {"max_invalid_attempts_per_decision", "waiting_for_players_timeout_secs", "player_action_timeout_secs", "max_game_turns"})
    values = {"schema_version": 1, **test, **lifecycle}
    integer(values["seed_use_limit"], 1)
    integer(values["max_invalid_attempts_per_decision"], 1, 2**32 - 1)
    integer(values["max_game_turns"], 150, 2**32 - 1)
    for key in ("waiting_for_players_timeout_secs", "player_action_timeout_secs", "graceful_shutdown_timeout_secs"):
        integer(values[key], 1, 86400)
    integer(values["server_idle_timeout_secs"], 0, 86400)
    if count is not None:
        values["seed_use_limit"] = count
        values["server_idle_timeout_secs"] = 0
    return values


class ManagedServer:
    def __init__(self, binary: str, config: str, seed: str, count: int, directory: Path,
                 startup_seconds: float = 15, shutdown_seconds: float = 30,
                 cleanup_deadline: float = float("inf"), *,
                 strict_cleanup_deadline: bool = False) -> None:
        if type(strict_cleanup_deadline) is not bool or strict_cleanup_deadline and (
                type(cleanup_deadline) not in (int, float) or not math.isfinite(cleanup_deadline)
                or cleanup_deadline <= 0):
            raise ContractError("strict server cleanup requires a finite original absolute deadline")
        self.binary = str(Path(binary).resolve())
        self.config = str(Path(config).resolve())
        self.seed = seed
        self.count = count
        self.directory = directory.resolve()
        self.startup_seconds = startup_seconds
        self.shutdown_seconds = shutdown_seconds
        self.cleanup_deadline = cleanup_deadline
        self.strict_cleanup_deadline = strict_cleanup_deadline
        self._creation: asyncio.Task[asyncio.subprocess.Process] | None = None
        self.expected = server_effective(config, count)
        self.process: asyncio.subprocess.Process | None = None
        self.stopped = False
        self.drainers: list[asyncio.Task[None]] = []
        self.startup: asyncio.Future[Json] | None = None
        self.health: Json = {"exit_code": None, "shutdown": None, "log_error": None}

    async def _drain(self, stream: asyncio.StreamReader, path: Path, startup: bool) -> None:
        try:
            with path.open("wb") as output:
                while line := await stream.readline():
                    output.write(line)
                    output.flush()
                    if startup and self.startup is not None and not self.startup.done():
                        try:
                            value = json.loads(line)
                            if isinstance(value, dict) and "base_url" in value:
                                self.startup.set_result(value)
                        except (ValueError, UnicodeDecodeError):
                            pass
        except (OSError, ValueError) as exc:
            self.health["log_error"] = str(exc)
            if self.startup is not None and not self.startup.done():
                self.startup.set_exception(exc)
        finally:
            if startup and self.startup is not None and not self.startup.done():
                self.startup.set_exception(ContractError("server stdout ended before startup"))

    async def start(self) -> Json:
        self.directory.mkdir(parents=True, exist_ok=False)
        self.startup = asyncio.get_running_loop().create_future()
        try:
            creation = asyncio.create_subprocess_exec(
                self.binary, "--config", self.config, "--seed", self.seed,
                "--seed-use-limit", str(self.count), "--server-idle-timeout-secs", "0",
                "--data-dir", str(self.directory / "simulator-data"),
                "--http-log-dir", str(self.directory / "http"), "--shutdown-on-stdin-eof",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=1024 * 1024,
            )
            if self.strict_cleanup_deadline:
                self._creation = asyncio.create_task(creation)
                async with asyncio.timeout(max(0., self.cleanup_deadline - time.monotonic())):
                    self.process = await asyncio.shield(self._creation)
            else:
                self.process = await creation
            assert self.process.stdout and self.process.stderr
            self.drainers = [asyncio.create_task(self._drain(self.process.stdout, self.directory / "stdout.log", True)),
                             asyncio.create_task(self._drain(self.process.stderr, self.directory / "stderr.log", False))]
            startup_seconds = (min(self.startup_seconds, max(0., self.cleanup_deadline - time.monotonic()))
                if self.strict_cleanup_deadline else self.startup_seconds)
            async with asyncio.timeout(startup_seconds):
                startup = await self.startup
            closed(startup, {"base_url", "http_log_path", "data_dir", "test_server_config", "test_server_info"})
            info = validate_info(startup["test_server_info"])
            if info["seed_mode"] != "repeated" or info["seed_fingerprint"] != seed_fingerprint(self.seed):
                raise ContractError("startup seed mismatch")
            if {k: info[k] for k in EFFECTIVE_KEYS} != self.expected or canonical(startup["test_server_config"]) != canonical(self.expected):
                raise ContractError("startup config does not match requested effective config")
            from urllib.parse import urlsplit
            parsed = urlsplit(startup["base_url"])
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None or parsed.path or parsed.query or parsed.fragment:
                raise ContractError("unexpected managed server URL")
            return startup
        except BaseException as error:
            if self.strict_cleanup_deadline:
                vars(error)["cleanup_owner"] = self
                try:
                    await self.stop()
                except BaseException as cleanup:  # noqa: BLE001 -- preserve initiating startup failure and real cleanup owner
                    error.add_note("original strict server cleanup remains owned: " + type(cleanup).__name__)
            else:
                await self.stop()
            raise

    async def stop(self, remaining_seconds: float | None = None) -> Json:
        if self.strict_cleanup_deadline:
            return await self._stop_strict(remaining_seconds)
        if self.stopped:
            return self.health
        process = self.process
        if process is None:
            self.stopped = True
            return self.health
        cleanup_started = time.monotonic()
        deadline = min(self.cleanup_deadline, cleanup_started + remaining_seconds if remaining_seconds is not None else float("inf"))

        def remaining(cap: float) -> float:
            return min(cap, max(0.0, deadline - time.monotonic()))

        if process.stdin:
            process.stdin.close()
        try:
            async with asyncio.timeout(remaining(self.shutdown_seconds)):
                await process.wait()
            self.health["shutdown"] = "stdin_eof"
        except TimeoutError:
            self.health["shutdown"] = "terminate"
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                async with asyncio.timeout(remaining(1.0)):
                    await process.wait()
            except TimeoutError:
                self.health["shutdown"] = "kill"
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                # Bounded mandatory reap allowance may extend a spent wall
                # budget; record this separately instead of silently overrunning.
                try:
                    async with asyncio.timeout(1.0):
                        await process.wait()
                except TimeoutError:
                    self.health["reap_failure"] = "process did not reap within kill allowance"
        self.health["exit_code"] = process.returncode
        if self.drainers:
            try:
                async with asyncio.timeout(remaining(1.0)):
                    await asyncio.gather(*self.drainers)
            except TimeoutError:
                for task in self.drainers:
                    task.cancel()
                await asyncio.gather(*self.drainers, return_exceptions=True)
                self.health["log_error"] = "log drain timeout"
        if time.monotonic() > deadline:
            self.health["budget_overrun_seconds"] = time.monotonic() - deadline
        self.stopped = True
        return self.health

    async def _stop_strict(self, remaining_seconds: float | None) -> Json:
        """Original deadline only; uncertain owners remain reachable for containment."""
        if self.stopped:
            return self.health
        now = time.monotonic()
        deadline = min(self.cleanup_deadline,
            now + remaining_seconds if remaining_seconds is not None else self.cleanup_deadline)

        def remaining(cap: float = float("inf")) -> float:
            return min(cap, max(0., deadline - time.monotonic()))

        try:
            if self.process is None and self._creation is not None:
                if self._creation.done():
                    if self._creation.cancelled():
                        self.health["reap_failure"] = "original server creation cancellation is unconfirmed"
                        return self.health
                    self.process = self._creation.result()
                elif remaining() > 0:
                    try:
                        async with asyncio.timeout(remaining()):
                            self.process = await asyncio.shield(self._creation)
                    except TimeoutError:
                        self.health["reap_failure"] = "original server creation remains pending"
                        return self.health
                else:
                    self.health["reap_failure"] = "original server creation remains pending"
                    return self.health
            process = self.process
            if process is None:
                self.stopped = True
                return self.health
            if process.stdin:
                process.stdin.close()
            if process.returncode is None and remaining() > 0:
                try:
                    async with asyncio.timeout(remaining(self.shutdown_seconds)):
                        await process.wait()
                    self.health["shutdown"] = "stdin_eof"
                except TimeoutError:
                    pass
            elif process.returncode is not None and self.health["shutdown"] is None:
                self.health["shutdown"] = "stdin_eof"
            if process.returncode is None:
                self.health["shutdown"] = "terminate"
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                if remaining() > 0:
                    try:
                        async with asyncio.timeout(remaining(1.)):
                            await process.wait()
                    except TimeoutError:
                        pass
            if process.returncode is None:
                self.health["shutdown"] = "kill"
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                if remaining() > 0:
                    try:
                        async with asyncio.timeout(remaining()):
                            await process.wait()
                    except TimeoutError:
                        pass
            self.health["exit_code"] = process.returncode
            if process.returncode is None:
                self.health["reap_failure"] = "original server reap unconfirmed at strict cutoff"
            unfinished = [task for task in self.drainers if not task.done()]
            if unfinished and remaining() > 0:
                try:
                    async with asyncio.timeout(remaining(1.)):
                        await asyncio.gather(*unfinished)
                except TimeoutError:
                    self.health["log_error"] = self.health["log_error"] or "original log drain timed out"
                except BaseException:
                    self.health["log_error"] = self.health["log_error"] or "original log drain cancelled or failed"
                    raise
            for task in self.drainers:
                if task.done() and (task.cancelled() or task.exception() is not None):
                    self.health["log_error"] = self.health["log_error"] or "original log drain cancelled or failed"
            unfinished = [task for task in self.drainers if not task.done()]
            if unfinished:
                for task in unfinished:
                    task.cancel()
                self.health["log_error"] = "original log drain unconfirmed at strict cutoff"
            self.stopped = process.returncode is not None and not unfinished
            if time.monotonic() > deadline:
                self.health["budget_overrun_seconds"] = time.monotonic() - deadline
            return self.health
        except BaseException as error:
            process = self.process
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            for task in self.drainers:
                if not task.done():
                    task.cancel()
            vars(error)["cleanup_owner"] = self
            raise
