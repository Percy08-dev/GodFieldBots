"""Run-owned subprocess lifecycle, effective TestServer config and bounded cleanup."""
from __future__ import annotations

import asyncio
import json
import math
import time
import tomllib
from functools import partial
from pathlib import Path

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_simulator_client.http import EFFECTIVE_KEYS, seed_fingerprint, validate_info
from gf_simulator_client.http_diagnostics import HttpPairCapture


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
                 strict_cleanup_deadline: bool = False,
                 http_capture: HttpPairCapture | None = None) -> None:
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
        self.http_capture = http_capture
        self._http_settlement: asyncio.Task[Json] | None = None
        self._http_early_exit = False
        self._shutdown_started = False
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

    def _bind_process(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        assert process.stdout is not None and process.stderr is not None
        self.drainers = [asyncio.create_task(self._drain(process.stdout, self.directory / "stdout.log", True)),
                         asyncio.create_task(self._drain(process.stderr, self.directory / "stderr.log", False))]

    def _close_stdin(self, process: asyncio.subprocess.Process) -> None:
        if not self._shutdown_started:
            self._http_early_exit = process.returncode is not None
            self._shutdown_started = True
        if process.stdin:
            process.stdin.close()

    async def start(self) -> Json:
        self.directory.mkdir(parents=True, exist_ok=False)
        self.startup = asyncio.get_running_loop().create_future()
        try:
            http_directory = (self.directory / "http" if self.http_capture is None
                              else self.http_capture.prepare())
            creation = asyncio.create_subprocess_exec(
                self.binary, "--config", self.config, "--seed", self.seed,
                "--seed-use-limit", str(self.count), "--server-idle-timeout-secs", "0",
                "--data-dir", str(self.directory / "simulator-data"),
                "--http-log-dir", str(http_directory), "--shutdown-on-stdin-eof",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=1024 * 1024,
                env=_simulator_environment(),
            )
            if self.strict_cleanup_deadline or self.http_capture is not None:
                self._creation = asyncio.create_task(creation)
                if self.strict_cleanup_deadline:
                    async with asyncio.timeout(max(0., self.cleanup_deadline - time.monotonic())):
                        process = await asyncio.shield(self._creation)
                else:
                    process = await asyncio.shield(self._creation)
            else:
                process = await creation
            self._bind_process(process)
            startup_seconds = (min(self.startup_seconds, max(0., self.cleanup_deadline - time.monotonic()))
                if self.strict_cleanup_deadline else self.startup_seconds)
            async with asyncio.timeout(startup_seconds):
                startup = await self.startup
            closed(startup, {"base_url", "http_log_path", "data_dir", "test_server_config", "test_server_info"})
            if self.http_capture is not None and startup["http_log_path"] != str(http_directory / "http.jsonl"):
                raise ContractError("startup HTTP capture differs from its private pair path")
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
            if self.strict_cleanup_deadline or self.http_capture is not None:
                vars(error)["cleanup_owner"] = self
                try:
                    await self.stop()
                except BaseException as cleanup:  # noqa: BLE001 -- preserve initiating startup failure and real cleanup owner
                    error.add_note("original server cleanup remains owned: " + type(cleanup).__name__)
            else:
                await self.stop()
            raise

    async def stop(self, remaining_seconds: float | None = None) -> Json:
        if self.strict_cleanup_deadline:
            return await self._stop_strict(remaining_seconds)
        if self.stopped:
            return self.health
        cleanup_started = time.monotonic()
        deadline = min(self.cleanup_deadline, cleanup_started + remaining_seconds if remaining_seconds is not None else float("inf"))

        def remaining(cap: float) -> float:
            return min(cap, max(0.0, deadline - time.monotonic()))

        if self.process is None and self._creation is not None:
            if self._creation.cancelled():
                self.health["reap_failure"] = "server creation cancellation unconfirmed"
                return self.health
            if not self._creation.done():
                try:
                    async with asyncio.timeout(remaining(self.startup_seconds)):
                        self._bind_process(await asyncio.shield(self._creation))
                except TimeoutError:
                    self.health["reap_failure"] = "server creation remains pending"
                    return self.health
            elif self._creation.exception() is None:
                self._bind_process(self._creation.result())
        process = self.process
        if process is None:
            self.stopped = True
            return self.health
        self._close_stdin(process)
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
                    if self._creation.exception() is None:
                        self._bind_process(self._creation.result())
                elif remaining() > 0:
                    try:
                        async with asyncio.timeout(remaining()):
                            self._bind_process(await asyncio.shield(self._creation))
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
            self._close_stdin(process)
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

    async def settle_http(self, *, deadline: float, abnormal: bool,
                          responses: int = 0, error_responses: int = 0,
                          transport_errors: int = 0) -> Json:
        capture = self.http_capture
        assert capture is not None
        if self._http_settlement is None:
            stopped = (self.stopped and (self.process is None or self.process.returncode is not None)
                       and all(task.done() and not task.cancelled() and task.exception() is None
                               for task in self.drainers))
            settle = partial(capture.settle, stopped=stopped, deadline=deadline, responses=responses,
                error_responses=error_responses, transport_errors=transport_errors,
                abnormal=(abnormal or self._http_early_exit or self.health["exit_code"] != 0
                          or self.health["shutdown"] != "stdin_eof" or bool(self.health["log_error"])))
            if not stopped or time.monotonic() >= deadline:
                return settle()
            self._http_settlement = asyncio.create_task(asyncio.to_thread(settle))
        settlement = self._http_settlement
        if settlement.done():
            return settlement.result()
        try:
            async with asyncio.timeout_at(deadline):
                return await asyncio.shield(settlement)
        except TimeoutError:
            if settlement.done():
                return settlement.result()
            return {"status": "settlement_unconfirmed",
                    "residual": str(capture.directory or capture.ram_root)}


def _simulator_environment() -> dict[str, str]:
    import os

    from gf_simulator_client.performance_diagnostics import external_environment
    return external_environment(os.environ)
