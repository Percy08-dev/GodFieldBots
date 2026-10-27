"""Run-owned subprocess lifecycle, effective TestServer config and bounded cleanup."""
from __future__ import annotations

import asyncio
import json
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
                 cleanup_deadline: float = float("inf")) -> None:
        self.binary = str(Path(binary).resolve())
        self.config = str(Path(config).resolve())
        self.seed = seed
        self.count = count
        self.directory = directory.resolve()
        self.startup_seconds = startup_seconds
        self.shutdown_seconds = shutdown_seconds
        self.cleanup_deadline = cleanup_deadline
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
            self.process = await asyncio.create_subprocess_exec(
                self.binary, "--config", self.config, "--seed", self.seed,
                "--seed-use-limit", str(self.count), "--server-idle-timeout-secs", "0",
                "--data-dir", str(self.directory / "simulator-data"),
                "--http-log-dir", str(self.directory / "http"), "--shutdown-on-stdin-eof",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=1024 * 1024,
            )
            assert self.process.stdout and self.process.stderr
            self.drainers = [asyncio.create_task(self._drain(self.process.stdout, self.directory / "stdout.log", True)),
                             asyncio.create_task(self._drain(self.process.stderr, self.directory / "stderr.log", False))]
            async with asyncio.timeout(self.startup_seconds):
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
        except BaseException:
            await self.stop()
            raise

    async def stop(self, remaining_seconds: float | None = None) -> Json:
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
