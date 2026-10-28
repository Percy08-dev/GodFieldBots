"""Fresh participant interpreters with clean argv/environment and bounded framing."""
from __future__ import annotations

import asyncio
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import NoReturn

from gf_common.domain import (
    ActionConfig,
    CandidateSet,
    ContractError,
    DecisionInput,
    Json,
    Policy,
    PolicyDecision,
    validated_policy_options,
)
from gf_simulator_client.framing import read_frame, write_frame


def factory_descriptor(factory: Callable[[str], Policy]) -> Json:
    module = getattr(factory, "__module__", "")
    name = getattr(factory, "__qualname__", "")
    if module in ("__main__", "__mp_main__") or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) or not name.isidentifier() or getattr(sys.modules.get(module), name, None) is not factory:
        raise ContractError("worker factory must be an importable top-level function")
    return {"module": module, "name": name}


class ParticipantWorker:
    def __init__(self, factory: Callable[[str], Policy], model_id: str, sampling_seed: int,
                 actions: ActionConfig, timeout: float = 10, *, policy_options: Json | None = None) -> None:
        descriptor = factory_descriptor(factory)
        policy_options = validated_policy_options(policy_options) if policy_options is not None else None
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        self.process = subprocess.Popen([sys.executable, "-E", "-m", "gf_simulator_client.worker_bootstrap"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        env=environment, bufsize=0)
        self.initial = {"factory": descriptor, "model_id": model_id, "sampling_seed": sampling_seed,
                        "actions": asdict(actions)}
        if policy_options is not None:
            self.initial["policy_options"] = policy_options
        self.inference_elapsed_ms: float | None = None
        self.initialized = False
        self.timeout = timeout
        self.closed = False
        self.health: Json = {}
        self.io_thread: threading.Thread | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def _exchange(self, value: DecisionInput | None) -> Json:
        if self.closed:
            raise RuntimeError("worker closed")
        messages: queue.SimpleQueue[Json | BaseException] = queue.SimpleQueue()
        assert self.process.stdin is not None and self.process.stdout is not None
        source, destination = self.process.stdout, self.process.stdin

        def exchange() -> None:
            try:
                if not self.initialized:
                    write_frame(destination, self.initial)
                    ready = read_frame(source)
                    if ready != {"ready": True}:
                        messages.put(ready)
                        return
                if value is None:
                    messages.put({"ready": True})
                else:
                    write_frame(destination, value.to_dict())
                    messages.put(read_frame(source))
            except (OSError, EOFError, ValueError) as exc:
                messages.put(exc)

        self.io_thread = threading.Thread(target=exchange, daemon=True)
        self.io_thread.start()
        try:
            async with asyncio.timeout(self.timeout):
                while True:
                    try:
                        received = messages.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.01)
            if isinstance(received, BaseException):
                raise received
        except (TimeoutError, asyncio.CancelledError, OSError, EOFError, ValueError) as exc:
            await self._abort(exc)
        if "worker_error" in received:
            await self._abort(RuntimeError(f"{received['worker_error']}: {received['message']}"))
        self.initialized = True
        return received

    async def initialize(self) -> None:
        """Bounded native load/reset preflight without constructing a decision."""
        if self.initialized:
            return
        if await self._exchange(None) != {"ready": True}:
            await self._abort(ContractError("malformed worker initialization response"))

    async def select(self, value: DecisionInput) -> tuple[CandidateSet, PolicyDecision]:
        received = await self._exchange(value)
        try:
            metric = received.get("inference_elapsed_ms")
            if metric is not None and (type(metric) not in (int, float) or not math.isfinite(metric) or metric < 0):
                raise ContractError("invalid inference timing")
            self.inference_elapsed_ms = metric
            raw = received["candidates"]
            candidates = CandidateSet(tuple(raw.pop("actions")), **raw)
            decision = PolicyDecision(**received["decision"])
            return candidates, decision
        except (KeyError, TypeError, ValueError) as exc:
            error = ContractError("malformed worker response")
            error.__cause__ = exc
            await self._abort(error)

    async def _abort(self, original: BaseException) -> NoReturn:
        """Preserve the operation error even if its owned cleanup is cancelled."""
        try:
            await self.close(timeout=0)
        except BaseException as cleanup_error:  # noqa: BLE001 - original error survives owned cleanup
            self.health.setdefault("abort_cleanup_error", type(cleanup_error).__name__)
        raise original

    async def close(self, timeout: float = 2.0) -> None:
        # closed forbids new requests; the shared task, not this flag, proves
        # process/pipe/thread cleanup. Later calls never reset its first budget.
        if self._close_task is None:
            self.closed = True
            self._close_task = asyncio.create_task(self._close_owned(timeout, time.monotonic()))
        first_cancel: asyncio.CancelledError | None = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as exc:
                first_cancel = first_cancel or exc
            except BaseException:  # noqa: BLE001 - task.result below preserves the first cancellation
                break
        try:
            self._close_task.result()
        except BaseException as cleanup_error:
            if first_cancel is not None:
                self.health.setdefault("cleanup_failure", str(cleanup_error))
                raise first_cancel from cleanup_error
            raise
        if first_cancel is not None:
            raise first_cancel

    async def _close_owned(self, timeout: float, started: float) -> None:
        if timeout <= 0:
            self.health["emergency_grace"] = "cleanup budget exhausted"
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
        deadline = started + min(1.0, max(0.0, timeout))
        while self.process.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self.process.poll() is None:
            self.health["emergency_grace"] = "kill/reap allowance used"
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
            kill_deadline = time.monotonic() + 0.5
            while self.process.poll() is None and time.monotonic() < kill_deadline:
                await asyncio.sleep(0.01)
        # Unbuffered pipes: closing after process death releases any blocked I/O.
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        if self.io_thread is not None:
            self.io_thread.join(timeout=0.1)
            if self.io_thread.is_alive():
                raise RuntimeError("worker I/O thread did not exit after child termination")
        if self.process.poll() is None:
            raise RuntimeError("worker did not exit after kill")
        if time.monotonic() - started > max(0.0, timeout):
            self.health["budget_overrun_seconds"] = time.monotonic() - started - max(0.0, timeout)
