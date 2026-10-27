"""Fresh participant interpreters with clean argv/environment and bounded framing."""
from __future__ import annotations

import asyncio
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict

from gf_common.domain import (
    ActionConfig,
    CandidateSet,
    ContractError,
    DecisionInput,
    Json,
    Policy,
    PolicyDecision,
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
                 actions: ActionConfig, timeout: float = 10) -> None:
        descriptor = factory_descriptor(factory)
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        self.process = subprocess.Popen([sys.executable, "-E", "-m", "gf_simulator_client.worker_bootstrap"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        env=environment, bufsize=0)
        self.initial = {"factory": descriptor, "model_id": model_id, "sampling_seed": sampling_seed,
                        "actions": asdict(actions)}
        self.initialized = False
        self.timeout = timeout
        self.closed = False
        self.health: Json = {}
        self.io_thread: threading.Thread | None = None

    async def select(self, value: DecisionInput) -> tuple[CandidateSet, PolicyDecision]:
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
        except (TimeoutError, asyncio.CancelledError, OSError, EOFError, ValueError):
            await self.close(timeout=0)
            raise
        if "worker_error" in received:
            await self.close(timeout=0)
            raise RuntimeError(f"{received['worker_error']}: {received['message']}")
        try:
            self.initialized = True
            raw = received["candidates"]
            candidates = CandidateSet(tuple(raw.pop("actions")), **raw)
            decision = PolicyDecision(**received["decision"])
            return candidates, decision
        except (KeyError, TypeError, ValueError) as exc:
            await self.close(timeout=0)
            raise ContractError("malformed worker response") from exc

    async def close(self, timeout: float = 2.0) -> None:
        if self.closed:
            return
        self.closed = True
        started = time.monotonic()
        if timeout <= 0:
            self.health["emergency_grace"] = "cleanup budget exhausted"
        if self.process.poll() is None:
            self.process.terminate()
        deadline = time.monotonic() + min(1.0, max(0.0, timeout))
        while self.process.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self.process.poll() is None:
            self.health["emergency_grace"] = "kill/reap allowance used"
            self.process.kill()
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
