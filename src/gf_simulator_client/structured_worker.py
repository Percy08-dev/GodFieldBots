"""Protocol2 own-participant process with parent-validated frames and reservations."""
from __future__ import annotations

import asyncio
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from gf_common.diverse_actions import GeneratedCandidates
from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured import ActionConfigV2, DecisionFrame, EventBank, EventDelta, Selection
from gf_common.structured_wire import delta_dict, frame_dict, parse_generated, parse_selection
from gf_models.structured_binding import (
    check_input_binding,
    input_binding,
    policy_label,
    safe_options_v2,
)
from gf_models.structured_config import InferenceBindingV2, model_config_v2
from gf_models.structured_runtime import own_worker_settings
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.workers import ParticipantWorker, factory_descriptor


class StructuredParticipantWorker(ParticipantWorker):
    def __init__(self, factory: Callable[..., Any], model_id: str, sampling_seed: int,
                 actions: ActionConfigV2, timeout: float = 10, *, ledger: ReservationLedger,
                 policy_options: Json | None = None) -> None:
        descriptor = factory_descriptor(factory)
        integer(sampling_seed)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("positive structured worker timeout required")
        options = safe_options_v2(policy_options, action_contract=actions.action_contract, purpose="component", synthetic_capacity=False) if policy_options is not None else None
        self.caps, self.costs, residency, self.temporary_bytes = own_worker_settings(model_id, actions, options)
        if options is None:
            self.expected_policy = model_id + "-structured-v1"
        else:
            self.expected_policy = policy_label(model_config_v2(options["model"]), InferenceBindingV2(**options["inference"]),
                                                options["expected_manifest_digest"], cuda_visible_devices=options["cuda_visible_devices"])
        self.initial = {"protocol_version": 2, "factory": descriptor, "model_id": model_id,
                        "sampling_seed": sampling_seed, "actions": asdict(actions), "policy_options": options,
                        "caps": asdict(self.caps), "numeric_costs": asdict(self.costs),
                        **input_binding(actions.action_contract, "component", False)}
        self.model_id = model_id
        self.actions = actions
        search_seed = int(digest({"recipe": "structured-search-stream-v1", "sampling_seed": sampling_seed})[:16], 16)
        self._search_state = random.Random(search_seed).getstate()
        self.timeout, self.closed, self.initialized = timeout, False, False
        self.health: Json = {}
        self.io_thread: threading.Thread | None = None
        self._close_task: asyncio.Task[None] | None = None
        self.inference_elapsed_ms: float | None = None
        self.startup_elapsed_ms: float | None = None
        self.last_timings: Json = {}
        self._parent_bank: EventBank | None = None
        self._last_frame: DecisionFrame | None = None
        self.lease = ledger.reserve_worker(residency)  # before Popen/initialize, including preflight.
        self._spawn_started = time.monotonic()
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            self.process = subprocess.Popen([sys.executable, "-E", "-m", "gf_simulator_client.structured_bootstrap"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, bufsize=0)
        except BaseException:
            self.lease.cancel_before_spawn()
            raise
        self.lease.spawned(self.process)

    async def _roundtrip(self, payload: Json) -> Json:
        if self.closed:
            raise RuntimeError("structured worker closed")
        try:
            assert self.process.stdin is not None and self.process.stdout is not None
            source, destination = self.process.stdout, self.process.stdin
            messages: queue.SimpleQueue[Json | BaseException] = queue.SimpleQueue()
            def exchange() -> None:
                try:
                    write_frame(destination, payload)
                    messages.put(read_frame(source))
                except BaseException as exc:  # noqa: BLE001 - owned thread reports exact failure to parent
                    messages.put(exc)
            self.io_thread = threading.Thread(target=exchange, daemon=True)
            self.io_thread.start()
            async with asyncio.timeout(self.timeout):
                while True:
                    try:
                        received = messages.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(.01)
            if isinstance(received, BaseException):
                raise received
            if "worker_error" in received:
                raise RuntimeError(f"{received['worker_error']}: {received.get('message', '')}")
            return received
        except BaseException as exc:  # noqa: BLE001 - thread setup/exchange failures preserve error after owned reap
            await self._abort(exc)

    async def initialize(self) -> None:
        if self.initialized:
            return
        try:
            received = await self._roundtrip(self.initial)
            closed(received, {"protocol_version", "kind", "policy_version", "model_load_ms", "action_contract", "purpose", "synthetic_capacity"})
            check_input_binding({key: received[key] for key in ("action_contract", "purpose", "synthetic_capacity")},
                                action_contract=self.actions.action_contract, purpose="component", synthetic_capacity=False)
            if received["protocol_version"] != 2 or type(received["protocol_version"]) is not int or received["kind"] != "ready" or received["policy_version"] != self.expected_policy:
                raise ContractError("malformed structured initialize/ready identity")
            _timings({"model_load_ms": received["model_load_ms"]}, ready=True)
            self.startup_elapsed_ms = (time.monotonic() - self._spawn_started) * 1000
            self.initialized = True
        except BaseException as exc:  # noqa: BLE001 - every startup failure owns child reap
            await self._abort(exc)

    async def select_frame(self, frame: DecisionFrame, delta: EventDelta) -> tuple[GeneratedCandidates, Selection]:
        await self.initialize()  # deliberately outside decision timing.
        started = time.monotonic()
        try:
            integer(frame.boundary.end, 0, self.caps.max_episode_events)
            integer(frame.boundary.end - frame.interval_start, 0, self.caps.max_interval_events)
            if self._last_frame is not None and (frame.interval_start != self._last_frame.boundary.end or frame.revision <= self._last_frame.revision):
                raise ContractError("parent own-decision interval/revision continuity mismatch")
            if delta.actor != frame.actor or delta.episode != frame.episode or delta.start != frame.interval_start:
                raise ContractError("parent frame/delta actor/interval mismatch")
            if self._parent_bank is None:
                if frame.interval_start != 0:
                    raise ContractError("first parent participant interval must start at zero")
                self._parent_bank = EventBank(frame.actor, frame.episode, self.caps.max_episode_events)
            self._parent_bank.append(delta)
            if self._parent_bank.boundary != frame.boundary:
                raise ContractError("parent delta must end at exact original frame boundary")
            frame.bank_view(self._parent_bank.view())
            scope = self.lease.reserve_decision(self.temporary_bytes)
            received = await self._roundtrip({"protocol_version": 2, "kind": "select", "frame": frame_dict(frame), "delta": delta_dict(delta),
                                                **input_binding(self.actions.action_contract, "component", False)})
            closed(received, {"protocol_version", "kind", "generated", "selection", "timings", "action_contract", "purpose", "synthetic_capacity"})
            check_input_binding({key: received[key] for key in ("action_contract", "purpose", "synthetic_capacity")},
                                action_contract=self.actions.action_contract, purpose="component", synthetic_capacity=False)
            if type(received["protocol_version"]) is not int or received["protocol_version"] != 2 or received["kind"] != "selection":
                raise ContractError("malformed protocol2 selection response")
            generated, selected = parse_generated(received["generated"], action_contract=self.actions.action_contract), parse_selection(received["selection"])
            generated.table.validate(frame)
            if (generated.report.config_hash != digest(asdict(self.actions))
                    or generated.report.numeric_costs_hash != digest(asdict(self.costs))
                    or generated.report.rng_before != self._search_state
                    or generated.report.attempts > self.actions.max_expansions
                    or generated.report.discovered > self.actions.max_temporary_candidates):
                raise ContractError("worker search report/config/cost/RNG binding mismatch")
            if self.model_id in ("mlp", "gru", "transformer") and (selected.logprob is None or selected.value is None):
                raise ContractError("neural policy reply lacks finite logprob/value")
            if selected.policy_version != self.expected_policy:
                raise ContractError("worker policy identity does not match owned execution binding")
            selected.validate(generated.table, frame)  # original parent frame, never worker-supplied reconstruction.
            if len(generated.table.kinds) > self.caps.max_candidates or any(b - a > self.caps.max_positions for a, b in zip(generated.table.offsets[:-1], generated.table.offsets[1:], strict=True)):
                raise ContractError("worker candidate table exceeds configured presentation/capacity")
            _timings(received["timings"])
            self.last_timings = {**received["timings"], "parent_request_ms": (time.monotonic() - started) * 1000}
            self.last_timings["communication_decode_host_ms"] = max(0., self.last_timings["parent_request_ms"] - self.last_timings["worker_decision_ms"])
            self.inference_elapsed_ms = self.last_timings["policy_ms"]
            self._last_frame = frame
            self._search_state = generated.report.rng_after
            self.lease.reply(scope)
            return generated, selected
        except BaseException as exc:  # noqa: BLE001 - no temporary release while a failed child is still alive
            await self._abort(exc)

    async def _close_owned(self, timeout: float, started: float) -> None:
        try:
            # A constructor/start failure can leave an unstarted Thread object.
            # Never join it; the original setup error survives real child reap.
            if self.io_thread is not None and self.io_thread.ident is None:
                self.io_thread = None
            await super()._close_owned(timeout, started)
        finally:
            if self.process.poll() is not None:
                self.lease.reaped(self.process)


def _timings(values: Json, *, ready: bool = False) -> None:
    required = {"model_load_ms"} if ready else {"generation_ms", "policy_ms", "worker_decision_ms"}
    optional = set() if ready else {"tokenization_ms", "input_pack_transfer_host_ms", "forward_enqueue_host_ms",
                                    "sync_host_ms", "return_ms", "sampling_ms"}
    closed(values, required, optional)
    if any(type(value) not in (int, float) or value < 0 or value > sys.float_info.max or not math.isfinite(value) for value in values.values()):
        raise ContractError("invalid structured host timing metrics")
