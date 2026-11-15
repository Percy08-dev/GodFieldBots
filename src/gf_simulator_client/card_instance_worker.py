"""Protocol3 baseline participant; shared exchange/reap and reservation owner."""

from __future__ import annotations

import asyncio
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, cast

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import check_game_binding, game_binding
from gf_common.diverse_actions import GameGeneratedCandidatesV3
from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
from gf_common.structured import DecisionFrame, EventBank, EventDelta, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_wire import (
    delta_dict,
    frame_dict,
    parse_game_generated_v3,
    parse_selection,
)
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.card_instance_game_tree_runtime import (
    RetainedGamePolicyArtifacts,
    _configured,
    policy_label_tree_game_v3,
    safe_tree_options_game_v3,
)
from gf_models.structured_binding import (
    policy_label_game_v3,
    policy_label_game_v4,
    safe_options_game_v3,
    safe_options_game_v4,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV3,
    ModelConfigV4,
    model_config_v3,
)
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import (
    StructuredParticipantWorker,
    _failed_spawn,
    _timings,
)


class CardInstanceParticipantWorker(StructuredParticipantWorker):
    actions: ActionConfigV3  # type: ignore[assignment]

    def __init__(
        self,
        model_id: str,
        sampling_seed: int,
        actions: ActionConfigV3,
        timeout: float = 10,
        *,
        ledger: ReservationLedger,
        policy_options: Json | None = None,
        artifact_inputs: RetainedGameArtifacts | RetainedGamePolicyArtifacts | None = None,
        deadline: float | None = None,
    ) -> None:
        self.rules = card_instance_rules()
        self.binding = game_binding(
            actions.action_contract, rule_signature(actions.action_contract, self.rules)
        )
        self._tree = model_id == "lightgbm"
        v4 = type(policy_options) is dict and policy_options.get("version") == "4"
        parser = safe_tree_options_game_v3 if self._tree else safe_options_game_v4 if v4 else safe_options_game_v3
        refresh = model_id == "heuristic_refresh"
        options = (asdict(parse_refresh_options(policy_options if policy_options is not None else {})) if refresh else
            parser(policy_options, actions=actions, rules=self.rules) if policy_options is not None else None)
        if self._tree and (deadline is None or type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic()):
            raise ContractError("Tree3 worker requires the original finite live session deadline")
        self._session_deadline = deadline
        self._operation_deadline = deadline
        if self._tree:
            assert deadline is not None
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ContractError("positive protocol3 timeout required")
            self._construction_deadline = min(deadline, time.monotonic()+min(10., float(timeout)))
        self._inflight: asyncio.Task[Any] | None = None
        self._tree_failed = False
        self.caps, self.costs, residency, self.temporary_bytes = own_game_worker_settings_v3(
            model_id, actions, options, rules=self.rules)
        self.artifact_inputs = artifact_inputs
        self.neural_metadata: Json | None = None
        self.tree_metadata: Json | None = None
        self.baseline_options: Json | None = options if refresh else None
        if refresh:
            if artifact_inputs is not None:
                raise ContractError("configured baseline forbids artifact inputs")
            self.expected_policy = refresh_policy_label(actions, self.rules, parse_refresh_options(options if options is not None else {}))
        elif options is None:
            if artifact_inputs is not None:
                raise ContractError("baseline worker forbids neural artifact inputs")
            self.expected_policy = model_id + "-card-instance-v2"
        elif self._tree:
            if type(artifact_inputs) is not RetainedGamePolicyArtifacts:
                raise ContractError("Tree3 worker requires its retained actual GAME pair owner")
            assert options is not None and deadline is not None
            artifact_inputs.require_options(model_id, options)
            artifact_inputs.guard(deadline=self._construction_deadline)
            self.expected_policy = policy_label_tree_game_v3(options, actions=actions, rules=self.rules)
            self.tree_metadata = artifact_inputs.journal_metadata(self.expected_policy)
        else:
            if type(artifact_inputs) not in (RetainedGameArtifacts, RetainedGamePolicyArtifacts):
                raise ContractError("neural worker requires its retained original artifact owner")
            assert artifact_inputs is not None
            artifact_inputs.require_options(model_id, options)
            from gf_models.hpo_config import model_config_v4
            model_parser = model_config_v4 if v4 else model_config_v3
            model = model_parser(options["model"], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            label_options = {"action_contract": actions.action_contract, "rules": self.rules,
                "action_config_hash": digest(asdict(actions)), "cuda_visible_devices": options["cuda_visible_devices"]}
            if type(model) is ModelConfigV4:
                self.expected_policy = policy_label_game_v4(model, InferenceBindingV2(**options["inference"]),
                    options["expected_manifest_digest"], **label_options)
            else:
                assert type(model) is ModelConfigV3
                self.expected_policy = policy_label_game_v3(model, InferenceBindingV2(**options["inference"]),
                    options["expected_manifest_digest"], **label_options)
            self.neural_metadata = artifact_inputs.journal_metadata(self.expected_policy)
        integer(sampling_seed)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("positive protocol3 timeout required")
        self._tree_timeout = min(10., float(timeout))
        self.model_id, self.actions = model_id, actions
        self.initial: Json = {
            "protocol_version": 3,
            "model_id": model_id,
            "sampling_seed": sampling_seed,
            "actions": asdict(actions),
            "rules": asdict(self.rules),
            "caps": asdict(self.caps),
            "numeric_costs": asdict(self.costs),
            **self.binding,
        }
        if options is not None:
            self.initial["policy_options"] = options
        self._search_state = random.Random(
            int(
                digest({"recipe": "structured-search-stream-v1", "sampling_seed": sampling_seed})[
                    :16
                ],
                16,
            )
        ).getstate()
        self.timeout, self.closed, self.initialized = timeout, False, False
        self.health = {}
        self.io_thread = self._close_task = None
        self.inference_elapsed_ms = self.startup_elapsed_ms = None
        self.last_timings = {}
        self._parent_bank = None
        self._last_frame = None
        self._spawn_started = time.monotonic()
        self.process = cast("subprocess.Popen[bytes]", None)
        if self._tree:
            assert options is not None
            from gf_common.domain import canonical
            from gf_models.structured_tree_config import resource_estimate
            from gf_models.structured_tree_encoder import _owned
            from gf_simulator_client.framing import MAX_FRAME_BYTES, frame_decode_bytes
            self._tree_model = _configured(options)
            assert type(artifact_inputs) is RetainedGamePolicyArtifacts
            self._tree_retained = (artifact_inputs.retained_bytes + artifact_inputs.guard_scratch_bytes
                + _owned((self.initial, self.tree_metadata)))
            self._tree_reply_limit = min(MAX_FRAME_BYTES, 65536+actions.max_candidates*(512+18*8))
            resource_estimate(self._tree_model, model_bytes=options["tree_descriptor"]["model_bytes"],
                extra_bytes=self._tree_retained+256*_owned(self.initial)+65536)
            encoded_size = len(canonical(self.initial).encode())
            resource_estimate(self._tree_model, model_bytes=options["tree_descriptor"]["model_bytes"],
                extra_bytes=self._tree_retained+frame_decode_bytes(encoded_size)+frame_decode_bytes(self._tree_reply_limit))
            artifact_inputs.guard(deadline=self._construction_deadline)
        self._spawn(ledger, residency,
            [sys.executable, "-E", "-m", "gf_simulator_client.card_instance_bootstrap"]
            + ([str(options["resources"]["max_owned_bytes"])] if self._tree and options is not None else []),
            {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    def _spawn(self, ledger: ReservationLedger, residency: int, arguments: list[str], environment: dict[str, str]) -> None:
        """Original audited creation boundary; resident logical validators borrow transport."""
        boundary = self._creation_boundary = threading.Event()
        self._creation_observer_failure: BaseException | None = None
        self.lease = ledger.reserve_worker(residency)  # must precede Popen.
        self.lease._cleanup_owner = self
        if type(self.artifact_inputs) is RetainedGamePolicyArtifacts:
            self.artifact_inputs.retain_worker(self)
        process = None
        try:
            try:
                self.lease._creation_started(self, boundary)
            except BaseException as error:
                self._creation_observer_failure = error
                raise
            creation_error = observer_error = None
            try:
                process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0, env=environment)
                self.process = process
            except BaseException as error:
                creation_error = error
                raise
            finally:
                try:
                    self.lease._creation_finished(self, boundary, process, creation_error)
                except BaseException as error:  # noqa: BLE001 - keep local child for original cleanup
                    self._creation_observer_failure = error
                    if creation_error is not None:
                        creation_error.add_note('worker creation observation: ' + type(error).__name__)
                    else:
                        observer_error = error
                finally:
                    boundary.set()
            if observer_error is not None:
                raise observer_error
            self.lease.spawned(process)
        except BaseException as error:
            boundary.set()
            _failed_spawn(process, self.lease, error)
            if not self.lease.released:
                error._structured_startup_owner = self  # type: ignore[attr-defined]
            raise

    def _check(self, received: Json) -> None:
        check_game_binding(
            {key: received[key] for key in self.binding},
            self.actions.action_contract,
            self.binding["rule_signature"],
        )
        if type(received["protocol_version"]) is not int or received["protocol_version"] != 3:
            raise ContractError("protocol3 response required")

    async def _initialize_owned(self) -> None:
        if self.initialized:
            return
        try:
            received = await self._roundtrip(self.initial)
            closed(
                received,
                {"protocol_version", "kind", "policy_version", "model_load_ms"} | set(self.binding),
            )
            self._check(received)
            if received["kind"] != "ready" or received["policy_version"] != self.expected_policy:
                raise ContractError("protocol3 ready policy identity mismatch")
            _timings({"model_load_ms": received["model_load_ms"]}, ready=True)
            self.startup_elapsed_ms = (time.monotonic() - self._spawn_started) * 1000
            self.initialized = True
        except BaseException as error:  # noqa: BLE001 - reap before preserving original failure.
            await self._abort(error)

    async def _select_frame_owned(
        self, frame: DecisionFrame, delta: EventDelta
    ) -> tuple[GameGeneratedCandidatesV3, Selection]:
        await self._initialize_owned()
        started = time.monotonic()
        try:
            layout = CORRECTED_CARD_INSTANCE_LAYOUT
            if frame.layout is not layout or delta.layout is not layout:
                raise ContractError("protocol3 frame/delta profile mismatch")
            integer(frame.boundary.end, 0, self.caps.max_episode_events)
            integer(frame.boundary.end - frame.interval_start, 0, self.caps.max_interval_events)
            if self._last_frame is not None and (
                frame.interval_start != self._last_frame.boundary.end
                or frame.revision <= self._last_frame.revision
            ):
                raise ContractError("protocol3 original frame interval/revision mismatch")
            if (
                delta.actor != frame.actor
                or delta.episode != frame.episode
                or delta.start != frame.interval_start
            ):
                raise ContractError("protocol3 original frame/delta identity mismatch")
            if self._tree:
                from gf_models.structured_tree_config import resource_estimate
                from gf_models.structured_tree_encoder import _owned
                event_bytes = sum(len(value) for value in delta._events)
                if self._parent_bank is not None:
                    event_bytes += sum(len(value) for value in self._parent_bank.view()._events)
                resource_estimate(self._tree_model, model_bytes=self.initial["policy_options"]["tree_descriptor"]["model_bytes"],
                    extra_bytes=self._tree_retained+4*_owned((frame, delta, None if self._parent_bank is None else vars(self._parent_bank), self._last_frame))
                    +64*len(frame.payload)+32*event_bytes+65536)
            if self._parent_bank is None:
                if frame.interval_start:
                    raise ContractError("protocol3 first event prefix must start at zero")
                self._parent_bank = EventBank(
                    frame.actor, frame.episode, self.caps.max_episode_events, layout=layout
                )
            self._parent_bank.append(delta)
            if self._parent_bank.boundary != frame.boundary:
                raise ContractError("protocol3 event delta/frame boundary mismatch")
            frame.bank_view(self._parent_bank.view())
            scope = self.lease.reserve_decision(self.temporary_bytes)
            received = await self._roundtrip(
                {
                    "protocol_version": 3,
                    "kind": "select",
                    "frame": frame_dict(frame, layout=layout),
                    "delta": delta_dict(delta, layout=layout),
                    **self.binding,
                }
            )
            closed(
                received,
                {"protocol_version", "kind", "generated", "selection", "timings"}
                | set(self.binding),
            )
            self._check(received)
            if received["kind"] != "selection":
                raise ContractError("protocol3 selection required")
            generated = parse_game_generated_v3(
                received["generated"],
                action_contract=self.actions.action_contract,
                rules=self.rules,
            )
            selected = parse_selection(received["selection"])
            if (
                generated.report.config_hash != digest(asdict(self.actions))
                or generated.report.numeric_costs_hash != digest(asdict(self.costs))
                or generated.report.rng_before != self._search_state
                or generated.report.attempts > self.actions.max_expansions
                or generated.report.discovered > self.actions.max_temporary_candidates
                or selected.policy_version != self.expected_policy
                or self.neural_metadata is None and (selected.logprob is not None or selected.value is not None)
                or self.neural_metadata is not None and any(type(value) not in (int, float)
                    or value is None or not math.isfinite(value) for value in (selected.logprob, selected.value))
            ):
                raise ContractError("protocol3 search/baseline binding mismatch")
            if len(generated.table.kinds) > self.caps.max_candidates or any(
                b - a > self.caps.max_positions
                for a, b in zip(
                    generated.table.offsets[:-1], generated.table.offsets[1:], strict=True
                )
            ):
                raise ContractError("protocol3 candidate capacity exceeded")
            selected.validate(generated.table, frame)  # only the original parent frame.
            _timings(received["timings"])
            self.last_timings = {
                **received["timings"],
                "parent_request_ms": (time.monotonic() - started) * 1000,
            }
            self.last_timings["communication_decode_host_ms"] = max(
                0.0,
                self.last_timings["parent_request_ms"] - self.last_timings["worker_decision_ms"],
            )
            self.inference_elapsed_ms = self.last_timings["policy_ms"]
            self._last_frame, self._search_state = frame, generated.report.rng_after
            self.lease.reply(scope)
            return generated, selected
        except BaseException as error:  # noqa: BLE001 - reap before preserving original failure.
            await self._abort(error)

    def _admit_tree_operation(self) -> None:
        if self._inflight is not None:
            raise ContractError("one in-flight Tree3 participant operation required")
        if self.closed or self._tree_failed:
            raise ContractError("Tree3 participant is cleanup-only")
        assert self._session_deadline is not None
        self._operation_deadline = min(self._session_deadline, time.monotonic()+self._tree_timeout)
        self._inflight = asyncio.current_task()

    def _tree_guard(self) -> None:
        if not self._tree:
            return
        assert self._operation_deadline is not None
        if self.closed or time.monotonic() >= self._operation_deadline:
            raise TimeoutError("Tree3 original operation/session deadline exhausted or closed")
        assert type(self.artifact_inputs) is RetainedGamePolicyArtifacts
        self.artifact_inputs.guard(deadline=self._operation_deadline)

    async def initialize(self) -> None:
        if not self._tree:
            await self._initialize_owned()
            return
        self._admit_tree_operation()
        try:
            self._tree_guard()
            await self._initialize_owned()
            self._tree_guard()
        except BaseException as error:  # noqa: BLE001 - preserve cancellation after owned retirement.
            self._tree_failed = True
            await self._abort(error)
        finally:
            self._inflight = None

    async def select_frame(self, frame: DecisionFrame, delta: EventDelta) -> tuple[GameGeneratedCandidatesV3, Selection]:  # type: ignore[override]
        if not self._tree:
            return await self._select_frame_owned(frame, delta)
        self._admit_tree_operation()
        try:
            self._tree_guard()
            result = await self._select_frame_owned(frame, delta)
            self._tree_guard()
            return result
        except BaseException as error:  # noqa: BLE001 - preserve cancellation after owned retirement.
            self._tree_failed = True
            await self._abort(error)
        finally:
            self._inflight = None

    def _read_reply(self, source: Any) -> Json:
        if not self._tree:
            return super()._read_reply(source)
        from gf_models.structured_tree_config import resource_estimate
        from gf_models.structured_tree_encoder import _owned
        from gf_simulator_client.framing import read_frame_admitted
        retained = resource_estimate(self._tree_model,
            model_bytes=self.initial["policy_options"]["tree_descriptor"]["model_bytes"],
            extra_bytes=self._tree_retained+_owned((None if self._parent_bank is None else vars(self._parent_bank), self._last_frame)))["total_bytes"]
        return read_frame_admitted(source, max_owned_bytes=self._tree_model.resources.max_owned_bytes,
            retained_bytes=retained, max_frame_bytes=self._tree_reply_limit)

    async def _roundtrip(self, payload: Json) -> Json:
        if not self._tree:
            return await super()._roundtrip(payload)
        from gf_common.domain import canonical
        from gf_models.structured_tree_config import resource_estimate
        from gf_models.structured_tree_encoder import _owned
        from gf_simulator_client.framing import frame_decode_bytes
        self._tree_guard()
        assert self._operation_deadline is not None
        self.timeout = min(self._tree_timeout, self._operation_deadline-time.monotonic())
        resource_estimate(self._tree_model,
            model_bytes=self.initial["policy_options"]["tree_descriptor"]["model_bytes"],
            extra_bytes=self._tree_retained+256*_owned(payload)+65536)
        resource_estimate(self._tree_model,
            model_bytes=self.initial["policy_options"]["tree_descriptor"]["model_bytes"],
            extra_bytes=self._tree_retained+_owned((None if self._parent_bank is None else vars(self._parent_bank), self._last_frame, payload))
            +frame_decode_bytes(len(canonical(payload).encode()))+frame_decode_bytes(self._tree_reply_limit))
        received = await super()._roundtrip(payload)
        self._tree_guard()
        return received

    async def close(self, timeout: float = 2.) -> None:
        if self._tree:
            assert self._session_deadline is not None
            timeout = min(timeout, max(0., self._session_deadline-time.monotonic()))
        await super().close(timeout)
        if type(self.artifact_inputs) is RetainedGamePolicyArtifacts:
            self.artifact_inputs.retire_worker(self)
        if self._tree and not self._tree_failed:
            assert type(self.artifact_inputs) is RetainedGamePolicyArtifacts
            assert self._session_deadline is not None
            self.artifact_inputs.guard(deadline=self._session_deadline)
