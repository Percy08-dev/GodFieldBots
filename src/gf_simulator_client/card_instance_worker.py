"""Protocol3 baseline participant; shared exchange/reap and reservation owner."""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from typing import cast

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import check_game_binding, game_binding
from gf_common.diverse_actions import GameGeneratedCandidatesV3
from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured import DecisionFrame, EventBank, EventDelta, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_wire import (
    delta_dict,
    frame_dict,
    parse_game_generated_v3,
    parse_selection,
)
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.structured_binding import policy_label_game_v3, safe_options_game_v3
from gf_models.structured_config import InferenceBindingV2, model_config_v3
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
        artifact_inputs: RetainedGameArtifacts | None = None,
    ) -> None:
        self.rules = card_instance_rules()
        self.binding = game_binding(
            actions.action_contract, rule_signature(actions.action_contract, self.rules)
        )
        options = (safe_options_game_v3(policy_options, actions=actions, rules=self.rules)
            if policy_options is not None else None)
        self.caps, self.costs, residency, self.temporary_bytes = own_game_worker_settings_v3(
            model_id, actions, options, rules=self.rules)
        self.artifact_inputs = artifact_inputs
        self.neural_metadata: Json | None = None
        if options is None:
            if artifact_inputs is not None:
                raise ContractError("baseline worker forbids neural artifact inputs")
            self.expected_policy = model_id + "-card-instance-v2"
        else:
            if type(artifact_inputs) is not RetainedGameArtifacts:
                raise ContractError("neural worker requires its retained original artifact owner")
            artifact_inputs.require_options(model_id, options)
            self.expected_policy = policy_label_game_v3(model_config_v3(options["model"], layout=CORRECTED_CARD_INSTANCE_LAYOUT),
                InferenceBindingV2(**options["inference"]), options["expected_manifest_digest"],
                action_contract=actions.action_contract, rules=self.rules, action_config_hash=digest(asdict(actions)),
                cuda_visible_devices=options["cuda_visible_devices"])
            self.neural_metadata = artifact_inputs.journal_metadata(self.expected_policy)
        integer(sampling_seed)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ContractError("positive protocol3 timeout required")
        self.model_id, self.actions = model_id, actions
        self.initial = {
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
        self.lease = ledger.reserve_worker(residency)  # must precede Popen.
        self.lease._cleanup_owner = self
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-E", "-m", "gf_simulator_client.card_instance_bootstrap"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                env={
                    "PATH": os.defpath,
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONHASHSEED": "0",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            self.process = process
            self.lease.spawned(process)
        except BaseException as error:
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

    async def initialize(self) -> None:
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

    async def select_frame(  # type: ignore[override]
        self, frame: DecisionFrame, delta: EventDelta
    ) -> tuple[GameGeneratedCandidatesV3, Selection]:
        await self.initialize()
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
