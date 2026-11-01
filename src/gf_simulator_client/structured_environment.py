"""Actions2 episode adapter; CLI injects journals, policy factory and shared ledger."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from gf_common.action_contract import ActionContract
from gf_common.diverse_actions import GeneratedCandidates
from gf_common.domain import (
    ContractError,
    DecisionInput,
    EpisodeGroup,
    EpisodeResult,
    EpisodeSpec,
    Json,
    RuntimePolicyBinding,
    digest,
    integer,
)
from gf_common.structured import ActionConfigV2, DecisionFrame, EventBank, EventDelta, Selection
from gf_simulator_client.environment import (
    EnvironmentSettings,
    Journal,
    SimulatorEnvironment,
    _DecisionReceipt,
)
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from gf_simulator_client.workers import ParticipantWorker


class StructuredJournalSink(Journal, Protocol):
    last_storage_ms: float

    def begin_frame(self, frame: DecisionFrame, delta: EventDelta, generated: GeneratedCandidates,
                    selection: Selection, command: Json, timings: Json) -> int: ...


@dataclass
class StructuredEpisodeResult(EpisodeResult):
    action_contract: ActionContract = field(kw_only=True)
    purpose: str = field(kw_only=True)
    synthetic_capacity: bool = field(kw_only=True)
    journal_schema_version: int = 5
    feature_version: str = "participant-structured-v1"
    participant_runtime: str = "python-reference-v2"
    model_startup_ms: dict[int, float] = field(default_factory=dict)
    decision_timings: list[Json] = field(default_factory=list)


class StructuredSimulatorEnvironment(SimulatorEnvironment):
    """Own original frames and parent action reconstruction; no source authorization."""

    def __init__(self, settings: EnvironmentSettings, policy_factory: Callable[..., Any],
                 journal_factory: Callable[..., Journal], *, ledger: ReservationLedger) -> None:
        if not isinstance(settings.actions, ActionConfigV2):
            raise ContractError("structured environment requires actions2")
        super().__init__(settings, policy_factory, journal_factory)
        self.ledger = ledger
        self._include_input_time = True

    def _new_result(self, group: EpisodeGroup, spec: EpisodeSpec) -> EpisodeResult:
        assert isinstance(self.settings.actions, ActionConfigV2)
        return StructuredEpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                       action_contract=self.settings.actions.action_contract, purpose="component", synthetic_capacity=False)

    def _create_actor(self, directory: Path, spec: EpisodeSpec, viewer: int, game: str,
                      journals: dict[int, Journal], workers: dict[int, ParticipantWorker],
                      contexts: dict[int, Any], result: EpisodeResult) -> None:
        assert isinstance(self.settings.actions, ActionConfigV2)
        label = spec.policies[viewer - 1]
        binding = self.settings.policy_bindings.get(label, RuntimePolicyBinding(label))
        worker = StructuredParticipantWorker(self.policy_factory, binding.model_id, spec.sampling_seeds[viewer - 1],
            self.settings.actions, self.settings.worker_seconds, ledger=self.ledger, policy_options=binding.policy_options)
        workers[viewer] = worker  # retain ownership before journal creation can fail.
        if label != worker.expected_policy:
            raise ContractError("structured schedule label differs from owned policy execution identity")
        bank = EventBank(viewer, str(uuid.uuid4()), worker.caps.max_episode_events)
        contexts[viewer] = bank
        aggregate_window = 128 if binding.policy_options is None else binding.policy_options["model"]["event"]["aggregate_window"]
        try:
            journals[viewer] = self.journal_factory(directory, spec.episode_id, viewer, label,
                run_id=self.settings.run_id, game_id=game, action_config_hash=digest(asdict(self.settings.actions)),
                bank_episode=bank.episode, caps=asdict(worker.caps), numeric_costs=asdict(worker.costs),
                model_id=binding.model_id, model_version="1" if binding.policy_options is None else "2",
                aggregate_window=aggregate_window, sampling_seed=spec.sampling_seeds[viewer - 1], use_action_options=True,
                action_contract=self.settings.actions.action_contract, purpose="component", synthetic_capacity=False)
        except OSError as exc:
            result.health["save_failure"] = str(exc)
            raise

    async def _ready_actors(self, workers: dict[int, ParticipantWorker], result: EpisodeResult) -> None:
        assert isinstance(result, StructuredEpisodeResult)
        for viewer, worker in workers.items():
            assert isinstance(worker, StructuredParticipantWorker)
            await worker.initialize()
            assert worker.startup_elapsed_ms is not None
            result.model_startup_ms[viewer] = worker.startup_elapsed_ms

    async def _select_and_journal(self, value: DecisionInput, worker: ParticipantWorker, journal: Journal,
                                  context: Any, selection_started: float, result: EpisodeResult) -> _DecisionReceipt:
        assert isinstance(worker, StructuredParticipantWorker)
        assert isinstance(context, EventBank) and isinstance(result, StructuredEpisodeResult)
        sink = cast(StructuredJournalSink, journal)
        before = context.boundary
        integer(len(value.events), before.end, worker.caps.max_episode_events)
        integer(len(value.events) - before.end, 0, worker.caps.max_interval_events)
        delta = EventDelta(context.actor, context.episode, before.end, before.prefix_hash, value.events[before.end:])
        context.append(delta)
        frame = DecisionFrame.from_input(value, context, before.end)
        frame_ready = time.monotonic()
        generated, selected = await worker.select_frame(frame, delta)
        # The Action is always reconstructed against this original parent frame.
        action = selected.validate(generated.table, frame)
        command = {"command_id": str(uuid.uuid4()), "expected_revision": frame.revision, "action": action}
        timings = {**worker.last_timings, "input_frame_host_ms": (frame_ready - selection_started) * 1000,
                   "parent_decision_host_ms": (time.monotonic() - selection_started) * 1000}
        try:
            index = sink.begin_frame(frame, delta, generated, selected, command, timings)
        except OSError as exc:
            result.health["save_failure"] = str(exc)
            raise
        result.decision_timings.append({"actor": frame.actor, **timings, "storage_host_ms": sink.last_storage_ms})
        return _DecisionReceipt(command, index, len(generated.table.kinds), timings["generation_ms"],
                                generated.report.attempts, worker.inference_elapsed_ms)
