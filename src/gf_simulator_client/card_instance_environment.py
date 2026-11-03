"""Corrected baselines on the shared episode/server/cleanup ownership loop."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from gf_common.card_instance import ActionConfigV3, CardInstanceContract
from gf_common.card_instance_runtime import require_corrected_contract
from gf_common.card_instance_validation import decision_input, reconcile_replay, replay_closure
from gf_common.diverse_actions import GameGeneratedCandidatesV3
from gf_common.domain import (
    ContractError,
    DecisionInput,
    EpisodeGroup,
    EpisodeResult,
    EpisodeSpec,
    GroupResult,
    Json,
    RuntimePolicyBinding,
    integer,
)
from gf_common.structured import DecisionFrame, EventBank, EventDelta, Selection, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.environment import (
    EnvironmentSettings,
    Journal,
    SimulatorEnvironment,
    _DecisionReceipt,
)
from gf_simulator_client.http import AttestedClient
from gf_simulator_client.process import ManagedServer
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import _startup_cleanup_owner
from gf_simulator_client.workers import ParticipantWorker


class CardInstanceJournalSink(Journal, Protocol):
    last_storage_ms: float

    def begin_frame(self, frame: DecisionFrame, delta: EventDelta, generated: GameGeneratedCandidatesV3,
                    selection: Selection, command: Json, timings: Json) -> int: ...


@dataclass
class CardInstanceEpisodeResult(EpisodeResult):
    action_contract: CardInstanceContract = field(kw_only=True)
    purpose: str = "qualified_game"
    synthetic_capacity: bool = False
    journal_schema_version: int = 6
    feature_version: str = "participant-structured-v2"
    participant_runtime: str = "python-reference-v3"
    model_startup_ms: dict[int, float] = field(default_factory=dict)
    decision_timings: list[Json] = field(default_factory=list)


def _unused_factory(*args: Any, **kwargs: Any) -> Any:
    raise ContractError("corrected participant factory is fixed in the clean protocol3 bootstrap")


class CardInstanceSimulatorEnvironment(SimulatorEnvironment):
    def __init__(
        self,
        settings: EnvironmentSettings,
        journal_factory: Callable[..., Journal],
        *,
        ledger: ReservationLedger,
        runtime_evidence: CardInstanceRuntimeEvidence,
    ) -> None:
        if type(settings.actions) is not ActionConfigV3:
            raise ContractError("corrected environment requires exact actions3")
        require_corrected_contract(settings.actions.action_contract)
        if (
            settings.external_url is not None
            or settings.actions.max_sequence_length > 18
            or Path(settings.binary).absolute() != runtime_evidence.binary
        ):
            raise ContractError(
                "corrected execution requires the matching locally managed verified release"
            )
        runtime_evidence.preflight()  # before output, server or participant allocation.
        for label, binding in settings.policy_bindings.items():
            if (
                binding.model_id not in ("random", "heuristic")
                or binding.policy_options is not None
                or label != binding.model_id + "-card-instance-v2"
            ):
                raise ContractError("corrected execution currently supports fixed baselines only")
        self._initialize(settings, _unused_factory, journal_factory)
        self._include_input_time = True
        self.ledger, self.runtime_evidence = ledger, runtime_evidence
        self._managed_server: ManagedServer | None = None

    async def run_group(self, group: EpisodeGroup) -> GroupResult:
        # Each group owns its evidence context. Concurrent groups never overwrite
        # another live process's receipt, while they share the reservation ledger.
        for spec in group.episodes:
            for label in spec.policies:
                binding = self.settings.policy_bindings.get(label)
                if (
                    binding is None
                    or binding.model_id not in ("random", "heuristic")
                    or label != binding.model_id + "-card-instance-v2"
                ):
                    raise ContractError(
                        "corrected schedule requires explicit baseline execution bindings"
                    )
        own = CardInstanceSimulatorEnvironment(
            self.settings,
            self.journal_factory,
            ledger=self.ledger,
            runtime_evidence=CardInstanceRuntimeEvidence(
                self.runtime_evidence.binary,
                self.runtime_evidence.source_root,
                deadline=min(self.settings.wall_deadline, self.settings.work_deadline),
            ),
        )
        return await own._run_group_body(group)

    async def _episode(
        self, client: AttestedClient, definitions: Json, group: EpisodeGroup, spec: EpisodeSpec
    ) -> EpisodeResult:
        if self._managed_server is None or self._managed_server.process is None:
            raise ContractError("corrected game requires its owned live runtime")
        self.runtime_evidence.preflight()
        self.runtime_evidence.verify_executed(self._managed_server.process)
        return await self._episode_body(client, definitions, group, spec)

    def _server_started(self, server: ManagedServer) -> None:
        if server.process is None:
            raise ContractError("corrected runtime process owner missing")
        self._managed_server = server
        self.runtime_evidence.verify_executed(server.process)

    def _server_stopped(self, server: ManagedServer, health: Json) -> None:
        if server.process is None or server.process.returncode is None:
            raise ContractError("corrected runtime must be reaped before final evidence check")
        health["runtime_evidence"] = self.runtime_evidence.finalize()

    def _new_client(self, *args: Any, **kwargs: Any) -> AttestedClient:
        self.runtime_evidence.receipt(require_executed=True)
        assert type(self.settings.actions) is ActionConfigV3
        return AttestedClient(*args, **kwargs, contract=self.settings.actions.action_contract)

    def _decision_input(
        self,
        obs: Json,
        events: list[Json],
        cursor: str | None,
        definitions: Json,
        options: Json | None,
    ) -> DecisionInput:
        return decision_input(obs, events, cursor, definitions, options)

    def _reconcile_replay(
        self, replay: Json, viewer: int, accepted: list[Json], outcome: Any
    ) -> None:
        reconcile_replay(replay, viewer, accepted, outcome)

    def _replay_closure(self, replay: Json, game: str, viewer: int) -> Json | None:
        return replay_closure(replay, game, viewer)

    def _finish_journal(self, journal: Journal, result: EpisodeResult, final: Json | None) -> None:
        if result.end_kind == "natural" and not result.replay_reconciled:
            raise ContractError("journal6 natural reward requires both reconciled player closures")
        super()._finish_journal(journal, result, final)

    def _new_result(self, group: EpisodeGroup, spec: EpisodeSpec) -> EpisodeResult:
        assert type(self.settings.actions) is ActionConfigV3
        return CardInstanceEpisodeResult(
            spec.episode_id,
            spec.pair_id,
            group.seed,
            spec.policies,
            action_contract=self.settings.actions.action_contract,
        )

    def _create_actor(
        self,
        directory: Path,
        spec: EpisodeSpec,
        viewer: int,
        game: str,
        journals: dict[int, Journal],
        workers: dict[int, ParticipantWorker],
        contexts: dict[int, Any],
        result: EpisodeResult,
    ) -> None:
        assert type(self.settings.actions) is ActionConfigV3
        label = spec.policies[viewer - 1]
        binding = self.settings.policy_bindings.get(label, RuntimePolicyBinding(label))
        try:
            worker = CardInstanceParticipantWorker(
                binding.model_id,
                spec.sampling_seeds[viewer - 1],
                self.settings.actions,
                self.settings.worker_seconds,
                ledger=self.ledger,
            )
        except BaseException as error:
            owner = _startup_cleanup_owner(error, self.ledger)
            if owner is not None:
                workers[viewer] = owner
            raise
        workers[viewer] = worker
        if label != worker.expected_policy:
            raise ContractError("corrected schedule label differs from owned execution")
        bank = EventBank(
            viewer,
            str(uuid.uuid4()),
            worker.caps.max_episode_events,
            layout=CORRECTED_CARD_INSTANCE_LAYOUT,
        )
        contexts[viewer] = bank
        journals[viewer] = self.journal_factory(
            directory,
            spec.episode_id,
            viewer,
            label,
            run_id=self.settings.run_id,
            game_id=game,
            actions=self.settings.actions,
            bank_episode=bank.episode,
            caps=asdict(worker.caps),
            numeric_costs=asdict(worker.costs),
            model_id=binding.model_id,
            sampling_seed=spec.sampling_seeds[viewer - 1],
            runtime_evidence=self.runtime_evidence.receipt(require_executed=True),
        )

    async def _ready_actors(
        self, workers: dict[int, ParticipantWorker], result: EpisodeResult
    ) -> None:
        assert isinstance(result, CardInstanceEpisodeResult)
        for viewer, worker in workers.items():
            assert isinstance(worker, CardInstanceParticipantWorker)
            await worker.initialize()
            assert worker.startup_elapsed_ms is not None
            result.model_startup_ms[viewer] = worker.startup_elapsed_ms

    async def _select_and_journal(
        self,
        value: DecisionInput,
        worker: ParticipantWorker,
        journal: Journal,
        context: Any,
        selection_started: float,
        result: EpisodeResult,
    ) -> _DecisionReceipt:
        assert isinstance(worker, CardInstanceParticipantWorker)
        assert isinstance(context, EventBank) and isinstance(result, CardInstanceEpisodeResult)
        before = context.boundary
        integer(len(value.events), before.end, worker.caps.max_episode_events)
        integer(len(value.events) - before.end, 0, worker.caps.max_interval_events)
        delta = event_delta(
            context.actor,
            context.episode,
            before.end,
            before.prefix_hash,
            value.events[before.end :],
            layout=CORRECTED_CARD_INSTANCE_LAYOUT,
        )
        context.append(delta)
        frame = DecisionFrame.from_input(
            value, context, before.end, layout=CORRECTED_CARD_INSTANCE_LAYOUT
        )
        ready = time.monotonic()
        generated, selected = await worker.select_frame(frame, delta)
        command = {
            "command_id": str(uuid.uuid4()),
            "expected_revision": frame.revision,
            "action": selected.validate(generated.table, frame),
        }
        timings = {
            **worker.last_timings,
            "input_frame_host_ms": (ready - selection_started) * 1000,
            "parent_decision_host_ms": (time.monotonic() - selection_started) * 1000,
        }
        sink = cast(CardInstanceJournalSink, journal)
        index = sink.begin_frame(frame, delta, generated, selected, command, timings)
        result.decision_timings.append(
            {"actor": frame.actor, **timings, "storage_host_ms": sink.last_storage_ms}
        )
        return _DecisionReceipt(
            command,
            index,
            len(generated.table.kinds),
            timings["generation_ms"],
            generated.report.attempts,
            worker.inference_elapsed_ms,
        )
