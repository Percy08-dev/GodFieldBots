"""Corrected baselines on the shared episode/server/cleanup ownership loop."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from gf_common.artifacts import ArtifactSink, json_bound
from gf_common.card_instance import ActionConfigV3, CardInstanceContract
from gf_common.card_instance_rules import card_instance_rules
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
    canonical,
    integer,
)
from gf_common.hpo_capture import HpoCapture
from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
from gf_common.structured import DecisionFrame, EventBank, EventDelta, Selection, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.card_instance_game_tree_runtime import RetainedGamePolicyArtifacts
from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
from gf_models.structured_runtime import own_game_worker_settings_v3
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
        artifact_inputs: RetainedGameArtifacts | RetainedGamePolicyArtifacts | OwnedGameArtifacts | None = None,
        artifact_sink: ArtifactSink | None = None,
        server_output_root: Path | None = None,
        resident_pool: Any = None,
        capture: HpoCapture | None = None,
        game_observer: Any = None,
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
            own_game_worker_settings_v3(binding.model_id, settings.actions, binding.policy_options, rules=card_instance_rules())
            if binding.model_id == "heuristic_refresh":
                if label != refresh_policy_label(settings.actions, card_instance_rules(), parse_refresh_options(binding.policy_options if binding.policy_options is not None else {})):
                    raise ContractError("configured baseline label/options mismatch")
            elif binding.policy_options is None:
                if label != binding.model_id + "-card-instance-v2":
                    raise ContractError("baseline label differs from fixed execution")
            else:
                if type(artifact_inputs) not in (RetainedGameArtifacts, RetainedGamePolicyArtifacts, OwnedGameArtifacts):
                    raise ContractError("neural environment requires retained external inputs")
                assert artifact_inputs is not None
                artifact_inputs.require_options(binding.model_id, binding.policy_options)
                if artifact_inputs.journal_metadata(label)["policy_options"] != binding.policy_options:
                    raise ContractError("neural environment label/options mismatch")
        self._initialize(settings, _unused_factory, journal_factory)
        self._include_input_time = True
        self.ledger, self.runtime_evidence = ledger, runtime_evidence
        self.action_contract = settings.actions.action_contract
        self.artifact_inputs = artifact_inputs
        self.resident_pool = resident_pool
        self.capture = capture
        self.game_observer=game_observer
        self._observed_terminal: set[str]=set()
        self.artifact_sink, self.server_output_root = artifact_sink, server_output_root
        self._managed_server: ManagedServer | None = None
        self._storage_group_owner: CardInstanceSimulatorEnvironment | None = None

    async def run_group(self, group: EpisodeGroup) -> GroupResult:
        # Each group owns its evidence context. Concurrent groups never overwrite
        # another live process's receipt, while they share the reservation ledger.
        for spec in group.episodes:
            for label in spec.policies:
                binding = self.settings.policy_bindings.get(label)
                if (
                    binding is None
                    or binding.policy_options is None and label != binding.model_id + "-card-instance-v2"
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
                self.runtime_evidence.source_root, contract=cast(ActionConfigV3,self.settings.actions).action_contract,
                deadline=min(self.settings.wall_deadline, self.settings.work_deadline),
            ),
            artifact_inputs=self.artifact_inputs,
            artifact_sink=self.artifact_sink,
            server_output_root=self.server_output_root,
            resident_pool=self.resident_pool,
            capture=getattr(self,'capture',None),
            game_observer=getattr(self,'game_observer',None),
        )
        if self.artifact_sink is not None:
            self._storage_group_owner = own
        result = await own._run_group_body(group)
        capture=getattr(self,'capture',None)
        if capture is not None:
            try:
                capture.close_group(group, result)
            except ContractError as exc:
                result.health["capture_failure"] = str(exc)
        return result

    def _episode_opened(self, result: EpisodeResult) -> None:
        self._observe_game(result,'created')
        capture=getattr(self,'capture',None)
        if capture is not None:
            assert result.game_id is not None
            capture.open_episode(result.episode_id, result.game_id)

    def _act_acknowledged(self, result: EpisodeResult, viewer: int, command: Json,
                          acknowledgement: Json) -> None:
        if acknowledgement.get('status') in ('terminal','aborted'):
            self._observe_game(result,'terminal' if acknowledgement['status']=='terminal' else 'aborted')
        capture=getattr(self,'capture',None)
        if capture is not None:
            try:
                capture.accept_decision(result.episode_id, viewer, command["command_id"], acknowledgement)
            except ContractError as exc:
                result.health["capture_failure"] = str(exc)

    def _terminal_observed(self, result: EpisodeResult, viewer: int, observation: Json) -> None:
        self._observe_game(result,'terminal')
        capture=getattr(self,'capture',None)
        if capture is not None:
            capture.terminal_observation(result.episode_id, viewer, observation)

    def _episode_closed(self, result: EpisodeResult) -> None:
        self._observe_game(result,'retired')
        capture=getattr(self,'capture',None)
        if capture is not None and result.game_id is not None:
            try:
                capture.close_episode(result.episode_id, result)
            except ContractError as exc:
                result.health["capture_failure"] = str(exc)

    def _observe_game(self,result: EpisodeResult,event: str) -> None:
        observer=getattr(self,'game_observer',None)
        if observer is None or result.game_id is None:return
        from gf_common.hpo_progress import observe
        if event!='created':
            if result.game_id in self._observed_terminal:return
            self._observed_terminal.add(result.game_id)
        observe(observer,{'event':event,'game_id':result.game_id,'episode_id':result.episode_id,
            'monotonic':time.monotonic(),'scope':'actual-created-to-first-terminal-abort-or-retirement'})

    def _new_server(self, *args: Any, **kwargs: Any) -> ManagedServer:
        server = super()._new_server(*args, **kwargs)
        self._managed_server = server
        return server

    def storage_cleanup_complete(self, result: GroupResult) -> bool:
        owner = self._storage_group_owner
        if owner is None:
            return False
        server = owner._managed_server
        if server is not None:
            if any(not task.done() for task in server.drainers):
                return False
            if server.process is None:
                if not server.stopped:
                    return False
            elif server.process.returncode is None:
                return False
        process = result.health.get("process", {})
        diagnostic = result.health.get("http_diagnostics", {})
        return (not process.get("reap_failure")
            and diagnostic.get("status") not in ("stop_unconfirmed", "settlement_unconfirmed")
            and not result.health.get("cleanup_failure")
            and not any(episode.health.get("worker_cleanup_failure") for episode in result.episodes))

    def _create_episode_directory(self, directory: Path) -> None:
        if self.artifact_sink is None:
            super()._create_episode_directory(directory)

    def _save_artifact(self, path: Path, value: Any) -> None:
        if self.artifact_sink is None:
            super()._save_artifact(path, value)
        else:
            with self.artifact_sink.budget.reserve(8 * json_bound(value) + 65536):
                self.artifact_sink.write(path.parent.name + "/" + path.name,
                    (canonical(value) + "\n").encode())

    def _server_directory(self, group: EpisodeGroup) -> Path:
        if self.server_output_root is None:
            return super()._server_directory(group)
        return self.server_output_root / f"server-{group.episodes[0].episode_id}"

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
        if self.artifact_sink is not None:
            self.artifact_sink.hold_graph(f"participant-{obs['viewer_player_id']}",
                32 * (json_bound(obs) + json_bound(events) + json_bound(options)) + 65536)
        return decision_input(obs, events, cursor, definitions, options,
                              action_contract=self.action_contract)

    def _reconcile_replay(
        self, replay: Json, viewer: int, accepted: list[Json], outcome: Any
    ) -> None:
        if getattr(self,'capture',None) is None:
            reconcile_replay(replay, viewer, accepted, outcome, action_contract=self.action_contract)

    def _replay_closure(self, replay: Json, game: str, viewer: int) -> Json | None:
        if getattr(self, 'capture', None) is not None:
            return replay["episode_end"]
        return replay_closure(replay, game, viewer, action_contract=self.action_contract)

    def _replay_was_reconciled(self) -> bool:
        return getattr(self,'capture',None) is None

    def _finish_journal(self, journal: Journal, result: EpisodeResult, final: Json | None) -> None:
        if result.end_kind == "natural" and getattr(self,'capture',None) is None and not result.replay_reconciled:
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
            factory = CardInstanceParticipantWorker if self.resident_pool is None else lambda *args, **kwargs: self.resident_pool.worker(viewer, *args, **kwargs)
            shared_options={}
            if self.ledger.service is not None and self.resident_pool is None:
                from gf_simulator_client.shared_worker import SharedParticipantWorker
                factory=SharedParticipantWorker
                shared_options={'service':self.ledger.service.owner,'game':game,'viewer':viewer}
            elif self.resident_pool is not None and self.resident_pool.shared:
                shared_options={'game':game}
            worker = factory(
                binding.model_id,
                spec.sampling_seeds[viewer - 1],
                self.settings.actions,
                self.settings.worker_seconds,
                ledger=self.ledger,
                policy_options=binding.policy_options,
                **shared_options,
                artifact_inputs=self.artifact_inputs if binding.policy_options is not None and binding.model_id != "heuristic_refresh" else None,
                **({"deadline": self.settings.work_deadline} if binding.model_id == "lightgbm" else {}),
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
            action_contract=self.action_contract,
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
            **({"baseline_options": worker.baseline_options} if worker.baseline_options is not None else {}),
            **({"neural": worker.neural_metadata} if worker.neural_metadata is not None else {}),
            **({"tree": worker.tree_metadata} if worker.tree_metadata is not None else {}),
        )
        if self.capture is not None:
            self.capture.participant_created(spec.episode_id, game, viewer, bank.view())

    def _before_worker_close(self, workers: dict[int, ParticipantWorker], result: EpisodeResult) -> None:
        if self.resident_pool is not None and (result.end_kind == 'natural' and result.failure is None
                and (getattr(self,'capture',None) is not None or result.replay_reconciled) and not any(result.health.values())):
            from gf_simulator_client.resident_pool import ResidentParticipantWorker
            for worker in workers.values():
                if type(worker) is not ResidentParticipantWorker:
                    raise ContractError('resident natural completion changed original logical owner')
                worker.mark_completed()

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
            action_contract=context.raw_action_contract,
        )
        context.append(delta)
        frame = DecisionFrame.from_input(
            value, context, before.end, layout=CORRECTED_CARD_INSTANCE_LAYOUT
        )
        ready = time.monotonic()
        admission = (nullcontext() if self.artifact_sink is None else
            self.artifact_sink.budget.reserve(4096 * worker.caps.max_candidates
                + 128 * worker.caps.max_candidates * worker.caps.max_positions + 65536))
        with admission:
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
            capture=getattr(self,'capture',None)
            if capture is not None:
                capture.begin_decision(result.episode_id, frame.actor, frame, delta,
                    generated.table, selected, command)
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
