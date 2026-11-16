"""Private live collection owner; no conversion from public rollouts or replay.

The private construction seam is for the coordinator's actual lock/snapshot
owners. Production composition and encoded banks belong to the next increment.
"""
from __future__ import annotations

import json
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any, NoReturn

from gf_common.artifacts import ArtifactAdmission, ArtifactReservation, json_bound
from gf_common.card_instance_validation import observation as validate_observation
from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeResult,
    GroupResult,
    Json,
    canonical,
    closed,
    group_health_failures,
    integer,
)
from gf_common.structured import (
    CandidateTable,
    DecisionFrame,
    EventBank,
    EventBankView,
    EventBoundary,
    EventDelta,
    Selection,
)
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, StructuredLayout
from gf_training.collection import CollectionPlan

_TOKEN = object()


def _release(reservations: list[ArtifactReservation]) -> None:
    for reservation in reservations:
        reservation.close()


class _Retention:
    def __init__(self, budget: ArtifactAdmission) -> None:
        self.budget = budget
        self.reservations: list[ArtifactReservation] = []
        weakref.finalize(self, _release, self.reservations)

    def reserve(self, size: int) -> None:
        self.reservations.append(self.budget.reserve(size))

    def retain(self, value: Any) -> None:
        # An exported leaf can outlive the batch, so it also keeps admission.
        weakref.finalize(value, lambda retention: None, self)


@dataclass(frozen=True)
class _OwnedHistoryView(EventBankView):
    """Corrected participant view whose derived prefixes retain admission."""

    _retention: _Retention = field(repr=False, compare=False)

    @property
    def layout(self) -> StructuredLayout:
        return CORRECTED_CARD_INSTANCE_LAYOUT

    def prefix(self, boundary: EventBoundary) -> _OwnedHistoryView:
        integer(boundary.end, 0, len(self._events))
        if self._prefixes[boundary.end] != boundary.prefix_hash:
            raise ContractError("bank boundary mismatch")
        # Shared event bytes/prefix strings remain charged to their original
        # owner; admit the additional shallow tuples before creating them.
        self._retention.reserve(1024 + 16 * (boundary.end + 1))
        return _OwnedHistoryView(self.actor, self.episode, self._events[:boundary.end],
            self._prefixes[:boundary.end + 1], self._retention)


@dataclass(frozen=True)
class CapturedDecision:
    frame: DecisionFrame
    delta: EventDelta
    table: CandidateTable
    selection: Selection
    _command: bytes = field(repr=False)
    _acknowledgement: bytes = field(repr=False)
    _retention: _Retention = field(repr=False, compare=False)

    @property
    def command(self) -> Json:
        return json.loads(self._command)

    @property
    def acknowledgement(self) -> Json:
        return json.loads(self._acknowledgement)


@dataclass(frozen=True)
class CapturedParticipant:
    viewer: int
    decisions: tuple[CapturedDecision, ...]
    history: EventBankView
    _terminal: bytes = field(repr=False)
    _retention: _Retention = field(repr=False, compare=False)

    @property
    def terminal_observation(self) -> Json:
        return json.loads(self._terminal)


@dataclass(frozen=True)
class CapturedEpisode:
    episode_id: str
    game_id: str
    learner_viewer: int
    participants: tuple[CapturedParticipant, CapturedParticipant]
    _outcome: bytes = field(repr=False)
    replay_reconciled: bool = False

    @property
    def learner(self) -> CapturedParticipant:
        return self.participants[self.learner_viewer - 1]

    @property
    def outcome(self) -> Any:
        return json.loads(self._outcome)


class OwnedHpoBatch:
    """Live retained graph only; this foundation does not encode/train/persist."""

    origin = "producer-owned-v1"

    def __init__(self, token: object, context: _CaptureContext,
                 episodes: tuple[CapturedEpisode, ...]) -> None:
        if token is not _TOKEN:
            raise ContractError("owned batch requires its actual private capture")
        self.run_id, self.generation = context.run_id, context.generation
        self._owner, self._plan, self._snapshot_owner = context._owner, context._plan, context._snapshot_owner
        self._episodes = episodes
        self._retention = context._retention
        self._closed = False

    @property
    def episodes(self) -> tuple[CapturedEpisode, ...]:
        if self._closed:
            raise ContractError("owned batch is closed")
        return self._episodes

    def require_owner(self, owner: object, *, plan: CollectionPlan, snapshot_owner: object) -> None:
        if self._closed or owner is not self._owner or plan is not self._plan or snapshot_owner is not self._snapshot_owner:
            raise ContractError("owned batch differs from actual run/plan/snapshot owner")

    def close(self) -> None:
        self._episodes = ()
        self._retention = None
        self._closed = True


@dataclass
class _Pending:
    frame: DecisionFrame
    delta: EventDelta
    table: CandidateTable
    selection: Selection
    command: bytes
    acknowledgement: bytes | None = None


@dataclass
class _Episode:
    game_id: str
    banks: dict[int, EventBank]
    commands: dict[tuple[int, str], _Pending] = field(default_factory=dict)
    pending: dict[int, str] = field(default_factory=dict)
    terminals: dict[int, bytes] = field(default_factory=dict)
    result: EpisodeResult | None = None
    revision: int = 0


def _new_capture(owner: object, *, run_id: str, plan: CollectionPlan, generation: int,
                 snapshot_owner: object, learner_seats: dict[str, int],
                 learner_policy_version: str, budget: ArtifactAdmission) -> _CaptureContext:
    """Private composition seam; owner objects never come from serialized flags."""
    return _CaptureContext(_TOKEN, owner, run_id=run_id, plan=plan, generation=generation,
        snapshot_owner=snapshot_owner, learner_seats=learner_seats,
        learner_policy_version=learner_policy_version, budget=budget)


class _CaptureContext:
    def __init__(self, token: object, owner: object, *, run_id: str,
                 plan: CollectionPlan, generation: int, snapshot_owner: object,
                 learner_seats: dict[str, int], learner_policy_version: str,
                 budget: ArtifactAdmission) -> None:
        if token is not _TOKEN or owner is None or snapshot_owner is None:
            raise ContractError("capture requires actual run and learner snapshot owners")
        integer(generation)
        if not run_id or not learner_policy_version or type(plan) is not CollectionPlan:
            raise ContractError("capture requires explicit run/plan/model identity")
        specs = tuple(spec for group in plan.groups for spec in group.episodes)
        if (not specs or len({spec.episode_id for spec in specs}) != len(specs)
                or set(learner_seats) != {spec.episode_id for spec in specs}):
            raise ContractError("capture requires complete unique plan and explicit learner seats")
        for viewer in learner_seats.values():
            integer(viewer, 1, 2)
        for spec in specs:
            if spec.learner_viewer is not None and spec.learner_viewer != learner_seats[spec.episode_id]:
                raise ContractError("capture learner seat differs from actual plan")
        self._owner, self._plan, self._snapshot_owner = owner, plan, snapshot_owner
        self.run_id, self.generation = run_id, generation
        self._specs, self._seats = specs, dict(learner_seats)
        self._policy_version = learner_policy_version
        self._retention: _Retention | None = _Retention(budget)
        self._retention.reserve(65536 + 4096 * len(specs))
        self._episodes: dict[str, _Episode] = {}
        self._groups: dict[int, GroupResult] = {}
        self._state = "BUILDING"

    def _fail(self, message: str) -> NoReturn:
        self._state = "FAILED"
        raise ContractError(message)

    def _check(self) -> _Retention:
        if self._state != "BUILDING" or self._retention is None:
            raise ContractError("capture is not building")
        return self._retention

    def _episode(self, episode_id: str, *, open_only: bool = True) -> _Episode:
        self._check()
        episode = self._episodes.get(episode_id)
        if episode is None or open_only and episode.result is not None:
            self._fail("capture episode is missing or closed")
        return episode

    def open_episode(self, episode_id: str, game_id: str) -> None:
        self._check()
        if (episode_id not in self._seats or episode_id in self._episodes
                or type(game_id) is not str or not game_id
                or any(e.game_id == game_id for e in self._episodes.values())):
            self._fail("capture episode/game differs from original plan")
        self._episodes[episode_id] = _Episode(game_id, {})

    def participant_created(self, episode_id: str, game_id: str, viewer: int,
                            bank: EventBankView) -> None:
        """Retain the actual journal/context's initial participant identity."""
        episode = self._episode(episode_id)
        retention = self._check()
        try:
            integer(viewer, 1, 2)
            if (game_id != episode.game_id or not isinstance(bank, EventBankView)
                    or bank.actor != viewer or bank.layout is not CORRECTED_CARD_INSTANCE_LAYOUT
                    or viewer in episode.banks or episode.commands or episode.terminals
                    or bank._events or len(bank._prefixes) != 1):
                raise ContractError("capture participant creation identity/order mismatch")
            # Preserve the original EventBank identity contract, including
            # fixed non-UUID fixture identities. Production passes its UUID.
            if type(bank.episode) is not str or not bank.episode:
                raise ContractError("capture original bank identity required")
            retention.reserve(65536 + 8 * len(bank.episode))
            original = EventBank(viewer, bank.episode, layout=bank.layout)
            if original.boundary != bank.boundary:
                raise ContractError("capture participant initial prefix differs")
            episode.banks[viewer] = original
        except BaseException:
            self._state = "FAILED"
            raise

    def begin_decision(self, episode_id: str, viewer: int, frame: DecisionFrame,
                       delta: EventDelta, table: CandidateTable, selection: Selection,
                       command: Json) -> None:
        episode = self._episode(episode_id)
        retention = self._check()
        try:
            integer(viewer, 1, 2)
            closed(command, {"command_id", "expected_revision", "action"})
            integer(command["expected_revision"])
            if str(uuid.UUID(command["command_id"])) != command["command_id"]:
                raise ContractError("noncanonical command UUID")
            if (frame.actor != viewer or delta.actor != viewer or delta.episode != frame.episode
                    or frame.layout is not CORRECTED_CARD_INSTANCE_LAYOUT
                    or delta.layout is not frame.layout or table.layout is not frame.layout
                    or viewer in episode.pending or episode.pending
                    or viewer in episode.terminals or frame.revision != episode.revision
                    or (viewer, command["command_id"]) in episode.commands
                    or command["expected_revision"] != frame.revision
                    or selection.validate(table, frame) != command["action"]
                    or viewer == self._seats[episode_id] and selection.policy_version != self._policy_version):
                raise ContractError("capture decision participant/model/command mismatch")
            if viewer == self._seats[episode_id] and (selection.logprob is None or selection.value is None):
                raise ContractError("learner capture requires original finite policy statistics")
            # Reserve before retaining/copying any new graph, including event
            # bank growth and immutable command serialization.
            retention.reserve(65536 + len(frame.payload) * 4
                + sum(len(event) * 4 + 256 for event in delta._events)
                + 4096 * len(table.kinds) + 128 * len(table.slots) + 8 * json_bound(command))
            bank = episode.banks.get(viewer)
            if bank is None:
                bank = EventBank(viewer, frame.episode, layout=frame.layout)
                episode.banks[viewer] = bank
            bank.append(delta)
            frame.bank_view(bank.view())
            for value in (frame, delta, table, selection):
                retention.retain(value)
            episode.commands[viewer, command["command_id"]] = _Pending(
                frame, delta, table, selection, canonical(command).encode())
            episode.pending[viewer] = command["command_id"]
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            self._fail(str(error))

    def accept_decision(self, episode_id: str, viewer: int, command_id: str,
                        acknowledgement: Json) -> None:
        episode = self._episode(episode_id)
        try:
            integer(viewer, 1, 2)
            closed(acknowledgement, {"command_id", "previous_revision", "state_revision", "status", "result"})
            integer(acknowledgement["previous_revision"])
            record = episode.commands.get((viewer, command_id))
            if record is None:
                raise ContractError("ACK lacks original command")
            integer(acknowledgement["state_revision"], record.frame.revision + 1)
            if (acknowledgement["command_id"] != command_id
                    or acknowledgement["previous_revision"] != record.frame.revision
                    or acknowledgement["status"] not in ("active", "terminal")
                    or (acknowledgement["status"] == "terminal") != (acknowledgement["result"] is not None)):
                raise ContractError("ACK is rejected/stale or has wrong original command")
            raw = canonical(acknowledgement).encode()
            if record.acknowledgement is not None:
                if record.acknowledgement != raw:
                    raise ContractError("conflicting duplicate ACK")
                return
            if episode.pending.get(viewer) != command_id:
                raise ContractError("ACK is not the pending original command")
            self._check().reserve(65536 + 8 * json_bound(acknowledgement))
            record.acknowledgement = raw
            episode.revision = acknowledgement["state_revision"]
            del episode.pending[viewer]
        except (ValueError, TypeError, KeyError) as error:
            self._fail(str(error))

    def terminal_observation(self, episode_id: str, viewer: int, observation: Json) -> None:
        episode = self._episode(episode_id)
        try:
            integer(viewer, 1, 2)
            if (observation["viewer_player_id"] != viewer or observation["phase"] != "terminal"
                    or observation["result"] is None or observation["actionable"]
                    or observation["state_revision"] < episode.revision):
                raise ContractError("actual terminal participant observation required")
            self._check().reserve(65536 + 16 * json_bound(observation))
            validated = validate_observation(observation, viewer)
            raw = canonical(validated).encode()
            if viewer in episode.terminals and episode.terminals[viewer] != raw:
                raise ContractError("terminal observation changed")
            episode.terminals[viewer] = raw
        except (ValueError, TypeError, KeyError) as error:
            self._fail(str(error))

    def close_episode(self, episode_id: str, result: EpisodeResult) -> None:
        episode = self._episode(episode_id)
        if (result.episode_id != episode_id or result.game_id != episode.game_id
                or result.end_kind != "natural" or result.failure is not None
                or any(result.health.values()) or episode.pending
                or set(episode.terminals) != {1, 2}
                or result.decisions != len(episode.commands)
                or any(record.acknowledgement is None for record in episode.commands.values())):
            self._fail("incomplete/unhealthy episode cannot complete capture")
        projections = [tuple(json.loads(episode.terminals[v])[key] for key in
            ("state_revision", "result", "phase", "turn", "first_player", "active_player", "current_player", "end_times")) for v in (1, 2)]
        if projections[0] != projections[1] or projections[0][1] != result.outcome:
            self._fail("actual terminal observations disagree")
        episode.result = result

    def close_group(self, group: EpisodeGroup, result: GroupResult) -> None:
        self._check()
        if (not any(group is planned for planned in self._plan.groups)
                or id(group) in self._groups or group_health_failures(result.health)
                or result.health.get("runtime_evidence_failure")
                or [r.episode_id for r in result.episodes] != [spec.episode_id for spec in group.episodes]):
            self._fail("actual collection group failed or changed ownership")
        for item in result.episodes:
            episode = self._episode(item.episode_id, open_only=False)
            if episode.result is not item or item.failure is not None or any(item.health.values()):
                self._fail("late episode/group cleanup failed")
        self._groups[id(group)] = result

    def seal(self, owner: object, *, plan: CollectionPlan, snapshot_owner: object) -> OwnedHpoBatch:
        retention = self._check()
        if owner is not self._owner or plan is not self._plan or snapshot_owner is not self._snapshot_owner:
            self._fail("capture run/plan/generation/snapshot owner mismatch")
        if len(self._groups) != len(self._plan.groups):
            raise ContractError("all actual collection groups must close before seal")
        if any(group_health_failures(group.health) or group.health.get("runtime_evidence_failure")
                for group in self._groups.values()):
            self._fail("late collection group cleanup failed")
        episodes = []
        for spec in self._specs:
            episode = self._episode(spec.episode_id, open_only=False)
            if episode.result is None or any(episode.result.health.values()) or episode.result.failure is not None:
                self._fail("capture contains incomplete/unhealthy episode")
            if set(episode.banks) != {1, 2}:
                self._fail("capture requires both original participant bank identities")
            participants = []
            for viewer in (1, 2):
                bank = episode.banks[viewer]
                view = bank.view()
                history = _OwnedHistoryView(view.actor, view.episode, view._events,
                    view._prefixes, retention)
                decisions = tuple(CapturedDecision(p.frame, p.delta, p.table, p.selection,
                    p.command, p.acknowledgement, retention)
                    for (actor, _), p in episode.commands.items()
                    if actor == viewer and p.acknowledgement is not None)
                participants.append(CapturedParticipant(viewer, decisions, history, episode.terminals[viewer], retention))
            episodes.append(CapturedEpisode(spec.episode_id, episode.game_id,
                self._seats[spec.episode_id], (participants[0], participants[1]),
                canonical(json.loads(episode.terminals[1])["result"]).encode()))
        batch = OwnedHpoBatch(_TOKEN, self, tuple(episodes))
        self._episodes.clear()
        self._groups.clear()
        self._retention = None
        self._state = "SEALED"
        return batch

    def close(self) -> None:
        self._episodes.clear()
        self._groups.clear()
        self._retention = None
        self._state = "FAILED"
