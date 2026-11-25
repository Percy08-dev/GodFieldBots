"""Same-host original custody and Python Journal6 content encoding.

These owners issue no collection/Source/learning authority. The dedicated thread
owns SQLite and the unchanged Pack business operations; native captures retain
one recording lease through every alias. Private validation workspace is separate.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import random
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

import gf_native

from gf_common.artifacts import ArtifactAdmission
from gf_common.candidate_columns import (
    CODEC_V2,
    _codebook,
    column_layout,
    encode_candidate_columns_v2,
)
from gf_common.card_instance import ActionConfigV3, is_shared_contract
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.card_instance_runtime import validate_runtime_receipt
from gf_common.card_instance_validation import act_response, reconcile_replay, validate_events
from gf_common.diverse_actions import SearchReportV3
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.group_pack import (
    FORMAT,
    MAX_JSON,
    MIB,
    PackMember,
    StorageSettings,
    canonical_json,
    encode_pack,
    member_index,
    safe_name,
)
from gf_common.structured import (
    DecisionFrame,
    EventBoundary,
    EventDelta,
    Selection,
    _VerifiedJournalFrame,
)
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import (
    parse_delta,
    parse_frame,
    parse_game_generated_v3,
    parse_report_v3,
)
from gf_training.candidate_storage import candidate_basename
from gf_training.card_instance_journal import CardInstanceJournal
from gf_training.group_pack_db import DATABASE, fsync_directory, validate_descriptor
from gf_training.group_pack_resources import remaining
from gf_training.group_pack_source import CollectionLocator, OriginalFile
from gf_training.group_pack_worker import GroupPackDatabase, _retain_failure
from gf_training.group_pack_writer import PendingDescriptorOwner


@dataclass(frozen=True)
class PolicyJournalContext:
    native: Json
    journal: Json


@dataclass(frozen=True)
class EpisodeEncodingContext:
    run_id: str
    trial: str
    episode_id: str
    actions: ActionConfigV3
    rules: CardInstanceRules
    numeric_costs: Json  # GameSpec DEFAULT; each policy may explicitly override it.
    runtime_evidence: Json
    policies: tuple[PolicyJournalContext, PolicyJournalContext]
    deadline: float
    clock: Json | None = None
    learner_viewer: int | None = None


@dataclass(frozen=True)
class EpisodeEncodingPlan:
    """Complete immutable controls digest, excluding the not-yet-created process."""

    episode_id: str
    controls_sha256: str


def episode_encoding_plan(context: EpisodeEncodingContext) -> EpisodeEncodingPlan:
    controls = asdict(context)
    del controls["runtime_evidence"]
    if context.learner_viewer is None:
        del controls["learner_viewer"]
    return EpisodeEncodingPlan(context.episode_id, digest(controls))


@dataclass(frozen=True)
class StorageGroup:
    group_id: str
    episodes: tuple[str, ...]
    record: Json


class RecordingBudget:
    """Additional allocations only; native originals already belong to account."""

    def __init__(self, account: Any, deadline: float) -> None:
        self.account, self.deadline = account, deadline

    @property
    def used(self) -> int:
        return int(self.account.used)

    @property
    def peak(self) -> int:
        return int(self.account.peak)

    @property
    def limit(self) -> int:
        return int(self.account.limit)

    def reserve(self, size: int, *, wait: bool = False) -> _RecordingReservation:
        remaining(self.deadline)
        if wait:
            raise ValueError("recording transforms never wait on their own retained sources")
        if type(size) is not int or size < 0:
            raise ValueError("invalid recording reservation")
        return _RecordingReservation(self.account.reserve(size))


class _RecordingReservation:
    def __init__(self, lease: Any) -> None:
        self.lease = lease

    def resize(self, size: int) -> None:
        self.lease.resize(size)

    def close(self) -> None:
        self.lease.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _utf8_size(text: str) -> int:
    # Bounded individual chunks; the private transform loan owns this scratch.
    return sum(
        len(text[start : start + 4096].encode("utf-8")) for start in range(0, len(text), 4096)
    )


class RecordingArtifacts:
    """Owned canonical chunks and exact byte leases, without filesystem authority."""

    def __init__(self, budget: RecordingBudget) -> None:
        self.budget = budget
        self.chunks: dict[str, list[tuple[bytes, _RecordingReservation]]] = {}
        self.closed = False
        self._cleanup_children: list[Any] = []

    def contains(self, name: str) -> bool:
        safe_name(name)
        return name in self.chunks

    def write(
        self, name: str, data: bytes, reservation: _RecordingReservation | None = None
    ) -> None:
        safe_name(name)
        if self.closed or name in self.chunks or type(data) is not bytes:
            raise ValueError("artifact destination is closed, duplicate or invalid")
        lease = self.budget.reserve(len(data)) if reservation is None else reservation
        self.chunks[name] = [(data, lease)]

    def append_text(self, name: str, text: str) -> None:
        lease = self.budget.reserve(_utf8_size(text))
        try:
            data = text.encode("utf-8")
            self.chunks[name].append((data, lease))
        except BaseException as error:
            _retain_failure(self, error, clear_frames=True)
            if "data" in locals():
                del data
            lease.close()
            raise

    def json(self, name: str, value: Json) -> None:
        text = canonical(value)
        lease = self.budget.reserve(_utf8_size(text))
        try:
            self.write(name, text.encode("utf-8"), lease)
        except BaseException as error:
            _retain_failure(self, error, clear_frames=True)
            lease.close()
            raise

    def rename(self, old: str, new: str) -> None:
        safe_name(new)
        if self.closed or new in self.chunks:
            raise ValueError("artifact rename destination unavailable")
        self.chunks[new] = self.chunks.pop(old)

    def members(self) -> tuple[PackMember, ...]:
        # Join only fragmented outputs. Reserve the actual additional copy before
        # joining, then retire the old chunks after the new backing is reachable.
        for name, chunks in self.chunks.items():
            if len(chunks) != 1:
                lease = self.budget.reserve(sum(len(data) for data, _ in chunks))
                try:
                    joined = b"".join(data for data, _ in chunks)
                except BaseException as error:
                    _retain_failure(self, error, clear_frames=True)
                    lease.close()
                    raise
                self.chunks[name] = [(joined, lease)]
                for _, old in chunks:
                    old.close()
                chunks.clear()
        return tuple(
            PackMember(_member_kind(name), name, chunks[0][0])
            for name, chunks in sorted(self.chunks.items())
        )

    def close(self) -> None:
        from gf_training.group_pack_resources import close_resources

        close_resources(self, [])
        # Drop actual buffers before refunding their leases.
        leases = [lease for chunks in self.chunks.values() for _, lease in chunks]
        self.chunks.clear()
        for lease in leases:
            lease.close()
        self.closed = True


def _member_kind(name: str) -> str:
    return (
        "candidates"
        if name.endswith(".columns")
        else "journal"
        if name.endswith(".jsonl")
        else "replay"
        if "/replay-player-" in name
        else "result"
    )


class _Text:
    def __init__(self, sink: RecordingArtifacts, name: str) -> None:
        self.sink, self.name, self.closed = sink, name, False
        sink.write(name, b"")

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("journal output closed")
        self.sink.append_text(self.name, text)
        return len(text)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("journal output closed")

    def close(self) -> None:
        self.closed = True


class _Journal(CardInstanceJournal):
    def __init__(self, *args: Any, sink: RecordingArtifacts, **kwargs: Any) -> None:
        self.sink = sink
        super().__init__(*args, **kwargs)

    def _name(self, path: Path) -> str:
        return safe_name(path.parent.name + "/" + path.name)

    def _allocate_output(self) -> None:
        self.output = _Text(self.sink, self._name(self.path))  # type: ignore[assignment]

    def _final_exists(self) -> bool:
        return self.sink.contains(self._name(self.final_path))

    def _begin_native_frame(
        self, bundle: gf_native.StorageBundle, frame: DecisionFrame | _VerifiedJournalFrame, delta: EventDelta,
        report: SearchReportV3, selection: Selection, command: Json,
    ) -> int:
        started = time.perf_counter()
        self._context.validate_report(report, selection)
        payload: bytes | None = None
        descriptor: Json | None = None
        lease: _RecordingReservation | None = None

        def reconstruct() -> Json:
            nonlocal payload, descriptor, lease
            columns, size = column_layout(report.selected, report.selected_elements)
            lease = self.sink.budget.reserve(2 * size)  # native codec Vec and returned bytes coexist
            try:
                frame_hash, table_hash, selected, count, elements, action, payload, sha256 = (
                    bundle._journal_candidates(self.caps["max_positions"], size)
                )
            except ValueError as error:
                raise ContractError(str(error)) from error
            lease.resize(len(payload))
            if (
                frame_hash != frame.frame_hash
                or table_hash != selection.table_hash
                or selected != selection.index
                or count != report.selected
                or elements != report.selected_elements
            ):
                raise ContractError("native writer candidate/frame/report mismatch")
            descriptor = {
                "schema_version": 2,
                "codec": CODEC_V2,
                "file": candidate_basename(frame.actor, self.count),
                "count": count,
                "elements": elements,
                "bytes": size,
                "sha256": sha256,
                "frame_hash": frame_hash,
                "table_hash": table_hash,
                "columns": columns,
                "action_contract": asdict(self.action_contract),
                "purpose": self.purpose,
                "synthetic_capacity": False,
                "codebook_sha256": _codebook(LAYOUT),
            }
            return json.loads(action)

        def publish() -> Json:
            nonlocal lease
            assert lease is not None and payload is not None and descriptor is not None
            self.sink.write(self.path.parent.name + "/" + descriptor["file"], payload, lease)
            lease = None  # recording sink owns the only lease from here
            return descriptor

        try:
            return self._begin_frame(
                frame, delta, report, selection, command, {}, started,
                candidate_count=report.selected, action=reconstruct, publish=publish, owned_bank=True,
            )
        except BaseException as error:
            _retain_failure(self.sink, error, clear_frames=True)
            if lease is not None:
                payload = None
                lease.close()
            raise

    def _publish_candidates(self, frame: Any, table: Any, index: int, **kwargs: Any) -> Json:
        _, size = column_layout(len(table.kinds), len(table.slots))
        lease = self.sink.budget.reserve(2 * size)  # codec bytearray and final bytes overlap
        try:
            payload, descriptor = encode_candidate_columns_v2(
                table,
                candidate_basename(frame.actor, index),
                layout=LAYOUT,
                max_positions=kwargs["max_positions"],
            )
            lease.resize(len(payload))
            self.sink.write(self.path.parent.name + "/" + descriptor["file"], payload, lease)
            return descriptor
        except BaseException as error:
            _retain_failure(self.sink, error, clear_frames=True)
            if "payload" in locals():
                del payload
            lease.close()
            raise

    def acknowledge(self, index: int, acknowledgement: Json) -> None:
        super().acknowledge(index, acknowledgement)
        del self.commands[index]  # only pending commands are needed by this encoder

    def _publish_final_output(self) -> None:
        self.sink.rename(self._name(self.path), self._name(self.final_path))


def _verify_training_trajectory(native: Any, context: EpisodeEncodingContext, game: str,
                                outcome: Any, rows: list[_AcceptedTrainingRow]) -> None:
    viewer = integer(context.learner_viewer, 1, 2)
    policy = context.policies[viewer - 1].native
    winner = None if outcome == 'draw' else outcome['win']['winner']
    try:
        native.require_identity(context.run_id, context.trial, context.episode_id, game,
            viewer, policy['checkpoint'], policy['epoch'], len(rows), winner)
        for decision, row in enumerate(rows):
            native.require_row(decision, row.frame.revision, row.frame.frame_hash, row.selection.table_hash,
                row.candidates, row.selection.index, row.selection.logprob, row.selection.value)
    except ValueError as error:
        raise ContractError("native learner/Journal trajectory differs") from error


@dataclass(frozen=True)
class _AcceptedTrainingRow:
    frame: DecisionFrame | _VerifiedJournalFrame
    selection: Selection
    candidates: int


class WriterTrainingEpisode:
    """Numeric capability and raw learner facts from one committed Journal."""

    def __init__(self, budget: RecordingBudget) -> None:
        self._lease = budget.reserve(0)
        self._active_bytes = 0
        self._action_bytes = 0
        self.actions: tuple[list[Json], list[Json]] = ([], [])
        self.rows: list[_AcceptedTrainingRow] = []
        self.events: tuple[bytes, ...] = ()
        self.boundary: Any = None
        self.outcome: Any = None
        self.facts: Json = {}
        self.native: Any = None

    def grow(self, balance: int) -> None:
        self._lease.resize(self._active_bytes + balance)
        self._active_bytes += balance

    def accept(self, viewer: int, action: Json, row: _AcceptedTrainingRow | None) -> None:
        from gf_training.structured_rollout import _owned_bytes
        action_bytes = _owned_bytes(action)
        self.grow(action_bytes + (_owned_bytes(row) if row is not None else 0))
        self._action_bytes += action_bytes
        self.actions[viewer - 1].append(action)
        if row is not None:
            self.rows.append(row)

    def seal(self, bundle: Any, context: EpisodeEncodingContext, game: str,
             journal: _Journal, outcome: Any, replays: list[Json]) -> None:
        from gf_training.structured_rollout import _episode_records_digest, _owned_bytes

        viewers = {replay["viewer_player_id"]: replay for replay in replays}
        if set(viewers) != {1, 2} or len(replays) != 2 or any(replay['game_id'] != game for replay in replays):
            raise ContractError("training capture requires both exact participant Replays")
        for viewer, actions in enumerate(self.actions, 1):
            reconcile_replay(viewers[viewer], viewer, actions, outcome)
        del actions
        if journal.count != len(self.rows):
            raise ContractError("accepted learner count differs from actual Journal")
        native = bundle._training_episode()
        if type(native) is not gf_native.PlayerPpoEpisode:
            raise ContractError("native learner training capability missing from healthy terminal")
        _verify_training_trajectory(native, context, game, outcome, self.rows)
        self.boundary = journal.bank.boundary
        # Snapshot exact decision-prefix bytes before retiring the writer Journal.
        # These events are used only to preserve the numeric format's raw fields.
        self.grow(sum(sys.getsizeof(raw) + 8 for raw in journal.bank._events))
        self.events = tuple(journal.bank._events)
        self.outcome = outcome
        viewer = integer(context.learner_viewer, 1, 2)
        policy = context.policies[viewer - 1].native
        self.facts = {"episode_id": context.episode_id, "game_id": game, "viewer": viewer,
            "checkpoint": policy["checkpoint"], "epoch": policy["epoch"], "decisions": len(self.rows),
            "episode_digest": _episode_records_digest(context.actions.action_contract, "qualified_game", False,
                viewer, journal.bank.episode, self.boundary,
                ((row.frame, row.selection.table_hash, row.selection) for row in self.rows),
                "natural", outcome, None)}
        self.grow(_owned_bytes((self.boundary, outcome, self.facts)))
        for actions in self.actions:
            actions.clear()
        self.grow(-self._action_bytes)
        self._action_bytes = 0
        self.native = native

    def close(self) -> None:
        self.native = self.boundary = self.outcome = None
        self.facts = {}
        self.events = ()
        for actions in self.actions:
            actions.clear()
        self.rows.clear()
        self._lease.close()


class CanonicalEpisodeEncoder:
    """Encode actual native facts; this is content validation, never a live Source."""

    def __init__(
        self, context: EpisodeEncodingContext, sink: RecordingArtifacts,
        *, _writer_workspace: _WriterWorkspace | None = None,
    ) -> None:
        self.context, self.sink = context, sink
        self._writer_workspace = _writer_workspace
        self._native_candidates = is_shared_contract(context.actions.action_contract)
        self._action_contract = asdict(context.actions.action_contract)
        self.game: str | None = None
        self.journals: dict[int, _Journal] = {}
        self.pending: tuple[int, str] | None = None
        self.revision: int | None = None
        self.policy_states: dict[int, Json] = {}
        self.terminal: Json | None = None
        self.originals: list[Any] = []
        self.seen: set[int] = set()
        self.training = WriterTrainingEpisode(sink.budget) if context.learner_viewer is not None else None
        self._training_pending: _AcceptedTrainingRow | None = None

    def _open(self, game: str) -> None:
        if self.game is not None:
            if game != self.game:
                raise ContractError("original native game identity changed")
            return
        self.game = game
        c = self.context
        for policy in c.policies:
            effective = policy.native.get("numeric_costs")
            if effective is None:
                effective = c.numeric_costs
            if canonical(effective) != canonical(policy.journal["numeric_costs"]):
                raise ContractError("per-policy effective numeric costs differ from canonical journal")
        for viewer, policy in enumerate(c.policies, 1):
            self.journals[viewer] = _Journal(
                Path(c.episode_id),
                c.episode_id,
                viewer,
                run_id=c.run_id,
                game_id=game,
                actions=c.actions,
                rules=c.rules,
                bank_episode=c.episode_id,
                runtime_evidence=c.runtime_evidence,
                sink=self.sink,
                **policy.journal,
            )

    def consume(self, bundle: Any) -> None:
        self._consume(bundle, decode_original(bundle))

    def _consume(self, bundle: Any, data: Json) -> None:
        c = self.context
        if (
            type(bundle) is not gf_native.StorageBundle
            or not self.sink.budget.account.owns(bundle)
            or bundle.episode != c.episode_id
            or bundle.id in self.seen
            or self.terminal is not None
        ):
            raise ContractError("foreign, duplicate or late original storage bundle")
        kind = bundle.kind
        if data.get("schema") != f"rust-player-{kind}-v1" or data.get("episode") != c.episode_id:
            raise ContractError("native capture discriminant/episode mismatch")
        if kind != "terminal":
            if data.get("game") != bundle.game:
                raise ContractError("native capture game mismatch")
            self._open(bundle.game)
        if kind == "pending":
            closed(
                data,
                {
                    "schema",
                    "run",
                    "trial",
                    "game",
                    "episode",
                    "actor",
                    "frame",
                    "delta",
                    "generated",
                    "selected",
                    "scores",
                    "command",
                    "search_rng_before",
                    "search_rng_after",
                    "policy_state_before",
                    "policy_state_after",
                    "sampling_seed",
                    "policy",
                },
            )
            viewer = data["actor"]
            integer(viewer, 1, 2)
            policy = c.policies[viewer - 1]
            journal = self.journals[viewer]
            if (
                self.pending is not None
                or data["run"] != c.run_id
                or data["trial"] != c.trial
                or data["policy"] != policy.native
                or data["sampling_seed"] != policy.journal["sampling_seed"]
            ):
                raise ContractError("native pending chronology/policy context mismatch")
            frame: DecisionFrame | _VerifiedJournalFrame
            if self._native_candidates and self._writer_workspace is not None:
                scratch = self._writer_workspace.frame_scratch
                try:
                    frame = _verified_pending_frame(bundle, scratch, c.deadline)
                except MemoryError:
                    # The existing workspace is a soft loan. A conservative
                    # native JSON estimate can exceed it for an admissible frame.
                    remaining(c.deadline)
                    frame = parse_frame(data["frame"], layout=LAYOUT)
                    remaining(c.deadline)
            else:
                frame = parse_frame(data["frame"], layout=LAYOUT)
            delta = parse_delta(data["delta"], layout=LAYOUT)
            if self._native_candidates:
                raw = closed(data["generated"], {"table", "report"})
                generated = None
                report = parse_report_v3(
                    raw["report"], action_contract=c.actions.action_contract, rules=c.rules
                )
                table = raw["table"]
                if (
                    not isinstance(table, dict)
                    or table.get("action_contract") != self._action_contract
                ):
                    raise ContractError("native writer candidate table action contract mismatch")
                candidate_count = report.selected
                table_hash = report.table_hash
            else:
                generated = parse_game_generated_v3(
                    data["generated"], action_contract=c.actions.action_contract, rules=c.rules
                )
                report = generated.report
                candidate_count = len(generated.table.kinds)
                table_hash = generated.table.table_hash
            if (
                self.revision is not None
                and frame.revision != self.revision
                or report.rng_before != _state(data["search_rng_before"])
                or report.rng_after != _state(data["search_rng_after"])
            ):
                raise ContractError("native revision/search RNG chronology mismatch")
            scores = data["scores"]
            if policy.native["kind"] == "mlp":
                closed(scores, {"identity", "logprob", "sample", "value"})
                expected = {
                    "run": c.run_id,
                    "trial": c.trial,
                    "game": self.game,
                    "player": viewer,
                    "epoch": policy.native["epoch"],
                    "request": journal.count,
                    "revision": frame.revision,
                    "checkpoint": policy.native["checkpoint"],
                    "deadline": c.deadline,
                }
                if scores["identity"] != expected or scores["sample"] != data["selected"]:
                    raise ContractError("native selected score identity mismatch")
                logprob, value = scores["logprob"], scores["value"]
                if (
                    data["policy_state_before"] is not None
                    or data["policy_state_after"] is not None
                ):
                    raise ContractError("MLP capture has baseline state")
            else:
                if scores is not None:
                    raise ContractError("baseline capture has neural scores")
                logprob = value = None
                previous = self.policy_states.get(viewer)
                before, after = data["policy_state_before"], data["policy_state_after"]
                closed(before, {"rng", "refresh"})
                closed(after, {"rng", "refresh"})
                rng = random.Random(policy.journal["sampling_seed"])
                if previous is None:
                    if _state(before["rng"]) != rng.getstate():
                        raise ContractError("initial baseline RNG mismatch")
                elif before != previous:
                    raise ContractError("baseline state stream discontinuity")
                rng.setstate(_state(before["rng"]))
                if (
                    policy.native["kind"] == "random"
                    and rng.randrange(candidate_count) != data["selected"]
                ):
                    raise ContractError("Random selection differs from original own RNG")
                if rng.getstate() != _state(after["rng"]):
                    raise ContractError("baseline RNG transition mismatch")
                self.policy_states[viewer] = after
            selection = Selection(
                table_hash,
                data["selected"],
                policy.journal["policy"],
                logprob,
                value,
            )
            if generated is None:
                index = journal._begin_native_frame(bundle, frame, delta, report, selection, data["command"])
            else:
                assert isinstance(frame, DecisionFrame)
                index = journal.begin_frame(frame, delta, generated, selection, data["command"], {})
            if viewer == c.learner_viewer:
                self._training_pending = _AcceptedTrainingRow(frame, selection, candidate_count)
            self.pending = viewer, data["command"]["command_id"]
            assert index == journal.count - 1
        elif kind == "ack":
            closed(data, {"schema", "game", "episode", "actor", "command", "acknowledgement"})
            viewer = data["actor"]
            if self.pending != (viewer, data["command"]["command_id"]):
                raise ContractError("ACK lacks the unique original pending command")
            journal = self.journals[viewer]
            index = journal.count - 1
            if data["command"] != journal.commands[index]:
                raise ContractError("ACK command differs from pending original")
            response = act_response(data["acknowledgement"], data["command"], viewer)
            journal.acknowledge(
                index,
                {
                    key: response[key]
                    for key in (
                        "command_id",
                        "previous_revision",
                        "state_revision",
                        "status",
                        "result",
                    )
                },
            )
            if self.training is not None and response["status"] in ("active", "terminal"):
                self.training.accept(viewer, data["command"]["action"], self._training_pending)
            self._training_pending = None
            self.revision = response["state_revision"]
            self.pending = None
        elif kind == "terminal":
            closed(
                data,
                {
                    "schema",
                    "report",
                    "run",
                    "trial",
                    "episode",
                    "game",
                    "runtime_evidence",
                    "actions",
                    "rules",
                    "numeric_costs",
                    "clock",
                    "policies",
                    "terminal_observations",
                    "replays",
                    "banks",
                },
            )
            if (
                data["run"] != c.run_id
                or data["trial"] != c.trial
                or digest(data["actions"]) != digest(asdict(c.actions))
                or data["rules"] != asdict(c.rules)
                or data["numeric_costs"] != c.numeric_costs
                or data["clock"] != c.clock
                or data["runtime_evidence"] != c.runtime_evidence
                or data["policies"] != [p.native for p in c.policies]
                or data["game"] != (bundle.game or None)
            ):
                raise ContractError("terminal controls differ from actual collector context")
            if data["game"] is not None:
                self._open(data["game"])
            report = data["report"]
            if report["episode"] != c.episode_id or report["game"] != data["game"]:
                raise ContractError("terminal report identity mismatch")
            if len(data["banks"]) != 2:
                raise ContractError("both original participant banks required")
            observations = {obs["viewer_player_id"]: obs for obs in data["terminal_observations"]}
            for viewer, bank in enumerate(data["banks"], 1):
                if bank["actor"] != viewer or bank["episode"] != c.episode_id:
                    raise ContractError("terminal original bank identity mismatch")
                final_journal = self.journals.get(viewer)
                if final_journal is not None:
                    prefix = [json.loads(event) for event in final_journal.bank._events]
                    if bank["events"][: len(prefix)] != prefix:
                        raise ContractError("terminal bank differs from original decision prefix")
                    # A failure between draining Events and publishing pending
                    # has no J6 decision. Preserve/validate that actual tail below;
                    # never manufacture its frame, ACK or transition.
                    validate_events(bank["events"])
                    del prefix
                    final_journal.finish(
                        report["end_kind"],
                        report["outcome"],
                        observations.get(viewer) if report["end_kind"] == "natural" else None,
                        report["simulator_abort_reason"],
                    )
                elif bank["events"]:
                    raise ContractError("events without a created participant journal")
            if (self.training is not None and report["end_kind"] == "natural"
                    and report["error"] is None and not report["cleanup_errors"] and report["inference_released"]):
                self.training.seal(bundle, c, bundle.game,
                    self.journals[integer(c.learner_viewer, 1, 2)], report["outcome"], data["replays"])
            for replay in data["replays"]:
                viewer = replay.get("viewer_player_id", replay.get("viewer"))
                integer(viewer, 1, 2)
                self.sink.json(f"{c.episode_id}/replay-player-{viewer}.json", replay)
            self.sink.json(
                f"{c.episode_id}/native-terminal.json",
                {key: data[key] for key in ("report", "banks", "terminal_observations")},
            )
            self.terminal = report
            # Terminal validation is complete; only actual originals/canonical
            # bytes remain needed for the group. Retire the private bank graph
            # now, including for a staged incomplete seat-swap pair.
            self.journals.clear()
            self.policy_states.clear()
            self.seen.clear()
        else:
            raise ContractError("unknown native storage transport kind")
        self.seen.add(bundle.id)
        self.originals.append(bundle)  # SAME capability/reservation; custody before ACK

    def retire(self) -> None:
        self._writer_workspace = None
        self.originals.clear()
        self.journals.clear()
        self.policy_states.clear()
        self._training_pending = None
        if self.training is not None:
            self.training.close()
            self.training = None


def _verified_pending_frame(
    bundle: gf_native.StorageBundle, scratch: gf_native._JournalFrameScratch, deadline: float,
) -> _VerifiedJournalFrame:
    try:
        metadata, payload = bundle._journal_frame(scratch, deadline)
    except ValueError as error:
        raise ContractError(str(error)) from error
    actor, episode, revision, start, boundary, start_hash, frame_hash = metadata
    return _VerifiedJournalFrame(
        actor, episode, revision, start, EventBoundary(*boundary), start_hash, payload, frame_hash,
    )


def _state(value: Json) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError("closed own RNG state required")
    return value[0], tuple(value[1]), value[2]


# An explicit caller-owned SOFT Source workspace loan covers decoding, typed
# validation and retained incremental banks. No per-record formula/scan claims a
# hard RSS limit; recording buffers above use their distinct exact byte leases.
def decode_original(bundle: Any) -> Json:
    if type(bundle) is not gf_native.StorageBundle:
        raise ContractError("actual immutable native StorageBundle required")
    view = bundle.view()
    buffer = memoryview(view)
    if hashlib.sha256(buffer).hexdigest() != bundle.transport_sha256:
        raise ContractError("native transport SHA changed")
    return json.loads(
        codecs.decode(buffer, "utf-8"),
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite native JSON")),
    )


class RustPendingDescriptor:
    """Actual pending FD owner and descriptor backing, including surviving views.

    Python3.14's buffer protocol retains this wrapper until the LAST memoryview
    releases; closing the file owner cannot refund bytes still exposed by views.
    No inner legacy writer or Source capability is exposed.
    """

    def __init__(self, owner: PendingDescriptorOwner, lease: _RecordingReservation) -> None:
        self._owner: PendingDescriptorOwner | None = owner
        self._lease: _RecordingReservation | None = lease
        self._exports = 0
        self._files_closed = False
        self._lock = threading.RLock()
        self.path, self.directory = owner.path, owner.directory
        self.descriptor_sha256 = hashlib.sha256(owner.raw).hexdigest()

    def recheck(self) -> None:
        with self._lock:
            if self._owner is None or self._files_closed:
                raise RuntimeError("pending descriptor closed")
            self._owner.recheck()

    def verify_training_pack(self, *, manifest_sha256: str, logical_sha256: str, run_id: str) -> Json:
        from gf_training.rust_ppo_pack import check_sealed_pack
        with self._lock:
            self.recheck()
            assert self._owner is not None
            result = check_sealed_pack(self._owner.raw, self._owner.database,
                descriptor_sha256=self.descriptor_sha256, manifest_sha256=manifest_sha256,
                logical_sha256=logical_sha256, run_id=run_id)
            self.recheck()
            return result

    def __buffer__(self, flags: int) -> memoryview:
        with self._lock:
            if self._owner is None or self._files_closed:
                raise RuntimeError("pending descriptor closed")
            self._exports += 1
            return memoryview(self._owner.raw)

    def __release_buffer__(self, view: memoryview) -> None:
        with self._lock:
            self._exports -= 1
            self._retire()

    def view(self) -> memoryview:
        return memoryview(self)

    def _retire(self) -> None:
        if self._files_closed and self._exports == 0:
            self._owner = None
            if self._lease is not None:
                self._lease.close()
                self._lease = None

    def close(self) -> None:
        with self._lock:
            if self._owner is not None and not self._files_closed:
                self._owner.close()
                self._files_closed = True
            self._retire()

    def _invalidate(self) -> None:
        with self._lock:
            if self._owner is None:
                return
            owner = self._owner
            info = os.fstat(owner.fd)
            if (info.st_dev, info.st_ino) != owner.identity:
                raise ValueError("pending descriptor original FD changed")
            marker = b"GFB-QUARANTINED-DESCRIPTOR\n"
            os.ftruncate(owner.fd, 0)
            if os.pwrite(owner.fd, marker, 0) != len(marker):
                raise OSError("pending descriptor invalidation incomplete")
            os.fsync(owner.fd)
            fsync_directory(owner.directory)


@dataclass
class _Command:
    kind: str
    value: Any
    done: threading.Event
    accepted: bool = False
    result: Any = None
    diagnostic_ticket: Any = None


class _WriterWorkspace:
    """The existing grant shared only by its original writer and inventory."""

    def __init__(self, reservation: Any, *, frame_scratch_bytes: int = 0) -> None:
        self.reservation = reservation
        self._frame_scratch_bytes = frame_scratch_bytes
        self._frame_scratch: gf_native._JournalFrameScratch | None = None
        self.writer_live = True
        self.inventories = 0
        self.lock = threading.Lock()

    @property
    def frame_scratch(self) -> gf_native._JournalFrameScratch:
        # Only the live writer uses this, on its first eligible pending frame.
        if self._frame_scratch is None:
            self._frame_scratch = gf_native.StorageBundle._journal_frame_scratch(self._frame_scratch_bytes)
        return self._frame_scratch

    def _retire(self) -> None:
        if not self.writer_live and not self.inventories and self.reservation is not None:
            self._frame_scratch = None
            self.reservation.close()
            self.reservation = None

    def close(self) -> None:
        with self.lock:
            self.writer_live = False
            self._retire()


class _InventoryLoan:
    def __init__(self, workspace: _WriterWorkspace) -> None:
        self.workspace = workspace
        self.released = False
        with workspace.lock:
            if not workspace.writer_live:
                raise RuntimeError("writer workspace already retired")
            workspace.inventories += 1

    def close(self) -> None:
        with self.workspace.lock:
            if not self.released:
                self.workspace.inventories -= 1
                self.released = True
            self.workspace._retire()


_UNCERTAIN_INVENTORY_LOANS: list[_InventoryLoan] = []


def _finalize_inventory(loan: _InventoryLoan) -> None:
    try:
        loan.close()
    except BaseException:  # noqa: BLE001 -- actual grant remains reachable after uncertain close
        _UNCERTAIN_INVENTORY_LOANS.append(loan)


class RustWriterInventory:
    """Immutable DATA ONLY; rows are borrowed until this snapshot closes.

    Consume/serialize in a bounded scope and drop all row views and temporary
    dictionaries before close. Independently read-back Source data has its own
    original shared-account ownership; this snapshot grants no Source authority.
    """

    def __init__(self, rows: tuple[tuple[str, int, str], ...], workspace: _WriterWorkspace) -> None:
        self._rows = rows
        self._loan = _InventoryLoan(workspace)
        self._closed = False
        self._finalizer = weakref.finalize(self, _finalize_inventory, self._loan)

    @property
    def rows(self) -> tuple[tuple[str, int, str], ...]:
        if self._closed:
            raise RuntimeError("writer inventory snapshot closed")
        return self._rows

    def close(self) -> None:
        if self._closed:
            return
        self._rows = ()
        self._loan.close()
        self._closed = True
        self._finalizer.detach()

    def __reduce__(self) -> Any:
        raise TypeError("inventory workspace ownership cannot be copied or serialized")


class RustPlayerStorage:
    """A real dedicated SQLite thread, with explicit source-workspace ownership.

    ``accept`` completes validation/canonical encoding and adopts SAME originals
    before returning. Its caller then ACKs that original native bundle. A partial
    pair stays staged; no complete-group SQL transaction is needed to retire its
    first episode's game slot. ``seal`` is separate from acceptance and join.
    """

    def __init__(
        self,
        runtime: Any,
        directory: Path,
        *,
        run_id: str,
        contexts: tuple[EpisodeEncodingContext, ...],
        groups: tuple[StorageGroup, ...],
        settings: StorageSettings,
        workspace: ArtifactAdmission,
        workspace_bytes: int,
        deadline: float,
        cleanup_deadline: float,
        deferred_contexts: bool = False,
        context_plans: tuple[EpisodeEncodingPlan, ...] = (),
        training: bool = False,
        work_directory: Path | None = None,
        durable_mount: tuple[Path, int] | None = None,
    ) -> None:
        if (work_directory is None) != (durable_mount is None):
            raise ValueError("staged writer requires both work and durable mount ownership")
        if len({c.episode_id for c in contexts}) != len(contexts):
            raise ValueError("duplicate planned storage episode")
        planned = [episode for group in groups for episode in group.episodes]
        if (
            not groups
            or len(set(planned)) != len(planned)
            or (not {c.episode_id for c in contexts}.issubset(planned) if deferred_contexts
                else set(planned) != {c.episode_id for c in contexts})
            or len({g.group_id for g in groups}) != len(groups)
            or any(c.run_id != run_id for c in contexts)
        ):
            raise ValueError("storage groups/context differ from original plan")
        if type(deferred_contexts) is not bool:
            raise ValueError("explicit deferred-context mode required")
        if deferred_contexts:
            if (any(type(plan) is not EpisodeEncodingPlan for plan in context_plans)
                    or len(context_plans) != len(planned)
                    or {plan.episode_id for plan in context_plans} != set(planned)):
                raise ValueError("complete immutable episode context plans required")
        elif context_plans:
            raise ValueError("context plans require explicit deferred mode")
        self.deferred_contexts = deferred_contexts
        self.training = training
        self._training_episodes: dict[str, WriterTrainingEpisode] = {}
        self._sealed_logical_sha256: str | None = None
        self._training_seal_verified = False
        self._context_plans = {plan.episode_id: plan for plan in context_plans}
        self.runtime, self.directory, self.run_id = runtime, directory, run_id
        self._work_directory = directory if work_directory is None else work_directory
        self._durable_mount = durable_mount
        self.settings, self.deadline, self.cleanup_deadline = settings, deadline, cleanup_deadline
        self.budget = RecordingBudget(runtime.recording_account(), deadline)
        if self.budget.limit != settings.budget_bytes:
            raise ValueError("writer and native runtime require the SAME recording capacity")
        self.workspace, self.workspace_bytes = workspace, workspace_bytes
        self.groups = groups
        self.sinks = {g.group_id: RecordingArtifacts(self.budget) for g in groups}
        self.group_for = {episode: group for group in groups for episode in group.episodes}
        self.encoders = {
            c.episode_id: CanonicalEpisodeEncoder(
                c, self.sinks[self.group_for[c.episode_id].group_id]
            )
            for c in contexts
        }
        if deferred_contexts and any(
            {episode for episode in group.episodes if episode in self.encoders}
            not in (set(), set(group.episodes)) for group in groups
        ):
            raise ValueError("initial deferred contexts must bind complete planned groups")
        if deferred_contexts and any(episode_encoding_plan(c) != self._context_plans[c.episode_id]
                for c in contexts):
            raise ValueError("initial context differs from immutable original plan")
        if deferred_contexts and any(runtime.game_admission_attempted(c.run_id, c.trial, c.episode_id) for c in contexts):
            raise ValueError("original native admission preceded deferred context registration")
        self.committed: set[str] = set()
        self._condition = threading.Condition()
        from gf_common.performance_metrics import StorageMetrics
        from gf_simulator_client.performance_diagnostics import host_diagnostics
        self._performance = host_diagnostics()
        self._performance_queue: Any = (StorageMetrics(scope=self._performance.current_scope or self._performance.scope(phase='idle'),
            start_ns=time.monotonic_ns(), model='thread-fifo') if self._performance is not None else None)
        self._queue: deque[_Command] = deque()
        self._queued_bytes = 0
        self._active: _Command | None = None
        self._failure: str | None = None
        self._closing = False
        self._sealed = False
        self._retry = False
        self._ready = threading.Event()
        self._database: GroupPackDatabase | None = None
        self._publication_database: OriginalFile | None = None
        self._pending_owner: RustPendingDescriptor | None = None
        self._workspace_lease: Any = None
        self._publication_fd = -1
        self._publication_lease: _RecordingReservation | None = None
        self._temporary: tuple[Path, tuple[int, int]] | None = None
        self._published: tuple[Path, tuple[int, int]] | None = None
        self._uncertain_releases: list[Any] = []
        self._cleanup_children: list[Any] = []
        self.cleanup_errors: tuple[str, ...] = ()
        self.thread_identity: int | None = None
        self._request = 0
        self._thread = threading.Thread(target=self._run, name="gf-rust-player-storage")
        try:
            # start() may launch its native thread before its startup wait is
            # interrupted. Once attempted, absence of ident/alive is not proof
            # of non-creation: retain this SAME owner on every launch failure.
            self._thread.start()
            if not self._ready.wait(remaining(deadline)):
                raise TimeoutError("storage initialization deadline")
            self._check()
        except BaseException as error:
            # Launch may already have transferred real I/O/workspace custody.
            # A failed assignment must return THAT owner, including uncertainty.
            _retain_failure(self, error)
            error.__dict__["cleanup_owner"] = self
            try:
                self.cancel("storage initialization failed")
            except BaseException as cleanup:  # noqa: BLE001 -- preserve original construction error
                _retain_failure(self, cleanup, clear_frames=True)
                error.add_note(
                    f"initialization cancellation remains owned ({type(cleanup).__name__})"
                )
            raise

    def _check(self) -> None:
        if self._failure is not None:
            error = RuntimeError(self._failure)
            error.__dict__["cleanup_owner"] = self
            raise error

    def _rpc(self, kind: str, value: Any = None) -> Any:
        command = _Command(kind, value, threading.Event())
        try:
            size = value.reserved_bytes if kind == "bundle" else 0
            if size > self.settings.queue_bytes:
                self.cancel("native record exceeds storage queue byte bound")
                raise ValueError("native record exceeds storage queue byte bound")
            if self._performance_queue is not None:
                command.diagnostic_ticket = self._performance_queue.attempt(time.monotonic_ns())
            with self._condition:
                self._check()
                while kind == "bundle" and self._queued_bytes + size > self.settings.queue_bytes:
                    self._condition.wait(remaining(self.deadline))
                    self._check()
                    if self._closing:
                        raise RuntimeError("storage admission closed")
                if self._closing or self._sealed:
                    raise RuntimeError("storage admission closed")
                self._queued_bytes += size
                self._queue.append(command)
                if command.diagnostic_ticket is not None:
                    self._performance_queue.admit(command.diagnostic_ticket, time.monotonic_ns(), size)
                self._condition.notify_all()
        except BaseException:
            if command.diagnostic_ticket is not None:
                with self._condition:
                    self._performance_queue.fail(command.diagnostic_ticket, time.monotonic_ns())
            raise
        if not command.done.wait(remaining(self.deadline)):
            self.cancel("storage operation deadline")
            raise TimeoutError("storage operation deadline")
        if not command.accepted:
            self._check()
            raise RuntimeError("storage command failed")
        return command.result

    def accept(self, bundle: Any) -> None:
        if type(bundle) is not gf_native.StorageBundle or not self.budget.account.owns(bundle):
            self.cancel("foreign storage account/capability")
            raise ContractError("storage custody requires original SAME-account native capability")
        self._rpc("bundle", bundle)

    def store_control(self, name: str, data: bytes) -> None:
        safe_name(name)
        if type(data) is not bytes or len(data) > min(self.settings.max_pack_bytes, 64 * MIB):
            raise ValueError("invalid control bytes")
        self._rpc("control", (name, data))

    def inventory(self) -> RustWriterInventory:
        """Ordered raw-member index; partial groups cannot issue a snapshot."""
        value = self._rpc("inventory")
        if value is None:
            raise ContractError("complete committed groups required for writer inventory")
        return value

    def training_evidence(self, inventory: RustWriterInventory, episode_ids: tuple[str, ...]) -> list[Json]:
        if (not self.training or inventory._closed or inventory._loan.workspace is not self._workspace_lease
                or set(episode_ids) != self._training_episodes.keys() or len(set(episode_ids)) != len(episode_ids)):
            raise ContractError("complete committed learner trajectories required")
        rows = [self._training_episodes[episode].facts for episode in episode_ids]
        if len({row['game_id'] for row in rows}) != len(rows):
            raise ContractError("duplicate learner trajectory game identity")
        return rows

    def bind_contexts(self, contexts: tuple[EpisodeEncodingContext, ...]) -> None:
        """Bind one actual complete group before any original native admission."""
        if not self.deferred_contexts:
            raise ContractError("late contexts require explicit deferred mode")
        self._rpc("contexts", contexts)

    def _bind_contexts(self, contexts: tuple[EpisodeEncodingContext, ...]) -> None:
        if type(contexts) is not tuple or not contexts or any(type(c) is not EpisodeEncodingContext for c in contexts):
            raise ContractError("actual immutable episode contexts required")
        names = {c.episode_id for c in contexts}
        if len(names) != len(contexts) or any(name not in self.group_for for name in names):
            raise ContractError("duplicate or unplanned episode context")
        group = self.group_for[contexts[0].episode_id]
        if names != set(group.episodes) or any(name in self.encoders for name in names):
            raise ContractError("complete previously unbound group contexts required")
        if any(self.runtime.game_admission_attempted(c.run_id, c.trial, c.episode_id) for c in contexts):
            raise ContractError("actual native admission already attempted before context binding")
        if any(c.run_id != self.run_id or episode_encoding_plan(c) != self._context_plans[c.episode_id]
                for c in contexts):
            raise ContractError("actual group context differs from original controls/policy plan")
        staged: dict[str, CanonicalEpisodeEncoder] = {}
        for context in contexts:
            # Own immutable controls, not caller-mutated receipt/policy dicts.
            policies = tuple(PolicyJournalContext(clone(p.native), clone(p.journal))
                for p in context.policies)
            context = replace(context, numeric_costs=clone(context.numeric_costs),
                runtime_evidence=clone(context.runtime_evidence), clock=clone(context.clock),
                policies=(policies[0], policies[1]))
            receipt = validate_runtime_receipt(context.runtime_evidence, require_executed=True)
            if receipt["action_contract"] != asdict(context.actions.action_contract):
                raise ContractError("executed receipt differs from planned participant profile")
            staged[context.episode_id] = CanonicalEpisodeEncoder(
                context, self.sinks[group.group_id], _writer_workspace=self._workspace_lease)
        self.encoders.update(staged)

    def seal(self) -> CollectionLocator:
        return self._rpc("seal")

    def take_pending_owner(self) -> RustPendingDescriptor:
        self.join()
        if not self._sealed or self._pending_owner is None:
            raise RuntimeError("no actual pending descriptor owner")
        result, self._pending_owner = self._pending_owner, None
        return result

    def verify_training_seal(self, manifest_sha256: str) -> Json:
        if not self.training or not self._sealed or self._thread.is_alive() or self._pending_owner is None or self._sealed_logical_sha256 is None:
            raise ContractError("sealed joined writer with actual training witness required")
        with self.budget.reserve(2 * 64 * 1024):
            observed = self._pending_owner.verify_training_pack(manifest_sha256=manifest_sha256,
                logical_sha256=self._sealed_logical_sha256, run_id=self.run_id)
        self._training_seal_verified = True
        return observed

    def take_training_episodes(self, episode_ids: tuple[str, ...]) -> tuple[WriterTrainingEpisode, ...]:
        if (not self.training or not self._sealed or self._thread.is_alive() or not self._training_seal_verified
                or set(episode_ids) != self._training_episodes.keys()):
            raise ContractError("actual sealed learner trajectories required")
        result = tuple(self._training_episodes[episode] for episode in episode_ids)
        self._training_episodes.clear()
        return result

    def close_unclaimed_training(self) -> None:
        if (not self.training and self._durable_mount is None) or self._thread.is_alive():
            return
        for trajectory in self._training_episodes.values():
            trajectory.close()
        self._training_episodes.clear()
        if self._pending_owner is not None:
            self._pending_owner.close()
            self._pending_owner = None

    def staging_input_retired(self) -> bool:
        return (self._thread.ident is not None and not self._thread.is_alive()
                and not self.cleanup_errors and self._publication_database is None)

    def cancel(self, reason: str) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = "storage failed: " + reason
            self._closing = True
            for command in self._queue:
                command.done.set()
            self._condition.notify_all()
        self.runtime.cancel("storage failed")

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._retry = True
            self._condition.notify_all()
        self.join()

    def join(self) -> dict[str, Any]:
        self._thread.join(remaining(self.cleanup_deadline))
        if self._thread.is_alive():
            raise TimeoutError("original storage thread join unconfirmed")
        if self._performance_queue is not None and not getattr(self, '_performance_flushed', False):
            if self._failure is not None or self.cleanup_errors:
                self._performance_queue.metrics.error('writer_retirement_incomplete')
            assert self._performance is not None
            self._performance.metrics(self._performance_queue.metrics, role='writer')
            self._performance_flushed = True
            self._performance_queue = None
        if self.cleanup_errors:
            raise RuntimeError(
                "storage cleanup remains unconfirmed: " + ",".join(self.cleanup_errors)
            )
        self._check()
        return {
            "joined": True,
            "thread_identity": self.thread_identity,
            "sealed": self._sealed,
            "committed_groups": len(self.committed),
            "recording_native_bytes": self.budget.account.native_bytes,
            "recording_external_bytes": self.budget.account.external_bytes,
            "recording_peak_bytes": self.budget.peak,
        }

    def _db(self, command: str, payload: bytes | None = None, **fields: Any) -> Any:
        assert self._database is not None
        result = self._database.execute(
            dict(version=1, request_id=self._request, command=command, **fields), payload
        )
        self._request += 1
        return result

    def _commit(self, group: StorageGroup) -> None:
        if group.group_id in self.committed or any(
            e not in self.encoders or self.encoders[e].terminal is None for e in group.episodes
        ):
            return
        sink = self.sinks[group.group_id]
        members = sink.members()
        index = canonical_json({"format": "group-pack-v1", "files": member_index(members)})
        raw_size = 16 + len(index) + sum(len(member.data) for member in members)
        if raw_size > self.settings.max_pack_bytes:
            raise ValueError("canonical group exceeds Pack byte bound")
        # Existing compressor returns chunks plus their final joined backing.
        # The format's output bound is raw_size+MIB, not max-Pack/game preallocation.
        compressed = self.budget.reserve(2 * (raw_size + MIB))
        pack = None
        try:
            pack = encode_pack(members, max_uncompressed_bytes=raw_size)
            compressed.resize(len(pack))
            episodes = [
                {
                    "episode_id": e,
                    "game_id": self.encoders[e].game,
                    "record": self.encoders[e].terminal,
                }
                for e in group.episodes
            ]
            self._db(
                "submit",
                pack,
                group_id=group.group_id,
                raw_bytes=raw_size,
                record=group.record,
                episodes=episodes,
            )
            del episodes
        except BaseException as error:
            # Do not refund while a failed DB/codec traceback still owns bytes.
            _retain_failure(self, error, clear_frames=True)
            raise
        finally:
            pack = None
            compressed.close()
        del members, index
        self.committed.add(group.group_id)
        for episode in group.episodes:
            encoder = self.encoders[episode]
            if encoder.training is not None and encoder.training.native is not None:
                self._training_episodes[episode], encoder.training = encoder.training, None
            encoder.retire()
        sink.close()

    def _publish(self, descriptor: Json) -> CollectionLocator:
        from uuid import uuid4

        descriptor = validate_descriptor(descriptor)
        database = self._publication_database
        assert database is not None
        database.freeze_database(descriptor["database_sha256"])
        database.recheck()
        if (
            database.identity.st_size != descriptor["database_bytes"]
            or descriptor["run_id"] != self.run_id
        ):
            raise ValueError("sealed descriptor differs from original held DB")
        raw = canonical_json(descriptor, limit=MAX_JSON)
        self._publication_lease = self.budget.reserve(len(raw))
        if self._durable_mount is not None:
            from gf_training.rust_player_promotion import promote_collection

            mount_root, mount_device = self._durable_mount
            pending = promote_collection(database, directory=self.directory, descriptor=descriptor, raw=raw,
                settings=self.settings, deadline=self.deadline, max_owned_bytes=self.workspace_bytes,
                mount_root=mount_root, mount_device=mount_device)
            self._pending_owner = RustPendingDescriptor(pending, self._publication_lease)
            self._publication_lease = None
            database.close()
            self._publication_database = None
            self._pending_owner.recheck()
            return CollectionLocator(FORMAT, pending.path)
        target = self.directory / "collection.pending.json"
        temporary = self.directory / f".descriptor-{uuid4().hex}.tmp"
        import os

        self._publication_fd = os.open(
            temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        info = os.fstat(self._publication_fd)
        identity = info.st_dev, info.st_ino
        self._temporary = temporary, identity
        cursor = 0
        while cursor < len(raw):
            remaining(self.deadline)
            written = os.write(self._publication_fd, raw[cursor : cursor + 65536])
            if written <= 0:
                raise OSError("descriptor write incomplete")
            cursor += written
        os.fsync(self._publication_fd)
        os.link(temporary, target, follow_symlinks=False)
        self._published = target, identity
        if (temporary.stat().st_dev, temporary.stat().st_ino) != identity:
            raise ValueError("temporary descriptor original changed")
        temporary.unlink()
        self._temporary = None
        fsync_directory(self.directory)
        if (target.stat().st_dev, target.stat().st_ino) != identity or os.pread(
            self._publication_fd, len(raw) + 1, 0
        ) != raw:
            raise ValueError("pending descriptor original changed")
        # Actual retained creation FD and original DB; no legacy process receipt.
        pending = PendingDescriptorOwner(
            self.directory, target, identity, raw, descriptor, self._publication_fd, database
        )
        assert self._publication_lease is not None
        self._pending_owner = RustPendingDescriptor(pending, self._publication_lease)
        self._publication_fd = -1
        self._publication_database = None
        self._publication_lease = None
        self._pending_owner.recheck()
        self._published = None
        return CollectionLocator(FORMAT, target)

    def _cleanup(self) -> tuple[str, ...]:
        from gf_training.group_pack_resources import (
            close_raw_descriptor,
            close_resources,
            require_certain_releases,
        )

        errors = []
        if self._database is not None and not self._sealed:
            errors.extend(self._database.fail())
        for attribute in ("_publication_database",):
            original = getattr(self, attribute)
            if original is not None:
                try:
                    original.close()
                except BaseException as error:  # noqa: BLE001 -- actual owner cleanup/failure boundary
                    _retain_failure(self, error, clear_frames=True)
                    errors.append(type(error).__name__)
                else:
                    setattr(self, attribute, None)
        try:
            if self._pending_owner is not None and not self._sealed:
                # A completed durable catalog may already reference this exact
                # seal even when cancellation wins before Source handoff.
                if self._durable_mount is None:
                    self._pending_owner._invalidate()
                self._pending_owner.close()
                self._pending_owner = None
                self._published = None
            if self._published is not None:
                import os

                path, identity = self._published
                if self._publication_fd < 0:
                    self._publication_fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                info = os.fstat(self._publication_fd)
                if (info.st_dev, info.st_ino) != identity:
                    raise ValueError("foreign published descriptor replacement")
                marker = b"GFB-QUARANTINED-DESCRIPTOR\n"
                os.ftruncate(self._publication_fd, 0)
                if os.pwrite(self._publication_fd, marker, 0) != len(marker):
                    raise OSError("descriptor invalidation incomplete")
                os.fsync(self._publication_fd)
                fsync_directory(self.directory)
                self._published = None
            close_raw_descriptor(self, "_publication_fd")
            require_certain_releases(self)
            close_resources(self, [])
        except BaseException as error:  # noqa: BLE001 -- actual owner cleanup/failure boundary
            _retain_failure(self, error, clear_frames=True)
            errors.append(type(error).__name__)
        if not errors:
            if not self._sealed:
                for trajectory in self._training_episodes.values():
                    trajectory.close()
                self._training_episodes.clear()
            for encoder in self.encoders.values():
                encoder.retire()
            for sink in self.sinks.values():
                sink.close()
            if self._workspace_lease is not None:
                self._workspace_lease.close()
                self._workspace_lease = None
            if self._publication_lease is not None:
                self._publication_lease.close()
                self._publication_lease = None
        return tuple(errors)

    def _run(self) -> None:
        self.thread_identity = threading.get_ident()
        command = None
        try:
            self._workspace_lease = _WriterWorkspace(
                self.workspace.reserve(self.workspace_bytes), frame_scratch_bytes=self.workspace_bytes,
            )
            for initial_encoder in self.encoders.values():
                initial_encoder._writer_workspace = self._workspace_lease
            self._work_directory.mkdir(parents=True, exist_ok=False)
            self._database = GroupPackDatabase(max_owned_bytes=self.workspace_bytes)
            values = asdict(self.settings) | {
                "db_root": str(self.settings.db_root if self._durable_mount is None else self._work_directory.parent),
                "ram_root": str(self.settings.ram_root),
            }
            self._db(
                "init",
                directory=str(self._work_directory),
                run_id=self.run_id,
                settings=values,
                deadline=self.deadline,
            )
            self._publication_database = OriginalFile(
                self._work_directory / DATABASE, self.deadline, mutable=True
            )
            self._ready.set()
            while True:
                with self._condition:
                    while not self._queue and not self._closing:
                        self._condition.wait(remaining(self.deadline))
                    if self._closing:
                        break
                    command = self._queue.popleft()
                    self._active = command
                    if command.diagnostic_ticket is not None:
                        self._performance_queue.dispatch(command.diagnostic_ticket, time.monotonic_ns())
                if command.kind == "bundle":
                    bundle = command.value
                    encoder = self.encoders.get(bundle.episode)
                    if encoder is None:
                        raise ContractError("unplanned native episode")
                    encoder.consume(bundle)
                    del bundle
                    # Cancellation before custody acceptance cannot become a
                    # successful command merely because encoding just returned.
                    self._check()
                    # Custody acceptance is published BEFORE complete-group SQL.
                    command.accepted = True
                    command.done.set()
                    self._commit(self.group_for[command.value.episode])
                elif command.kind == "control":
                    name, data = command.value
                    self._db("control", data, name=name)
                    del data
                    command.accepted = True
                    command.done.set()
                elif command.kind == "contexts":
                    self._bind_contexts(command.value)
                    command.accepted = True
                    command.done.set()
                elif command.kind == "inventory":
                    if len(self.committed) == len(self.groups):
                        assert self._database is not None and self._database.db is not None
                        rows = tuple(self._database.db.execute(
                            "SELECT name,length,sha256 FROM members ORDER BY name"
                        ))
                        command.result = RustWriterInventory(rows, self._workspace_lease)
                        del rows
                    # An incomplete observational request is refused without
                    # corrupting the actual still-writing partial pair.
                    command.accepted = True
                    command.done.set()
                elif command.kind == "seal":
                    if len(self.encoders) != len(self.group_for) or len(self.committed) != len(self.groups) or any(
                        encoder.pending is not None for encoder in self.encoders.values()
                    ):
                        raise ValueError("cannot seal incomplete original groups")
                    descriptor = self._db("seal-producer" if self.training else "seal-metrics", publication="pending")["descriptor"]
                    if self.training:
                        assert self._database is not None
                        self._sealed_logical_sha256 = self._database._sealed_logical_sha256
                    locator = self._publish(descriptor)
                    # Cancellation and successful seal advertisement have ONE
                    # linearization owner. A completed publish alone is not it.
                    with self._condition:
                        self._check()
                        self._sealed = True
                        command.result = locator
                        command.accepted = True
                        command.done.set()
                    break
                else:
                    raise ValueError("unknown storage command")
                with self._condition:
                    if command.kind == "bundle":
                        self._queued_bytes -= command.value.reserved_bytes
                    if command.diagnostic_ticket is not None:
                        self._performance_queue.release(command.diagnostic_ticket)
                    self._active = None
                    self._condition.notify_all()
                command = None
            if not self._sealed and self._failure is None:
                self._db("close")
        except BaseException as error:  # noqa: BLE001 -- actual owner cleanup/failure boundary
            # Adopt every actual nested resource before discarding exception
            # chains. Keep only generic status, never dead private buffer frames.
            _retain_failure(self, error, clear_frames=True)
            self.cancel(type(error).__name__ + ": " + str(error))
        finally:
            self._ready.set()
            if command is not None:
                command.done.set()
            with self._condition:
                for waiting in self._queue:
                    waiting.done.set()
                self._condition.notify_all()
            command = None
            self._active = None
            self.cleanup_errors = self._cleanup()
            # A failed safe close can be retried only on THIS original thread.
            while self.cleanup_errors and time.monotonic() < self.cleanup_deadline:
                with self._condition:
                    if not self._retry:
                        self._condition.wait(max(0, self.cleanup_deadline - time.monotonic()))
                    self._retry = False
                self.cleanup_errors = self._cleanup()
            if not self.cleanup_errors:
                self._queue.clear()
                self._queued_bytes = 0
