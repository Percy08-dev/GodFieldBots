"""Participant-owned structured records. No engine, transport, array or model I/O.

Action tables describe typed syntax and own references, never game legality.
The append chain authenticates continuity within one participant/episode owner;
it is an integrity check, not evidence of a server-authenticated history.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.domain import ContractError, DecisionInput, Json, canonical, closed, digest, integer
from gf_common.validation import validate_events, validate_worker_input

INPUT_RECIPE = "participant-structured-v1"
GENERATOR_V2 = "public-atomic-diverse-v4"
ACTION_KINDS = ("pray", "play_sequence", "respond", "respond_sequence", "discard", "sacrifice",
                "exchange", "sell", "buy", "buy_decision")
TARGETS = ("omitted", "null", "self", "opponent")
NUMERIC_FIELDS = ("hp", "mp", "money", "accept")


def _bytes(value: Any) -> bytes:
    return canonical(value).encode("utf-8")


def _hash(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("invalid SHA256 digest")


def _owner(actor: int, episode: str) -> None:
    integer(actor, 1, 2)
    if not isinstance(episode, str) or not episode:
        raise ContractError("episode identity required")


@dataclass(frozen=True)
class ActionConfigV2:
    version: str = "2"
    generator: str = GENERATOR_V2
    use_action_options: bool = True
    targets: tuple[str, ...] = ("opponent", "self")
    max_sequence_length: int = 18
    max_candidates: int = 32768
    max_expansions: int = 262144
    max_temporary_candidates: int = 262144
    count_states: int = 16384
    count_milliseconds: int = 50
    search_milliseconds: int = 1000
    channel_weights: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)

    def __post_init__(self) -> None:
        if self.version != "2" or self.generator != GENERATOR_V2:
            raise ContractError("unsupported structured action recipe")
        if type(self.use_action_options) is not bool or not self.use_action_options:
            raise ContractError("structured v2 requires acting-player atomic hints")
        if (type(self.targets) is not tuple or not self.targets or len(set(self.targets)) != len(self.targets)
                or set(self.targets) - {"self", "opponent"}):
            raise ContractError("invalid structured target configuration")
        for name, maximum in (("max_sequence_length", 128), ("max_candidates", 65536),
                              ("max_expansions", 2**31 - 1),
                              ("max_temporary_candidates", 2**31 - 1),
                              ("count_states", 2**31 - 1), ("count_milliseconds", 1000),
                              ("search_milliseconds", 10000)):
            integer(getattr(self, name), 1, maximum)
        if not isinstance(self.channel_weights, tuple) or len(self.channel_weights) != 7:
            raise ContractError("seven channel weights required")
        for weight in self.channel_weights:
            integer(weight, 1, 2**31 - 1)


def action_config_v2(value: Json) -> ActionConfigV2:
    closed(value, set(asdict(ActionConfigV2())))
    if not isinstance(value["channel_weights"], (list, tuple)):
        raise ContractError("channel weights array required")
    if not isinstance(value["targets"], (list, tuple)):
        raise ContractError("target array required")
    return ActionConfigV2(**{**value, "channel_weights": tuple(value["channel_weights"]), "targets": tuple(value["targets"])})


@dataclass(frozen=True)
class EventBoundary:
    end: int
    prefix_hash: str

    def __post_init__(self) -> None:
        integer(self.end, 0, 2**31 - 1)
        _hash(self.prefix_hash)


@dataclass(frozen=True, init=False)
class EventDelta:
    actor: int
    episode: str
    start: int
    previous_hash: str
    _events: tuple[bytes, ...] = field(repr=False)

    def __init__(self, actor: int, episode: str, start: int, previous_hash: str,
                 events: tuple[Json, ...] | list[Json]) -> None:
        _owner(actor, episode)
        integer(start, 0, 2**31 - 1)
        _hash(previous_hash)
        if not isinstance(events, (list, tuple)):
            raise ContractError("event array required")
        validate_events(list(events))
        for key, value in (("actor", actor), ("episode", episode), ("start", start),
                           ("previous_hash", previous_hash),
                           ("_events", tuple(_bytes(event) for event in events))):
            object.__setattr__(self, key, value)

    @property
    def events(self) -> tuple[Json, ...]:
        return tuple(json.loads(event) for event in self._events)


def _initial(actor: int, episode: str) -> str:
    return digest([INPUT_RECIPE, "event-bank", actor, episode])


def _next(previous: str, event: bytes) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + event).hexdigest()


@dataclass(frozen=True)
class EventBankView:
    actor: int
    episode: str
    _events: tuple[bytes, ...] = field(repr=False)
    _prefixes: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _owner(self.actor, self.episode)
        if (not isinstance(self._events, tuple) or not isinstance(self._prefixes, tuple)
                or len(self._events) > 2**31 - 1 or len(self._prefixes) != len(self._events) + 1
                or self._prefixes[0] != _initial(self.actor, self.episode)):
            raise ContractError("invalid bank view")
        previous = self._prefixes[0]
        for raw, expected in zip(self._events, self._prefixes[1:], strict=True):
            if type(raw) is not bytes:
                raise ContractError("bank events must be owned bytes")
            event = json.loads(raw)
            validate_events([event])
            if _bytes(event) != raw or _next(previous, raw) != expected:
                raise ContractError("event prefix hash mismatch")
            previous = expected

    @property
    def boundary(self) -> EventBoundary:
        return EventBoundary(len(self._events), self._prefixes[-1])

    def events(self, start: int = 0, end: int | None = None) -> tuple[Json, ...]:
        end = len(self._events) if end is None else end
        integer(start, 0, len(self._events))
        integer(end, start, len(self._events))
        return tuple(json.loads(event) for event in self._events[start:end])

    def prefix(self, boundary: EventBoundary) -> EventBankView:
        integer(boundary.end, 0, len(self._events))
        if self._prefixes[boundary.end] != boundary.prefix_hash:
            raise ContractError("bank boundary mismatch")
        return EventBankView(self.actor, self.episode, self._events[:boundary.end],
                             self._prefixes[:boundary.end + 1])


class EventBank:
    def __init__(self, actor: int, episode: str, max_events: int = 262144) -> None:
        _owner(actor, episode)
        integer(max_events, 1, 2**31 - 1)
        self._actor, self._episode, self._max_events = actor, episode, max_events
        self._events: list[bytes] = []
        self._prefixes = [_initial(actor, episode)]
        self._last_frame: DecisionFrame | None = None

    @property
    def actor(self) -> int:
        return self._actor

    @property
    def episode(self) -> str:
        return self._episode

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def boundary(self) -> EventBoundary:
        return EventBoundary(len(self._events), self._prefixes[-1])

    def append(self, delta: EventDelta) -> EventBoundary:
        if ((delta.actor, delta.episode) != (self.actor, self.episode)
                or delta.start != len(self._events) or delta.previous_hash != self._prefixes[-1]):
            raise ContractError("event delta owner/continuity mismatch")
        if len(self._events) + len(delta._events) > self.max_events:
            raise ContractError("episode event cap exceeded")
        for event in delta._events:
            self._prefixes.append(_next(self._prefixes[-1], event))
            self._events.append(event)
        return self.boundary

    def view(self, boundary: EventBoundary | None = None) -> EventBankView:
        boundary = self.boundary if boundary is None else boundary
        integer(boundary.end, 0, len(self._events))
        if self._prefixes[boundary.end] != boundary.prefix_hash:
            raise ContractError("bank boundary mismatch")
        return EventBankView(self.actor, self.episode, tuple(self._events[:boundary.end]),
                             tuple(self._prefixes[:boundary.end + 1]))


@dataclass(frozen=True)
class DecisionFrame:
    actor: int
    episode: str
    revision: int
    interval_start: int
    boundary: EventBoundary
    interval_start_hash: str
    payload: bytes = field(repr=False)
    frame_hash: str

    def __post_init__(self) -> None:
        _owner(self.actor, self.episode)
        integer(self.revision)
        integer(self.interval_start, 0, self.boundary.end)
        if self.boundary.end - self.interval_start > 65536:
            raise ContractError("event interval cap exceeded")
        _hash(self.interval_start_hash)
        _hash(self.frame_hash)
        if type(self.payload) is not bytes:
            raise ContractError("frame payload must be owned bytes")
        value = json.loads(self.payload)
        closed(value, {"observation", "typed_metadata", "catalog", "action_options", "history_end"})
        rebuilt = validate_worker_input(DecisionInput(value["observation"], (), value["history_end"],
                                                     tuple(value["typed_metadata"]), value["catalog"],
                                                     value["action_options"]), True)
        if (rebuilt.observation["viewer_player_id"] != self.actor
                or rebuilt.observation["current_player"] != self.actor
                or not rebuilt.observation["actionable"]
                or rebuilt.observation["state_revision"] != self.revision
                or len(rebuilt.observation["own"]["hand"])
                + len(rebuilt.observation["own"]["learned_miracles"]) > 18
                or len(rebuilt.observation["opponent"]["learned_miracles"]) > 18
                or self.payload != _bytes(value)):
            raise ContractError("frame actor/revision/shape mismatch")
        if any(key != definition["definition_id"] for key, definition in rebuilt.catalog.items()):
            raise ContractError("perceived catalog key/definition mismatch")
        if (rebuilt.observation["own"]["status"]["fog"]
                and set(rebuilt.observation["opponent"]) - {"player_id", "learned_miracles", "ascended"}):
            raise ContractError("Fog opponent visibility mismatch")
        if self.frame_hash != digest(self._binding()):
            raise ContractError("frame hash mismatch")

    def _binding(self) -> Json:
        return {"actor": self.actor, "episode": self.episode, "revision": self.revision,
                "interval_start": self.interval_start, "interval_start_hash": self.interval_start_hash,
                "boundary": asdict(self.boundary), "payload": json.loads(self.payload)}

    @property
    def observation(self) -> Json:
        return json.loads(self.payload)["observation"]

    @property
    def raw(self) -> Json:
        return json.loads(self.payload)

    @property
    def references(self) -> tuple[Json, ...]:
        own = self.observation["own"]
        return tuple([{"source": "hand_artifact", "instance_id": card["instance_id"]}
                      for card in own["hand"]]
                     + [{"source": "learned_miracle", "definition_id": name}
                        for name in own["learned_miracles"]])

    def bank_view(self, bank: EventBankView) -> EventBankView:
        if (bank.actor, bank.episode) != (self.actor, self.episode):
            raise ContractError("foreign event viewer/episode")
        view = bank.prefix(self.boundary)
        if view._prefixes[self.interval_start] != self.interval_start_hash:
            raise ContractError("interval prefix mismatch")
        return view

    @classmethod
    def from_input(cls, value: DecisionInput, bank: EventBank,
                   interval_start: int) -> DecisionFrame:
        value = validate_worker_input(value, True)
        obs = value.observation
        if obs["viewer_player_id"] != bank.actor:
            raise ContractError("foreign input viewer")
        if tuple(_bytes(e) for e in value.events) != tuple(bank._events):
            raise ContractError("complete input history does not match bank")
        previous = bank._last_frame
        if interval_start != (0 if previous is None else previous.boundary.end):
            raise ContractError("decision interval continuity mismatch")
        if previous is not None and obs["state_revision"] <= previous.revision:
            raise ContractError("decision revision must advance")
        integer(interval_start, 0, bank.boundary.end)
        raw = value.to_dict()
        del raw["events"]
        binding = {"actor": bank.actor, "episode": bank.episode, "revision": obs["state_revision"],
                   "interval_start": interval_start, "interval_start_hash": bank._prefixes[interval_start],
                   "boundary": asdict(bank.boundary), "payload": raw}
        frame = cls(bank.actor, bank.episode, obs["state_revision"], interval_start, bank.boundary,
                    bank._prefixes[interval_start], _bytes(raw), digest(binding))
        bank._last_frame = frame
        return frame


@dataclass(frozen=True)
class CandidateTable:
    frame_hash: str
    kinds: tuple[str, ...]
    targets: tuple[str, ...]
    numerics: tuple[tuple[int, ...], ...]
    presence: tuple[tuple[bool, ...], ...]
    offsets: tuple[int, ...]
    slots: tuple[int, ...]
    table_hash: str

    def __post_init__(self) -> None:
        _hash(self.frame_hash)
        _hash(self.table_hash)
        count = len(self.kinds)
        if not 1 <= count <= 65536 or any(type(v) is not tuple for v in
                (self.kinds, self.targets, self.numerics, self.presence, self.offsets, self.slots)):
            raise ContractError("invalid compact candidate columns")
        if not (len(self.targets) == len(self.numerics) == len(self.presence) == count
                and len(self.offsets) == count + 1 and self.offsets[0] == 0
                and self.offsets[-1] == len(self.slots)):
            raise ContractError("candidate shape/offset mismatch")
        for slot in self.slots:
            integer(slot, 0, 17)
        for index, kind in enumerate(self.kinds):
            if kind not in ACTION_KINDS or self.targets[index] not in TARGETS:
                raise ContractError("unknown action kind/relative target")
            integer(self.offsets[index], 0, len(self.slots))
            integer(self.offsets[index + 1], self.offsets[index],
                    min(len(self.slots), self.offsets[index] + 128))
            if (type(self.numerics[index]) is not tuple or type(self.presence[index]) is not tuple
                    or len(self.numerics[index]) != 4 or len(self.presence[index]) != 4):
                raise ContractError("invalid candidate numeric row")
            for number, present in zip(self.numerics[index], self.presence[index], strict=True):
                integer(number, -(2**31), 2**32 - 1)
                if type(present) is not bool or (not present and number != 0):
                    raise ContractError("invalid numeric presence")
        if digest(self._content()) != self.table_hash:
            raise ContractError("candidate table hash mismatch")

    def _content(self) -> Json:
        return {key: value for key, value in asdict(self).items() if key != "table_hash"}

    @classmethod
    def from_actions(cls, frame: DecisionFrame, actions: tuple[Json, ...] | list[Json],
                     max_positions: int = 128) -> CandidateTable:
        integer(max_positions, 1, 128)
        context = _ActionContext.from_frame(frame)
        refs = {canonical(ref): i for i, ref in enumerate(context.references)}
        kinds: list[str] = []
        targets: list[str] = []
        numerics: list[tuple[int, ...]] = []
        presence: list[tuple[bool, ...]] = []
        offsets = [0]
        slots: list[int] = []
        for action in actions:
            _validate_action_context(action, context)
            kind = action["type"]
            elements = _action_refs(action)
            if len(elements) > max_positions:
                raise ContractError("candidate exceeds position capacity")
            try:
                slots.extend(refs[canonical(ref)] for ref in elements)
            except KeyError as exc:
                raise ContractError("candidate contains foreign own reference") from exc
            kinds.append(kind)
            target = ("omitted" if "target" not in action else "null" if action["target"] is None
                      else "self" if action["target"] == frame.actor else "opponent")
            targets.append(target)
            numerics.append(tuple(int(action.get(key, 0)) for key in NUMERIC_FIELDS))
            presence.append(tuple(key in action for key in NUMERIC_FIELDS))
            offsets.append(len(slots))
        content: Json = {"frame_hash": frame.frame_hash, "kinds": tuple(kinds), "targets": tuple(targets),
                   "numerics": tuple(numerics), "presence": tuple(presence),
                   "offsets": tuple(offsets), "slots": tuple(slots)}
        result = cls(**content, table_hash=digest(content))
        result._validate_context(context)
        return result

    def validate(self, frame: DecisionFrame) -> None:
        if self.frame_hash != frame.frame_hash:
            raise ContractError("candidate frame mismatch")
        self._validate_context(_ActionContext.from_frame(frame))

    def _validate_context(self, context: _ActionContext) -> None:
        if any(slot >= len(context.references) for slot in self.slots):
            raise ContractError("candidate slot not in original own view")
        for index in range(len(self.kinds)):
            action = self._to_action(index, context.actor, context.references)
            _validate_action_context(action, context)
            if (tuple(int(action.get(key, 0)) for key in NUMERIC_FIELDS) != self.numerics[index]
                    or tuple(key in action for key in NUMERIC_FIELDS) != self.presence[index]):
                raise ContractError("noncanonical candidate numeric encoding")

    def to_action(self, index: int, original_frame: DecisionFrame, *, validate: bool = True) -> Json:
        if original_frame.frame_hash != self.frame_hash:
            raise ContractError("selection original frame mismatch")
        context = _ActionContext.from_frame(original_frame)
        action = self._to_action(index, context.actor, context.references)
        if validate:
            _validate_action_context(action, context)
        return action

    def _to_action(self, index: int, actor: int, refs: tuple[Json, ...]) -> Json:
        integer(index, 0, len(self.kinds) - 1)
        row = self.slots[self.offsets[index]:self.offsets[index + 1]]
        if any(slot >= len(refs) for slot in row):
            raise ContractError("foreign candidate reference")
        elements = [refs[slot] for slot in row]
        kind = self.kinds[index]
        action: Json = {"type": kind}
        if kind in ("play_sequence", "respond_sequence"):
            action["ordered_elements" if kind == "play_sequence" else "ordered_defenses"] = elements
        elif kind in ("respond", "discard", "sacrifice"):
            if any(ref["source"] != "hand_artifact" for ref in elements):
                raise ContractError("action requires hand references")
            action["ordered_defenses" if kind == "respond" else "instance_ids"] = [e["instance_id"] for e in elements]
        elif kind in ("exchange", "buy", "sell"):
            if len(elements) != (2 if kind == "sell" else 1) or any(e["source"] != "hand_artifact" for e in elements):
                raise ContractError("invalid trade carrier references")
            action["artifact"] = elements[0]["instance_id"]
            if kind == "sell":
                action["sold_artifact"] = elements[1]["instance_id"]
        elif elements:
            raise ContractError("unexpected action references")
        target = self.targets[index]
        if target != "omitted":
            action["target"] = None if target == "null" else actor if target == "self" else 3 - actor
        for key, number, present in zip(NUMERIC_FIELDS, self.numerics[index], self.presence[index], strict=True):
            if present:
                if key == "accept" and number not in (0, 1):
                    raise ContractError("noncanonical buy acceptance encoding")
                action[key] = bool(number) if key == "accept" else number
        return action


def _action_refs(action: Json) -> list[Json]:
    kind = action["type"]
    if kind in ("play_sequence", "respond_sequence"):
        elements = action["ordered_elements" if kind == "play_sequence" else "ordered_defenses"]
        if not isinstance(elements, list):
            raise ContractError("ordered element list required")
        return elements
    keys = ("artifact", "sold_artifact") if kind == "sell" else ("artifact",) if kind in ("buy", "exchange") else ()
    ids = ([action[key] for key in keys] if keys else
           action.get("instance_ids", action.get("ordered_defenses", [])))
    if not isinstance(ids, list):
        raise ContractError("ordered reference list required")
    return [{"source": "hand_artifact", "instance_id": item} for item in ids]


@dataclass(frozen=True)
class _ActionContext:
    actor: int
    references: tuple[Json, ...]
    owned: frozenset[str]
    descriptors: Json

    @classmethod
    def from_frame(cls, frame: DecisionFrame) -> _ActionContext:
        # One call-local owned decode. Never retained across participants or
        # frames, and never exposed as mutable policy/global cache state.
        raw = frame.raw
        own = raw["observation"]["own"]
        references = tuple([{"source": "hand_artifact", "instance_id": card["instance_id"]}
                            for card in own["hand"]]
                           + [{"source": "learned_miracle", "definition_id": name}
                              for name in own["learned_miracles"]])
        return cls(frame.actor, references, frozenset(canonical(ref) for ref in references),
                   {item["type"]: item for item in raw["typed_metadata"]})


def _validate_action_context(action: Json, context: _ActionContext) -> None:
    descriptors = context.descriptors
    if not isinstance(action, dict) or action.get("type") not in descriptors:
        raise ContractError("action kind not in typed metadata")
    desc = descriptors[action["type"]]
    closed(action, {"type"} | set(desc["required_fields"]), set(desc["optional_fields"]))
    for field_name in ("ordered_elements", "ordered_defenses", "instance_ids"):
        if field_name in action and not isinstance(action[field_name], list):
            raise ContractError("ordered action array required")
    for ref in _action_refs(action):
        if canonical(ref) not in context.owned:
            raise ContractError("action reference not owned")
    if "target" in action and action["target"] is not None:
        integer(action["target"], 1, 2)
    if "target" in desc["required_fields"] and action["target"] is None:
        raise ContractError("required target cannot be null")
    for key in ("hp", "mp", "money"):
        if key in action:
            integer(action[key], -(2**31) if key == "hp" else 0,
                    2**31 - 1 if key == "hp" else 2**32 - 1)
    if "accept" in action and type(action["accept"]) is not bool:
        raise ContractError("buy acceptance must be boolean")


@dataclass(frozen=True)
class Selection:
    table_hash: str
    index: int
    policy_version: str
    logprob: float | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        _hash(self.table_hash)
        integer(self.index, 0, 65535)
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ContractError("policy identity required")
        for number in (self.logprob, self.value):
            if number is None:
                continue
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ContractError("nonfinite selection output")
        if self.logprob is not None and self.logprob > 0:
            raise ContractError("positive log probability")

    def validate(self, table: CandidateTable, frame: DecisionFrame) -> Json:
        if self.table_hash != table.table_hash:
            raise ContractError("selection table mismatch")
        table.validate(frame)
        return table.to_action(self.index, frame)
