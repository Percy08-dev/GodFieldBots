"""Rollout3 raw participant banks/tokens with separate natural-trajectory targets.

No latent or repeated candidate-by-frame/card arrays. This is a numeric component
format, not a formal source/parent eligibility or journal authorization shortcut.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.structured import (
    ACTION_KINDS,
    TARGETS,
    CandidateTable,
    DecisionFrame,
    EventBank,
    EventBankView,
    EventBoundary,
    EventDelta,
    Selection,
)
from gf_common.structured_tokens import CandidateTokens, FrameTokens, tokenize_frame, tokenize_table
from gf_common.validation import result as validate_result
from gf_models.arrays import MAX_ARRAY_BYTES
from gf_models.sharded_arrays import (
    MAX_MANIFEST_BYTES,
    create_output_directory,
    inspect_bundle,
    json_bytes,
    plan_bundle,
    read_bundle,
    read_json,
    write_bundle,
    write_owned_json,
)
from gf_models.structured_artifacts import semantics
from gf_models.structured_config import ModelConfigV2, architecture_identity
from gf_training.ppo import gae, normalize_advantages

FRAME_FIELDS = ("state_numerics", "state_present", "state_categories", "state_category_present", "card_numerics",
                "card_present", "card_categories", "card_mask", "opponent_learned", "history")
EVENT_FIELDS = ("interval_numerics", "interval_present", "interval_categories")
CANDIDATE_FIELDS = ("kinds", "targets", "numerics", "present", "offsets", "slots")


def _owned_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Pinned CPython owned-object estimate, with shared immutable refs once."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sys.getsizeof(getattr(value, "__dict__", {})) + sum(_owned_bytes(getattr(value, item.name), seen) for item in fields(value))
    if isinstance(value, (tuple, list)):
        return size + sum(_owned_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return size + sum(_owned_bytes(key, seen) + _owned_bytes(item, seen) for key, item in value.items())
    if isinstance(value, np.ndarray):
        return size + (0 if value.flags.owndata else value.nbytes)
    if value is None or type(value) in (str, bytes, int, float, bool):
        return size
    raise ContractError("unknown owned rollout object type")


def _scalar_count(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    return sum(_scalar_count(item) for item in value) if isinstance(value, tuple) else 1


@dataclass(frozen=True, init=False)
class EpisodeRecords:
    bank: EventBankView
    frames: tuple[DecisionFrame, ...]
    tables: tuple[CandidateTable, ...]
    selections: tuple[Selection, ...]
    end_kind: str
    _outcome: bytes = field(repr=False)
    reason: str | None = None

    def __init__(self, bank: EventBankView, frames: tuple[DecisionFrame, ...],
                 tables: tuple[CandidateTable, ...], selections: tuple[Selection, ...],
                 end_kind: str, outcome: Any = None, reason: str | None = None) -> None:
        for key, value in (("bank", bank), ("frames", frames), ("tables", tables), ("selections", selections),
                           ("end_kind", end_kind), ("_outcome", canonical(outcome).encode()), ("reason", reason)):
            object.__setattr__(self, key, value)
        self.__post_init__()

    @property
    def outcome(self) -> Any:
        return json.loads(self._outcome)

    def __post_init__(self) -> None:
        if type(self.bank) is not EventBankView:
            raise ContractError("owned participant event bank required")
        if not (type(self.frames) is type(self.tables) is type(self.selections) is tuple
                and len(self.frames) == len(self.tables) == len(self.selections)):
            raise ContractError("episode frame/table/selection ownership mismatch")
        if self.end_kind not in ("natural", "truncated", "aborted", "unsupported", "infrastructure_failure"):
            raise ContractError("unknown structured episode closure")
        if self.end_kind == "natural":
            validate_result(self.outcome)
            if self.outcome is None or self.reason is not None:
                raise ContractError("natural episode requires outcome and no interruption reason")
        elif self.outcome is not None or not isinstance(self.reason, str) or not self.reason:
            raise ContractError("interrupted episode requires reason and no invented outcome")
        previous_end, previous_revision = 0, -1
        for frame, table, selection in zip(self.frames, self.tables, self.selections, strict=True):
            frame.bank_view(self.bank)
            if frame.interval_start != previous_end or frame.revision <= previous_revision:
                raise ContractError("episode frame continuity mismatch")
            selection.validate(table, frame)
            previous_end, previous_revision = frame.boundary.end, frame.revision


@dataclass(frozen=True)
class EpisodeTokens:
    records: EpisodeRecords
    frames: tuple[FrameTokens, ...]
    tables: tuple[CandidateTokens, ...]
    advantage: Any
    target: Any


class StructuredRollout:
    def __init__(self, records: tuple[EpisodeRecords, ...], model: ModelConfigV2, action_config_hash: str, *,
                 _reserved_bytes: int = 0) -> None:
        if type(records) is not tuple or any(type(record) is not EpisodeRecords for record in records):
            raise ContractError("owned episode record tuple required")
        if not isinstance(action_config_hash, str) or len(action_config_hash) != 64 or set(action_config_hash) - set("0123456789abcdef"):
            raise ContractError("rollout action hash required")
        integer(_reserved_bytes, 0, model.resources.max_owned_bytes)
        for record in records:
            integer(len(record.frames), 0, model.temporal.max_decisions)
            integer(record.bank.boundary.end, 0, model.event.max_episode)
            for frame, table in zip(record.frames, record.tables, strict=True):
                integer(frame.boundary.end, 0, model.event.max_episode)
                integer(frame.boundary.end - frame.interval_start, 0, model.event.max_interval)
                integer(len(table.kinds), 1, 65536)
                if any(stop - start > model.max_positions for start, stop in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
                    raise ContractError("rollout candidate position capacity exceeded before tokenization")
        # Check a conservative raw-token ownership envelope before tokenization;
        # learned latents are never part of this estimate or stored input.
        predicted_tokens = sum(len(record.frames) * 1000 * 64
                               + (record.frames[-1].boundary.end if record.frames else 0) * 40 * 64
                               for record in records)
        # CandidateTokens makes new kind/target tuple owners even though compact
        # numeric/reference columns are shared. Charge these before tokenization.
        predicted_tokens += sum(len(table.kinds) * 64 + 512 for record in records for table in record.tables)
        if _owned_bytes(records) + predicted_tokens + _reserved_bytes > model.resources.max_owned_bytes:
            raise ContractError("rollout raw/token ownership reservation exceeded before encoding")
        self.model, self.action_config_hash = model, action_config_hash
        episodes = []
        raw_advantages = []
        labels: set[str] = set()
        self.locations: list[tuple[int, int]] = []
        owners: set[tuple[int, str]] = set()
        for episode_index, record in enumerate(records):
            owner = (record.bank.actor, record.bank.episode)
            if owner in owners:
                raise ContractError("duplicate participant episode bank")
            owners.add(owner)
            integer(len(record.frames), 0, model.temporal.max_decisions)
            if record.bank.boundary.end > model.event.max_episode:
                raise ContractError("rollout episode event cap exceeded")
            frame_tokens = tuple(tokenize_frame(frame, record.bank, aggregate_window=model.event.aggregate_window) for frame in record.frames)
            candidates = tuple(tokenize_table(table, frame, model.max_positions, aggregate_window=model.event.aggregate_window)
                               for frame, table in zip(record.frames, record.tables, strict=True))
            for selection in record.selections:
                labels.add(selection.policy_version)
            if record.end_kind == "natural" and record.frames:
                reward = (0 if record.outcome == "draw" else 1 if record.outcome["win"]["winner"] == record.bank.actor else -1)
                advantage, target = gae([selection.value for selection in record.selections], reward)
                raw_advantages.extend(advantage.tolist())
                target = target.astype(np.float32)
            else:
                advantage, target = None, None
            # Preserve accepted own-decision endpoints even for interrupted
            # episodes. Their absence of targets forbids training/bootstrapping.
            self.locations.extend((episode_index, index) for index in range(len(record.frames)))
            episodes.append(EpisodeTokens(record, frame_tokens, candidates, advantage, target))
        if len(labels) > 1:
            raise ContractError("rollout mixes frozen policy identities")
        self.policy_version = next(iter(labels), None)
        normalized = normalize_advantages(raw_advantages) if raw_advantages else np.empty(0, np.float32)
        offset = 0
        output = []
        for episode in episodes:
            if episode.advantage is not None:
                count = len(episode.frames)
                advantage = normalized[offset:offset + count].copy()
                offset += count
                advantage.flags.writeable = False
                episode.target.flags.writeable = False
                episode = EpisodeTokens(episode.records, episode.frames, episode.tables, advantage, episode.target)
            output.append(episode)
        self.episodes = tuple(output)
        self.eligible = all(episode.records.end_kind == "natural" for episode in self.episodes) and bool(self.locations)
        self.resident_bytes = _owned_bytes((self.model, self.episodes, self.locations, self.action_config_hash, self.policy_version))
        if self.resident_bytes + _reserved_bytes > model.resources.max_owned_bytes:
            raise ContractError("rollout owned memory reservation exceeded")

    def __len__(self) -> int:
        return len(self.locations)

    def require_learning(self) -> None:
        if not self.eligible:
            raise ContractError("natural nonempty complete rollout required; interrupted sources cannot bootstrap")

    def endpoint(self, index: int) -> tuple[tuple[FrameTokens, ...], CandidateTokens]:
        integer(index, 0, len(self.locations) - 1)
        episode_index, position = self.locations[index]
        episode = self.episodes[episode_index]
        return episode.frames[:position + 1], episode.tables[position]

    def vector(self, index: int) -> tuple[int, float, float, float, float]:
        episode_index, position = self.locations[index]
        episode = self.episodes[episode_index]
        if episode.advantage is None or episode.target is None:
            raise ContractError("interrupted endpoint has no learning targets")
        selection = episode.records.selections[position]
        return selection.index, float(selection.logprob), float(selection.value), float(episode.advantage[position]), float(episode.target[position])


def _array(value: Any, dtype: Any) -> Any:
    # Shape is known raw token state, never an untrusted descriptor. Check before
    # constructing the NumPy buffer; large candidate refs are chunk/shard bounded.
    if _scalar_count(value) * np.dtype(dtype).itemsize > MAX_ARRAY_BYTES:
        raise ContractError("rollout tensor exceeds64MiB before allocation")
    return np.asarray(value, dtype=dtype)


def _token_dtype(name: str) -> Any:
    return np.uint8 if "present" in name or "mask" in name else np.int64


def _blob(value: bytes) -> Any:
    if len(value) > MAX_ARRAY_BYTES:
        raise ContractError("rollout raw record exceeds64MiB before allocation")
    return np.frombuffer(value, dtype=np.uint8).copy()


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    if not isinstance(value, tuple):
        return ()
    if not value:
        return (0,)
    shape = _shape(value[0])
    if any(_shape(item) != shape for item in value[1:]):
        raise ContractError("ragged raw token shape")
    return (len(value), *shape)


def write_rollout(directory: Path, rollout: StructuredRollout, *, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    create_output_directory(directory, "rollout.json")
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    planned: dict[str, tuple[Any, Any, bool]] = {}
    planned_bytes = 0
    planning_bytes = (65536 * (1 + sum(len(episode.frames) for episode in rollout.episodes))
                      + 32 * sum(_owned_bytes(episode.records.bank) for episode in rollout.episodes))
    if rollout.resident_bytes + planning_bytes > rollout.model.resources.max_owned_bytes:
        raise ContractError("rollout metadata planning exceeds reservation")
    def plan(key: str, value: Any, dtype: Any, *, blob: bool = False) -> None:
        nonlocal planned_bytes, planning_bytes
        size = len(value) if blob else _scalar_count(value) * np.dtype(dtype).itemsize
        if size > shard_bytes:
            raise ContractError("rollout tensor exceeds shard bound before allocation")
        planned_bytes += size
        planning_bytes += 1024 + len(key.encode())
        if rollout.resident_bytes + planning_bytes + 2 * planned_bytes > rollout.model.resources.max_owned_bytes:
            raise ContractError("rollout packing plus owned inputs exceeds reservation")
        planned[key] = (value, dtype, blob)
    episodes: list[Json] = []
    for episode_index, episode in enumerate(rollout.episodes):
        record = episode.records
        prefix = f"e.{episode_index}"
        raw_blocks = []
        # Raw participant events are stored once in bounded blocks. Later frames
        # reference exact prefixes; terminal/future tail never enters earlier inputs.
        for start in range(0, record.bank.boundary.end, 1024):
            end = min(start + 1024, record.bank.boundary.end)
            key = prefix + f".raw-events.{start}"
            plan(key, canonical(record.bank.events(start, end)).encode(), np.uint8, blob=True)
            raw_blocks.append({"start": start, "end": end, "key": key})
        frame_records = []
        for position, (frame, tokens, table, candidates) in enumerate(zip(record.frames, episode.frames, record.tables, episode.tables, strict=True)):
            frame_prefix = prefix + f".f.{position}"
            payload_key = frame_prefix + ".payload"
            plan(payload_key, frame.payload, np.uint8, blob=True)
            for name in FRAME_FIELDS:
                plan(frame_prefix + "." + name, getattr(tokens, name), _token_dtype(name))
            event_prefix = prefix + f".event-tokens.{position}"
            for name in EVENT_FIELDS:
                plan(event_prefix + "." + name, getattr(tokens, name), _token_dtype(name))
            for name in CANDIDATE_FIELDS:
                plan(frame_prefix + ".candidate." + name, getattr(candidates, name), _token_dtype(name))
            frame_records.append({"actor": frame.actor, "episode": frame.episode, "revision": frame.revision,
                                  "interval_start": frame.interval_start, "boundary": asdict(frame.boundary),
                                  "interval_start_hash": frame.interval_start_hash, "frame_hash": frame.frame_hash,
                                  "payload_key": payload_key, "token_prefix": frame_prefix,
                                  "event_token_prefix": event_prefix, "table_hash": table.table_hash})
        for name, values in (("selected", tuple(selection.index for selection in record.selections)),
                             ("old_logprob", tuple(selection.logprob for selection in record.selections)),
                             ("old_value", tuple(selection.value for selection in record.selections))):
            plan(prefix + "." + name, values, np.int64 if name == "selected" else np.float32)
        if episode.advantage is not None:
            plan(prefix + ".advantage", episode.advantage, np.float32)
            plan(prefix + ".target", episode.target, np.float32)
        episodes.append({"actor": record.bank.actor, "episode": record.bank.episode,
                         "boundary": asdict(record.bank.boundary), "raw_blocks": raw_blocks,
                         "frames": frame_records, "end_kind": record.end_kind,
                         "outcome": record.outcome, "reason": record.reason,
                         "selections": [asdict(selection) for selection in record.selections],
                         "has_targets": episode.advantage is not None})
    if not planned:
        # Zero-decision episode closure is preserved without synthetic transition.
        plan("empty", (), np.uint8)
    descriptors = {key: {"dtype": str(np.dtype(dtype)), "shape": [len(value)] if blob else list(_shape(value)),
                         "bytes": len(value) if blob else _scalar_count(value) * np.dtype(dtype).itemsize}
                   for key, (value, dtype, blob) in planned.items()}
    bundle_plan = plan_bundle("rollout", descriptors, shard_bytes=shard_bytes)
    manifest = {"schema_version": 3, "kind": "structured-rollout", **semantics(rollout.model),
                "architecture_identity": architecture_identity(rollout.model),
                "action_config_hash": rollout.action_config_hash, "policy_version": rollout.policy_version,
                "decision_count": len(rollout), "episodes": episodes, "data": bundle_plan}
    metadata_bytes = json_bytes(manifest, limit=MAX_MANIFEST_BYTES)
    if (rollout.resident_bytes + planning_bytes + 2 * planned_bytes
            + 32 * metadata_bytes + 65536 > rollout.model.resources.max_owned_bytes):
        raise ContractError("complete rollout JSON/packing reservation exceeded")
    arrays = {key: _blob(value) if blob else _array(value, dtype)
              for key, (value, dtype, blob) in planned.items()}
    manifest["data"] = write_bundle(directory, "rollout", arrays, shard_bytes=shard_bytes)
    if json_bytes(manifest, limit=MAX_MANIFEST_BYTES) != metadata_bytes:
        raise ContractError("actual complete manifest differs from admitted size plan")
    write_owned_json(directory / "rollout.json", manifest)
    return manifest


def _read_reservation(manifest: Json, descriptors: Json) -> int:
    """Conservative verified-input reconstruction envelope, not an RSS bound.

    Reuse the project's32x raw JSON +64KiB/document admission policy. Charge
    retained decoded arrays, two shard scratch copies, Python token/ref owners
    and fixed per-frame state before decoding or parsing any raw frame/event.
    """
    decoded = sum(item["bytes"] for item in descriptors.values())
    raw = [item for name, item in descriptors.items() if name.endswith(".payload") or ".raw-events." in name]
    raw_bytes = sum(item["bytes"] for item in raw)
    largest_shard = max(sum(item["bytes"] for item in shard["metadata"]["arrays"].values())
                        for shard in manifest["data"]["shards"])
    metadata_bytes = json_bytes(manifest, limit=MAX_MANIFEST_BYTES)
    frame_count = sum(len(episode["frames"]) for episode in manifest["episodes"])
    return (decoded + 2 * largest_shard + 32 * (raw_bytes + metadata_bytes)
            + 65536 * (len(raw) + 1) + 8 * (decoded - raw_bytes) + 65536 * frame_count)


def read_rollout(directory: Path, expected_digest: str, model: ModelConfigV2) -> StructuredRollout:
    manifest = read_json(directory / "rollout.json", expected_digest=expected_digest,
                         max_owned_bytes=model.resources.max_owned_bytes)
    closed(manifest, {"schema_version", "kind", "architecture_identity", "action_config_hash", "policy_version",
                      "decision_count", "episodes", "data"} | set(semantics(model)))
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 3
            or manifest["kind"] != "structured-rollout"
            or manifest["architecture_identity"] != architecture_identity(model)
            or any(canonical(manifest[key]) != canonical(value) for key, value in semantics(model).items())):
        raise ContractError("rollout recipe/architecture/runtime mismatch")
    integer(manifest["decision_count"], 0, 2**31 - 1)
    if not isinstance(manifest["episodes"], list):
        raise ContractError("episode records required")
    for episode in manifest["episodes"]:
        closed(episode, {"actor", "episode", "boundary", "raw_blocks", "frames", "end_kind", "outcome", "reason", "selections", "has_targets"})
        if not isinstance(episode["frames"], list) or not isinstance(episode["selections"], list) or len(episode["frames"]) != len(episode["selections"]):
            raise ContractError("episode frame/selection metadata shape mismatch")
        integer(len(episode["frames"]), 0, model.temporal.max_decisions)
        boundary_meta = closed(episode["boundary"], {"end", "prefix_hash"})
        integer(boundary_meta["end"], 0, model.event.max_episode)
    descriptors = inspect_bundle(directory, manifest["data"], max_decoded_bytes=model.resources.max_owned_bytes)
    if _read_reservation(manifest, descriptors) > model.resources.max_owned_bytes:
        raise ContractError("joint rollout decode/raw-owner reservation exceeded before reconstruction")
    arrays = read_bundle(directory, manifest["data"], max_decoded_bytes=model.resources.max_owned_bytes)
    used: set[str] = set()
    def get(key: str, dtype: Any) -> Any:
        if not isinstance(key, str) or key not in arrays or key in used or arrays[key].dtype != np.dtype(dtype):
            raise ContractError("rollout array key/dtype/ownership mismatch")
        used.add(key)
        return arrays[key]
    records = []
    stored_tokens: list[tuple[Json, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    train_vectors = []
    if not isinstance(manifest["episodes"], list):
        raise ContractError("episode records required")
    for episode_index, metadata in enumerate(manifest["episodes"]):
        closed(metadata, {"actor", "episode", "boundary", "raw_blocks", "frames", "end_kind", "outcome", "reason", "selections", "has_targets"})
        boundary = EventBoundary(**closed(metadata["boundary"], {"end", "prefix_hash"}))
        bank = EventBank(metadata["actor"], metadata["episode"], max_events=max(1, boundary.end))
        prefix = f"e.{episode_index}"
        for block in metadata["raw_blocks"]:
            closed(block, {"start", "end", "key"})
            if block["key"] != prefix + f".raw-events.{block['start']}" or block["start"] != bank.boundary.end:
                raise ContractError("event bank block continuity mismatch")
            raw = get(block["key"], np.uint8)
            if raw.ndim != 1:
                raise ContractError("raw event bytes must be a vector")
            events = json.loads(raw.tobytes())
            if canonical(events).encode() != raw.tobytes() or len(events) != block["end"] - block["start"]:
                raise ContractError("raw event block shape/canonical mismatch")
            bank.append(EventDelta(bank.actor, bank.episode, block["start"], bank.boundary.prefix_hash, events))
        if bank.boundary != boundary:
            raise ContractError("complete episode bank hash mismatch")
        frames, tables = [], []
        for position, frame_meta in enumerate(metadata["frames"]):
            closed(frame_meta, {"actor", "episode", "revision", "interval_start", "boundary", "interval_start_hash",
                               "frame_hash", "payload_key", "token_prefix", "event_token_prefix", "table_hash"})
            frame_prefix, event_prefix = prefix + f".f.{position}", prefix + f".event-tokens.{position}"
            if (frame_meta["payload_key"] != frame_prefix + ".payload" or frame_meta["token_prefix"] != frame_prefix
                    or frame_meta["event_token_prefix"] != event_prefix):
                raise ContractError("frame/token ownership path mismatch")
            payload = get(frame_meta["payload_key"], np.uint8)
            if payload.ndim != 1:
                raise ContractError("frame payload must be byte vector")
            frame = DecisionFrame(frame_meta["actor"], frame_meta["episode"], frame_meta["revision"],
                                  frame_meta["interval_start"], EventBoundary(**frame_meta["boundary"]),
                                  frame_meta["interval_start_hash"], payload.tobytes(), frame_meta["frame_hash"])
            frame_arrays = {name: get(frame_prefix + "." + name, _token_dtype(name)) for name in FRAME_FIELDS}
            event_arrays = {name: get(event_prefix + "." + name, _token_dtype(name)) for name in EVENT_FIELDS}
            candidate_arrays = {name: get(frame_prefix + ".candidate." + name, _token_dtype(name)) for name in CANDIDATE_FIELDS}
            for code in candidate_arrays["kinds"]:
                integer(int(code), 2, len(ACTION_KINDS) + 1)
            for code in candidate_arrays["targets"]:
                integer(int(code), 2, len(TARGETS) + 1)
            if not np.isin(candidate_arrays["present"], (0, 1)).all():
                raise ContractError("candidate presence must be0/1")
            table = CandidateTable(frame.frame_hash, tuple(ACTION_KINDS[int(code) - 2] for code in candidate_arrays["kinds"]),
                tuple(TARGETS[int(code) - 2] for code in candidate_arrays["targets"]),
                tuple(tuple(int(value) for value in row) for row in candidate_arrays["numerics"]),
                tuple(tuple(bool(value) for value in row) for row in candidate_arrays["present"]),
                tuple(int(value) for value in candidate_arrays["offsets"]), tuple(int(value) for value in candidate_arrays["slots"]), frame_meta["table_hash"])
            frames.append(frame)
            tables.append(table)
            stored_tokens.append((frame_meta, frame_arrays, event_arrays, candidate_arrays))
        selections = tuple(Selection(**value) for value in metadata["selections"])
        record = EpisodeRecords(bank.view(), tuple(frames), tuple(tables), selections, metadata["end_kind"], metadata["outcome"], metadata["reason"])
        records.append(record)
        count = len(frames)
        scalar = {}
        for name in ("selected", "old_logprob", "old_value"):
            value = get(prefix + "." + name, np.int64 if name == "selected" else np.float32)
            if value.shape != (count,):
                raise ContractError("rollout scalar shape mismatch")
            scalar[name] = value
        if type(metadata["has_targets"]) is not bool:
            raise ContractError("target presence must be boolean")
        for name in ("advantage", "target") if metadata["has_targets"] else ():
            scalar[name] = get(prefix + "." + name, np.float32)
            if scalar[name].shape != (count,):
                raise ContractError("rollout target shape mismatch")
        train_vectors.append(scalar)
    rollout = StructuredRollout(tuple(records), model, manifest["action_config_hash"],
                                _reserved_bytes=manifest["data"]["decoded_bytes"])
    if manifest["decision_count"] != len(rollout) or manifest["policy_version"] != rollout.policy_version:
        raise ContractError("rollout endpoint/policy identity mismatch")
    for episode, scalars in zip(rollout.episodes, train_vectors, strict=True):
        for name, source in (("selected", [s.index for s in episode.records.selections]),
                             ("old_logprob", [s.logprob for s in episode.records.selections]),
                             ("old_value", [s.value for s in episode.records.selections])):
            if not np.array_equal(scalars[name], np.asarray(source, dtype=scalars[name].dtype)):
                raise ContractError("selection/scalar binding mismatch")
        if (episode.advantage is not None) != ("advantage" in scalars):
            raise ContractError("natural/interrupted target presence mismatch")
        if episode.advantage is not None and (not np.array_equal(scalars["advantage"], episode.advantage)
                                               or not np.array_equal(scalars["target"], episode.target)):
            raise ContractError("GAE/normalized target mismatch")
    expected_tokens = [(frame, table) for episode in rollout.episodes for frame, table in zip(episode.frames, episode.tables, strict=True)]
    for (tokens, candidates), (_, frame_arrays, event_arrays, candidate_arrays) in zip(expected_tokens, stored_tokens, strict=True):
        for values, names, owner in ((frame_arrays, FRAME_FIELDS, tokens), (event_arrays, EVENT_FIELDS, tokens),
                                     (candidate_arrays, CANDIDATE_FIELDS, candidates)):
            for name in names:
                expected = _array(getattr(owner, name), _token_dtype(name))
                if values[name].shape != expected.shape or not np.array_equal(values[name], expected):
                    raise ContractError("stored tokens differ from original participant frame/bank")
    if "empty" in arrays and not used:
        if arrays["empty"].dtype != np.uint8 or arrays["empty"].shape != (0,):
            raise ContractError("invalid empty rollout marker")
        used.add("empty")
    if used != set(arrays):
        raise ContractError("unowned/unknown rollout tensor")
    return rollout
