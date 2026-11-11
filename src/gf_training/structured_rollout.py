"""Rollout3 raw participant banks/tokens with separate natural-trajectory targets.

No latent or repeated candidate-by-frame/card arrays. This is a numeric component
format, not a formal source/parent eligibility or journal authorization shortcut.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.action_contract import ActionContract
from gf_common.card_instance import ActionConfigV3, CardInstanceContract
from gf_common.card_instance_rules import CardInstanceRules, card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.structured import (
    CandidateTable,
    DecisionFrame,
    EventBank,
    EventBankView,
    EventBoundary,
    Selection,
    _CorrectedCardInstanceEventBankView,
    _frame_type,
    event_delta,
)
from gf_common.structured_layout import (
    CORRECTED_CARD_INSTANCE_LAYOUT,
    LEGACY_LAYOUT,
    StructuredLayout,
)
from gf_common.structured_tokens import CandidateTokens, FrameTokens, tokenize_frame, tokenize_table
from gf_common.validation import result as validate_result
from gf_models.arrays import MAX_ARRAY_BYTES, _source_guard
from gf_models.sharded_arrays import (
    MAX_MANIFEST_BYTES,
    create_output_directory,
    inspect_bundle,
    json_bytes,
    plan_bundle,
    read_bundle,
    read_json,
    validate_bundle_metadata,
    write_bundle,
    write_owned_json,
)
from gf_models.structured_artifacts import semantics, semantics_v3
from gf_models.structured_binding import check_input_binding, input_binding, input_binding_game_v3
from gf_models.structured_config import (
    ModelConfigV3,
    aggregate_recipe,
    recorded_architecture_identity,
    validate_config,
)
from gf_models.structured_sealed import (
    SEMANTIC_KEYS,
    SEMANTIC_KEYS_V3,
    validate_recorded_semantics,
    validate_recorded_semantics_v3,
)
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


@dataclass(frozen=True)
class _RolloutContext:
    actions: ActionConfigV3
    rules: CardInstanceRules
    layout: StructuredLayout = CORRECTED_CARD_INSTANCE_LAYOUT

    def __post_init__(self) -> None:
        input_binding_game_v3(self.actions.action_contract, self.rules)
        if self.layout is not CORRECTED_CARD_INSTANCE_LAYOUT or self.actions.max_sequence_length > 18:
            raise ContractError('rollout4 requires corrected real18 game context')

    def classification(self) -> Json:
        return input_binding_game_v3(self.actions.action_contract, self.rules)

    def check(self, raw: Json) -> None:
        if canonical(raw) != canonical(self.classification()):
            raise ContractError('rollout4 game classification mismatch')


def _record_binding(contract: Any, purpose: str, synthetic: bool) -> Json:
    if type(contract) is CardInstanceContract:
        expected = input_binding_game_v3(contract, card_instance_rules())
        if purpose != 'qualified_game' or synthetic is not False:
            raise ContractError('game episode binding mismatch')
        return expected
    return input_binding(contract, purpose, synthetic)


@dataclass(frozen=True, init=False)
class EpisodeRecords:
    action_contract: ActionContract | CardInstanceContract
    purpose: str
    synthetic_capacity: bool
    bank: EventBankView
    frames: tuple[DecisionFrame, ...]
    tables: tuple[CandidateTable, ...]
    selections: tuple[Selection, ...]
    end_kind: str
    _outcome: bytes = field(repr=False)
    reason: str | None = None

    def __init__(self, bank: EventBankView, frames: tuple[DecisionFrame, ...],
                 tables: tuple[CandidateTable, ...], selections: tuple[Selection, ...],
                 end_kind: str, outcome: Any = None, reason: str | None = None, *,
                 action_contract: Any, purpose: str, synthetic_capacity: bool,
                 _context: _RolloutContext | None = None) -> None:
        if _context is None:
            input_binding(action_contract, purpose, synthetic_capacity)
        elif action_contract != _context.actions.action_contract or _record_binding(action_contract, purpose, synthetic_capacity) != _context.classification():
            raise ContractError("game episode context mismatch")
        for key, value in (("action_contract", action_contract), ("purpose", purpose), ("synthetic_capacity", synthetic_capacity), ("bank", bank), ("frames", frames), ("tables", tables), ("selections", selections),
                           ("end_kind", end_kind), ("_outcome", canonical(outcome).encode()), ("reason", reason)):
            object.__setattr__(self, key, value)
        self.__post_init__()

    @property
    def outcome(self) -> Any:
        return json.loads(self._outcome)

    def __post_init__(self) -> None:
        if type(self.bank) is not (_CorrectedCardInstanceEventBankView if type(self.action_contract) is CardInstanceContract else EventBankView):
            raise ContractError("owned participant event bank required")
        if not (type(self.frames) is type(self.tables) is type(self.selections) is tuple
                and len(self.frames) == len(self.tables) == len(self.selections)):
            raise ContractError("episode frame/table/selection ownership mismatch")
        if self.end_kind not in ("natural", "truncated", "aborted", "cancelled", "incomplete", "unsupported", "infrastructure_failure"):
            raise ContractError("unknown structured episode closure")
        if self.end_kind == "natural":
            validate_result(self.outcome)
            if self.outcome is None or self.reason is not None:
                raise ContractError("natural episode requires outcome and no interruption reason")
        elif self.outcome is not None or not isinstance(self.reason, str) or not self.reason:
            raise ContractError("interrupted episode requires reason and no invented outcome")
        previous_end, previous_revision = 0, -1
        for frame, table, selection in zip(self.frames, self.tables, self.selections, strict=True):
            if _record_binding(table.action_contract, table.purpose, table.synthetic_capacity) != _record_binding(self.action_contract, self.purpose, self.synthetic_capacity):
                raise ContractError("episode table binding mismatch")
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


def _predicted_token_bytes(records: tuple[EpisodeRecords, ...]) -> int:
    return (sum(len(record.frames) * 1000 * 64
                + (record.frames[-1].boundary.end if record.frames else 0) * 40 * 64 for record in records)
            + sum(len(table.kinds) * 64 + 512 for record in records for table in record.tables))


class StructuredRollout:
    def __init__(self, records: tuple[EpisodeRecords, ...], model: Any, action_config_hash: str, *,
                 action_contract: Any, purpose: str, synthetic_capacity: bool, _reserved_bytes: int = 0,
                 _historical: bool = False, _context: _RolloutContext | None = None) -> None:
        self._context = _context
        self.layout = LEGACY_LAYOUT if _context is None else _context.layout
        if not _historical:
            validate_config(model, layout=self.layout)
        self.input_binding = (input_binding(action_contract, purpose, synthetic_capacity) if _context is None else _context.classification())
        if _context is not None and (type(model) is not ModelConfigV3 or action_config_hash != digest(asdict(_context.actions))
                or action_contract != _context.actions.action_contract or purpose != "qualified_game" or synthetic_capacity is not False):
            raise ContractError("rollout4 model/actions/game context mismatch")
        self.action_contract, self.purpose, self.synthetic_capacity = action_contract, purpose, synthetic_capacity
        if type(records) is not tuple or any(type(record) is not EpisodeRecords for record in records):
            raise ContractError("owned episode record tuple required")
        if not isinstance(action_config_hash, str) or len(action_config_hash) != 64 or set(action_config_hash) - set("0123456789abcdef"):
            raise ContractError("rollout action hash required")
        integer(_reserved_bytes, 0, model.resources.max_owned_bytes)
        for record in records:
            if _record_binding(record.action_contract, record.purpose, record.synthetic_capacity) != self.input_binding:
                raise ContractError("rollout episode binding mismatch")
            if any(selection.logprob is None or selection.value is None for selection in record.selections):
                raise ContractError("PPO learner rows require finite logprob and value before ingestion")
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
        # CandidateTokens makes new kind/target tuple owners even though compact
        # numeric/reference columns are shared. Charge these before tokenization.
        predicted_tokens = _predicted_token_bytes(records)
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
            frame_tokens = tuple(tokenize_frame(frame, record.bank, aggregate_window=model.event.aggregate_window,
                                                aggregate_recipe=aggregate_recipe(model), layout=self.layout) for frame in record.frames)
            candidates = tuple(tokenize_table(table, frame, model.max_positions, aggregate_window=model.event.aggregate_window,
                                              aggregate_recipe=aggregate_recipe(model), layout=self.layout)
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
        self.resident_bytes = _owned_bytes((self.model, self.episodes, self.locations, self.action_config_hash, self.policy_version, self.input_binding))
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
        if selection.logprob is None or selection.value is None:
            raise ContractError("PPO learner row lacks logprob/value")
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


def _episode_correspondence_digest(record: EpisodeRecords) -> str:
    """Exact original raw-record binding; supplies no Source/parent authority."""
    return digest({"recipe": "structured-source-episode-correspondence-v1",
        **_record_binding(record.action_contract, record.purpose, record.synthetic_capacity),
        "actor": record.bank.actor, "episode": record.bank.episode, "boundary": asdict(record.bank.boundary),
        "frames": [{"frame_hash": frame.frame_hash, "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
                    "revision": frame.revision, "interval_start": frame.interval_start,
                    "interval_start_hash": frame.interval_start_hash, "boundary": asdict(frame.boundary),
                    "table_hash": table.table_hash, "selection": asdict(selection)}
                   for frame, table, selection in zip(record.frames, record.tables, record.selections, strict=True)],
        "end_kind": record.end_kind, "outcome": record.outcome, "reason": record.reason})


def _source_correspondence(control_fingerprint: str, declaration: Json, episode_digests: tuple[str, ...], *, game: bool = False, storage_binding: Json | None = None, hpo: bool = False) -> Json:
    """Build a bounded ephemeral cohort binding from an ORIGINAL native owner.

    Callers must obtain the digests/fingerprint from their validated original
    Source owner. This value is not persisted and cannot authorize an input.
    """
    from gf_models.structured_sealed import checksum
    checksum(control_fingerprint)
    closed(declaration, SEMANTIC_KEYS_V3 - {'reward_policy_version'} if hpo else SEMANTIC_KEYS_V3 if game else SEMANTIC_KEYS)
    if type(episode_digests) is not tuple:
        raise ContractError('ordered original episode digest tuple required')
    for value in episode_digests:
        checksum(value)
    if storage_binding is not None:
        from gf_common.group_pack_game import checked_storage_binding
        checked_storage_binding(storage_binding)
        if not game:
            raise ContractError('packed correspondence requires corrected game context')
    content = {"recipe": ('card-instance-hpo-packed-source-correspondence-v1' if storage_binding is not None else 'card-instance-hpo-source-correspondence-v1') if hpo else "card-instance-packed-source-correspondence-v1" if storage_binding is not None else "structured-source-cohort-correspondence-v2" if game else "structured-source-cohort-correspondence-v1", "control_fingerprint": control_fingerprint,
               "semantics_sha256": digest(declaration), "episode_count": len(episode_digests),
               "episode_digests": episode_digests}
    return {**content, "binding_sha256": digest(content)}


def _check_source_correspondence(value: Json, declaration: Json, *, game: bool = False,
                                 storage_binding: Json | None = None, hpo: bool = False) -> tuple[str, ...]:
    from gf_models.structured_sealed import checksum
    closed(value, {'recipe', 'control_fingerprint', 'semantics_sha256', 'episode_count', 'episode_digests', 'binding_sha256'})
    if storage_binding is not None:
        from gf_common.group_pack_game import checked_storage_binding
        checked_storage_binding(storage_binding)
        if not game:
            raise ContractError('packed correspondence requires corrected game context')
    expected_recipe = ('card-instance-hpo-packed-source-correspondence-v1' if storage_binding is not None else 'card-instance-hpo-source-correspondence-v1') if hpo else ('card-instance-packed-source-correspondence-v1' if storage_binding is not None else 'structured-source-cohort-correspondence-v2' if game else 'structured-source-cohort-correspondence-v1')
    if value['recipe'] != expected_recipe or value['semantics_sha256'] != digest(declaration):
        raise ContractError('original Source correspondence semantic recipe mismatch')
    for key in ('control_fingerprint', 'semantics_sha256', 'binding_sha256'):
        checksum(value[key])
    integer(value['episode_count'], 0)
    if not isinstance(value['episode_digests'], (tuple, list)) or len(value['episode_digests']) != value['episode_count']:
        raise ContractError('original Source correspondence count/order required')
    for item in value['episode_digests']:
        checksum(item)
    if digest({key: item for key, item in value.items() if key != 'binding_sha256'}) != value['binding_sha256']:
        raise ContractError('original Source correspondence fingerprint binding mismatch')
    return tuple(value['episode_digests'])


def write_rollout(directory: Path, rollout: StructuredRollout, *, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    if type(rollout) is not StructuredRollout:
        raise ContractError('legacy rollout3 writer requires its closed StructuredRollout')
    if rollout._context is not None:
        raise ContractError("legacy rollout writer rejects game4")
    return _write_rollout(directory, rollout, shard_bytes=shard_bytes)


def _legacy_table_contract(table: CandidateTable) -> ActionContract:
    if type(table.action_contract) is not ActionContract:
        raise ContractError("numeric rollout3 requires an exact legacy ActionContract")
    return table.action_contract


def _write_rollout(directory: Path, rollout: StructuredRollout, *, shard_bytes: int = MAX_ARRAY_BYTES,
                   extra_owned_bytes: int = 0, max_owned_bytes: int | None = None,
                   check: Callable[[], None] | None = None,
                   reserve: Callable[[int], None] = lambda _: None, _hpo: Json | None = None) -> Json:
    poll = check or (lambda: None)
    context = rollout._context
    if (rollout.model.version == '4') != (_hpo is not None):
        raise ContractError('numeric5 requires its explicit complete selection context')
    validate_config(rollout.model, layout=rollout.layout)
    for episode in rollout.episodes:
        for table in episode.records.tables:
            if context is None:
                _legacy_table_contract(table)  # Reject new semantics before output/admission/planning.
            elif table.action_contract != context.actions.action_contract:
                raise ContractError("rollout4 table contract mismatch")
    poll()
    limit = rollout.model.resources.max_owned_bytes if max_owned_bytes is None else min(max_owned_bytes, rollout.model.resources.max_owned_bytes)
    integer(extra_owned_bytes, 0, limit)
    def admit(size: int) -> None:
        poll()
        if size + extra_owned_bytes > limit:
            raise ContractError('joint rollout planning/packing/publication reservation exceeded')
        reserve(size)
    if context is None:
        check_input_binding(rollout.input_binding, action_contract=rollout.action_contract,
                            purpose=rollout.purpose, synthetic_capacity=rollout.synthetic_capacity)
    else:
        context.check(rollout.input_binding)
    create_output_directory(directory, "rollout.json")
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    planned: dict[str, tuple[Any, Any, bool]] = {}
    planned_bytes = 0
    planning_bytes = (65536 * (1 + sum(len(episode.frames) for episode in rollout.episodes))
                      + 32 * sum(_owned_bytes(episode.records.bank) for episode in rollout.episodes))
    admit(rollout.resident_bytes + planning_bytes)
    def plan(key: str, value: Any, dtype: Any, *, blob: bool = False) -> None:
        poll()
        nonlocal planned_bytes, planning_bytes
        size = len(value) if blob else _scalar_count(value) * np.dtype(dtype).itemsize
        if size > shard_bytes:
            raise ContractError("rollout tensor exceeds shard bound before allocation")
        planned_bytes += size
        planning_bytes += 1024 + len(key.encode())
        admit(rollout.resident_bytes + planning_bytes + 2 * planned_bytes)
        planned[key] = (value, dtype, blob)
    episodes: list[Json] = []
    for episode_index, episode in enumerate(rollout.episodes):
        poll()
        record = episode.records
        prefix = f"e.{episode_index}"
        raw_blocks = []
        # Raw participant events are stored once in bounded blocks. Later frames
        # reference exact prefixes; terminal/future tail never enters earlier inputs.
        for start in range(0, record.bank.boundary.end, 1024):
            poll()
            end = min(start + 1024, record.bank.boundary.end)
            key = prefix + f".raw-events.{start}"
            plan(key, canonical(record.bank.events(start, end)).encode(), np.uint8, blob=True)
            raw_blocks.append({"start": start, "end": end, "key": key})
        frame_records = []
        for position, (frame, tokens, table, candidates) in enumerate(zip(record.frames, episode.frames, record.tables, episode.tables, strict=True)):
            poll()
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
                                  "event_token_prefix": event_prefix, "table_hash": table.table_hash,
                                  **_record_binding(_legacy_table_contract(table) if context is None else table.action_contract, table.purpose, table.synthetic_capacity)})
        for name, values in (("selected", tuple(selection.index for selection in record.selections)),
                             ("old_logprob", tuple(selection.logprob for selection in record.selections)),
                             ("old_value", tuple(selection.value for selection in record.selections))):
            plan(prefix + "." + name, values, np.int64 if name == "selected" else np.float32)
        has_targets = episode.advantage is not None and (_hpo is None or bool(episode.frames))
        if has_targets:
            plan(prefix + ".advantage", episode.advantage, np.float32)
            plan(prefix + ".target", episode.target, np.float32)
        episodes.append({**_record_binding(record.action_contract, record.purpose, record.synthetic_capacity),
                         "actor": record.bank.actor, "episode": record.bank.episode,
                         "boundary": asdict(record.bank.boundary), "raw_blocks": raw_blocks,
                         "frames": frame_records, "end_kind": record.end_kind,
                         "outcome": record.outcome, "reason": record.reason,
                         "selections": [asdict(selection) for selection in record.selections],
                         "has_targets": has_targets})
    if not planned:
        # Zero-decision episode closure is preserved without synthetic transition.
        plan("empty", (), np.uint8)
    return _publish_payload(directory, model=rollout.model, classification=rollout.input_binding,
        declaration=(_hpo_semantics(rollout.model, context) if _hpo is not None else semantics(rollout.model) if context is None else semantics_v3(rollout.model, layout=context.layout, action_contract=context.actions.action_contract, rules=context.rules)), action_config_hash=rollout.action_config_hash,
        policy_version=rollout.policy_version, decision_count=len(rollout), episodes=episodes,
        planned=planned, resident_bytes=rollout.resident_bytes, planning_bytes=planning_bytes,
        planned_bytes=planned_bytes, shard_bytes=shard_bytes, admit=admit, poll=poll, check=check,
        _context=context, _hpo=_hpo)


def _publish_payload(directory: Path, *, model: Any, classification: Json, declaration: Json,
                     action_config_hash: str, policy_version: str | None, decision_count: int,
                     episodes: list[Json], planned: dict[str, tuple[Any, Any, bool]],
                     resident_bytes: int, planning_bytes: int, planned_bytes: int, shard_bytes: int,
                     admit: Callable[[int], None], poll: Callable[[], None],
                     check: Callable[[], None] | None, _context: _RolloutContext | None = None,
                     _hpo: Json | None = None) -> Json:
    """Publish one admitted raw payload; callers own validation and input leases.

    This private seam neither authorizes Source nor constructs/retokenizes it.
    The active public wrapper and native adapter share the exact same file codec.
    """
    descriptors = {key: {"dtype": str(np.dtype(dtype)), "shape": [len(value)] if blob else list(_shape(value)),
                         "bytes": len(value) if blob else _scalar_count(value) * np.dtype(dtype).itemsize}
                   for key, (value, dtype, blob) in planned.items()}
    bundle_plan = plan_bundle("rollout", descriptors, shard_bytes=shard_bytes)
    manifest = {"schema_version": 5 if _hpo is not None else 3 if _context is None else 4, "kind": "structured-rollout", **classification, **declaration,
                "architecture_identity": recorded_architecture_identity(model, layout=LEGACY_LAYOUT if _context is None else _context.layout),
                "action_config_hash": action_config_hash, "policy_version": policy_version,
                "decision_count": decision_count, "episodes": episodes, "data": bundle_plan}
    if _hpo is not None:
        manifest.update(_hpo)
    metadata_bytes = json_bytes(manifest, limit=MAX_MANIFEST_BYTES)
    admit(resident_bytes + planning_bytes + 2 * planned_bytes + 32 * metadata_bytes + 65536)
    arrays = {}
    for key, (value, dtype, blob) in planned.items():
        poll()
        arrays[key] = _blob(value) if blob else _array(value, dtype)
    manifest["data"] = write_bundle(directory, "rollout", arrays, shard_bytes=shard_bytes, _check=check)
    if json_bytes(manifest, limit=MAX_MANIFEST_BYTES) != metadata_bytes:
        raise ContractError("actual complete manifest differs from admitted size plan")
    poll()
    write_owned_json(directory / "rollout.json", manifest)
    poll()
    return manifest



def _write_prepared_payload(directory: Path, metadata: Json, arrays: dict[str, Any], model: Any, *,
                            resident_bytes: int, extra_owned_bytes: int, max_owned_bytes: int,
                            shard_bytes: int = MAX_ARRAY_BYTES, check: Callable[[], None],
                            reserve: Callable[[int], None], _context: _RolloutContext | None = None) -> Json:
    """Private native payload publication, never a Source authority boundary.

    Input buffers stay owned by the caller's live leases through publication.
    resident_bytes covers additional Python ownership; extra_owned_bytes includes
    persistent native/sealed/control owners already admitted by that caller.
    The original-source decoder comparison remains mandatory after publication.
    """
    from gf_models.structured_sealed import checksum
    layout = LEGACY_LAYOUT if _context is None else _context.layout
    validate_config(model, layout=layout)
    semantic_keys = SEMANTIC_KEYS if _context is None else SEMANTIC_KEYS_V3
    poll = check
    poll()
    limit = min(max_owned_bytes, model.resources.max_owned_bytes)
    integer(resident_bytes, 0, limit)
    integer(extra_owned_bytes, 0, limit)
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    def admit(size: int) -> None:
        poll()
        if size + extra_owned_bytes > limit:
            raise ContractError('joint prepared payload planning/packing/publication reservation exceeded')
        reserve(size)
    admit(resident_bytes + 65536)
    class_keys = {'action_contract', 'purpose', 'synthetic_capacity'} if _context is None else set(_context.classification())
    closed(metadata, {'architecture_identity', 'action_config_hash', 'policy_version', 'decision_count', 'episodes'} | class_keys | set(semantic_keys))
    classification: Json = {key: metadata[key] for key in class_keys}
    from gf_common.action_contract import parse_action_contract
    contract = parse_action_contract(classification['action_contract']) if _context is None else _context.actions.action_contract
    def binding_check(value: Json, **_: Any) -> None:
        if _context is None:
            assert type(contract) is ActionContract
            check_input_binding(value, action_contract=contract, purpose='component', synthetic_capacity=False)
        else:
            _context.check(value)
    binding_check(classification)
    declaration = {key: metadata[key] for key in semantic_keys}
    if _context is None:
        validate_recorded_semantics(declaration, model)
        expected_semantics = semantics(model)
    else:
        validate_recorded_semantics_v3(declaration, model, layout=layout, action_contract=_context.actions.action_contract, rules=_context.rules)
        expected_semantics = semantics_v3(model, layout=layout, action_contract=_context.actions.action_contract, rules=_context.rules)
    if (canonical(declaration) != canonical(expected_semantics)
            or metadata['architecture_identity'] != recorded_architecture_identity(model, layout=layout)):
        raise ContractError('prepared active payload semantics/architecture mismatch')
    checksum(metadata['action_config_hash'])
    integer(metadata['decision_count'], 0)
    if metadata['policy_version'] is not None and (type(metadata['policy_version']) is not str or not metadata['policy_version']):
        raise ContractError('prepared policy identity required')
    if type(arrays) is not dict or type(metadata['episodes']) is not list:
        raise ContractError('owned prepared array map and episode list required')
    planned: dict[str, tuple[Any, Any, bool]] = {}
    planning_bytes = 65536
    planned_bytes = 0
    def take(key: str, dtype: Any, shape: tuple[int, ...] | None) -> Any:
        nonlocal planning_bytes, planned_bytes
        poll()
        planning_bytes += 1024 + len(key.encode())
        admit(resident_bytes + planning_bytes + 2 * planned_bytes)
        if key in planned or key not in arrays:
            raise ContractError('prepared named-array key ownership mismatch')
        value = arrays[key]
        if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous
                or value.flags.writeable or value.nbytes > shard_bytes
                or shape is not None and value.shape != shape or shape is None and value.ndim != 1):
            raise ContractError('prepared readonly array dtype/shape/size mismatch')
        prospective = planned_bytes + value.nbytes
        finite_scratch = min(value.size, 32768) if value.dtype == np.float32 else 0
        admit(resident_bytes + planning_bytes + 2 * prospective + finite_scratch)
        # Integer/u8 values are finite by construction. Bound the temporary
        # Boolean buffer for FP32 checks instead of scanning a whole bank.
        if value.dtype == np.float32:
            flat = value.reshape(-1)
            for start in range(0, len(flat), 32768):
                poll()
                if not np.isfinite(flat[start:start + 32768]).all():
                    raise ContractError('prepared nonfinite tensor')
        planned_bytes = prospective
        planned[key] = (value, dtype, False)
        return value
    owners = set()
    total = 0
    frame_shapes = {'state_numerics': (26,), 'state_present': (26,), 'state_categories': (10,),
        'state_category_present': (10,), 'card_numerics': (18, len(layout.card_numeric_fields)), 'card_present': (18, len(layout.card_numeric_fields)),
        'card_categories': (18, len(layout.card_categories)), 'card_mask': (18,), 'opponent_learned': (18,), 'history': (layout.history_width,)}
    for ei, episode in enumerate(metadata['episodes']):
        poll()
        closed(episode, {'actor', 'episode', 'boundary', 'raw_blocks', 'frames', 'end_kind', 'outcome', 'reason',
                        'selections', 'has_targets'} | set(classification))
        binding_check({key: episode[key] for key in classification}, action_contract=contract,
                            purpose='component', synthetic_capacity=False)
        integer(episode['actor'], 1, 2)
        if type(episode['episode']) is not str or not episode['episode'] or (episode['actor'], episode['episode']) in owners:
            raise ContractError('prepared participant bank identity missing/duplicated')
        owners.add((episode['actor'], episode['episode']))
        boundary = EventBoundary(**closed(episode['boundary'], {'end', 'prefix_hash'}))
        integer(boundary.end, 0, model.event.max_episode)
        if type(episode['frames']) is not list or type(episode['selections']) is not list:
            raise ContractError('prepared ordered frame/selection metadata required')
        count = len(episode['frames'])
        integer(count, 0, model.temporal.max_decisions)
        if len(episode['selections']) != count:
            raise ContractError('prepared frame/selection count mismatch')
        total += count
        prefix = f'e.{ei}'
        end = 0
        if type(episode['raw_blocks']) is not list:
            raise ContractError('prepared ordered event blocks required')
        for block in episode['raw_blocks']:
            closed(block, {'start', 'end', 'key'})
            integer(block['start'], end, end)
            integer(block['end'], end + 1, min(end + 1024, boundary.end))
            key = prefix + f".raw-events.{end}"
            if block['key'] != key:
                raise ContractError('prepared raw event block path mismatch')
            take(key, np.uint8, None)
            end = block['end']
        if end != boundary.end:
            raise ContractError('prepared complete bank block boundary mismatch')
        previous_end, previous_revision = 0, -1
        selections = []
        for fi, (frame, saved) in enumerate(zip(episode['frames'], episode['selections'], strict=True)):
            closed(frame, {'actor', 'episode', 'revision', 'interval_start', 'boundary', 'interval_start_hash',
                           'frame_hash', 'payload_key', 'token_prefix', 'event_token_prefix', 'table_hash'} | set(classification))
            binding_check({key: frame[key] for key in classification}, action_contract=contract,
                                purpose='component', synthetic_capacity=False)
            b = EventBoundary(**closed(frame['boundary'], {'end', 'prefix_hash'}))
            integer(frame['revision'], previous_revision + 1)
            integer(frame['interval_start'], previous_end, previous_end)
            integer(b.end, previous_end, min(boundary.end, previous_end + model.event.max_interval))
            if frame['actor'] != episode['actor'] or frame['episode'] != episode['episode']:
                raise ContractError('prepared frame bank owner mismatch')
            for key in ('interval_start_hash', 'frame_hash', 'table_hash'):
                checksum(frame[key])
            fp, ep = prefix + f'.f.{fi}', prefix + f'.event-tokens.{fi}'
            if (frame['payload_key'] != fp + '.payload' or frame['token_prefix'] != fp
                    or frame['event_token_prefix'] != ep):
                raise ContractError('prepared frame/token path mismatch')
            take(fp + '.payload', np.uint8, None)
            for name in FRAME_FIELDS:
                take(fp + '.' + name, _token_dtype(name), frame_shapes[name])
            interval = b.end - previous_end
            for name, width in (('interval_numerics', 11), ('interval_present', 11), ('interval_categories', 12)):
                take(ep + '.' + name, _token_dtype(name), (interval, width) if interval else (0,))
            kinds = take(fp + '.candidate.kinds', np.int64, None)
            candidates = len(kinds)
            integer(candidates, 1, 65536)
            take(fp + '.candidate.targets', np.int64, (candidates,))
            take(fp + '.candidate.numerics', np.int64, (candidates, 4))
            take(fp + '.candidate.present', np.uint8, (candidates, 4))
            offsets = take(fp + '.candidate.offsets', np.int64, (candidates + 1,))
            slots = take(fp + '.candidate.slots', np.int64, None)
            if offsets[0] != 0 or offsets[-1] != len(slots) or np.any(np.diff(offsets) < 0) or np.any(np.diff(offsets) > model.max_positions):
                raise ContractError('prepared candidate offsets/capacity mismatch')
            selected = Selection(**closed(saved, set(Selection.__dataclass_fields__)))
            integer(selected.index, 0, candidates - 1)
            if (selected.logprob is None or selected.value is None or selected.policy_version != metadata['policy_version']
                    or selected.table_hash != frame['table_hash']):
                raise ContractError('prepared neural Selection identity/statistics mismatch')
            selections.append(selected)
            previous_end, previous_revision = b.end, frame['revision']
        for name, dtype, full in (('selected', np.int64, [s.index for s in selections]),
                                  ('old_logprob', np.float32, [s.logprob for s in selections]),
                                  ('old_value', np.float32, [s.value for s in selections])):
            value = take(prefix + '.' + name, dtype, (count,))
            with np.errstate(over='ignore', invalid='ignore'):
                expected = np.asarray(full, dtype=dtype)
            if not np.isfinite(expected).all() or value.tobytes() != expected.tobytes():
                raise ContractError('prepared Selection/scalar precision binding mismatch')
        kind = episode['end_kind']
        if kind not in ('natural', 'truncated', 'aborted', 'cancelled', 'incomplete', 'unsupported', 'infrastructure_failure'):
            raise ContractError('prepared unknown episode closure')
        if kind == 'natural':
            validate_result(episode['outcome'])
            if episode['outcome'] is None or episode['reason'] is not None:
                raise ContractError('prepared natural outcome missing')
        elif episode['outcome'] is not None or type(episode['reason']) is not str or not episode['reason']:
            raise ContractError('prepared interrupted outcome/reason mismatch')
        if type(episode['has_targets']) is not bool or episode['has_targets'] != (kind == 'natural' and count > 0):
            raise ContractError('prepared natural target presence mismatch')
        if episode['has_targets']:
            take(prefix + '.advantage', np.float32, (count,))
            take(prefix + '.target', np.float32, (count,))
    if total != metadata['decision_count'] or total and metadata['policy_version'] is None:
        raise ContractError('prepared decision count/policy identity mismatch')
    if not planned:
        take('empty', np.uint8, (0,))
    if set(planned) != set(arrays):
        raise ContractError('prepared unknown or missing array')
    create_output_directory(directory, 'rollout.json')
    return _publish_payload(directory, model=model, classification=classification, declaration=declaration,
        action_config_hash=metadata['action_config_hash'], policy_version=metadata['policy_version'],
        decision_count=total, episodes=metadata['episodes'], planned=planned, resident_bytes=resident_bytes,
        planning_bytes=planning_bytes, planned_bytes=planned_bytes, shard_bytes=shard_bytes,
        admit=admit, poll=poll, check=check, _context=_context)

def _write_prepared_game_payload(directory: Path, metadata: Json, arrays: dict[str, Any], model: ModelConfigV3, *,
                                 actions: ActionConfigV3, rules: CardInstanceRules, resident_bytes: int,
                                 extra_owned_bytes: int, max_owned_bytes: int, shard_bytes: int,
                                 check: Callable[[], None], reserve: Callable[[int], None]) -> Json:
    return _write_prepared_payload(directory, metadata, arrays, model, resident_bytes=resident_bytes,
        extra_owned_bytes=extra_owned_bytes, max_owned_bytes=max_owned_bytes, shard_bytes=shard_bytes,
        check=check, reserve=reserve, _context=_RolloutContext(actions, rules))

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


def read_rollout(directory: Path, expected_digest: str, model: Any, *,
                 action_contract: Any, purpose: str, synthetic_capacity: bool) -> StructuredRollout:
    input_binding(action_contract, purpose, synthetic_capacity)
    validate_config(model)
    return _read_rollout(directory, expected_digest, model, action_contract=action_contract,
        purpose=purpose, synthetic_capacity=synthetic_capacity, expected_semantics=semantics(model))


def _read_recorded_rollout(directory: Path, expected_digest: str, model: Any, *,
                           action_contract: ActionContract, declaration: Json,
                           expected_records: tuple[EpisodeRecords, ...] | None, max_owned_bytes: int,
                           extra_owned_bytes: int, check: Callable[[], None],
                           reserve: Callable[[int], None], expected_correspondence: Json | None = None) -> StructuredRollout:
    """Historical component correspondence, never current runtime/Source authority.

    The caller binds declaration to its validated original generation inference.
    No producer field is invented for rollout3; artifact3 still requires one.
    """
    closed(declaration, SEMANTIC_KEYS)
    validate_recorded_semantics(declaration, model)
    if expected_correspondence is not None:
        if expected_records is not None:
            raise ContractError('original record and digest correspondence modes are exclusive')
        _check_source_correspondence(expected_correspondence, declaration)
    if expected_records is not None and (type(expected_records) is not tuple or any(type(item) is not EpisodeRecords for item in expected_records)):
        raise ContractError('ordered original participant records required')
    check()
    limit = min(max_owned_bytes, model.resources.max_owned_bytes)
    integer(extra_owned_bytes, 0, limit)
    manifest_path = directory / 'rollout.json'
    fd = os.open(manifest_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        manifest_identity = tuple(getattr(info, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
        guard = _source_guard(manifest_path, source, manifest_identity, check)
        guard()
        manifest_parse = 32 * info.st_size + 65536
        if manifest_parse + extra_owned_bytes > limit:
            raise ContractError('joint recorded JSON reservation exceeded before manifest parse')
        reserve(manifest_parse)
        manifest = read_json(manifest_path, expected_digest=expected_digest, max_owned_bytes=limit - extra_owned_bytes)
        guard()
    descriptors = validate_bundle_metadata(manifest['data'], max_decoded_bytes=limit - extra_owned_bytes)
    manifest_extra = 32 * max(0, manifest_identity[2] - json_bytes(manifest, limit=MAX_MANIFEST_BYTES))
    base = _read_reservation(manifest, descriptors) + manifest_extra + 512 * len(manifest['data']['files'])
    snapshots: dict[str, tuple[int, ...]] = {}
    encoded = sidecar_encoded = 0
    for name in manifest['data']['files']:
        check()
        if base + 32 * encoded + 32 * sidecar_encoded + 65536 + extra_owned_bytes > limit:
            raise ContractError('joint recorded encoded archive reservation exceeded before open')
        fd = os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ContractError('recorded bundle requires regular original file')
            identity = tuple(getattr(info, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
            _source_guard(directory / name, source, identity, check)()
        snapshots[name] = identity
        if name.endswith('.npz'):
            encoded = max(encoded, info.st_size)
        else:
            sidecar_encoded = max(sidecar_encoded, info.st_size)
    sidecar_parse = 32 * sidecar_encoded + 65536
    reservation = base + 32 * encoded + sidecar_parse
    if reservation + extra_owned_bytes > limit:
        raise ContractError('joint recorded encoded archive/JSON reservation exceeded before open')
    reserve(reservation)
    result = _read_rollout(directory, expected_digest, model, action_contract=action_contract,
        purpose='component', synthetic_capacity=False, expected_semantics=declaration,
        expected_records=expected_records, expected_correspondence=expected_correspondence, max_owned_bytes=max_owned_bytes,
        extra_owned_bytes=extra_owned_bytes, check=check, reserve=reserve,
        original_manifest=manifest, snapshots=snapshots,
        encoded_scratch=manifest_extra + 32 * encoded + 512 * len(snapshots) + sidecar_parse,
        json_budget=limit - extra_owned_bytes - base - 32 * encoded, historical=True)
    for name, identity in chain(snapshots.items(), (('rollout.json', manifest_identity),)):
        check()
        fd = os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as source:
            _source_guard(directory / name, source, identity, check)()
    check()
    return result


def _read_rollout(directory: Path, expected_digest: str, model: Any, *,
                  action_contract: Any, purpose: str, synthetic_capacity: bool,
                  expected_semantics: Json, expected_records: tuple[EpisodeRecords, ...] | None = None,
                  expected_correspondence: Json | None = None,
                  max_owned_bytes: int | None = None, extra_owned_bytes: int = 0,
                  check: Callable[[], None] | None = None,
                  reserve: Callable[[int], None] = lambda _: None,
                  original_manifest: Json | None = None, snapshots: dict[str, tuple[int, ...]] | None = None,
                  encoded_scratch: int = 0, json_budget: int | None = None, historical: bool = False,
                  _context: _RolloutContext | None = None, storage_binding: Json | None = None,
                  _hpo: Json | None = None) -> StructuredRollout:
    poll = check or (lambda: None)
    poll()
    limit = model.resources.max_owned_bytes if max_owned_bytes is None else min(max_owned_bytes, model.resources.max_owned_bytes)
    integer(extra_owned_bytes, 0, limit)
    layout = LEGACY_LAYOUT if _context is None else _context.layout
    classification = input_binding(action_contract, purpose, synthetic_capacity) if _context is None else _context.classification()
    def binding_check(raw: Json, **_: Any) -> None:
        if _context is None:
            check_input_binding(raw, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
        else:
            _context.check(raw)
    manifest = original_manifest if original_manifest is not None else read_json(directory / "rollout.json",
        expected_digest=expected_digest, max_owned_bytes=limit - extra_owned_bytes)
    poll()
    closed(manifest, {"schema_version", "kind", "architecture_identity", "action_config_hash", "policy_version",
                      "decision_count", "episodes", "data"} | set(classification) | set(expected_semantics) | (set(_hpo) if _hpo is not None else set()))
    binding_check({key: manifest[key] for key in classification}, action_contract=action_contract,
                        purpose=purpose, synthetic_capacity=synthetic_capacity)
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != (5 if _hpo is not None else 3 if _context is None else 4)
            or manifest["kind"] != "structured-rollout"
            or manifest["architecture_identity"] != recorded_architecture_identity(model, layout=layout)
            or any(canonical(manifest[key]) != canonical(value) for key, value in expected_semantics.items())):
        raise ContractError("rollout recipe/architecture/runtime mismatch")
    integer(manifest["decision_count"], 0, 2**31 - 1)
    if _hpo is not None:
        assert _context is not None
        if manifest['action_config_hash'] != digest(asdict(_context.actions)):
            raise ContractError('numeric5 action configuration mismatch')
    if not isinstance(manifest["episodes"], list):
        raise ContractError("episode records required")
    expected_digests = None
    if expected_correspondence is not None:
        if expected_records is not None:
            raise ContractError('original record and digest correspondence modes are exclusive')
        expected_digests = _check_source_correspondence(expected_correspondence, expected_semantics, game=_context is not None, storage_binding=storage_binding, hpo=_hpo is not None)
        if len(manifest['episodes']) != len(expected_digests):
            raise ContractError('numeric episode count differs from ORIGINAL Source digest order')
    if expected_records is not None and len(manifest['episodes']) != len(expected_records):
        raise ContractError('numeric/original Source episode count differs')
    for episode in manifest["episodes"]:
        poll()
        closed(episode, {"actor", "episode", "boundary", "raw_blocks", "frames", "end_kind", "outcome", "reason", "selections", "has_targets"} | set(classification))
        binding_check({key: episode[key] for key in classification}, action_contract=action_contract,
                            purpose=purpose, synthetic_capacity=synthetic_capacity)
        if not isinstance(episode["frames"], list) or not isinstance(episode["selections"], list) or len(episode["frames"]) != len(episode["selections"]):
            raise ContractError("episode frame/selection metadata shape mismatch")
        integer(len(episode["frames"]), 0, model.temporal.max_decisions)
        for frame_meta in episode["frames"]:
            poll()
            closed(frame_meta, {"actor", "episode", "revision", "interval_start", "boundary", "interval_start_hash",
                               "frame_hash", "payload_key", "token_prefix", "event_token_prefix", "table_hash"} | set(classification))
            binding_check({key: frame_meta[key] for key in classification}, action_contract=action_contract,
                                purpose=purpose, synthetic_capacity=synthetic_capacity)
        boundary_meta = closed(episode["boundary"], {"end", "prefix_hash"})
        integer(boundary_meta["end"], 0, model.event.max_episode)
    descriptors = inspect_bundle(directory, manifest["data"], max_decoded_bytes=limit - extra_owned_bytes,
                                 _check=check, _snapshots=snapshots, _json_budget=json_budget)
    reservation = _read_reservation(manifest, descriptors) + encoded_scratch
    if reservation + extra_owned_bytes > limit:
        raise ContractError("joint rollout decode/raw-owner reservation exceeded before reconstruction")
    reserve(reservation)
    poll()
    arrays = read_bundle(directory, manifest["data"], max_decoded_bytes=limit - extra_owned_bytes,
                         _check=check, _snapshots=snapshots, _json_budget=json_budget)
    poll()
    used: set[str] = set()
    def get(key: str, dtype: Any) -> Any:
        poll()
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
        poll()
        closed(metadata, {"actor", "episode", "boundary", "raw_blocks", "frames", "end_kind", "outcome", "reason", "selections", "has_targets"} | set(classification))
        boundary = EventBoundary(**closed(metadata["boundary"], {"end", "prefix_hash"}))
        bank = EventBank(metadata["actor"], metadata["episode"], max_events=max(1, boundary.end), layout=layout)
        prefix = f"e.{episode_index}"
        for block in metadata["raw_blocks"]:
            poll()
            closed(block, {"start", "end", "key"})
            if block["key"] != prefix + f".raw-events.{block['start']}" or block["start"] != bank.boundary.end:
                raise ContractError("event bank block continuity mismatch")
            raw = get(block["key"], np.uint8)
            if raw.ndim != 1:
                raise ContractError("raw event bytes must be a vector")
            events = json.loads(raw.tobytes())
            if canonical(events).encode() != raw.tobytes() or len(events) != block["end"] - block["start"]:
                raise ContractError("raw event block shape/canonical mismatch")
            bank.append(event_delta(bank.actor, bank.episode, block["start"], bank.boundary.prefix_hash, events, layout=layout))
        if bank.boundary != boundary:
            raise ContractError("complete episode bank hash mismatch")
        frames, tables = [], []
        for position, frame_meta in enumerate(metadata["frames"]):
            poll()
            closed(frame_meta, {"actor", "episode", "revision", "interval_start", "boundary", "interval_start_hash",
                               "frame_hash", "payload_key", "token_prefix", "event_token_prefix", "table_hash"} | set(classification))
            binding_check({key: frame_meta[key] for key in classification}, action_contract=action_contract,
                                purpose=purpose, synthetic_capacity=synthetic_capacity)
            frame_prefix, event_prefix = prefix + f".f.{position}", prefix + f".event-tokens.{position}"
            if (frame_meta["payload_key"] != frame_prefix + ".payload" or frame_meta["token_prefix"] != frame_prefix
                    or frame_meta["event_token_prefix"] != event_prefix):
                raise ContractError("frame/token ownership path mismatch")
            payload = get(frame_meta["payload_key"], np.uint8)
            if payload.ndim != 1:
                raise ContractError("frame payload must be byte vector")
            frame = _frame_type(layout)(frame_meta["actor"], frame_meta["episode"], frame_meta["revision"],
                                  frame_meta["interval_start"], EventBoundary(**frame_meta["boundary"]),
                                  frame_meta["interval_start_hash"], payload.tobytes(), frame_meta["frame_hash"])
            frame_arrays = {name: get(frame_prefix + "." + name, _token_dtype(name)) for name in FRAME_FIELDS}
            event_arrays = {name: get(event_prefix + "." + name, _token_dtype(name)) for name in EVENT_FIELDS}
            candidate_arrays = {name: get(frame_prefix + ".candidate." + name, _token_dtype(name)) for name in CANDIDATE_FIELDS}
            for code in candidate_arrays["kinds"]:
                integer(int(code), 2, len(layout.action_kinds) + 1)
            for code in candidate_arrays["targets"]:
                integer(int(code), 2, len(layout.targets) + 1)
            if not np.isin(candidate_arrays["present"], (0, 1)).all():
                raise ContractError("candidate presence must be0/1")
            table = CandidateTable(frame.frame_hash, tuple(layout.action_kinds[int(code) - 2] for code in candidate_arrays["kinds"]),
                tuple(layout.targets[int(code) - 2] for code in candidate_arrays["targets"]),
                tuple(tuple(int(value) for value in row) for row in candidate_arrays["numerics"]),
                tuple(tuple(bool(value) for value in row) for row in candidate_arrays["present"]),
                tuple(int(value) for value in candidate_arrays["offsets"]), tuple(int(value) for value in candidate_arrays["slots"]), frame_meta["table_hash"],
                action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
            frames.append(frame)
            tables.append(table)
            stored_tokens.append((frame_meta, frame_arrays, event_arrays, candidate_arrays))
        selections = tuple(Selection(**value) for value in metadata["selections"])
        record = EpisodeRecords(bank.view(), tuple(frames), tuple(tables), selections, metadata["end_kind"], metadata["outcome"], metadata["reason"],
                                action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity, _context=_context)
        if expected_digests is not None and _episode_correspondence_digest(record) != expected_digests[episode_index]:
            raise ContractError('numeric raw record differs from ORIGINAL native Source correspondence')
        if expected_records is not None:
            original = expected_records[episode_index]
            if (record != original or any(canonical(asdict(a)) != canonical(asdict(b))
                    for a, b in zip(record.selections, original.selections, strict=True))):
                raise ContractError('numeric raw record differs from ORIGINAL participant Source')
            record = original  # retain no second complete raw Source copy.
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
    rollout: StructuredRollout
    if _hpo is not None:
        from gf_training.hpo_collection import replay_selection
        from gf_training.hpo_config import hpo_training_config
        from gf_training.hpo_rollout import HpoRollout
        assert _context is not None
        for key, value in _hpo.items():
            if canonical(manifest[key]) != canonical(value):
                raise ContractError('numeric5 training/selection/step correspondence mismatch')
        locations = replay_selection(_hpo['selection'], tuple(len(record.frames) for record in records))
        rollout = HpoRollout(tuple(records), model, hpo_training_config(_hpo['training_recipe']),
            actions=_context.actions, rules=_context.rules, selected_locations=locations,
            _reserved_bytes=manifest['data']['decoded_bytes'] + extra_owned_bytes)
    else:
        rollout = StructuredRollout(tuple(records), model, manifest["action_config_hash"],
                                    _reserved_bytes=manifest["data"]["decoded_bytes"] + extra_owned_bytes, action_contract=action_contract,
                                    purpose=purpose, synthetic_capacity=synthetic_capacity, _historical=historical, _context=_context)
    if manifest["decision_count"] != len(rollout) or manifest["policy_version"] != rollout.policy_version:
        raise ContractError("rollout endpoint/policy identity mismatch")
    for episode, scalars in zip(rollout.episodes, train_vectors, strict=True):
        poll()
        for name, source in (("selected", [s.index for s in episode.records.selections]),
                             ("old_logprob", [s.logprob for s in episode.records.selections]),
                             ("old_value", [s.value for s in episode.records.selections])):
            if not np.array_equal(scalars[name], np.asarray(source, dtype=scalars[name].dtype)):
                raise ContractError("selection/scalar binding mismatch")
        has_targets = episode.advantage is not None and (_hpo is None or bool(episode.frames))
        if has_targets != ("advantage" in scalars):
            raise ContractError("natural/interrupted target presence mismatch")
        if has_targets and (not np.array_equal(scalars["advantage"], episode.advantage)
                                               or not np.array_equal(scalars["target"], episode.target)):
            raise ContractError("GAE/normalized target mismatch")
    expected_tokens = [(frame, table) for episode in rollout.episodes for frame, table in zip(episode.frames, episode.tables, strict=True)]
    for (tokens, candidates), (_, frame_arrays, event_arrays, candidate_arrays) in zip(expected_tokens, stored_tokens, strict=True):
        poll()
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
    poll()
    return rollout


def _hpo_semantics(model: Any, context: _RolloutContext | None) -> Json:
    from gf_models.structured_artifacts import _game_context_v4
    if context is None:
        raise ContractError('numeric5 requires corrected participant context')
    return _game_context_v4(context.actions.action_contract, context.rules,
        digest(asdict(context.actions))).semantics(model)


def game_episode_records(bank: EventBankView, frames: tuple[DecisionFrame, ...], tables: tuple[CandidateTable, ...],
                         selections: tuple[Selection, ...], end_kind: str, outcome: Any = None,
                         reason: str | None = None, *, actions: ActionConfigV3, rules: CardInstanceRules) -> EpisodeRecords:
    context = _RolloutContext(actions, rules)
    if bank.layout is not context.layout:
        raise ContractError('episode4 original bank layout mismatch')
    return EpisodeRecords(bank, frames, tables, selections, end_kind, outcome, reason,
        action_contract=actions.action_contract, purpose='qualified_game', synthetic_capacity=False, _context=context)


def game_rollout(records: tuple[EpisodeRecords, ...], model: ModelConfigV3, *, actions: ActionConfigV3,
                 rules: CardInstanceRules, reserved_bytes: int = 0) -> StructuredRollout:
    return StructuredRollout(records, model, digest(asdict(actions)), action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False, _reserved_bytes=reserved_bytes,
        _context=_RolloutContext(actions, rules))


def read_game_rollout(directory: Path, expected_digest: str, model: ModelConfigV3, *, actions: ActionConfigV3,
                      rules: CardInstanceRules, declaration: Json, expected_correspondence: Json,
                      max_owned_bytes: int | None = None, extra_owned_bytes: int = 0,
                      check: Callable[[], None] | None = None, reserve: Callable[[int], None] = lambda _: None,
                      storage_binding: Json | None = None) -> StructuredRollout:
    context = _RolloutContext(actions, rules)
    validate_config(model, layout=context.layout)
    closed(declaration, SEMANTIC_KEYS_V3)
    validate_recorded_semantics_v3(declaration, model, layout=context.layout,
        action_contract=actions.action_contract, rules=rules)
    _check_source_correspondence(expected_correspondence, declaration, game=True, storage_binding=storage_binding)
    return _read_rollout(directory, expected_digest, model, action_contract=actions.action_contract,
        purpose='qualified_game', synthetic_capacity=False, expected_semantics=declaration,
        expected_correspondence=expected_correspondence, max_owned_bytes=max_owned_bytes,
        extra_owned_bytes=extra_owned_bytes, check=check, reserve=reserve, _context=context, storage_binding=storage_binding)
