"""Private final64 packet decoding; no Source, parent, loan or permission issuer."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from gf_common.candidate_columns import decode_candidate_columns_v2
from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.structured import _CorrectedCardInstanceEventBankView
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.structured_wire import parse_frame, parse_selection
from gf_models.structured_binding import input_binding_game_v3
from gf_training.structured_rollout import (
    CANDIDATE_FIELDS,
    EVENT_FIELDS,
    FRAME_FIELDS,
    EpisodeRecords,
    _episode_correspondence_digest,
    _token_dtype,
    game_episode_records,
)

LIMIT = 64 * 1024**2
PACKET_KEYS = {'schema_version', 'kind', 'summary_sha256', 'episode_index', 'episode_id', 'viewer',
    'episode_digest', 'binding', 'endpoint', 'bank', 'decisions', 'arrays', 'correspondence'}


def _json(raw: bytes) -> Json:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError('duplicate final64 DTO JSON key')
            result[key] = value
        return result

    def nonfinite(_: str) -> Any:
        raise ContractError('nonfinite final64 DTO JSON')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if type(value) is not dict or canonical(value).encode() != raw:
        raise ContractError('canonical final64 DTO object required')
    return value


def _decode_final_validation_packet(payload: Any, *, summary: Json, episode_index: int,
        actions: ActionConfigV3, rules: CardInstanceRules, aggregate_window: int,
        max_owned_bytes: int, retained_owned_bytes: int, max_record_bytes: int,
        reserve: Callable[[int], None], check: Callable[[], None]) -> EpisodeRecords:
    """Decode one real original episode under caller-owned joint admission/guards.

    ``retained_owned_bytes`` includes all caller/native/view/prior-record ownership.
    ``reserve(n)`` must admit the proposed additional decoder ownership BEFORE it
    is allocated, preserving earlier reservations. This function never releases a
    caller reservation or opens an original, native session or resource loan.
    """
    integer(episode_index, 0, 63)
    integer(aggregate_window, 1, 65536)
    integer(max_record_bytes, 1, LIMIT)
    integer(retained_owned_bytes, 0, 2**63 - 1)
    integer(max_owned_bytes, 1, 2**63 - 1)
    if type(payload) is not tuple or len(payload) != 4 or type(payload[0]) is not bytes:
        raise ContractError('closed final64 typed-bank packet required')
    if len(payload[0]) > max_record_bytes:
        raise ContractError('final64 packet metadata exceeds record bound')
    growth = 0

    def admit(extra: int) -> None:
        nonlocal growth
        check()
        integer(extra, 0, 2**63 - 1)
        proposed = growth + extra
        if retained_owned_bytes + proposed > max_owned_bytes:
            raise ContractError('final64 joint ownership exhausted BEFORE decode')
        reserve(proposed)
        growth = proposed

    admit(64 * len(payload[0]) + 65536)
    metadata = closed(_json(payload[0]), PACKET_KEYS)
    if (type(metadata['schema_version']) is not int or metadata['schema_version'] != 1
            or metadata['kind'] != 'card-instance-final-validation-episode-payload'
            or type(metadata['episode_index']) is not int or metadata['episode_index'] != episode_index
            or metadata['summary_sha256'] != digest(summary)):
        raise ContractError('final64 packet differs from retained structural summary')
    expected = summary['episodes'][episode_index]
    for name in ('episode_id', 'viewer', 'episode_digest'):
        if canonical(metadata[name]) != canonical(expected[name]):
            raise ContractError('final64 ordered episode/role/correspondence differs')
    if canonical(metadata['binding']) != canonical(input_binding_game_v3(actions.action_contract, rules)):
        raise ContractError('final64 original five-field binding differs')
    if type(metadata['decisions']) is not list or len(metadata['decisions']) != expected['decisions']:
        raise ContractError('final64 complete original decision population differs')
    groups = dict(zip(('uint8', 'int64', 'float32'), payload[1:], strict=True))
    for dtype, banks in groups.items():
        if type(banks) is not tuple:
            raise ContractError('immutable final64 bank tuple required')
        for bank in banks:
            if (type(bank) is not np.ndarray or bank.dtype != np.dtype(dtype) or bank.ndim != 1
                    or bank.flags.writeable or not bank.flags.c_contiguous or bank.nbytes > LIMIT):
                raise ContractError('final64 bank dtype/readonly/shape/bound differs')
    if type(metadata['arrays']) is not list:
        raise ContractError('ordered final64 array descriptors required')
    admit(1024 * (len(metadata['arrays']) + 1))
    arrays: dict[str, Any] = {}
    spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for descriptor in metadata['arrays']:
        check()
        closed(descriptor, {'key', 'dtype', 'shape', 'bank_index', 'element_offset', 'element_count'})
        key, dtype = descriptor['key'], descriptor['dtype']
        if (type(key) is not str or not key or not key.isascii()
                or any(not (char.isalnum() or char in '._-') for char in key)
                or key in arrays or type(dtype) is not str or dtype not in groups):
            raise ContractError('final64 typed array identity differs')
        index = integer(descriptor['bank_index'], 0, len(groups[dtype]) - 1)
        bank = groups[dtype][index]
        offset = integer(descriptor['element_offset'], 0, len(bank))
        count = integer(descriptor['element_count'], 0, len(bank) - offset)
        shape = descriptor['shape']
        if type(shape) is not list or len(shape) > 8:
            raise ContractError('final64 typed shape required')
        for axis in shape:
            integer(axis, 0, LIMIT)
        if math.prod(shape) != count or count * bank.dtype.itemsize > LIMIT:
            raise ContractError('final64 typed shape/count/bound mismatch')
        spans.setdefault((dtype, index), []).append((offset, offset + count))
        arrays[key] = bank[offset:offset + count].reshape(shape)
    # Exhaust every byte/element of every supplied bank; no hidden/foreign banks.
    for dtype, banks in groups.items():
        for index, bank in enumerate(banks):
            position = 0
            for start, end in sorted(spans.get((dtype, index), [])):
                if start != position:
                    raise ContractError('final64 overlapping/unowned typed bank range')
                position = end
            if position != len(bank) or (not len(bank) and (dtype, index) not in spans):
                raise ContractError('final64 incomplete typed bank inventory')
    used: set[str] = set()

    def get(key: str, dtype: Any) -> Any:
        check()
        if type(key) is not str or key not in arrays or key in used or arrays[key].dtype != np.dtype(dtype):
            raise ContractError('final64 array ownership/type mismatch')
        used.add(key)
        return arrays[key]

    def raw(key: str, *, record: bool = True) -> bytes:
        value = get(key, np.uint8)
        if value.ndim != 1 or value.nbytes > (max_record_bytes if record else LIMIT):
            raise ContractError('final64 DTO record/source bound exceeded')
        admit(64 * value.nbytes + 65536)
        return value.tobytes()

    bank = closed(metadata['bank'], {'actor', 'episode', 'initial_prefix_hash', 'event_records', 'terminal_prefix_hash'})
    if bank['actor'] != metadata['viewer'] or type(bank['event_records']) is not list:
        raise ContractError('final64 original own event bank differs')
    admit(1024 * (len(bank['event_records']) + 1))
    events, prefixes = [], [bank['initial_prefix_hash']]
    for index, event in enumerate(bank['event_records']):
        closed(event, {'array_key', 'prefix_hash'})
        if event['array_key'] != f'dto.event.{index}':
            raise ContractError('final64 ordered event DTO identity differs')
        encoded = raw(event['array_key'])
        _json(encoded)
        events.append(encoded)
        prefixes.append(event['prefix_hash'])
    view = _CorrectedCardInstanceEventBankView(bank['actor'], bank['episode'], tuple(events), tuple(prefixes))
    if view.boundary.prefix_hash != bank['terminal_prefix_hash']:
        raise ContractError('final64 original terminal bank prefix differs')
    frames, tables, selections = [], [], []
    for index, decision in enumerate(metadata['decisions']):
        closed(decision, {'frame_array_key', 'candidate_descriptor', 'candidate_array_key', 'selection'})
        if decision['frame_array_key'] != f'dto.frame.{index}' or decision['candidate_array_key'] != f'dto.columns.{index}':
            raise ContractError('final64 ordered decision DTO identity differs')
        frame = parse_frame(_json(raw(decision['frame_array_key'])), layout=LAYOUT)
        table = decode_candidate_columns_v2(raw(decision['candidate_array_key'], record=False), decision['candidate_descriptor'],
            action_contract=actions.action_contract, purpose='qualified_game', synthetic_capacity=False,
            layout=LAYOUT, expected_frame_hash=frame.frame_hash, max_positions=18, max_bytes=LIMIT)
        selection = parse_selection(decision['selection'])
        selection.validate(table, frame)
        fp, ep = f'e.0.f.{index}', f'e.0.event-tokens.{index}'
        if _json(raw(fp + '.payload')) != frame.raw:
            raise ContractError('final64 duplicate raw frame payload differs')
        admit(64 * (sum(value.nbytes for name, value in arrays.items() if name.startswith((fp + '.', ep + '.')))) + 65536)
        token = tokenize_frame(frame, view, aggregate_window=aggregate_window, layout=LAYOUT)
        candidate = tokenize_table(table, frame, 18, aggregate_window=aggregate_window, layout=LAYOUT)
        expected_frame, expected_candidates = asdict(token), asdict(candidate)
        for fields, prefix, reference in ((FRAME_FIELDS, fp, expected_frame), (EVENT_FIELDS, ep, expected_frame),
                                         (CANDIDATE_FIELDS, fp + '.candidate', expected_candidates)):
            for name in fields:
                value = get(prefix + '.' + name, _token_dtype(name))
                expected_array = np.asarray(reference[name], dtype=value.dtype)
                if value.shape != expected_array.shape or not np.array_equal(value, expected_array):
                    raise ContractError('final64 raw token/reference dtype/shape/value differs')
        frames.append(frame)
        tables.append(table)
        selections.append(selection)
    endpoint = closed(metadata['endpoint'], {'end_kind', 'outcome', 'reason'})
    if endpoint['end_kind'] != 'natural' or endpoint['reason'] is not None or canonical(endpoint['outcome']) != canonical(expected['outcome']):
        raise ContractError('final64 original natural endpoint differs')
    admit(1024 * (len(frames) + 1))
    record = game_episode_records(view, tuple(frames), tuple(tables), tuple(selections), 'natural',
        endpoint['outcome'], actions=actions, rules=rules)
    correspondence = closed(metadata['correspondence'], {'episode_digest', 'frames'})
    if (used != set(arrays) or _episode_correspondence_digest(record) != metadata['episode_digest']
            or correspondence['episode_digest'] != metadata['episode_digest']):
        raise ContractError('final64 complete original record correspondence differs')
    # Exact frame/table/full-precision selected identities, not a token-only hash.
    actual_frames = [{'frame_hash': frame.frame_hash, 'payload_sha256': hashlib.sha256(frame.payload).hexdigest(),
        'revision': frame.revision, 'interval_start': frame.interval_start,
        'interval_start_hash': frame.interval_start_hash, 'boundary': asdict(frame.boundary),
        'table_hash': table.table_hash, 'selection': asdict(selection)}
        for frame, table, selection in zip(frames, tables, selections, strict=True)]
    if canonical(correspondence['frames']) != canonical(actual_frames):
        raise ContractError('final64 ordered frame/table/selection correspondence differs')
    check()
    return record
