"""Concealed draw markers remain in corrected raw histories only."""

import copy
import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common import card_instance_validation
from gf_common.candidate_columns import decode_candidate_columns_v2, encode_candidate_columns_v2
from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig
from gf_common.structured import CandidateTable, EventBank, Selection, event_delta
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT, LEGACY_LAYOUT
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature, tokenize_frame, validate_frame_tokens
from gf_common.structured_wire import parse_frame
from gf_models.arrays import sha256
from gf_models.sharded_arrays import read_bundle
from gf_models.structured_artifacts import semantics_v3
from gf_training import native_backend
from gf_training.structured_rollout import (
    EVENT_FIELDS,
    _write_prepared_game_payload,
    _write_rollout,
    game_episode_records,
    game_rollout,
    read_game_rollout,
)
from tests.model3_support import model3
from tests.test_card_instance_learning import ACTIONS, RULES, correspondence
from tests.test_card_instance_native_fixture_tokens import CASES, construct, seal, unpack
from tests.test_guardian_event_namespace import own_frame

MARKER = {"type": "artifact_drawn", "player": 1, "definition_id": "concealed"}
DRAW = {"type": "artifact_drawn", "player": 1, "definition_id": "bronze-club"}
SAFE = {"type": "safe", "player": 2}
MARKERS = (MARKER,) * 8


@pytest.mark.parametrize(
    "events, visible, count, truncated",
    [
        pytest.param((DRAW, *MARKERS, SAFE), (DRAW, SAFE), 2, 0, id="mixed"),
        pytest.param(MARKERS, (), 0, 0, id="all-concealed"),
        pytest.param((DRAW,) * 127 + MARKERS + (SAFE,), (DRAW,) * 127 + (SAFE,), 128, 0, id="full"),
        pytest.param((DRAW,) * 128 + MARKERS + (SAFE,), (DRAW,) * 128 + (SAFE,), 128, 1, id="truncated"),
    ],
)
def test_concealed_draws_do_not_consume_interval_or_history_rows(events, visible, count, truncated):
    _, bank, actual = own_frame(events)
    _, _, expected = own_frame(visible)
    assert bank.view().events() == events
    assert actual.event_start == 0 and actual.event_end == len(events)
    previous = bank.view()._prefixes[0]
    for row in events:
        previous = hashlib.sha256(bytes.fromhex(previous) + canonical(row).encode()).hexdigest()
    assert bank.boundary.prefix_hash == previous
    assert actual.history[369:371] == (count, truncated)
    for field in ("interval_numerics", "interval_present", "interval_categories", "history"):
        assert getattr(actual, field) == getattr(expected, field)
    validate_frame_tokens(actual, layout=LAYOUT)
    with pytest.raises(ContractError):
        validate_frame_tokens(replace(actual, interval_present=((False,) * 11,) + actual.interval_present), layout=LAYOUT)


def test_hidden_rows_cannot_extend_the_raw_frame_interval_limit():
    _, _, tokens = own_frame(())
    with pytest.raises(ContractError):
        validate_frame_tokens(replace(tokens, event_end=65537), layout=LAYOUT)


@pytest.mark.parametrize("events,rows", [(MARKERS, 0), ((DRAW, *MARKERS[:6], DRAW), 2)])
def test_prepared_rollout_preserves_filtered_shapes_and_revalidates_original_rows(tmp_path, events, rows):
    frame, bank, _ = own_frame(events)
    table = CandidateTable.from_actions(frame, [{"type": "discard", "card_refs": [frame.references[0]]}],
        action_contract=ACTIONS.action_contract, purpose="qualified_game", synthetic_capacity=False)
    selection = Selection(table.table_hash, 0, "p65-marker-fixture", 0.0, 0.0)
    record = game_episode_records(bank.view(), (frame,), (table,), (selection,), "natural",
        {"win": {"winner": 1}}, actions=ACTIONS, rules=RULES)
    model = model3()
    rollout = game_rollout((record,), model, actions=ACTIONS, rules=RULES)
    original = tmp_path / "original"
    manifest = _write_rollout(original, rollout, shard_bytes=65536)
    metadata = {key: value for key, value in manifest.items() if key not in ("schema_version", "kind", "data")}
    arrays = read_bundle(original, manifest["data"])
    keys = [f"e.0.event-tokens.0.{field}" for field in EVENT_FIELDS]
    for key, width in zip(keys, (11, 11, 12), strict=True):
        assert arrays[key].shape == ((rows, width) if rows else (0,))

    def prepare(path, values):
        _write_prepared_game_payload(path, metadata, values, model, actions=ACTIONS, rules=RULES,
            resident_bytes=0, extra_owned_bytes=0, max_owned_bytes=64 * 1024**2, shard_bytes=65536,
            check=lambda: None, reserve=lambda _: None)

    def read(path):
        return read_game_rollout(path, sha256(path / "rollout.json"), model, actions=ACTIONS, rules=RULES,
            expected_correspondence=correspondence(rollout),
            declaration=semantics_v3(model, layout=LAYOUT, action_contract=ACTIONS.action_contract, rules=RULES))

    saved = tmp_path / "prepared"
    prepare(saved, arrays)
    assert read(saved).episodes[0].records.bank.events() == events
    if rows:
        short = {key: arrays[key][:1] for key in keys}
        with pytest.raises(ContractError, match="shape"):
            prepare(tmp_path / "unequal", arrays | {keys[0]: short[keys[0]]})
        tampered = tmp_path / "tampered"
        prepare(tampered, arrays | short)
        with pytest.raises(ContractError, match="stored tokens differ"):
            read(tampered)
        changed = arrays[keys[-1]].copy()
        changed[0, 0] += 1
        changed.flags.writeable = False
        altered = tmp_path / "altered-categories"
        prepare(altered, arrays | {keys[-1]: changed})
        with pytest.raises(ContractError, match="stored tokens differ"):
            read(altered)


@pytest.mark.parametrize("events", [MARKERS, (DRAW,) * 128 + MARKERS + (SAFE,)])
def test_native_export_matches_python_filtered_events(events):
    native = pytest.importorskip("gf_native")
    case = next(row for row in CASES if row['envelope']['viewer'] == 1 and len(row['envelope']['decisions']) == 1)
    raw = copy.deepcopy(case['envelope'])
    raw['events'] = events
    row = raw['decisions'][0]
    old_frame = parse_frame(row['frame'], layout=LAYOUT)
    old_table = decode_candidate_columns_v2(bytes.fromhex(case['columns_hex'][0]), row['columns'],
        layout=LAYOUT, action_contract=LAYOUT.action_contract, purpose='component', synthetic_capacity=False)
    actions = [old_table.to_action(i, old_frame) for i in range(len(old_table.kinds))]
    bank = EventBank(1, raw['episode'], layout=LAYOUT)
    bank.append(event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, events, layout=LAYOUT))
    row['frame']['boundary'] = asdict(bank.boundary)
    seal(row['frame'])
    frame = parse_frame(row['frame'], layout=LAYOUT)
    table = CandidateTable.from_actions(frame, actions, action_contract=LAYOUT.action_contract,
        purpose='component', synthetic_capacity=False)
    payload, row['columns'] = encode_candidate_columns_v2(table, row['columns']['file'], layout=LAYOUT)
    expected = tokenize_frame(frame, bank.view(), layout=LAYOUT)
    owner = construct(native, case, raw=canonical(raw).encode(), payloads=(payload,), aggregate_window=128)
    try:
        _, arrays = unpack(owner.export_tokens())
        for key, array in arrays.items():
            if '.candidate.' not in key:
                wanted = np.asarray(getattr(expected, key.rsplit('.', 1)[1]), dtype=array.dtype)
                np.testing.assert_array_equal(array, wanted, err_msg=key)
    finally:
        owner.close()


@pytest.mark.parametrize("row", [MARKER | {"type": "artifact_used"}, MARKER | {"definition_id": "unknown-public-id"}])
def test_corrected_event_dto_and_bank_reject_other_markers_and_unknown_ids(row):
    with pytest.raises(ContractError):
        card_instance_validation.validate_events([row])
    with pytest.raises(ContractError):
        own_frame((row,))


def test_historical_banks_keep_their_original_marker_behavior():
    legacy = EventBank(1, "old-draw", layout=LEGACY_LAYOUT)
    legacy.append(event_delta(1, legacy.episode, 0, legacy.boundary.prefix_hash, MARKERS, layout=LEGACY_LAYOUT))
    assert legacy.view().events() == MARKERS
    old = EventBank(1, "old-draw", layout=CARD_INSTANCE_LAYOUT)
    with pytest.raises(ContractError):
        event_delta(1, old.episode, 0, old.boundary.prefix_hash, MARKERS, layout=CARD_INSTANCE_LAYOUT)


def test_old_native_reader_is_allowed_only_for_pre_marker_recorded_build(monkeypatch):
    info = {
        "card_instance_contract": native_backend.CARD_INSTANCE_CONTRACT,
        "card_instance_signature": signature(128, layout=LAYOUT),
        "card_instance_models": "mlp,gru,transformer",
        "card_instance_guardian_acted": native_backend.CARD_INSTANCE_GUARDIAN_ACTED,
        "card_instance_defense_priority": native_backend.CARD_INSTANCE_DEFENSE_PRIORITY,
        "native_source_sha256": "a" * 64,
    }
    methods = (
        "control_summary", "original_config_inputs", "authorize", "export_cohort",
        "retire_payload", "bind_numeric", "begin_update", "prepare_commit", "finalize_read",
        "recheck", "close", "used_bytes", "peak_bytes", "cancel_handle", "control_reserved_bytes",
    )
    reader = SimpleNamespace(
        build_info=lambda: info,
        CardInstanceNativeSession=type("PreviousReader", (), dict.fromkeys(methods, lambda: None)),
        CancelHandle=type("Cancel", (), {"cancel": lambda: None}),
    )
    settings = PreparationConfig(backend="rust", reader="participant-structured-v2")
    identities = []

    def identity(module):
        identities.append(module)
        return {"build": info, "gil_disabled": False}

    monkeypatch.setattr(native_backend, "binary_identity", identity)
    with pytest.raises(ContractError, match="concealed draw"):
        native_backend.card_instance_reader_settings(settings, reader)
    assert not identities
    assert native_backend.card_instance_reader_settings(settings, reader, recorded_build=info) == asdict(settings)
    assert identities == [reader]
    current = info | {"card_instance_concealed_draw": native_backend.CARD_INSTANCE_CONCEALED_DRAW}
    with pytest.raises(ContractError, match="concealed draw"):
        native_backend.card_instance_reader_settings(settings, reader, recorded_build=current)
    assert identities == [reader]
