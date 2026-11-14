"""ADR0040 closed public Acted union; component evidence grants no Source."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common import structured_vocab as vocab
from gf_common.candidate_columns import decode_candidate_columns_v2, encode_candidate_columns_v2
from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig
from gf_common.structured import CandidateTable, EventBank, event_delta
from gf_common.structured_layout import (
    CARD_INSTANCE_LAYOUT,
    CORRECTED_CARD_INSTANCE_LAYOUT,
    LEGACY_LAYOUT,
)
from gf_common.structured_tokens import tokenize_frame
from gf_common.structured_wire import parse_frame
from gf_training import native_backend
from tests.test_card_instance_native_fixture_tokens import CASES, construct, seal, unpack
from tests.test_guardian_event_namespace import assert_event_time_tokens, own_frame
from tests.test_hpo_native import candidate  # noqa: F401 - explicit private candidate

FIXTURE = Path(__file__).parent / 'fixtures/guardian_acted_public_91d5926.json'
CAPABILITY = 'card-or-guardian-v1'


def test_retained_public_events_preserve_raw_ids_viewers_order_and_prefix():
    evidence = json.loads(FIXTURE.read_text())
    assert evidence['simulator_revision'] == '91d5926a7572db327eaea3751e7c24f428120996'
    assert evidence['source_sha256'] == '661a6c91b7e699eadaa4083a4cc43333fb0535eba5a681d281b570519744aecb'
    assert [(r['viewer'], r['event']['player'], r['event']['definition_id'])
        for r in evidence['records']] == [(1, 2, 'moon'), (2, 1, 'earth')]
    for row in evidence['records']:
        _, bank, tokens = own_frame((row['event'],), row['viewer'])
        assert bank.view().events() == (row['event'],)
        index = vocab.GUARDIANS.index(row['event']['definition_id'])
        assert tokens.interval_categories[0][1] == 298 + index
        assert tokens.interval_categories[0][4] == 3
        assert tokens.history[359 + index] == 1


@pytest.mark.parametrize('actor', [1, 2])
@pytest.mark.parametrize('name', (*vocab.DEFINITIONS, *vocab.GUARDIANS))
def test_all306_acted_ids_have_exact_existing_tokens_counts_order_and_prefix(name, actor):
    if name in vocab.GUARDIANS:
        index = vocab.GUARDIANS.index(name)
        category, history = 298 + index, 359 + index
    else:
        index = vocab.DEFINITIONS.index(name)
        category, history = 2 + index, 63 + index
    assert_event_time_tokens(name, 'guardian_acted', actor, category, history)


@pytest.mark.parametrize('layout', [LEGACY_LAYOUT, CARD_INSTANCE_LAYOUT])
@pytest.mark.parametrize('name', vocab.GUARDIANS)
def test_old_layouts_keep_their_previous_acted_domains(layout, name):
    bank = EventBank(1, 'old-acted', layout=layout)
    events = [{'type': 'guardian_acted', 'player': 1, 'definition_id': name}]
    if layout is LEGACY_LAYOUT:
        bank.append(event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, events, layout=layout))
        assert bank.view().events() == tuple(events)
    else:
        with pytest.raises(ContractError):
            event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, events, layout=layout)


@pytest.mark.parametrize('marker', [None, '', 'card-only-v1', 'card-or-guardian-v2'])
@pytest.mark.parametrize('entry', ['reader', 'refresh'])
def test_current_preflights_refuse_old_validator_before_identity_or_ownership(monkeypatch, marker, entry):
    from gf_common.structured_tokens import signature
    info = {'card_instance_refresh': native_backend.CARD_INSTANCE_REFRESH_BASELINE,
        'card_instance_contract': native_backend.CARD_INSTANCE_CONTRACT,
        'card_instance_models': 'mlp,gru,transformer',
        'card_instance_signature': signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        'native_source_sha256': 'a'*64}
    if marker is not None:
        info['card_instance_guardian_acted'] = marker
    module = SimpleNamespace(build_info=lambda: info.items())
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *_: pytest.fail('identity reached'))
    with pytest.raises(ContractError, match='GuardianActed'):
        if entry == 'reader':
            native_backend.card_instance_reader_settings(
                PreparationConfig(backend='rust', reader='participant-structured-v2'), module)
        else:
            native_backend.require_native_refresh_baseline(module)


def test_current_refresh_preflight_accepts_exact_both_markers():
    module = SimpleNamespace(build_info=lambda: {'card_instance_refresh':
        native_backend.CARD_INSTANCE_REFRESH_BASELINE, 'card_instance_guardian_acted': CAPABILITY}.items())
    native_backend.require_native_refresh_baseline(module)


@pytest.mark.parametrize('viewer', [1, 2])
@pytest.mark.parametrize('offset', range(0, 326, 50))
def test_native_python_union_tokens_match_without_changing_existing_recipe(candidate, viewer, offset):  # noqa: F811
    assert dict(candidate.build_info())['card_instance_guardian_acted'] == CAPABILITY
    events = [
        {'type': family, 'player': 1 + i % 2, 'definition_id': name}
        for i, (family, name) in enumerate([
            *(('guardian_acted', name) for name in (*vocab.DEFINITIONS, *vocab.GUARDIANS)),
            *((family, name) for family in ('guardian_set', 'guardian_left') for name in vocab.GUARDIANS),
        ][offset:offset+50])]
    case = next(c for c in CASES if c['envelope']['viewer'] == viewer and len(c['envelope']['decisions']) == 1)
    raw = copy.deepcopy(case['envelope'])
    raw['events'] = events
    bank = EventBank(viewer, raw['episode'], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    bank.append(event_delta(viewer, bank.episode, 0, bank.boundary.prefix_hash, events,
        layout=CORRECTED_CARD_INSTANCE_LAYOUT))
    row = raw['decisions'][0]
    old_frame = parse_frame(row['frame'], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    old_table = decode_candidate_columns_v2(bytes.fromhex(case['columns_hex'][0]), row['columns'],
        layout=CORRECTED_CARD_INSTANCE_LAYOUT, action_contract=CORRECTED_CARD_INSTANCE_LAYOUT.action_contract,
        purpose='component', synthetic_capacity=False)
    actions = [old_table.to_action(i, old_frame) for i in range(len(old_table.kinds))]
    row['frame']['boundary'] = asdict(bank.boundary)
    seal(row['frame'])
    frame = parse_frame(row['frame'], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    expected = tokenize_frame(frame, bank.view(), aggregate_window=128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    table = CandidateTable.from_actions(frame, actions, action_contract=CORRECTED_CARD_INSTANCE_LAYOUT.action_contract,
        purpose='component', synthetic_capacity=False)
    payload, row['columns'] = encode_candidate_columns_v2(table, row['columns']['file'],
        layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    owner = construct(candidate, case, raw=canonical(raw).encode(), payloads=(payload,), aggregate_window=128)
    try:
        _, arrays = unpack(owner.export_tokens())
        for key, array in arrays.items():
            if '.candidate.' not in key:
                assert array.tobytes() == np.asarray(getattr(expected, key.rsplit('.', 1)[1]), dtype=array.dtype).tobytes(), key
        assert bank.view().events() == tuple(events)
    finally:
        owner.close()


@pytest.mark.parametrize('family,name', [
    ('guardian_acted', 'unknown-public-id'), ('guardian_acted', 'sun'),
    ('guardian_set', 'fireball'), ('guardian_left', 'seafood-soup'),
    ('artifact_used', 'earth'),
])
def test_both_validators_refuse_unknown_or_wrong_namespace(candidate, family, name):  # noqa: F811
    events = [{'type': family, 'player': 1, 'definition_id': name}]
    with pytest.raises(ContractError):
        own_frame(events)
    case = CASES[0]
    raw = copy.deepcopy(case['envelope'])
    raw['events'] = events
    frame = raw['decisions'][0]['frame']
    previous = frame['interval_start_hash']
    frame['boundary'] = {'end': 1, 'prefix_hash': hashlib.sha256(
        bytes.fromhex(previous) + canonical(events[0]).encode()).hexdigest()}
    seal(frame)
    with pytest.raises((ValueError, RuntimeError), match='namespace'):
        construct(candidate, case, raw=canonical(raw).encode())
