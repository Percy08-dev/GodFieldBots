"""Controlled Tree3 RAW schema/token checks; no pair, model or Source authority."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.card_instance_game_tree_runtime import validate_tree_metadata
from gf_training.card_instance_journal import read_card_instance_journal

ROOT = Path(__file__).parents[1]
GOLDEN = json.loads((ROOT/'native/tests/fixtures/card_instance_tree3_raw_metadata_rules_v2.json').read_text())
CASE = json.loads((ROOT/'native/tests/fixtures/card_instance_tree3_raw_journal_rules_v2.json').read_text())['journals'][0]


def write_case(directory, rows=None):
    rows = CASE['rows'] if rows is None else rows
    path = directory/'player-1.jsonl'
    path.write_text(''.join(canonical(row)+'\n' for row in rows))
    (directory/CASE['rows'][2]['table']['file']).write_bytes(bytes.fromhex(CASE['bytes_hex']))
    return path


def test_tree3_controlled_metadata_full_python_reader_and_raw_reference(tmp_path):
    actions = action_config_v3(GOLDEN['actions'])
    _, label = validate_tree_metadata(GOLDEN['tree'], actions=actions, rules=card_instance_rules())
    assert label == GOLDEN['policy_label']
    original = read_card_instance_journal(write_case(tmp_path), actions=actions)
    assert original.header['tree'] == GOLDEN['tree'] and len(original.accepted) == 1
    for decision in original.decisions:
        assert decision.selection.logprob is None and decision.selection.value is None
        assert json.loads(canonical(asdict(tokenize_frame(decision.frame, original.bank, aggregate_window=128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)))) == CASE['frame_tokens']
        assert json.loads(canonical(asdict(tokenize_table(decision.generated.table, decision.frame, aggregate_window=128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)))) == CASE['candidate_tokens']


@pytest.mark.parametrize('field', ['logprob', 'value'])
def test_tree3_python_rejects_each_finite_selection_stat(tmp_path, field):
    rows = copy.deepcopy(CASE['rows'])
    rows[2]['selection'][field] = 0.0
    with pytest.raises(ContractError):
        read_card_instance_journal(write_case(tmp_path, rows), actions=action_config_v3(GOLDEN['actions']))


@pytest.fixture
def candidate():
    native = pytest.importorskip('gf_native')
    if not callable(getattr(getattr(native, 'StructuredJournalTokens', None), 'card_instance', None)):
        pytest.skip('isolated corrected RAW candidate required; installed extension unchanged')
    return native


def raw_settings(**override):
    return canonical({'memory_bytes': 512*1024**2, 'max_json_bytes': 64*1024**2,
        'max_record_bytes': 8*1024**2, **override}).encode()


def open_raw(candidate, path, **kwargs):
    import time
    return candidate.StructuredJournalTokens.card_instance(str(path.parent), path.name,
        kwargs.pop('settings', raw_settings()), kwargs.pop('deadline', time.monotonic()+60), **kwargs)


ALL_CASES = [*json.loads((ROOT/'native/tests/fixtures/card_instance_journal6_rules_v2.json').read_text())['journals'], CASE]


@pytest.mark.parametrize('case', ALL_CASES, ids=lambda case: case['label'])
def test_corrected_raw_ffi_full_originals_tokens_order_prefix_and_metadata(candidate, case, tmp_path):
    from gf_common.structured_wire import frame_dict
    directory = tmp_path
    path = directory/'player-1.jsonl'
    path.write_text(''.join(canonical(row)+'\n' for row in case['rows']))
    (directory/case['rows'][2]['table']['file']).write_bytes(bytes.fromhex(case['bytes_hex']))
    original = read_card_instance_journal(path, actions=action_config_v3(case['rows'][0]['actions']))
    owner = open_raw(candidate, path)
    try:
        raw = json.loads(owner.diagnostic())
        assert raw['validation_scope'] == 'journal6-codec2-rawtokens-only'
        assert raw['qualified'] is False and raw['parent_qualification'] == 'not_performed'
        assert raw['header'] == original.header and raw['accepted'] == list(original.accepted)
        assert raw['bank']['events'] == list(original.bank.events())
        assert raw['bank']['prefixes'] == list(original.bank._prefixes)
        assert raw['end'] == original.end
        for decision, actual in zip(original.decisions, raw['decisions'], strict=True):
            assert actual['frame'] == frame_dict(decision.frame, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            assert actual['table'] == json.loads(canonical(asdict(decision.generated.table)))
            assert actual['selection'] == asdict(decision.selection)
            assert actual['search_report'] == json.loads(canonical(asdict(decision.generated.report)))
            assert actual['frame_tokens'] == case['frame_tokens']
            assert actual['candidate_tokens'] == case['candidate_tokens']
        assert owner.used_bytes() >= owner.control_reserved_bytes() > 0
        # The first decoded packet remains jointly charged. A second export is
        # an ordinary refusal; the same original guard still remains available.
        with pytest.raises(RuntimeError):
            owner.diagnostic()
        owner.recheck(len(canonical(raw)))
        assert not hasattr(owner, 'authorize') and not hasattr(owner, 'export_cohort')
    finally:
        owner.close()
    assert owner.used_bytes() == 0
    owner.close()
    with pytest.raises(RuntimeError):
        owner.recheck()
    assert raw['header'] == original.header  # immutable bytes survive owner close
    with pytest.raises(ValueError):
        candidate.StructuredJournalTokens(str(path.parent), path.name, raw_settings(), __import__('time').monotonic()+60)


@pytest.mark.parametrize('operation', ['diagnostic', 'recheck'])
def test_corrected_raw_quota_failure_then_smaller_retry_never_revives(candidate, tmp_path, operation):
    owner = open_raw(candidate, write_case(tmp_path))
    with pytest.raises(MemoryError):
        getattr(owner, operation)(2**40)
    for method in (owner.diagnostic, owner.recheck, owner.close, owner.close):
        with pytest.raises(MemoryError):
            method()
    assert owner.used_bytes() == 0


def test_corrected_raw_changed_sidecar_restore_cannot_revive(candidate, tmp_path):
    path = write_case(tmp_path)
    columns = tmp_path/CASE['rows'][2]['table']['file']
    previous = columns.read_bytes()
    owner = open_raw(candidate, path)
    columns.write_bytes(previous[:-1]+bytes([previous[-1]^1]))
    with pytest.raises(ValueError):
        owner.recheck()
    columns.write_bytes(previous)
    for method in (owner.diagnostic, owner.recheck, owner.close, owner.close):
        with pytest.raises(ValueError):
            method()
    assert owner.used_bytes() == 0


@pytest.mark.parametrize('expired', [False, True])
def test_corrected_raw_cancel_deadline_priority_and_cleanup(candidate, tmp_path, expired):
    import time
    path = write_case(tmp_path)
    with pytest.raises(TimeoutError):
        open_raw(candidate, path, deadline=time.monotonic()-1)
    with pytest.raises(MemoryError):
        open_raw(candidate, path, settings=raw_settings(memory_bytes=1))
    handle = candidate.CancelHandle(time.monotonic()+60)
    if expired:
        owner = open_raw(candidate, path, deadline=time.monotonic()+.1)
        time.sleep(.12)
        expected = TimeoutError
    else:
        owner = open_raw(candidate, path, cancel_handle=handle)
        handle.cancel()
        expected = RuntimeError
    # Cleanup preserves the initiating error and retires physical original owners
    # even when no subsequent operational guard can succeed.
    with pytest.raises(expected):
        owner.diagnostic()
    with pytest.raises(expected):
        owner.recheck()
    with pytest.raises(expected):
        owner.close()
    assert owner.used_bytes() == 0
    handle.cancel()
    with pytest.raises(RuntimeError, match='cancel'):
        open_raw(candidate, path, cancel_handle=handle, deadline=time.monotonic()-1)


@pytest.mark.parametrize('mutation', ['mixed', 'component', 'finite-logprob', 'finite-value', 'foreign-prefix', 'private-mapping', 'version2'])
def test_tree_raw_ffi_closed_mutations_before_payload(candidate, tmp_path, mutation):
    rows = copy.deepcopy(CASE['rows'])
    if mutation == 'mixed':
        rows[0]['neural'] = rows[0]['tree']
    elif mutation == 'component':
        rows[0]['tree']['policy_options']['source_origin']['purpose'] = 'component'
    elif mutation.startswith('finite-'):
        rows[2]['selection'][mutation.removeprefix('finite-')] = 0.0
    elif mutation == 'foreign-prefix':
        rows[2]['frame']['boundary']['prefix_hash'] = '0'*64
    elif mutation == 'private-mapping':
        rows[0]['tree']['inference_manifest']['feature_layout']['raw_field_mapping'][0]['segments'] = ['other_private']
    else:
        rows[0]['model_version'] = '2'
    with pytest.raises(ValueError):
        open_raw(candidate, write_case(tmp_path, rows))


def test_tree_raw_exact_published_marker_and_live_preflight_grant_no_source(candidate):
    from gf_common.preprocessing import PreparationConfig
    from gf_training.native_backend import CARD_INSTANCE_TREE_RAW, tree_raw_reader_settings
    assert dict(candidate.build_info())['card_instance_tree_raw'] == CARD_INSTANCE_TREE_RAW
    config = PreparationConfig(backend='rust', reader='participant-structured-v2')
    assert tree_raw_reader_settings(config, candidate) == asdict(config)
    assert not hasattr(candidate.StructuredJournalTokens, 'authorize')
    assert not hasattr(candidate.StructuredJournalTokens, 'export_cohort')


@pytest.mark.parametrize('mutation', ['none', 'marker', 'signature', 'abi', 'method', 'cancel', 'binary'])
def test_tree_raw_preflight_closed_stub_contract_before_any_reader_allocation(candidate, monkeypatch, mutation):
    from types import SimpleNamespace

    from gf_common.preprocessing import PreparationConfig
    from gf_training import native_backend

    info = dict(candidate.build_info())
    info['card_instance_tree_raw'] = native_backend.CARD_INSTANCE_TREE_RAW
    reader = type('UNADMITTEDReader', (), {key: lambda *a: None for key in
        ('card_instance', 'diagnostic', 'recheck', 'close', 'cancel_handle', 'used_bytes', 'peak_bytes', 'control_reserved_bytes')})
    module = SimpleNamespace(build_info=lambda: info.items(), StructuredJournalTokens=reader,
        CardInstanceNativeSession=candidate.CardInstanceNativeSession, CancelHandle=candidate.CancelHandle)
    identity = {'build': info, 'gil_disabled': False, 'python_abi': 'cpython-314'}
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *a: identity)
    if mutation == 'marker':
        info.pop('card_instance_tree_raw')
    elif mutation == 'signature':
        info['card_instance_signature'] = '0'*64
    elif mutation == 'abi':
        info['abi'] = 'cp313'
    elif mutation == 'method':
        reader.card_instance = None
    elif mutation == 'cancel':
        module.CancelHandle = object
    elif mutation == 'binary':
        identity['gil_disabled'] = True
    config = PreparationConfig(backend='rust', reader='participant-structured-v2')
    if mutation == 'none':
        assert native_backend.tree_raw_reader_settings(config, module) == asdict(config)
    else:
        with pytest.raises(ContractError):
            native_backend.tree_raw_reader_settings(config, module)


@pytest.mark.parametrize('window', [1, 65536])
def test_tree_raw_actual_own_history_window_uses_full_original_bank(candidate, tmp_path, window):
    from gf_models.card_instance_game_tree_runtime import policy_label_tree_game_v3
    from gf_models.structured_tree_config import architecture_config_v3, architecture_identity_v3
    from gf_models.structured_tree_encoder import feature_layout_v3

    rows = copy.deepcopy(CASE['rows'])
    header = rows[0]
    raw = header['tree']
    safe, manifest = raw['policy_options'], raw['inference_manifest']
    safe['model']['encoder']['aggregate_window'] = window
    architecture = architecture_config_v3(safe['model'])
    layout = feature_layout_v3(architecture.encoder)
    safe['feature_signature'] = layout['feature_signature']
    manifest['model'] = copy.deepcopy(safe['model'])
    manifest['feature_layout'] = layout
    for key in ('feature_signature', 'token_signature', 'feature_names', 'categorical_indices'):
        manifest[key] = layout[key]
    manifest['architecture_identity'] = architecture_identity_v3(architecture)
    manifest['semantics']['token_signature'] = layout['token_signature']
    actions = action_config_v3(header['actions'])
    label = policy_label_tree_game_v3(safe, actions=actions, rules=card_instance_rules())
    header.update(aggregate_window=window, token_signature=layout['token_signature'], policy=label)
    rows[2]['selection']['policy_version'] = label
    path = write_case(tmp_path, rows)
    original = read_card_instance_journal(path, actions=actions)
    owner = open_raw(candidate, path)
    try:
        actual = json.loads(owner.diagnostic())
        assert actual['projection']['aggregate_window'] == window
        for decision, raw_decision in zip(original.decisions, actual['decisions'], strict=True):
            expected = tokenize_frame(decision.frame, original.bank, aggregate_window=window,
                layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            table = tokenize_table(decision.generated.table, decision.frame, aggregate_window=window,
                layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            assert raw_decision['frame_tokens'] == json.loads(canonical(asdict(expected)))
            assert raw_decision['candidate_tokens'] == json.loads(canonical(asdict(table)))
    finally:
        owner.close()


@pytest.mark.parametrize('zero', [True, False])
def test_tree_raw_zero_natural_and_pending_truncated_keep_real_distinction(candidate, tmp_path, zero):
    rows = copy.deepcopy(CASE['rows'])
    if zero:
        rows = [rows[0], rows[-1]]
    else:
        end = rows[-1]
        end.update(end_kind='truncated', outcome=None, terminal_reward=None, final_observation=None,
            pending_decisions=[0], simulator_abort_reason='turn_limit_exceeded')
        rows = [*rows[:3], end]
    path = write_case(tmp_path, rows)
    original = read_card_instance_journal(path, actions=action_config_v3(rows[0]['actions']))
    owner = open_raw(candidate, path)
    try:
        actual = json.loads(owner.diagnostic())
        assert actual['accepted'] == []
        assert actual['end'] == original.end
        assert actual['end']['end_kind'] == ('natural' if zero else 'truncated')
        assert len(actual['decisions']) == (0 if zero else 1)
        if not zero:
            assert actual['end']['outcome'] is actual['end']['terminal_reward'] is None
    finally:
        owner.close()


def test_old_tree2_raw_constructor_and_null_statistics_stay_exact(candidate, make_input, tmp_path):
    from gf_common.action_contract import engineering_contract
    from gf_common.structured_wire import frame_dict
    from gf_training.structured_journal import read_structured_journal
    from tests.test_native_structured_journal import journal

    path = journal(make_input, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(model_id='lightgbm', model_version='2', generation_schema=None,
        policy='UNADMITTED-legacy-tree2-raw')
    rows[2]['selection'].update(policy_version='UNADMITTED-legacy-tree2-raw', logprob=None, value=None)
    path.write_text(''.join(canonical(row)+'\n' for row in rows))
    original = read_structured_journal(path, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    import time
    owner = candidate.StructuredJournalTokens(str(path.parent), path.name, raw_settings(), time.monotonic()+60)
    try:
        raw = json.loads(owner.diagnostic())
        assert raw['validation_scope'] == 'journal5-columns-rawtokens-only'
        assert raw['header'] == original.header
        assert raw['decisions'][0]['frame'] == frame_dict(original.decisions[0].frame)
        assert raw['decisions'][0]['selection'] == asdict(original.decisions[0].selection)
        assert raw['decisions'][0]['frame_tokens'] == json.loads(canonical(asdict(
            tokenize_frame(original.decisions[0].frame, original.bank))))
    finally:
        owner.close()
    with pytest.raises(ValueError):
        open_raw(candidate, path)


def test_tree_raw_portable_execution_boundary_differs_from_python_physical_graph_guard(candidate, tmp_path):
    """RAW allocates no Booster; this boundary is NOT physical-admission parity."""
    from gf_common.domain import digest
    from gf_models.card_instance_game_tree_runtime import (
        game_tree_settings,
        policy_label_tree_game_v3,
    )
    from gf_models.structured_tree_config import (
        TreeDescriptor,
        configured_tree_v3,
        resource_estimate,
    )

    rows = copy.deepcopy(CASE['rows'])
    header = rows[0]
    safe = header['tree']['policy_options']
    actions = action_config_v3(header['actions'])
    model = configured_tree_v3({**safe['model'], 'checkpoint': safe['checkpoint'],
        'inference': safe['inference'], 'resources': safe['resources']})
    descriptor = TreeDescriptor(**safe['tree_descriptor'])
    minimum = resource_estimate(model, model_bytes=descriptor.model_bytes)['total_bytes']
    safe['resources']['max_owned_bytes'] = minimum
    safe['resources']['max_array_bytes'] = resource_estimate(model)['array_bytes']
    caps, costs, _, _ = game_tree_settings(safe, actions=actions, rules=card_instance_rules())
    label = policy_label_tree_game_v3(safe, actions=actions, rules=card_instance_rules())
    header.update(policy=label, caps=asdict(caps), numeric_costs=asdict(costs))
    rows[2]['selection']['policy_version'] = label
    rows[2]['search_report'].update(numeric_costs=asdict(costs), numeric_costs_hash=digest(asdict(costs)))
    path = write_case(tmp_path, rows)
    # Python's original physical `_owned` metadata guard is left unchanged.
    # Native independently admits actual metadata/copies under its global quota.
    with pytest.raises(ContractError, match='owned resource'):
        read_card_instance_journal(path, actions=actions)
    owner = open_raw(candidate, path)
    try:
        raw = json.loads(owner.diagnostic())
        assert raw['qualified'] is False and raw['header']['numeric_costs'] == asdict(costs)
        assert raw['decisions'][0]['frame_tokens'] == CASE['frame_tokens']
    finally:
        owner.close()
