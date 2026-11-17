"""Corrected FFI diagnostics; no game or Source promotion."""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.card_instance import action_config_v3
from gf_common.domain import ContractError, canonical
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_training.native_backend import card_instance_reader_settings
from gf_training.native_card_instance import _NativeGameInputs
from gf_training.native_structured_session import _payload_views
from tests.card_instance_tree_fixture_support import require_current_capture_binding

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'runs/card-instance-v2-91d5926/ppo-acceptance/mlp-cpu-resume-1'


@pytest.fixture
def candidate():
    native = pytest.importorskip('gf_native')
    if not hasattr(native, 'CardInstanceNativeSession'):
        pytest.skip('isolated additive candidate required; installed extension is not replaced')
    return native


def test_public_corrected_reader_rejects_class_without_real_capability():
    native = SimpleNamespace(build_info=lambda: [('abi', 'cp314-gil')],
        CardInstanceNativeSession=type('DiagnosticClassOnly', (), {}))
    with pytest.raises(ContractError, match='capability'):
        card_instance_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), native)


def test_current_reader_refuses_otherwise_valid_pre_defense_native(monkeypatch):
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import signature
    from gf_training import native_backend

    info = {'card_instance_contract': native_backend.CARD_INSTANCE_CONTRACT,
        'card_instance_signature': signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        'card_instance_models': 'mlp,gru,transformer', 'native_source_sha256': 'a' * 64,
        'card_instance_guardian_acted': native_backend.CARD_INSTANCE_GUARDIAN_ACTED}
    methods = ('control_summary', 'original_config_inputs', 'authorize', 'export_cohort',
        'retire_payload', 'bind_numeric', 'begin_update', 'prepare_commit', 'finalize_read',
        'recheck', 'close', 'used_bytes', 'peak_bytes', 'cancel_handle', 'control_reserved_bytes')
    session = type('OldSession', (), dict.fromkeys(methods, lambda: None))
    module = SimpleNamespace(build_info=lambda: info, CardInstanceNativeSession=session,
        CancelHandle=type('Cancel', (), {'cancel': lambda: None}))
    monkeypatch.setattr(native_backend, 'binary_identity', lambda _: {'build': info, 'gil_disabled': False})
    with pytest.raises(ContractError, match='defense.*capability'):
        card_instance_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), module)


def test_payload_context_cannot_relabel_recorded_export_as_live():
    metadata = {'schema_version': 2, 'kind': 'card-instance-recorded-cohort-payload', 'collection_index': 0,
        'metadata': {}, 'arrays': [], 'source_proof': {}, 'correspondence': {}}
    payload = canonical(metadata).encode(), (), (), ()
    with pytest.raises(ContractError):
        _payload_views(payload, game=True)
    parsed, arrays = _payload_views(payload, game=True, recorded=True)
    assert parsed == metadata and arrays == {}
    with pytest.raises(ContractError):
        _payload_views(payload)


def test_activated_capabilities_and_cancel_keep_old_markers(candidate):
    from dataclasses import asdict

    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import signature
    from gf_training.native_backend import CARD_INSTANCE_CONTRACT, STRUCTURED_CONTRACT
    info = dict(candidate.build_info())
    assert info['card_instance_contract'] == CARD_INSTANCE_CONTRACT
    assert info['card_instance_signature'] == signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    assert info['card_instance_models'] == 'mlp,gru,transformer'
    assert info['structured_preprocessing_contract'] == STRUCTURED_CONTRACT
    assert info['structured_default_signature'] == signature(128)
    assert info['typed_reader_contract'] == 'typed-borrowed-v1'
    assert info['ppo_policy_models'] == 'mlp,gru,transformer'
    settings = PreparationConfig(backend='rust', reader='participant-structured-v2')
    assert card_instance_reader_settings(settings, candidate) == asdict(settings)
    cancel = candidate.CancelHandle(time.monotonic() + 30)
    cancel.cancel()
    settings = canonical({'memory_bytes': 128 * 1024**2, 'scratch_bytes': 1024**2,
        'max_source_file_bytes': 1024**2, 'max_record_bytes': 1024**2, 'chunk_target_bytes': 1024**2}).encode()
    with pytest.raises((InterruptedError, RuntimeError), match='cancel'):
        candidate.CardInstanceNativeSession(b'{}', settings, time.monotonic() + 30, cancel)


@pytest.mark.parametrize('mutation', ['contract', 'signature', 'models', 'missing-source', 'invalid-source', 'other-source'])
def test_activated_reader_still_refuses_mixed_capability_or_loaded_source(candidate, monkeypatch, mutation):
    from gf_training import native_backend
    identity = native_backend.binary_identity(candidate)
    information = dict(candidate.build_info())
    if mutation == 'contract':
        information['card_instance_contract'] = native_backend.STRUCTURED_CONTRACT
    elif mutation == 'signature':
        information['card_instance_signature'] = '0' * 64
    elif mutation == 'models':
        information['card_instance_models'] = 'mlp'
    elif mutation == 'missing-source':
        del information['native_source_sha256']
    elif mutation == 'invalid-source':
        information['native_source_sha256'] = 'not-sha256'
    else:
        information['native_source_sha256'] = '0' * 64
    mixed = SimpleNamespace(build_info=lambda: information,
        CardInstanceNativeSession=candidate.CardInstanceNativeSession, CancelHandle=candidate.CancelHandle)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda native: identity)
    with pytest.raises(ContractError):
        card_instance_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), mixed)


def original_configuration():
    if os.environ.get('GF_RUN_CARD_INSTANCE_ORIGINALS') != '1':
        pytest.skip('opt-in genuine original Source read-only diagnostic')
    if not ORIGINAL.is_dir():
        pytest.fail('approved genuine originals absent; opt-in check cannot count as PASS')
    state = json.loads((ORIGINAL / 'run.json').read_text())
    require_current_capture_binding(state, subject='MLP parent originals')
    settings = replace(preparation_config(state['effective_config']['preprocessing']), backend='rust')
    actions = action_config_v3(state['effective_config']['actions'])
    plan = {'schema_version': 3, 'mode': 'historical_parent',
        'parents': [{'directory': str(ORIGINAL), 'require_complete': True}], 'collections': []}
    return plan, settings, actions


@pytest.mark.integration
def test_genuine_recorded_cohorts_ffi_arrays_gae_numeric4_match_originals(candidate, tmp_path):
    from gf_models.sharded_arrays import read_bundle
    plan, settings, actions = original_configuration()
    with _NativeGameInputs(plan, settings, actions=actions, max_owned_bytes=8 * 1024**3,
            deadline=time.monotonic() + 360, native=candidate) as provider:
        assert provider.summary['parent']['npz_validated'] is False
        assert provider.summary['qualified'] is False
        assert provider.summary['parent']['state']['status'] == 'complete'
        assert provider.plan['mode'] == 'historical_parent'
        original_identity = json.loads((ORIGINAL / 'run.json').read_text())['identity']['preprocessing']
        assert original_identity['backend'] == 'python'  # Never rewrite recorded backend identity.
        expected_inputs = json.loads((ORIGINAL / 'run.json').read_text())['config_inputs']
        assert provider.original_config_inputs(0) == expected_inputs
        for index in range(2):
            destination = tmp_path / f'comparison-{index}'
            manifest = provider.compare_recorded(index, destination)
            original = ORIGINAL / 'numeric' / f'batch-{index:06}'
            expected = json.loads((original / 'rollout.json').read_text())
            assert canonical(manifest) == canonical(expected)
            arrays = read_bundle(destination, manifest['data'])
            reference = read_bundle(original, expected['data'])
            assert set(arrays) == set(reference)
            for key in arrays:
                assert arrays[key].dtype == reference[key].dtype
                assert arrays[key].shape == reference[key].shape
                assert arrays[key].tobytes() == reference[key].tobytes(), key
            del arrays, reference
            provider.recheck()
        with pytest.raises(ContractError, match='outside'):
            provider.compare_recorded(0, ORIGINAL / 'must-not-be-created')
        assert not (ORIGINAL / 'must-not-be-created').exists()
        with pytest.raises(ContractError, match='mode'):
            provider.publish_numeric(tmp_path / 'not-live')
        assert not (tmp_path / 'not-live').exists()


class _DiagnosticStopped(RuntimeError):
    pass


@pytest.mark.integration
def test_recorded_export_view_lifetime_and_failed_ffi_accounting_are_sticky(candidate):
    plan, settings, actions = original_configuration()
    with pytest.raises(_DiagnosticStopped), _NativeGameInputs(plan, settings, actions=actions, max_owned_bytes=8 * 1024**3,
            deadline=time.monotonic() + 300, native=candidate) as provider:
        session = provider._session
        payload = provider._executor.call(session.export_recorded_cohort, 0, provider._held())
        metadata, arrays = _payload_views(payload, game=True, recorded=True)
        assert all(not value.flags.writeable and value.flags.c_contiguous for value in arrays.values())
        held = next(value for value in arrays.values() if value.size)
        first_bytes = held.tobytes()
        with pytest.raises(ValueError, match='views'):
            provider._executor.call(session.retire_payload, provider._held())
        assert held.tobytes() == first_bytes
        del held, metadata, arrays, payload
        provider._executor.call(session.retire_payload, provider._held())
        provider.recheck()
        # Mode/phase refusals remain ordinary; they cannot authorize live work.
        with pytest.raises(ValueError):
            provider._executor.call(session.begin_update, provider._held())
        with pytest.raises(ValueError):
            provider._executor.call(session.prepare_commit, provider._held())
        provider.recheck()
        # An admitted allocation failure is terminal even after caller repair.
        with pytest.raises(MemoryError):
            provider._executor.call(session.export_recorded_cohort, 1, 12 * 1024**3)
        with pytest.raises(RuntimeError, match='closed'):
            provider._executor.call(session.export_recorded_cohort, 1, 0)
        assert provider.summary['qualified'] is False  # Capability never qualifies historical input.
        # The intentionally failed owner must leave via cleanup, not finalize.
        raise _DiagnosticStopped('expected terminal accounting failure; cleanup only')

@pytest.mark.parametrize('temporal', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('hand', ['direct', 'deepsets', 'attention'])
def test_corrected_prepared_writer_keeps_all_model_hand_numeric4_bytes(tmp_path, temporal, hand):
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.sharded_arrays import read_bundle
    from gf_training.structured_rollout import (
        _owned_bytes,
        _write_prepared_game_payload,
        _write_rollout,
    )
    from tests.test_card_instance_learning import ACTIONS, learner_for, rollout_for
    learner = learner_for(temporal, hand, 'cpu')
    rollout = rollout_for(learner)
    manifest = _write_rollout(tmp_path / 'reference', rollout)
    arrays = read_bundle(tmp_path / 'reference', manifest['data'])
    metadata = {key: value for key, value in manifest.items() if key not in ('schema_version', 'kind', 'data')}
    prepared = _write_prepared_game_payload(tmp_path / 'prepared', metadata, arrays, learner.model,
        actions=ACTIONS, rules=card_instance_rules(), resident_bytes=_owned_bytes((learner.model, metadata)),
        extra_owned_bytes=0, max_owned_bytes=learner.model.resources.max_owned_bytes,
        shard_bytes=64 * 1024**2, check=lambda: None, reserve=lambda _: None)
    assert canonical(prepared) == canonical(manifest)
    for name in ('rollout.json', *manifest['data']['files']):
        assert (tmp_path / 'prepared' / name).read_bytes() == (tmp_path / 'reference' / name).read_bytes()


def test_prepared_native_provider_exit_is_memory_only(tmp_path):
    from gf_training.card_instance_source import _GameBatchSource
    from gf_training.structured_lineage import _ChainScan
    from tests.test_card_instance_learning import ACTIONS
    settings = PreparationConfig(backend='rust', reader='participant-structured-v2')
    plan = {'schema_version': 3, 'mode': 'historical_parent',
        'parents': [{'directory': str(tmp_path), 'require_complete': False}], 'collections': []}
    provider = _NativeGameInputs(plan, settings, actions=ACTIONS, max_owned_bytes=1024**3,
        deadline=time.monotonic() + 30)
    class NoIO:
        def close(self):
            pytest.fail('prepared release attempted reportable I/O')
    provider._session = provider._executor = provider.sealed = NoIO()
    provider._prepared = True
    provider.__exit__(None, None, None)
    assert provider._closed and provider._session is None and provider._executor is None
    controls = _ChainScan(directory=tmp_path, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    with pytest.raises(ContractError, match='mutually exclusive'):
        _GameBatchSource(None, controls, {}, None, None, ACTIONS, {}, '', None, (), native_inputs=provider)
    # A native representation never fabricates an empty Python EpisodeRecords tuple.
    source = _GameBatchSource(None, controls, {}, None, None, ACTIONS, {}, '', None, None, native_inputs=provider)
    assert source.snapshot is None and source.records is None
    with pytest.raises(ContractError, match='one live online Source'):
        source.issue(tmp_path, {})


def test_recorded_native_backend_identity_is_closed_and_source_bound():
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.structured_binding import input_binding_game_v3
    from gf_training.native_backend import (
        CARD_INSTANCE_CONTRACT,
        recorded_card_instance_preprocessing,
    )
    from tests.test_card_instance_learning import ACTIONS
    binding = input_binding_game_v3(ACTIONS.action_contract, card_instance_rules())
    value = {'backend': 'rust', 'reader_contract': 'participant-structured-v2',
        'implementation': 'rust-player-card-refs-v2', **binding,
        'binary': {'backend': 'native', 'extension_path': '/diagnostic/gf_native.so',
            'extension_sha256': 'a' * 64, 'python_abi': 'cpython-314', 'gil_disabled': False,
            'build': {'abi': 'cp314-gil', 'card_instance_contract': CARD_INSTANCE_CONTRACT,
                'card_instance_signature': binding['layout_signature'],
                'card_instance_models': 'mlp,gru,transformer', 'native_source_sha256': 'b' * 64}}}
    recorded_card_instance_preprocessing(value, binding=binding)  # Recorded shape only, never live permission.
    import copy
    for mutation in ('backend', 'source', 'signature', 'abi', 'extra'):
        bad = copy.deepcopy(value)
        if mutation == 'backend':
            bad['backend'] = 'python'
        elif mutation == 'source':
            bad['binary']['build']['native_source_sha256'] = 'not-a-source-hash'
        elif mutation == 'signature':
            bad['binary']['build']['card_instance_signature'] = '0' * 64
        elif mutation == 'abi':
            bad['binary']['gil_disabled'] = True
        else:
            bad['authority'] = True
        with pytest.raises(ContractError):
            recorded_card_instance_preprocessing(bad, binding=binding)
