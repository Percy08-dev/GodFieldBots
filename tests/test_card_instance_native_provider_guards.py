"""Unadmitted provider guard fixtures: no executed/live Source is fabricated."""
from __future__ import annotations

import copy
import hashlib
import time
from types import SimpleNamespace

import pytest

from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_common.preprocessing import PreparationConfig
from gf_models.structured_binding import input_binding_game_v3
from gf_training import native_backend
from gf_training.native_card_instance import _NativeGameInputs
from gf_training.native_structured_session import NativeStructuredSource


def owner(tmp_path, *, mode='historical_parent', native=None):
    return _NativeGameInputs({'schema_version': 3, 'mode': mode, 'parents': [
        {'directory': str(tmp_path), 'require_complete': False}], 'collections': []},
        PreparationConfig(backend='rust', reader='participant-structured-v2'),
        actions=ActionConfigV3(action_contract=corrected_card_instance_contract()),
        max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30, native=native)


def test_direct_public_provider_runs_capability_gate_before_source_allocation(tmp_path, monkeypatch):
    module = SimpleNamespace(build_info=list, CardInstanceNativeSession=lambda *a: pytest.fail('allocated'))
    monkeypatch.setattr(native_backend, 'load_native', lambda: module)
    with pytest.raises(ContractError, match='capability'):
        owner(tmp_path).__enter__()


def test_explicit_candidate_cannot_bypass_live_entry(tmp_path):
    with pytest.raises(ContractError, match='historical diagnostic only'):
        owner(tmp_path, mode='live_cohort', native=SimpleNamespace()).__enter__()


def identity_for(path):
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import signature
    return {'backend': 'native', 'extension_path': str(path),
        'extension_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'python_abi': 'cpython-314',
        'gil_disabled': False, 'build': {'abi': 'cp314-gil',
            'card_instance_contract': native_backend.CARD_INSTANCE_CONTRACT,
            'card_instance_signature': signature(128, layout=CORRECTED_CARD_INSTANCE_LAYOUT),
            'card_instance_models': 'mlp,gru,transformer', 'native_source_sha256': 'a' * 64}}


@pytest.mark.parametrize('missing', ['export_cohort', 'cancel'])
def test_direct_public_provider_checks_methods_and_cancel_abi(tmp_path, monkeypatch, missing):
    binary = tmp_path / 'unadmitted-evidence'
    binary.write_bytes(b'ABI refusal only')
    current = identity_for(binary)
    required = ('control_summary', 'original_config_inputs', 'authorize', 'export_cohort', 'retire_payload',
        'bind_numeric', 'begin_update', 'prepare_commit', 'finalize_read', 'recheck', 'close', 'used_bytes',
        'peak_bytes', 'cancel_handle', 'control_reserved_bytes')
    session = type('UnadmittedSession', (), {key: lambda *a: None for key in required if key != missing})
    cancel = type('UnadmittedCancel', (), {} if missing == 'cancel' else {'cancel': lambda *a: None})
    module = SimpleNamespace(build_info=lambda: current['build'], CardInstanceNativeSession=session, CancelHandle=cancel)
    monkeypatch.setattr(native_backend, 'load_native', lambda: module)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *a: pytest.fail('reached binary allocation stage'))
    with pytest.raises(ContractError, match='cancellation ABI'):
        owner(tmp_path).__enter__()


@pytest.mark.parametrize('field', ['extension_path', 'extension_sha256', 'python_abi', 'gil_disabled',
    'source', 'build', 'backend'])
def test_live_exact_recorded_binary_mismatch_precedes_allocation(tmp_path, monkeypatch, field):
    # Deliberately incomplete Run metadata can exercise refusal, never admission.
    binary = tmp_path / 'owned-binary-evidence'
    binary.write_bytes(b'unadmitted metadata fixture')
    current = identity_for(binary)
    recorded = copy.deepcopy(current)
    if field == 'source':
        recorded['build']['native_source_sha256'] = 'b' * 64
    elif field == 'build':
        recorded['build']['extra_evidence'] = 'different'
    else:
        recorded[field] = {'extension_path': str(tmp_path / 'foreign'), 'extension_sha256': 'b' * 64,
            'python_abi': 'foreign', 'gil_disabled': True, 'backend': 'foreign'}[field]
    provider = owner(tmp_path, mode='live_cohort')
    binding = input_binding_game_v3(provider.actions.action_contract, card_instance_rules())
    preprocessing = {'backend': 'rust', 'reader_contract': 'participant-structured-v2',
        'implementation': 'rust-player-card-refs-v2', 'binary': recorded, **binding}
    (tmp_path / 'run.json').write_text(canonical({'identity': {'preprocessing': preprocessing}}))
    monkeypatch.setattr(native_backend, 'load_native', lambda: SimpleNamespace())
    # Pure preflight already has separate method/cancel tests; bypass only that
    # check here to isolate exact-parent binary refusal, never reach native core.
    monkeypatch.setattr(native_backend, 'card_instance_reader_settings', lambda *a: None)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *a: current)
    monkeypatch.setattr(NativeStructuredSource, '__enter__', lambda self: pytest.fail('allocated'))
    with pytest.raises(ContractError):
        provider.__enter__()


@pytest.mark.parametrize('change', ['modify', 'replace'])
def test_retained_binary_evidence_rejects_later_changes(tmp_path, monkeypatch, change):
    binary = tmp_path / 'owned-binary-evidence'
    binary.write_bytes(b'unadmitted binary evidence')
    current = identity_for(binary)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *a: current)
    monkeypatch.setattr(NativeStructuredSource, '__enter__', lambda self: self)
    provider = owner(tmp_path, native=SimpleNamespace()).__enter__()
    assert provider.extra >= provider._binary_scan.peak
    if change == 'modify':
        binary.write_bytes(b'changed')
    else:
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(binary.read_bytes())
        replacement.replace(binary)
    # Recheck, preallocation, and precommit share this retained guard.
    provider.identity = current
    with pytest.raises(ContractError, match='sealed'):
        provider._before_session_construct()
    with pytest.raises(ContractError, match='sealed'):
        provider.recheck()


def test_last_precommit_binary_guard_cannot_produce_prepared_after_replacement(tmp_path, monkeypatch):
    # Deliberately unadmitted primitives reach ONLY the refusing last guard.
    binary = tmp_path / 'owned-binary-evidence'
    binary.write_bytes(b'precommit evidence')
    current = identity_for(binary)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *a: current)
    monkeypatch.setattr(NativeStructuredSource, '__enter__', lambda self: self)
    provider = owner(tmp_path, native=SimpleNamespace()).__enter__()
    provider.plan['mode'] = 'live_cohort'
    provider._begun = True
    provider._numeric_files = {'rollout.json': 'a' * 64}
    provider.summary = {'parent': {'state': {'optimizer_steps': 0}}, 'collections': [{
        'learner_decisions': 1, 'source_proof': {'manifest_sha256': 'b' * 64}, 'correspondence': {}}]}
    receipt = {'schema_version': 1, 'kind': 'card-instance-source-prepared',
        'summary_sha256': digest(provider.summary), 'collection_manifest_sha256': 'b' * 64,
        'source_correspondence': {}, 'numeric_manifest_sha256': 'a' * 64,
        'numeric_files_sha256': digest(provider._numeric_files), 'optimizer_steps': 4}
    def late_change(*args):
        binary.write_bytes(b'changed during native completion')
        return canonical(receipt)
    provider._session = SimpleNamespace(prepare_commit=lambda *a: None)
    provider._executor = SimpleNamespace(call=late_change, close=lambda: pytest.fail('prepared shutdown reached'))
    monkeypatch.setattr(provider, 'recheck', provider._binary_scan.finalize)
    with pytest.raises(ContractError, match='sealed'):
        provider.prepare_commit()
    assert not provider._prepared and not provider._session_closed


@pytest.mark.parametrize('field', ['schema_version', 'kind', 'summary_sha256', 'parent_index', 'run_sha256', 'config_inputs'])
def test_corrected_config_query_is_bound_to_singular_summary(tmp_path, field):
    provider = owner(tmp_path)
    provider.summary = {'parent': {'run_sha256': 'a' * 64}}
    response = {'schema_version': 3, 'kind': 'card-instance-original-config-inputs',
        'summary_sha256': digest(provider.summary), 'parent_index': 0, 'run_sha256': 'a' * 64,
        'config_inputs': [{'path': 'unadmitted-origin', 'sha256': 'b' * 64}]}
    provider._session = SimpleNamespace(original_config_inputs=lambda *a: canonical(response))
    calls = []
    provider._executor = SimpleNamespace(call=lambda fn, *args: (calls.append(args), fn(*args))[1])
    assert provider.original_config_inputs(0) == response['config_inputs']
    assert calls == [(0, 0)]
    with pytest.raises(ContractError):
        provider.original_config_inputs(1)
    assert len(calls) == 1
    response[field] = {'schema_version': 2, 'kind': 'structured-original-config-inputs',
        'summary_sha256': 'b' * 64, 'parent_index': 1, 'run_sha256': 'b' * 64, 'config_inputs': {}}[field]
    with pytest.raises(ContractError, match='input-origin'):
        provider.original_config_inputs(0)


def test_old_component_config_query_and_preallocation_hook_remain_schema2(tmp_path):
    provider = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': []}, PreparationConfig(backend='rust', reader='structured-v1'),
        max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    # The unchanged old context does not require corrected binary evidence.
    provider._before_session_construct()
    provider.summary = {'parents': [{'run_sha256': 'a' * 64}], 'control_fingerprint': 'b' * 64}
    response = {'schema_version': 2, 'kind': 'structured-original-config-inputs',
        'control_fingerprint': 'b' * 64, 'parent_index': 0, 'run_sha256': 'a' * 64,
        'config_inputs': [{'path': 'old-origin', 'sha256': 'c' * 64}]}
    provider._session = SimpleNamespace(original_config_inputs=lambda *a: canonical(response))
    provider._executor = SimpleNamespace(call=lambda fn, *args: fn(*args))
    assert provider.original_config_inputs(0) == response['config_inputs']


@pytest.mark.parametrize('destination', ['foreign', 'original', 'existing', 'file', 'symlink', 'dangling'])
def test_live_publication_refuses_before_export_or_writer(tmp_path, monkeypatch, destination):
    # Unadmitted owner exercises only pure early refusals, never a live chain.
    provider = owner(tmp_path, mode='live_cohort')
    provider.summary = {'parent': {'directory': str(tmp_path)}, 'collections': [{'generation': 0}]}
    derived = tmp_path / 'numeric/batch-000000'
    derived.parent.mkdir()
    target = derived
    if destination == 'foreign':
        target = tmp_path / 'foreign'
    elif destination == 'original':
        target = tmp_path
    elif destination == 'existing':
        derived.mkdir()
    elif destination == 'file':
        derived.write_bytes(b'foreign existing file')
    else:
        original = tmp_path / 'other'
        if destination == 'symlink':
            original.mkdir()
        derived.symlink_to(original, target_is_directory=True)
    provider._session = SimpleNamespace(export_cohort=lambda *a: pytest.fail('exported'))
    provider._executor = SimpleNamespace(call=lambda *a: pytest.fail('allocated/exported'))
    from gf_training import structured_rollout
    monkeypatch.setattr(structured_rollout, '_write_prepared_game_payload', lambda *a, **k: pytest.fail('wrote'))
    before = sorted(str(path) for path in tmp_path.rglob('*'))
    with pytest.raises(ContractError, match='numeric publication'):
        provider.publish_numeric(target)
    assert sorted(str(path) for path in tmp_path.rglob('*')) == before
