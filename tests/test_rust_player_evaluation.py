"""Rust evaluation DATA boundaries; synthetic records never issue live authority."""
from copy import deepcopy

import pytest

from gf_common.domain import ContractError


def metadata_data():
    return {'schema_version': 3, 'kind': 'hpo-rust-parent-metadata-validation',
        'origin': 'producer-owned-v1', 'encoder': 'canonical-python', 'run_id': 'run',
        'run_path': '/parents/run', 'training_seed': 17, 'generation': 2, 'used_decisions': 64,
        **{key: 'a' * 64 for key in ('config_sha256', 'state_sha256', 'generation_sha256',
            'generations_sha256', 'inference_sha256', 'effective_settings_sha256',
            'runtime_binding_sha256', 'completion_receipt_sha256', 'prepared_sha256',
            'exit_sha256', 'source_validation_sha256')}}


def seed_data():
    metadata = metadata_data()
    return {'schema_version': 2, 'kind': 'hpo-rust-seed-training',
        **{key: metadata[key] for key in ('training_seed', 'run_path', 'run_id', 'config_sha256',
            'state_sha256', 'generation', 'generation_sha256', 'inference_sha256', 'used_decisions',
            'runtime_binding_sha256', 'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256')},
        'proposal_sha256': 'b' * 64, 'source_receipt_sha256': 'c' * 64, 'gpu_termination_sha256': 'd' * 64}


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('schema_version', True),
    ('kind', 'hpo-owned-parent-metadata-validation'), ('training_seed', True), ('generation', False),
    ('used_decisions', -1), ('run_path', 'relative/run'), ('runtime_binding_sha256', 'A' * 64),
    ('origin', 'legacy'), ('extra', None)])
def test_parent_metadata3_closed_selector_and_types(field, value):
    from gf_training.rust_player_evaluation import validate_parent_metadata
    row = metadata_data()
    assert validate_parent_metadata(row) is row
    bad = deepcopy(row)
    bad[field] = value
    with pytest.raises(ContractError):
        validate_parent_metadata(bad)


@pytest.mark.parametrize('field,value', [('schema_version', 1), ('schema_version', True),
    ('kind', 'hpo-seed-training'), ('training_seed', True), ('used_decisions', True),
    ('completion_receipt_sha256', 'bad'), ('run_path', '../foreign'), ('extra', None)])
def test_seed_training2_closed_selector_and_types(field, value):
    from gf_training.rust_player_evaluation import validate_seed_training
    row = seed_data()
    assert validate_seed_training(row) is row
    bad = deepcopy(row)
    bad[field] = value
    with pytest.raises(ContractError):
        validate_seed_training(bad)



@pytest.mark.parametrize('field', ['run_id', 'run_path', 'training_seed', 'config_sha256', 'state_sha256',
    'generation', 'generation_sha256', 'inference_sha256', 'used_decisions', 'runtime_binding_sha256',
    'completion_receipt_sha256', 'prepared_sha256', 'exit_sha256', 'source_receipt_sha256', 'gpu_termination_sha256'])
def test_closed_seed_binds_exact_parent_source_and_physical_data(field):
    from gf_training.rust_player_evaluation import validate_seed_binding
    row, metadata = seed_data(), metadata_data()
    validate_seed_binding(row, metadata, source_sha256='c' * 64, termination_sha256='d' * 64)
    changed = deepcopy(row)
    if field.endswith('_sha256'): changed[field] = '0' * 64
    elif field == 'run_path': changed[field] = '/foreign/run'
    elif field == 'run_id': changed[field] = 'foreign'
    else: changed[field] += 1
    with pytest.raises(ContractError):
        validate_seed_binding(changed, metadata, source_sha256='c' * 64, termination_sha256='d' * 64)


@pytest.mark.parametrize('status,used,variant', [('paused', 64, 'hpo-ppo-owned-rust-player-v1'),
    ('incomplete', 0, 'hpo-ppo-owned-rust-player-v1'), ('complete', 63, 'hpo-ppo-owned-rust-player-v1'),
    ('complete', 65, 'hpo-ppo-owned-rust-player-v1'), ('complete', 64, 'hpo-ppo-owned-v1')])
def test_parent_metadata_requires_complete_new_mode_and_exact_D_data(status, used, variant):
    from gf_training.rust_player_evaluation import validate_parent_binding
    metadata = metadata_data()
    state = {'run_variant': variant, 'status': status, 'run_id': metadata['run_id'], 'seed': 17,
        'config_sha256': metadata['config_sha256'], 'generation_sha256': metadata['generation_sha256'],
        'runtime_binding_sha256': metadata['runtime_binding_sha256'], 'updates': 2, 'used_decisions': used,
        'effective_config': {'hpo_spec': {'schema_version': 6, 'decision_budget': 64,
            'effective_settings_sha256': metadata['effective_settings_sha256']}}}
    with pytest.raises(ContractError):
        validate_parent_binding(metadata, state, run_path='/parents/run')


@pytest.mark.parametrize('field,value', [('token', '../foreign'), ('token', '.hidden'), ('token', ''),
    ('run_id', 'foreign'), ('generation', True), ('generation', 3), ('inference_sha256', '0' * 64),
    ('generation_sha256', '0' * 64), ('extra', None)])
def test_reference1_binding_uses_exact_five_artifact_fields(field, value):
    from gf_training.rust_player_evaluation import validate_reference_binding
    state, generation = {'run_id': 'run'}, {'generation': 2, 'inference_sha256': 'a' * 64}
    row = {'run_id': 'run', 'token': 'evaluation-1', 'generation': 2,
        'inference_sha256': 'a' * 64, 'generation_sha256': 'b' * 64}
    assert validate_reference_binding(row, state, generation, generation_sha256='b' * 64) is row
    row[field] = value
    with pytest.raises(ContractError):
        validate_reference_binding(row, state, generation, generation_sha256='b' * 64)


def evaluation_account():
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack import StorageSettings
    from gf_training.group_pack_resources import StorageBudget
    budget = StorageBudget(StorageSettings(), time.monotonic() + 30)
    return budget, ArtifactPhaseAdmission(budget, 8 * 1024**2)


def inference_files(tmp_path, *, large=False):
    import hashlib
    files = {'inference.json': b'{"fixture":"DATA only"}', 'weights-000.npz': b'x' * (2 * 1024**2 if large else 100)}
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}


def test_independent_readonly_originals_survive_other_owner_close_without_weight_heap_copy(tmp_path):
    import os

    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_evaluation import _RetainedInferenceFiles
    expected = inference_files(tmp_path, large=True)
    budget, phase = evaluation_account()
    old = [OriginalFile(tmp_path / name, budget.deadline) for name in expected]
    owner = _RetainedInferenceFiles(tmp_path, expected, budget=budget, parent_budget=phase)
    try:
        fds = [file.fd for file in owner._held.values()]
        for original in old: original.close()
        owner.require()
        for fd in fds: os.fstat(fd)
        assert budget.peak < (tmp_path / 'weights-000.npz').stat().st_size
        assert owner.files == expected
        owner.close()
        for fd in fds:
            with pytest.raises(OSError): os.fstat(fd)
    finally:
        owner.close()
        for original in old: original.close()
    assert budget.used == 0


@pytest.mark.parametrize('fault', ['replacement', 'wrong-sha', 'nested-constructor'])
def test_readonly_original_constructor_failure_retains_nested_owner_and_funding(tmp_path, monkeypatch, fault):
    from gf_training import rust_player_evaluation as module
    expected = inference_files(tmp_path)
    budget, phase = evaluation_account()
    actual = module.OriginalFile
    if fault == 'wrong-sha': expected['weights-000.npz'] = '0' * 64
    elif fault == 'replacement':
        def replaced(path, deadline):
            original = actual(path, deadline)
            if path.name == 'weights-000.npz':
                path.rename(path.with_suffix('.old'))
                path.write_bytes(b'foreign')
            return original
        monkeypatch.setattr(module, 'OriginalFile', replaced)
    else:
        def failed(path, deadline):
            original = actual(path, deadline)
            if path.name == 'weights-000.npz':
                error = OSError('after actual nested original constructor')
                error.cleanup_owner = original
                raise error
            return original
        monkeypatch.setattr(module, 'OriginalFile', failed)
    with pytest.raises((ContractError, OSError, ValueError)) as failed:
        module._RetainedInferenceFiles(tmp_path, expected, budget=budget, parent_budget=phase)
    owner = failed.value.cleanup_owner
    assert budget.used > 0
    if fault == 'nested-constructor': assert owner._cleanup_children
    owner.close()
    assert budget.used == 0


def test_readonly_original_positive_close_retry_keeps_actual_charge(tmp_path):
    import os

    from gf_training.rust_player_evaluation import _RetainedInferenceFiles
    budget, phase = evaluation_account()
    owner = _RetainedInferenceFiles(tmp_path, inference_files(tmp_path), budget=budget, parent_budget=phase)
    original = owner._held['weights-000.npz']
    fd, actual = original.fd, original.close
    def fail(): raise OSError('before original terminal close')
    original.close = fail
    before = budget.used
    with pytest.raises(OSError): owner.close()
    os.fstat(fd)
    assert budget.used == before
    original.close = actual
    owner.close()
    assert budget.used == 0


def test_private_scan_injection_preserves_exact_legacy_provider_type_and_checks(tmp_path):
    import time

    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_sealed import SealedScan
    from tests.test_hpo_artifacts import ACTIONS, RULES, configured_for, retained, saved
    learner, directory, _ = saved(tmp_path, trained=False)
    configured = configured_for(learner, directory)
    baseline = retained(configured)
    label, binding = baseline.binding(configured)
    deadline, limit = time.monotonic() + 30, 256 * 1024**2
    scan = SealedScan(max_owned_bytes=limit, deadline=deadline)
    try:
        owner = RetainedGameArtifacts.for_hpo_registry({label: binding}, ACTIONS, pool_size=1,
            rules=RULES, max_owned_bytes=limit, preflight_deadline=deadline,
            finalization_deadline=deadline, _scan=scan)
        try:
            assert type(owner) is RetainedGameArtifacts and owner._scan is scan
            assert owner.journal_metadata(label) == baseline.journal_metadata(label)
            owner.finalize(deadline=deadline)
        finally:
            owner.release()
    finally:
        baseline.release()


@pytest.mark.parametrize('scan', [{}, object()])
def test_private_provider_scan_refuses_shape_equivalent_metadata_before_file_io(scan, tmp_path, monkeypatch):
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    def fail(*args, **kwargs): raise AssertionError('no file I/O for forged scan')
    monkeypatch.setattr('os.open', fail)
    with pytest.raises(ContractError, match='SealedScan'):
        RetainedGameArtifacts.for_hpo_registry({}, None, pool_size=1, rules=None, max_owned_bytes=1024,
            preflight_deadline=1., finalization_deadline=1., _scan=scan)


def test_completed_audit_originals_refuses_metadata_before_any_file_or_pool_open(monkeypatch, tmp_path):
    from gf_training.rust_player_source import RustCompletedOriginals
    def fail(*args, **kwargs): raise AssertionError('no file work for forged audit')
    monkeypatch.setattr('os.open', fail)
    with pytest.raises(ContractError, match='actual'):
        RustCompletedOriginals(metadata_data(), tmp_path / 'collection.pending.json', 'a' * 64,
            settings=None, deadline=1.)


def test_completed_audit_owner_is_private_and_not_recreated_from_json():
    from gf_training.rust_player_evaluation import RustCompletedParentAudit
    with pytest.raises(ContractError, match='actual'):
        RustCompletedParentAudit(metadata_data(), None, None, loaded=None, pool=None,
            parent_budget=None, deadline=1., record_root=None)


@pytest.mark.parametrize('cleanup_fault', [None, 'nested', 'refund'])
def test_shared_numeric_audit_failure_keeps_actual_array_loan_until_stopped_frame_close(tmp_path, monkeypatch, cleanup_fault):
    """Synthetic audit DATA fault, actual NumPy/pool/loan; no Source issuance."""
    import time
    import weakref
    from types import SimpleNamespace

    import numpy as np

    from gf_training import hpo_rollout, structured_rollout
    from gf_training import rust_player_completion as module
    from gf_training.group_pack_pool import PackedArtifactPool
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    refs = []
    nested = []
    class ArrayOwner:
        blocked = True
        def __init__(self, array): self.array = array
        def close(self):
            if self.blocked: raise OSError('nested array owner not yet retired')
            self.array = None
    monkeypatch.setattr(module, '_audit_raw_originals', lambda *a, **k: (1, 1, ('a' * 64,), '2' * 32))
    monkeypatch.setattr(module, '_audit_source_device_plan', lambda *a, **k: None)
    monkeypatch.setattr(structured_rollout, '_hpo_semantics', lambda *a, **k: {})
    monkeypatch.setattr(structured_rollout, '_source_correspondence', lambda *a, **k: {})
    from tests.test_card_instance_learning import ACTIONS
    def fail_numeric(*args, reserve, **kwargs):
        decoded = np.arange(512, dtype=np.int64)
        reserve(decoded.nbytes)
        refs.append(weakref.ref(decoded))
        error = ContractError('late numeric correspondence error')
        if cleanup_fault == 'nested':
            owner = ArrayOwner(decoded)
            nested.append(owner)
            error.cleanup_owner = owner
        raise error
    monkeypatch.setattr(hpo_rollout, 'read_hpo_rollout', fail_numeric)
    loaded = SimpleNamespace(config=SimpleNamespace(storage=SimpleNamespace(db_root=tmp_path),
        actions=ACTIONS, model=SimpleNamespace(architecture=lambda **k: None),
        hpo_spec={'schema_version': 6, 'training_recipe': {}, 'operating': {'numeric_budget_bytes': 1024**2}}))
    monkeypatch.setattr('gf_training.hpo_config.hpo_training_config', lambda x: None)
    state = {'run_id': 'run'}
    current = {'generation': 1}
    update = {'source_proof': {'selection': {}, 'snapshot_identity': 'snapshot'},
        'numeric': 'numeric/batch-000000', 'numeric_files': {}, 'numeric_sha256': 'a' * 64}
    before = pool.budget.used
    try:
        with pytest.raises(ContractError, match='late numeric') as failed:
            module._audit_update_data(object(), SimpleNamespace(directory=tmp_path), pool, loaded, state,
                {}, current, update, pool.deadline, record_root=tmp_path, control_loans=[])
        assert refs[0]() is not None
        assert pool.budget.used == before + 4096
        owner = failed.value.cleanup_owner
        if cleanup_fault == 'nested':
            with pytest.raises(OSError, match='nested array'): owner.close()
            assert pool.budget.used == before + 4096 and refs[0]() is not None
            nested[0].blocked = False
        elif cleanup_fault == 'refund':
            actual = owner.reservation.close
            def fail_refund(): raise OSError('before numeric reservation close')
            owner.reservation.close = fail_refund
            with pytest.raises(OSError, match='reservation close'): owner.close()
            assert pool.budget.used == before + 4096 and not owner.closed
            owner.reservation.close = actual
        owner.close()
        owner.close()
        assert str(failed.value) == 'late numeric correspondence error'
        assert refs[0]() is None and pool.budget.used == before
    finally:
        pool.close()


def test_phase_shrink_keeps_actual_source_phase_counters_and_historical_peak():
    budget, phase = evaluation_account()
    loan = phase.reserve(4096)
    loan.shrink(1024)
    assert budget.used == phase.used == loan.size == loan.actual.size == 1024
    assert budget.peak == phase.peak == 4096
    loan.shrink(1024)
    loan.shrink(0)
    assert budget.used == phase.used == 0 and phase.peak == 4096
    loan.close()
    loan.close()
    with pytest.raises(ValueError): loan.shrink(0)


@pytest.mark.parametrize('size', [True, -1, 1025, 1.0])
def test_phase_shrink_refuses_invalid_or_growth_before_backend_action(size):
    budget, phase = evaluation_account()
    loan = phase.reserve(1024)
    actual = loan.actual.resize
    calls = []
    def resize(value):
        calls.append(value)
        actual(value)
    loan.actual.resize = resize
    with pytest.raises(ValueError): loan.shrink(size)
    assert not calls and budget.used == phase.used == loan.size == 1024
    loan.close()


def test_phase_shrink_backend_failure_keeps_counters_and_no_replacement_allocation():
    budget, phase = evaluation_account()
    loan = phase.reserve(1024)
    def fail(size): raise OSError('backend shrink refused before release')
    loan.actual.resize = fail
    with pytest.raises(OSError): loan.shrink(512)
    assert budget.used == phase.used == loan.size == loan.actual.size == 1024
    assert budget.peak == phase.peak == 1024
    loan.close()
    assert budget.used == phase.used == 0


def test_phase_shrink_does_not_extend_close_only_backend_protocol():
    from gf_common.artifacts import ArtifactPhaseAdmission
    class CloseOnly:
        closed = False
        def close(self): self.closed = True
    class Source:
        def reserve(self, size): return CloseOnly()
    phase = ArtifactPhaseAdmission(Source(), 1024)
    loan = phase.reserve(512)
    with pytest.raises(TypeError): loan.shrink(256)
    assert phase.used == loan.size == 512 and not loan.actual.closed
    loan.close()
    assert phase.used == 0 and loan.actual.closed


def test_funded_scan_uses_max_concurrent_estimate_then_shrinks_without_funding_gap():
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _FundedEvaluationScan
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 1024**2)
    custodian = object()
    hold = pool.hold_close(custodian)
    before = pool.budget.used
    scan = _FundedEvaluationScan(pool, phase, max_owned_bytes=phase.limit, deadline=pool.deadline)
    try:
        scan.charge(1024, temporary=True)
        scan.charge(128)
        scan.charge(512, temporary=True)
        assert pool.budget.used == before + 1024 and phase.used == 1024
        assert scan.peak == 1024  # not final128 plus historical1024
        scan.settle()
        assert pool.budget.used == before + 128 and phase.used == 128
        scan.charge(4096)
        scan.charge(128, temporary=True)
        assert pool.budget.used == before + 4352
        scan.settle()
        assert phase.used == 4224 and phase.peak == 4352
        with pytest.raises(RuntimeError): pool.close()
        scan.close()
        assert pool.budget.used == before and phase.used == 0
    finally:
        scan.close()
        hold.retire(custodian)
        pool.close()


def test_funded_scan_rejects_other_account_and_preserves_metadata_on_failed_scratch_admission():
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _FundedEvaluationScan
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    other, _ = evaluation_account()
    foreign = ArtifactPhaseAdmission(other, 1024)
    with pytest.raises(ContractError):
        _FundedEvaluationScan(pool, foreign, max_owned_bytes=1024, deadline=pool.deadline)
    phase = ArtifactPhaseAdmission(pool.budget, 1024)
    scan = _FundedEvaluationScan(pool, phase, max_owned_bytes=2048, deadline=pool.deadline)
    before = pool.budget.used
    try:
        scan.charge(128)
        with pytest.raises(ValueError) as failed:
            scan.charge(1024, temporary=True)
        assert failed.value.cleanup_owner is scan
        assert pool.budget.used == before + 128 and phase.used == 128
        scan.close()
        assert pool.budget.used == before
    finally:
        scan.close()
        pool.close()


def reference_parent_data(directory):
    """Synthetic Run5/reg8/artifact DATA only; does not qualify a real Parent."""
    import hashlib
    import json
    from pathlib import Path

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.domain import canonical, digest
    from gf_common.rust_player import runtime_binding
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    from gf_training.rust_player_lineage import record_fields
    from tests.test_rust_player_config import RUN_ID, campaign_raw
    campaign = campaign_config(campaign_raw(), origin=directory)
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    loaded = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID)
    state = json.loads((Path(__file__).parent / 'fixtures/rust_player_run5_common_metadata.json').read_bytes())
    bound = runtime_binding(loaded.config.hpo_spec['runtime'], state['identity']['player_runtime']['producer'])
    state['identity']['player_runtime'] = bound
    state.update(effective_config=loaded.effective, config_sha256=loaded.sha256,
        runtime_binding_sha256=digest(bound), hpo_spec_sha256=digest(loaded.config.hpo_spec),
        run_variant='hpo-ppo-owned-rust-player-v1', model_version='4', updates=0,
        used_decisions=loaded.config.hpo_spec['decision_budget'], collected_decisions=loaded.config.hpo_spec['decision_budget'],
        stop_reason=None, status='complete')
    artifacts = directory / 'artifacts/000000'
    artifacts.mkdir(parents=True)
    files = inference_files(artifacts)
    record = dict.fromkeys(record_fields('generation', hpo=True))
    record.update(schema_version=6, origin='producer-owned-v1', encoder='canonical-python',
        run_id=state['run_id'], generation=0, parent_sha256=None, inference_sha256=files['inference.json'],
        artifact_files=files, runtime_binding_sha256=state['runtime_binding_sha256'])
    raw = (canonical(record) + '\n').encode()
    generation_sha = hashlib.sha256(raw).hexdigest()
    (directory / 'generations').mkdir()
    (directory / 'generations/000000.json').write_bytes(raw)
    state['generation_sha256'] = generation_sha
    raw = (canonical(state) + '\n').encode()
    (directory / 'run.json').write_bytes(raw)
    state_sha = hashlib.sha256(raw).hexdigest()
    registry = {'schema_version': 8, 'run_id': state['run_id'], 'runtime_binding_sha256': state['runtime_binding_sha256'],
        'origin': 'producer-owned-v1', 'encoder': 'canonical-python', 'evaluation_references': {}, 'service_references': {},
        'generations': {'0': {'inference_sha256': files['inference.json'], 'snapshot_identity': record['snapshot_identity'],
            'files': {name: {'sha256': sha, 'status': 'retained', 'deletion_intent': None, 'deletion_receipt': None} for name, sha in files.items()}}}}
    (directory / 'artifact-registry.json').write_text(canonical(registry) + '\n')
    return {'run_id': state['run_id'], 'token': 'evaluation-1', 'generation': 0,
        'inference_sha256': files['inference.json'], 'generation_sha256': generation_sha}, state_sha


def test_actual_owned_reference_transaction_preserves_reference1_and_releases_after_handoff(tmp_path):
    import json
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.lineage import RunOwnership, run_lock
    from gf_training.rust_player_evaluation import _reference_transition
    binding, state_sha = reference_parent_data(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 16 * 1024**2)
    try:
        with run_lock(tmp_path) as owner:
            with pytest.raises(ContractError):
                _reference_transition(RunOwnership(owner.directory, owner.token), binding, 'acquire',
                    state_sha256=state_sha, pool=pool, parent_budget=phase, deadline=pool.deadline)
            acquired = _reference_transition(owner, binding, 'acquire', state_sha256=state_sha,
                pool=pool, parent_budget=phase, deadline=pool.deadline)
            assert acquired.phase == 'acquire' and acquired.binding == binding
        # Later transaction uses its own actual owner AFTER an already protected handoff.
        with run_lock(tmp_path) as owner:
            released = _reference_transition(owner, binding, 'release', state_sha256=state_sha,
                pool=pool, parent_budget=phase, deadline=pool.deadline, acquisition_sha256=acquired.receipt_sha256)
            assert released.phase == 'release'
        registry = json.loads((tmp_path / 'artifact-registry.json').read_bytes())
        assert registry['evaluation_references'] == {}
        assert not phase.used
    finally:
        pool.close()


def test_reference_receipt_failure_keeps_protective_intent_and_refuses_json_replay(tmp_path, monkeypatch):
    import json
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.lineage import run_lock
    from gf_training.rust_player_evaluation import _reference_transition
    binding, state_sha = reference_parent_data(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 16 * 1024**2)
    actual = OwnedControlWriter.write
    def fail(writer, path, value, **kwargs):
        if path.name == 'acquire.receipt.json': raise OSError('reference receipt durability failure')
        return actual(writer, path, value, **kwargs)
    try:
        with run_lock(tmp_path) as owner:
            with monkeypatch.context() as patch:
                patch.setattr(OwnedControlWriter, 'write', fail)
                with pytest.raises(OSError) as failed:
                    _reference_transition(owner, binding, 'acquire', state_sha256=state_sha,
                        pool=pool, parent_budget=phase, deadline=pool.deadline)
            assert pool._close_holds
            failed.value.cleanup_owner.close()
            registry = json.loads((tmp_path / 'artifact-registry.json').read_bytes())
            assert registry['evaluation_references'][binding['token']]['generation'] == 0
            with pytest.raises(ContractError, match='interrupted|replay') as refused:
                _reference_transition(owner, binding, 'acquire', state_sha256=state_sha,
                    pool=pool, parent_budget=phase, deadline=pool.deadline)
            refused.value.cleanup_owner.close()
            assert registry == json.loads((tmp_path / 'artifact-registry.json').read_bytes())
    finally:
        pool.close()


def test_forged_parent_refused_before_provider_allocation_or_file_mutation(tmp_path):
    from gf_training.rust_player_evaluation import RustEvaluationData
    class NoAllocation:
        def reserve(self, *args, **kwargs):
            raise AssertionError('untrusted metadata cannot allocate evaluation data')
    with pytest.raises(ContractError, match='actual'):
        RustEvaluationData.from_parent(metadata_data(), None, 'label', evaluation_job=None,
            pool=NoAllocation(), parent_budget=NoAllocation(), pin='pin')
    assert list(tmp_path.iterdir()) == []


def test_closed_seed_constructor_refuses_metadata_parent_and_physical_dict_before_store_write(tmp_path, monkeypatch):
    from gf_cli.rust_player_evaluation import persist_rust_training_receipt
    def fail(*args, **kwargs): raise AssertionError('no external work for forged Parent/physical receipt')
    monkeypatch.setattr('os.open', fail)
    with pytest.raises(ContractError, match='actual'):
        persist_rust_training_receipt(None, None, {}, None, training_seed=17, run_id='run', directory=tmp_path,
            termination={'schema_version': 3}, deadline=1., completion=metadata_data())


def test_closed_data_import_refuses_json_current_control_before_any_new_owner_allocation(tmp_path, monkeypatch):
    from gf_cli.rust_player_evaluation import open_closed_rust_parent
    def fail(*args, **kwargs): raise AssertionError('forged control cannot open originals')
    monkeypatch.setattr('os.open', fail)
    with pytest.raises(ContractError, match='actual'):
        open_closed_rust_parent(None, None, {}, None, 17, control={'campaign_bootstrap': {}},
            ownership=None, pool=None, parent_budget=None, deadline=1.)


def test_historical_root_comes_only_from_current_campaign_and_strict_old_supervision_id():
    from pathlib import Path

    from gf_cli.rust_player_evaluation import historical_record_root
    bootstrap = {'campaign_directory': '/campaign'}
    physical = {'schema_version': 3, 'kind': 'rust-player-gpu-termination-v1',
        'ticket': '1' * 32, 'exit_record': {'supervision_id': '2' * 32}}
    assert historical_record_root(bootstrap, physical) == Path('/campaign/supervision/' + '2' * 32)
    for bad in ('../foreign', '', '3' * 31, '/absolute'):
        with pytest.raises(ContractError):
            historical_record_root(bootstrap, physical | {'exit_record': {'supervision_id': bad}})


def test_legacy_physical_reader_does_not_treat_new_rust_data_as_old_termination(monkeypatch):
    from gf_cli.hpo_evaluation import verify_gpu_termination
    from gf_training import hpo_gpu_lease as gpu
    monkeypatch.setattr(gpu, '_directory', lambda *a: None)
    monkeypatch.setattr(gpu, '_admission', lambda *a: None)
    monkeypatch.setattr(gpu, '_read', lambda *a: {})
    monkeypatch.setattr(gpu, '_termination', lambda *a: {'schema_version': 3, 'kind': 'rust-player-gpu-termination-v1'})
    with pytest.raises(ContractError, match='legacy'):
        verify_gpu_termination({'ticket': '1' * 32}, campaign_id='campaign', operation_id='operation',
            run_id='run', evaluation_id=None)


def test_legacy_physical_reader_keeps_its_existing_dataclass_positive(monkeypatch):
    """Synthetic namespace DATA, no actual job/termination capability issued."""
    from dataclasses import asdict

    from gf_cli.hpo_evaluation import verify_gpu_termination
    from gf_training import hpo_gpu_lease as gpu
    receipt = gpu.TerminationReceipt(1, 'hpo-gpu-termination', '1' * 32, 'a' * 64,
        'b' * 64, 'boot', 1., ())
    context = {'campaign_id': 'campaign', 'operation_id': 'operation', 'kind': 'training',
        'run_id': 'run', 'evaluation_id': None}
    monkeypatch.setattr(gpu, '_directory', lambda *a: None)
    monkeypatch.setattr(gpu, '_admission', lambda *a: None)
    monkeypatch.setattr(gpu, '_read', lambda *a: {'context': context})
    monkeypatch.setattr(gpu, '_termination', lambda *a: receipt)
    verify_gpu_termination(asdict(receipt), campaign_id='campaign', operation_id='operation', run_id='run', evaluation_id=None)


@pytest.mark.parametrize('copy_kind', ['copy', 'deepcopy', 'pickle'])
def test_new_completed_originals_cannot_copy_or_serialize_live_custody(copy_kind):
    """Partial cleanup object has no audit/capture authority to fabricate."""
    import copy
    import pickle

    from gf_training.rust_player_source import RustCompletedOriginals
    partial = RustCompletedOriginals.__new__(RustCompletedOriginals)
    operation = {'copy': copy.copy, 'deepcopy': copy.deepcopy, 'pickle': pickle.dumps}[copy_kind]
    with pytest.raises(TypeError): operation(partial)


def test_real_funded_provider_and_independent_fd_custody_overlap_and_positive_close(tmp_path):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_evaluation import _FundedEvaluationScan, _RetainedInferenceFiles
    from tests.test_hpo_artifacts import ACTIONS, RULES, configured_for, retained, saved
    learner, directory, _ = saved(tmp_path, trained=False)
    configured = configured_for(learner, directory)
    old_provider = retained(configured)
    label, binding = old_provider.binding(configured)
    expected = {row['path']: row['sha256'] for row in old_provider.journal_metadata(label)['files']}
    old_fds = [OriginalFile(directory / name, time.monotonic() + 30) for name in expected]
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 32 * 1024**2)
    custodian = object()
    hold = pool.hold_close(custodian)
    baseline = pool.budget.used
    files = _RetainedInferenceFiles(directory, expected, budget=pool.budget, parent_budget=phase)
    scan = _FundedEvaluationScan(pool, phase, max_owned_bytes=phase.limit - phase.used, deadline=pool.deadline)
    provider = None
    try:
        provider = RetainedGameArtifacts.for_hpo_registry({label: binding}, ACTIONS, pool_size=1,
            rules=RULES, max_owned_bytes=scan.limit, preflight_deadline=scan.deadline,
            finalization_deadline=pool.deadline, _scan=scan)
        scan.settle()
        steady = phase.used
        assert steady > scan.retained and pool.budget.used == baseline + steady
        old_provider.release()
        for original in old_fds: original.close()
        files.require()
        assert provider.journal_metadata(label)['policy_options'] == binding.policy_options
        provider.finalize(deadline=pool.deadline)
        scan.settle()
        with pytest.raises(RuntimeError): pool.close()
        provider.release()
        files.close()
        scan.close()
        assert phase.used == 0 and pool.budget.used == baseline
    finally:
        old_provider.release()
        for original in old_fds: original.close()
        if provider is not None: provider.release()
        files.close()
        scan.close()
        hold.retire(custodian)
        pool.close()


def test_fd_owner_preserves_original_before_fallible_dictionary_transfer(tmp_path, monkeypatch):
    import os

    from gf_training import rust_player_evaluation as module
    budget, phase = evaluation_account()
    expected = inference_files(tmp_path)
    owner = module._RetainedInferenceFiles.__new__(module._RetainedInferenceFiles)
    actual = module.OriginalFile
    fds = []
    class RefuseTransfer(dict):
        def __setitem__(self, key, value): raise MemoryError('dict transfer failed')
    def construct(path, deadline):
        original = actual(path, deadline)
        fds.append(original.fd)
        if path.name == 'weights-000.npz': owner._held = RefuseTransfer(owner._held)
        return original
    monkeypatch.setattr(module, 'OriginalFile', construct)
    with pytest.raises(MemoryError) as failed:
        owner.__init__(tmp_path, expected, budget=budget, parent_budget=phase)
    assert failed.value.cleanup_owner is owner and owner._pending_original is not None
    for fd in fds: os.fstat(fd)
    owner.close()
    for fd in fds:
        with pytest.raises(OSError): os.fstat(fd)
    assert budget.used == 0


def test_reference_owner_preserves_coupon_before_fallible_list_transfer(tmp_path):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.lineage import run_lock
    from gf_training.rust_player_evaluation import _ReferenceJournal
    reference_parent_data(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 16 * 1024**2)
    class RefuseTransfer(list):
        def append(self, value): raise MemoryError('coupon transfer failed')
    try:
        with run_lock(tmp_path) as ownership:
            owner = _ReferenceJournal(ownership, pool=pool, parent_budget=phase, deadline=pool.deadline)
            owner._loans = RefuseTransfer(owner._loans)
            with pytest.raises(MemoryError) as failed:
                owner.read(tmp_path / 'run.json')
            assert failed.value.cleanup_owner is owner and owner._pending_loan is not None
            assert phase.used > 128 * 1024
            owner.close()
            assert phase.used == 0
    finally:
        pool.close()


def sqlite_media_fixture(tmp_path):
    """Real tiny sealed SQLite/media bytes in-process, not engine/Source evidence."""
    import hashlib
    import time
    from dataclasses import asdict

    from gf_common.group_pack import (
        PackMember,
        StorageSettings,
        canonical_json,
        encode_pack,
        unpack_raw,
    )
    from gf_training.group_pack_db import descriptor_name
    from gf_training.group_pack_worker import GroupPackDatabase
    directory = tmp_path / 'originals/evaluation'
    directory.mkdir(parents=True)
    settings = StorageSettings(db_root=tmp_path / 'catalog', ram_root=tmp_path / 'ram')
    options = asdict(settings)
    options.update(db_root=str(settings.db_root), ram_root=str(settings.ram_root))
    db = GroupPackDatabase(max_owned_bytes=512 * 1024**2)
    ordinal = 0
    def command(kind, payload=None, **fields):
        nonlocal ordinal
        result = db.execute({'version': 1, 'request_id': ordinal, 'command': kind, **fields}, payload)
        ordinal += 1
        return result
    try:
        command('init', directory=str(directory), run_id='synthetic-media', settings=options, deadline=time.monotonic() + 30)
        pack = encode_pack((PackMember('journal', 'group/e0', b'fixture'),))
        command('submit', pack, group_id='group', raw_bytes=len(unpack_raw(pack)), record={},
            episodes=[{'episode_id': 'fixture', 'game_id': None, 'record': {}}])
        command('control', b'{}\n', name='manifest.json')
        command('control', b'{}\n', name='report.json')
        descriptor = command('seal', publication='published')
    finally:
        db.close()
    path = directory / descriptor_name('published')
    path.write_bytes(canonical_json(descriptor))
    envelope = tmp_path / 'envelope'
    envelope.mkdir()
    (envelope / 'collection.locator.json').write_bytes(canonical_json({'schema_version': 1,
        'format': 'sqlite-group-pack-locator-v1', 'descriptor': str(path),
        'descriptor_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    return envelope, directory


def test_borrowed_packed_wrapper_closes_own_fds_while_outer_actual_pool_hold_remains(tmp_path):
    import time

    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.group_pack_pool import PackedArtifactPool
    envelope, directory = sqlite_media_fixture(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    custodian = object()
    hold = pool.hold_close(custodian)
    before = pool.budget.used
    owner = PackedSourceOwner(envelope, PreparationConfig(), pool.deadline, storage_binding=storage_binding(),
        max_owned_bytes=pool.budget.limit, pool=pool, expected_descriptor_root=directory)
    try:
        assert owner.pool is pool and not owner._owns_pool
        with pytest.raises(RuntimeError): owner.close()
        owner.close(_audit_hold=hold, _audit_owner=custodian)
        assert owner._closed and pool.budget.used == before
        with pytest.raises(RuntimeError): pool.close()
    finally:
        hold.retire(custodian)
        owner.close()
        pool.close()


def test_unknown_terminal_fd_close_is_quarantined_without_number_retry_or_refund(tmp_path, monkeypatch):
    import os

    from gf_training.group_pack_resources import CleanupError
    from gf_training.rust_player_evaluation import _RetainedInferenceFiles
    budget, phase = evaluation_account()
    owner = _RetainedInferenceFiles(tmp_path, inference_files(tmp_path), budget=budget, parent_budget=phase)
    target = owner._held['weights-000.npz']
    number = target.fd
    actual_close = os.close
    attempts = []
    def uncertain(fd):
        actual_close(fd)
        if fd == number:
            attempts.append(fd)
            raise OSError('actual kernel close completed but terminal acknowledgment uncertain')
    before = budget.used
    with monkeypatch.context() as patch:
        patch.setattr(os, 'close', uncertain)
        with pytest.raises(OSError): owner.close()
    assert target.fd == -1 and budget.used == before and not owner._closed
    with pytest.raises(CleanupError): owner.close()
    assert attempts == [number] and budget.used == before
    with pytest.raises(OSError): os.fstat(number)
    # Intentionally retain the logically charged quarantine. No raw FD remains;
    # there is no fabricated positive receipt or operator recovery in this test.


def test_copied_evaluation_metadata_has_independent_graphs_and_actual_overlap_funding():
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _RetainedEvaluationMetadata
    manifest, proof = {'nested': {'values': [1, 2]}}, {'manifest_sha256': 'a' * 64}
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 4 * 1024**2)
    holder = object()
    hold = pool.hold_close(holder)
    old = pool.budget.reserve(32 * json_bound((manifest, proof)) + 65536)
    before = pool.budget.used
    copied = _RetainedEvaluationMetadata(manifest, proof, pool=pool, parent_budget=phase)
    try:
        assert copied.value['manifest'] == manifest and copied.value['manifest'] is not manifest
        assert copied.value['manifest']['nested']['values'] is not manifest['nested']['values']
        assert pool.budget.used == before + phase.used and phase.used > 0
        manifest.clear()
        old.close()
        assert copied.value['manifest']['nested']['values'] == [1, 2]
        copied.close()
        assert phase.used == 0
    finally:
        copied.close()
        old.close()
        hold.retire(holder)
        pool.close()


@pytest.mark.parametrize('fault', ['wrong-owner', 'foreign', 'other-hold', 'retired', 'construct-uncertain', 'retire-uncertain', 'group-loan'])
def test_borrowed_verifier_exemption_refuses_foreign_or_uncertain_custody_without_effect(tmp_path, monkeypatch, fault):
    import time

    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.group_pack_pool import PackedArtifactPool
    envelope, directory = sqlite_media_fixture(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    owner = PackedSourceOwner(envelope, PreparationConfig(), pool.deadline, storage_binding=storage_binding(),
        max_owned_bytes=pool.budget.limit, pool=pool, expected_descriptor_root=directory)
    holder, other_holder = object(), object()
    other_pool = None
    actual_acquire = pool.budget.acquire
    if fault == 'construct-uncertain':
        def fail(size, deadline, **kwargs):
            actual_acquire(size, deadline, **kwargs)
            raise OSError('hold construction acknowledgment uncertain')
        with monkeypatch.context() as patch:
            patch.setattr(pool.budget, 'acquire', fail)
            with pytest.raises(OSError): pool.hold_close(holder)
        hold = pool._close_holds[0]
    else:
        hold = pool.hold_close(holder)
    passed, passed_owner = hold, holder
    if fault == 'wrong-owner': passed_owner = other_holder
    elif fault == 'foreign':
        other_pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=pool.deadline)
        passed = other_pool.hold_close(holder)
    elif fault == 'other-hold': pool.hold_close(other_holder)
    elif fault == 'retired': hold.retire(holder)
    elif fault == 'retire-uncertain':
        def fail(size): raise OSError('before hold refund')
        with monkeypatch.context() as patch:
            patch.setattr(pool.budget, 'release', fail)
            with pytest.raises(OSError): hold.retire(holder)
    elif fault == 'group-loan':
        assert owner.artifacts is not None
        loan_context = owner.artifacts.borrow_group('group')
        loan_context.__enter__()
    before = pool.budget.used
    try:
        with pytest.raises(RuntimeError): owner.close(_audit_hold=passed, _audit_owner=passed_owner)
        assert pool.budget.used == before and not owner._closed
    finally:
        if fault == 'group-loan': loan_context.__exit__(None, None, None)
        for actual in tuple(pool._close_holds): actual.retire(actual.owner)
        owner.close()
        pool.close()
        if other_pool is not None:
            for actual in tuple(other_pool._close_holds): actual.retire(actual.owner)
            other_pool.close()


def test_owning_wrapper_and_default_borrowed_close_keep_all_terminal_hold_guards(tmp_path):
    import time

    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    envelope, directory = sqlite_media_fixture(tmp_path)
    owner = PackedSourceOwner(envelope, PreparationConfig(), time.monotonic() + 30,
        storage_binding=storage_binding(), max_owned_bytes=128 * 1024**2, expected_descriptor_root=directory)
    holder = object()
    hold = owner.pool.hold_close(holder)
    before = owner.pool.budget.used
    try:
        with pytest.raises(RuntimeError): owner.close()
        with pytest.raises(RuntimeError): owner.close(_audit_hold=hold, _audit_owner=holder)
        assert owner.pool.budget.used == before and not owner._closed
    finally:
        hold.retire(holder)
        owner.close()


def test_actual_borrowed_retirement_adapter_positive_retry_keeps_pool_and_copy_funding(tmp_path):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import (
        _BorrowedVerifierRetirement,
        _RetainedEvaluationMetadata,
    )
    envelope, directory = sqlite_media_fixture(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 8 * 1024**2)
    holder = object()
    hold = pool.hold_close(holder)
    before = pool.budget.used
    owner = PackedSourceOwner(envelope, PreparationConfig(), pool.deadline, storage_binding=storage_binding(),
        max_owned_bytes=pool.budget.limit, pool=pool, expected_descriptor_root=directory)
    copy = _RetainedEvaluationMetadata({'manifest': 'owned-copy'}, {'proof': 'owned-copy'}, pool=pool, parent_budget=phase)
    adapter = _BorrowedVerifierRetirement(owner, hold, holder)
    actual = owner.locator.close
    def fail(): raise OSError('before actual verifier locator close')
    owner.locator.close = fail
    allocated = pool.budget.used
    def stopped_failed_retirement():
        with pytest.raises(OSError) as failed:
            adapter.close()
        adapter.capture(failed.value)
    try:
        stopped_failed_retirement()
        # The independent packed child positively closed; its own index can
        # refund. The still-live wrapper/locator and new copy remain funded.
        assert before + copy.quota_bytes + owner._charge_bytes == pool.budget.used <= allocated
        assert not owner._closed and not adapter.closed
        assert phase.used == copy.quota_bytes
        owner.locator.close = actual
        adapter.close()
        adapter.close()
        assert owner._closed and pool.budget.used == before + copy.quota_bytes
        assert copy.value['manifest']['manifest'] == 'owned-copy'
        with pytest.raises(RuntimeError): pool.close()
        copy.close()
        assert pool.budget.used == before
    finally:
        owner.locator.close = actual
        adapter.close()
        copy.close()
        hold.retire(holder)
        pool.close()


def test_shared_saved_reader_requires_real_audit_owner_before_file_work(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gf_cli.hpo_evaluation import verify_saved_evaluation
    from gf_common.preprocessing import PreparationConfig
    loaded = SimpleNamespace(base=SimpleNamespace(config=SimpleNamespace(preprocessing=PreparationConfig())),
        config=SimpleNamespace(storage=SimpleNamespace(originals_root=str(tmp_path))))
    def fail(*args, **kwargs): raise AssertionError('forged shared reader cannot open files')
    monkeypatch.setattr('os.open', fail)
    with pytest.raises(ContractError, match='actual'):
        verify_saved_evaluation(tmp_path, loaded, {'evaluation_id': 'eval'}, deadline=1., _pool=object(), _audit_owner=metadata_data())


@pytest.mark.parametrize('case', ['test_actual_borrowed_scan_and_custody_pool_terminal_closes_refuse_until_join',
    'test_independent_actual_holds_and_wrong_owner_cannot_release_another'])
def test_legacy_native_hold_contract_pure_regression_with_inprocess_media(tmp_path, monkeypatch, case):
    """Reuse original assertions; no FFI submission/native worker or writer process."""
    from tests import test_native_group_pack_close_hold as legacy
    monkeypatch.setattr(legacy, 'packed', lambda root: (sqlite_media_fixture(root)[0], None))
    getattr(legacy, case)(tmp_path)


@pytest.mark.parametrize('alias', ['pool', 'budget'])
def test_live_parent_receiver_rejects_old_pool_or_budget_without_any_effect(alias):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _require_independent_evaluation_pool
    first = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    second = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=first.deadline)
    original_second = second.budget
    before = first.budget.used, second.budget.used
    try:
        if alias == 'pool':
            receiving = first
        else:
            # Only a pure negative identity check: both pools are real and their
            # original account is restored before any close/acquire can occur.
            second.budget = first.budget
            receiving = second
        with pytest.raises(ContractError, match='independent'):
            _require_independent_evaluation_pool(first, receiving)
        assert (first.budget.used, original_second.used) == before
        assert not first._close_holds and not second._close_holds
    finally:
        second.budget = original_second
        first.close()
        second.close()


def test_distinct_receiving_pool_identity_check_preserves_accounts():
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _require_independent_evaluation_pool
    first = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    second = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=first.deadline)
    before = first.budget.used, second.budget.used
    try:
        _require_independent_evaluation_pool(first, second)
        assert (first.budget.used, second.budget.used) == before
    finally:
        first.close()
        second.close()


def chained_array_failure(kind, refs, *, child=None, cycle=False):
    """Real chained stopped frames/arrays; no Source/Parent/job authority."""
    import weakref

    import numpy as np
    def inner():
        decoded = np.arange(512, dtype=np.int64)
        refs.append(weakref.ref(decoded))
        failure = ValueError('inner array decode failure')
        if child is not None:
            child.array = decoded
            failure.cleanup_owner = child
        raise failure
    def outer():
        try:
            inner()
        except ValueError as cause:
            if kind == 'cause': raise ContractError('outer controlled failure') from cause
            if kind == 'suppressed': raise ContractError('outer controlled failure') from None
            raise ContractError('outer controlled failure')
    try:
        outer()
    except ContractError as failure:
        if cycle:
            linked = failure.__cause__ or failure.__context__
            linked.__context__ = failure
        return failure


class ChainedDataOwner:
    """Actual array consumer attached only to an inner DATA exception."""
    def __init__(self, refs, *, loan=None, blocked=False):
        self.array, self.loan, self.blocked, self.calls = None, loan, blocked, 0
        self.refs = refs
    def close(self):
        self.calls += 1
        if self.blocked: raise OSError('actual chained data owner still live')
        self.array = None
        assert self.refs[0]() is None
        if self.loan is not None: self.loan.close()


@pytest.mark.parametrize('kind', ['cause', 'context', 'suppressed'])
@pytest.mark.parametrize('nested_refusal', [False, True])
def test_numeric_chained_frames_and_cause_only_owner_retire_before_actual_loan(kind, nested_refusal):
    import time

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import _FailedNumericAudit
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    before = pool.budget.used
    loan = pool.budget.reserve(4096)
    refs = []
    child = ChainedDataOwner(refs, blocked=True) if nested_refusal else None
    error = chained_array_failure(kind, refs, child=child)
    inner = error.__cause__ or error.__context__
    original_cause, original_context, suppressed = error.__cause__, error.__context__, error.__suppress_context__
    owner = _FailedNumericAudit(loan, error)
    actual = loan.close
    def refund():
        assert refs[0]() is None
        actual()
    loan.close = refund
    try:
        if child is not None:
            with pytest.raises(OSError, match='chained data owner'): owner.close()
            assert pool.budget.used == before + 4096 and not loan.closed and not owner.closed
            assert refs[0]() is not None
            child.blocked = False
        owner.close()
        owner.close()
        assert refs[0]() is None and pool.budget.used == before
        assert error.__cause__ is original_cause and error.__context__ is original_context
        assert error.__suppress_context__ is suppressed and inner is not None
        if child is not None: assert child.calls == 2
    finally:
        # Failure evidence stays honest but tests leave no actual allocation.
        import traceback
        traceback.clear_frames(inner.__traceback__)
        if child is not None: child.blocked = False; child.array = None
        loan.close = actual
        owner.close()
        pool.close()


@pytest.mark.parametrize('kind', ['cause', 'context', 'suppressed'])
@pytest.mark.parametrize('nested_refusal', [False, True])
def test_borrowed_adapter_clears_all_chained_metadata_frames_before_refund(tmp_path, kind, nested_refusal):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_evaluation import _BorrowedVerifierRetirement
    envelope, directory = sqlite_media_fixture(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 8 * 1024**2)
    holder = object()
    hold = pool.hold_close(holder)
    before = pool.budget.used
    verifier = PackedSourceOwner(envelope, PreparationConfig(), pool.deadline, storage_binding=storage_binding(),
        max_owned_bytes=pool.budget.limit, pool=pool, expected_descriptor_root=directory)
    loan = phase.reserve(4096)
    refs = []
    child = ChainedDataOwner(refs, loan=loan, blocked=nested_refusal)
    error = chained_array_failure(kind, refs, child=child)
    inner = error.__cause__ or error.__context__
    adapter = _BorrowedVerifierRetirement(verifier, hold, holder)
    adapter.capture(error)
    try:
        if nested_refusal:
            charged = pool.budget.used
            with pytest.raises(OSError): adapter.close()
            assert pool.budget.used == charged and phase.used == 4096 and not verifier._closed
            assert not loan.closed and refs[0]() is not None
            child.blocked = False
        adapter.close()
        adapter.close()
        assert refs[0]() is None and phase.used == 0 and pool.budget.used == before
        assert child.calls == (2 if nested_refusal else 1)
    finally:
        import traceback
        traceback.clear_frames(inner.__traceback__)
        child.blocked = False
        child.array = None
        adapter.close()
        loan.close()
        hold.retire(holder)
        pool.close()


@pytest.mark.parametrize('owner_kind', ['numeric', 'borrowed'])
def test_failure_graph_duplicate_links_cycles_and_recapture_do_not_duplicate_cleanup(tmp_path, owner_kind):
    import time

    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import _FailedNumericAudit
    from gf_training.rust_player_evaluation import _BorrowedVerifierRetirement
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    holder = object(); hold = pool.hold_close(holder)
    refs = []; child = ChainedDataOwner(refs)
    error = chained_array_failure('cause', refs, child=child, cycle=True)
    inner = error.__cause__
    loan = pool.budget.reserve(4096)
    if owner_kind == 'numeric': owner = _FailedNumericAudit(loan, error)
    else:
        envelope, directory = sqlite_media_fixture(tmp_path)
        verifier = PackedSourceOwner(envelope, PreparationConfig(), pool.deadline, storage_binding=storage_binding(),
            max_owned_bytes=pool.budget.limit, pool=pool, expected_descriptor_root=directory)
        owner = _BorrowedVerifierRetirement(verifier, hold, holder)
        owner.capture(error)
        owner.capture(error)
    try:
        owner.close(); owner.close()
        assert refs[0]() is None and child.calls == 1
    finally:
        import traceback
        traceback.clear_frames(inner.__traceback__)
        child.array = None
        owner.close(); loan.close(); hold.retire(holder); pool.close()


@pytest.mark.parametrize('late_consumer', ['none', 'positive', 'refusal'])
def test_failure_recapture_retires_new_funded_caller_frame_and_late_owner(late_consumer):
    import time
    import traceback
    import weakref

    import numpy as np

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import _capture_failure_graph, _FailedNumericAudit
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    before = pool.budget.used
    loan = pool.budget.reserve(4096)
    refs, captured = [], {}
    child = None if late_consumer == 'none' else ChainedDataOwner(refs, blocked=late_consumer == 'refusal')
    def lower():
        try:
            raise ContractError('actual lower DATA read failed')
        except ContractError as error:
            captured['owner'] = _FailedNumericAudit(loan, error)
            raise
    def higher():
        payload = np.arange(512, dtype=np.int64)
        refs.append(weakref.ref(payload))
        try:
            lower()
        except ContractError as error:
            if child is not None:
                child.array = payload
                error.cleanup_owner = child  # first attached AFTER lower capture
            raise
    def receive():
        try:
            higher()
        except ContractError as error:
            _capture_failure_graph(captured['owner'], error)
            return error
    error = receive()
    owner = captured['owner']
    refund = loan.close
    def checked_refund():
        assert refs[0]() is None  # including newly prepended higher/receive frames
        refund()
    loan.close = checked_refund
    try:
        if late_consumer == 'refusal':
            with pytest.raises(OSError, match='chained data owner'):
                owner.close()
            assert pool.budget.used == before + 4096 and not loan.closed and not owner.closed
            assert refs[0]() is not None
            child.blocked = False
        owner.close()
        owner.close()
        assert refs[0]() is None and pool.budget.used == before
        assert error.cleanup_owner is owner
        if child is not None:
            assert child.calls == (2 if late_consumer == 'refusal' else 1)
    finally:
        traceback.clear_frames(error.__traceback__)
        if child is not None:
            child.blocked = False
            child.array = None
        loan.close = refund
        owner.close()
        pool.close()


@pytest.mark.parametrize('late_link', ['cause', 'context', 'suppressed'])
def test_failure_recapture_revisits_new_links_and_refuses_active_frames(late_link):
    import time
    import traceback

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_completion import _capture_failure_graph, _FailedNumericAudit
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    before = pool.budget.used
    loan = pool.budget.reserve(4096)
    refs, captured = [], {}
    child = ChainedDataOwner(refs)
    def lower():
        try:
            raise ContractError('lower DATA failure before linked producer fails')
        except ContractError as error:
            captured['owner'] = _FailedNumericAudit(loan, error)
            raise
    def higher():
        try:
            lower()
        except ContractError as error:
            linked = chained_array_failure('context', refs, child=child)
            if late_link == 'cause':
                error.__cause__ = linked
            else:
                error.__context__ = linked
                error.__suppress_context__ = late_link == 'suppressed'
            inner = linked.__cause__ or linked.__context__
            inner.__context__ = error  # cyclic arm preserves the actual inner producer
            captured['linked'] = linked
            raise
    def receive():
        try:
            higher()
        except ContractError as error:
            _capture_failure_graph(captured['owner'], error)
            # Current receiver is part of the newly captured traceback and is
            # still executing: its funded graph cannot positively retire yet.
            with pytest.raises(RuntimeError, match='still executing'):
                captured['owner'].close()
            assert pool.budget.used == before + 4096 and not loan.closed
            return error
    error = receive()
    owner, linked = captured['owner'], captured['linked']
    original_links = error.__cause__, error.__context__, error.__suppress_context__
    refund = loan.close
    def checked_refund():
        assert refs[0]() is None
        refund()
    loan.close = checked_refund
    try:
        owner.close()
        owner.close()
        assert child.calls == 1 and refs[0]() is None and pool.budget.used == before
        assert (error.__cause__, error.__context__, error.__suppress_context__) == original_links
    finally:
        traceback.clear_frames(error.__traceback__)
        traceback.clear_frames(linked.__traceback__)
        if linked.__cause__ is not None: traceback.clear_frames(linked.__cause__.__traceback__)
        if linked.__context__ is not None: traceback.clear_frames(linked.__context__.__traceback__)
        child.array = None
        loan.close = refund
        owner.close()
        pool.close()


@pytest.mark.parametrize('wrapper_kind', ['numeric', 'raw'])
@pytest.mark.parametrize('link', ['cause', 'context', 'suppressed'])
@pytest.mark.parametrize('transitive', [False, True])
def test_failure_capture_rejects_actual_owner_back_edge_before_positive_retry(tmp_path, wrapper_kind, link, transitive):
    import time
    import traceback
    import weakref

    import numpy as np

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_completion import (
        _capture_failure_graph,
        _FailedNumericAudit,
        _FailedRawAudit,
    )
    from gf_training.rust_player_evaluation import _RetainedEvaluationMetadata
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 8 * 1024**2)
    before = pool.budget.used
    loan = pool.budget.reserve(4096)
    refs = []
    outer = _RetainedEvaluationMetadata({'outer': 'actual DATA'}, {}, pool=pool, parent_budget=phase)
    middle = (_RetainedEvaluationMetadata({'middle': 'actual DATA'}, {}, pool=pool, parent_budget=phase)
        if transitive else None)
    late = _RetainedEvaluationMetadata({'late': 'independent DATA'}, {}, pool=pool, parent_budget=phase)
    class ActualArrayConsumer:
        def __init__(self):
            self.array, self.error, self.closed, self.blocked = None, None, False, True
        def close(self):
            if self.closed: return
            if self.blocked:
                payload = self.array  # funded alias in the real failed-close frame
                assert payload is not None
                try:
                    raise self.error
                except ContractError as original:
                    error = OSError('actual child cleanup still blocked')
                    error.cleanup_owner = late  # first published ONLY on cleanup failure
                    if link == 'cause': raise error from original
                    if link == 'suppressed': raise error from None
                    raise error
            self.array, self.closed = None, True
    child = ActualArrayConsumer()
    def failed_data():
        payload = np.arange(512, dtype=np.int64)
        refs.append(weakref.ref(payload))
        child.array = payload
        error = ContractError('original actual funded DATA audit failure')
        error.cleanup_owner = child
        raise error
    def capture_data():
        try:
            failed_data()
        except ContractError as error:
            return error
    error = capture_data()
    child.error = error
    original = None
    if wrapper_kind == 'numeric':
        wrapper = _FailedNumericAudit(loan, error)
    else:
        path = tmp_path / 'raw-original.json'
        path.write_bytes(b'{"actual":"original DATA"}')
        original = OriginalFile(path, pool.deadline)
        class ActualRawOriginals:
            def __init__(self): self.closed = False
            def close(self):
                if self.closed: return
                original.close()
                loan.close()
                self.closed = True
        wrapper = _FailedRawAudit(ActualRawOriginals(), error)
    if middle is not None: _capture_failure_graph(middle, error)
    _capture_failure_graph(outer, error)
    expected_parent = middle if middle is not None else outer
    assert wrapper in expected_parent._cleanup_children
    if middle is not None: assert middle in outer._cleanup_children
    refund = loan.close
    def checked_refund():
        assert child.closed and late._closed and refs[0]() is None
        refund()
    loan.close = checked_refund
    charged, phase_charged = pool.budget.used, phase.used
    try:
        with pytest.raises(OSError, match='actual child cleanup still blocked'):
            outer.close()
        assert pool.budget.used == charged and phase.used == phase_charged
        assert not wrapper.closed and not loan.closed and not outer._closed
        assert not late._closed and refs[0]() is not None
        if original is not None: assert original.fd >= 0
        assert outer not in wrapper._cleanup_children  # A→(middle→)L must not become L→A
        assert late in wrapper._cleanup_children  # omit ONLY the reverse owner
        child.blocked = False
        outer.close()
        outer.close()
        assert wrapper.closed and child.closed and late._closed and loan.closed
        assert refs[0]() is None and phase.used == 0 and pool.budget.used == before
        if original is not None: assert original.fd == -1
    finally:
        # The RED reproduction is bounded: remove its actual reverse edge before
        # positive cleanup rather than recursively running the broken graph.
        wrapper._cleanup_children[:] = [owner for owner in wrapper._cleanup_children
            if owner is not outer and owner is not middle]
        child.blocked = False
        traceback.clear_frames(error.__traceback__)
        outer.close()
        if middle is not None: middle.close()
        wrapper.close()
        late.close()
        loan.close = refund
        loan.close()
        pool.close()
