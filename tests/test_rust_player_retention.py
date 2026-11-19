"""Artifact policy/data checks; records never confer live deletion authority."""
from copy import deepcopy

import pytest

from gf_common.domain import ContractError


def records(count):
    return {g: {'generation': g, 'used_decisions': g * 100,
        'artifact_files': {n: 'a' * 64 for n in ('inference.json', 'weights-000.npz', 'weights-001.npz', 'resume.json', 'state-000.npz')},
        'parent_controls': {'snapshot_pool': [0, count - 1]}} for g in range(count)}


def test_ordinary_predecessor_policy_uses_declared_shards_and_active_references():
    from gf_training.rust_player_retention import protected_files
    rows = records(13)
    registry = {'evaluation_references': {}, 'service_references': {}}
    keep = protected_files(rows, registry, generation=12)
    assert keep[11] == {'inference.json'}
    assert keep[10] == set(rows[10]['artifact_files'])
    assert keep[0] == set(rows[0]['artifact_files'])
    registry['service_references']['live'] = {'status': 'published', 'binding': {'generation': 11}}
    assert protected_files(rows, registry, generation=12)[11] == set(rows[11]['artifact_files'])
    registry['service_references']['live']['status'] = 'released'
    assert protected_files(rows, registry, generation=12)[11] == {'inference.json'}


def test_hpo_policy_is_exact_existing_pure_recipe_with_truthful_references():
    from gf_training.hpo_lineage import protected_hpo_files_for_operating
    from gf_training.rust_player_retention import protected_files
    rows = records(13)
    operating = {'resume_latest': 2, 'inference_latest': 3, 'inference_periodic_decisions': 400}
    registry = {'evaluation_references': {'eval': {'generation': 3}},
        'service_references': {'cache': {'status': 'published', 'binding': {'generation': 5}},
            'old': {'status': 'released', 'binding': {'generation': 6}}}}
    expected = protected_hpo_files_for_operating(operating, list(rows.values()),
        {'evaluation_references': registry['evaluation_references'], 'cache_references': {'cache': {'generation': 5}}}, [{'generation': 7}])
    assert protected_files(rows, registry, generation=12, operating=operating, pending=[7]) == expected


def intent():
    return {'schema_version': 1, 'kind': 'rust-player-artifact-retirement-intent-v1',
        'run_id': 'run', 'operation_id': '1' * 32, 'invocation_id': '2' * 32, 'retirement_id': '3' * 32,
        'generation': 4, 'file': 'weights-001.npz', 'sha256': 'a' * 64,
        'original_identity': [1, 2, 3, 4, 5], 'staging_basename': '.prune-' + '3' * 32,
        'parent_identity': [1, 6], 'source_directory_identity': [1, 7], 'stage_directory_identity': [1, 8],
        'runtime_binding_sha256': 'b' * 64}


@pytest.mark.parametrize('field,value', [('generation', True), ('file', '../raw'), ('original_identity', [1, 2, True, 4, 5]),
    ('parent_identity', [1, -1]), ('staging_basename', '.prune-foreign'), ('sha256', 'A' * 64), ('extra', None)])
def test_strict_retirement_intent_rejects_untrusted_fields(field, value):
    from gf_training.rust_player_retention import validate_retirement_record
    row = intent()
    assert validate_retirement_record(row, receipt=False) is row
    bad = deepcopy(row)
    bad[field] = value
    with pytest.raises(ContractError):
        validate_retirement_record(bad, receipt=False)


def test_json_or_equal_field_owner_cannot_authorize_retirement(tmp_path):
    from gf_training.rust_player_retention import RustArtifactRetirement
    with pytest.raises(ContractError, match='actual'):
        RustArtifactRetirement(intent(), 4, 'weights-001.npz', {}, 1.)


@pytest.mark.parametrize('changed', ['sha256', 'operation_id', 'invocation_id', 'file', 'original_identity', 'intent_sha256'])
def test_receipt_must_bind_same_exact_immutable_intent(changed):
    from gf_training.rust_player_retention import validate_retirement_pair
    row = intent()
    receipt = {key: value for key, value in row.items() if key not in ('staging_basename',
        'parent_identity', 'source_directory_identity', 'stage_directory_identity')}
    receipt.update(kind='rust-player-artifact-retirement-receipt-v1', intent_sha256='c' * 64, completed_unix=1.)
    validate_retirement_pair(row, receipt, intent_sha256='c' * 64)
    receipt[changed] = [1, 2, 4, 4, 5] if changed == 'original_identity' else ('0' * 64 if changed.endswith('sha256') else 'different')
    with pytest.raises(ContractError):
        validate_retirement_pair(row, receipt, intent_sha256='c' * 64)


def run_retention_control(control, ownership):
    """Genuine CPU admission/lock/registry, synthetic artifact DATA only.

    Never calls collection, local.update/save, Prepared or any Source/Parent issuer.
    Each fault gets its own supervisor/parent; a failed parent is never reset.
    """
    import hashlib
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from gf_cli.config import load_rust_player_effective
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_common.rust_player import deadline_seconds
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player import RustPlayerInference
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_resources import CleanupError
    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.rust_player_learning import LocalGameLearner
    from gf_training.rust_player_lineage import record_fields
    from gf_training.rust_player_registry import RustRunRegistry
    from gf_training.rust_player_retention import RustArtifactRetirement, validate_retirement_pair

    case = os.environ.get('GFB_RETENTION_CONTROL_CASE', 'success')
    assert case in {'success', 'replacement', 'intent-write', 'after-intent', 'rename', 'unlink',
        'source-fsync', 'stage-fsync', 'root-fsync', 'root-fsync-after-rmdir', 'close', 'receipt-write',
        'sha', 'stage-notempty', 'stage-replacement', 'close-before-release'}
    loaded = load_rust_player_effective(Path(control.operation['config_path']), expected_sha256=control.operation['config_sha256'])
    bound = json.loads((control.root / 'component-runtime.json').read_bytes())
    assert digest(bound) == control.operation['runtime_binding_sha256']
    admission = control.admit({'managed_ram_bytes': 24 * 1024**3, 'managed_cuda_bytes': 0,
        'source_workspace_bytes': 2 * 1024**3, 'storage_bytes': 1024**3,
        'game_concurrency': 1, 'player_worker_threads': 1, 'max_batch_size': 1}, bound['producer'])
    actual = admission.observation
    deadline = deadline_seconds(control.operation['deadlines']['execution_bits'])
    pool = PackedArtifactPool(max_owned_bytes=2 * 1024**3, deadline=deadline)
    writer = OwnedControlWriter(ownership, budget=pool.budget, limit=64 * 1024**2)
    local = LocalGameLearner(loaded.config.model.architecture(layout=layout), 17, loaded.config.model.inference,
        'c' * 64, actions=loaded.config.actions, rules=card_instance_rules(), admission=admission)
    service = RustPlayerInference(device='cpu', cache_entries=1, cache_weight_bytes=2 * 1024**3)
    state = {'run_id': actual['run_id'], 'updates': 12, 'config_sha256': loaded.sha256,
        'identity': {'player_runtime': bound}, 'runtime_binding_sha256': digest(bound)}
    registry = RustRunRegistry(ownership, state, config=loaded.config, admission=admission,
        pool=pool, writer=writer, local=local, service=service)
    files = ('inference.json', 'weights-000.npz', 'weights-001.npz', 'resume.json', 'state-000.npz')
    for generation in range(13):
        directory = ownership.directory / 'artifacts' / f'{generation:06}'
        directory.mkdir(parents=True)
        artifact_files = {}
        for name in files:
            payload = f'data-only-{generation}-{name}'.encode()
            (directory / name).write_bytes(payload)
            artifact_files[name] = hashlib.sha256(payload).hexdigest()
        row = dict.fromkeys(record_fields('generation'))
        row.update(schema_version=4, run_id=actual['run_id'], generation=generation,
            artifact_files=artifact_files, inference_sha256=artifact_files['inference.json'],
            resume_sha256=artifact_files['resume.json'], snapshot_identity=digest(artifact_files),
            parent_controls={'snapshot_pool': [0, 12]}, runtime_binding_sha256=digest(bound))
        registry.records[generation] = row
        registry.registry['generations'][str(generation)] = {'inference_sha256': row['inference_sha256'],
            'snapshot_identity': row['snapshot_identity'], 'files': {name: {'sha256': sha, 'status': 'retained',
                'deletion_intent': None, 'deletion_receipt': None} for name, sha in artifact_files.items()}}
    registry.writer.write(ownership.directory / 'artifact-registry.json', registry.registry, deadline=deadline)
    # Competitor must not acquire the actual parent's original flock.
    result = subprocess.run([sys.executable, '-c',
        'import fcntl,sys; f=open(sys.argv[1],"r+b");\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(3)\nsys.exit(0)',
        str(ownership.directory / '.lock')], check=False, capture_output=True, timeout=5)
    assert result.returncode == 3, result.stderr
    assert local.learner.steps == 0
    target_name = 'weights-001.npz'
    row = registry.registry['generations']['11']['files'][target_name]
    owner = RustArtifactRetirement(registry, 11, target_name, row, deadline)
    target = ownership.directory / 'artifacts' / '000011' / target_name
    if case == 'sha':
        target.write_bytes(b'changed artifact content')
    events = []
    actual_write, actual_rename, actual_unlink, actual_fsync = writer.write, os.rename, os.unlink, os.fsync
    root_syncs = 0
    injected = False

    def write(path, value, **kwargs):
        nonlocal injected
        name = path.name
        events.append('write:' + name)
        if name == 'receipt.json':
            assert owner.closed and not owner.fds
            assert not target.exists() and not (target.parent.parent / owner.staging).exists()
        if not injected and ((case == 'intent-write' and name == 'intent.json')
                or (case == 'after-intent' and name == 'artifact-registry.json')
                or (case == 'receipt-write' and name == 'receipt.json')):
            injected = True
            raise OSError('injected ' + case)
        if case == 'replacement' and name == 'intent.json':
            target.rename(target.with_suffix('.old'))
            target.write_bytes(b'foreign replacement')
        checksum = actual_write(path, value, **kwargs)
        if case == 'stage-replacement' and name == 'intent.json':
            stage = target.parent.parent / owner.staging
            stage.rename(stage.with_name(stage.name + '-original'))
            stage.mkdir()
        return checksum

    def rename(*args, **kwargs):
        nonlocal injected
        if kwargs.get('src_dir_fd') == owner.fds.get('source', SimpleNamespace(fd=-1)).fd:
            events.append('rename')
            if case == 'rename':
                injected = True
                raise OSError('injected rename')
        return actual_rename(*args, **kwargs)

    def unlink(*args, **kwargs):
        nonlocal injected
        if kwargs.get('dir_fd') == owner.fds.get('stage', SimpleNamespace(fd=-1)).fd:
            events.append('unlink')
            if case == 'unlink':
                injected = True
                raise OSError('injected unlink')
        return actual_unlink(*args, **kwargs)

    def fsync(fd):
        nonlocal root_syncs, injected
        role = next((name for name, held in owner.fds.items() if held.fd == fd), None)
        if role:
            events.append('fsync:' + role)
            if case == 'stage-notempty' and role == 'stage' and owner.phase == 'unlinked':
                foreign = os.open('foreign', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
                os.close(foreign)
            if role == 'parent':
                root_syncs += 1
            if not injected and owner.phase in ('unlinked', 'removed') and (case == role + '-fsync' or case == 'root-fsync' and role == 'parent'
                    or case == 'root-fsync-after-rmdir' and role == 'parent' and owner.phase == 'removed'):
                # Writer durability also syncs these directories; inject only
                # after the actual artifact has been destructively unlinked.
                injected = True
                raise OSError('injected ' + case)
        return actual_fsync(fd)

    from gf_common import owned_completion
    actual_close = owned_completion._close_owned_descriptor
    from gf_training.group_pack_resources import OwnedDescriptor
    actual_owner_close = OwnedDescriptor.close

    def owner_close(held):
        nonlocal injected
        if case == 'close-before-release' and not injected and held is owner.fds.get('original'):
            injected = True
            os.fstat(held.fd)
            raise OSError('injected before owned terminal-close attempt')
        return actual_owner_close(held)

    def close(held, fd, identity, role):
        nonlocal injected
        actual_close(held, fd, identity, role)
        if case == 'close' and held is owner.fds.get('original'):
            injected = True
            error = OSError('injected uncertain terminal close after actual kernel close')
            held._uncertain_releases.append((role, fd, identity, error))
            raise error

    try:
        with patch.object(writer, 'write', write), patch('os.rename', rename), patch('os.unlink', unlink), \
                patch('os.fsync', fsync), patch.object(owned_completion, '_close_owned_descriptor', close), \
                patch.object(OwnedDescriptor, 'close', owner_close):
            if case == 'success':
                owner.run()
            else:
                with pytest.raises((OSError, ContractError)):
                    owner.run()
        if case == 'success':
            with pytest.raises(ContractError, match='single use'):
                owner.run()
            assert row['status'] == 'deleted' and owner.closed and owner.phase == 'complete'
            assert not target.exists() and not (target.parent.parent / owner.staging).exists()
            receipt_path = ownership.directory / row['deletion_receipt']['path']
            intent_path = ownership.directory / row['deletion_intent']['path']
            intent_data = json.loads(intent_path.read_bytes())
            receipt_data = json.loads(receipt_path.read_bytes())
            validate_retirement_pair(intent_data, receipt_data, intent_sha256=row['deletion_intent']['sha256'])
            assert receipt_data['operation_id'] == actual['operation_id']
            assert receipt_data['invocation_id'] == actual['invocation_id']
            assert events.index('rename') < events.index('unlink') < events.index('write:receipt.json')
            assert root_syncs >= 2
            # Existing ordinary predecessor recipe now retires every other
            # declared shard, never a raw or numeric original.
            registry._retain_artifact_files(deadline=deadline)
            assert all(registry.registry['generations']['11']['files'][name]['status'] == 'deleted' for name in files if name != 'inference.json')
            assert (target.parent / 'inference.json').exists()
            assert all((ownership.directory / 'artifacts' / '000010' / name).exists() for name in files)
            assert not (ownership.directory / 'numeric').exists()
            assert not (ownership.directory / 'collections').exists()
        else:
            assert registry.failed
            assert row['status'] == ('retained' if case in ('intent-write', 'sha') else 'deleting')
            assert row['deletion_receipt'] is None
            assert not list((ownership.directory / 'retirement').rglob('receipt.json'))
            with pytest.raises(ContractError):
                registry.require()
            if case == 'replacement':
                assert target.read_bytes() == b'foreign replacement'
            if case == 'close':
                before = pool.budget.used
                with pytest.raises(CleanupError):
                    owner.close()
                assert pool.budget.used == before and not owner.closed
            else:
                owner.close()  # explicit cleanup only; does not reset deleting/failed
                assert row['status'] != 'deleted'
        assert local.learner.steps == 0 and not local._saved
        output = {'case': case, 'events': events, 'registry_failed': registry.failed,
            'status': row['status'], 'retirement_id': owner.retirement_id,
            'admission_sha256': digest(actual), 'run_id': actual['run_id'],
            'operation_id': actual['operation_id'], 'invocation_id': actual['invocation_id'],
            'optimizer_steps': local.learner.steps, 'source_issued': False, 'prepared_issued': False}
        (control.root / 'retention-control-result.json').write_text(json.dumps(output, sort_keys=True))
    finally:
        if case != 'close':
            owner.close()
            pool.close()


@pytest.mark.parametrize('present', ['artifact', 'stage', 'symlink-stage', 'source-replaced'])
def test_cold_absence_checks_exact_held_directories_and_both_pathnames(tmp_path, present):
    import os

    from gf_training.rust_player_retention import _audit_deleted_absence
    root = tmp_path / 'parent'
    source = root / 'artifacts' / '000004'
    source.mkdir(parents=True)
    row = intent()
    row['parent_identity'] = [getattr(source.parent.stat(), key) for key in ('st_dev', 'st_ino')]
    row['source_directory_identity'] = [getattr(source.stat(), key) for key in ('st_dev', 'st_ino')]
    _audit_deleted_absence(root, 4, row['file'], row)
    if present == 'artifact':
        (source / row['file']).write_bytes(b'replaced')
    elif present == 'stage':
        (source.parent / row['staging_basename']).mkdir()
    elif present == 'symlink-stage':
        os.symlink('/nonexistent', source.parent / row['staging_basename'])
    else:
        source.rename(source.with_name('old-source'))
        source.mkdir()
    with pytest.raises(ContractError):
        _audit_deleted_absence(root, 4, row['file'], row)


def _artifact_inventory_data(tmp_path, *, hpo=False, deleted_name='weights-000.npz', deleted_generation=1, count=3):
    """Synthetic immutable data inventory, never a live registry/retirement cap."""
    import hashlib
    import json
    import time
    from types import SimpleNamespace

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_lineage import record_fields
    root = tmp_path / 'parent'
    registry = {'generations': {}, 'evaluation_references': {}, 'service_references': {}}
    deleted_intent = None
    for generation in range(count):
        directory = root / 'artifacts' / f'{generation:06}'
        directory.mkdir(parents=True)
        files = {}
        for name in ('inference.json', 'weights-000.npz', 'resume.json'):
            path = directory / name
            path.write_bytes(f'fixture-{generation}-{name}'.encode())
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict.fromkeys(record_fields('generation', hpo=hpo))
        record.update(schema_version=6 if hpo else 4, run_id='run', generation=generation, artifact_files=files,
            inference_sha256=files['inference.json'], snapshot_identity='s' + str(generation),
            parent_controls={'snapshot_pool': [0, count - 1]})
        if hpo:
            record.update(origin='producer-owned-v1', encoder='canonical-python', used_decisions=generation * 100)
        (root / 'generations').mkdir(exist_ok=True)
        (root / 'generations' / f'{generation:06}.json').write_text(json.dumps(record))
        rows = {name: {'sha256': sha, 'status': 'retained', 'deletion_intent': None, 'deletion_receipt': None} for name, sha in files.items()}
        registry['generations'][str(generation)] = {'inference_sha256': files['inference.json'],
            'snapshot_identity': record['snapshot_identity'], 'files': rows}
        if generation == deleted_generation:
            name = deleted_name
            row = intent() | {'generation': generation, 'file': name, 'sha256': files[name]}
            row['original_identity'] = [getattr((directory / name).stat(), key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')]
            row['parent_identity'] = [getattr(directory.parent.stat(), key) for key in ('st_dev', 'st_ino')]
            row['source_directory_identity'] = [getattr(directory.stat(), key) for key in ('st_dev', 'st_ino')]
            (directory / name).unlink()  # fixture DATA setup, not an authorized deletion transaction
            stem = 'retirement/artifacts/' + row['retirement_id']
            (root / stem).mkdir(parents=True)
            payload = json.dumps(row).encode()
            sha = hashlib.sha256(payload).hexdigest()
            (root / stem / 'intent.json').write_bytes(payload)
            receipt = {key: value for key, value in row.items() if key not in ('staging_basename', 'parent_identity', 'source_directory_identity', 'stage_directory_identity')}
            receipt.update(kind='rust-player-artifact-retirement-receipt-v1', intent_sha256=sha, completed_unix=1.)
            payload = json.dumps(receipt).encode()
            (root / stem / 'receipt.json').write_bytes(payload)
            rows[name].update(status='deleted', deletion_intent={'path': stem + '/intent.json', 'sha256': sha},
                deletion_receipt={'path': stem + '/receipt.json', 'sha256': hashlib.sha256(payload).hexdigest()})
            deleted_intent = row
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    return SimpleNamespace(operation={'parent_directory': str(root)}), {'run_id': 'run', 'updates': count - 1,
        'runtime_binding_sha256': 'b' * 64}, registry, pool, deleted_intent


@pytest.mark.parametrize('mutation', [None, 'deleting', 'missing-receipt', 'foreign-path', 'stage-present', 'retained-missing'])
def test_cold_inventory_audits_retained_and_exact_retired_files_without_mutating_data(tmp_path, monkeypatch, mutation):
    import traceback
    from pathlib import Path

    from gf_training import rust_player_retention as retention
    control, state, registry, pool, original_intent = _artifact_inventory_data(tmp_path)
    # Independently test the filesystem/record reader here. Historical binding
    # is separately exercised below; this DATA stub grants no live capability.
    histories = []
    monkeypatch.setattr(retention, '_retirement_history', lambda c, s, p, i, *a, **k: histories.append(i['invocation_id']))
    row = registry['generations']['1']['files']['weights-000.npz']
    root = Path(control.operation['parent_directory'])
    if mutation == 'deleting': row['status'] = 'deleting'
    elif mutation == 'missing-receipt': row['deletion_receipt'] = None
    elif mutation == 'foreign-path': row['deletion_receipt']['path'] = '../foreign.json'
    elif mutation == 'stage-present': (root / 'artifacts' / original_intent['staging_basename']).mkdir()
    elif mutation == 'retained-missing': (root / 'artifacts/000000/resume.json').unlink()
    loans = []
    try:
        if mutation is None:
            retention.audit_artifact_files(control, state, {}, registry, pool, loans, pool.deadline, hpo=False)
            assert histories == [original_intent['invocation_id']]
            assert not loans
        else:
            with pytest.raises((ContractError, OSError)) as failed:
                retention.audit_artifact_files(control, state, {}, registry, pool, loans, pool.deadline, hpo=False)
            cleanup = vars(failed.value).get('cleanup_owner')
            traceback.clear_frames(failed.value.__traceback__)
            if cleanup is not None:
                cleanup.close()
    finally:
        for loan in loans: loan.close()
        pool.close()


@pytest.mark.parametrize('mutation', [None, 'foreign-operation', 'failed-frozen-state', 'missing-history',
    'wrong-admission-sha', 'noncooperative-exit', 'completion-unknown', 'completion-schema',
    'completion-schema-bool', 'completion-eligible', 'completion-eligible-type', 'completion-update',
    'completion-update-bool', 'completion-metadata', 'completion-audit-path', 'completion-audit-sha',
    *['completion-missing:' + key for key in ('schema_version', 'kind', 'supervision_id', 'invocation_id',
        'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256', 'operation_sha256',
        'prepared_sha256', 'exit_sha256', 'state_sha256', 'generation_sha256', 'source_manifest_sha256',
        'state', 'committed_update', 'audit_path', 'audit_sha256', 'eligible', 'parent_metadata_validation')],
    *['completion-foreign:' + key for key in ('supervision_id', 'invocation_id', 'operation_id', 'run_id',
        'config_sha256', 'runtime_binding_sha256', 'source_manifest_sha256')],
    'audit-missing', 'audit-unknown', 'audit-schema', 'audit-identity', 'audit-prepared', 'audit-generation',
    'audit-saved-pair', 'audit-updates', 'audit-optimizer', 'audit-full-content'])
def test_deleted_history_binds_retirement_invocation_not_current_reader(tmp_path, mutation):
    """Immutable historical DATA records, with no reconstructed live capability."""
    import hashlib
    import json
    import time
    import traceback
    from pathlib import Path
    from types import SimpleNamespace

    from gf_common.domain import canonical, digest
    from gf_common.rust_player import deadline_bits
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_retention import _retirement_history
    from gf_training.rust_player_supervision import write_control_record
    from tests.test_rust_player_supervisor import _new_namespace_records
    prior = json.loads((Path(__file__).parent / 'fixtures/rust_player_run5_common_metadata.json').read_bytes())
    protocol = tmp_path / 'protocol'
    protocol.mkdir()
    parent = tmp_path / 'parent'
    parent.mkdir()
    old, current = '2' * 32, '4' * 32
    admit = _new_namespace_records()[0]['host_admission']
    admit.update(operation_id='retiring-operation', run_id=prior['run_id'], context=None, global_ticket=None,
        producer=prior['identity']['player_runtime']['producer'], config_sha256=prior['config_sha256'],
        runtime_binding_sha256=prior['runtime_binding_sha256'])
    operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2',
        'operation_id': admit['operation_id'], 'run_id': prior['run_id'], 'mode': 'ordinary',
        'operation_type': 'ordinary-train', 'config_path': str(parent / 'config.json'),
        'config_sha256': prior['config_sha256'], 'runtime_binding_sha256': prior['runtime_binding_sha256'],
        'deadlines': admit['deadlines'], 'parent_directory': str(parent), 'expected_context': None,
        'execution_request': {'mode': 'fresh', 'pause_after_update': None}}
    operation_path, admission_path = protocol / 'operation.json', protocol / ('admission-' + old + '.json')
    operation_sha = write_control_record(operation_path, operation)
    admission_sha = write_control_record(admission_path, admit)
    entry = {'operation_path': str(operation_path), 'operation_sha256': operation_sha,
        'execution_request': operation['execution_request'], 'admission_path': str(admission_path),
        'admission_sha256': admission_sha, 'mode': 'fresh', 'elapsed_seconds': 1., 'host_health': {}}
    prior.update(status='paused', invocations={old: entry})
    state_bytes = (canonical(prior) + '\n').encode()
    (parent / ('state-' + old + '.json')).write_bytes(state_bytes)
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    prepared = {'schema_version': 1, 'kind': 'rust-player-prepared-v1',
        **{key: admit[key] for key in ('supervision_id', 'invocation_id', 'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256')},
        'admission_sha256': admission_sha, 'state': 'paused', 'committed_update': prior['updates'],
        'generation': prior['updates'], 'state_sha256': state_sha, 'generation_sha256': prior['generation_sha256'],
        'artifacts_sha256': digest({'inference.json': 'a' * 64, 'resume.json': 'b' * 64}),
        'source_manifest_sha256': 'b' * 64,
        'close_ack_path': str(protocol / 'unused-close.json'), 'close_ack_sha256': 'c' * 64}
    prepared_sha = write_control_record(protocol / ('prepared-' + old + '.json'), prepared)
    exit_record = {'schema_version': 1, 'kind': 'rust-player-exit-v1',
        **{key: admit[key] for key in ('supervision_id', 'invocation_id', 'nonce')},
        'admission_sha256': admission_sha, 'prepared_sha256': prepared_sha, 'host_original': admit['host'],
        'wait': {'exit_code': 0, 'signal': None, 'core_dumped': False}, 'descendants': [], 'signals': [],
        'descendants_reaped': True, 'cooperative': True, 'finished_unix_ns': time.time_ns()}
    exit_path = protocol / ('exit-' + old + '.json')
    exit_sha = write_control_record(exit_path, exit_record)
    audit = {'schema_version': 1, 'kind': 'rust-player-cold-audit-v1', 'run_id': prior['run_id'],
        'invocation_id': old, 'operation_sha256': operation_sha, 'prepared_sha256': prepared_sha,
        'exit_sha256': exit_sha, 'state_sha256': state_sha, 'generation_sha256': prior['generation_sha256'],
        'updates': [{'generation': g, 'manifest_sha256': prepared['source_manifest_sha256'],
            'numeric_sha256': 'a' * 64, 'learner_decisions': 1, 'episode_digests': ['a' * 64],
            'source_set_sha256': 'b' * 64} for g in range(1, prior['updates'] + 1)],
        'saved_pair': {'generation': prior['updates'], 'inference_sha256': 'a' * 64, 'resume_sha256': 'b' * 64,
            'artifact_files': {'inference.json': 'a' * 64, 'resume.json': 'b' * 64},
            'optimizer_steps': prior['optimizer_steps'], 'full_content_validated': True},
        'runtime_binding_sha256': prior['runtime_binding_sha256']}
    audit_path = protocol / ('cold-audit-' + old + '.json')
    audit_sha = write_control_record(audit_path, audit)
    completion = {'schema_version': 1, 'kind': 'rust-player-training-completion-v1',
        **{key: admit[key] for key in ('supervision_id', 'invocation_id', 'operation_id',
            'run_id', 'config_sha256', 'runtime_binding_sha256')},
        'operation_sha256': operation_sha, 'prepared_sha256': prepared_sha,
        'exit_sha256': exit_sha, 'state_sha256': state_sha, 'generation_sha256': prior['generation_sha256'],
        'source_manifest_sha256': prepared['source_manifest_sha256'], 'state': 'paused',
        'committed_update': prior['updates'], 'audit_path': str(protocol / ('cold-audit-' + old + '.json')),
        'audit_sha256': audit_sha, 'eligible': False, 'parent_metadata_validation': None}
    completion_path = protocol / ('completion-' + old + '.json')
    write_control_record(completion_path, completion)
    state = deepcopy(prior)
    state['invocations'][current] = {'operation_path': 'current reader is a different operation'}
    row = intent() | {'invocation_id': old, 'operation_id': admit['operation_id'],
        'runtime_binding_sha256': prior['runtime_binding_sha256']}
    if mutation == 'foreign-operation':
        completion_path.write_bytes(canonical(completion | {'operation_sha256': '0' * 64}).encode())
    elif mutation == 'failed-frozen-state':
        (parent / ('state-' + old + '.json')).write_text(canonical(prior | {'status': 'failed'}))
    elif mutation == 'missing-history':
        del state['invocations'][old]
    elif mutation == 'wrong-admission-sha':
        state['invocations'][old]['admission_sha256'] = '0' * 64
    elif mutation == 'noncooperative-exit':
        exit_path.write_text(canonical(exit_record | {'cooperative': False}))
    elif mutation is not None and mutation.startswith('completion-'):
        changed = deepcopy(completion)
        if mutation.startswith('completion-missing:'): del changed[mutation.split(':', 1)[1]]
        elif mutation.startswith('completion-foreign:'):
            key = mutation.split(':', 1)[1]
            changed[key] = '0' * (32 if key in ('supervision_id', 'invocation_id') else 64) if key.endswith('sha256') or key in ('supervision_id', 'invocation_id') else 'foreign'
        elif mutation == 'completion-unknown': changed['admission_sha256'] = admission_sha
        elif mutation == 'completion-schema': changed['schema_version'] = 2
        elif mutation == 'completion-schema-bool': changed['schema_version'] = True
        elif mutation == 'completion-eligible': changed['eligible'] = True
        elif mutation == 'completion-eligible-type': changed['eligible'] = 0
        elif mutation == 'completion-update': changed['committed_update'] += 1
        elif mutation == 'completion-update-bool': changed['committed_update'] = True
        elif mutation == 'completion-metadata': changed['parent_metadata_validation'] = 'claimed'
        elif mutation == 'completion-audit-path': changed['audit_path'] = str(protocol / 'foreign-audit.json')
        elif mutation == 'completion-audit-sha': changed['audit_sha256'] = '0' * 64
        completion_path.write_text(canonical(changed))
    elif mutation is not None and mutation.startswith('audit-'):
        changed = deepcopy(audit)
        if mutation == 'audit-missing': del changed['runtime_binding_sha256']
        elif mutation == 'audit-unknown': changed['extra'] = None
        elif mutation == 'audit-schema': changed['schema_version'] = 2
        elif mutation == 'audit-identity': changed['invocation_id'] = current
        elif mutation == 'audit-prepared': changed['prepared_sha256'] = '0' * 64
        elif mutation == 'audit-generation': changed['generation_sha256'] = '0' * 64
        elif mutation == 'audit-saved-pair': changed['saved_pair']['artifact_files']['inference.json'] = '0' * 64
        elif mutation == 'audit-updates': changed['updates'] = []
        elif mutation == 'audit-optimizer': changed['saved_pair']['optimizer_steps'] += 1
        elif mutation == 'audit-full-content': changed['saved_pair']['full_content_validated'] = False
        data = canonical(changed).encode()
        audit_path.write_bytes(data)
        completion_path.write_text(canonical(completion | {'audit_sha256': hashlib.sha256(data).hexdigest()}))
    control = SimpleNamespace(root=protocol, operation={'parent_directory': str(parent)},
        launch={'session_deadlines': {'finish_bits': deadline_bits(time.monotonic() + 60)}})
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    loans = []
    try:
        if mutation is None:
            _retirement_history(control, state, {'invocation_id': current}, row, pool, loans, pool.deadline, hpo=False)
        else:
            with pytest.raises((ContractError, OSError)) as failed:
                _retirement_history(control, state, {'invocation_id': current}, row, pool, loans, pool.deadline, hpo=False)
            traceback.clear_frames(failed.value.__traceback__)
    finally:
        for loan in loans: loan.close()
        pool.close()


@pytest.mark.parametrize('case,hpo,target,generation,count,protected', [
    ('ordinary-unprotected', False, 'weights-000.npz', 1, 3, False),
    ('ordinary-unprotected-resume', False, 'resume.json', 1, 3, False),
    ('ordinary-inference-manifest', False, 'inference.json', 1, 3, True),
    ('ordinary-evaluation', False, 'weights-000.npz', 1, 3, True),
    ('ordinary-g0', False, 'weights-000.npz', 0, 3, True),
    ('ordinary-tenth', False, 'resume.json', 10, 12, True),
    ('ordinary-current', False, 'resume.json', 2, 3, True),
    ('hpo-unprotected', True, 'weights-000.npz', 1, 3, False),
    ('hpo-resume', True, 'resume.json', 1, 3, True),
    ('hpo-periodic', True, 'weights-000.npz', 1, 3, True),
    ('hpo-inference-latest', True, 'weights-000.npz', 1, 3, True),
    ('hpo-evaluation', True, 'weights-000.npz', 1, 3, True)])
def test_cold_requires_applicable_pure_protection_outside_latest_pool(tmp_path, monkeypatch, case, hpo, target, generation, count, protected):
    """Synthetic record/filesystem DATA; no retirement or Source capability."""
    import traceback

    from gf_training import rust_player_retention as retention
    control, state, registry, pool, _ = _artifact_inventory_data(tmp_path, hpo=hpo,
        deleted_name=target, deleted_generation=generation, count=count)
    if 'evaluation' in case:
        registry['evaluation_references']['evaluation'] = {'generation': generation,
            'run_id': state['run_id'], 'inference_sha256': registry['generations'][str(generation)]['inference_sha256'],
            'generation_sha256': 'a' * 64, 'token': 'evaluation'}
    operating = {'resume_latest': 2 if case == 'hpo-resume' else 1,
        'inference_latest': 2 if case == 'hpo-inference-latest' else 1,
        'inference_periodic_decisions': 100 if case == 'hpo-periodic' else 10000}
    # Isolate protection from separately checked historical DATA validation.
    monkeypatch.setattr(retention, '_retirement_history', lambda *a, **k: None)
    loans = []
    try:
        options = {'operating': operating} if hpo else {}
        if protected:
            with pytest.raises(ContractError, match='required|protected') as failed:
                retention.audit_artifact_files(control, state, {}, registry, pool, loans, pool.deadline, hpo=hpo, **options)
            traceback.clear_frames(failed.value.__traceback__)
        else:
            retention.audit_artifact_files(control, state, {}, registry, pool, loans, pool.deadline, hpo=hpo, **options)
    finally:
        for loan in loans: loan.close()
        pool.close()
