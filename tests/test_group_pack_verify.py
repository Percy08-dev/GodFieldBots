from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.test_group_pack_storage import submit, writer


def test_full_audit_reads_payloads_and_never_publishes_pending(tmp_path: Path) -> None:
    from gf_training.group_pack_verify import verify_storage
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal(publication='pending')
    before = {p.name: p.read_bytes() for p in locator.path.parent.iterdir()}
    result = verify_storage(locator=locator.path)
    assert result['passed'] and result['cleanup_confirmed'] and result['complete']
    assert result['counts']['packs'] == 1
    assert result['scope'] == 'diagnostic-storage-only'
    assert before == {p.name: p.read_bytes() for p in locator.path.parent.iterdir()}


def test_unrequested_pack_corruption_is_deferred_but_full_audit_fails(tmp_path: Path) -> None:
    import time

    from gf_training.group_pack_source import PackedSource
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    w = writer(tmp_path)
    submit(w, 'a')
    submit(w, 'b')
    locator = w.seal()
    with sqlite3.connect(locator.path.parent/'collection.sqlite3') as db:
        value = db.execute("SELECT pack FROM groups WHERE group_id='b'").fetchone()[0]
        db.execute("UPDATE groups SET pack=? WHERE group_id='b'", (b'!' + value[1:],))
    source = PackedSource(locator.path, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2)
    with source.borrow_group('a') as loan:
        assert loan.read('a/e0') == b'\xff\x00'
    with pytest.raises(ValueError, match='BLOB'), source.borrow_group('b'):
        pass
    source.close()
    with pytest.raises(StorageAuditError) as failure:
        verify_storage(locator=locator.path)
    assert not failure.value.result['passed']
    assert failure.value.result['cleanup_confirmed']


def test_cli_audit_json_and_explicit_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from gf_cli import main
    path = tmp_path/'not-a-locator'
    path.write_text('{}')
    assert main(['verify-storage', '--locator', str(path)]) != 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema_version'] == 1 and not result['passed']


@pytest.mark.parametrize('conflict', [False, True])
def test_same_inode_scans_once_but_checks_each_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: bool) -> None:
    import os

    from gf_common.group_pack import canonical_json, parse_json
    from gf_training.group_pack_source import OriginalFile
    from gf_training.group_pack_verify import _Audit
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    alias = tmp_path/'alias'
    alias.mkdir()
    os.link(locator.path.parent/'collection.sqlite3', alias/'collection.sqlite3')
    descriptor = parse_json(locator.path.read_bytes())
    if conflict:
        descriptor['database_sha256'] = '0'*64
    second = alias/locator.path.name
    second.write_bytes(canonical_json(descriptor))
    hashes = []
    original_hash = OriginalFile.hash
    def hash_file(file: OriginalFile) -> str:
        if file.path.name == 'collection.sqlite3':
            hashes.append(file.path)
        return original_hash(file)
    monkeypatch.setattr(OriginalFile, 'hash', hash_file)
    audit = _Audit(locator.path, 'locator', 512*1024**2, 30)
    try:
        audit.reference(locator.path)
        if conflict:
            with pytest.raises(ValueError, match='conflicting'):
                audit.reference(second)
        else:
            audit.reference(second)
            assert audit.result['counts']['references'] == 2
            assert audit.result['counts']['unique_databases'] == 1
        assert len(hashes) == 1
    finally:
        audit.close()


@pytest.mark.parametrize('fault', ['cancel', 'timeout', 'close'])
def test_audit_failure_and_cleanup_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from gf_training.group_pack_source import OriginalFile, PackedOriginals
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    if fault == 'close':
        close = PackedOriginals.close
        def uncertain(owner: PackedOriginals) -> None:
            close(owner)
            raise RuntimeError('uncertain actual close')
        monkeypatch.setattr(PackedOriginals, 'close', uncertain)
    else:
        hash_file = OriginalFile.hash
        def fail(owner: OriginalFile) -> str:
            if owner.path.name == 'collection.sqlite3':
                if fault == 'cancel':
                    raise KeyboardInterrupt()
                raise TimeoutError('audit deadline')
            return hash_file(owner)
        monkeypatch.setattr(OriginalFile, 'hash', fail)
    with pytest.raises(StorageAuditError) as failed:
        verify_storage(locator=locator.path)
    result = failed.value.result
    assert not result['passed'] and not result['complete']
    assert result['cleanup_confirmed'] is (fault != 'close')
    assert failed.value.cleanup_owner is not None


@pytest.mark.parametrize('field', ['memory_budget_mib', 'wall_seconds'])
@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True])
def test_limits_are_strictly_positive_finite(field: str, value: float) -> None:
    from gf_training.group_pack_verify import verify_storage
    with pytest.raises(ValueError):
        verify_storage(locator=Path('/not-read'), **{field: value})


def diagnostic_dataset4(root: Path, *, prune_middle: bool = False) -> Path:
    """Unadmitted synthetic controls plus actual storage; never a learning Source."""
    from gf_common import card_instance_game_tree as game
    from gf_common.domain import digest as object_digest
    from gf_common.group_pack import canonical_json, digest
    from tests.test_group_pack_dataset_contract import packed_dto
    from tests.test_group_pack_game_source import owner
    value = packed_dto()
    if prune_middle:
        from copy import deepcopy
        for parent in value['parents']:
            parent['final_generation'] = 2
            parent['generations'].append({**parent['generations'][-1], 'generation': 2})
        expanded = []
        for row in value['collections']:
            if row['split'] == 'train':
                expanded.extend((row, {**deepcopy(row), 'generation': 1}))
            else:
                row['generation'] = 2
                expanded.append(row)
        value['collections'] = expanded
        value['counts'] = {split: {key: sum(row['counts'][key] for row in expanded if row['split'] == split)
            for key in game.COUNT_FIELDS} for split in ('train', 'validation')}
    def write(path: Path, data: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json(data)
        path.write_bytes(raw)
        return digest(raw)
    for parent in value['parents']:
        directory = root/f"parent-{parent['parent_index']}"
        parent['directory'] = str(directory)
        receipts = {}
        for row in parent['generations']:
            number = row['generation']
            artifacts = directory/'artifacts'/f'{number:06}'
            files = {name: write(artifacts/name, {}) for name in ('inference.json', 'resume.json')}
            if prune_middle:
                for name in ('weights-000000.npz', 'weights-000000.json', 'resume-000000.npz', 'resume-000000.json'):
                    files[name] = write(artifacts/name, {'UNADMITTED': True})
                files['inference.json'] = write(artifacts/'inference.json', {'weights': {'files': {
                    k: v for k, v in files.items() if k.startswith('weights-')}}})
            row.update(inference_sha256=files['inference.json'], resume_sha256=files['resume.json'],
                update_sha256=None if number == 0 else write(directory/'updates'/f'{number:06}.json', {}),
                artifact_files_sha256=object_digest(files))
            record = {k: row[k] for k in ('generation', 'inference_sha256', 'resume_sha256', 'update_sha256')}
            record['artifact_files'] = files
            row['generation_sha256'] = write(directory/'generations'/f'{number:06}.json', record)
            receipt = {'status': 'retained', 'inference_sha256': files['inference.json'], 'original_files': files,
                'deletion_intent': None, 'deletion_receipt': None}
            if prune_middle and number == 1:
                targets = {k: v for k, v in files.items() if k != 'inference.json'}
                receipt.update(status='deleted', deletion_intent={'files': targets, 'staging_basename': 'prune-'+'1'*32},
                    deletion_receipt={'files': targets})
                for name in targets:
                    (artifacts/name).unlink()
            receipts[str(number)] = receipt
        parent['final_generation_sha256'] = parent['generations'][-1]['generation_sha256']
        parent['final_inference_sha256'] = parent['generations'][-1]['inference_sha256']
        parent['config_inputs_sha256'] = object_digest({})
        state = {'schema_version': 4, 'run_id': parent['run_id'], 'seed': parent['seed'],
            'updates': parent['final_generation'], 'generation_sha256': parent['final_generation_sha256'],
            'resume_identity': parent['resume_identity'], 'config_sha256': parent['config_sha256'],
            'config_inputs': {}, 'formal_declaration': {'sha256': parent['formal_declaration_sha256']},
            'storage_binding': value['storage_binding'], 'diagnostic_only': 'UNADMITTED'}
        parent['run_sha256'] = write(directory/'run.json', state)
        parent['registry_sha256'] = write(directory/'artifact-registry.json', {'schema_version': 3, 'generations': receipts})
    for index, row in enumerate(value['collections']):
        parent = value['parents'][row['parent_index']]
        envelope = root/f'envelope-{index}'
        envelope.mkdir()
        row['directory'] = str(envelope)
        manifest = {'collection_role': row['collection_role'], 'schedule': [], 'policy_bindings': {},
            'linkage': {'parent_run_id': parent['run_id'], 'parent_run_path': parent['directory'],
                'final_generation' if row['split'] == 'validation' else 'collection_generation': row['generation']}}
        raw = canonical_json(manifest)
        w = writer(root/f'physical-{index}')
        submit(w, 'a')
        w.store_control('manifest.json', raw)
        locator = w.seal()
        write(envelope/'collection.locator.json', {'format': 'sqlite-group-pack-locator-v1', 'schema_version': 1,
            'descriptor': str(locator.path), 'descriptor_sha256': digest(locator.path.read_bytes())})
        original = owner(envelope)
        try:
            row['storage'], row['custody'] = original.storage, original.custody
        finally:
            original.close()
        # A historical value binds its own original fingerprint and saved aggregate.
        row['custody']['recipe'] = 'packed-original-custody-v1'
        row['custody']['fingerprint'] = object_digest({k: v for k, v in row['custody'].items() if k != 'fingerprint'})
        row.update(manifest_sha256=digest(raw), schedule_sha256=object_digest([]), bindings_sha256=object_digest({}))
        if row['split'] == 'validation':
            row['original_correspondence'] = game.final_correspondence(row['manifest_sha256'], ['e'*64]*64, packed=True)
        else:
            numeric = root/f'numeric-{index}'
            sha = write(numeric/'rollout.json', {})
            row['numeric'] = {'directory': str(numeric), 'sha256': sha, 'files': {'rollout.json': sha}}
    value['source_set_sha256'] = game.source_identity(value['parents'], value['collections'])
    value['shared_generation'] = game.shared_generation(value['parents'], value['collections'])
    value['content_sha256'] = object_digest({k: v for k, v in value.items() if k != 'content_sha256'})
    path = root/'dataset.json'
    write(path, value)
    return path


def test_dataset4_audit_follows_actual_storage_and_every_parent_reference(tmp_path: Path) -> None:
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    path = diagnostic_dataset4(tmp_path)
    result = verify_storage(dataset=path)
    assert result['passed'] and result['counts']['references'] == 6
    assert result['counts']['unique_databases'] == 6
    assert result['scope'] == 'diagnostic-storage-only'
    (tmp_path/'parent-1/artifacts/000000/resume.json').unlink()
    with pytest.raises(StorageAuditError) as failed:
        verify_storage(dataset=path)
    assert not failed.value.result['passed'] and failed.value.result['cleanup_confirmed']


def test_earlier_audited_original_is_rechecked_at_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from gf_training import group_pack_verify as module
    path = diagnostic_dataset4(tmp_path)
    checked = module._Audit.check_completed
    def changed(audit: module._Audit) -> None:
        first = next(file for file in audit.completed_files if file.path.name == 'collection.sqlite3')
        os.utime(first.path, ns=(first.identity.st_atime_ns, first.identity.st_mtime_ns+1))
        checked(audit)
    monkeypatch.setattr(module._Audit, 'check_completed', changed)
    with pytest.raises(module.StorageAuditError) as failed:
        module.verify_storage(dataset=path)
    assert not failed.value.result['passed'] and failed.value.result['cleanup_confirmed']
    assert 'identity changed' in str(failed.value)


def test_deadline_observed_after_actual_close_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_training import group_pack_verify as module
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    close = module._Audit.close
    def late(audit: module._Audit) -> None:
        close(audit)
        audit.deadline = 0
    monkeypatch.setattr(module._Audit, 'close', late)
    with pytest.raises(module.StorageAuditError) as failed:
        module.verify_storage(locator=locator.path)
    assert isinstance(failed.value.error, TimeoutError)
    assert failed.value.result['cleanup_confirmed'] and not failed.value.result['passed']


def test_insufficient_positive_budget_reports_incomplete_scope(tmp_path: Path) -> None:
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    with pytest.raises(StorageAuditError) as failed:
        verify_storage(locator=tmp_path/'unread', memory_budget_mib=.001)
    assert failed.value.result['schema_version'] == 1
    assert not failed.value.result['complete'] and failed.value.result['cleanup_confirmed']


def test_dataset4_audit_accepts_only_confirmed_optional_generation_deletions(tmp_path: Path) -> None:
    import json

    from gf_common import card_instance_game_tree as game
    from gf_common.domain import digest as object_digest
    from gf_common.group_pack import canonical_json, digest
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    path = diagnostic_dataset4(tmp_path, prune_middle=True)
    assert verify_storage(dataset=path)['passed']
    value = json.loads(path.read_bytes())
    parent = value['parents'][0]
    registry_path = Path(parent['directory'])/'artifact-registry.json'
    registry = json.loads(registry_path.read_bytes())
    registry['generations']['1']['deletion_receipt']['files']['resume.json'] = 'f'*64
    raw = canonical_json(registry)
    registry_path.write_bytes(raw)
    parent['registry_sha256'] = digest(raw)
    value['source_set_sha256'] = game.source_identity(value['parents'], value['collections'])
    value['shared_generation'] = game.shared_generation(value['parents'], value['collections'])
    value['content_sha256'] = object_digest({k: v for k, v in value.items() if k != 'content_sha256'})
    path.write_bytes(canonical_json(value))
    with pytest.raises(StorageAuditError, match='deletion'):
        verify_storage(dataset=path)


@pytest.mark.parametrize('stage', ['dataset', 'parent'])
def test_control_constructor_failure_keeps_actual_uncertain_owner_and_charge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    from gf_training.group_pack_source import OriginalFile
    from gf_training.group_pack_verify import StorageAuditError, verify_storage
    path = diagnostic_dataset4(tmp_path) if stage == 'parent' else tmp_path/'dataset.json'
    if stage == 'dataset':
        path.write_bytes(b'{}')
    target = path if stage == 'dataset' else tmp_path/'parent-0/run.json'
    real_hash, real_close = OriginalFile.hash, OriginalFile.close
    actuals = []
    def hash_file(file: OriginalFile) -> str:
        if file.path == target:
            actuals.append(file)
            raise ValueError('control constructor hash fault')
        return real_hash(file)
    def close(file: OriginalFile) -> None:
        if file.path == target:
            raise RuntimeError('actual control close uncertain')
        real_close(file)
    monkeypatch.setattr(OriginalFile, 'hash', hash_file)
    monkeypatch.setattr(OriginalFile, 'close', close)
    with pytest.raises(StorageAuditError) as failure:
        verify_storage(dataset=path)
    audit = failure.value.cleanup_owner
    assert actuals and actuals[0].fd >= 0
    try:
        assert not failure.value.result['cleanup_confirmed']
        assert actuals[0] in audit.owners or actuals[0] in audit._cleanup_children
        assert audit.budget.used > 65536
        assert len(audit.errors) >= 2
    finally:
        monkeypatch.undo()
        audit.close()
    assert actuals[0].fd == -1
