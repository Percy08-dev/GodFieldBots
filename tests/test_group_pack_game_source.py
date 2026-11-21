from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_common.domain import ContractError, digest
from gf_common.group_pack import MIB, canonical_json
from gf_common.preprocessing import PreparationConfig
from gf_training.group_pack_source import CollectionLocator
from tests.test_group_pack_storage import submit, writer


def packed(root: Path) -> tuple[Path, CollectionLocator]:
    w = writer(root)
    submit(w, 'first')
    submit(w, 'second')
    w.store_control('manifest.json', b'{"hello":"original"}\n')
    w.store_control('report.json', b'{}\n')
    locator = w.seal()
    envelope = root / 'envelope'
    envelope.mkdir()
    (envelope / 'collection.locator.json').write_bytes(canonical_json({
        'format': 'sqlite-group-pack-locator-v1', 'schema_version': 1,
        'descriptor': str(locator.path), 'descriptor_sha256': __import__('hashlib').sha256(locator.path.read_bytes()).hexdigest()}))
    return envelope, locator


def owner(root: Path, pool: object = None):
    from gf_common.group_pack_game import storage_binding
    from gf_training.group_pack_game_source import PackedSourceOwner
    return PackedSourceOwner(root, PreparationConfig(), time.monotonic() + 30,
        storage_binding=storage_binding(), max_owned_bytes=512 * MIB, pool=pool)


def test_checked_binding_is_closed_and_detached() -> None:
    from gf_common.group_pack_game import checked_storage_binding, storage_binding
    binding = storage_binding()
    assert checked_storage_binding(binding) == binding
    binding['provider'] = 'arbitrary-reader'
    with pytest.raises(ContractError):
        checked_storage_binding(binding)
    with pytest.raises(ContractError):
        checked_storage_binding({**storage_binding(), 'authority': True})


def test_exact_logical_bytes_stable_identity_and_persisted_custody(tmp_path: Path) -> None:
    root, _ = packed(tmp_path)
    first = owner(root)
    assert first.read(root / 'first/e0') == b'\xff\x00'
    assert first.read(root / 'manifest.json') == b'{"hello":"original"}\n'
    assert first.inventory('**/*') == ('first/e0', 'first/e1', 'manifest.json', 'report.json', 'second/e0', 'second/e1')
    files, storage, custody = first.files.copy(), first.storage, first.custody
    assert files['first/e0']['identity']['format'] == 'packed-member-v1'
    assert set(custody) == {'recipe', 'files', 'directories', 'fingerprint'}
    assert custody['fingerprint'] == digest({k: v for k, v in custody.items() if k != 'fingerprint'})
    assert {row['role'] for row in custody['files']} == {'locator', 'descriptor', 'database'}
    first.finalize()
    first.close()
    second = owner(root)
    second.read(root / 'first/e0')
    second.read(root / 'manifest.json')
    assert second.files == files
    assert second.storage == storage
    assert second.custody == custody
    second.close()


@pytest.mark.parametrize('role', ['locator', 'descriptor', 'database'])
def test_identical_inode_replacement_rejects_saved_custody(tmp_path: Path, role: str) -> None:
    root, locator = packed(tmp_path)
    first = owner(root)
    custody = first.custody
    first.close()
    path = {'locator': root / 'collection.locator.json', 'descriptor': locator.path,
        'database': locator.path.parent / 'collection.sqlite3'}[role]
    temporary = path.with_suffix('.replacement')
    shutil.copyfile(path, temporary)
    temporary.replace(path)
    second = owner(root)
    with pytest.raises(ContractError, match='custody'):
        second.require_custody(custody)
    second.close()


def test_shared_pool_one_connection_group_and_stale_live_loans(tmp_path: Path) -> None:
    from gf_training.group_pack_pool import PackedArtifactPool
    a, _ = packed(tmp_path / 'a')
    b, _ = packed(tmp_path / 'b')
    pool = PackedArtifactPool(max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    first, second = owner(a, pool), owner(b, pool)
    with first.borrow_group('first') as loan:
        assert loan.read('first/e0') == b'\xff\x00'
        with pytest.raises(RuntimeError, match='loan'):
            second.read(b / 'second/e0')
        with pytest.raises(RuntimeError, match='loan'):
            first.close()
    second.read(b / 'second/e0')
    assert first.artifacts._originals.db is None
    assert second.artifacts._originals.db is not None
    assert not first.artifacts._cache
    assert first.read(a / 'first/e1') == b'\xff\x01'
    assert not second.artifacts._cache and second.artifacts._originals.db is None
    with pytest.raises(RuntimeError):
        loan.read('first/e0')
    first.close()
    second.close()
    pool.close()
    assert pool.budget.used == 0


def test_bare_descriptor_cannot_be_original_game_root(tmp_path: Path) -> None:
    _, locator = packed(tmp_path)
    with pytest.raises((ContractError, FileNotFoundError)):
        owner(locator.path.parent)


def pack_original(directory: Path, destination: Path) -> None:
    from gf_common.group_pack import PackMember, encode_pack, unpack_raw
    from gf_training.group_pack_writer import GroupPackWriter
    from tests.test_group_pack_storage import settings
    w = GroupPackWriter(destination, 'original-fixture', settings=settings(destination.parent),
        deadline=time.monotonic() + 60)
    manifest = __import__('json').loads((directory / 'manifest.json').read_bytes())
    episode_rows = {row['episode_id']: row for row in manifest['episodes']}
    for schedule, health in zip(manifest['schedule'], manifest['groups'], strict=True):
        names = {row['episode_id'] for row in schedule['episodes']}
        members = tuple(PackMember('candidates' if p.suffix == '.columns' else 'journal' if p.suffix == '.jsonl' else 'diagnostic',
            p.relative_to(directory).as_posix(), p.read_bytes())
            for p in sorted(directory.rglob('*')) if p.is_file() and p.relative_to(directory).parts[0] in names)
        data = encode_pack(members)
        w.submit(schedule['episodes'][0]['episode_id'], data, raw_bytes=len(unpack_raw(data)), record=health,
            episodes=[{'episode_id': name, 'game_id': episode_rows[name]['game_id'],
                'record': {key: episode_rows[name][key] for key in ('end_kind', 'failure')}} for name in sorted(names)])
    for name in ('manifest.json', 'report.json'):
        w.store_control(name, (directory / name).read_bytes())
    locator = w.seal()
    shutil.rmtree(directory)
    directory.mkdir()
    (directory / 'collection.locator.json').write_bytes(canonical_json({
        'format': 'sqlite-group-pack-locator-v1', 'schema_version': 1,
        'descriptor': str(locator.path), 'descriptor_sha256': __import__('hashlib').sha256(locator.path.read_bytes()).hexdigest()}))


def test_packed_proof2_and_full_own_input_numeric_parity(tmp_path: Path) -> None:
    from gf_common.domain import canonical
    from gf_training.card_instance_source import recorded_plan
    from gf_training.lineage import run_lock
    from gf_training.structured_source import _inspect_component_collection, _SourceOwner
    from tests.card_instance_learning_support import initial_parent, original_collection
    from tests.test_card_instance_learning import ACTIONS
    parent = tmp_path / 'parent'
    with run_lock(parent, create=True):
        learner, state, records, settings = initial_parent(parent)
        plan, _, _ = recorded_plan(parent, state, records, 0, time.monotonic() + 60)
        original_collection(plan, state, learner)
        reference = _inspect_component_collection(_SourceOwner(plan.output, settings, time.monotonic() + 60), ACTIONS, 'train', _game=True)
        from gf_cli.config import GroupPackStorageConfig
        manifest = reference.manifest
        manifest['config']['storage'] = asdict(GroupPackStorageConfig(db_root=str(tmp_path / 'db'), ram_root=str(tmp_path / 'ram')))
        manifest['config_hash'] = digest(manifest['config'])
        (plan.output / 'manifest.json').write_text(canonical(manifest))
        pack_original(plan.output, tmp_path / 'external')
        original = owner(plan.output)
        actual = _inspect_component_collection(original, ACTIONS, 'train', _game=True)
        assert actual.manifest == manifest
        assert actual.episodes == reference.episodes
        from gf_training.structured_rollout import _write_rollout, game_rollout
        from tests.test_card_instance_learning import RULES
        seats = plan.linkage['learner_seats']
        for label, collection in (('reference', reference), ('packed', actual)):
            selected = tuple(row.records for row in collection.episodes if seats[row.episode_id] == row.viewer)
            rollout = game_rollout(selected, learner.model, actions=ACTIONS, rules=RULES)
            _write_rollout(tmp_path / label, rollout)
        from gf_models.arrays import sha256
        files = lambda folder: {p.name: sha256(p) for p in folder.iterdir()}
        assert files(tmp_path / 'reference') == files(tmp_path / 'packed')
        assert actual.proof['schema_version'] == 2
        from gf_common.group_pack_game import checked_packed_proof
        checked_packed_proof(actual.proof)
        with pytest.raises(ContractError):
            checked_packed_proof(actual.proof | {'service_usage':{}})
        assert actual.proof['kind'] == 'card-instance-packed-original-source-proof'
        assert actual.proof['source_set_sha256'] == digest(actual.proof['files'])
        assert actual.proof['custody'] == original.custody
        assert set(actual.proof) == set(reference.proof) | {'storage', 'custody'}
        for name, entry in reference.proof['files'].items():
            if name == 'manifest.json':
                continue  # Explicit storage Config changed the control, never policy bytes.
            assert entry['sha256'] == actual.proof['files'][name]['sha256']
            assert entry['bytes'] == actual.proof['files'][name]['bytes']
        original.close()
        reopened = owner(plan.output)
        again = _inspect_component_collection(reopened, ACTIONS, 'train', _game=True)
        assert canonical(again.proof) == canonical(actual.proof)
        reopened.close()


def test_external_disk_bytes_are_deduplicated_and_include_failed_wal(tmp_path: Path) -> None:
    from gf_training.ppo_run import owned_bytes
    parent, external = tmp_path / 'parent', tmp_path / 'external'
    parent.mkdir()
    external.mkdir()
    (parent / 'original').write_bytes(b'123')
    (external / 'failed.sqlite3-wal').write_bytes(b'12345')
    __import__('os').link(parent / 'original', external / 'linked')
    assert owned_bytes(parent, external_roots=(external, parent)) == 8
    (external / 'foreign-link').symlink_to(parent / 'original')
    with pytest.raises(ContractError, match='symlink'):
        owned_bytes(parent, external_roots=(external,))


def test_precommit_retires_same_borrowers_and_future_close_has_no_original_io(tmp_path: Path, monkeypatch) -> None:
    from gf_training.group_pack_game_source import PackedBatchCustody
    from gf_training.structured_lineage import _ChainScan
    root, _ = packed(tmp_path)
    original = owner(root)
    custody = PackedBatchCustody(original)
    scans = [_ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30) for _ in range(3)]
    for scan in scans:
        assert custody.borrow(scan) is original
    proof = {'storage': original.storage, 'custody': original.custody}
    custody.prepare(proof)
    assert custody.retired and custody.frozen_proof is not None
    assert original._closed and original.pool.closed
    assert original.locator.fd == -1 and original.artifacts._originals.database_file.fd == -1
    assert all(not scan._packed_owners for scan in scans)
    def forbidden():
        pytest.fail('reportable original I/O after successful prepare')
    monkeypatch.setattr(original, 'finalize', forbidden)
    monkeypatch.setattr(original, 'close', forbidden)
    custody.close()
    for scan in scans:
        scan.close(recheck=False)
    assert original.pool.budget.used == 0


def test_live_group_and_close_failure_preserve_actual_batch_owner(tmp_path: Path, monkeypatch) -> None:
    from gf_training.group_pack_game_source import PackedBatchCustody
    root, _ = packed(tmp_path)
    original = owner(root)
    custody = PackedBatchCustody(original)
    proof = {'storage': original.storage, 'custody': original.custody}
    with original.borrow_group('first'):
        with pytest.raises(RuntimeError, match='loan'):
            custody.prepare(proof)
        assert not custody.retired and original.locator.fd >= 0
    close = original.artifacts._originals.database_file.close
    def fail_close():
        raise OSError('injected actual original FD close failure')
    monkeypatch.setattr(original.artifacts._originals.database_file, 'close', fail_close)
    with pytest.raises(OSError) as failure:
        custody.prepare(proof)
    assert not custody.retired and custody.frozen_proof is None
    assert failure.value.cleanup_owner is original
    assert original.artifacts._originals.database_file.fd >= 0
    monkeypatch.setattr(original.artifacts._originals.database_file, 'close', close)
    original.close()


def test_failure_recheck_still_retires_actual_fds_and_preserves_failure(tmp_path: Path, monkeypatch) -> None:
    from gf_training.group_pack_game_source import PackedBatchCustody
    root, _ = packed(tmp_path)
    original = owner(root)
    custody = PackedBatchCustody(original)
    def fail():
        raise ContractError('initiating original recheck failure')
    monkeypatch.setattr(original, 'finalize', fail)
    with pytest.raises(ContractError, match='initiating original recheck'):
        custody.close()
    assert custody.retired and original._closed and original.pool.budget.used == 0
    assert original.locator.fd == -1


def test_bad_locator_constructor_retires_opened_originals(tmp_path: Path) -> None:
    import json
    root, _ = packed(tmp_path)
    path = root / 'collection.locator.json'
    pointer = json.loads(path.read_bytes())
    pointer['descriptor_sha256'] = '0' * 64
    path.write_bytes(canonical_json(pointer))
    before = len(tuple(Path('/proc/self/fd').iterdir()))
    with pytest.raises(ContractError, match='hash'):
        owner(root)
    assert len(tuple(Path('/proc/self/fd').iterdir())) == before


def test_optional_predecessor_is_separate_actual_original_custody(tmp_path: Path) -> None:
    import json
    root, locator = packed(tmp_path)
    descriptor = json.loads(locator.path.read_bytes())
    predecessor = locator.path.parent / 'collection.pending.json'
    predecessor.write_bytes(canonical_json({**descriptor, 'publication': 'pending'}))
    first = owner(root)
    custody = first.custody
    assert {row['role'] for row in custody['files']} == {'locator', 'descriptor', 'database', 'predecessor'}
    first.close()
    replacement = predecessor.with_suffix('.replacement')
    replacement.write_bytes(predecessor.read_bytes())
    replacement.replace(predecessor)
    second = owner(root)
    with pytest.raises(ContractError, match='custody'):
        second.require_custody(custody)
    second.close()
