from __future__ import annotations

import time
from pathlib import Path

import pytest

from gf_training import group_pack_db as source_module
from gf_training.group_pack_source import OriginalFile, PackedSource
from tests.test_group_pack_storage import submit, writer


def test_ordinary_database_is_not_hashed_and_groups_decode_on_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = writer(tmp_path)
    submit(w, 'a')
    submit(w, 'b')
    locator = w.seal()
    hashes: list[str] = []
    decodes: list[int] = []
    original_hash, original_decode = OriginalFile.hash, source_module.decode_pack

    def counted_hash(self: OriginalFile) -> str:
        hashes.append(self.path.name)
        return original_hash(self)

    def counted_decode(data: bytes, limit: int):  # type: ignore[no-untyped-def]
        decodes.append(len(data))
        return original_decode(data, limit)

    monkeypatch.setattr(OriginalFile, 'hash', counted_hash)
    monkeypatch.setattr(source_module, 'decode_pack', counted_decode)
    source = PackedSource(locator.path, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2)
    assert 'collection.sqlite3' not in hashes
    assert decodes == []
    with source.borrow_group('a') as loan:
        assert len(decodes) == 1
        original_recheck = source.recheck
        monkeypatch.setattr(source, 'recheck', lambda: pytest.fail('cached loan performed I/O'))
        assert loan.read('a/e0') == b'\xff\x00'
        assert loan.names() == ('a/e0', 'a/e1')
    monkeypatch.setattr(source, 'recheck', original_recheck)
    source.evict_group()
    with source.borrow_group('a') as loan:
        assert loan.read('a/e0') == b'\xff\x00'
    assert len(decodes) == 2
    source.close()
    assert 'collection.sqlite3' not in hashes


def test_database_specific_expected_hash_does_not_weaken_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_common.group_pack import digest
    path = tmp_path/'control'
    path.write_bytes(b'control')
    original = OriginalFile(path, time.monotonic()+30)
    monkeypatch.setattr(original, 'hash', lambda: '0'*64)
    with pytest.raises(ValueError, match='content'):
        original.recheck()
    original.close()
    database = OriginalFile(path, time.monotonic()+30, database_expected_sha256=digest(b'control'))
    monkeypatch.setattr(database, 'hash', lambda: pytest.fail('expected DB checksum was recomputed'))
    database.recheck()
    database.close()


@pytest.mark.parametrize('identity_field', [0, 3])
def test_historical_custody_is_checked_before_cross_version_comparison(tmp_path: Path, identity_field: int) -> None:
    from copy import deepcopy

    from gf_common.domain import ContractError, digest
    from gf_common.group_pack_game import same_storage_custody
    from tests.test_group_pack_game_source import owner, packed
    root, _ = packed(tmp_path)
    current = owner(root)
    try:
        v2 = current.custody
        assert v2['recipe'] == 'packed-original-custody-v2'
        v1 = deepcopy(v2)
        v1['recipe'] = 'packed-original-custody-v1'
        v1['fingerprint'] = digest({k: v for k, v in v1.items() if k != 'fingerprint'})
        assert same_storage_custody(current.storage, v1, current.storage, v2)
        v1['files'][0]['identity'][identity_field] += 1
        with pytest.raises(ContractError):
            same_storage_custody(current.storage, v1, current.storage, v2)
        v1['fingerprint'] = digest({k: v for k, v in v1.items() if k != 'fingerprint'})
        assert not same_storage_custody(current.storage, v1, current.storage, v2)
    finally:
        current.close()


def test_cached_members_through_owner_reader_and_native_provider_are_memory_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_training.native_group_pack import OriginalArtifactProvider
    from tests.test_group_pack_game_source import owner, packed
    root, _ = packed(tmp_path)
    current = owner(root)
    provider = OriginalArtifactProvider((current,))
    try:
        lease = provider.borrow(str(root), 'first/e0')
        assert lease.data() == b'\xff\x00'
        lease.close()
        assert current.artifacts is not None
        monkeypatch.setattr(current.artifacts, 'recheck', lambda: pytest.fail('cached member filesystem check'))
        monkeypatch.setattr(current, 'finalize', lambda: pytest.fail('cached member finalize'))
        monkeypatch.setattr('gf_training.native_group_pack.hashlib.sha256', lambda *_: pytest.fail('cached member hash'))
        assert current.reader.read('first/e1') == b'\xff\x01'
        assert current.artifacts.read('first/e0') == b'\xff\x00'
        with current.borrow_group('first') as loan:
            assert loan.read('first/e1') == b'\xff\x01'
        lease = provider.borrow(str(root), 'first/e1')
        assert lease.data() == b'\xff\x01'
        lease.close()
    finally:
        monkeypatch.undo()
        provider.close()
        current.close()


def test_historical_dataset4_checks_saved_aggregates_and_all_roles() -> None:
    from copy import deepcopy

    from gf_common import card_instance_game_tree as game
    from gf_common.domain import ContractError, digest
    from gf_common.group_pack_game import same_packed_dataset
    from tests.test_group_pack_dataset_contract import packed_dto
    old = packed_dto()
    current = deepcopy(old)
    for row in current['collections']:
        value = row['custody']
        value['recipe'] = 'packed-original-custody-v2'
        value['fingerprint'] = digest({k: v for k, v in value.items() if k != 'fingerprint'})
    current['source_set_sha256'] = game.source_identity(current['parents'], current['collections'])
    current['content_sha256'] = digest({k: v for k, v in current.items() if k != 'content_sha256'})
    assert same_packed_dataset(old, current)
    old['content_sha256'] = 'f'*64
    with pytest.raises(ContractError):
        same_packed_dataset(old, current)


def test_native_policy_refuses_before_general_native_reader_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from gf_common.domain import ContractError
    from gf_common.preprocessing import PreparationConfig
    from gf_training import native_backend
    monkeypatch.setattr(native_backend, 'card_instance_reader_settings', lambda *_: pytest.fail('late refusal'))
    for marker in (None, 'pack-checksum-v0;custody1'):
        native = SimpleNamespace(build_info=lambda marker=marker: {'card_instance_group_pack': 'provider1;proof2;plan4',
            'card_instance_pack_checksum': marker})
        with pytest.raises(ContractError, match='checksum'):
            native_backend.group_pack_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), native)


def test_writer_parent_publication_never_rehashes_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_hash = OriginalFile.hash
    def hash_file(file: OriginalFile) -> str:
        if file.path.name == 'collection.sqlite3':
            pytest.fail('parent publication rehashed the sealed DB')
        return original_hash(file)
    monkeypatch.setattr(OriginalFile, 'hash', hash_file)
    w = writer(tmp_path)
    submit(w, 'a')
    assert w.seal().path.exists()


def test_admission_never_runs_integrity_or_fetches_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    statements: list[str] = []
    connect = sqlite3.connect
    def recording(*args, **kwargs):  # type: ignore[no-untyped-def]
        db = connect(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db
    monkeypatch.setattr(sqlite3, 'connect', recording)
    source = PackedSource(locator.path, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2)
    assert not any('integrity_check' in s or 'foreign_key_check' in s or 'SELECT pack FROM' in s for s in statements)
    with source.borrow_group('a') as loan:
        assert loan.read('a/e0') == b'\xff\x00'
    assert sum('SELECT pack FROM' in s for s in statements) == 1
    source.close()


def test_unbound_recovery_is_metadata_only_and_cannot_supply_descriptor_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_training.group_pack_db import reconcile_collection
    from gf_training.group_pack_source import PackedOriginals, inspect_packed
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    raw_descriptor = locator.path.read_bytes()
    locator.path.unlink()
    hash_file = OriginalFile.hash
    def no_database_hash(file: OriginalFile) -> str:
        if file.path.name == 'collection.sqlite3':
            pytest.fail('diagnostic recovery hashed an unbound DB')
        return hash_file(file)
    monkeypatch.setattr(OriginalFile, 'hash', no_database_hash)
    monkeypatch.setattr(source_module, 'decode_pack', lambda *_: pytest.fail('diagnostic recovery decoded a Pack'))
    database = locator.path.parent/'collection.sqlite3'
    result = inspect_packed(database, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2,
        without_descriptor=True)
    assert result['database_sha256'] is None
    assert reconcile_collection(database.parent, deadline=time.monotonic()+30,
        max_owned_bytes=512*1024**2)['state'] == 'sealed-without-descriptor'
    unbound = OriginalFile(database, time.monotonic()+30, _database_index_only=True)
    locator.path.write_bytes(raw_descriptor)
    try:
        with pytest.raises(ValueError, match='identity mismatch'):
            PackedOriginals(locator.path, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2,
                database_original=unbound)
    finally:
        unbound.close()


def test_diagnostic_close_uncertainty_is_not_reported_as_unsealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_training.group_pack_db import reconcile_collection
    w = writer(tmp_path)
    submit(w, 'a')
    locator = w.seal()
    locator.path.unlink()
    close = OriginalFile.close
    def uncertain(file: OriginalFile) -> None:
        if file.path.name == 'collection.sqlite3':
            raise ValueError('uncertain diagnostic database close')
        close(file)
    monkeypatch.setattr(OriginalFile, 'close', uncertain)
    with pytest.raises(ValueError, match='uncertain') as failed:
        reconcile_collection(locator.path.parent, deadline=time.monotonic()+30, max_owned_bytes=512*1024**2)
    actual = vars(failed.value)['cleanup_owner']
    assert not actual._closed
    monkeypatch.undo()
    actual.close()
