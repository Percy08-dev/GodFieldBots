"""Physical hash verification and default diagnostic projection."""

import hashlib
import importlib.util
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import pytest

from gf_common.domain import ContractError
from gf_common.group_pack import canonical_json
from gf_training.group_pack_resources import StorageBudget, StorageSettings
from gf_training.rust_ppo_pack import verify_numeric_files, verify_saved_pack


def seal(directory, content):
    database = directory / 'collection.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE evidence (value TEXT)')
        connection.execute('INSERT INTO evidence VALUES (?)', (content,))
    manifest = 'a' * 64
    physical = {'database_sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
        'database_bytes': database.stat().st_size, 'seal_id': str(uuid4())}
    descriptor = {'format': 'sqlite-group-pack-v1', 'schema_version': 1, 'run_id': 'run',
        **physical, 'database': database.name, 'publication': 'pending',
        'controls': [{'name': 'manifest.json', 'bytes': 8, 'sha256': manifest}]}
    path = directory / 'collection.pending.json'
    data = canonical_json(descriptor)
    path.write_bytes(data)
    return path, physical | {'descriptor_sha256': hashlib.sha256(data).hexdigest(), 'logical_sha256': 'b' * 64}


def test_sealed_verification_hashes_actual_database_without_opening_sqlite(tmp_path, monkeypatch):
    path, evidence = seal(tmp_path, 'first collection')
    deadline = time.monotonic() + 10
    budget = StorageBudget(StorageSettings(), deadline)

    def unexpected_sqlite(*args, **kwargs):
        pytest.fail('normal Source verification opened SQLite')

    monkeypatch.setattr(sqlite3, 'connect', unexpected_sqlite)
    with_args = {'run_id': 'run', 'manifest_sha256': 'a' * 64, 'deadline': deadline, 'budget': budget}
    assert verify_saved_pack(path, evidence, **with_args) == evidence
    database = path.with_name('collection.sqlite3')
    raw = database.read_bytes()
    database.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ContractError, match='database checksum'):
        verify_saved_pack(path, evidence, **with_args)
    assert budget.used == 0


def test_swapping_valid_sealed_collection_is_rejected(tmp_path):
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir()
    right.mkdir()
    _, expected = seal(left, 'first collection')
    path, _ = seal(right, 'other collection')
    deadline = time.monotonic() + 10
    budget = StorageBudget(StorageSettings(), deadline)
    with pytest.raises(ContractError, match='descriptor checksum'):
        verify_saved_pack(path, expected, run_id='run', manifest_sha256='a' * 64,
            deadline=deadline, budget=budget)


def test_normal_numeric_verification_streams_payload_and_rejects_missing_or_changed_shards(tmp_path):
    path = tmp_path / 'rollout-00000.npz'
    path.write_bytes(b'original')
    deadline = time.monotonic() + 10
    budget = StorageBudget(StorageSettings(), deadline)
    files = {path.name: hashlib.sha256(b'original').hexdigest()}
    verify_numeric_files(tmp_path, files, deadline=deadline, budget=budget)
    path.write_bytes(b'changed!')
    with pytest.raises(ContractError, match='shard checksum'):
        verify_numeric_files(tmp_path, files, deadline=deadline, budget=budget)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_numeric_files(tmp_path, files, deadline=deadline, budget=budget)
    assert budget.used == 0


def test_default_operation_projection_uses_verified_writer_counts(monkeypatch):
    from gf_training import group_pack_source

    spec = importlib.util.spec_from_file_location('verify_ppo_operation',
        Path(__file__).parents[1] / 'tools/rust_player_verify_operation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def unexpected_pack(*args, **kwargs):
        pytest.fail('normal operation projection opened packed members')

    monkeypatch.setattr(group_pack_source, 'PackedSource', unexpected_pack)
    update = {'batch_id': 'batch-000000', 'pack_verification': {'database_sha256': 'b' * 64},
        'actual_games': 16, 'planned_games': 16, 'collection_seconds': 2.0, 'update_seconds': 1.0,
        'generation': 1, 'source_proof': {'accepted_decisions': 64, 'closure_counts': {'natural': 16}}}
    data, = module._project_collections({'updates': [update],
        'measurement_scope': {'start_generation': 0, 'end_generation': 1}},
        limit=10**7, deadline=10, retained=(), native=True)
    assert data['accepted_decisions'] == 64 and data['natural_games'] == 16
    assert data['pack_verification'] == update['pack_verification']
