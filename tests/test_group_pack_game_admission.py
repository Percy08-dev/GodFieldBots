"""Realistic packed byte/graph admission; fixture Sources remain unqualified."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace

import pytest

from gf_cli.config import GroupPackStorageConfig
from gf_common.domain import ContractError, canonical, digest
from gf_common.group_pack import MIB, PackMember, canonical_json, encode_pack, unpack_raw
from gf_common.group_pack_game import storage_binding
from gf_training.group_pack_game_source import PackedSourceOwner
from gf_training.group_pack_writer import GroupPackWriter
from gf_training.lineage import run_lock
from gf_training.structured_source import _inspect_component_collection
from tests.card_instance_learning_support import initial_parent, original_collection
from tests.test_card_instance_learning import ACTIONS
from tests.test_group_pack_game_source import owner, pack_original, packed
from tests.test_group_pack_storage import settings as storage_settings


def fixture_original(tmp_path):
    parent = tmp_path / 'parent'
    with run_lock(parent, create=True):
        learner, state, records, settings = initial_parent(parent)
        from gf_training.card_instance_source import recorded_plan
        plan, _, _ = recorded_plan(parent, state, records, 0, time.monotonic() + 90)
        original_collection(plan, state, learner)
        manifest = json.loads((plan.output / 'manifest.json').read_bytes())
        settings = replace(settings, memory_budget_mib=2048)
        manifest['config']['preprocessing'] = asdict(settings)
        manifest['config']['storage'] = asdict(GroupPackStorageConfig(db_root=str(tmp_path / 'db'), ram_root=str(tmp_path / 'ram')))
        manifest['config_hash'] = digest(manifest['config'])
        return plan, manifest, settings


def update_inventory(directory, manifest):
    manifest['files'] = [{'path': p.relative_to(directory).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ('manifest.json', 'report.json')]
    (directory / 'manifest.json').write_text(canonical(manifest))


def test_realistic_sixteen_game_inventory_admits_under_unchanged_two_gib(tmp_path):
    plan, manifest, settings = fixture_original(tmp_path)
    # Eight ordinary group-local diagnostics stress original inventory bytes;
    # opaque bytes need custody, not thirty-two-fold JSON decoding reservations.
    for group in plan.groups:
        (plan.output / group.episodes[0].episode_id / 'diagnostic.log').write_bytes(b'x' * (8 * MIB))
    update_inventory(plan.output, manifest)
    pack_original(plan.output, tmp_path / 'external')
    original = PackedSourceOwner(plan.output, settings, time.monotonic() + 90,
        storage_binding=storage_binding(), max_owned_bytes=2 * 1024**3)
    try:
        actual = _inspect_component_collection(original, ACTIONS, 'train', _game=True)
        assert actual.proof['games'] == 16 and actual.proof['qualified'] is False
        assert actual.proof['source_set_sha256'] == digest(actual.proof['files'])
        assert len(actual.episodes) == 32
        assert original.pool.budget.limit == 2 * 1024**3
        assert original.pool.budget.used < 512 * MIB
        # Constructor validation temporarily reserves all available capacity;
        # that account peak is not an actual-residency measurement.
    finally:
        original.close()
    assert original.pool.closed and original.pool.budget.used == 0


def raw_original(tmp_path):
    size = 4 * MIB
    w = GroupPackWriter(tmp_path / 'external', 'raw', settings=storage_settings(tmp_path), deadline=time.monotonic() + 30)
    pack = encode_pack((PackMember('diagnostic', 'episode/raw', b'x' * size),))
    w.submit('group', pack, raw_bytes=len(unpack_raw(pack)), record={}, episodes=[])
    locator = w.seal()
    root = tmp_path / 'envelope'
    root.mkdir()
    (root / 'collection.locator.json').write_bytes(canonical_json({'format': 'sqlite-group-pack-locator-v1',
        'schema_version': 1, 'descriptor': str(locator.path), 'descriptor_sha256': hashlib.sha256(locator.path.read_bytes()).hexdigest()}))
    return owner(root), size


def test_raw_reader_and_distinct_decode_graphs_are_separately_admitted(tmp_path):
    original, size = raw_original(tmp_path)
    try:
        before = original._charge_bytes
        a = original.reader.read('episode/raw')
        first = original._charge_bytes - before
        assert len(a) == size and 2 * size <= first < 3 * size
        before = original._charge_bytes
        b = original.reader.read('episode/raw')
        second = original._charge_bytes - before
        assert a == b and 2 * size <= second < 3 * size
        # Repeated reads/parse graphs of the SAME filename remain independently
        # admitted; only the stable logical evidence metadata is shared.
        before = original._charge_bytes
        original.reader.admit_validation(12345)
        original.reader.admit_validation(12345)
        assert original._charge_bytes - before == 24690
    finally:
        original.close()


def test_inventory_evidence_is_admitted_once_before_identity_allocation(tmp_path, monkeypatch):
    root, _ = packed(tmp_path)
    original = owner(root)
    try:
        original.remember_inventory_entry('first/e0', 2, hashlib.sha256(b'\xff\x00').hexdigest())
        before = original._charge_bytes
        original.remember_inventory_entry('first/e0', 2, hashlib.sha256(b'\xff\x00').hexdigest())
        assert original._charge_bytes == before
        original.pool.budget.limit = original.pool.budget.used + 1
        def forbidden(name):
            pytest.fail('identity allocated before evidence admission')
        monkeypatch.setattr(original, 'member_identity', forbidden)
        with pytest.raises(ValueError, match='budget|capacity'):
            original.remember_inventory_entry('first/e1', 2, hashlib.sha256(b'\xff\x01').hexdigest())
        assert 'first/e1' not in original.files
    finally:
        original.close()
    assert original.pool.budget.used == 0


@pytest.mark.parametrize('field,value', [('bytes', 3), ('sha256', '0' * 64), ('path', 'missing/file')])
def test_untrusted_manifest_entry_cannot_replace_constructor_evidence(tmp_path, field, value):
    root, _ = packed(tmp_path)
    original = owner(root)
    try:
        entry = {'path': 'first/e0', 'bytes': 2, 'sha256': hashlib.sha256(b'\xff\x00').hexdigest(), field: value}
        with pytest.raises(ContractError):
            original.remember_inventory_entry(entry['path'], entry['bytes'], entry['sha256'])
        assert not original.files
    finally:
        original.close()


def test_packed_journals_do_not_enter_outer_json_parse(tmp_path, monkeypatch):
    from gf_training import structured_source
    plan, manifest, settings = fixture_original(tmp_path)
    update_inventory(plan.output, manifest)
    pack_original(plan.output, tmp_path / 'external')
    parse = structured_source._json
    def checked(raw):
        value = parse(raw)
        if isinstance(value, dict) and value.get('kind') in ('header', 'decision', 'events', 'end'):
            pytest.fail('journal decoded by duplicate outer Source parser')
        return value
    monkeypatch.setattr(structured_source, '_json', checked)
    original = PackedSourceOwner(plan.output, settings, time.monotonic() + 60,
        storage_binding=storage_binding(), max_owned_bytes=2 * 1024**3)
    try:
        result = _inspect_component_collection(original, ACTIONS, 'train', _game=True)
        assert len(result.episodes) == 32
    finally:
        original.close()


@pytest.mark.parametrize('malformation', ['duplicate-key', 'nonfinite', 'nonobject'])
def test_single_packed_journal_parse_preserves_strict_json_refusal(tmp_path, malformation):
    plan, manifest, settings = fixture_original(tmp_path)
    path = next(plan.output.glob('*/player-1.jsonl'))
    raw = path.read_text()
    if malformation == 'duplicate-key':
        raw = raw.replace('{', '{"kind":"header",', 1)
    elif malformation == 'nonfinite':
        raw = raw.replace('{', '{"unexpected_nonfinite":NaN,', 1)
    else:
        raw = '[]\n' + raw
    path.write_text(raw)
    update_inventory(plan.output, manifest)
    pack_original(plan.output, tmp_path / 'external')
    original = PackedSourceOwner(plan.output, settings, time.monotonic() + 60,
        storage_binding=storage_binding(), max_owned_bytes=2 * 1024**3)
    try:
        with pytest.raises(ContractError, match='duplicate|nonfinite|objects'):
            _inspect_component_collection(original, ACTIONS, 'train', _game=True)
    finally:
        original.close()
    assert original.pool.budget.used == 0


def test_generic_object_keeps_separate_live_json_decode_admission(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    try:
        a = original.object(root / 'manifest.json')
        before = original._charge_bytes
        b = original.object(root / 'manifest.json')
        assert a == b and a is not b
        assert original._charge_bytes - before >= 32 * len(b'{"hello":"original"}\n') + 65536
    finally:
        original.close()


def test_validated_inventory_evidence_does_not_reread_opaque_bytes_and_rechecks_original(tmp_path, monkeypatch):
    root, _ = packed(tmp_path)
    original = owner(root)
    file = original.artifacts._originals.database_file
    try:
        def forbidden(*args, **kwargs):
            pytest.fail('inventory repeated original member read')
        monkeypatch.setattr(original.artifacts, 'read', forbidden)
        monkeypatch.setattr(original.artifacts, 'borrow_group', forbidden)
        original.remember_inventory_entry('first/e0', 2, hashlib.sha256(b'\xff\x00').hexdigest())
        assert original.files['first/e0']['sha256'] == hashlib.sha256(b'\xff\x00').hexdigest()
        data = file.path.read_bytes()
        file.path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with pytest.raises(ValueError, match='original|identity|content'):
            original.finalize()
    finally:
        original.close()
    assert file.fd == -1 and original.pool.closed and original.pool.budget.used == 0


@pytest.mark.parametrize('size_mib', [1, 2])
def test_complete_packed_source_preserves_opaque_file_cap_without_reread(tmp_path, monkeypatch, size_mib):
    plan, manifest, settings = fixture_original(tmp_path)
    settings = replace(settings, max_source_file_mib=1, max_record_mib=1)
    manifest['config']['preprocessing'] = asdict(settings)
    manifest['config_hash'] = digest(manifest['config'])
    name = plan.groups[0].episodes[0].episode_id + '/bounded-diagnostic.log'
    (plan.output / name).write_bytes(b'x' * (size_mib * MIB))
    update_inventory(plan.output, manifest)
    pack_original(plan.output, tmp_path / 'external')
    original = PackedSourceOwner(plan.output, settings, time.monotonic() + 60,
        storage_binding=storage_binding(), max_owned_bytes=2 * 1024**3)
    read = original._read_original
    def no_opaque_read(path, **kwargs):
        if path == plan.output / name:
            pytest.fail('opaque inventory payload reread instead of bounded evidence')
        return read(path, **kwargs)
    monkeypatch.setattr(original, '_read_original', no_opaque_read)
    try:
        if size_mib > 1:
            with pytest.raises(ContractError, match='bounded original logical file required'):
                _inspect_component_collection(original, ACTIONS, 'train', _game=True)
            assert name not in original.files
        else:
            actual = _inspect_component_collection(original, ACTIONS, 'train', _game=True)
            assert len(actual.episodes) == 32 and actual.proof['qualified'] is False
            assert actual.proof['files'][name]['bytes'] == MIB
            assert actual.proof['files'][name]['sha256'] == hashlib.sha256(b'x' * MIB).hexdigest()
    finally:
        original.close()
    assert original._closed and original.pool.closed and original.pool.budget.used == 0
    assert original.locator.fd == -1
    assert original.artifacts._originals.database_file.fd == -1
    assert original.artifacts._originals.descriptor_file.fd == -1
    assert original.artifacts._originals.db is None
