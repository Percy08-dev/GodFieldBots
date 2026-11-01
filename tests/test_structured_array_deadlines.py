"""Cooperative deadlines through bounded numerical I/O, no Torch needed."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import gf_models.sharded_arrays as shards
from gf_models import arrays


def expired():
    raise TimeoutError('injected caller deadline')


@pytest.mark.parametrize('operation', ['hash', 'inspect', 'read', 'write'])
def test_expired_entry_fails_before_empty_or_invalid_io(tmp_path, operation):
    with pytest.raises(TimeoutError, match='injected caller deadline'):
        if operation == 'hash':
            arrays.sha256(tmp_path / 'absent', _check=expired)
        elif operation == 'inspect':
            shards.inspect_bundle(tmp_path, {}, _check=expired)
        elif operation == 'read':
            shards.read_bundle(tmp_path, {}, _check=expired)
        else:
            shards.write_bundle(tmp_path / 'absent', 'x', {}, _check=expired)
    assert not (tmp_path / 'absent').exists()


def test_padded_small_decoded_npz_stops_during_chunk_hash_before_archive(tmp_path, monkeypatch):
    path = tmp_path / 'tensor.npz'
    metadata = arrays.save_arrays(path, {'x': np.ones(1, np.float32)})
    path.write_bytes(b'padding' * 400000 + path.read_bytes())
    metadata['sha256'] = arrays.sha256(path)
    original_open = Path.open
    opened = []
    def capture(self, *args, **kwargs):
        source = original_open(self, *args, **kwargs)
        if self == path:
            opened.append(source)
        return source
    def check():
        if opened and not opened[-1].closed and opened[-1].tell() >= 1024**2:
            raise TimeoutError('hash chunk deadline')
    monkeypatch.setattr(Path, 'open', capture)
    with pytest.raises(TimeoutError, match='hash chunk deadline'):
        arrays.load_arrays(path, metadata, _check=check)
    assert opened[-1].closed


def test_many_shards_preserve_deadline_from_first_numpy_decode(tmp_path, monkeypatch):
    manifest = shards.write_bundle(tmp_path, 'many', {f'x{i}': np.ones(4, np.uint8) for i in range(8)}, shard_bytes=4)
    assert len(manifest['shards']) == 8
    original_load, calls = np.load, []
    def loaded(*args, **kwargs):
        value = original_load(*args, **kwargs)
        calls.append(1)
        return value
    def check():
        if calls:
            raise TimeoutError('array member deadline')
    monkeypatch.setattr(np, 'load', loaded)
    with pytest.raises(TimeoutError, match='array member deadline'):
        shards.read_bundle(tmp_path, manifest, _check=check)
    assert calls == [1]


def test_save_deadline_leaves_only_unpublished_npz_evidence(tmp_path, monkeypatch):
    original_save, saved = np.savez, []
    def save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        saved.append(1)
        return result
    def check():
        if saved:
            raise TimeoutError('after bounded save')
    monkeypatch.setattr(np, 'savez', save)
    with pytest.raises(TimeoutError, match='after bounded save'):
        shards.write_bundle(tmp_path / 'output', 'one', {'x': np.ones(1, np.float32)}, _check=check)
    assert (tmp_path / 'output/one-000000.npz').is_file()
    assert not (tmp_path / 'output/one-000000.json').exists()


def test_default_load_still_wraps_os_timeout_while_injected_deadline_survives(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError('legacy filesystem error')
    monkeypatch.setattr(arrays, '_load_arrays', timeout)
    from gf_common.domain import ContractError
    with pytest.raises(ContractError, match='typed-array'):
        arrays.load_arrays(tmp_path / 'x')
    with pytest.raises(TimeoutError, match='legacy filesystem error'):
        arrays.load_arrays(tmp_path / 'x', _check=lambda: None)


def test_caller_owned_fd_is_used_for_archive_and_position_is_restored(tmp_path, monkeypatch):
    path = tmp_path / 'x.npz'
    metadata = arrays.save_arrays(path, {'x': np.arange(8, dtype=np.float32)})
    original = arrays.zipfile.ZipFile
    seen = []
    with path.open('rb') as source:
        info = os.fstat(source.fileno())
        identity = tuple(getattr(info, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
        source.seek(7)
        def archive(actual, *args, **kwargs):
            assert actual is source
            seen.append(1)
            return original(actual, *args, **kwargs)
        monkeypatch.setattr(arrays.zipfile, 'ZipFile', archive)
        loaded = arrays.load_arrays(path, metadata, _source=source, _snapshot=identity, _check=lambda: None)
        np.testing.assert_array_equal(loaded['x'], np.arange(8, dtype=np.float32))
        assert source.tell() == 7 and not source.closed and seen == [1]
