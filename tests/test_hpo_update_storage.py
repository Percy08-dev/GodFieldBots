"""Physical HPO update publications, with sealed-format and decoder checks."""
import hashlib
import subprocess
import sys
import time
from compression import zstd
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, canonical
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414


def diagnostic_storage(root, mode):
    return {
        'http_log_ram_root': str(root / 'ram'),
        'http_log_archive_root': str(root / 'archive'),
        'http_log_retention': 'errors',
        'http_log_compression': 'zstd',
        'update_compression': mode,
    }


def sealed_state(root, mode):
    storage = {} if mode is None else {'hpo_diagnostics': diagnostic_storage(root, mode)}
    return {'effective_config': {'hpo_spec': {'storage': storage}}}


@pytest.mark.parametrize('mode', [None, 'none', 'zstd'])
def test_sealed_update_roundtrip_and_physical_digest(tmp_path, mode):
    from gf_training.hpo_update_storage import read_hpo_update

    updates = tmp_path / 'updates'
    updates.mkdir()
    value = {'numbers': [0.0, 1, 2e-8], 'label': '更新'}
    raw = (canonical(value) + '\n').encode()
    stored = zstd.compress(raw) if mode == 'zstd' else raw
    path = updates / ('000001.json.zst' if mode == 'zstd' else '000001.json')
    path.write_bytes(stored)
    digest = hashlib.sha256(stored).hexdigest()
    assert read_hpo_update(tmp_path, 1, sealed_state(tmp_path, mode), digest) == value
    with pytest.raises(ContractError, match='hash'):
        read_hpo_update(tmp_path, 1, sealed_state(tmp_path, mode), '0' * 64)
    other_mode = 'none' if mode == 'zstd' else 'zstd'
    with pytest.raises(FileNotFoundError):
        read_hpo_update(tmp_path, 1, sealed_state(tmp_path, other_mode), digest)


@pytest.mark.parametrize('cold', [False, True])
def test_only_explicit_hpo_read_decodes_zstd(tmp_path, cold):
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack import StorageSettings
    from gf_models.structured_sealed import SealedScan
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_restore import _ColdScan
    from gf_training.hpo_update_storage import read_hpo_update

    path = tmp_path / 'updates' / '000001.json.zst'
    path.parent.mkdir()
    stored = zstd.compress(b'{"update":1}\n')
    path.write_bytes(stored)
    expected = hashlib.sha256(stored).hexdigest()
    deadline = time.monotonic() + 10
    limit = 8 * 1024**2
    if cold:
        budget = StorageBudget(StorageSettings(), deadline)
        scan = _ColdScan(SimpleNamespace(directory=tmp_path), ArtifactPhaseAdmission(budget, limit), deadline)
    else:
        scan = SealedScan(max_owned_bytes=limit, deadline=deadline)
    try:
        state = sealed_state(tmp_path, 'zstd')
        first = read_hpo_update(tmp_path, 1, state, expected, scan=scan)
        assert first == {'update': 1}
        if cold:
            assert read_hpo_update(tmp_path, 1, state, expected, scan=scan) is first
        with pytest.raises(ValueError):
            scan.object(path, expected)
        with pytest.raises(ContractError, match='size exceeds bound'), scan.file(path, max_bytes=len(stored) - 1):
            pytest.fail('generic read admitted more bytes than requested')
    finally:
        if isinstance(scan, _ColdScan):
            scan.close()


def test_raw_hpo_update_import_and_read_with_no_python_zstd(tmp_path):
    subprocess.run([sys.executable, '-c', '''
import hashlib
import sys
from pathlib import Path
sys.modules['compression'] = None
from gf_common.hpo_update_bytes import encode_update
from gf_training.hpo_update_storage import read_hpo_update
parent = Path(sys.argv[1])
path = parent / 'updates' / '000001.json'
path.parent.mkdir()
raw = encode_update({'update': 1}, compressed=False)
path.write_bytes(raw)
state = {'effective_config': {'hpo_spec': {'storage': {}}}}
assert read_hpo_update(parent, 1, state, hashlib.sha256(raw).hexdigest()) == {'update': 1}
''', str(tmp_path)], check=True)


def test_sealed_recheck_refuses_grown_source_before_hashing(tmp_path, monkeypatch):
    from gf_models.structured_sealed import SealedScan

    path = tmp_path / 'bounded.json'
    path.write_bytes(b'{}')
    scan = SealedScan(max_owned_bytes=1024**2, deadline=time.monotonic() + 10)
    with scan.file(path, max_bytes=2):
        pass
    path.write_bytes(b'{"extra":1}')
    monkeypatch.setattr(hashlib, 'sha256', lambda: pytest.fail('grown file reached hashing'))
    with pytest.raises(ContractError, match='changed since verified snapshot'):
        scan.finalize()


@pytest.mark.parametrize('fault', ['truncated', 'junk', 'second-frame', 'stored-limit',
    'expanded-limit', 'unknown-expanded-limit', 'window-limit', 'known-window-limit', 'duplicate-key'])
def test_rehashed_untrusted_zstd_is_bounded_and_rejected(tmp_path, monkeypatch, fault):
    from gf_common import hpo_update_bytes
    from gf_training.hpo_update_storage import read_hpo_update

    updates = tmp_path / 'updates'
    updates.mkdir()
    raw = b'{"x":1}\n'
    body = zstd.compress(raw)
    if fault == 'truncated':
        body = body[:-1]
    elif fault == 'junk':
        body += b'junk'
    elif fault == 'second-frame':
        body += zstd.compress(b'')
    elif fault == 'stored-limit':
        monkeypatch.setattr(hpo_update_bytes, 'UPDATE_MAX_BYTES', len(body) - 1)
    elif fault.endswith('expanded-limit'):
        body = zstd.compress(b'{"x":"' + b'a' * 128 + b'"}\n', options={
            zstd.CompressionParameter.content_size_flag: 0 if fault.startswith('unknown') else 1})
        monkeypatch.setattr(hpo_update_bytes, 'UPDATE_MAX_BYTES', 64)
    elif fault.endswith('window-limit'):
        # A 16 MiB window with a small final raw block and optional content size.
        header = b'\x80\x70' + len(raw).to_bytes(4, 'little') if fault.startswith('known') else b'\x00\x70'
        body = b'\x28\xb5\x2f\xfd' + header + bytes([len(raw) * 8 + 1, 0, 0]) + raw
    else:
        body = zstd.compress(b'{"x":1,"x":2}\n')
    (updates / '000001.json.zst').write_bytes(body)
    with pytest.raises(ContractError):
        read_hpo_update(tmp_path, 1, sealed_state(tmp_path, 'zstd'), hashlib.sha256(body).hexdigest())


def test_actual_common_hpo_writer_and_full_structural_chain(global_root, tmp_path, monkeypatch):
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_update_storage import read_hpo_update
    from tests.test_resident_training import run_actual_source_case

    root = run_actual_source_case(global_root, tmp_path, monkeypatch, packed=False,
        diagnostic_storage=diagnostic_storage(tmp_path, 'zstd'))
    state, records = validate_hpo_chain(root, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    assert sorted(path.name for path in (root / 'updates').iterdir()) == ['000001.json.zst', '000002.json.zst']
    raw = zstd.decompress((root / 'updates/000001.json.zst').read_bytes())
    assert raw == (canonical(read_hpo_update(root, 1, state, records[1]['update_sha256'])) + '\n').encode()
    (root / 'updates/000001.json').write_bytes(raw)
    with pytest.raises(ContractError, match='inventory'):
        validate_hpo_chain(root, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_rust_player_cold_read_holds_decoded_control_loan(tmp_path):
    from gf_common.group_pack import StorageSettings
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_update_storage import check_cold_update_inventory
    from gf_training.rust_player_completion import _read

    directory = tmp_path / 'updates'
    directory.mkdir()
    value = {'kind': 'HPO', 'observations': [2, 3, 5] * 1_000}
    raw = (canonical(value) + '\n').encode()
    stored = zstd.compress(raw)
    path = directory / '000001.json.zst'
    path.write_bytes(stored)
    deadline = time.monotonic() + 30
    budget = StorageBudget(StorageSettings(), deadline)
    check_cold_update_inventory(tmp_path, 1, compressed=True, deadline=deadline)
    sha = hashlib.sha256(stored).hexdigest()
    loans = []
    result, digest = _read(path, budget, deadline, sha, loans=loans, hpo_update=True)
    assert result == value and digest == sha
    assert len(loans) == 1 and budget.used == 32 * len(raw) + 65536
    del result
    loans.pop().close()
    assert budget.used == 0
    (directory / '000001.json').write_bytes(raw)
    with pytest.raises(ContractError, match='inventory'):
        check_cold_update_inventory(tmp_path, 1, compressed=True, deadline=deadline)
