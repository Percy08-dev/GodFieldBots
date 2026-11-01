"""NumPy-only sealed failure boundaries; no optional Torch fixture imports."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError
from gf_models.sharded_arrays import validate_bundle_metadata, write_bundle
from gf_models.structured_sealed import (
    SealedScan,
    inspect_pruned_inference,
    inspect_retained_artifact,
)


@pytest.mark.parametrize('case', ['one_byte', 'expired', 'few_fit'])
def test_inventory_admission_stops_before_thousand(tmp_path, monkeypatch, case):
    import gf_models.structured_sealed as module
    consumed = []
    class Entries:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def __iter__(self): return self
        def __next__(self):
            index = len(consumed)
            if index >= 1000: raise StopIteration
            consumed.append(index)
            return SimpleNamespace(name=f'e{index:04}.json', is_file=lambda **kwargs: True)
    monkeypatch.setattr(module.os, 'scandir', lambda path: Entries())
    scan = SealedScan(max_owned_bytes=1 if case == 'one_byte' else 600000,
                      deadline=-1. if case == 'expired' else float('inf'))
    with pytest.raises((ContractError, TimeoutError)):
        scan.inventory(tmp_path, {f'e{index:04}.json' for index in range(1000)})
    assert len(consumed) < 1000
    if case != 'few_fit':
        assert not consumed
    else:
        assert consumed


def test_empty_expired_finalize_and_late_inventory_reject(tmp_path, monkeypatch):
    with pytest.raises(TimeoutError):
        SealedScan(max_owned_bytes=100000, deadline=-1).finalize()
    (tmp_path / 'a.json').write_text('{}')
    scan = SealedScan(max_owned_bytes=1000000, deadline=float('inf'))
    scan.object(tmp_path / 'a.json')
    scan.inventory(tmp_path, {'a.json'})
    original = scan.file
    @contextmanager
    def late(path, expected=None):
        with original(path, expected) as value:
            yield value
        (tmp_path / 'late.json').write_text('{}')
    monkeypatch.setattr(scan, 'file', late)
    with pytest.raises(ContractError, match='inventory'):
        scan.finalize()


@pytest.mark.parametrize('expected', [None, '', '0', True])
def test_expected_digest_mandatory_before_any_io(tmp_path, monkeypatch, expected):
    scan = SealedScan(max_owned_bytes=1000000, deadline=float('inf'))
    monkeypatch.setattr(scan, 'object', lambda *args, **kwargs: pytest.fail('invalid required digest reached IO'))
    with pytest.raises(ContractError, match='SHA256'):
        inspect_retained_artifact(tmp_path, expected_inference_digest=expected, action_contract=engineering_contract(),
                                  purpose='component', synthetic_capacity=False, scan=scan)


def test_json_sidecar_hash_and_pruned_resume_digest_are_required(tmp_path):
    manifest = write_bundle(tmp_path / 'bundle', 'weights', {'a': np.zeros(2, dtype=np.float32)})
    manifest['files']['weights-000000.json'] = None
    with pytest.raises(ContractError, match='digest'):
        validate_bundle_metadata(manifest)
    with pytest.raises(ContractError, match='SHA256'):
        inspect_pruned_inference(tmp_path, expected_inference_digest='a' * 64, expected_resume_digest=None,
            original_files={}, deletion_receipt={}, action_contract=engineering_contract(), purpose='component', synthetic_capacity=False,
            scan=SealedScan(max_owned_bytes=1000000, deadline=float('inf')))


def test_second_tensor_read_loop_checks_deadline(tmp_path, monkeypatch):
    import gf_models.structured_sealed as module
    manifest = write_bundle(tmp_path / 'arrays', 'weights', {'a': np.zeros(2, dtype=np.float32), 'b': np.zeros(2, dtype=np.float32)})
    now, loads = [0.], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    original = module.np.load
    def slow(*args, **kwargs):
        loads.append(1)
        value = original(*args, **kwargs)
        now[0] = 20.
        return value
    monkeypatch.setattr(module.np, 'load', slow)
    with pytest.raises(TimeoutError):
        SealedScan(max_owned_bytes=1000000, deadline=10.).bundle(tmp_path / 'arrays', manifest)
    assert loads == [1]
