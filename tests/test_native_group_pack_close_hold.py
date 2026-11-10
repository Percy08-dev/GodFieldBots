from __future__ import annotations

from types import SimpleNamespace

import pytest

from gf_training.group_pack_game_source import PackedBatchCustody
from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from gf_training.structured_lineage import _ChainScan
from tests.test_group_pack_game_source import owner, packed


def test_actual_borrowed_scan_and_custody_pool_terminal_closes_refuse_until_join(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    custody = PackedBatchCustody(original)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    scan.charge(4096)
    custody.borrow(scan)
    assert not scan._owns_packed_pool and scan._pool_control_charge > 0
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    used, charge = original.pool.budget.used, scan._pool_control_charge
    fds = original.locator.fd, original.artifacts._originals.database_file.fd
    for close in (original.close, original.pool.close, custody.close, lambda: scan.close(recheck=False)):
        with pytest.raises(RuntimeError, match='hold'):
            close()
        assert original.pool.budget.used == used and scan._pool_control_charge == charge
        assert scan._packed_borrows[root] is custody and not scan._closed
        assert (original.locator.fd, original.artifacts._originals.database_file.fd) == fds
    executor.close()
    provider.close()
    scan.close(recheck=False)
    custody.close()
    assert original.pool.closed and original.pool.budget.used == 0


def test_independent_actual_holds_and_wrong_owner_cannot_release_another(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    first = OriginalArtifactProvider((original,))
    second = OriginalArtifactProvider((original,))
    a = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), first)
    b = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), second)
    with pytest.raises(RuntimeError, match='actual'):
        a._hold.retire(b)
    a.close()
    first.close()
    with pytest.raises(RuntimeError, match='hold'):
        original.close()
    b.close()
    second.close()
    original.close()
    assert original.pool.budget.used == 0


@pytest.mark.parametrize('after', [False, True])
def test_hold_registration_account_fault_is_preowned_and_retired_without_executor_effects(tmp_path, monkeypatch, after):
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    before = original.pool.budget.used
    acquire = original.pool.budget.acquire
    def fault(amount, deadline, **kwargs):
        if amount != 4096:
            return acquire(amount, deadline, **kwargs)
        if after:
            acquire(amount, deadline, **kwargs)
        raise OSError('hold admission fault')
    monkeypatch.setattr(original.pool.budget, 'acquire', fault)
    with pytest.raises(OSError, match='admission'):
        _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    actual = provider._executor_owner
    assert actual._closed and actual._worker is None and actual._hold is None
    assert original.pool.budget.used == before and not original.pool._close_holds
    provider.close()
    original.close()


def test_partial_executor_init_after_actual_object_creation_does_not_start_cleanup_worker(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    initialize = ThreadPoolExecutor.__init__
    def fault(actual, *args, **kwargs):
        initialize(actual, *args, **kwargs)
        raise OSError('executor initialized then failed')
    monkeypatch.setattr(ThreadPoolExecutor, '__init__', fault)
    with pytest.raises(OSError):
        _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    actual = provider._executor_owner
    assert actual._closed and actual._worker is None and not original.pool._close_holds
    provider.close()
    original.close()


def test_executor_init_before_attributes_has_confirmed_no_worker_effects(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    initialize = ThreadPoolExecutor.__init__
    def fault(actual, *args, **kwargs):
        assert actual.__dict__ == {}
        raise OSError('executor failed before attributes')
    monkeypatch.setattr(ThreadPoolExecutor, '__init__', fault)
    try:
        with pytest.raises(OSError, match='before attributes'):
            _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
        actual = provider._executor_owner
        assert actual._closed and actual._worker is None and not original.pool._close_holds
    finally:
        # Disposable fault cleanup, never production evidence: initialize the
        # SAME demonstrably empty object so this RED test cannot strand a hold.
        actual = provider._executor_owner
        if not actual._closed and actual._executor.__dict__ == {}:
            initialize(actual._executor, max_workers=1)
            with pytest.raises(OSError):
                actual.close()
        if actual._closed:
            provider.close()
            original.close()
