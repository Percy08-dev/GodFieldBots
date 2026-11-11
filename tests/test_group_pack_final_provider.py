"""Actual shared pool/executor final-provider ownership, without Source eligibility."""
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from tests.test_group_pack_game_source import owner, packed


def test_final_provider_borrows_same_executor_hold_pool_and_keeps_base_registry_immutable(tmp_path):
    base_root, _ = packed(tmp_path / 'base')
    final_root, _ = packed(tmp_path / 'final')
    base = owner(base_root)
    provider = OriginalArtifactProvider((base,))
    registry = provider.registry()
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    final = final_provider = None
    try:
        executor.prepare()  # Actual idle creator-retirement gate BEFORE new opens/query.
        final = owner(final_root, base.pool)
        final_provider = OriginalArtifactProvider((final,))
        executor.attach_final_provider(final_provider)
        assert final_provider._executor_owner is executor
        assert final_provider.pool is provider.pool
        assert len(base.pool._close_holds) == 1
        assert provider.registry() == registry
        assert executor.call(lambda: final_provider.borrow(str(final_root), 'first/e0').data()) == b'\xff\x00'
        assert not final_provider._leases and base.pool.active is None
        with pytest.raises(ContractError, match='executor'):
            final_provider.close()
        with pytest.raises(RuntimeError, match='hold'):
            final.close()
        with pytest.raises(ContractError):
            executor.attach_final_provider(final_provider)
    finally:
        executor.close()
        provider.close()
        if final_provider is not None:
            final_provider.close()
        if final is not None:
            final.close()
        base.close()
    assert base.pool.closed and base.pool.budget.used == 0


def test_final_provider_failure_fences_both_callbacks_and_retains_same_actual_hold_until_join(tmp_path):
    base_root, _ = packed(tmp_path / 'base')
    final_root, _ = packed(tmp_path / 'final')
    base = owner(base_root)
    provider = OriginalArtifactProvider((base,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    executor.prepare()
    final = owner(final_root, base.pool)
    final_provider = OriginalArtifactProvider((final,))
    first = OSError('final operation failed')
    try:
        executor.attach_final_provider(final_provider)
        with pytest.raises(OSError) as error:
            executor.call(lambda: (_ for _ in ()).throw(first))
        assert error.value is first
        assert executor._hold is not None
        assert final_provider._failure is first and provider._failure is first
        with pytest.raises(RuntimeError, match='hold'):
            final.close()
        with pytest.raises(OSError) as closed:
            executor.close()
        assert closed.value is first and executor._closed and executor._hold is None
        assert provider._fenced and final_provider._fenced
    finally:
        if not executor._closed:
            if executor._failure is not None:
                with pytest.raises(OSError):
                    executor.close()
            else:
                executor.close()
        provider.close()
        final_provider.close()
        final.close()
        base.close()
    assert base.pool.budget.used == 0


def test_external_growth_and_final_working_delta_are_never_credited_as_shared_controls(tmp_path):
    import time

    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.structured_lineage import _ChainScan
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    scan._packed_pool = original.pool
    scan.charge(4096)
    actual = _NativeGameInputs.__new__(_NativeGameInputs)
    actual._closed, actual.deadline = False, time.monotonic() + 30
    actual._artifact_provider, actual._shared_controls = provider, scan
    actual.sealed, actual._persistent_python = None, 0
    actual._final_working_delta = 13
    try:
        actual.caller_growth(7, external_growth=11)
        assert actual._held() == 24
        actual._shared_controls = None
        assert actual._held() == 31
        actual.caller_growth(7)
        assert actual._held() == 20
    finally:
        provider.close()
        scan.close(recheck=False)
        original.close()
