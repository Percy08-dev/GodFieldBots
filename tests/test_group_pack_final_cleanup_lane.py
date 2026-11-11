"""Actual fixed cleanup worker ownership, without an admitted formal Source."""
import threading
from types import SimpleNamespace

import pytest

from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from tests.test_group_pack_game_source import owner, packed


def test_failed_final_slot_uses_same_worker_fences_and_cancels_before_native_close_without_join(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    cancelled = []
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: cancelled.append(True)), provider)
    first = OSError('initial admitted final operation failed')
    calls = []
    class Native:
        closed = False
        slot_closed = False
        def operation(self):
            calls.append(('operation', threading.get_ident()))
            raise first
        def close_final_validation(self):
            assert provider._fenced and cancelled
            calls.append(('slot', threading.get_ident()))
            if not self.slot_closed:
                self.slot_closed = True
                raise RuntimeError('first cancelled guard, actual slot retired')
        def close(self):
            self.closed = True
    native = Native()
    executor.bind_native(native)
    before = original.pool.budget.used
    try:
        with pytest.raises(OSError) as failed:
            executor.call(native.operation)
        assert failed.value is first
        with pytest.raises(RuntimeError, match='cancelled guard'):
            executor.close_final_validation()
        executor.close_final_validation()
        assert native.slot_closed and not native.closed
        assert {worker for _, worker in calls} == {executor._worker}
        assert executor._worker != threading.get_ident()
        assert executor._hold is not None and not executor._closed
        assert executor._failure is first and executor._active is None
        assert original.pool.budget.used == before
        assert original.locator.fd >= 0
        with pytest.raises(RuntimeError, match='hold'):
            original.close()
    finally:
        with pytest.raises(OSError) as closed:
            executor.close()
        assert closed.value is first
        provider.close()
        original.close()
    assert original.pool.budget.used == 0


def test_fixed_cleanup_lane_accepts_no_selected_callback(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    try:
        with pytest.raises(TypeError):
            executor.close_final_validation(lambda: None)
    finally:
        executor.close()
        provider.close()
        original.close()
