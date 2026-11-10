from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from tests.test_group_pack_game_source import owner, packed


def test_real_sqlite_alternates_caller_worker_and_release_requires_actual_join(tmp_path):
    from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
    root, _ = packed(tmp_path)
    original = owner(root)
    original.read_raw(root / 'first/e0')
    provider = OriginalArtifactProvider((original,))
    assert original.pool.active is None
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    with pytest.raises(ContractError, match='executor'):
        provider.close()
    caller = threading.get_ident()
    def work():
        assert threading.get_ident() != caller
        lease = provider.borrow(str(root), 'first/e0')
        assert lease.data() == b'\xff\x00'
        lease.close()
        return threading.get_ident()
    worker = executor.call(work)
    assert original.pool.active is None
    original.read_raw(root / 'second/e0')
    assert executor.call(work) == worker
    assert original.pool.active is None
    executor.close()
    provider.close()
    original.close()
    assert original.pool.budget.used == 0


def test_worker_close_uncertainty_keeps_actual_worker_and_outer_provider(tmp_path, monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    held = []
    def work():
        lease = provider.borrow(str(root), 'first/e0')
        lease.close()
        actual = original.artifacts._originals.db
        class Fault:
            def close(self):
                raise OSError('creator close uncertain')
        original.artifacts._originals.db = Fault()
        held.append(actual)
    with pytest.raises(OSError, match='uncertain'):
        executor.call(work)
    with pytest.raises(OSError, match='uncertain'):
        executor.close()
    assert not executor._closed
    with pytest.raises(ContractError, match='executor'):
        provider.close()
    # Repair only the injected wrapper; cleanup retries the SAME actual SQLite
    # on the SAME live executor. No Source health or normal work is restored.
    original.artifacts._originals.db = held[0]
    with pytest.raises(OSError, match='uncertain'):
        executor.close()
    assert executor._closed and original.pool.active is None
    provider.close()
    original.close()


@pytest.mark.parametrize('after', [False, True])
def test_later_caller_close_failure_stays_on_creator_and_cleanup_never_reads(tmp_path, monkeypatch, after):
    from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    executor.call(lambda: None)
    original.read_raw(root / 'first/e0')
    actual = original.artifacts._originals.db
    caller = threading.get_ident()
    seen = []
    class Fault:
        def close(self):
            seen.append(threading.get_ident())
            if after:
                actual.close()
            raise OSError('caller close fault')
    original.artifacts._originals.db = Fault()
    with pytest.raises(OSError, match='caller close'):
        executor.call(lambda: pytest.fail('failed caller IO ownership transferred'))
    assert seen == [caller] and executor._creator == 'caller'
    monkeypatch.setattr(original, 'finalize', lambda: pytest.fail('cleanup performed original read/hash'))
    with pytest.raises(OSError):
        executor.close()
    assert not executor._closed and all(thread == caller for thread in seen)
    original.artifacts._originals.db = actual
    with pytest.raises(OSError):
        executor.close()
    assert executor._closed
    provider.close()
    original.close()


def test_async_cancel_joins_actual_callback_and_stays_cleanup_only(tmp_path):
    import asyncio

    from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    started, cancelled = threading.Event(), threading.Event()
    executor = _PackedNativeExecution(SimpleNamespace(cancel=cancelled.set), provider)
    def work():
        lease = provider.borrow(str(root), 'first/e0')
        started.set()
        assert cancelled.wait(5)
        lease.close()
    async def run():
        task = asyncio.create_task(executor.acall(work))
        async with asyncio.timeout(6):
            while not started.is_set():
                if task.done():
                    await task
                await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert executor._active is None and original.pool.active is None and not provider._leases
        with pytest.raises(ContractError, match='cleanup-only'):
            executor.call(lambda: None)
        with pytest.raises(asyncio.CancelledError):
            await executor.aclose()
        assert executor._closed
    try:
        asyncio.run(run())
    finally:
        if not executor._closed:
            try:
                executor.close()
            except BaseException:  # noqa: BLE001, S110 -- lab preserves the first failure and attempts checked retirement.
                pass
        if executor._closed:
            provider.close()
            original.close()


def test_preargument_account_tracks_growth_after_populated_cache_and_replaces_last_hold(tmp_path):
    import time

    from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
    from gf_common.group_pack_game import storage_binding
    from gf_common.preprocessing import PreparationConfig
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    original.read_raw(root / 'first/e0')
    assert original.artifacts._cache_charge > 0
    provider = OriginalArtifactProvider((original,))
    assert original.pool.active is None and original.artifacts._cache_charge == 0
    plan = {'schema_version': 4, 'mode': 'historical_parent',
        'parents': [{'directory': str(root.parent), 'require_complete': False}], 'collections': [],
        'storage_binding': storage_binding(), 'artifact_registry_sha256': provider.registry_sha256}
    inputs = _NativeGameInputs(plan, PreparationConfig(backend='rust', reader='participant-structured-v2'),
        actions=ActionConfigV3(action_contract=corrected_card_instance_contract()), max_owned_bytes=1024**3,
        deadline=time.monotonic()+30, artifact_provider=provider)
    assert inputs.budget == inputs.limit - inputs.extra - inputs.base
    inputs._executor = inputs._make_executor(SimpleNamespace(cancel=lambda: None))
    class Native:
        held = 0
        def used_bytes(self):
            return 64 + self.held
        def recheck(self, held):
            self.held = held
        def close(self):
            pass
    native = Native()
    inputs._session = native
    inputs._executor.bind_native(native)
    original.charge(32)  # Smaller than the retired caller cache, still positive D.
    inputs.budget = 64 + 32 + 17
    built = []
    def builder(held):
        built.append(held)
        return (held,)
    inputs._native_call(native.recheck, builder, envelope=17)
    inputs._native_call(native.recheck, builder, envelope=17)
    assert built == [49, 49] and inputs._last_held == 49
    inputs.budget -= 1
    with pytest.raises(ContractError, match='BEFORE'):
        inputs._native_call(native.recheck, lambda _: pytest.fail('encoding preceded admission'), envelope=17)
    assert inputs._last_held == 49
    with pytest.raises(ContractError):
        inputs.close()
    assert inputs.cleanup_complete
    original.close()


def test_failed_native_close_fences_callbacks_and_confirms_outer_retirement_separately(tmp_path, monkeypatch):
    from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    cancelled = []
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: cancelled.append(True)), provider)
    reads = []
    monkeypatch.setattr(original, 'finalize', lambda: reads.append(True))
    class Native:
        closed = False
        calls = 0
        def close(self):
            self.calls += 1
            if self.closed:
                return
            assert cancelled and provider._fenced
            with pytest.raises(ContractError, match='fenced'):
                provider.recheck(str(root))
            self.closed = True  # Model actual Closed-with-error, not outer confirmation.
            raise RuntimeError('native failed check while closing')
    native = Native()
    executor.bind_native(native)
    failure = ValueError('original operation failed')
    with pytest.raises(ValueError):
        executor.call(lambda: (_ for _ in ()).throw(failure))
    with pytest.raises(ValueError) as first:
        executor.close()
    assert first.value is failure and native.closed and not executor._closed and not executor._native_retired
    with pytest.raises(ContractError, match='executor'):
        provider.close()
    with pytest.raises(ValueError) as second:
        executor.close()
    assert second.value is failure and executor._closed and executor._native_retired and native.calls == 2
    assert reads == []
    provider.close()
    original.close()
