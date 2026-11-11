"""Actual unadmitted wrapper accounts, without formal Source admission."""
from types import SimpleNamespace

import pytest

from gf_common.group_pack import SOURCE_BASE
from gf_common.group_pack_game import storage_binding
from gf_common.preprocessing import PreparationConfig
from gf_training.card_instance_formal import FormalGameParent, _PackedFinalOriginals
from gf_training.group_pack_game_source import PackedSourceOwner
from gf_training.group_pack_resources import StorageBudget
from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from tests.test_group_pack_game_source import owner, packed


@pytest.mark.parametrize('after', [False, True])
@pytest.mark.parametrize('context', ['ordinary-own', 'shared', 'final-slot'])
def test_actual_wrapper_initial_acquire_failure_keeps_exact_effect_and_retires_once(tmp_path, monkeypatch, after, context):
    final_root, _ = packed(tmp_path/'final')
    base = executor = provider = slot = None
    if context != 'ordinary-own':
        base_root, _ = packed(tmp_path/'base')
        base = owner(base_root)
    baseline = base.pool.budget.used if base is not None else 0
    if context == 'final-slot':
        provider = OriginalArtifactProvider((base,))
        executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
        executor.prepare()
        parent = FormalGameParent(tmp_path, max_owned_bytes=512*1024**2, deadline=base.deadline)
        slot = _PackedFinalOriginals(parent)
        parent._packed_final_originals = slot
        slot.pool = base.pool
    actual = PackedSourceOwner.__new__(PackedSourceOwner)
    if slot is not None:
        slot.owner = actual  # Actual preowned cleanup component, never admitted Source authority.
    acquire = StorageBudget.acquire
    first = MemoryError('wrapper initial acquisition fault')
    attempted = []
    def fault(budget, size, deadline, **options):
        if size == 65536 and not attempted:
            attempted.append(budget)
            if after:
                acquire(budget, size, deadline, **options)
            raise first
        return acquire(budget, size, deadline, **options)
    monkeypatch.setattr(StorageBudget, 'acquire', fault)
    try:
        with pytest.raises(MemoryError) as caught:
            PackedSourceOwner.__init__(actual, final_root, PreparationConfig(),
                __import__('time').monotonic()+30, storage_binding=storage_binding(),
                max_owned_bytes=512*1024**2, pool=base.pool if base is not None else None)
        assert caught.value is first and caught.value.cleanup_owner is actual
        if slot is not None:
            assert not actual._closed and not slot.cleanup_complete
            assert actual._charge_bytes == (65536 if after else 0)
            executor.close()
            provider.close()
            slot.close()
            assert slot.cleanup_complete
        assert actual._closed and actual._charge_bytes == 0
        if base is None:
            assert actual.pool.closed and actual.pool.budget.used == 0
        else:
            assert base.pool.budget.used == baseline and base.locator.fd >= 0
        actual.close()
        assert actual.pool.budget.used == baseline
    finally:
        monkeypatch.setattr(StorageBudget, 'acquire', acquire)
        if executor is not None:
            executor.close()
            provider.close()
        actual.close()
        if base is not None:
            base.close()


@pytest.mark.parametrize('after', [False, True])
@pytest.mark.parametrize('context', ['ordinary-own', 'shared', 'final-slot'])
def test_actual_wrapper_refund_failure_never_refunds_live_base_twice(tmp_path, monkeypatch, after, context):
    final_root, _ = packed(tmp_path/'final')
    base = slot = None
    if context != 'ordinary-own':
        base_root, _ = packed(tmp_path/'base')
        base = owner(base_root)
    baseline = base.pool.budget.used if base is not None else SOURCE_BASE
    actual = owner(final_root, base.pool if base is not None else None)
    if context == 'final-slot':
        parent = FormalGameParent(tmp_path, max_owned_bytes=512*1024**2, deadline=actual.deadline)
        slot = _PackedFinalOriginals(parent)
        slot.owner, slot.pool = actual, actual.pool
        parent._packed_final_originals = slot
    close = slot.close if slot is not None else actual.close
    amount = actual._charge_bytes
    release = actual.pool.budget.release
    first = OSError('wrapper refund fault')
    attempted = []
    def fault(size):
        if size == amount and not attempted:
            attempted.append(size)
            if after:
                release(size)
            raise first
        return release(size)
    monkeypatch.setattr(actual.pool.budget, 'release', fault)
    try:
        with pytest.raises(OSError) as caught:
            close()
        assert caught.value is first and not actual._closed
        assert actual._charge_bytes == (0 if after else amount)
        assert actual.pool.budget.used == baseline + (0 if after else amount)
        assert actual._failure is first
        monkeypatch.setattr(actual.pool.budget, 'release', release)
        close()
        close()
        assert actual._closed and actual._charge_bytes == 0
        if base is not None:
            assert base.locator.fd >= 0 and actual.pool.budget.used == baseline
        else:
            assert actual.pool.closed and actual.pool.budget.used == 0
    finally:
        monkeypatch.setattr(actual.pool.budget, 'release', release)
        actual.close()
        if base is not None:
            base.close()
    assert actual.pool.budget.used == 0


@pytest.mark.parametrize('after', [False, True])
@pytest.mark.parametrize('operation', ['late-acquire', 'decoded-refund'])
def test_later_wrapper_charge_stages_keep_effect_evidence_and_first_error(tmp_path, monkeypatch, after, operation):
    root, _ = packed(tmp_path)
    actual = owner(root)
    if operation == 'decoded-refund':
        actual.read(root/'manifest.json')
    budget = actual.pool.budget
    before, charge = budget.used, actual._charge_bytes
    amount = 1234 if operation == 'late-acquire' else max(0, charge - (
        65536 + 1024*len(actual.files) + sum(len(str(key))*32 for key in actual.files)))
    assert amount > 0
    original = budget.acquire if operation == 'late-acquire' else budget.release
    first = OSError('later wrapper account stage')
    def fault(size, *args, **options):
        assert size == amount
        if after:
            original(size, *args, **options)
        raise first
    attribute = 'acquire' if operation == 'late-acquire' else 'release'
    monkeypatch.setattr(budget, attribute, fault)
    try:
        with pytest.raises(OSError) as caught:
            actual.charge(amount) if operation == 'late-acquire' else actual.retire_decoded()
        assert caught.value is first and actual._failure is first
        delta = amount if operation == 'late-acquire' else -amount
        assert budget.used == before + (delta if after else 0)
        assert actual._charge_bytes == charge + (delta if after else 0)
        with pytest.raises(OSError) as sticky:
            actual.read(root/'manifest.json')
        assert sticky.value is first
    finally:
        monkeypatch.setattr(budget, attribute, original)
        actual.close()
    assert budget.used == 0 and actual.pool.closed


@pytest.mark.parametrize('operation', ['acquire', 'release'])
def test_unknown_wrapper_effect_preserves_actual_account_and_cleanup_owner(tmp_path, monkeypatch, operation):
    root, _ = packed(tmp_path)
    actual = owner(root)
    budget = actual.pool.budget
    before, charge = budget.used, actual._charge_bytes
    original = getattr(budget, operation)
    first = OSError('partial account effect')
    def partial(size, *args, **options):
        original(17, *args, **options)
        raise first
    monkeypatch.setattr(budget, operation, partial)
    try:
        with pytest.raises(OSError) as caught:
            actual.charge(1234) if operation == 'acquire' else actual.retire_decoded()
        assert caught.value is first and first.cleanup_owner is actual
        assert actual._charge_uncertain and not actual._closed
        assert actual._charge_bytes == charge
        assert budget.used == before + (17 if operation == 'acquire' else -17)
        monkeypatch.setattr(budget, operation, original)
        with pytest.raises(OSError) as retirement:
            actual.close()
        assert retirement.value is first and not actual._closed
        retained = budget.used
        with pytest.raises(OSError):
            actual.close()
        assert budget.used == retained
        assert not actual.pool.closed
    finally:
        monkeypatch.setattr(budget, operation, original)
        # Test-only disposal of exactly the concrete injected reservations. The
        # production owner keeps uncertainty and never claims terminal cleanup.
        actual.pool.budget.release(budget.used - SOURCE_BASE)
        actual.pool.close()
    assert actual._charge_uncertain and not actual._closed
