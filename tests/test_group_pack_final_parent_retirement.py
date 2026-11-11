"""Actual physical cleanup owners on an unadmitted parent, never Source authority."""
from types import SimpleNamespace

import pytest

from gf_training.card_instance_formal import FormalGameParent
from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from gf_training.structured_lineage import _ChainScan
from tests.test_group_pack_game_source import owner, packed


def test_parent_retires_scan_originals_accounts_only_after_actual_executor_join(tmp_path, monkeypatch):
    root, _ = packed(tmp_path)
    original = owner(root)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    scan._packed_pool = original.pool
    scan._packed_owners[root] = original
    scan.charge(4096)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    # Start the actual worker and join it at parent retirement.
    assert executor.call(lambda: 1) == 1
    events = []
    close_original = original.close
    def close():
        assert executor._closed and executor._hold is None
        events.append('original')
        close_original()
    monkeypatch.setattr(original, 'close', close)
    class NativeRetirement:
        cleanup_complete = False
        def close(self):
            events.append('native')
            executor.close()
            events.append('joined')
            provider.close()
            self.cleanup_complete = True
    parent = FormalGameParent(tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    parent.scan = scan
    parent._native_inputs = NativeRetirement()
    try:
        assert parent._retire_inputs() is None
        assert scan._closed
        assert original.pool.closed and original.pool.budget.used == 0
        assert original.locator.fd < 0
        assert events[:3] == ['native', 'joined', 'original']
    finally:
        executor.close()
        provider.close()
        if not scan._closed:
            scan.close(recheck=False)


def test_failed_native_join_keeps_actual_scan_fds_pool_charge_and_parent_cleanup_owner(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    scan._packed_pool = original.pool
    scan._packed_owners[root] = original
    scan.charge(4096)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    class UncertainJoin:
        cleanup_complete = False
        def close(self):
            raise OSError('actual native join remains unconfirmed')
    parent = FormalGameParent(tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    parent.scan, parent._native_inputs = scan, UncertainJoin()
    before = original.pool.budget.used
    try:
        with pytest.raises(OSError, match='join') as error:
            parent._retire_inputs()
        assert vars(error.value)['_formal_parent_cleanup_owner'] is parent
        assert not scan._closed and original.pool.budget.used == before
        assert original.locator.fd >= 0 and executor._hold is not None
        assert not parent.cleanup_complete
    finally:
        executor.close()
        provider.close()
        scan.close(recheck=False)


def test_final_original_slot_retains_provider_wrapper_and_control_charge_under_same_actual_hold(tmp_path):
    from gf_training.card_instance_formal import _PackedFinalOriginals
    base_root, _ = packed(tmp_path / 'base')
    final_root, _ = packed(tmp_path / 'final')
    original = owner(base_root)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    executor.prepare()
    final = owner(final_root, original.pool)
    final_provider = OriginalArtifactProvider((final,))
    executor.attach_final_provider(final_provider)
    parent = FormalGameParent(tmp_path, max_owned_bytes=512 * 1024**2, deadline=original.deadline)
    slot = _PackedFinalOriginals(parent)
    slot.owner, slot.provider, slot.pool = final, final_provider, original.pool
    parent._packed_final_originals = slot
    before = original.pool.budget.used
    try:
        with pytest.raises(RuntimeError, match='hold'):
            slot.close()
        assert final.locator.fd >= 0 and original.pool.budget.used == before
        assert not slot.cleanup_complete
        executor.close()
        provider.close()
        slot.close()
        assert slot.cleanup_complete and final.locator.fd < 0
        assert not original.pool.closed
    finally:
        executor.close()
        provider.close()
        if not slot.cleanup_complete:
            slot.close()
        original.close()
    assert original.pool.budget.used == 0


@pytest.mark.parametrize('prefix', ['ucd', 'type', 'duplicate'])
def test_no_effect_partial_provider_cannot_block_actual_final_original_and_account_cleanup(tmp_path, monkeypatch, prefix):
    from gf_training.card_instance_formal import _PackedFinalOriginals
    base_root, _ = packed(tmp_path/'base')
    final_root, _ = packed(tmp_path/'final')
    original = owner(base_root)
    baseline = original.pool.budget.used
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    executor.prepare()
    final = owner(final_root, original.pool)
    parent = FormalGameParent(tmp_path, max_owned_bytes=512*1024**2, deadline=original.deadline)
    slot = _PackedFinalOriginals(parent)
    slot.pool, slot.owner = original.pool, final
    # Run an ACTUAL concrete constructor prefix. Every case precedes any
    # acquisition/IO/executor binding, without inventing a retired flag.
    import gf_training.native_group_pack as module
    from gf_common.domain import ContractError
    slot.provider = OriginalArtifactProvider.__new__(OriginalArtifactProvider)
    owners = (final, final) if prefix == 'duplicate' else (object(),) if prefix == 'type' else (final,)
    if prefix == 'ucd':
        monkeypatch.setattr(module.unicodedata, 'unidata_version', '99.0.0')
    with pytest.raises(ContractError):
        OriginalArtifactProvider.__init__(slot.provider, owners)
    assert not hasattr(slot.provider, '_acquisition_states')
    slot.pool.budget.acquire(65536, original.deadline)
    slot._controls = 65536
    try:
        with pytest.raises(RuntimeError, match='hold'):
            slot.close()
        executor.close()
        provider.close()
        slot.close()
        assert slot.cleanup_complete and final.locator.fd < 0
        assert original.pool.budget.used == baseline
    finally:
        executor.close()
        provider.close()
        if not slot.cleanup_complete:
            # Controlled disposal for the RED run, no production ownership inference.
            slot.provider = None
            slot.close()
        original.close()
    assert original.pool.budget.used == 0


def test_cleanup_only_public_source_facade_retires_while_actual_parent_hold_fds_and_metadata_remain(tmp_path, monkeypatch):
    from gf_training.card_instance_final_validation import GameFinalValidationSource
    from gf_training.card_instance_formal import _PackedFinalOriginals
    from gf_training.native_card_instance import _NativeFinalValidationInputs
    from tests.test_native_final_validation_forwarding import unadmitted_loan
    # Existing UNADMITTED resource primitive has an actual parent lock/exclusion,
    # no run.json, admitted Source, final-slot effects, records or fitting rights.
    parent, native, _, _, token, _, scope = unadmitted_loan(tmp_path, monkeypatch)
    base_root, _ = packed(tmp_path/'base')
    final_root, _ = packed(tmp_path/'public-final')
    original = owner(base_root)
    provider = OriginalArtifactProvider((original,))
    executor = _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
    executor.prepare()
    final = owner(final_root, original.pool)
    final_provider = OriginalArtifactProvider((final,))
    executor.attach_final_provider(final_provider)
    slot = _PackedFinalOriginals(parent)
    slot.pool, slot.owner, slot.provider = original.pool, final, final_provider
    parent._packed_final_originals = slot
    native._artifact_provider, native._executor = provider, executor
    loan = _NativeFinalValidationInputs(native,parent,token,deadline=parent.deadline)
    parent._native_final_inputs = loan
    source = GameFinalValidationSource(parent,None,None,{'scope':'UNADMITTED cleanup only','qualified':False},native=loan)
    loan.bind_source(source)
    source._management, source._borrow_token = final, token
    first = OSError('cleanup-only source failure')
    source._failure = first
    before = original.pool.budget.used
    descriptors = final.locator.fd, final.artifacts._originals.database_file.fd
    try:
        with pytest.raises(OSError) as error:
            source.close()
        assert error.value is first and source.cleanup_complete and loan.cleanup_complete
        assert executor._hold is not None and not executor._closed
        assert (final.locator.fd,final.artifacts._originals.database_file.fd) == descriptors
        assert original.pool.budget.used == before and not slot.cleanup_complete
        assert not (parent.directory/'run.json').exists()
        executor.close()
        provider.close()
        slot.close()
        assert slot.cleanup_complete and final.locator.fd < 0
    finally:
        executor.close()
        provider.close()
        if not slot.cleanup_complete:
            slot.close()
        original.close()
        scope.__exit__(None,None,None)
    assert original.pool.budget.used == 0
