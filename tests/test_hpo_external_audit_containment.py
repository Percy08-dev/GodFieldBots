"""Private external audit capacity loans, actual last use and exact refunds."""
import gc
import time
from types import SimpleNamespace

import pytest

GIB=1024**3
MIB=1024**2
NUMERIC=583949647


def prepared(tmp_path,external=4*GIB,original=2*GIB,model_limit=12*GIB):
    from gf_common.group_pack_game import storage_binding
    from gf_training.hpo_owned_restore import _prepare_external_audit
    from gf_training.structured_lineage import _ChainScan
    state={'schema_version':4,'storage_binding':storage_binding(),'effective_config':{
        'storage':{'budget_mib':original//MIB},'preprocessing':{'memory_budget_mib':original//MIB}}}
    scan=_ChainScan(directory=tmp_path,max_owned_bytes=external,deadline=time.monotonic()+30)
    scan.configure_packed(state)
    scan.charge(8*MIB)
    owner=_prepare_external_audit(scan,state,SimpleNamespace(resources=SimpleNamespace(max_owned_bytes=model_limit)))
    return scan,owner,state


@pytest.mark.parametrize('external',[GIB,2*GIB,4*GIB])
def test_flat_nested_and_equality_preserve_original_pool_capacity(tmp_path,external):
    scan,owner,_=prepared(tmp_path,external)
    pool=scan.ensure_packed_pool()
    assert pool.budget.limit==min(external,2*GIB)
    assert owner.nested is (external>2*GIB)
    if owner.nested:
        assert owner.account.limit==external and owner.account.used==2*GIB
    else:assert owner.account is None
    assert scan._pool_control_charge==scan.retained  # Controls funded in this child exactly once.
    scan.close(recheck=False)
    assert scan._closed and pool.closed
    if owner.nested:assert owner.account.used==0


def test_model_cap_also_selects_flat_path_without_resizing_child(tmp_path):
    scan,owner,_=prepared(tmp_path,4*GIB,model_limit=GIB)
    assert not owner.nested and scan.ensure_packed_pool().budget.limit==GIB
    scan.close(recheck=False)


def numeric_owner(scan,owner):
    from gf_training.hpo_owned_restore import (
        _AuditCohortAnchor,
        _AuditCohortRetirement,
        _AuditNumericPeak,
    )
    pool=scan.ensure_packed_pool()
    original=SimpleNamespace(pool=pool,check=lambda:None)
    retirement=_AuditCohortRetirement(original)
    peak=_AuditNumericPeak(original,capacity=owner)
    retirement.add_numeric(peak)
    anchor=_AuditCohortAnchor(retirement)
    return peak,retirement,anchor


def test_four_two_split_numeric_absolute_peak_and_child_growth_are_not_added_twice(tmp_path):
    scan,owner,_=prepared(tmp_path)
    peak,retirement,anchor=numeric_owner(scan,owner)
    pool=scan._packed_pool
    child_before=pool.budget.used
    peak(NUMERIC);peak(NUMERIC);peak(NUMERIC+4096)
    assert owner.account.used==2*GIB+NUMERIC+4096
    assert pool.budget.used==child_before
    pool.budget.acquire(64*MIB,scan.deadline)
    assert owner.account.used==2*GIB+NUMERIC+4096
    pool.budget.release(64*MIB)
    del anchor
    gc.collect()
    retirement.require_retired()
    assert owner.account.used==2*GIB
    scan.close(recheck=False)
    assert owner.account.used==0


def test_aggregate_numeric_refusal_precedes_array_allocation_and_keeps_original_loan(tmp_path):
    scan,owner,_=prepared(tmp_path,external=2*GIB+NUMERIC-1)
    peak,retirement,anchor=numeric_owner(scan,owner)
    with pytest.raises(ValueError):peak(NUMERIC)
    assert peak.peak==0 and owner.account.used==2*GIB
    del anchor
    gc.collect()
    retirement.require_retired()
    scan.close(recheck=False)
    assert owner.account.used==0


@pytest.mark.parametrize('borrowed',[False,True])
def test_existing_or_borrowed_pool_never_gets_resized_or_a_new_loan(tmp_path,borrowed):
    from gf_training.hpo_owned_restore import _prepare_external_audit
    scan,_owner,state=prepared(tmp_path,external=2*GIB)
    pool=scan.ensure_packed_pool()
    scan._owned_audit_capacity=None
    scan.limit=4*GIB
    scan._owns_packed_pool=not borrowed
    assert _prepare_external_audit(scan,state,SimpleNamespace(resources=SimpleNamespace(max_owned_bytes=12*GIB))) is None
    assert pool.budget.limit==2*GIB and scan._owned_audit_capacity is None
    scan.close(recheck=False)
    if borrowed:
        assert not pool.closed
        pool.close()


@pytest.mark.parametrize('traceback',[False,True])
def test_decoded_alias_or_failure_traceback_retains_numeric_and_original_capacity(tmp_path,traceback):
    import numpy as np
    scan,owner,_=prepared(tmp_path)
    peak,retirement,anchor=numeric_owner(scan,owner)
    peak(NUMERIC)
    decoded=np.arange(4,dtype=np.float32)
    anchor.bind(decoded)
    if traceback:
        def failed(value):raise ValueError('original decoded failure')
        try:failed(decoded)
        except ValueError as caught:error=caught
        del decoded,anchor
    else:
        alias=decoded[1:]
        del decoded,anchor
    gc.collect()
    with pytest.raises((BufferError,RuntimeError)):scan.close(recheck=False)
    assert not scan._closed and owner.account.used==2*GIB+NUMERIC
    if traceback:error.__traceback__=None;del error
    else:del alias
    gc.collect()
    retirement.require_retired()
    scan.close(recheck=False)
    assert owner.account.used==0


@pytest.mark.parametrize('stage',['pool','controls','loan','numeric'])
@pytest.mark.parametrize('after_effect',[False,True])
def test_failed_cleanup_keeps_actual_owner_and_retry_refunds_once(tmp_path,monkeypatch,stage,after_effect):
    scan,owner,_=prepared(tmp_path)
    pool=scan.ensure_packed_pool()
    peak=retirement=anchor=None
    if stage=='numeric':
        peak,retirement,anchor=numeric_owner(scan,owner)
        peak(NUMERIC)
    calls=[]
    if stage=='pool':
        original=pool.close
        def failed():
            calls.append('pool')
            if after_effect:original()
            raise OSError('injected pool close')
        monkeypatch.setattr(pool,'close',failed)
    else:
        budget=pool.budget if stage=='controls' else owner.account
        original=budget.release
        target=scan._pool_control_charge if stage=='controls' else NUMERIC if stage=='numeric' else 2*GIB
        def failed(amount):
            if amount!=target:return original(amount)
            calls.append(amount)
            if after_effect:original(amount)
            raise OSError('injected '+stage+' release')
        monkeypatch.setattr(budget,'release',failed)
    if stage=='numeric':
        del anchor
        gc.collect()
        with pytest.raises(OSError):retirement.require_retired()
        assert owner.account.used==2*GIB+(0 if after_effect else NUMERIC)
    else:
        with pytest.raises(OSError):scan.close(recheck=False)
        assert not scan._closed
        if stage in ('pool','controls'):assert owner.account.used==2*GIB
    monkeypatch.undo()
    scan.close(recheck=False)
    assert scan._closed and owner.account.used==0
    before=len(calls)
    scan.close(recheck=False)
    assert len(calls)==before


def test_full_original_loan_is_attached_and_admitted_before_concrete_pool_construction(tmp_path,monkeypatch):
    from gf_training.group_pack_pool import PackedArtifactPool
    scan,owner,_=prepared(tmp_path)
    initialize=PackedArtifactPool.__init__
    def failed(pool,**options):
        assert scan._owned_audit_capacity is owner and owner.loan.amount==2*GIB
        initialize(pool,**options)
        raise OSError('after concrete pool allocation')
    monkeypatch.setattr(PackedArtifactPool,'__init__',failed)
    with pytest.raises(OSError) as caught:scan.ensure_packed_pool()
    assert caught.value.cleanup_owner is scan
    assert owner.account.used==2*GIB and not scan._packed_pool.closed
    scan.close(recheck=False)
    assert owner.account.used==0


def test_parent_admission_effect_before_error_remains_owned_and_refunds_once(tmp_path,monkeypatch):
    scan,owner,_=prepared(tmp_path)
    acquire=owner.account.acquire
    def failed(size,deadline,**options):
        acquire(size,deadline,**options)
        raise OSError('after parent original loan admission')
    monkeypatch.setattr(owner.account,'acquire',failed)
    with pytest.raises(OSError) as caught:scan.ensure_packed_pool()
    assert caught.value.cleanup_owner is scan and scan._packed_pool is None
    assert owner.account.used==owner.loan.amount==2*GIB
    scan.close(recheck=False)
    assert owner.account.used==0
    scan.close(recheck=False)
    assert owner.account.used==0


def test_actual_native_cleanup_failure_retains_scan_loan_until_positive_close(tmp_path):
    import os
    scan,owner,_=prepared(tmp_path)
    pool=scan.ensure_packed_pool()
    reader,writer=os.pipe()
    class Native:
        failed=True
        closed=False
        def close(self):
            if self.closed:return
            if self.failed:raise OSError('native originals still running')
            os.close(reader);os.close(writer)
            self.closed=True
    native=Native()
    pool._cleanup_children.append(native)
    try:
        with pytest.raises(OSError):scan.close(recheck=False)
        assert not scan._closed and owner.account.used==2*GIB
        os.fstat(reader);os.fstat(writer)
        native.failed=False
        scan.close(recheck=False)
        assert native.closed and owner.account.used==0
        with pytest.raises(OSError):os.fstat(reader)
    finally:
        if not native.closed:
            native.failed=False
            native.close()
