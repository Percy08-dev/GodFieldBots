"""Offline audit reserves one real decoded lifetime and retires only at last use."""
import gc
import time
import weakref

import numpy as np
import pytest


def originals(tmp_path):
    from gf_training.group_pack_pool import PackedArtifactPool
    from tests.test_group_pack_game_source import owner, packed
    envelope,_=packed(tmp_path)
    pool=PackedArtifactPool(max_owned_bytes=512*1024**2,deadline=time.monotonic()+30)
    return owner(envelope,pool=pool),pool


def test_audit_numeric_reserve_is_absolute_highwater_and_failed_growth_does_not_advance(tmp_path):
    from gf_training.hpo_owned_restore import _AuditNumericPeak
    original,pool=originals(tmp_path)
    try:
        before=pool.budget.used
        reserve=_AuditNumericPeak(original)
        reserve(100000); reserve(100000); reserve(10); reserve(150000)
        assert pool.budget.used-before==150000 and reserve.peak==150000
        limit=pool.budget.limit; pool.budget.limit=pool.budget.used
        with pytest.raises(ValueError): reserve(200000)
        assert reserve.peak==150000 and pool.budget.used-before==150000
        pool.budget.limit=limit
    finally:
        original.close(); pool.close()
    assert pool.budget.used==0


def test_audit_success_last_graph_use_precedes_refund_and_metadata_floor_survives(tmp_path):
    from gf_training.hpo_owned_restore import _AuditCohortAnchor, _AuditCohortRetirement
    original,pool=originals(tmp_path)
    graph_refs=[]
    metadata=original.object(original.root/'manifest.json')
    floor=original._control_object_charge
    retirement=_AuditCohortRetirement(original)
    def work():
        anchor=_AuditCohortAnchor(retirement)
        original.charge(4*1024**2)
        array=np.ones(16,np.float32); anchor.bind(array); graph_refs.append(weakref.ref(array))
        assert not retirement.retired
        return 2,3
    assert work()==(2,3)
    assert graph_refs[0]() is None and retirement.retired
    before=original._charge_bytes
    original.retire_decoded()
    assert original._charge_bytes<before and original._charge_bytes>=floor
    assert metadata=={'hello':'original'} and original._control_object_charge==floor
    assert not original._closed and original.locator.fd>=0
    original.finalize()
    original.close(); pool.close()
    assert pool.budget.used==0


def test_failed_audit_trace_and_array_alias_block_enclosing_scan_refund(tmp_path):
    from gf_training.hpo_owned_restore import _AuditCohortAnchor, _AuditCohortRetirement
    from gf_training.structured_lineage import _ChainScan
    original,pool=originals(tmp_path)
    scan=_ChainScan(directory=tmp_path,max_owned_bytes=512*1024**2,deadline=time.monotonic()+30)
    scan._packed_pool=pool; scan._packed_owners[original.root]=original
    aliases=[]; refs=[]; primary=ValueError('original audit validation')
    retirement=_AuditCohortRetirement(original)
    def work():
        anchor=_AuditCohortAnchor(retirement)
        original.charge(4*1024**2)
        array=np.ones(16,np.float32); anchor.bind(array)
        aliases.append(array); refs.append(weakref.ref(array))
        raise primary
    with pytest.raises(ValueError) as failed, scan: work()
    assert failed.value is primary and failed.value.cleanup_owner is scan
    assert not scan._closed and not original._closed and not retirement.retired
    charged=pool.budget.used
    with pytest.raises(RuntimeError): scan.close(recheck=False)
    assert pool.budget.used==charged
    failed.value.__traceback__=None; del failed
    primary.__traceback__=None; gc.collect()
    assert not retirement.retired and refs[0]() is not None and pool.budget.used==charged
    aliases.clear(); gc.collect()
    assert retirement.retired and refs[0]() is None
    scan.close(recheck=False)
    assert scan._closed and original._closed and pool.budget.used==0
