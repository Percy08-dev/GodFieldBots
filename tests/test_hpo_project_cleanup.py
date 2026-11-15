"""No-I/O exception-order tests; stubs cannot authorize a Source or GPU update."""
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError


@pytest.mark.parametrize('used',[0,1])
@pytest.mark.parametrize('primary_fails,cleanup_fails',[(True,True),(True,False),(False,True)])
def test_project_preserves_primary_and_reachable_cleanup_owner(monkeypatch,tmp_path,used,primary_fails,cleanup_fails):
    import gf_training.hpo_ppo as module
    primary=RuntimeError('original optimizer/parity failure')
    cleanup=OSError('original batch cleanup failure')
    owner=object()
    handle=object()
    closed=[]
    rollout=SimpleNamespace(require_learning=lambda:None)
    class Rollout:
        def require_learning(self): return rollout.require_learning()
        def __len__(self): return 1
    source=SimpleNamespace(proof={'manifest_sha256':'a'*64},learner_label='learner',
        selection={'used_decisions':used},write_numeric=lambda _: {'data':{'files':{}}},
        read_numeric=lambda *_:Rollout(),issue=lambda *_:handle)
    @contextmanager
    def batch(*args,**kwargs): yield source
    monkeypatch.setattr(module,'hpo_batch_source',batch)
    monkeypatch.setattr('gf_models.arrays.sha256',lambda *_:'a'*64)
    operations=object.__new__(module._HpoOperations)
    operations.ledger=object()
    operations.online=SimpleNamespace(packed_batch=owner)
    operations.projection=None
    operations.resident_runtime=None  # Explicit fresh mode; this fixture bypasses __init__.
    operations.cap=lambda _:1024
    def close():
        closed.append(owner)
        operations.online.packed_batch=None  # A failed close may have changed its slot.
        if cleanup_fails: raise cleanup
    operations.close_batch=close
    expected=primary if primary_fails else cleanup
    with (pytest.raises(type(expected)) as raised,
            operations.project(None,tmp_path/'numeric',None,'learner',0,time.monotonic()+30,
                expected_manifest_sha256='a'*64) as projected):
        assert projected[0]==used
        if primary_fails: raise primary
    assert raised.value is expected
    assert operations.projection is None and closed==[owner]
    if cleanup_fails:
        assert raised.value.cleanup_owner is owner
        if primary_fails:
            assert any('batch cleanup failure' in note for note in primary.__notes__)
        operations.job=SimpleNamespace(confirm_termination=lambda:pytest.fail('unresolved cleanup confirmed termination'))
        with pytest.raises(ContractError,match='uncertain'):
            operations.finish_cleanup(raised.value)


def test_existing_primary_cleanup_owner_is_not_lost_on_batch_close_failure(monkeypatch,tmp_path):
    import gf_training.hpo_ppo as module
    primary=RuntimeError('primary')
    first=object()
    batch_owner=object()
    primary.cleanup_owner=first
    @contextmanager
    def batch(*args,**kwargs):
        yield SimpleNamespace(proof={'manifest_sha256':'a'*64},learner_label='learner',selection={'used_decisions':0})
    monkeypatch.setattr(module,'hpo_batch_source',batch)
    operations=object.__new__(module._HpoOperations)
    operations.ledger=object()
    operations.online=SimpleNamespace(packed_batch=batch_owner)
    operations.projection=None
    operations.cap=lambda _:1024
    def close(): raise OSError('close')
    operations.close_batch=close
    with (pytest.raises(RuntimeError) as raised,
            operations.project(None,tmp_path/'numeric',None,'learner',0,time.monotonic()+30,
                expected_manifest_sha256='a'*64)):
        raise primary
    assert raised.value is primary and primary.cleanup_owner is first
    assert primary._hpo_batch_cleanup_owner is batch_owner
