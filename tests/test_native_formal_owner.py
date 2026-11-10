"""UNADMITTED ownership/budget primitives only, never a formal Source/runtime."""
import time
from types import SimpleNamespace

import pytest

from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training.card_instance_formal import FormalGameParent
from gf_training.lineage import _OWNED_LOCKS, run_lock
from gf_training.native_card_instance import _NativeGameInputs
from tests.test_formal_ppo_control import unadmitted_parent_lifecycle


def test_intrinsic_and_external_budget_are_separate_and_failure_has_cleanup_only_zero(tmp_path,monkeypatch):
    directory=tmp_path/'unadmitted'
    with run_lock(directory,create=True) as ownership:
        owner,_,_,checks=unadmitted_parent_lifecycle(directory,ownership,monkeypatch)
        intrinsic=owner.resident_bytes
        owner.admit_external(4096)
        assert owner.resident_bytes==intrinsic and owner._external_bytes==4096
        before=len(checks)
        owner.admit_external(0)
        assert len(checks)==before and owner._external_bytes==0
        with pytest.raises(ContractError,match='joint'):
            owner.admit_external(owner.limit-intrinsic-owner._guard_scratch()+1)
        owner.admit_external(0)
        with pytest.raises(ContractError,match='poisoned'):
            owner.admit_external(1)
        with pytest.raises(ContractError,match='poisoned'):
            _=owner.resident_bytes
        assert not owner.cleanup_complete
        with pytest.raises(ContractError):
            owner.__exit__(None,None,None)
        assert owner.cleanup_complete


def test_expired_parent_allows_only_zero_memory_release_and_cleanup_confirmation(tmp_path,monkeypatch):
    directory=tmp_path/'unadmitted'
    with run_lock(directory,create=True) as ownership:
        owner,_,_,checks=unadmitted_parent_lifecycle(directory,ownership,monkeypatch)
        owner.admit_external(100)
        owner.deadline=time.monotonic()-1
        before=len(checks)
        owner.admit_external(0)
        assert len(checks)==before
        with pytest.raises(TimeoutError):
            owner.admit_external(1)
        with pytest.raises(ContractError,match='poisoned'):
            owner.final_binding()
        owner.admit_external(0)
        with pytest.raises(TimeoutError):
            owner.__exit__(None,None,None)
        assert owner.cleanup_complete
        with pytest.raises(ContractError):
            owner.admit_external(False)


def test_unconfirmed_native_retirement_keeps_actual_parent_lock_until_retry(tmp_path,monkeypatch):
    import gf_training.card_instance_source as source
    directory=tmp_path/'unadmitted-owned-lock'
    owner=FormalGameParent(directory,max_owned_bytes=1024**3,deadline=time.monotonic()+60)
    lock=owner._scope.enter_context(run_lock(directory,create=True))
    owner._owns_lock=True;owner._token=lock.token;owner._entered=owner._active=True
    owner.scan=SimpleNamespace(retained=0,files={},inventories={})
    monkeypatch.setattr(source,'recheck_game_parent_snapshot',lambda scan:None)
    events=[]
    first=OSError('unconfirmed native core/spool')
    class Native:
        cleanup_complete=False
        allowed=False
        def close(self):
            assert _OWNED_LOCKS[directory]==lock.token
            events.append('native-close')
            if not self.allowed:raise first
            self.cleanup_complete=True
            events.append('native-confirmed')
    native=Native();owner._native_inputs=native
    owner._inputs_scope.callback(lambda:events.append('inputs-stack'))
    with pytest.raises(OSError) as failed:
        owner.__exit__(None,None,None)
    assert failed.value is first and first._formal_parent_cleanup_owner is owner
    assert _OWNED_LOCKS[directory]==lock.token and not owner.cleanup_complete
    native.allowed=True
    with pytest.raises(OSError) as retried:
        owner.__exit__(None,None,None)
    assert retried.value is first and directory not in _OWNED_LOCKS and owner.cleanup_complete
    assert events==['inputs-stack','native-close','native-close','native-confirmed']


def test_native_cleanup_confirmation_uses_actual_session_executor_even_expired(tmp_path):
    native=_NativeGameInputs({'schema_version':3,'mode':'historical_parent','parents':[],'collections':[]},
        PreparationConfig(backend='rust',reader='participant-structured-v2'),
        actions=ActionConfigV3(action_contract=corrected_card_instance_contract()),max_owned_bytes=1024**3,
        deadline=time.monotonic()+60)
    native._session=object();native._executor=SimpleNamespace(_closed=False,_active=None)
    native._closed=True;native.deadline=time.monotonic()-1
    assert not native.cleanup_complete
    native._session_closed=True
    assert not native.cleanup_complete
    native._executor._closed=True
    assert native.cleanup_complete
    native._executor._active=object()
    assert not native.cleanup_complete


def test_same_native_owner_is_retained_by_parent_context_until_scope_exit(tmp_path,monkeypatch):
    import gf_training.card_instance_source as source
    import gf_training.native_card_instance as native_module
    from gf_common.domain import digest
    root=tmp_path/'unadmitted-traversal';root.mkdir()
    actions=ActionConfigV3(action_contract=corrected_card_instance_contract())
    state={'schema_version':3,'identity':{'fixture':'NOT-runtime'},'effective_config':{'preprocessing':{
        **PreparationConfig().__dict__,'backend':'rust','reader':'participant-structured-v2'}},'seed':17}
    # This seam fixture bypasses all admission; it tests only object ownership.
    rng=__import__('numpy').random.Generator(__import__('numpy').random.PCG64(17))
    records=[{'parent_controls':{'schedule_opponent_rng':rng.bit_generator.state,'snapshot_pool':[0],'collection_position':0}}]
    events=[];held=[]
    class Scan:
        def __init__(self):
            self.root=root
            self.limit=1024**3
            self.retained=4096
            self.files={root/'run.json':((), 'a'*64)}
        def finalize(self):events.append('python-final')
    class Native:
        def __init__(self,*args,**kwargs):held.append(self)
        def __enter__(self):
            self.summary={'parent':{'identity_sha256':digest(state['identity']),'run_sha256':'a'*64,'state':{}}}
            events.append('native-enter');return self
        def __exit__(self,*args):events.append('native-retire-join')
    monkeypatch.setattr(source,'_ChainScan',Scan)
    monkeypatch.setattr(source,'_validate_component_chain',lambda *args,**kwargs:(state,records))
    monkeypatch.setattr(source,'_state',lambda *args,**kwargs:(actions,None))
    monkeypatch.setattr(native_module,'_NativeGameInputs',Native)
    with source._open_game_parent_inputs(root,max_owned_bytes=1024**3,deadline=time.monotonic()+60,_scan=Scan()) as (actual,_,native):
        assert actual is state and native is held[0] and 'native-retire-join' not in events
    assert events[-1]=='native-retire-join'


def test_native_joint_resident_excludes_external_once_and_zero_keeps_failed_lease_conservative(tmp_path,monkeypatch):
    directory=tmp_path/'unadmitted-native-accounting'
    with run_lock(directory,create=True) as ownership:
        owner,_,_,_=unadmitted_parent_lifecycle(directory,ownership,monkeypatch)
        owner._native_scan_base=0;owner.scan.retained=2000
        class Native:
            growth=0
            cleanup_complete=False
            def __init__(self):self.calls=[]
            def caller_growth(self,value):self.growth=value;self.calls.append(('caller',value))
            def recheck(self):self.calls.append(('guard',self.growth))
            @property
            def joint_owned_bytes(self):return 100*1024**2+self.growth
            def close(self):self.cleanup_complete=True;self.calls.append(('closed',0))
        native=Native();owner._native_inputs=native
        intrinsic=owner.resident_bytes
        owner.admit_external(4096)
        assert owner.resident_bytes==intrinsic==100*1024**2+2000
        assert native.growth==2000+4096
        initiating=ContractError('unadmitted original guard failed')
        owner._poison(initiating)
        before=list(native.calls)
        owner.admit_external(0)
        assert native.calls==before and native.growth==2000+4096 and not owner.cleanup_complete
        with pytest.raises(ContractError,match='poisoned'):
            owner.admit_external(1)
        with pytest.raises(ContractError) as closed:
            owner.__exit__(None,None,None)
        assert closed.value is initiating and owner.cleanup_complete


def test_nonzero_external_over_total_limit_poison_is_not_a_retryable_smaller_claim(tmp_path,monkeypatch):
    directory=tmp_path/'unadmitted-total-limit'
    with run_lock(directory,create=True) as ownership:
        owner,_,_,_=unadmitted_parent_lifecycle(directory,ownership,monkeypatch)
        with pytest.raises(ContractError,match='joint'):
            owner.admit_external(owner.limit+1)
        owner.admit_external(0)
        with pytest.raises(ContractError,match='poisoned'):
            owner.admit_external(1)
        with pytest.raises(ContractError):
            owner.__exit__(None,None,None)
        assert owner.cleanup_complete


def test_formal_preflight_fresh_interpreter_blocks_optional_imports(tmp_path):
    import subprocess
    import sys
    code = '''
import importlib.abc,sys
class BlockOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','lightgbm'):raise AssertionError('optional import '+name)
sys.meta_path.insert(0,BlockOptional())
from tests.test_formal_ppo_contract import config_for,loaded_for
from gf_training.card_instance_formal import formal_preflight
for model in ('mlp','gru','transformer'):
 formal_preflight(loaded_for(config_for(model)))
assert 'torch' not in sys.modules and 'lightgbm' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30,check=False)
    assert result.returncode==0,result.stderr


def test_failed_parent_inputs_receive_original_error_and_cannot_finalize_read(tmp_path,monkeypatch):
    from contextlib import contextmanager
    directory=tmp_path/'unadmitted-error-handoff'
    with run_lock(directory,create=True) as ownership:
        owner,_,_,_=unadmitted_parent_lifecycle(directory,ownership,monkeypatch)
        original=ContractError('initiating metadata failure')
        events=[]
        @contextmanager
        def inputs():
            try:
                yield None
            except BaseException as error:
                events.append(('cleanup-only',error))
                raise
            else:
                pytest.fail('failed parent attempted successful final-read handoff')
        owner._inputs_scope.enter_context(inputs())
        owner._poison(original)
        with pytest.raises(ContractError) as closed:
            owner.__exit__(None,None,None)
        assert closed.value is original and events==[('cleanup-only',original)] and owner.cleanup_complete


class _NoteAllocationFailure(ContractError):
    def add_note(self, note):
        raise MemoryError('inert diagnostic allocation failure')


def _unadmitted_enter_probe(tmp_path,monkeypatch,state,native,*,limit=32*1024**2,enter_error=None):
    """Real lock + inert control/traversal, no Source admission or native execution."""
    from contextlib import contextmanager

    import gf_training.card_instance_formal as formal
    import gf_training.card_instance_source as source
    import gf_training.structured_lineage as lineage
    directory=tmp_path/'unadmitted-enter-probe';directory.mkdir()
    class Scan:
        retained=0
        def __init__(self,**kwargs):self.files={};self.inventories={}
        def object(self,*args):
            return {'run_variant':'formal-parent-v1','effective_config':{'preprocessing':{'backend':'python'}}}
        def charge(self,size):self.retained+=size
    @contextmanager
    def inputs(*args,**kwargs):
        if enter_error is not None:
            enter_error._structured_source_cleanup_owner=native
            raise enter_error
        try:
            yield state,[{'generation':0}],native
        finally:
            native.close()
    monkeypatch.setattr(lineage,'_ChainScan',Scan)
    monkeypatch.setattr(formal,'verify_kernel_ast',lambda *args:None)
    monkeypatch.setattr(formal,'read_formal_control',lambda *args:None)
    monkeypatch.setattr(source,'_open_game_parent_inputs',inputs)
    monkeypatch.setattr(source,'recheck_game_parent_snapshot',lambda *args:None)
    return FormalGameParent(directory,max_owned_bytes=limit,deadline=time.monotonic()+60)


@pytest.mark.parametrize('phase',['enter','exit','retire'])
@pytest.mark.parametrize('diagnostic',['note','format'])
def test_note_allocation_failure_keeps_first_error_and_same_unconfirmed_owner(tmp_path,monkeypatch,phase,diagnostic):
    from contextlib import contextmanager

    import gf_training.card_instance_source as source
    primary = (_NoteAllocationFailure if diagnostic == 'note' else ContractError)('first unadmitted original failure')
    class BadCleanupString(OSError):
        def __str__(self):raise MemoryError('inert formatting allocation failure')
    class Native:
        cleanup_complete=False
        allowed=False
        def close(self):
            if not self.allowed:
                raise (OSError if diagnostic == 'note' else BadCleanupString)('inert native cleanup unconfirmed')
            self.cleanup_complete=True
    native=Native()
    if phase=='enter':
        owner=_unadmitted_enter_probe(tmp_path,monkeypatch,{},native,enter_error=primary)
        operation=owner.__enter__
    else:
        directory=tmp_path/'unadmitted-note-lock'
        owner=FormalGameParent(directory,max_owned_bytes=1024**3,deadline=time.monotonic()+60)
        lock=owner._scope.enter_context(run_lock(directory,create=True))
        owner._owns_lock=True;owner._token=lock.token;owner._entered=owner._active=True
        owner.scan=SimpleNamespace(retained=0,files={},inventories={})
        owner._native_inputs=native
        monkeypatch.setattr(source,'recheck_game_parent_snapshot',lambda *args:None)
        if phase=='retire':
            @contextmanager
            def inputs():
                try:yield
                finally:raise primary
            owner._inputs_scope.enter_context(inputs())
        owner._poison(primary)
        operation=lambda:owner.__exit__(None,None,None)
    try:
        with pytest.raises(type(primary)) as failed:
            operation()
        assert failed.value is primary and primary._formal_parent_cleanup_owner is owner
        assert _OWNED_LOCKS.get(owner.directory)==owner._token
        owner.deadline=time.monotonic()-1
        assert not owner.cleanup_complete and not native.cleanup_complete
    finally:
        native.allowed=True
        try:owner.__exit__(None,None,None)
        except type(primary) as cleanup:
            assert cleanup is primary
    assert owner.cleanup_complete and owner.directory not in _OWNED_LOCKS


def test_joint_native_materialization_refuses_before_both_canonical_copies(tmp_path,monkeypatch):
    import gf_training.card_instance_formal as formal
    limit=32*1024**2
    class Native:
        cleanup_complete=False
        growth=0
        @property
        def joint_owned_bytes(self):return limit-65536+self.growth
        def caller_growth(self,size):self.growth=size
        def recheck(self):pass
        def close(self):self.cleanup_complete=True
    native=Native()
    state={'status':'complete','updates':1,'controlled-payload':'x'*(100*1024)}
    owner=_unadmitted_enter_probe(tmp_path,monkeypatch,state,native,limit=limit)
    calls=[]
    def forbidden(*args):
        calls.append(True)
        pytest.fail('canonical copies allocated before joint native/caller admission')
    monkeypatch.setattr(formal,'canonical',forbidden)
    with pytest.raises(ContractError,match='joint'):
        owner.__enter__()
    assert calls==[] and native.cleanup_complete and owner.cleanup_complete
    assert owner.directory not in _OWNED_LOCKS


def test_failed_owned_lock_close_cannot_confirm_from_map_absence_or_empty_retry(tmp_path,monkeypatch):
    import fcntl
    from contextlib import contextmanager

    import gf_training.card_instance_source as source
    from gf_training.lineage import RunOwnership
    directory=tmp_path/'unadmitted-held-lock';directory.mkdir()
    held=(directory/'.lock').open('a')
    fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token='unadmitted-real-flock-only'
    close_failure=OSError('unconfirmed original flock/file close')
    attempts=[]
    @contextmanager
    def uncertain_lock():
        _OWNED_LOCKS[directory]=token
        try:
            yield RunOwnership(directory,token)
        finally:
            # Fault models the real registry removal preceding fallible unlock/
            # descriptor close. Keep the actual OS flock held for the probe.
            _OWNED_LOCKS.pop(directory,None)
            attempts.append(True)
            raise close_failure
    owner=FormalGameParent(directory,max_owned_bytes=1024**3,deadline=time.monotonic()+60)
    owner._scope.enter_context(uncertain_lock())
    owner._token=token;owner._owns_lock=True;owner._entered=owner._active=True
    owner.scan=SimpleNamespace(retained=0,files={},inventories={})
    monkeypatch.setattr(source,'recheck_game_parent_snapshot',lambda *args:None)
    try:
        with pytest.raises(OSError) as failed:
            owner.__exit__(None,None,None)
        assert failed.value is close_failure and close_failure._formal_parent_cleanup_owner is owner
        assert directory not in _OWNED_LOCKS
        with (directory/'.lock').open('a') as independent, pytest.raises(BlockingIOError):
            fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.deadline=time.monotonic()-1
        assert not owner.cleanup_complete
        with pytest.raises(OSError) as retry:
            owner.__exit__(None,None,None)
        assert retry.value is close_failure and attempts==[True] and not owner.cleanup_complete
    finally:
        # Fresh test-owned handle, never an uncertain production FD retry.
        held.close()
    assert not owner.cleanup_complete


def test_normal_owned_and_borrowed_locks_confirm_only_the_owners_cleanup(tmp_path,monkeypatch):
    import gf_training.card_instance_source as source
    monkeypatch.setattr(source,'recheck_game_parent_snapshot',lambda *args:None)
    directory=tmp_path/'unadmitted-normal-owned'
    owner=FormalGameParent(directory,max_owned_bytes=1024**3,deadline=time.monotonic()+60)
    owned=owner._scope.enter_context(run_lock(directory,create=True))
    owner._token=owned.token;owner._owns_lock=True;owner._entered=owner._active=True
    owner.scan=SimpleNamespace(retained=0,files={},inventories={})
    owner.__exit__(None,None,None)
    assert owner.cleanup_complete and owner._lock_retired and directory not in _OWNED_LOCKS
    borrowed_directory=tmp_path/'unadmitted-normal-borrowed'
    with run_lock(borrowed_directory,create=True) as borrowed:
        child,_,_,_=unadmitted_parent_lifecycle(borrowed_directory,borrowed,monkeypatch)
        child.__exit__(None,None,None)
        assert child.cleanup_complete and _OWNED_LOCKS[borrowed_directory]==borrowed.token
