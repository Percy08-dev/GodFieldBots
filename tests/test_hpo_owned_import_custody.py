"""Cold partial owners and decode envelopes survive actual aliases/failure traces."""
import asyncio
import gc
import os
import time
import weakref

import pytest


@pytest.fixture(scope='module')
def paused_parent(tmp_path_factory):
    from tests.hpo_owned_support import owned_runtime, run_owned
    root=tmp_path_factory.mktemp('cold-custody')
    with pytest.MonkeyPatch.context() as patch:
        runtime,run_id,_,_=owned_runtime(root,patch)
        report,ops=asyncio.run(run_owned(runtime,root/'parents'/run_id,pause_after_update=1))
        assert report['status']=='paused' and ops.budget.used==0
    return root/'parents'/run_id


def cold(parent):
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.hpo_owned_restore import OwnedColdImport
    from gf_training.lineage import run_lock
    from tests.test_hpo_owned_numeric import budget
    lock=run_lock(parent); owner=lock.__enter__()
    account=budget(); phase=ArtifactPhaseAdmission(account,1024**3)
    imported=OwnedColdImport(owner,budget=account,parent_budget=phase,deadline=time.monotonic()+30)
    return lock,imported,account,phase


def load(imported):
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.structured_config import InferenceBindingV2
    from gf_training.hpo_lineage import hpo_state
    actions,model,_=hpo_state(imported.state)
    imported.load(model=model,binding=InferenceBindingV2(**imported.state['execution']['learner']['binding']),actions=actions,rules=card_instance_rules())


def test_partial_current_bank_view_keeps_primary_and_every_other_owner_retires(paused_parent,monkeypatch):
    import gf_models.hpo_owned_inference as inference
    baseline=len(os.listdir('/proc/self/fd'))
    lock,imported,account,phase=cold(paused_parent)
    views=[]; banks=[]
    original=inference._validate_owned_inference
    primary=ValueError('original held current validation failure')
    def invalid(bankset,metadata,**options):
        if metadata['generation']==1:
            views.append(bankset.array(bankset.names[0])); banks.append(bankset)
            raise primary
        return original(bankset,metadata,**options)
    monkeypatch.setattr(inference,'_validate_owned_inference',invalid)
    try:
        with pytest.raises(ValueError) as failed: load(imported)
        assert failed.value is primary and failed.value.cleanup_owner is imported
        assert not imported._closed
        assert account.used>0
        # Resume and earlier pool inference must retire despite current inference alias.
        assert len(os.listdir('/proc/self/fd'))==baseline+3  # run lock + original FD + mmap duplicate
        views.clear()
        failed.value.__traceback__=None
        del failed
        primary.__traceback__=None
        gc.collect()
        imported.close()
        assert imported._closed and account.used==0 and phase.used==0
    finally:
        views.clear()
        try: imported.close()
        finally: lock.__exit__(None,None,None)
    assert len(os.listdir('/proc/self/fd'))==baseline


def test_decode_envelope_remains_live_until_actual_array_alias_releases(paused_parent,monkeypatch):
    from gf_common.ram_bank import ReadonlyRamBankSet
    lock,imported,account,phase=cold(paused_parent)
    aliases=[]
    build=ReadonlyRamBankSet.build.__func__
    def retained(cls,arrays,*args,**kwargs):
        aliases.append(next(iter(arrays.values())))
        return build(cls,arrays,*args,**kwargs)
    monkeypatch.setattr(ReadonlyRamBankSet,'build',classmethod(retained))
    try:
        load(imported)
        receipts=imported._decodes
        assert receipts and any(not receipt.closed for receipt in receipts)
        assert sum(value.nbytes for value in aliases)>0 and account.used>0
        with pytest.raises(BufferError): imported.close()
        assert not imported._closed
        aliases.clear(); gc.collect()
        imported.close()
        assert imported._closed and account.used==0 and phase.used==0
    finally:
        aliases.clear(); gc.collect()
        try: imported.close()
        finally: lock.__exit__(None,None,None)


@pytest.mark.parametrize('failure',['callback','callback-timeout','finite'])
def test_actual_decoder_failure_keeps_envelope_through_arrays_and_trace(tmp_path,failure):
    import numpy as np

    from gf_models.arrays import load_arrays, sha256
    from gf_training.hpo_owned_restore import _DecodeEnvelope, _DecodeReceipt
    from tests.test_hpo_owned_numeric import budget
    account=budget()
    path=tmp_path/'values.npz'
    np.savez(path,weight=np.array([float('nan') if failure=='finite' else 1],dtype=np.float32))
    metadata={'sha256':sha256(path),'arrays':{'weight':{'dtype':'float32','shape':[1],'bytes':4}}}
    receipt=_DecodeReceipt(); envelope=_DecodeEnvelope(account.reserve(1024**2),receipt)
    refs=[]; aliases=[]
    primary=TimeoutError('exact callback timeout') if failure=='callback-timeout' else ValueError('exact callback error')
    def observe(array):
        refs.append(weakref.ref(array))
        assert envelope is not None
        envelope.hold(array); aliases.append(array)
        if failure.startswith('callback'): raise primary
    from gf_common.domain import ContractError
    with pytest.raises(type(primary) if failure.startswith('callback') else ContractError) as failed:
        load_arrays(path,metadata,_on_array=observe)
    if failure.startswith('callback'): assert failed.value is primary
    envelope=None
    assert refs and refs[0]() is not None and not receipt.closed
    assert account.used==1024**2
    limit=account.limit; account.limit=1024**2
    with pytest.raises(ValueError): account.reserve(1)
    account.limit=limit
    aliases.clear(); gc.collect()
    assert not receipt.closed  # Decoder traceback still owns its actual value/encoded payload.
    failed.value.__traceback__=None
    del failed
    primary.__traceback__=None
    gc.collect()
    assert refs[0]() is None and receipt.closed and account.used==0
    # Default decoder behavior still returns the original writable typed arrays.
    if failure.startswith('callback'):
        values=load_arrays(path,metadata)
        np.testing.assert_array_equal(values['weight'],[1])
        assert values['weight'].flags.writeable


def test_decode_admission_failure_does_not_leave_unbacked_live_receipt(paused_parent,monkeypatch):
    import inspect
    lock,imported,account,phase=cold(paused_parent)
    reserve=account.reserve
    primary=ValueError('decode admission denied before allocation')
    def denied(size,**options):
        if inspect.currentframe().f_back.f_code.co_name=='_decode_bank': raise primary
        return reserve(size,**options)
    monkeypatch.setattr(account,'reserve',denied)
    try:
        with pytest.raises(ValueError) as failed: load(imported)
        assert failed.value is primary and imported._closed
        assert account.used==0 and phase.used==0 and not imported._decodes
    finally:
        imported.close(); lock.__exit__(None,None,None)


def test_optional_legacy_capture_hooks_default_to_strict_replay():
    from gf_common.domain import ContractError, EpisodeResult
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    environment=object.__new__(CardInstanceSimulatorEnvironment)
    episode=EpisodeResult('e','p','s',('random','heuristic'))
    environment._episode_opened(episode)
    environment._act_acknowledged(episode,1,{}, {})
    environment._terminal_observed(episode,1,{})
    environment._episode_closed(episode)
    with pytest.raises(ContractError): environment._reconcile_replay({},1,[],None)
    with pytest.raises(ContractError): environment._replay_closure({},'game',1)


def test_import_uncertain_original_fd_close_is_charged_and_never_retried(paused_parent,monkeypatch):
    import gf_common.ram_bank as ram
    import gf_models.hpo_owned_inference as inference
    baseline=len(os.listdir('/proc/self/fd'))
    lock,imported,account,phase=cold(paused_parent)
    real_close=os.close
    called=[]; target=[]
    primary=ValueError('original validation before unknown close')
    validate=inference._validate_owned_inference
    def uncertain(fd):
        if target and fd==target[0]:
            called.append(fd)
            real_close(fd)
            raise OSError('injected close completion uncertainty')
        return real_close(fd)
    def invalid(bankset,metadata,**options):
        if metadata['generation']==1:
            target.append(bankset.banks[0].fileno())
            monkeypatch.setattr(ram.os,'close',uncertain)
            raise primary
        return validate(bankset,metadata,**options)
    monkeypatch.setattr(inference,'_validate_owned_inference',invalid)
    try:
        with pytest.raises(ValueError) as failed: load(imported)
        assert failed.value is primary and failed.value.cleanup_owner is imported
        assert called==target and not imported._closed and account.used>0 and phase.used>0
        assert len(imported._partial_banks)==1
        bank=imported._partial_banks[0].banks[0]
        assert bank._close_uncertain and bank._fd==-1
        with pytest.raises(ram.RamBankCleanupError): imported.close()
        assert called==target and account.used>0
    finally:
        monkeypatch.setattr(ram.os,'close',real_close)
        lock.__exit__(None,None,None)
    # The injected syscall really completed; no historical descriptor is retried.
    # Its ORIGINAL charged uncertain owner deliberately stays unretired.
    assert len(os.listdir('/proc/self/fd'))==baseline


@pytest.fixture(scope='module')
def completed_parent(tmp_path_factory):
    from tests.hpo_owned_support import owned_runtime, run_owned
    root=tmp_path_factory.mktemp('complete-cold-custody')
    with pytest.MonkeyPatch.context() as patch:
        runtime,run_id,_,_=owned_runtime(root,patch)
        _,ops=asyncio.run(run_owned(runtime,root/'parents'/run_id))
        ops.completion.close()
    return root/'parents'/run_id


def test_external_import_unwind_keeps_actual_held_view_primary(completed_parent,monkeypatch):
    import gf_models.hpo_owned_inference as inference
    from gf_training.hpo_owned_restore import import_owned_final_policy
    baseline=len(os.listdir('/proc/self/fd'))
    views=[]; primary=ValueError('external exact original validation')
    def invalid(bankset,metadata,**options):
        views.append(bankset.array(bankset.names[0])); raise primary
    monkeypatch.setattr(inference,'_validate_owned_inference',invalid)
    with pytest.raises(ValueError) as failed:
        import_owned_final_policy(completed_parent,deadline=time.monotonic()+30,max_owned_bytes=2*1024**3,parent_budget_bytes=1024**3)
    assert failed.value is primary
    imported=failed.value.cleanup_owner
    assert not imported._closed
    views.clear(); failed.value.__traceback__=None; del failed
    primary.__traceback__=None; gc.collect()
    imported.close()
    assert imported._closed and imported.budget.used==0 and imported.parent_budget.used==0
    assert len(os.listdir('/proc/self/fd'))==baseline


def test_detached_candidate_retires_even_when_original_import_exit_fails(completed_parent,monkeypatch):
    import gf_training.hpo_owned_completion as completion
    from gf_training.hpo_owned_restore import _ColdSnapshot, import_owned_final_policy
    baseline=len(os.listdir('/proc/self/fd')); candidates=[]
    detach=completion._detach_external_policy
    close=_ColdSnapshot.close
    primary=ValueError('exact original import close fault')
    def detached(*args,**options):
        data=detach(*args,**options); candidates.append(data); return data
    def failed(self): raise primary
    monkeypatch.setattr(completion,'_detach_external_policy',detached)
    monkeypatch.setattr(_ColdSnapshot,'close',failed)
    try:
        with pytest.raises(ValueError) as rejected:
            import_owned_final_policy(completed_parent,deadline=time.monotonic()+30,max_owned_bytes=2*1024**3,parent_budget_bytes=1024**3)
        assert rejected.value is primary and candidates[0]._closed
        imported=rejected.value.cleanup_owner
        assert not imported._closed and imported.budget.used>0
    finally: monkeypatch.setattr(_ColdSnapshot,'close',close)
    rejected.value.__traceback__=None; del rejected
    primary.__traceback__=None; gc.collect()
    imported.close()
    assert imported.budget.used==0 and imported.parent_budget.used==0
    assert len(os.listdir('/proc/self/fd'))==baseline


def test_actual_ppo_automatic_exit_preserves_import_primary(tmp_path,monkeypatch):
    import gf_models.hpo_owned_inference as inference
    from tests.hpo_owned_support import owned_runtime, run_owned
    root=tmp_path
    runtime,run_id,_,_=owned_runtime(root,monkeypatch)
    directory=root/'parents'/run_id
    baseline=len(os.listdir('/proc/self/fd'))
    views=[]; primary=ValueError('exact automatic PPO import error')
    validate=inference._validate_owned_inference
    def invalid(bankset,metadata,**options):
        if metadata['generation']==1:
            views.append(bankset.array(bankset.names[0])); raise primary
        return validate(bankset,metadata,**options)
    async def run():
        await run_owned(runtime,directory,pause_after_update=1)
        monkeypatch.setattr(inference,'_validate_owned_inference',invalid)
        with pytest.raises(ValueError) as rejected: await run_owned(runtime,directory,mode='resume')
        assert rejected.value is primary
        imported=rejected.value.cleanup_owner
        assert not imported._closed
        views.clear(); rejected.value.__traceback__=None; del rejected
        primary.__traceback__=None; gc.collect()
        imported.close()
        assert imported._closed and imported.budget.used==0 and imported.parent_budget.used==0
    asyncio.run(run())
    assert len(os.listdir('/proc/self/fd'))==baseline
