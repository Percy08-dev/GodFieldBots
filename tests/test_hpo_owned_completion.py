"""Completion retains inference alone after actual execution scope retirement."""
import asyncio
import copy

import numpy as np
import pytest

from gf_common.domain import ContractError


def test_inference_bank_independent_retention_survives_original_close():
    from gf_common.ram_bank import RamBankBuilder
    from tests.test_hpo_owned_numeric import budget
    account=budget()
    bank=RamBankBuilder({'w':np.arange(4,dtype=np.float32)},account).seal()
    retained=bank.retain(account)
    bank.close()
    np.testing.assert_array_equal(retained.array('w'),[0,1,2,3])
    retained.close()
    assert account.used==0
    with pytest.raises(ContractError): bank.retain(account)


def test_completed_run_handoff_retains_only_inference_and_cannot_be_copied(tmp_path,monkeypatch):
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    async def run():
        report,ops=await run_owned(runtime,tmp_path/'parents'/run_id)
        completion=ops.completion
        assert report['status']=='complete' and completion is not None
        completion.require()
        assert ops.owned_registry._closed
        assert ops.owned_registry.process.current_snapshot.resume.banks[0]._fd==-1
        assert not hasattr(completion,'resume')
        assert all(not name.startswith(('rng.','adam.')) for name in completion.inference.names)
        assert completion.state['status']=='complete' and completion.state['used_decisions']==32
        assert completion.actor_capability.require() is completion
        with pytest.raises(TypeError): copy.copy(completion)
        completion.close()
        assert ops.budget.used==0
        with pytest.raises(ContractError): completion.require()
    asyncio.run(run())


def test_final_state_persistence_failure_never_mints_completion(tmp_path,monkeypatch):
    import json

    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    observed=[]
    candidates=[]
    persist=_OwnedHpoOperations.persist_control
    def fail_complete(self,path,value,*args,**kwargs):
        if path.name=='run.json' and value.get('status')=='complete':
            observed.append(self)
            candidates.append(self._prepared_completion)
            assert self.completion is None
            assert self._prepared_completion is not None
            with pytest.raises(ContractError): self._prepared_completion.require()
            raise OSError('final state persistence failed')
        return persist(self,path,value,*args,**kwargs)
    monkeypatch.setattr(_OwnedHpoOperations,'persist_control',fail_complete)
    async def run():
        import os
        baseline=len(os.listdir('/proc/self/fd'))
        with pytest.raises(OSError,match='final state persistence failed') as raised:
            await run_owned(runtime,directory)
        assert observed[-1].completion is None
        assert observed[-1]._prepared_completion is None
        candidate=candidates[-1]
        assert candidate._closed and candidate._reservation.closed
        assert not hasattr(candidate,'resume')
        assert candidate._records==candidate._manifest==candidate._actor_metadata==b''
        assert all(bank._fd==-1 and bank._mapping.closed for bank in candidate.inference.banks)
        original=observed[-1].owned_registry.process.current_snapshot
        assert original._closed and original._control.value=={}
        assert all(bank._fd==-1 and bank._mapping.closed for bank in original.resume.banks)
        assert len(os.listdir('/proc/self/fd'))==baseline
        trace=raised.value.__traceback__
        while trace is not None and trace.tb_frame.f_code.co_name!='_run_ppo': trace=trace.tb_next
        assert trace is not None
        archive=trace.tb_frame.f_locals['collection']
        assert archive.batch._closed and archive.batch._episodes==() and archive.batch._retention is None
        assert observed[-1].budget.used>0  # Actual admitted diagnostic manifest still reachable.
        del archive,trace,original
        # The original exception legitimately retains common-loop diagnostic
        # objects. Releasing that traceback must release their real admissions.
        raised.value.__traceback__=None
        del raised
        import gc
        gc.collect()
        assert observed[-1].budget.used==0 and observed[-1].writer.parent_budget.used==0
        assert json.loads((directory/'run.json').read_text())['status']=='failed'
    asyncio.run(run())


@pytest.mark.parametrize('account',['source','parent'])
def test_preparation_admits_before_any_snapshot_metadata_copy(tmp_path,monkeypatch,account):
    from gf_training.hpo_owned_completion import _prepare_completion
    from gf_training.hpo_owned_process import RemoteModelSnapshot
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    original=_OwnedHpoOperations.prepare_completion
    checked=[]
    def prepare(self,state):
        budget=self.budget if account=='source' else self.writer.parent_budget
        before=budget.used; limit=budget.limit
        budget.limit=before
        def forbidden(self): raise AssertionError('copied metadata before admission')
        try:
            with monkeypatch.context() as patch:
                patch.setattr(RemoteModelSnapshot,'actor_metadata',property(forbidden))
                patch.setattr(RemoteModelSnapshot,'inference_manifest',property(forbidden))
                with pytest.raises(ValueError): _prepare_completion(self)
            assert budget.used==before
            checked.append(True)
        finally: budget.limit=limit
        return original(self,state)
    monkeypatch.setattr(_OwnedHpoOperations,'prepare_completion',prepare)
    async def run():
        _,ops=await run_owned(runtime,tmp_path/'parents'/run_id)
        ops.completion.close()
        assert checked==[True] and ops.budget.used==0
    asyncio.run(run())


@pytest.mark.parametrize('mutation',['stale-running','supplied-state','wrong-writer','wrong-receipt','wrong-identity'])
def test_completion_requires_actual_final_writer_digest_identity(tmp_path,monkeypatch,mutation):
    import copy

    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.lineage import generation_path
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    original=_OwnedHpoOperations.finalize_completion
    seen=[]
    def finalize(self,state,error):
        if error is not None or state['status']!='complete': return original(self,state,error)
        seen.append(self)
        path=self.ownership.directory/'run.json'
        writer=self.writer
        receipt=writer._completed[path]
        identity=writer.files[path]
        supplied=state
        if mutation=='stale-running':
            writer.write(path,{**state,'status':'running','completed_updates':None},deadline=self.budget.deadline)
        elif mutation=='supplied-state': supplied={**state,'collection_seconds':state['collection_seconds']+1}
        elif mutation=='wrong-writer': self.writer=copy.copy(writer)
        elif mutation=='wrong-receipt': writer._completed[path]=writer._completed[generation_path(self.ownership.directory,2)]
        else: writer.files[path]=(0,0,0,0,0)
        try: return original(self,supplied,error)
        finally:
            self.writer=writer
            if mutation in ('wrong-receipt','wrong-identity'):
                writer._completed[path]=receipt
                writer.files[path]=identity
    monkeypatch.setattr(_OwnedHpoOperations,'finalize_completion',finalize)
    async def run():
        with pytest.raises(ContractError) as rejected:
            await run_owned(runtime,tmp_path/'parents'/run_id)
        assert seen[-1].completion is None and seen[-1]._prepared_completion is None
        rejected.value.__traceback__=None
        del rejected
        import gc
        gc.collect()
        assert seen[-1].budget.used==0 and seen[-1].writer.parent_budget.used==0
    asyncio.run(run())


def test_actual_control_write_receipt_cannot_shadow_its_digest(tmp_path):
    from dataclasses import FrozenInstanceError

    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.lineage import run_lock
    from tests.test_hpo_owned_numeric import budget
    with run_lock(tmp_path) as ownership:
        writer=OwnedControlWriter(ownership,budget=budget(),limit=1024**2)
        path=tmp_path/'run.json'
        sha=writer.write(path,{'status':'running'},deadline=__import__('time').monotonic()+10)
        receipt=writer._completed[path]
        receipt.require(writer,path,sha)
        with pytest.raises(FrozenInstanceError): receipt.sha256='a'*64


@pytest.mark.parametrize('account',['source','parent'])
def test_external_detach_admits_before_copying_cold_snapshot_metadata(tmp_path,monkeypatch,account):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_owned_completion import _detach_external_policy
    from gf_training.hpo_owned_restore import OwnedColdImport, _ColdSnapshot
    from gf_training.lineage import run_lock
    from tests.hpo_owned_support import owned_runtime, run_owned
    from tests.test_card_instance_learning import RULES
    from tests.test_hpo_owned_numeric import budget
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        _,ops=await run_owned(runtime,directory); ops.completion.close()
        source=budget(); phase=ArtifactPhaseAdmission(source,runtime.config.hpo_spec['operating']['parent_budget_bytes'])
        with run_lock(directory) as ownership:
            imported=OwnedColdImport(ownership,budget=source,parent_budget=phase,deadline=time.monotonic()+30,require_complete=True)
            try:
                actions,model,_=hpo_state(imported.state)
                from gf_models.structured_config import InferenceBindingV2
                imported.load(model=model,binding=InferenceBindingV2(**imported.state['execution']['inference']['binding']),actions=actions,rules=RULES,final_only=True)
                constrained=source if account=='source' else phase
                before=constrained.used; old=constrained.limit; constrained.limit=before
                def forbidden(self): raise AssertionError('cold metadata copied before admission')
                try:
                    with monkeypatch.context() as patch:
                        patch.setattr(_ColdSnapshot,'actor_metadata',property(forbidden))
                        patch.setattr(_ColdSnapshot,'inference_manifest',property(forbidden))
                        with pytest.raises(ValueError): _detach_external_policy(imported)
                    assert constrained.used==before
                finally: constrained.limit=old
            finally: imported.close()
        assert source.used==0 and phase.used==0
    asyncio.run(run())
