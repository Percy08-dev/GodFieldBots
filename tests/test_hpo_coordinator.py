"""Actual common-loop CPU components; no GPU/Simulator qualification."""
import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field

import pytest

pytest.importorskip('torch')

from gf_cli.config import ExperimentConfig, LoadedConfig
from gf_common.domain import ContractError, digest
from gf_models.sharded_arrays import read_json
from gf_training.collection import FinalizedCollection
from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
from tests.hpo_lineage_support import hpo_state_fixture
from tests.hpo_support import original_hpo_collection
from tests.test_card_instance_learning import ACTIONS, RULES


@dataclass(frozen=True)
class ControlledHpoConfig(ExperimentConfig):
    hpo_spec: dict=field(default_factory=dict)


def controlled_run(tmp_path,monkeypatch,*,budget=3,max_updates=4,numeric_budget=None,storage_cap=None):
    import gf_training.hpo_gpu_lease as gpu
    import gf_training.hpo_source as source
    from gf_cli.config import CorrectedServerConfigV2, RunConfigV2
    from gf_common.card_instance import action_config_v3
    from gf_common.preprocessing import preparation_config
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_models.config import PPOTrainingConfig
    from gf_models.structured_binding import configured_model_v4
    from gf_training.hpo_ppo import _HpoOperations
    from tests.test_card_instance_baseline_runtime import receipt
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'gpu')
    state,model,recipe,binding=hpo_state_fixture(tmp_path/'parent',decision_budget=budget)
    raw=state['effective_config']
    run_id=str(uuid.uuid4())
    raw['run']['name']=run_id
    raw['run']['output_dir']=raw['hpo_spec']['storage']['parent_root']
    raw['training']['max_updates']=max_updates
    raw['hpo_spec']['operating']['max_updates']=max_updates
    raw['hpo_spec']['storage']['numeric_root']=raw['hpo_spec']['storage']['parent_root']
    if numeric_budget is not None: raw['hpo_spec']['operating']['numeric_budget_bytes']=numeric_budget
    if storage_cap is not None:
        raw['hpo_spec']['storage'].update(destination_capacity_bytes=storage_cap,admission_reservation_bytes=storage_cap)
        raw['training'].update(max_owned_bytes=storage_cap,admission_reservation_bytes=storage_cap)
    config=ControlledHpoConfig(run=RunConfigV2(**raw['run']),server=CorrectedServerConfigV2(**raw['server']),
        actions=action_config_v3(raw['actions']),model=configured_model_v4(raw['model'],layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        training=PPOTrainingConfig(**raw['training']),preprocessing=preparation_config(raw['preprocessing']),hpo_spec=raw['hpo_spec'])
    raw=asdict(config)
    loaded=LoadedConfig(config,raw,digest(raw),())
    # The sole explicitly replaced production gates keep honest CPU metadata.
    # Original journals/arrays/Source issuance and actual subprocesses remain real.
    monkeypatch.setattr(source,'require_hpo_production',lambda original: None)
    def preflight(self,config,*,deadline):
        class Runtime:
            def preflight(self): return receipt()
        self.runtime=Runtime()
    monkeypatch.setattr(_HpoOperations,'preflight',preflight)
    original_integration=_HpoOperations.integration_identity
    def integration(self,config,deadline):
        if self.runtime is None: preflight(self,config,deadline=deadline)
        return original_integration(self,config,deadline)
    monkeypatch.setattr(_HpoOperations,'integration_identity',integration)
    return loaded,model,recipe,binding,run_id


def collector_for(loaded,model,recipe,binding,*,decisions=True):
    async def collect(plan,ledger):
        from gf_training.hpo_numeric import HpoLearner
        from gf_training.lineage import checkpoint_path
        state=read_json(plan.output.parent.parent/'run.json')
        learner=HpoLearner(model,recipe,17,binding,state['resume_identity'],actions=ACTIONS,rules=RULES)
        learner.restore(checkpoint_path(plan.output.parent.parent,state['updates']))
        original_hpo_collection(plan,state,learner,decisions=decisions)
        return FinalizedCollection(plan.output,{'comparison_valid':True})
    return collect


@pytest.mark.parametrize('budget,decisions,status,used',[(1,True,'complete',1),(9,True,'complete',9),(3,False,'incomplete',0)])
def test_actual_common_loop_decision_budget_and_empty(tmp_path,monkeypatch,budget,decisions,status,used):
    from gf_training.hpo_ppo import train_hpo_ppo
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=budget)
    job=admit_gpu_job(JobContext('fixture','operation','training',run_id,None),
        global_cuda_bytes=loaded.config.run.global_cuda_budget_bytes,max_workers=4)
    try:
        result=asyncio.run(train_hpo_ppo(loaded,collector_for(loaded,model,recipe,binding,decisions=decisions),
            tmp_path/'parent',gpu_job=job))
        assert result.report['status']==status and result.report['used_decisions']==used
        assert result.gpu_termination.ticket==job.ticket
        assert all(resource.outcome=='reaped' for resource in result.gpu_termination.resources)
        state=read_json(tmp_path/'parent'/'run.json')
        assert state['device']=='cpu' and state['used_decisions']==used
        assert state['run_id']==run_id
    finally: job.close()


def test_foreign_job_context_refuses_before_parent_creation(tmp_path,monkeypatch):
    from gf_training.hpo_ppo import train_hpo_ppo
    loaded,model,recipe,binding,_=controlled_run(tmp_path,monkeypatch)
    job=admit_gpu_job(JobContext('fixture','operation','training',str(uuid.uuid4()),None),global_cuda_bytes=12*1024**3,max_workers=4)
    try:
        with pytest.raises(ContractError):
            asyncio.run(train_hpo_ppo(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',gpu_job=job))
        assert not (tmp_path/'parent').exists()
        job.confirm_termination()
    finally: job.close()


def invoke(loaded,collector,directory,run_id,*,mode='fresh',pause=None):
    from gf_training.hpo_ppo import train_hpo_ppo
    job=admit_gpu_job(JobContext('fixture',str(uuid.uuid4()),'training',run_id,None),
        global_cuda_bytes=loaded.config.run.global_cuda_budget_bytes,max_workers=4)
    try:
        return asyncio.run(train_hpo_ppo(loaded,collector,directory,gpu_job=job,mode=mode,pause_after_update=pause))
    finally: job.close()


def test_actual_pause_resume_uses_same_original_deadlines_and_arrays(tmp_path,monkeypatch):
    from gf_models.sharded_arrays import read_bundle
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=9)
    collector=collector_for(loaded,model,recipe,binding)
    paused=invoke(loaded,collector,tmp_path/'parent',run_id,pause=1)
    assert paused.report['status']=='paused' and paused.report['used_decisions']==8
    original=read_json(tmp_path/'parent'/'run.json')
    resumed=invoke(loaded,collector,tmp_path/'parent',run_id,mode='resume')
    assert resumed.report['status']=='complete' and resumed.report['used_decisions']==9
    restored=read_json(tmp_path/'parent'/'run.json')
    for key in ('created_unix','wall_deadline_unix','execution_deadline_unix','admission_deadline_unix'):
        assert restored[key]==original[key]
    assert [row['mode'] for row in restored['invocations']]==['fresh','resume']
    other=tmp_path/'continuous'
    other.mkdir()
    loaded2,model2,recipe2,binding2,run_id2=controlled_run(other,monkeypatch,budget=9)
    invoke(loaded2,collector_for(loaded2,model2,recipe2,binding2),other/'parent',run_id2)
    for filename,member in (('inference.json','weights'),('resume.json','state')):
        left=tmp_path/'parent'/'artifacts'/'000002'
        right=other/'parent'/'artifacts'/'000002'
        la=read_bundle(left,read_json(left/filename)[member])
        ra=read_bundle(right,read_json(right/filename)[member])
        assert la.keys()==ra.keys()
        import numpy as np
        for key in la: np.testing.assert_array_equal(la[key],ra[key],err_msg=key)


def test_update_cap_is_incomplete_with_real_termination_receipt(tmp_path,monkeypatch):
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=9,max_updates=1)
    result=invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    assert result.report['status']=='incomplete' and result.report['stop_reason']=='update_cap'
    assert result.report['used_decisions']==8
    assert result.gpu_termination.resources


def test_commit_failure_after_real_update_quarantines_whole_parent(tmp_path,monkeypatch):
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_ppo import _HpoOperations
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1)
    original=_HpoOperations.commit
    def fault(self,*args,**kwargs):
        if kwargs['generation']==1: raise OSError('injected after actual update and full save')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(_HpoOperations,'commit',fault)
    with pytest.raises(OSError,match='after actual update'):
        invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    state=read_json(tmp_path/'parent'/'run.json')
    assert state['status']=='failed' and state['quarantine'] is True
    assert (tmp_path/'parent'/'artifacts'/'000001'/'resume.json').exists()
    with pytest.raises(ContractError):
        validate_hpo_chain(tmp_path/'parent',max_owned_bytes=1024**3,deadline=time.monotonic()+30)


@pytest.mark.parametrize('invalid_job',[None,object()])
def test_missing_or_fabricated_job_refuses_before_parent(tmp_path,monkeypatch,invalid_job):
    from gf_training.hpo_ppo import train_hpo_ppo
    loaded,model,recipe,binding,_=controlled_run(tmp_path,monkeypatch)
    with pytest.raises(ContractError,match='actual caller GpuJob'):
        asyncio.run(train_hpo_ppo(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',gpu_job=invalid_job))
    assert not (tmp_path/'parent').exists()


def test_storage_admission_stop_can_preserve_healthy_g0_without_objective(tmp_path,monkeypatch):
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,storage_cap=128*1024**2)
    result=invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    assert result.report['status']=='incomplete' and result.report['stop_reason']=='storage_cap'
    assert result.report['updates']==result.report['used_decisions']==0
    assert (tmp_path/'parent'/'artifacts'/'000000'/'resume.json').is_file()
    assert not list((tmp_path/'parent'/'collections').iterdir())


def test_numeric_phase_refusal_precedes_optimizer_side_effects(tmp_path,monkeypatch):
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,numeric_budget=1)
    with pytest.raises(ContractError,match='numeric'):
        invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    state=read_json(tmp_path/'parent'/'run.json')
    assert state['status']=='failed' and state['updates']==state['optimizer_steps']==0
    assert not (tmp_path/'parent'/'artifacts'/'000001').exists()
    assert list((tmp_path/'parent'/'collections').iterdir())  # Complete originals stay.


def test_final_write_failure_never_turns_gpu_receipt_into_data_qualification(tmp_path,monkeypatch):
    from gf_training.hpo_gpu_lease import TerminationReceipt
    from gf_training.hpo_ppo import _HpoOperations
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1)
    original=_HpoOperations.persist_control
    def fault(self,path,value,*args,**kwargs):
        if path.name=='run.json' and value.get('status')=='complete':
            raise OSError('injected final write failure')
        return original(self,path,value,*args,**kwargs)
    monkeypatch.setattr(_HpoOperations,'persist_control',fault)
    with pytest.raises(OSError,match='final write') as raised:
        invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    assert type(raised.value.hpo_gpu_termination) is TerminationReceipt
    state=read_json(tmp_path/'parent'/'run.json')
    assert state['status']=='failed' and state['quarantine'] is True


def test_runtime_refuses_parent_outside_declared_colocated_root(tmp_path,monkeypatch):
    from dataclasses import replace

    from gf_training.hpo_ppo import train_hpo_ppo
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch)
    spec=dict(loaded.config.hpo_spec)
    spec['storage']={**spec['storage'],'parent_root':str(tmp_path/'declared'),'numeric_root':str(tmp_path/'declared')}
    config=replace(loaded.config,hpo_spec=spec,run=replace(loaded.config.run,output_dir=str(tmp_path/'declared')))
    raw=asdict(config)
    loaded=LoadedConfig(config,raw,digest(raw),loaded.inputs)
    job=admit_gpu_job(JobContext('fixture','operation','training',run_id,None),global_cuda_bytes=12*1024**3,max_workers=4)
    try:
        with pytest.raises(ContractError,match='root'):
            asyncio.run(train_hpo_ppo(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'outside',gpu_job=job))
        assert not (tmp_path/'outside').exists()
        job.confirm_termination()
    finally: job.close()
