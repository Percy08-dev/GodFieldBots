"""Actual common-loop CPU components; no GPU/Simulator qualification."""
import asyncio
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

import pytest

pytest.importorskip('torch')

from gf_cli.config import ExperimentConfig, GroupPackExperimentConfig, LoadedConfig
from gf_common.domain import ContractError, digest
from gf_models.sharded_arrays import read_json
from gf_training.collection import FinalizedCollection
from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
from tests.hpo_lineage_support import hpo_state_fixture
from tests.hpo_support import original_hpo_collection
from tests.test_card_instance_learning import ACTIONS, RULES
from tests.test_hpo_native import candidate as native_save_candidate  # noqa: F401 - scoped private-binary fixture


@pytest.fixture(scope='module')
def native_save_backend(request):
    if os.environ.get('GFB_HPO_NATIVE_CANDIDATE'):
        return request.getfixturevalue('native_save_candidate')
    from gf_training.native_backend import load_native
    return load_native()


@dataclass(frozen=True)
class ControlledHpoConfig(ExperimentConfig):
    hpo_spec: dict=field(default_factory=dict)


@dataclass(frozen=True)
class ControlledPackedHpoConfig(GroupPackExperimentConfig):
    hpo_spec: dict=field(default_factory=dict)


def controlled_run(tmp_path,monkeypatch,*,budget=3,max_updates=4,numeric_budget=None,storage_cap=None,native=False,packed=False):
    import gf_training.hpo_gpu_lease as gpu
    import gf_training.hpo_source as source
    from gf_cli.config import CorrectedServerConfigV2, GroupPackStorageConfig, RunConfigV2
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
    if native:
        raw['preprocessing']['backend']='rust'
    storage={}
    if packed:
        storage={'storage':GroupPackStorageConfig(db_root=str(tmp_path/'database'),ram_root=str(tmp_path/'ram'),budget_mib=1024)}
    config=(ControlledPackedHpoConfig if packed else ControlledHpoConfig)(run=RunConfigV2(**raw['run']),server=CorrectedServerConfigV2(**raw['server']),
        actions=action_config_v3(raw['actions']),model=configured_model_v4(raw['model'],layout=CORRECTED_CARD_INSTANCE_LAYOUT),
        training=PPOTrainingConfig(**raw['training']),preprocessing=preparation_config(raw['preprocessing']),hpo_spec=raw['hpo_spec'],**storage)
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
        if hasattr(loaded.config,'storage'):
            import json
            from pathlib import Path

            from gf_models.arrays import sha256
            from gf_training.collection import PackedFinalizedCollection
            from gf_training.group_pack_source import CollectionLocator
            from tests.test_group_pack_game_source import pack_original
            destination=Path(loaded.config.storage.db_root)/state['run_id']/'collections'/plan.linkage['batch_id']
            pack_original(plan.output,destination)
            pointer=json.loads((plan.output/'collection.locator.json').read_bytes())
            return PackedFinalizedCollection(plan.output,{'comparison_valid':True,'healthy':True},
                CollectionLocator('sqlite-group-pack-v1',Path(pointer['descriptor'])),
                sha256(plan.output/'collection.locator.json'),{'publication_scope':'published'})
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


@pytest.mark.parametrize('packed',[False,True])
def test_actual_cpu_native_common_loop_ack_save_two_updates_and_resume(tmp_path,monkeypatch,packed,native_save_backend):
    """Isolated CPU leases and held originals grant no real GPU recovery/authority."""
    from gf_common.hpo import NATIVE_CAPABILITY
    from gf_training.card_instance_source import IssuedGameProjection
    from gf_training.native_backend import load_native
    from gf_training.native_card_instance import _NativeGameInputs
    native=load_native()
    assert native is native_save_backend
    if dict(native.build_info()).get('card_instance_hpo')!=NATIVE_CAPABILITY:
        pytest.skip('actual installed HPO native support required; no fallback')
    seen=[]
    original=_NativeGameInputs._authorize_owned
    def observed(self):
        assert self._hpo and not self._explicit_candidate
        assert self.native is native
        original(self)
        if self.plan['mode']=='live_cohort':
            seen.append(self)
    monkeypatch.setattr(_NativeGameInputs,'_authorize_owned',observed)
    charged=[]
    original_record=IssuedGameProjection._record_hpo_save_ack
    original_prepare=IssuedGameProjection._prepare_commit
    def record(self,process,reply):
        assert process._issued_handle is self and not process.ledger.updating and not self.completed
        original_record(self,process,reply)
        assert self.source.controls.retained>=self._hpo_ack_source_floor>=self._hpo_ack_charge>0
        with pytest.raises(AttributeError): self._hpo_save_ack=b'forged'
        charged.append(self)
    def prepare(self):
        original_prepare(self)
        control=self.source.controls
        assert control.retained>=self._hpo_ack_source_floor
        if control._packed_pool is not None:
            assert control._pool_control_charge>=control.retained
        assert self.source.native_inputs.sealed is not control
        assert self.source.native_inputs.sealed.artifact_count==control.artifact_count
    monkeypatch.setattr(IssuedGameProjection,'_record_hpo_save_ack',record)
    monkeypatch.setattr(IssuedGameProjection,'_prepare_commit',prepare)
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=9,native=True,packed=packed)
    collector=collector_for(loaded,model,recipe,binding)
    first=invoke(loaded,collector,tmp_path/'parent',run_id,pause=1)
    assert first.report['status']=='paused' and first.report['updates']==1
    resumed=invoke(loaded,collector,tmp_path/'parent',run_id,mode='resume')
    assert resumed.report['status']=='complete' and resumed.report['updates']==2
    continuous=tmp_path/'continuous'
    continuous.mkdir()
    loaded2,model2,recipe2,binding2,run_id2=controlled_run(continuous,monkeypatch,budget=9,native=True,packed=packed)
    result=invoke(loaded2,collector_for(loaded2,model2,recipe2,binding2),continuous/'parent',run_id2)
    assert result.report['status']=='complete' and result.report['updates']==2
    from tests.test_structured_numeric_worker import assert_artifacts_equal
    assert_artifacts_equal(continuous/'parent/artifacts/000002',tmp_path/'parent/artifacts/000002')
    assert len(seen)>=4 and all(owner._prepared and owner.cleanup_complete for owner in seen)
    assert len(charged)==4 and all(projection._hpo_save_phase=='admitted' for projection in charged)
    for generation in (1,2):
        record=read_json(tmp_path/'parent/generations'/f'{generation:06}.json')
        update=read_json(tmp_path/'parent/updates'/f'{generation:06}.json')
        assert record['used_decisions']==min(generation*8,9)
        assert update['expected_optimizer_step_delta']==update['metrics']['optimizer_step_delta']


@pytest.mark.parametrize('fault',['missing','duplicate_ack','foreign_process','foreign_owner','generation',
    'steps','map','handle','extra','symlink','replaced_root','old_file','saved_payload','partial','cancel'])
def test_actual_cpu_native_ack_transition_faults_never_publish(tmp_path,monkeypatch,fault,native_save_backend):
    from dataclasses import replace

    from gf_training.card_instance_source import IssuedGameProjection
    from gf_training.native_backend import load_native
    assert load_native() is native_save_backend
    if 'card_instance_hpo' not in dict(load_native().build_info()):
        pytest.skip('actual HPO native required')
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1,native=True)
    captured=[]
    original_record=IssuedGameProjection._record_hpo_save_ack
    original_admit=IssuedGameProjection._admit_hpo_saved_output
    def record(self,process,reply):
        captured.append(self)
        assert self.source.native_inputs is not None and process._issued_handle is self
        if fault=='missing': return
        original_record(self,process,reply)
        if fault=='duplicate_ack': original_record(self,process,reply)
    def admit(self,process,generation,files,ownership):
        if fault=='foreign_process': process=object()
        elif fault=='foreign_owner': ownership=replace(ownership,token='foreign')
        elif fault=='generation': generation+=1
        elif fault=='steps': process.steps+=1
        elif fault=='map':
            files={**files,'inference.json':'f'*64}
        elif fault=='handle': process._issued_handle=object()
        artifacts=tmp_path/'parent/artifacts'
        if fault=='extra': (artifacts/'000002').mkdir()
        elif fault=='symlink':
            (artifacts/'000001').rename(tmp_path/'saved-elsewhere')
            (artifacts/'000001').symlink_to(tmp_path/'saved-elsewhere',target_is_directory=True)
        elif fault=='replaced_root':
            artifacts.rename(tmp_path/'original-artifacts')
            artifacts.mkdir()
        elif fault=='old_file':
            with (artifacts/'000000/inference.json').open('ab') as out: out.write(b' ')
        elif fault=='saved_payload':
            name=next(name for name in files if name.startswith('weights-') and name.endswith('.npz'))
            with (artifacts/'000001'/name).open('ab') as out: out.write(b'changed')
        try:
            return original_admit(self,process,generation,files,ownership)
        finally:
            if fault=='handle': self.process._issued_handle=None
    monkeypatch.setattr(IssuedGameProjection,'_record_hpo_save_ack',record)
    monkeypatch.setattr(IssuedGameProjection,'_admit_hpo_saved_output',admit)
    if fault in ('partial','cancel'):
        # Fault after the actual Rust permit and Python namespace admission,
        # before the native capsule can publish a successful saved-pair phase.
        import gf_training.structured_lineage as lineage
        original_append=lineage._admit_hpo_artifact_append
        def append(*args,**kwargs):
            original_append(*args,**kwargs)
            if fault=='cancel': raise asyncio.CancelledError('admission cancellation')
            raise OSError('partial admission fault')
        monkeypatch.setattr(lineage,'_admit_hpo_artifact_append',append)
    with pytest.raises((ContractError,ValueError,RuntimeError,OSError,asyncio.CancelledError)):
        invoke(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',run_id)
    assert captured
    projection=captured[0]
    assert not projection.prepared and projection._hpo_save_error is not None
    error=projection._hpo_save_error
    with pytest.raises(BaseException) as retry:
        original_admit(projection,projection.process,1,{},projection.source.online.ownership)
    assert retry.value is error
    assert not (tmp_path/'parent/generations/000001.json').exists()
    assert not (tmp_path/'parent/updates/000001.json').exists()
    state=read_json(tmp_path/'parent/run.json')
    if state['status']!='failed':
        # Poisoned physical namespaces can also prevent the failure-marker
        # write. They must remain structurally ineligible, never be repaired.
        assert fault in ('symlink','replaced_root')
        from gf_training.hpo_lineage import validate_hpo_chain
        with pytest.raises(ContractError):
            validate_hpo_chain(tmp_path/'parent',require_complete=False,max_owned_bytes=1024**3,deadline=time.monotonic()+30)


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


@pytest.mark.parametrize('diagnostics',[None,True])
def test_progress_observes_actual_commits_and_cleanup_without_authority(tmp_path,monkeypatch,diagnostics):
    from gf_training.hpo_ppo import train_hpo_ppo
    loaded,model,recipe,binding,run_id=controlled_run(tmp_path,monkeypatch,budget=1)
    import io

    from gf_cli.hpo_progress import ProgressOwner
    from gf_cli.hpo_storage import CampaignStore
    store=CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=8*1024**2)
    stderr=io.StringIO()
    owner=ProgressOwner(store,mode='off',stream=stderr)
    def failed_diagnostic_write(): raise OSError('diagnostic write failure')
    monkeypatch.setattr(owner,'_write',failed_diagnostic_write)
    observations=[]
    durable_matches=[]
    def progress(value):
        observations.append(value)
        owner.emit(value)
        if 'committed_used' in value:
            durable_matches.append(read_json(tmp_path/'parent'/'run.json')['used_decisions']==value['committed_used'])
        if value.get('phase')=='source':
            raise RuntimeError('display failure must not abort Source')
    with admit_gpu_job(JobContext('fixture','progress','training',run_id,None),
            global_cuda_bytes=loaded.config.run.global_cuda_budget_bytes,max_workers=4) as job:
        result=asyncio.run(train_hpo_ppo(loaded,collector_for(loaded,model,recipe,binding),tmp_path/'parent',
            gpu_job=job,progress=progress,**({} if diagnostics is None else {'diagnostics':True})))
        assert result.report['status']=='complete'
    phases=[value['phase'] for value in observations if 'phase' in value]
    assert phases.index('collect')<phases.index('source')<phases.index('update')<phases.index('cleanup')
    assert ('save' in phases)==bool(diagnostics)
    commits=[value for value in observations if 'committed_used' in value]
    assert [value['committed_used'] for value in commits]==[0,1]
    assert durable_matches==[True,True]
    assert commits[-1]['latest_metrics']['entropy'] is not None
    assert observations[-1]['status']=='complete'
    assert any(value.get('diagnostic_invocation') for value in observations)==bool(diagnostics)
    assert any('cuda_allocator' in value for value in observations)==bool(diagnostics)
    if diagnostics is None:
        assert all('GFB_HPO_DIAGNOSTIC_V1' not in path.read_text() for path in (tmp_path/'parent').glob('learner-*.log'))
    assert stderr.getvalue().count('persistence disabled')==1
    store.put('survives-learning.json',{'ok':True})
    store.close()
