"""Final data enters a separately admitted evaluation scope after training closes."""
import asyncio
import time
import uuid
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, digest


@pytest.mark.parametrize('session_gap,expired_training',[(False,False),(False,True),(True,False)])
def test_actual_completion_uses_new_evaluation_job_and_inference_only_actor9(tmp_path,monkeypatch,session_gap,expired_training):
    from gf_cli.hpo_evaluation import _final_policy_reference, completion_inspection, execution_plan
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import RuntimePolicyBinding
    from gf_common.group_pack import StorageSettings
    from gf_common.schedule import serialize_schedule
    from gf_common.structured import event_delta
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_completion import completion_artifacts, retain_for_evaluation
    from gf_training.hpo_resident_runtime import ResidentEvaluationRuntime
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_owned_support import owned_runtime, run_owned
    from tests.model3_support import raw_inputs
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        _,training=await run_owned(runtime,directory)
        original=training.completion
        assert all('parent_controls' not in record for record in original.records)
        if expired_training: training.budget.deadline=time.monotonic()-1
        old_deadline=training.budget.deadline
        if session_gap: original.close()
        with pytest.raises(ContractError): training.job._require_open()
        evaluation_id=str(uuid.uuid4())
        with gpu.admit_gpu_job(gpu.JobContext('fixture',evaluation_id,'evaluation',run_id,evaluation_id),
                global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes,max_workers=4) as job:
            assert job is not training.job and job.ticket!=training.job.ticket
            deadline=time.monotonic()+30
            account=StorageBudget(StorageSettings(),deadline+1)
            phase=ArtifactPhaseAdmission(account,runtime.config.hpo_spec['operating']['parent_budget_bytes'])
            if session_gap:
                from gf_training.hpo_owned_restore import import_owned_final_policy
                data=import_owned_final_policy(directory,deadline=deadline,max_owned_bytes=account.limit,parent_budget_bytes=phase.limit,
                    budget=account,parent_budget=phase)
            else:
                data=retain_for_evaluation(original,budget=account,parent_budget=phase)
                original.close()
            assert training.budget.used==0 and training.budget.deadline==old_deadline
            parent,binding=_final_policy_reference(completion_inspection(data,directory),'a'*64)
            label=parent['policy_label']
            bindings={label:binding,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
            reference=pin_hpo_evaluation(directory,job.ticket,2,parent['inference_sha256'],max_owned_bytes=2*1024**3,deadline=deadline)
            link={'schema_version':1,'kind':'hpo-evaluation-linkage','campaign_id':'fixture','evaluation_id':evaluation_id,
                'gpu_ticket':job.ticket,'purpose':'objective','conditions_sha256':'b'*64,'left_parent':parent,'right_parent':None,
                'game_seeds':['00000000-0000-0000-0000-000000000001'],'learner_seats':{},'schedule_digest':'0'*64,
                'policy_registry_digest':digest({name:asdict(value) for name,value in bindings.items()})}
            groups=hpo_evaluation_groups(link); schedule=serialize_schedule(groups,4)
            link.update(learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']},schedule_digest=digest(schedule))
            plan=execution_plan(groups,bindings,tmp_path/'evaluation',link,runtime=runtime.config.hpo_spec['runtime'],pins=((directory,reference),),
                started_monotonic=time.monotonic(),admission_deadline=deadline-2,execution_deadline=deadline-1,work_deadline=deadline,wall_deadline=deadline+1)
            artifacts=completion_artifacts(data,binding,label)
            resident=ResidentEvaluationRuntime(job,plan,artifacts,runtime.config)
            try:
                for viewer in (1,1):
                    logical=resident.pool.worker(viewer,binding.model_id,17+viewer,runtime.config.actions,
                        ledger=job.ledger,policy_options=binding.policy_options,artifact_inputs=artifacts)
                    await logical.initialize()
                    _,_,frames,bank,_=raw_inputs(count=1); frame=frames[0]
                    delta=event_delta(1,frame.episode,frame.interval_start,frame.interval_start_hash,
                        bank.view().events(frame.interval_start,frame.boundary.end),layout=frame.layout)
                    generated,selection=await logical.select_frame(frame,delta)
                    assert selection.validate(generated.table,frame) is not None
                    logical.mark_completed(); await logical.close()
            finally:
                await resident.close(); artifacts.release()
            release_hpo_evaluation(directory,reference.binding,max_owned_bytes=2*1024**3,deadline=deadline)
            assert job.confirm_termination().ticket==job.ticket
        data.require(); data.close()
        assert account.used==0 and phase.used==0 and training.budget.used==0
    asyncio.run(run())


@pytest.mark.parametrize('mutation',['source','snapshot','outer','selected','set'])
def test_evaluation_use_requires_original_set_source_and_outer_epoch(mutation):
    from gf_common.hpo_actor_snapshot import checked_evaluation_use
    parent={'run_id':'parent','run_path':'/parent','training_seed':17,'generation':2,'generation_sha256':'a'*64,
        'inference_sha256':'b'*64,'source_receipt_sha256':'c'*64,'policy_label':'policy'}
    selected={'parent':parent,'snapshot_identity':'d'*64}
    request={'run_id':'evaluation','generation':0,'command':'BEGIN','viewer':1,'request_epoch':3,'lease_epoch':2}
    use={'schema_version':1,'kind':'owned-evaluation-final-policy-use-v1','evaluation_id':'evaluation',
        'finals':[selected],'selected':selected,'outer_request':request}
    metadata={'generation':2,'snapshot_identity':'d'*64,'inference_digest':'b'*64}
    options={'checkpoint':'/parent/artifacts/000002','expected_manifest_digest':'b'*64}
    assert checked_evaluation_use(use,metadata,request,options)
    import copy
    changed=copy.deepcopy(use)
    if mutation=='source': changed['selected']['parent']['generation']=3
    elif mutation=='snapshot': changed['selected']['snapshot_identity']='e'*64
    elif mutation=='outer': changed['outer_request']['lease_epoch']=3
    elif mutation=='selected': changed['selected']={**selected,'snapshot_identity':'e'*64}
    else: changed['finals']=[]
    with pytest.raises(ContractError): checked_evaluation_use(changed,metadata,request,options)


def test_actual_completion_and_session_receipt_validation_stay_metadata_only(tmp_path,monkeypatch):
    import json

    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_evaluation import persist_training_receipt, verify_training_receipt
    from gf_cli.hpo_storage import CampaignStore
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,campaign_config,settings=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        _,ops=await run_owned(runtime,directory)
        completion=ops.completion
        state=completion.state
        campaign={'campaign_id':'fixture','conditions':{'producer_identity':state['identity'],
            'integration_identity':state['identity']['integration'],'inference_binding':state['execution']['inference']['binding'],
            'cuda_visible_devices':state['execution']['inference']['cuda_visible_devices']}}
        proposal=Proposal(0,0,0,'a'*64,settings,'{}','{}')
        def forbidden(*args,**kwargs): raise AssertionError('historical samples in receipt path')
        monkeypatch.setattr('gf_training.hpo_source.validate_hpo_parent',forbidden)
        with CampaignStore(tmp_path/'campaign',mode='fresh',max_owned_bytes=256*1024**2) as store:
            receipt=persist_training_receipt(store,campaign_config,campaign,proposal,training_seed=17,run_id=run_id,
                directory=directory,termination=ops.receipt,deadline=time.monotonic()+30,completion=completion)
            source=store.get('proposals/000000/seeds/17/source-validation.json')[0]
            assert source['schema_version']==2 and source['kind']=='hpo-owned-parent-metadata-validation'
            completion.close()
            assert ops.budget.used==0
            restored=verify_training_receipt(store,campaign_config,campaign,proposal,17,deadline=time.monotonic()+30)
            assert restored==receipt and type(restored) is dict
            assert not hasattr(restored,'actor_capability')
            assert json.loads((directory/'run.json').read_text())['status']=='complete'
    asyncio.run(run())


def test_owned_origin_cannot_enter_legacy_source_factory():
    from gf_training.card_instance_source import _open_batch
    with (pytest.raises(ContractError,match='legacy Proof4'),
            _open_batch(None,{'run_variant':'hpo-ppo-owned-v1'},None,None,None,
                online=None,max_owned_bytes=1024,deadline=1)):
        pytest.fail('old Source issued owned authority')


@pytest.mark.parametrize('mixed',[False,True])
def test_comparison_profile_selects_owned_import_or_strict_external_artifacts(tmp_path,monkeypatch,mixed):

    from gf_cli.hpo_config import campaign_config
    from gf_cli.hpo_evaluation import (
        _comparison_artifacts,
        _final_policy_reference,
        completion_inspection,
    )
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,left,_settings=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        _,ops=await run_owned(runtime,directory)
        parent,binding=_final_policy_reference(completion_inspection(ops.completion,directory),'a'*64)
        ops.completion.close()
        right=left
        if mixed:
            from tests.test_hpo_config import campaign_raw
            raw=campaign_raw()
            raw.update(schema_version=3,runtime=left.effective['runtime'],storage=left.effective['storage'])
            right=campaign_config(raw,origin=tmp_path)
        owner,data=_comparison_artifacts(left,right,[(directory,parent['policy_label'],binding)],
            {parent['policy_label']:binding},runtime.config,deadline=time.monotonic()+30,work=time.monotonic()+40,max_owned_bytes=256*1024**2)
        try:
            assert type(owner) is (RetainedGameArtifacts if mixed else OwnedGameArtifacts)
            owner.require_options(binding.model_id,binding.policy_options)
            assert owner.finalize(deadline=time.monotonic()+30)['policies'][parent['policy_label']]['inference_manifest']['schema_version']==5
            if mixed: assert data==[] and not hasattr(owner,'owned_snapshots')
            else:
                assert len(data)==1 and owner.owned_snapshots[0].generation==2
                assert owner.owned_snapshots[0].require() is data[0]
        finally:
            owner.release()
            for item in data: item.close()
        assert ops.budget.used==0
    asyncio.run(run())


@pytest.mark.parametrize('same_parent',[False,True])
def test_paired_comparison_alternates_two_authorized_final_snapshots(tmp_path,monkeypatch,same_parent):
    from dataclasses import replace

    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_evaluation import (
        _comparison_artifacts,
        _final_policy_reference,
        completion_inspection,
        execution_plan,
    )
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.schedule import serialize_schedule
    from gf_common.structured import event_delta
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_resident_runtime import ResidentEvaluationRuntime
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_owned_support import owned_runtime, run_owned
    from tests.model3_support import raw_inputs
    runtime,left_id,profile,_=owned_runtime(tmp_path,monkeypatch)
    right_id='00000000-0000-0000-0000-000000000029'
    config=replace(runtime.config,run=replace(runtime.config.run,name=right_id),training=replace(runtime.config.training,seeds=(29,)))
    raw=asdict(config)
    right=LoadedConfig(config,raw,digest(raw),runtime.inputs)
    async def run():
        refs=[]; sources=[]; bindings={}
        cases=((runtime,left_id),) if same_parent else ((runtime,left_id),(right,right_id))
        for loaded,run_id in cases:
            directory=tmp_path/'parents'/run_id
            _,ops=await run_owned(loaded,directory)
            reference,binding=_final_policy_reference(completion_inspection(ops.completion,directory),'a'*64)
            refs.append(reference); sources.append((directory,reference['policy_label'],binding)); bindings[reference['policy_label']]=binding
            ops.completion.close()
        if same_parent:
            refs.append(refs[0]); sources.append(sources[0])
        owner,data=_comparison_artifacts(profile,profile,sources,bindings,runtime.config,
            deadline=time.monotonic()+60,work=time.monotonic()+70,max_owned_bytes=256*1024**2)
        assert len(data)==(1 if same_parent else 2) and len(owner.owned_snapshots)==(1 if same_parent else 2)
        evaluation_id=str(uuid.uuid4())
        with gpu.admit_gpu_job(gpu.JobContext('fixture',evaluation_id,'evaluation',left_id,evaluation_id),
                global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes,max_workers=4) as job:
            deadline=time.monotonic()+40
            pins=tuple((path,pin_hpo_evaluation(path,job.ticket,2,ref['inference_sha256'],max_owned_bytes=2*1024**3,deadline=deadline))
                for (path,_,_),ref in zip(sources[:1] if same_parent else sources,refs[:1] if same_parent else refs,strict=True))
            link={'schema_version':1,'kind':'hpo-evaluation-linkage','campaign_id':'fixture','evaluation_id':evaluation_id,
                'gpu_ticket':job.ticket,'purpose':'comparison','conditions_sha256':'b'*64,'left_parent':refs[0],'right_parent':refs[1],
                'game_seeds':['00000000-0000-0000-0000-000000000001'],'learner_seats':{},'schedule_digest':'0'*64,
                'policy_registry_digest':digest({name:asdict(value) for name,value in bindings.items()})}
            groups=hpo_evaluation_groups(link); schedule=serialize_schedule(groups,4)
            link.update(learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']},schedule_digest=digest(schedule))
            plan=execution_plan(groups,bindings,tmp_path/'comparison',link,runtime=runtime.config.hpo_spec['runtime'],pins=pins,
                started_monotonic=time.monotonic(),admission_deadline=deadline-2,execution_deadline=deadline-1,work_deadline=deadline,wall_deadline=deadline+1)
            resident=ResidentEvaluationRuntime(job,plan,owner,runtime.config)
            seen=[]
            try:
                # Viewer1's policy switches on the paired seat swap. The second
                # BEGIN uses the SAME fresh physical process and frozen final set.
                for group in groups:
                    for spec in group.episodes:
                        binding=bindings[spec.policies[0]]; seen.append(spec.policies[0])
                        logical=resident.pool.worker(1,binding.model_id,spec.sampling_seeds[0],runtime.config.actions,
                            ledger=job.ledger,policy_options=binding.policy_options,artifact_inputs=owner)
                        await logical.initialize()
                        _,_,frames,bank,_=raw_inputs(count=1); frame=frames[0]
                        delta=event_delta(1,frame.episode,frame.interval_start,frame.interval_start_hash,
                            bank.view().events(frame.interval_start,frame.boundary.end),layout=frame.layout)
                        await logical.select_frame(frame,delta)
                        logical.mark_completed(); await logical.close()
                assert set(seen)==set(bindings)
                assert len(resident.pool.owners)==1
                assert resident.pool.custody.evaluation_scope._owned_final_set is not None
                with pytest.raises(ContractError):
                    resident.custody.evaluation_scope.owned_use({},object(),binding)
            finally:
                await resident.close(); owner.release()
            for path,pin in pins: release_hpo_evaluation(path,pin.binding,max_owned_bytes=2*1024**3,deadline=deadline)
            job.confirm_termination()
        for item in data:
            account=item._budget
            item.close()
        assert account.used==0
    asyncio.run(run())


def test_owned_handoff_actual_collector_publication_and_strict_eval_audit(tmp_path,monkeypatch):
    """Controlled public endpoints; real actors/writer/Source; no qualified game claim."""
    from dataclasses import replace

    import gf_cli.card_instance as core
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_evaluation import (
        _final_policy_reference,
        _HpoEvaluationCleanupObservation,
        completion_inspection,
        execution_plan,
        verify_saved_evaluation,
    )
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import RuntimePolicyBinding
    from gf_common.group_pack import StorageSettings
    from gf_common.schedule import serialize_schedule
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_completion import completion_artifacts, retain_for_evaluation
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_owned_support import owned_runtime, run_owned
    from tests.test_hpo_owned_run import install_owned_fixture_runtime
    runtime,run_id,profile,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    install_owned_fixture_runtime(monkeypatch)
    # The training capture fixture deliberately uses diagnostic-only Replay.
    # Evaluation exercises strict reconciliation, so emit both actual own views.
    import copy

    from tests.test_group_pack_runtime_environment import PublicClient
    replay=PublicClient.replay
    async def own_replay(self,game,viewer):
        result=copy.deepcopy(await replay(self,game,viewer))
        for turn in result['turns']:
            for decision in turn['decisions']:
                observation=decision['observation']
                observation.update(viewer_player_id=viewer,current_player=decision['actor'],active_player=decision['actor'])
                observation['own']['player_id']=viewer
                observation['opponent']['player_id']=3-viewer
                decision['action_options']['actor']=decision['actor']
        return result
    monkeypatch.setattr(PublicClient,'replay',own_replay)
    def cpu_component(binding):
        assert binding.policy_options['version']=='4' and binding.policy_options['inference']['device']=='cpu'
    monkeypatch.setattr(core,'_require_hpo_evaluation_policy',cpu_component)
    async def run():
        _,training=await run_owned(runtime,directory)
        evaluation_id=str(uuid.uuid4())
        config=replace(runtime.config,run=replace(runtime.config.run,split='validation',name=evaluation_id,
            output_dir=str(tmp_path/'evaluations'),seeds=profile.config.evaluation_game_seeds,
            sampling_seeds=(profile.config.sampler_seed,),pairs_per_seed=1,
            wall_seconds=120,execution_seconds=90,admission_seconds=20,finalization_seconds=30))
        raw=asdict(config); projected=LoadedConfig(config,raw,digest(raw),runtime.inputs)
        start=time.monotonic(); execution=start+90; wall=start+120; work=wall-config.run.report_reserve_seconds
        account=StorageBudget(StorageSettings(),work); phase=ArtifactPhaseAdmission(account,profile.config.operating.parent_budget_bytes)
        data=retain_for_evaluation(training.completion,budget=account,parent_budget=phase)
        training.completion.close()
        parent,binding=_final_policy_reference(completion_inspection(data,directory),'a'*64)
        label=parent['policy_label']; bindings={label:binding,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
        with gpu.admit_gpu_job(gpu.JobContext('fixture',evaluation_id,'evaluation',run_id,evaluation_id),
                global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=4) as job:
            pin=pin_hpo_evaluation(directory,job.ticket,2,parent['inference_sha256'],max_owned_bytes=2*1024**3,deadline=execution)
            link={'schema_version':1,'kind':'hpo-evaluation-linkage','campaign_id':'fixture','evaluation_id':evaluation_id,
                'gpu_ticket':job.ticket,'purpose':'objective','conditions_sha256':'b'*64,'left_parent':parent,'right_parent':None,
                'game_seeds':list(profile.config.evaluation_game_seeds),'learner_seats':{},'schedule_digest':'0'*64,
                'policy_registry_digest':digest({name:asdict(value) for name,value in bindings.items()})}
            groups=hpo_evaluation_groups(link); schedule=serialize_schedule(groups,4)
            link.update(learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']},schedule_digest=digest(schedule))
            plan=execution_plan(groups,bindings,tmp_path/'evaluations'/evaluation_id,link,runtime=config.hpo_spec['runtime'],pins=((directory,pin),),
                started_monotonic=start,admission_deadline=start+20,execution_deadline=execution,work_deadline=work,wall_deadline=wall)
            artifacts=completion_artifacts(data,binding,label)
            observation=_HpoEvaluationCleanupObservation(plan,job,artifacts)
            try:
                result=await core.collect_hpo_evaluation(projected,plan,gpu_job=job,artifact_inputs=artifacts,_cleanup_observation=observation)
                assert result.report['healthy'] and result.storage_metrics['publication_scope']=='published'
                audit_profile=replace(profile,base=replace(profile.base,config=replace(profile.base.config,actions=config.actions)))
                manifest,proof=verify_saved_evaluation(result.directory,audit_profile,link,deadline=work,
                    extra_owned_bytes=artifacts.retained_bytes,_cleanup_observation=observation,evaluation_runtime=config.hpo_spec['runtime'])
                assert proof['closure_counts']=={'natural':2} and manifest['linkage']==link
                observation.release_artifacts()
            except BaseException as failure:
                failure.add_note('controlled episode failures: '+repr([[(episode.failure,episode.health) for episode in row[3].episodes] for row in observation._groups.values() if row[3] is not None]))
                raise
            finally:
                artifacts.release()
                release_hpo_evaluation(directory,pin.binding,max_owned_bytes=2*1024**3,deadline=work)
            assert job.confirm_termination().resources
        data.close()
        assert account.used==0 and phase.used==0 and training.budget.used==0
    asyncio.run(run())
