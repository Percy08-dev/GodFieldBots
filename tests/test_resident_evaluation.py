"""Actual eval pool/Pack/publication with hand-authored unqualified game facts."""
import asyncio
import json
import time
from dataclasses import asdict, replace

import pytest

from gf_common.domain import RuntimePolicyBinding, clone, digest


@pytest.mark.parametrize('resident_parent,resident_execution',[(False,True),(True,False),(True,True)])
def test_actual_eval_collector_crosses_producer_and_explicit_runtime_independently(tmp_path,monkeypatch,resident_parent,resident_execution):
    import gf_cli.card_instance as core
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_evaluation import (
        _HpoEvaluationCleanupObservation,
        evaluation_receipt_runtime_fields,
        execution_plan,
    )
    from gf_common.card_instance_collection import (
        checked_hpo_evaluation_manifest,
        hpo_evaluation_groups,
    )
    from gf_common.schedule import serialize_schedule
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.sharded_arrays import read_json
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    from gf_training.group_pack_source import open_artifact_source
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.test_hpo_artifacts import configured_for
    from tests.test_hpo_eval_collector import scenario
    from tests.test_hpo_eval_failure_cleanup import install_component_execution
    from tests.test_resident_episode import ACTIONS, RULES

    loaded,old_plan,old_job,old_owner,_,_=scenario(tmp_path,monkeypatch)
    old_owner.release()
    old_job.confirm_termination()
    old_job.close()
    state,records,learner=initial_hpo_chain(tmp_path/'parent',resident=resident_parent)
    checkpoint=tmp_path/'parent/artifacts/000000'
    configured=replace(configured_for(learner,checkpoint),inference=replace(learner.binding,device='cuda:0'))
    start=time.monotonic()
    owner=RetainedGameArtifacts((configured,),ACTIONS,rules=RULES,max_owned_bytes=512*1024**2,
        preflight_deadline=start+30,finalization_deadline=start+90)
    label,binding=owner.binding(configured)
    bindings={label:binding,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
    config=replace(loaded.config,storage=replace(loaded.config.storage,db_root=str(tmp_path/'db'),ram_root=str(tmp_path/'ram')))
    if resident_parent:
        config=replace(config,hpo_spec=clone(state['effective_config']['hpo_spec']))
    raw=clone(asdict(config))
    loaded=LoadedConfig(config,raw,digest(raw),loaded.inputs)
    link=old_plan.linkage
    parent=link['left_parent']
    parent.update(run_id=state['run_id'],run_path=str(tmp_path/'parent'),training_seed=17,generation=0,
        generation_sha256=state['generation_sha256'],inference_sha256=records[0]['inference_sha256'],policy_label=label)
    job=admit_gpu_job(JobContext(link['campaign_id'],link['evaluation_id'],'evaluation',state['run_id'],link['evaluation_id']),
        global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=4)
    pin=pin_hpo_evaluation(tmp_path/'parent',job.ticket,0,records[0]['inference_sha256'],
        max_owned_bytes=1024**3,deadline=start+30)
    link['gpu_ticket']=job.ticket
    groups=hpo_evaluation_groups(link)
    schedule=serialize_schedule(groups,4)
    link.update(schedule_digest=digest(schedule),policy_registry_digest=digest({key:asdict(value) for key,value in bindings.items()}),
        learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']})
    runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728} if resident_execution else None
    plan=execution_plan(groups,bindings,tmp_path/'evaluation',link,runtime=runtime,pins=((tmp_path/'parent',pin),),
        started_monotonic=start,admission_deadline=start+20,execution_deadline=start+90,work_deadline=start+119,wall_deadline=start+120)
    install_component_execution(monkeypatch,cpu_processes=False)
    facts=CardInstanceSimulatorEnvironment.run_group
    async def component_group(self,group):
        # Own the real resident processes/leases; facts below remain explicitly
        # hand-authored and cannot pass replay/Source qualification.
        for spec in group.episodes:
            from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
            workers=[(self.resident_pool.worker(viewer,'heuristic',spec.sampling_seeds[viewer-1],ACTIONS,ledger=self.ledger)
                if self.resident_pool is not None else CardInstanceParticipantWorker('heuristic',spec.sampling_seeds[viewer-1],ACTIONS,ledger=self.ledger))
                for viewer in (1,2)]
            for worker in workers:
                await worker.initialize()
                if self.resident_pool is not None: worker.mark_completed()
            await asyncio.gather(*(worker.close() for worker in workers))
        return await facts(self,group)
    monkeypatch.setattr(CardInstanceSimulatorEnvironment,'run_group',component_group)
    observation=_HpoEvaluationCleanupObservation(plan,job,owner)
    progress=[]
    try:
        result=asyncio.run(core.collect_hpo_evaluation(loaded,plan,gpu_job=job,artifact_inputs=owner,_cleanup_observation=observation,progress=progress.append))
        phases=[value['phase'] for value in progress if 'phase' in value]
        assert phases[0]=='evaluate' and phases[-1]=='cleanup'
        assert any(value.get('transient_games',0)>0 for value in progress)
        assert all(value.get('transient_learner_decisions') is None for value in progress)
        observation.require_core_retired(error=ValueError('component completed'))
        assert not job.ledger.workers
        if resident_execution:
            owners=observation._resident_runtime.pool.owners
            assert len(owners)==4 and job.ledger.serial==4
            assert all(sum(owner.viewer==viewer for owner in owners)==2 for viewer in (1,2))
        assert job.ticket in read_json(tmp_path/'parent/artifact-registry.json')['evaluation_references']
        artifacts=open_artifact_source(result.locator,deadline=time.monotonic()+30,max_owned_bytes=512*1024**2)
        try:
            manifest=json.loads(artifacts.read('manifest.json'))
            checked_hpo_evaluation_manifest(manifest)
            assert manifest['schema_version']==(7 if resident_execution else 6)
            assert manifest['config']['hpo_spec']['schema_version']==(3 if resident_parent else 1)
            if resident_execution:
                assert manifest['execution_runtime']==plan.execution_runtime
                assert manifest['logical_retirement']['job_ticket']==job.ticket
                assert all(row['end_ack']['generation']==0 for row in manifest['logical_retirement']['owners'])
            else:
                assert 'execution_runtime' not in manifest and 'logical_retirement' not in manifest
            receipt_fields=evaluation_receipt_runtime_fields(manifest)
            assert receipt_fields['schema_version']==(2 if resident_execution else 1)
            if resident_execution: assert receipt_fields['execution_runtime_sha256']==digest(manifest['execution_runtime'])
            else: assert receipt_fields=={'schema_version':1}
        finally:
            artifacts.close()
        registry=read_json(tmp_path/'parent/artifact-registry.json')
        assert registry['schema_version']==(5 if resident_parent else 4)
        assert registry.get('cache_references',{})=={}
        if not resident_parent: assert 'cache_references' not in registry
    finally:
        owner.release()
        release_hpo_evaluation(tmp_path/'parent',pin.binding,max_owned_bytes=1024**3,deadline=start+119)
        job.confirm_termination()
        job.close()


@pytest.mark.parametrize('case',['duplicate-left','missing-right','conflicting-path','full','self'])
def test_evaluation_scope_requires_exact_distinct_genuine_parent_pin_coverage(tmp_path,monkeypatch,case):
    from gf_cli.hpo_evaluation import ResidentHpoEvaluationPlan
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import ContractError
    from gf_common.schedule import serialize_schedule
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_resident import EvaluationPoolScope
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from gf_training.hpo_source import hpo_binding
    from tests.hpo_support import variable_initial_hpo_chain
    from tests.test_hpo_evaluation_role import binding as linkage_fixture
    from tests.test_resident_episode import ACTIONS, RULES
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'admission')
    parents=[]
    bindings={}
    for index,kind in enumerate(('mlp','gru')):
        directory=tmp_path/f'parent-{index}'
        state,records,_=variable_initial_hpo_chain(directory,games=8,model_id=kind)
        label,binding=hpo_binding(directory,state,records[0],0)
        bindings[label]=binding
        parents.append({'run_id':state['run_id'],'run_path':str(directory),'training_seed':state['seed'],
            'generation':0,'generation_sha256':state['generation_sha256'],'inference_sha256':records[0]['inference_sha256'],
            'source_receipt_sha256':str(index+1)*64,'policy_label':label})
    if case=='self':
        parents[1]=clone(parents[0])
        bindings={parents[0]['policy_label']:bindings[parents[0]['policy_label']]}
    elif case=='conflicting-path':
        parents[1]=clone(parents[0])
        parents[1]['training_seed']+=1
        bindings={parents[0]['policy_label']:bindings[parents[0]['policy_label']]}
    link,_,_=linkage_fixture(comparison=True)
    link.update(left_parent=parents[0],right_parent=parents[1])
    with gpu.admit_gpu_job(gpu.JobContext(link['campaign_id'],link['evaluation_id'],'evaluation',parents[0]['run_id'],link['evaluation_id']),
            global_cuda_bytes=12*1024**3,max_workers=4) as job:
        link['gpu_ticket']=job.ticket
        groups=hpo_evaluation_groups(link)
        schedule=serialize_schedule(groups,4)
        link.update(schedule_digest=digest(schedule),policy_registry_digest=digest({key:asdict(value) for key,value in bindings.items()}),
            learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']})
        from pathlib import Path
        start=time.monotonic()
        actual=[]
        for parent in parents:
            if case in ('duplicate-left','missing-right') and actual: break
            path=Path(parent['run_path'])
            if any(path==previous[0] for previous in actual): continue
            acquired=pin_hpo_evaluation(path,job.ticket,0,parent['inference_sha256'],max_owned_bytes=1024**3,deadline=start+30)
            actual.append((path,acquired))
        supplied=tuple(actual)
        if case=='duplicate-left': supplied=(actual[0],actual[0])
        elif case=='missing-right': supplied=(actual[0],)
        plan=ResidentHpoEvaluationPlan.create_resident(groups,bindings,tmp_path/'evaluation',link,
            runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728},pins=supplied,
            started_monotonic=start,admission_deadline=start+20,execution_deadline=start+30,work_deadline=start+39,wall_deadline=start+40)
        owner=RetainedGameArtifacts.for_hpo_registry(bindings,ACTIONS,rules=RULES,pool_size=2,max_owned_bytes=512*1024**2,
            preflight_deadline=start+30,finalization_deadline=start+39)
        try:
            if case in ('full','self'):
                scope=EvaluationPoolScope(job,plan,owner,supplied,max_owned_bytes=1024**3,deadline=start+30)
                scope.require(job,plan)
            else:
                with pytest.raises(ContractError):
                    EvaluationPoolScope(job,plan,owner,supplied,max_owned_bytes=1024**3,deadline=start+30)
        finally:
            owner.release()
            for path,pin in actual:
                release_hpo_evaluation(path,pin.binding,max_owned_bytes=1024**3,deadline=start+39)
        job.confirm_termination()


def test_production_evaluation_policy_guard_still_rejects_cpu():
    from gf_cli.card_instance import _require_hpo_evaluation_policy
    from gf_common.domain import ContractError
    with pytest.raises(ContractError,match='CUDA'):
        _require_hpo_evaluation_policy(RuntimePolicyBinding('mlp',{'version':'4','inference':{'device':'cpu'}}))
