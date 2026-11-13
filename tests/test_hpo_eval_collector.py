"""Core dispatch/custody/deadline checks; the collection I/O is a test-owned spy.

CPU-saved inference weights support a planned CUDA binding without running GPU
inference. No returned spy result is treated as Source or evaluation authority.
"""
import asyncio
import time
import uuid
from dataclasses import asdict, replace

import pytest

pytest.importorskip('torch')

from gf_common.domain import ContractError, RuntimePolicyBinding, digest
from gf_common.schedule import serialize_schedule
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_training.collection import PackedRunResult
from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
from tests.test_card_instance_learning import ACTIONS, RULES


def scenario(tmp_path,monkeypatch):
    import gf_training.hpo_gpu_lease as gpu
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_cli.hpo_evaluation import HpoEvaluationPlan
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_training.hpo_config import effective_settings
    from tests.model3_support import LAYOUT
    from tests.test_hpo_artifacts import configured_for, saved
    from tests.test_hpo_config import campaign_raw
    from tests.test_hpo_evaluation_role import binding
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'gpu')
    learner,checkpoint,receipt=saved(tmp_path,trained=False)
    configured=replace(configured_for(learner,checkpoint),inference=replace(learner.binding,device='cuda:0'))
    owner=RetainedGameArtifacts((configured,),ACTIONS,rules=RULES,max_owned_bytes=256*1024**2,
        preflight_deadline=time.monotonic()+60,finalization_deadline=time.monotonic()+180)
    label,execution=owner.binding(configured)
    registry={label:execution,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
    link,_,_=binding()
    link['left_parent'].update(policy_label=label,inference_sha256=receipt['inference_digest'])
    job=admit_gpu_job(JobContext(link['campaign_id'],link['evaluation_id'],'evaluation',link['left_parent']['run_id'],link['evaluation_id']),
        global_cuda_bytes=12*1024**3,max_workers=4)
    link['gpu_ticket']=job.ticket
    groups=hpo_evaluation_groups(link)
    serialized=serialize_schedule(groups,4)
    link.update(schedule_digest=digest(serialized),policy_registry_digest=digest({key:asdict(value) for key,value in registry.items()}),
        learner_seats={spec['episode_id']:spec['learner_viewer'] for group in serialized for spec in group['episodes']})
    loaded_campaign=campaign_config(campaign_raw(),origin=tmp_path)
    settings=effective_settings(learner.model,learner.training,{},layout=LAYOUT)
    loaded=materialize_hpo_run(loaded_campaign,settings,learning_seed=17,run_id=str(uuid.uuid4()))
    config=replace(loaded.config,actions=ACTIONS,run=replace(loaded.config.run,split='validation',wall_seconds=120,
        admission_seconds=20,execution_seconds=90,finalization_seconds=30))
    raw=asdict(config)
    loaded=LoadedConfig(config,raw,digest(raw),loaded.inputs)
    start=time.monotonic()
    plan=HpoEvaluationPlan.create(groups,registry,tmp_path/'evaluation',link,started_monotonic=start,
        admission_deadline=start+20,execution_deadline=start+90,work_deadline=start+119,wall_deadline=start+120)
    return loaded,plan,job,owner,configured,start


@pytest.mark.parametrize('finish_seconds,accept',[(30,True),(100,False)])
def test_admitted_eval_can_finish_after_admission_but_never_after_execution(tmp_path,monkeypatch,finish_seconds,accept):
    import gf_cli.card_instance as core
    from gf_training.group_pack_source import CollectionLocator
    loaded,plan,job,owner,configured,start=scenario(tmp_path,monkeypatch)
    clock=[start]
    class Runtime:
        def __init__(self,*args,**kwargs): pass
        def preflight(self): return {}
    async def spy(actual_loaded,command,bindings,labels,**options):
        assert actual_loaded is loaded and command=='evaluate'
        assert options['artifact_inputs'] is owner and options['owned_ledger'] is job.ledger
        assert options['collection'] is plan
        assert bindings==plan.bindings and set(labels)==set(bindings)
        clock[0]=start+finish_seconds
        return PackedRunResult(plan.output,{'healthy':True,'comparison_valid':True},
            CollectionLocator('sqlite-group-pack-v1',loaded.config.storage.settings().db_root/'evaluation'/'collection.published.json'),
            'a'*64,{'publication_scope':'published'})
    monkeypatch.setattr(core,'CardInstanceRuntimeEvidence',Runtime)
    monkeypatch.setattr(core,'_run_corrected',spy)
    monkeypatch.setattr(core.time,'monotonic',lambda:clock[0])
    try:
        if accept:
            result=asyncio.run(core.collect_hpo_evaluation(loaded,plan,gpu_job=job,artifact_inputs=owner))
            assert result.directory==plan.output
        else:
            with pytest.raises(TimeoutError):
                asyncio.run(core.collect_hpo_evaluation(loaded,plan,gpu_job=job,artifact_inputs=owner))
        assert owner.binding(configured)[0]==plan.linkage['left_parent']['policy_label']
        assert not job._sealed  # Outer owns pin/artifact retirement and confirmation.
    finally:
        owner.release()
        job.confirm_termination()
        job.close()


def test_eval_collector_refuses_path_or_dict_instead_of_actual_retained_owner(tmp_path,monkeypatch):
    from gf_cli.card_instance import collect_hpo_evaluation
    loaded,plan,job,owner,_,_=scenario(tmp_path,monkeypatch)
    try:
        with pytest.raises(ContractError):
            asyncio.run(collect_hpo_evaluation(loaded,plan,gpu_job=job,artifact_inputs={'checkpoint':'not-an-owner'}))
    finally:
        owner.release()
        job.confirm_termination()
        job.close()
