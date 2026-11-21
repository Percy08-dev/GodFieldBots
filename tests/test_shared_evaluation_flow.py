"""Synthetic full-D parent and fresh evaluation: real owners, no learning claim."""
import asyncio
import copy
import os
import time
import uuid
from dataclasses import asdict, replace

import pytest


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
def test_shared_final_reader_fresh_evaluation_and_saved_cold_verification(tmp_path,monkeypatch):
    import gf_cli.card_instance as core
    import gf_simulator_client.card_instance_environment as environment
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_evaluation import (
        _final_policy_reference,
        _HpoEvaluationCleanupObservation,
        completion_inspection,
        execution_plan,
        inspect_training_parent,
        verify_saved_evaluation,
    )
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.group_pack import StorageSettings
    from gf_common.schedule import serialize_schedule
    from gf_common.shared_inference import evaluation_runtime
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_completion import completion_artifacts, retain_for_evaluation
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from gf_training.ppo_run import _run_ppo
    from tests.shared_inference_support import fixture_runtime_receipt, shared_owned_configuration
    from tests.test_group_pack_runtime_environment import Evidence, PublicClient
    from tests.test_hpo_owned_run import install_owned_fixture_runtime, owned_fixture_collection
    runtime,run_id,profile,settings=shared_owned_configuration(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    install_owned_fixture_runtime(monkeypatch)
    class ProfileEvidence(Evidence):
        def __init__(self,*args,contract=None,**kwargs):
            super().__init__(*args,**kwargs)
            self.contract=runtime.config.actions.action_contract if contract is None else contract
        def preflight(self):return fixture_runtime_receipt(self.contract)
        def receipt(self,**kwargs):return fixture_runtime_receipt(self.contract)
        def finalize(self):return fixture_runtime_receipt(self.contract)
    monkeypatch.setattr(core,'CardInstanceRuntimeEvidence',ProfileEvidence)
    monkeypatch.setattr(environment,'CardInstanceRuntimeEvidence',ProfileEvidence)
    original_replay=PublicClient.replay
    async def own_replay(self,game,viewer):
        value=copy.deepcopy(await original_replay(self,game,viewer))
        for turn in value['turns']:
            for decision in turn['decisions']:
                observation=decision['observation']
                observation.update(viewer_player_id=viewer,current_player=decision['actor'],active_player=decision['actor'])
                observation['own']['player_id']=viewer
                observation['opponent']['player_id']=3-viewer
                decision['action_options']['actor']=decision['actor']
        return value
    monkeypatch.setattr(PublicClient,'replay',own_replay)
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
            global_cuda_bytes=runtime.config.run.global_cuda_budget_bytes,max_workers=2) as training_job:
            training=_OwnedHpoOperations(training_job)
            async def collect(plan,ledger):return await owned_fixture_collection(training,runtime.config,plan,ledger)
            await _run_ppo(runtime,collect,directory,_game_ops=training)
        assert training.receipt.schema_version==2
        inspected=inspect_training_parent(directory,profile,settings,training_seed=17,run_id=run_id,
            deadline=time.monotonic()+30,completion=training.completion)
        assert inspected.state['run_variant']=='hpo-ppo-owned-shared-v1'
        del inspected  # Its explicitly charged management snapshot is no longer live.
        evaluation_id=str(uuid.uuid4())
        config=replace(runtime.config,run=replace(runtime.config.run,split='validation',name=evaluation_id,
            output_dir=str(tmp_path/'evaluations'),seeds=profile.config.evaluation_game_seeds,
            sampling_seeds=(profile.config.sampler_seed,),pairs_per_seed=1))
        projected=LoadedConfig(config,asdict(config),digest(asdict(config)),runtime.inputs)
        start=time.monotonic();execution=start+90;work=start+119;wall=start+120
        account=StorageBudget(StorageSettings(),work)
        phase=ArtifactPhaseAdmission(account,profile.config.operating.parent_budget_bytes)
        data=retain_for_evaluation(training.completion,budget=account,parent_budget=phase)
        training.completion.close()
        parent,binding=_final_policy_reference(completion_inspection(data,directory),'a'*64)
        label=parent['policy_label']
        bindings={label:binding,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
        with gpu.admit_gpu_job(gpu.JobContext('fixture',evaluation_id,'evaluation',run_id,evaluation_id),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=2) as job:
            pin=pin_hpo_evaluation(directory,job.ticket,2,parent['inference_sha256'],max_owned_bytes=2*1024**3,deadline=execution)
            link={'schema_version':1,'kind':'hpo-evaluation-linkage','campaign_id':'fixture','evaluation_id':evaluation_id,
                'gpu_ticket':job.ticket,'purpose':'objective','conditions_sha256':'b'*64,'left_parent':parent,'right_parent':None,
                'game_seeds':list(profile.config.evaluation_game_seeds),'learner_seats':{},'schedule_digest':'0'*64,
                'policy_registry_digest':digest({name:asdict(value) for name,value in bindings.items()})}
            groups=hpo_evaluation_groups(link);schedule=serialize_schedule(groups,4)
            link.update(learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']},schedule_digest=digest(schedule))
            assert evaluation_runtime(profile.config) is None
            plan=execution_plan(groups,bindings,tmp_path/'evaluations'/evaluation_id,link,runtime=None,pins=((directory,pin),),
                started_monotonic=start,admission_deadline=start+20,execution_deadline=execution,work_deadline=work,wall_deadline=wall)
            artifacts=completion_artifacts(data,binding,label)
            observation=_HpoEvaluationCleanupObservation(plan,job,artifacts)
            try:
                result=await core.collect_hpo_evaluation(projected,plan,gpu_job=job,artifact_inputs=artifacts,_cleanup_observation=observation)
                assert result.report['healthy'] and job.ledger.service is None
                manifest,proof=verify_saved_evaluation(result.directory,profile,link,deadline=work,
                    extra_owned_bytes=artifacts.retained_bytes,_cleanup_observation=observation,evaluation_runtime=None)
                assert 'execution_runtime' not in manifest and proof['closure_counts']=={'natural':2}
                observation.release_artifacts()
            finally:
                artifacts.release()
                release_hpo_evaluation(directory,pin.binding,max_owned_bytes=2*1024**3,deadline=work)
            assert job.confirm_termination().schema_version==1
        data.close()
        cold=inspect_training_parent(directory,profile,settings,training_seed=17,run_id=run_id,deadline=time.monotonic()+30)
        assert cold.state['used_decisions']==32
        assert account.used==0 and training.budget.used==0
    asyncio.run(run())
