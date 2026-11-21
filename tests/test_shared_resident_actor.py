"""CPU resident transport uses explicit drain evidence and the common MLP service."""
import asyncio
import time
from dataclasses import asdict
from types import SimpleNamespace

from tests.test_hpo_gpu_lease import admit
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414


def test_actual_resident_cpu_actor_has_no_cuda_drain_claim(global_root,tmp_path):
    from gf_common.domain import RuntimePolicyBinding
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_common.shared_inference import ServingBudget, shared_runtime
    from gf_models.structured_binding import input_binding_game_v3, safe_options_game_v4
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_simulator_client.shared_service import SharedInferenceService
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_resident_episode import initial
    learner=learner_for('mlp')
    checkpoint='b'*64
    options=safe_options_game_v4({'version':'4','checkpoint':str(tmp_path/'absent'),
        'expected_manifest_digest':checkpoint,'model':asdict(learner.model),'inference':asdict(learner.binding),
        'cuda_visible_devices':'','action_config_hash':learner.action_config_hash,
        **input_binding_game_v3(ACTIONS.action_contract,RULES)},actions=ACTIONS,rules=RULES)
    binding=RuntimePolicyBinding('mlp',options)
    banks=ReadonlyRamBankSet.build({name:value.detach().numpy() for name,value in learner.network.state_dict().items()},ServingBudget(128*1024**2))
    async def run():
        with admit() as job:
            service=SharedInferenceService(job.ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
                execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu',job_id=job.ticket)
            service.publish(banks,model=asdict(learner.model),binding=asdict(learner.binding),checkpoint=checkpoint,generation=0,identity='c'*64)
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='d'*32,shared=True)
            pool=ResidentParticipantPool(custody,resident_bytes=0,deadline=time.monotonic()+60,shared=True)
            pool.service=custody.service=service
            pool.admit(CollectionPlan((),{'model':binding},tmp_path,{'parent_run_id':'run-a',
                'collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            owner=ResidentPhysicalWorker(pool,1)
            pool.owners.append(owner)
            try:
                await owner.bootstrap(timeout=15)
                custody.begin_admitted_lease(pool.admission,owner.lease,binding=binding)
                request=custody.episode_request(owner.lease,'BEGIN')
                value=initial('mlp',options=options)
                response=await owner.exchange({'request':request,'payload':value},timeout=15)
                custody.acknowledge_episode(owner.lease,response)
                from tests.test_card_instance_baseline_runtime import game
                from tests.test_resident_episode import message
                policy_version=response['payload']['policy_version']
                owner.loan=SimpleNamespace(model_id='mlp',game='actual-component-game-id')
                request=custody.episode_request(owner.lease,'SELECT')
                frame,_bank,delta,_=game()
                payload=message(frame,delta)
                response=await owner.exchange({'request':request,'payload':payload},timeout=15)
                actual=custody.acknowledge_episode(owner.lease,response)
                assert actual['selection']['policy_version']==policy_version
                owner.loan=None
                for command in ('END','PARK'):
                    await owner.control(command,checkpoint=checkpoint,timeout=15)
                row=custody._row(owner.lease)
                assert row.park_ack['cpu_only'] is True and 'cuda_synchronized' not in row.park_ack
            finally:
                owner.loan=None
                await pool.close()
                service.close()
            assert job.confirm_termination().schema_version==2
    try:asyncio.run(run())
    finally:banks.close()
