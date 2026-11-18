"""Actual exec participant keeps search/sampling local and never initializes CUDA."""
import asyncio
import time
from dataclasses import asdict, replace

import pytest

torch=pytest.importorskip('torch')


def test_real_cpu_participant_uses_private_broker_scores_without_weights(tmp_path):
    from gf_common.card_instance import ActionConfigV3, shared_card_instance_contract
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_common.shared_inference import ServingBudget, shared_runtime
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_artifacts import save_game_inference_v3
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from gf_simulator_client.shared_service import SharedInferenceService
    from gf_simulator_client.shared_worker import SharedParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.model3_support import LAYOUT, model3
    from tests.test_card_instance_baseline_runtime import game
    actions=ActionConfigV3(action_contract=shared_card_instance_contract(),max_candidates=64)
    model=model3()
    binding=InferenceBindingV2(max_candidates=64,candidate_chunk=2)
    network=make_network(model,17,binding,layout=LAYOUT)
    artifact=tmp_path/'model'
    save_game_inference_v3(artifact,network,model,digest(asdict(actions)),
        action_contract=actions.action_contract,rules=card_instance_rules())
    configured=ConfiguredNeuralModelV3(**{key:getattr(model,key) for key in model.__dataclass_fields__},checkpoint=str(artifact),inference=replace(binding,device='cuda:0'))
    owner=RetainedGameArtifacts((configured,),actions,rules=card_instance_rules(),max_owned_bytes=512*1024**2,
        preflight_deadline=time.monotonic()+30,finalization_deadline=time.monotonic()+60)
    _,policy=owner.binding(configured)
    budget=ServingBudget(128*1024**2)
    banks=ReadonlyRamBankSet.build({name:value.detach().numpy() for name,value in network.state_dict().items()},budget)
    ledger=ReservationLedger(1024**3,2)
    service=SharedInferenceService(ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
        execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu')
    service.publish(banks,model=asdict(model),binding=asdict(configured.inference),
        checkpoint=policy.policy_options['expected_manifest_digest'],generation=0,identity='b'*64)
    async def run():
        worker=SharedParticipantWorker('mlp',17,actions,ledger=ledger,policy_options=policy.policy_options,
            artifact_inputs=owner,service=service,game='real-actor-game',viewer=1)
        try:
            await worker.initialize()
            frame,_bank,delta,_=game()
            generated,selection=await worker.select_frame(frame,delta)
            selection.validate(generated.table,frame)
            assert worker.last_timings['shared_request_roundtrip_ms']>0
            assert worker.lease.resident_bytes==0
        finally:await worker.close()
    try:asyncio.run(run())
    finally:
        service.close();banks.close();owner.release()
