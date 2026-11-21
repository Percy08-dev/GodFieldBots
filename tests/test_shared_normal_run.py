"""Unqualified Model3 Source/update loop with the real shared publication owner."""
import asyncio
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
@pytest.mark.parametrize('fault',[None,'uncertain_close','reaped_close'])
def test_normal_model3_two_updates_publish_and_close_local_service(tmp_path,monkeypatch,fault):
    from gf_cli.config import LoadedConfig, SharedRunConfigV2
    from gf_common.card_instance import (
        ActionConfigV3,
        reviewed_specs,
        shared_card_instance_contract,
    )
    from gf_common.domain import digest
    from gf_common.preprocessing import PreparationConfig
    from gf_common.shared_inference import shared_runtime
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    from gf_training.card_instance_ppo import _GameOperations, train_card_instance_ppo
    from gf_training.card_instance_source import validate_game_parent
    from gf_training.collection import FinalizedCollection
    from gf_training.structured_numeric import GameLearner
    from tests.card_instance_learning_support import original_collection
    from tests.structured_ppo_support import configuration
    from tests.test_card_instance_baseline_runtime import receipt
    from tests.test_card_instance_learning import RULES, learner_for
    base=configuration(tmp_path)
    tiny=learner_for()
    binding=replace(tiny.binding,device='cuda:0')
    actions=ActionConfigV3(action_contract=shared_card_instance_contract(),max_candidates=64)
    model=ConfiguredNeuralModelV3(**{key:getattr(tiny.model,key) for key in tiny.model.__dataclass_fields__},inference=binding)
    run=SharedRunConfigV2(**asdict(base.config.run),inference_service=shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=base.config.run.concurrency))
    config=replace(base.config,run=run,actions=actions,model=model,training=replace(base.config.training,device='cuda:0'),
        server=replace(base.config.server,simulator_revision=actions.action_contract.simulator_revision),
        preprocessing=PreparationConfig(reader='participant-structured-v2',memory_budget_mib=1024))
    loaded=LoadedConfig(config,asdict(config),digest(asdict(config)),())
    monkeypatch.setattr(_GameOperations,'preflight',lambda self,config,deadline:None)
    def integration(self,config,deadline):
        from gf_models.arrays import sha256
        return {'binary_sha256':receipt()['binary']['sha256'],'server_config_sha256':sha256(Path(config.server.config)),
            'declared_simulator_revision':actions.action_contract.simulator_revision,
            'source_revision':actions.action_contract.simulator_revision,'source_spec_sha256':reviewed_specs(actions.action_contract),'source_dirty':False}
    monkeypatch.setattr(_GameOperations,'integration_identity',integration)
    services=[]
    children=[]
    original_close=[]
    async def collector(plan,ledger):
        child=ledger.learner._cleanup_owner
        from gf_models.sharded_arrays import read_json
        state=read_json(child.directory/'run.json')
        generation=plan.linkage['collection_generation']
        frozen=GameLearner(tiny.model,17,binding,state['resume_identity'],actions=actions,rules=RULES)
        frozen.restore(child.directory/'artifacts'/f'{generation:06}')
        original_collection(plan,state,frozen)
        services.append(ledger.service.owner)
        children.append(child)
        if fault is not None and generation==1:
            service=services[-1]
            original_close.append(service.close)
            def broken_close():
                if fault=='reaped_close':original_close[0]()
                else:service.uncertain=True
                raise RuntimeError('fixture service close failure')
            service.close=broken_close
        return FinalizedCollection(plan.output,{'comparison_valid':True})
    parent=tmp_path/'normal-shared'
    if fault is not None:
        async def failure_case():
            try:
                with pytest.raises(RuntimeError,match='fixture service close failure'):
                    await train_card_instance_ppo(loaded,collector,parent)
                assert children[-1].lease.released
                assert children[-1].process.returncode==0
                assert services[-1].lease.released is (fault=='reaped_close')
                if fault=='reaped_close':assert not services[-1].controller.pins
            finally:
                if original_close:
                    services[-1].close=original_close[0]
                    if fault=='uncertain_close':
                        # A fixture-induced uncertain owner stays quarantined.
                        # Retire the exact known test children over the untouched
                        # original private pipe without manufacturing a lease proof.
                        from gf_simulator_client.framing import read_frame, write_frame
                        from gf_simulator_client.shared_service import _DeadlineReader
                        service=services[-1]
                        deadline=time.monotonic()+20
                        write_frame(service.process.stdin,{'command':'ABORT_CLOSE','deadline':deadline})
                        retired=read_frame(_DeadlineReader(service.process.stdout,deadline))
                        service.process.wait(timeout=20)
                        assert retired['gpu_exitcode']==0 and retired['relays_joined'] is True
                        assert service.process.returncode==0
                        service.channel.close()
                        for left,_ in service._pairs:left.close()
                        service.process.stdin.close();service.process.stdout.close()
                    else:
                        services[-1].controller.close()
                for child in children:await child.close(time.monotonic()+20)
            assert services[-1].lease.released is (fault=='reaped_close')
        asyncio.run(failure_case())
        return
    report=asyncio.run(train_card_instance_ppo(loaded,collector,parent))
    assert report['updates']==2
    assert len(services)==2 and services[0] is services[1] and services[0].closed
    assert services[0].lease.released
    state,records=validate_game_parent(parent,max_owned_bytes=1024**3,deadline=time.monotonic()+30)
    assert state['status']=='complete' and len(records)==3
