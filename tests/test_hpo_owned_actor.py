"""Actor9 receives inference backings only through original resident transport."""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict

import pytest

from tests.test_hpo_gpu_lease import admit
from tests.test_hpo_numeric import learner_for
from tests.test_hpo_owned_numeric import budget
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414


@pytest.mark.parametrize('kind', ['mlp','gru','transformer'])
def test_actual_owned_actor_begin_end_uses_fd_after_checkpoint_removed(global_root,tmp_path,kind):
    from gf_common.domain import ContractError, RuntimePolicyBinding, canonical
    from gf_models.structured_binding import input_binding_game_v3, safe_options_game_v4
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_resident_episode import initial
    learner=learner_for(kind)
    snapshot=learner.snapshot_owned(budget=budget())
    snapshot.save(tmp_path/'saved')
    checkpoint=snapshot.save_ack.files['inference.json']
    options=safe_options_game_v4({'version':'4','checkpoint':str(tmp_path/'absent'),
        'expected_manifest_digest':checkpoint,'model':asdict(learner.model),'inference':asdict(learner.binding),
        'cuda_visible_devices':'','action_config_hash':learner.action_config_hash,
        **input_binding_game_v3(ACTIONS.action_contract,RULES)},actions=ACTIONS,rules=RULES)
    binding=RuntimePolicyBinding(kind,options)
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=time.monotonic()+30,owned_snapshots=True)
            pool.bind_owned_snapshot(snapshot.actor_capability)
            pool.admit(CollectionPlan((),{'model':binding},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            owner=ResidentPhysicalWorker(pool,1)
            pool.owners.append(owner)
            try:
                await owner.bootstrap(timeout=10)
                custody.begin_admitted_lease(pool.admission,owner.lease,binding=binding)
                request=custody.episode_request(owner.lease,'BEGIN')
                response=await owner.exchange({'request':request,'payload':initial(kind,options=options)},timeout=10)
                ready=custody.acknowledge_episode(owner.lease,response)
                assert ready['kind']=='ready' and ready['protocol_version']==3
                with pytest.raises(ContractError,match='cached'):
                    pool.unbind_owned_snapshot(snapshot.actor_capability)
                assert not owner.owned_channel.outstanding
                import os
                resume_inodes={os.fstat(bank.fileno()).st_ino for bank in snapshot.resume.banks}
                child_inodes=set()
                for fd in __import__('pathlib').Path(f'/proc/{owner.process.pid}/fd').iterdir():
                    try: child_inodes.add(fd.stat().st_ino)
                    except FileNotFoundError: pass
                assert resume_inodes.isdisjoint(child_inodes)
                # FD transfers completed; the permanent snapshot still owns both
                # isolated backings for the current policy/opponent pool.
                assert all(bank.fileno()>=0 for bank in (*snapshot.inference.banks,*snapshot.resume.banks))
                from gf_common.structured import event_delta
                from gf_models.hpo_owned_inference import load_owned_inference
                from gf_models.structured_policy import GameStructuredPolicyV4
                from gf_simulator_client.card_instance_episode import CardInstanceEpisode
                from tests.model3_support import raw_inputs
                from tests.test_resident_episode import message
                network=load_owned_inference(snapshot.inference,snapshot.actor_metadata,
                    model=learner.model,binding=learner.binding,actions=ACTIONS,rules=RULES)
                policy=GameStructuredPolicyV4(network,learner.model,learner.binding,checkpoint,
                    action_contract=ACTIONS.action_contract,rules=RULES,action_config_hash=learner.action_config_hash)
                direct=CardInstanceEpisode(initial(kind,options=options),policy_loader=lambda *args:policy)
                _,_,frames,bank,_=raw_inputs(count=3)
                for frame in frames:
                    delta=event_delta(1,frame.episode,frame.interval_start,frame.interval_start_hash,
                        bank.view().events(frame.interval_start,frame.boundary.end),layout=frame.layout)
                    payload=message(frame,delta)
                    expected=direct.select(payload)
                    request=custody.episode_request(owner.lease,'SELECT')
                    response=await owner.exchange({'request':request,'payload':payload},timeout=10)
                    actual=custody.acknowledge_episode(owner.lease,response)
                    assert expected['selection']==actual['selection']
                    assert canonical(expected['generated']['table'])==canonical(actual['generated']['table'])
                    for key in expected['generated']['report']:
                        if key in ('count_ms','search_ms','selection_ms'):
                            import math
                            assert math.isfinite(actual['generated']['report'][key]) and actual['generated']['report'][key]>=0
                            continue
                        assert canonical(expected['generated']['report'][key])==canonical(actual['generated']['report'][key]),key
                direct.reset()
                request=custody.request(owner.lease,'END',checkpoint_sha256=checkpoint)
                response=await owner.exchange({'request':request},timeout=10)
                custody.acknowledge(owner.lease,response)
                assert response['private_state_empty']
                request=custody.request(owner.lease,'PARK',checkpoint_sha256=checkpoint)
                response=await owner.exchange({'request':request},timeout=10)
                custody.acknowledge(owner.lease,response)
            finally:
                await pool.close()
                if checkpoint in pool._owned_snapshots:
                    pool.unbind_owned_snapshot(snapshot.actor_capability)
            assert owner.process.poll()==0 and owner.lease.released
            job.confirm_termination()
    asyncio.run(run())
    snapshot.close()


@pytest.mark.parametrize('field',['generation','model','snapshot'])
def test_owned_actor_external_model_metadata_guard(tmp_path,field):
    from gf_common.domain import ContractError, canonical
    from gf_models.hpo_owned_inference import load_owned_inference
    from tests.test_card_instance_learning import ACTIONS, RULES
    learner=learner_for()
    snapshot=learner.snapshot_owned(budget=budget())
    snapshot.save(tmp_path/'saved')
    metadata=snapshot.actor_metadata
    encoded=canonical(metadata)
    assert all(private not in encoded for private in ('adam.','rng.','optimizer_steps','shuffle_rng','resume_identity'))
    if field=='generation': metadata['generation']=True
    elif field=='model': metadata['inference']['model']['model_id']='gru'
    else: metadata['snapshot_identity']='not-a-snapshot'
    with pytest.raises(ContractError):
        load_owned_inference(snapshot.inference,metadata,model=learner.model,binding=learner.binding,actions=ACTIONS,rules=RULES)
    snapshot.close()


def test_common_actor_capability_immutable_exact_origin_and_binding_owner(tmp_path,monkeypatch):
    import ast
    import builtins
    import copy
    import inspect
    from types import SimpleNamespace

    from gf_common.domain import ContractError
    from gf_common.hpo_actor_snapshot import OwnedActorSnapshot
    from gf_simulator_client import resident_pool
    learner=learner_for()
    account=budget()
    snapshot=learner.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'saved')
    capability=snapshot.actor_capability
    assert type(capability) is OwnedActorSnapshot
    assert capability is snapshot.actor_capability
    assert all(not name.startswith(('adam.','rng.')) for name in capability.inference.names)
    with pytest.raises((AttributeError,TypeError)): capability.generation=99
    with pytest.raises(ContractError): copy.copy(capability).require()
    changed=capability.actor_metadata
    changed['generation']=99
    assert capability.actor_metadata['generation']==0
    tree=ast.parse(inspect.getsource(resident_pool))
    assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith('gf_training') for node in ast.walk(tree))
    pool=resident_pool.ResidentParticipantPool(SimpleNamespace(ledger=SimpleNamespace(global_cuda_bytes=1024)),
        resident_bytes=0,deadline=float('inf'),owned_snapshots=True)
    real_import=builtins.__import__
    def no_training(name,*args,**kw):
        if name.startswith('gf_training'): raise AssertionError('client imported training')
        return real_import(name,*args,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'__import__',no_training)
        with pytest.raises(ContractError): pool.bind_owned_snapshot(snapshot)
        with pytest.raises(ContractError): pool.bind_owned_snapshot({})
        pool.bind_owned_snapshot(capability)
        assert pool.owned_snapshot(capability.checkpoint) is capability
        lease=pool._owned_snapshot_bindings[capability.checkpoint]
        with pytest.raises(ContractError): lease.close(object())
        with pytest.raises(BufferError): snapshot.close()
        pool.unbind_owned_snapshot(capability)
    snapshot.close()
    with pytest.raises(ContractError): capability.require()
    assert account.used==0


def test_actor_binding_keeps_actual_permanent_snapshot_until_unbound(tmp_path):
    import copy
    import gc
    import weakref
    learner=learner_for()
    account=budget()
    snapshot=learner.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'saved')
    capability=snapshot.actor_capability
    holder=object()
    binding=capability.bind(holder)
    from gf_common.domain import ContractError
    with pytest.raises(ContractError): copy.copy(binding).close(holder)
    ref=weakref.ref(snapshot)
    learner._owned_snapshot=None
    del snapshot
    gc.collect()
    assert ref() is not None and capability.require() is ref()
    actual=ref()
    binding.close(holder)
    actual.close()
    assert account.used==0
