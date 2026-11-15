"""Unqualified CPU originals plus actual wire6/parked owners, never a game run."""
import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, digest
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('failed_post_commit_observation',[False,True])
def test_actual_source_update_save_commit_pin_sync_order_with_parked_workers(global_root,tmp_path,monkeypatch,packed,failed_post_commit_observation):
    from gf_models.arrays import sha256
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_binding import configured_model_v4
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.card_instance_source import (
        _OnlineGameOwner,
        game_batch_source,
        validate_owned_collection,
    )
    from gf_training.hpo_learning_process import HpoLearningProcess
    from gf_training.hpo_lineage import commit_hpo_generation, validate_hpo_chain
    from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.lineage import checkpoint_path, run_lock
    from tests.hpo_support import original_hpo_collection, variable_initial_hpo_chain
    from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES

    # This one CPU-only fixture bypasses solely the production CUDA/data gate.
    # Actual locks, wire6, originals, live capabilities, pins and reaps are real.
    def require_cpu_fixture(state):
        assert state['device']=='cpu' and state['run_id']=='2de8b1c1-6db1-5cc9-a2e7-933ce3d10001'
    monkeypatch.setattr('gf_training.hpo_source.require_hpo_production',require_cpu_fixture)
    root=tmp_path/'parent'
    state,records,learner=variable_initial_hpo_chain(root,games=8,decision_budget=12,resident=True,packed=packed)
    from gf_models.arrays import atomic_json
    state['run_id']=records[0]['run_id']='2de8b1c1-6db1-5cc9-a2e7-933ce3d10001'
    atomic_json(root/'generations/000000.json',records[0])
    state['generation_sha256']=sha256(root/'generations/000000.json')
    atomic_json(root/'run.json',state)
    atomic_json(root/'latest.json',{'generation':0,'sha256':state['generation_sha256']})
    spec=state['effective_config']['hpo_spec']
    configured=configured_model_v4(state['effective_config']['model'],layout=LAYOUT)
    config=SimpleNamespace(model=configured,actions=ACTIONS,hpo_spec=spec)
    async def run(ownership):
        deadline=time.monotonic()+90
        with gpu.admit_gpu_job(gpu.JobContext('campaign','operation','training',state['run_id'],None),
                global_cuda_bytes=spec['operating']['managed_cuda_bytes'],max_workers=4) as job:
            child=HpoLearningProcess(root,1024**2,learner.model,learner.training,learner.binding,
                digest(asdict(ACTIONS)),state['resume_identity'],actions=ACTIONS,ledger=job.ledger,
                cuda_visible_devices='',cleanup_deadline=deadline,storage_binding=state.get('storage_binding'),phase_limits={key:spec['operating'][key]
                    for key in ('numeric_budget_bytes','save_budget_bytes')})
            runtime=ResidentTrainingRuntime(job,ownership,config,state,deadline=deadline)
            online=_OnlineGameOwner(ownership,state,child,job.ledger,hpo_job=job,resident_runtime=runtime)
            child.online_owner=online
            await child.start()
            owner=None
            try:
                await child.initialize(17,root/'unused',restore=checkpoint_path(root,0),deadline=deadline)
                committed_records=list(records)
                previous_authority=None
                for generation in range(2):
                    plan,label,after=recorded_hpo_plan(root,state,committed_records,generation,deadline)
                    if generation:
                        assert all(row.sync_ack['generation']==generation and row.end_ack['generation']==generation-1
                            for row in runtime.custody._owners.values())
                        assert previous_authority is not None
                        with pytest.raises(ContractError): previous_authority.require(job.ledger)
                    runtime.pool.admit(plan)
                    validate_owned_collection(plan,job.ledger,max_owned_bytes=1024**3,deadline=deadline,
                        resident_phase=runtime.pool.admission)
                    with (pytest.raises(ContractError),
                            game_batch_source(plan,job.ledger,max_owned_bytes=1024**3,deadline=deadline)):
                        pass
                    owner=RetainedGameArtifacts.for_hpo_registry(plan.bindings,ACTIONS,rules=RULES,pool_size=2,
                        max_owned_bytes=512*1024**2,preflight_deadline=deadline,finalization_deadline=deadline)
                    for viewer in (1,2):
                        binding=plan.bindings[label]
                        if generation and viewer==2:
                            binding=next(value for value in plan.bindings.values() if value.policy_options is not None
                                and value.policy_options.get('checkpoint')==str(checkpoint_path(root,0)))
                        logical=runtime.pool.worker(viewer,binding.model_id,17,ACTIONS,ledger=job.ledger,
                            policy_options=binding.policy_options,artifact_inputs=owner)
                        await logical.initialize()
                        if viewer==1:
                            from gf_common.structured import event_delta
                            from tests.model3_support import raw_inputs
                            _,_,frames,bank,_=raw_inputs(count=1)
                            frame=frames[0]
                            delta=event_delta(1,frame.episode,0,bank.view()._prefixes[0],bank.view().events(),layout=frame.layout)
                            await logical.select_frame(frame,delta)
                        logical.mark_completed()
                        await logical.close()
                    authority=runtime.pool.retire_collection()
                    assert all(row['end_ack']['generation']==generation for row in authority.evidence['owners'])
                    if generation:
                        frozen=next(row for row in authority.evidence['owners'] if row['viewer']==2)
                        assert frozen['end_ack']['checkpoint_sha256']==committed_records[0]['inference_sha256']
                    original_hpo_collection(plan,state,learner,logical_retirement=authority.evidence)
                    if packed:
                        from gf_common.preprocessing import preparation_config
                        from gf_training.group_pack_game_source import (
                            PackedBatchCustody,
                            PackedSourceOwner,
                        )
                        from tests.test_group_pack_game_source import pack_original
                        destination=Path(state['effective_config']['storage']['db_root'])/state['run_id']/'collections'/plan.linkage['batch_id']
                        pack_original(plan.output,destination)
                        original=PackedSourceOwner(plan.output,preparation_config(state['effective_config']['preprocessing']),deadline,
                            storage_binding=state['storage_binding'],max_owned_bytes=1024**3,expected_descriptor_root=destination)
                        online.packed_batch=PackedBatchCustody(original)
                    with game_batch_source(plan,job.ledger,max_owned_bytes=1024**3,deadline=deadline) as source:
                        numeric=root/'numeric'/plan.linkage['batch_id']
                        manifest=source.write_numeric(numeric)
                        files={'rollout.json':sha256(numeric/'rollout.json'),**manifest['data']['files']}
                        handle=source.issue(numeric,files)
                        assert handle.resident_authority is authority
                        assert len(job.ledger.workers)==2
                        acknowledgement=await child.update(handle,checkpoint_path(root,generation+1),deadline=deadline)
                        assert len(job.ledger.workers)==2 and not job.ledger.updating
                        selection=source.selection
                        state.update(accepted_decisions=state['accepted_decisions']+selection['used_decisions'],used_decisions=state['used_decisions']+selection['used_decisions'],
                            collected_decisions=state['collected_decisions']+selection['collected_decisions'],optimizer_steps=child.steps)
                        update={'schema_version':5,'status':'complete','run_id':state['run_id'],'generation':generation+1,
                            'parent_sha256':state['generation_sha256'],'batch_id':plan.linkage['batch_id'],
                            'collection':'collections/'+plan.linkage['batch_id'],'collection_manifest_sha256':source.proof['manifest_sha256'],
                            'numeric':'numeric/'+plan.linkage['batch_id'],'numeric_sha256':files['rollout.json'],
                            'accepted_decisions':selection['used_decisions'],'source_proof':source.proof,
                            'metrics':acknowledgement['metrics'],'planned_games':8,'actual_games':8,'health':{},
                            'policy_bindings':{key:asdict(value) for key,value in plan.bindings.items()},
                            'learner_seats':plan.linkage['learner_seats'],'collection_seconds':0.,'update_seconds':0.,
                            'model_id':state['model_id'],'model_version':'4','input_recipe':state['input_recipe'],
                            'numeric_files':files,'hpo_spec_sha256':state['hpo_spec_sha256'],
                            'collected_decisions':selection['collected_decisions'],'used_decisions':selection['used_decisions'],
                            'overshoot_decisions':selection['collected_decisions']-selection['used_decisions'],
                            'expected_optimizer_step_delta':source.proof['expected_optimizer_step_delta'],
                            **{key:state[key] for key in ('action_contract','purpose','synthetic_capacity','rule_signature','layout_signature')}}
                        if packed: update['storage_binding']=state['storage_binding']
                        record=commit_hpo_generation(root,state,generation=generation+1,update=update,
                            controls={'schedule_opponent_rng':after,'snapshot_pool':[generation,generation+1],'collection_position':generation+1},
                            artifact_files=acknowledgement['artifact']['files'],max_owned_bytes=1024**3,
                            deadline=deadline,projection=handle,ownership=ownership)
                        assert handle.prepared
                    if online.packed_batch is not None:
                        online.packed_batch.close()
                        online.packed_batch=None
                    # Source and original registry snapshots retire before new pins.
                    if failed_post_commit_observation and generation==0:
                        def failed_observe(receipt): raise OSError('new committed pin observation failed')
                        with monkeypatch.context() as patch:
                            patch.setattr(runtime.custody,'observe_cache_pin',failed_observe)
                            with pytest.raises(OSError,match='observation failed'):
                                await runtime.after_commit(generation+1,record)
                        runtime.custody.fail()
                        break
                    await runtime.after_commit(generation+1,record)
                    previous_authority=authority
                    assert all(row.generation==generation+1 and row.sync_ack['generation']==generation+1 and row.end_ack['generation']==generation for row in runtime.custody._owners.values())
                    assert len(job.ledger.workers)==2
                    for entries in runtime.entries.values():
                        assert {pin.binding['generation'] for _,pin in entries.values()}==set(range(generation+2))
                    _,committed_records=validate_hpo_chain(root,max_owned_bytes=1024**3,deadline=deadline)
                    learner.restore(checkpoint_path(root,generation+1))
                    owner.release()
                    owner=None
            finally:
                await runtime.close()
                await child.close(deadline)
                if owner is not None: owner.release()
            from gf_models.sharded_arrays import read_json
            assert read_json(root/'artifact-registry.json')['cache_references']=={}
            job.confirm_termination()
    with run_lock(root) as ownership:
        asyncio.run(run(ownership))


@pytest.mark.parametrize('fault',['observe','acquire_intent','acquire_registry'])
def test_original_pin_acquisition_attempt_survives_observation_and_transaction_failure(global_root,tmp_path,monkeypatch,fault):
    from gf_models.sharded_arrays import read_json
    from gf_models.structured_binding import configured_model_v4
    from gf_training import hpo_gpu_lease as gpu
    from gf_training import hpo_retention as retention
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
    from gf_training.hpo_source import hpo_binding
    from gf_training.lineage import run_lock
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.test_card_instance_learning import ACTIONS, LAYOUT
    root=tmp_path/'parent'
    state,records,_=initial_hpo_chain(root,resident=True)
    spec=state['effective_config']['hpo_spec']
    config=SimpleNamespace(model=configured_model_v4(state['effective_config']['model'],layout=LAYOUT),actions=ACTIONS,hpo_spec=spec)
    label,binding=hpo_binding(root,state,records[0],0)
    async def run(ownership):
        deadline=time.monotonic()+30
        with gpu.admit_gpu_job(gpu.JobContext('campaign','operation','training',state['run_id'],None),global_cuda_bytes=12*1024**3,max_workers=4) as job:
            runtime=ResidentTrainingRuntime(job,ownership,config,state,deadline=deadline)
            runtime.pool.admit(CollectionPlan((),{label:binding},tmp_path/'collection',
                {'parent_run_id':state['run_id'],'collection_generation':0,'batch_id':'batch-000000'},deadline,deadline+1))
            # A real owner and bootstrap; no model needs to load for this fault.
            from gf_simulator_client.resident_pool import ResidentPhysicalWorker
            owner=ResidentPhysicalWorker(runtime.pool,1)
            runtime.pool.owners.append(owner)
            await owner.bootstrap(timeout=10)
            runtime.custody.begin_admitted_lease(runtime.pool.admission,owner.lease,binding=binding)
            original=retention._write_control
            def interrupted(path,*args,**kwargs):
                value=original(path,*args,**kwargs)
                if (fault=='acquire_intent' and path.name=='acquire.intent.json'
                        or fault=='acquire_registry' and path.name=='artifact-registry.json'):
                    raise OSError('actual acquire interrupted')
                return value
            def observe_failed(*args): raise OSError('original observation failed')
            with monkeypatch.context() as patch:
                if fault=='observe': patch.setattr(runtime.custody,'observe_cache_pin',observe_failed)
                else: patch.setattr(retention,'_write_control',interrupted)
                with pytest.raises(OSError): runtime._pin(owner,binding)
            runtime.custody.fail()
            await runtime.close()
            assert read_json(root/'artifact-registry.json')['cache_references']=={}
            journals=list((root/'cache-reference-receipts').iterdir())
            assert len(journals)==1 and (journals[0]/'release.receipt.json').is_file()
            runtime.require_closed()
            job.confirm_termination()
    with run_lock(root) as ownership:
        asyncio.run(run(ownership))
