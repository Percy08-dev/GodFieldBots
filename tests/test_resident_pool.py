"""Production resident transport with actual audited subprocess and executor owners."""
import asyncio

import pytest

from tests.test_hpo_gpu_lease import admit
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414 - fixture export
from tests.test_resident_episode import ACTIONS


def test_actual_bootstrap_multi_episode_reuses_physical_owner_with_positive_io_retirement(global_root, tmp_path):
    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b'*32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, deadline=1e12)
            plan = CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id':'run-a', 'collection_generation':0, 'batch_id':'batch-000000'},1e12,1e12)
            pool.admit(plan)
            try:
                owners = []
                for seed in (17, 29, 17):
                    logical = pool.worker(1, 'random', seed, ACTIONS, ledger=job.ledger)
                    await logical.initialize()
                    owners.append(logical.owner)
                    assert custody._owners[logical.lease.key].bootstrap is not None
                    assert not custody._owners[logical.lease.key].io
                    logical.mark_completed()
                    await logical.close()
                    assert not logical.lease.released and logical.process.poll() is None
                    assert not custody._owners[logical.lease.key].io
                assert owners[0] is owners[1] is owners[2]
                authority = pool.retire_collection()
                authority.require(job.ledger)
                assert len(job.ledger.workers) == job.ledger.serial == 1
            finally:
                await pool.close()
            assert not job.ledger.workers
            job.confirm_termination()
    asyncio.run(run())


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_fresh_resident_all_model4_fixed_input_selection_logprob_value_A_B_A(global_root, tmp_path, kind):
    import time
    from dataclasses import asdict

    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.structured import event_delta
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_binding import (
        input_binding_game_v3,
        policy_label_game_v4,
        safe_options_game_v4,
    )
    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.model3_support import raw_inputs
    from tests.test_hpo_numeric import learner_for
    from tests.test_resident_episode import RULES

    learner = learner_for(kind)
    bindings = []
    registry = {}
    original = None
    for name in ('A', 'B'):
        path = tmp_path / name
        receipt = learner.save(path)
        options = safe_options_game_v4({'version':'4','checkpoint':str(path),
            'expected_manifest_digest':receipt['inference_digest'], 'model':asdict(learner.model),
            'inference':asdict(learner.binding), 'cuda_visible_devices':'',
            'action_config_hash':learner.action_config_hash, **input_binding_game_v3(ACTIONS.action_contract, RULES)},
            actions=ACTIONS, rules=RULES)
        bindings.append(RuntimePolicyBinding(kind, options))
        label = policy_label_game_v4(learner.model, learner.binding, receipt["inference_digest"],
            action_contract=ACTIONS.action_contract, rules=RULES, action_config_hash=learner.action_config_hash,
            cuda_visible_devices="")
        registry[label] = bindings[-1]
        if original is None:
            original = {key: value.clone() for key,value in learner.network.state_dict().items()}
            with learner.torch.no_grad():
                next(learner.network.parameters()).add_(.125)
    from gf_models.arrays import sha256
    immutable_files={path:sha256(path) for name in ('A','B') for path in (tmp_path/name).iterdir() if path.is_file()}
    for path in immutable_files:
        path.chmod(0o444)
    owner = RetainedGameArtifacts.for_hpo_registry(registry, ACTIONS,
        rules=RULES, pool_size=2, max_owned_bytes=512*1024**2,
        preflight_deadline=time.monotonic()+30, finalization_deadline=time.monotonic()+60)
    numerical_records=[]
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('campaign-a','operation','training','run-a',None),
                global_cuda_bytes=12*1024**3,max_workers=2) as job:
            pool = ResidentParticipantPool(ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32),
                resident_bytes=512*1024**2,deadline=time.monotonic()+60)
            plan = CollectionPlan((),registry,tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12)
            pool.admit(plan)
            answers = []
            try:
                for binding in (bindings[0], bindings[1], bindings[0]):
                    logical = pool.worker(1,kind,17,ACTIONS,ledger=job.ledger,
                        policy_options=binding.policy_options,artifact_inputs=owner)
                    fresh = CardInstanceParticipantWorker(kind,17,ACTIONS,ledger=ReservationLedger(),
                        policy_options=binding.policy_options,artifact_inputs=owner)
                    _, _, frames, bank, _ = raw_inputs(count=3)
                    selected = []
                    fresh_tables,pooled_tables,fresh_selections,pooled_selections=[],[],[],[]
                    try:
                        for frame in frames:
                            delta = event_delta(frame.actor,frame.episode,frame.interval_start,bank.view()._prefixes[frame.interval_start],
                                bank.view().events(frame.interval_start,frame.boundary.end),layout=frame.layout)
                            a, b = await fresh.select_frame(frame,delta), await logical.select_frame(frame,delta)
                            assert a[1] == b[1]
                            assert a[0].table == b[0].table
                            selected.append(asdict(b[1]))
                            fresh_tables.append(a[0].table)
                            pooled_tables.append(b[0].table)
                            fresh_selections.append(a[1])
                            pooled_selections.append(b[1])
                        if not numerical_records:
                            from gf_training.structured_rollout import game_episode_records
                            for tables,selections in ((fresh_tables,fresh_selections),(pooled_tables,pooled_selections)):
                                numerical_records.append(game_episode_records(bank.view(),tuple(frames),tuple(tables),tuple(selections),
                                    'natural','draw',actions=ACTIONS,rules=RULES))
                        logical.mark_completed()
                    finally:
                        await fresh.close()
                        await logical.close()
                    answers.append(selected)
                assert answers[0] == answers[2]
                assert len(pool.owners) == 1
                pool.retire_collection().require(job.ledger)
            finally:
                await pool.close()
            job.confirm_termination()
        return answers
    try:
        answers = asyncio.run(run())
        assert digest(answers[0]) == digest(answers[2])
        assert all(sha256(path)==expected and path.stat().st_mode&0o222==0 for path,expected in immutable_files.items())
        import numpy as np

        from gf_models.sharded_arrays import read_bundle, read_json
        from gf_training.hpo_rollout import HpoRollout
        from tests.test_structured_learning import assert_same_learners
        left,right=learner_for(kind),learner_for(kind)
        for numeric,record in zip((left,right),numerical_records,strict=True):
            numeric.update(HpoRollout((record,),numeric.model,numeric.training,actions=ACTIONS,rules=RULES,
                selected_locations=((0,0),(0,1),(0,2))))
        assert_same_learners(left,right)
        destinations=(tmp_path/'fresh-updated',tmp_path/'resident-updated')
        left.save(destinations[0]);right.save(destinations[1])
        for filename,field in (('inference.json','weights'),('resume.json','state')):
            bundles=[read_bundle(path,read_json(path/filename)[field]) for path in destinations]
            assert bundles[0].keys()==bundles[1].keys()
            for key,value in bundles[0].items(): np.testing.assert_array_equal(value,bundles[1][key])
    finally:
        owner.release()


@pytest.mark.parametrize('resident_parent', [False, True])
def test_evaluation_pool_uses_genuine_outer_refs_for_registry4_and5(global_root, tmp_path, resident_parent):
    import time
    from dataclasses import asdict

    from gf_cli.hpo_evaluation import ResidentHpoEvaluationPlan
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.schedule import serialize_schedule
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.sharded_arrays import read_json
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_resident import EvaluationPoolScope
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from gf_training.hpo_source import hpo_binding
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.test_hpo_evaluation_role import binding as link_fixture
    from tests.test_resident_episode import RULES

    root = tmp_path/'parent'
    state, records, _ = initial_hpo_chain(root,resident=resident_parent)
    label, binding = hpo_binding(root,state,records[0],0)
    registry = {label:binding,'heuristic-card-instance-v2':RuntimePolicyBinding('heuristic')}
    link, _, _ = link_fixture()
    parent = {'run_id':state['run_id'],'run_path':str(root),'training_seed':17,'generation':0,
        'generation_sha256':state['generation_sha256'],'inference_sha256':records[0]['inference_sha256'],
        'source_receipt_sha256':'c'*64,'policy_label':label}
    link.update(left_parent=parent,policy_registry_digest=digest({key:asdict(value) for key,value in registry.items()}))
    monkey_context = gpu.JobContext('campaign','evaluation','evaluation',state['run_id'],'evaluation')
    with gpu.admit_gpu_job(monkey_context,global_cuda_bytes=12*1024**3,max_workers=4) as job:
        link['gpu_ticket'] = job.ticket
        groups = hpo_evaluation_groups(link)
        schedule = serialize_schedule(groups,4)
        link.update(schedule_digest=digest(schedule),learner_seats={spec['episode_id']:spec['learner_viewer']
            for group in schedule for spec in group['episodes']})
        pin = pin_hpo_evaluation(root,job.ticket,0,records[0]['inference_sha256'],max_owned_bytes=1024**3,deadline=time.monotonic()+30)
        start = time.monotonic()
        plan = ResidentHpoEvaluationPlan.create_resident(groups,registry,tmp_path/'evaluation',link,
            runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728},pins=((root,pin),),
            started_monotonic=start,admission_deadline=start+10,execution_deadline=start+20,
            work_deadline=start+25,wall_deadline=start+30)
        owner = RetainedGameArtifacts.for_hpo_registry(registry,ACTIONS,rules=RULES,pool_size=2,max_owned_bytes=512*1024**2,
            preflight_deadline=start+20,finalization_deadline=start+25)
        try:
            with pytest.raises(Exception,match='parent-set'):
                EvaluationPoolScope(job,plan,owner,(),max_owned_bytes=1024**3,deadline=start+20)
            scope = EvaluationPoolScope(job,plan,owner,((root,pin),),max_owned_bytes=1024**3,deadline=start+20)
            from types import SimpleNamespace

            from gf_training.hpo_resident_runtime import ResidentEvaluationRuntime
            config = SimpleNamespace(actions=ACTIONS,run=SimpleNamespace(concurrency=2),hpo_spec=state['effective_config']['hpo_spec'])
            runtime = ResidentEvaluationRuntime(job,plan,owner,config)
            assert runtime.custody.evaluation_scope.plan is scope.plan
            async def use_pool():
                logical = runtime.pool.worker(1,'heuristic',17,ACTIONS,ledger=job.ledger)
                await logical.initialize()
                logical.mark_completed()
                await logical.close()
                runtime.pool.retire_collection().require(job.ledger)
                assert job.ticket in read_json(root/'artifact-registry.json')['evaluation_references']
                await runtime.close()
            asyncio.run(use_pool())
            pool = runtime.custody
            registry_value = read_json(root/'artifact-registry.json')
            assert registry_value['schema_version'] == (5 if resident_parent else 4)
            assert registry_value.get('cache_references',{}) == {}
            assert job.ticket in registry_value['evaluation_references']
            pool.require_resource_retirement(job,job.ledger)
        finally:
            owner.release()
            release_hpo_evaluation(root,pin.binding,max_owned_bytes=1024**3,deadline=start+25)
        job.confirm_termination()


@pytest.mark.parametrize('fault', ['epoch', 'lost-sync-ack'])
def test_actual_lost_ack_retains_staging_until_all_original_owners_reaped(global_root,tmp_path,fault):
    from gf_common.domain import ContractError, RuntimePolicyBinding, digest
    from gf_simulator_client.card_instance_episode import episode_initial
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_resident_episode import RULES
    async def run():
        with admit() as job:
            custody = ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool = ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12)
            binding = RuntimePolicyBinding('random')
            pool.admit(CollectionPlan((),{'random':binding},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            for viewer in (1,2):
                logical=pool.worker(viewer,'random',17,ACTIONS,ledger=job.ledger)
                await logical.initialize()
                logical.mark_completed()
                await logical.close()
            authority=pool.retire_collection()
            owner=pool.owners[0]
            checkpoint=digest({'model_id':'random','policy_options':None})
            transition=job.ledger.reserve_resident_transition(100,authority=authority,target_generation=1,
                target_checkpoint_sha256=checkpoint,affected_workers=(owner.lease,))
            original=owner._read_reply
            def corrupt(source):
                reply=original(source)
                if fault=='epoch': reply['lease_epoch']+=1
                else: raise EOFError('lost original SYNC ACK after child reset/copy')
                return reply
            owner._read_reply=corrupt
            try:
                with pytest.raises((ContractError,EOFError)), transition:
                    await owner.control('SYNC',checkpoint=checkpoint,generation=1,
                        payload=episode_initial(binding,0,ACTIONS,RULES),timeout=10.)
                assert job.ledger.cuda_bytes >= 116
                with pytest.raises(ContractError): transition.retire_failed_transition()
                with pytest.raises(ContractError): job.confirm_termination()
            finally:
                await pool.close()
                transition.retire_failed_transition()
            assert not job.ledger.workers and job.ledger.cuda_bytes==0
            job.confirm_termination()
    asyncio.run(run())


def test_actual_cancellation_keeps_original_executor_io_until_positive_retirement(global_root,tmp_path):
    import threading

    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
            owner=logical.owner
            gate,entered=threading.Event(),threading.Event()
            original=owner._read_reply
            def hold(source):
                reply=original(source)
                entered.set()
                gate.wait(timeout=5)
                return reply
            owner._read_reply=hold
            request=asyncio.create_task(logical.initialize())
            try:
                while not entered.is_set(): await asyncio.sleep(.001)
                request.cancel()
                await asyncio.sleep(.01)
                row=custody._owners[owner.lease.key]
                assert row.io and any(not future.done() for future in row.io.values())
                with pytest.raises(ContractError): job.confirm_termination()
                gate.set()
                with pytest.raises(asyncio.CancelledError): await request
                await pool.close()
                assert not row.io and not owner._operations
                job.confirm_termination()
            finally:
                gate.set()
                if not request.done():
                    await asyncio.gather(request,return_exceptions=True)
                await pool.close()
    asyncio.run(run())


@pytest.mark.parametrize('fault',['constructor','registration','reset'])
def test_actual_constructor_registration_and_reset_failures_keep_owned_cleanup(global_root,tmp_path,monkeypatch,fault):
    import subprocess

    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            if fault=='constructor':
                def fail(*args,**kwargs): raise OSError('actual Popen constructor fault')
                monkeypatch.setattr(subprocess,'Popen',fail)
            elif fault=='registration':
                def fail(*args,**kwargs): raise ContractError('original registration fault')
                monkeypatch.setattr(custody,'register',fail)
            try:
                if fault in ('constructor','registration'):
                    with pytest.raises((OSError,ContractError),match='fault'):
                        pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
                else:
                    logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
                    await logical.initialize()
                    original=logical.owner._read_reply
                    def bad_reset(source):
                        reply=original(source)
                        reply['private_state_empty']=False
                        return reply
                    logical.owner._read_reply=bad_reset
                    logical.mark_completed()
                    with pytest.raises(ContractError): await logical.close()
                with pytest.raises(ContractError): pool.retire_collection()
            finally:
                await pool.close()
            assert not job.ledger.workers
            job.confirm_termination()
    asyncio.run(run())


@pytest.mark.parametrize('fault',['timeout','cleanup'])
def test_actual_timeout_and_failed_cleanup_have_no_work_authority_but_keep_reap_authority(global_root,tmp_path,monkeypatch,fault):
    import threading

    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
            await logical.initialize()
            logical.mark_completed()
            owner=logical.owner
            if fault=='timeout':
                gate=threading.Event()
                original=owner._read_reply
                def hold(source):
                    reply=original(source)
                    gate.wait(timeout=5)
                    return reply
                monkeypatch.setattr(owner,'_read_reply',hold)
                timer=threading.Timer(.1,gate.set)
                timer.start()
                try:
                    with pytest.raises(TimeoutError): await logical.close(timeout=.02)
                finally:
                    gate.set()
                    timer.join(timeout=1)
                await pool.close()
            else:
                await logical.close()
                authority=pool.retire_collection()
                original=owner._close_owned
                async def fail_after_actual_reap(timeout,started):
                    await original(timeout,started)
                    raise OSError('reported close failure after original audited reap')
                monkeypatch.setattr(owner,'_close_owned',fail_after_actual_reap)
                with pytest.raises(OSError,match='audited reap'): await pool.close()
                with pytest.raises(ContractError): authority.require(job.ledger)
            assert custody._failed and not job.ledger.workers
            assert not owner._operations and not custody._owners[owner.lease.key].io
            with pytest.raises(ContractError): pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
            job.confirm_termination()
    asyncio.run(run())


@pytest.mark.parametrize('fault',['task-before','task-after','submit-before','submit-after','eager','cancel-setup','pending-setup-reap','executor-closed'])
def test_gated_setup_owns_original_task_future_before_any_framing(global_root,tmp_path,monkeypatch,fault):
    import threading
    import time

    import gf_simulator_client.resident_pool as transport
    from gf_common.domain import RuntimePolicyBinding
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=transport.ResidentParticipantPool(custody,resident_bytes=16,deadline=time.monotonic()+10)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
            owner=logical.owner
            writes,violations=[],[]
            entered=threading.Event()
            original_write=transport.write_frame
            def observed_write(stream,payload):
                row=custody._owners[owner.lease.key]
                writes.append(payload)
                if not row.io or not getattr(row,'setups',set()): violations.append('framing began without original owners')
                entered.set()
                return original_write(stream,payload)
            monkeypatch.setattr(transport,'write_frame',observed_write)
            if fault=='executor-closed':
                owner.executor.shutdown(wait=True)
            original_submit=owner.executor.submit
            def submitted(*args,**kwargs):
                if fault=='submit-before': raise RuntimeError('actual submit boundary fault before return')
                future=original_submit(*args,**kwargs)
                # Old transport deterministically writes before capturing a Task.
                entered.wait(timeout=.03)
                if fault=='submit-after': raise RuntimeError('submit fault after real executor submission')
                return future
            monkeypatch.setattr(owner.executor,'submit',submitted)
            loop=asyncio.get_running_loop()
            previous=loop.get_task_factory()
            def factory(loop,coroutine,**options):
                if coroutine.cr_code.co_name.endswith('wait_original'):
                    if fault=='task-before': raise RuntimeError('original Task factory fault')
                    task=asyncio.Task(coroutine,loop=loop,**options)
                    if fault=='task-after': raise RuntimeError('Task factory transferred original coroutine then failed')
                    if fault=='pending-setup-reap':
                        from gf_common.domain import ContractError
                        owner.process.terminate()
                        owner.process.wait(timeout=2)
                        owner.lease.reaped(owner.process)
                        with pytest.raises(ContractError,match='setup'):
                            job.confirm_termination()
                        raise RuntimeError('factory fault with original setup pending after actual reap')
                    if fault=='cancel-setup': asyncio.current_task(loop=loop).cancel()
                    return task
                return asyncio.Task(coroutine,loop=loop,**options)
            loop.set_task_factory(asyncio.eager_task_factory if fault=='eager' else factory)
            try:
                if fault=='eager':
                    await logical.initialize()
                    logical.mark_completed()
                    await logical.close()
                elif fault=='cancel-setup':
                    with pytest.raises(asyncio.CancelledError): await logical.initialize()
                else:
                    with pytest.raises(RuntimeError,match='fault|factory|shutdown'): await logical.initialize()
                assert not violations
                if fault!='eager': assert not writes
            finally:
                loop.set_task_factory(previous)
                await pool.close()
            assert not owner._operations and not owner._setups
            assert not custody._owners[owner.lease.key].io and not custody._owners[owner.lease.key].setups
            job.confirm_termination()
    asyncio.run(run())


def test_uncertain_setup_executor_shutdown_retains_original_custody_until_positive_join(global_root,tmp_path,monkeypatch):
    import time

    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=time.monotonic()+10)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
            owner=logical.owner
            submit,shutdown=owner.executor.submit,owner.executor.shutdown
            def fail_submit(*args,**kwargs):
                submit(*args,**kwargs)
                raise RuntimeError('actual submitted setup Future return lost')
            def fail_shutdown(*args,**kwargs): raise OSError('original executor shutdown unconfirmed')
            monkeypatch.setattr(owner.executor,'submit',fail_submit)
            monkeypatch.setattr(owner.executor,'shutdown',fail_shutdown)
            with pytest.raises(RuntimeError,match='return lost'): await logical.initialize()
            assert owner.process.poll() is not None and owner.lease.released
            assert owner._setups and custody._owners[owner.lease.key].setups
            with pytest.raises(ContractError,match='setup'): job.confirm_termination()
            monkeypatch.setattr(owner.executor,'shutdown',shutdown)
            # SAME surviving cleanup owner: actually stop/join its original
            # executor, then retire its actual never-authorized setup token.
            await owner.retire_setups(time.monotonic()+2)
            assert not owner._setups and not custody._owners[owner.lease.key].setups
            with pytest.raises(OSError,match='shutdown'): await pool.close()
            job.confirm_termination()
    asyncio.run(run())


def test_actual_diagnostics_follow_validated_ack_and_do_not_enter_authority(global_root,tmp_path):
    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    values=[]
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12,progress=values.append)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            try:
                for seed in (17,29):
                    logical=pool.worker(1,'random',seed,ACTIONS,ledger=job.ledger)
                    await logical.initialize()
                    logical.mark_completed(); await logical.close()
                assert pool.starts==1 and pool.reuses==1
                facts=[value for value in values if 'pool_workers' in value]
                assert facts[-1]['pool_workers'][0]['loads']==1
                assert facts[-1]['pool_workers'][0]['cache_reuses']==1
                assert facts[-1]['pool_workers'][0]['cuda_allocator'] is None
                assert facts[-1]['managed_cuda_reservation_bytes']==job.ledger.cuda_bytes
                authority=pool.retire_collection()
                authority.require(job.ledger)
            finally: await pool.close()
            job.confirm_termination()
    asyncio.run(run())


@pytest.mark.parametrize('corruption',['malformed-diagnostic','oversized-diagnostic','foreign-ready'])
def test_outer_diagnostics_never_weaken_management_ack_or_publish_foreign_facts(global_root,tmp_path,monkeypatch,corruption):
    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    original=ResidentPhysicalWorker._read_reply
    values=[]
    def read(owner,source):
        response=original(owner,source)
        if response.get('kind')=='READY':
            if corruption=='foreign-ready': response['pid']+=1
            else: response['_diagnostic']={'loads':'malformed'} if corruption=='malformed-diagnostic' else {'extra':'x'*4097}
        return response
    monkeypatch.setattr(ResidentPhysicalWorker,'_read_reply',read)
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12,progress=values.append)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            try:
                logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
                if corruption=='foreign-ready':
                    with pytest.raises(ContractError): await logical.initialize()
                    assert values==[]
                else:
                    await logical.initialize()
                    logical.mark_completed(); await logical.close()
                    assert values and all(value['pool_workers'][0]['loads']==1 for value in values)
            finally: await pool.close()
            job.confirm_termination()
    asyncio.run(run())


def test_replacement_diagnostics_show_current_live_pool_without_erasing_audit_history(global_root,tmp_path):
    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_gpu_lease import JobContext, admit_gpu_job
    from gf_training.hpo_resident import ResidentPoolCustody
    values=[]
    async def run():
        with admit_gpu_job(JobContext('campaign-a','operation','training','run-a',None),global_cuda_bytes=1024,max_workers=4) as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12,progress=values.append)
            try:
                for generation,per_viewer in ((0,2),(1,1),(2,2)):
                    pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                        {'parent_run_id':'run-a','collection_generation':generation,'batch_id':f'batch-{generation:06}'},1e12,1e12))
                    workers=[pool.worker(viewer,'random',17,ACTIONS,ledger=job.ledger)
                        for viewer in (1,2) for _ in range(per_viewer)]
                    for logical in workers:
                        await logical.initialize()
                        logical.mark_completed()
                    for logical in workers: await logical.close()
                    await pool.retire_unused()
                    pool.retire_collection().require(job.ledger)
                    pool.observe_diagnostic()
                    live=[owner for owner in pool.owners if not owner.lease.released]
                    assert len(values[-1]['pool_workers'])==len(live)==2*per_viewer
                assert len(pool.owners)==len(custody._owners)==6
                assert len(job.ledger.workers)==4
                assert values[-1]['pool_starts']==6 and values[-1]['pool_live_workers']==4
                assert all(len(value['pool_workers'])<=4 for value in values)
            finally: await pool.close()
            assert len(custody._owners)==6  # original audited retirement history survives
            job.confirm_termination()
    asyncio.run(run())


@pytest.mark.parametrize('command',['BEGIN','SELECT'])
def test_foreign_inner_policy_cannot_publish_matching_outer_diagnostic(global_root,tmp_path,monkeypatch,command):
    from gf_common.domain import ContractError, RuntimePolicyBinding
    from gf_common.structured import event_delta
    from gf_simulator_client.resident_pool import ResidentParticipantPool, ResidentPhysicalWorker
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.model3_support import raw_inputs
    original=ResidentPhysicalWorker._read_reply
    values=[]
    def read(owner,source):
        reply=original(owner,source)
        if reply.get('request',{}).get('command')==command:
            payload=reply['payload']
            if command=='BEGIN': payload['policy_version']='foreign-policy'
            else: payload['selection']['policy_version']='foreign-policy'
            reply['_diagnostic']={'schema_version':1,'loads':999,'cache_reuses':0,
                'cpu_weight_restore_bytes':0,'cuda_weight_restore_bytes':0,'cuda_allocator':None}
        return reply
    monkeypatch.setattr(ResidentPhysicalWorker,'_read_reply',read)
    async def run():
        with admit() as job:
            custody=ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
            pool=ResidentParticipantPool(custody,resident_bytes=16,deadline=1e12,progress=values.append)
            pool.admit(CollectionPlan((),{'random':RuntimePolicyBinding('random')},tmp_path,
                {'parent_run_id':'run-a','collection_generation':0,'batch_id':'batch-000000'},1e12,1e12))
            try:
                logical=pool.worker(1,'random',17,ACTIONS,ledger=job.ledger)
                await logical.owner.bootstrap(timeout=10.)
                values.clear()
                if command=='BEGIN':
                    with pytest.raises(ContractError,match='policy identity'): await logical.initialize()
                else:
                    await logical.initialize()
                    assert values[-1]['pool_workers'][0]['loads']==1
                    values.clear()
                    _,_,frames,bank,_=raw_inputs(count=1)
                    frame=frames[0]
                    delta=event_delta(frame.actor,frame.episode,frame.interval_start,bank.view()._prefixes[frame.interval_start],
                        bank.view().events(frame.interval_start,frame.boundary.end),layout=frame.layout)
                    with pytest.raises(ContractError,match='binding mismatch'): await logical.select_frame(frame,delta)
                assert values==[]
            finally: await pool.close()
            job.confirm_termination()
    asyncio.run(run())
