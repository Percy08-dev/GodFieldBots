"""SAME audited ownership, real process lifecycle and conservative parked memory."""
import pickle

import pytest

from gf_common.domain import ContractError
from tests.test_hpo_gpu_lease import admit, spawn_worker


@pytest.fixture
def global_root(tmp_path, monkeypatch):
    from gf_training import hpo_gpu_lease as gpu
    root = tmp_path/"global-admission"
    monkeypatch.setattr(gpu,"_GLOBAL_ROOT",root)
    return root


def ack(custody, worker, command, checkpoint='c'*64):
    request = custody.request(worker, command, checkpoint_sha256=checkpoint)
    response = {**request, 'clean_reset':True, 'private_state_empty':True,
        'pending_io_empty':True, 'cuda_synchronized':True, 'reset_version':'resident-reset-v1'}
    custody.acknowledge(worker, response)
    return response


def setup_pool(job):
    from gf_training.hpo_resident import ResidentPoolCustody
    owner, lease, process = spawn_worker(job, script='import time;time.sleep(30)', residency=16)
    pool = ResidentPoolCustody(job, run_id='run-a', pool_id='b'*32)
    pool.register(lease, owner, viewer=1)
    pool.begin_lease(lease, generation=0)
    ack(pool, lease, 'END')
    ack(pool, lease, 'PARK')
    return pool, owner, lease, process


def cleanup(lease, process):
    process.terminate()
    process.wait(timeout=5)
    lease.reaped(process)


def test_only_original_parked_authority_allows_update_with_charged_workers(global_root):
    with admit() as job:
        pool, _owner, lease, process = setup_pool(job)
        try:
            authority = pool.parked_authority(generation=0, collection_id='batch-000000')
            assert authority.evidence['owners'][0]['pid'] == process.pid
            assert authority.evidence['owners'][0]['end_ack']['command'] == 'END'
            assert authority.evidence['owners'][0]['park_ack']['command'] == 'PARK'
            with pytest.raises(ContractError):
                job.ledger.begin_update(10)
            with pytest.raises(ContractError):
                job.ledger.begin_resident_update(10, authority=authority.evidence)
            with pytest.raises(TypeError):
                pickle.dumps(authority)
            job.ledger.begin_resident_update(10, authority=authority)
            assert job.ledger.cuda_bytes == 26
            assert job.ledger.workers[lease.key] is lease
            with pytest.raises(ContractError):
                pool.begin_lease(lease, generation=0)
            job.ledger.finish_update()
            assert job.ledger.cuda_bytes == 16
            pool.begin_lease(lease, generation=1)
            with pytest.raises(ContractError):
                authority.require(job.ledger)
        finally:
            cleanup(lease, process)
        job.confirm_termination()


def test_missing_wrong_or_pending_ack_cannot_retire_and_lease_viewer_is_fixed(global_root):
    from gf_training.hpo_resident import ResidentPoolCustody
    with admit() as job:
        owner, lease, process = spawn_worker(job, script='import time;time.sleep(30)')
        pool = ResidentPoolCustody(job, run_id='run-a', pool_id='b'*32)
        pool.register(lease, owner, viewer=1)
        try:
            with pytest.raises(ContractError):
                pool.register(lease, owner, viewer=2)
            pool.begin_lease(lease, generation=0)
            request = pool.request(lease, 'END', checkpoint_sha256='c'*64)
            with pytest.raises(ContractError):
                pool.parked_authority(generation=0, collection_id='batch-000000')
            response = {**request, 'viewer':2, 'clean_reset':True,'private_state_empty':True,
                'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'}
            with pytest.raises(ContractError):
                pool.acknowledge(lease, response)
            with pytest.raises(ContractError):
                pool.parked_authority(generation=0, collection_id='batch-000000')
        finally:
            cleanup(lease, process)
        # Failed work remains ineligible; actual original cleanup still terminates resources.
        assert pool._failed
        job.confirm_termination()


def test_sync_and_owner_set_change_invalidate_authority_and_stage_memory(global_root):
    with admit() as job:
        pool, _owner, lease, process = setup_pool(job)
        try:
            authority = pool.parked_authority(generation=0, collection_id='batch-000000')
            with job.ledger.reserve_resident_transition(100, authority=authority,
                    target_generation=1,target_checkpoint_sha256='d'*64,affected_workers=(lease,)):
                assert job.ledger.cuda_bytes == 116
                with pytest.raises(ContractError):
                    job.ledger.begin_resident_update(1000, authority=authority)
                request = pool.request(lease,'SYNC',checkpoint_sha256='d'*64,generation=1)
                with pytest.raises(ContractError):
                    authority.require(job.ledger)
                pool.acknowledge(lease,{**request,'clean_reset':True,'private_state_empty':True,
                    'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'})
                ack(pool,lease,'PARK',checkpoint='d'*64)
            assert job.ledger.cuda_bytes == 16
        finally:
            cleanup(lease, process)
        job.confirm_termination()


@pytest.mark.parametrize('release_mode', ['evict','reap','failed-reap'])
def test_cache_release_requires_original_eviction_or_exact_audited_reap(global_root, tmp_path, release_mode):
    import time

    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_retention import pin_hpo_cache, release_hpo_cache
    from tests.hpo_lineage_support import initial_hpo_chain
    root = tmp_path/'parent'
    state, records, _ = initial_hpo_chain(root, resident=True)
    with gpu.admit_gpu_job(gpu.JobContext('campaign-a','operation','training',state['run_id'],None), global_cuda_bytes=1024,max_workers=2) as job:
        from gf_training.hpo_resident import ResidentPoolCustody
        owner, worker, process = spawn_worker(job, script='import time;time.sleep(30)')
        pool = ResidentPoolCustody(job, run_id=state['run_id'],pool_id='b'*32)
        pool.register(worker, owner, viewer=1)
        identity = pool.begin_lease(worker,generation=0)
        ack(pool,worker,'END')
        ack(pool,worker,'PARK')
        options = {'max_owned_bytes':1024**3,'deadline':time.monotonic()+30}
        pin = pin_hpo_cache(root,'cache-1',0,records[0]['inference_sha256'],owner=identity,**options)
        pool.observe_cache_pin(pin)
        with pytest.raises(ContractError):
            pool.cache_release(pin.binding)
        try:
            if release_mode == 'evict':
                request = pool.request(worker,'EVICT',checkpoint_sha256=pin.binding['inference_sha256'],cache_identity=pin.binding['cache_identity'])
                with pytest.raises(ContractError):
                    pool.cache_release(pin.binding)
                pool.acknowledge(worker,{**request,'clean_reset':True,'private_state_empty':True,
                    'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'})
                authority = pool.cache_release(pin.binding)
            else:
                if release_mode == 'failed-reap':
                    request = pool.request(worker,'SYNC',checkpoint_sha256='d'*64)
                    with pytest.raises(ContractError):
                        pool.acknowledge(worker,{**request,'viewer':2,'clean_reset':True,'private_state_empty':True,
                            'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'})
                    with pytest.raises(ContractError):
                        job.ledger.reserve_worker(1)
                with pytest.raises(ContractError):
                    pool.cache_release_after_reap(pin.binding)
                cleanup(worker,process)
                authority = pool.cache_release_after_reap(pin.binding)
            released = release_hpo_cache(root,pin.binding,authority=authority,**options)
            assert released.phase == 'release'
            from gf_models.sharded_arrays import read_json
            release_intent = read_json(root/'cache-reference-receipts/cache-1/release.intent.json')
            assert release_intent['release_evidence']['kind'] == ('hpo-resident-cache-eviction' if release_mode == 'evict' else 'hpo-resident-cache-reaped')
            assert released == release_hpo_cache(root,pin.binding,authority=authority,**options)
        finally:
            if process.poll() is None:
                cleanup(worker,process)
        job.confirm_termination()


def test_uncertain_transition_retains_charge_until_actual_owner_reap(global_root):
    with admit() as job:
        pool, _owner, worker, process = setup_pool(job)
        authority = pool.parked_authority(generation=0,collection_id='batch-000000')
        transition = job.ledger.reserve_resident_transition(100,authority=authority,
            target_generation=1,target_checkpoint_sha256='d'*64,affected_workers=(worker,))
        try:
            with pytest.raises(RuntimeError), transition:
                pool.request(worker,'SYNC',checkpoint_sha256='d'*64,generation=1)
                raise RuntimeError('lost original SYNC ACK')
            assert job.ledger.cuda_bytes == 116
            with pytest.raises(ContractError):
                transition.retire_failed_transition()
            cleanup(worker,process)
            transition.retire_failed_transition()
            assert job.ledger.cuda_bytes == 0
        finally:
            if process.poll() is None:
                cleanup(worker,process)
        job.confirm_termination()



def test_cancelled_async_wrapper_does_not_retire_original_executor_io(global_root):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    async def scenario():
        with admit() as job:
            pool, _owner, worker, process = setup_pool(job)
            gate = threading.Event()
            started = threading.Event()
            def blocking_io():
                started.set()
                gate.wait(timeout=10)
            with ThreadPoolExecutor(max_workers=1) as executor:
                completion = executor.submit(blocking_io)
                async def join_io():
                    await asyncio.wrap_future(completion)
                task = asyncio.create_task(join_io())
                pool.track_io(worker,task,completion)
                try:
                    while not started.is_set():
                        await asyncio.sleep(0)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.done() and completion.running()
                    with pytest.raises(ContractError):
                        pool.retire_io(worker,task)
                    with pytest.raises(ContractError):
                        pool.parked_authority(generation=0,collection_id='batch-000000')
                finally:
                    gate.set()
                    completion.result(timeout=5)
                    pool.retire_io(worker,task)
                    cleanup(worker,process)
            job.confirm_termination()
    asyncio.run(scenario())



def test_transition_syncs_only_bound_affected_owners_and_keeps_other_bindings(global_root):
    from gf_training.hpo_resident import ResidentPoolCustody
    with admit() as job:
        owner1, worker1, process1 = spawn_worker(job,script='import time;time.sleep(30)')
        owner2, worker2, process2 = spawn_worker(job,script='import time;time.sleep(30)')
        pool = ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
        try:
            for viewer,owner,worker in ((1,owner1,worker1),(2,owner2,worker2)):
                pool.register(worker,owner,viewer=viewer)
                pool.begin_lease(worker,generation=0)
                ack(pool,worker,'END')
                ack(pool,worker,'PARK')
            authority = pool.parked_authority(generation=0,collection_id='batch-000000')
            frozen = authority.evidence['owners'][1]['park_ack']
            with job.ledger.reserve_resident_transition(100,authority=authority,target_generation=1,
                    target_checkpoint_sha256='d'*64,affected_workers=(worker1,)):
                request = pool.request(worker1,'SYNC',checkpoint_sha256='d'*64,generation=1)
                pool.acknowledge(worker1,{**request,'clean_reset':True,'private_state_empty':True,
                    'pending_io_empty':True,'cuda_synchronized':True,'reset_version':'resident-reset-v1'})
                ack(pool,worker1,'PARK',checkpoint='d'*64)
            assert pool._owners[worker2.key].park_ack == frozen
            assert job.ledger.cuda_bytes == 32
        finally:
            cleanup(worker1,process1)
            cleanup(worker2,process2)
        job.confirm_termination()


def test_custody_cannot_cross_actual_training_job_parent(global_root):
    from gf_training.hpo_resident import ResidentPoolCustody
    with admit() as job:
        with pytest.raises(ContractError):
            ResidentPoolCustody(job,run_id='foreign-parent',pool_id='b'*32)
        assert job.ledger._resident_pool is None
        job.confirm_termination()


def test_training_custody_cannot_stand_in_for_unbound_evaluation_panel(global_root):
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_resident import ResidentPoolCustody
    with gpu.admit_gpu_job(gpu.JobContext('campaign','operation','evaluation','run-a','evaluation-a'),global_cuda_bytes=1024,max_workers=2) as job:
        with pytest.raises(ContractError,match='evaluation panel'):
            ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
        assert job.ledger._resident_pool is None
        job.confirm_termination()


@pytest.mark.parametrize('cancelled', [False, True])
@pytest.mark.parametrize('phase', ['parked', 'active'])
def test_prior_original_io_blocks_lease_reuse_and_decision(global_root, cancelled, phase):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    async def scenario():
        with admit() as job:
            pool, _owner, worker, process = setup_pool(job)
            if phase == 'active':
                pool.begin_lease(worker,generation=1)
            epoch = pool._owners[worker.key].lease_epoch
            gate, started = threading.Event(), threading.Event()
            def blocking_io():
                started.set()
                gate.wait(timeout=10)
            with ThreadPoolExecutor(max_workers=1) as executor:
                completion = executor.submit(blocking_io)
                async def join_io():
                    await asyncio.wrap_future(completion)
                task = asyncio.create_task(join_io())
                pool.track_io(worker,task,completion)
                try:
                    while not started.is_set():
                        await asyncio.sleep(0)
                    if cancelled:
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    assert completion.running()
                    if phase == 'parked':
                        with pytest.raises(ContractError):
                            pool.begin_lease(worker,generation=1)
                        assert pool._owners[worker.key].lease_epoch == epoch
                    else:
                        with pytest.raises(ContractError):
                            worker.reserve_decision(1)
                        assert not worker.decisions
                    gate.set()
                    completion.result(timeout=5)
                    if not cancelled:
                        await asyncio.wait_for(task,5)
                    if phase == 'parked':
                        with pytest.raises(ContractError):
                            pool.begin_lease(worker,generation=1)
                    else:
                        with pytest.raises(ContractError):
                            worker.reserve_decision(1)
                    pool.retire_io(worker,task)
                    if phase == 'parked':
                        pool.begin_lease(worker,generation=1)
                    decision = worker.reserve_decision(1)
                    worker.reply(decision)
                finally:
                    gate.set()
                    completion.result(timeout=5)
                    if not cancelled:
                        await asyncio.wait_for(task,5)
                    if task in pool._owners[worker.key].io:
                        pool.retire_io(worker,task)
                    cleanup(worker,process)
            job.confirm_termination()
    asyncio.run(scenario())


@pytest.mark.parametrize('cancelled', [False, True])
def test_actual_reap_does_not_terminate_job_before_original_io_retirement(global_root, cancelled):
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    async def scenario():
        with admit() as job:
            pool, _owner, worker, process = setup_pool(job)
            gate, started = threading.Event(), threading.Event()
            def blocking_io():
                started.set()
                gate.wait(timeout=10)
            with ThreadPoolExecutor(max_workers=1) as executor:
                completion = executor.submit(blocking_io)
                async def join_io():
                    await asyncio.wrap_future(completion)
                task = asyncio.create_task(join_io())
                pool.track_io(worker,task,completion)
                try:
                    while not started.is_set():
                        await asyncio.sleep(0)
                    if cancelled:
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    pool.fail()
                    cleanup(worker,process)
                    assert worker.released and completion.running()
                    with pytest.raises(ContractError):
                        job.confirm_termination()
                    assert job._receipt is None
                finally:
                    gate.set()
                    completion.result(timeout=5)
                    if not cancelled:
                        await asyncio.wait_for(task,5)
                    if process.poll() is None:
                        cleanup(worker,process)
                    # The RED implementation seals after prematurely minting a receipt.
                    # Always stop actual I/O; only a still-open job can retire its token.
                    if job._receipt is None:
                        with pytest.raises(ContractError):
                            job.confirm_termination()
                        pool.retire_io(worker,task)
            receipt = job.confirm_termination()
            assert receipt.resources[0].pid == process.pid
    asyncio.run(scenario())


@pytest.mark.parametrize('crash_stage', ['acquire_intent','acquire_registry','release_intent','release_registry'])
def test_cache_transactions_borrow_original_coordinator_lock_and_retry(global_root, tmp_path, monkeypatch, crash_stage):
    import time
    from dataclasses import replace

    from gf_models.sharded_arrays import read_json
    from gf_training import hpo_gpu_lease as gpu
    from gf_training import hpo_retention as retention
    from gf_training.hpo_resident import ResidentPoolCustody
    from gf_training.lineage import run_lock
    from tests.hpo_lineage_support import initial_hpo_chain
    root = tmp_path/'parent'
    state, records, _learner = initial_hpo_chain(root,resident=True)
    with gpu.admit_gpu_job(gpu.JobContext('campaign','operation','training',state['run_id'],None),global_cuda_bytes=1024,max_workers=2) as job:
        owner, worker, process = spawn_worker(job,script='import time;time.sleep(30)')
        pool = ResidentPoolCustody(job,run_id=state['run_id'],pool_id='b'*32)
        pool.register(worker,owner,viewer=1)
        identity = pool.begin_lease(worker,generation=0)
        ack(pool,worker,'END')
        ack(pool,worker,'PARK')
        original = retention._write_control
        def interrupted(path,*args,**kwargs):
            result = original(path,*args,**kwargs)
            if (crash_stage.endswith('intent') and path.name == crash_stage.split('_')[0]+'.intent.json'
                    or crash_stage.endswith('registry') and path.name == 'artifact-registry.json'):
                raise OSError('injected cache transaction interruption')
            return result
        try:
            with run_lock(root) as ownership:
                options = {'max_owned_bytes':1024**3,'deadline':time.monotonic()+30,'ownership':ownership}
                pin_args = (root,'cache-1',0,records[0]['inference_sha256'])
                with pytest.raises(ContractError):
                    retention.pin_hpo_cache(*pin_args,owner=identity,**{**options,'ownership':replace(ownership)})
                if crash_stage.startswith('acquire'):
                    with monkeypatch.context() as patch:
                        patch.setattr(retention,'_write_control',interrupted)
                        with pytest.raises(OSError):
                            retention.pin_hpo_cache(*pin_args,owner=identity,**options)
                pin = retention.pin_hpo_cache(*pin_args,owner=identity,**options)
                assert retention.pin_hpo_cache(*pin_args,owner=identity,**options) == pin
                pool.observe_cache_pin(pin)
                cleanup(worker,process)
                authority = pool.cache_release_after_reap(pin.binding)
                with pytest.raises(ContractError):
                    retention.release_hpo_cache(root,pin.binding,authority=authority,
                        **{**options,'ownership':replace(ownership)})
                if crash_stage.startswith('release'):
                    with monkeypatch.context() as patch:
                        patch.setattr(retention,'_write_control',interrupted)
                        with pytest.raises(OSError):
                            retention.release_hpo_cache(root,pin.binding,authority=authority,**options)
                release = retention.release_hpo_cache(root,pin.binding,authority=authority,**options)
                assert release.phase == 'release'
                assert retention.release_hpo_cache(root,pin.binding,authority=authority,**options) == release
                assert read_json(root/'artifact-registry.json')['cache_references'] == {}
        finally:
            if process.poll() is None:
                cleanup(worker,process)
        job.confirm_termination()
