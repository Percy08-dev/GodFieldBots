"""Original managed creation-task authority, with bounded CPU fixtures only."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from gf_common.domain import ContractError
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.hpo_owned_process import HpoOwnedLearningProcess
from tests.test_card_instance_learning import ACTIONS
from tests.test_hpo_numeric import learner_for
from tests.test_hpo_owned_numeric import budget


def child_for(directory, *, ledger=None):
    learner=learner_for()
    account=budget()
    child=HpoOwnedLearningProcess(directory,1024**2,learner.model,learner.training,learner.binding,
        learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger or ReservationLedger(),
        cuda_visible_devices='',cleanup_deadline=time.monotonic()+15,budget=account)
    return child,account


def admit(child):
    child.lease=child.ledger.reserve_learner(child.resident_bytes,child)
    child.lease.creation_pending=True
    return child.lease


def close_fixture_endpoints(child):
    # Negative tests never issue a managed no-child/reap proof. Their CPU-only
    # reservation remains quarantined; this closes only the known test sockets.
    child.channel.close()
    child._inherited.close()


def forbid_effects(monkeypatch,child):
    reached=[]
    def forbidden(*args,**kwargs):
        reached.append(True)
        raise AssertionError('startup side effect before exact task admission')
    monkeypatch.setattr(child._inherited,'inherited_endpoint',forbidden)
    monkeypatch.setattr(child,'_environment',forbidden)
    return reached


def test_direct_async_creator_is_rejected_before_endpoint_export(tmp_path,monkeypatch):
    async def run():
        before=len(os.listdir('/proc/self/fd'))
        child,account=child_for(tmp_path)
        effects=forbid_effects(monkeypatch,child)
        try:
            with pytest.raises(ContractError): await child._create_owned_process()
            assert effects==[] and not child._creation_no_child
            await child.close(time.monotonic()+5)
            assert len(os.listdir('/proc/self/fd'))==before and account.used==0
        finally: close_fixture_endpoints(child)
    asyncio.run(run())


def test_unadmitted_factory_does_not_issue_a_coroutine(tmp_path):
    async def run():
        child,_account=child_for(tmp_path)
        coroutine=None
        try:
            with pytest.raises(ContractError): coroutine=child._create_process()
            assert coroutine is None and not child._creation_no_child
            await child.close(time.monotonic()+5)
        finally:
            if coroutine is not None: coroutine.close()
            close_fixture_endpoints(child)
    asyncio.run(run())


def test_admitted_request_in_foreign_task_cannot_spawn_or_settle(tmp_path,monkeypatch):
    async def run():
        child,_account=child_for(tmp_path)
        lease=admit(child)
        effects=forbid_effects(monkeypatch,child)
        coroutine=child._create_process()
        try:
            task=asyncio.create_task(coroutine)  # Deliberately NOT child.creation_task.
            with pytest.raises(ContractError): await task
            with pytest.raises(ContractError): await child.close(time.monotonic()+1)
            assert effects==[] and not child._creation_no_child
            assert child.creation_task is None and lease.creation_pending and not lease.released
        finally:
            coroutine.close()
            close_fixture_endpoints(child)
    asyncio.run(run())


def test_duplicate_factory_and_unsettled_close_refuse_without_clearing_pending(tmp_path,monkeypatch):
    async def run():
        child,_account=child_for(tmp_path)
        lease=admit(child)
        effects=forbid_effects(monkeypatch,child)
        coroutine=child._create_process()
        duplicate=None
        try:
            with pytest.raises(ContractError): duplicate=child._create_process()
            with pytest.raises(ContractError): await child.close(time.monotonic()+1)
            coroutine.close()  # Closing an arbitrary request is not the original task-failure observation.
            with pytest.raises(ContractError): await child.close(time.monotonic()+1)
            assert effects==[] and lease.creation_pending and child.ledger.learner is lease
            assert not child._creation_no_child and not lease.released
        finally:
            coroutine.close()
            if duplicate is not None: duplicate.close()
            close_fixture_endpoints(child)
    asyncio.run(run())


@pytest.mark.parametrize('eager',[False,True])
def test_original_managed_task_spawns_exactly_one_real_cpu_child_and_reaps(tmp_path,monkeypatch,eager):
    real=asyncio.create_subprocess_exec
    calls=[]
    async def counted(*args,**kwargs):
        calls.append(args)
        return await real(*args,**kwargs)
    monkeypatch.setattr(asyncio,'create_subprocess_exec',counted)
    async def run():
        before=len(os.listdir('/proc/self/fd'))
        loop=asyncio.get_running_loop()
        old=loop.get_task_factory()
        if eager: loop.set_task_factory(asyncio.eager_task_factory)
        child,account=child_for(tmp_path)
        snapshot=None
        try:
            await child.start()
            snapshot=await child.initialize_owned(17,tmp_path/'g0',deadline=time.monotonic()+10)
            with pytest.raises(ContractError): child._create_process()
            with pytest.raises(ContractError): await child._create_owned_process()
            assert len(calls)==1
        finally:
            await child.close(time.monotonic()+5)
            if snapshot is not None: snapshot.close()
            loop.set_task_factory(old)
        assert child.process.returncode==0 and child.lease.released and account.used==0
        assert len(os.listdir('/proc/self/fd'))==before
    asyncio.run(run())


@pytest.mark.parametrize('failure',['task_creation','stdlib'])
def test_original_creation_failures_keep_original_no_child_observations(tmp_path,monkeypatch,failure):
    from gf_training import hpo_gpu_lease as gpu
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'global')
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture','operation','training','startup-proof',None),
                global_cuda_bytes=12*1024**3,max_workers=4) as job:
            child,account=child_for(tmp_path,ledger=job.ledger)
            before=len(os.listdir('/proc/self/fd'))
            real_task=asyncio.create_task
            if failure=='task_creation':
                def failed_task(coroutine,*args,**kwargs):
                    request=child._owned_factory_request
                    if coroutine is getattr(request,'coroutine',None):
                        raise RuntimeError('actual original task creation failure')
                    return real_task(coroutine,*args,**kwargs)
                monkeypatch.setattr(asyncio,'create_task',failed_task)
                expected=RuntimeError
            else:
                monkeypatch.setattr('gf_training.hpo_owned_process.sys.executable','/no-such-owned-startup-executable')
                expected=FileNotFoundError
            try:
                with pytest.raises(expected): await child.start()
                assert child._creation_no_child and child.process is None
                assert child.lease.released and child.ledger.learner is None
                assert len(os.listdir('/proc/self/fd'))==before-2 and account.used==0
                job.confirm_termination()
            finally:
                close_fixture_endpoints(child)
    asyncio.run(run())


def test_environment_rejection_does_not_mint_no_child_settlement(tmp_path,monkeypatch):
    async def run():
        child,_account=child_for(tmp_path)
        def rejected(): raise RuntimeError('controlled environment rejection')
        monkeypatch.setattr(child,'_environment',rejected)
        try:
            with pytest.raises(RuntimeError,match='environment'): await child.start()
            assert not child._creation_no_child
            assert child.lease.creation_pending and not child.lease.released
            with pytest.raises(ContractError): await child.close(time.monotonic()+1)
            assert child.lease.creation_pending and child.ledger.learner is child.lease
        finally: close_fixture_endpoints(child)
    asyncio.run(run())


def test_cancelled_original_task_does_not_mint_no_child_settlement(tmp_path,monkeypatch):
    async def run():
        child,_account=child_for(tmp_path)
        effects=forbid_effects(monkeypatch,child)
        loop=asyncio.get_running_loop()
        old=loop.get_task_factory()
        def eager_cancel(loop,coroutine,**kwargs):
            task=asyncio.eager_task_factory(loop,coroutine,**kwargs)
            if coroutine is getattr(child._owned_factory_request,'coroutine',None): task.cancel()
            return task
        loop.set_task_factory(eager_cancel)
        try:
            with pytest.raises(asyncio.CancelledError): await child.start()
            assert effects==[] and not child._creation_no_child
            assert child.lease.creation_pending and not child.lease.released
            with pytest.raises(ContractError): await child.close(time.monotonic()+1)
        finally:
            loop.set_task_factory(old)
            close_fixture_endpoints(child)
    asyncio.run(run())


def test_parent_start_cancellation_preserves_original_creation_and_reap(tmp_path,monkeypatch):
    real=asyncio.create_subprocess_exec
    async def run():
        before=len(os.listdir('/proc/self/fd'))
        child,account=child_for(tmp_path)
        entered,release=asyncio.Event(),asyncio.Event()
        calls=[]
        async def paused(*args,**kwargs):
            calls.append(args)
            entered.set()
            await release.wait()
            return await real(*args,**kwargs)
        monkeypatch.setattr(asyncio,'create_subprocess_exec',paused)
        task=asyncio.create_task(child.start())
        await asyncio.wait_for(entered.wait(),5)
        task.cancel()
        release.set()
        try:
            with pytest.raises(asyncio.CancelledError): await task
            assert len(calls)==1 and child.process.returncode is not None
            assert child.lease.released and child.ledger.learner is None
            assert child.current_snapshot is None and account.used==0
            assert len(os.listdir('/proc/self/fd'))==before
        finally:
            if child.process is not None and child.process.returncode is None:
                await child.close(time.monotonic()+3)
            close_fixture_endpoints(child)
    asyncio.run(run())
