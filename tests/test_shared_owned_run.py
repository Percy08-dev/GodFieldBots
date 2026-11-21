"""Unqualified fixed-input lifecycle: real CUDA learner/service, capture and commits."""
import asyncio
import os

import pytest


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
def test_shared_owned_two_updates_commit_publish_cpu_drain_and_actual_cleanup(tmp_path,monkeypatch):
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.ppo_run import _run_ppo
    from tests.shared_inference_support import shared_owned_configuration
    from tests.test_hpo_owned_run import owned_fixture_collection
    runtime,run_id,_campaign,_settings=shared_owned_configuration(tmp_path,monkeypatch)
    config=runtime.config
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=2) as job:
            ops=_OwnedHpoOperations(job)
            async def collect(plan,ledger):return await owned_fixture_collection(ops,config,plan,ledger)
            report=await _run_ppo(runtime,collect,tmp_path/'parents'/run_id,_game_ops=ops)
            assert report['updates']==2 and report['used_decisions']==32
            assert ops.receipt.schema_version==2 and len(ops.receipt.services)==1
            assert ops.shared_controller.service.closed
            assert not ops.shared_controller.pins
            if ops.completion is not None:ops.completion.close()
            assert ops.budget.used==0
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
def test_shared_owned_pause_resume_uses_new_service_and_strict_cold_proofs(tmp_path,monkeypatch):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack import StorageSettings
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_restore import OwnedColdImport
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.lineage import run_lock
    from gf_training.ppo_run import _run_ppo
    from tests.shared_inference_support import shared_owned_configuration
    from tests.test_hpo_owned_run import owned_fixture_collection
    runtime,run_id,_campaign,_settings=shared_owned_configuration(tmp_path,monkeypatch)
    config=runtime.config
    directory=tmp_path/'parents'/run_id
    async def invocation(mode,pause):
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=2) as job:
            ops=_OwnedHpoOperations(job)
            async def collect(plan,ledger):return await owned_fixture_collection(ops,config,plan,ledger)
            report=await _run_ppo(runtime,collect,directory,mode=mode,pause_after_update=pause,_game_ops=ops)
            service=ops.shared_controller.service
            assert service.closed and ops.receipt.schema_version==2
            if ops.completion is not None:ops.completion.close()
            assert ops.budget.used==0
            return report,service.service_id
    paused,first=asyncio.run(invocation('fresh',1))
    assert paused['status']=='paused' and paused['updates']==1
    completed,second=asyncio.run(invocation('resume',None))
    assert completed['status']=='complete' and completed['updates']==2 and first!=second
    deadline=time.monotonic()+30
    account=StorageBudget(StorageSettings(),deadline)
    phase=ArtifactPhaseAdmission(account,config.hpo_spec['operating']['parent_budget_bytes'])
    with run_lock(directory) as ownership:
        imported=OwnedColdImport(ownership,budget=account,parent_budget=phase,deadline=deadline,require_complete=True)
        try:
            assert imported.state['run_variant']=='hpo-ppo-owned-shared-v1'
            assert imported.registry['schema_version']==7
            assert all(row['phase']=='release' for row in imported.registry['service_references'].values())
        finally:imported.close()
    assert account.used==0
    # Cold/resume/final entry readers must require every original service even
    # when the editable cache-reference map is empty or partially omitted.
    import json

    from gf_common.domain import ContractError, clone
    from gf_training.journal import atomic_json
    original=json.loads((directory/'artifact-registry.json').read_text())
    assert len(original['service_references'])>=2
    for missing in ('all_refs','one_ref','all_services','one_service','proof'):
        altered=clone(original)
        moved=None
        if missing=='all_refs':altered['service_references'].clear()
        elif missing=='one_ref':altered['service_references'].pop(next(iter(altered['service_references'])))
        else:
            root=directory/'services'
            first_service=next(iter(root.iterdir()))
            source=root if missing=='all_services' else first_service if missing=='one_service' else first_service/'proof.json'
            moved=(source,tmp_path/('held-'+missing))
            source.rename(moved[1])
        atomic_json(directory/'artifact-registry.json',altered)
        try:
            with run_lock(directory) as ownership, pytest.raises((ContractError,OSError)):
                OwnedColdImport(ownership,budget=account,parent_budget=phase,deadline=time.monotonic()+30,require_complete=True)
            assert account.used==0
        finally:
            atomic_json(directory/'artifact-registry.json',original)
            if moved is not None:moved[1].rename(moved[0])


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
@pytest.mark.parametrize('fault',['write','published_then_error'])
def test_real_owned_preparation_failure_reaps_originals_and_releases_source_bindings(tmp_path,monkeypatch,fault):
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.ppo_run import _run_ppo
    from tests.shared_inference_support import shared_owned_configuration
    runtime,run_id,_campaign,_settings=shared_owned_configuration(tmp_path,monkeypatch)
    config=runtime.config
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=2) as job:
            ops=_OwnedHpoOperations(job)
            snapshots=[]
            async def collect(plan,ledger):
                controller=ops.shared_controller
                snapshots.append(ops.owned_registry.current.actor_capability)
                def failed(*args,**kwargs):raise RuntimeError('fixture shared '+fault)
                if fault=='write':controller._write=failed
                else:
                    publish=controller.service.publish
                    def published(*args,**kwargs):
                        publish(*args,**kwargs)
                        failed()
                    controller.service.publish=published
                artifacts=ops.owned_registry.artifacts(plan.bindings)
                try:controller.ensure(plan.bindings,artifacts)
                finally:artifacts.release()
            with pytest.raises(RuntimeError,match='fixture shared '+fault):
                await _run_ppo(runtime,collect,tmp_path/'parents'/run_id,_game_ops=ops)
            assert ops.shared_controller.service.closed and not ops.shared_controller.pins
            assert ops.shared_controller.service.lease.released
            assert ops.owned_registry.process.lease.released
            assert all(not snapshot._state.bindings for snapshot in snapshots)
            assert ops.budget.used==0
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('GF_RUN_CUDA')!='1',reason='separate actual CUDA component check')
def test_real_noop_resume_records_never_created_service_and_keeps_first_invocation_proofs(tmp_path,monkeypatch):
    import time

    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.group_pack import StorageSettings
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_owned_restore import OwnedColdImport
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.lineage import run_lock
    from gf_training.ppo_run import _run_ppo
    from tests.shared_inference_support import shared_owned_configuration
    from tests.test_hpo_owned_run import owned_fixture_collection
    runtime,run_id,_campaign,_settings=shared_owned_configuration(tmp_path,monkeypatch,decision_budget=16)
    config=runtime.config
    directory=tmp_path/'parents'/run_id
    async def invoke(mode):
        with gpu.admit_gpu_job(gpu.JobContext('fixture',run_id,'training',run_id,None),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=2) as job:
            ops=_OwnedHpoOperations(job)
            async def collect(plan,ledger):
                assert mode=='fresh','no-op resume must not create/admit a service or collect'
                return await owned_fixture_collection(ops,config,plan,ledger)
            result=await _run_ppo(runtime,collect,directory,mode=mode,pause_after_update=1 if mode=='fresh' else None,_game_ops=ops)
            if mode=='resume':assert ops.shared_controller is None
            if ops.completion is not None:ops.completion.close()
            assert ops.budget.used==0
            return result
    assert asyncio.run(invoke('fresh'))['status']=='paused'
    assert asyncio.run(invoke('resume'))['status']=='complete'
    deadline=time.monotonic()+30
    account=StorageBudget(StorageSettings(),deadline)
    phase=ArtifactPhaseAdmission(account,config.hpo_spec['operating']['parent_budget_bytes'])
    with run_lock(directory) as owner:
        imported=OwnedColdImport(owner,budget=account,parent_budget=phase,deadline=deadline,require_complete=True)
        try:
            first,second=(row['shared_service'] for row in imported.state['invocations'])
            assert first['service_id'] is not None and second['service_id'] is None
            assert second['start_generation']==second['end_generation']==1 and second['publications']==[]
        finally:imported.close()
    assert account.used==0
