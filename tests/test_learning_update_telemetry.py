"""Bounded telemetry is observational and never supplies lifecycle authority."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, canonical
from gf_training.hpo_training_execution import (
    TrainingExecutionSpec,
    read_execution_spec,
    write_execution_spec,
)


def test_spec_v1_bytes_stay_closed_and_v2_observation_is_explicit(tmp_path):
    spec=TrainingExecutionSpec(variant='packed')
    legacy=asdict(spec)
    legacy.pop('observe_actual',None)
    path=tmp_path/'legacy.json'
    sha=write_execution_spec(path,spec)
    assert path.read_bytes()==canonical(legacy).encode()
    assert not read_execution_spec(path,sha).observe_actual
    observed=TrainingExecutionSpec(schema_version=2,variant='packed',observe_actual=True,
        diagnostic_directory=str(tmp_path/'observations'))
    path2=tmp_path/'observed.json'
    sha2=write_execution_spec(path2,observed)
    assert read_execution_spec(path2,sha2)==observed
    assert json.loads(path2.read_bytes())['observe_actual'] is True
    legacy['observe_actual']=False
    raw=canonical(legacy).encode()
    (tmp_path/'wrong-v1.json').write_bytes(raw)
    with pytest.raises(ContractError):
        read_execution_spec(tmp_path/'wrong-v1.json',hashlib.sha256(raw).hexdigest())
    for kwargs in ({'schema_version':1,'observe_actual':True},
            {'schema_version':2,'observe_actual':1},
            {'schema_version':2,'observe_actual':True},
            {'schema_version':2,'observe_actual':True,'diagnostic_directory':str(tmp_path),'profile_start':0}):
        with pytest.raises(ContractError):
            TrainingExecutionSpec(**kwargs)


def bindings(role='worker'):
    return {'role':role,'model_id':'mlp','variant':'baseline','resume_identity':'a'*64,
        'spec_sha256':'b'*64,'source_digest':'c'*64,'pid':os.getpid(),
        'generation':0 if role=='worker' else None,'transaction':'original-transaction' if role=='worker' else None,
        'run_id':None,'context_id':None}


def test_exclusive_observer_lease_samples_and_releases_only_after_publication(tmp_path):
    from gf_training.hpo_execution_observation import (
        LEASE_BYTES,
        ProcessObservation,
        read_observation,
    )
    owner=ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    with pytest.raises(ContractError,match='exclusive'):
        ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    with owner.phase('receive_validation'):
        time.sleep(.06)
    for name in ('update_call','input_retirement','snapshot_export','snapshot_save','snapshot_transfer'):
        with owner.phase(name):
            pass
    owner.numeric_loop_update_seconds=.01
    outcome=owner.finish_publish(tmp_path/'worker.json',original_completed=True)
    assert outcome['published'] and outcome['diagnostic_complete']
    value=read_observation(tmp_path/'worker.json')
    assert value['lease_bytes']==LEASE_BYTES
    assert value['sampler']['stopped'] and value['sampler']['sample_count']>=4
    assert value['sampler']['max_pss_bytes']>0 and value['sampler']['max_rss_bytes']>0
    assert value['sampler']['max_gap_seconds']>=0
    assert value['phases']['receive_validation']['wall_seconds']>=.05
    next_owner=ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    next_owner.start()
    next_owner.finish_publish(tmp_path/'next.json',original_completed=False)


def test_insufficient_allowance_refuses_before_thread_creation(monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation.threading,'Thread',lambda **kwargs:pytest.fail('thread before lease admission'))
    with pytest.raises(ContractError,match='allowance'):
        observation.ProcessObservation(bindings(),allowance_bytes=observation.LEASE_BYTES-1)


def test_sampler_timeout_retains_exclusive_owner_and_never_joins_unbounded(tmp_path,monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    calls=[]
    class Stuck:
        def __init__(self,**kwargs):
            pass
        def start(self):
            pass
        def join(self,timeout):
            calls.append(timeout)
        def is_alive(self):
            return True
    monkeypatch.setattr(observation.threading,'Thread',Stuck)
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    result=owner.finish_publish(tmp_path/'timeout.json',original_completed=False)
    assert calls==[.25] and not result['diagnostic_complete']
    assert observation._POOL.owner is owner
    with pytest.raises(ContractError,match='exclusive'):
        observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)


def test_proc_failure_is_explicit_and_original_exception_propagates(tmp_path,monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    def failed():
        raise OSError('proc unavailable')
    monkeypatch.setattr(observation,'_read_proc',failed)
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    primary=RuntimeError('actual original failure')
    with pytest.raises(RuntimeError) as caught:
        try:
            with owner.phase('update_call'):
                raise primary
        finally:
            result=owner.finish_publish(tmp_path/'failure.json',original_completed=False)
    assert caught.value is primary and not result['diagnostic_complete']
    record=observation.read_observation(tmp_path/'failure.json')
    assert record['failures']['count']>0 and record['sampler']['max_pss_bytes'] is None
    assert not record['original_pipeline_completed']


def test_publication_is_exclusive_bounded_and_failure_holds_lease(tmp_path,monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    path=tmp_path/'existing.json'
    path.write_bytes(b'original')
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    result=owner.finish_publish(path,original_completed=False)
    assert not result['published'] and not result['diagnostic_complete']
    assert path.read_bytes()==b'original' and observation._POOL.owner is owner
    with pytest.raises(ContractError):
        observation.ProcessObservation({**bindings(),'bad':object()},allowance_bytes=256*1024**2)


@pytest.mark.parametrize('mode',['off','observed','publication_failure'])
def test_original_cpu_worker_observation_and_close_boundary(tmp_path,mode):
    _owned_observation_lifecycle(tmp_path,mode)


def _owned_observation_lifecycle(tmp_path,mode,*,device='cpu',ledger=None):
    import asyncio
    from dataclasses import replace

    from gf_models.structured_config import resource_estimate
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_execution_observation import read_observation, validate_observation
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget, captured_record
    from tests.test_hpo_owned_wire import recapture
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1),device=device)
    account=budget()
    diagnostic=tmp_path/'observer'
    diagnostic.mkdir()
    execution=None
    if mode!='off':
        path=tmp_path/'spec.json'
        sha=write_execution_spec(path,TrainingExecutionSpec(schema_version=2,observe_actual=True,
            diagnostic_directory=str(diagnostic)))
        execution=(str(path),sha)
    existing=diagnostic/'update-000000.json'
    if mode=='publication_failure':
        existing.write_bytes(b'preserved original')
    async def run():
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ReservationLedger() if ledger is None else ledger,
            cuda_visible_devices='' if device=='cpu' else os.environ.get('CUDA_VISIBLE_DEVICES','0'),cleanup_deadline=time.monotonic()+60,budget=account,
            _training_execution=execution)
        if execution is not None and device=='cpu':
            # CPU fixture reserves the same real original resource envelope so
            # argv carries a positive genuine lease, without a creator override.
            child.resident_bytes=resource_estimate(learner.model,learner.binding,
                suffix_decisions=learner.training.tbptt_length,layout=learner.network.layout)['total_bytes']
        deadline=time.monotonic()+60
        await child.start()
        initial=updated=None
        try:
            initial=await child.initialize_owned(17,tmp_path/'g0',deadline=deadline)
            local,owner,plan,local_snapshot,_local_budget,record=captured_record(learner)
            local.close()
            captured,owner,plan=recapture(record,initial,account)
            selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
            numeric=encode_owned_batch(captured,owner=owner,plan=plan,snapshot_owner=initial,
                model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
                selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
            numeric.persist(tmp_path/'numeric')
            updated=await child.update_owned(numeric,tmp_path/'g1',deadline=deadline)
            assert updated.generation==1 and not child.channel.outstanding
            numeric.close()
            numeric.require_retired()
            captured.close()
            local_snapshot.close()
        finally:
            await child.close(deadline)
            if initial is not None: initial.close()
            if updated is not None: updated.close()
        assert child.process.returncode==0 and child.lease.released
        return child.process.pid
    pid=asyncio.run(run())
    if mode=='off':
        assert not list(diagnostic.iterdir())
    elif mode=='publication_failure':
        assert existing.read_bytes()==b'preserved original'
    else:
        value=read_observation(existing)
        validate_observation(value,expected={'role':'worker','pid':pid,'generation':0,
            'resume_identity':learner.resume_identity,'spec_sha256':execution[1]},require_cuda=device!='cpu')
        if device=='cpu':
            assert value['cuda']=={'status':'not_requested'}
        else:
            assert value['cuda']['status']=='observed' and value['cuda']['update_call_event_ms']>0
            assert all(value['cuda']['allocator_after_retirement'][name]>0
                for name in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'))
        assert value['numeric_loop_update_seconds']>=0 and value['sampler']['stopped']
        assert existing.stat().st_size<=64*1024
    assert account.used==0
    return pid


@pytest.mark.skipif(os.environ.get('GFB_RUN_TELEMETRY_CUDA')!='1',reason='explicit root-owned CUDA telemetry check')
def test_original_cuda_worker_telemetry_and_genuine_receipt(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('host CUDA unavailable')
    from gf_training import hpo_gpu_lease as gpu
    # Isolate durable test admission files while retaining the original audited
    # job/creator/FD/close/reap proof. This is not a performance/game population.
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'gpu-job')
    directory=tmp_path/'worker'
    directory.mkdir()
    with gpu.admit_gpu_job(gpu.JobContext('telemetry-fixture','operation','training','worker-fixture',None),
            global_cuda_bytes=12*1024**3,max_workers=4) as job:
        pid=_owned_observation_lifecycle(directory,'observed',device='cuda:0',ledger=job.ledger)
        receipt=job.confirm_termination()
        assert job.require_confirmed_termination(receipt) is receipt
        assert len(receipt.resources)==1
        resource=receipt.resources[0]
        assert resource.kind=='learner' and resource.outcome=='reaped' and resource.pid==pid


def test_phase_clock_and_stop_errors_do_not_mask_original(tmp_path,monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    original=RuntimeError('authoritative')
    real_clock=observation.time.perf_counter
    def bad_clock():
        raise OSError('clock failure')
    with monkeypatch.context() as patch:
        patch.setattr(observation.time,'perf_counter',bad_clock)
        with pytest.raises(RuntimeError) as caught, owner.phase('update_call'):
            raise original
        assert caught.value is original
    assert observation.time.perf_counter is real_clock
    result=owner.finish_publish(tmp_path/'clock-error.json',original_completed=False)
    assert not result['diagnostic_complete']


def test_no_input_aliases_and_bounded_proc_reads(tmp_path,monkeypatch):
    import weakref

    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    class Input:
        pass
    token=Input()
    reference=weakref.ref(token)
    raw={**bindings(),'resume_identity':token}
    with pytest.raises(ContractError):
        observation.ProcessObservation(raw,allowance_bytes=256*1024**2)
    assert observation._POOL.owner is None
    del raw,token
    assert reference() is None
    reads=[]
    real=observation.os.read
    def read(fd,limit):
        reads.append(limit)
        return real(fd,limit)
    monkeypatch.setattr(observation.os,'read',read)
    pss,rss=observation._read_proc()
    assert pss>0 and rss>0 and reads==[8192]
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    for _ in range(100):
        owner.sample()
    owner.finish_publish(tmp_path/'bounded.json',original_completed=False)
    value=observation.read_observation(tmp_path/'bounded.json')
    assert value['sampler']['sample_count']>=100 and value['sampler']['retained_samples']==0
    assert (tmp_path/'bounded.json').stat().st_size<=64*1024


def test_explicit_diagnostic_flag_labels_request_and_original_failure_separately(tmp_path,monkeypatch):
    from benchmarks.learning_update import actual
    from tests.test_learning_update_benchmarks import (
        test_actual_request_reaches_observer_pre_gpu_hash_consumer,
    )
    parse=actual.argparse.ArgumentParser.parse_args
    def observed_args(*args,**kwargs):
        result=parse(*args,**kwargs)
        result.diagnostic_telemetry=True
        return result
    monkeypatch.setattr(actual.argparse.ArgumentParser,'parse_args',observed_args)
    # Reuse the ORIGINAL historical preflight-consumer regression. It raises
    # before real GPU admission/coordinator; telemetry must not replace that.
    test_actual_request_reaches_observer_pre_gpu_hash_consumer(tmp_path,monkeypatch)
    value=json.loads((tmp_path/'report'/'result.json').read_bytes())
    assert not value['timing_eligible'] and value['diagnostic_complete'] is False
    assert value['failure']['type']=='BeforeGpu'
    request=value['request']
    assert request['diagnostic_telemetry'] and not request['timing_eligible']
    spec=read_execution_spec(__import__('pathlib').Path(request['execution_path']),request['execution_sha256'])
    assert spec.schema_version==2 and spec.observe_actual


@pytest.mark.parametrize('mode,pause',[('resume',None),('fresh',1)])
def test_observed_spec_cannot_enter_resume_or_pause(tmp_path,mode,pause):
    import asyncio
    from types import SimpleNamespace

    from gf_training.hpo_ppo import train_hpo_ppo
    path=tmp_path/'observed.json'
    sha=write_execution_spec(path,TrainingExecutionSpec(schema_version=2,observe_actual=True,
        diagnostic_directory=str(tmp_path/'records')))
    loaded=SimpleNamespace(config=SimpleNamespace(hpo_spec={'schema_version':4}))
    with pytest.raises(ContractError,match='fresh producer-owned'):
        asyncio.run(train_hpo_ppo(loaded,None,tmp_path/'parent',gpu_job=None,mode=mode,
            pause_after_update=pause,_training_execution=(str(path),sha)))


def test_sampler_start_stop_errors_retain_owner_and_preserve_work(tmp_path,monkeypatch):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    class Broken:
        def __init__(self,**kwargs):
            pass
        def start(self):
            raise OSError('start failure')
        def join(self,timeout):
            assert timeout==.25
            raise OSError('join failure')
    monkeypatch.setattr(observation.threading,'Thread',Broken)
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    ran=[]
    with owner.phase('update_call'):
        ran.append('original work')
    result=owner.finish_publish(tmp_path/'failed.json',original_completed=False)
    assert ran==['original work'] and not result['diagnostic_complete']
    assert observation._POOL.owner is owner


def test_helper_requires_all_bound_records_and_real_cuda_fields(tmp_path):
    from benchmarks.learning_update.actual import verify_telemetry
    from gf_training.hpo_execution_observation import ProcessObservation
    records=tmp_path/'records'
    (records/'learner').mkdir(parents=True)
    parent=tmp_path/'parent'
    parent.mkdir()
    (parent/'run.json').write_text(json.dumps({'resume_identity':'a'*64}))
    for role,generation in (('coordinator',None),('worker',0),('worker',1)):
        metadata=bindings(role)
        metadata.update(run_id='run' if role=='coordinator' else None,
            context_id='context' if role=='coordinator' else None,generation=generation,
            transaction=None if role=='coordinator' else f'transaction-{generation}')
        owner=ProcessObservation(metadata,allowance_bytes=256*1024**2)
        owner.start()
        for phase in (('coordinator_call',) if role=='coordinator' else (
                'receive_validation','update_call','input_retirement','snapshot_export','snapshot_save','snapshot_transfer')):
            with owner.phase(phase):
                pass
        if role=='worker':
            # Synthetic validator fixture only; no CUDA execution/evidence claim.
            owner.numeric_loop_update_seconds=.1
            owner.cuda={'status':'observed','update_call_event_ms':1.,'allocator_after_retirement':{
                'allocated_bytes':10,'reserved_bytes':20,'peak_allocated_bytes':30,'peak_reserved_bytes':40}}
        path=records/'coordinator-observation.json' if role=='coordinator' else records/'learner'/f'update-{generation:06d}.json'
        assert owner.finish_publish(path,original_completed=True)['diagnostic_complete']
    result={'probe_expectation_met':True,'gpu_termination_context_verified':True,
        'production_original_cleanup_receipt_revalidated':True,
        'gpu_termination':{'resources':[{'kind':'learner','outcome':'reaped','pid':os.getpid()}]}}
    arguments={'parent':parent,'spec_sha':'b'*64,'source_digest':'c'*64,'variant':'baseline',
        'model_id':'mlp','run_id':'run','context_id':'context'}
    assert verify_telemetry(result,records,**arguments)['diagnostic_complete']
    path=records/'learner'/'update-000001.json'
    value=json.loads(path.read_bytes())
    value['cuda']={'status':'not_requested'}
    path.write_text(json.dumps(value))
    assert not verify_telemetry(result,records,**arguments)['diagnostic_complete']
    path.unlink()
    assert not verify_telemetry(result,records,**arguments)['diagnostic_complete']


@pytest.mark.parametrize('observe,failure',[(False,None),(True,None),(True,'update'),(True,'retirement'),(True,'save')])
def test_worker_off_has_no_observer_markers_and_original_phase_order(tmp_path,monkeypatch,observe,failure):
    import io
    import weakref
    from types import SimpleNamespace

    from gf_training import hpo_execution_observation as observation
    from gf_training import hpo_numeric, hpo_owned_worker, structured_numeric_worker
    from gf_training.hpo_numeric_worker import wire_identity
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    identity=wire_identity(learner.model,learner.training,learner.binding,learner.action_config_hash,
        learner.resume_identity,actions=ACTIONS,cuda_visible_devices='')
    diagnostic=tmp_path/'observations'
    events=[]
    aliases=[]
    original_error=OSError('original-'+str(failure))
    class Numeric:
        def __del__(self):
            events.append('del_numeric')
    class Delivery:
        def complete(self,sync):
            events.append('delivery_complete')
            if failure=='retirement':
                raise original_error
            sync()
    class Control:
        def close(self):
            events.append('control_close')
    class Snapshot:
        identity='snapshot'
        def save(self,path):
            events.append('save')
            if failure=='save' and learner._owned_generation:
                raise original_error
    def receive(*args,**kwargs):
        events.append('receive')
        numeric=Numeric()
        aliases.append(weakref.ref(numeric))
        return numeric,[Delivery()],Control()
    def update(numeric):
        events.append('update')
        if failure=='update':
            raise original_error
        learner._owned_generation+=1
        return {'update_seconds':.01}
    def export(**kwargs):
        events.append('export')
        return Snapshot()
    def send(*args,**kwargs):
        events.append('send')
        # Full telemetry publication may never precede original snapshot send.
        assert not (diagnostic/'update-000000.json').exists()
    def close():
        events.append('close')
        if observe:
            assert (diagnostic/'update-000000.json').is_file()
    monkeypatch.setattr(hpo_numeric,'HpoLearner',lambda *args,**kwargs:learner)
    monkeypatch.setattr(learner,'update_owned',update)
    monkeypatch.setattr(learner,'snapshot_owned',export)
    monkeypatch.setattr(learner,'synchronize',lambda:events.append('sync'))
    monkeypatch.setattr(hpo_owned_worker,'_receive_numeric',receive)
    monkeypatch.setattr(hpo_owned_worker,'_send_snapshot',send)
    monkeypatch.setattr(hpo_owned_worker.RamBankChannel,'_from_inherited',lambda *args:SimpleNamespace(close=close))
    monkeypatch.setattr(structured_numeric_worker,'_select_visibility',lambda *args:None)
    monkeypatch.setattr(learner.torch.cuda,'Event',lambda **kwargs:pytest.fail('CPU/off event marker'))
    if not observe:
        monkeypatch.setattr(observation,'ProcessObservation',lambda *args,**kwargs:pytest.fail('off sampler created'))
    initial={'protocol_version':7,'command':'initialize_owned','identity':identity,'model':asdict(learner.model),
        'seed':17,'transaction':'initial','output':str(tmp_path/'g0')}
    request={'protocol_version':7,'command':'update_owned','identity':identity,'transaction':'update',
        'generation':0,'expected_steps':0,'batch_identity':'a'*64,'bank_count':1,
        'policy_version':'fixture','output':str(tmp_path/'g1')}
    closing={'protocol_version':7,'command':'close','identity':identity}
    monkeypatch.setattr('sys.stdin',io.StringIO('\n'.join(json.dumps(row) for row in (initial,request,closing))+'\n'))
    argv=['worker','-1','fixture']
    if observe:
        path=tmp_path/'observed.json'
        sha=write_execution_spec(path,TrainingExecutionSpec(schema_version=2,observe_actual=True,
            diagnostic_directory=str(diagnostic)))
        argv.extend([str(path),sha,str(learner.binding.managed_budget_bytes)])
    monkeypatch.setattr('sys.argv',argv)
    if failure is not None:
        with pytest.raises(OSError) as caught:
            hpo_owned_worker.main()
        assert caught.value is original_error
        # Drop retained test traceback aliases before checking input destruction.
        caught.value.__traceback__=None
        del caught
        value=observation.read_observation(diagnostic/'update-000000.json')
        assert value['sampler']['stopped'] and not value['original_pipeline_completed']
        assert not value['diagnostic_complete']
    else:
        hpo_owned_worker.main()
        assert events==['export','save','send','receive','update','del_numeric','sync',
            'delivery_complete','sync','control_close','export','save','send','close']
    assert all(reference() is None for reference in aliases)


@pytest.mark.parametrize('mode',['off','observed','cancel','publication_failure','oversized_binding'])
def test_coordinator_original_cleanup_precedes_observer_publication(tmp_path,monkeypatch,mode):
    import asyncio
    from dataclasses import dataclass
    from types import SimpleNamespace

    from benchmarks.learning_update import actual
    from benchmarks.learning_update.common import prior_runner
    from gf_cli import hpo_config
    from gf_training import hpo_execution_observation as observation
    from gf_training import hpo_ppo
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    monkeypatch.setattr(actual,'REPO',tmp_path)
    config=tmp_path/'config.toml'
    config.write_text('noGPU coordinator fixture')
    parent_root=tmp_path/'runs/learning-update-optimization-v1/telemetry-v1/parents'
    run_id='00000000-0000-4000-8000-000000000017'
    parent=parent_root/run_id
    output=tmp_path/'report'
    @dataclass
    class Config:
        storage: object
        model: object=None
        training: object=None
    loaded=SimpleNamespace(config=Config(SimpleNamespace(parent_root=str(parent_root))))
    runtime=SimpleNamespace(sha256='a'*64,config=SimpleNamespace(hpo_spec={'operating':{'parent_budget_bytes':1024**3}}))
    settings=SimpleNamespace(model=SimpleNamespace(model_id='mlp'),training=None)
    events=[]
    original_error=asyncio.CancelledError('original cancellation')
    async def original(*args,**kwargs):
        events.append('original-start')
        parent.mkdir(parents=True)
        (parent/'run.json').write_text(json.dumps({'resume_identity':'a'*64,
            **({'padding':'x'*70000} if mode=='oversized_binding' else {})}))
        if mode=='publication_failure':
            (output/'coordinator-observation.json').write_bytes(b'preserved')
        try:
            await asyncio.sleep(0)
            if mode=='cancel':
                raise original_error
            return SimpleNamespace(report={'status':'incomplete'})
        finally:
            events.append('original-cleanup')
    async def execute(args,result,diagnostic):
        try:
            await hpo_ppo.train_hpo_ppo(runtime,None,parent,gpu_job=None,mode='fresh',pause_after_update=None)
        except asyncio.CancelledError as error:
            assert error is original_error and events[-1]=='original-cleanup'
            raise
        events.append('returned')
        if mode!='off':
            assert (output/'coordinator-observation.json').exists()
        result['probe_expectation_met']=True
        result['gpu_termination_context_verified']=True
        result['production_original_cleanup_receipt_revalidated']=True
        result['gpu_termination']={'resources':[]}
    runner=prior_runner()
    monkeypatch.setattr(runner,'materialize',lambda *args:(loaded,settings,runtime))
    monkeypatch.setattr(runner,'production_patches',lambda *args:[])
    monkeypatch.setattr(runner,'execute',execute)
    monkeypatch.setattr(hpo_ppo,'train_hpo_ppo',original)
    monkeypatch.setattr(actual,'prior_runner',lambda:runner)
    monkeypatch.setattr(actual,'protocol',lambda *args:None)
    monkeypatch.setattr(actual,'verify_freeze',lambda *args:{'files':{}})
    monkeypatch.setattr(hpo_config,'joint_resource_admission',lambda *args,**kwargs:{})
    if mode=='off':
        monkeypatch.setattr(observation,'ProcessObservation',lambda *args,**kwargs:pytest.fail('off coordinator observer'))
    argv=['actual','--config',str(config),'--output',str(output),'--run-id',run_id,
        '--context-id','00000000-0000-4000-8000-000000000018','--variant','baseline',
        '--freeze',str(tmp_path/'freeze.json'),'--execute']
    if mode!='off':
        argv.append('--diagnostic-telemetry')
    monkeypatch.setattr('sys.argv',argv)
    assert actual.main()==(0 if mode=='off' else 2)
    result=json.loads((output/'result.json').read_bytes())
    assert result['train_calls']==1
    if mode=='cancel':
        assert result['failure']['type']=='CancelledError'
        record=observation.read_observation(output/'coordinator-observation.json')
        assert record['sampler']['stopped'] and not record['diagnostic_complete']
    else:
        assert result['probe_expectation_met'] and 'failure' not in result
    if mode=='observed':
        record=observation.read_observation(output/'coordinator-observation.json')
        assert record['sampler']['stopped'] and record['diagnostic_complete']
        # Fake fixture has no real learner records/receipt: overall diagnostic
        # failure cannot be disguised as original training failure or success.
        assert not result['diagnostic_complete']
    elif mode=='oversized_binding':
        record=observation.read_observation(output/'coordinator-observation.json')
        assert record['sampler']['stopped'] and not record['diagnostic_complete']
        assert record['failures']['count']>0 and record['bindings']['resume_identity'] is None
    elif mode=='publication_failure':
        assert (output/'coordinator-observation.json').read_bytes()==b'preserved'
        assert not result['coordinator_observation']['published']
    elif mode=='off':
        assert not (output/'coordinator-observation.json').exists()


@pytest.mark.parametrize('operation',['close','unlink'])
def test_uncertain_publication_keeps_pending_and_lease(tmp_path,monkeypatch,operation):
    from gf_training import hpo_execution_observation as observation
    monkeypatch.setattr(observation,'_POOL',observation._ObservationPool())
    owner=observation.ProcessObservation(bindings(),allowance_bytes=256*1024**2)
    owner.start()
    owner.numeric_loop_update_seconds=.01
    for phase in observation.WORKER_PHASES:
        with owner.phase(phase):
            pass
    path=tmp_path/'worker.json'
    real_close=observation.os.close
    real_unlink=observation.os.unlink
    pending=tmp_path/'.observation-worker.json.pending'
    def failed_close(fd):
        if fd==owner._pending_fd:
            raise OSError('uncertain close')
        return real_close(fd)
    def failed_unlink(name,*args,**kwargs):
        if str(name)==str(pending):
            raise OSError('uncertain unlink')
        return real_unlink(name,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(observation.os,'close',failed_close)
        if operation=='unlink':
            patch.setattr(observation.os,'close',real_close)
            patch.setattr(observation.os,'unlink',failed_unlink)
        result=owner.finish_publish(path,original_completed=True)
    assert not result['diagnostic_complete'] and not result['published']
    assert pending.is_file() and observation._POOL.owner is owner
    assert path.exists()==(operation=='unlink')
    if operation=='close':
        assert owner._pending_fd is not None
        # The test injection definitively did not close this diagnostic FD;
        # release it explicitly here. Production never retries uncertain close.
        real_close(owner._pending_fd)
    else:
        assert owner._pending_fd is None
        observation.validate_observation(observation.read_observation(path),expected=bindings(),require_cuda=False)
        from benchmarks.learning_update.actual import verify_telemetry
        value=verify_telemetry({'probe_expectation_met':True,'gpu_termination_context_verified':True,
            'production_original_cleanup_receipt_revalidated':True},tmp_path,parent=tmp_path/'missing',
            spec_sha='b'*64,source_digest='c'*64,variant='baseline',model_id='mlp',run_id='run',context_id='context')
        assert not value['diagnostic_complete'] and 'pending residue' in value['failure']


def test_pending_residue_overrides_valid_looking_complete_records(tmp_path,monkeypatch):
    from benchmarks.learning_update.actual import verify_telemetry
    records=tmp_path/'records'
    (records/'learner').mkdir(parents=True)
    (records/'coordinator-observation.json').write_bytes(b'valid-looking bytes must not be consulted')
    (records/'.observation-coordinator-observation.json.pending').write_bytes(b'pending')
    result={'probe_expectation_met':True,'gpu_termination_context_verified':True,
        'production_original_cleanup_receipt_revalidated':True}
    value=verify_telemetry(result,records,parent=tmp_path/'nonexistent',spec_sha='b'*64,
        source_digest='c'*64,variant='baseline',model_id='mlp',run_id='run',context_id='context')
    assert not value['diagnostic_complete'] and 'pending residue' in value['failure']
