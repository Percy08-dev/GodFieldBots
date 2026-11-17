"""Protocol population, deterministic selection and immutable launch checks."""
from copy import deepcopy
from pathlib import Path

import pytest


def test_options_require_all_closed_fields(tmp_path):
    from benchmarks.collection_cpu.common import immutable_json, options
    p = tmp_path/'options.json'
    immutable_json(p, {'parent':False,'transfer':'T0','http':False,'diagnostic':False})
    assert options(p).transfer == 'T0'
    with pytest.raises(FileExistsError):
        immutable_json(p, {})


def test_trial_budget_order_and_independent_confirmation():
    from benchmarks.collection_cpu.trials import materialize
    pilot = materialize('12345678-1234-5678-1234-567812345678', 'pilot', [], 'T2')
    assert [(r['model'], r['arm']) for r in pilot] == [
        ('mlp','C'),('mlp','P'),('mlp','T'),('mlp','H'),
        ('gru','H'),('gru','T'),('gru','P'),('gru','C'),
        ('transformer','P'),('transformer','C'),('transformer','H'),('transformer','T')]
    confirm = materialize('12345678-1234-5678-1234-567812345678', 'confirm', ['P','H'], 'T2')
    combo = materialize('12345678-1234-5678-1234-567812345678', 'combo', ['P','H'], 'T2')
    assert len(pilot)+len(confirm)+len(combo) == 54
    assert [r['arm'] for r in confirm[:4]] == ['P','C','C','P']
    assert [r['arm'] for r in combo[:6]] == ['C','B','B','C','C','B']
    assert len({r['run_id'] for r in pilot+confirm+combo}) == 54
    assert len({r['context_id'] for r in pilot+confirm+combo}) == 54
    assert not {r['run_id'] for r in pilot}&{r['run_id'] for r in confirm}
    assert materialize('12345678-1234-5678-1234-567812345678', 'combo', ['P'], 'T2') == []


def test_selection_signed_sum_ties_and_negative_candidates():
    from benchmarks.collection_cpu.trials import select_candidates
    rows = {model:{'C':100.,'P':102.,'T':101.,'H':103.} for model in ('mlp','gru','transformer')}
    result = select_candidates(rows, {'P':True,'T':True,'H':True})
    assert result['selected'] == ['P','T']  # signed differences within 1% unresolved; protocol preference
    assert result['scores']['P'] == -6.
    assert result['timing_win_claimed'] is False
    rows['mlp']['H'] = 1.
    assert select_candidates(rows, {'P':True,'T':True,'H':True})['selected'][0] == 'H'
    broken = deepcopy(rows)
    del broken['gru']['C']
    with pytest.raises(ValueError, match='complete'):
        select_candidates(broken, {'P':True,'T':True,'H':True})


def test_transfer_selection_common_all_models_three_pairs():
    from benchmarks.collection_cpu.trials import select_transfer
    rows = {model:{mode:[{'control':10.,'candidate':11. if mode == 'reuse' else 10.95,
                         'population':'same','valid':True,'pair':i,
                         'reports':[f'{model}-{mode}-{i}-c',f'{model}-{mode}-{i}-v']} for i in (1,2,3)]
                   for mode in ('reuse','T1','T2','T3')} for model in ('mlp','gru','transformer')}
    result = select_transfer(rows)
    assert result['selected'] == 'reuse' and result['T0_faster'] is True
    rows['gru']['T2'][0]['population'] = 'different'
    with pytest.raises(ValueError, match='population'):
        select_transfer(rows)


def test_profile_endpoints_are_small_and_recorded_tail_full_prefix():
    from benchmarks.collection_cpu.replay import profile_endpoints
    batches = [{'sequences':[{'frames':[{}]*3,'candidates':[{'kinds':[0]*n} for n in (1,50,8)]},
                             {'frames':[{}]*3,'candidates':[{'kinds':[0]*n} for n in (2,40,3)]}]}]
    assert profile_endpoints(batches) == [(0,0,0),(0,1,0),(0,1,2),(0,0,2),(0,1,1),(0,0,1)]


def test_selection_can_exclude_mechanically_invalid_missing_candidate():
    from benchmarks.collection_cpu.trials import select_candidates
    rows = {model:{'C':10.,'P':11.,'H':12.} for model in ('mlp','gru','transformer')}
    assert select_candidates(rows,{'P':True,'T':False,'H':True})['selected']==['P','H']


def test_replay_aggregation_keeps_missing_trials_explicit(tmp_path):
    from benchmarks.collection_cpu.aggregate import replay_ledger
    plan = {'pairs':[{'model':'mlp','transfer':'T1','pair':1,
                      'reports':[str(tmp_path/'missing-c'),str(tmp_path/'missing-v')]}]}
    report = replay_ledger(plan)
    assert report['ledger'][0]['status']=='FAILED_OR_NOT_RUN'
    assert report['rows']['mlp']['T1']==[{'valid':False,'population':None}]


def test_diagnostics_preserve_original_result_failure_and_only_scalar_counts():
    from benchmarks.collection_cpu.diagnostics import Counters
    counts = Counters()
    private = object()
    def original(value):
        assert value is private
        return private
    def fail(value):
        raise RuntimeError('original failure')
    assert counts.wrap('original',original)(private) is private
    with pytest.raises(RuntimeError,match='original failure'):
        counts.wrap('fail',fail)(private)
    payload = counts.payload()
    assert sorted(row['count'] for row in payload['rows'].values())==[1,1]
    assert sum(row['failures'] for row in payload['rows'].values())==1
    assert 'object at' not in str(payload)


def test_profile_summary_uses_actual_copy_events_not_wrapper_counts(tmp_path):
    from benchmarks.collection_cpu.common import immutable_json
    from benchmarks.collection_cpu.replay import trace_summary
    path = tmp_path/'trace.json'
    immutable_json(path,{'traceEvents':[{'name':'Memcpy HtoD','args':{'bytes':24}},
        {'name':'Memcpy DtoH','args':{}},{'name':'kernel','cat':'kernel'},
        {'name':'cudaStreamSynchronize','cat':'cuda_runtime'}]})
    report = trace_summary([path])
    assert report['copies']['H2D']['bytes']==24
    assert report['copies']['D2H']['bytes'] is None
    assert report['kernel_activities']=={'kernel':1}


def test_actual_installer_charge_symmetry_and_original_bootstrap_arguments(tmp_path):
    from benchmarks.collection_cpu.actual import execution_patches
    from benchmarks.collection_cpu.common import Options
    from gf_models import structured_runtime
    from gf_simulator_client.resident_pool import ResidentPhysicalWorker
    base = {'workers':4,'ram_phases':{'collection':100,'save':200},'worker_bytes':30,
            'peak_cuda_bytes':50,'managed_ram_bytes':2**40,'managed_cuda_bytes':2**40}
    for settings in (Options(),Options(parent=True,transfer='T2',http=True)):
        replacements = execution_patches(settings,tmp_path)
        wrapped = {(owner,name):fn for owner,name,fn in replacements}
        assert (ResidentPhysicalWorker,'_spawn') in wrapped
        # Existing charged_admission helper is responsible for both actual pool
        # residency and preflight. Prove the exact same envelope for both arms.
        from benchmarks.collection_cpu.common import inference_helper
        charged = inference_helper('actual').charged_admission(lambda *a,**k:base,settings)
        from gf_models.structured_config import InferenceBindingV2
        result = charged(None,InferenceBindingV2())
        assert result['experimental_storage_per_worker']==96*1024**2
        assert result['ram_phases']['collection']==100+4*96*1024**2
        assert (structured_runtime,'hpo_resident_training_admission') in wrapped


@pytest.mark.parametrize('mode',['T1','T2','T3'])
def test_all_six_cached_transfer_identities_scrub_before_resident_ack(mode):
    from types import SimpleNamespace

    import numpy as np

    from benchmarks.collection_cpu.transfer import install
    from gf_models.resident_cache import ResidentPolicyCache
    from gf_models.structured_network import make_network
    from tests.model3_support import LAYOUT
    from tests.test_hpo_network import small_model4
    cache,retained = ResidentPolicyCache(),[]
    for index in range(6):
        network = make_network(small_model4(),index,layout=LAYOUT)
        experiment = install(network,mode)
        experiment.packer.pack({'refs':tuple(range(20))},{'refs':np.int64},group='frame')
        experiment.consumed()
        retained.extend(pair for pair in experiment.packer.buffers.values())
        policy = SimpleNamespace(network=network,reset=lambda seed,e=experiment:e.reset())
        cache.acquire(str(index)*64,weight_bytes=1,load=lambda p=policy:p,validate=lambda:None)
        cache.active = None
    cache.reset_all()
    assert cache.private_state_empty()
    assert all(not cpu.count_nonzero() and not device.count_nonzero() for cpu,device in retained)


def test_transfer_selection_does_not_invent_fallback_for_failed_mode():
    from benchmarks.collection_cpu.trials import select_transfer
    rows = {model:{mode:[{'control':10.,'candidate':9.,'population':model,'valid':True,'pair':i,
        'reports':[f'{model}-{mode}-{i}-c',f'{model}-{mode}-{i}-v']} for i in (1,2,3)]
        for mode in ('reuse','T1','T2','T3')} for model in ('mlp','gru','transformer')}
    rows['mlp']['reuse'] = [{'valid':False,'population':None}]*3
    result = select_transfer(rows)
    assert result['selected']=='T1' and 'reuse' in result['invalid']


def test_actual_ledger_rejects_changed_pilot_conditions_and_duplicate_independence(monkeypatch):
    from pathlib import Path

    import benchmarks.collection_cpu.aggregate as aggregation
    from benchmarks.collection_cpu.trials import materialize
    rows = materialize('12345678-1234-5678-1234-567812345678','pilot',[],'T1')
    requests = {}
    for row in rows:
        row['config'] = {'sha256':'config'}
        requests[Path(row['output'])/'request.json'] = {'run_id':row['run_id'],'context_id':row['context_id'],
            'options':row['options'],'config_file_sha256':'config','parent':'/parents/'+row['run_id']}
    plan = {'namespace':'12345678-1234-5678-1234-567812345678','phase':'pilot','selected':[],
            'transfer':'T1','trials':rows}
    def summary(path,**kwargs):
        condition = 'different' if str(path)==rows[1]['output'] else 'same'
        row=next(row for row in rows if row['output']==str(path))
        return {'collection_seconds':10.,'options':row['options'],
            'launch_identity':{'run_id':row['run_id'],'context_id':row['context_id'],'options':row['options'],
                'config_sha256':'config','parent':'/parents/'+row['run_id'],'output':str(path)},
            'matched_conditions':{'condition':condition,'materialized':{'hpo_spec':{'storage':{'parent_root':'/parents'}}}}}
    monkeypatch.setattr(aggregation,'actual_summary',summary)
    monkeypatch.setattr(aggregation,'read_json',lambda path:requests[path])
    result = aggregation.actual_ledger([plan])
    assert result['ledger'][1]['status']=='FAILED_OR_NOT_RUN'
    assert 'P' not in result['rows']['mlp']
    with pytest.raises(ValueError,match='repeated'):
        aggregation.actual_ledger([plan,plan])


def test_one_full_warm_replay_per_process_supports_three_independent_pairs(tmp_path):
    from benchmarks.collection_cpu.aggregate import replay_summary
    from benchmarks.collection_cpu.common import immutable_json
    immutable_json(tmp_path/'request.json',{'decisions':314,'input_manifest_sha256':'same',
        'device':'cuda:0','code':{},'transfer':'T1','pair':1,'position':2})
    immutable_json(tmp_path/'receipt.json',{'returncode':0,'failure':None,'actual_child_waited':True})
    immutable_json(tmp_path/'result.json',{'device':'cuda:0','profiled':False,'torch':'pinned',
        'cuda_runtime':'pinned','threads':1,'loops':[{'population':population,'decisions':314,
        'candidates':1000,'host_wall_seconds':seconds} for population,seconds in [('warmup-first',50.),('warm',42.)]]})
    assert replay_summary(tmp_path)['warm_seconds']==42.


def test_frozen_launch_binds_exact_ids_options_materialization_output(tmp_path):
    from benchmarks.collection_cpu.actual import require_launch
    from benchmarks.collection_cpu.common import Options
    row = {'run_id':'run','context_id':'context','materialized_sha256':'config',
           'options':Options().identity(),'output':str(tmp_path)}
    require_launch({'materializations':[row]},run_id='run',context_id='context',
                   materialized_sha256='config',selected=Options(),output=tmp_path)
    for changed in ({'context_id':'other'},{'options':Options(parent=True).identity()},
                    {'output':str(tmp_path/'other')},{'materialized_sha256':'other'}):
        with pytest.raises(ValueError,match='root-frozen launch'):
            require_launch({'materializations':[{**row,**changed}]},run_id='run',context_id='context',
                           materialized_sha256='config',selected=Options(),output=tmp_path)


@pytest.mark.parametrize('settings',[{}, {'parent':True}, {'diagnostic':True}])
def test_composed_actual_layers_preserve_original_capture_measurement_and_validation(tmp_path,monkeypatch,settings):
    import asyncio
    from types import SimpleNamespace

    from benchmarks.collection_cpu import actual, parent
    from benchmarks.collection_cpu.common import Options, inference_helper
    from gf_common.structured import CandidateTable
    from gf_training.card_instance_journal import CardInstanceJournal
    from gf_training.hpo_owned_capture import _CaptureContext
    inherited_before = 'begin_frame' in vars(CardInstanceJournal)
    runner = inference_helper('actual').prior_runner()
    measure = runner.Measurement()
    calls,proofs = [],[]
    frame,table = SimpleNamespace(actor=1,revision=1),object()
    def validate(t,f):
        calls.append('full')
    def capture(*args):
        CandidateTable.validate(table,frame)
    monkeypatch.setattr(CandidateTable,'validate',validate)
    monkeypatch.setattr(_CaptureContext,'begin_decision',capture)
    original_patches = parent.patches
    def expose():
        replacements,proof = original_patches()
        proofs.append(proof)
        return replacements,proof
    monkeypatch.setattr(parent,'patches',expose)
    selected = Options(**settings)
    prepared = actual.execution_patches(selected,tmp_path)
    base = [(_CaptureContext,'begin_decision',measure.wrap('capture_begin_decision',capture))]
    async def execute(*args):
        from contextlib import nullcontext
        with proofs[0].decision() if proofs else nullcontext():
            CandidateTable.validate(table,frame)
            _CaptureContext.begin_decision(object())
            if proofs:
                CandidateTable.validate(table,frame)
    async def check():
        with runner.patches([*base,*prepared]):
            await actual.layered_execute(execute,selected,runner,None,{},None,tmp_path)
    asyncio.run(check())
    assert ('begin_frame' in vars(CardInstanceJournal)) is inherited_before
    assert calls==['full','full']
    assert [row['label'] for row in measure.payload()['rows']]==['capture_begin_decision']


def test_replay_signal_exit_race_always_waits_and_publishes_original_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace

    from benchmarks.collection_cpu import replay
    source,out = tmp_path/'input',tmp_path/'out'
    source.mkdir()
    (source/'manifest.json').write_text('{}')
    waits = []
    class Child:
        pid=123
        def poll(self):
            return None
        def wait(self,timeout=None):
            waits.append(timeout)
            return -15
    monkeypatch.setattr(replay,'namespace',lambda path:path)
    monkeypatch.setattr(replay,'read_json',lambda path:{'files':{},'simulator_files':{},'population':'learner','decisions':314})
    monkeypatch.setattr(replay,'runtime_identity',dict)
    monkeypatch.setattr(replay,'code_files',dict)
    monkeypatch.setattr(replay.subprocess,'Popen',lambda *a,**k:Child())
    ticks=iter((0.,2.,3.,4.))
    monkeypatch.setattr(replay.time,'monotonic',lambda:next(ticks))
    def race(*args):
        raise ProcessLookupError('exited between poll and signal')
    monkeypatch.setattr(replay.os,'killpg',race)
    args=SimpleNamespace(input=source,output=out,warm_loops=1,timeout=1,transfer='T0',
                         device='cuda:0',pair=1,position=1,profile=False)
    assert replay.run(args)==2
    import json
    receipt=json.loads((out/'receipt.json').read_text())
    assert waits and receipt['actual_child_waited'] is True
    assert receipt['failure']['type']=='TimeoutError'
    assert receipt['cleanup_failures'][0]['type']=='ProcessLookupError'


def test_replay_duplicate_pair_or_report_cannot_be_measured(monkeypatch,tmp_path):

    from benchmarks.collection_cpu import aggregate
    def summary(path):
        name=path.name
        return {'transfer':'T0' if name.endswith('c') else 'T1','pair':1,'position':1 if name.endswith('c') else 2,
                'population':{'input':'same'},'warm_seconds':1.}
    monkeypatch.setattr(aggregate,'replay_summary',summary)
    row={'model':'mlp','transfer':'T1','pair':1,'reports':[str(tmp_path/'c'),str(tmp_path/'v')],
         'input_manifest_sha256':'same'}
    report=aggregate.replay_ledger({'pairs':[dict(row) for _ in range(3)]})
    assert all(row['status']=='FAILED_OR_NOT_RUN' for row in report['ledger'])
    assert not any(row['valid'] for row in report['rows']['mlp']['T1'])


def test_actual_saved_request_cannot_relabel_another_result(tmp_path,monkeypatch):
    from types import SimpleNamespace

    from benchmarks.collection_cpu import aggregate
    from benchmarks.collection_cpu.common import immutable_json
    immutable_json(tmp_path/'request.json',{'options':{'parent':True},'run_id':'P'})
    immutable_json(tmp_path/'result.json',{'request':{'population':'performance','options':{'parent':False},'run_id':'C'},
                                        'parent':str(tmp_path/'parent')})
    monkeypatch.setattr(aggregate,'inference_helper',lambda name:SimpleNamespace(actual_summary=lambda path:{}))
    with pytest.raises(ValueError,match='embedded.*saved request'):
        aggregate.actual_summary(tmp_path)


def test_transfer_selector_rejects_repeated_physical_pair_even_with_three_rows():
    from benchmarks.collection_cpu.trials import select_transfer
    rows={model:{mode:[{'control':1.,'candidate':1.,'population':model,'valid':True,'pair':i,
        'reports':[f'{model}-{mode}-{i}-c',f'{model}-{mode}-{i}-v']} for i in (1,2,3)]
        for mode in ('reuse','T1','T2','T3')} for model in ('mlp','gru','transformer')}
    rows['mlp']['T1'][1]['reports']=rows['mlp']['T1'][0]['reports']
    with pytest.raises(ValueError,match='physical replay report identity repeated'):
        select_transfer(rows)


def test_replay_pair_order_and_unique_file_identity_are_enforced(tmp_path,monkeypatch):
    import os

    from benchmarks.collection_cpu import aggregate
    reports,rows={},[]
    for pair in (1,2,3):
        paths=[]
        order=['T0','T1'] if pair!=2 else ['T1','T0']
        for position,transfer in enumerate(order,1):
            path=tmp_path/f'{pair}-{position}'
            path.mkdir()
            (path/'receipt.json').write_text('{}')
            paths.append(str(path))
            reports[str(path)]={'transfer':transfer,'pair':pair,'position':position,'report':str(path),
                'population':{'input':'same'},'warm_seconds':1.}
        rows.append({'model':'mlp','transfer':'T1','pair':pair,'reports':paths,'input_manifest_sha256':'same'})
    monkeypatch.setattr(aggregate,'replay_summary',lambda path:reports[str(path)])
    assert all(row['status']=='MEASURED' for row in aggregate.replay_ledger({'pairs':rows})['ledger'])
    reports[rows[1]['reports'][0]]['transfer']='T0'
    reports[rows[1]['reports'][1]]['transfer']='T1'
    assert aggregate.replay_ledger({'pairs':rows})['ledger'][1]['status']=='FAILED_OR_NOT_RUN'
    source=tmp_path/'1-1'/'receipt.json'
    target=tmp_path/'3-1'/'receipt.json'
    target.unlink()
    os.link(source,target)
    ledger=aggregate.replay_ledger({'pairs':rows})['ledger']
    assert ledger[0]['status']==ledger[2]['status']=='FAILED_OR_NOT_RUN'


def test_replay_failed_wait_never_claims_reaped(tmp_path,monkeypatch):
    from types import SimpleNamespace

    from benchmarks.collection_cpu import replay
    source,out=tmp_path/'input',tmp_path/'out'
    source.mkdir()
    (source/'manifest.json').write_text('{}')
    waits=[]
    class Child:
        pid=123
        def poll(self):
            return None
        def wait(self,timeout=None):
            waits.append(timeout)
            raise OSError('wait unavailable')
    monkeypatch.setattr(replay,'namespace',lambda path:path)
    monkeypatch.setattr(replay,'read_json',lambda path:{'files':{},'simulator_files':{},'population':'learner','decisions':314})
    monkeypatch.setattr(replay,'runtime_identity',dict)
    monkeypatch.setattr(replay,'code_files',dict)
    monkeypatch.setattr(replay.subprocess,'Popen',lambda *a,**k:Child())
    ticks=iter((0.,2.,3.,4.))
    monkeypatch.setattr(replay.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(replay.os,'killpg',lambda *args:None)
    args=SimpleNamespace(input=source,output=out,warm_loops=1,timeout=1,transfer='T0',
                         device='cuda:0',pair=1,position=1,profile=False)
    assert replay.run(args)==2
    import json
    receipt=json.loads((out/'receipt.json').read_text())
    assert len(waits)==2 and receipt['actual_child_waited'] is False
    assert receipt['returncode'] is None and receipt['unconfirmed_child_pid']==123
    assert receipt['failure']['type']=='TimeoutError'


@pytest.mark.parametrize('path_type',[str,Path])
def test_freeze_build_normalizes_materialized_asset_paths(tmp_path,monkeypatch,path_type):
    import json
    from types import SimpleNamespace

    from benchmarks.collection_cpu import freeze
    from benchmarks.collection_cpu.common import Options, sha256
    from benchmarks.collection_cpu.trials import materialize
    from gf_common import card_instance
    from gf_training import native_backend
    assets={}
    for name in ('runner.py','measure.py','analyze.py','native.so','server','server.toml','spec.md','input.toml','options.json'):
        path=tmp_path/name
        path.write_text(name)
        assets[name]=path
    rows=materialize('12345678-1234-5678-1234-567812345678','diagnostic',[],'reuse')
    for row in rows:
        row['config']={'path':str(assets['input.toml']),'sha256':sha256(assets['input.toml'])}
        row['options_file']={'path':str(assets['options.json']),'sha256':sha256(assets['options.json'])}
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'namespace':'12345678-1234-5678-1234-567812345678',
        'phase':'diagnostic','selected':[],'transfer':'reuse','trials':rows}))
    server=SimpleNamespace(binary=path_type(assets['server']),config=path_type(assets['server.toml']),
        source_dir=path_type(tmp_path),simulator_revision='sim-revision')
    loaded=SimpleNamespace(inputs=[{'path':path_type(assets['input.toml']),'sha256':sha256(assets['input.toml'])}])
    runtime=SimpleNamespace(config=SimpleNamespace(server=server),sha256='materialized')
    runner=SimpleNamespace(materialize=lambda *args:(loaded,None,runtime))
    helper=SimpleNamespace(PRIOR=tmp_path,prior_runner=lambda:runner)
    monkeypatch.setattr(freeze,'code_files',dict)
    monkeypatch.setattr(freeze,'inference_helper',lambda name:helper)
    def git(command,**kwargs):
        value='' if 'status' in command else 'bots-revision' if command[2]==str(freeze.REPO) else 'sim-revision'
        return SimpleNamespace(stdout=value)
    monkeypatch.setattr(freeze.subprocess,'run',git)
    monkeypatch.setattr(native_backend,'binary_identity',lambda:{'extension_path':path_type(assets['native.so']),
        'extension_sha256':sha256(assets['native.so'])})
    monkeypatch.setattr(card_instance,'CORRECTED_REVIEWED_SPEC_HASHES',{'spec.md':sha256(assets['spec.md'])})
    result=freeze.build([plan])
    for name in ('server','server.toml','native.so','input.toml','spec.md'):
        assert result['files'][str(assets[name])]==sha256(assets[name])
    assert len(result['materializations'])==3
    assert result['materializations'][0]['options']==Options(diagnostic=True).identity()
    json.dumps(result,allow_nan=False)
