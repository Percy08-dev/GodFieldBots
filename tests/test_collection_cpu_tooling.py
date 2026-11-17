"""Protocol population, deterministic selection and immutable launch checks."""
from copy import deepcopy

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
                         'population':'same','valid':True} for _ in range(3)]
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
    rows = {model:{mode:[{'control':10.,'candidate':9.,'population':model,'valid':True} for _ in range(3)]
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
            'options':row['options'],'config_file_sha256':'config'}
    plan = {'namespace':'12345678-1234-5678-1234-567812345678','phase':'pilot','selected':[],
            'transfer':'T1','trials':rows}
    def summary(path,**kwargs):
        condition = 'different' if str(path)==rows[1]['output'] else 'same'
        return {'collection_seconds':10.,'matched_conditions':{'condition':condition}}
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
