"""No-GPU contracts for preparation, original observer, profile windows and summaries."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.learning_update.common import PRODUCER, immutable_json, read_json, sha256


def test_prepare_resolves_each_numeric_directory_and_preserves_repeated_basenames(tmp_path,monkeypatch):
    from benchmarks.learning_update import prepare
    from gf_training import hpo_lineage, hpo_rollout, hpo_source
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    inventory={'models':{}}
    expected={}
    for kind in ('mlp','gru','transformer'):
        parent=tmp_path/kind
        parent.mkdir()
        immutable_json(parent/'run.json',{'model':kind})
        generations=[]
        updates=[]
        for generation in (0,1):
            directory=parent/'artifacts'/f'{generation:06d}'
            directory.mkdir(parents=True)
            immutable_json(directory/'inference.json',{'schema_version':5,'producer':{'source_digest':PRODUCER}})
            immutable_json(directory/'resume.json',{'schema_version':5,'producer':{'source_digest':PRODUCER},
                'optimizer_steps':generation*2})
            generations.append({'generation':generation,'files':{str(p):sha256(p) for p in directory.iterdir()}})
            numeric=parent/'numeric'/f'batch-{generation:06d}'
            numeric.mkdir(parents=True)
            immutable_json(numeric/'rollout.json',{'generation':generation})
            blob=numeric/'rollout-000000.json'
            immutable_json(blob,{'unique':f'{kind}-{generation}'})
            expected[str(blob)]=sha256(blob)
            update=parent/f'update-{generation}.json'
            immutable_json(update,{'generation':generation})
            updates.append({'numeric':str(numeric/'rollout.json'),'numeric_sha256':sha256(numeric/'rollout.json'),
                'numeric_files':{blob.name:sha256(blob)},'update_record':str(update),'record_sha256':sha256(update),
                'selection':{},'snapshot_identity':'a'*64,'expected_source_generation':generation})
        inventory['models'][kind]={'parent':str(parent),'run_json_sha256':sha256(parent/'run.json'),
            'recipe':asdict(learner.training),'operating':{},'generation_inputs':generations,'updates':updates}
    source=tmp_path/'inventory.json'
    immutable_json(source,inventory)
    monkeypatch.setattr(prepare,'INVENTORY',source)
    monkeypatch.setattr(prepare,'code_files',dict)
    monkeypatch.setattr(prepare,'revision',lambda:'a'*40)
    state={'status':'incomplete','updates':2,'resume_identity':learner.resume_identity,
        'execution':{'learner':{'binding':asdict(learner.binding)}}}
    monkeypatch.setattr(hpo_source,'validate_hpo_parent',lambda *args,**kwargs:(state,()))
    monkeypatch.setattr(hpo_lineage,'hpo_state',lambda *args:(ACTIONS,learner.model,None))
    observed=[]
    class Rollout:
        collected_decisions=2
        def __len__(self):
            return 2
    def read(directory,*args,**kwargs):
        observed.append(directory)
        return Rollout()
    monkeypatch.setattr(hpo_rollout,'read_hpo_rollout',read)
    value=prepare.prepare(tmp_path/'prepared.json')
    pinned={name:digest for model in value['models'].values() for name,digest in model['files'].items() if name in expected}
    assert pinned==expected and len(observed)==6
    assert len(set(pinned.values()))==6
    from benchmarks.learning_update.common import check_files
    check_files(pinned)
    first=Path(next(iter(expected)))
    first.write_text('changed')
    with pytest.raises(ValueError,match='frozen file changed'):
        prepare.prepare(tmp_path/'must-not-exist.json')
    assert len(observed)==6 and not (tmp_path/'must-not-exist.json').exists()


def test_actual_request_reaches_observer_pre_gpu_hash_consumer(tmp_path,monkeypatch):
    from benchmarks.learning_update import actual
    from benchmarks.learning_update.common import prior_runner
    from gf_cli import hpo, hpo_config, hpo_storage
    from gf_training import hpo_gpu_lease
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(actual,'REPO',tmp_path)
    config=tmp_path/'config.toml'
    config.write_text('unqualified no-GPU fixture')
    @dataclass
    class Config:
        storage: Any
        operating: Any
        run_wall_seconds: int=600
        model: Any=None
        training: Any=None
    loaded=SimpleNamespace(config=Config(
        SimpleNamespace(parent_root=str(tmp_path/'runs/learning-update-optimization-v1/parents')),
        SimpleNamespace(managed_cuda_bytes=12*1024**3,max_participant_workers=4)),
        sha256='a'*64,effective={},inputs=({'path':str(config),'sha256':sha256(config)},))
    settings=SimpleNamespace(model=SimpleNamespace(model_id='mlp'),training=None,identity='a'*64)
    runtime=SimpleNamespace(sha256='a'*64,effective={})
    observed=[]
    class BeforeGpu(Exception):
        pass
    def never_admit(*args,**kwargs):
        observed.append('original-config-consumer-passed-before-GPU')
        raise BeforeGpu('bounded no-GPU original preflight consumer reached')
    # Exercise the ORIGINAL historical execute through its request hash guard;
    # only unrelated external storage/conditions/admission are bounded stubs.
    runner=prior_runner()
    monkeypatch.setattr(runner,'materialize',lambda *args:(loaded,settings,runtime))
    monkeypatch.setattr(runner,'production_patches',lambda *args:[])
    monkeypatch.setattr(actual,'prior_runner',lambda:runner)
    monkeypatch.setattr(actual,'protocol',lambda *args:None)
    monkeypatch.setattr(actual,'verify_freeze',lambda *args:{'files':{}})
    monkeypatch.setattr(hpo_config,'joint_resource_admission',lambda *args,**kwargs:{})
    monkeypatch.setattr(hpo_config,'require_hpo_runtime_execution',lambda *args:None)
    monkeypatch.setattr(hpo,'_conditions',lambda *args:{})
    monkeypatch.setattr(hpo,'_originals',lambda *args:tmp_path/'originals')
    monkeypatch.setattr(hpo_storage,'check_storage',lambda *args,**kwargs:{})
    monkeypatch.setattr(hpo_gpu_lease,'admit_gpu_job',never_admit)
    output=tmp_path/'report'
    monkeypatch.setattr('sys.argv',['actual','--config',str(config),'--output',str(output),
        '--run-id','00000000-0000-4000-8000-000000000017',
        '--context-id','00000000-0000-4000-8000-000000000018',
        '--variant','baseline','--freeze',str(tmp_path/'freeze.json'),'--execute'])
    assert actual.main()==2
    assert observed==['original-config-consumer-passed-before-GPU']
    result=read_json(output/'result.json')
    assert result['failure']['type']=='BeforeGpu' and result['train_calls']==0
    assert result['request']['config_file_sha256']==sha256(config)


def _write_population(directory: Path,models=('mlp',)) -> None:
    directory.mkdir()
    from benchmarks.learning_update.summarize import ORDER
    immutable_json(directory/'request.json',{'schema_version':1,'kind':'fixed-update-core-comparison',
        'models':list(models),'repeats':3,'ordering':'forward/reverse/forward'})
    count=0
    for kind in models:
        for batch in (0,1):
            for repeat in range(3):
                for position,variant in enumerate(ORDER[:4] if kind=='mlp' else ORDER):
                    count+=1
                    immutable_json(directory/f'{kind}-b{batch}-r{repeat}-{variant}.json',{
                        'model':kind,'batch':batch,'repeat':repeat,'variant':variant,
                        'wall_seconds':10.-position,'process_cpu_seconds':1.,'cuda_event_ms':100.,
                        'peak_allocated_bytes':100,'peak_reserved_bytes':200,'metrics':{}})
    immutable_json(directory/'result.json',{'status':'complete','measured_updates':count,
        'expected_updates':count,'quality_eligible':False})


def test_summary_requires_actual_baseline_and_declared_complete_population(tmp_path):
    from benchmarks.learning_update.summarize import core_summary
    directory=tmp_path/'partial'
    _write_population(directory)
    for path in directory.glob('*-baseline.json'):
        path.unlink()
    with pytest.raises(ValueError,match='population|baseline'):
        core_summary(directory)


@pytest.mark.parametrize('defect',['no_request','no_completion','failed','failure_file','missing','duplicate','foreign','count'])
def test_summary_rejects_incomplete_failed_duplicate_or_foreign_rows(tmp_path,defect):
    from benchmarks.learning_update.summarize import core_summary
    directory=tmp_path/'population'
    _write_population(directory)
    if defect in ('no_request','no_completion'):
        (directory/('request.json' if defect=='no_request' else 'result.json')).unlink()
    elif defect in ('failed','count'):
        result=read_json(directory/'result.json')
        result['status' if defect=='failed' else 'measured_updates']='failed' if defect=='failed' else 84
        (directory/'result.json').write_text(json.dumps(result))
    elif defect=='failure_file':
        immutable_json(directory/'failure.json',{'status':'failed'})
    else:
        path=next(directory.glob('*-baseline.json'))
        row=read_json(path)
        if defect=='missing':
            path.unlink()
        elif defect=='duplicate':
            immutable_json(directory/'duplicate-b0-r0-baseline.json',row)
        else:
            row['model']='gru'
            path.write_text(json.dumps(row))
    with pytest.raises((ValueError,FileNotFoundError)):
        core_summary(directory)


@pytest.mark.parametrize('models,expected,complete',[(('mlp',),24,False),(('mlp','gru','transformer'),84,True)])
def test_summary_reports_explicit_limited_population_and_baseline_effects(tmp_path,models,expected,complete):
    from benchmarks.learning_update.summarize import core_summary
    directory=tmp_path/'population'
    _write_population(directory,models)
    result=core_summary(directory)
    assert result['population']['measured_updates']==expected
    assert result['population']['models']==list(models)
    assert result['population']['complete_84_update_protocol']==complete
    metrics=next(row for row in result['effects'] if row['variant']=='metrics')
    assert metrics['baseline_seconds_saved']==metrics['incremental_seconds_saved']==1.


def test_profile_group_window_includes_own_packing_and_excludes_next(tmp_path,monkeypatch):
    from dataclasses import replace

    from gf_training import hpo_update_execution
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    from gf_training.hpo_update_diagnostics import ProfileWindow
    from tests.test_hpo_numeric import learner_for, rollout_for
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1,minibatch_size=2))
    rollout=rollout_for(learner,count=2)
    learner._training_execution=TrainingExecutionSpec(variant='packed')
    learner._training_reservation_bytes=learner.binding.managed_budget_bytes
    window=ProfileWindow(learner,tmp_path,0,1,False)
    observed=[]
    pack=hpo_update_execution.pack_window
    def capture(*args,**kwargs):
        observed.append(('packing',window.active,window.seen))
        return pack(*args,**kwargs)
    monkeypatch.setattr(hpo_update_execution,'pack_window',capture)
    with window.patches():
        learner._update_numeric(rollout,{},0.,learner.network)
        window.finish()
    assert observed==[('packing',True,1),('packing',False,2)]
    events=read_json(tmp_path/'trace.json')['traceEvents']
    assert sum(event.get('name')=='input packing/HtoD' for event in events)==1


def test_trace_counts_report_observed_categories_and_unavailable_cuda(tmp_path):
    from gf_training.hpo_update_diagnostics import trace_counts
    path=tmp_path/'trace.json'
    immutable_json(path,{'traceEvents':[{'cat':'kernel','name':'a'},
        {'cat':'gpu_memcpy','name':'Memcpy HtoD'},{'cat':'gpu_memcpy','name':'Memcpy DtoH'},
        {'cat':'cuda_runtime','name':'cudaLaunchKernel'},
        {'cat':'cuda_runtime','name':'cudaMemcpyAsync'},
        {'cat':'cuda_runtime','name':'cudaEventSynchronize'},
        {'cat':'cuda_driver','name':'cuCtxSynchronize'}]})
    result=trace_counts(path)
    assert result['kernel_count']==1 and result['device_copy_count']==2
    assert result['runtime_kernel_launch_count']==1 and result['runtime_copy_count']==1
    assert result['runtime_sync_count']==1 and result['driver_sync_count']==1
    assert not result['missing_cuda_activity']
    immutable_json(tmp_path/'cpu.json',{'traceEvents':[{'cat':'cpu_op','name':'aten::linear'}]})
    result=trace_counts(tmp_path/'cpu.json')
    assert result['missing_cuda_activity']
    assert result['kernel_count'] is None and result['device_copy_count'] is None
    assert result['runtime_sync_count'] is None


def test_streamed_trace_crosses_read_chunks_and_rejects_oversized_records(tmp_path):
    from gf_training.hpo_update_diagnostics import trace_counts
    path=tmp_path/'many-events.json'
    immutable_json(path,{'header':{},'traceEvents':[{'cat':'cpu_op','name':'aten::linear'}]*10000})
    assert path.stat().st_size>65536
    assert trace_counts(path)['event_count']==10000
    oversized=tmp_path/'oversized-event.json'
    immutable_json(oversized,{'traceEvents':[{'cat':'cpu_op','name':'x'*(2*1024**2)}]})
    with pytest.raises(ValueError,match='counting bound'):
        trace_counts(oversized)


def test_baseline_profile_retains_original_endpoint_boundary(tmp_path,monkeypatch):
    from dataclasses import replace

    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    from gf_training.hpo_update_diagnostics import ProfileWindow
    from tests.test_hpo_numeric import learner_for, rollout_for
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1,minibatch_size=2))
    rollout=rollout_for(learner,count=2)
    learner._training_execution=TrainingExecutionSpec(variant='baseline')
    window=ProfileWindow(learner,tmp_path,0,1,False)
    observed=[]
    forward=learner.network._forward_calculate
    def capture(*args,**kwargs):
        observed.append((window.active,window.seen))
        return forward(*args,**kwargs)
    monkeypatch.setattr(learner.network,'_forward_calculate',capture)
    with window.patches():
        learner._update_numeric(rollout,{},0.,learner.network)
        window.finish()
    assert observed==[(True,1),(False,2)]
    assert window.summary['captured_groups']==1
