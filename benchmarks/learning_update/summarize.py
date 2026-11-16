"""Separate cumulative/incremental core effects from inclusive whole-run phases."""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

from benchmarks.learning_update.common import immutable_json, read_json

ORDER=('baseline','metrics','packed','microbatch','shared-prefix')


def core_summary(directory: Path) -> dict[str,Any]:
    request=read_json(directory/'request.json')
    result=read_json(directory/'result.json')
    models=request.get('models')
    if (type(request.get('schema_version')) is not int or request.get('schema_version')!=1 or request.get('kind')!='fixed-update-core-comparison'
            or request.get('repeats')!=3 or request.get('ordering')!='forward/reverse/forward'
            or type(models) is not list or not models
            or any(kind not in ('mlp','gru','transformer') for kind in models)
            or len(set(models))!=len(models)):
        raise ValueError('valid declared fixed core population required')
    identities={(kind,batch,variant,repeat) for kind in models for batch in (0,1)
        for variant in (ORDER[:4] if kind=='mlp' else ORDER) for repeat in range(3)}
    expected=len(identities)
    if (result.get('status')!='complete' or type(result.get('measured_updates')) is not int
            or type(result.get('expected_updates')) is not int or result.get('measured_updates')!=expected
            or result.get('expected_updates')!=expected or result.get('quality_eligible') is not False
            or (directory/'failure.json').exists()):
        raise ValueError('successful completed declared core population required')
    rows={}
    for path in directory.glob('*.json'):
        if path.name in ('request.json','result.json'):
            continue
        row=read_json(path)
        identity=(row.get('model'),row.get('batch'),row.get('variant'),row.get('repeat'))
        if (type(row.get('batch')) is not int or type(row.get('repeat')) is not int
                or identity not in identities or identity in rows):
            raise ValueError('foreign or duplicate core population row')
        kind,batch,variant,repeat=identity
        if path.name!=f'{kind}-b{batch}-r{repeat}-{variant}.json':
            raise ValueError('core population row filename/identity mismatch')
        for name in ('wall_seconds','process_cpu_seconds','cuda_event_ms',
                'peak_allocated_bytes','peak_reserved_bytes'):
            value=row.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('core population requires finite nonnegative measurements')
        if row['wall_seconds']==0:
            raise ValueError('core baseline/variant duration must be positive')
        rows[identity]=row
    if set(rows)!=identities:
        raise ValueError('incomplete core population or missing actual baseline rows')
    summary=[]
    for kind in models:
        for batch in (0,1):
            baseline=[rows[kind,batch,'baseline',repeat] for repeat in range(3)]
            base=statistics.median(row['wall_seconds'] for row in baseline)
            previous=None
            for variant in (ORDER[:4] if kind=='mlp' else ORDER):
                selected=[rows[kind,batch,variant,repeat] for repeat in range(3)]
                wall=statistics.median(r['wall_seconds'] for r in selected)
                summary.append({'model':kind,'batch':batch,'variant':variant,'median_wall_seconds':wall,
                    'baseline_seconds_saved':base-wall,'baseline_percent_saved':100*(base-wall)/base,
                    'incremental_seconds_saved':None if previous is None else previous-wall,
                    'incremental_percent_saved':None if previous is None else 100*(previous-wall)/previous,
                    'median_process_cpu_seconds':statistics.median(r['process_cpu_seconds'] for r in selected),
                    'median_cuda_event_ms':statistics.median(r['cuda_event_ms'] for r in selected),
                    'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in selected),
                    'max_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in selected),
                    'diagnostics':[r.get('diagnostics') for r in selected],
                    'metrics':[r['metrics'] for r in selected]})
                previous=wall
    return {'population':{'models':models,'measured_updates':expected,
        'complete_84_update_protocol':expected==84 and set(models)=={'mlp','gru','transformer'},
        'status':'complete','quality_eligible':False},'effects':summary}


def actual_summary(plan: Any) -> dict[str,Any]:
    rows=[]
    for trial in plan['trials']:
        result=read_json(Path(trial['report'])/'result.json')
        if not result.get('probe_expectation_met') or result.get('failure'):
            rows.append({**trial,'status':'FAILED','failure':result.get('failure')})
            continue
        phases: dict[str,float]={}
        with (Path(trial['report'])/'timings.jsonl').open() as stream:
            import json
            for line in stream:
                row=json.loads(line)
                label=row['label']
                phases[label]=phases.get(label,0.)+row['elapsed_seconds']
        rows.append({**trial,'status':'healthy-incomplete/update_cap',
            'helper_elapsed_seconds':result['helper_elapsed_seconds'],
            'training_seconds':result['training_finished_monotonic']-result['training_started_monotonic'],
            'inclusive_phase_seconds':phases,
            'parent_phase_boundaries':result['measurement'].get('parent_phases'),
            'measurement_excluded':result['measurement'].get('exclude_from_performance'),
            'learner_allocator':result['diagnostics'].get('learner_allocator')})
    comparisons=[]
    for kind in ('mlp','gru','transformer'):
        pairs=[]
        for pair in range(3):
            selected=[r for r in rows if r['model']==kind and r['pair']==pair]
            if len(selected)!=2 or any(r['status']=='FAILED' for r in selected):
                pairs.append({'pair':pair,'status':'INELIGIBLE'})
                continue
            baseline=next(r for r in selected if r['route']=='baseline')
            candidate=next(r for r in selected if r['route']=='candidate')
            base,cand=baseline['training_seconds'],candidate['training_seconds']
            pairs.append({'pair':pair,'status':'DESCRIPTIVE_UNTIL_STRICT_AUDITS',
                'training_seconds_saved':base-cand,'training_percent_saved':100*(base-cand)/base,
                'helper_seconds_saved':baseline['helper_elapsed_seconds']-candidate['helper_elapsed_seconds']})
        comparisons.append({'model':kind,'pairs':pairs})
    return {'runs':rows,'comparisons':comparisons,
        'phase_scope':'inclusive named walls may overlap; never subtract unrelated clocks to infer I/O wait'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--actual-plan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={'schema_version':1,'core':core_summary(args.core),'quality_eligible':False,
        'recommendation':'Retain production defaults pending complete independent review, physical population and strict audits.'}
    if args.actual_plan is not None:
        result['actual']=actual_summary(read_json(args.actual_plan))
    immutable_json(args.output,result)


if __name__=='__main__':
    main()
