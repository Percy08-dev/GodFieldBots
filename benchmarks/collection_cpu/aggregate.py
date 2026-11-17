"""Read-only complete ledgers, matched timing populations and honest failures."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import (
    MODELS,
    immutable_json,
    inference_helper,
    namespace,
    read_json,
    sha256,
)
from benchmarks.collection_cpu.trials import positive, validate_plan


def replay_summary(path: Path) -> dict[str, Any]:
    request,result,receipt = (read_json(path/name) for name in ('request.json','result.json','receipt.json'))
    if result['device'] != 'cuda:0':
        raise ValueError('approved transfer selection requires actual CUDA replay')
    if receipt['returncode'] or receipt['failure'] or not receipt['actual_child_waited'] or result['profiled']:
        raise ValueError('successful unprofiled physically reaped replay required')
    rows = result['loops']
    warm = [row for row in rows if row['population'] == 'warm']
    if len(warm) < 1 or len(rows) != len(warm)+1 or rows[0]['population'] != 'warmup-first':
        raise ValueError('separate first/warmup and >=1 full warm loop required')
    if any(row['decisions'] != request['decisions'] for row in rows):
        raise ValueError('full learner population differs')
    population = {'input':request['input_manifest_sha256'],'device':request['device'],'code':request['code'],
                  'decisions':[r['decisions'] for r in rows],'candidates':[r['candidates'] for r in rows],
                  'torch':result['torch'],'cuda':result['cuda_runtime'],'threads':result['threads']}
    return {'warm_seconds':statistics.median(positive(row['host_wall_seconds']) for row in warm),
            'population':population,'valid':True,'transfer':request['transfer'],
            'pair':request['pair'],'position':request['position'],'report':str(path)}


def replay_ledger(plan: Any) -> dict[str, Any]:
    grouped: dict[str,Any] = {model:{mode:[] for mode in ('reuse','T1','T2','T3')} for model in MODELS}
    ledger = []
    for row in plan['pairs']:
        entry = dict(row)
        try:
            reports = [replay_summary(Path(path)) for path in row['reports']]
            control = next(r for r in reports if r['transfer']=='T0')
            candidate = next(r for r in reports if r['transfer']==row['transfer'])
            if (control['population'] != candidate['population'] or control['pair'] != row['pair']
                    or candidate['pair'] != row['pair'] or [r['position'] for r in reports] != [1,2]
                    or control['population']['input'] != row['input_manifest_sha256']):
                raise ValueError('original alternating replay pair population differs')
            grouped[row['model']][row['transfer']].append({'control':control['warm_seconds'],
                'candidate':candidate['warm_seconds'],'population':control['population'],'valid':True})
            entry.update(status='MEASURED',control=control,candidate=candidate)
        except (OSError,ValueError,KeyError,StopIteration,TypeError) as error:
            grouped[row['model']][row['transfer']].append({'valid':False,'population':None})
            entry.update(status='FAILED_OR_NOT_RUN',error={'type':type(error).__name__,'message':str(error)})
        ledger.append(entry)
    return {'rows':grouped,'ledger':ledger,'scope':'T selection only; P/H never ranked from actor replay'}


def actual_summary(path: Path, *, diagnostic: bool = False) -> dict[str, Any]:
    original = inference_helper('aggregate')
    summary = original.actual_summary(path)
    result = read_json(path/'result.json')
    request = result['request']
    if request.get('population') != ('passive-diagnostic' if diagnostic else 'performance'):
        raise ValueError('passive/profiler population must remain separate from performance')
    parent = Path(result['parent'])
    updates = [read_json(parent/f'updates/{i:06}.json') for i in (1,2)]
    primary = sum(positive(update['collection_seconds']) for update in updates)
    rows = [json.loads(line) for line in (path/'timings.jsonl').read_text().splitlines()]
    intervals = [row['elapsed_seconds'] for row in rows if row['label']=='collector_callback']
    if len(intervals) != 2:
        raise ValueError('original two coordinator collection intervals required')
    return {**summary,'collection_seconds':primary,'coordinator_collection_seconds':sum(intervals),
            'collection_batches':[{'collection_seconds':u['collection_seconds'],'decisions':u['collected_decisions'],
                'used_decisions':u['used_decisions'],'numeric_sha256':u['numeric_sha256'],
                'source_proof_sha256':u.get('source_proof_sha256'),'actual_games':u['actual_games']} for u in updates],
            'games':[row['details'] for row in rows if row['label']=='game_operations'],
            'memory':result['diagnostics'],'trajectory_equality':'not assumed; decision/candidate/cutoff differences retained',
            'request_sha256':sha256(path/'request.json'),'result_sha256':sha256(path/'result.json')}


def actual_ledger(plans: list[Any]) -> dict[str, Any]:
    ledger = []
    seen = set()
    for plan in plans:
        validate_plan(plan)
        for row in plan['trials']:
            if row['run_id'] in seen:
                raise ValueError('actual independent trial identity repeated')
            seen.add(row['run_id'])
            entry = dict(row)
            try:
                summary = actual_summary(Path(row['output']),diagnostic=plan['phase']=='diagnostic')
                request = read_json(Path(row['output'])/'request.json')
                if (request['run_id'] != row['run_id'] or request['context_id'] != row['context_id']
                        or request['options'] != row['options'] or request['config_file_sha256'] != row['config']['sha256']):
                    raise ValueError('actual immutable trial identity/options/config differ')
                entry.update(status='MEASURED',summary=summary)
            except (OSError,ValueError,KeyError,TypeError) as error:
                entry.update(status='FAILED_OR_NOT_RUN',error={'type':type(error).__name__,'message':str(error)})
            ledger.append(entry)
    for model in MODELS:
        pilots = [row for row in ledger if row['phase']=='pilot' and row['model']==model and row['status']=='MEASURED']
        controls = [row for row in pilots if row['arm']=='C']
        if len(controls)==1:
            for row in pilots:
                try:
                    inference_helper('aggregate').require_matched_actual(controls[0]['summary'],row['summary'])
                except ValueError as error:
                    row.update(status='FAILED_OR_NOT_RUN',error={'type':type(error).__name__,'message':str(error)})
    pairs = []
    for phase in ('confirm','combo'):
        candidates = sorted({row['candidate'] for row in ledger if row['phase']==phase})
        for candidate in candidates:
            for model in MODELS:
                chosen = [row for row in ledger if row['phase']==phase and row['model']==model and row['candidate']==candidate]
                for number in sorted({row['pair'] for row in chosen}):
                    pair = [row for row in chosen if row['pair']==number]
                    value: dict[str,Any] = {'phase':phase,'candidate':candidate,'model':model,'pair':number,'status':'FAILED_OR_NOT_RUN'}
                    if len(pair)==2 and all(row['status']=='MEASURED' for row in pair):
                        base = next(row for row in pair if row['arm']=='C')['summary']
                        variant = next(row for row in pair if row['arm']!='C')['summary']
                        try:
                            inference_helper('aggregate').require_matched_actual(base,variant)
                        except ValueError as error:
                            value['error'] = {'type':type(error).__name__,'message':str(error)}
                            pairs.append(value)
                            continue
                        delta = base['collection_seconds']-variant['collection_seconds']
                        value.update(status='MEASURED',saving_seconds=delta,percent=100*delta/base['collection_seconds'],
                            control=base,variant=variant)
                    pairs.append(value)
    pilot: dict[str,Any] = {model:{} for model in MODELS}
    for row in ledger:
        if row['phase']=='pilot' and row['status']=='MEASURED':
            pilot[row['model']][row['arm']] = row['summary']['collection_seconds']
    medians = []
    for phase,candidate,model in sorted({(p['phase'],p['candidate'],p['model']) for p in pairs}):
        chosen = [p for p in pairs if (p['phase'],p['candidate'],p['model'])==(phase,candidate,model)]
        expected = 2 if phase=='confirm' else 3
        measured = [p for p in chosen if p['status']=='MEASURED']
        medians.append({'phase':phase,'candidate':candidate,'model':model,'expected_pairs':expected,
            'measured_pairs':len(measured),'median_saving_seconds':statistics.median(p['saving_seconds'] for p in measured) if len(measured)==expected else None,
            'median_percent':statistics.median(p['percent'] for p in measured) if len(measured)==expected else None,
            'consistent_positive':len(measured)==expected and all(p['saving_seconds']>0 for p in measured)})
    return {'ledger':ledger,'pairs':pairs,'medians':medians,'rows':pilot,
            'selection_bias':'exploratory shared control separate from TWO new independent singleton confirmations',
            'performance_runs':sum(row['phase']!='diagnostic' for row in ledger),
            'diagnostic_runs':sum(row['phase']=='diagnostic' for row in ledger),
            'qualification':'timing evidence only; no quality claim or automatic default adoption'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('actual','replay'),required=True)
    parser.add_argument('--plan',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    plans = [read_json(path) for path in args.plan]
    result = actual_ledger(plans) if args.kind=='actual' else replay_ledger(plans[0])
    if args.kind=='actual' and (result['performance_runs']>54 or result['diagnostic_runs']>3):
        raise ValueError('bounded actual protocol population exceeded')
    output = namespace(args.output)
    immutable_json(output,{'schema_version':1,'plans':{str(p):sha256(p) for p in args.plan},**result})
    print(json.dumps({'output':str(args.output),'status':'AGGREGATED_WITH_EXPLICIT_MISSING_OR_FAILED'}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
