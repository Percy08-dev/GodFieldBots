"""Deterministic bounded trial ordering and selection, never performance execution."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import (
    MODELS,
    ROOT,
    Options,
    immutable_json,
    namespace,
    read_json,
    sha256,
)

PREFERENCE = ('P','T','H')


def positive(value: Any) -> float:
    if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive measured duration required')
    return float(value)


def select_transfer(rows: dict[str, Any]) -> dict[str, Any]:
    if set(rows) != set(MODELS):
        raise ValueError('complete all-model replay population required')
    totals: dict[str,float] = {}
    controls: dict[str,float] = {}
    invalid: dict[str,str] = {}
    populations: dict[str,Any] = {}
    report_identities: set[str] = set()
    for mode in ('reuse','T1','T2','T3'):
        times, bases = [], []
        for model in MODELS:
            pairs = rows[model].get(mode, [])
            if len(pairs) != 3:
                invalid[mode] = 'requires three original alternating pairs per model'
                continue
            if any(pair.get('valid') is not True for pair in pairs):
                invalid[mode] = 'mechanically invalid or missing replay pair'
                continue
            if {pair.get('pair') for pair in pairs} != {1,2,3}:
                raise ValueError('independent replay pair identities1,2,3 required')
            for pair in pairs:
                reports = pair.get('reports',[])
                if len(reports)!=2:
                    raise ValueError('two independent physical replay report identities required')
                for name in reports:
                    identity = str(Path(name).resolve())
                    if identity in report_identities:
                        raise ValueError('independent physical replay report identity repeated')
                    report_identities.add(identity)
            population = pairs[0]['population']
            if model in populations and populations[model] != population:
                raise ValueError('replay population differs across transfer options')
            populations[model] = population
            if any(pair['population'] != population for pair in pairs):
                raise ValueError('replay pair population differs')
            if any(pair.get('valid') is not True for pair in pairs):
                invalid[mode] = 'mechanically invalid replay pair'
                continue
            times.append(statistics.median(positive(pair['candidate']) for pair in pairs))
            bases.append(statistics.median(positive(pair['control']) for pair in pairs))
        if mode not in invalid:
            totals[mode], controls[mode] = sum(times), sum(bases)
    if not totals:
        return {'selected':None,'status':'NO_VALID_TRANSFER','invalid':invalid,'totals':{}}
    minimum = min(totals.values())
    selected = next(mode for mode in ('reuse','T1','T2','T3') if mode in totals and totals[mode] <= minimum*1.01)
    return {'selected':selected,'status':'SELECTED','totals':totals,'invalid':invalid,
            'control_totals':controls,'T0_faster':controls[selected] < totals[selected],
            'criterion':'sum per-model median full G0/G1 warm replay wall; within1% historical/T1/T2/T3 preference'}


def select_candidates(rows: dict[str, Any], valid: dict[str,bool]) -> dict[str, Any]:
    if set(rows) != set(MODELS) or set(valid) != set(PREFERENCE):
        raise ValueError('complete all-model exploratory and validity declaration required')
    required = {'C',*(arm for arm in PREFERENCE if valid[arm] is True)}
    if any(not required <= set(row) or set(row)-{'C','P','T','H'} for row in rows.values()):
        raise ValueError('complete exploratory block required; failed/missing trial cannot be imputed')
    denominator = sum(positive(rows[model]['C']) for model in MODELS)
    scores = {arm:sum(positive(rows[model]['C'])-positive(rows[model][arm]) for model in MODELS)
              for arm in PREFERENCE if valid[arm] is True}
    remaining = dict(scores)
    selected: list[str] = []
    ties = []
    while remaining and len(selected) < 2:
        best = max(remaining.values())
        tied = [arm for arm in PREFERENCE if arm in remaining and best-remaining[arm] <= denominator*.01]
        chosen = tied[0]
        ties.append(tied)
        selected.append(chosen)
        del remaining[chosen]
    return {'selected':selected,'scores':scores,'control_sum_seconds':denominator,
            'signed_percent':{arm:value/denominator*100 for arm,value in scores.items()},
            'unresolved_tie_groups':ties,'timing_win_claimed':False,
            'selection_bias':'one shared exploratory control; confirmations are two NEW independent pairs'}


def arm_options(arm: str, selected: list[str], transfer: str, *, diagnostic: bool = False) -> dict[str, Any]:
    arms = selected if arm == 'B' else [arm]
    return Options(parent='P' in arms, http='H' in arms,
                   transfer=transfer if 'T' in arms else 'T0', diagnostic=diagnostic).identity()


def materialize(seed: str, phase: str, selected: list[str], transfer: str) -> list[dict[str, Any]]:
    scope = uuid.UUID(seed)
    if str(scope) != seed or len(set(selected)) != len(selected) or set(selected)-set(PREFERENCE):
        raise ValueError('canonical namespace UUID and unique selected candidates required')
    if transfer not in ('reuse','T1','T2','T3'):
        raise ValueError('screened non-T0 transfer required')
    schedule: list[tuple[str,str,str,int]] = []
    if phase == 'diagnostic':
        schedule = [(model,'C','passive',0) for model in MODELS]
    elif phase == 'pilot':
        for model, arms in zip(MODELS, ('CPTH','HTPC','PCHT'), strict=True):
            schedule.extend((model,arm,'exploratory',0) for arm in arms)
    elif phase == 'confirm':
        if len(selected) > 2:
            raise ValueError('at most two selected singleton candidates')
        for candidate in selected:
            for model in MODELS:
                schedule.extend((model,arm,candidate,pair) for pair,arms in enumerate(((candidate,'C'),('C',candidate)),1) for arm in arms)
    elif phase == 'combo':
        if len(selected) != 2:
            return []
        for model in MODELS:
            schedule.extend((model,arm,'+'.join(selected),pair) for pair,arms in enumerate((('C','B'),('B','C'),('C','B')),1) for arm in arms)
    else:
        raise ValueError('unknown bounded protocol phase')
    rows = []
    for index,(model,arm,candidate,pair) in enumerate(schedule,1):
        key = f'{phase}/{index}/{model}/{arm}/{candidate}/{pair}'
        run_id = str(uuid.uuid5(scope,key+'/run'))
        rows.append({'index':index,'phase':phase,'model':model,'arm':arm,'candidate':candidate,'pair':pair,
                     'run_id':run_id,'context_id':str(uuid.uuid5(scope,key+'/context')),
                     'options':arm_options(arm,selected,transfer,diagnostic=phase=='diagnostic'),
                     'output':str(ROOT/'actual'/phase/f'{index:02}-{model}-{arm}-{run_id}'),
                     'status':'NOT_RUN'})
    return rows


def validate_plan(plan: Any) -> None:
    expected = materialize(plan['namespace'],plan['phase'],plan['selected'],plan['transfer'])
    if len(expected) != len(plan['trials']):
        raise ValueError('complete deterministic phase trial ledger required')
    for declared,actual in zip(expected,plan['trials'],strict=True):
        if any(actual.get(key) != value for key,value in declared.items()):
            raise ValueError('immutable deterministic trial ordering/identity/options/output differs')


def write_plan(args: Any) -> int:
    out = namespace(args.output)
    selected = []
    inputs = {}
    if args.selection:
        declaration = read_json(args.selection)
        selected = declaration['selected']
        inputs[str(args.selection.resolve())] = sha256(args.selection)
    if args.phase == 'combo' and (len(selected) != 2 or not args.compatible):
        raise ValueError('combination requires two mechanically compatible admitted candidates')
    if args.phase in ('confirm','combo') and not args.selection:
        raise ValueError('confirmation requires immutable exploratory selection')
    rows = materialize(args.namespace,args.phase,selected,args.transfer)
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        model_config = (args.configs/f"{row['model']}.toml").resolve(strict=True)
        row['config'] = {'path':str(model_config),'sha256':sha256(model_config)}
        option_path = out/f"{row['index']:02}-options.json"
        immutable_json(option_path,row['options'])
        row['options_file'] = {'path':str(option_path),'sha256':sha256(option_path)}
        row['command'] = [sys.executable,'benchmarks/collection_cpu/actual.py',
                          '--config',str(model_config),'--options',str(option_path),
                          '--run-id',row['run_id'],'--context-id',row['context_id'],
                          '--output',row['output']]
    immutable_json(out/'plan.json',{'schema_version':1,'phase':args.phase,'namespace':args.namespace,
        'selected':selected,'transfer':args.transfer,'selection_files':inputs,'trials':rows,
        'max_performance_runs':54,'max_diagnostic_runs':3,'selection_bias':'shared pilot + two fresh singleton confirmation pairs'})
    print(json.dumps({'status':'MATERIALIZED','output':str(out/'plan.json'),'trials':len(rows)}))
    return 0


def replay_plan(args: Any) -> int:
    output = namespace(args.output)
    pairs, profiles = [], []
    for model in MODELS:
        source = (args.inputs/model).resolve(strict=True)
        manifest_hash = sha256(source/'manifest.json')
        for transfer in ('reuse','T1','T2','T3'):
            for pair in (1,2,3):
                order = ('T0',transfer) if pair != 2 else (transfer,'T0')
                reports,commands = [], []
                for position,mode in enumerate(order,1):
                    report = ROOT/'replays'/f'{model}-{transfer}-pair{pair}-{position}-{mode}'
                    reports.append(str(report))
                    commands.append([sys.executable,'benchmarks/collection_cpu/replay.py','run',
                        '--input',str(source),'--output',str(report),'--transfer',mode,
                        '--pair',str(pair),'--position',str(position),'--device','cuda:0','--warm-loops','1'])
                pairs.append({'model':model,'transfer':transfer,'pair':pair,'reports':reports,
                              'commands':commands,'input_manifest_sha256':manifest_hash,'status':'NOT_RUN'})
        for transfer in ('T0','reuse','T1','T2','T3'):
            report = ROOT/'profiles'/f'{model}-{transfer}'
            profiles.append({'model':model,'transfer':transfer,'output':str(report),'status':'NOT_RUN',
                'command':[sys.executable,'benchmarks/collection_cpu/replay.py','run',
                    '--input',str(source),'--output',str(report),'--transfer',transfer,
                    '--pair','1','--position','1','--device','cuda:0','--profile']})
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'replay-plan.json',{'schema_version':1,'pairs':pairs,'separate_profiles':profiles,
                                            'qualification':'learner-only replay; profiles excluded from timing'})
    print(json.dumps({'output':str(output/'replay-plan.json'),'pairs':len(pairs),'profiles':len(profiles)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    screen = commands.add_parser('materialize-replay')
    screen.add_argument('--inputs',type=Path,required=True)
    screen.add_argument('--output',type=Path,required=True)
    mat = commands.add_parser('materialize')
    mat.add_argument('--namespace',required=True)
    mat.add_argument('--phase',choices=('diagnostic','pilot','confirm','combo'),required=True)
    mat.add_argument('--transfer',choices=('reuse','T1','T2','T3'),required=True)
    mat.add_argument('--configs',type=Path,required=True)
    mat.add_argument('--selection',type=Path)
    mat.add_argument('--compatible',action='store_true')
    mat.add_argument('--output',type=Path,required=True)
    for name in ('select-transfer','select-candidates'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--input',type=Path,required=True)
        cmd.add_argument('--output',type=Path,required=True)
        if name == 'select-candidates':
            cmd.add_argument('--validity',type=Path,required=True,help='root reviewed P/T/H mechanical validity declaration')
    args = parser.parse_args()
    if args.command == 'materialize-replay':
        return replay_plan(args)
    if args.command == 'materialize':
        return write_plan(args)
    data = read_json(args.input)
    result = select_transfer(data['rows']) if args.command == 'select-transfer' else select_candidates(data['rows'],read_json(args.validity)['valid'])
    output = namespace(args.output)
    immutable_json(output,{**result,'input_sha256':sha256(args.input),
                          'validity_sha256':sha256(args.validity) if args.command=='select-candidates' else None})
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
