"""After ALL timing: original full Source/numeric, exact receipts and SHA/stat."""
from __future__ import annotations

import argparse
import stat
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import immutable_json, namespace, read_json, sha256
from benchmarks.collection_cpu.trials import validate_plan


def snapshot(directory: Path) -> dict[str,Any]:
    rows = {}
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('original audit directory required')
    for path in sorted(directory.rglob('*')):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('original audit symlinks unsupported')
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('original audit regular files required')
        rows[str(path)] = [info.st_dev,info.st_ino,info.st_mode,info.st_size,
                           info.st_mtime_ns,info.st_ctime_ns,sha256(path)]
    return rows


def audit(row: Any) -> dict[str,Any]:
    from gf_cli.hpo_evaluation import verify_gpu_termination
    from gf_training.hpo_source import validate_hpo_parent
    report = Path(row['output'])
    result,request = read_json(report/'result.json'),read_json(report/'request.json')
    if (result['exit_code'] or not result['probe_expectation_met'] or result['train_calls']!=1
            or result.get('failure') or request['run_id']!=row['run_id'] or request['context_id']!=row['context_id']
            or result['request']!=request):
        raise ValueError('actual invocation identity/failure differs')
    parent = Path(result['parent'])
    state_before = read_json(parent/'run.json')
    original = Path(state_before['effective_config']['hpo_spec']['storage']['originals_root'])/row['run_id']
    before = snapshot(parent)|snapshot(original)
    started = time.monotonic()
    state,records = validate_hpo_parent(parent,require_complete=False,
        max_owned_bytes=2*1024**3,deadline=time.monotonic()+180)
    if (state['status']!='incomplete' or state['stop_reason']!='update_cap'
            or state['updates']!=2 or len(records)!=3):
        raise ValueError('healthy original incomplete/update_cap two-update parent required')
    receipt = read_json(report/'actual-termination.json')
    if receipt != result['gpu_termination']:
        raise ValueError('exact original invocation termination receipt differs')
    verify_gpu_termination(receipt,campaign_id=row['context_id'],operation_id=row['run_id'],
                           run_id=row['run_id'],evaluation_id=None)
    after = snapshot(parent)|snapshot(original)
    if before!=after:
        raise ValueError('original SHA/stat identity changed during full audit')
    return {'phase':row['phase'],'index':row['index'],'model':row['model'],'run_id':row['run_id'],
            'status':'PASS_FULL_ORIGINAL_SOURCE_NUMERIC_EXACT_RECEIPT','original_files':len(before),
            'original_identity_unchanged':True,'identity_before':before,'seconds':time.monotonic()-started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all-timing-finished',action='store_true',required=True)
    args = parser.parse_args()
    plans = [read_json(path) for path in args.plan]
    for plan in plans:
        validate_plan(plan)
    rows = [row for plan in plans for row in plan['trials']]
    if len(rows)>57 or len({row['run_id'] for row in rows})!=len(rows):
        raise ValueError('bounded unique actual population required')
    results = []
    for row in rows:
        try:
            results.append(audit(row))
        except (OSError,ValueError,RuntimeError,TimeoutError,KeyError) as error:
            results.append({'phase':row['phase'],'index':row['index'],'run_id':row['run_id'],
                            'status':'FAIL_OR_NOT_RUN','error':{'type':type(error).__name__,'message':str(error)}})
    passed = all(row['status']=='PASS_FULL_ORIGINAL_SOURCE_NUMERIC_EXACT_RECEIPT' for row in results)
    output = namespace(args.output)
    immutable_json(output,{'schema_version':1,'plans':{str(p):sha256(p) for p in args.plan},
        'status':'PASS' if passed else 'FAIL_OR_NOT_RUN','audits':results})
    print('PASS' if passed else 'FAIL_OR_NOT_RUN')
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
