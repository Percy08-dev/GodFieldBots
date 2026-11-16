"""Post-timing strict complete-original Source/numeric/receipt verification."""
from __future__ import annotations

import argparse
import stat
import time
from pathlib import Path
from typing import Any

from benchmarks.learning_update.common import immutable_json, read_json, sha256


def snapshot(directory: Path) -> dict[str,Any]:
    values={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('original audit links unsupported')
        if not path.is_file():
            continue
        info=path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('original audit regular files required')
        values[str(path)]=[info.st_dev,info.st_ino,info.st_mode,info.st_size,
            info.st_mtime_ns,info.st_ctime_ns,sha256(path)]
    return values


def audit_trial(row: Any) -> dict[str,Any]:
    from gf_cli.hpo_evaluation import verify_gpu_termination
    from gf_training.hpo_source import validate_hpo_parent
    parent,original=Path(row['parent']),Path(row['original'])
    before=snapshot(parent)|snapshot(original)
    started=time.monotonic()
    state,records=validate_hpo_parent(parent,require_complete=False,
        max_owned_bytes=2*1024**3,deadline=time.monotonic()+180)
    if state['status']!='incomplete' or state['updates']!=2 or len(records)!=3 or state['stop_reason']!='update_cap':
        raise ValueError('audited original must be healthy incomplete/update_cap with2updates3generations')
    result=read_json(Path(row['report'])/'result.json')
    if not result['probe_expectation_met'] or result['train_calls']!=1 or result.get('failure'):
        raise ValueError('actual timed invocation failed')
    verify_gpu_termination(result['gpu_termination'],campaign_id=row['context_id'],
        operation_id=row['run_id'],run_id=row['run_id'],evaluation_id=None)
    after=snapshot(parent)|snapshot(original)
    if before!=after:
        raise ValueError('strict audit changed original byte/stat identity')
    return {'index':row['index'],'model':row['model'],'route':row['route'],
        'status':'PASS_FULL_ORIGINAL_SOURCE_NUMERIC_AND_RECEIPT','updates':2,'generations':3,
        'original_identity_unchanged':True,'original_files':len(before),'seconds':time.monotonic()-started}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=read_json(args.plan)
    if len(plan['trials'])!=18:
        raise ValueError('complete18run population required')
    rows=[]
    for row in plan['trials']:
        rows.append(audit_trial(row))
    immutable_json(args.output,{'schema_version':1,'plan_sha256':sha256(args.plan),'audits':rows,
        'status':'PASS_ALL18_STRICT_ORIGINAL_SOURCE_NUMERIC_RECEIPTS'})


if __name__=='__main__':
    main()
