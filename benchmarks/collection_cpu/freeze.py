"""Root-invoked immutable source/config/options/native/Simulator freeze creation.

This records evidence; only independent review and root's execution decision
permit subsequent actual invocations. It never starts training or a listener.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import (
    REPO,
    check_files,
    code_files,
    immutable_json,
    inference_helper,
    namespace,
    read_json,
    sha256,
)
from benchmarks.collection_cpu.trials import validate_plan


def build(plans: list[Path]) -> dict[str,Any]:
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_training.native_backend import binary_identity
    files = code_files()
    # Refuse a code freeze pretending to name a reviewed committed source tree.
    result = subprocess.run(['git','-C',str(REPO),'status','--porcelain','--',*files],
                            check=True,capture_output=True,text=True)
    if result.stdout.strip():
        raise ValueError('commit reviewed application/benchmark source before root freeze')
    revision = subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],check=True,
                              capture_output=True,text=True).stdout.strip()
    original = inference_helper('actual')
    runner = original.prior_runner()
    for name in ('runner.py','measure.py','analyze.py'):
        path = original.PRIOR/name
        files[str(path)] = sha256(path)
    native = binary_identity()
    files[native['extension_path']] = native['extension_sha256']
    materializations = []
    for path in plans:
        plan = read_json(path)
        validate_plan(plan)
        files[str(path.resolve())] = sha256(path)
        for row in plan['trials']:
            config = Path(row['config']['path']).resolve(strict=True)
            option = Path(row['options_file']['path']).resolve(strict=True)
            if sha256(config)!=row['config']['sha256'] or sha256(option)!=row['options_file']['sha256']:
                raise ValueError('original planned config/options bytes changed')
            loaded,_,runtime = runner.materialize(config,row['run_id'],'C')
            files[str(config)],files[str(option)] = sha256(config),sha256(option)
            for item in loaded.inputs:
                files[item['path']] = item['sha256']
            server = runtime.config.server
            files[str(server.binary)],files[str(server.config)] = sha256(server.binary),sha256(server.config)
            simulator = Path(server.source_dir)
            simulator_revision = subprocess.run(['git','-C',str(simulator),'rev-parse','HEAD'],
                check=True,capture_output=True,text=True).stdout.strip()
            if simulator_revision != server.simulator_revision:
                raise ValueError('current configured Simulator revision differs')
            for name,expected in CORRECTED_REVIEWED_SPEC_HASHES.items():
                files[str(simulator/name)] = expected
            materializations.append({'run_id':row['run_id'],'context_id':row['context_id'],
                'materialized_sha256':runtime.sha256,'options':row['options'],'output':row['output']})
    if len(materializations)>57 or len({row['run_id'] for row in materializations}) != len(materializations):
        raise ValueError('at most57 distinct declared actual invocations per root freeze')
    check_files(files)
    return {'schema_version':1,'git_revision':revision,'files':files,'materializations':materializations,
            'qualification':'root-issued experiment execution evidence; no production Source/resume/quality authority'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = namespace(args.output)
    freeze = build(args.plan)
    immutable_json(output,freeze)
    print(json.dumps({'status':'FROZEN','output':str(output),'sha256':sha256(output),
                      'files':len(freeze['files']),'materializations':len(freeze['materializations'])}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
