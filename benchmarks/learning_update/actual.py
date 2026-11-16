"""Fresh actual comparison using the original coordinator, creator and observer."""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.learning_update.common import (
    PRIOR,
    REPO,
    check_files,
    code_files,
    immutable_json,
    prior_runner,
    read_json,
    revision,
    sha256,
    tree_bytes,
)


def verify_freeze(path: Path,loaded: Any,runtime: Any) -> dict[str,Any]:
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_training.native_backend import binary_identity
    freeze=read_json(path)
    if (freeze['kind']!='root-reviewed-update-source-freeze' or freeze['schema_version']!=1
            or freeze['git_revision']!=revision()):
        raise ValueError('reviewed source freeze differs from current revision')
    required={str((REPO/name).resolve()) for name in code_files()}
    required.update(str(PRIOR/name) for name in ('runner.py','measure.py','analyze.py'))
    required.update(row['path'] for row in loaded.inputs)
    required.update((str(runtime.config.server.binary),str(runtime.config.server.config)))
    native=binary_identity()
    required.add(native['extension_path'])
    simulator=Path(runtime.config.server.source_dir)
    required.update(str(simulator/name) for name in CORRECTED_REVIEWED_SPEC_HASHES)
    if revision(simulator)!=runtime.config.server.simulator_revision:
        raise ValueError('configured Simulator source changed')
    covered={str((REPO/name).resolve()) if not Path(name).is_absolute() else name for name in freeze['files']}
    if not required<=covered:
        raise ValueError('freeze lacks code/config/native/observer/Simulator coverage')
    check_files(freeze['files'])
    return {'sha256':sha256(path),**freeze}


def protocol(loaded: Any,settings: Any,runtime: Any) -> None:
    training=runtime.config.hpo_spec['training_recipe']
    operating=runtime.config.hpo_spec['operating']
    model=runtime.config.model
    if (loaded.config.learning_seeds!=(17,) or loaded.config.decision_budget!=1000000
            or loaded.config.run_wall_seconds!=600 or operating['max_updates']!=2
            or operating['max_participant_workers']!=4 or operating['game_concurrency']!=2
            or training['collection_games']!=16 or training['epochs']!=1
            or model.temporal.hidden_size!=192 or model.temporal.num_layers!=2
            or settings.model.model_id not in ('mlp','gru','transformer')):
        raise ValueError('actual comparison must retain the accepted fixed protocol')
    limits={'managed_ram_bytes':24*1024**3,'managed_cuda_bytes':12*1024**3,
        'source_budget_bytes':2*1024**3,'numeric_budget_bytes':512*1024**2,
        'collector_budget_bytes':2*1024**3,'parent_budget_bytes':1024**3,
        'save_budget_bytes':1024**3}
    if any(operating[key]!=value for key,value in limits.items()):
        raise ValueError('actual original phase limits differ')


async def monitored(runner: Any,args: Any,result: Any,diagnostic: Any) -> None:
    task=asyncio.create_task(runner.execute(args,result,diagnostic))
    try:
        while not task.done():
            await asyncio.wait({task},timeout=1.)
            if tree_bytes(args.report_dir)>1024**3:
                raise ValueError('actual diagnostic output exceeds1GiB')
        await task
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def main() -> int:
    from dataclasses import asdict, replace

    from gf_cli.hpo_config import joint_resource_admission
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.hpo_training_execution import TrainingExecutionSpec, write_execution_spec
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--context-id',required=True)
    parser.add_argument('--variant',choices=('baseline','metrics','packed','microbatch','shared-prefix'),required=True)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if Path.cwd().resolve()!=REPO:
        raise ValueError('invoke from repository root')
    runner=prior_runner()
    runner.canonical_uuid(args.run_id);runner.canonical_uuid(args.context_id)
    args.arm,args.mode,args.pause_after_update='C','fresh',None
    args.config=args.config.resolve(strict=True)
    args.report_dir=args.output.absolute()
    if args.report_dir.exists() or args.report_dir.is_symlink():
        raise ValueError('fresh actual report directory required')
    loaded,settings,runtime=runner.materialize(args.config,args.run_id,'C')
    protocol(loaded,settings,runtime)
    if args.variant=='shared-prefix' and settings.model.model_id=='mlp':
        raise ValueError('shared-prefix is sequence-only')
    parent=Path(loaded.config.storage.parent_root)/args.run_id
    root=REPO/'runs/learning-update-optimization-v1'
    if not parent.is_relative_to(root) or parent.exists() or parent.is_symlink():
        raise ValueError('fresh actual parent under runs/learning-update-optimization-v1 required')
    admission=joint_resource_admission(replace(loaded.config,model=settings.model,training=settings.training),
        layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if not args.execute:
        print({'status':'PLAN_ONLY','parent':str(parent),'admission':admission,'variant':args.variant})
        return 0
    if args.freeze is None:
        raise ValueError('actual execution requires root-reviewed source freeze')
    identity=verify_freeze(args.freeze,loaded,runtime)
    args.report_dir.mkdir(parents=True,exist_ok=False)
    spec=TrainingExecutionSpec(variant=args.variant,diagnostic_directory=str(args.report_dir/'learner'))
    spec_path=args.report_dir/'worker-execution.json'
    spec_sha=write_execution_spec(spec_path,spec)
    request={'schema_version':1,'kind':'experimental-update-actual','mode':'fresh',
        'quality_eligible':False,'completion_objective':None,'run_id':args.run_id,
        'context_id':args.context_id,'parent':str(parent),'execution':asdict(spec),
        'execution_path':str(spec_path),'execution_sha256':spec_sha,'freeze':identity,
        'config_file_sha256':sha256(args.config),'materialized_sha256':runtime.sha256,
        'base_identity_scope':'ordinary model/runtime only; experiment sidecar separate',
        'admission':admission,'python':sys.version}
    immutable_json(args.report_dir/'request.json',request)
    result: dict[str,Any]={'schema_version':1,'request':request,'train_calls':0,
        'probe_expectation_met':False,'gpu_termination':None,
        'gpu_termination_context_verified':False,'raw_training_report':None}
    measure=runner.Measurement()
    diagnostic=runner.BoundedDiagnostics(args.report_dir/'diagnostics.jsonl')
    started=time.monotonic()
    code=0
    try:
        from gf_training import hpo_ppo
        original=hpo_ppo.train_hpo_ppo
        async def once(*pos: Any,**kw: Any) -> Any:
            result['train_calls']+=1
            if result['train_calls']!=1 or kw.get('mode')!='fresh' or kw.get('pause_after_update') is not None:
                raise ValueError('exactly one fresh original coordinator call required')
            return await original(*pos,**kw,_training_execution=(str(spec_path),spec_sha))
        # These historical patches only observe actual boundaries and ownership;
        # no inference bootstrap, worker replacement, or96MiB charge is used.
        patches=runner.production_patches(measure,'C')+[(hpo_ppo,'train_hpo_ppo',once)]
        check_files(identity['files'])
        with runner.patches(patches):
            asyncio.run(monitored(runner,args,result,diagnostic))
        if result['train_calls']!=1:
            raise ValueError('actual coordinator call count differs')
    except BaseException as error:  # noqa: BLE001 - preserve original attempt/cleanup evidence
        result['failure']=runner.error_value(error)
        code=130 if isinstance(error,KeyboardInterrupt) else 2
    finally:
        result['helper_elapsed_seconds']=time.monotonic()-started
        result['diagnostics']=diagnostic.finish()
        result['measurement']=measure.payload()
        rows=result['measurement'].pop('rows')
        with (args.report_dir/'timings.jsonl').open('x') as stream:
            import json
            for row in rows:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        immutable_json(args.report_dir/'result.json',result)
    check_files(identity['files'])
    if not result.get('probe_expectation_met'):
        return 2
    return code


if __name__=='__main__':
    raise SystemExit(main())
