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


def verify_telemetry(result: dict[str,Any],directory: Path,*,parent: Path,
                     spec_sha: str,source_digest: str,variant: str,model_id: str,
                     run_id: str,context_id: str) -> dict[str,Any]:
    """Read only AFTER the original call has closed/reaped its real owners."""
    import os

    from gf_training.hpo_execution_observation import read_observation, validate_observation
    try:
        if (not result.get('probe_expectation_met') or result.get('failure')
                or result.get('gpu_termination_context_verified') is not True
                or result.get('production_original_cleanup_receipt_revalidated') is not True):
            raise ValueError('original actual invocation did not complete healthy cleanup')
        if any(next(root.glob('.observation-*.pending'),None) is not None
                for root in (directory,directory/'learner')):
            raise ValueError('uncertain observation publication pending residue')
        state=read_observation(parent/'run.json')
        resume=state['resume_identity']
        receipt=result['gpu_termination']
        pids=[row['pid'] for row in receipt['resources'] if row['kind']=='learner' and row['outcome']=='reaped']
        if len(pids)!=1:
            raise ValueError('one actually reaped learner required for telemetry binding')
        common={'model_id':model_id,'variant':variant,'spec_sha256':spec_sha,
            'source_digest':source_digest,'resume_identity':resume}
        coordinator=directory/'coordinator-observation.json'
        validate_observation(read_observation(coordinator),expected={**common,'role':'coordinator',
            'pid':os.getpid(),'generation':None,'transaction':None,'run_id':run_id,'context_id':context_id},require_cuda=False)
        files={str(coordinator):sha256(coordinator)}
        transactions=set()
        expected_files={directory/'learner'/f'update-{generation:06d}.json' for generation in (0,1)}
        if set((directory/'learner').glob('update-*.json'))!=expected_files:
            raise ValueError('exactly two observed worker update records required')
        for generation in (0,1):
            path=directory/'learner'/f'update-{generation:06d}.json'
            value=read_observation(path)
            validate_observation(value,expected={**common,'role':'worker','generation':generation,
                'pid':pids[0],'run_id':None,'context_id':None},require_cuda=True)
            transaction=value['bindings']['transaction']
            if type(transaction) is not str or not transaction or transaction in transactions:
                raise ValueError('distinct actual worker update transactions required')
            transactions.add(transaction)
            files[str(path)]=sha256(path)
        return {'diagnostic_complete':True,'files':files,'record_count':3,
            'scope':'diagnostic-only; original training/receipt health remains separate'}
    except BaseException as error:  # noqa: BLE001 - diagnostic failure must not rewrite original outcome
        return {'diagnostic_complete':False,'failure':(type(error).__name__+':'+str(error))[:512]}


def main() -> int:
    from dataclasses import replace

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
    parser.add_argument('--diagnostic-telemetry',action='store_true',
        help='fresh diagnostic-only process telemetry; excluded from timing populations')
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
    spec=TrainingExecutionSpec(schema_version=2 if args.diagnostic_telemetry else 1,
        variant=args.variant,diagnostic_directory=str(args.report_dir/'learner'),
        observe_actual=args.diagnostic_telemetry)
    spec_path=args.report_dir/'worker-execution.json'
    spec_sha=write_execution_spec(spec_path,spec)
    request={'schema_version':1,'kind':'experimental-update-actual','mode':'fresh',
        'quality_eligible':False,'completion_objective':None,'run_id':args.run_id,
        'context_id':args.context_id,'parent':str(parent),'execution':spec.payload(),
        'timing_eligible':not args.diagnostic_telemetry,'diagnostic_telemetry':args.diagnostic_telemetry,
        'execution_path':str(spec_path),'execution_sha256':spec_sha,'freeze':identity,
        'config_file_sha256':sha256(args.config),'materialized_sha256':runtime.sha256,
        'base_identity_scope':'ordinary model/runtime only; experiment sidecar separate',
        'admission':admission,'python':sys.version}
    immutable_json(args.report_dir/'request.json',request)
    result: dict[str,Any]={'schema_version':1,'request':request,'train_calls':0,
        'probe_expectation_met':False,'gpu_termination':None,
        'gpu_termination_context_verified':False,'raw_training_report':None,
        'timing_eligible':not args.diagnostic_telemetry,'diagnostic_complete':None}
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
            observation=None
            original_completed=False
            if args.diagnostic_telemetry:
                try:
                    import os

                    from gf_models.structured_artifacts import producer_identity
                    from gf_training.hpo_execution_observation import ProcessObservation
                    observation=ProcessObservation({'role':'coordinator','generation':None,'transaction':None,
                        'resume_identity':None,'model_id':settings.model.model_id,'variant':args.variant,
                        'spec_sha256':spec_sha,'source_digest':producer_identity()['source_digest'],
                        'pid':os.getpid(),'run_id':args.run_id,'context_id':args.context_id},
                        allowance_bytes=runtime.config.hpo_spec['operating']['parent_budget_bytes'])
                    observation.start()
                except BaseException as error:  # noqa: BLE001 - observer creation cannot prevent original call
                    result['coordinator_observation_failure']=(type(error).__name__+':'+str(error))[:512]
            try:
                if observation is not None:
                    with observation.phase('coordinator_call'):
                        trained=await original(*pos,**kw,_training_execution=(str(spec_path),spec_sha))
                else:
                    trained=await original(*pos,**kw,_training_execution=(str(spec_path),spec_sha))
                original_completed=True
                return trained
            finally:
                if observation is not None:
                    try:
                        from gf_training.hpo_execution_observation import read_observation
                        observation.bind_resume(read_observation(parent/'run.json')['resume_identity'])
                    except BaseException as error:  # noqa: BLE001 - missing original parent means incomplete telemetry
                        observation.failure('resume binding',error)
                    try:
                        result['coordinator_observation']=observation.finish_publish(
                            args.report_dir/'coordinator-observation.json',original_completed=original_completed)
                    except BaseException as error:  # noqa: BLE001 - preserve original authoritative exception/outcome
                        result['coordinator_observation_failure']=(type(error).__name__+':'+str(error))[:512]
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
        if args.diagnostic_telemetry:
            try:
                from gf_models.structured_artifacts import producer_identity
                result['telemetry']=verify_telemetry(result,args.report_dir,parent=parent,spec_sha=spec_sha,
                    source_digest=producer_identity()['source_digest'],variant=args.variant,model_id=settings.model.model_id,
                    run_id=args.run_id,context_id=args.context_id)
            except BaseException as error:  # noqa: BLE001 - observational source/read failure preserves original outcome
                result['telemetry']={'diagnostic_complete':False,'failure':(type(error).__name__+':'+str(error))[:512]}
            result['diagnostic_complete']=result['telemetry']['diagnostic_complete']
        immutable_json(args.report_dir/'result.json',result)
    check_files(identity['files'])
    if not result.get('probe_expectation_met') or args.diagnostic_telemetry and not result.get('diagnostic_complete'):
        return 2
    return code


if __name__=='__main__':
    raise SystemExit(main())
