"""Pin the six complete-original numeric6 inputs and their G0/G1 start states."""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.learning_update.common import (
    INVENTORY,
    PRIOR,
    PRODUCER,
    REPO,
    check_files,
    code_files,
    immutable_json,
    prior_runner,
    read_json,
    revision,
    sha256,
)


def prepare(output: Path) -> dict[str,Any]:
    from gf_common.card_instance_rules import card_instance_rules
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_rollout import read_hpo_rollout
    from gf_training.hpo_source import validate_hpo_parent
    inventory=read_json(INVENTORY)
    models={}
    for kind in ('mlp','gru','transformer'):
        recorded=inventory['models'][kind]
        parent=Path(recorded['parent'])
        if sha256(parent/'run.json')!=recorded['run_json_sha256']:
            raise ValueError('investigated original parent changed')
        state,_records=validate_hpo_parent(parent,require_complete=False,max_owned_bytes=2*1024**3,
            deadline=time.monotonic()+600)
        if state['status']!='incomplete' or state['updates']!=2:
            raise ValueError('fixed original must be healthy two-update parent')
        actions,model,_=hpo_state(state)
        from gf_models.structured_config import InferenceBindingV2
        binding=InferenceBindingV2(**state['execution']['learner']['binding'])
        training=hpo_training_config(recorded['recipe'])
        files={str(parent/'run.json'):sha256(parent/'run.json')}
        batches=[]
        starts=[]
        for generation in recorded['generation_inputs']:
            check_files(generation['files'])
            directory=parent/'artifacts'/f"{generation['generation']:06d}"
            inference=read_json(directory/'inference.json')
            resume=read_json(directory/'resume.json')
            if (inference['schema_version']!=5 or resume['schema_version']!=5
                    or inference['producer']['source_digest']!=PRODUCER
                    or resume['producer']['source_digest']!=PRODUCER
                    or resume['optimizer_steps']!=generation['generation']*2):
                raise ValueError('pinned historical start identity/schema/Adam steps differ')
            files.update(generation['files'])
            starts.append({'generation':generation['generation'],'directory':str(directory),
                'steps':resume['optimizer_steps'],'inference_sha256':sha256(directory/'inference.json'),
                'resume_sha256':sha256(directory/'resume.json'),'recorded_producer':resume['producer']})
        for update in recorded['updates']:
            path=Path(update['numeric'])
            numeric_files={str(path.parent/name):expected for name,expected in update['numeric_files'].items()}
            check_files(numeric_files)
            if sha256(path)!=update['numeric_sha256'] or sha256(Path(update['update_record']))!=update['record_sha256']:
                raise ValueError('pinned original update/numeric changed')
            original=read_json(Path(update['update_record']))
            rollout=read_hpo_rollout(path.parent,update['numeric_sha256'],model,training,
                actions=actions,rules=card_instance_rules(),expected_selection=update['selection'],
                expected_snapshot_identity=update['snapshot_identity'],
                max_owned_bytes=2*1024**3,numeric_budget_bytes=512*1024**2)
            files.update(numeric_files)
            files[str(path)]=update['numeric_sha256']
            files[update['update_record']]=update['record_sha256']
            batches.append({'numeric':str(path),'numeric_sha256':update['numeric_sha256'],
                'selection':update['selection'],'snapshot_identity':update['snapshot_identity'],
                'generation':update['expected_source_generation'],'used_decisions':len(rollout),
                'collected_decisions':rollout.collected_decisions,'numeric_schema':6,
                'original_update':original})
        check_files(files)
        models[kind]={'parent':str(parent),'model':asdict(model),'training':asdict(training),
            'actions':asdict(actions),'binding':asdict(binding),'resume_identity':state['resume_identity'],
            'operating':recorded['operating'],'files':files,'starts':starts,'batches':batches}
    value={'schema_version':1,'kind':'fixed-update-diagnostic-inputs','models':models,
        'inventory_sha256':sha256(INVENTORY),'recorded_producer_source':PRODUCER,
        'preparation_code':code_files(),'preparation_revision':revision(),
        'qualification':'train-only unqualified diagnostic; no Source/capability or completion objective',
        'core_excludes':['full Source verification','ownership/FD management','snapshot/save','input retirement']}
    immutable_json(output,value)
    return value


def freeze(output: Path,configs: list[Path],reviewed_revision: str) -> dict[str,Any]:
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_training.native_backend import binary_identity
    if revision()!=reviewed_revision:
        raise ValueError('reviewed revision must equal current HEAD')
    files=code_files()
    tracked=subprocess.run(['git','-C',str(REPO),'ls-files','--error-unmatch','--',*files],
        capture_output=True,text=True,check=False)
    clean=subprocess.run(['git','-C',str(REPO),'diff','--quiet','HEAD','--',*files],capture_output=True,check=False)
    if tracked.returncode or clean.returncode:
        raise ValueError('freeze requires committed reviewed application/benchmark source')
    for name in ('runner.py','measure.py','analyze.py'):
        files[str(PRIOR/name)]=sha256(PRIOR/name)
    runner=prior_runner()
    native=binary_identity()
    files[native['extension_path']]=sha256(Path(native['extension_path']))
    simulator_revision=None
    for config in configs:
        loaded,_,runtime=runner.materialize(config.resolve(strict=True),'00000000-0000-4000-8000-000000000001','C')
        for row in loaded.inputs:
            files[row['path']]=sha256(Path(row['path']))
        simulator=Path(runtime.config.server.source_dir)
        actual=revision(simulator)
        if actual!=runtime.config.server.simulator_revision:
            raise ValueError('Simulator source revision differs from configured pin')
        simulator_revision=actual
        for name in CORRECTED_REVIEWED_SPEC_HASHES:
            files[str(simulator/name)]=sha256(simulator/name)
        for name in (runtime.config.server.binary,runtime.config.server.config):
            files[str(name)]=sha256(Path(name))
    check_files(files)
    value={'schema_version':1,'kind':'root-reviewed-update-source-freeze','git_revision':reviewed_revision,
        'files':files,'simulator_revision':simulator_revision,'native':native}
    immutable_json(output,value)
    return value


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    command=commands.add_parser('inputs');command.add_argument('--output',type=Path,required=True)
    command=commands.add_parser('freeze');command.add_argument('--output',type=Path,required=True)
    command.add_argument('--config',type=Path,action='append',required=True)
    command.add_argument('--reviewed-revision',required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=REPO:
        raise ValueError('invoke from repository root')
    if args.command=='inputs':
        prepare(args.output)
    else:
        freeze(args.output,args.config,args.reviewed_revision)


if __name__=='__main__':
    main()
