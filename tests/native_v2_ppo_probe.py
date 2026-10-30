"""Diagnostic current-v2 saved PPO ingestion; no collection, resume or optimizer.

Use a private candidate wheel via PYTHONPATH. Default is a read-only plan; add
--execute to create a NEW owned diagnostic tree. Its gen0 clone remains running,
has zero updates, and is never complete/offline eligible. Immutable generation
and artifact bytes are copied unchanged. Only mutable clone controls, the copied
parent path, and copied policy checkpoint paths change.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from gf_common.domain import ContractError, Json
from gf_common.preprocessing import PreparationConfig
from gf_models.config import MLPModelConfig, model_config
from gf_training.collection import OnlinePPOContext
from gf_training.journal import atomic_json
from gf_training.lineage import read_json, run_lock, validate_chain
from gf_training.native_ppo import preprocessing_identity, stage_rollout
from gf_training.ppo import ingest_collection, write_rollout


def sha(path:Path)->str:
    with path.open('rb')as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def array_proof(path:Path)->Json:
    import numpy as np

    with np.load(path,allow_pickle=False)as arrays:
        return {name:{'shape':list(arrays[name].shape),'dtype':str(arrays[name].dtype),
                      'sha256':hashlib.sha256(arrays[name].tobytes(order='C')).hexdigest()}
                for name in arrays.files}


def reference(path:Path)->Json:
    value=read_json(path)
    expected=(ROOT/'runs/ppo-comparison-v2/train-seed17/collections/batch-000000').resolve()
    if Path(value['source_collection']).resolve()!=expected:
        raise ContractError('probe requires the explicitly pinned current-v2 batch0 reference')
    parent=expected.parents[1]
    if read_json(parent/'run.json')['status']!='complete' or value['parent_status']!='complete':
        raise ContractError('the original current-v2 parent must remain complete')
    for filename,proof in value['files'].items():
        source=Path(filename)
        if source.stat().st_size!=proof['bytes'] or sha(source)!=proof['sha256']:
            raise ContractError('pinned current-v2 reference bytes changed')
    shard=Path(value['numeric_source'])/'shard-000000.npz'
    if array_proof(shard)!=value['arrays']:
        raise ContractError('actual original PPO array bytes differ from reference manifest')
    return value


def clone_gen0(value:Json,target:Path)->tuple[Path,Json,list[Json],list[Json]]:
    source=Path(value['source_collection']);parent=source.parents[1]
    target.mkdir(parents=True,exist_ok=False)
    for name in ('generations','updates','artifacts','collections'):
        (target/name).mkdir()
    copies:list[Json]=[]

    def copy(source:Path,destination:Path)->None:
        if source.is_symlink() or not source.is_file():
            raise ContractError('diagnostic clone requires ordinary immutable source files')
        before=source.stat();checksum=sha(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        if sha(destination)!=checksum or sha(source)!=checksum or destination.samefile(source):
            raise ContractError('diagnostic copy changed bytes or aliases original storage')
        copies.append({'source':str(source),'copy':str(destination),'sha256':checksum,
                       'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'inode':before.st_ino})

    generation=read_json(parent/'generations/000000.json')
    copy(parent/'generations/000000.json',target/'generations/000000.json')
    for name,checksum in generation['artifact_hashes'].items():
        original=parent/'artifacts/000000'/name
        if Path(name).name!=name or sha(original)!=checksum:
            raise ContractError('immutable generation0 artifact hash mismatch')
        copy(original,target/'artifacts/000000'/name)
    manifest=read_json(source/'manifest.json')
    collection=target/'collections/batch-000000'
    copy(source/'manifest.json',collection/'manifest.json')
    copy(source/'report.json',collection/'report.json')
    for episode in manifest['episodes']:
        identifier=episode['episode_id']
        if Path(identifier).name!=identifier:
            raise ContractError('invalid original episode path')
        for name in ('player-1.jsonl','player-2.jsonl','replay-player-1.json','replay-player-2.json'):
            copy(source/identifier/name,collection/identifier/name)
    state=read_json(parent/'run.json')
    state.update(status='running',updates=0,completed_updates=0,accepted_decisions=0,optimizer_steps=0,
                 generation_sha256=sha(target/'generations/000000.json'),elapsed_seconds=0.,
                 last_unix=state['created_unix'],collection_seconds=0.,update_seconds=0.,invocations=[])
    # Keep original creation/deadline/config/identity bytes as values. This is a
    # diagnostic pre-batch snapshot, not a fresh training budget or resumed run.
    atomic_json(target/'run.json',state)
    atomic_json(target/'latest.json',{'generation':0,'sha256':state['generation_sha256']})
    registry=read_json(parent/'artifact-registry.json')
    atomic_json(target/'artifact-registry.json',{'0':registry['0']})
    rewrites=[{'field':'ppo.parent_run_path','original':manifest['ppo']['parent_run_path'],
               'copy':str(target)}]
    manifest['ppo']['parent_run_path']=str(target)
    for name,binding in manifest['policy_bindings'].items():
        if binding['model_id']=='mlp':
            checkpoint=Path(binding['policy_options']['checkpoint'])
            relative=checkpoint.relative_to(parent)
            new=target/relative
            if not new.is_dir():
                raise ContractError('generation0 clone cannot authorize a later snapshot')
            rewrites.append({'field':f'policy_bindings.{name}.policy_options.checkpoint',
                             'original':str(checkpoint),'copy':str(new)})
            binding['policy_options']['checkpoint']=str(new)
    atomic_json(collection/'manifest.json',manifest)
    # Ordinary guards, with immutable generation record/parent_controls intact.
    validate_chain(target,require_complete=False,require_artifacts=True)
    try:
        validate_chain(target,require_complete=True)
    except ContractError:
        pass
    else:
        raise ContractError('diagnostic clone must never become offline eligible')
    return collection,manifest,copies,rewrites


def check_original(copies:list[Json])->None:
    for item in copies:
        source=Path(item['source']);stat=source.stat()
        if ((stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(item['bytes'],item['mtime_ns'],item['inode'])
                or sha(source)!=item['sha256']):
            raise ContractError('original source changed during diagnostic probe')


def compare_numeric(directory:Path,value:Json)->Json:
    manifest=read_json(directory/'rollout.json')
    if manifest['decision_count']!=224 or len(manifest['shards'])!=1:
        raise ContractError('saved PPO probe must retain all224 learner decisions in one prescribed shard')
    actual=array_proof(directory/'shard-000000.npz')
    if actual!=value['arrays']:
        mismatches=[name for name in set(actual)|set(value['arrays'])if actual.get(name)!=value['arrays'].get(name)]
        raise ContractError(f'actual saved PPO array mismatch: {sorted(mismatches)}')
    return actual


async def run(args:argparse.Namespace,value:Json)->Json:
    deadline=time.monotonic()+args.seconds
    settings=[PreparationConfig(backend='rust',reader=reader,memory_budget_mib=2048,
                cache_budget_mib=512,chunk_target_mib=16,workers=args.workers,max_in_flight=args.workers)
              for reader in ('legacy','typed-borrowed-v1')]
    identities=[preprocessing_identity(config,deadline=deadline)for config in settings]
    args.output.mkdir(parents=True,exist_ok=False)
    parent=args.output/'diagnostic-parent'
    collection,manifest,copies,rewrites=clone_gen0(value,parent)
    model=model_config(read_json(parent/'run.json')['effective_config']['model'])
    if not isinstance(model,MLPModelConfig):
        raise ContractError('current-v2 PPO reference requires its exact MLP recipe')
    results:Json={}
    with run_lock(parent)as owner:
        context=OnlinePPOContext(parent,manifest['ppo']['parent_run_id'],'batch-000000',0,owner.token)
        # Independent unchanged Python feature/GAE/target oracle on the same owned
        # diagnostic source. No policy selection, model inference or optimizer.
        decisions,proof=ingest_collection(collection,model,manifest['ppo'],deadline=deadline,online_context=context)
        python=args.output/'python-numeric'
        write_rollout(python,decisions,deadline=deadline)
        results['python']={'arrays':compare_numeric(python,value),'proof':proof}
        del decisions
        for config,identity in zip(settings,identities,strict=True):
            numeric=args.output/(config.reader+'-numeric')
            count,proof=await stage_rollout(collection,numeric,model,manifest['ppo'],config,
                deadline=deadline,wall_deadline=deadline,online_context=context,expected_identity=identity)
            results[config.reader]={'decisions':count,'arrays':compare_numeric(numeric,value),
                                    'proof':proof,'preprocessing_identity':identity}
        if not results['python']['proof']==results['legacy']['proof']==results['typed-borrowed-v1']['proof']:
            raise ContractError('reader-specific source/manifest/decision provenance differs')
    check_original(copies)
    reference(args.reference)
    clone=read_json(parent/'run.json')
    if clone['status']!='running' or clone['updates']!=0:
        raise ContractError('diagnostic clone status unexpectedly changed')
    same_policy=[spec for group in manifest['schedule']for spec in group['episodes']if spec['policies'][0]==spec['policies'][1]]
    return {'purpose':'diagnostic only; no collection/update/resume/offline eligibility',
            'status':'PASS','reference_manifest':str(args.reference),'reference_manifest_sha256':sha(args.reference),
            'helper_sha256':sha(Path(__file__)),'source_collection':value['source_collection'],
            'source_numeric':value['numeric_source'],'generation':0,'model_inference_sha256':read_json(parent/'generations/000000.json')['inference_sha256'],
            'workers':args.workers,'episodes':len(manifest['episodes']),'learner_decisions':224,
            'same_policy_id_episodes':len(same_policy),'same_policy_learner_seats':sorted({spec['learner_viewer']for spec in same_policy}),
            'diagnostic_clone':str(parent),'clone_status':'running','clone_updates':0,
            'copied_files':copies,'manifest_rewrites':rewrites,'results':results}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'runs/assessments/combined-borrowed-20261030/current-v2-ppo-reference.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,4),default=1)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.seconds<=0 or not(args.output.is_relative_to(Path('/tmp'))or args.output.is_relative_to(ROOT/'runs/assessments')):
        raise ValueError('positive deadline and new owned /tmp or assessments output required')
    value=reference(args.reference)
    if not args.execute:
        print(json.dumps({'execute':False,'reference':str(args.reference),'output':str(args.output),
                          'readers':['python','legacy','typed-borrowed-v1'],'workers':args.workers,
                          'decisions':224,'episodes':16}))
        return
    result=asyncio.run(run(args,value))
    atomic_json(args.output/'result.json',result)
    print(json.dumps({key:result[key]for key in ('status','episodes','learner_decisions','same_policy_id_episodes','same_policy_learner_seats')}))


if __name__=='__main__':
    main()
