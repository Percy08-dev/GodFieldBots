"""SOURCE/RECIPE manifest2; no cached-feature or fitting authority."""
from __future__ import annotations

import math
import stat
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.action_contract import (
    engineering_contract,
    parse_action_contract,
    require_qualified_contract,
)
from gf_common.domain import ContractError, Json, canonical, closed, digest
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_models.structured_sealed import SealedScan
from gf_training.native_tree_dataset import NativeTreeDataset
from gf_training.structured_lineage import _ChainScan, _write
from gf_training.structured_rollout import _owned_bytes

FIELDS={'schema_version','kind','selection','label_recipe','input_binding','input_recipe','tree_projection',
        'neural_cohort','parents','collections','counts','source_set_sha256','shared_generation','content_sha256'}
SELECTION='all-explicit-learner-accepted-natural-return-v2'
LABEL='behavior-selected-natural-return-gamma1-v1'


def _outside_output(output:Path,sources:list[Path])->tuple[Path,dict[Path,tuple[int,int]]]:
    """Freeze existing nonsymlink directory ancestors before any source work."""
    lexical=Path(output).absolute()
    roots=[]
    for source in sources:
        source=Path(source).absolute()
        for path in (source,*source.parents):
            try:info=path.lstat()
            except FileNotFoundError:continue
            if not stat.S_ISDIR(info.st_mode):raise ContractError('source/output ancestors must be nonsymlink directories')
        roots.append(source.resolve(strict=False))
    for path in (lexical,*lexical.parents):
        try:info=path.lstat()
        except FileNotFoundError:continue
        if not stat.S_ISDIR(info.st_mode):raise ContractError('source/output ancestors must be nonsymlink directories')
    canonical=lexical.resolve(strict=False)
    if any(canonical==root or root in canonical.parents for root in roots):
        raise ContractError('output must be outside immutable original source directories')
    if canonical.exists():raise ContractError('owned output must not already exist')
    if not canonical.parent.is_dir():raise ContractError('owned output parent must already exist')
    snapshots={}
    for path in canonical.parents:
        info=path.lstat()
        if not stat.S_ISDIR(info.st_mode):raise ContractError('output ancestors must be nonsymlink directories')
        snapshots[path]=(info.st_dev,info.st_ino)
    return canonical,snapshots


def _check_output_ancestors(snapshots:dict[Path,tuple[int,int]])->None:
    for path,identity in snapshots.items():
        info=path.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=identity:
            raise ContractError('owned output ancestor differs from original snapshot')


def _content(owner:NativeTreeDataset)->Json:
    assert owner.summary is not None
    proof=owner.verify_complete()
    if canonical(proof['collections'])!=canonical(owner.collections):
        raise ContractError('TreeDataset native/Python complete collection receipts differ')
    counts={split:{'games':0,'participants':0,'selected_decisions':0,'zero_endpoint_games':0} for split in ('train','validation')}
    collections=[]
    for receipt,work in zip(owner.collections,owner.summary['worklist'],strict=True):
        collections.append({**receipt,'directory':work['directory']})
        for key,value in receipt['counts'].items():counts[receipt['split']][key]+=value
    parents=[]
    for item in owner.summary['parents']:
        state=item['state'];last=item['generations'][-1]
        generations=[{key:g[key] for key in ('generation','generation_sha256','update_sha256','inference_sha256',
            'resume_sha256','artifact_files_sha256')} for g in item['generations']]
        parents.append({'directory':item['directory'],'run_id':state['run_id'],'seed':state['seed'],
            'run_sha256':item['run_sha256'],'config_sha256':state['config_sha256'],'resume_identity':state['resume_identity'],
            'registry_sha256':item['registry_sha256'],'final_generation':last['generation'],
            'final_generation_sha256':last['generation_sha256'],'final_inference_sha256':last['inference_sha256'],
            'generations':generations,'elapsed_seconds':item['elapsed_seconds']})
    if [p['seed'] for p in parents]!=[17,29,43] or counts['validation']['games']!=192:
        raise ContractError('TreeDataset requires all three ordered parents/final64-game validations')
    if counts['train']['games']!=16*sum(p['final_generation'] for p in parents):
        raise ContractError('TreeDataset omitted a committed original train population')
    first=owner.summary['parents'][0]
    cohort={'model':first['model'],'actions_sha256':first['actions_sha256'],'execution':first['execution'],
        'semantics':first['semantics'],'input_binding':owner.summary['input_binding'],
        'qualified':False,'parent_qualification':'not_performed'}
    source_identity={'recipe':'structured-tree-dataset-originals-v2','parents':parents,
        'collections':[{key:c[key] for key in ('parent_index','collection_kind','generation','manifest_sha256',
            'source_set_sha256','schedule_sha256','bindings_sha256','numeric','selected_roles','counts')} for c in collections]}
    validations=[c for c in collections if c['collection_kind']=='ppo_validation']
    shared=sum(p['elapsed_seconds'] for p in parents)+sum(c['elapsed_seconds'] for c in validations)
    if not math.isfinite(shared) or shared<0:raise ContractError('TreeDataset original generation/validation cost invalid')
    key=digest({'parents':[{name:p[name] for name in ('run_id','run_sha256','final_generation_sha256')} for p in parents],
        'validation':[{name:c[name] for name in ('parent_index','manifest_sha256','source_set_sha256')} for c in validations]})
    value={'schema_version':2,'kind':'structured-fixed-sources','selection':SELECTION,'label_recipe':LABEL,
        'input_binding':owner.summary['input_binding'],'input_recipe':'participant-structured-v1',
        'tree_projection':owner.summary['tree_projection'],'neural_cohort':cohort,'parents':parents,'collections':collections,
        'counts':counts,'source_set_sha256':digest(source_identity),
        'shared_generation':{'seconds':shared,'cost_key':key,'scope':'whole PPO parents plus final validation once; excludes per-fit preparation/fitting'}}
    return {**value,'content_sha256':digest(value)}


def _consume(owner:NativeTreeDataset)->None:
    while owner.stage_next():
        assert owner.current is not None
        count=0
        for sample in owner.selected_inputs():
            if sample.label not in (-1.,0.,1.):raise ContractError('TreeDataset selected natural label invalid')
            count+=1
            del sample
        if count!=owner.current['counts']['selected_decisions']:
            raise ContractError('TreeDataset private projection lost selected rows')
        owner.retire_current()


def build_component_dataset(plan:Json,output:Path,settings:PreparationConfig,*,max_owned_bytes:int,deadline:float,
                            native:Any=None,max_scratch_bytes:int|None=None)->Json:
    """Known engineering-component path; never authorization for real fitting."""
    contract=parse_action_contract(plan['action_contract'])
    if contract!=engineering_contract() or plan['purpose']!='component' or plan['synthetic_capacity'] is not False:
        raise ContractError('explicit known component/false-capacity Dataset2 required')
    output,ancestors=_outside_output(output,[Path(item[key]) for item in plan['parents'] for key in ('directory','validation_directory')])
    _check_output_ancestors(ancestors)
    with NativeTreeDataset(plan,settings,max_owned_bytes=max_owned_bytes,deadline=deadline,native=native,
                           max_scratch_bytes=max_scratch_bytes) as owner:
        _consume(owner);manifest=_content(owner)
        payload_bytes=len(canonical(manifest).encode())+1
        if payload_bytes>owner.scratch or payload_bytes>settings.max_source_file_mib*1024**2:
            raise ContractError('Dataset2 manifest exceeds remaining output scratch/source-file allowance')
        owner._reserve(32*payload_bytes+65536+_owned_bytes(manifest))
        remaining=owner.budget-owner._session.used_bytes()-_owned_bytes(manifest)
        if remaining<=0:raise ContractError('joint TreeDataset publication ownership exhausted')
        scan=_ChainScan(directory=output,max_owned_bytes=remaining,deadline=owner.deadline)
        for parent in (output.parent,*output.parent.parents):
            if parent.is_symlink() or not parent.is_dir():raise ContractError('Dataset2 output requires owned nonsymlink ancestors')
        _check_output_ancestors(ancestors)
        output.mkdir(mode=0o700,exist_ok=False)
        _write(output/'manifest.json',manifest,scan=scan,immutable=True)
        # Publication is no fitting authority: keep the SAME originals until a
        # post-publication whole-source check and final retirement completes.
        owner.verify_complete();scan.finalize();_check_output_ancestors(ancestors);owner.finalize()
        return manifest


def build_dataset(plan:Json,output:Path,settings:PreparationConfig,*,max_owned_bytes:int,deadline:float)->Json:
    require_qualified_contract(parse_action_contract(plan['action_contract']))
    return build_component_dataset(plan,output,settings,max_owned_bytes=max_owned_bytes,deadline=deadline)


def inspect_component_dataset(path:Path,*,expected_digest:str,settings:PreparationConfig,max_owned_bytes:int,
                              deadline:float,native:Any=None)->Json:
    settings=preparation_config(asdict(settings))
    max_owned_bytes=min(max_owned_bytes,settings.memory_budget_mib*1024**2)
    deadline=min(deadline,time.monotonic()+settings.wall_seconds-settings.cleanup_seconds)
    scan=SealedScan(max_owned_bytes=max_owned_bytes,deadline=deadline)
    manifest=scan.object(path,expected_digest);closed(manifest,FIELDS)
    if (type(manifest['schema_version']) is not int or manifest['schema_version']!=2
            or manifest['kind']!='structured-fixed-sources' or manifest['selection']!=SELECTION or manifest['label_recipe']!=LABEL):
        raise ContractError('source-only Dataset2 recipe/schema required')
    if digest({key:value for key,value in manifest.items() if key!='content_sha256'})!=manifest['content_sha256']:
        raise ContractError('Dataset2 canonical content checksum mismatch')
    parents=[]
    for pi,parent in enumerate(manifest['parents']):
        validations=[c for c in manifest['collections'] if c['parent_index']==pi and c['collection_kind']=='ppo_validation']
        if len(validations)!=1:raise ContractError('Dataset2 exact final validation membership required')
        parents.append({'directory':parent['directory'],'validation_directory':validations[0]['directory']})
    plan={'schema_version':2,'kind':'structured-tree-dataset-plan','parents':parents,
        'encoder':manifest['tree_projection']['encoder'],**manifest['input_binding']}
    with NativeTreeDataset(plan,settings,max_owned_bytes=max_owned_bytes,deadline=deadline,
                           extra_owned_bytes=scan.retained,native=native) as owner:
        _consume(owner);actual=_content(owner)
        if canonical(actual)!=canonical(manifest):raise ContractError('Dataset2 original sources/recipe/membership changed')
        scan.finalize();owner.finalize()
    return manifest
