"""Stdlib Tree2 fit-record semantics and explicit owned Training3 pairing.

No training orchestration, Booster or Torch imports belong in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_models.structured_tree_config import MAX_ARRAY_BYTES, TreeArchitectureV2, feature_width

ENVELOPE_RECIPE='tree-fit-model-envelope-v1'


def model_envelope(model:TreeArchitectureV2,rounds:int)->Json:
    if type(model) is not TreeArchitectureV2:raise ContractError('pure Tree2 architecture required for fit envelope')
    integer(rounds,1,2**31-1)
    width=feature_width(model.encoder.max_positions);leaves=model.network.num_leaves
    nodes=rounds*(2*leaves-1);size=65536+512*width+1024*nodes
    if size>MAX_ARRAY_BYTES:raise ContractError('pre-admitted Tree2 model envelope exceeds64MiB')
    return {'recipe':ENVELOPE_RECIPE,'feature_width':width,'rounds':rounds,'num_leaves':leaves,'node_bound':nodes,'text_bytes':size}


def validate_envelope(value:Any,model:TreeArchitectureV2,rounds:int)->Json:
    closed(value,{'recipe','feature_width','rounds','num_leaves','node_bound','text_bytes'})
    expected=model_envelope(model,rounds)
    for key in ('feature_width','rounds','num_leaves','node_bound','text_bytes'):integer(value[key],1)
    if canonical(value)!=canonical(expected):raise ContractError('Tree2 model envelope differs from exact configured recipe')
    return expected


REQUEST_FIELDS={'schema_version','kind','run_id','model','execution','fit_settings','seed','input_binding','feature_signature',
    'label_recipe','dataset_manifest_sha256','dataset_content_sha256','source_set_sha256','input_files','row_counts','producer','runtime','model_envelope'}
COMPLETION_FIELDS={'schema_version','kind','run_id','status','request_sha256','input_binding','seed','dataset_manifest_sha256',
    'dataset_content_sha256','source_set_sha256','feature_signature','label_recipe','producer','runtime','health','metrics','costs','outputs'}
PROVENANCE_FIELDS={'schema_version','kind','training_run_id','request_sha256','dataset_manifest_sha256','input_binding'}
FIT_FIELDS={'num_boost_round','early_stopping_rounds','learning_rate','min_data_in_leaf','num_threads'}


def _sha(value:Any)->str:
    import re
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:raise ContractError('Tree2 full SHA256 required')
    return value


def _relative(value:Any)->str:
    from pathlib import PurePosixPath
    if type(value) is not str or not value or not value.isascii() or '\\' in value:
        raise ContractError('Tree2 owned relative file name required')
    path=PurePosixPath(value)
    if path.is_absolute() or any(part in ('','.','..') for part in value.split('/')):
        raise ContractError('Tree2 unsafe relative file map')
    return value


def validate_fit_settings(value:Any)->Json:
    import math
    raw=closed(value,FIT_FIELDS)
    for key in ('num_boost_round','early_stopping_rounds','min_data_in_leaf'):integer(raw[key],1,2**31-1)
    integer(raw['num_threads'],1,64)
    rate=raw['learning_rate']
    if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=1:raise ContractError('Tree2 fit learning rate invalid')
    return dict(raw)


def validate_input_binding(value:Any)->Json:
    from gf_common.action_contract import parse_action_contract, validate_record_purpose
    raw=closed(value,{'action_contract','purpose','synthetic_capacity'})
    contract=parse_action_contract(raw['action_contract']);validate_record_purpose(raw['purpose'],contract)
    if type(raw['synthetic_capacity']) is not bool:raise ContractError('Tree2 input capacity classification must be explicit boolean')
    return dict(raw)


def validate_request(value:Any)->Json:
    from gf_models.structured_tree_config import architecture_config, configured_tree_v2
    raw=closed(value,REQUEST_FIELDS)
    if type(raw['schema_version']) is not int or raw['schema_version']!=3 or raw['kind']!='structured-tree-fit-request':
        raise ContractError('Tree2 immutable fit request3 required')
    import re
    if type(raw['run_id']) is not str or re.fullmatch('[0-9a-f]{32}',raw['run_id']) is None:raise ContractError('Tree2 training run identity invalid')
    model=architecture_config(raw['model']);execution=closed(raw['execution'],{'inference','resources'})
    configured_tree_v2({**asdict(model),'checkpoint':None,**execution})
    fit=validate_fit_settings(raw['fit_settings']);validate_envelope(raw['model_envelope'],model,fit['num_boost_round'])
    integer(raw['seed'],0,2**31-1);validate_input_binding(raw['input_binding'])
    for key in ('feature_signature','dataset_manifest_sha256','dataset_content_sha256','source_set_sha256'):_sha(raw[key])
    if raw['label_recipe']!='behavior-selected-natural-return-gamma1-v1':raise ContractError('Tree2 selected natural label recipe required')
    rows=closed(raw['row_counts'],{'train','validation'})
    for size in rows.values():integer(size,1,2**63-1)
    files=raw['input_files']
    if type(files) is not dict or 'inputs/manifest.json' not in files:raise ContractError('Tree2 exact typed input descriptor/map required')
    for name,sha in files.items():
        _relative(name);_sha(sha)
        if not name.startswith('inputs/') or name.count('/')!=1 or not name.endswith(('.npz','.json')):
            raise ContractError('Tree2 request owns numeric input files only')
    closed(raw['producer'],{'code_revision','source_digest'});_sha(raw['producer']['source_digest'])
    revision=raw['producer']['code_revision']
    if revision is not None and (type(revision) is not str or re.fullmatch('[0-9a-f]{40}',revision) is None):raise ContractError('Tree2 producer revision invalid')
    runtime=closed(raw['runtime'],{'numpy_version','lightgbm_version','native_library_sha256'})
    if runtime['numpy_version']!='2.5.4' or runtime['lightgbm_version']!='4.7.0':raise ContractError('Tree2 unsupported fit native runtime')
    _sha(runtime['native_library_sha256'])
    return dict(raw)


@dataclass(frozen=True)
class TrainedProvenance:
    """Canonical immutable owned provenance; no mutable nested dict alias."""
    encoded: bytes

    @classmethod
    def parse(cls,value:Any)->TrainedProvenance:
        raw=closed(value,PROVENANCE_FIELDS)
        if type(raw['schema_version']) is not int or raw['schema_version']!=3 or raw['kind']!='structured-fixed-dataset2-fit':
            raise ContractError('Tree2 closed trained provenance3 required')
        import re
        if type(raw['training_run_id']) is not str or re.fullmatch('[0-9a-f]{32}',raw['training_run_id']) is None:raise ContractError('Tree2 trained run identity invalid')
        for key in ('request_sha256','dataset_manifest_sha256'):_sha(raw[key])
        validate_input_binding(raw['input_binding'])
        return cls(canonical(raw).encode())

    def to_dict(self)->Json:
        import json
        return json.loads(self.encoded)


def provenance_for_request(request:Json,request_sha:str)->TrainedProvenance:
    validate_request(request);_sha(request_sha)
    return TrainedProvenance.parse({'schema_version':3,'kind':'structured-fixed-dataset2-fit','training_run_id':request['run_id'],
        'request_sha256':request_sha,'dataset_manifest_sha256':request['dataset_manifest_sha256'],'input_binding':request['input_binding']})


def fit_admission(model:Any,fit:Json,rows:Json,*,retained_bytes:int)->Json:
    from gf_models.structured_tree_config import ConfiguredTreeModelV2
    if type(model) is not ConfiguredTreeModelV2:raise ContractError('actual configured Tree2 required')
    fit=validate_fit_settings(fit);integer(retained_bytes,0,2**63-1)
    width=feature_width(model.encoder.max_positions);envelope=model_envelope(model.architecture(),fit['num_boost_round'])
    closed(rows,{'train','validation'})
    for count in rows.values():integer(count,1,2**63-1)
    if any(count*4>model.resources.max_array_bytes for count in rows.values()):raise ContractError('Tree2 fit label array exceeds admitted64MiB tensor bound')
    raw=sum(rows.values())*(width*4+4)
    threads=max(model.inference.num_threads,fit['num_threads'])
    fixed=256*1024**2+threads*64*1024**2+16*envelope['text_bytes']
    packing=model.inference.candidate_chunk*(12*width+32)+(1042+18*32)*4+65536
    total=retained_bytes+5*raw+fixed+packing
    if total>model.resources.max_owned_bytes:raise ContractError('Tree2 full Source/raw/Dataset/workspace/model fit admission exceeded')
    return {'raw_bytes':raw,'model_envelope':envelope,'fixed_bytes':fixed,'packing_bytes':packing,'retained_bytes':retained_bytes,'total_bytes':total,'native_threads':threads}


@dataclass(frozen=True)
class TrainingBinding:
    directory: str
    request_sha256: str
    completion_sha256: str

    def __post_init__(self)->None:
        if type(self.directory) is not str or not self.directory:raise ContractError('explicit owned training directory required')
        _sha(self.request_sha256);_sha(self.completion_sha256)


def validate_pair_records(request:Any,completion:Any,inference:Json,*,request_sha256:str,
                          inference_sha256:str,model_sha256:str)->None:
    import math
    request=validate_request(request);completion=closed(completion,COMPLETION_FIELDS)
    for value in (request_sha256,inference_sha256,model_sha256):_sha(value)
    if (type(completion['schema_version']) is not int or completion['schema_version']!=3
            or completion['kind']!='structured-tree-training' or completion['status']!='complete'):
        raise ContractError('complete healthy Tree2 Training3 required before binding')
    if completion['request_sha256']!=request_sha256:raise ContractError('Tree2 request/completion hash link differs')
    for key in ('run_id','input_binding','seed','dataset_manifest_sha256','dataset_content_sha256','source_set_sha256',
                'feature_signature','label_recipe','producer','runtime'):
        if canonical(completion[key])!=canonical(request[key]):raise ContractError('Tree2 request/completion identity mismatch: '+key)
    origin=TrainedProvenance.parse(inference['provenance'])
    if inference['fit_origin']!='trained' or origin!=provenance_for_request(request,request_sha256):
        raise ContractError('Tree2 trained inference provenance differs from immutable request')
    if canonical(inference['model'])!=canonical(request['model']) or inference['feature_signature']!=request['feature_signature']:
        raise ContractError('Tree2 request/inference architecture/features differ')
    creation=closed(inference['creation_execution'],{'inference','resources','tree_descriptor'})
    if canonical({key:creation[key] for key in ('inference','resources')})!=canonical(request['execution']):
        raise ContractError('Tree2 original creation execution differs from pre-fit request')
    classification={key:inference[key] for key in ('action_contract','purpose','synthetic_capacity')}
    if canonical(classification)!=canonical(request['input_binding']):raise ContractError('Tree2 trained classification promotion forbidden')
    descriptor=inference['tree_descriptor']
    if (descriptor['model_sha256']!=model_sha256 or descriptor['model_bytes']>request['model_envelope']['text_bytes']
            or descriptor['tree_count']>request['fit_settings']['num_boost_round']):
        raise ContractError('Tree2 actual model exceeds admitted fit envelope')
    health=closed(completion['health'],{'exit_code','reaped','stdout_bytes','stderr_bytes'})
    if type(health['exit_code']) is not int or health['exit_code']!=0 or health['reaped'] is not True:
        raise ContractError('Tree2 native child success and confirmed reap required')
    integer(health['stdout_bytes'],0,2**63-1);integer(health['stderr_bytes'],0,2**63-1)
    metrics=closed(completion['metrics'],{'configured_rounds','consumed_iterations','best_iteration','retained_tree_count','validation_l2'})
    cap=request['fit_settings']['num_boost_round']
    integer(metrics['configured_rounds'],cap,cap);integer(metrics['consumed_iterations'],1,cap)
    integer(metrics['best_iteration'],1,metrics['consumed_iterations']);integer(metrics['retained_tree_count'],1,metrics['best_iteration'])
    if metrics['retained_tree_count']!=descriptor['tree_count']:raise ContractError('Tree2 completion retained-tree count differs')
    loss=metrics['validation_l2']
    if type(loss) not in (int,float) or not math.isfinite(loss) or loss<0:raise ContractError('Tree2 finite validation metric required')
    costs=closed(completion['costs'],{'preparation_seconds','child_seconds','final_validation_seconds','cleanup_seconds','elapsed_seconds','shared_generation'})
    for key in ('preparation_seconds','child_seconds','final_validation_seconds','cleanup_seconds','elapsed_seconds'):
        number=costs[key]
        if type(number) not in (int,float) or not math.isfinite(number) or number<0:raise ContractError('Tree2 finite measured costs required')
    if costs['elapsed_seconds']>7200:raise ContractError('Tree2 original per-fit7200-second budget exceeded')
    shared=closed(costs['shared_generation'],{'seconds','cost_key','scope'});_sha(shared['cost_key'])
    if type(shared['seconds']) not in (int,float) or not math.isfinite(shared['seconds']) or shared['seconds']<0 or type(shared['scope']) is not str:
        raise ContractError('Tree2 shared generation accounting invalid')
    outputs=closed(completion['outputs'],{'files','inference_sha256','model_sha256'})
    if outputs['inference_sha256']!=inference_sha256 or outputs['model_sha256']!=model_sha256:
        raise ContractError('Tree2 completion/inference/model hashes differ')
    if type(outputs['files']) is not dict or 'training.json' in outputs['files']:
        raise ContractError('Tree2 acyclic output file map excludes completion itself')
    for name,sha in outputs['files'].items():_relative(name);_sha(sha)
    if (outputs['files'].get('request.json')!=request_sha256 or outputs['files'].get('artifact/inference.json')!=inference_sha256
            or outputs['files'].get('artifact/model.txt')!=model_sha256):
        raise ContractError('Tree2 required exact request/artifact output map differs')
    if any(outputs['files'].get(name)!=sha for name,sha in request['input_files'].items()):
        raise ContractError('Tree2 consumed numeric inputs differ from immutable request')


def validate_input_envelope(value:Any,request:Json)->tuple[Json,Any]:
    """Pure envelope/typed-descriptor link, before allocating native inputs."""
    from gf_models.sharded_arrays import validate_bundle_metadata
    from gf_models.structured_tree_config import configured_tree_v2
    from gf_models.structured_tree_encoder import feature_layout
    raw=closed(value,{'schema_version','kind','action_config_hash','feature_signature','label_recipe','input_binding','row_counts','data'})
    if type(raw['schema_version']) is not int or raw['schema_version']!=3 or raw['kind']!='structured-tree-fit-inputs':
        raise ContractError('Tree2 closed numeric input envelope3 required')
    _sha(raw['action_config_hash'])
    for key in ('feature_signature','label_recipe','input_binding','row_counts'):
        if canonical(raw[key])!=canonical(request[key]):raise ContractError('Tree2 input/request common fields differ: '+key)
    model=configured_tree_v2({**request['model'],'checkpoint':None,**request['execution']})
    if feature_layout(model.encoder)['feature_signature']!=raw['feature_signature']:
        raise ContractError('Tree2 numeric input recipe/architecture mismatch')
    admission=fit_admission(model,request['fit_settings'],request['row_counts'],retained_bytes=0)
    descriptors=validate_bundle_metadata(raw['data'],max_decoded_bytes=admission['raw_bytes'])
    expected={};chunk=model.inference.candidate_chunk;width=feature_width(model.encoder.max_positions)
    # Descriptor totals constrain the declared population before constructing keys.
    if raw['data']['decoded_bytes']!=admission['raw_bytes']:
        raise ContractError('Tree2 typed input decoded population differs')
    if len(descriptors)!=2*sum((n+chunk-1)//chunk for n in request['row_counts'].values()):
        raise ContractError('Tree2 exact numeric block membership differs')
    for split,count in request['row_counts'].items():
        for index,start in enumerate(range(0,count,chunk)):
            rows=min(chunk,count-start)
            for kind,shape,size in (('x',[rows,width],rows*width*4),('y',[rows],rows*4)):
                expected[f'{split}.{kind}.{index:06}']={'dtype':'float32','shape':shape,'bytes':size}
    if canonical(descriptors)!=canonical(expected):raise ContractError('Tree2 typed feature/label blocks differ from exact row layout')
    files={'inputs/'+name:sha for name,sha in raw['data']['files'].items()}
    files['inputs/manifest.json']=request['input_files']['inputs/manifest.json']
    if files!=request['input_files']:raise ContractError('Tree2 exact input file map/descriptor mismatch')
    return raw,model


def inspect_fit_inputs(directory:Any,request:Json,scan:Any,*,retain:bool=False)->tuple[Json,dict[str,list[Any]],dict[str,list[Any]]]:
    """Verify bounded typed blocks; retain arrays only in the own numeric child."""
    import numpy as np

    from gf_models.arrays import load_arrays
    from gf_models.structured_tree_encoder import categorical_limits, feature_layout
    request=validate_request(request)
    envelope=scan.object(directory/'inputs'/'manifest.json',request['input_files']['inputs/manifest.json'])
    envelope,model=validate_input_envelope(envelope,request)
    fit_admission(model,request['fit_settings'],request['row_counts'],retained_bytes=scan.retained)
    data=envelope['data'];scan.inventory(directory/'inputs',set(data['files'])|{'manifest.json'})
    blocks:dict[str,dict[int,Any]]={'train':{},'validation':{}};labels:dict[str,dict[int,Any]]={'train':{},'validation':{}}
    kept=0;layout=feature_layout(model.encoder);limits=categorical_limits(model.encoder)
    for shard in data['shards']:
        scan.check();name=shard['file'];side=name[:-4]+'.json'
        if canonical(scan.object(directory/'inputs'/side,data['files'][side]))!=canonical(shard['metadata']):
            raise ContractError('Tree2 typed sidecar differs')
        with scan.file(directory/'inputs'/name,data['files'][name]) as (source,info):
            decoded=sum(item['bytes'] for item in shard['metadata']['arrays'].values())
            scan.charge(kept+32*info.st_size+3*decoded+65536,temporary=True)
            arrays=load_arrays(directory/'inputs'/name,shard['metadata'],_check=scan.check,_source=source,
                               _snapshot=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns))
            for key in sorted(arrays):
                scan.check();array=arrays[key];split,kind,block_id=key.split('.');index=int(block_id)
                if kind=='x':
                    for start in range(0,len(array),model.inference.candidate_chunk):
                        scan.check();chunk=array[start:start+model.inference.candidate_chunk]
                        if not np.isfinite(chunk).all():raise ContractError('nonfinite Tree2 feature block')
                        for feature_index,maximum in zip(layout['categorical_indices'],limits,strict=True):
                            values=chunk[:,feature_index]
                            if np.any(values<0) or np.any(values>maximum) or np.any(values!=np.floor(values)):
                                raise ContractError('Tree2 input categories outside exact integer domain')
                elif not np.all(np.isin(array,(-1.,0.,1.))):raise ContractError('Tree2 selected labels must be natural ±1/0')
                if retain:
                    array.flags.writeable=False
                    destination=(blocks if kind=='x' else labels)[split]
                    if index in destination:raise ContractError('Tree2 duplicate logical numeric block')
                    destination[index]=array;kept+=array.nbytes
            del arrays
    scan.check()
    if retain:
        for split,features in blocks.items():
            if set(features)!=set(labels[split]) or any(len(a)!=len(labels[split][i]) for i,a in features.items()):
                raise ContractError('Tree2 feature/label logical block pair mismatch')
    return envelope,{split:[items[i] for i in sorted(items)] for split,items in blocks.items()}, {split:[items[i] for i in sorted(items)] for split,items in labels.items()}


@dataclass
class PairedTraining:
    """Retained original request/completion/input/artifact snapshots, no Source authority."""
    binding:TrainingBinding
    request:Json
    completion:Json
    inference:Json
    scan:Any
    directories:dict[Any,tuple[int,...]]
    ancestors:dict[Any,tuple[int,int]]

    def finalize(self)->None:
        from gf_models.structured_sealed import _identity
        self.scan.finalize()
        for path,identity in self.ancestors.items():
            self.scan.check();info=path.lstat()
            if (info.st_dev,info.st_ino)!=identity or path.is_symlink():raise ContractError('Tree2 paired ancestor replaced')
        for path,snapshot in self.directories.items():
            self.scan.check()
            if _identity(path.lstat())!=snapshot or path.is_symlink():
                raise ContractError('Tree2 paired owned directory/ancestor changed')


def _container(scan:Any,directory:Any,expected:dict[str,str])->dict[Any,tuple[int,...]]:
    import os
    import stat

    from gf_models.structured_sealed import _identity
    result={}
    for path in (directory,*directory.parents):
        scan.check();info=path.lstat()
        if not stat.S_ISDIR(info.st_mode):raise ContractError('Tree2 training root requires nonsymlink real ancestors')
        # Ancestor siblings may legitimately change; pin their object identities.
        result[path]=_identity(info)
    seen=set();scan.charge(1024+512*len(expected))
    with os.scandir(directory) as entries:
        for entry in entries:
            scan.check()
            kind=expected.get(entry.name)
            if (kind is None or entry.name in seen or
                    (kind=='file' and not entry.is_file(follow_symlinks=False)) or
                    (kind=='directory' and not entry.is_dir(follow_symlinks=False))):
                raise ContractError('Tree2 training bundle exact owned inventory required')
            scan.charge(512+4*len(entry.name));seen.add(entry.name)
    if seen!=set(expected):raise ContractError('Tree2 training bundle incomplete inventory')
    return result


def inspect_training(directory:Any,*,expected_binding:TrainingBinding|None=None,max_owned_bytes:int=2*1024**3,
                     deadline:float=float('inf'))->PairedTraining:
    """Explicit Training3 root only; no ancestor search, Booster or Torch work."""
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan, _identity
    directory=Path(directory).absolute();scan=SealedScan(max_owned_bytes=max_owned_bytes,deadline=deadline)
    directories=_container(scan,directory,{'request.json':'file','training.json':'file','inputs':'directory','artifact':'directory',
        'child-result.json':'file','stdout.log':'file','stderr.log':'file'})
    # Only root/artifact/input metadata is immutable; outside ancestor ctimes may
    # vary without replacing the actual directory. Check those identities below.
    ancestor_ids={p:(s[0],s[1]) for p,s in directories.items() if p!=directory}
    directories={directory:directories[directory]}
    request=scan.object(directory/'request.json',None if expected_binding is None else expected_binding.request_sha256)
    completion=scan.object(directory/'training.json',None if expected_binding is None else expected_binding.completion_sha256)
    validate_request(request)
    if expected_binding is not None and Path(expected_binding.directory).absolute()!=directory:
        raise ContractError('Tree2 explicit paired directory differs from binding')
    scan.inventory(directory/'artifact',{'inference.json','model.txt'})
    directories[directory/'artifact']=_identity((directory/'artifact').lstat())
    directories[directory/'inputs']=_identity((directory/'inputs').lstat())
    inference=scan.object(directory/'artifact'/'inference.json')
    request_sha=scan.files[directory/'request.json'][1];inference_sha=scan.files[directory/'artifact'/'inference.json'][1]
    with scan.file(directory/'artifact'/'model.txt'):
        pass
    model_sha=scan.files[directory/'artifact'/'model.txt'][1]
    from gf_common.action_contract import parse_action_contract
    from gf_models.structured_tree_artifacts import inspect_artifact
    from gf_models.structured_tree_config import configured_tree_v2
    model=configured_tree_v2({**request['model'],'checkpoint':None,**request['execution']})
    classification=request['input_binding']
    inference,text=inspect_artifact(directory/'artifact',expected_digest=inference_sha,model=model,
        action_config_hash=inference['action_config_hash'],action_contract=parse_action_contract(classification['action_contract']),
        purpose=classification['purpose'],synthetic_capacity=classification['synthetic_capacity'],deadline=deadline,_scan=scan)
    validate_fitted_parameters(text,request,model)
    validate_pair_records(request,completion,inference,request_sha256=request_sha,inference_sha256=inference_sha,model_sha256=model_sha)
    envelope,_,_=inspect_fit_inputs(directory,request,scan)
    if inference['action_config_hash']!=envelope['action_config_hash']:
        raise ContractError('Tree2 actual own action hash differs from verified numeric descriptor')
    expected={'request.json','artifact/inference.json','artifact/model.txt','child-result.json','stdout.log','stderr.log'}|set(request['input_files'])
    if set(completion['outputs']['files'])!=expected:raise ContractError('Tree2 completion exact file membership differs')
    for name,sha in completion['outputs']['files'].items():
        with scan.file(directory/name,sha):pass
    receipt=scan.object(directory/'child-result.json',completion['outputs']['files']['child-result.json'])
    closed(receipt,{'schema_version','kind','request_sha256','input_files','metrics','inference_sha256','model_sha256'})
    if (type(receipt['schema_version']) is not int or receipt['schema_version']!=3 or receipt['kind']!='structured-tree-fit-result'
            or receipt['request_sha256']!=request_sha or receipt['input_files']!=request['input_files']
            or canonical(receipt['metrics'])!=canonical(completion['metrics'])
            or receipt['inference_sha256']!=inference_sha or receipt['model_sha256']!=model_sha):
        raise ContractError('Tree2 actual numeric child receipt differs from completion')
    for name,key in (('stdout.log','stdout_bytes'),('stderr.log','stderr_bytes')):
        if scan.files[directory/name][0][2]!=completion['health'][key]:raise ContractError('Tree2 bounded child log byte count differs')
    binding=TrainingBinding(str(directory),request_sha,scan.files[directory/'training.json'][1])
    pair=PairedTraining(binding,request,completion,inference,scan,directories,ancestor_ids)
    pair.finalize()
    return pair


def tree_fit_producer(code_revision:str|None=None)->Json:
    """Own Python source identity without importing the optional neural stack."""
    import hashlib
    import re
    from pathlib import Path
    if code_revision is not None and (type(code_revision) is not str or re.fullmatch('[0-9a-f]{40}',code_revision) is None):
        raise ContractError('Tree2 producer revision invalid')
    root=Path(__file__).resolve().parent.parent;source=hashlib.sha256()
    for package in ('gf_common','gf_models','gf_training'):
        for path in sorted((root/package).rglob('*.py')):
            source.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return {'code_revision':code_revision,'source_digest':source.hexdigest()}


def resolve_checkpoint(directory:Any,*,max_owned_bytes:int=2*1024**3,deadline:float=float('inf'))->tuple[Any,str,TrainingBinding|None]:
    """Inspect only the explicitly supplied toy artifact or Training3 root."""
    import stat
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan
    directory=Path(directory).absolute()
    entries={}
    for name in ('inference.json','training.json'):
        try:info=(directory/name).lstat()
        except FileNotFoundError:continue
        if not stat.S_ISREG(info.st_mode):raise ContractError('Tree2 root discriminator must be a regular owned file')
        entries[name]=True
    if len(entries)!=1:raise ContractError('explicit Tree2 toy artifact/Training3 root is ambiguous or incomplete')
    if 'training.json' in entries:
        pair=inspect_training(directory,max_owned_bytes=max_owned_bytes,deadline=deadline)
        return directory/'artifact',pair.completion['outputs']['inference_sha256'],pair.binding
    scan=SealedScan(max_owned_bytes=max_owned_bytes,deadline=deadline)
    _container(scan,directory,{'inference.json':'file','model.txt':'file'})
    manifest=scan.object(directory/'inference.json')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version']!=3
            or manifest.get('kind')!='structured-tree-inference' or manifest.get('fit_origin')!='toy' or manifest.get('provenance') is not None):
        raise ContractError('bare trained/unknown Tree2 artifact cannot bind without explicit paired Training3 root')
    scan.finalize()
    return directory,scan.files[directory/'inference.json'][1],None


def copy_training_bundle(source:Any,target:Any,*,max_owned_bytes:int=2*1024**3,deadline:float=float('inf'))->TrainingBinding:
    """Copy unchanged paired bytes; own completion remains revocable until success."""
    import os
    import stat
    from pathlib import Path

    from gf_models.structured_tree_artifacts import _publish, _written
    source=Path(source).absolute();target=Path(target).absolute()
    # Reject aliasing BEFORE source parsing, mkdir or publication. Lexical '..'
    # and symlink ancestors cannot grant write access inside immutable originals.
    for path in (source,*source.parents,target,*target.parents):
        try:info=path.lstat()
        except FileNotFoundError:continue
        if not stat.S_ISDIR(info.st_mode):raise ContractError('paired copy requires nonsymlink real directory ancestors')
    source=source.resolve(strict=True);target=target.resolve(strict=False)
    if target==source or source in target.parents:raise ContractError('paired copy requires a separate owned target outside immutable source')
    ancestors={}
    for parent in target.parents:
        info=parent.lstat()
        if not stat.S_ISDIR(info.st_mode):raise ContractError('paired copy requires nonsymlink directory ancestors')
        ancestors[parent]=(info.st_dev,info.st_ino)
    def check()->None:
        pair.scan.check()
        for parent,identity in ancestors.items():
            current=parent.lstat()
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=identity:
                raise ContractError('paired copy ancestor replaced')
    pair=inspect_training(source,max_owned_bytes=max_owned_bytes,deadline=deadline)
    check();target.mkdir(mode=0o700,exist_ok=False)
    root_info=target.lstat();root_identity=(root_info.st_dev,root_info.st_ino)
    root_fd=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    held:dict[str,int]={'':root_fd};identities={'':root_identity}
    try:
        if (os.fstat(root_fd).st_dev,os.fstat(root_fd).st_ino)!=root_identity:
            raise ContractError('paired-copy original target root replaced')
        for name in ('inputs','artifact'):
            check()
            os.mkdir(name,mode=0o700,dir_fd=root_fd)
            created=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
            fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
            held[name]=fd;identities[name]=(created.st_dev,created.st_ino)
        def owned_directories()->None:
            check()
            for name,fd in held.items():
                path=target/name if name else target;info=path.lstat();actual=os.fstat(fd)
                if (not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=identities[name]
                        or (actual.st_dev,actual.st_ino)!=identities[name]):
                    raise ContractError('paired-copy original target directory replaced')
        owned_directories()
        # Every destination is relative to an ORIGINAL retained directory FD;
        # reopening destination.parent by pathname would follow a swapped root.
        for name,expected in pair.completion['outputs']['files'].items():
            owned_directories()
            with pair.scan.file(source/name,expected) as (stream,info):
                pair.scan.charge(32*info.st_size+65536,temporary=True)
                raw=stream.read(info.st_size+1)
                if len(raw)!=info.st_size:raise ContractError('paired copy original length changed')
                destination=target/name;parent=Path(name).parent.as_posix()
                directory_fd=held['' if parent=='.' else parent]
                owned_directories()
                fd,snapshot,actual=_publish(destination.parent,directory_fd,destination.name,raw,deadline)
                try:
                    if actual!=expected:raise ContractError('paired copy changed immutable bytes')
                    _written(destination.parent,directory_fd,destination.name,fd,snapshot,expected,deadline)
                    pair.scan.charge(512+len(str(destination).encode()))
                    pair.scan.files[destination]=(snapshot,expected)
                    owned_directories()
                finally:os.close(fd)
            pair.finalize();owned_directories()
        pair.finalize();owned_directories()
        with pair.scan.file(source/'training.json',pair.binding.completion_sha256) as (stream,info):
            pair.scan.charge(32*info.st_size+65536,temporary=True)
            raw=stream.read(info.st_size+1)
            if len(raw)!=info.st_size:raise ContractError('paired completion length changed')
        owner=_CopiedCompletion(target,pair,ancestors,root_identity,deadline,max_owned_bytes,root_fd)
        return owner.commit(raw)
    finally:
        for fd in reversed(tuple(held.values())):os.close(fd)


class _CopiedCompletion:
    """Narrow models-level exact-copy owner; no training orchestration import."""
    def __init__(self,target:Any,pair:PairedTraining,ancestors:Any,root_identity:Any,deadline:float,limit:int,root_fd:int)->None:
        self.target,self.pair,self.ancestors,self.root_identity=target,pair,ancestors,root_identity
        self.deadline,self.limit=deadline,limit;self.fd:int|None=None;self.directory_fd:int|None=None
        self.snapshot:Any=None;self.exposed=False;self.revoked=False;self.root_fd=root_fd

    def _namespace(self)->None:
        import os
        self.pair.scan.check()
        if self.directory_fd is not None and (os.fstat(self.directory_fd).st_dev,os.fstat(self.directory_fd).st_ino)!=self.root_identity:
            raise ContractError('paired-copy held target root differs')
        for path,expected in {**self.ancestors,self.target:self.root_identity}.items():
            info=path.lstat()
            if path.is_symlink() or (info.st_dev,info.st_ino)!=expected:raise ContractError('paired-copy namespace changed')

    def _postcheck(self)->TrainingBinding:
        self.pair.finalize();self._namespace()
        remaining=self.limit-self.pair.scan.retained
        if remaining<=0:raise ContractError('paired-copy joint metadata ownership exhausted')
        copied=inspect_training(self.target,max_owned_bytes=remaining,deadline=self.deadline)
        copied.finalize();self.pair.finalize();self._namespace()
        return copied.binding

    def _release(self)->None:
        import os
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.directory_fd is not None:os.close(self.directory_fd);self.directory_fd=None

    def revoke(self)->None:
        import os
        import time
        if self.revoked or self.fd is None:return
        if time.monotonic()>=self.deadline:raise ContractError('completion revocation unconfirmed: original copy deadline exhausted')
        held=os.fstat(self.fd)
        if (held.st_dev,held.st_ino)!=self.snapshot[:2] or os.pwrite(self.fd,b'!',0)!=1 or os.pread(self.fd,1,0)!=b'!':
            raise ContractError('completion revocation unconfirmed: copied owned FD invalidation failed')
        self.revoked=True;self._release()

    def commit(self,raw:bytes)->TrainingBinding:
        import os
        import uuid

        from gf_models.structured_tree_artifacts import _identity, _publish, _written
        temporary='copy-completion-'+uuid.uuid4().hex+'.tmp'
        self.directory_fd=os.dup(self.root_fd)
        try:
            self._namespace()
            self.fd,self.snapshot,sha=_publish(self.target,self.directory_fd,temporary,raw,self.deadline)
            self.exposed=True
            # Exclusive publication never overwrites an existing final object.
            os.link(temporary,'training.json',src_dir_fd=self.directory_fd,dst_dir_fd=self.directory_fd,follow_symlinks=False)
            linked=os.fstat(self.fd)
            if (linked.st_dev,linked.st_ino,linked.st_size,linked.st_mtime_ns)!=self.snapshot[:4]:
                raise ContractError('paired completion changed across publication')
            self.snapshot=_identity(linked)
            _written(self.target,self.directory_fd,temporary,self.fd,self.snapshot,sha,self.deadline)
            os.unlink(temporary,dir_fd=self.directory_fd)
            unlinked=os.fstat(self.fd)
            if (unlinked.st_dev,unlinked.st_ino,unlinked.st_size,unlinked.st_mtime_ns)!=self.snapshot[:4]:
                raise ContractError('paired completion changed during temporary retirement')
            self.snapshot=_identity(unlinked)
            _written(self.target,self.directory_fd,'training.json',self.fd,self.snapshot,sha,self.deadline)
            self.pair.scan.files[self.target/'training.json']=(self.snapshot,sha)
            binding=self._postcheck()
            _written(self.target,self.directory_fd,'training.json',self.fd,self.snapshot,sha,self.deadline)
            self._namespace()
        except BaseException as original:
            if self.fd is not None and self.exposed:
                try:self.revoke()
                except BaseException as cleanup:  # noqa: BLE001 - exact unconfirmed copy owner stays reachable
                    vars(original)['_structured_copy_cleanup_owner']=self
                    try:original.add_note('completion revocation unconfirmed: '+type(cleanup).__name__)
                    except BaseException:pass  # noqa: BLE001, S110 - diagnostics cannot replace original failure
            else:self._release()
            raise
        self._release()
        return binding


def validate_fitted_parameters(text:str,request:Json,model:Any)->None:
    from gf_common.domain import digest
    from gf_models.structured_tree_artifacts import _parameters
    params=_parameters(text);fit=request['fit_settings']
    derived=int(digest({'recipe':model.initialization.recipe,'model_seed':request['seed']})[:8],16)%(2**31-1)
    expected_params={'num_iterations':str(fit['num_boost_round']),'min_data_in_leaf':str(fit['min_data_in_leaf']),
        'num_threads':str(fit['num_threads']),'seed':str(derived),'data_random_seed':str(derived),
        'feature_fraction_seed':str(derived),'bagging_seed':str(derived),'deterministic':'1','force_col_wise':'1','feature_pre_filter':'0'}
    if any(params.get(key)!=value for key,value in expected_params.items()) or float(params.get('learning_rate','nan'))!=fit['learning_rate']:
        raise ContractError('Tree2 actual native fitted parameters differ from immutable request')
