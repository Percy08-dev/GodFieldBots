"""Stdlib Tree2 fit-record semantics and explicit owned Training3 pairing.

No training orchestration, Booster or Torch imports belong in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_models.structured_tree_config import (
    MAX_ARRAY_BYTES,
    TREE2_CONTEXT,
    TREE3_CONTEXT,
    ConfiguredTreeModelV2,
    ConfiguredTreeModelV3,
    TreeArchitectureV2,
    TreeArchitectureV3,
    architecture_config,
    architecture_config_v3,
    architecture_identity,
    architecture_identity_v3,
    configured_tree_v2,
    configured_tree_v3,
)


@dataclass(frozen=True)
class _TreeRecords:
    modern: bool
    game: bool = False
    packed: bool = False

    @property
    def schema(self) -> int:
        return 5 if self.packed else 4 if self.modern else 3

    @property
    def numeric_schema(self) -> int:
        return 4 if self.modern else 3

    @property
    def inference_schema(self) -> int:
        return self.numeric_schema

    @property
    def numeric_extra_fields(self) -> set[str]:
        return {'source_kind', 'source_origin'} if self.modern else set()

    @property
    def source_fields(self) -> tuple[str, ...]:
        return ('source_kind', 'source_origin', 'storage_binding') if self.packed else ('source_kind', 'source_origin')

    def source_identity(self, raw: Json) -> tuple[Any, ...]:
        self.source(raw)
        return tuple(raw[key] for key in self.source_fields)

    def numeric_source(self, raw: Json, request: Json) -> None:
        self.source(request)
        for key in self.numeric_extra_fields:
            if canonical(raw[key]) != canonical(request[key]):
                raise ContractError('numeric Tree source differs from SAME immutable request')

    @property
    def features(self) -> Any:
        return TREE3_CONTEXT if self.modern else TREE2_CONTEXT

    @property
    def extra_fields(self) -> set[str]:
        return ({'source_kind', 'source_origin', 'storage_binding'} if self.packed
            else {'source_kind', 'source_origin'} if self.modern else set())

    @property
    def label(self) -> str:
        return ('behavior-selected-natural-return-gamma1-v1' if self.game or not self.modern
                else 'card-instance-declared-terminal-v1')

    @property
    def shared_cost(self) -> str:
        return 'shared_fixture_construction' if self.modern and not self.game else 'shared_generation'

    @property
    def request_fields(self) -> set[str]:
        return ({'shared_generation', 'managed_memory'} if self.game else
                {'shared_fixture_construction'} if self.modern else set())

    def architecture(self, raw: Any) -> Any:
        return (architecture_config_v3 if self.modern else architecture_config)(raw)

    def configured(self, raw: Any) -> Any:
        return (configured_tree_v3 if self.modern else configured_tree_v2)(raw)

    def identity(self, model: Any) -> str:
        return (architecture_identity_v3 if self.modern else architecture_identity)(model)

    def layout(self, encoder: Any) -> Json:
        from gf_models.structured_tree_encoder import feature_layout, feature_layout_v3
        return (feature_layout_v3 if self.modern else feature_layout)(encoder)

    def limits(self, encoder: Any) -> tuple[int, ...]:
        from gf_models.structured_tree_encoder import categorical_limits, categorical_limits_v3
        return (categorical_limits_v3 if self.modern else categorical_limits)(encoder)

    def binding(self, raw: Any) -> Json:
        if not self.modern:
            return validate_input_binding(raw)
        if self.game:
            from gf_common.card_instance_game_tree import input_binding as game_binding
            expected = game_binding()
        else:
            from gf_common.card_instance_fixture import input_binding as fixture_binding
            expected = fixture_binding()
        closed(raw, set(expected))
        if canonical(raw) != canonical(expected):
            raise ContractError('Tree3 exact corrected component input binding required')
        return dict(raw)

    def source(self, raw: Json, *, toy: bool = False) -> None:
        if not self.modern:
            return
        if self.game:
            from gf_common.card_instance_game_tree import checked_source
            if toy:
                raise ContractError('qualified game Tree cannot be toy/component origin')
            checked_source(raw, packed=self.packed)
            return
        if toy:
            if raw['source_kind'] is not None or raw['source_origin'] is not None:
                raise ContractError('Tree3 toy cannot claim source-backed origin')
        else:
            from gf_common.card_instance_fixture import origin
            if (type(raw['source_kind']) is not str or raw['source_kind'] != 'card-instance-tree-fixture-originals'
                    or canonical(raw['source_origin']) != canonical(origin())):
                raise ContractError('Tree3 trained fit requires exact declared fixture source origin')

    def require_model(self, model: Any, *, pure: bool = False) -> None:
        expected = ((TreeArchitectureV3 if self.modern else TreeArchitectureV2) if pure else
                    (ConfiguredTreeModelV3 if self.modern else ConfiguredTreeModelV2))
        if type(model) is not expected:
            raise ContractError('closed Tree model differs from explicit record context')


TREE2_RECORDS = _TreeRecords(False)
TREE3_RECORDS = _TreeRecords(True)
TREE3_GAME_RECORDS = _TreeRecords(True, True)
TREE3_PACKED_GAME_RECORDS = _TreeRecords(True, True, True)


def _records(value: _TreeRecords) -> _TreeRecords:
    if value is not TREE2_RECORDS and value is not TREE3_RECORDS and value is not TREE3_GAME_RECORDS and value is not TREE3_PACKED_GAME_RECORDS:
        raise ContractError('explicit closed Tree record context required')
    return value

def game_records(raw: Json) -> _TreeRecords:
    """Closed source-value dispatch only. It grants no Source or fit authority."""
    from gf_common.card_instance_game_tree import PACKED_SOURCE_KIND, SOURCE_KIND
    kind = raw.get('source_kind')
    if kind == PACKED_SOURCE_KIND:
        context = TREE3_PACKED_GAME_RECORDS
    elif kind == SOURCE_KIND:
        context = TREE3_GAME_RECORDS
    else:
        raise ContractError('exact known game Tree source subtype required')
    context.source(raw)
    return context


ENVELOPE_RECIPE='tree-fit-model-envelope-v1'


def model_envelope(model:Any,rounds:int,*,_context:_TreeRecords=TREE2_RECORDS)->Json:
    context=_records(_context);context.require_model(model,pure=True)
    integer(rounds,1,2**31-1)
    width=context.features.feature_width(model.encoder.max_positions);leaves=model.network.num_leaves
    nodes=rounds*(2*leaves-1);size=65536+512*width+1024*nodes
    if size>MAX_ARRAY_BYTES:raise ContractError('pre-admitted Tree2 model envelope exceeds64MiB')
    return {'recipe':ENVELOPE_RECIPE,'feature_width':width,'rounds':rounds,'num_leaves':leaves,'node_bound':nodes,'text_bytes':size}


def validate_envelope(value:Any,model:Any,rounds:int,*,_context:_TreeRecords=TREE2_RECORDS)->Json:
    closed(value,{'recipe','feature_width','rounds','num_leaves','node_bound','text_bytes'})
    expected=model_envelope(model,rounds,_context=_context)
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


def validate_request(value:Any,*,_context:_TreeRecords=TREE2_RECORDS)->Json:
    context=_records(_context)
    raw=closed(value,REQUEST_FIELDS|context.extra_fields|context.request_fields)
    context.source(raw)
    if context.game:
        from gf_common.card_instance_game_tree import checked_memory, checked_shared
        checked_shared(raw['shared_generation'])
        checked_memory(raw['managed_memory'], child_limit_bytes=raw['execution']['resources']['max_owned_bytes'])
    elif context.modern:
        _fixture_construction_cost(raw['shared_fixture_construction'])
    if type(raw['schema_version']) is not int or raw['schema_version']!=context.schema or raw['kind']!='structured-tree-fit-request':
        raise ContractError('Tree2 immutable fit request3 required')
    import re
    if type(raw['run_id']) is not str or re.fullmatch('[0-9a-f]{32}',raw['run_id']) is None:raise ContractError('Tree2 training run identity invalid')
    model=context.architecture(raw['model']);execution=closed(raw['execution'],{'inference','resources'})
    context.configured({**asdict(model),'checkpoint':None,**execution})
    fit=validate_fit_settings(raw['fit_settings']);validate_envelope(raw['model_envelope'],model,fit['num_boost_round'],_context=context)
    integer(raw['seed'],0,2**31-1);context.binding(raw['input_binding'])
    for key in ('feature_signature','dataset_manifest_sha256','dataset_content_sha256','source_set_sha256'):_sha(raw[key])
    if raw['label_recipe']!=context.label:raise ContractError('Tree2 selected natural label recipe required')
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


def _fixture_construction_cost(value:Any)->None:
    import math
    raw=closed(value,{'seconds','cost_key','scope'})
    _sha(raw['cost_key'])
    if (type(raw['seconds']) not in (int,float) or not math.isfinite(raw['seconds']) or raw['seconds']<0
            or raw['scope']!='original fixture construction once; excludes PPO, per-fit preparation and fitting'):
        raise ContractError('Tree3 exact original fixture construction cost receipt required')


@dataclass(frozen=True)
class TrainedProvenance:
    """Canonical immutable owned provenance; no mutable nested dict alias."""
    encoded: bytes

    @classmethod
    def parse(cls,value:Any)->TrainedProvenance:
        return _parse_provenance(cls,value,TREE2_RECORDS)

    def to_dict(self)->Json:
        import json
        return json.loads(self.encoded)


@dataclass(frozen=True)
class TrainedProvenanceV4(TrainedProvenance):
    @classmethod
    def parse(cls,value:Any)->TrainedProvenanceV4:
        return _parse_provenance(cls,value,TREE3_RECORDS)


def _parse_provenance(cls:Any,value:Any,context:_TreeRecords)->Any:
    import re
    raw=closed(value,PROVENANCE_FIELDS|context.extra_fields)
    expected_kind=('structured-fixed-dataset4-fit' if context.packed else
        'structured-fixed-dataset3-fit' if context.modern else 'structured-fixed-dataset2-fit')
    if type(raw['schema_version']) is not int or raw['schema_version']!=context.schema or raw['kind']!=expected_kind:
        raise ContractError('closed trained provenance/context mismatch')
    if type(raw['training_run_id']) is not str or re.fullmatch('[0-9a-f]{32}',raw['training_run_id']) is None:
        raise ContractError('Tree trained run identity invalid')
    for key in ('request_sha256','dataset_manifest_sha256'):_sha(raw[key])
    context.binding(raw['input_binding']);context.source(raw)
    return cls(canonical(raw).encode())


class GameTrainedProvenanceV4(TrainedProvenanceV4):
    @classmethod
    def parse(cls, value: Any) -> GameTrainedProvenanceV4:
        return _parse_provenance(cls, value, TREE3_GAME_RECORDS)


class PackedGameTrainedProvenanceV5(TrainedProvenanceV4):
    @classmethod
    def parse(cls, value: Any) -> PackedGameTrainedProvenanceV5:
        return _parse_provenance(cls, value, TREE3_PACKED_GAME_RECORDS)


def _provenance(value:Any,context:_TreeRecords)->TrainedProvenance:
    return (PackedGameTrainedProvenanceV5 if context.packed else GameTrainedProvenanceV4 if context.game else TrainedProvenanceV4 if context.modern else TrainedProvenance).parse(value)


def provenance_for_request(request:Json,request_sha:str,*,_context:_TreeRecords=TREE2_RECORDS)->TrainedProvenance:
    context=_records(_context);validate_request(request,_context=context);_sha(request_sha)
    value={'schema_version':context.schema,'kind':('structured-fixed-dataset4-fit' if context.packed else
        'structured-fixed-dataset3-fit' if context.modern else 'structured-fixed-dataset2-fit'),
        'training_run_id':request['run_id'],'request_sha256':request_sha,'dataset_manifest_sha256':request['dataset_manifest_sha256'],
        'input_binding':request['input_binding'],**{key:request[key] for key in context.extra_fields}}
    return _provenance(value,context)


def fit_admission(model:Any,fit:Json,rows:Json,*,retained_bytes:int,_context:_TreeRecords=TREE2_RECORDS)->Json:
    context=_records(_context);context.require_model(model)
    fit=validate_fit_settings(fit);integer(retained_bytes,0,2**63-1)
    width=context.features.feature_width(model.encoder.max_positions);envelope=model_envelope(model.architecture(),fit['num_boost_round'],_context=context)
    closed(rows,{'train','validation'})
    for count in rows.values():integer(count,1,2**63-1)
    if any(count*4>model.resources.max_array_bytes for count in rows.values()):raise ContractError('Tree2 fit label array exceeds admitted64MiB tensor bound')
    raw=sum(rows.values())*(width*4+4)
    threads=max(model.inference.num_threads,fit['num_threads'])
    fixed=256*1024**2+threads*64*1024**2+16*envelope['text_bytes']
    packing=model.inference.candidate_chunk*(12*width+32)+(context.features.context_width+18*context.features.card_width)*4+65536
    total=retained_bytes+5*raw+fixed+packing
    if context.game:
        arrays=2*sum((count+model.inference.candidate_chunk-1)//model.inference.candidate_chunk for count in rows.values())
        # Exact configured controls + conservative existing4096B-per-shard
        # metadata/document overlap; no Source paths or observed maxima.
        control=len(canonical({'model':asdict(model.architecture()),'execution':{'inference':asdict(model.inference),
            'resources':asdict(model.resources)},'fit':fit,'rows':rows,'envelope':envelope}).encode())
        own=32*(control+4096*arrays)+4*65536
        child=5*raw+fixed+packing+max(own,retained_bytes)
        if child>model.resources.max_owned_bytes:
            raise ContractError('game Tree CHILD-LOCAL5raw/fixed/packing/control envelope exceeds configured cap')
        return {'raw_bytes':raw,'model_envelope':envelope,'fixed_bytes':fixed,'packing_bytes':packing,
            'retained_bytes':retained_bytes,'total_bytes':retained_bytes+child,'native_threads':threads,
            'child_local_bytes':child,'child_own_bytes':own}
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
                          inference_sha256:str,model_sha256:str,_context:_TreeRecords=TREE2_RECORDS)->None:
    import math
    context=_records(_context)
    request=validate_request(request,_context=context);completion=closed(completion,COMPLETION_FIELDS|context.extra_fields)
    context.source(completion)
    for value in (request_sha256,inference_sha256,model_sha256):_sha(value)
    if (type(completion['schema_version']) is not int or completion['schema_version']!=context.schema
            or completion['kind']!='structured-tree-training' or completion['status']!='complete'):
        raise ContractError('complete healthy Tree2 Training3 required before binding')
    if completion['request_sha256']!=request_sha256:raise ContractError('Tree2 request/completion hash link differs')
    for key in ('run_id','input_binding','seed','dataset_manifest_sha256','dataset_content_sha256','source_set_sha256',
                'feature_signature','label_recipe','producer','runtime',*context.extra_fields):
        if canonical(completion[key])!=canonical(request[key]):raise ContractError('Tree2 request/completion identity mismatch: '+key)
    origin=_provenance(inference['provenance'],context)
    if inference['fit_origin']!='trained' or origin!=provenance_for_request(request,request_sha256,_context=context):
        raise ContractError('Tree2 trained inference provenance differs from immutable request')
    if canonical(inference['model'])!=canonical(request['model']) or inference['feature_signature']!=request['feature_signature']:
        raise ContractError('Tree2 request/inference architecture/features differ')
    creation=closed(inference['creation_execution'],{'inference','resources','tree_descriptor'})
    if canonical({key:creation[key] for key in ('inference','resources')})!=canonical(request['execution']):
        raise ContractError('Tree2 original creation execution differs from pre-fit request')
    classification={key:inference[key] for key in request['input_binding']}
    context.source(inference)
    if context.modern and any(canonical(inference[key])!=canonical(request[key]) for key in context.extra_fields):
        raise ContractError('Tree3 source origin promotion forbidden')
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
    costs=closed(completion['costs'],{'preparation_seconds','child_seconds','final_validation_seconds','cleanup_seconds','elapsed_seconds',context.shared_cost})
    for key in ('preparation_seconds','child_seconds','final_validation_seconds','cleanup_seconds','elapsed_seconds'):
        number=costs[key]
        if type(number) not in (int,float) or not math.isfinite(number) or number<0:raise ContractError('Tree2 finite measured costs required')
    if costs['elapsed_seconds']>7200:raise ContractError('Tree2 original per-fit7200-second budget exceeded')
    shared=closed(costs[context.shared_cost],{'seconds','cost_key','scope'});_sha(shared['cost_key'])
    if type(shared['seconds']) not in (int,float) or not math.isfinite(shared['seconds']) or shared['seconds']<0 or type(shared['scope']) is not str:
        raise ContractError('Tree2 shared generation accounting invalid')
    if context.modern and canonical(shared)!=canonical(request[context.shared_cost]):
        raise ContractError('Tree3 completion fixture construction cost differs from immutable request')
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


def validate_input_envelope(value:Any,request:Json,*,_context:_TreeRecords=TREE2_RECORDS)->tuple[Json,Any]:
    """Pure envelope/typed-descriptor link, before allocating native inputs."""
    from gf_models.sharded_arrays import validate_bundle_metadata
    context=_records(_context)
    raw=closed(value,{'schema_version','kind','action_config_hash','feature_signature','label_recipe','input_binding','row_counts','data'}|context.numeric_extra_fields)
    context.numeric_source(raw, request)
    if type(raw['schema_version']) is not int or raw['schema_version']!=context.numeric_schema or raw['kind']!='structured-tree-fit-inputs':
        raise ContractError('Tree2 closed numeric input envelope3 required')
    _sha(raw['action_config_hash'])
    for key in ('feature_signature','label_recipe','input_binding','row_counts',*context.numeric_extra_fields):
        if canonical(raw[key])!=canonical(request[key]):raise ContractError('Tree2 input/request common fields differ: '+key)
    model=context.configured({**request['model'],'checkpoint':None,**request['execution']})
    if context.layout(model.encoder)['feature_signature']!=raw['feature_signature']:
        raise ContractError('Tree2 numeric input recipe/architecture mismatch')
    admission=fit_admission(model,request['fit_settings'],request['row_counts'],retained_bytes=0,_context=context)
    descriptors=validate_bundle_metadata(raw['data'],max_decoded_bytes=admission['raw_bytes'])
    expected={};chunk=model.inference.candidate_chunk;width=context.features.feature_width(model.encoder.max_positions)
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


def inspect_fit_inputs(directory:Any,request:Json,scan:Any,*,retain:bool=False,_context:_TreeRecords=TREE2_RECORDS)->tuple[Json,dict[str,list[Any]],dict[str,list[Any]]]:
    """Verify bounded typed blocks; retain arrays only in the own numeric child."""
    import numpy as np

    from gf_models.arrays import load_arrays
    context=_records(_context)
    request=validate_request(request,_context=context)
    envelope=scan.object(directory/'inputs'/'manifest.json',request['input_files']['inputs/manifest.json'])
    envelope,model=validate_input_envelope(envelope,request,_context=context)
    fit_admission(model,request['fit_settings'],request['row_counts'],retained_bytes=scan.retained,_context=context)
    data=envelope['data'];scan.inventory(directory/'inputs',set(data['files'])|{'manifest.json'})
    blocks:dict[str,dict[int,Any]]={'train':{},'validation':{}};labels:dict[str,dict[int,Any]]={'train':{},'validation':{}}
    kept=0;layout=context.layout(model.encoder);limits=context.limits(model.encoder)
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
                     deadline:float=float('inf'),_context:_TreeRecords=TREE2_RECORDS)->PairedTraining:
    """Explicit Training3 root only; no ancestor search, Booster or Torch work."""
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan, _identity
    context=_records(_context)
    directory=Path(directory).absolute();scan=SealedScan(max_owned_bytes=max_owned_bytes,deadline=deadline)
    directories=_container(scan,directory,{'request.json':'file','training.json':'file','inputs':'directory','artifact':'directory',
        'child-result.json':'file','stdout.log':'file','stderr.log':'file'})
    # Only root/artifact/input metadata is immutable; outside ancestor ctimes may
    # vary without replacing the actual directory. Check those identities below.
    ancestor_ids={p:(s[0],s[1]) for p,s in directories.items() if p!=directory}
    directories={directory:directories[directory]}
    request=scan.object(directory/'request.json',None if expected_binding is None else expected_binding.request_sha256)
    completion=scan.object(directory/'training.json',None if expected_binding is None else expected_binding.completion_sha256)
    validate_request(request,_context=context)
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
    from gf_common.card_instance import parse_card_instance_contract
    from gf_models.structured_tree_artifacts import inspect_artifact
    model=context.configured({**request['model'],'checkpoint':None,**request['execution']})
    classification=request['input_binding']
    inference,text=inspect_artifact(directory/'artifact',expected_digest=inference_sha,model=model,
        action_config_hash=inference['action_config_hash'],action_contract=(parse_card_instance_contract if context.modern else parse_action_contract)(classification['action_contract']),
        purpose=classification['purpose'],synthetic_capacity=classification['synthetic_capacity'],deadline=deadline,_scan=scan,_context=context,
        _expected_source=None if not context.modern else context.source_identity(request))
    validate_fitted_parameters(text,request,model)
    validate_pair_records(request,completion,inference,request_sha256=request_sha,inference_sha256=inference_sha,model_sha256=model_sha,_context=context)
    envelope,_,_=inspect_fit_inputs(directory,request,scan,_context=context)
    if inference['action_config_hash']!=envelope['action_config_hash']:
        raise ContractError('Tree2 actual own action hash differs from verified numeric descriptor')
    expected={'request.json','artifact/inference.json','artifact/model.txt','child-result.json','stdout.log','stderr.log'}|set(request['input_files'])
    if set(completion['outputs']['files'])!=expected:raise ContractError('Tree2 completion exact file membership differs')
    for name,sha in completion['outputs']['files'].items():
        with scan.file(directory/name,sha):pass
    receipt=scan.object(directory/'child-result.json',completion['outputs']['files']['child-result.json'])
    closed(receipt,{'schema_version','kind','request_sha256','input_files','metrics','inference_sha256','model_sha256'}|context.numeric_extra_fields)
    context.numeric_source(receipt, request)
    if (type(receipt['schema_version']) is not int or receipt['schema_version']!=context.numeric_schema or receipt['kind']!='structured-tree-fit-result'
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


def resolve_checkpoint(directory:Any,*,max_owned_bytes:int=2*1024**3,deadline:float=float('inf'),_context:_TreeRecords=TREE2_RECORDS)->tuple[Any,str,TrainingBinding|None]:
    """Inspect only the explicitly supplied toy artifact or Training3 root."""
    import stat
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan
    context=_records(_context)
    directory=Path(directory).absolute()
    entries={}
    for name in ('inference.json','training.json'):
        try:info=(directory/name).lstat()
        except FileNotFoundError:continue
        if not stat.S_ISREG(info.st_mode):raise ContractError('Tree2 root discriminator must be a regular owned file')
        entries[name]=True
    if len(entries)!=1:raise ContractError('explicit Tree2 toy artifact/Training3 root is ambiguous or incomplete')
    if 'training.json' in entries:
        pair=inspect_training(directory,max_owned_bytes=max_owned_bytes,deadline=deadline,_context=context)
        return directory/'artifact',pair.completion['outputs']['inference_sha256'],pair.binding
    scan=SealedScan(max_owned_bytes=max_owned_bytes,deadline=deadline)
    _container(scan,directory,{'inference.json':'file','model.txt':'file'})
    manifest=scan.object(directory/'inference.json')
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version']!=context.inference_schema
            or manifest.get('kind')!='structured-tree-inference' or manifest.get('fit_origin')!='toy' or manifest.get('provenance') is not None):
        raise ContractError('bare trained/unknown Tree2 artifact cannot bind without explicit paired Training3 root')
    context.source(manifest,toy=True)
    scan.finalize()
    return directory,scan.files[directory/'inference.json'][1],None


from gf_common.owned_completion import _close_owned_descriptor, _invalidate_owned_completion


def copy_training_bundle(source:Any,target:Any,*,max_owned_bytes:int=2*1024**3,deadline:float=float('inf'),_context:_TreeRecords=TREE2_RECORDS)->TrainingBinding:
    """Copy unchanged paired bytes; own completion remains revocable until success."""
    import os
    import stat
    from pathlib import Path

    from gf_models.structured_tree_artifacts import _publish, _written
    context=_records(_context)
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
    pair=inspect_training(source,max_owned_bytes=max_owned_bytes,deadline=deadline,_context=context)
    check();target.mkdir(mode=0o700,exist_ok=False)
    root_info=target.lstat();root_identity=(root_info.st_dev,root_info.st_ino)
    root_fd=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    held:dict[str,int]={'':root_fd};identities={'':root_identity}
    owner=_CopiedCompletion(target,pair,ancestors,root_identity,deadline,max_owned_bytes,root_fd,_context=context)
    owner._retained_directories=held
    try:
        actual_root=os.fstat(root_fd)
        owner._retained_identities['']=(actual_root.st_dev,actual_root.st_ino)
        if (os.fstat(root_fd).st_dev,os.fstat(root_fd).st_ino)!=root_identity:
            raise ContractError('paired-copy original target root replaced')
        for name in ('inputs','artifact'):
            check()
            os.mkdir(name,mode=0o700,dir_fd=root_fd)
            created=os.stat(name,dir_fd=root_fd,follow_symlinks=False)
            fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
            held[name]=fd;identities[name]=(created.st_dev,created.st_ino)
            actual_fd=os.fstat(fd);owner._retained_identities[name]=(actual_fd.st_dev,actual_fd.st_ino)
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
                owner._copy_fd=fd;owner._copy_identity=snapshot[:2]
                try:
                    if actual!=expected:raise ContractError('paired copy changed immutable bytes')
                    _written(destination.parent,directory_fd,destination.name,fd,snapshot,expected,deadline)
                    pair.scan.charge(512+len(str(destination).encode()))
                    pair.scan.files[destination]=(snapshot,expected)
                    owned_directories()
                except BaseException as original:
                    try:owner._retire_file()
                    except BaseException as cleanup:  # noqa: BLE001 - keep initiating error and exact FD owner
                        vars(original)['_structured_copy_cleanup_owner']=owner
                        original.add_note('copied file retirement unconfirmed: '+str(cleanup))
                    raise
                owner._retire_file()
            pair.finalize();owned_directories()
        pair.finalize();owned_directories()
        with pair.scan.file(source/'training.json',pair.binding.completion_sha256) as (stream,info):
            pair.scan.charge(32*info.st_size+65536,temporary=True)
            raw=stream.read(info.st_size+1)
            if len(raw)!=info.st_size:raise ContractError('paired completion length changed')
        return owner.commit(raw)
    except BaseException as original:
        if not owner.exposed:
            try:owner._release()
            except BaseException as cleanup:  # noqa: BLE001 - retain original error and same descriptor owner
                vars(original)['_structured_copy_cleanup_owner']=owner
                original.add_note('copy descriptor retirement unconfirmed: '+str(cleanup))
        raise


class _CopiedCompletion:
    """Narrow models-level exact-copy owner; no training orchestration import."""
    def __init__(self,target:Any,pair:PairedTraining,ancestors:Any,root_identity:Any,deadline:float,limit:int,root_fd:int,*,_context:_TreeRecords=TREE2_RECORDS)->None:
        self.context=_records(_context)
        self.target,self.pair,self.ancestors,self.root_identity=target,pair,ancestors,root_identity
        self.deadline,self.limit=deadline,limit;self.fd:int|None=None;self.directory_fd:int|None=None
        self.snapshot:Any=None;self.exposed=False;self.revoked=False;self.root_fd=root_fd
        self._completion_fd_identity:Any=None;self._copy_fd:int|None=None;self._copy_identity:Any=None
        self._retained_directories:dict[str,int]={};self._retained_identities:dict[str,Any]={}
        self._uncertain_releases:list[tuple[str,int,Any,BaseException]]=[]
        self._commit_started=False

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
        copied=inspect_training(self.target,max_owned_bytes=remaining,deadline=self.deadline,_context=self.context)
        copied.finalize();self.pair.finalize();self._namespace()
        return copied.binding

    def _retire_file(self)->None:
        if self._copy_fd is not None:
            fd=self._copy_fd;self._copy_fd=None
            _close_owned_descriptor(self,fd,self._copy_identity,'copied-file')

    def _retire_directories(self)->None:
        for name in reversed(tuple(self._retained_directories)):
            fd=self._retained_directories.pop(name)
            _close_owned_descriptor(self,fd,self._retained_identities.get(name),'directory:'+name)

    def _release(self)->None:
        self._retire_file();self._retire_directories()
        # Keep completion revocable until every other descriptor has retired.
        if self.directory_fd is not None:
            fd=self.directory_fd;self.directory_fd=None
            _close_owned_descriptor(self,fd,self.root_identity,'completion-directory')
        if self.fd is not None:
            fd=self.fd;self.fd=None
            _close_owned_descriptor(self,fd,self._completion_fd_identity,'completion')
        if self._uncertain_releases:
            raise ContractError('terminal copy descriptor release remains unconfirmed; quarantined numbers are not retryable')

    def revoke(self)->None:
        import time
        if not self.revoked and self.exposed:
            if time.monotonic()>=self.deadline:raise ContractError('completion revocation unconfirmed: original copy deadline exhausted')
            _invalidate_owned_completion(self,self.target/'training.json')
            self.revoked=True
        self._release()  # Also retry retirement after an already-confirmed invalidation.

    def commit(self,raw:bytes)->TrainingBinding:
        import os
        import uuid

        from gf_models.structured_tree_artifacts import _identity, _publish, _written
        if self._commit_started or self._uncertain_releases:
            raise ContractError('copy publication is one-use; failed attempts are cleanup-only')
        self._commit_started=True
        temporary='copy-completion-'+uuid.uuid4().hex+'.tmp'
        try:
            self.directory_fd=os.dup(self.root_fd)
            # No outer finally may fail after a healthy completion is committed.
            self._retire_directories()
            self._namespace()
            self.fd,self.snapshot,sha=_publish(self.target,self.directory_fd,temporary,raw,self.deadline)
            self._completion_fd_identity=self.snapshot[:2]
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
            self._release()
        except BaseException as original:
            try:
                if self.exposed:self.revoke()
                else:self._release()
            except BaseException as cleanup:  # noqa: BLE001 - uncertain revocation/descriptor owner stays reachable
                vars(original)['_structured_copy_cleanup_owner']=self
                try:original.add_note('completion revocation unconfirmed: '+type(cleanup).__name__)
                except BaseException:pass  # noqa: BLE001, S110 - diagnostics cannot replace original failure
            raise
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


# Explicit corrected wrappers; legacy entry points keep their original discriminator.
def model_envelope_v3(model:TreeArchitectureV3,rounds:int)->Json:
    return model_envelope(model,rounds,_context=TREE3_RECORDS)


def validate_request_v3(value:Any)->Json:
    return validate_request(value,_context=TREE3_RECORDS)


def fit_admission_v3(model:ConfiguredTreeModelV3,fit:Json,rows:Json,*,retained_bytes:int)->Json:
    return fit_admission(model,fit,rows,retained_bytes=retained_bytes,_context=TREE3_RECORDS)


def inspect_training_v3(directory:Any,**options:Any)->PairedTraining:
    return inspect_training(directory,_context=TREE3_RECORDS,**options)


def resolve_checkpoint_v3(directory:Any,**options:Any)->tuple[Any,str,TrainingBinding|None]:
    return resolve_checkpoint(directory,_context=TREE3_RECORDS,**options)


def copy_training_bundle_v3(source:Any,target:Any,**options:Any)->TrainingBinding:
    return copy_training_bundle(source,target,_context=TREE3_RECORDS,**options)



def game_fit_admission(model: ConfiguredTreeModelV3, fit: Json, rows: Json, *,
                       retained_bytes: int, total_bytes: int, controller_extra_bytes: int = 0) -> Json:
    """Two independent equations: full configured child AND joint controller+child."""
    from gf_common.card_instance_game_tree import managed_memory
    local = fit_admission(model, fit, rows, retained_bytes=0, _context=TREE3_GAME_RECORDS)
    integer(controller_extra_bytes, 0, 2**63-1)
    # Conservative duplicate overlap is intentional: full declared child cap
    # PLUS controller5raw/fixed/packing, never an observed successful peak.
    controller = retained_bytes + 5*local['raw_bytes'] + local['fixed_bytes'] + local['packing_bytes'] + controller_extra_bytes
    memory = managed_memory(total_bytes, controller, model.resources.max_owned_bytes)
    return {**local, 'retained_bytes': retained_bytes, 'total_bytes': memory['reserved_peak_bytes'],
        'managed_memory': memory, 'controller_reserved_bytes': controller}


def validate_game_request(value: Any) -> Json:
    return validate_request(value, _context=game_records(value))



def _game_pair_context(directory: Any, options: Json) -> _TreeRecords:
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan
    path = Path(directory)/'request.json'
    if not path.exists():
        return TREE3_GAME_RECORDS  # Existing absent/bare-root refusal remains in the actual reader.
    scan = SealedScan(max_owned_bytes=options.get('max_owned_bytes', 2*1024**3), deadline=options.get('deadline', float('inf')))
    raw = scan.object(path)
    context = game_records(raw)
    validate_request(raw, _context=context)
    scan.finalize()
    return context


def inspect_training_game_v3(directory: Any, **options: Any) -> PairedTraining:
    """Retain ACTUAL healthy real pair custody; serialized binding is not its owner."""
    return inspect_training(directory, _context=_game_pair_context(directory,options), **options)


def resolve_checkpoint_game_v3(directory: Any, **options: Any) -> tuple[Any, str, TrainingBinding]:
    artifact, inference_sha256, binding = resolve_checkpoint(directory, _context=_game_pair_context(directory,options), **options)
    if type(binding) is not TrainingBinding:
        raise ContractError('real game Tree requires a trained complete pair')
    return artifact, inference_sha256, binding
