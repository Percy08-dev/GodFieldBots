"""One owned native TreeDataset2 worklist, actual seals/numeric authorization.

This component owner grants no real-I/O, fitting or qualified-Source authority.
It never runs a Python Source reader underneath the native traversal.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from gf_common.action_contract import parse_action_contract
from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_common.schedule import serialize_schedule
from gf_common.structured import Selection
from gf_common.structured_tokens import (
    CandidateTokens,
    FrameTokens,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.sharded_arrays import _pairs
from gf_models.structured_binding import recorded_policy_label
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_sealed import SealedScan
from gf_training.collection import validation_groups
from gf_training.native_structured_session import _verify_parent_controls
from gf_training.structured_rollout import _owned_bytes, _read_recorded_rollout

CONTROL_FIELDS = {'schema_version','kind','scope','qualified','parent_qualification','control_fingerprint',
                  'input_binding','tree_projection','parents','worklist'}
CURRENT_FIELDS = {'schema_version','kind','control_fingerprint','work_index','parent_index','split','collection_kind',
    'generation','manifest_sha256','source_set_sha256','schedule_sha256','bindings_sha256','original_correspondence',
    'numeric','selected_roles','counts','tree_projection','elapsed_seconds'}


@dataclass(frozen=True)
class SelectedInput:
    frame: FrameTokens
    candidate: CandidateTokens
    original_selection: Selection
    label: float


class NativeTreeDataset:
    def __init__(self,plan:Json,settings:PreparationConfig,*,max_owned_bytes:int,deadline:float,
                 extra_owned_bytes:int=0,max_scratch_bytes:int|None=None,native:Any=None)->None:
        self.settings=preparation_config(asdict(settings))
        if self.settings.backend!='rust' or self.settings.reader!='structured-v1':
            raise ContractError('explicit Rust structured TreeDataset consumer required')
        self.plan={**plan,'parents':[{**entry,'directory':str(Path(entry['directory']).absolute()),
            'validation_directory':str(Path(entry['validation_directory']).absolute())} for entry in plan['parents']]}
        self.native=native
        self.limit=min(max_owned_bytes,self.settings.memory_budget_mib*1024**2)
        self.deadline=min(deadline,time.monotonic()+self.settings.wall_seconds-self.settings.cleanup_seconds)
        self.base=65536+_owned_bytes((plan,self.settings));self.extra=extra_owned_bytes
        self.budget=self.limit-self.base-self.extra
        self.scratch=self.settings.scratch_budget_mib*1024**2
        if max_scratch_bytes is not None:
            integer(max_scratch_bytes,0);self.scratch=min(self.scratch,max_scratch_bytes)
        if self.budget<=0 or self.scratch<=0:
            raise ContractError('TreeDataset caller memory/scratch exhausted before I/O')
        self._locks=ExitStack();self._executor:Any=None;self._session:Any=None
        self._session_closed=False;self._closed=False
        self.summary:Json|None=None;self.current:Json|None=None;self.sealed:SealedScan|None=None
        self.models:dict[int,Any]={};self.schedules:dict[tuple[int,int],str]={};self.registries:dict[tuple[int,int],str]={}
        self.collections:list[Json]=[];self._position=0;self._exhausted=False;self._final:Json|None=None
        self._selected:Any=None
        self._selection_complete=False
        self._sink_bytes=0
        self._active_sample:SelectedInput|None=None

    def _check(self)->None:
        if self._closed:raise ContractError('TreeDataset owner is closed')
        if time.monotonic()>=self.deadline:raise TimeoutError('TreeDataset owned deadline')

    def _held(self,extra:int=0)->int:
        return ((self.sealed.retained if self.sealed is not None else 0)+self._sink_bytes
            +_owned_bytes((self.models,self.schedules,self.registries,self.collections,self._selected,self._active_sample))+extra)

    def reserve_sink(self,total_bytes:int)->None:
        """Update total external/sink ownership BEFORE its caller allocation.

        This is resource admission, not Source authorization. Release it only
        after those caller-owned buffers/files no longer occupy memory.
        """
        integer(total_bytes,0,self.budget)
        if self._session is None:raise ContractError('open TreeDataset before growing retained sink ownership')
        self._sink_bytes=total_bytes
        self._call('recheck')

    def _call(self,name:str,*arguments:Any,extra:int=0)->Any:
        self._check()
        return self._executor.call(getattr(self._session,name),*arguments,self._held(extra))

    def __enter__(self)->Self:
        from gf_training.lineage import _OWNED_LOCKS, run_lock
        from gf_training.native_backend import binary_identity, load_native
        from gf_training.native_execution import NativeExecution
        try:
            self._check();self.native=self.native or load_native()
            if not callable(getattr(self.native,'TreeDatasetSession',None)):
                raise ContractError('native binary lacks concrete TreeDatasetSession')
            self.identity=binary_identity(self.native)
            self.base+=_owned_bytes(self.identity);self.budget=self.limit-self.base-self.extra
            if self.budget<=0:raise ContractError('TreeDataset control/binary ownership exhausted')
            paths={Path(entry[key]).resolve() for entry in self.plan['parents'] for key in ('directory','validation_directory')}
            for directory in sorted(paths):
                if directory not in _OWNED_LOCKS:self._locks.enter_context(run_lock(directory))
            settings=canonical({'memory_bytes':self.budget,'scratch_bytes':self.scratch,
                'max_source_file_bytes':self.settings.max_source_file_mib*1024**2,
                'max_record_bytes':self.settings.max_record_mib*1024**2,
                'chunk_target_bytes':min(self.settings.chunk_target_mib*1024**2,64*1024**2)}).encode()
            cancel=self.native.CancelHandle(self.deadline)
            self._executor=NativeExecution(cancel,thread_name_prefix='gf-tree-dataset')
            def construct()->None:
                self._session=self.native.TreeDatasetSession(canonical(self.plan).encode(),settings,self.deadline,cancel)
            self._executor.call(construct)
            self.summary=json.loads(self._call('control_summary'),object_pairs_hook=_pairs)
            summary=closed(self.summary,CONTROL_FIELDS)
            if (type(summary['schema_version']) is not int or summary['schema_version']!=2
                    or summary['kind']!='structured-tree-dataset-controls' or summary['scope']!='engineering-component-originals'
                    or summary['qualified'] is not False or summary['parent_qualification']!='not_performed'
                    or canonical(summary['input_binding'])!=canonical({key:self.plan[key] for key in ('action_contract','purpose','synthetic_capacity')})):
                raise ContractError('closed component TreeDataset control summary required')
            remaining=self.budget-self._session.used_bytes()
            if remaining<=0:raise ContractError('joint TreeDataset/sealed ownership exhausted before decode')
            self.sealed=SealedScan(max_owned_bytes=remaining,deadline=self.deadline)
            contract=parse_action_contract(summary['input_binding']['action_contract'])
            parents,self.models,self.schedules,self.registries=_verify_parent_controls(summary['parents'],contract,self.sealed)
            acks=[{**{key:p[key] for key in ('parent_index','config_sha256','resume_identity','controls_sha256')},
                   'sealed_sha256':digest(p['sealed'])} for p in parents]
            self.sealed.finalize()
            auth={'schema_version':2,'kind':'structured-tree-dataset-controls-authorization',
                'control_fingerprint':summary['control_fingerprint'],'tree_projection_sha256':digest(summary['tree_projection']),
                'parents':acks,'worklist_sha256':digest(summary['worklist'])}
            self._call('authorize_controls',canonical(auth).encode(),extra=_owned_bytes(auth))
            return self
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:  # noqa: BLE001 - preserve initiating error and reachable cleanup owner
                error.add_note('TreeDataset entry cleanup: '+type(cleanup).__name__)
                if getattr(cleanup,'_structured_source_cleanup_owner',None) is self:
                    vars(error)['_structured_source_cleanup_owner']=self
            raise

    def _validation_conditions(self,parent_index:int)->tuple[str,str]:
        assert self.summary is not None
        parent=self.summary['parents'][parent_index];generation=parent['generations'][-1]
        model=self.models[parent_index];binding=InferenceBindingV2(**parent['execution']['inference']['binding'])
        visibility=parent['execution']['inference']['cuda_visible_devices']
        label=recorded_policy_label(model,binding,generation['inference_sha256'],cuda_visible_devices=visibility)
        contract=parse_action_contract(self.summary['input_binding']['action_contract'])
        options={'version':'2','checkpoint':str(Path(parent['directory'])/'artifacts'/f"{generation['generation']:06}"),
            'expected_manifest_digest':generation['inference_sha256'],'model':asdict(model),'inference':asdict(binding),
            'cuda_visible_devices':visibility,'action_contract':asdict(contract),'purpose':'component','synthetic_capacity':False}
        registry={label:RuntimePolicyBinding(model.model_id,options),'random-structured-v1':RuntimePolicyBinding('random'),
                  'heuristic-structured-v1':RuntimePolicyBinding('heuristic')}
        groups=validation_groups(parent['state']['seed'],label,actions_version=2)
        return digest(serialize_schedule(groups,4)),digest({name:asdict(value) for name,value in registry.items()})

    def stage_next(self)->bool:
        if self.current is not None or self._exhausted:raise ContractError('TreeDataset current inputs must retire before one advance')
        ready=self._call('stage_next')
        if type(ready) is not bool:raise ContractError('TreeDataset stage result must be boolean')
        if not ready:
            assert self.summary is not None
            if self._position!=len(self.summary['worklist']):raise ContractError('TreeDataset exhausted a partial worklist')
            self._exhausted=True;return False
        self.current=json.loads(self._call('collection_summary'),object_pairs_hook=_pairs)
        current=closed(self.current,CURRENT_FIELDS);assert self.summary is not None
        if (type(current['schema_version']) is not int or current['schema_version']!=2
                or current['kind']!='structured-tree-dataset-collection' or current['work_index']!=self._position
                or current['control_fingerprint']!=self.summary['control_fingerprint']
                or canonical(current['tree_projection'])!=canonical(self.summary['tree_projection'])):
            raise ContractError('TreeDataset current collection/control mismatch')
        work=self.summary['worklist'][self._position]
        for key in ('parent_index','collection_kind','generation','manifest_sha256','numeric'):
            if canonical(current[key])!=canonical(work[key]):raise ContractError('TreeDataset original work membership changed')
        pi,gi=current['parent_index'],current['generation']
        if current['collection_kind']=='ppo_batch':
            expected=(self.schedules[pi,gi],self.registries[pi,gi]);numeric=current['numeric']
            if current['split']!='train' or numeric is None:raise ContractError('train work requires original numeric cohort')
        elif current['collection_kind']=='ppo_validation':
            expected=self._validation_conditions(pi);numeric=None
            if current['split']!='validation' or current['numeric'] is not None:raise ContractError('validation numeric must be explicitly null')
        else:raise ContractError('unknown TreeDataset collection kind')
        if expected!=(current['schedule_sha256'],current['bindings_sha256']):
            raise ContractError('TreeDataset original schedule/final-policy binding differs')
        if numeric is not None:
            parent=self.summary['parents'][pi];declaration=parent['semantics']
            if digest(declaration)!=current['original_correspondence']['semantics_sha256']:
                raise ContractError('TreeDataset original numerical declaration mismatch')
            self._call('recheck')
            rollout=_read_recorded_rollout(Path(numeric['directory']),numeric['sha256'],self.models[pi],
                action_contract=parse_action_contract(self.summary['input_binding']['action_contract']),
                declaration=declaration,expected_records=None,expected_correspondence=current['original_correspondence'],
                max_owned_bytes=self.budget,extra_owned_bytes=self._session.used_bytes(),check=self._check,
                reserve=lambda size:self._reserve(size))
            rollout.require_learning()
            if len(rollout)!=current['counts']['selected_decisions']:raise ContractError('TreeDataset recorded selected count differs')
            del rollout
        self._call('recheck')
        auth={key:current[key] for key in ('schema_version','control_fingerprint','work_index','parent_index',
            'manifest_sha256','source_set_sha256','schedule_sha256','bindings_sha256')}
        auth.update(kind='structured-tree-dataset-collection-authorization',selected_roles_sha256=digest(current['selected_roles']),
            correspondence_sha256=current['original_correspondence']['binding_sha256'],numeric_sha256=None if numeric is None else numeric['sha256'],
            tree_projection_sha256=digest(current['tree_projection']))
        self._call('authorize_collection',canonical(auth).encode(),extra=_owned_bytes(auth))
        current['tree_projection']=self.summary['tree_projection']
        self._selection_complete=False
        return True

    def _reserve(self,size:int)->None:
        self._check()
        if size+self._session.used_bytes()>self.budget:raise ContractError('joint TreeDataset/numeric reservation exceeded')

    def selected_inputs(self)->Iterator[SelectedInput]:
        if self.current is None or self._selected is not None:raise ContractError('one current selected-input export required')
        assert self.summary is not None
        payload=self._call('export_current')
        metadata,arrays=_selected_payload(payload)
        if (metadata['work_index']!=self._position or metadata['control_fingerprint']!=self.summary['control_fingerprint']
                or canonical(metadata['tree_projection'])!=canonical(self.summary['tree_projection'])):
            raise ContractError('TreeDataset selected payload belongs to different work')
        self._selected=(metadata,arrays)
        try:
            seen=0
            for episode,role in zip(metadata['episodes'],self.current['selected_roles'],strict=True):
                if (episode['actor']!=role['viewer'] or episode['end_kind']!='natural'
                        or canonical(episode['outcome'])!=canonical(role['outcome']) or len(episode['frames'])!=role['accepted_count']):
                    raise ContractError('TreeDataset selected episode/closure/count differs')
                for index,frame in enumerate(episode['frames']):
                    prefix=frame['token_prefix'];events=frame['event_token_prefix']
                    elements=sum(arrays[prefix+'.'+name].size for name in (
                        'payload','state_numerics','state_present','state_categories','state_category_present','card_numerics',
                        'card_present','card_categories','card_mask','opponent_learned','history',
                        *(f'candidate.{field}' for field in ('kinds','targets','numerics','present','offsets','slots'))))
                    elements+=sum(arrays[events+'.'+name].size for name in ('interval_numerics','interval_present','interval_categories'))
                    self._reserve(128*elements+65536)
                    sample=_selected_input(episode,frame,episode['original_selections'][index],arrays,metadata['tree_projection'])
                    self._active_sample=sample
                    self._reserve(_owned_bytes(sample))
                    seen+=1;yield sample
                    self._active_sample=None
                    del sample
            if seen!=self.current['counts']['selected_decisions']:raise ContractError('TreeDataset selected payload lost decisions')
            self._selection_complete=True
        finally:
            self._selected=None
            self._active_sample=None
            arrays.clear();del arrays,metadata,payload
        # A consumer must close/exhaust the iterator before this call; no NumPy
        # borrowed arrays or a second full reconstructed Source survives it.

    def retire_current(self)->None:
        if self.current is None or self._selected is not None or not self._selection_complete:raise ContractError('TreeDataset all selected inputs/views must retire first')
        self._call('retire_current')
        receipt={key:value for key,value in self.current.items() if key!='tree_projection'}
        receipt['tree_projection_sha256']=digest(self.current['tree_projection'])
        self.collections.append(receipt);self.current=None;self._position+=1

    def verify_complete(self)->Json:
        if not self._exhausted or self.current is not None:raise ContractError('TreeDataset full worklist required')
        assert self.sealed is not None
        self.sealed.finalize()
        result=json.loads(self._call('verify_complete'),object_pairs_hook=_pairs)
        self.sealed.finalize()
        return result

    def finalize(self)->Json:
        if self._final is None:
            self.verify_complete()
            self._final=json.loads(self._call('finalize'),object_pairs_hook=_pairs)
        return self._final

    def close(self)->None:
        if self._closed:return
        failure=None
        if self._executor is not None and self._session is not None and not self._session_closed:
            try:self._executor.call(self._session.close);self._session_closed=True
            except BaseException as error:  # noqa: BLE001 - join owned native work before propagation
                failure=error
                try:self._executor.call(self._session.close);self._session_closed=True
                except BaseException as cleanup:  # noqa: BLE001 - preserve initiating error and reachable cleanup owner
                    failure.add_note('TreeDataset unconfirmed native cleanup: '+type(cleanup).__name__)
                    vars(failure)['_structured_source_cleanup_owner']=self;raise failure
        if self._executor is not None:
            try:self._executor.close()
            except BaseException as error:  # noqa: BLE001 - join owned native work before propagation
                if failure is None:failure=error
                else:failure.add_note('TreeDataset executor shutdown: '+type(error).__name__)
                if not self._executor._closed or self._executor._active is not None:
                    vars(failure)['_structured_source_cleanup_owner']=self;raise failure
        self._locks.close();self._closed=True
        if failure is not None:raise failure

    def __exit__(self,kind:object,value:object,traceback:object)->None:
        failure=None
        if value is None:
            try:self.finalize()
            except BaseException as error:  # noqa: BLE001 - join owned native work before propagation
                failure=error
        try:self.close()
        except BaseException as cleanup:
            original=value if isinstance(value,BaseException) else failure
            if original is None:raise
            original.add_note('TreeDataset cleanup: '+type(cleanup).__name__)
            if getattr(cleanup,'_structured_source_cleanup_owner',None) is self:
                vars(original)['_structured_source_cleanup_owner']=self
        if failure is not None:raise failure


def _selected_payload(payload:Any)->tuple[Json,dict[str,Any]]:
    import math

    import numpy as np
    if type(payload) is not tuple or len(payload)!=3 or any(type(value) is not tuple for value in payload[1:]):
        raise ContractError('TreeDataset exact3tuple U8/I64 selected-input ABI required')
    metadata=json.loads(payload[0],object_pairs_hook=_pairs)
    closed(metadata,{'schema_version','kind','control_fingerprint','work_index','tree_projection','episodes','arrays'})
    if type(metadata['schema_version']) is not int or metadata['schema_version']!=1 or metadata['kind']!='structured-tree-selected-inputs':
        raise ContractError('TreeDataset selected-input metadata kind/version mismatch')
    banks={'uint8':payload[1],'int64':payload[2]}
    for dtype,values in banks.items():
        for value in values:
            if (type(value) is not np.ndarray or value.dtype!=np.dtype(dtype) or value.ndim!=1
                    or value.flags.writeable or not value.flags.c_contiguous or value.nbytes>64*1024**2):
                raise ContractError('TreeDataset bank dtype/readonly/size mismatch')
    names=set();ranges:dict[tuple[str,int],list[tuple[int,int]]]={};plans=[]
    for item in metadata['arrays']:
        closed(item,{'key','dtype','shape','bank_index','element_offset','element_count'})
        key,dtype=item['key'],item['dtype']
        if type(key) is not str or not key or key in names or dtype not in banks:
            raise ContractError('TreeDataset unique typed array names required')
        names.add(key)
        shape=item['shape']
        if type(shape) is not list or len(shape)>8:raise ContractError('TreeDataset array shape required')
        for size in shape:integer(size,0,64*1024**2)
        bank=integer(item['bank_index'],0,len(banks[dtype])-1)
        start=integer(item['element_offset'],0,len(banks[dtype][bank]))
        count=integer(item['element_count'],0,len(banks[dtype][bank])-start)
        if count!=math.prod(shape) or count*np.dtype(dtype).itemsize>64*1024**2:
            raise ContractError('TreeDataset array bounds/shape mismatch')
        ranges.setdefault((dtype,bank),[]).append((start,start+count))
        plans.append((key,dtype,bank,start,count,shape))
    for spans in ranges.values():
        end=0
        for start,stop in sorted(spans):
            if start<end and start!=stop:raise ContractError('TreeDataset array descriptors overlap')
            end=max(end,stop)
    arrays={key:banks[dtype][bank][start:start+count].reshape(shape) for key,dtype,bank,start,count,shape in plans}
    return metadata,arrays


def _selected_input(episode:Json,frame:Json,selection:Json,arrays:dict[str,Any],projection:Json)->SelectedInput:
    from gf_training._raw_token_arrays import raw_token_tuple
    from gf_training.journal import _reward
    def row(key:str,boolean:bool=False)->tuple[Any,...]:
        return raw_token_tuple(arrays[key], boolean=boolean)
    prefix=frame['token_prefix'];event=frame['event_token_prefix']
    fields={name:row(prefix+'.'+name,'present' in name or 'mask' in name) for name in (
        'state_numerics','state_present','state_categories','state_category_present','card_numerics','card_present',
        'card_categories','card_mask','opponent_learned')}
    fields.update({name:row(event+'.'+name,'present' in name) for name in ('interval_numerics','interval_present','interval_categories')})
    fields['history']=row(prefix+'.history')
    tokens=FrameTokens(frame['frame_hash'],projection['token_signature'],projection['encoder']['aggregate_window'],
        episode['actor'],episode['episode'],frame['revision'],frame['interval_start'],frame['boundary']['end'],**fields)
    candidates=CandidateTokens(frame['table_hash'],frame['frame_hash'],projection['token_signature'],
        **{name:row(prefix+'.candidate.'+name,name=='present') for name in ('kinds','targets','numerics','present','offsets','slots')})
    validate_frame_tokens(tokens);validate_candidate_tokens(candidates,tokens,projection['encoder']['max_positions'])
    original=Selection(**closed(selection,{'table_hash','index','policy_version','logprob','value'}))
    if (len(candidates.kinds)!=1 or frame['selected_index']!=0 or original.index!=frame['original_selection_index']
            or original.table_hash!=frame['table_hash']):
        raise ContractError('TreeDataset selected row/local0/original index binding mismatch')
    return SelectedInput(tokens,candidates,original,float(_reward(episode['outcome'],episode['actor'])))
