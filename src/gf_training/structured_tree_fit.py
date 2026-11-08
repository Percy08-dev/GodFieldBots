"""Concrete Dataset2-backed fresh Tree2 fitting; no arbitrary row/proof entry."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, canonical
from gf_models.structured_tree_config import ConfiguredTreeModelV2, ConfiguredTreeModelV3
from gf_models.structured_tree_encoder import PreparedTreeFrame, PreparedTreeFrameV3
from gf_models.structured_tree_pairing import (
    TREE2_RECORDS,
    TREE3_GAME_RECORDS,
    TREE3_RECORDS,
    _records,
    _TreeRecords,
    fit_admission,
    validate_fit_settings,
)
from gf_training.structured_tree_data import VerifiedTreeDataset


def _dataset_view(dataset:Any,context:_TreeRecords)->Json:
    raw=dataset.manifest
    if not context.modern or context.game:
        return raw
    from gf_common.domain import digest
    proof=raw['source']
    return {**raw,'tree_projection':proof['tree_projection'],'counts':proof['counts'],
        'input_binding':proof['input_binding'],'neural_cohort':{'actions_sha256':digest(proof['actions'])},
        'source_kind':proof['source_kind'],'source_origin':proof['origin'],
        'shared_fixture_construction':{'seconds':proof['construction_cost']['seconds'],
            'cost_key':digest({'fixture_manifest_sha256':proof['manifest_sha256'],'construction_cost':proof['construction_cost']}),
            'scope':'original fixture construction once; excludes PPO, per-fit preparation and fitting'}}


def _dataset_paths(dataset:Any,metadata:Json,context:_TreeRecords)->list[Any]:
    from pathlib import Path
    if context.modern:
        return [dataset.path.parent,*dataset.source.original_roots]
    return [dataset.path.parent]+[Path(c['directory']) for c in metadata['collections']]+[Path(p['directory']) for p in metadata['parents']]


def _fit_scratch_admission(prepared:Any,log_bytes:int)->int:
    if not prepared.context.modern:
        return 0
    from gf_common.domain import integer
    integer(log_bytes,1,2**63-1)
    chunk=prepared.model.inference.candidate_chunk
    arrays=2*sum((rows+chunk-1)//chunk for rows in prepared.rows.values())
    metadata=len(canonical(prepared.context.layout(prepared.model.encoder)).encode())
    # This Python fixture Source/index retains readonly originals and in-memory
    # snapshots, with no disk spool. Keep the entire private numeric population,
    # bounded model text, full feature/control metadata and both bounded logs.
    # A second complete copy covers writer/publication temporary overlap; do not
    # count a hardlink as a release or assume compression will reduce storage.
    persisted=(prepared.admission['raw_bytes']+prepared.admission['model_envelope']['text_bytes']
               +4*metadata+2*log_bytes+16*65536+4096*arrays)
    peak=2*persisted
    if peak>prepared.dataset.settings.scratch_budget_mib*1024**2:
        raise ContractError('Tree3 full Source/private-fit/publication scratch admission exceeded')
    return peak


def _admit(prepared:Any,*,extra_bytes:int=0)->Json:
    """SAME numeric envelope; game additionally holds one joint full-child cap."""
    dataset=prepared.dataset
    if not prepared.context.game:
        return fit_admission(prepared.model,prepared.fit,prepared.rows,
            retained_bytes=dataset.retained_without_sink()+extra_bytes,_context=prepared.context)
    from gf_models.structured_tree_pairing import game_fit_admission
    extra_bytes+=getattr(prepared,'_control_extra',0)
    result=game_fit_admission(prepared.model,prepared.fit,prepared.rows,
        retained_bytes=dataset.retained_without_sink(),total_bytes=dataset.limit,controller_extra_bytes=extra_bytes)
    dataset.reserve_fit(result['fixed_bytes']+result['packing_bytes']+extra_bytes)
    return {**result,'managed_memory':dataset.fit_memory()}


class _PreparedTreeFit:
    """Only this concrete active Source owner can create its private fit rows."""
    def __init__(self,dataset:Any,model:Any,fit_settings:Json,*,_context:_TreeRecords=TREE2_RECORDS)->None:
        from gf_training.card_instance_game_tree_data import VerifiedGameTreeDataset3
        from gf_training.card_instance_tree_data import VerifiedFixtureTreeDataset3
        context=_records(_context)
        expected=VerifiedGameTreeDataset3 if context.game else VerifiedFixtureTreeDataset3 if context.modern else VerifiedTreeDataset
        if type(dataset) is not expected or not dataset._active or dataset.source is None or dataset.manifest is None:
            raise ContractError('actual active concrete verified Dataset owner required; arrays/provenance cannot enter')
        context.require_model(model)
        if model.checkpoint is not None:
            raise ContractError('fresh configured Tree architecture required')
        metadata=_dataset_view(dataset,context)
        if canonical(metadata['tree_projection']['encoder'])!=canonical(asdict(model.encoder)):
            raise ContractError('Tree fit encoder differs from verified Dataset raw projection')
        if metadata['tree_projection']['feature_signature']!=context.layout(model.encoder)['feature_signature']:
            raise ContractError('Tree fit feature signature differs from verified Dataset')
        self.dataset:Any=dataset
        self.model,self.fit=model,validate_fit_settings(fit_settings)
        self.context,self.metadata=context,metadata
        self.rows={split:metadata['counts'][split]['selected_decisions'] for split in ('train','validation')}
        if context.game:
            from gf_training.card_instance_game_tree_data import _verified_fit_rows
            assert type(dataset) is VerifiedGameTreeDataset3
            self.rows=_verified_fit_rows(metadata,dataset._verified_rows)
        self.admission=_admit(self)
        self.blocks:dict[str,list[np.ndarray[Any,Any]]]={split:[] for split in self.rows}
        self.labels:dict[str,list[np.ndarray[Any,Any]]]={split:[] for split in self.rows}
        self.counts={split:0 for split in self.rows};self._complete=False
        self._packing_arrays:dict[str,Any]={}
        self._row_refs:list[Any]=[]

    def retire_rows(self)->None:
        """Game-only physical row retirement, before ANY sink/accounting release."""
        if not self.context.game or getattr(self,'_rows_retired',False):return
        import weakref
        self._row_refs.extend(weakref.ref(block) for values in (*self.blocks.values(),*self.labels.values()) for block in values)
        self._packing_arrays.clear()
        for values in (*self.blocks.values(),*self.labels.values()):values.clear()
        if any(reference() is not None for reference in self._row_refs):
            raise ContractError('game fit row/packing views remain alive; sink reservation cannot retire')
        self._row_refs.clear()
        self.dataset._retire_fit_sink(self)
        self._rows_retired=True

    def prepare(self)->None:
        if self._complete:raise ContractError('Tree2 private Source rows prepare exactly once')
        dataset=self.dataset;assert dataset.source is not None
        width=self.context.features.feature_width(self.model.encoder.max_positions)
        # Reserve the entire known selected population and Dataset copy allowance
        # before allocation. Never trim rows/chunk/threads to satisfy admission.
        dataset.reserve_sink(5*self.admission['raw_bytes'])
        worklist=dataset.selected_inputs()
        parts:Any=None;row:Any=None;frame:Any=None;sample:Any=None
        failure:BaseException|None=None
        try:
            for split,sample in worklist:
                self.admission=_admit(self)
                index=self.counts[split]
                if index>=self.rows[split]:raise ContractError('Source produced more selected rows than immutable Dataset2')
                chunk=self.model.inference.candidate_chunk
                if index%chunk==0:
                    size=min(chunk,self.rows[split]-index)
                    if size*(width*4+4)>64*1024**2:raise ContractError('private feature+label shard decoded size exceeds64MiB')
                    dataset.reserve_sink(5*self.admission['raw_bytes'])
                    self.blocks[split].append(np.empty((size,width),dtype=np.float32))
                    self.labels[split].append(np.empty(size,dtype=np.float32))
                source_owned=(dataset.source.current_feature_bytes if self.context.game else
                    dataset.source.retained_bytes if self.context.modern else
                    dataset.source.base+dataset.source.extra+dataset.source._session.used_bytes())
                frame=(PreparedTreeFrameV3 if self.context.modern else PreparedTreeFrame).create(
                    sample.tokens if self.context.modern else sample.frame,self.model,extra_owned_bytes=source_owned)
                parts=frame.iter_chunks(sample.selected_tokens() if self.context.modern else sample.candidate)
                start,row=next(parts)
                if start!=0 or row.shape!=(1,width):raise ContractError('Tree2 fit consumes only one actually selected candidate')
                self.blocks[split][-1][index%chunk]=row[0]
                self.labels[split][-1][index%chunk]=sample.label
                self.counts[split]+=1
                del parts,row,frame,sample
        except BaseException as error:
            failure=error
            raise
        finally:
            if self.context.game:
                # Clear the current row/sample before unwinding actual raw
                # borrows; a failed body must not pin views in its traceback.
                parts=row=frame=sample=None  # noqa: F841 - explicit traceback-held view retirement before borrow close
                try:worklist.close()
                except BaseException as cleanup:
                    if failure is None:raise
                    vars(failure)['_game_tree_dataset_cleanup_owner']=dataset
                    from gf_training.card_instance_game_tree_data import _note
                    _note(failure,'game original worklist retirement: ',cleanup)
        if self.counts!=self.rows:raise ContractError('Tree2 private preparation omitted original selected rows')
        dataset.verify_complete()
        self.admission=_admit(self)
        self._complete=True


def _prepare_request(prepared:_PreparedTreeFit,directory:Any,*,seed:int,deadline:float)->tuple[Json,str,Any]:
    """Persist only this concrete owner's actual private numeric blocks."""
    import hashlib
    import uuid
    from pathlib import Path

    from gf_common.domain import integer
    from gf_models.sharded_arrays import write_bundle
    from gf_models.structured_tree_artifacts import runtime_metadata
    from gf_models.structured_tree_pairing import (
        tree_fit_producer,
        validate_input_envelope,
        validate_request,
    )
    from gf_training.structured_lineage import _ChainScan, _write
    from gf_training.structured_tree_data import _check_output_ancestors, _outside_output
    if type(prepared) is not _PreparedTreeFit or not prepared._complete:raise ContractError('actual fully prepared verified Source rows required')
    integer(seed,0,2**31-1)
    dataset=prepared.dataset;assert dataset.manifest is not None and dataset.source is not None
    context=prepared.context;metadata=prepared.metadata
    _fit_scratch_admission(prepared,getattr(prepared,'log_bytes',1024**2))
    source_paths=_dataset_paths(dataset,metadata,context)
    directory,ancestors=_outside_output(Path(directory),source_paths)
    model=prepared.model
    arrays={f'{split}.{kind}.{index:06}':block for split in ('train','validation')
        for kind,values in (('x',prepared.blocks[split]),('y',prepared.labels[split])) for index,block in enumerate(values)}
    if context.game:prepared._packing_arrays=arrays
    # Existing bundle writer may hold a shard/packing temporary. Reserve its
    # conservative encoded scratch plus metadata while all Source owners remain.
    scratch=32*(prepared.admission['raw_bytes']+4096*len(arrays))+65536+4096*len(arrays)
    _admit(prepared,extra_bytes=scratch)
    dataset.reserve_sink(5*prepared.admission['raw_bytes']+(0 if context.game else scratch))
    directory.mkdir(mode=0o700)
    directory_identity=directory.stat()
    publisher:Any=_ChainScan(directory=directory,max_owned_bytes=(model.resources.max_owned_bytes if context.game else
        model.resources.max_owned_bytes-dataset.retained_without_sink()-5*prepared.admission['raw_bytes']-scratch),deadline=deadline)
    publisher._tree_fit_directory_identity=(directory_identity.st_dev,directory_identity.st_ino)
    publisher._tree_fit_ancestors=ancestors
    def check()->None:
        publisher.check();_check_output_ancestors(ancestors)
        _admit(prepared,extra_bytes=scratch+publisher.retained)
        dataset.reserve_sink(5*prepared.admission['raw_bytes']+(0 if context.game else scratch+publisher.retained))
    check()
    data=write_bundle(directory/'inputs','features',arrays,_check=check)
    envelope={'schema_version':context.schema,'kind':'structured-tree-fit-inputs',
        'action_config_hash':metadata['neural_cohort']['actions_sha256'],
        'feature_signature':metadata['tree_projection']['feature_signature'],'label_recipe':metadata['label_recipe'],
        'input_binding':metadata['input_binding'],'row_counts':prepared.rows,'data':data,
        **{key:metadata[key] for key in context.extra_fields}}
    check();envelope_sha=_write(directory/'inputs'/'manifest.json',envelope,scan=publisher,immutable=True)
    files={'inputs/'+name:sha for name,sha in data['files'].items()};files['inputs/manifest.json']=envelope_sha
    request={'schema_version':context.schema,'kind':'structured-tree-fit-request','run_id':uuid.uuid4().hex,
        'model':asdict(model.architecture()),'execution':{'inference':asdict(model.inference),'resources':asdict(model.resources)},
        'fit_settings':prepared.fit,'seed':seed,'input_binding':envelope['input_binding'],'feature_signature':envelope['feature_signature'],
        'label_recipe':envelope['label_recipe'],'dataset_manifest_sha256':dataset.expected_digest,
        'dataset_content_sha256':dataset.manifest['content_sha256'],'source_set_sha256':dataset.manifest['source_set_sha256'],
        'input_files':files,'row_counts':prepared.rows,'producer':tree_fit_producer(),
        'runtime':runtime_metadata(deadline=deadline),'model_envelope':prepared.admission['model_envelope'],
        **{key:metadata[key] for key in context.extra_fields},
        **({'shared_generation':metadata['shared_generation'],'managed_memory':dataset.fit_memory()} if context.game else
           {'shared_fixture_construction':metadata['shared_fixture_construction']} if context.modern else {})}
    validate_request(request,_context=context);validate_input_envelope(envelope,request,_context=context)
    check();request_sha=_write(directory/'request.json',request,scan=publisher,immutable=True)
    for name,sha in files.items():
        check()
        with publisher.file(directory/name,sha):pass
    dataset.verify_complete();publisher.finalize();check()
    # Packing scratch is released; retain original numeric blocks and immutable
    # request/file metadata throughout the child and final Source verification.
    dataset.reserve_sink(5*prepared.admission['raw_bytes']+(0 if context.game else publisher.retained))
    assert hashlib.sha256((canonical(request)+'\n').encode()).hexdigest()==request_sha
    return request,request_sha,publisher


class _OwnedTreeFitChild:
    """Reachable pending creation/process/drainers; confirmed reap precedes release."""
    def __init__(self,directory:Any,deadline:float,log_bytes:int,*,child_limit_bytes:int|None=None)->None:
        if child_limit_bytes is not None:
            from gf_common.card_instance_game_tree import MAX_OWNED_BYTES
            from gf_common.domain import integer
            integer(child_limit_bytes,1,MAX_OWNED_BYTES)
        self.child_limit_bytes=child_limit_bytes
        self.directory,self.deadline,self.log_bytes=directory,deadline,log_bytes
        self.creation:Any=None;self.process:Any=None;self.drainers:list[Any]=[];self.health:Json={}
        self.closed=False

    async def start(self,request_sha:str)->None:
        import asyncio
        import os
        import sys

        from gf_training.lightgbm_run import _drain
        environment={'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','CUDA_VISIBLE_DEVICES':'',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        arguments=[str(self.directory),request_sha,str(self.deadline)]
        if self.child_limit_bytes is not None:
            arguments.append(str(self.child_limit_bytes))
        self.creation=asyncio.create_task(asyncio.create_subprocess_exec(sys.executable,'-E','-m','gf_training.structured_tree_worker',
            *arguments,env=environment,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE))
        initiating:BaseException|None=None
        try:
            while not self.creation.done():
                try:
                    async with asyncio.timeout_at(self.deadline):await asyncio.shield(self.creation)
                except asyncio.CancelledError as error:initiating=initiating or error
            self.process=self.creation.result()
            assert self.process.stdout is not None and self.process.stderr is not None
            self.drainers=[asyncio.create_task(_drain(self.process.stdout,self.directory/'stdout.log',self.log_bytes,self.health)),
                asyncio.create_task(_drain(self.process.stderr,self.directory/'stderr.log',self.log_bytes,self.health))]
        except BaseException as error:
            if not self.creation.done() or self.process is not None:
                vars(error)['_structured_tree_fit_child_cleanup_owner']=self
            if initiating is not None:
                initiating.add_note('Tree2 startup after cancellation: '+type(error).__name__+': '+str(error))
                if not self.creation.done() or self.process is not None:
                    vars(initiating)['_structured_tree_fit_child_cleanup_owner']=self
                raise initiating
            raise
        if initiating is not None:
            vars(initiating)['_structured_tree_fit_child_cleanup_owner']=self
            raise initiating

    async def wait(self)->None:
        import asyncio
        async with asyncio.timeout_at(self.deadline):
            while self.process.returncode is None:
                if self.health.get('log_limit_exceeded'):raise ContractError('Tree2 native logs exceeded bounded allowance')
                await asyncio.sleep(.01)
            await self.process.wait()
            await asyncio.gather(*self.drainers)
        if self.process.returncode!=0 or self.health.get('log_limit_exceeded'):raise ContractError('Tree2 fresh numeric child failed')

    async def close(self,deadline:float)->None:
        import asyncio
        import time

        from gf_training.lightgbm_run import _stop
        if self.closed:return
        if self.creation is not None and self.process is None:
            if self.creation.done() and (self.creation.cancelled() or self.creation.exception() is not None):
                # Resolved failed creation proves no returned child. Retire its
                # task without replaying the already-preserved initiating error.
                self.closed=True
                return
            try:
                async with asyncio.timeout_at(deadline):self.process=await asyncio.shield(self.creation)
            except BaseException:
                if self.creation.done() and (self.creation.cancelled() or self.creation.exception() is not None):
                    self.closed=True
                    return
                raise
            # Spawn may have completed after start's timeout. Own its pipes now.
            if not self.drainers:
                from gf_training.lightgbm_run import _drain
                self.drainers=[asyncio.create_task(_drain(self.process.stdout,self.directory/'stdout.log',self.log_bytes,self.health)),
                    asyncio.create_task(_drain(self.process.stderr,self.directory/'stderr.log',self.log_bytes,self.health))]
        if self.process is not None:
            try:await _stop(self.process,self.drainers,max(0,deadline-time.monotonic()),self.health)
            except BaseException as error:
                if self.process.returncode is not None and all(task.done() for task in self.drainers):
                    self.health['exit_code']=self.process.returncode;self.health['reaped']=True
                    self.health['cleanup_failure']=type(error).__name__;self.closed=True
                raise
            if self.process.returncode is None or any(not task.done() for task in self.drainers):
                raise ContractError('Tree2 child cleanup not confirmed')
            self.health['reaped']=True
        self.closed=True


class _TreeFitRun:
    """Concrete per-fit owner, retaining Dataset2 originals until child cleanup."""
    def __init__(self,dataset:Any,directory:Any,model:Any,training:Any,seed:int,*,_context:_TreeRecords=TREE2_RECORDS)->None:
        import time
        self.context=_records(_context)
        self.dataset,self.directory,self.model,self.training,self.seed=dataset,directory,model,training,seed
        self.started=time.monotonic();self.deadline=min(dataset.deadline,self.started+training.wall_seconds)
        self.work_deadline=self.deadline-max(training.cleanup_seconds,dataset.settings.cleanup_seconds)
        self.child:_OwnedTreeFitChild|None=None;self.publisher:Any=None;self.publication:Any=None;self.request:Json|None=None
        self.request_sha:str|None=None;self.completion:Json|None=None;self.prepared:Any=None;self.closed=False

    def _note_failure(self,primary:BaseException,label:str,secondary:BaseException)->None:
        if self.context.game:
            from gf_training.card_instance_game_tree_data import _note
            _note(primary,label,secondary)
        else:
            primary.add_note(label+str(secondary))

    async def _cleanup_child(self)->None:
        import asyncio
        if self.child is None:return
        cleanup=asyncio.create_task(self.child.close(self.deadline))
        initiating:BaseException|None=None
        while not cleanup.done():
            try:await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:initiating=initiating or error
            except BaseException as error:  # game task errors preserve original cancellation
                if not self.context.game:raise
                if not cleanup.done():
                    primary=initiating if initiating is not None else error
                    vars(primary)['_structured_tree_fit_cleanup_owner']=self
                    if initiating is not None:self._note_failure(primary,'game shield cleanup failure: ',error)
                    raise primary
                break  # retrieve task.result below, preserving saved primary
        try:cleanup.result()
        except BaseException as error:
            if not self.child.closed:vars(error)['_structured_tree_fit_cleanup_owner']=self
            if initiating is not None:
                self._note_failure(initiating,'Tree2 cleanup error: ',error)
                if not self.child.closed:vars(initiating)['_structured_tree_fit_cleanup_owner']=self
                raise initiating
            raise
        if initiating is not None:raise initiating

    async def close(self)->None:
        if self.closed:return
        if self.publication is not None and not self.publication.committed:self.publication.revoke()
        terminal_error:BaseException|None=None
        try:await self._cleanup_child()
        except BaseException as error:
            if self.child is not None and not self.child.closed:raise
            terminal_error=error
        initiating=terminal_error or RuntimeError('failed fit retirement')
        if self.context.game and self.prepared is not None:self.prepared.retire_rows()
        if self.dataset._active:self.dataset.__exit__(type(initiating),initiating,None)
        self.closed=not self.dataset._active
        if terminal_error is not None:raise terminal_error

    async def execute(self)->Json:
        import time

        from gf_common.action_contract import engineering_contract, parse_action_contract
        from gf_models.structured_tree_artifacts import inspect_artifact
        from gf_models.structured_tree_pairing import (
            inspect_fit_inputs,
            validate_pair_records,
        )
        from gf_training.structured_lineage import _write
        dataset=self.dataset;entered=False;original:BaseException|None=None
        prep_seconds=child_seconds=validation_seconds=cleanup_seconds=0.
        try:
            dataset.__enter__();entered=True
            assert dataset.manifest is not None and dataset.source is not None
            context=self.context;metadata=_dataset_view(dataset,context)
            classification=metadata['input_binding']
            if not context.modern and classification!={'action_contract':asdict(engineering_contract()),'purpose':'component','synthetic_capacity':False}:
                raise ContractError('private Tree2 component fit requires known engineering/false-capacity Dataset2')
            self.prepared=_PreparedTreeFit(dataset,self.model,{key:getattr(self.training,key) for key in
                ('num_boost_round','early_stopping_rounds','learning_rate','min_data_in_leaf','num_threads')},_context=context)
            self.prepared.log_bytes=self.training.max_log_bytes
            if context.game:
                self.prepared._control_extra=2*self.training.max_log_bytes+4*65536
                self.prepared.admission=_admit(self.prepared)
            _fit_scratch_admission(self.prepared,self.training.max_log_bytes)
            self.prepared.prepare()
            self.request,self.request_sha,self.publisher=_prepare_request(self.prepared,self.directory,seed=self.seed,deadline=self.work_deadline)
            request=self.request;publisher=self.publisher
            base={key:request[key] for key in ('run_id','input_binding','seed','dataset_manifest_sha256','dataset_content_sha256',
                'source_set_sha256','feature_signature','label_recipe','producer','runtime',*context.extra_fields)}
            self.completion={'schema_version':context.schema,'kind':'structured-tree-training',**base,'status':'running','request_sha256':self.request_sha,
                'health':{},'metrics':{},'costs':{},'outputs':{}}
            _write(self.directory/'training.json',self.completion,scan=publisher)
            dataset.verify_complete();publisher.finalize()
            prep_seconds=time.monotonic()-self.started
            self.child=_OwnedTreeFitChild(self.directory,self.work_deadline,self.training.max_log_bytes,
                **({'child_limit_bytes':self.model.resources.max_owned_bytes} if context.game else {}))
            child_started=time.monotonic()
            await self.child.start(self.request_sha);await self.child.wait()
            cleanup_started=time.monotonic();await self._cleanup_child();cleanup_seconds=time.monotonic()-cleanup_started
            child_seconds=time.monotonic()-child_started-cleanup_seconds
            # Same native + Dataset2 index + numeric snapshots through validation.
            validation_started=time.monotonic()
            proof=publisher.object(self.directory/'child-result.json')
            from gf_common.domain import closed
            closed(proof,{'schema_version','kind','request_sha256','input_files','metrics','inference_sha256','model_sha256'}|context.extra_fields)
            context.source(proof)
            if (type(proof['schema_version']) is not int or proof['schema_version']!=context.schema or proof['kind']!='structured-tree-fit-result'
                    or proof['request_sha256']!=self.request_sha or proof['input_files']!=request['input_files']):
                raise ContractError('Tree2 own child request/input receipt mismatch')
            envelope,_,_=inspect_fit_inputs(self.directory,request,publisher,_context=context)
            inference,text=inspect_artifact(self.directory/'artifact',expected_digest=proof['inference_sha256'],model=self.model,
                action_config_hash=envelope['action_config_hash'],action_contract=(context.features.layout.action_contract if context.modern else parse_action_contract(classification['action_contract'])),
                purpose=classification['purpose'],synthetic_capacity=classification['synthetic_capacity'],deadline=self.work_deadline,_context=context,
                _expected_source=None if not context.modern else (request['source_kind'],request['source_origin']))
            from gf_models.structured_tree_pairing import validate_fitted_parameters
            validate_fitted_parameters(text,request,self.model)
            files={**request['input_files'],'request.json':self.request_sha,'artifact/inference.json':proof['inference_sha256'],
                'artifact/model.txt':proof['model_sha256']}
            for name in ('stdout.log','stderr.log','child-result.json'):
                with publisher.file(self.directory/name):pass
                files[name]=publisher.files[self.directory/name][1]
            for name,sha in files.items():
                with publisher.file(self.directory/name,sha):pass
            publisher.finalize();dataset.verify_complete()
            # Terminal native sweep + post-native original-index verification and
            # confirmed Source cleanup MUST finish before complete is published.
            if context.modern:
                if context.game:self.prepared.retire_rows()
                dataset.finalize();dataset.close()
            else:
                dataset.__exit__(None,None,None)
            entered=False
            validation_seconds=time.monotonic()-validation_started
            health={'exit_code':self.child.health['exit_code'],'reaped':self.child.health['reaped'],
                'stdout_bytes':self.child.health['stdout.log_bytes'],'stderr_bytes':self.child.health['stderr.log_bytes']}
            self.completion.update(status='complete',health=health,metrics=proof['metrics'],
                costs={'preparation_seconds':prep_seconds,'child_seconds':child_seconds,'final_validation_seconds':validation_seconds,
                    'cleanup_seconds':cleanup_seconds,'elapsed_seconds':time.monotonic()-self.started,
                    context.shared_cost:metadata[context.shared_cost]},
                outputs={'files':files,'inference_sha256':proof['inference_sha256'],'model_sha256':proof['model_sha256']})
            validate_pair_records(request,self.completion,inference,request_sha256=self.request_sha,
                inference_sha256=proof['inference_sha256'],model_sha256=proof['model_sha256'],_context=context)
            publisher.finalize()
            self.publication=_CompletionPublication(self.directory,publisher,self.deadline,self.model.resources.max_owned_bytes,_context=context)
            try:self.publication.commit(self.completion)
            except BaseException as error:
                if self.publication.fd is not None:vars(error)['_structured_tree_fit_cleanup_owner']=self
                raise
            # FD release is the commit point. No later fallible work/deadline
            # check may turn this already-committed return into a failed fit.
            self.closed=True
            if context.modern:dataset.release()  # Memory-only retirement after the completion commit.
            return self.completion
        except BaseException as error:
            original=error
            try:await self._cleanup_child()
            except BaseException as cleanup:  # noqa: BLE001 - keep exact owner and initiating cancellation
                self._note_failure(original,'Tree2 fit cleanup: ',cleanup)
                if self.child is not None and not self.child.closed:
                    vars(original)['_structured_tree_fit_cleanup_owner']=self
                    raise original
            if entered:
                if self.context.game and self.prepared is not None:
                    try:self.prepared.retire_rows()
                    except BaseException as retirement:  # noqa: BLE001 - retain live actual arrays/originals
                        vars(original)['_structured_tree_fit_cleanup_owner']=self
                        from gf_training.card_instance_game_tree_data import _note
                        _note(original,'game fit row retirement: ',retirement)
                        raise original
                dataset.__exit__(type(original),original,original.__traceback__)
                if dataset._active:vars(original)['_structured_tree_fit_cleanup_owner']=self
            if self.context.modern and dataset._closed:
                dataset.release()
            if self.publisher is not None and self.completion is not None and not (self.publication is not None and self.publication.exposed):
                self.completion['status']='failed'
                try:_write(self.directory/'training.json',self.completion,scan=self.publisher)
                except BaseException as publication:  # noqa: BLE001 - preserve initiating fit failure
                    if self.context.game:
                        self._note_failure(original,'Tree2 failed record publication: ',publication)
                    else:
                        original.add_note('Tree2 failed record publication: '+type(publication).__name__)
            raise


async def _fit_component_dataset(manifest:Any,*,expected_digest:str,directory:Any,model:ConfiguredTreeModelV2,
                                 settings:Any,training:Any,seed:int,native:Any=None)->Json:
    """Private concrete component workflow; no qualified Source/public fit gate."""
    import time
    from pathlib import Path

    from gf_common.domain import integer
    from gf_models.config import TrainingConfig, training_config
    if type(training) is not TrainingConfig:raise ContractError('existing closed fresh-fit training settings required')
    training_config(asdict(training));integer(seed,0,2**31-1)
    if model.checkpoint is not None:raise ContractError('Tree2 Source fitting always initializes fresh')
    if training.wall_seconds>7200:raise ContractError('original Tree2 per-fit wall budget cannot extend7200seconds')
    dataset=VerifiedTreeDataset(Path(manifest),expected_digest,settings,max_owned_bytes=model.resources.max_owned_bytes,
        deadline=time.monotonic()+training.wall_seconds,native=native)
    return await _TreeFitRun(dataset,Path(directory).absolute(),model,training,seed).execute()


async def fit_dataset(manifest:Any,*,action_contract:Any,**options:Any)->Json:
    from gf_common.action_contract import require_qualified_contract
    require_qualified_contract(action_contract)
    return await _fit_component_dataset(manifest,**options)


class _CompletionPublication:
    """One owned completion inode; late failure invalidates only its held FD.

    A foreign current-path replacement is preserved and reported. This owner
    cannot revoke authority of an independently supplied coherent foreign copy.
    """
    def __init__(self,directory:Any,publisher:Any,cleanup_deadline:float,max_owned_bytes:int,*,_context:_TreeRecords=TREE2_RECORDS)->None:
        self.context=_records(_context)
        self.directory,self.publisher,self.cleanup_deadline=directory,publisher,cleanup_deadline
        self.limit=min(max_owned_bytes,publisher.limit-publisher.retained)
        self.fd:int|None=None;self.directory_fd:int|None=None;self.snapshot:Any=None
        self.exposed=False;self.committed=False;self.revoked=False
        self._completion_fd_identity:Any=None
        self._uncertain_releases:list[tuple[str,int,Any,BaseException]]=[]
        self._commit_started=False

    def _namespace(self)->None:
        import os
        import stat
        info=self.directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=self.publisher._tree_fit_directory_identity
                or self.directory_fd is not None and (os.fstat(self.directory_fd).st_dev,os.fstat(self.directory_fd).st_ino)!=(info.st_dev,info.st_ino)):
            raise ContractError('Tree2 completion owned directory namespace changed')
        from gf_training.structured_tree_data import _check_output_ancestors
        _check_output_ancestors(self.publisher._tree_fit_ancestors)

    def _postcheck(self)->None:
        from gf_models.structured_tree_pairing import inspect_training
        self.publisher.finalize()
        remaining=min(self.limit,self.publisher.limit-self.publisher.retained)
        if remaining<=0:raise ContractError('completion joint retained metadata ownership exhausted')
        pair=inspect_training(self.directory,max_owned_bytes=remaining,deadline=self.publisher.deadline,_context=self.context)
        pair.finalize()
        self._namespace()

    def _release(self)->None:
        from gf_models.structured_tree_pairing import _close_owned_descriptor
        # Keep completion revocable until every other descriptor has retired.
        if self.directory_fd is not None:
            fd=self.directory_fd;self.directory_fd=None
            _close_owned_descriptor(self,fd,self.publisher._tree_fit_directory_identity,'completion-directory')
        if self.fd is not None:
            fd=self.fd;self.fd=None
            _close_owned_descriptor(self,fd,self._completion_fd_identity,'completion')
        if self._uncertain_releases:
            raise ContractError('terminal fit descriptor release remains unconfirmed; quarantined numbers are not retryable')

    def revoke(self)->None:
        import time

        from gf_models.structured_tree_pairing import _invalidate_owned_completion
        if not self.revoked and self.exposed:
            if time.monotonic()>=self.cleanup_deadline:raise ContractError('completion revocation unconfirmed: original cleanup deadline exhausted')
            _invalidate_owned_completion(self,self.directory/'training.json')
            self.revoked=True
        # Invalidation can be confirmed even when the previous release failed.
        self._release()

    def commit(self,completion:Json)->None:
        import hashlib
        import os

        from gf_models.structured_tree_artifacts import _identity, _publish, _written
        if self._commit_started or self._uncertain_releases:
            raise ContractError('fit publication is one-use; failed attempts are cleanup-only')
        self._commit_started=True
        self.publisher.check();self._namespace()
        raw=(canonical(completion)+'\n').encode();self.publisher.charge(32*len(raw)+65536)
        self.directory_fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            self._namespace()
            original=self.publisher.files[self.directory/'training.json'][0]
            if _identity((self.directory/'training.json').lstat())!=original:
                raise ContractError('Tree2 original running completion path changed')
            # _publish can close its FD on failure, but this name has NO complete
            # authority. Transfer the returned FD to this owner BEFORE rename.
            self.fd,self.snapshot,sha=_publish(self.directory,self.directory_fd,'training.json.complete.tmp',raw,self.publisher.deadline)
            self._completion_fd_identity=self.snapshot[:2]
            self.publisher.check();self._namespace()
            if _identity((self.directory/'training.json').lstat())!=original:
                raise ContractError('Tree2 running completion replaced before commit')
            self.exposed=True
            os.replace('training.json.complete.tmp','training.json',src_dir_fd=self.directory_fd,dst_dir_fd=self.directory_fd)
            renamed=os.fstat(self.fd)
            if (renamed.st_dev,renamed.st_ino,renamed.st_size,renamed.st_mtime_ns)!=self.snapshot[:4]:
                raise ContractError('Tree2 held completion changed across rename')
            self.snapshot=_identity(renamed)
            _written(self.directory,self.directory_fd,'training.json',self.fd,self.snapshot,sha,self.publisher.deadline)
            self.publisher.files[self.directory/'training.json']=(self.snapshot,hashlib.sha256(raw).hexdigest())
            self._postcheck()
            _written(self.directory,self.directory_fd,'training.json',self.fd,self.snapshot,sha,self.publisher.deadline)
            self._namespace()
            # Final close is fallible and remains within the rollback boundary.
            self._release()
        except BaseException as original:
            if self.exposed:
                # The first failure-path operation is OWNED-FD invalidation.
                # Namespace diagnostics/formatting must never delay revocation.
                try:self.revoke()
                except BaseException as cleanup:  # noqa: BLE001 - unconfirmed authority retirement stays reachable
                    vars(original)['_structured_completion_cleanup_owner']=self
                    try:original.add_note('completion revocation unconfirmed: '+type(cleanup).__name__+': '+str(cleanup))
                    except BaseException:pass  # noqa: BLE001, S110 - diagnostic allocation cannot replace the initiating error
                try:
                    current=(self.directory/'training.json').lstat()
                    if (current.st_dev,current.st_ino)!=(self.snapshot[0],self.snapshot[1]):
                        original.add_note('completion namespace changed; foreign current path preserved')
                except BaseException:  # noqa: BLE001 - namespace/diagnostic failures preserve the original exception
                    try:original.add_note('completion namespace diagnostic unavailable; only held inode is owned')
                    except BaseException:pass  # noqa: BLE001, S110 - diagnostics are best effort after revocation
            else:
                try:self._release()
                except BaseException as cleanup:  # noqa: BLE001 - retain original failure and exact descriptor owner
                    vars(original)['_structured_completion_cleanup_owner']=self
                    original.add_note('completion descriptor retirement unconfirmed: '+str(cleanup))
            raise
        # No fallible I/O remains after confirmed descriptor retirement.
        self.committed=True


async def _fit_component_fixture_dataset(manifest:Any,*,expected_digest:str,directory:Any,model:ConfiguredTreeModelV3,
                                         settings:Any,training:Any,seed:int)->Json:
    """Private concrete declared-fixture fresh fit; never a qualified Source entry."""
    import time
    from pathlib import Path

    from gf_common.domain import integer
    from gf_models.config import TrainingConfig, training_config
    from gf_training.card_instance_tree_data import VerifiedFixtureTreeDataset3
    TREE3_RECORDS.require_model(model)
    if type(training) is not TrainingConfig:
        raise ContractError('existing closed fresh-fit training settings required')
    training_config(asdict(training));integer(seed,0,2**31-1)
    if model.checkpoint is not None or training.wall_seconds>7200:
        raise ContractError('fixture Tree3 requires fresh bounded training configuration')
    dataset=VerifiedFixtureTreeDataset3(Path(manifest),expected_digest,settings,max_owned_bytes=model.resources.max_owned_bytes,
        deadline=time.monotonic()+training.wall_seconds)
    return await _TreeFitRun(dataset,Path(directory).absolute(),model,training,seed,_context=TREE3_RECORDS).execute()


async def fit_game_dataset(manifest:Any,*,expected_digest:str,directory:Any,model:ConfiguredTreeModelV3,
                          settings:Any,training:Any,seed:int,deadline:float,
                          max_owned_bytes:int=8*1024**3,ownership:Any=None)->Json:
    """Fresh qualified Tree3 fit over retained actual three-parent originals.

    This uses the SAME prepared rows/worker/pair publication workflow. Metadata,
    arrays, fixtures and an inference origin alone cannot enter that workflow.
    """
    import math
    import time
    from pathlib import Path

    from gf_common.domain import integer
    from gf_models.config import TrainingConfig, training_config
    from gf_training.card_instance_game_tree_data import VerifiedGameTreeDataset3
    TREE3_GAME_RECORDS.require_model(model)
    if type(training) is not TrainingConfig:
        raise ContractError('existing closed fresh-fit training configuration required')
    training_config(asdict(training));integer(seed,0,2**31-1)
    if (model.checkpoint is not None or training.wall_seconds>7200 or type(deadline) not in (int,float)
            or not math.isfinite(deadline) or deadline<=time.monotonic()):
        raise ContractError('qualified Tree3 requires fresh bounded configuration and original live caller deadline')
    dataset=VerifiedGameTreeDataset3(Path(manifest),expected_digest,settings,max_owned_bytes=max_owned_bytes,
        deadline=min(deadline,time.monotonic()+training.wall_seconds),child_limit_bytes=model.resources.max_owned_bytes,
        ownership=ownership)
    return await _TreeFitRun(dataset,Path(directory).absolute(),model,training,seed,_context=TREE3_GAME_RECORDS).execute()
