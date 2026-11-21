"""Invocation-local producer lineage; JSON records never reconstruct live authority."""
from __future__ import annotations

import hashlib
import os
import stat
import time
import uuid
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, clone
from gf_common.hpo import require_owned_record_version
from gf_training.hpo_lineage import sync_hpo_ancestry
from gf_training.lineage import checkpoint_path, generation_path, require_hpo_run_owner

_TOKEN = object()


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ContractError('owned output must remain a regular original file')
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns


@dataclass(frozen=True,slots=True,init=False)
class OwnedControlWriteReceipt:
    """Actual final write-return digest, bound to this writer/path/full stat."""
    _writer: Any
    path: Path
    identity: tuple[int,int,int,int,int]
    sha256: str

    def __init__(self, token: object, writer: Any, path: Path, identity: Any, sha256: str) -> None:
        if token is not _TOKEN: raise ContractError('actual completed control write required')
        for name,value in (('_writer',weakref.ref(writer)),('path',path),('identity',identity),('sha256',sha256)):
            object.__setattr__(self,name,value)

    def require(self, writer: Any, path: Path, expected_digest: str) -> None:
        require_hpo_run_owner(writer.owner.directory,writer.owner)
        if (self._writer() is not writer or writer._completed.get(path) is not self or self.path!=path
                or writer.failed or writer._uncertain_closes or writer.files.get(path)!=self.identity
                or _identity(path)!=self.identity or self.sha256!=expected_digest):
            raise ContractError('final state differs from actual completed writer/path/stat/digest')

    def __reduce__(self) -> Any: raise TypeError('actual control write receipt cannot be copied/persisted')


class OwnedControlWriter:
    """Durable writes tracked from their actual completion, without a chain scanner."""
    def __init__(self, owner: Any, *, budget: Any, limit: int) -> None:
        self.owner = require_hpo_run_owner(owner.directory,owner)
        self.budget,self.limit = budget,limit
        self.parent_budget=ArtifactPhaseAdmission(budget,limit)
        self.files: dict[Path,tuple[int,int,int,int,int]] = {}
        self._completed: dict[Path,OwnedControlWriteReceipt]={}
        self.failed = False
        self._uncertain_closes: list[Any]=[]

    def write(self, path: Path, value: Json, *, deadline: float, immutable: bool = False) -> str:
        path=Path(os.path.abspath(path))
        require_hpo_run_owner(self.owner.directory,self.owner)
        if self.failed or not path.is_relative_to(self.owner.directory) or time.monotonic()>=deadline:
            raise ContractError('owned control writer failed, expired or outside original parent')
        allowance = 32*(json_bound(value)+1)+65536
        if allowance>self.limit:
            raise ContractError('owned parent control envelope exceeded')
        reservation=self.parent_budget.reserve(allowance)
        descriptor = None
        try:
            with reservation:
                raw = (canonical(value)+'\n').encode()
                temporary = path.with_name('.owned-'+uuid.uuid4().hex+'.tmp')
                sync_hpo_ancestry(path.parent,ownership=self.owner)
                if (path.exists() or path.is_symlink()) and (immutable or path not in self.files or _identity(path)!=self.files[path]):
                    raise ContractError('foreign or replaced owned control output')
                descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                cursor=0
                while cursor<len(raw):
                    if time.monotonic()>=deadline: raise TimeoutError('owned control write deadline')
                    count=os.write(descriptor,raw[cursor:cursor+65536])
                    if count<=0: raise OSError('owned control write incomplete')
                    cursor+=count
                os.fsync(descriptor)
                info=os.fstat(descriptor)
                original=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
                fd,descriptor=descriptor,None
                try: os.close(fd)
                except BaseException as close_error:
                    self._uncertain_closes.append((fd,original,close_error))
                    raise
                if immutable:
                    os.link(temporary,path,follow_symlinks=False)
                    temporary.unlink()
                else:
                    os.replace(temporary,path)
                published=_identity(path)
                if published[:4]!=original[:4]: raise ContractError('owned output identity changed at publication')
                original=published  # Our link/rename advances this SAME inode's ctime.
                sync_hpo_ancestry(path.parent,ownership=self.owner)
                if time.monotonic()>=deadline: raise TimeoutError('owned control durability deadline')
                self.files[path]=original
                sha=hashlib.sha256(raw).hexdigest()
                self._completed[path]=OwnedControlWriteReceipt(_TOKEN,self,path,original,sha)
                return sha
        except BaseException as error:
            self.failed=True
            vars(error)['_hpo_storage_uncertain']=str(path)
            vars(error)['_owned_storage_writer']=self
            if descriptor is not None:
                fd,descriptor=descriptor,None
                try: os.close(fd)
                except OSError as cleanup:
                    self._uncertain_closes.append((fd,None,cleanup))
                    error.add_note('owned control descriptor close: '+str(cleanup))
            raise


class OwnedUpdateTransaction:
    """Single-use exact capture/archive/numeric/process/parent/save composition."""
    def __init__(self, token: object, registry: OwnedRunRegistry, collection: Any, numeric: Any,
                 files: Json, authority: Any) -> None:
        if token is not _TOKEN: raise ContractError('actual owned update issuer required')
        self.registry,self.collection,self.numeric=registry,collection,numeric
        self.files,self.authority=clone(files),authority
        self.parent=registry.current
        self.result: Any=None
        self.state='PERSISTED'

    async def dispatch(self, output: Path, *, deadline: float) -> Any:
        registry=self.registry
        registry.require()
        if self.state!='PERSISTED' or registry.pending is not self or registry.current is not self.parent:
            raise ContractError('owned update transaction is stale or already dispatched')
        self.collection.require(registry,self.collection.plan)
        self.authority.require(registry.process.ledger)
        self.state='DISPATCHED'  # No retry after any possible Adam side effect.
        try:
            self.result=await registry.process.update_owned(self.numeric,output,deadline=deadline,
                resident_authority=self.authority)
            self.result.require_process(registry.process)
            if self.result.generation!=self.parent.generation+1:
                raise ContractError('owned update returned wrong next generation')
            self.state='SAVED'
            return self.result
        except BaseException:
            self.state='FAILED'
            registry.failed=True
            raise

    def __reduce__(self) -> Any:
        raise TypeError('owned update transactions cannot be persisted')


class OwnedRunRegistry:
    """One actual flock + Source account + process, with live commits and pins."""
    def __init__(self, ownership: Any, state: Json, *, budget: Any, parent_budget_bytes: int,
                 writer: OwnedControlWriter | None = None) -> None:
        self.ownership=require_hpo_run_owner(ownership.directory,ownership)
        self.directory,self.state,self.budget=ownership.directory,state,budget
        self.writer=writer or OwnedControlWriter(ownership,budget=budget,limit=parent_budget_bytes)
        if self.writer.owner is not ownership or self.writer.budget is not budget:
            raise ContractError('registry requires SAME actual parent writer and Source account')
        self.parent_budget=self.writer.parent_budget
        if self.parent_budget.source is not budget or self.parent_budget.limit!=parent_budget_bytes:
            raise ContractError('registry requires SAME parent phase admission')
        self.external_controls: Any=None
        self.operations: Any=None
        self._parent_limit=parent_budget_bytes
        self._metadata_bytes=0
        self._metadata_charges: list[Any]=[]
        self.process: Any=None
        self.current: Any=None
        self.snapshots: dict[int,Any]={}
        self.records: list[Json]=[]
        self.artifact_identity: Any=None
        self.generation_directories: dict[int,Any]={}
        self._retirement_owners: list[Any]=[]
        self.file_identities: dict[tuple[int,str],tuple[int,int,int,int,int]]={}
        self.registry: Json={'schema_version':7 if state.get('run_variant')=='hpo-ppo-owned-shared-v1' else 6,'origin':'producer-owned-v1','encoder':'canonical-python','generations':{},
            'evaluation_references':{},'cache_references':{}}
        if state.get('run_variant')=='hpo-ppo-owned-shared-v1':self.registry['service_references']={}
        self._cache_attempts: dict[str,Any]={}
        self._encoded: Any=None
        self._encoded_collection: Any=None
        self.pending: OwnedUpdateTransaction | None=None
        self.capture: Any=None
        self.plan: Any=None
        self.failed=False
        self._closed=False
        self._charge_metadata(32*json_bound(state)+65536)

    def __reduce__(self) -> Any:
        raise TypeError('owned registry cannot be persisted')

    def _charge_metadata(self, size: int) -> None:
        if self._metadata_bytes+size>self._parent_limit:
            raise ContractError('owned retained parent metadata envelope exceeded')
        reservation=self.parent_budget.reserve(size)
        try: self._metadata_charges.append(reservation)
        except BaseException:
            reservation.close()
            raise
        self._metadata_bytes+=size

    def require(self) -> None:
        require_hpo_run_owner(self.directory,self.ownership)
        if self.failed or self._closed: raise ContractError('owned registry is failed or closed')
        if self.process is not None and getattr(self.process,'_owned_registry',None) is not self:
            raise ContractError('owned registry differs from actual process attachment')

    def attach_process(self, process: Any) -> None:
        from gf_training.hpo_owned_process import HpoOwnedLearningProcess
        self.require()
        if (self.process is not None or type(process) is not HpoOwnedLearningProcess
                or process.directory.resolve()!=self.directory or process.budget is not self.budget
                or getattr(process,'_owned_registry',None) is not None):
            raise ContractError('registry requires SAME actual fresh Wire7 process/Source account')
        self.process=process
        vars(process)['_owned_registry']=self

    def import_committed(self, imported: Any, snapshot: Any) -> None:
        from gf_training.hpo_owned_process import OwnedRestoreAck
        from gf_training.hpo_owned_restore import OwnedColdImport
        self.require()
        if (type(imported) is not OwnedColdImport or imported.ownership is not self.ownership
                or imported.budget is not self.budget or imported._process is not self.process
                or imported.state is not self.state or self.current is not None or self.records
                or type(snapshot.restore_ack) is not OwnedRestoreAck):
            raise ContractError('registry import requires SAME actual restored parent transaction')
        snapshot.require_process(self.process)
        if snapshot.generation!=self.state['updates'] or snapshot.optimizer_steps!=self.state['optimizer_steps']:
            raise ContractError('registry import counters differ from committed generation')
        self.records=imported.records
        self.registry=imported.registry
        self.snapshots=dict(imported.snapshots)
        self.snapshots[snapshot.generation]=snapshot
        self.current=snapshot
        from gf_training.structured_lineage import _stat_identity
        self.artifact_identity=_stat_identity((self.directory/'artifacts').lstat())
        for number,record in enumerate(self.records):
            directory=checkpoint_path(self.directory,number)
            self.generation_directories[number]=_stat_identity(directory.lstat())
            for name,row in self.registry['generations'][str(number)]['files'].items():
                if row['status']=='retained': self.file_identities[number,name]=_identity(directory/name)

    def _require_snapshot(self, snapshot: Any) -> None:
        from gf_training.hpo_owned_process import RemoteModelSnapshot
        self.require()
        if type(snapshot) is not RemoteModelSnapshot or self.process is None:
            raise ContractError('registry commit requires actual completed Wire7 snapshot')
        snapshot.require_process(self.process)
        if self.process.current_snapshot is not snapshot or snapshot.actor_metadata['inference']['model']['model_id']!=self.state['model_id']:
            raise ContractError('registry snapshot model/current process differs')

    def commit_initial(self, snapshot: Any, *, controls: Json, deadline: float) -> Json:
        self._require_snapshot(snapshot)
        if (self.current is not None or self.records or snapshot.generation!=0 or snapshot.optimizer_steps!=0
                or self.state['updates']!=0 or self.state['generation_sha256'] is not None):
            raise ContractError('owned initial commit must be the actual fresh generation zero')
        return self._commit(snapshot,None,controls,deadline)

    def encode_and_prepare(self, collection: Any, directory: Path, *, model: Any, training: Any,
                           actions: Any, rules: Any, numeric_budget_bytes: int, authority: Any) -> OwnedUpdateTransaction:
        from gf_common.hpo import selected_locations
        from gf_training.hpo_owned_rollout import encode_owned_batch
        self.require()
        collection.require(self,self.plan)
        if self._encoded is not None or self.pending is not None:
            raise ContractError('owned registry already has an original encoded batch')
        selection=collection.selection()
        # The registry observes THIS exact encoder return and completed writer,
        # never accepts a caller's equivalent numeric or persisted file map.
        self._encoded=encode_owned_batch(collection.batch,owner=self,plan=self.plan,snapshot_owner=self.current,
            model=model,training=training,actions=actions,rules=rules,selection=selection,
            selected_locations=selected_locations(selection),budget=self.budget,numeric_budget_bytes=numeric_budget_bytes)
        self._encoded_collection=collection
        files=self._encoded.persist(directory)['files']
        return self.prepare_update(collection,self._encoded,files,authority=authority)

    def prepare_update(self, collection: Any, numeric: Any, files: Json, *, authority: Any) -> OwnedUpdateTransaction:
        from gf_training.hpo_owned_collection import OwnedFinalizedCollection
        from gf_training.hpo_owned_rollout import OwnedNumericBatch
        from gf_training.hpo_resident import ParkedAuthority
        self.require()
        if (type(collection) is not OwnedFinalizedCollection or type(numeric) is not OwnedNumericBatch
                or type(authority) is not ParkedAuthority or self.pending is not None
                or numeric is not self._encoded or collection is not self._encoded_collection):
            raise ContractError('actual collection/numeric/PARK transaction required')
        collection.require(self,collection.plan)
        authority.require(self.process.ledger)
        if (numeric._state!='PERSISTED' or numeric._budget is not self.budget
                or numeric.generation!=self.current.generation or files!=numeric._files
                or authority.evidence['generation']!=numeric.generation
                or authority.evidence['collection_id']!=collection.plan.linkage['batch_id']):
            raise ContractError('owned update persistence/physical generation differs')
        collection.batch.require_owner(self,plan=collection.plan,snapshot_owner=self.current)
        self.pending=OwnedUpdateTransaction(_TOKEN,self,collection,numeric,files,authority)
        return self.pending

    def commit_update(self, transaction: OwnedUpdateTransaction, update: Json, *, controls: Json, deadline: float) -> Json:
        self.require()
        if (type(transaction) is not OwnedUpdateTransaction or self.pending is not transaction
                or transaction.registry is not self or transaction.state!='SAVED'
                or transaction.parent is not self.current):
            raise ContractError('owned commit requires SAME saved single-use transaction')
        require_owned_record_version(update,'update')
        require_owned_record_version(update['source_proof'],'proof')
        require_owned_record_version(transaction.numeric.metadata,'numeric')
        require_owned_record_version(transaction.collection.manifest,'collection')
        self._require_snapshot(transaction.result)
        if (update['generation']!=transaction.result.generation or update['numeric_files']!=transaction.files
                or update['accepted_decisions']!=transaction.numeric.used_decisions
                or update['source_proof']!=transaction.collection.proof(transaction.numeric)
                or self.state['optimizer_steps']!=transaction.result.optimizer_steps
                or self.state['used_decisions']!=self.records[-1]['used_decisions']+transaction.numeric.used_decisions
                or self.state['collected_decisions']!=self.records[-1]['collected_decisions']+transaction.numeric.collected_decisions
                or self.state['accepted_decisions']!=self.state['used_decisions']):
            raise ContractError('owned commit record differs from actual dispatched batch/save')
        try:
            result=self._commit(transaction.result,update,controls,deadline)
            transaction.state='COMMITTED'
            self.pending=None
            self._encoded=None
            self._encoded_collection=None
            return result
        except BaseException:
            transaction.state='FAILED'
            self.failed=True
            raise

    def _commit(self, snapshot: Any, update: Json | None, controls: Json, deadline: float) -> Json:
        generation=snapshot.generation
        state=self.state
        files=snapshot.files
        identities={name:_identity(checkpoint_path(self.directory,generation)/name) for name in files}
        from gf_training.structured_lineage import _stat_identity
        actual_artifacts=_stat_identity((self.directory/'artifacts').lstat())
        if self.artifact_identity is not None and actual_artifacts[:2]!=self.artifact_identity[:2]:
            raise ContractError('owned artifacts parent changed from actual original')
        self.artifact_identity=actual_artifacts
        self.generation_directories[generation]=_stat_identity(checkpoint_path(self.directory,generation).lstat())
        update_sha=None if update is None else hashlib.sha256((canonical(update)+'\n').encode()).hexdigest()
        self._charge_metadata(65536+8192*len(files)+32*json_bound(controls))
        record={'schema_version':5,'origin':'producer-owned-v1','encoder':'canonical-python',
            'run_id':state['run_id'],'generation':generation,'parent_sha256':state['generation_sha256'],
            'snapshot_identity':snapshot.identity,'inference_sha256':files['inference.json'],
            'resume_sha256':files['resume.json'],'artifact_files':files,'update_sha256':update_sha,
            'parent_controls':clone(controls),'optimizer_steps':snapshot.optimizer_steps,
            'used_decisions':state['used_decisions'],'collected_decisions':state['collected_decisions'],
            'accepted_decisions':state['accepted_decisions']}
        try:
            if update is not None: self.writer.write(self.directory/'updates'/f'{generation:06}.json',update,deadline=deadline,immutable=True)
            require_owned_record_version(record,'generation')
            require_owned_record_version(self.registry,'registry')
            sha=self.writer.write(generation_path(self.directory,generation),record,deadline=deadline,immutable=True)
            self.registry['generations'][str(generation)]={'files':{name:{'sha256':value,'status':'retained',
                'deletion_intent':None,'deletion_receipt':None} for name,value in files.items()}}
            self.writer.write(self.directory/'artifact-registry.json',self.registry,deadline=deadline)
            self.writer.write(self.directory/'latest.json',{'generation':generation,'sha256':sha},deadline=deadline)
            state.update(updates=generation,generation_sha256=sha)
            self.writer.write(self.directory/'run.json',state,deadline=deadline)
        except BaseException:
            self.failed=True
            raise
        self.records.append(clone(record))
        self.snapshots[generation]=snapshot
        self.file_identities.update({(generation,name):identity for name,identity in identities.items()})
        self.current=snapshot
        return clone(record)

    def binding(self, generation: int) -> Any:
        from gf_common.card_instance import action_config_v3
        from gf_common.card_instance_rules import card_instance_rules
        from gf_common.domain import RuntimePolicyBinding, digest
        from gf_models.structured_binding import (
            input_binding_game_v3,
            policy_label_game_v4,
            safe_options_game_v4,
        )
        from gf_models.structured_config import InferenceBindingV2
        self.require()
        snapshot=self.snapshots.get(generation)
        if snapshot is None: raise ContractError('owned generation has no live snapshot owner')
        snapshot.require_process(self.process)
        # Exact saved metadata; no checkpoint deserialization or historical chain.
        model=self.process.model
        actions=action_config_v3(self.state['effective_config']['actions'])
        execution=self.state['execution']['inference']
        binding=InferenceBindingV2(**execution['binding'])
        sha=snapshot.files['inference.json']
        safe=safe_options_game_v4({'version':'4','checkpoint':str(checkpoint_path(self.directory,generation)),
            'expected_manifest_digest':sha,'model':asdict(model),'inference':asdict(binding),
            'cuda_visible_devices':execution['cuda_visible_devices'],'action_config_hash':digest(asdict(actions)),
            **input_binding_game_v3(actions.action_contract,card_instance_rules())},actions=actions,rules=card_instance_rules())
        label=policy_label_game_v4(model,binding,sha,action_contract=actions.action_contract,rules=card_instance_rules(),
            action_config_hash=digest(asdict(actions)),cuda_visible_devices=execution['cuda_visible_devices'])
        return label,RuntimePolicyBinding(model.model_id,safe)

    def artifacts(self, bindings: Any) -> Any:
        from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
        from gf_training.structured_lineage import _stat_identity
        preparation=65536
        for binding in bindings.values():
            if binding.policy_options is None or binding.model_id=='heuristic_refresh': continue
            snapshot=self.snapshots[int(Path(binding.policy_options['checkpoint']).name)]
            original=snapshot._control.value
            preparation+=32*json_bound(original['inference_manifest'])+8192*len(original['files'])+32*json_bound(binding.policy_options)
        with self.parent_budget.reserve(preparation):
            entries={}
            for label,binding in bindings.items():
                if binding.policy_options is None or binding.model_id=='heuristic_refresh': continue
                generation=int(Path(binding.policy_options['checkpoint']).name)
                expected_label,expected=self.binding(generation)
                if label!=expected_label or binding!=expected: raise ContractError('foreign owned artifact binding')
                snapshot=self.snapshots[generation]
                inference=snapshot.inference_manifest
                files={'inference.json':snapshot.files['inference.json'],**inference['weights']['files']}
                entries[label]=(snapshot.actor_capability,binding,[{'path':name,'sha256':sha,
                    'identity':list(_stat_identity((checkpoint_path(self.directory,generation)/name).lstat()))}
                    for name,sha in sorted(files.items())],inference)
            return OwnedGameArtifacts(entries,budget=self.budget,parent_budget=self.parent_budget)

    def pin_cache(self, attempt: Any, *, deadline: float) -> Any:
        from gf_common.domain import digest
        from gf_common.hpo_resident import cache_owner
        from gf_training.hpo_resident import CacheAcquisition
        from gf_training.hpo_retention import HpoReferenceReceipt
        require_hpo_run_owner(self.directory,self.ownership)
        if type(attempt) is not CacheAcquisition or attempt.pool._acquisitions.get(attempt.token) is not attempt:
            raise ContractError('owned cache requires actual acquisition transaction')
        previous=self.registry['cache_references'].get(attempt.token)
        if previous is not None:
            if getattr(self,'_cache_attempts',{}).get(attempt.token) is not attempt or previous['phase']!='acquire':
                raise ContractError('owned cache acquisition already released or foreign')
            return self._cache_attempts[previous['binding']['token']].receipt
        label,policy=self.binding(attempt.generation)
        del label
        record=self.records[attempt.generation]
        if record['inference_sha256']!=attempt.inference_sha256:
            raise ContractError('owned cache acquisition checkpoint differs from committed snapshot')
        policy_sha=digest(asdict(policy))
        weights=digest({name:sha for name,sha in record['artifact_files'].items() if name.startswith('weights-')})
        binding={'run_id':self.state['run_id'],'token':attempt.token,'generation':attempt.generation,
            'inference_sha256':attempt.inference_sha256,
            'generation_sha256':self._generation_hash(attempt.generation),**cache_owner(attempt.owner_identity),
            'policy_sha256':policy_sha,'weights_sha256':weights,'snapshot_identity':self.snapshots[attempt.generation].identity,
            'cache_identity':digest({'recipe':'owned-inference-cache-v1','policy_sha256':policy_sha,
                'inference_sha256':attempt.inference_sha256,'snapshot_identity':self.snapshots[attempt.generation].identity})}
        self._charge_metadata(65536+32*json_bound(binding))
        self._cache_attempts[attempt.token]=attempt
        root=self.directory/'cache-references'/attempt.token
        root.mkdir(parents=True,exist_ok=False)
        intent={'schema_version':2,'origin':'producer-owned-v1','encoder':'canonical-python','phase':'acquire','binding':binding}
        self.writer.write(root/'acquire.intent.json',intent,deadline=deadline,immutable=True)
        sha=self.writer.write(root/'acquire.receipt.json',intent,deadline=deadline,immutable=True)
        receipt=HpoReferenceReceipt(binding,sha,'acquire')
        attempt.receipt=receipt
        self.registry['cache_references'][attempt.token]={'phase':'acquire','binding':clone(binding),'receipt_sha256':sha}
        self.writer.write(self.directory/'artifact-registry.json',self.registry,deadline=deadline)
        return receipt

    def _generation_hash(self, generation: int) -> str:
        return hashlib.sha256((canonical(self.records[generation])+'\n').encode()).hexdigest()

    def release_cache(self, binding: Json, *, authority: Any, deadline: float) -> None:
        from gf_training.hpo_resident import CacheReleaseAuthority
        require_hpo_run_owner(self.directory,self.ownership)
        if type(authority) is not CacheReleaseAuthority: raise ContractError('actual ACK/reap cache release required')
        authority.require(binding)
        row=self.registry['cache_references'].get(binding['token'])
        if row is None or row['phase']!='acquire' or row['binding']!=binding:
            raise ContractError('owned cache release lacks SAME original live pin')
        value={'schema_version':2,'origin':'producer-owned-v1','encoder':'canonical-python','phase':'release',
            'binding':clone(binding),'evidence':authority.evidence}
        root=self.directory/'cache-references'/binding['token']
        self.writer.write(root/'release.intent.json',value,deadline=deadline,immutable=True)
        sha=self.writer.write(root/'release.receipt.json',value,deadline=deadline,immutable=True)
        row.update(phase='release',receipt_sha256=sha)
        self.writer.write(self.directory/'artifact-registry.json',self.registry,deadline=deadline)

    def protected_files(self, operating: Json) -> dict[int,set[str]]:
        from gf_training.hpo_lineage import protected_hpo_files_for_operating
        references={'evaluation_references':self.registry['evaluation_references'],
            'cache_references':{token:row['binding'] for token,row in self.registry['cache_references'].items()
                if row['phase']=='acquire'}}
        references['cache_references'].update({token:row['binding'] for token,row in self.registry.get('service_references',{}).items() if row['phase']=='acquire'})
        pending=[{'generation':attempt.generation} for token,attempt in self._cache_attempts.items()
            if token not in self.registry['cache_references']]
        return protected_hpo_files_for_operating(operating,self.records,references,pending)

    def retain(self, operating: Json, *, deadline: float, pool: Any) -> None:
        self.require()
        current=self.current.generation
        controls=self.records[-1]['parent_controls']
        protected=self.protected_files(operating)
        live=set(controls['snapshot_pool'])|{current}|{row['binding']['generation'] for row in self.registry['cache_references'].values() if row['phase']=='acquire'}
        live.update(row['binding']['generation'] for row in self.registry.get('service_references',{}).values() if row['phase']=='acquire')
        for generation,record in enumerate(self.records):
            for name in record['artifact_files']:
                row=self.registry['generations'][str(generation)]['files'][name]
                if name in protected[generation] or row['status']!='retained': continue
                retirement=_OwnedRetirement(self,generation,name,row,deadline)
                try:
                    retirement.run()
                except BaseException as error:
                    self.failed=True
                    if retirement.phase in ('unstarted','held','intent'):
                        try: retirement.retire_unmoved()
                        except BaseException as cleanup:  # noqa: BLE001 - keep exact failed staging custody and original error
                            error.add_note('owned retirement cleanup retained: '+str(cleanup))
                    vars(error)['_hpo_retention_pending']=retirement
                    vars(error)['cleanup_owner']=retirement
                    raise
            if generation not in live and generation in self.snapshots:
                snapshot=self.snapshots[generation]
                pool.unbind_owned_snapshot(snapshot.actor_capability)
                snapshot.close()
                del self.snapshots[generation]

    def close(self) -> None:
        require_hpo_run_owner(self.directory,self.ownership)
        if self.process is not None and self.process.process is not None and self.process.process.returncode is None:
            raise ContractError('owned registry requires actual learner reap before close')
        if any(not owner.closed for owner in self._retirement_owners):
            raise ContractError('owned retirement physical custody remains unresolved')
        owned={id(snapshot):snapshot for snapshot in self.snapshots.values()}
        if self.process is not None and self.process.current_snapshot is not None:
            owned[id(self.process.current_snapshot)]=self.process.current_snapshot
        if self.pending is not None and self.pending.result is not None:
            owned[id(self.pending.result)]=self.pending.result
        for snapshot in owned.values(): snapshot.close()
        self.pending=None
        self.snapshots.clear()
        self.current=None
        for charge in self._metadata_charges: charge.close()
        self._metadata_charges.clear()
        self._closed=True


class _OwnedRetirement:
    """Held-original staging; persisted intents never authorize pathname unlink."""
    def __init__(self, registry: OwnedRunRegistry, generation: int, name: str, row: Json, deadline: float) -> None:
        self.registry,self.generation,self.name,self.row=registry,generation,name,row
        self.deadline=deadline
        self.parent=registry.directory/'artifacts'
        self.path=checkpoint_path(registry.directory,generation)
        self.staging='prune-'+uuid.uuid4().hex
        self.fds: dict[str,int]={}
        self.closed=False
        self.phase='unstarted'
        self.reservation: Any=None
        registry._retirement_owners.append(self)  # Retain before any reservation/open can fail.

    def check(self) -> None:
        require_hpo_run_owner(self.registry.directory,self.registry.ownership)
        if time.monotonic()>=self.deadline: raise TimeoutError('owned retirement deadline')

    def directory(self, role: str, expected: Any, *, advance: bool = False) -> Any:
        from gf_training.structured_lineage import _directory_entry, _stat_identity
        self.check()
        fd=self.fds[role]
        held=os.fstat(fd)
        current=(self.parent.lstat() if role=='parent' else
            os.stat(self.path.name if role=='source' else self.staging,dir_fd=self.fds['parent'],follow_symlinks=False))
        if (not stat.S_ISDIR(current.st_mode) or _stat_identity(held)!=_stat_identity(current)
                or _stat_identity(held)[:2]!=expected[:2]
                or not advance and _stat_identity(held)!=expected):
            raise ContractError('owned retirement directory differs from original full stat fence')
        if role!='parent': _directory_entry(self.fds['parent'],self.path.name if role=='source' else self.staging,fd)
        return _stat_identity(held)

    def close(self) -> None:
        failures=[]
        for role,fd in tuple(self.fds.items()):
            del self.fds[role]  # A close attempt consumes the number, including uncertainty.
            try: os.close(fd)
            except OSError as error:
                failures.append((role,fd,error))
        if failures:
            self.uncertain=failures
            raise ContractError('owned retirement descriptor closure uncertain')
        if getattr(self,'uncertain',None): raise ContractError('owned retirement uncertain descriptors remain')
        if self.reservation is not None: self.reservation.close()
        self.closed=True

    def retire_unmoved(self) -> None:
        from gf_training.structured_lineage import _directory_entry, _stat_identity
        if self.phase not in ('unstarted','held','intent'):
            raise ContractError('owned move/unlink effect requires original quarantine')
        if 'stage' in self.fds:
            _directory_entry(self.fds['parent'],self.staging,self.fds['stage'])
            os.rmdir(self.staging,dir_fd=self.fds['parent'])
            os.fsync(self.fds['parent'])
            # Only our empty private staging name was removed; no file moved.
            held=os.fstat(self.fds['parent'])
            current=self.parent.lstat()
            if _stat_identity(held)==_stat_identity(current):
                self.registry.artifact_identity=_stat_identity(held)
        self.close()

    def run(self) -> None:
        from gf_training.structured_lineage import _directory_entry, _stat_identity
        registry=self.registry
        self.reservation=registry.parent_budget.reserve(65536)
        self.check()
        flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
        self.fds['parent']=os.open(self.parent,flags)
        parent=self.directory('parent',registry.artifact_identity)
        self.fds['source']=os.open(self.path.name,flags,dir_fd=self.fds['parent'])
        source=self.directory('source',registry.generation_directories[self.generation])
        self.fds['original']=os.open(self.name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW,dir_fd=self.fds['source'])
        self.phase='held'
        original=registry.file_identities[self.generation,self.name]
        before=os.fstat(self.fds['original'])
        named=os.stat(self.name,dir_fd=self.fds['source'],follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before)!=original or _stat_identity(named)!=original:
            raise ContractError('owned retirement file differs from original full stat fence')
        os.mkdir(self.staging,0o700,dir_fd=self.fds['parent'])
        parent=self.directory('parent',parent,advance=True)
        registry.artifact_identity=parent
        self.fds['stage']=os.open(self.staging,flags,dir_fd=self.fds['parent'])
        stage=_stat_identity(os.fstat(self.fds['stage']))
        self.directory('stage',stage)
        intent={'schema_version':1,'origin':'producer-owned-v1','generation':self.generation,
            'name':self.name,'sha256':self.row['sha256'],'original_identity':list(original),'staging_basename':self.staging}
        root=registry.directory/'retention'
        root.mkdir(exist_ok=True)
        stem=f'{self.generation:06}-{self.name}'
        self.phase='intent'
        self.row['deletion_intent']=registry.writer.write(root/(stem+'.intent.json'),intent,deadline=self.deadline,immutable=True)
        registry.writer.write(registry.directory/'artifact-registry.json',registry.registry,deadline=self.deadline)
        self.directory('parent',parent)
        self.directory('source',source)
        self.directory('stage',stage)
        if (_stat_identity(os.fstat(self.fds['original']))!=original
                or _stat_identity(os.stat(self.name,dir_fd=self.fds['source'],follow_symlinks=False))!=original):
            raise ContractError('owned retention original changed after intent')
        self.phase='rename_requested'
        os.rename(self.name,self.name,src_dir_fd=self.fds['source'],dst_dir_fd=self.fds['stage'])
        # The only accepted ctime advance is our move of the SAME held inode.
        parent=self.directory('parent',parent)
        source=self.directory('source',source,advance=True)
        stage=self.directory('stage',stage,advance=True)
        moved=os.stat(self.name,dir_fd=self.fds['stage'],follow_symlinks=False)
        held=os.fstat(self.fds['original'])
        if (_stat_identity(held)[:4]!=original[:4] or _stat_identity(moved)!=_stat_identity(held)
                or not stat.S_ISREG(moved.st_mode)):
            raise ContractError('owned staged file differs from held original')
        self.phase='staged'
        self.directory('parent',parent)
        self.directory('source',source)
        self.directory('stage',stage)
        if _stat_identity(os.stat(self.name,dir_fd=self.fds['stage'],follow_symlinks=False))!=_stat_identity(held):
            raise ContractError('owned private staged file changed before unlink')
        self.phase='unlink_requested'
        os.unlink(self.name,dir_fd=self.fds['stage'])
        self.phase='unlinked'
        stage=self.directory('stage',stage,advance=True)
        self.directory('source',source)
        self.directory('parent',parent)
        for role in ('source','stage','parent'): os.fsync(self.fds[role])
        _directory_entry(self.fds['parent'],self.staging,self.fds['stage'])
        os.rmdir(self.staging,dir_fd=self.fds['parent'])
        registry.artifact_identity=self.directory('parent',parent,advance=True)
        registry.generation_directories[self.generation]=source
        os.fsync(self.fds['parent'])
        self.close()
        self.row['deletion_receipt']=registry.writer.write(root/(stem+'.receipt.json'),intent,deadline=self.deadline,immutable=True)
        self.row['status']='deleted'
        registry.writer.write(registry.directory/'artifact-registry.json',registry.registry,deadline=self.deadline)
