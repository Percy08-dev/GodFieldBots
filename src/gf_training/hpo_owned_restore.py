"""One external owned checkpoint import under the actual new lock/admission.

Metadata validation is deliberately separate from full historical sample audits.
Only current resume and required pool inference arrays are decoded here.
"""
from __future__ import annotations

import os
import re
import stat
import weakref
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.artifacts import ArtifactPhaseAdmission, json_bound
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer
from gf_common.hpo import require_owned_record_version, selected_locations
from gf_common.ram_bank import MAX_BANK_BYTES, ReadonlyRamBankSet
from gf_models.structured_sealed import SealedScan, checksum
from gf_training.lineage import checkpoint_path, generation_path, require_hpo_run_owner

_TOKEN=object()


class _ColdScan(SealedScan):
    """Cached JSON once; original external decoders and physical fences remain."""
    def __init__(self, ownership: Any, phase: ArtifactPhaseAdmission, deadline: float) -> None:
        self.root=ownership.directory
        self.phase=phase
        self._charges: list[Any]=[]
        self._scratch=0
        self._objects: dict[Path,Json]={}
        self._directories: dict[Path,tuple[Any,...]]={}
        super().__init__(max_owned_bytes=phase.limit,deadline=deadline)

    def charge(self, size: int, *, temporary: bool=False) -> None:
        super().charge(size,temporary=temporary)
        amount=max(0,size-self._scratch) if temporary else size
        if amount: self._charges.append(self.phase.reserve(amount))
        if temporary: self._scratch=max(self._scratch,size)

    def object(self, path: Path, expected: str | None=None) -> Json:
        if path in self._objects:
            if expected is not None and self.verified_digest(path)!=expected:
                raise ContractError('cold cached original hash mismatch')
            return self._objects[path]
        value=super().object(path,expected)
        self._objects[path]=value
        return value

    def finalize(self) -> None:
        # Original decoded objects/arrays are retained; identity fences catch
        # mutation without decoding or reopening every historical file again.
        self.check()
        _check_archive_presence(self)
        from gf_models.structured_sealed import _identity
        for path,(identity,_sha) in self.files.items():
            if _identity(path.lstat())!=identity or path.is_symlink():
                raise ContractError('cold imported original changed')
        for directory,(identity,names) in self._directories.items():
            if _identity(directory.lstat())!=identity: raise ContractError('cold parent directory changed')
            with os.scandir(directory) as entries:
                if tuple(sorted(entry.name for entry in entries))!=names: raise ContractError('cold directory inventory changed')
        for directory,names in self.inventories.items():
            current,_=self._inventory(directory,names)
            if current!=names: raise ContractError('cold namespace changed')

    def directories(self, path: Path, expected: set[str]) -> None:
        import stat

        from gf_models.structured_sealed import _identity
        before=path.lstat()
        if not stat.S_ISDIR(before.st_mode): raise ContractError('cold parent namespace must be a directory')
        self.charge(1024+1024*len(expected))
        with os.scandir(path) as entries:
            names=[]
            for entry in entries:
                self.check()
                if not entry.is_dir(follow_symlinks=False) or entry.name not in expected:
                    raise ContractError('foreign cold parent directory entry')
                names.append(entry.name)
        if set(names)!=expected or len(names)!=len(expected): raise ContractError('cold parent directory inventory mismatch')
        identity=_identity(path.lstat())
        if identity!=_identity(before): raise ContractError('cold parent namespace changed')
        self._directories[path]=(identity,tuple(sorted(expected)))

    def close(self) -> None:
        self._objects.clear()
        for charge in self._charges: charge.close()
        self._charges.clear()


def _regular_info(path: Path) -> Any:
    try: info=path.lstat()
    except FileNotFoundError: info=None
    if info is None or not stat.S_ISREG(info.st_mode):
        raise ContractError('required owned original regular file missing or replaced')
    return info


def _bounded_archive_object(scan: Any, path: Path, expected: str | None=None) -> Json:
    from gf_common.group_pack import MAX_JSON
    if _regular_info(path).st_size>MAX_JSON: raise ContractError('owned archive control exceeds original bound')
    return scan.object(path,expected)


def _check_archive_presence(scan: Any) -> None:
    from gf_models.structured_sealed import _identity
    for path,identity in getattr(scan,'_owned_archive_presence',{}).items():
        if _identity(_regular_info(path))!=identity:
            raise ContractError('owned archive original changed after presence validation')


def _archive_presence(scan: Any, state: Json, update: Json) -> None:
    """Control-only archive fence: no SQLite/member/decompression/DB hash access."""
    from gf_models.structured_sealed import _identity
    from gf_training.group_pack_db import DATABASE, descriptor_name, validate_descriptor
    from gf_training.group_pack_source import _no_symlink_path
    locator_path=scan.root/update['collection']/'collection.locator.json'
    locator=closed(_bounded_archive_object(scan,locator_path),{'format','schema_version','descriptor','descriptor_sha256'})
    integer(locator['schema_version'],1,1)
    if locator['format']!='sqlite-group-pack-locator-v1' or type(locator['descriptor']) is not str:
        raise ContractError('owned archive requires exact original locator')
    root=Path(state['effective_config']['storage']['db_root'])/state['run_id']/'collections'/update['batch_id']
    expected=root/descriptor_name('published')
    if locator['descriptor']!=str(expected): raise ContractError('owned locator targets foreign archive namespace')
    missing=False
    try: _no_symlink_path(root)
    except FileNotFoundError: missing=True
    if missing: raise ContractError('owned archive directory missing')
    descriptor=validate_descriptor(_bounded_archive_object(scan,expected,checksum(locator['descriptor_sha256'])))
    if descriptor['publication']!='published': raise ContractError('owned archive descriptor is not published')
    controls={row['name']:row for row in descriptor['controls']}
    if ('manifest.json' not in controls or controls['manifest.json']['sha256']!=update['collection_manifest_sha256']
            or controls['manifest.json']['bytes']<=0):
        raise ContractError('owned published descriptor differs from committed collection manifest')
    names={DATABASE,expected.name}
    predecessor=root/descriptor_name('pending')
    if predecessor.exists() or predecessor.is_symlink():
        pending=_bounded_archive_object(scan,predecessor)
        validate_descriptor(pending)
        if pending!={**descriptor,'publication':'pending'}:
            raise ContractError('owned published predecessor differs from same original seal')
        names.add(predecessor.name)
    scan.inventory(root,names)
    database=root/DATABASE
    info=_regular_info(database)
    if info.st_size!=descriptor['database_bytes']:
        raise ContractError('owned archive physical database size differs from published descriptor')
    scan.charge(1024+4*len(str(database)))
    if not hasattr(scan,'_owned_archive_presence'): scan._owned_archive_presence={}
    identity=_identity(info)
    previous=scan._owned_archive_presence.get(database)
    if previous is not None and previous!=identity: raise ContractError('owned archive changed between control inspections')
    scan._owned_archive_presence[database]=identity
    _check_archive_presence(scan)


def _owned_retirement_status(scan: Any, number: int, name: str, row: Json, expected: str) -> str:
    """Registry6 SHA-pointer records, independent of legacy inline _receipt."""
    from gf_training.hpo_lineage import _identity as validate_identity
    closed(row,{'sha256','status','deletion_intent','deletion_receipt'})
    if checksum(row['sha256'])!=expected: raise ContractError('owned retirement original file hash differs')
    if row['status']=='retained':
        if row['deletion_intent'] is not None or row['deletion_receipt'] is not None:
            raise ContractError('owned retained file contains uncertain retirement evidence')
        return 'retained'
    if row['status']!='deleted': raise ContractError('owned uncertain retirement cannot import')
    root=scan.root/'retention'
    stem=f'{number:06}-{name}'
    intent=scan.object(root/(stem+'.intent.json'),checksum(row['deletion_intent']))
    receipt=scan.object(root/(stem+'.receipt.json'),checksum(row['deletion_receipt']))
    keys={'schema_version','origin','generation','name','sha256','original_identity','staging_basename'}
    closed(intent,keys); closed(receipt,keys)
    integer(intent['schema_version'],1,1); integer(intent['generation'],number,number)
    integer(receipt['schema_version'],1,1); integer(receipt['generation'],number,number)
    if (intent['origin']!='producer-owned-v1' or intent['name']!=name or checksum(intent['sha256'])!=expected
            or receipt!=intent):
        raise ContractError('owned retirement completion differs from original intent/file/generation')
    validate_identity(intent['original_identity'])
    validate_identity(receipt['original_identity'])
    staging=intent['staging_basename']
    if type(staging) is not str or re.fullmatch(r'prune-[0-9a-f]{32}',staging) is None:
        raise ContractError('owned retirement staging namespace invalid')
    for path in (scan.root/'artifacts'/staging,checkpoint_path(scan.root,number)/name):
        if path.exists() or path.is_symlink(): raise ContractError('owned completed retirement remains physically present')
    return 'deleted'


_DECODE_QUARANTINE: list[Any]=[]


class _DecodeReceipt:
    def __init__(self) -> None:
        self.closed=True
        self.error: BaseException | None=None


def _finish_decode(reservation: Any, receipt: _DecodeReceipt) -> None:
    try: reservation.close()
    except BaseException as error:  # noqa: BLE001 - finalizer keeps actual uncertain admission, never retries
        receipt.error=error
        _DECODE_QUARANTINE.append((reservation,receipt,error))
    else: receipt.closed=True


def _keep_decode_alive(owner: Any) -> None:
    # weakref.finalize holds this owner, never the ndarray itself.
    pass


class _DecodeEnvelope:
    """Actual reservation lives through decoded arrays AND decoder traceback frames."""
    def __init__(self, reservation: Any, receipt: _DecodeReceipt) -> None:
        self.reservation=reservation
        receipt.closed=False
        try: weakref.finalize(self,_finish_decode,reservation,receipt)
        except BaseException:
            _finish_decode(reservation,receipt)
            raise

    def hold(self, array: Any) -> None:
        weakref.finalize(array,_keep_decode_alive,self)


def _owned_update_metadata(update: Json, state: Json, previous: Json, current: Json) -> None:
    from gf_training.hpo_config import finite_number, hpo_training_config
    from gf_training.hpo_lineage import _CLASSIFICATION, HPO_UPDATE_FIELDS, _selection, hpo_spec
    from gf_training.hpo_source import learner_context_costs
    require_owned_record_version(update,'update')
    closed(update,HPO_UPDATE_FIELDS|{'origin','encoder'}|({'storage_binding'} if 'storage_binding' in state else set()))
    generation=current['generation']
    integer(update['generation'],generation,generation)
    for key in ('run_id','model_id','model_version','input_recipe','hpo_spec_sha256',*_CLASSIFICATION):
        if canonical(update[key])!=canonical(state[key]): raise ContractError('owned update identity mismatch')
    batch=f'batch-{generation-1:06}'
    if (update['status']!='complete' or update['health']!={} or update['parent_sha256']!=current['parent_sha256']
            or update['batch_id']!=batch or update['collection']!='collections/'+batch or update['numeric']!='numeric/'+batch):
        raise ContractError('owned update lineage/namespace/health mismatch')
    if 'storage_binding' in state and update['storage_binding']!=state['storage_binding']:
        raise ContractError('owned update storage binding mismatch')
    proof=closed(update['source_proof'],{'schema_version','origin','encoder','manifest_sha256','games','generation','batch_id',
        'selection','selection_sha256','replay_reconciled','snapshot_identity','expected_optimizer_step_delta','context_costs'})
    require_owned_record_version(proof,'proof')
    if (proof['generation']!=generation-1 or type(proof['generation']) is not int
            or proof['batch_id']!=batch or proof['replay_reconciled'] is not False
            or proof['snapshot_identity']!=previous['snapshot_identity']
            or checksum(proof['manifest_sha256'])!=update['collection_manifest_sha256']
            or digest(proof['selection'])!=proof['selection_sha256']):
        raise ContractError('owned capture proof parent/selection mismatch')
    selection=_selection(proof['selection'],state,previous,generation-1)
    recipe=hpo_training_config(hpo_spec(state)['training_recipe'])
    used=selection['used_decisions']; collected=selection['collected_decisions']
    delta=recipe.epochs*((used+recipe.minibatch_size-1)//recipe.minibatch_size)
    for key,value in {'used_decisions':used,'accepted_decisions':used,'collected_decisions':collected,
            'overshoot_decisions':collected-used,'expected_optimizer_step_delta':delta,
            'actual_games':recipe.collection_games,'planned_games':recipe.collection_games}.items():
        integer(update[key],value,value)
    integer(proof['games'],recipe.collection_games,recipe.collection_games)
    integer(proof['expected_optimizer_step_delta'],delta,delta)
    if proof['context_costs']!=learner_context_costs(selection,recipe):
        raise ContractError('owned proof context counts mismatch')
    for key,amount in (('used_decisions',used),('collected_decisions',collected),('optimizer_steps',delta)):
        if current[key]!=previous[key]+amount: raise ContractError('owned cumulative counters mismatch')
    seats=update['learner_seats']
    if type(seats) is not dict or len(seats)!=recipe.collection_games: raise ContractError('owned full learner seats required')
    for name,seat in seats.items():
        from gf_models.structured_sealed import basename
        basename(name); integer(seat,1,2)
    # Same external numerical count/finite validator used at actual Wire7 ACK.
    from types import SimpleNamespace

    from gf_training.hpo_owned_process import _owned_metrics
    metrics=update['metrics']
    extras={'elapsed_seconds','save_seconds','peak_cpu_memory_bytes','snapshot_seconds','batch_completion_wait_seconds','batch_send_seconds','snapshot_receive_seconds'}
    for key in extras & set(metrics):
        if key=='peak_cpu_memory_bytes': integer(metrics[key],0)
        else: finite_number(metrics[key],0,1e20)
    _owned_metrics({key:value for key,value in metrics.items() if key not in extras},batch=SimpleNamespace(collected_decisions=collected,used_decisions=used,
        locations=selected_locations(selection)),training=recipe,expected_steps=current['optimizer_steps'])
    metrics=update['metrics']
    for key,value in learner_context_costs(selection,recipe).items():
        integer(metrics[key],value,value)
    if metrics['parity']!={'status':'not_run'}: raise ContractError('owned omitted parity required')
    for key in ('collection_seconds','update_seconds'): finite_number(update[key],0,1e20)
    files=update['numeric_files']
    if type(files) is not dict or files.get('rollout.json')!=checksum(update['numeric_sha256']):
        raise ContractError('owned numeric file-map mismatch')
    for name,sha in files.items():
        if name!='rollout.json' and re.fullmatch(r'rollout-[0-9]{6}\.(json|npz)',name) is None:
            raise ContractError('owned numeric namespace mismatch')
        checksum(sha)


def _owned_cache_journal(scan: Any, state: Json, records: list[Json], registry: Json) -> None:
    from gf_common.hpo_resident import CACHE_OWNER_KEYS, cache_owner, checked_cache_release
    from gf_training.hpo_source import hpo_binding
    root=scan.root/'cache-references'
    refs=registry['cache_references']
    if type(refs) is not dict: raise ContractError('owned cache reference map required')
    if not root.exists():
        if refs: raise ContractError('owned cache journal missing')
        return
    if root.is_symlink() or not root.is_dir(): raise ContractError('owned cache namespace invalid')
    with os.scandir(root) as entries:
        names=set()
        for entry in entries:
            scan.charge(1024)
            if not entry.is_dir(follow_symlinks=False): raise ContractError('foreign owned cache namespace entry')
            names.add(entry.name)
    if names!=set(refs): raise ContractError('owned cache transaction inventory mismatch')
    for token,row in refs.items():
        if type(token) is not str or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',token) is None:
            raise ContractError('owned cache token invalid')
        closed(row,{'phase','binding','receipt_sha256'})
        binding=closed(row['binding'],{'run_id','token','generation','inference_sha256','generation_sha256',
            'policy_sha256','weights_sha256','snapshot_identity','cache_identity'}|CACHE_OWNER_KEYS)
        number=integer(binding['generation'],0,len(records)-1)
        record=records[number]
        if (binding['run_id']!=state['run_id'] or binding['token']!=token
                or binding['inference_sha256']!=record['inference_sha256']
                or binding['generation_sha256']!=scan.verified_digest(generation_path(scan.root,number))
                or binding['snapshot_identity']!=record['snapshot_identity']):
            raise ContractError('owned historical cache source identity mismatch')
        cache_owner({key:binding[key] for key in CACHE_OWNER_KEYS})
        _,policy=hpo_binding(scan.root,state,record,number)
        policy_sha=digest(asdict(policy))
        weights=digest({key:sha for key,sha in record['artifact_files'].items() if key.startswith('weights-')})
        if (binding['policy_sha256']!=policy_sha or binding['weights_sha256']!=weights
                or binding['cache_identity']!=digest({'recipe':'owned-inference-cache-v1','policy_sha256':policy_sha,
                    'inference_sha256':record['inference_sha256'],'snapshot_identity':record['snapshot_identity']})):
            raise ContractError('owned historical cache binding mismatch')
        phase=row['phase']
        if phase!='release': raise ContractError('cold parent retains unresolved cache custody')
        names={'acquire.intent.json','acquire.receipt.json','release.intent.json','release.receipt.json'}
        scan.inventory(root/token,names)
        for current in ('acquire','release'):
            intent=scan.object(root/token/(current+'.intent.json'))
            receipt=scan.object(root/token/(current+'.receipt.json'))
            require_owned_record_version(intent,'cache'); require_owned_record_version(receipt,'cache')
            closed(intent,{'schema_version','origin','encoder','phase','binding'}|({'evidence'} if current=='release' else set()))
            if intent!=receipt or intent['phase']!=current or intent['binding']!=binding:
                raise ContractError('owned cache transaction completion mismatch')
            if current=='release': checked_cache_release(binding,intent['evidence'])
        if row['receipt_sha256']!=scan.verified_digest(root/token/'release.receipt.json'):
            raise ContractError('owned cache receipt hash mismatch')


def _read_owned_metadata(scan: Any, *, require_complete: bool=False) -> tuple[Json,list[Json],Json]:
    from gf_training.hpo_lineage import (
        _generation,
        hpo_state,
        protected_hpo_files_for_operating,
        reference_journal,
    )
    state=scan.object(scan.root/'run.json')
    hpo_state(state)
    from gf_training.structured_lineage import _artifact_directories
    _artifact_directories(scan.root,state['updates'],scan)
    if state['run_variant']!='hpo-ppo-owned-v1': raise ContractError('owned import requires exact owned Run5')
    if require_complete and state['status']!='complete': raise ContractError('complete owned final-D parent required')
    count=state['updates']
    if type(scan) is _ColdScan:
        scan.directories(scan.root/'artifacts',{f'{number:06}' for number in range(count+1)})
        for namespace in ('collections','numeric'):
            scan.directories(scan.root/namespace,{f'batch-{number:06}' for number in range(count)})
    for directory,numbers in (('generations',range(count+1)),('updates',range(1,count+1))):
        scan.inventory(scan.root/directory,{f'{number:06}.json' for number in numbers})
    latest=closed(scan.object(scan.root/'latest.json'),{'generation','sha256'})
    if latest!={'generation':count,'sha256':state['generation_sha256']}: raise ContractError('owned latest/run mismatch')
    records: list[Json]=[]
    parent=None; pool=[0]
    for number in range(count+1):
        record=scan.object(generation_path(scan.root,number))
        pool=_generation(record,state,number,parent,records[-1] if records else None,pool)
        if number:
            update=scan.object(scan.root/'updates'/f'{number:06}.json',record['update_sha256'])
            _owned_update_metadata(update,state,records[-1],record)
            for namespace in ('collections','numeric'):
                path=scan.root/namespace/f'batch-{number-1:06}'
                if not path.is_dir() or path.is_symlink(): raise ContractError('owned historical output namespace missing')
            # Preserve numeric namespace/presence without reading old numerics.
            scan.inventory(scan.root/update['numeric'],set(update['numeric_files']))
            scan.inventory(scan.root/update['collection'],{'collection.locator.json'})
            _archive_presence(scan,state,update)
        records.append(record); parent=scan.verified_digest(generation_path(scan.root,number))
    if parent!=state['generation_sha256']: raise ContractError('owned generation chain digest mismatch')
    for key in ('used_decisions','collected_decisions','accepted_decisions','optimizer_steps'):
        if records[-1][key]!=state[key]: raise ContractError('owned run/generation counters differ')
    from gf_training.hpo_source import _initial_schedule, recorded_hpo_plan
    _initial_schedule(state,records)
    for number in range(count):
        plan,_label,after=recorded_hpo_plan(scan.root,state,records,number,scan.deadline)
        update=scan.object(scan.root/'updates'/f'{number+1:06}.json',records[number+1]['update_sha256'])
        if (after!=records[number+1]['parent_controls']['schedule_opponent_rng']
                or canonical(update['policy_bindings'])!=canonical({label:asdict(binding) for label,binding in plan.bindings.items()})
                or update['learner_seats']!=plan.linkage['learner_seats']):
            raise ContractError('owned recorded schedule/registry/seats mismatch')
    registry=scan.object(scan.root/'artifact-registry.json')
    require_owned_record_version(registry,'registry')
    closed(registry,{'schema_version','origin','encoder','generations','evaluation_references','cache_references'})
    if set(registry['generations'])!={str(number) for number in range(count+1)}:
        raise ContractError('owned registry generation inventory mismatch')
    _owned_cache_journal(scan,state,records,registry)
    pending=reference_journal(scan,state,records,registry)
    protected=protected_hpo_files_for_operating(state['effective_config']['hpo_spec']['operating'],records,
        {'evaluation_references':registry['evaluation_references'],'cache_references':{}},pending)
    retirement_files=set()
    for number,record in enumerate(records):
        entry=closed(registry['generations'][str(number)],{'files'})['files']
        if set(entry)!=set(record['artifact_files']): raise ContractError('owned retained file-map mismatch')
        retained=set()
        for name,value in entry.items():
            status=_owned_retirement_status(scan,number,name,value,record['artifact_files'][name])
            if status=='deleted': retirement_files.update({f'{number:06}-{name}.intent.json',f'{number:06}-{name}.receipt.json'})
            if status=='retained': retained.add(name)
            elif status!='deleted': raise ContractError('owned uncertain retirement cannot import')
        if not protected[number]<=retained: raise ContractError('owned protected file missing')
        inference_files={name for name in record['artifact_files'] if name=='inference.json' or name.startswith('weights-')}
        if ('resume.json' in retained and not set(record['artifact_files'])<=retained
                or 'inference.json' in retained and not inference_files<=retained):
            raise ContractError('owned retained checkpoint dependency pair incomplete')
        scan.inventory(checkpoint_path(scan.root,number),retained)
    retention=scan.root/'retention'
    if retirement_files or retention.exists() or retention.is_symlink(): scan.inventory(retention,retirement_files)
    _check_archive_presence(scan)
    scan.finalize()
    return state,records,registry


def _close_import_bankset(bankset: ReadonlyRamBankSet) -> None:
    """Retire independent backing owners even if another has an export/uncertainty."""
    failures=[]
    for bank in bankset.banks:
        try: bank.close()
        except BaseException as error: failures.append(error)  # noqa: BLE001 - all original bank owners must be attempted
    if failures:
        first=failures[0]
        for other in failures[1:]: first.add_note('additional imported backing retirement: '+str(other))
        raise first


class _ColdSnapshot:
    """Inference-only pool owner or current pair from an actual validated import."""
    def __init__(self, token: object, importer: OwnedColdImport, record: Json, inference: Any,
                 resume: Any, control: Any) -> None:
        if token is not _TOKEN: raise ContractError('actual cold import required')
        self._importer=importer
        self.inference,self.resume,self._control=inference,resume,control
        self.generation,self.identity=record['generation'],record['snapshot_identity']
        self.optimizer_steps=record['optimizer_steps']
        self._closed=False; self._actor_capability: Any=None; self._process: Any=None

    def __reduce__(self) -> Any: raise TypeError('cold snapshot ownership cannot be copied')

    def require_process(self, process: Any) -> None:
        if self._closed or self._process is not process or self._importer.snapshots.get(self.generation) is not self:
            raise ContractError('cold pool owner differs from actual import/process')
        self._importer.require(allow_consumed=True)

    @property
    def files(self) -> Json: return dict(self._control.value['files'])
    @property
    def inference_manifest(self) -> Json: return clone(self._control.value['inference_manifest'])
    @property
    def actor_metadata(self) -> Json: return clone(self._control.value['actor_metadata'])
    @property
    def actor_capability(self) -> Any:
        from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
        if self._actor_capability is None:
            self._actor_capability=_issue_actor_snapshot(self,inference=self.inference,metadata=self.actor_metadata,
                check_origin=_check_cold_origin,budget=self._importer.budget)
        self._actor_capability.require()
        return self._actor_capability

    def close(self) -> None:
        if self._closed: return
        if self._actor_capability is not None and self._actor_capability._state.bindings:
            raise BufferError('cold actor bindings remain live')
        failures=[]
        for owner in (self.inference,self.resume,self._control):
            if owner is None: continue
            try:
                if type(owner) is ReadonlyRamBankSet: _close_import_bankset(owner)
                else: owner.close()
            except BaseException as error: failures.append(error)  # noqa: BLE001 - every other actual storage owner must retire
        if failures:
            first=failures[0]
            for other in failures[1:]: first.add_note('additional cold snapshot retirement: '+str(other))
            raise first
        if self._actor_capability is not None: self._actor_capability._retire_origin(self)
        self._closed=True


def _check_cold_origin(owner: _ColdSnapshot) -> None:
    if type(owner) is not _ColdSnapshot or owner._closed: raise ContractError('retired cold inference owner')
    owner._importer.require(allow_consumed=True)
    if owner._importer.snapshots.get(owner.generation) is not owner: raise ContractError('cold owner not actually imported')


class OwnedColdImport:
    def __init__(self, ownership: Any, *, budget: Any, parent_budget: ArtifactPhaseAdmission,
                 deadline: float, require_complete: bool=False) -> None:
        self.ownership=require_hpo_run_owner(ownership.directory,ownership)
        if parent_budget.source is not budget: raise ContractError('cold import requires SAME Source account')
        self.budget,self.parent_budget=budget,parent_budget
        self.scan=_ColdScan(ownership,parent_budget,deadline)
        self.snapshots: dict[int,Any]={}
        self._partial_banks: list[ReadonlyRamBankSet]=[]
        self._partial_controls: list[Any]=[]
        self._decodes: list[_DecodeReceipt]=[]
        self._failed=False
        self._closed=False; self._consumed=False
        self._process=None
        try:
            self.state,self.records,self.registry=_read_owned_metadata(self.scan,require_complete=require_complete)
            self._for_resume=not require_complete
            if self._for_resume and (self.state['status']!='paused' or not self.state['updates']):
                raise ContractError('cold restore requires an explicitly paused committed owned parent')
        except BaseException as error:
            self._abort(error)
            raise

    def __reduce__(self) -> Any: raise TypeError('cold import cannot be persisted')

    def require(self, *, allow_consumed: bool=False) -> None:
        require_hpo_run_owner(self.ownership.directory,self.ownership)
        if self._closed or self._failed or self._consumed and not allow_consumed: raise ContractError('cold import retired or consumed')

    def _abort(self, error: BaseException) -> None:
        self._failed=True
        try: self.close()
        except BaseException as cleanup:  # noqa: BLE001 - original validation/I/O failure survives every cleanup failure
            vars(error)['cleanup_owner']=self
            vars(error)['_hpo_owned_import_cleanup_owner']=self
            error.add_note('owned import retirement remains owned: '+str(cleanup))

    def _scope_exit(self, _kind: Any, error: BaseException | None, _traceback: Any) -> bool:
        """Actual cleanup exit; never suppress/replace an existing primary."""
        if error is None:
            self.close()  # Clean return cannot hide incomplete retirement.
        elif not self._closed:
            self._failed=True
            if (getattr(error,'cleanup_owner',None) is self
                    or getattr(error,'_hpo_owned_import_cleanup_owner',None) is self):
                vars(error)['_hpo_owned_import_cleanup_owner']=self
            else:
                self._abort(error)
        return False

    def _decode_bank(self, directory: Path, bundle: Json, *, model: Any, maximum: int) -> ReadonlyRamBankSet:
        from gf_models.arrays import load_arrays
        from gf_models.sharded_arrays import validate_bundle_metadata
        validate_bundle_metadata(bundle,max_decoded_bytes=model.resources.max_owned_bytes)
        envelope=35*max(((directory/row['file']).lstat().st_size for row in bundle['shards']),default=0)+3*bundle['decoded_bytes']+65536
        receipt=_DecodeReceipt()
        self._decodes.append(receipt)
        reservation=self.budget.reserve(envelope)
        decode=_DecodeEnvelope(reservation,receipt)
        arrays: dict[str,Any]={}
        decoded=None
        for shard in bundle['shards']:
            sidecar=self.scan.object(directory/Path(shard['file']).with_suffix('.json'),bundle['files'][Path(shard['file']).with_suffix('.json').name])
            if sidecar!=shard['metadata']: raise ContractError('cold shard sidecar mismatch')
            with self.scan.file(directory/shard['file'],bundle['files'][shard['file']]) as (source,info):
                decoded=load_arrays(directory/shard['file'],sidecar,_check=self.scan.check,_on_array=decode.hold,
                    _source=source,_snapshot=tuple(getattr(info,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')))
                arrays.update(decoded)
                decoded=None
        bankset=ReadonlyRamBankSet.build(arrays,self.budget,max_bytes=maximum)
        self._partial_banks.append(bankset)  # Before any later validation/copy can fail.
        arrays.clear()  # No decoded dictionary/array survives envelope retirement on success.
        return bankset

    def load(self, *, model: Any, binding: Any, actions: Any, rules: Any, final_only: bool=False) -> None:
        self.require()
        if self.snapshots: raise ContractError('cold arrays already imported')
        from gf_models.structured_sealed import validate_game_inference_metadata_v4
        from gf_training.hpo_owned_worker import _Control
        numbers={self.state['updates']}
        if not final_only: numbers.update(self.records[-1]['parent_controls']['snapshot_pool'])
        try:
            for number in sorted(numbers):
                record=self.records[number]; directory=checkpoint_path(self.scan.root,number)
                manifest=self.scan.object(directory/'inference.json',record['inference_sha256'])
                validate_game_inference_metadata_v4(manifest,actions=actions,rules=rules)
                full=number==self.state['updates'] and not final_only
                resume_manifest=self.scan.object(directory/'resume.json',record['resume_sha256']) if full else None
                inference_files={'inference.json':record['inference_sha256'],**manifest['weights']['files']}
                original_inference={name:sha for name,sha in record['artifact_files'].items() if name=='inference.json' or name.startswith('weights-')}
                if inference_files!=original_inference:
                    raise ContractError('cold inference bundle differs from committed original file map')
                if full:
                    assert resume_manifest is not None
                    if (resume_manifest['state'].get('prefix')!='resume'
                            or resume_manifest.get('inference_digest')!=record['inference_sha256']
                            or resume_manifest.get('optimizer_steps')!=record['optimizer_steps']
                            or {**inference_files,'resume.json':record['resume_sha256'],**resume_manifest['state']['files']}!=record['artifact_files']):
                        raise ContractError('cold resume bundle differs from committed pair/file map')
                banks: list[ReadonlyRamBankSet]=[]
                for field,document in [('weights',manifest)]+([('state',resume_manifest)] if full else []):
                    assert document is not None
                    maximum=min(MAX_BANK_BYTES,self.state['effective_config']['hpo_spec']['operating']['save_budget_bytes']-sum(bankset.admitted_bytes for bankset in banks))
                    banks.append(self._decode_bank(directory,document[field],model=model,maximum=maximum))
                metadata={'protocol_version':9,'command':'SYNC_OWNED','generation':number,'snapshot_identity':record['snapshot_identity'],
                    'inference_digest':record['inference_sha256'],'inference':{key:value for key,value in manifest.items() if key!='weights'},
                    'arrays':[asdict(descriptor) for bank in banks[0].banks for descriptor in bank.descriptors]}
                from gf_models.hpo_owned_inference import _validate_owned_inference
                _validate_owned_inference(banks[0],metadata,model=model,binding=binding,actions=actions,rules=rules)
                value={'actor_metadata':metadata,'inference_manifest':manifest,'files':record['artifact_files'],
                    'generation':number,'snapshot_identity':record['snapshot_identity'],'optimizer_steps':record['optimizer_steps'],
                    'resume_manifest':resume_manifest}
                control=_Control(value,self.parent_budget.reserve(32*json_bound(value)+65536),32*json_bound(value)+65536)
                self._partial_controls.append(control)
                self.snapshots[number]=_ColdSnapshot(_TOKEN,self,record,banks[0],banks[1] if full else None,control)
                for bankset in banks: self._partial_banks.remove(bankset)
                self._partial_controls.remove(control)
            self.scan.finalize()
        except BaseException as error:
            self._abort(error)
            raise

    def close(self) -> None:
        if self._closed: return
        failures=[]
        for number,snapshot in tuple(self.snapshots.items()):
            try: snapshot.close()
            except BaseException as error: failures.append(error)  # noqa: BLE001 - drain all exact independent owners
            else: del self.snapshots[number]
        for collection in (self._partial_banks,self._partial_controls):
            for owner in tuple(collection):
                try:
                    if type(owner) is ReadonlyRamBankSet: _close_import_bankset(owner)
                    else: owner.close()
                except BaseException as error: failures.append(error)  # noqa: BLE001 - known view or uncertainty stays tracked
                else: collection.remove(owner)
        for receipt in tuple(self._decodes):
            if receipt.closed: self._decodes.remove(receipt)
            elif receipt.error is not None: failures.append(receipt.error)
            else: failures.append(BufferError('actual decode buffers/traceback still retain admission'))
        if failures:
            first=failures[0]
            vars(first)['cleanup_owner']=self
            for other in failures[1:]: first.add_note('additional import retirement: '+str(other))
            raise first
        self.scan.close()
        self._closed=True


def import_owned_final_policy(directory: Path, *, deadline: float, max_owned_bytes: int,
                              parent_budget_bytes: int, budget: Any=None, parent_budget: Any=None) -> Any:
    """New external final-policy owner; no historical samples or execution grant."""
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.group_pack import StorageSettings
    from gf_models.structured_config import InferenceBindingV2
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_owned_completion import _detach_external_policy
    from gf_training.lineage import run_lock
    if budget is None:
        budget=StorageBudget(StorageSettings(budget_bytes=max_owned_bytes),deadline)
    if budget.limit!=max_owned_bytes: raise ContractError('external final import Source limit mismatch')
    phase=ArtifactPhaseAdmission(budget,parent_budget_bytes) if parent_budget is None else parent_budget
    if type(phase) is not ArtifactPhaseAdmission or phase.source is not budget or phase.limit!=parent_budget_bytes:
        raise ContractError('external final import requires SAME Source/parent phase')
    with run_lock(directory) as ownership:
        imported=OwnedColdImport(ownership,budget=budget,parent_budget=phase,deadline=deadline,require_complete=True)
        candidate=None
        try:
            actions,model,_=hpo_state(imported.state)
            imported.load(model=model,binding=InferenceBindingV2(**imported.state['execution']['inference']['binding']),
                actions=actions,rules=card_instance_rules(),final_only=True)
            candidate=_detach_external_policy(imported)
            imported._scope_exit(None,None,None)
            return candidate
        except BaseException as error:
            if candidate is not None:
                try: candidate.close()
                except BaseException as cleanup:  # noqa: BLE001 - detached owner is independent of importer cleanup
                    vars(error)['hpo_final_policy_cleanup_owner']=candidate
                    error.add_note('detached final policy remains owned: '+str(cleanup))
            imported._scope_exit(type(error),error,error.__traceback__)
            raise


def audit_owned_artifacts(scan: Any, state: Json, records: list[Json], registry: Json) -> None:
    """Explicit offline structural audit of all retained checkpoints, no grant."""
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.structured_artifacts import adam_recipe
    from gf_models.structured_sealed import (
        inspect_retained_game_artifact_v4,
        inspect_retained_game_inference_v4,
    )
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_lineage import hpo_state
    actions,model,_=hpo_state(state)
    recipe=hpo_training_config(state['effective_config']['hpo_spec']['training_recipe'])
    optimizer=adam_recipe(recipe.adam_learning_rate,recipe.adam_beta1,recipe.adam_beta2,recipe.adam_epsilon)
    for number,record in enumerate(records):
        retained={name for name,row in registry['generations'][str(number)]['files'].items() if row['status']=='retained'}
        directory=checkpoint_path(scan.root,number)
        for name in retained:
            with scan.file(directory/name,record['artifact_files'][name]): pass
        if 'resume.json' in retained:
            artifact=inspect_retained_game_artifact_v4(directory,expected_inference_digest=record['inference_sha256'],
                actions=actions,rules=card_instance_rules(),training_recipe=asdict(recipe),optimizer_recipe=optimizer,scan=scan)
            resume=artifact.resume
            if (resume is None or resume['resume_identity']!=state['resume_identity'] or resume['optimizer_steps']!=record['optimizer_steps']
                    or canonical(resume['model'])!=canonical(asdict(model)) or resume['binding']!=state['execution']['learner']['binding']):
                raise ContractError('owned external checkpoint resume mismatch')
        elif 'inference.json' in retained:
            artifact=inspect_retained_game_inference_v4(directory,expected_inference_digest=record['inference_sha256'],actions=actions,rules=card_instance_rules(),scan=scan)
        else: continue
        if artifact.files!={name:record['artifact_files'][name] for name in retained}:
            raise ContractError('owned external retained artifact map mismatch')


def audit_owned_samples(scan: Any, state: Json, records: list[Json], *, deadline: float) -> Json:
    """Full external journals/Replay/numeric audit, NEVER a Source/live producer."""
    from gf_common.card_instance_collection import hpo_batch_role
    from gf_common.card_instance_rules import card_instance_rules
    from gf_training.card_instance_source import _check_collection_parent
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_rollout import read_hpo_rollout
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.structured_source import _inspect_component_collection
    actions,model,_=hpo_state(state)
    recipe=hpo_training_config(state['effective_config']['hpo_spec']['training_recipe'])
    scan.configure_packed(state)
    games=decisions=0
    for number in range(state['updates']):
        plan,label,_=recorded_hpo_plan(scan.root,state,records,number,deadline)
        original=scan.packed_owner(plan.output)
        original.charge(scan.retained)
        collection=_inspect_component_collection(original,actions,'train',_game=True,_game_role=hpo_batch_role(recipe.collection_games))
        manifest=collection.manifest
        _check_collection_parent(manifest,state,plan)
        update=scan.object(scan.root/'updates'/f'{number+1:06}.json',records[number+1]['update_sha256'])
        if original.files['manifest.json']['sha256']!=update['collection_manifest_sha256']:
            raise ContractError('owned archive manifest differs from saved update')
        selected=tuple(row.records for row in collection.episodes if plan.linkage['learner_seats'][row.episode_id]==row.viewer)
        counts=[len(record.frames) for record in selected]
        if len(selected)!=recipe.collection_games or counts!=update['source_proof']['selection']['episode_counts']:
            raise ContractError('owned complete original learner population differs')
        rollout=read_hpo_rollout(scan.root/update['numeric'],update['numeric_sha256'],model,recipe,
            actions=actions,rules=card_instance_rules(),expected_selection=update['source_proof']['selection'],
            expected_records=selected,expected_snapshot_identity=records[number]['snapshot_identity'],
            numeric_budget_bytes=state['effective_config']['hpo_spec']['operating']['numeric_budget_bytes'],
            max_owned_bytes=scan.limit,extra_owned_bytes=scan.retained+original.used,check=original.check,
            reserve=lambda amount,owner=original:owner.charge(amount))
        rollout.require_learning()
        if rollout.policy_version!=label: raise ContractError('owned numeric original learner identity differs')
        games+=len(selected); decisions+=len(rollout)
        del rollout,selected,collection
        original.finalize()
    scan.finalize()
    return {'schema_version':1,'kind':'hpo-owned-external-sample-audit','origin':'producer-owned-v1',
        'encoder':'canonical-python','qualified':False,'parent_qualification':'not_performed',
        'games':games,'used_decisions':decisions,'generation_sha256':state['generation_sha256']}
