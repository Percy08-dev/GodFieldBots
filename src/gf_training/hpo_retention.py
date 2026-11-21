"""HPO-only owned file retirement and evaluation-reference write-ahead records."""
from __future__ import annotations

import os
import secrets
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, integer
from gf_training.hpo_lineage import (
    CACHE_REFERENCE_DIRECTORY,
    REFERENCE_DIRECTORY,
    cache_reference_journal,
    checked_cache_reference_binding,
    checked_reference_binding,
    protected_hpo_files,
    reference_journal,
    sync_hpo_ancestry,
    validate_hpo_chain,
    write_hpo_control,
)
from gf_training.lineage import (
    RunOwnership,
    checkpoint_path,
    generation_path,
    require_hpo_run_owner,
    run_lock,
)
from gf_training.structured_lineage import (
    _artifact_directories,
    _ChainScan,
    _directory_entry,
    _prune_parent,
    _stat_identity,
)
from gf_training.structured_lineage import (
    _retire_staged_file as _delete_staged_file,
)


@dataclass(frozen=True)
class HpoReferenceReceipt:
    binding: Json
    receipt_sha256: str
    phase: str


def _write_control(path: Path, value: Json, **options: Any) -> str:
    return write_hpo_control(path, value, **options)


def _mkdir(path: Path, *, ownership: RunOwnership) -> None:
    require_hpo_run_owner(ownership.directory,ownership)
    if not path.is_relative_to(ownership.directory):
        raise ContractError('HPO reference namespace lies outside actual owned parent')
    if not path.exists():
        path.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ContractError('HPO reference control path is not an owned directory')
    sync_hpo_ancestry(path,ownership=ownership)


def _reference_parent(scan: Any, *, max_owned_bytes: int, deadline: float) -> tuple[Json,list[Json]]:
    state=scan.object(scan.root/'run.json')
    if state.get('run_variant') in ('hpo-ppo-owned-v1','hpo-ppo-owned-shared-v1'):
        from gf_training.hpo_owned_restore import _read_owned_metadata
        state,records,_registry=_read_owned_metadata(scan)
        return state,records
    return validate_hpo_chain(scan.root,max_owned_bytes=max_owned_bytes,deadline=deadline,scan=scan)


def _transition(directory: Path, binding: Json, phase: str, *, max_owned_bytes: int,
                deadline: float, ownership: RunOwnership | None = None, cache: bool = False, release_evidence: Json | None = None) -> HpoReferenceReceipt:
    with ExitStack() as scope:
        if ownership is None:
            ownership=scope.enter_context(run_lock(directory))
        ownership=require_hpo_run_owner(directory,ownership)
        scan=scope.enter_context(_ChainScan(directory=directory,max_owned_bytes=max_owned_bytes,deadline=deadline))
        state, records = _reference_parent(scan,max_owned_bytes=max_owned_bytes,deadline=deadline)
        checker = checked_cache_reference_binding if cache else checked_reference_binding
        journal = cache_reference_journal if cache else reference_journal
        checker(binding,state,records,scan)
        registry=scan.object(directory/'artifact-registry.json')
        token=binding['token']
        root=directory/(CACHE_REFERENCE_DIRECTORY if cache else REFERENCE_DIRECTORY)
        kind = 'hpo-cache-reference-' if cache else 'hpo-evaluation-reference-'
        refs = registry['cache_references' if cache else 'evaluation_references']
        folder=root/token
        # An existing namespace is admitted by the reader. Do not adopt a foreign
        # empty token directory as a successful transaction.
        if cache:
            scan._hpo_cache_reference_snapshot=None
        else:
            scan._hpo_reference_snapshot=None
        if phase=='acquire':
            _mkdir(root,ownership=ownership)
            _mkdir(folder,ownership=ownership)
        elif not folder.is_dir() or folder.is_symlink():
            raise ContractError('HPO reference release requires original acquisition evidence')
        intent_path=folder/(phase+'.intent.json')
        receipt_path=folder/(phase+'.receipt.json')
        release_receipt=folder/'release.receipt.json'
        if phase=='acquire' and (release_receipt.exists() or (folder/'release.intent.json').exists()):
            raise ContractError('released HPO reference token is terminal')
        if intent_path.exists():
            intent=scan.object(intent_path)
            if canonical(intent['binding'])!=canonical(binding):
                raise ContractError('HPO reference retry changed its exact binding')
        else:
            if phase=='release' and not (folder/'acquire.receipt.json').exists():
                raise ContractError('HPO reference release lacks completed acquisition')
            intent={'schema_version':1,'kind':kind+phase+'-intent',
                'operation_id':str(uuid.uuid4()),'binding':binding}
            if phase=='release':
                scan.object(folder/'acquire.receipt.json')
                intent['acquire_receipt_sha256']=scan.verified_digest(folder/'acquire.receipt.json')
                if cache:
                    from gf_common.hpo_resident import checked_cache_release
                    intent['release_evidence'] = checked_cache_release(binding,release_evidence)
            _write_control(intent_path,intent,scan=scan,immutable=True,ownership=ownership)
        intended=binding if cache else {'generation':binding['generation'],'inference_sha256':binding['inference_sha256']}
        actual=refs.get(token)
        if receipt_path.exists():
            receipt=scan.object(receipt_path)
            # Full journal validation binds the intent and immutable receipt.
            journal(scan,state,records,registry)
            if phase=='acquire' and canonical(actual)!=canonical(intended) or phase=='release' and actual is not None:
                raise ContractError('completed HPO reference transaction changed registry fact')
            scan.finalize()
            return HpoReferenceReceipt(dict(binding),scan.verified_digest(receipt_path),phase)
        if actual is not None and canonical(actual)!=canonical(intended):
            raise ContractError('HPO active reference has foreign binding')
        if phase=='acquire':
            refs[token]=intended
        else:
            refs.pop(token,None)
        # The matching durable intent is the ONLY authority to reconcile an
        # already-mutated registry when the prior process stopped before receipt.
        registry_sha=_write_control(directory/'artifact-registry.json',registry,scan=scan,ownership=ownership)
        receipt={'schema_version':1,'kind':kind+phase+'-receipt','binding':binding,
            'intent_sha256':scan.verified_digest(intent_path),'registry_sha256':registry_sha,'confirmed_unix':time.time()}
        sha=_write_control(receipt_path,receipt,scan=scan,immutable=True,ownership=ownership)
        # Own writes change token directory inventories. Rebuild only these
        # known namespace snapshots, then perform the last reportable checks.
        scan.inventories.pop(folder,None)
        journal(scan,state,records,registry)
        scan.finalize()
        return HpoReferenceReceipt(dict(binding),sha,phase)


def pin_hpo_evaluation(directory: Path, token: str, generation: int, inference_sha256: str, *,
                       max_owned_bytes: int, deadline: float) -> HpoReferenceReceipt:
    directory=directory.resolve()
    with (run_lock(directory) as ownership,
            _ChainScan(directory=directory,max_owned_bytes=max_owned_bytes,deadline=deadline) as scan):
        state,records=_reference_parent(scan,max_owned_bytes=max_owned_bytes,deadline=deadline)
        integer(generation,0,len(records)-1)
        binding={'run_id':state['run_id'],'token':token,'generation':generation,
            'inference_sha256':inference_sha256,'generation_sha256':scan.verified_digest(generation_path(directory,generation))}
        checked_reference_binding(binding,state,records,scan)
        registry=scan.object(directory/'artifact-registry.json')
        expected={name for name in records[generation]['artifact_files'] if name=='inference.json' or name.startswith('weights-')}
        if any(registry['generations'][str(generation)]['files'][name]['status']!='retained' for name in expected):
            raise ContractError('HPO evaluation pin requires actual retained inference and weights')
        scan.finalize()
        return _transition(directory,binding,'acquire',max_owned_bytes=max_owned_bytes,deadline=deadline,ownership=ownership)


def release_hpo_evaluation(directory: Path, binding: Json, *, max_owned_bytes: int,
                           deadline: float) -> HpoReferenceReceipt:
    return _transition(directory.resolve(),binding,'release',max_owned_bytes=max_owned_bytes,deadline=deadline)


def retain_hpo_generations(directory: Path, generation: int, *, max_owned_bytes: int, deadline: float,
                            ownership: RunOwnership | None = None) -> None:
    owner=require_hpo_run_owner(directory,ownership)
    directory=owner.directory
    with _ChainScan(directory=directory,max_owned_bytes=max_owned_bytes,deadline=deadline) as scan:
        state,records=validate_hpo_chain(directory,max_owned_bytes=max_owned_bytes,deadline=deadline,scan=scan)
        integer(generation,state['updates'],state['updates'])
        registry=scan.object(directory/'artifact-registry.json')
        pending=reference_journal(scan,state,records,registry)
        if 'cache_references' in registry:
            pending += cache_reference_journal(scan,state,records,registry)
        protected=protected_hpo_files(state,records,registry,pending)
        for number,record in enumerate(records):
            targets={name:sha for name,sha in record['artifact_files'].items()
                if name not in protected[number] and registry['generations'][str(number)]['files'][name]['status']=='retained'}
            if targets:
                _delete_files(directory,number,targets,registry,scan,ownership=owner)


def _delete_files(directory: Path, number: int, targets: Json, registry: Json, scan: _ChainScan, *,
                    ownership: RunOwnership) -> None:
    require_hpo_run_owner(directory,ownership)
    parent=directory/'artifacts'
    path=checkpoint_path(directory,number)
    staging='prune-'+secrets.token_hex(16)
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    parent_fd=os.open(parent,flags)
    source_fd=stage_fd=None
    assert scan.artifact_snapshot is not None
    original_parent=(scan.artifact_snapshot[0],scan.artifact_snapshot[1])
    try:
        scan.finalize()
        _prune_parent(parent,parent_fd,original_parent,scan)
        source_fd=os.open(path.name,flags,dir_fd=parent_fd)
        _directory_entry(parent_fd,path.name,source_fd)
        os.mkdir(staging,0o700,dir_fd=parent_fd)
        _prune_parent(parent,parent_fd,original_parent,scan,advance=True)
        stage_fd=os.open(staging,flags,dir_fd=parent_fd)
        entries=registry['generations'][str(number)]['files']
        operation=str(uuid.uuid4())
        for name,sha in targets.items():
            with scan.file(path/name,sha) as (_source,info):
                entries[name].update(status='deleting',deletion_intent={'operation_id':operation,
                    'staging_basename':staging,'sha256':sha,'original_identity':list(_stat_identity(info))},deletion_receipt=None)
        _write_control(directory/'artifact-registry.json',registry,scan=scan,ownership=ownership)
        for name,sha in targets.items():
            require_hpo_run_owner(directory,ownership)
            _delete_staged_file(parent,parent_fd,original_parent,path,source_fd,staging,stage_fd,name,sha,scan)
        os.fsync(source_fd)
        os.fsync(stage_fd)
        _directory_entry(parent_fd,staging,stage_fd)
        require_hpo_run_owner(directory,ownership)
        os.rmdir(staging,dir_fd=parent_fd)
        os.fsync(parent_fd)
        _prune_parent(parent,parent_fd,original_parent,scan,advance=True)
        scan.inventories[path]=tuple(sorted(name for name,value in entries.items() if value['status']=='retained'))
        _artifact_directories(directory,scan.artifact_count,scan)
        scan.finalize()
        for name in targets:
            intent=entries[name]['deletion_intent']
            entries[name].update(status='deleted',deletion_receipt={key:intent[key]
                for key in ('operation_id','sha256','original_identity')}|{'completed_unix':time.time()})
        _write_control(directory/'artifact-registry.json',registry,scan=scan,ownership=ownership)
        scan.finalize()
    except BaseException as error:
        error.__dict__['_hpo_retention_pending']=str(path)
        raise
    finally:
        for fd in (stage_fd,source_fd,parent_fd):
            if fd is not None: os.close(fd)


def pin_hpo_cache(directory: Path, token: str, generation: int, inference_sha256: str, *,
                  owner: Json, max_owned_bytes: int, deadline: float,
                  ownership: RunOwnership | None = None) -> HpoReferenceReceipt:
    """Durably protect exact immutable cache bytes BEFORE the worker reads them."""
    from dataclasses import asdict

    from gf_common.domain import digest
    from gf_common.hpo_resident import cache_owner
    from gf_training.hpo_source import hpo_binding
    directory=directory.resolve()
    with ExitStack() as scope:
        if ownership is None:
            ownership=scope.enter_context(run_lock(directory))
        ownership=require_hpo_run_owner(directory,ownership)
        scan=scope.enter_context(_ChainScan(directory=directory,max_owned_bytes=max_owned_bytes,deadline=deadline))
        state,records=validate_hpo_chain(directory,max_owned_bytes=max_owned_bytes,deadline=deadline,scan=scan)
        integer(generation,0,len(records)-1)
        _,policy=hpo_binding(directory,state,records[generation],generation)
        policy_sha=digest(asdict(policy))
        weights=digest({key:sha for key,sha in records[generation]['artifact_files'].items() if key.startswith('weights-')})
        binding={'run_id':state['run_id'],'token':token,'generation':generation,
            'inference_sha256':inference_sha256,'generation_sha256':scan.verified_digest(generation_path(directory,generation)),
            **cache_owner(owner),'policy_sha256':policy_sha,'weights_sha256':weights,
            'cache_identity':digest({'policy_sha256':policy_sha,'inference_sha256':inference_sha256,'weights_sha256':weights})}
        checked_cache_reference_binding(binding,state,records,scan)
        registry=scan.object(directory/'artifact-registry.json')
        expected={name for name in records[generation]['artifact_files'] if name=='inference.json' or name.startswith('weights-')}
        if any(registry['generations'][str(generation)]['files'][name]['status']!='retained' for name in expected):
            raise ContractError('HPO cache pin requires actual retained inference and weights')
        scan.finalize()
        return _transition(directory,binding,'acquire',max_owned_bytes=max_owned_bytes,deadline=deadline,ownership=ownership,cache=True)


def release_hpo_cache(directory: Path, binding: Json, *, authority: Any,
                      max_owned_bytes: int, deadline: float,
                      ownership: RunOwnership | None = None) -> HpoReferenceReceipt:
    """Only SAME live custody's eviction or audited reap permits pin retirement."""
    from gf_training.hpo_resident import CacheReleaseAuthority
    if type(authority) is not CacheReleaseAuthority:
        raise ContractError('original live resident cache-release authority required')
    authority.require(binding)
    return _transition(directory.resolve(),binding,'release',max_owned_bytes=max_owned_bytes,deadline=deadline,ownership=ownership,cache=True,release_evidence=authority.evidence)
