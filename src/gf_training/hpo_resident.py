"""Live resident ownership atop the SAME HPO audited job; no policy or transport I/O.

The original transport owner drives requests/ACKs. Persisted management facts are
never sufficient to reconstruct these unpersistable capabilities.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from typing import Any, Self, cast

from gf_common.domain import ContractError, Json, canonical, clone, digest, integer
from gf_common.hpo_resident import (
    CACHE_OWNER_KEYS,
    REQUEST_KEYS,
    RESIDENT_MODE,
    cache_owner,
    checked_ack,
    checked_cache_release,
    checked_retirement,
)
from gf_training.hpo_gpu_lease import GpuJob, _actual_process, _AuditedLedger, _job, _Worker

_ISSUER = object()


@dataclass
class _Owner:
    worker: _Worker
    owner: Any
    viewer: int
    lease_epoch: int = 0
    generation: int = 0
    state: str = 'idle'
    request_epoch: int = 0
    pending: Json | None = None
    end_ack: Json | None = None
    park_ack: Json | None = None
    sync_ack: Json | None = None
    io: dict[asyncio.Task[Any], Future[Any]] = field(default_factory=dict)
    setups: set[Any] = field(default_factory=set)
    bootstrap: Json | None = None
    checkpoint: str | None = None
    collection: Any = None
    episode_started: bool = False
    binding_identity: str | None = None


class ParkedAuthority:
    def __init__(self, issuer: object, custody: ResidentPoolCustody, epoch: int, evidence: Json) -> None:
        if issuer is not _ISSUER:
            raise ContractError('original resident custody issuer required')
        self._custody, self._epoch, self._evidence = custody, epoch, clone(evidence)

    @property
    def evidence(self) -> Json:
        return clone(self._evidence)

    def require(self, ledger: Any) -> None:
        pool = self._custody
        pool._require()
        if (ledger is not pool.job.ledger or pool._epoch != self._epoch
                or pool._authority is not self):
            raise ContractError('foreign or invalidated resident PARK authority')
        pool._check_parked(self._evidence['generation'])
        if canonical(pool._evidence(self._evidence['generation'], self._evidence['collection_id'])) != canonical(self._evidence):
            raise ContractError('resident PARK original owner/ACK evidence changed')

    def __reduce__(self) -> Any:
        raise TypeError('resident PARK authority cannot be persisted')


def _plan_identity(plan: Any) -> Json:
    return {'groups': [asdict(group) for group in plan.groups],
        'bindings': {label: asdict(binding) for label, binding in plan.bindings.items()},
        'output': str(plan.output), 'linkage': plan.linkage,
        'execution_deadline': plan.execution_deadline, 'wall_deadline': plan.wall_deadline}


class CollectionAdmission:
    """Live precollection scope; never completed Source/update/commit authority."""
    def __init__(self, issuer: object, pool: ResidentPoolCustody, plan: Any) -> None:
        if issuer is not _ISSUER:
            raise ContractError('original resident collection admission required')
        self.pool, self.plan, self.original = pool, plan, canonical(_plan_identity(plan))
        self.generation = plan.linkage['collection_generation'] if pool.evaluation_scope is None else 0
        self.collection_id = plan.linkage['batch_id'] if pool.evaluation_scope is None else plan.linkage['evaluation_id']
        self.phase = 'admitting'
        self.used: set[int] = set()

    def require(self, plan: Any, ledger: Any) -> None:
        self.pool._require()
        if (self.pool._admission is not self or self.phase not in ('admitting', 'collecting')
                or plan is not self.plan or ledger is not self.pool.ledger
                or canonical(_plan_identity(plan)) != self.original):
            raise ContractError('resident collection original plan/ledger/admission changed or retired')

    def __reduce__(self) -> Any:
        raise TypeError('resident collection admission cannot be persisted')


class CacheReleaseAuthority:
    def __init__(self, issuer: object, pool: ResidentPoolCustody, binding: Json, worker: _Worker, *, reaped: bool, evidence: Json) -> None:
        if issuer is not _ISSUER:
            raise ContractError('original resident cache-release issuer required')
        self._pool, self._binding, self._worker, self._reaped = pool, clone(binding), worker, reaped
        self._evidence = clone(checked_cache_release(binding, evidence))

    def require(self, binding: Json) -> None:
        self._pool._require(cleanup=self._reaped)
        if (canonical(binding) != canonical(self._binding)
                or self._pool._releases.get(binding['token']) is not self):
            raise ContractError('foreign resident cache release or changed pin')
        if self._reaped:
            self._pool._check_reaped(self._worker)
        elif binding['cache_identity'] in self._pool._cached.get(self._worker.key, set()):
            raise ContractError('evicted resident cache identity became active again')

    @property
    def evidence(self) -> Json:
        return clone(self._evidence)

    def __reduce__(self) -> Any:
        raise TypeError('resident cache release authority cannot be persisted')


class CacheAcquisition:
    """Original live acquire attempt, registered before any durable write."""
    def __init__(self,issuer: object,pool: ResidentPoolCustody,worker: Any,token: str,generation: int,inference_sha256: str) -> None:
        if issuer is not _ISSUER:
            raise ContractError('original cache acquisition issuer required')
        row=pool._row(worker)
        pool._live(row)
        self.pool,self.worker,self.owner=pool,worker,row.owner
        self.token,self.generation,self.inference_sha256=token,generation,inference_sha256
        self._owner_identity=clone(pool._identity(row))
        cache_owner(self._owner_identity)
        self.receipt: Any = None

    @property
    def owner_identity(self) -> Json:
        return clone(self._owner_identity)

    def require_receipt(self,receipt: Any) -> None:
        from gf_training.hpo_retention import HpoReferenceReceipt
        if (self.pool._acquisitions.get(self.token) is not self or type(receipt) is not HpoReferenceReceipt
                or receipt.phase!='acquire' or receipt.binding['token']!=self.token
                or receipt.binding['generation']!=self.generation
                or receipt.binding['inference_sha256']!=self.inference_sha256
                or receipt.binding['run_id']!=self.pool.run_id
                or canonical({key:receipt.binding[key] for key in CACHE_OWNER_KEYS})!=canonical(self._owner_identity)):
            raise ContractError('cache acquisition receipt differs from SAME original attempt')

    def __reduce__(self) -> Any:
        raise TypeError('live cache acquisition cannot be persisted')


class EvaluationPoolScope:
    """Actual panel admission plus genuine outer evaluation pins/artifact custody."""
    def __init__(self, job: GpuJob, plan: Any, artifacts: Any, pins: tuple[Any, ...], *,
                 max_owned_bytes: int, deadline: float) -> None:
        from pathlib import Path

        from gf_common.card_instance_collection import checked_linkage, hpo_evaluation_role
        from gf_common.hpo_resident import checked_evaluation_runtime
        from gf_common.schedule import serialize_schedule
        from gf_models.card_instance_artifacts import RetainedGameArtifacts
        from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
        from gf_training.hpo_lineage import REFERENCE_DIRECTORY, validate_hpo_chain
        from gf_training.hpo_retention import HpoReferenceReceipt
        from gf_training.structured_lineage import _ChainScan
        if type(job) is not GpuJob or type(artifacts) not in (RetainedGameArtifacts,OwnedGameArtifacts) or type(pins) is not tuple:
            raise ContractError('resident evaluation requires actual job/artifacts/evaluation references')
        plan.require_job(job)
        linkage = plan.linkage
        expected_context = {'campaign_id':linkage['campaign_id'],'operation_id':linkage['evaluation_id'],
            'kind':'evaluation','run_id':linkage['left_parent']['run_id'],'evaluation_id':linkage['evaluation_id']}
        if (job._admission['context'] != expected_context or job.ticket != linkage['gpu_ticket']
                or job.ledger.workers or job.ledger.learner is not None or job.ledger.updating or job.ledger.cuda_bytes):
            raise ContractError('resident evaluation differs from SAME actual idle admission')
        registry = {name:asdict(value) for name,value in plan.bindings.items()}
        checked_linkage(linkage,role=hpo_evaluation_role(sum(len(group.episodes) for group in plan.groups)),
            schedule=serialize_schedule(plan.groups,4),registry=registry)
        runtime = checked_evaluation_runtime(plan.execution_runtime,linkage,registry)
        from gf_models.resident_cache import resident_registry_execution
        resident_registry_execution(plan.bindings)
        parents = [linkage['left_parent']] + ([] if linkage['right_parent'] is None else [linkage['right_parent']])
        unique: dict[str,Json] = {}
        for parent in parents:
            path=str(Path(parent['run_path']).resolve(strict=True))
            normalized={**parent,'run_path':path}
            if path in unique and canonical(unique[path]) != canonical(normalized):
                raise ContractError('resident evaluation has conflicting same-path parent/checkpoint declarations')
            unique[path]=normalized
        parents=list(unique.values())  # Identical immutable self-comparison refs share one final capability.
        supplied=[str(Path(directory).resolve(strict=True)) for directory,_ in pins]
        if len(supplied)!=len(set(supplied)) or set(supplied)!=set(unique):
            raise ContractError('resident evaluation requires exact distinct immutable parent-set pins')
        if len(pins) != len(unique):
            raise ContractError('resident evaluation lacks complete immutable parent-set pins')
        saved_snapshot_ids: dict[str,str]={}
        for directory, receipt in pins:
            if type(receipt) is not HpoReferenceReceipt or receipt.phase != 'acquire':
                raise ContractError('resident evaluation needs original acquisition receipts')
            directory=Path(directory).resolve(strict=True)
            parent = unique.get(str(directory))
            expected = None if parent is None else {'run_id':parent['run_id'],'token':job.ticket,
                'generation':parent['generation'],'inference_sha256':parent['inference_sha256'],
                'generation_sha256':parent['generation_sha256']}
            if expected is None or canonical(receipt.binding) != canonical(expected):
                raise ContractError('resident evaluation reference differs from actual immutable parent')
            assert parent is not None
            from contextlib import nullcontext

            from gf_training.lineage import generation_path
            paths=[directory/REFERENCE_DIRECTORY/job.ticket/'acquire.receipt.json',directory/'artifact-registry.json',generation_path(directory,parent['generation'])]
            admission=(artifacts.parent_budget.reserve(32*sum(path.lstat().st_size for path in paths)+1024**2+65536)
                if type(artifacts) is OwnedGameArtifacts else nullcontext())
            with admission, _ChainScan(directory=Path(directory),max_owned_bytes=max_owned_bytes,deadline=deadline) as scan:
                if type(artifacts) is OwnedGameArtifacts:
                    from gf_common.hpo import require_owned_record_version
                    from gf_models.structured_sealed import checksum
                    record=scan.object(generation_path(directory,parent['generation']),parent['generation_sha256'])
                    require_owned_record_version(record,'generation')
                    if record['generation']!=parent['generation'] or record['inference_sha256']!=parent['inference_sha256']:
                        raise ContractError('evaluation final source generation differs')
                    saved_snapshot_ids[str(directory)]=checksum(record['snapshot_identity'])
                else:
                    _, _records = validate_hpo_chain(Path(directory),max_owned_bytes=max_owned_bytes,deadline=deadline,scan=scan)
                acquired = scan.object(Path(directory)/REFERENCE_DIRECTORY/job.ticket/'acquire.receipt.json',receipt.receipt_sha256)
                registry_value = scan.object(Path(directory)/'artifact-registry.json')
                if (canonical(acquired.get('binding')) != canonical(expected)
                        or registry_value['evaluation_references'].get(job.ticket) != {
                            'generation':parent['generation'],'inference_sha256':parent['inference_sha256']}):
                    raise ContractError('resident evaluation reference is not durably active')
                scan.finalize()
        self._owned_use_reservation: Any=None
        self._owned_final_set: bytes | None=None
        self._owned_cap_entries: dict[int,Json]={}
        self._owned_closed=False
        if type(artifacts) is OwnedGameArtifacts:
            import weakref

            from gf_common.artifacts import json_bound
            # Admission precedes copies/serialization of the immutable final set.
            self._owned_use_reservation=artifacts.parent_budget.reserve(32*json_bound(parents)+65536)
            weakref.finalize(self,self._owned_use_reservation.close)
            finals=[]
            for cap in artifacts.owned_snapshots:
                cap.require()
                metadata=cap.actor_metadata
                matched=[]
                for parent in parents:
                    if parent['inference_sha256']==cap.checkpoint and parent['generation']==cap.generation:
                        directory=Path(parent['run_path'])
                        if saved_snapshot_ids[str(directory)]==cap.identity:
                            matched.append(parent)
                if len(matched)!=1:
                    raise ContractError('owned evaluation capability lacks exact committed final reference')
                entry={'parent':clone(matched[0]),'snapshot_identity':metadata['snapshot_identity']}
                finals.append(entry); self._owned_cap_entries[id(cap)]=entry
            if not 1<=len(finals)<=2: raise ContractError('owned evaluation requires one/two exact finals')
            self._owned_final_set=canonical(finals).encode()
        for binding in plan.bindings.values():
            if binding.policy_options is not None:
                artifacts.require_options(binding.model_id,binding.policy_options)
        artifacts.finalize(deadline=deadline)
        self.job,self.plan,self.artifacts,self.pins = job,plan,artifacts,pins
        self.original = canonical(_plan_identity(plan))
        self.runtime = runtime
        self._pin_identity = canonical([(str(directory),asdict(receipt)) for directory,receipt in pins])

    def require(self, job: GpuJob, plan: Any) -> None:
        job._require_open()
        if (job is not self.job or plan is not self.plan or canonical(_plan_identity(plan)) != self.original
                or job.ticket != self.runtime['job_ticket']
                or job._admission['context']['evaluation_id'] != self.runtime['evaluation_id']
                or canonical(plan.execution_runtime) != canonical(self.runtime)
                or canonical([(str(directory),asdict(receipt)) for directory,receipt in self.pins]) != self._pin_identity
                or self.artifacts._released or self._owned_closed):
            raise ContractError('resident evaluation actual admission/panel scope changed')

    def owned_use(self, request: Json, cap: Any, binding: Any) -> tuple[Json,Any]:
        self.require(self.job,self.plan)
        if self._owned_final_set is None or id(cap) not in self._owned_cap_entries:
            raise ContractError('selected actor capability is outside actual evaluation final set')
        cap.require()
        entry=self._owned_cap_entries[id(cap)]
        if (entry['parent']['policy_label'] not in self.plan.bindings
                or self.plan.bindings[entry['parent']['policy_label']]!=binding):
            raise ContractError('evaluation capability differs from frozen selected policy')
        import json

        from gf_common.artifacts import json_bound
        reservation=self.artifacts.parent_budget.reserve(32*json_bound(request)+32*len(self._owned_final_set)+65536)
        try:
            value={'schema_version':1,'kind':'owned-evaluation-final-policy-use-v1',
                'evaluation_id':self.plan.linkage['evaluation_id'],'finals':json.loads(self._owned_final_set),
                'selected':clone(entry),'outer_request':clone(request)}
            from gf_common.hpo_actor_snapshot import checked_evaluation_use
            checked_evaluation_use(value,cap.actor_metadata,request,binding.policy_options)
            return value,reservation
        except BaseException:
            reservation.close()
            raise

    def release_owned_metadata(self) -> None:
        if self._owned_closed: return
        if self.job.ledger.workers or self.job.ledger.learner is not None or self.job.ledger.updating:
            raise ContractError('evaluation final metadata still has physical work owners')
        if self._owned_use_reservation is not None: self._owned_use_reservation.close()
        self._owned_final_set=None; self._owned_cap_entries.clear(); self._owned_closed=True

    def __reduce__(self) -> Any:
        raise TypeError('resident evaluation actual scope cannot be persisted')


class ResidentPoolCustody:
    """One actual-job lifetime; physical viewers never change or disappear on PARK."""
    def __init__(self, job: GpuJob, *, run_id: str, pool_id: str, evaluation_scope: EvaluationPoolScope | None = None) -> None:
        if type(job) is not GpuJob or type(job.ledger) is not _AuditedLedger:
            raise ContractError('resident custody requires actual HPO job/audited ledger')
        job._require_open()
        context = job._admission['context']
        if context['kind'] != 'training':
            if type(evaluation_scope) is not EvaluationPoolScope:
                raise ContractError('resident evaluation panel scope is not activated')
            evaluation_scope.require(job,evaluation_scope.plan)
            if run_id != evaluation_scope.runtime['evaluation_id']:
                raise ContractError('resident evaluation namespace differs from actual panel')
        elif evaluation_scope is not None:
            raise ContractError('training cannot borrow evaluation backing')
        if context['kind'] == 'training' and context['run_id'] != run_id:
            raise ContractError('resident custody cannot cross the actual training job parent')
        if type(run_id) is not str or not run_id or len(run_id) > 128:
            raise ContractError('resident bounded run identity required')
        cache_owner({'job_ticket':job.ticket,'pool_id':pool_id,'pool_epoch':1,
            'worker_key':1,'pid':1,'viewer':1,'lease_epoch':1})
        if getattr(job.ledger, '_resident_pool', None) is not None:
            raise ContractError('actual HPO job already has resident custody')
        self.job, self.run_id, self.pool_id = job, run_id, pool_id
        self.evaluation_scope = evaluation_scope
        self.ledger = cast(_AuditedLedger, job.ledger)
        self.pool_epoch = 1
        self._epoch = 0
        self._failed = False
        self._owners: dict[int, _Owner] = {}
        self._authority: ParkedAuthority | None = None
        self._admission: CollectionAdmission | None = None
        self._pins: dict[str, Json] = {}
        self._acquisitions: dict[str,CacheAcquisition] = {}
        self._cached: dict[int, set[str]] = {}
        self._releases: dict[str, CacheReleaseAuthority] = {}
        job.ledger._resident_pool = self

    def _require(self, *, cleanup: bool = False) -> None:
        self.job._require_open()
        if self._failed and not cleanup:
            raise ContractError('resident work custody failed; only original audited reap cleanup remains')
        if self.ledger._resident_pool is not self:
            raise ContractError('resident custody is not SAME live job owner')
        if not cleanup and self.evaluation_scope is not None:
            self.evaluation_scope.require(self.job,self.evaluation_scope.plan)

    def fail(self) -> None:
        """Sticky timeout/cancel/transport failure; preserve original cleanup ownership."""
        self._require(cleanup=True)
        self._failed = True
        self._invalidate()

    def _invalidate(self) -> None:
        self._epoch += 1
        self._authority = None

    def _row(self, worker: Any) -> _Owner:
        self._require()
        key = getattr(worker, 'key', None)
        if type(key) is not int:
            raise ContractError('foreign resident physical reservation')
        row = self._owners.get(key)
        if row is None or row.worker is not worker or _job(worker) is not self.job:
            raise ContractError('foreign resident physical reservation')
        return row

    def _live(self, row: _Owner) -> None:
        worker = row.worker
        if worker.released or self.job.ledger.workers.get(worker.key) is not worker or worker.creation_pending:
            raise ContractError('resident physical reservation is not live')
        worker._settled(never_started=False)
        _actual_process(worker._process, row.owner)
        if worker._owner is not row.owner or worker._process.poll() is not None:
            raise ContractError('resident original physical owner/process changed or exited')

    def register(self, worker: Any, owner: Any, *, viewer: int) -> None:
        self._require()
        integer(viewer, 1, 2)
        if (type(worker) is not _Worker or _job(worker) is not self.job
                or worker.key in self._owners or worker._owner is not owner
                or self.job.ledger.updating or self.ledger._resident_transition):
            raise ContractError('resident registration requires unbound original physical owner')
        if sum(row.viewer == viewer and not row.worker.released for row in self._owners.values()) >= self.ledger.max_workers // 2:
            raise ContractError('resident physical viewer partition is full')
        row = _Owner(worker, owner, viewer)
        self._live(row)
        self._owners[worker.key] = row
        self._cached[worker.key] = set()
        self._invalidate()

    def admit_collection(self, plan: Any, ledger: Any) -> CollectionAdmission:
        from gf_training.collection import CollectionPlan
        self._require()
        evaluation = self.evaluation_scope is not None
        if self.evaluation_scope is not None:
            self.evaluation_scope.require(self.job,plan)
        if ((not evaluation and type(plan) is not CollectionPlan) or ledger is not self.ledger or ledger.updating
                or self.ledger._resident_transition
                or self._admission is not None and self._admission.phase != 'retired'
                or not evaluation and plan.linkage.get('parent_run_id') != self.run_id):
            raise ContractError('resident admission requires SAME original collection/job/ledger')
        if not evaluation:
            integer(plan.linkage.get('collection_generation'), 0, 2**63-1)
        if any(row.state == 'active' or row.pending is not None or row.io or row.setups or row.worker.decisions
                for row in self._owners.values()):
            raise ContractError('resident collection admission overlaps original work')
        from gf_models.resident_cache import resident_registry_execution
        resident_registry_execution(plan.bindings)
        self._admission = CollectionAdmission(_ISSUER, self, plan)
        return self._admission

    def bootstrap_request(self, worker: Any) -> Json:
        row = self._row(worker)
        self._live(row)
        if row.bootstrap is not None or row.pending is not None or row.io or row.setups or row.lease_epoch:
            raise ContractError('resident bootstrap requires new original owner')
        request = {'schema_version': 1, 'protocol': RESIDENT_MODE, 'command': 'READY',
            'run_id': self.run_id, **self._identity(row)}
        row.pending = request
        return clone(request)

    def observe_bootstrap_ready(self, worker: Any, response: Json) -> None:
        row = self._row(worker)
        self._live(row)
        try:
            expected = {**(row.pending or {}), 'kind': 'READY', 'clean_reset': True,
                'private_state_empty': True, 'pending_io_empty': True, 'cuda_synchronized': True,
                'reset_version': 'resident-reset-v1'}
            if (row.bootstrap is not None or row.io or row.setups or row.worker.decisions
                    or not row.pending or row.pending.get('command') != 'READY'
                    or canonical(response) != canonical(expected)):
                raise ContractError('resident clean bootstrap ready identity/state differs')
            row.bootstrap, row.pending = clone(response), None
        except BaseException:
            self.fail()
            raise

    def begin_admitted_lease(self, admission: CollectionAdmission, worker: Any, *, binding: Any) -> Json:
        admission.require(admission.plan, self.ledger)
        row = self._row(worker)
        self._live(row)
        if (row.bootstrap is None or row.state not in ('idle', 'parked')
                or row.lease_epoch and row.park_ack is None or row.pending is not None
                or row.io or row.setups or worker.decisions):
            raise ContractError('resident owner lacks original clean ready/PARK state')
        if canonical(asdict(binding)) not in [canonical(asdict(value)) for value in admission.plan.bindings.values()]:
            raise ContractError('resident selected binding differs from original collection')
        result = self.begin_lease(worker, generation=admission.generation)
        row.collection = admission
        row.episode_started = False
        row.binding_identity = digest(asdict(binding))
        options = binding.policy_options
        row.checkpoint = options['expected_manifest_digest'] if options is not None and 'expected_manifest_digest' in options else digest(asdict(binding))
        admission.phase = 'collecting'
        admission.used.add(worker.key)
        return result

    def owned_evaluation_use(self, worker: Any, request: Json, cap: Any, binding: Any) -> tuple[Json,Any]:
        self._require()
        row=self._row(worker)
        self._live(row)
        if (self.evaluation_scope is None or row.pending is None or canonical(row.pending)!=canonical(request)
                or row.binding_identity!=digest(asdict(binding)) or row.state!='active'
                or row.collection is not self._admission or row.generation!=0):
            raise ContractError('evaluation use lacks SAME actual outer owner/epoch/selected binding')
        return self.evaluation_scope.owned_use(request,cap,binding)

    def episode_request(self, worker: Any, command: str) -> Json:
        row = self._row(worker)
        self._live(row)
        if row.collection is None:
            raise ContractError('resident episode lacks original collection admission')
        row.collection.require(row.collection.plan, self.ledger)
        if (command not in ('BEGIN', 'SELECT') or row.state != 'active' or row.pending is not None
                or row.io or row.setups or self.ledger.updating or command == 'BEGIN' and worker.decisions):
            raise ContractError('resident episode request overlaps original work')
        row.request_epoch += 1
        request = {**self._identity(row), 'schema_version': 1, 'protocol': RESIDENT_MODE,
            'run_id': self.run_id, 'generation': row.generation, 'command': command,
            'request_epoch': row.request_epoch, 'checkpoint_sha256': row.checkpoint}
        row.pending = request
        self._invalidate()
        return clone(request)

    def acknowledge_episode(self, worker: Any, response: Json) -> Json:
        row = self._row(worker)
        self._live(row)
        try:
            from gf_common.domain import closed
            closed(response, {'request', 'payload'})
            if (row.pending is None or row.pending.get('command') not in ('BEGIN', 'SELECT')
                    or row.io or row.setups or canonical(response['request']) != canonical(row.pending)
                    or type(response['payload']) is not dict):
                raise ContractError('resident episode original response identity differs')
            if row.pending['command'] == 'BEGIN':
                row.episode_started = True
                if row.binding_identity is not None and row.checkpoint == row.binding_identity:
                    self._cached[worker.key].add(row.binding_identity)
            row.pending = None
            return response['payload']
        except BaseException:
            self.fail()
            raise

    def retire_collection(self, admission: CollectionAdmission) -> ParkedAuthority:
        admission.require(admission.plan, self.ledger)
        self._check_parked(admission.generation)
        if set(self.ledger.workers) != admission.used:
            raise ContractError('resident collection retains unused physical owner')
        admission.phase = 'retired'
        return self.parked_authority(generation=admission.generation, collection_id=admission.collection_id)

    def begin_lease(self, worker: Any, *, generation: int) -> Json:
        row = self._row(worker)
        self._live(row)
        integer(generation, 0, 2**63-1)
        if (row.state not in ('idle','parked') or row.pending is not None or row.io or row.setups or worker.decisions
                or self.job.ledger.updating or self.ledger._resident_transition):
            raise ContractError('resident BEGIN overlaps work or charged transition')
        row.lease_epoch += 1
        row.generation, row.state = generation, 'active'
        row.end_ack = row.park_ack = row.sync_ack = None
        self._invalidate()
        return self._identity(row)

    def _identity(self, row: _Owner) -> Json:
        return {'job_ticket':self.job.ticket, 'pool_id':self.pool_id, 'pool_epoch':self.pool_epoch,
            'worker_key':row.worker.key,'pid':row.worker.pid,'viewer':row.viewer,'lease_epoch':row.lease_epoch}

    def require_decision(self, worker: Any) -> None:
        row = self._row(worker)
        self._live(row)
        if (row.state != 'active' or row.pending is not None or row.io or row.setups or worker.decisions
                or row.collection is not None and not row.episode_started or self.job.ledger.updating):
            raise ContractError('resident decision lacks exclusive active logical lease')

    def request(self, worker: Any, command: str, *, checkpoint_sha256: str, cache_identity: str | None = None, generation: int | None = None) -> Json:
        row = self._row(worker)
        self._live(row)
        allowed = ('END','EVICT') if row.state == 'active' else ('PARK','SYNC','EVICT') if row.state in ('idle','parked') else ()
        if (command not in allowed or row.pending is not None or row.io or row.setups or worker.decisions or not row.lease_epoch
                or self.job.ledger.updating or command == 'EVICT' and row.state == 'active' and row.episode_started):
            raise ContractError('resident command overlaps request/decision or invalid lifecycle')
        if generation is not None:
            integer(generation, 0, 2**63-1)
            if command != 'SYNC' or generation != row.generation + 1:
                raise ContractError('resident SYNC target must be exactly next generation')
        row.request_epoch += 1
        request = {**self._identity(row), 'schema_version':1, 'protocol':RESIDENT_MODE,
            'run_id':self.run_id,'generation':row.generation if generation is None else generation,'command':command,
            'request_epoch':row.request_epoch,'checkpoint_sha256':checkpoint_sha256,'cache_identity':cache_identity}
        checked_ack({**request, 'clean_reset':True,'private_state_empty':True,'pending_io_empty':True,
            'cuda_synchronized':True,'reset_version':'resident-reset-v1'})
        row.pending = request
        row.park_ack = None
        self._invalidate()
        return clone(request)

    def acknowledge(self, worker: Any, acknowledgement: Json) -> None:
        row = self._row(worker)
        self._live(row)
        try:
            raw = checked_ack(acknowledgement)
            if row.pending is None or canonical({key:raw[key] for key in REQUEST_KEYS}) != canonical(row.pending):
                raise ContractError('resident original ACK identity/epoch/request differs')
            if raw['command'] == 'END':
                row.end_ack, row.state = clone(raw), 'idle'
            elif raw['command'] == 'PARK':
                previous = row.sync_ack if row.sync_ack is not None else row.end_ack
                if previous is None or raw['checkpoint_sha256'] != previous['checkpoint_sha256']:
                    raise ContractError('resident PARK lacks same-policy END/reset')
                row.park_ack, row.state = clone(raw), 'parked'
            else:
                if raw['command'] == 'SYNC':
                    row.sync_ack = clone(raw)
                    row.generation = raw['generation']
                    row.checkpoint = raw['checkpoint_sha256']
                if raw['command'] == 'EVICT':
                    identity = raw['cache_identity']
                    if identity not in self._cached[worker.key]:
                        raise ContractError('resident eviction lacks original pinned cache identity')
                    self._cached[worker.key].remove(identity)
                    for token,binding in self._pins.items():
                        if binding['worker_key'] == worker.key and binding['cache_identity'] == identity:
                            self._releases[token] = CacheReleaseAuthority(_ISSUER,self,binding,worker,reaped=False, evidence={
                                'schema_version':1,'kind':'hpo-resident-cache-eviction','binding_sha256':digest(binding),
                                'ack':clone(raw),'ack_sha256':digest(raw)})
                if raw['command'] != 'EVICT' or row.state != 'active':
                    row.state = 'idle'
            row.pending = None
        except BaseException:
            self._failed = True
            self._invalidate()
            raise

    def close_request(self,worker: Any) -> Json:
        row = self._row(worker)
        self._live(row)
        if row.state not in ('idle','parked') or row.pending is not None or row.io or row.setups or worker.decisions:
            raise ContractError('resident CLOSE requires actual idle transport')
        row.request_epoch += 1
        request = {**self._identity(row),'schema_version':1,'protocol':RESIDENT_MODE,'run_id':self.run_id,
            'generation':row.generation,'command':'CLOSE','request_epoch':row.request_epoch}
        row.pending = request
        self._invalidate()
        return clone(request)

    def acknowledge_close(self,worker: Any,response: Json) -> None:
        row = self._row(worker)
        # A valid CLOSE response may precede observed process death. Original
        # process/owner identity and BOTH I/O completions remain mandatory.
        if (row.pending is None or row.pending['command'] != 'CLOSE' or row.io or row.setups
                or canonical(response) != canonical({'request':row.pending,'kind':'CLOSED'})):
            self.fail()
            raise ContractError('resident original CLOSE response differs')
        row.pending,row.state = None,'closing'

    def track_setup(self,worker: Any,setup: Any) -> None:
        from gf_simulator_client.resident_pool import _ExchangeSetup
        row = self._row(worker)
        self._live(row)
        if (type(setup) is not _ExchangeSetup or setup.owner is not row.owner
                or setup.executor is not row.owner.executor or setup not in row.owner._setups
                or setup.future is not None or setup.submission_started or row.io or row.setups):
            raise ContractError('resident original gated setup required before submission')
        row.setups.add(setup)
        self._invalidate()

    def authorize_setup(self,worker: Any,setup: Any) -> None:
        row = self._row(worker)
        self._live(row)
        if (setup not in row.setups or setup.owner is not row.owner or setup.task is None
                or setup.future is None or row.io.get(setup.task) is not setup.future
                or canonical(setup.payload.get('request')) != canonical(row.pending)
                or setup.payload.get('request',{}).get('command') != 'CLOSE'
                    and (row.owner.closed or row.owner.pool.closed)):
            raise ContractError('resident framing requires SAME original setup/Task/Future/request custody')
        setup.authorize()

    def retire_setup(self,worker: Any,setup: Any) -> None:
        from gf_simulator_client.resident_pool import _ExchangeSetup
        self._require(cleanup=True)
        row = self._owners.get(getattr(worker,'key',-1))
        if (row is None or row.worker is not worker or type(setup) is not _ExchangeSetup
                or setup.owner is not row.owner or setup.executor is not row.owner.executor
                or setup not in row.owner._setups):
            raise ContractError('resident SAME original setup cleanup owner required')
        setup.require_retired()
        if setup.task in row.io:
            if row.io[setup.task] is not setup.future:
                raise ContractError('resident original setup Future identity changed')
            self.retire_io(worker,setup.task)
        elif setup.authorized:
            raise ContractError('resident framed setup lost its original registered Task/Future')
        row.setups.discard(setup)

    def track_io(self, worker: Any, task: asyncio.Task[Any], completion: Future[Any]) -> None:
        row = self._row(worker)
        self._live(row)
        if (not isinstance(task, asyncio.Task) or task.done() or task in row.io
                or not isinstance(completion, Future)):
            raise ContractError('original active resident I/O Task required')
        row.io[task] = completion
        self._invalidate()

    def retire_io(self, worker: Any, task: asyncio.Task[Any]) -> None:
        self._require(cleanup=True)
        row = self._owners.get(getattr(worker,'key',-1))
        if row is None or row.worker is not worker or task not in row.io or not task.done() or not row.io[task].done():
            raise ContractError('SAME completed resident I/O Task required for retirement')
        del row.io[task]

    def require_resource_retirement(self, job: GpuJob, ledger: _AuditedLedger) -> None:
        """Cleanup proof for actual termination, independent of failed work state."""
        self.job._require_open()
        if (job is not self.job or ledger is not self.ledger or ledger is not job.ledger
                or ledger._resident_pool is not self):
            raise ContractError('resident resource retirement requires SAME actual job/custody/ledger')
        for row in self._owners.values():
            if (row.io or row.setups or getattr(row.owner,'_operations',None) or getattr(row.owner,'_setups',None)):
                raise ContractError('resident original transport I/O/setup has not been explicitly retired')
            self._check_reaped(row.worker)
            _actual_process(row.worker._process,row.owner)
            if row.worker._owner is not row.owner:
                raise ContractError('resident cleanup changed original physical owner')
        # Only original reap plus explicit I/O retirement proves that outstanding
        # commands can no longer deliver responses. Never retire them from EOF,
        # cancellation, worker-map absence or a persisted PID alone.
        for row in self._owners.values():
            row.pending, row.state = None, 'reaped'

    def require_confirmed_resource_retirement(self,job: GpuJob,ledger: _AuditedLedger,receipt: Any) -> None:
        """Verify the existing exact global proof; grants no open/work authority."""
        if job is not self.job or ledger is not self.ledger or ledger is not job.ledger or ledger._resident_pool is not self:
            raise ContractError('confirmed retirement requires SAME original job/custody/ledger')
        job.require_confirmed_termination(receipt)
        for row in self._owners.values():
            worker=row.worker
            if (row.io or row.setups or getattr(row.owner,'_operations',None) or getattr(row.owner,'_setups',None)
                    or not worker.released or worker.key in ledger.workers or worker._proof is None
                    or not any(worker._proof is resource for resource in receipt.resources)
                    or worker._proof.outcome!='reaped' or worker._process is None or worker._process.poll() is None
                    or worker._proof.pid!=worker._process.pid or worker._owner is not row.owner
                    or row.pending is not None or row.state!='reaped'):
                raise ContractError('confirmed original resident resource retirement changed')
            _actual_process(worker._process,row.owner)

    def _check_parked(self, generation: int) -> None:
        live = {key: row for key, row in self._owners.items() if not row.worker.released}
        if not live or set(live) != set(self.job.ledger.workers):
            raise ContractError('resident PARK authority must cover every actual physical owner')
        for row in live.values():
            self._live(row)
            if (row.state != 'parked' or row.pending is not None or row.io or row.setups or row.worker.decisions
                    or row.generation != generation or row.end_ack is None or row.park_ack is None):
                raise ContractError('resident owner has unretired lease/I/O/decision/ACK')

    def _evidence(self, generation: int, collection_id: str) -> Json:
        return checked_retirement({'schema_version':1,'kind':'hpo-resident-logical-retirement',
            'run_id':self.run_id,'collection_id':collection_id,'job_ticket':self.job.ticket,
            'pool_id':self.pool_id,'pool_epoch':self.pool_epoch,'generation':generation,
            'owners':[{**self._identity(row),'end_ack':row.end_ack,'end_ack_sha256':digest(row.end_ack),
                'park_ack':row.park_ack,'park_ack_sha256':digest(row.park_ack)} for _,row in sorted(self._owners.items()) if not row.worker.released]}, max_workers=self.ledger.max_workers)

    def parked_authority(self, *, generation: int, collection_id: str) -> ParkedAuthority:
        self._require()
        self._check_parked(generation)
        self._authority = ParkedAuthority(_ISSUER,self,self._epoch,self._evidence(generation,collection_id))
        return self._authority

    def observe_evaluation_cache(self,worker: Any,plan: Any,binding: Any,identity: str) -> None:
        from gf_common.hpo_resident import _sha
        self._require()
        if self.evaluation_scope is None:
            raise ContractError('training cache requires Registry5 pins')
        self.evaluation_scope.require(self.job,plan)
        row = self._row(worker)
        self._live(row)
        if canonical(asdict(binding)) not in [canonical(asdict(value)) for value in plan.bindings.values()]:
            raise ContractError('evaluation cache differs from complete immutable panel policy set')
        _sha(identity)
        self._cached[worker.key].add(identity)

    def begin_cache_acquisition(self,worker: Any,token: str,generation: int,inference_sha256: str) -> CacheAcquisition:
        from gf_common.hpo_resident import _sha
        self._require()
        integer(generation,0,2**63-1)
        _sha(inference_sha256)
        if (type(token) is not str or len(token)!=32 or any(char not in '0123456789abcdef' for char in token)
                or token in self._acquisitions):
            raise ContractError('unique original cache acquisition token required')
        attempt=CacheAcquisition(_ISSUER,self,worker,token,generation,inference_sha256)
        self._acquisitions[token]=attempt
        return attempt

    def require_acquisition_reap(self,attempt: CacheAcquisition) -> None:
        self._require(cleanup=True)
        if (type(attempt) is not CacheAcquisition or attempt.pool is not self
                or self._acquisitions.get(attempt.token) is not attempt
                or self._owners[attempt.worker.key].owner is not attempt.owner):
            raise ContractError('original cache acquisition cleanup owner required')
        self._check_reaped(attempt.worker)

    def cache_release_acquisition_after_reap(self,attempt: CacheAcquisition,receipt: Any) -> CacheReleaseAuthority:
        self.require_acquisition_reap(attempt)
        attempt.require_receipt(receipt)
        if attempt.receipt is not receipt:
            raise ContractError('cache cleanup requires original retained acquisition receipt')
        # A failed observation cannot erase the original transaction. This
        # cleanup-only adoption is backed by its pre-write live attempt, genuine
        # idempotent receipt and original audited reap, never persisted JSON/PID.
        self._pins[attempt.token]=clone(receipt.binding)
        return self.cache_release_after_reap(receipt.binding)

    def observe_cache_pin(self, receipt: Any) -> None:
        from gf_training.hpo_retention import HpoReferenceReceipt
        self._require()
        if type(receipt) is not HpoReferenceReceipt or receipt.phase != 'acquire':
            raise ContractError('resident cache read requires original durable acquisition receipt')
        binding = receipt.binding
        key = binding.get('worker_key')
        if type(key) is not int:
            raise ContractError('resident cache pin has invalid physical owner')
        row = self._owners.get(key)
        if row is None or binding.get('run_id') != self.run_id:
            raise ContractError('resident cache pin has foreign run/physical owner')
        self._live(row)
        if canonical({key:binding[key] for key in CACHE_OWNER_KEYS}) != canonical(self._identity(row)):
            raise ContractError('resident cache pin has stale original owner/lease epoch')
        previous = self._pins.get(binding['token'])
        if previous is not None and canonical(previous) != canonical(binding):
            raise ContractError('resident cache pin token changed binding')
        self._pins[binding['token']] = clone(binding)
        self._cached[row.worker.key].add(binding['cache_identity'])
        self._releases.pop(binding['token'], None)

    def _check_reaped(self, worker: _Worker) -> None:
        if (_job(worker) is not self.job or not worker.released or worker.key in self.job.ledger.workers
                or worker._proof is None or worker._proof.outcome != 'reaped'
                or worker._process is None or worker._process.poll() is None
                or worker._proof.pid != worker._process.pid):
            raise ContractError('resident cache release requires actual original audited process reap')
        row = self._owners.get(worker.key)
        if row is not None and (row.io or row.setups or getattr(row.owner,'_operations',None) or getattr(row.owner,'_setups',None)):
            raise ContractError('resident original reap retains unretired transport setup/I/O')

    def cache_release_after_reap(self, binding: Json) -> CacheReleaseAuthority:
        self._require(cleanup=True)
        if canonical(self._pins.get(binding['token'])) != canonical(binding):
            raise ContractError('resident cache release has no matching original pin')
        worker = self._owners[binding['worker_key']].worker
        self._check_reaped(worker)
        assert worker._proof is not None
        authority = CacheReleaseAuthority(_ISSUER,self,binding,worker,reaped=True,evidence={
            'schema_version':1,'kind':'hpo-resident-cache-reaped','binding_sha256':digest(binding),
            'resource':asdict(worker._proof)})
        self._releases[binding['token']] = authority
        return authority

    def cache_release(self, binding: Json) -> CacheReleaseAuthority:
        self._require()
        authority = self._releases.get(binding['token'])
        if authority is None:
            raise ContractError('resident cache pin lacks original eviction ACK')
        authority.require(binding)
        return authority


class ResidentTransition:
    """One staging charge released by exact SYNC/PARK or original reaped cleanup."""
    def __init__(self, ledger: _AuditedLedger, temporary_bytes: int, authority: ParkedAuthority, *,
                 target_generation: int, target_checkpoint_sha256: str, affected_workers: tuple[_Worker,...]) -> None:
        ledger._parked(authority)
        integer(temporary_bytes,0,ledger.global_cuda_bytes)
        integer(target_generation,authority.evidence['generation']+1,authority.evidence['generation']+1)
        # Pure ACK validation includes the target digest without allocating anything.
        from gf_common.hpo_resident import _sha
        _sha(target_checkpoint_sha256)
        self.ledger, self.pool, self.authority = ledger, authority._custody, authority
        self.bytes, self.generation, self.checkpoint = temporary_bytes,target_generation,target_checkpoint_sha256
        if (type(affected_workers) is not tuple or not affected_workers
                or len({id(worker) for worker in affected_workers}) != len(affected_workers)):
            raise ContractError('resident transition requires exact affected physical-owner subset')
        self.affected = frozenset(self.pool._row(worker).worker.key for worker in affected_workers)
        self.owners = tuple((row,row.worker._process,row.request_epoch,clone(row.park_ack)) for row in self.pool._owners.values() if not row.worker.released)
        self.active = self.failed = self.finished = False

    def __enter__(self) -> Self:
        self.authority.require(self.ledger)
        if (self.active or self.finished or self.ledger.updating or self.ledger._resident_transition
                or self.ledger.cuda_bytes+self.bytes>self.ledger.global_cuda_bytes):
            raise ContractError('resident transition overlaps or exceeds shared CUDA budget')
        self.ledger._resident_transition = self
        self.ledger._resident_temporary = self.bytes
        self.ledger.peak_cuda_bytes = max(self.ledger.peak_cuda_bytes,self.ledger.cuda_bytes)
        self.active = True
        return self

    def _release(self) -> None:
        if self.ledger._resident_transition is not self or not self.active:
            raise ContractError('foreign or already retired resident transition')
        self.ledger._resident_temporary = 0
        self.ledger._resident_transition = None
        self.active = False
        self.finished = True

    def _synced(self) -> None:
        self.pool._require()
        if self.ledger.updating or {key for key,row in self.pool._owners.items() if not row.worker.released} != set(self.ledger.workers):
            raise ContractError('resident SYNC changed physical owner set or overlaps update')
        for row,process,epoch,park_ack in self.owners:
            self.pool._live(row)
            if (self.pool._owners.get(row.worker.key) is not row or row.worker._process is not process
                    or row.pending is not None or row.io or row.setups or row.worker.decisions or row.state != 'parked'):
                raise ContractError('resident transition has active original physical owner/I/O')
            if row.worker.key not in self.affected:
                if row.request_epoch != epoch or canonical(row.park_ack) != canonical(park_ack):
                    raise ContractError('unaffected resident owner changed its original PARK/binding')
                continue
            if (row.sync_ack is None or row.park_ack is None
                    or row.sync_ack['request_epoch'] <= epoch
                    or row.park_ack['request_epoch'] <= row.sync_ack['request_epoch']):
                raise ContractError('resident staging release lacks actual completed original SYNC/PARK')
            for ack in (row.sync_ack,row.park_ack):
                checked_ack(ack)
                if ack['generation'] != self.generation or ack['checkpoint_sha256'] != self.checkpoint:
                    raise ContractError('resident staging release target digest/generation differs')

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        try:
            if kind is not None:
                raise ContractError('resident transition interrupted before confirmed copy retirement')
            self._synced()
        except BaseException:
            self.failed = self.pool._failed = True
            self.pool._invalidate()
            if kind is None:
                raise
        else:
            self._release()

    def retire_failed_transition(self) -> None:
        self.pool._require(cleanup=True)
        if (not self.failed or not self.active or self.ledger.updating
                or self.ledger._resident_transition is not self):
            raise ContractError('original failed resident transition required')
        for row,process,_epoch,_park_ack in self.owners:
            if row.worker._process is not process or row.io or row.setups or row.worker.decisions:
                raise ContractError('resident failed transition retains original I/O/process/decision')
            self.pool._check_reaped(row.worker)
        # Reap plus retired original I/O establishes that no ACK can still arrive.
        for row,_process,_epoch,_park_ack in self.owners:
            row.pending, row.state = None, 'reaped'
        self._release()
