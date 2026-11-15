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


class ResidentPoolCustody:
    """One actual-job lifetime; physical viewers never change or disappear on PARK."""
    def __init__(self, job: GpuJob, *, run_id: str, pool_id: str) -> None:
        if type(job) is not GpuJob or type(job.ledger) is not _AuditedLedger:
            raise ContractError('resident custody requires actual HPO job/audited ledger')
        job._require_open()
        context = job._admission['context']
        if context['kind'] != 'training':
            raise ContractError('resident evaluation panel scope is not activated')
        if context['kind'] == 'training' and context['run_id'] != run_id:
            raise ContractError('resident custody cannot cross the actual training job parent')
        if type(run_id) is not str or not run_id or len(run_id) > 128:
            raise ContractError('resident bounded run identity required')
        cache_owner({'job_ticket':job.ticket,'pool_id':pool_id,'pool_epoch':1,
            'worker_key':1,'pid':1,'viewer':1,'lease_epoch':1})
        if getattr(job.ledger, '_resident_pool', None) is not None:
            raise ContractError('actual HPO job already has resident custody')
        self.job, self.run_id, self.pool_id = job, run_id, pool_id
        self.ledger = cast(_AuditedLedger, job.ledger)
        self.pool_epoch = 1
        self._epoch = 0
        self._failed = False
        self._owners: dict[int, _Owner] = {}
        self._authority: ParkedAuthority | None = None
        self._pins: dict[str, Json] = {}
        self._cached: dict[int, set[str]] = {}
        self._releases: dict[str, CacheReleaseAuthority] = {}
        job.ledger._resident_pool = self

    def _require(self, *, cleanup: bool = False) -> None:
        self.job._require_open()
        if self._failed and not cleanup:
            raise ContractError('resident work custody failed; only original audited reap cleanup remains')
        if self.ledger._resident_pool is not self:
            raise ContractError('resident custody is not SAME live job owner')

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
        if sum(row.viewer == viewer for row in self._owners.values()) >= 2:
            raise ContractError('resident physical viewer partition is full')
        row = _Owner(worker, owner, viewer)
        self._live(row)
        self._owners[worker.key] = row
        self._cached[worker.key] = set()
        self._invalidate()

    def begin_lease(self, worker: Any, *, generation: int) -> Json:
        row = self._row(worker)
        self._live(row)
        integer(generation, 0, 2**63-1)
        if (row.state not in ('idle','parked') or row.pending is not None or row.io or worker.decisions
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
        if row.state != 'active' or row.pending is not None or row.io or self.job.ledger.updating:
            raise ContractError('resident decision lacks exclusive active logical lease')

    def request(self, worker: Any, command: str, *, checkpoint_sha256: str, cache_identity: str | None = None, generation: int | None = None) -> Json:
        row = self._row(worker)
        self._live(row)
        allowed = ('END',) if row.state == 'active' else ('PARK','SYNC','EVICT') if row.state in ('idle','parked') else ()
        if (command not in allowed or row.pending is not None or row.io or worker.decisions or not row.lease_epoch
                or self.job.ledger.updating):
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
                row.state = 'idle'
            row.pending = None
        except BaseException:
            self._failed = True
            self._invalidate()
            raise

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
            if row.io:
                raise ContractError('resident original transport I/O has not been explicitly retired')
            self._check_reaped(row.worker)
            _actual_process(row.worker._process,row.owner)
            if row.worker._owner is not row.owner:
                raise ContractError('resident cleanup changed original physical owner')
        # Only original reap plus explicit I/O retirement proves that outstanding
        # commands can no longer deliver responses. Never retire them from EOF,
        # cancellation, worker-map absence or a persisted PID alone.
        for row in self._owners.values():
            row.pending, row.state = None, 'reaped'

    def _check_parked(self, generation: int) -> None:
        if not self._owners or set(self._owners) != set(self.job.ledger.workers):
            raise ContractError('resident PARK authority must cover every actual physical owner')
        for row in self._owners.values():
            self._live(row)
            if (row.state != 'parked' or row.pending is not None or row.io or row.worker.decisions
                    or row.generation != generation or row.end_ack is None or row.park_ack is None):
                raise ContractError('resident owner has unretired lease/I/O/decision/ACK')

    def _evidence(self, generation: int, collection_id: str) -> Json:
        return checked_retirement({'schema_version':1,'kind':'hpo-resident-logical-retirement',
            'run_id':self.run_id,'collection_id':collection_id,'job_ticket':self.job.ticket,
            'pool_id':self.pool_id,'pool_epoch':self.pool_epoch,'generation':generation,
            'owners':[{**self._identity(row),'end_ack':row.end_ack,'end_ack_sha256':digest(row.end_ack),
                'park_ack':row.park_ack,'park_ack_sha256':digest(row.park_ack)} for _,row in sorted(self._owners.items())]})

    def parked_authority(self, *, generation: int, collection_id: str) -> ParkedAuthority:
        self._require()
        self._check_parked(generation)
        self._authority = ParkedAuthority(_ISSUER,self,self._epoch,self._evidence(generation,collection_id))
        return self._authority

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
        self.owners = tuple((row,row.worker._process,row.request_epoch,clone(row.park_ack)) for row in self.pool._owners.values())
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
        if self.ledger.updating or set(self.pool._owners) != set(self.ledger.workers):
            raise ContractError('resident SYNC changed physical owner set or overlaps update')
        for row,process,epoch,park_ack in self.owners:
            self.pool._live(row)
            if (self.pool._owners.get(row.worker.key) is not row or row.worker._process is not process
                    or row.pending is not None or row.io or row.worker.decisions or row.state != 'parked'):
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
            if row.worker._process is not process or row.io or row.worker.decisions:
                raise ContractError('resident failed transition retains original I/O/process/decision')
            self.pool._check_reaped(row.worker)
        # Reap plus retired original I/O establishes that no ACK can still arrive.
        for row,_process,_epoch,_park_ack in self.owners:
            row.pending, row.state = None, 'reaped'
        self._release()
