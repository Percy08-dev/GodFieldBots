"""Bounded process-only observations; no model, Source, I/O or lifecycle authority.

The named exclusive lease is a subset of an EXISTING reserved workspace/management
allowance, never extra headroom. On sampler/publication uncertainty the process
retains the owner and lease until exit. Linux proc/file calls retain OS blocking
limitations; sampler join alone has a250ms bound. No input objects are retained.
"""
from __future__ import annotations

import json
import math
import os
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, canonical, closed, integer

LEASE_BYTES=16*1024**2
SIDECAR_BYTES=64*1024
CADENCE_SECONDS=.05
JOIN_SECONDS=.25
BINDING_KEYS={'role','model_id','variant','resume_identity','spec_sha256','source_digest',
    'pid','generation','transaction','run_id','context_id'}
WORKER_PHASES=('receive_validation','update_call','input_retirement','snapshot_export','snapshot_save','snapshot_transfer')
COORDINATOR_PHASES=('coordinator_call',)


class _ObservationPool:
    def __init__(self) -> None:
        self.lock=threading.Lock()
        self.owner: ProcessObservation | None=None

    def acquire(self,owner: ProcessObservation,allowance: int) -> None:
        integer(allowance,1,2**63-1)
        with self.lock:
            if allowance<LEASE_BYTES:
                raise ContractError('existing observation allowance below16MiB')
            if self.owner is not None:
                raise ContractError('exclusive process observation lease already held')
            self.owner=owner

    def release(self,owner: ProcessObservation) -> None:
        with self.lock:
            if self.owner is not owner:
                raise ContractError('observation lease owner mismatch')
            self.owner=None


_POOL=_ObservationPool()


def _read_proc() -> tuple[int,int]:
    fd=os.open('/proc/self/smaps_rollup',os.O_RDONLY|os.O_NONBLOCK)
    try:
        raw=os.read(fd,8192)
    finally:
        os.close(fd)
    if len(raw)>=8192:
        raise ContractError('fixed proc sample exceeds8KiB')
    values={}
    for line in raw.decode('ascii').splitlines():
        name,_,value=line.partition(':')
        if name in ('Pss','Rss'):
            parts=value.split()
            if len(parts)!=2 or parts[1]!='kB':
                raise ContractError('proc memory sample unit mismatch')
            values[name]=integer(int(parts[0]),1,2**63-1)*1024
    if set(values)!={'Pss','Rss'}:
        raise ContractError('proc memory sample missing PSS/RSS')
    return values['Pss'],values['Rss']


def _hash(value: Any) -> bool:
    return type(value) is str and len(value)==64 and not set(value)-set('0123456789abcdef')


def _bindings(value: Any) -> dict[str,Any]:
    closed(value,BINDING_KEYS)
    if any(type(item) not in (str,int,type(None)) for item in value.values()):
        raise ContractError('telemetry retains primitive bindings only')
    if any(type(item) is str and len(item)>256 for item in value.values()):
        raise ContractError('bounded observation bindings required')
    if (value['role'] not in ('worker','coordinator') or value['model_id'] not in ('mlp','gru','transformer')
            or not _hash(value['spec_sha256']) or not _hash(value['source_digest'])
            or value['resume_identity'] is not None and not _hash(value['resume_identity'])):
        raise ContractError('closed observation binding mismatch')
    integer(value['pid'],1,2**63-1)
    if value['variant'] not in ('baseline','metrics','packed','microbatch','shared-prefix'):
        raise ContractError('closed observation variant required')
    if value['role']=='worker':
        integer(value['generation'],0,2**63-1)
        if type(value['transaction']) is not str or not value['transaction']:
            raise ContractError('worker observation transaction required')
    elif (value['generation'] is not None or value['transaction'] is not None
            or not all(type(value[name]) is str and value[name] for name in ('run_id','context_id'))):
        raise ContractError('coordinator observation run/context required')
    return dict(value)


class ProcessObservation:
    def __init__(self,bindings: dict[str,Any],*,allowance_bytes: int) -> None:
        self.bindings=_bindings(bindings)
        if self.bindings['pid']!=os.getpid():
            raise ContractError('observer must own current process only')
        self._pool=_POOL
        self._pool.acquire(self,allowance_bytes)  # BEFORE observer/thread creation.
        self.allowance_bytes=allowance_bytes
        self.stop=threading.Event()
        self.lock=threading.Lock()
        self.thread: Any=None
        self._pending_fd: int | None=None
        self._pending_path: str | None=None
        self.started=False
        self.finished=False
        self.failures: list[str]=[]
        self.failure_count=0
        self.sample_count=0
        self.sample_failure_count=0
        self.first: float | None=None
        self.last: float | None=None
        self.max_gap=0.
        self.max_pss: int | None=None
        self.max_rss: int | None=None
        self.phases: dict[str,Any]={}
        self.cuda: dict[str,Any]={'status':'not_requested'}
        self.numeric_loop_update_seconds: float | None=None
        self.wall_start=time.perf_counter()
        self.cpu_start=time.process_time()

    def failure(self,where: str,error: BaseException) -> None:
        with self.lock:
            self.failure_count+=1
            if len(self.failures)<8:
                self.failures.append((where+':'+type(error).__name__+':'+str(error))[:256])

    def sample(self) -> None:
        try:
            pss,rss=_read_proc()
            with self.lock:
                now=time.perf_counter()
                self.sample_count+=1
                self.first=now if self.first is None else self.first
                if self.last is not None:
                    self.max_gap=max(self.max_gap,now-self.last)
                self.last=now
                self.max_pss=pss if self.max_pss is None else max(pss,self.max_pss)
                self.max_rss=rss if self.max_rss is None else max(rss,self.max_rss)
        except BaseException as error:  # noqa: BLE001 - sampling cannot mask original work
            self.sample_failure_count+=1
            self.failure('proc',error)

    def _run(self) -> None:
        try:
            while not self.stop.wait(CADENCE_SECONDS):
                self.sample()
        except BaseException as error:  # noqa: BLE001 - retain observation failure, not inputs
            self.failure('sampler',error)

    def start(self) -> None:
        try:
            if self.started:
                raise ContractError('observation start is single-use')
            self.started=True
            self.sample()
            self.thread=threading.Thread(target=self._run,daemon=True,name='gfb-process-observation')
            self.thread.start()
        except BaseException as error:  # noqa: BLE001 - original lifecycle remains authoritative
            self.failure('start',error)

    @contextmanager
    def phase(self,name: str) -> Any:
        started=None
        try:
            self.sample()
            started=(time.perf_counter(),time.process_time())
        except BaseException as error:  # noqa: BLE001 - failed clocks must still execute original body
            self.failure('phase start',error)
        try:
            yield
        finally:
            try:
                if started is None:
                    raise ContractError('phase clocks unavailable')
                wall,cpu=started
                values={'wall_seconds':time.perf_counter()-wall,'process_cpu_seconds':time.process_time()-cpu}
                allowed=WORKER_PHASES if self.bindings['role']=='worker' else COORDINATOR_PHASES
                if name not in allowed or name in self.phases:
                    raise ContractError('closed single-use observation phase required')
                self.phases[name]=values
                self.sample()
            except BaseException as error:  # noqa: BLE001 - never replace a phase's original exception
                self.failure('phase',error)

    def bind_resume(self,value: str) -> None:
        if not _hash(value) or self.bindings['resume_identity'] not in (None,value):
            raise ContractError('observation resume binding mismatch')
        self.bindings['resume_identity']=value

    def finish_publish(self,path: Path,*,original_completed: bool) -> dict[str,Any]:
        """Best effort; no exception/ACK/transition authority. Never retry publish."""
        if self.finished:
            return {'published':False,'diagnostic_complete':False,'failure':'already finished'}
        self.finished=True
        stopped=False
        published=False
        try:
            self.sample()
            self.stop.set()
            if self.thread is not None:
                self.thread.join(timeout=JOIN_SECONDS)
                stopped=not self.thread.is_alive()
            if not stopped:
                self.failure('stop',RuntimeError('sampler did not stop within250ms'))
            wall,cpu=time.perf_counter()-self.wall_start,time.process_time()-self.cpu_start
            required=WORKER_PHASES if self.bindings['role']=='worker' else COORDINATOR_PHASES
            complete=(stopped and original_completed and self.failure_count==0 and self.sample_count>=2
                and set(self.phases)==set(required) and _hash(self.bindings['resume_identity'])
                and (self.bindings['role']!='worker' or type(self.numeric_loop_update_seconds) is float
                    and math.isfinite(self.numeric_loop_update_seconds) and self.numeric_loop_update_seconds>=0))
            value={'schema_version':2,'kind':'actual-process-observation','bindings':dict(self.bindings),
                'lease_bytes':LEASE_BYTES,'lease_pool':'process-observation-workspace-subset-v1',
                'existing_allowance_bytes':self.allowance_bytes,'cadence_seconds':CADENCE_SECONDS,
                'scope':'process threads only; sampled PSS/RSS maxima, never simultaneous tree/device peak',
                'interval_definitions':{'coordinator_call':'original call through return/cancellation cleanup',
                    'worker_request':'parsed valid update receive through retirement and original snapshot/send',
                    'update_call':'original learner.update_owned call; CUDA event includes host submission gaps',
                    'input_retirement':'del numeric, existing sync, delivery.complete(sync), control.close'},
                'sampler':{'stopped':stopped,'sample_count':self.sample_count,
                    'sample_failure_count':self.sample_failure_count,'first_sample':self.first,'last_sample':self.last,
                    'max_gap_seconds':self.max_gap,'max_pss_bytes':self.max_pss,'max_rss_bytes':self.max_rss,
                    'retained_samples':0},'phases':dict(self.phases),'wall_seconds':wall,'process_cpu_seconds':cpu,
                'numeric_loop_update_seconds':self.numeric_loop_update_seconds,'cuda':dict(self.cuda),
                'failures':{'count':self.failure_count,'retained':list(self.failures),
                    'dropped':max(0,self.failure_count-len(self.failures))},
                'original_pipeline_completed':original_completed,'diagnostic_complete':complete}
            raw=canonical(value).encode()
            if len(raw)>SIDECAR_BYTES:
                raise ContractError('observation sidecar exceeds64KiB before create')
            pending=path.with_name('.observation-'+path.name+'.pending')
            fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            self._pending_fd=fd
            self._pending_path=str(pending)
            try:
                offset=0
                while offset<len(raw):
                    written=os.write(fd,raw[offset:])
                    if written<=0:
                        raise OSError('observation publication made no progress')
                    offset+=written
            finally:
                # A close error leaves FD custody quarantined, with no retry.
                os.close(fd)
                self._pending_fd=None
            # Link creates the final name exclusively, only AFTER successful
            # write+close. Unlink uncertainty leaves a visible pending residue
            # that the helper MUST reject even if final bytes look complete.
            os.link(pending,path,follow_symlinks=False)
            os.unlink(pending)
            self._pending_path=None
            published=True
            if stopped:
                self._pool.release(self)
            return {'published':True,'diagnostic_complete':complete,'sampler_stopped':stopped}
        except BaseException as error:  # noqa: BLE001 - original operation exceptions remain unchanged
            self.failure('publication/stop',error)
            return {'published':published,'diagnostic_complete':False,'sampler_stopped':stopped,
                'failure':(type(error).__name__+':'+str(error))[:256]}


def read_observation(path: Path) -> dict[str,Any]:
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=SIDECAR_BYTES:
            raise ContractError('bounded regular observation sidecar required')
        raw=os.read(fd,SIDECAR_BYTES+1)
    finally:
        os.close(fd)
    if len(raw)>SIDECAR_BYTES:
        raise ContractError('observation read exceeds64KiB')
    def pairs(items: Any) -> dict[str,Any]:
        result={}
        for name,item in items:
            if name in result:
                raise ContractError('duplicate observation key')
            result[name]=item
        return result
    value=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ContractError('nonfinite observation')))
    if type(value) is not dict:
        raise ContractError('observation object required')
    return value


def validate_observation(value: dict[str,Any],*,expected: dict[str,Any],require_cuda: bool) -> None:
    def finite(number: Any,positive: bool=False) -> bool:
        return type(number) in (int,float) and math.isfinite(number) and (number>0 if positive else number>=0)
    closed(value,{'schema_version','kind','bindings','lease_bytes','lease_pool','existing_allowance_bytes',
        'cadence_seconds','scope','interval_definitions','sampler','phases','wall_seconds','process_cpu_seconds',
        'numeric_loop_update_seconds','cuda','failures','original_pipeline_completed','diagnostic_complete'})
    binding=_bindings(value.get('bindings'))
    sampler=value.get('sampler',{})
    required=WORKER_PHASES if binding['role']=='worker' else COORDINATOR_PHASES
    if (type(value.get('schema_version')) is not int or value.get('schema_version')!=2
            or value.get('kind')!='actual-process-observation'
            or any(binding.get(name)!=item for name,item in expected.items())
            or type(value.get('lease_bytes')) is not int or value.get('lease_bytes')!=LEASE_BYTES
            or type(value.get('existing_allowance_bytes')) is not int or value['existing_allowance_bytes']<LEASE_BYTES
            or value.get('lease_pool')!='process-observation-workspace-subset-v1'
            or value.get('cadence_seconds')!=CADENCE_SECONDS
            or value.get('diagnostic_complete') is not True or value.get('original_pipeline_completed') is not True
            or sampler.get('stopped') is not True or type(sampler.get('sample_count')) is not int
            or not finite(sampler.get('sample_count'),True)
            or sampler['sample_count']<2 or type(sampler.get('sample_failure_count')) is not int
            or sampler.get('sample_failure_count')!=0
            or not all(finite(sampler.get(name),True) for name in ('max_pss_bytes','max_rss_bytes'))
            or not all(finite(sampler.get(name)) for name in ('first_sample','last_sample','max_gap_seconds'))
            or sampler['last_sample']<sampler['first_sample'] or sampler.get('retained_samples')!=0
            or type(value.get('failures',{}).get('count')) is not int
            or value['failures']['count']!=0 or set(value.get('phases',{}))!=set(required)
            or not all(finite(value.get(name)) for name in ('wall_seconds','process_cpu_seconds'))):
        raise ContractError('incomplete or mismatched observation')
    if binding['role']=='worker' and not finite(value.get('numeric_loop_update_seconds')):
        raise ContractError('original numeric-loop clock unavailable')
    for phase in value['phases'].values():
        if set(phase)!={'wall_seconds','process_cpu_seconds'} or not all(finite(number) for number in phase.values()):
            raise ContractError('invalid observation phase clocks')
    if require_cuda:
        cuda=value.get('cuda',{})
        if (cuda.get('status')!='observed' or not finite(cuda.get('update_call_event_ms'),True)
                or not all(finite(cuda.get('allocator_after_retirement',{}).get(name),True)
                    for name in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'))):
            raise ContractError('required real CUDA observation unavailable')
