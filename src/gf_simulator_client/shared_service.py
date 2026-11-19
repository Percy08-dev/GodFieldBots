"""Invocation-owned clean broker, bounded private endpoints and composite reap."""
from __future__ import annotations

import math
import os
import select
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, digest
from gf_common.ram_bank import RamBankChannel
from gf_common.shared_inference import shared_runtime
from gf_common.shared_requests import RequestAuthority
from gf_simulator_client.framing import read_frame, write_frame


class _DeadlineReader:
    def __init__(self,source: Any,deadline: float) -> None:
        self.source,self.deadline=source,deadline

    def read(self,size: int) -> bytes:
        remaining=self.deadline-time.monotonic()
        if remaining<=0 or not select.select([self.source],[],[],remaining)[0]:
            raise TimeoutError('shared control response deadline')
        return os.read(self.source.fileno(),size)


class SharedInferenceService:
    """Coordinator owns no Torch objects and never computes over numeric rows."""
    def __init__(self,ledger: Any,runtime: Json,*,execution: str,directory: Path,
                 slots: int,estimate: Json,visibility: str | None=None,job_id: str | None=None,
                 device: str='cuda:0',deadline: float=float('inf'),execution_deadline: float | None=None,run_id: str | None=None,config_sha256: str | None=None) -> None:
        from gf_simulator_client.performance_diagnostics import host_diagnostics
        self.diagnostics = host_diagnostics()
        self._measurement_scope: Json | None = None
        self.ledger,self.runtime=ledger,shared_runtime(runtime,game_concurrency=slots//2)
        self.execution,self.service_id=execution,uuid.uuid4().hex
        self.directory=directory/'services'/self.service_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.deadline=deadline
        self.execution_deadline=deadline if execution_deadline is None else execution_deadline
        self.failed=False
        self.estimate=estimate
        self.controller: Any=None
        self.lock=threading.RLock()
        self.authority=RequestAuthority(execution)
        self.registration: Any=None
        self.retirement: Any=None
        self.proof: Any=None
        self.process: Any=None
        self.closed=False
        self.uncertain=False
        self.loaded: dict[str,Json]={}
        self._endpoint_owners: dict[int,Any]={}
        self.lease=ledger.reserve_service(estimate['cuda_bytes'],self)
        self._pairs: list[Any]=[]
        self.channel: Any=None
        self._inherited: Any=None
        self._spawn_attempted=False
        self._popen_failure: BaseException | None=None
        self._no_child_error: BaseException | None=None
        try:
            for _ in range(slots):self._pairs.append(socket.socketpair())
            self.channel,self._inherited=RamBankChannel.pair(timeout=30.)
            bank_fd,bank_invocation=self._inherited.inherited_endpoint()
            inherited=tuple(right.fileno() for _,right in self._pairs)+(bank_fd,)
            self._spawn_attempted=True
            try:
                self.process=subprocess.Popen([sys.executable,'-E','-m','gf_simulator_client.shared_bootstrap'],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,
                    env={'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},pass_fds=inherited)
            except Exception as error:
                self._popen_failure=error  # Actual synchronous creator exception, not a PID inference.
                raise
            self.channel.bind_child(self.process)
            write_frame(self.process.stdin,{'service_id':self.service_id,'execution':execution,
                'job_id':job_id or execution,'run_id':run_id or execution,'config_sha256':config_sha256,'epoch':1,'runtime':self.runtime,'device':device,'visibility':visibility,
                **({'performance_diagnostics': {'directory': str(self.diagnostics.directory),
                    'observation_id': self.diagnostics.observation_id, 'profile_recipe': self.diagnostics.profile_recipe}}
                    if self.diagnostics is not None else {}),
                'endpoints':[right.fileno() for _,right in self._pairs],'bank_fd':bank_fd,
                'bank_invocation':bank_invocation,'cleanup_deadline':deadline if math.isfinite(deadline) else None,'host_bytes':max(estimate.get('host_bytes',0),
                    runtime['cache_weight_bytes']*4+estimate.get('batch_input_bytes',0)*34+65536)})
            self._inherited.close()
            for _,right in self._pairs:right.close()
            registration=read_frame(_DeadlineReader(self.process.stdout,min(deadline,time.monotonic()+40.)))
            if (registration.get('kind')!='shared-service-registration' or registration.get('service_id')!=self.service_id
                    or registration.get('execution')!=execution or registration.get('broker_pid')!=self.process.pid
                    or type(registration.get('gpu_pid')) is not int):
                raise ContractError('shared broker registration differs from original launch')
            self.registration=registration
            self._persist('registration.json',registration)
        except BaseException as error:
            vars(error)['_shared_service_owner']=self
            if self.process is None and (not self._spawn_attempted or error is self._popen_failure):
                try:self._confirm_no_child(error,job_id or execution,run_id or execution,config_sha256,device)
                except BaseException as cleanup:  # noqa: BLE001 - keep original construction failure and custody.
                    self.uncertain=True
                    error.add_note('shared no-child cleanup unconfirmed: '+type(cleanup).__name__)
                raise
            self.uncertain=True
            # Returned broker without nested reap proof remains live custody.
            if self.process is not None and self.process.poll() is None:
                self.process.terminate();self.process.wait(timeout=10.)
            raise

    def _confirm_no_child(self,error: BaseException,job_id: str,run_id: str,config_sha256: str | None,device: str) -> None:
        if self.process is not None or self._spawn_attempted and error is not self._popen_failure:
            raise ContractError('original shared no-child creation observation required')
        for left,right in self._pairs:left.close();right.close()
        if self.channel is not None:self.channel.close()
        if self._inherited is not None:self._inherited.close()
        registration={'schema_version':1,'kind':'shared-service-no-child','service_id':self.service_id,
            'execution':self.execution,'job_id':job_id,'run_id':run_id,'config_sha256':config_sha256,
            'device':device,'failure_type':type(error).__name__,'broker_pid':None,'gpu_pid':None}
        retirement={'schema_version':1,'kind':'shared-service-no-child-retirement','service_id':self.service_id,
            'registration_sha256':digest(registration),'pending':0,'active_relays':0,'relays_joined':True,'successful':False}
        proof={'key':1,'service_id':self.service_id,'broker_pid':None,'broker_outcome':'never_started',
            'gpu_pid':None,'gpu_outcome':'never_started','gpu_exitcode':None,
            'registration_sha256':digest(registration),'retirement_sha256':digest(retirement)}
        self._persist('registration.json',registration)
        self._persist('retirement.json',retirement)
        self._persist('proof.json',proof)
        self.registration,self.retirement,self.proof=registration,retirement,proof
        self._no_child_error=error
        self.closed=self.failed=True
        self.lease.retired(self,proof)

    def _persist(self,name: str,value: Json) -> None:
        from gf_common.domain import canonical
        encoded=(canonical(value)+'\n').encode()
        with (self.directory/name).open('xb') as destination:
            destination.write(encoded);destination.flush();os.fsync(destination.fileno())

    def _exchange(self,value: Json,*,timeout: float=35.) -> Json:
        with self.lock:
            if not self.closed and (self.process is None or self.process.poll() is not None):self.uncertain=True
            if self.closed or self.uncertain:
                raise ContractError('shared broker is closed, failed or uncertain')
            try:
                write_frame(self.process.stdin,value)
                response=read_frame(_DeadlineReader(self.process.stdout,min(self.deadline,time.monotonic()+timeout)))
            except BaseException as error:
                self.uncertain=True
                vars(error)['_shared_service_owner']=self
                raise
            if 'error' in response:
                self.uncertain=True
                raise ContractError('shared inference infrastructure failure')
            return response

    def measurement_begin(self, scope: Json) -> None:
        if self.diagnostics is None:
            return
        try:
            response = self._exchange({'command': 'DIAGNOSTIC_BEGIN', 'scope': scope})
            if response != {'accepted': True}:
                raise ValueError('diagnostic start not positively quiescent')
            self._measurement_scope = dict(scope)
        except Exception as error:  # noqa: BLE001 - diagnostic failure is never policy/Source authority.
            self.diagnostics.error('legacy_measurement_begin:' + type(error).__name__)

    def measurement_end(self) -> None:
        if self.diagnostics is None or self._measurement_scope is None:
            return
        try:
            response = self._exchange({'command': 'DIAGNOSTIC_END', 'scope': self._measurement_scope})
            if response.get('accepted') is not True:
                raise ValueError('diagnostic end not positively quiescent')
        except Exception as error:  # noqa: BLE001 - diagnostic failure is never policy/Source authority.
            self.diagnostics.error('legacy_measurement_end:' + type(error).__name__)
        finally:
            self._measurement_scope = None

    def acquire_endpoint(self,owner: Any) -> tuple[int,int]:
        for index,(left,_) in enumerate(self._pairs):
            if index not in self._endpoint_owners:
                self._endpoint_owners[index]=owner
                return index,left.fileno()
        raise ContractError('shared participant endpoint capacity exhausted')

    def release_endpoint(self,index: int,owner: object) -> None:
        if self._endpoint_owners.get(index) is not owner:
            raise ContractError('shared endpoint release requires original participant owner')
        if self._endpoint_owners[index].process.returncode is None:
            raise ContractError('shared endpoint requires confirmed participant reap')
        self.authority.retire(str(index))
        del self._endpoint_owners[index]

    def authorize(self,index: int,owner: Any,*,game: str,participant: int,revision: int,checkpoint: str,deadline: float) -> Json:
        if self._endpoint_owners.get(index) is not owner or self.ledger.updating or self.failed:
            raise ContractError('shared request lacks actual idle participant capability')
        self.lease.invalidate(self)
        routing=self.authority.begin(str(index),game=game,participant=participant,revision=revision,checkpoint=checkpoint)
        response=self._exchange({'command':'AUTHORIZE','endpoint':str(index),'routing':routing,'deadline':deadline})
        if response.get('accepted') is False and response.get('reason')=='expired':
            self.authority.retire(str(index))
            self.failed=True
            raise TimeoutError('shared request expired before acceptance')
        if response!={'accepted':True}:raise ContractError('ambiguous shared request acceptance')
        return routing

    def complete(self,index: int,routing: Json,*,completed: bool) -> None:
        self.authority.match(str(index),routing)
        if not completed:self.failed=True
        response=self._exchange({'command':'RETIRE','endpoint':str(index),'completed':completed})
        if response!={'accepted':True}:raise ContractError('ambiguous shared request retirement')
        self.authority.retire(str(index),completed=completed)

    def publish(self,banks: Any,*,model: Json,binding: Json,checkpoint: str,generation: int,identity: str) -> Json:
        if self.failed:raise ContractError('failed shared invocation cannot publish weights')
        if checkpoint in self.loaded:return self.loaded[checkpoint]
        metadata={'command':'PUBLISH','checkpoint':checkpoint,'model':model,'binding':binding}
        with self.lock:
            if self.process is None:raise ContractError('shared publication before original broker creation')
            self.lease.invalidate(self)
            write_frame(self.process.stdin,{'command':'PUBLISH','metadata':metadata,'bank_count':len(banks.banks),
                'generation':generation,'identity':identity})
            leases=[]
            try:
                for index,bank in enumerate(banks.banks):
                    leases.append(self.channel.send(bank,purpose='model',generation=generation,identity=identity+f':{index}'))
                for lease in leases:self.channel.accept_completion(lease)
                response=read_frame(_DeadlineReader(self.process.stdout,min(self.deadline,time.monotonic()+35.)))
                if response.get('accepted') is not True or response.get('checkpoint')!=checkpoint:
                    raise ContractError('ambiguous committed shared model publication')
                self.loaded[checkpoint]=response
                return response
            except BaseException:
                self.uncertain=True
                raise

    def evict(self,checkpoint: str) -> Json:
        if any(row['checkpoint']==checkpoint for row in self.authority.pending.values()):
            raise ContractError('immutable checkpoint remains pinned by valid request')
        response=self._exchange({'command':'EVICT','checkpoint':checkpoint})
        if response!={'accepted':True,'checkpoint':checkpoint}:raise ContractError('shared checkpoint eviction unconfirmed')
        del self.loaded[checkpoint]
        return response

    def quiesce(self) -> object:
        if not self.authority.empty:raise ContractError('shared participant requests remain pending')
        response=self._exchange({'command':'QUIESCE'})
        if (self.failed or response.get('accepted') is not True or response.get('pending')!=0
                or response.get('active_relays')!=0 or response.get('failures')!=0):
            raise ContractError('shared request/relay retirement failed')
        self.telemetry=response
        return self.lease.quiescent(self)

    def close(self) -> Json:
        if self.proof is not None:return self.proof
        abort=self.failed or not self.authority.empty
        if not abort:
            try:self.quiesce()
            except ContractError:
                abort=True
                self.failed=True
        if abort:
            self.failed=True
            for endpoint in tuple(self.authority.pending):self.authority.retire(endpoint)
        response=self._exchange({'command':'ABORT_CLOSE' if abort else 'CLOSE','deadline':min(self.deadline,time.monotonic()+35.)})
        if (response.get('kind')!='shared-service-retirement' or response.get('service_id')!=self.service_id
                or response.get('registration_sha256')!=digest(self.registration) or response.get('relays_joined') is not True
                or response.get('pending')!=0 or response.get('active_relays')!=0 or response.get('gpu_pid')!=self.registration['gpu_pid']
                or type(response.get('gpu_exitcode')) is not int):
            self.uncertain=True
            raise ContractError('original shared GPU service reap unconfirmed')
        self.retirement=response
        if response.get('successful') is not True:self.failed=True
        self._persist('retirement.json',response)
        self.process.wait(timeout=max(0.,min(10.,self.deadline-time.monotonic())))
        if self.process.returncode!=0:raise ContractError('original broker reap failed')
        for left,_ in self._pairs:left.close()
        self.channel.close()
        self.process.stdin.close();self.process.stdout.close()
        self.closed=True
        proof={'key':1,'service_id':self.service_id,'broker_pid':self.process.pid,'broker_outcome':'reaped',
            'gpu_pid':response['gpu_pid'],'gpu_outcome':'reaped','gpu_exitcode':response['gpu_exitcode'],
            'registration_sha256':digest(self.registration),'retirement_sha256':digest(response)}
        self.proof=proof
        self.lease.retired(self,proof)
        self._persist('proof.json',proof)
        return proof

    def require_retirement(self,proof: Any) -> None:
        if (self._no_child_error is not None and proof is self.proof and self.closed and not self.uncertain
                and self.process is None and (not self._spawn_attempted or self._no_child_error is self._popen_failure)):
            return
        if (proof is not self.proof or not self.closed or self.uncertain or self.retirement is None
                or self.process is None or self.process.returncode!=0 or not self.authority.empty
                or self.channel.outstanding):
            raise ContractError('SAME original composite shared service retirement required')
