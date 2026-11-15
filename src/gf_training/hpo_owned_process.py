"""Wire7 adapter retaining the existing actual learner creation/reap lifecycle."""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.ram_bank import MAX_BANK_BYTES, RamBankChannel, ReadonlyRamBankSet
from gf_training.hpo_learning_process import HpoLearningProcess
from gf_training.hpo_owned_worker import _control_bank, _receive_control
from gf_training.ppo_run import LearningProcess

_TOKEN = object()


async def _ready(channel: RamBankChannel, *, write: bool = False, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    fd = channel.socket.fileno()
    def signal() -> None:
        if not future.done():
            future.set_result(None)
    add, remove = (loop.add_writer, loop.remove_writer) if write else (loop.add_reader, loop.remove_reader)
    add(fd, signal)
    try:
        async with asyncio.timeout_at(deadline):
            await future
    finally:
        remove(fd)
    if time.monotonic() >= deadline:
        raise TimeoutError('owned channel deadline')


def _saved_files(files: Any) -> None:
    if type(files) is not dict or not {'inference.json','resume.json'} <= set(files):
        raise ContractError('Wire7 complete saved file map required')
    for name,value in files.items():
        if (type(name) is not str or name not in ('inference.json','resume.json')
                and re.fullmatch(r'(weights|resume)-[0-9]{6}\.(npz|json)',name) is None
                or type(value) is not str or len(value)!=64 or set(value)-set('0123456789abcdef')):
            raise ContractError('Wire7 invalid saved file map')
    for prefix in ('weights','resume'):
        npz = {name[:-4] for name in files if name.startswith(prefix+'-') and name.endswith('.npz')}
        sidecars = {name[:-5] for name in files if name.startswith(prefix+'-') and name.endswith('.json')}
        if not npz or npz != sidecars or npz != {f'{prefix}-{index:06}' for index in range(len(npz))}:
            raise ContractError('Wire7 incomplete saved shard inventory')


def _snapshot_facts(facts: Json, device: str) -> None:
    hardware=closed(facts['hardware'],{'device','name','cuda_runtime','device_count',
        'cuda_driver_api_version','exact_driver_build'})
    if hardware['device']!=device or type(hardware['name']) is not str or not hardware['name']:
        raise ContractError('Wire7 actual child hardware identity mismatch')
    integer(hardware['device_count'],0 if device=='cpu' else 1,0 if device=='cpu' else 2**31-1)
    if (hardware['cuda_runtime'] is not None and type(hardware['cuda_runtime']) is not str
            or hardware['exact_driver_build']!='unverified'
            or device=='cpu' and hardware['cuda_driver_api_version'] is not None):
        raise ContractError('Wire7 invalid actual child runtime/driver facts')
    if device!='cpu': integer(hardware['cuda_driver_api_version'],1)
    timings=closed(facts['timings'],{'snapshot_seconds','save_seconds'})
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in timings.values()):
        raise ContractError('Wire7 nonfinite/negative actual snapshot timings')


def _owned_metrics(value: Json, *, batch: Any, training: Any, expected_steps: int) -> None:
    if batch is None:
        closed(value,set())
        return
    losses={'policy_loss','value_mse','entropy','approx_kl','total_loss','update_seconds'}
    counters={'optimizer_steps','optimizer_step_delta','collected_decisions','used_decisions',
        'tbptt_length','epochs','endpoint_evaluations','context_decisions','suffix_decisions'}
    closed(value,losses|counters|{'parity'})
    if value['parity']!={'status':'not_run'}:
        raise ContractError('Wire7 parity must truthfully record not_run')
    for name in losses:
        number=value[name]
        if type(number) not in (int,float) or not math.isfinite(number) or name=='update_seconds' and number<0:
            raise ContractError('Wire7 nonfinite/invalid numerical metric')
    for name in counters: integer(value[name],0,2**63-1)
    delta=training.epochs*((batch.used_decisions+training.minibatch_size-1)//training.minibatch_size)
    expected={'optimizer_steps':expected_steps,'optimizer_step_delta':delta,
        'collected_decisions':batch.collected_decisions,'used_decisions':batch.used_decisions,
        'tbptt_length':training.tbptt_length,'epochs':training.epochs,
        'endpoint_evaluations':training.epochs*batch.used_decisions,
        'context_decisions':training.epochs*sum(position+1 for _,position in batch.locations),
        'suffix_decisions':training.epochs*sum(min(position+1,training.tbptt_length) for _,position in batch.locations)}
    if any(value[name]!=number for name,number in expected.items()):
        raise ContractError('Wire7 numeric counts differ from actual captured update')


class RemoteSaveAck:
    """Actual Wire7 final writer ACK bound to its original child transaction."""
    def __init__(self, token: object, snapshot: RemoteModelSnapshot, transaction: object, reply: Json) -> None:
        if token is not _TOKEN:
            raise ContractError('actual Wire7 writer ACK required')
        self._snapshot, self._transaction, self._reply = snapshot, transaction, canonical(reply).encode()
        self.generation, self.optimizer_steps = snapshot.generation, snapshot.optimizer_steps
        self._identity = snapshot.identity
        from gf_training.hpo_owned_snapshot import _backings
        self._inference, self._resume = _backings(snapshot.inference), _backings(snapshot.resume)

    @property
    def files(self) -> Json:
        return self._snapshot.files

    def require(self, snapshot: RemoteModelSnapshot) -> None:
        if (snapshot is not self._snapshot or snapshot.save_ack is not self
                or snapshot._transaction is not self._transaction or snapshot._closed
                or snapshot.identity != self._identity or snapshot.generation != self.generation
                or snapshot.optimizer_steps != self.optimizer_steps):
            raise ContractError('remote save ACK differs from actual child transaction')
        from gf_training.hpo_owned_snapshot import _backings
        if self._inference != _backings(snapshot.inference) or self._resume != _backings(snapshot.resume):
            raise ContractError('remote save ACK original backings changed')


class RemoteModelSnapshot:
    """Permanent SAME-backing ownership minted by one actual Wire7 transaction."""
    def __init__(self, token: object, process: HpoOwnedLearningProcess, transaction: object,
                 inference: ReadonlyRamBankSet, resume: ReadonlyRamBankSet, control: Any) -> None:
        if token is not _TOKEN or process.process is None or process.process.returncode is not None:
            raise ContractError('actual live Wire7 snapshot issuer required')
        self._process, self._transaction = process, transaction
        self.inference, self.resume, self._control = inference, resume, control
        self.identity, self.generation = control.value['snapshot_identity'], control.value['generation']
        self.optimizer_steps = control.value['optimizer_steps']
        self._closed = False
        self._transport_timings: dict[str,float] = {}
        self.save_ack: RemoteSaveAck | None = None
        self._actor_capability: Any = None

    @property
    def actor_metadata(self) -> Json:
        self._require_saved()
        return json.loads(canonical(self._control.value['actor_metadata']))

    @property
    def files(self) -> Json:
        self._require_saved()
        return dict(self._control.value['files'])

    @property
    def timings(self) -> Json:
        self._require_saved()
        return {**self._control.value['timings'],**getattr(self,'_transport_timings',{})}

    @property
    def metrics(self) -> Json:
        self._require_saved()
        return json.loads(canonical(self._control.value['metrics']))

    def _require_saved(self) -> None:
        if self._closed or self.save_ack is None:
            raise ContractError('remote snapshot lacks complete actual save ACK')

    def require_process(self, process: HpoOwnedLearningProcess) -> None:
        self._require_saved()
        assert self.save_ack is not None
        self.save_ack.require(self)
        if process is not self._process:
            raise ContractError('remote snapshot differs from actual learner process')

    @property
    def actor_capability(self) -> Any:
        from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
        _check_actor_origin(self)
        if self._actor_capability is None:
            self._actor_capability=_issue_actor_snapshot(self,inference=self.inference,
                metadata=self.actor_metadata,check_origin=_check_actor_origin,budget=self._process.budget)
        self._actor_capability.require()
        return self._actor_capability

    def close(self) -> None:
        if not self._closed:
            self.inference.close()
            self.resume.close()
            self._control.close()
            if self._actor_capability is not None:
                self._actor_capability._retire_origin(self)
            self._closed = True


def _check_actor_origin(owner: RemoteModelSnapshot) -> None:
    if type(owner) is not RemoteModelSnapshot:
        raise ContractError('actor capability requires actual Wire7 snapshot completion')
    owner._require_saved()
    assert owner.save_ack is not None
    owner.save_ack.require(owner)


class HpoOwnedLearningProcess(HpoLearningProcess):
    """Separate initialize_owned/update_owned entry; raw requests cannot update."""
    def __init__(self, *args: Any, budget: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.protocol, self.budget = 7, budget
        from gf_training.hpo_artifacts import save_estimate
        limits=self.identity['phase_limits']
        for value in limits.values(): integer(value,1,MAX_BANK_BYTES)
        if save_estimate(self.model)['additional_bytes']>limits['save_budget_bytes']:
            raise ContractError('Wire7 additional save envelope insufficient before process/network allocation')
        self.channel, self._inherited = RamBankChannel.pair()
        self._owned_failed = False
        self.generation = 0
        self.current_snapshot: RemoteModelSnapshot | None = None
        self._partial_banks: list[Any] = []
        self._partial_controls: list[Any] = []
        self._control_exports: list[Any] = []
        self._inflight_batch: Any = None

    async def _create_process(self) -> Any:
        fd, invocation = self._inherited.inherited_endpoint()
        try:
            return await asyncio.create_subprocess_exec(sys.executable, '-E', '-m', 'gf_training.hpo_owned_worker',
                str(fd), invocation, pass_fds=(fd,), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._environment(), limit=1024*1024)
        except Exception as error:
            self._creation_no_child = True
            self._creation_failed(error)
            raise

    def _process_created(self) -> None:
        super()._process_created()
        self.channel.bind_child(self.process)
        self._inherited.close()

    async def request(self, request: Json, deadline: float) -> Json:
        # Only original cleanup can issue a plain control request. Data authority
        # is always an actual OwnedNumericBatch/RemoteModelSnapshot object below.
        closed(request, {'protocol_version', 'command', 'identity'})
        if (request['protocol_version'] != 7 or request['command'] != 'close'
                or canonical(request['identity']) != canonical(self.identity)):
            raise ContractError('Wire7 raw request cannot issue numerical authority')
        reply = await LearningProcess.request(self, request, deadline)
        self._validate_close(reply)
        return reply

    def _validate_close(self, result: Json) -> None:
        if canonical(result) != canonical({'protocol_version':7,'command':'close','identity':self.identity,'closed':True}):
            raise ContractError('Wire7 original close acknowledgement required')

    def _require_live(self) -> None:
        if (self._owned_failed or self.process is None or self.process.returncode is not None
                or self.lease is None or self.lease.released):
            raise ContractError('Wire7 actual learner is not healthy/live/owned')

    async def initialize_owned(self, seed: int, output: Path, *, deadline: float) -> RemoteModelSnapshot:
        self._require_live()
        if self.steps is not None:
            raise ContractError('Wire7 initialization is single use')
        request = {'protocol_version':7,'command':'initialize_owned','identity':self.identity,
            'model':asdict(self.model),'seed':integer(seed,0,2**63-1),'output':str(output),'transaction':uuid.uuid4().hex}
        return await self._exchange_owned(request, deadline=deadline)

    async def update_owned(self, batch: Any, output: Path, *, deadline: float, resident_authority: Any = None) -> RemoteModelSnapshot:
        from gf_training.hpo_owned_rollout import OwnedNumericBatch
        self._require_live()
        if type(batch) is not OwnedNumericBatch or self.current_snapshot is None:
            raise ContractError('Wire7 requires actual captured batch/current snapshot')
        self.current_snapshot.require_process(self)
        if batch._budget is not self.budget:
            raise ContractError('Wire7 batch and model require SAME invocation Source account')
        if (batch._snapshot_owner is not self.current_snapshot or batch.generation != self.generation
                or batch.metadata['snapshot_identity'] != self.current_snapshot.identity):
            raise ContractError('Wire7 batch differs from original current snapshot')
        if batch._state != 'PERSISTED':
            raise ContractError('Wire7 update requires persisted batch, single use')
        request = {'protocol_version':7,'command':'update_owned','identity':self.identity,
            'transaction':uuid.uuid4().hex,'generation':self.generation,'expected_steps':self.steps,
            'batch_identity':batch.identity,'bank_count':len(batch.banks.banks),
            'policy_version':batch.policy_version,'output':str(output)}
        # Preserve the existing ledger's actual update ownership boundary.
        if resident_authority is None:
            self.ledger.begin_update(0)
        else:
            from gf_training.hpo_gpu_lease import _AuditedLedger
            if type(self.ledger) is not _AuditedLedger:
                raise ContractError('Wire7 resident update requires SAME audited ledger')
            self.ledger.begin_resident_update(0,authority=resident_authority)
        batch._begin_update()
        self._inflight_batch = batch
        try:
            result = await self._exchange_owned(request, deadline=deadline, batch=batch)
            batch._state = 'UPDATED'
            self._inflight_batch = None
            return result
        finally:
            if not self._owned_failed and self.ledger.updating:
                self.ledger.finish_update()

    async def _exchange_owned(self, request: Json, *, deadline: float, batch: Any = None) -> RemoteModelSnapshot:
        transaction = object()
        transport_timings: dict[str,float] = {}
        try:
            if batch is None:
                ready = await LearningProcess.request(self, request, deadline)
            else:
                async with asyncio.timeout_at(deadline):
                    self.process.stdin.write((canonical(request)+'\n').encode())
                    await self.process.stdin.drain()
                control = _control_bank(batch.metadata, self.budget)
                self._control_exports.append(control)
                leases = []
                send_started=time.perf_counter()
                for index, bank in enumerate((control, *batch.banks.banks)):
                    await _ready(self.channel, write=True, deadline=deadline)
                    suffix = ':metadata' if index == 0 else f':{index-1}'
                    leases.append(self.channel.send(bank, purpose='batch', generation=batch.generation,
                        identity=batch.identity+suffix))
                transport_timings['batch_send_seconds']=time.perf_counter()-send_started
                completion_started=time.perf_counter()
                for lease in leases:
                    await _ready(self.channel, deadline=deadline)
                    self.channel.accept_completion(lease)
                transport_timings['batch_completion_wait_seconds']=time.perf_counter()-completion_started
                control.close()
                self._control_exports.remove(control)
                ready = await self._read_reply(deadline)
            closed(ready, {'protocol_version','kind','transaction','identity','generation','snapshot_identity'})
            integer(ready['protocol_version'],7,7)
            integer(ready['generation'],0,2**63-1)
            generation = 0 if batch is None else self.generation+1
            if (ready['protocol_version'] != 7 or ready['kind'] != 'SNAPSHOT_READY'
                    or ready['transaction'] != request['transaction'] or ready['generation'] != generation
                    or canonical(ready['identity']) != canonical(self.identity)):
                raise ContractError('Wire7 snapshot READY differs from actual request')
            snapshot_identity = ready['snapshot_identity']
            snapshot_started=time.perf_counter()
            await _ready(self.channel, deadline=deadline)
            control = _receive_control(self.channel, purpose='save', generation=generation,
                identity=snapshot_identity+':metadata', budget=self.budget)
            self._partial_controls.append(control)
            facts = closed(control.value, {'actor_metadata','files','optimizer_steps','generation','snapshot_identity',
                'metrics','inference_banks','resume_banks','hardware','timings'})
            integer(facts['generation'],0,2**63-1)
            integer(facts['optimizer_steps'],0,2**31-1)
            _saved_files(facts['files'])
            _snapshot_facts(facts,self.binding.device)
            if facts['snapshot_identity'] != snapshot_identity or facts['generation'] != generation:
                raise ContractError('Wire7 snapshot control identity mismatch')
            banksets = []
            total = 0
            for kind in ('inference','resume'):
                banks = []
                for index in range(integer(facts[kind+'_banks'],1,65536)):
                    await _ready(self.channel, deadline=deadline)
                    delivery = self.channel.receive(purpose='save', generation=generation,
                        identity=snapshot_identity+f':{kind}:{index}', budget=self.budget,
                        max_bytes=min(MAX_BANK_BYTES, self.identity['phase_limits']['save_budget_bytes'])-total)
                    total += delivery.bank.size
                    retained = delivery.retain(self.budget)
                    self._partial_banks.append(retained)
                    banks.append(retained)
                    delivery.complete(lambda: None)
                banksets.append(ReadonlyRamBankSet(tuple(banks)))
            from gf_common.card_instance_rules import card_instance_rules
            from gf_models.hpo_owned_inference import _validate_owned_inference
            _validate_owned_inference(banksets[0], facts['actor_metadata'], model=self.model,
                binding=self.binding, actions=self.actions, rules=card_instance_rules())
            if (facts['actor_metadata']['snapshot_identity'] != snapshot_identity
                    or facts['actor_metadata']['generation'] != generation
                    or facts['actor_metadata']['inference_digest'] != facts['files']['inference.json']):
                raise ContractError('Wire7 inference metadata differs from actual snapshot save')
            snapshot = RemoteModelSnapshot(_TOKEN,self,transaction,banksets[0],banksets[1],control)
            final = await self._read_reply(deadline)
            closed(final, {'protocol_version','kind','transaction','identity','generation','optimizer_steps','snapshot_identity','files_sha256'})
            integer(final['protocol_version'],7,7)
            integer(final['generation'],0,2**63-1)
            integer(final['optimizer_steps'],0,2**31-1)
            expected_steps = 0 if batch is None else self.steps + self.hpo_training.epochs * ((len(batch)+self.hpo_training.minibatch_size-1)//self.hpo_training.minibatch_size)
            _owned_metrics(facts['metrics'],batch=batch,training=self.hpo_training,expected_steps=expected_steps)
            if (final['protocol_version'] != 7 or final['kind'] != 'SAVED'
                    or final['transaction'] != request['transaction'] or final['generation'] != generation
                    or final['snapshot_identity'] != snapshot_identity or final['optimizer_steps'] != expected_steps
                    or facts['optimizer_steps'] != expected_steps or final['files_sha256'] != digest(facts['files'])
                    or canonical(final['identity']) != canonical(self.identity)):
                raise ContractError('Wire7 actual complete writer ACK mismatch')
            snapshot.save_ack = RemoteSaveAck(_TOKEN,snapshot,transaction,final)
            transport_timings['snapshot_receive_seconds']=time.perf_counter()-snapshot_started
            snapshot._transport_timings=transport_timings
            self.hardware=dict(facts['hardware'])
            self._partial_controls.remove(control)
            for bankset in banksets:
                for bank in bankset.banks:
                    self._partial_banks.remove(bank)
            self.steps,self.generation,self.current_snapshot = expected_steps,generation,snapshot
            return snapshot
        except BaseException as error:
            self._owned_failed = True
            try:
                self.channel.retire_received(lambda: None)
            except BaseException as cleanup:  # noqa: BLE001 - preserve original failure and retained receive owner
                error.add_note('Wire7 receive cleanup retained: '+type(cleanup).__name__)
            raise

    async def _read_reply(self, deadline: float) -> Json:
        async with asyncio.timeout_at(deadline):
            raw = await self.process.stdout.readline()
        if not raw or self.health.get('log_limit_exceeded'):
            raise RuntimeError('Wire7 numerical child failed; inspect bounded learner log')
        reply = json.loads(raw)
        if type(reply) is not dict:
            raise ContractError('Wire7 JSON acknowledgement object required')
        return reply

    async def close(self, deadline: float) -> None:
        # Known exported aliases can keep receive cleanup deferred after actual
        # reap; retry those owners without recreating a child or retrying work.
        if (self.close_task is not None and self.close_task.done() and self.process is not None
                and self.process.returncode is not None and (self.channel.received_pending
                    or self._partial_banks or self._partial_controls or self._control_exports)):
            self.close_task = None
        await super().close(deadline)

    async def _close(self, deadline: float) -> None:
        try:
            await super()._close(deadline)
        finally:
            active_error=sys.exception()
            failures: list[BaseException]=[]
            def attempt(operation: Any) -> bool:
                try:
                    operation()
                    return True
                except BaseException as cleanup:  # noqa: BLE001 - every original cleanup owner must survive failure
                    failures.append(cleanup)
                    return False
            if self.process is not None and self.process.returncode is not None:
                attempt(lambda:self.channel.retire_received(lambda:None))
                for lease in self.channel.outstanding:
                    attempt(lambda lease=lease:lease.release_after_reap(self.process))
                for bank in tuple(self._partial_banks):
                    if attempt(bank.close): self._partial_banks.remove(bank)
                for control in tuple(self._partial_controls):
                    if attempt(control.close): self._partial_controls.remove(control)
                for bank in tuple(self._control_exports):
                    if attempt(bank.close): self._control_exports.remove(bank)
                if self._inflight_batch is not None:
                    self._inflight_batch._state='FAILED'
                    self._inflight_batch=None
                attempt(self.channel.close)
                attempt(self._inherited.close)
            elif self.process is None and self._creation_no_child:
                attempt(self.channel.close)
                attempt(self._inherited.close)
            if failures:
                if active_error is None:
                    raise failures[0]
                for cleanup in failures:
                    active_error.add_note('Wire7 actual cleanup retained: '+type(cleanup).__name__)
