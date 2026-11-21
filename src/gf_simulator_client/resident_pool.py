"""Viewer-partitioned physical transport; parent validators remain episode-local."""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from gf_common.domain import ContractError, Json, RuntimePolicyBinding, digest, integer
from gf_common.hpo_progress import ProgressCallback, observe, pool_diagnostic
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.framing import write_frame
from gf_simulator_client.structured_worker import StructuredParticipantWorker


class _ExchangeSetup:
    """Original gated executor submission, including its partial-construction owner."""
    def __init__(self,owner: ResidentPhysicalWorker,payload: Json,deadline: float) -> None:
        self.owner,self.executor,self.payload,self.deadline = owner,owner.executor,payload,deadline
        self._evaluation_use_reservation: Any=None
        self.snapshot: Any = None
        self.owned_cache_identity: str | None = None
        request = payload['request']
        policy_options = payload.get('payload', {}).get('policy_options')
        if owner.pool.owned_snapshots and not owner.pool.shared and request['command'] in ('BEGIN', 'SYNC') and policy_options is not None:
            checkpoint = policy_options.get('expected_manifest_digest')
            if checkpoint is not None:
                self.snapshot = owner.pool.owned_snapshot(checkpoint)
                wrapper={'metadata':self.snapshot.actor_metadata,'bank_count':len(self.snapshot.inference.banks)}
                if owner.pool.custody.evaluation_scope is not None:
                    binding=RuntimePolicyBinding(payload['payload']['model_id'],policy_options)
                    use,reservation=owner.pool.custody.owned_evaluation_use(owner.lease,request,self.snapshot,binding)
                    self._evaluation_use_reservation=reservation
                    wrapper['evaluation_use']=use
                elif self.snapshot.generation>request['generation']:
                    raise ContractError('training source snapshot exceeds actual control generation')
                try: self.payload={**payload,'owned':wrapper}
                except BaseException:
                    if self._evaluation_use_reservation is not None: self._evaluation_use_reservation.close()
                    raise
                self.owned_cache_identity=digest({'recipe':'owned-inference-cache-v1',
                    'policy_sha256':digest({'model_id':payload['payload']['model_id'],'policy_options':policy_options}),
                    'inference_sha256':checkpoint,'snapshot_identity':self.snapshot.identity})
        self.gate,self.completed,self.submission_finished = threading.Event(),threading.Event(),threading.Event()
        self.future: Future[Any] | None = None
        self.task: asyncio.Task[Any] | None = None
        self.tasks: tuple[asyncio.Task[Any],...] = ()
        self.coroutine: Any = None
        self.authorized = self.aborted = self.submission_started = False
        self.submission_error: BaseException | None = None
        self._abort_result = object()
        self._executor_threads: tuple[threading.Thread,...] | None = None
        self._executor_shutdown_observed = False

    def _exchange(self) -> Any:
        try:
            self.gate.wait()
            if not self.authorized:
                return self._abort_result
            assert self.owner.process.stdin is not None and self.owner.process.stdout is not None
            if self.owner._profile_io:
                from gf_simulator_client.framing import write_frame_profiled
                self.owner._last_io_write=write_frame_profiled(self.owner.process.stdin,self.payload)
            else:write_frame(self.owner.process.stdin,self.payload)
            leases = []
            if self.snapshot is not None:
                assert self.owner.owned_channel is not None
                for index, bank in enumerate(self.snapshot.inference.banks):
                    leases.append(self.owner.owned_channel.send(bank, purpose='model',
                        generation=self.snapshot.generation, identity=self.snapshot.identity+f':inference:{index}'))
            response = self.owner._read_reply(self.owner.process.stdout)
            for lease in leases:
                assert self.owner.owned_channel is not None
                self.owner.owned_channel.accept_completion(lease)
            if self.snapshot is not None:
                assert self.owned_cache_identity is not None
                self.owner._owned_cache_identities[self.owned_cache_identity]=self.snapshot.checkpoint
            return response
        finally:
            self.completed.set()

    def submit(self) -> None:
        self.submission_started = True
        try:
            self.future = self.executor.submit(self._exchange)
        except BaseException as error:
            self.submission_error = error
            raise
        finally:
            self.submission_finished.set()

    async def _wait_original(self) -> Any:
        assert self.future is not None
        while not self.future.done():
            await asyncio.sleep(.001)
        return self.future.result()

    def make_task(self) -> asyncio.Task[Any]:
        self.coroutine = self._wait_original()
        try:
            task = asyncio.create_task(self.coroutine)
            if not isinstance(task,asyncio.Task) or task.get_coro() is not self.coroutine:
                raise ContractError('resident setup requires original returned I/O Task')
        except BaseException:
            # A factory can transfer the exact coroutine and then raise. Recover
            # actual Tasks by that original object identity, never by name/flags.
            self.tasks = tuple(task for task in asyncio.all_tasks()
                if task.get_coro() is self.coroutine)
            if len(self.tasks)==1:
                self.task = self.tasks[0]
            if not self.tasks and inspect.getcoroutinestate(self.coroutine)==inspect.CORO_CREATED:
                self.coroutine.close()
            raise
        self.task,self.tasks = task,(task,)
        return task

    def authorize(self) -> None:
        if (self.aborted or self.gate.is_set() or self.future is None or self.task is None
                or self.task.done() or self.future.done() or time.monotonic()>=self.deadline):
            raise ContractError('resident framing setup cannot open without current original owners/deadline')
        self.authorized = True
        self.gate.set()

    def abort(self) -> None:
        self.aborted = True
        # Authorization is one-way: abort cannot pretend that already-started
        # framing disappeared. Its original Future/Task still must really retire.
        self.gate.set()

    def shutdown_unreturned_submission(self,deadline: float) -> None:
        if self.future is not None or self.authorized or not self.aborted:
            raise ContractError('resident no-Future cleanup requires original never-authorized setup')
        if self._executor_threads is None:
            self._executor_threads = tuple(self.executor._threads)
        # Abort gate FIRST. Cancel queued never-authorized work and positively
        # join the original executor's threads, including a submit that queued
        # the original callable and then raised before returning its Future.
        self.executor.shutdown(wait=False,cancel_futures=True)
        for thread in self._executor_threads:
            if not isinstance(thread,threading.Thread):
                raise ContractError('resident original executor thread identity changed')
            thread.join(timeout=max(0.,deadline-time.monotonic()))
            if thread.is_alive():
                raise ContractError('resident original setup executor shutdown/join unconfirmed')
        self._executor_shutdown_observed = True

    def require_retired(self) -> None:
        if self.submission_started and not self.submission_finished.is_set():
            raise ContractError('resident original submit observation remains pending')
        if any(not task.done() for task in self.tasks):
            raise ContractError('resident original setup Task has not retired')
        if self.coroutine is not None and not self.tasks and inspect.getcoroutinestate(self.coroutine)!=inspect.CORO_CLOSED:
            raise ContractError('resident original setup coroutine retirement unconfirmed')
        if self.future is None:
            if (self.authorized or not self.aborted or not self._executor_shutdown_observed
                    or self._executor_threads is None or any(thread.is_alive() for thread in self._executor_threads)):
                raise ContractError('resident original no-Future setup retirement unconfirmed')
        else:
            if not self.future.done() or not (self.completed.is_set() or self.future.cancelled()):
                raise ContractError('resident original gated executor Future has not retired')
            if not self.authorized and not self.future.cancelled() and self.future.result() is not self._abort_result:
                raise ContractError('resident original setup lacks its actual no-framing completion')
        if self._evaluation_use_reservation is not None:
            self._evaluation_use_reservation.close()


class ResidentPhysicalWorker(StructuredParticipantWorker):
    """One actual Popen/ledger owner and one explicitly tracked framed-I/O executor."""
    def __init__(self, pool: ResidentParticipantPool, viewer: int) -> None:
        self.pool, self.viewer = pool, viewer
        self._profile_io=pool.profile_io
        self.closed, self.initialized = False, False
        self.health = {}
        self.io_thread = self._close_task = None
        self.loan: ResidentParticipantWorker | None = None
        self.ready = False
        self._pending_diagnostic: Json | None = None
        self.diagnostic: Json | None = None
        self.protected_identity: str | None = None
        self._owned_cache_identities: dict[str,str] = {}
        self.used_collection = None
        self._operations: dict[asyncio.Task[Any], Future[Any]] = {}
        self._setups: set[_ExchangeSetup] = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='resident-framing')
        self.owned_channel = self._owned_inherited = None
        arguments = [sys.executable, '-E', '-m', 'gf_simulator_client.resident_bootstrap', str(viewer)]
        pass_fds: tuple[int, ...] = ()
        self.shared_endpoint=None
        if pool.shared:
            self.shared_endpoint,fd=pool.service.acquire_endpoint(self)
            pass_fds=(fd,)
            arguments.extend(('--shared',str(fd)))
        elif pool.owned_snapshots:
            from gf_common.ram_bank import RamBankChannel
            self.owned_channel, self._owned_inherited = RamBankChannel.pair()
            fd, invocation = self._owned_inherited.inherited_endpoint()
            pass_fds = (fd,)
            arguments.extend((str(fd), invocation))
        try:
            self._spawn(pool.ledger, pool.resident_bytes, arguments,
                {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '0',
                    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    **({'GF_BOTS_PERFORMANCE_TIMING': '1'} if pool.shared and pool.service.diagnostics is not None else {})},
                **({'pass_fds': pass_fds} if pass_fds else {}))
            if self.owned_channel is not None:
                self.owned_channel.bind_child(self.process)
                assert self._owned_inherited is not None
                self._owned_inherited.close()
        except BaseException:
            if self.owned_channel is not None and not self.owned_channel.outstanding:
                self.owned_channel.close()
                assert self._owned_inherited is not None
                self._owned_inherited.close()
            raise
        pool.custody.register(self.lease, self, viewer=viewer)

    async def exchange(self, payload: Json, *, timeout: float, closing: bool = False) -> Json:
        if self.closed and not closing or self.process.poll() is not None or self._operations or self._setups:
            raise ContractError('resident physical transport not exclusively ready')
        operation_deadline = self.pool.deadline if closing else self.pool.execution_deadline
        request_deadline = min(operation_deadline,time.monotonic()+timeout)
        if timeout<=0 or time.monotonic()>=request_deadline:
            raise TimeoutError('resident original operation deadline expired before I/O')
        routing=None
        if self.pool.shared and payload['request']['command']=='SELECT' and self.loan is not None and self.loan.model_id=='mlp':
            routing=self.pool.service.authorize(self.shared_endpoint,self,game=self.loan.game,participant=self.viewer,
                revision=payload['payload']['frame']['revision'],checkpoint=payload['request']['checkpoint_sha256'],deadline=request_deadline)
            payload={**payload,'shared_request':routing,'shared_deadline':request_deadline}
        setup = _ExchangeSetup(self,payload,request_deadline)
        self._setups.add(setup)
        try:
            self.pool.custody.track_setup(self.lease,setup)
            setup.submit()
            task = setup.make_task()
            assert setup.future is not None
            self._operations[task] = setup.future
            self.pool.custody.track_io(self.lease,task,setup.future)
            # Serve cancellation while the real framing gate is still closed.
            async with asyncio.timeout_at(request_deadline):
                await asyncio.sleep(0)
                self.pool.custody.authorize_setup(self.lease,setup)
                response = await asyncio.shield(task)
            self._retire_setup(setup)
            if type(response) is not dict or 'worker_error' in response:
                raise ContractError(f'resident worker error: {response}')
            self._pending_diagnostic = pool_diagnostic(response.pop('_diagnostic',None))
            if routing is not None:
                self.pool.service.complete(self.shared_endpoint,routing,completed=True)
                routing=None
            return response
        except BaseException as error:
            if routing is not None:
                try:
                    self.pool.service.complete(self.shared_endpoint,routing,completed=False)
                    routing=None
                except BaseException as cleanup:  # noqa: BLE001 - preserve original failure and continue actor reap.
                    error.add_note('shared request invalidation unconfirmed: '+type(cleanup).__name__)
            setup.abort()  # Always before waiting on original Future/executor shutdown.
            self.pool.custody.fail()
            cleanup_deadline = min(self.pool.deadline,time.monotonic()+2.)
            if not setup.authorized:
                try:
                    await self.retire_setups(cleanup_deadline)
                except BaseException as cleanup:  # noqa: BLE001 - preserve initiating setup/cancel failure
                    error.add_note('resident gated setup retirement: '+str(cleanup))
            if closing:
                raise
            try:
                await self.close(timeout=max(0.,cleanup_deadline-time.monotonic()))
            except BaseException as cleanup:  # noqa: BLE001 - preserve original after owned cleanup
                error.add_note('resident original physical cleanup: '+str(cleanup))
            raise

    def observe_diagnostic(self) -> None:
        # Caller has just validated the original reply against current custody.
        if self._pending_diagnostic is not None:
            self.diagnostic = self._pending_diagnostic
            self._pending_diagnostic = None
            self.pool.observe_diagnostic()

    def _retire_setup(self,setup: _ExchangeSetup) -> None:
        self.pool.custody.retire_setup(self.lease,setup)
        self._setups.remove(setup)
        for task in setup.tasks:
            self._operations.pop(task,None)
            if not task.cancelled():
                task.exception()

    async def retire_setups(self,deadline: float) -> None:
        for setup in tuple(self._setups):
            setup.abort()
            if setup.future is None:
                setup.shutdown_unreturned_submission(deadline)
            while ((setup.future is not None and not setup.future.done()) or any(not task.done() for task in setup.tasks)):
                if time.monotonic()>=deadline:
                    raise ContractError('resident original setup/Task/Future retirement unconfirmed')
                await asyncio.sleep(.001)
            self._retire_setup(setup)

    async def bootstrap(self, *, timeout: float) -> None:
        if self.ready:
            return
        request = self.pool.custody.bootstrap_request(self.lease)
        response = await self.exchange({'request': request}, timeout=timeout)
        self.pool.custody.observe_bootstrap_ready(self.lease, response)
        self.ready = True
        self.observe_diagnostic()

    async def control(self, command: str, *, checkpoint: str, timeout: float,
                      cache_identity: str | None = None, generation: int | None = None,
                      payload: Json | None = None) -> None:
        request = self.pool.custody.request(self.lease, command, checkpoint_sha256=checkpoint,
            cache_identity=cache_identity, generation=generation)
        response = await self.exchange({'request': request, **({} if payload is None else {'payload': payload})}, timeout=timeout)
        self.pool.custody.acknowledge(self.lease, response)
        if command=='EVICT' and cache_identity is not None:
            self._owned_cache_identities.pop(cache_identity,None)
        self.observe_diagnostic()

    async def _close_owned(self, timeout: float, started: float) -> None:
        if (self.lease.released and getattr(self.lease,'_proof',None) is not None and not self._operations and not self._setups
                and self.lease.key not in self.pool.ledger._resident_begin_imports
                and self.process is getattr(self.lease,'_process',object())
                and (self.process is None or self.process.poll() is not None)):
            self.executor.shutdown(wait=False,cancel_futures=False)
            return  # Original constructor cleanup already published its actual proof.
        close_error = None
        if (self.ready and self.loan is None and not self._operations and not self._setups and not self.pool.custody._failed
                and self.process.poll() is None and timeout > 0):
            try:
                request = self.pool.custody.close_request(self.lease)
                response = await self.exchange({'request':request},timeout=max(0.,started+timeout-time.monotonic()),closing=True)
                self.pool.custody.acknowledge_close(self.lease,response)
                self.observe_diagnostic()
                while self.process.poll() is None and time.monotonic()<started+timeout:
                    await asyncio.sleep(.001)
            except BaseException as error:  # noqa: BLE001 - reap remains authorized after bad CLOSE
                self.pool.custody.fail()
                close_error = error
        await super()._close_owned(max(0.,started+timeout-time.monotonic()),time.monotonic())
        # Actual process reap does not retire its original gated setup or I/O.
        await self.retire_setups(started+timeout)
        if self.owned_channel is not None and self.process.poll() is not None:
            for transfer in self.owned_channel.outstanding:
                transfer.release_after_reap(self.process)
            self.owned_channel.close()
            assert self._owned_inherited is not None
            self._owned_inherited.close()
        if self.lease.key in self.pool.ledger._resident_begin_imports:
            self.pool.custody.retire_failed_begin_import(self.lease)
        self.executor.shutdown(wait=False, cancel_futures=False)
        if self.pool.shared and self.shared_endpoint is not None and self.process.poll() is not None:
            self.pool.service.release_endpoint(self.shared_endpoint,self)
            self.shared_endpoint=None
        if close_error is not None:
            raise close_error


class ResidentParticipantPool:
    def __init__(self, custody: Any, *, resident_bytes: int, deadline: float,
                 acquire_cache: Any = None, release_cache: Any = None, progress: ProgressCallback | None = None,
                 owned_snapshots: bool = False, begin_import_bytes: int = 0, shared: bool = False, profile_io: bool = False) -> None:
        integer(resident_bytes, 0, custody.ledger.global_cuda_bytes)
        integer(begin_import_bytes, 0, custody.ledger.global_cuda_bytes)
        if begin_import_bytes and custody.evaluation_scope is not None:
            raise ContractError('BEGIN import admission is training-only')
        self.begin_import_bytes = begin_import_bytes
        self.shared=shared
        self.profile_io=profile_io
        self.service: Any=None
        self.custody, self.ledger = custody, custody.ledger
        self.progress = progress
        if type(owned_snapshots) is not bool:
            raise ContractError('explicit owned snapshot dispatch required')
        self.owned_snapshots = owned_snapshots
        self._owned_snapshots: dict[str, Any] = {}
        self._owned_snapshot_bindings: dict[str,Any] = {}
        self.starts = self.reuses = 0
        self.resident_bytes, self.deadline = resident_bytes, deadline
        self.execution_deadline = deadline
        self.acquire_cache, self.release_cache = acquire_cache, release_cache
        self.owners: list[ResidentPhysicalWorker] = []
        self.admission: Any = None
        self.authority: Any = None
        self.closed = False
        self._close_task: asyncio.Task[None] | None = None

    def bind_owned_snapshot(self, snapshot: Any) -> None:
        from gf_common.hpo_actor_snapshot import OwnedActorSnapshot
        if not self.owned_snapshots or type(snapshot) is not OwnedActorSnapshot:
            raise ContractError('actual common actor snapshot capability required for actor9')
        snapshot.require()
        checkpoint=snapshot.checkpoint
        existing=self._owned_snapshots.get(checkpoint)
        if existing is not None:
            if existing is not snapshot:
                raise ContractError('actor snapshot checkpoint owner changed')
            self._owned_snapshot_bindings[checkpoint].require(self,snapshot)
            return
        binding=snapshot.bind(self)
        self._owned_snapshots[checkpoint]=snapshot
        self._owned_snapshot_bindings[checkpoint]=binding

    def owned_snapshot(self, checkpoint: str) -> Any:
        snapshot=self._owned_snapshots.get(checkpoint)
        if snapshot is None:
            raise ContractError('actor9 checkpoint lacks actual bound snapshot')
        self._owned_snapshot_bindings[checkpoint].require(self,snapshot)
        return snapshot

    def unbind_owned_snapshot(self, snapshot: Any) -> None:
        checkpoint=snapshot.checkpoint
        if self._owned_snapshots.get(checkpoint) is not snapshot:
            raise ContractError('actor9 snapshot unbind differs from bound owner')
        if any(owner._operations or owner._setups or owner.owned_channel is not None
                and owner.owned_channel.outstanding for owner in self.owners):
            raise ContractError('actor9 transfer custody remains live')
        if any(owner.process.poll() is None and checkpoint in owner._owned_cache_identities.values() for owner in self.owners):
            raise ContractError('actor9 snapshot still has a live cached actor use')
        self._owned_snapshot_bindings[checkpoint].close(self)
        del self._owned_snapshot_bindings[checkpoint]
        del self._owned_snapshots[checkpoint]

    def observe_diagnostic(self) -> None:
        # Original custody retains every historical owner for actual retirement;
        # this optional current-pool view never traverses that history as rows.
        live = [owner for owner in self.owners if not owner.lease.released
            and owner.process is not None and owner.process.poll() is None]
        observe(self.progress, {'pool_starts':self.starts,'pool_reuses':self.reuses,
            'pool_live_workers':len(live),
            'pool_counter_scope':'invocation cumulative successful starts/reuses; rows are current live processes',
            'managed_cuda_reservation_bytes':self.ledger.cuda_bytes,
            'pool_workers':[{'owner_key':owner.lease.key,'pid':owner.process.pid,'viewer':owner.viewer,
                **(owner.diagnostic or {})} for owner in live if getattr(owner,'diagnostic',None) is not None],
            'pool_memory_scope':'last existing reset/sync sample; process lifetime maxima; not a simultaneous device peak'})

    def admit(self, plan: Any) -> None:
        self.admission = self.custody.admit_collection(plan, self.ledger)
        self.execution_deadline = min(self.deadline,plan.execution_deadline)
        self.authority = None

    def worker(self, viewer: int, *args: Any, **kwargs: Any) -> ResidentParticipantWorker:
        return ResidentParticipantWorker(self, viewer, *args, **kwargs)

    def acquire(self, logical: ResidentParticipantWorker, viewer: int) -> ResidentPhysicalWorker:
        integer(viewer, 1, 2)
        if self.closed or self.admission is None:
            raise ContractError('resident actual collection admission required')
        self.admission.require(self.admission.plan, self.ledger)
        partition = [owner for owner in self.owners if owner.viewer == viewer and not owner.lease.released]
        available = [owner for owner in partition if owner.loan is None and not owner.closed]
        # Prefer unused physical owners in this collection. Do not infer usage
        # from configured concurrency or assume every configured physical slot participates.
        available.sort(key=lambda owner: owner.lease.key in self.admission.used)
        if available:
            owner = available[0]
        else:
            if len(partition) >= self.ledger.max_workers // 2 or len(self.ledger.workers) >= self.ledger.max_workers:
                raise ContractError('resident viewer partition has no physical slot')
            try:
                owner = ResidentPhysicalWorker.__new__(ResidentPhysicalWorker)
                self.owners.append(owner)
                ResidentPhysicalWorker.__init__(owner,self,viewer)
                self.starts += 1
            except BaseException as error:
                partial = getattr(error, '_structured_startup_owner', None)
                if partial is not None and all(partial is not existing for existing in self.owners):
                    self.owners.append(partial)
                if not hasattr(owner,'lease'):
                    self.owners.remove(owner)
                    owner.executor.shutdown(wait=False,cancel_futures=False)
                self.custody.fail()
                raise
        if owner.loan is not None or owner._operations or owner._setups or owner.process.poll() is not None:
            raise ContractError('resident physical owner is not positively retired')
        owner.loan = logical
        if available:
            self.reuses += 1
        return owner

    async def retire_unused(self) -> None:
        self.admission.require(self.admission.plan,self.ledger)
        for owner in self.owners:
            if not owner.lease.released and owner.lease.key not in self.admission.used:
                if owner.loan is not None:
                    raise ContractError('unused resident owner has original logical lease')
                await owner.close(timeout=max(0., min(2., self.deadline-time.monotonic())))
                if self.release_cache is not None:
                    self.release_cache(owner, None)

    def retire_collection(self) -> Any:
        if self.shared:
            self.service.quiesce()
            data=self.service.telemetry
            self.custody.service_quiescence={'service_id':self.service.service_id,'pending':0,
                'publications':data['publications'],'requests':data['requests']}
        self.authority = self.custody.retire_collection(self.admission)
        return self.authority

    async def close(self) -> None:
        if self._close_task is None:
            self.closed = True
            self._close_task = asyncio.create_task(self._close_owned())
        first_cancel = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as error:
                first_cancel = first_cancel or error
            except BaseException:  # noqa: BLE001 - retrieve original shared task failure below
                break
        try:
            self._close_task.result()
        except BaseException as error:
            if first_cancel is not None:
                raise first_cancel from error
            raise
        if first_cancel is not None:
            raise first_cancel

    async def _close_owned(self) -> None:
        results = await asyncio.gather(*(owner.close(timeout=max(0., min(2., self.deadline-time.monotonic())))
            for owner in self.owners), return_exceptions=True)
        first = next((result for result in results if isinstance(result,BaseException)),None)
        for owner in self.owners:
            try:
                no_child = owner.process is None and owner.lease.pid is None and owner.lease._proof is not None and owner.lease._proof.outcome == 'never_started'
                if (not owner.lease.released or owner._operations or owner._setups
                        or not no_child and (owner.process is None or owner.process.poll() is None)):
                    raise ContractError('resident actual pool reap/I/O retirement unconfirmed')
                if self.release_cache is not None:
                    self.release_cache(owner,None)
            except BaseException as error:  # noqa: BLE001 - attempt all independently owned reaps/releases
                first = first or error
        if first is not None:
            self.custody.fail()
            raise first


class ResidentParticipantWorker(CardInstanceParticipantWorker):
    """Fresh protocol3 parent validation state borrowing one physical viewer owner."""
    def __init__(self, pool: ResidentParticipantPool, viewer: int, *args: Any, **kwargs: Any) -> None:
        if kwargs.get('ledger') is not pool.ledger:
            raise ContractError('resident logical participant requires SAME ledger')
        self.pool, self.viewer = pool, viewer
        self._performance_shared = pool.shared
        self.game=kwargs.pop("game",None)
        self._completed = False
        self._begin_import_request: Json | None = None
        super().__init__(*args, **kwargs)
        self.binding_value = RuntimePolicyBinding(self.model_id, self.initial.get('policy_options'))


    def _spawn(self, ledger: Any, residency: int, arguments: list[str], environment: dict[str, str],
               *, pass_fds: tuple[int, ...] = ()) -> None:
        self.owner = self.pool.acquire(self, self.viewer)
        self.process, self.lease = self.owner.process, self.owner.lease

    def _require_loan(self, *, closing: bool = False) -> None:
        if (self.owner.loan is not self or self.owner.closed or self.pool.closed
                or self.closed and not closing or self.lease.released):
            raise ContractError('resident logical owner is stale or retired')

    async def _roundtrip(self, payload: Json) -> Json:
        self._require_loan()
        try:
            pending_identity = None
            if not self.initialized:
                await self.owner.bootstrap(timeout=self.timeout)
                self.pool.custody.begin_admitted_lease(self.pool.admission, self.lease, binding=self.binding_value)
                if self.pool.acquire_cache is not None:
                    pending_identity = await self.pool.acquire_cache(self.owner, self.binding_value)
                else:
                    pending_identity = None
            request = self.pool.custody.episode_request(self.lease, 'SELECT' if self.initialized else 'BEGIN')
            if request['command'] == 'BEGIN' and self.pool.begin_import_bytes:
                self.pool.custody.reserve_begin_import(self.lease, request, self.pool.begin_import_bytes)
                self._begin_import_request = request
            response = await self.owner.exchange({'request': request, 'payload': payload}, timeout=self.timeout)
            result = self.pool.custody.acknowledge_episode(self.lease, response)
            if not self.initialized and pending_identity is not None:
                self.owner.protected_identity = pending_identity
            return result
        except BaseException as error:  # noqa: BLE001 - actual original lease owns failed cleanup
            self.pool.custody.fail()
            await self._abort(error)

    async def _initialize_owned(self) -> None:
        await super()._initialize_owned()
        # The complete existing protocol3 ready/model/binding/timing validation
        # has returned. A matching outer BEGIN alone cannot publish load facts.
        if self._begin_import_request is not None:
            self.pool.custody.retire_begin_import(self.lease, self._begin_import_request)
            self._begin_import_request = None
        self.owner.observe_diagnostic()

    async def _select_frame_owned(self, frame: Any, delta: Any) -> Any:
        result = await super()._select_frame_owned(frame,delta)
        # Original candidate/selection/policy/timing validation and reservation
        # reply retirement precede diagnostic publication as well.
        self.owner.observe_diagnostic()
        if self._profile_io:self.last_timings.update({**getattr(self.owner,"_last_io_write",{}),**getattr(self.owner,"_last_io_read",{})})
        return result

    def mark_completed(self) -> None:
        self._require_loan()
        self._completed = True

    async def _close_owned(self, timeout: float, started: float) -> None:
        owner = self.owner
        try:
            if not self._completed or not self.initialized or self.lease.decisions or timeout <= 0:
                self.pool.custody.fail()
                await owner.close(timeout=timeout)
                return
            self._require_loan(closing=True)
            row = self.pool.custody._row(self.lease)
            for command in ('END', 'PARK'):
                await owner.control(command, checkpoint=row.checkpoint,
                    timeout=min(self.timeout, max(0., started+timeout-time.monotonic())))
            if time.monotonic() > started + timeout:
                raise TimeoutError('resident reset exceeded original logical close deadline')
            self._parent_bank = self._last_frame = None
            self.last_timings = {}
            self.inference_elapsed_ms = None
        except BaseException as error:
            self.pool.custody.fail()
            try:
                await owner.close(timeout=max(0., started+timeout-time.monotonic()))
            except BaseException as cleanup:  # noqa: BLE001 - preserve original after owned cleanup
                error.add_note('resident logical cleanup: '+str(cleanup))
            raise
        finally:
            if owner.loan is self:
                owner.loan = None
