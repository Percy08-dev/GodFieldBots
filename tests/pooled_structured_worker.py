"""Explicit pytest CPU collection loans; real processes and leases stay physical."""
from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from pathlib import Path

from gf_common.domain import ContractError, closed, integer
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from tests.pooled_structured_bootstrap import check_envelope, envelope

CURRENT_POOL = ContextVar('pytest_cpu_participant_pool', default=None)
CURRENT_VIEWER = ContextVar('pytest_cpu_participant_viewer', default=None)


class PhysicalWorker(StructuredParticipantWorker):
    def __init__(self, *args, viewer, **kwargs):
        self.viewer, self.epoch, self.loan = viewer, 0, None
        self.poisoned = False
        self.resets = self.loads = self.reuses = 0
        super().__init__(*args, **kwargs)

    def _spawn(self, ledger, residency, arguments, environment):
        arguments = [*arguments[:-1], 'tests.pooled_structured_bootstrap', str(self.viewer)]
        super()._spawn(ledger, residency, arguments, environment)

    async def exchange(self, payload):
        try:
            result = await super()._roundtrip(payload)
            # The queue result can precede actual thread retirement. A loan must
            # never hand off until this concrete thread has really returned.
            if self.io_thread is not None:
                self.io_thread.join(timeout=.1)
                if self.io_thread.is_alive():
                    raise RuntimeError('pytest physical I/O thread retirement unconfirmed')
            return result
        except BaseException:
            self.poisoned = True
            raise


class CpuParticipantPool:
    def __init__(self, ledger, *, cleanup_deadline=float('inf')):
        self.ledger = ledger
        self.cleanup_deadline = cleanup_deadline
        self.owners = []
        self.logical_count = 0
        self._close_task = None
        self.closed = False
        self.peak_summed_rss_kib = self.peak_summed_pss_kib = 0

    def acquire(self, logical, residency, args, kwargs):
        viewer = CURRENT_VIEWER.get()
        integer(viewer, 1, 2)
        if self.closed or kwargs['ledger'] is not self.ledger:
            raise ContractError('pytest pool requires open SAME ledger')
        if logical.model_id not in ('random', 'heuristic', 'mlp', 'gru', 'transformer'):
            raise ContractError('pytest CPU pool model unsupported')
        options = logical.initial['policy_options']
        if options is not None and (options['inference']['device'] != 'cpu' or options['cuda_visible_devices'] != ''):
            raise ContractError('pytest pool requires CPU with empty CUDA visibility')
        owners = [owner for owner in self.owners if owner.viewer == viewer]
        owner = next((owner for owner in owners if owner.loan is None and not owner.closed and not owner.poisoned), None)
        if owner is None:
            if len(owners) >= 2 or len(self.owners) >= 4:
                raise ContractError('pytest viewer pool has no nonoverlapping physical slot')
            try:
                owner = PhysicalWorker(*args, viewer=viewer, **kwargs)
            except BaseException as error:
                startup = getattr(error, '_structured_startup_owner', None)
                if startup is not None:
                    startup.poisoned = True
                    self.owners.append(startup)
                raise
            self.owners.append(owner)
        if owner.lease.resident_bytes != residency or owner.lease.decisions or (owner.io_thread is not None and owner.io_thread.is_alive()):
            raise ContractError('pytest owner residency or handoff state mismatch')
        if owner.lease._cleanup_owner is not owner or owner.process.poll() is not None:
            raise ContractError('pytest physical ownership lost')
        owner.loan = logical
        owner.epoch += 1
        self.logical_count += 1
        self.observe_memory()
        return owner

    def observe_memory(self):
        rss = pss = 0
        for owner in self.owners:
            if owner.process is not None and owner.process.poll() is None:
                try:
                    values = {}
                    for line in Path(f'/proc/{owner.process.pid}/smaps_rollup').read_text().splitlines():
                        if line.startswith(('Rss:', 'Pss:')):
                            key, number, _ = line.split()
                            values[key] = int(number)
                    rss += values['Rss:']
                    pss += values['Pss:']
                except (OSError, KeyError, ValueError):
                    pass  # Optional OS observation, never a resource-admission input.
        self.peak_summed_rss_kib = max(self.peak_summed_rss_kib, rss)
        self.peak_summed_pss_kib = max(self.peak_summed_pss_kib, pss)

    async def close(self):
        if self._close_task is None:
            self.closed = True
            self._close_task = asyncio.create_task(self._close_owned(time.monotonic()))
        first_cancel = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as error:
                first_cancel = first_cancel or error
            except BaseException:  # noqa: BLE001 - retrieve the shared close task failure below
                break
        try:
            result = self._close_task.result()
        except BaseException as error:
            if first_cancel is not None:
                raise first_cancel from error
            raise
        if first_cancel is not None:
            raise first_cancel
        return result

    async def _close_owned(self, started):
        self.observe_memory()
        for owner in self.owners:
            if not owner.closed and (owner.process is None or owner.process.poll() is not None):
                owner.poisoned = True  # An idle crash is not intentional physical termination.
        unhealthy = any(owner.poisoned or owner.loan is not None for owner in self.owners)
        deadline = min(self.cleanup_deadline, started + 2.)
        unhealthy |= started >= self.cleanup_deadline
        results = await asyncio.gather(*(owner.close(timeout=max(0., deadline - time.monotonic()))
                                        for owner in self.owners), return_exceptions=True)
        failures = [value for value in results if isinstance(value, BaseException)]
        if failures:
            raise failures[0]
        if self.ledger.workers or any((owner.process is not None and owner.process.poll() is None) or not owner.lease.released
                or (owner.io_thread is not None and owner.io_thread.is_alive()) for owner in self.owners):
            raise RuntimeError('pytest collection physical cleanup unconfirmed')
        unhealthy |= time.monotonic() > deadline or any(any(owner.health.values()) for owner in self.owners)
        if unhealthy:
            raise RuntimeError('pytest collection contains poisoned loan, expired deadline or unhealthy cleanup')
        return {'mode': 'pytest-cpu-participant-v1', 'physical_workers': len(self.owners),
                'logical_participants': self.logical_count, 'resets': sum(owner.resets for owner in self.owners),
                'model_loads': sum(owner.loads for owner in self.owners),
                'model_reuses': sum(owner.reuses for owner in self.owners), 'closed': True,
                'peak_observed_summed_rss_kib': self.peak_summed_rss_kib,
                'peak_observed_summed_pss_kib': self.peak_summed_pss_kib,
                'memory_note': 'sampled Linux /proc; summed RSS double-counts shared pages; zero if unavailable',
                'owners': [{'viewer': owner.viewer, 'pid': owner.process.pid, 'resets': owner.resets,
                            'model_loads': owner.loads, 'model_reuses': owner.reuses} for owner in self.owners]}


class PooledStructuredParticipantWorker(StructuredParticipantWorker):
    """Fresh parent validator/search/bank state, borrowing only physical transport."""

    def __init__(self, *args, **kwargs):
        self.pool = CURRENT_POOL.get()
        if self.pool is None:
            raise ContractError('explicit pytest collection pool context required')
        self._construction = (args, kwargs)
        self._poisoned = False
        super().__init__(*args, **kwargs)

    def _spawn(self, ledger, residency, arguments, environment):
        self.owner = self.pool.acquire(self, residency, *self._construction)
        self.epoch, self.viewer = self.owner.epoch, self.owner.viewer
        self.process, self.lease = self.owner.process, self.owner.lease

    def _require_current_loan(self):
        owner = self.owner
        if (self.closed or self.pool.closed or owner.closed or owner.poisoned
                or owner.loan is not self or owner.epoch != self.epoch
                or self.lease.released or self.lease._cleanup_owner is not owner
                or self.process is not owner.process or self.process.poll() is not None):
            raise RuntimeError('pytest logical worker has no current live loan')

    async def initialize(self):
        self._require_current_loan()
        await super().initialize()

    async def select_frame(self, frame, delta):
        self._require_current_loan()
        return await super().select_frame(frame, delta)

    async def _roundtrip(self, payload):
        # A retired logical handle must not mutate or poison a later borrower.
        self._require_current_loan()
        try:
            self.owner.timeout = self.timeout
            operation = 'begin' if not self.initialized else 'select'
            response = await self.owner.exchange(envelope(operation, self.viewer, self.epoch, payload=payload))
            self.io_thread = self.owner.io_thread
            check_envelope(response, 'reply', self.viewer, self.epoch, ('payload',))
            reply = response['payload']
            if type(reply) is not dict or 'worker_error' in reply:
                raise ContractError(f'pytest original worker error: {reply}')
            return reply
        except BaseException as error:  # noqa: BLE001 - preserve failure after physical reap
            await self._abort(error)

    async def _abort(self, original):
        self._poisoned = self.owner.poisoned = True
        await super()._abort(original)

    async def _close_owned(self, timeout, started):
        owner = self.owner
        deadline = min(self.pool.cleanup_deadline, started + max(0., timeout))
        try:
            if self._poisoned or owner.poisoned or not self.initialized or self.lease.decisions or timeout <= 0:
                owner.poisoned = True
                await owner.close(timeout=max(0., deadline - time.monotonic()))
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('pytest logical close deadline expired before reset')
            owner.timeout = min(self.timeout, max(.001, deadline - time.monotonic()))
            response = await owner.exchange(envelope('end', self.viewer, self.epoch))
            self.io_thread = owner.io_thread
            check_envelope(response, 'reset', self.viewer, self.epoch, ('loads', 'reuses'))
            closed(response, {'pytest_pool', 'operation', 'viewer', 'epoch', 'loads', 'reuses'})
            integer(response['loads'], 1, owner.epoch)
            integer(response['reuses'], 0, owner.epoch)
            if response['loads'] + response['reuses'] != owner.epoch or self.lease.decisions:
                raise ContractError('pytest reset/cache accounting mismatch')
            self.pool.observe_memory()
            # An early queued ACK does not extend the original close allowance:
            # include actual I/O-thread retirement and final diagnostic sampling.
            if time.monotonic() > deadline:
                raise TimeoutError('pytest reset exceeded original logical close deadline')
            owner.loads, owner.reuses = response['loads'], response['reuses']
            owner.resets += 1
        except BaseException as original:
            owner.poisoned = True
            try:
                await owner.close(timeout=max(0., deadline - time.monotonic()))
            except BaseException as cleanup_error:  # noqa: BLE001 - reset/protocol failure remains primary
                if cleanup_error is not original:
                    original.add_note(f'pytest physical cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}')
                self.health.setdefault('abort_cleanup_error', type(cleanup_error).__name__)
            raise
        finally:
            if owner.loan is self and owner.epoch == self.epoch:
                owner.loan = None
