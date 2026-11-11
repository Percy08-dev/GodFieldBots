"""Concrete original-byte callbacks; these objects never issue Source authority.

The native owner admits ``external_bytes`` before entering FFI. Callbacks use
only the SAME retained original pool and never call a session or parent lock.
"""
from __future__ import annotations

import hashlib
import threading
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Self

from gf_common.artifacts import json_bound
from gf_common.domain import ContractError
from gf_common.group_pack import canonical_json, safe_name
from gf_training.group_pack_game_source import PackedSourceOwner
from gf_training.native_execution import NativeExecution

FORMAT = 'python-original-bytes-v1'
METADATA_LIMIT = 8 * 1024**2


class _OriginalByteLease:
    """Preowned staged loan; attempted exits/releases require actual evidence."""
    def __init__(self, provider: OriginalArtifactProvider, owner: PackedSourceOwner, generation: int) -> None:
        self._provider, self._owner, self._generation = provider, owner, generation
        self._context: Any = None
        self._loan: Any = None
        self._data: bytes | None = None
        self._charge = 0
        self._context_entered = self._exit_attempted = self._context_done = False
        self._budget_attempted = False
        self._budget_before = 0
        self._closed = False
        assert owner.artifacts is not None
        self._loan_baseline = owner.artifacts._loans

    def data(self) -> bytes:
        with self._provider._callback():
            if self._closed or self._provider._leases.get(self._generation) is not self:
                raise ContractError('original byte lease retired')
            if self._loan is not None:
                self._loan._check()
            else:
                self._owner.finalize()
            if type(self._data) is not bytes:
                raise ContractError('original byte lease not completely admitted')
            return self._data

    def _actual_loan(self) -> Any:
        if self._loan is None and self._context is not None:
            frame = getattr(getattr(self._context, 'gen', None), 'gi_frame', None)
            candidate = None if frame is None else frame.f_locals.get('loan')
            from gf_training.group_pack_source import GroupLoan
            if type(candidate) is GroupLoan and candidate._source is self._owner.artifacts:
                self._loan = candidate
        return self._loan

    def close(self) -> None:
        if self._closed:
            return
        try:
            loan = self._actual_loan()
            source = self._owner.artifacts
            assert source is not None
            if self._context_entered and not self._context_done:
                if loan is None and getattr(getattr(self._context, 'gen', None), 'gi_frame', None) is None and source._loans == self._loan_baseline:
                    self._context_done = True
                # A previous failed __exit__ is not retried after its generator
                # disappeared. The actual loan's inactive state/count is evidence.
                if self._context_done or self._exit_attempted and loan is not None and not loan._active and source._loans == self._loan_baseline:
                    self._context_done = True
                else:
                    if self._exit_attempted:
                        frame = getattr(getattr(self._context, 'gen', None), 'gi_frame', None)
                        if frame is None or loan is None or not loan._active:
                            raise ContractError('original context retirement remains uncertain')
                    self._exit_attempted = True
                    try:
                        self._context.__exit__(None, None, None)
                    finally:
                        if loan is not None and not loan._active and source._loans == self._loan_baseline:
                            self._context_done = True
                    if not self._context_done:
                        raise ContractError('actual original group loan retirement unconfirmed')
            self._data = None
            if self._charge:
                budget = self._provider.pool.budget
                if self._budget_attempted:
                    if budget.used == self._budget_before - self._charge:
                        self._charge = 0
                    elif budget.used != self._budget_before:
                        raise ContractError('original copy account retirement uncertain')
                if self._charge:
                    self._budget_before = budget.used
                    self._budget_attempted = True
                    amount = self._charge
                    try:
                        budget.release(amount)
                    finally:
                        if budget.used == self._budget_before - amount:
                            self._charge = 0
            self._provider._leases.pop(self._generation)
            self._closed = True
        except BaseException as error:
            self._provider._remember(error)
            raise


class OriginalArtifactProvider:
    """Fixed versioned adapter over actual Python-owned packed originals.

    Closing retires adapter metadata, never independently owned Source FDs. The
    enclosing owner closes those only after its FFI executor has actually joined.
    """
    def __init__(self, owners: Sequence[PackedSourceOwner], *, pool: Any = None,
                 shared_controls: Any = None) -> None:
        if unicodedata.unidata_version != '16.0.0':
            raise ContractError('packed native names require Unicode16.0.0')
        if any(type(owner) is not PackedSourceOwner for owner in owners):
            raise ContractError('concrete retained packed original owners required')
        self._empty_controls: Any = None
        if not owners:
            from gf_training.group_pack_pool import PackedArtifactPool
            from gf_training.structured_lineage import _ChainScan
            if (type(shared_controls) is not _ChainScan or type(pool) is not PackedArtifactPool
                    or shared_controls._packed_pool is not pool or not shared_controls._owns_packed_pool
                    or shared_controls._closed or shared_controls._pool_control_uncertain
                    or shared_controls._pool_control_charge < shared_controls.retained
                    or shared_controls._packed_owners or shared_controls._packed_borrows
                    or pool.sources or pool.active is not None):
                raise ContractError('zero originals require the actual admitted generation0 scan/pool')
            state = shared_controls._packed_state
            if (state is None or type(state.get('schema_version')) is not int or state['schema_version'] != 4
                    or type(state.get('updates')) is not int or state['updates'] != 0
                    or shared_controls.root/'run.json' not in shared_controls.files
                    or shared_controls.root/'generations/000000.json' not in shared_controls.files
                    or shared_controls.verified_digest(shared_controls.root/'generations/000000.json') != state['generation_sha256']):
                raise ContractError('actual run4 parent/generation0 binding required for empty registry')
            pool.check()
            self.pool = pool
            self._empty_controls = shared_controls
        else:
            self.pool = owners[0].pool
            if pool is not None and pool is not self.pool:
                raise ContractError('explicit provider pool differs from actual originals')
        if any(owner.pool is not self.pool for owner in owners):
            raise ContractError('SAME original pool required for all native collections')
        self._owners = {str(owner.root): owner for owner in owners}
        if len(self._owners) != len(owners):
            raise ContractError('duplicate original provider root')
        self._leases: dict[int, _OriginalByteLease] = {}
        self._generation = 0
        self._entered = self._closed = False
        self._failure: BaseException | None = None
        self._descriptions: dict[str, bytes] = {}
        self._metadata_charge = 0
        self._query_charge = 0
        self._query_release_attempted = False
        self._query_before = 0
        self._acquisition_states = {'metadata': 'unattempted', 'query': 'unattempted'}
        self._metadata_release_attempted = False
        self._metadata_before = 0
        self._executor_owner: Any = None
        self._fenced = False
        self._cache_uncertain: set[Any] = set()
        # Admit graphs, canonical strings and retained bytes BEFORE constructing
        # them. The already verified index remains in its original pool account.
        estimate = 65536
        maximum_group = 0
        for owner in owners:
            owner.check()
            assert owner.artifacts is not None
            estimate += 64 * len(str(owner.root)) + 65536
            for name, entry in owner._entries.items():
                size = entry.get('length', entry.get('bytes'))
                if size > owner.settings.max_source_file_mib * 1024**2:
                    raise ContractError('bounded original provider file required')
                estimate += 16384 + 128 * len(name)
            for group in owner.artifacts._groups.values():
                # zstd frame bound plus decoded/cache/copy overlap. The physical
                # reader independently accounts its actual allocation in SAME pool.
                maximum_group = max(maximum_group, 8 * group['raw_bytes'] + 1024**2)
                estimate += 2048 + 8 * (json_bound(group['group_id']) + json_bound(group['record']))
            for episode in owner.artifacts._index['episodes']:
                estimate += 2048 + 8 * sum(json_bound(episode[key])
                    for key in ('episode_id', 'group_id', 'game_id', 'record'))
        try:
            self._acquire_owned('metadata', estimate)
            roots = []
            for root, owner in sorted(self._owners.items()):
                storage, custody = owner.storage, owner.custody
                roots.append({'root': root, 'storage': storage, 'custody': custody})
                files = [{'name': name, 'bytes': entry.get('length', entry.get('bytes')),
                    'sha256': entry['sha256'], 'identity': owner.member_identity(name)}
                    for name, entry in sorted(owner._entries.items())]
                assert owner.artifacts is not None
                original_index = owner.artifacts._index
                index = {'groups': [{'group_id': row['group_id'], 'record': row['record']}
                    for row in sorted(original_index['groups'], key=lambda row: row['group_id'])],
                    'episodes': [{key: row[key] for key in ('episode_id', 'group_id', 'game_id', 'record')}
                        for row in sorted(original_index['episodes'], key=lambda row: row['episode_id'])]}
                description = canonical_json({'format': FORMAT, 'root': root,
                    'storage': storage, 'custody': custody, 'files': files, 'index': index}, limit=METADATA_LIMIT)
                self._descriptions[root] = description
            self._registry = canonical_json({'format': FORMAT, 'roots': roots}, limit=METADATA_LIMIT)
            self.registry_sha256 = hashlib.sha256(self._registry).hexdigest()
            # Actual metadata-only queries on the SAME admitted originals. No
            # Pack payload or second connection/account is read/constructed.
            self._retire_io()
            query_charge = 65536 + 1024 * sum(len(o.artifacts._groups) for o in owners if o.artifacts is not None)
            maximum_group = 0
            query_failure: BaseException | None = None
            try:
                self._acquire_owned('query', query_charge)
                for owner in owners:
                    source = owner.artifacts
                    assert source is not None
                    owner.finalize()
                    self.pool.activate(source)
                    rows = source._originals.db.execute('SELECT group_id,length(pack) FROM groups ORDER BY group_id').fetchall()
                    if [row[0] for row in rows] != sorted(source._groups):
                        raise ContractError('original compressed group index changed')
                    for name, compressed in rows:
                        maximum_group = max(maximum_group, 4 * source._groups[name]['raw_bytes'] + 2 * compressed)
                    owner.finalize()
                    self._retire_io()
            except BaseException as error:  # noqa: BLE001 -- acquisition itself is inside the owned stage.
                query_failure = error
                self._remember(error)  # The initiating failure owns cleanup BEFORE any refund can fail.
            for cleanup in (self._retire_io, lambda: self._release_owned('query')):
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001 -- independently retire actual query account and IO.
                    if query_failure is not None:
                        query_failure.add_note('original query cleanup: ' + str(error))
                    query_failure = query_failure or error
            if query_failure is not None:
                raise query_failure
            controls = max((2 * row['bytes'] for owner in owners if owner.artifacts is not None
                for row in owner.artifacts._index['controls']), default=0)
            self.working_bytes = maximum_group + controls + 1024**2
            self.baseline_bytes = self.pool.budget.used
            self.external_bytes = self.baseline_bytes + self.working_bytes
        except BaseException as error:
            self._remember(error)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- retain actual partial constructor ownership.
                error.add_note('original provider constructor cleanup: ' + str(cleanup))
            raise

    def _acquire_owned(self, kind: str, amount: int) -> None:
        """Exclusive SAME-account evidence, recorded before acquire effects."""
        self._acquisition_states[kind] = 'attempting'
        before = self.pool.budget.used
        try:
            self.pool.budget.acquire(amount, self.pool.deadline)
        finally:
            current = self.pool.budget.used
            if current == before + amount:
                setattr(self, '_' + kind + '_charge', amount)
                self._acquisition_states[kind] = 'admitted'
            elif current == before:
                self._acquisition_states[kind] = 'no-effect'
            else:
                self._acquisition_states[kind] = 'uncertain'
        if self._acquisition_states[kind] != 'admitted':
            raise ContractError('original ' + kind + ' admission lacks exact actual evidence')

    def _release_owned(self, kind: str) -> None:
        if self._acquisition_states[kind] == 'uncertain':
            raise ContractError('original ' + kind + ' admission remains uncertain')
        amount = getattr(self, '_' + kind + '_charge')
        if not amount:
            return
        attempted_name, before_name = '_' + kind + '_release_attempted', '_' + kind + '_before'
        attempted, before = getattr(self, attempted_name), getattr(self, before_name)
        if attempted:
            if self.pool.budget.used == before - amount:
                setattr(self, '_' + kind + '_charge', 0)
                return
            if self.pool.budget.used != before:
                raise ContractError('original ' + kind + ' release remains uncertain')
        setattr(self, before_name, self.pool.budget.used)
        setattr(self, attempted_name, True)
        try:
            self.pool.budget.release(amount)
        except BaseException as error:
            self._remember(error)
            raise
        finally:
            if self.pool.budget.used == getattr(self, before_name) - amount:
                setattr(self, '_' + kind + '_charge', 0)

    def _remember(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        vars(error)['_native_original_provider_cleanup_owner'] = self
        vars(self._failure)['_native_original_provider_cleanup_owner'] = self

    def _check(self) -> None:
        if self._fenced:
            raise ContractError('original provider callbacks fenced for cleanup')
        if self._executor_owner is not None and self._executor_owner._running and threading.get_ident() != self._executor_owner._worker:
            raise ContractError('caller cannot access active native original IO')
        if self._failure is not None:
            raise ContractError('original provider is permanently cleanup-only') from self._failure
        if self._closed:
            raise ContractError('original provider is closed')
        self.pool.check()

    @contextmanager
    def _callback(self) -> Iterator[None]:
        self._check()
        if self._entered:
            error = ContractError('original provider callback reentry')
            self._remember(error)
            raise error
        self._entered = True
        try:
            yield
            self._check()
        except BaseException as error:
            self._remember(error)
            raise
        finally:
            self._entered = False

    def _owner(self, root: str) -> PackedSourceOwner:
        if type(root) is not str or root not in self._owners:
            raise ContractError('unregistered original provider root')
        return self._owners[root]

    def registry(self) -> bytes:
        with self._callback():
            return self._registry

    def describe(self, root: str) -> bytes:
        with self._callback():
            self._owner(root)
            return self._descriptions[root]

    def recheck(self, root: str) -> None:
        with self._callback():
            self._owner(root).finalize()

    def borrow(self, root: str, name: str) -> _OriginalByteLease:
        with self._callback():
            owner = self._owner(root)
            safe_name(name)
            if name not in owner._entries:
                raise ContractError('unregistered original provider member')
            if self._leases:
                raise ContractError('one original byte lease at a time')
            source = owner.artifacts
            assert source is not None
            self._generation += 1
            lease = _OriginalByteLease(self, owner, self._generation)
            self._leases[self._generation] = lease  # Before ANY borrow effects.
            try:
                if name in source._member_groups:
                    lease._context = owner.borrow_group(source._member_groups[name])
                    lease._context_entered = True
                    lease._loan = lease._context.__enter__()
                    raw = lease._loan.read(name)
                else:
                    amount = 2 * owner._entries[name]['bytes']
                    before = self.pool.budget.used
                    try:
                        self.pool.budget.acquire(amount, self.pool.deadline)
                    finally:
                        if self.pool.budget.used == before + amount:
                            lease._charge = amount
                    self.pool.activate(source)
                    raw = source._originals.db.execute('SELECT data FROM controls WHERE name=?', (name,)).fetchone()[0]
                lease._data = raw
                entry = owner._entries[name]
                if (type(raw) is not bytes or len(raw) != entry.get('length', entry.get('bytes'))
                        or hashlib.sha256(raw).hexdigest() != entry['sha256']):
                    raise ContractError('original provider member content/type changed')
                return lease
            except BaseException as error:
                try:
                    lease.close()
                except BaseException as cleanup:  # noqa: BLE001 -- preowned actual stages remain reachable.
                    error.add_note('original loan cleanup: ' + str(cleanup))
                raise

    def _retire_io(self) -> None:
        """Exclusive creator-thread cleanup; no reads, hashes or admission."""
        primary: BaseException | None = None
        for lease in tuple(self._leases.values()):
            try:
                lease.close()
            except BaseException as error:  # noqa: BLE001 -- close SQLite independently of loan failure.
                primary = primary or error
        source = self.pool.active
        if source is not None:
            # Close the SAME actual connection even if cache/loan release failed.
            if source._originals.db is not None:
                connection = source._originals.db
                try:
                    connection.close()
                except BaseException as error:  # noqa: BLE001 -- retain the actual same-object retry owner.
                    primary = primary or error
                else:
                    source._originals.db = None
            if source._loans:
                primary = primary or ContractError('original cache still has actual group loans')
            else:
                amount = source._cache_charge
                if amount and source not in self._cache_uncertain:
                    budget = source._budget
                    before, parent_before = budget.used, self.pool.budget.used
                    try:
                        budget.release(amount)
                    except BaseException as error:  # noqa: BLE001 -- only exact exclusive-account evidence proves release.
                        primary = primary or error
                    if budget.used == before - amount and self.pool.budget.used == parent_before - amount:
                        source._cache_charge = 0
                    elif budget.used != before or self.pool.budget.used != parent_before:
                        self._cache_uncertain.add(source)
                        primary = primary or ContractError('original cache account retirement uncertain')
                if not source._cache_charge:
                    source._cache.clear()
                    if source._group_id is not None:
                        source._group_id = None
                        source._generation += 1
                if source._cache_charge:
                    primary = primary or ContractError('original cache charge retirement unconfirmed')
            if source._originals.db is None and not source._loans and not source._cache_charge:
                self.pool.active = None
        if primary is not None:
            self._remember(primary)
            raise primary

    def close(self) -> None:
        if self._closed:
            return
        if self._executor_owner is not None and (not self._executor_owner._closed or self._executor_owner._hold is not None):
            raise ContractError('actual native executor must retire before provider release')
        if self._entered:
            error = ContractError('original provider close during callback reentry')
            self._remember(error)
            raise error
        if self._leases:
            error = ContractError('original provider has active or uncertain byte leases')
            vars(error)['_native_original_provider_cleanup_owner'] = self
            raise error
        if self._executor_owner is None:
            self._retire_io()
        # Query admission must also retire when acquire raised after effects.
        # Unknown acquisition evidence retains BOTH stages and this actual owner.
        self._release_owned('query')
        self._descriptions.clear()
        self._registry = b''
        self._release_owned('metadata')
        self._closed = True

    def __enter__(self) -> Self:
        self._check()
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if not isinstance(value, BaseException):
                raise
            value.add_note('native original provider cleanup: ' + str(cleanup))
            vars(value)['_native_original_provider_cleanup_owner'] = self


class _PackedNativeExecution(NativeExecution):
    """One creator-aware IO interval around the existing concrete FFI executor."""
    def __init__(self, cancel_handle: Any, provider: OriginalArtifactProvider) -> None:
        from concurrent.futures import ThreadPoolExecutor
        if not callable(getattr(cancel_handle, 'cancel', None)) or provider._executor_owner is not None:
            raise ContractError('independent cancel handle and unbound actual provider required')
        self.provider = provider
        self._providers = [provider]
        self._final_provider: OriginalArtifactProvider | None = None
        self._closed = False
        self._active: Any = None
        self._cancel_handle = cancel_handle
        self._caller = threading.get_ident()
        self._worker: int | None = None
        self._creator: str | None = None
        self._running = False
        self._failure: BaseException | None = None
        self._native: Any = None
        self._native_retired = True
        self._executor: Any = None
        self._hold: Any = None
        self._executor_initialized = False
        self._submission_attempted = False
        provider._executor_owner = self  # Reachable BEFORE hold/executor effects.
        try:
            self._hold = provider.pool.hold_close(self)
            # Preown the actual object before __init__ can partially fail.
            self._executor = ThreadPoolExecutor.__new__(ThreadPoolExecutor)
            ThreadPoolExecutor.__init__(self._executor, max_workers=1, thread_name_prefix='gf-packed-source')
            self._executor_initialized = True
        except BaseException as error:
            if self._hold is None:
                self._hold = next((hold for hold in provider.pool._close_holds if hold.owner is self), None)
            self._remember(error)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- partial construction keeps actual hold/executor reachable.
                error.add_note('packed executor constructor cleanup: ' + str(cleanup))
            raise

    def attach_final_provider(self, provider: OriginalArtifactProvider) -> None:
        """Attach the parent's concrete final registry to this SAME executor."""
        self.prepare()
        if (type(provider) is not OriginalArtifactProvider or provider.pool is not self.provider.pool
                or provider._executor_owner is not None or self._final_provider is not None
                or len(provider._owners) != 1 or set(provider._owners) & set(self.provider._owners)):
            raise ContractError('one distinct final provider on SAME actual executor/pool required')
        provider._check()
        self._final_provider = provider
        self._providers.append(provider)
        provider._executor_owner = self

    def _retire_io(self) -> None:
        first: BaseException | None = None
        # Retire the actual active source's byte loan before another provider
        # can evict that SAME pool cache/connection.
        active = self.provider.pool.active
        ordered = sorted(self._providers,
            key=lambda provider: not any(owner.artifacts is active for owner in provider._owners.values()))
        for provider in ordered:
            try:
                provider._retire_io()
            except BaseException as error:  # noqa: BLE001 - retain every actual provider/loan/creator owner.
                if first is None:
                    first = error
                elif first is not error:
                    first.add_note('final provider IO retirement: ' + str(error))
        if first is not None:
            raise first

    def bind_native(self, native: Any) -> None:
        if self._native is not None:
            raise ContractError('actual native session already bound')
        self._native, self._native_retired = native, False

    def call(self, function: Any, *arguments: Any) -> Any:
        try:
            return super().call(function, *arguments)
        except BaseException as error:
            self._remember(error)
            raise

    async def acall(self, function: Any, *arguments: Any) -> Any:
        try:
            return await super().acall(function, *arguments)
        except BaseException as error:
            self._remember(error)
            raise

    def _remember(self, error: BaseException) -> None:
        self._failure = self._failure or error
        for provider in self._providers:
            provider._remember(error)
        vars(self._failure)['_structured_source_executor_cleanup_owner'] = self

    def prepare(self) -> None:
        if threading.get_ident() != self._caller or self._running or self._active is not None or self._closed:
            raise ContractError('native original caller interval unavailable')
        if self._failure is not None:
            raise ContractError('packed executor is permanently cleanup-only') from self._failure
        if self._creator == 'worker':
            raise ContractError('worker original IO retirement remains uncertain')
        for hold in self.provider.pool._close_holds:
            other = hold.owner
            if other is not self and (getattr(other, '_running', False) or getattr(other, '_active', None) is not None or getattr(other, '_creator', None) == 'worker'):
                raise ContractError('another actual pool executor owns active or uncertain IO')
        self._creator = 'caller'
        try:
            self._retire_io()
            self._creator = None
            for provider in self._providers:
                provider._check()
        except BaseException as error:
            self._remember(error)
            raise

    def _submit(self, function: Any, *arguments: Any) -> Any:
        self.prepare()
        self._running = True
        def run() -> Any:
            self._worker = threading.get_ident()
            self._creator = 'worker'  # BEFORE constructor/callback/connection work.
            primary: BaseException | None = None
            result: Any = None
            try:
                result = function(*arguments)
                if getattr(function, '__self__', None) is self._native and getattr(function, '__name__', '') in ('prepare_commit', 'finalize_read', 'close'):
                    self._native_retired = True
            except BaseException as error:  # noqa: BLE001 -- IO retirement precedes every Future completion.
                primary = error
            try:
                self._retire_io()
                self._creator = None
            except BaseException as cleanup:  # noqa: BLE001 -- SAME creating worker remains reachable.
                if primary is not None:
                    primary.add_note('native worker IO retirement: ' + str(cleanup))
                primary = primary or cleanup
            self._running = False
            if primary is not None:
                self._remember(primary)
                raise primary
            return result
        try:
            self._submission_attempted = True
            return super()._submit(run)
        except BaseException as error:
            self._running = False
            self._remember(error)
            raise

    def close_final_validation(self) -> None:
        """Fixed cleanup-only slot operation, retaining this executor and hold."""
        if threading.get_ident() != self._caller or self._closed or self._native is None:
            raise ContractError('SAME live native parent executor/caller required for final cleanup')
        if self._failure is None and self._active is None:
            self.call(self._native.close_final_validation)
            return
        if self._active is not None:
            first = self._failure or ContractError('final cleanup joined an active admitted operation')
            self._remember(first)
            self._cancel_for(first)
            self._join_sync(self._active, first)
            if not self._active.done():
                raise first
            self._active = None
        for provider in self._providers:
            provider._fenced = True
        # Slot.recheck starts with Control.check, before ANY original IO.
        self.cancel()
        if self._creator != 'worker':
            self._creator = 'caller'
            try:
                self._retire_io()
            except BaseException as error:
                self._remember(error)
                raise
            self._creator = None
        def close_slot() -> None:
            current = threading.get_ident()
            if self._creator == 'worker' and self._worker != current:
                raise ContractError('actual final-slot creator worker unavailable')
            self._worker = current
            self._creator = 'worker'
            primary: BaseException | None = None
            try:
                self._native.close_final_validation()
            except BaseException as error:  # noqa: BLE001 - IO retirement remains independently reportable.
                primary = error
            try:
                self._retire_io()
                self._creator = None
            except BaseException as error:  # noqa: BLE001 - retain SAME creator/loan/connection on failure.
                if primary is not None and primary is not error:
                    self._note(primary, 'final-slot creator IO cleanup', error)
                primary = primary or error
            self._running = False
            if primary is not None:
                raise primary
        self._running = True
        try:
            # Bypass normal admission ONLY for this fixed storage-free operation.
            future = NativeExecution._submit(self, close_slot)
            try:
                future.result()
            except KeyboardInterrupt as cancellation:
                self._cancel_for(cancellation)
                self._join_sync(future, cancellation)
                raise
            finally:
                if future.done():
                    self._active = None
        except BaseException as error:
            self._remember(error)
            raise

    def close(self) -> None:
        if self._closed:
            if self._hold is not None:
                self._hold.retire(self)
                self._hold = None
            if self._failure is not None:
                raise self._failure
            return
        if self._active is not None:
            error = ContractError('packed executor close during active work')
            self._cancel_for(error)
            self._join_sync(self._active, error)
            self._active = None
            self._remember(error)
        # Healthy native close keeps normal final checks. Failed/expired cleanup
        # first fences callbacks AND cancels Control before ANY native recheck.
        if self._failure is None:
            try:
                self.prepare()
                if not self._native_retired:
                    self.call(self._native.close)
            except BaseException as error:
                self._remember(error)
                raise
        else:
            for provider in self._providers:
                provider._fenced = True
            self.cancel()
            if self._creator != 'worker':
                self._creator = 'caller'
                try:
                    self._retire_io()
                    self._creator = None
                except BaseException as cleanup:  # noqa: BLE001 -- do not transfer caller-owned uncertainty.
                    self._failure.add_note('caller cleanup: ' + str(cleanup))
            def cleanup_worker() -> None:
                current = threading.get_ident()
                if self._creator == 'worker' and self._worker != current:
                    raise ContractError('actual creator worker unavailable for SQLite cleanup')
                self._worker = current
                primary: BaseException | None = None
                try:
                    if not self._native_retired:
                        self._native.close()
                        self._native_retired = True
                except BaseException as error:  # noqa: BLE001 -- SQLite close is independent of native closure.
                    primary = error
                try:
                    if self._creator != 'caller':
                        self._creator = 'worker'
                        self._retire_io()
                        self._creator = None
                except BaseException as error:  # noqa: BLE001 -- preserve native failure and actual IO owner.
                    primary = primary or error
                if primary is not None:
                    raise primary
            needs_worker = not self._native_retired or self._creator == 'worker'
            future = None if self._executor is None or not needs_worker else self._executor.submit(cleanup_worker)
            self._active = future
            try:
                if future is not None:
                    future.result()
            except BaseException as cleanup:  # noqa: BLE001 -- no shutdown with uncertain native/worker IO.
                self._failure.add_note('worker cleanup: ' + str(cleanup))
            finally:
                self._active = None
            if not self._native_retired or self._creator is not None or any(provider._leases for provider in self._providers) or self.provider.pool.active is not None:
                raise self._failure
        no_effect_initialization = (not self._executor_initialized and not self._submission_attempted
            and self._worker is None and self._active is None and self._native is None
            and self._native_retired and self._creator is None and not self._running
            and not any(provider._leases for provider in self._providers) and self.provider.pool.active is None)
        if self._executor is None or no_effect_initialization:
            if self._executor is not None:
                fields = vars(self._executor)
                threads = fields.get('_threads')
                if threads:
                    raise ContractError('partial executor has actual worker effects; cannot discard')
                # CPython initialization starts no workers. A fully populated
                # shutdown state can be retired normally; a partial object has
                # no submit/session effects and must never receive these APIs.
                if all(name in fields for name in ('_shutdown_lock', '_shutdown', '_work_queue', '_threads')):
                    self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            self._closed = True
        else:
            self._shutdown()
        if self._hold is not None:
            try:
                self._hold.retire(self)
                self._hold = None
            except BaseException as error:  # noqa: BLE001 -- retain the actual hold and initiating cleanup failure.
                self._remember(error)
                assert self._failure is not None
                raise self._failure
        if self._failure is not None:
            raise self._failure

    async def aclose(self) -> None:
        import asyncio
        cancellation: BaseException | None = None
        if self._active is not None:
            future = self._active
            completed = asyncio.wrap_future(future)
            try:
                await asyncio.shield(completed)
            except asyncio.CancelledError as error:
                cancellation = error
                self._cancel_for(error)
                await self._join_async(completed, future, error)
                self._remember(error)
            finally:
                self._active = None
        try:
            self.close()
        except BaseException as error:
            if cancellation is not None:
                self._note(cancellation, 'packed cleanup', error)
                raise cancellation
            raise
