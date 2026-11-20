"""One checked SQLite connection/group account across retained original collections."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gf_common.group_pack import SOURCE_BASE, StorageSettings
from gf_training.group_pack_resources import StorageBudget, close_resources, remaining
from gf_training.group_pack_source import PackedSource


class PoolCloseHold:
    """Actual terminal-close capability; its owning executor proves retirement."""
    def __init__(self, pool: PackedArtifactPool, owner: object) -> None:
        self.pool, self.owner = pool, owner
        self.amount = 0
        self._attempted = False
        self._before = 0
        self.retired = False
        self._construction_certain = False

    def retire(self, owner: object) -> None:
        if owner is not self.owner:
            raise RuntimeError('only the actual close-hold owner may retire it')
        if self.retired:
            return
        if self.amount:
            if self._attempted and self.pool.budget.used == self._before - self.amount:
                self.amount = 0
            else:
                if self._attempted and self.pool.budget.used != self._before:
                    raise RuntimeError('close-hold account retirement uncertain')
                self._before, self._attempted = self.pool.budget.used, True
                amount = self.amount
                try:
                    self.pool.budget.release(amount)
                finally:
                    if self.pool.budget.used == self._before - amount:
                        self.amount = 0
        try:
            self.pool._close_holds.remove(self)
        finally:
            if self not in self.pool._close_holds:
                self.retired = True


class PackedArtifactPool:
    def __init__(self, *, max_owned_bytes: int, deadline: float,
                 storage_limit: int | None = None) -> None:
        self.deadline = deadline
        self.sources: list[PackedSource] = []
        self.active: PackedSource | None = None
        self.closed = False
        self._cleanup_children: list[Any] = []
        self._close_holds: list[PoolCloseHold] = []
        self._base_charge = 0
        self._base_uncertain = False
        self._base_release_attempted = False
        self._base_before = 0
        # The actual budget is reachable before its initializer/admission effects.
        self.budget = StorageBudget.__new__(StorageBudget)
        try:
            StorageBudget.__init__(self.budget, StorageSettings(), deadline)
            self.budget.limit = min(max_owned_bytes, storage_limit or max_owned_bytes)
            before = self.budget.used
            try:
                self.budget.acquire(SOURCE_BASE, deadline)
            finally:
                if self.budget.used == before + SOURCE_BASE:
                    self._base_charge = SOURCE_BASE
                elif self.budget.used != before:
                    self._base_uncertain = True
        except BaseException as error:
            error.__dict__['cleanup_owner'] = self
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- retain actual partial budget/pool and primary error.
                error.add_note('original pool initialization cleanup: ' + str(cleanup))
            raise

    def hold_close(self, owner: object) -> PoolCloseHold:
        self.check()
        hold = PoolCloseHold(self, owner)
        self._close_holds.append(hold)  # Reachable before admission can partially succeed.
        before = self.budget.used
        try:
            self.budget.acquire(4096, self.deadline)
        except BaseException as error:
            if self.budget.used == before + 4096:
                hold.amount = 4096
            error.__dict__['cleanup_owner'] = owner
            raise
        hold.amount = 4096
        hold._construction_certain = True
        return hold

    def require_terminal_close(self) -> None:
        if self._close_holds:
            failure = RuntimeError('actual native executor close hold still owns originals/account')
            failure.__dict__['cleanup_owner'] = self._close_holds[0].owner
            raise failure

    def _require_borrowed_verifier_close(self, hold: PoolCloseHold, owner: object) -> None:
        """Exempt one positively registered outer hold; never retire/drop it."""
        if (type(hold) is not PoolCloseHold or hold.pool is not self or hold.owner is not owner
                or len(self._close_holds) != 1 or self._close_holds[0] is not hold
                or hold.retired or hold._attempted or not hold._construction_certain
                or hold.amount != 4096 or self.closed):
            raise RuntimeError('actual exclusive certain outer verifier hold/owner required')

    def check(self) -> None:
        remaining(self.deadline)
        if self.closed:
            raise RuntimeError('packed original pool closed')

    def deactivate(self) -> None:
        self.check()
        self._retire_active()

    def _retire_active(self) -> None:
        """Terminal eviction/close does no read, admission or deadline extension."""
        source = self.active
        if source is None:
            return
        source._retire_group()
        if source._originals.db is not None:
            # Do not lose the actual connection on a close failure.
            source._originals.db.close()
            source._originals.db = None
        self.active = None

    def before_open(self) -> None:
        self.deactivate()

    def register(self, source: PackedSource) -> None:
        self.check()
        if source._originals.db is not None:
            source._originals.db.close()
            source._originals.db = None
        self.sources.append(source)
        if source in self._cleanup_children:
            self._cleanup_children.remove(source)

    def activate(self, source: PackedSource) -> None:
        self.check()
        if source not in self.sources:
            raise RuntimeError('foreign original Source pool')
        if self.active is source:
            return
        self.deactivate()
        source.recheck()
        original = source._originals.database_file
        assert original is not None
        original.recheck()
        connection = sqlite3.connect(f'file:/proc/self/fd/{original.fd}?mode=ro',
            uri=True, timeout=min(5., remaining(self.deadline)))
        source._originals.db = connection
        self.active = source  # Own the actual connection before PRAGMAs can fail.
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA cache_size=-32768')
        source.recheck()

    def forget(self, source: PackedSource) -> None:
        if self.active is source:
            self._retire_active()
        self.sources.remove(source)

    def recheck(self) -> None:
        self.check()
        for source in self.sources:
            source.recheck()

    def close(self) -> None:
        if self.closed:
            return
        self.require_terminal_close()
        # Each Source attempts its independently owned originals, even if a
        # different Source cannot retire. No work-deadline gate applies here.
        close_resources(self, list(self.sources))
        self.sources.clear()
        if self._base_uncertain:
            raise RuntimeError('original pool base admission remains uncertain')
        if self._base_charge:
            if self._base_release_attempted and self.budget.used == self._base_before - self._base_charge:
                self._base_charge = 0
            else:
                if self._base_release_attempted and self.budget.used != self._base_before:
                    raise RuntimeError('original pool base release remains uncertain')
                amount = self._base_charge
                self._base_before, self._base_release_attempted = self.budget.used, True
                try:
                    self.budget.release(amount)
                finally:
                    if self.budget.used == self._base_before - amount:
                        self._base_charge = 0
        self.closed = True


class OriginalArtifactReader:
    """Concrete byte-reader adapter; it does not issue Source/update permission."""
    def __init__(self, source: PackedSource, read: Callable[[Path], bytes],
                 charge: Callable[[int], None], root: Path) -> None:
        if type(source) is not PackedSource or source._pool is None:
            raise TypeError('actual pooled original bytes required')
        self._source, self._read, self._charge, self._root = source, read, charge, root

    def read(self, name: str) -> bytes:
        return self._read(self._root / name)

    def names(self) -> tuple[str, ...]:
        return self._source.names()

    def contains(self, name: str) -> bool:
        return name in self.names()

    def admit_validation(self, size: int) -> None:
        self._charge(size)
