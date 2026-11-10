"""One checked SQLite connection/group account across retained original collections."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gf_common.group_pack import SOURCE_BASE, StorageSettings
from gf_training.group_pack_resources import StorageBudget, close_resources, remaining
from gf_training.group_pack_source import PackedSource


class PackedArtifactPool:
    def __init__(self, *, max_owned_bytes: int, deadline: float,
                 storage_limit: int | None = None) -> None:
        self.deadline = deadline
        # StorageSettings is only used to establish the SAME managed byte account;
        # no directories, Writer or catalog are opened by this read-only pool.
        settings = StorageSettings()
        self.budget = StorageBudget(settings, deadline)
        self.budget.limit = min(max_owned_bytes, storage_limit or max_owned_bytes)
        self.budget.acquire(SOURCE_BASE, deadline)
        self.sources: list[PackedSource] = []
        self.active: PackedSource | None = None
        self.closed = False
        self._cleanup_children: list[Any] = []

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
        # Each Source attempts its independently owned originals, even if a
        # different Source cannot retire. No work-deadline gate applies here.
        close_resources(self, list(self.sources))
        self.sources.clear()
        self.budget.release(SOURCE_BASE)
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
