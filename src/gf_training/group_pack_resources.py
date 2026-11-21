"""Managed storage accounting and deadline-bounded binary pipe transport."""

from __future__ import annotations

import os
import selectors
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self

from gf_common.artifacts import budget_refusal
from gf_common.artifacts import json_bound as json_bound  # noqa: PLC0414 -- compatibility pure re-export.
from gf_common.group_pack import MAX_JSON, Json, StorageSettings, canonical_json, parse_json
from gf_common.group_pack import SOURCE_BASE as SOURCE_BASE  # noqa: PLC0414 -- compatibility pure re-export.
from gf_common.group_pack import WRITER_BASE as WRITER_BASE  # noqa: PLC0414 -- compatibility pure re-export.


class CleanupError(RuntimeError):
    """Failure which retains the concrete cleanup owner for a later retry."""

    def __init__(self, message: str, owner: object) -> None:
        super().__init__(message)
        self.cleanup_owner = owner


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("storage operation deadline expired")
    return value


class ByteBudget:
    def __init__(self, limit: int, *, base: int = 0, name: str = 'storage',
                 phase: str = 'storage') -> None:
        if type(limit) is not int:
            raise ValueError("storage budget cannot admit base reservation")
        if limit < base:
            raise ValueError(budget_refusal('storage budget cannot admit base reservation',
                name=name,phase=phase,used=0,request=base,limit=limit))
        self.limit = limit
        self.name,self.phase=name,phase
        self.used = base
        self.peak = base
        self._condition = threading.Condition()
        self._stopped = False

    def acquire(self, size: int, deadline: float, *, wait: bool = False) -> None:
        if type(size) is not int or size < 0:
            raise ValueError("reservation exceeds capacity")
        with self._condition:
            if size > self.limit:
                raise ValueError(budget_refusal('reservation exceeds capacity',
                    name=self.name,phase=self.phase,used=self.used,request=size,limit=self.limit))
            while True:
                remaining(deadline)
                if self._stopped:
                    raise RuntimeError("storage admission stopped")
                if self.used + size <= self.limit:
                    self.used += size
                    self.peak = max(self.used, self.peak)
                    return
                if not wait:
                    raise ValueError(budget_refusal('storage budget exhausted',
                        name=self.name,phase=self.phase,used=self.used,request=size,limit=self.limit))
                self._condition.wait(remaining(deadline))

    def release(self, size: int) -> None:
        with self._condition:
            if size < 0 or size > self.used:
                raise RuntimeError("invalid storage reservation release")
            self.used -= size
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


class StorageBudget(ByteBudget):
    """One thread-safe account for live Python, Writer and measured RAM bytes."""

    def __init__(self, settings: StorageSettings, deadline: float, *, name: str = 'storage',
                 phase: str = 'storage') -> None:
        super().__init__(settings.budget_bytes,name=name,phase=phase)
        self.settings, self.deadline = settings, deadline
        self.ram_bytes = 0

    @property
    def peak_bytes(self) -> int:
        with self._condition:
            return self.peak

    def reserve(self, size: int, *, wait: bool = False) -> StorageReservation:
        self.acquire(size, self.deadline, wait=wait)
        return StorageReservation(self, size)

    def measure_ram(self, size: int) -> None:
        if type(size) is not int or size < 0:
            raise ValueError("invalid RAM measurement")
        with self._condition:
            self.used += size - self.ram_bytes
            self.ram_bytes = size
            self.peak = max(self.peak, self.used)
            self._condition.notify_all()
            if self.used > self.limit:
                self._stopped = True
                raise ValueError(budget_refusal('storage budget exceeded by RAM files',
                    name=self.name,phase=self.phase,used=self.used,request=size,limit=self.limit))


class StorageReservation:
    def __init__(self, budget: StorageBudget, size: int) -> None:
        self.budget, self.size = budget, size
        self.closed = False

    def resize(self, size: int) -> None:
        with self.budget._condition:
            if self.closed or type(size) is not int or size < 0:
                raise ValueError("invalid storage reservation resize")
            delta = size - self.size
            if delta > 0:
                self.budget.acquire(delta, self.budget.deadline)
            elif delta < 0:
                self.budget.release(-delta)
            self.size = size

    def close(self) -> None:
        with self.budget._condition:
            if not self.closed:
                self.budget.release(self.size)
                self.size, self.closed = 0, True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class BorrowedBudget(ByteBudget):
    """Local bookkeeping backed by the SAME account; stop never stops its lender."""

    def __init__(self, parent: StorageBudget, *, base: int = 0, limit: int | None = None,
                 name: str = 'borrowed-storage', phase: str | None = None) -> None:
        super().__init__(parent.limit if limit is None else min(limit, parent.limit),
                         name=name,phase=parent.phase if phase is None else phase)
        self.parent = parent
        # The shared condition releases the SAME lock while waiting. A local
        # lock held across parent admission would prevent another borrower
        # operation from releasing bytes and deadlock a saturated queue.
        self._condition = parent._condition
        self.acquire(base, parent.deadline)

    def acquire(self, size: int, deadline: float, *, wait: bool = False) -> None:
        with self._condition:
            if type(size) is not int or size < 0:
                raise ValueError("reservation exceeds borrowed capacity")
            if size > self.limit:
                raise ValueError(budget_refusal('reservation exceeds borrowed capacity',
                    name=self.name,phase=self.phase,used=self.used,request=size,limit=self.limit))
            while self.used + size > self.limit:
                if self._stopped:
                    raise RuntimeError("storage admission stopped")
                if not wait:
                    raise ValueError(budget_refusal('borrowed storage budget exhausted',
                        name=self.name,phase=self.phase,used=self.used,request=size,limit=self.limit))
                self._condition.wait(remaining(deadline))
            if self._stopped:
                raise RuntimeError("storage admission stopped")
            self.parent.acquire(size, deadline, wait=wait)
            self.used += size
            self.peak = max(self.used, self.peak)

    def release(self, size: int) -> None:
        with self._condition:
            if size < 0 or size > self.used:
                raise RuntimeError("invalid borrowed reservation release")
            self.parent.release(size)
            self.used -= size
            self._condition.notify_all()


@contextmanager
def reserve(budget: ByteBudget, size: int, deadline: float) -> Iterator[None]:
    budget.acquire(size, deadline)
    try:
        yield
    finally:
        budget.release(size)


def _ready(fd: int, event: int, deadline: float) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(fd, event)
        if not selector.select(remaining(deadline)):
            raise TimeoutError("storage pipe deadline expired")


def write_bytes(fd: int, data: bytes, deadline: float) -> None:
    view = memoryview(data)
    cursor = 0
    while cursor < len(view):
        _ready(fd, selectors.EVENT_WRITE, deadline)
        try:
            count = os.write(fd, view[cursor : cursor + 64 * 1024])
        except BlockingIOError:
            continue
        if count <= 0:
            raise RuntimeError("storage pipe closed")
        cursor += count


def read_bytes(fd: int, size: int, deadline: float) -> bytes:
    chunks = []
    cursor = 0
    while cursor < size:
        _ready(fd, selectors.EVENT_READ, deadline)
        try:
            chunk = os.read(fd, min(size - cursor, 64 * 1024))
        except BlockingIOError:
            continue
        if not chunk:
            raise RuntimeError("truncated storage frame")
        chunks.append(chunk)
        cursor += len(chunk)
    return b"".join(chunks)


def send_frame(fd: int, metadata: Json, payload: bytes, deadline: float) -> None:
    data = canonical_json(metadata, limit=MAX_JSON)
    write_bytes(fd, struct.pack("<QQ", len(data), len(payload)), deadline)
    write_bytes(fd, data, deadline)
    write_bytes(fd, payload, deadline)


def receive_header(fd: int, deadline: float, max_payload: int) -> tuple[Json, int]:
    metadata_size, payload_size = struct.unpack("<QQ", read_bytes(fd, 16, deadline))
    if metadata_size > MAX_JSON or payload_size > max_payload:
        raise ValueError("storage frame exceeds bound")
    metadata = parse_json(read_bytes(fd, metadata_size, deadline), limit=MAX_JSON)
    if not isinstance(metadata, dict):
        raise TypeError("invalid storage frame metadata")
    return metadata, payload_size


def retain_cleanup(owner: Json, error: BaseException) -> None:
    """Keep an already-created nested cleanup owner reachable from the outer owner."""
    nested = error.__dict__.get("cleanup_owner")
    if (
        nested is not None
        and nested is not owner
        and all(child is not nested for child in owner._cleanup_children)
    ):
        owner._cleanup_children.append(nested)
    error.__dict__["cleanup_owner"] = owner


def close_resources(owner: Json, resources: list[Json]) -> None:
    """Attempt every independent cleanup while retaining the first error and owner."""
    first: BaseException | None = None
    seen: set[int] = set()
    for resource in [*resources, *owner._cleanup_children]:
        if resource is None or resource is owner or id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            resource.close()
        except BaseException as error:  # noqa: BLE001 -- all owners remain reachable
            retain_cleanup(owner, error)
            if first is None:
                first = error
            else:
                first.add_note(f"additional cleanup failure ({type(error).__name__})")
    if first is not None:
        raise first
    owner._cleanup_children.clear()


def close_raw_descriptor(owner: Json, attribute: str) -> None:
    """Consume an active number before its one terminal close attempt."""
    from gf_common.owned_completion import _close_owned_descriptor

    fd = getattr(owner, attribute)
    setattr(owner, attribute, -1)
    if fd >= 0:
        try:
            _close_owned_descriptor(owner, fd, None, attribute)
        except BaseException as error:
            retain_cleanup(owner, error)
            raise


def require_certain_releases(owner: Json) -> None:
    if owner._uncertain_releases:
        raise CleanupError("descriptor release remains uncertain; attempted numbers retired", owner)


class OwnedDescriptor:
    """An individual raw handle, including explicit terminal-release uncertainty."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._uncertain_releases: list[Json] = []
        self._cleanup_children: list[Json] = []

    def close(self) -> None:
        close_raw_descriptor(self, "fd")
        require_certain_releases(self)
