"""Managed storage accounting and deadline-bounded binary pipe transport."""

from __future__ import annotations

import os
import selectors
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from gf_common.group_pack import MAX_JSON, Json, canonical_json, parse_json

WRITER_BASE = 128 * 1024 * 1024
SOURCE_BASE = 64 * 1024 * 1024


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
    def __init__(self, limit: int, *, base: int = 0) -> None:
        if type(limit) is not int or limit < base:
            raise ValueError("storage budget cannot admit base reservation")
        self.limit = limit
        self.used = base
        self.peak = base
        self._condition = threading.Condition()
        self._stopped = False

    def acquire(self, size: int, deadline: float, *, wait: bool = False) -> None:
        if size < 0 or size > self.limit:
            raise ValueError("reservation exceeds capacity")
        with self._condition:
            while True:
                remaining(deadline)
                if self._stopped:
                    raise RuntimeError("storage admission stopped")
                if self.used + size <= self.limit:
                    self.used += size
                    self.peak = max(self.used, self.peak)
                    return
                if not wait:
                    raise ValueError("storage budget exhausted")
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
