"""Original logical artifact bytes, without filesystem or Source authority."""

from __future__ import annotations

import math
from types import TracebackType
from typing import Any, Protocol, Self


class ArtifactReader(Protocol):
    def read(self, name: str) -> bytes: ...
    def names(self) -> tuple[str, ...]: ...
    def contains(self, name: str) -> bool: ...


class ArtifactReservation(Protocol):
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None: ...


class ArtifactAdmission(Protocol):
    def reserve(self, size: int, *, wait: bool = False) -> ArtifactReservation: ...


class ArtifactSink(Protocol):
    @property
    def budget(self) -> ArtifactAdmission: ...
    def hold_graph(self, key: str, size: int) -> None: ...
    def write(self, name: str, data: bytes) -> None: ...
    def append(self, name: str, data: bytes) -> None: ...
    def rename(self, old: str, new: str) -> None: ...
    def discard(self, name: str) -> None: ...
    def snapshot(self) -> ArtifactReader: ...


def json_bound(value: Any, *, depth: int = 0) -> int:
    """Allocation-free conservative encoded-size estimate for admission."""
    if depth > 100:
        raise ValueError("JSON nesting exceeds admission")
    if value is None or type(value) is bool:
        return 5
    if type(value) is int:
        return max(1, value.bit_length() + 2)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON")
        return 32
    if isinstance(value, str):
        return len(value) * 6 + 2
    if isinstance(value, (list, tuple)):
        return 2 + sum(json_bound(v, depth=depth + 1) + 1 for v in value)
    if isinstance(value, dict):
        if any(type(k) not in (str, int, float, bool, type(None)) for k in value):
            raise ValueError("invalid JSON object key")
        return 2 + sum(
            json_bound(k, depth=depth + 1) + json_bound(v, depth=depth + 1) + 2
            for k, v in value.items()
        )
    raise ValueError("invalid JSON value")


class ArtifactPhaseAdmission:
    """One overlapping phase counter backed by the SAME artifact account."""
    def __init__(self, source: ArtifactAdmission, limit: int) -> None:
        import threading
        if type(limit) is not int or limit<1:
            raise ValueError('positive artifact phase capacity required')
        self.source,self.limit=source,limit
        self.used=self.peak=0
        self._lock=threading.RLock()

    def reserve(self, size: int, *, wait: bool = False) -> _PhaseReservation:
        if wait:
            raise ValueError('artifact phase admission does not wait')
        with self._lock:
            if type(size) is not int or size<0 or self.used+size>self.limit:
                raise ValueError('shared artifact phase budget exhausted')
            actual=self.source.reserve(size)
            try: reservation=_PhaseReservation(self,actual,size)
            except BaseException:
                actual.close()
                raise
            self.used+=size
            self.peak=max(self.peak,self.used)
            return reservation


class _PhaseReservation:
    def __init__(self, phase: ArtifactPhaseAdmission, actual: ArtifactReservation, size: int) -> None:
        self.phase,self.actual,self.size=phase,actual,size
        self.closed=False

    def close(self) -> None:
        with self.phase._lock:
            if self.closed: return
            self.actual.close()
            self.phase.used-=self.size
            self.closed=True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()
