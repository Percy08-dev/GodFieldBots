"""Bounded chunk artifacts and Journal6 outputs; no physical Source authority."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from gf_common.candidate_columns import column_layout, encode_candidate_columns_v2
from gf_common.domain import ContractError, Json
from gf_common.group_pack import safe_name
from gf_training.candidate_storage import candidate_basename
from gf_training.card_instance_journal import CardInstanceJournal
from gf_training.group_pack_resources import StorageBudget, StorageReservation
from gf_training.group_pack_resources import json_bound as _json_bound


class MemoryArtifactSnapshot:
    """Immutable original bytes with explicit lifetime and parse graph ownership."""

    def __init__(self, budget: StorageBudget, files: dict[str, bytes],
                 reservations: list[StorageReservation]) -> None:
        self.budget, self._files, self._reservations = budget, files, reservations
        self._validation: list[StorageReservation] = []
        self._closed = False

    def read(self, name: str) -> bytes:
        safe_name(name)
        if self._closed:
            raise RuntimeError("snapshot retired")
        return self._files[name]

    def names(self) -> tuple[str, ...]:
        if self._closed:
            raise RuntimeError("snapshot retired")
        return tuple(sorted(self._files))

    def contains(self, name: str) -> bool:
        safe_name(name)
        if self._closed:
            raise RuntimeError("snapshot retired")
        return name in self._files

    def admit_validation(self, size: int) -> None:
        if self._closed:
            raise RuntimeError("snapshot retired")
        self._validation.append(self.budget.reserve(size))

    def retire_validation(self) -> None:
        for reservation in self._validation:
            reservation.close()
        self._validation.clear()

    def close(self) -> None:
        self._files.clear()
        self.retire_validation()
        for reservation in self._reservations:
            reservation.close()
        self._reservations.clear()
        self._closed = True


class MemoryGroupArtifacts:
    def __init__(self, budget: StorageBudget) -> None:
        self.budget = budget
        self._chunks: dict[str, list[bytes]] = {}
        self._reservations: list[StorageReservation] = []
        self._frozen = False
        self._lock = threading.RLock()
        self._graphs: dict[str, StorageReservation] = {}

    def _check(self, name: str) -> None:
        safe_name(name)
        if self._frozen:
            raise RuntimeError("memory artifact store is frozen")

    def retain(self, size: int) -> None:
        with self._lock:
            self._reservations.append(self.budget.reserve(size))

    def _adopt(self, reservation: StorageReservation) -> None:
        with self._lock:
            if reservation.budget is not self.budget or reservation.closed:
                raise ValueError("memory artifact reservation must belong to the SAME account")
            self._reservations.append(reservation)

    def hold_graph(self, key: str, size: int) -> None:
        with self._lock:
            self._check(key)
            if key not in self._graphs:
                reservation = self.budget.reserve(size)
                self._graphs[key] = reservation
                self._reservations.append(reservation)
            else:
                self._graphs[key].resize(size)

    def write(self, name: str, data: bytes) -> None:
        with self._lock:
            self._check(name)
            if name in self._chunks or type(data) is not bytes:
                raise ValueError("duplicate or invalid memory artifact")
            self.retain(len(data) + 512 + 4 * len(name))
            self._chunks[name] = [data]

    def append(self, name: str, data: bytes) -> None:
        with self._lock:
            self._check(name)
            if name not in self._chunks or type(data) is not bytes:
                raise ValueError("append requires owned artifact bytes")
            self.retain(len(data) + 128)
            self._chunks[name].append(data)

    def rename(self, old: str, new: str) -> None:
        with self._lock:
            self._check(old)
            self._check(new)
            if old not in self._chunks or new in self._chunks:
                raise ValueError("memory rename requires exclusive destination")
            self.retain(4 * len(new) + 256)
            self._chunks[new] = self._chunks.pop(old)

    def discard(self, name: str) -> None:
        with self._lock:
            self._check(name)
            if not name.endswith((".tmp", ".partial.jsonl")):
                raise ValueError("only owned temporary members may be discarded")
            del self._chunks[name]
            # Conservative charges remain owned until group retirement.

    def contains(self, name: str) -> bool:
        with self._lock:
            safe_name(name)
            return name in self._chunks

    def snapshot(self) -> MemoryArtifactSnapshot:
        with self._lock:
            self._check("snapshot")
            joined = self.budget.reserve(sum(sum(map(len, chunks)) + 512 + 4 * len(name)
                for name, chunks in self._chunks.items()))
            try:
                files = {name: b"".join(chunks) for name, chunks in self._chunks.items()}
            except BaseException:
                joined.close()
                raise
            self._frozen = True
            reservations, self._reservations = [*self._reservations, joined], []
            self._chunks.clear()
            return MemoryArtifactSnapshot(self.budget, files, reservations)

    def close(self) -> None:
        with self._lock:
            self._frozen = True
            self._chunks.clear()
            for reservation in self._reservations:
                reservation.close()
            self._reservations.clear()


class _MemoryText:
    def __init__(self, sink: MemoryGroupArtifacts, name: str) -> None:
        self.sink, self.name, self.closed = sink, name, False
        sink.write(name, b"")

    def write(self, value: str) -> int:
        if self.closed:
            raise ValueError("journal output is closed")
        with self.sink.budget.reserve(4 * len(value) + 128):
            self.sink.append(self.name, value.encode("utf-8"))
        return len(value)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("journal output is closed")

    def close(self) -> None:
        self.closed = True


class MemoryCardInstanceJournal(CardInstanceJournal):
    def __init__(self, *args: Any, artifact_sink: MemoryGroupArtifacts, **kwargs: Any) -> None:
        self.artifact_sink = artifact_sink
        super().__init__(*args, **kwargs)

    def _name(self, path: Path) -> str:
        return safe_name(path.parent.name + "/" + path.name)

    def _final_exists(self) -> bool:
        return self.artifact_sink.contains(self._name(self.final_path))

    def _allocate_output(self) -> None:
        if self._final_exists():
            raise ContractError("journal6 final evidence already exists")
        self.output = _MemoryText(self.artifact_sink, self._name(self.path))  # type: ignore[assignment]

    def write(self, record: Json) -> None:
        # Admission precedes JSON serialization and retains live journal bank/commands.
        self.artifact_sink.retain(32 * _json_bound(record) + 1024)
        super().write(record)

    def begin_frame(self, frame: Any, delta: Any, generated: Any, *args: Any, **kwargs: Any) -> int:
        # EventBank copies/decoded frames/commands are live independently of encoded chunks.
        self.artifact_sink.retain(32 * sum(len(event) for event in delta._events)
            + 65536)
        with self.artifact_sink.budget.reserve(4096 * len(generated.table.kinds)
                + 128 * len(generated.table.slots) + 65536):
            return super().begin_frame(frame, delta, generated, *args, **kwargs)

    def _publish_candidates(self, frame: Any, table: Any, index: int, **kwargs: Any) -> Json:
        name = candidate_basename(frame.actor, index)
        _, size = column_layout(len(table.kinds), len(table.slots))
        with self.artifact_sink.budget.reserve(4 * size + 65536):
            payload, descriptor = encode_candidate_columns_v2(table, name,
                layout=table.layout, max_positions=kwargs["max_positions"])
            self.artifact_sink.write(self.path.parent.name + "/" + name, payload)
        return descriptor

    def _publish_final_output(self) -> None:
        self.artifact_sink.rename(self._name(self.path), self._name(self.final_path))
