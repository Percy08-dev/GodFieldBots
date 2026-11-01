"""Parent-owned process/decision CUDA reservations, released only at owned boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gf_common.domain import ContractError, integer


class ReservationLedger:
    def __init__(self, global_cuda_bytes: int = 12 * 1024**3, max_workers: int = 4, *, learner_bytes: int = 0) -> None:
        integer(global_cuda_bytes, 1, 12 * 1024**3)
        integer(max_workers, 1, 4)
        integer(learner_bytes, 0, global_cuda_bytes)
        self.global_cuda_bytes, self.max_workers, self.learner_bytes = global_cuda_bytes, max_workers, learner_bytes
        self.workers: dict[int, WorkerReservation] = {}
        self.serial = 0
        self.updating = False
        self.peak_cuda_bytes = learner_bytes

    @property
    def cuda_bytes(self) -> int:
        return self.learner_bytes + sum(worker.resident_bytes + sum(worker.decisions.values()) for worker in self.workers.values())

    def reserve_worker(self, resident_bytes: int) -> WorkerReservation:
        integer(resident_bytes, 0, self.global_cuda_bytes)
        if self.updating or len(self.workers) >= self.max_workers or self.cuda_bytes + resident_bytes > self.global_cuda_bytes:
            raise ContractError("participant startup residency cannot fit shared worker/CUDA budget")
        self.serial += 1
        worker = WorkerReservation(self, self.serial, resident_bytes, {})
        self.workers[worker.key] = worker
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, self.cuda_bytes)
        return worker

    def begin_update(self, temporary_bytes: int) -> None:
        integer(temporary_bytes, 0, self.global_cuda_bytes)
        if self.workers or self.updating or self.learner_bytes + temporary_bytes > self.global_cuda_bytes:
            raise ContractError("learner update overlaps collection or exceeds shared CUDA budget")
        self.updating = True
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, self.learner_bytes + temporary_bytes)

    def finish_update(self) -> None:
        if not self.updating:
            raise ContractError("learner update reservation is not active")
        self.updating = False


@dataclass
class WorkerReservation:
    ledger: ReservationLedger
    key: int
    resident_bytes: int
    decisions: dict[int, int]
    pid: int | None = None
    serial: int = 0
    released: bool = False

    def spawned(self, process: Any) -> None:
        if self.pid is not None or self.released:
            raise ContractError("worker reservation already consumed")
        integer(process.pid, 1)
        self.pid = process.pid

    def cancel_before_spawn(self) -> None:
        if self.pid is not None or self.released:
            raise ContractError("spawned worker requires confirmed reap")
        del self.ledger.workers[self.key]
        self.released = True

    def reserve_decision(self, temporary_bytes: int) -> int:
        integer(temporary_bytes, 0, self.ledger.global_cuda_bytes)
        if self.released or self.pid is None or self.decisions:
            raise ContractError("worker already has an active decision or is not live")
        if self.ledger.cuda_bytes + temporary_bytes > self.ledger.global_cuda_bytes:
            raise ContractError("decision cannot fit shared CUDA budget")
        self.serial += 1
        self.decisions[self.serial] = temporary_bytes
        self.ledger.peak_cuda_bytes = max(self.ledger.peak_cuda_bytes, self.ledger.cuda_bytes)
        return self.serial

    def reply(self, key: int) -> None:
        if self.released or key not in self.decisions:
            raise ContractError("unknown scoped decision reservation")
        del self.decisions[key]

    def reaped(self, process: Any) -> None:
        if self.released:
            return
        if self.pid != process.pid or process.poll() is None:
            raise ContractError("participant residency cannot release before confirmed reap")
        self.decisions.clear()
        del self.ledger.workers[self.key]
        self.released = True
