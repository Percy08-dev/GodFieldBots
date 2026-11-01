"""Parent-owned process/decision CUDA reservations, released only at owned boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field
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
        self.learner: LearnerReservation | None = None

    @property
    def cuda_bytes(self) -> int:
        return (self.learner_bytes + (self.learner.resident_bytes if self.learner is not None else 0)
                + sum(worker.resident_bytes + sum(worker.decisions.values()) for worker in self.workers.values()))

    def reserve_learner(self, resident_bytes: int, owner: Any) -> LearnerReservation:
        integer(resident_bytes, 0, self.global_cuda_bytes)
        if self.learner_bytes or self.learner is not None or self.workers or self.updating or owner is None:
            raise ContractError('learner reservation duplicates residency or overlaps active work')
        self.learner = LearnerReservation(self, resident_bytes, owner)
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, self.cuda_bytes)
        return self.learner

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
        if self.workers or self.updating or self.cuda_bytes + temporary_bytes > self.global_cuda_bytes:
            raise ContractError("learner update overlaps collection or exceeds shared CUDA budget")
        self.updating = True
        self.peak_cuda_bytes = max(self.peak_cuda_bytes, self.cuda_bytes + temporary_bytes)

    def finish_update(self) -> None:
        if not self.updating:
            raise ContractError("learner update reservation is not active")
        self.updating = False


@dataclass
class LearnerReservation:
    """One concrete async numerical owner; pending creation is still ownership."""
    ledger: ReservationLedger
    resident_bytes: int
    _cleanup_owner: Any = field(repr=False, compare=False)
    creation_pending: bool = False
    pid: int | None = None
    released: bool = False

    def spawned(self, process: Any) -> None:
        if self.released or self.pid is not None or self.ledger.learner is not self:
            raise ContractError('learner reservation already consumed')
        integer(process.pid, 1)
        self.pid = process.pid
        self.creation_pending = False

    def cancel_before_spawn(self) -> None:
        if (self.released or self.pid is not None or self.creation_pending
                or getattr(self._cleanup_owner, 'process', None) is not None):
            raise ContractError('pending/spawned learner requires confirmed creation/reap')
        self._release()

    def reaped(self, process: Any) -> None:
        if self.released:
            return
        if process is None or process.returncode is None or (self.pid is not None and self.pid != process.pid):
            raise ContractError('learner residency cannot release before confirmed reap')
        if (self.creation_pending or self.ledger.updating
                or process is not getattr(self._cleanup_owner, 'process', None)):
            raise ContractError('learner residency cannot release before confirmed reap')
        self._release()

    def _release(self) -> None:
        if self.ledger.learner is not self:
            raise ContractError('foreign learner reservation')
        self.ledger.learner = None
        self.released = True
        self._cleanup_owner = None


@dataclass
class WorkerReservation:
    ledger: ReservationLedger
    key: int
    resident_bytes: int
    decisions: dict[int, int]
    pid: int | None = None
    serial: int = 0
    released: bool = False
    _cleanup_owner: Any = field(default=None, repr=False, compare=False)

    def spawned(self, process: Any) -> None:
        if self.pid is not None or self.released:
            raise ContractError("worker reservation already consumed")
        integer(process.pid, 1)
        self.pid = process.pid

    def cancel_before_spawn(self) -> None:
        if self.pid is not None or self.released:
            raise ContractError("spawned worker requires confirmed reap")
        process = getattr(self._cleanup_owner, 'process', None)
        if process is not None and process.poll() is None:
            raise ContractError("owned startup process requires confirmed reap")
        del self.ledger.workers[self.key]
        self.released = True
        self._cleanup_owner = None

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
        self._cleanup_owner = None
