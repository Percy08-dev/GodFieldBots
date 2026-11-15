"""One durable per-user HPO GPU admission gate; existing owners still reap.

This module allocates no process/device and confers no data qualification. A live
capability audits the SAME resource ledger; persisted PIDs are diagnostic only.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import pwd
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Self, cast

from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_simulator_client.structured_resources import (
    LearnerReservation,
    ReservationLedger,
    WorkerReservation,
)

# HOME/XDG settings of a particular campaign must not split a user's namespace.
_GLOBAL_ROOT = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/state/godfieldbots/hpo-gpu-admission-v1"
RESTART_CONFIRMATION = "I confirm the environment restarted and all previous GPU jobs terminated."
RELEASE_CONFIRMATION = "I authorize GPU admission to resume without confirming termination of the previous job."


class AdmissionBusy(ContractError):
    """Another local campaign holds global GPU admission."""


class RecoveryRequired(ContractError):
    """Unconfirmed durable admission forbids further GPU work."""


@dataclass(frozen=True)
class JobContext:
    campaign_id: str
    operation_id: str
    kind: str
    run_id: str | None
    evaluation_id: str | None


@dataclass(frozen=True)
class ResourceTermination:
    kind: str
    key: int
    pid: int | None
    outcome: str


@dataclass(frozen=True)
class TerminationReceipt:
    schema_version: int
    kind: str
    ticket: str
    admission_sha256: str
    context_sha256: str
    boot_id: str
    confirmed_unix: float
    resources: tuple[ResourceTermination, ...]


@dataclass(frozen=True)
class RestartReceipt:
    schema_version: int
    kind: str
    ticket: str
    admission_sha256: str
    admitted_boot_id: str
    current_boot_id: str
    operator: str
    confirmation: str
    confirmed_unix: float


@dataclass(frozen=True)
class AdmissionReleaseReceipt:
    schema_version: int
    kind: str
    ticket: str
    admission_sha256: str
    admitted_boot_id: str
    current_boot_id: str
    operator: str
    reason: str
    confirmation: str
    released_unix: float


def _text(value: Any) -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise ContractError("nonempty GPU admission identity required")


def _ticket(value: Any) -> None:
    if type(value) is not str or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("invalid GPU admission ticket")


def _sha(value: Any) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError("invalid GPU admission digest")


def _time(value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ContractError("invalid GPU admission timestamp")


def _context(value: Any) -> None:
    value = closed(value, {"campaign_id", "operation_id", "kind", "run_id", "evaluation_id"})
    _text(value["campaign_id"])
    _text(value["operation_id"])
    if value["kind"] not in ("training", "evaluation"):
        raise ContractError("unknown GPU job kind")
    for key in ("run_id", "evaluation_id"):
        if value[key] is not None:
            _text(value[key])
    if value["run_id"] is None or (value["kind"] == "evaluation") != (value["evaluation_id"] is not None):
        raise ContractError("GPU context requires exact run/evaluation identities")


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    _text(value)
    return value


def _directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ContractError("GPU admission directory must be private, owned and nonsymlink")


def _ensure_root(path: Path) -> None:
    missing = []
    ancestor = path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        _sync_directory(directory.parent)
    _directory(path)
    # A prior process may have created any ancestor and failed its parent fsync.
    # Existence cannot turn that uncertainty into durable namespace admission.
    identities = []
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ContractError("non-directory or symlink in GPU admission ancestry")
        identities.append((ancestor, info.st_dev, info.st_ino))
        _sync_directory(ancestor)
    for ancestor, device, inode in identities:
        info = ancestor.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (device, inode):
            raise ContractError("GPU admission ancestry changed while syncing")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path: Path) -> Json:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise ContractError("unowned or nonregular GPU admission record")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (ValueError, UnicodeError) as error:
        raise ContractError("corrupt GPU admission record") from error
    if type(value) is not dict:
        raise ContractError("GPU admission object required")
    return value


def _write(path: Path, value: Json, *, immutable: bool) -> None:
    """Exclusive temp plus atomic publication, then containing-directory fsync."""
    if immutable and path.exists():
        if canonical(_read(path)) != canonical(value):
            raise ContractError("immutable GPU admission record differs")
        _resync(path)
        return
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(canonical(value) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if immutable:
            # Hard-link publication is atomic and cannot overwrite an existing
            # immutable record. The temporary is ours, under the held flock.
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _sync_directory(path.parent)


class _Lock:
    def __init__(self) -> None:
        self.root = _GLOBAL_ROOT
        _ensure_root(self.root)
        self.file: BinaryIO | None = None
        self._job: GpuJob | None = None
        self._admission: Json | None = None
        descriptor = os.open(self.root / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                raise ContractError("invalid GPU admission lock")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise AdmissionBusy("another campaign holds global GPU admission") from error
            self.file = os.fdopen(descriptor, "r+b")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            jobs = self.root / "jobs"
            jobs.mkdir(mode=0o700, exist_ok=True)
            _directory(jobs)
            _sync_directory(self.root)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def _admission(value: Json, ticket: str) -> None:
    closed(value, {"schema_version", "kind", "ticket", "context", "boot_id", "owner_pid", "admitted_unix"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "hpo-gpu-admission" or value["ticket"] != ticket:
        raise ContractError("mismatched GPU admission record")
    _ticket(ticket)
    _context(value["context"])
    _text(value["boot_id"])
    integer(value["owner_pid"], 1)
    _time(value["admitted_unix"])


def _termination(value: Json, admission: Json) -> TerminationReceipt:
    closed(value, {"schema_version", "kind", "ticket", "admission_sha256", "context_sha256", "boot_id", "confirmed_unix", "resources"})
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "hpo-gpu-termination"
            or value["ticket"] != admission["ticket"] or value["admission_sha256"] != digest(admission)
            or value["context_sha256"] != digest(admission["context"]) or value["boot_id"] != admission["boot_id"]):
        raise ContractError("GPU termination receipt identity mismatch")
    _time(value["confirmed_unix"])
    if type(value["resources"]) is not list:
        raise ContractError("invalid GPU termination proof")
    resources = []
    keys = set()
    for resource in value["resources"]:
        closed(resource, {"kind", "key", "pid", "outcome"})
        if resource["kind"] not in ("worker", "learner") or resource["outcome"] not in ("reaped", "never_started"):
            raise ContractError("invalid resource termination")
        integer(resource["key"], 0)
        key = (resource["kind"], resource["key"])
        if key in keys:
            raise ContractError("duplicate resource termination")
        keys.add(key)
        if resource["pid"] is not None:
            integer(resource["pid"], 1)
        if (resource["outcome"] == "reaped") != (resource["pid"] is not None):
            raise ContractError("resource outcome/PID mismatch")
        resources.append(ResourceTermination(**resource))
    return TerminationReceipt(**{**value, "resources": tuple(resources)})


def _restart(value: Json, admission: Json) -> RestartReceipt:
    closed(value, {"schema_version", "kind", "ticket", "admission_sha256", "admitted_boot_id", "current_boot_id", "operator", "confirmation", "confirmed_unix"})
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "hpo-gpu-environment-restart"
            or value["ticket"] != admission["ticket"] or value["admission_sha256"] != digest(admission)
            or value["admitted_boot_id"] != admission["boot_id"] or value["confirmation"] != RESTART_CONFIRMATION):
        raise ContractError("GPU restart receipt identity mismatch")
    _text(value["current_boot_id"])
    _text(value["operator"])
    _time(value["confirmed_unix"])
    return RestartReceipt(**value)


def _release(value: Json, admission: Json) -> AdmissionReleaseReceipt:
    closed(value, {"schema_version", "kind", "ticket", "admission_sha256", "admitted_boot_id",
        "current_boot_id", "operator", "reason", "confirmation", "released_unix"})
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "hpo-gpu-admission-release" or value["ticket"] != admission["ticket"]
            or value["admission_sha256"] != digest(admission) or value["admitted_boot_id"] != admission["boot_id"]
            or value["confirmation"] != RELEASE_CONFIRMATION):
        raise ContractError("GPU admission release receipt identity mismatch")
    _text(value["current_boot_id"])
    _text(value["operator"])
    _text(value["reason"])
    if not value["reason"].strip():
        raise ContractError("non-whitespace admission release reason required")
    _time(value["released_unix"])
    return AdmissionReleaseReceipt(**value)


def _scan(lock: _Lock, *, allow_missing_current: bool = False) -> dict[str, tuple[Json, bool]]:
    """Validate admissions/orphans; the boolean means admission resolved only.

    An operator release permits new admission without confirming termination.
    Data verifiers must independently require an actual termination receipt.
    """
    root = lock.root
    if any(path.name not in {"lock", "current.json", "jobs"} for path in root.iterdir()):
        raise ContractError("uncertain GPU admission storage entry")
    jobs = {}
    for directory in (root / "jobs").iterdir():
        _ticket(directory.name)
        _directory(directory)
        names = {path.name for path in directory.iterdir()}
        resolutions = {"termination.json", "restart.json", "release.json"}
        if "admission.json" not in names or not names <= {"admission.json"} | resolutions:
            raise ContractError("incomplete/corrupt GPU job records")
        if len(names & resolutions) > 1:
            raise ContractError("conflicting GPU admission resolution receipts")
        admission = _read(directory / "admission.json")
        _admission(admission, directory.name)
        resolved = False
        for name, check in (("termination.json", _termination), ("restart.json", _restart), ("release.json", _release)):
            if name in names:
                check(_read(directory / name), admission)
                # A visible exact receipt after a prior directory-fsync failure
                # can be conservatively reconciled only by fully syncing now.
                _resync(directory / name)
                resolved = True
        jobs[directory.name] = (admission, resolved)
    current = root / "current.json"
    if current.exists() or current.is_symlink():
        pointer = closed(_read(current), {"schema_version", "ticket", "admission_sha256"})
        if type(pointer["schema_version"]) is not int or pointer["schema_version"] != 1:
            raise ContractError("invalid GPU current schema")
        _ticket(pointer["ticket"])
        _sha(pointer["admission_sha256"])
        if pointer["ticket"] not in jobs or pointer["admission_sha256"] != digest(jobs[pointer["ticket"]][0]):
            raise ContractError("GPU current pointer identity mismatch")
        _resync(current)
    elif jobs:
        # Admissions without a pointer may be legitimate failed publications;
        # preserve them for explicit recovery rather than infer idle.
        if not allow_missing_current and not any(not resolved for _, resolved in jobs.values()):
            raise ContractError("missing GPU current pointer")
    return jobs


class _AuditedLedger(ReservationLedger):
    _MAX_GLOBAL_CUDA_BYTES = 2**63 - 1

    def __init__(self, job: GpuJob, global_cuda_bytes: int, max_workers: int) -> None:
        super().__init__(global_cuda_bytes, max_workers)
        self._job = job
        self._history: list[_Worker | _Learner] = []
        self._resident_pool: Any = None
        self._resident_temporary = 0
        self._resident_update = False
        self._resident_transition: Any = None

    def reserve_worker(self, resident_bytes: int) -> WorkerReservation:
        self._job._require_open()
        if self._resident_pool is not None:
            self._resident_pool._require()
        if self._resident_transition:
            raise ContractError('resident physical creation overlaps transition')
        token = super().reserve_worker(resident_bytes)
        try:
            audited = _Worker(**{field.name: getattr(token, field.name) for field in fields(WorkerReservation)})
            self.workers[token.key] = audited
            self._history.append(audited)
            return audited
        except BaseException:
            self._job._poisoned = True
            raise

    def reserve_learner(self, resident_bytes: int, owner: Any) -> LearnerReservation:
        self._job._require_open()
        if self._resident_pool is not None:
            self._resident_pool._require()
        token = super().reserve_learner(resident_bytes, owner)
        try:
            audited = _Learner(**{field.name: getattr(token, field.name) for field in fields(LearnerReservation)})
            audited._owner = owner
            boundary = getattr(owner, "_startup_boundary", None)
            if not isinstance(boundary, asyncio.Event) or boundary.is_set():
                raise ContractError("original pending learner startup Event required")
            audited._boundary = boundary
            audited._audit_key = 1 + sum(isinstance(item, _Learner) for item in self._history)
            self.learner = audited
            self._history.append(audited)
            return audited
        except BaseException:
            self._job._poisoned = True
            raise

    def begin_update(self, temporary_bytes: int) -> None:
        self._job._require_open()
        if self._resident_pool is not None:
            self._resident_pool._require()
        super().begin_update(temporary_bytes)


    @property
    def cuda_bytes(self) -> int:
        return super().cuda_bytes + getattr(self, '_resident_temporary', 0)

    def _parked(self, authority: Any) -> None:
        from gf_training.hpo_resident import ParkedAuthority
        self._job._require_open()
        if type(authority) is not ParkedAuthority:
            raise ContractError('SAME live resident PARK authority required')
        authority.require(self)

    def begin_resident_update(self, temporary_bytes: int, *, authority: Any) -> None:
        self._parked(authority)
        integer(temporary_bytes, 0, self.global_cuda_bytes)
        if self.updating or self._resident_transition or self._resident_temporary or self.cuda_bytes + temporary_bytes > self.global_cuda_bytes:
            raise ContractError('resident update/transition overlaps or exceeds shared CUDA budget')
        self._resident_temporary = temporary_bytes
        self._resident_update = self.updating = True
        self.peak_cuda_bytes = max(self.peak_cuda_bytes,self.cuda_bytes)

    def finish_update(self) -> None:
        if self._resident_update:
            self._job._require_open()
            self._resident_temporary = 0
            self._resident_update = False
        super().finish_update()

    def reserve_resident_transition(self, temporary_bytes: int, *, authority: Any,
                                    target_generation: int, target_checkpoint_sha256: str, affected_workers: tuple[_Worker,...]) -> Any:
        from gf_training.hpo_resident import ResidentTransition
        self._parked(authority)
        return ResidentTransition(self,temporary_bytes,authority,target_generation=target_generation,
            target_checkpoint_sha256=target_checkpoint_sha256,affected_workers=affected_workers)


def _job(token: Any) -> GpuJob:
    ledger = token.ledger
    if type(ledger) is not _AuditedLedger or not any(item is token for item in ledger._history):
        raise ContractError("foreign audited GPU reservation")
    job = ledger._job
    job._require_open()
    if job.ledger is not ledger:
        raise ContractError("foreign GPU job ledger")
    return job


def _actual_process(process: Any, owner: Any) -> None:
    if not isinstance(process, (subprocess.Popen, asyncio.subprocess.Process)) or process is not getattr(owner, "process", None):
        raise ContractError("exact owned process object required")


class _Worker(WorkerReservation):
    _owner: Any = None
    _process: Any = None
    _proof: ResourceTermination | None = None
    _boundary: threading.Event | None = None
    _finished: bool = False
    _creation_error: BaseException | None = None

    def _creation_started(self, owner: Any, boundary: Any) -> None:
        _job(self)
        if not isinstance(boundary, threading.Event):
            raise ContractError("original worker creation Event required")
        if (self._owner is not None or self._cleanup_owner is not owner
                or boundary.is_set() or boundary is not getattr(owner, "_creation_boundary", None)):
            raise ContractError("original worker creation owner/Event required")
        self._owner, self._boundary = owner, boundary

    def _creation_finished(self, owner: Any, boundary: Any, process: Any, error: BaseException | None) -> None:
        _job(self)
        if self._owner is not owner or self._boundary is not boundary or self._finished:
            raise ContractError("worker creation observation differs from original")
        if process is not None:
            _actual_process(process, owner)
        self._process, self._creation_error, self._finished = process, error, True

    def _settled(self, *, never_started: bool) -> None:
        boundary = self._boundary
        if boundary is None:
            raise ContractError("original worker creation completion is unconfirmed")
        if (self._owner is None or not self._finished
                or getattr(self._owner, "_creation_observer_failure", None) is not None
                or not boundary.is_set() or boundary is not getattr(self._owner, "_creation_boundary", None)):
            raise ContractError("original worker creation completion is unconfirmed")
        if never_started and (self._process is not None or not isinstance(self._creation_error, Exception)):
            raise ContractError("actual synchronous stdlib no-child exception required")

    def reserve_decision(self, temporary_bytes: int) -> int:
        job = _job(self)
        pool = job._ledger._resident_pool
        if pool is not None:
            pool.require_decision(self)
        return super().reserve_decision(temporary_bytes)

    def spawned(self, process: Any) -> None:
        _job(self)
        self._settled(never_started=False)
        _actual_process(process, self._cleanup_owner)
        if self._owner is not self._cleanup_owner or self._process is not process:
            raise ContractError("spawn differs from original creation observation")
        super().spawned(process)

    def reaped(self, process: Any) -> None:
        _job(self)
        if self._proof is not None:
            if process is not self._process:
                raise ContractError("different process on repeated reap")
            return
        if self._process is not process:
            raise ContractError("resource reap does not match original process object")
        _actual_process(process, self._owner)
        self._settled(never_started=False)
        super().reaped(process)
        self._proof = ResourceTermination("worker", self.key, process.pid, "reaped")

    def cancel_before_spawn(self) -> None:
        _job(self)
        owner = self._cleanup_owner
        if owner is not self._owner:
            raise ContractError("original worker cleanup owner required")
        process = getattr(owner, "process", None)
        self._settled(never_started=process is None)
        if process is not self._process:
            raise ContractError("worker process differs from actual creation result")
        if process is not None:
            _actual_process(process, owner)
            if process.returncode is None:
                raise ContractError("returned worker process has not reaped")
        super().cancel_before_spawn()
        self._owner, self._process = owner, process
        self._proof = ResourceTermination("worker", self.key, None if process is None else process.pid,
                                          "never_started" if process is None else "reaped")


class _Learner(LearnerReservation):
    _owner: Any = None
    _process: Any = None
    _proof: ResourceTermination | None = None
    _audit_key: int = 0
    _boundary: asyncio.Event | None = None
    _creation: asyncio.Future[Any] | None = None
    _creation_error: Exception | None = None
    _task_error: BaseException | None = None

    def _same_owner(self, owner: Any) -> None:
        _job(self)
        if self._owner is not owner or self._boundary is None or self._boundary is not getattr(owner, "_startup_boundary", None):
            raise ContractError("original learner creation owner/Event required")

    def _creation_task_created(self, owner: Any, creation: Any) -> None:
        self._same_owner(owner)
        if self._creation is not None or self._task_error is not None or not isinstance(creation, asyncio.Future):
            raise ContractError("original local learner creation Future required")
        self._creation = creation

    def _creation_failed(self, owner: Any, error: Exception) -> None:
        self._same_owner(owner)
        if not isinstance(error, Exception) or self._creation_error is not None:
            raise ContractError("actual default stdlib creation exception required")
        self._creation_error = error

    def _creation_task_failed(self, owner: Any, error: BaseException) -> None:
        self._same_owner(owner)
        if not isinstance(error, BaseException) or self._creation is not None or self._task_error is not None:
            raise ContractError("actual closed-unstarted-coroutine failure required")
        self._task_error = error

    def _settled(self, *, never_started: bool) -> None:
        self._same_owner(self._owner)
        if getattr(self._owner, "_creation_observer_failure", None) is not None:
            raise ContractError("learner creation observation failed")
        if self._boundary is None or not self._boundary.is_set():
            raise ContractError("original learner startup Event is unsettled")
        creation = self._creation
        if creation is None:
            if not never_started or self._task_error is None or getattr(self._owner, "creation_task", None) is not None:
                raise ContractError("positive original learner creation proof required")
            return
        if (not creation.done() or creation.cancelled() or creation is not getattr(self._owner, "creation_task", None)):
            raise ContractError("original learner creation Future is replaced or unsettled")
        creation = cast(asyncio.Future[Any], creation)
        if never_started:
            if self._creation_error is None or creation.exception() is not self._creation_error:
                raise ContractError("original Future lacks the SAME observed stdlib no-child exception")
        elif creation.exception() is not None or creation.result() is not self._process:
            raise ContractError("original Future did not return the exact owned process")

    def spawned(self, process: Any) -> None:
        _job(self)
        _actual_process(process, self._owner)
        self._process = process
        self._same_owner(self._owner)
        creation = self._creation
        if creation is None:
            raise ContractError("spawn lacks original returned-process Future")
        if (not creation.done() or creation.cancelled() or creation.exception() is not None
                or creation.result() is not process or creation is not getattr(self._owner, "creation_task", None)):
            raise ContractError("spawn lacks original returned-process Future")
        super().spawned(process)

    def reaped(self, process: Any) -> None:
        _job(self)
        if self._proof is not None:
            if process is not self._process:
                raise ContractError("different process on repeated reap")
            return
        if self._process is not process:
            raise ContractError("resource reap does not match original process object")
        _actual_process(process, self._owner)
        self._settled(never_started=False)
        super().reaped(process)
        self._proof = ResourceTermination("learner", self._audit_key, process.pid, "reaped")

    def cancel_before_spawn(self) -> None:
        _job(self)
        self._settled(never_started=True)
        super().cancel_before_spawn()
        self._proof = ResourceTermination("learner", self._audit_key, None, "never_started")


_ISSUER = object()


class GpuJob:
    """Unpersistable admission capability. close releases only the global flock."""

    def __init__(self, issuer: object, lock: _Lock, admission: Json, global_cuda_bytes: int, max_workers: int) -> None:
        if issuer is not _ISSUER or lock._job is not None or lock._admission is not admission:
            raise ContractError("GPU job requires live admission issuer")
        self._lock, self._admission = lock, admission
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._poisoned, self._sealed = False, False
        self._receipt: TerminationReceipt | None = None
        self._ledger = _AuditedLedger(self, global_cuda_bytes, max_workers)
        self._capability = object()
        self._original_capability = self._capability
        lock._job = self

    @property
    def ticket(self) -> str:
        return str(self._admission["ticket"])

    @property
    def ledger(self) -> ReservationLedger:
        return self._ledger

    def _require_owner(self) -> None:
        if (os.getpid() != self._pid or threading.get_ident() != self._thread or self._lock.file is None or self._lock.file.closed
                or self._capability is not self._original_capability or self._ledger._job is not self or self._lock._job is not self
                or self._lock._admission is not self._admission):
            raise ContractError("GPU admission is not the same live held job capability")

    def _require_open(self) -> None:
        self._require_owner()
        if self._sealed or self._poisoned:
            raise ContractError("GPU job is sealed or poisoned")

    def require_confirmed_termination(self,receipt: TerminationReceipt) -> TerminationReceipt:
        """Read-only validation of this still-owned job's exact confirmed proof."""
        self._require_owner()
        if self._poisoned or not self._sealed or self._receipt is None or receipt is not self._receipt:
            raise ContractError('SAME unpoisoned already-confirmed termination receipt required')
        ledger=self._ledger
        proofs=tuple(token._proof for token in ledger._history)
        if (ledger.workers or ledger.learner is not None or ledger.updating or ledger._resident_temporary
                or ledger._resident_transition or len(proofs)!=len(receipt.resources)
                or any(not token.released or token.creation_pending or token._proof is None for token in ledger._history)
                or any(proof is not recorded for proof,recorded in zip(proofs,receipt.resources,strict=True))):
            raise ContractError('confirmed termination original resource history changed')
        path=self._lock.root/'jobs'/self.ticket/'termination.json'
        if _termination(_read(path),self._admission)!=receipt:
            raise ContractError('original confirmed durable termination receipt changed')
        return receipt

    def confirm_termination(self) -> TerminationReceipt:
        self._require_owner()
        if self._receipt is not None:
            path = self._lock.root / "jobs" / self.ticket / "termination.json"
            if _termination(_read(path), self._admission) != self._receipt:
                raise ContractError("committed GPU receipt changed")
            _resync(path)
            return self._receipt
        self._require_open()
        ledger = self._ledger
        if ledger.workers or ledger.learner is not None or ledger.updating or ledger._resident_temporary or ledger._resident_transition or any(
                not token.released or token.creation_pending or token._proof is None for token in ledger._history):
            raise ContractError("GPU resource creation/reap/update remains unconfirmed")
        if ledger._resident_pool is not None:
            ledger._resident_pool.require_resource_retirement(self,ledger)
        self._sealed = True
        try:
            path = self._lock.root / "jobs" / self.ticket
            if canonical(_read(path / "admission.json")) != canonical(self._admission):
                raise ContractError("original GPU admission changed")
            current = closed(_read(self._lock.root / "current.json"), {"schema_version", "ticket", "admission_sha256"})
            if current != {"schema_version": 1, "ticket": self.ticket, "admission_sha256": digest(self._admission)}:
                raise ContractError("GPU job is no longer current")
            receipt = TerminationReceipt(1, "hpo-gpu-termination", self.ticket, digest(self._admission),
                digest(self._admission["context"]), self._admission["boot_id"], time.time(),
                tuple(token._proof for token in ledger._history if token._proof is not None))
            _write(path / "termination.json", asdict(receipt), immutable=True)
            self._receipt = receipt
            return receipt
        except BaseException:
            self._poisoned = True
            raise

    def close(self) -> None:
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ContractError("foreign process/thread cannot close GPU capability")
        self._sealed = True
        self._lock.close()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> Any:
        raise TypeError("GPU job capabilities cannot be serialized")


def admit_gpu_job(context: JobContext, *, global_cuda_bytes: int, max_workers: int) -> GpuJob:
    """Durably admit before any GPU work; every campaign shares this namespace."""
    if type(context) is not JobContext:
        raise ContractError("typed closed GPU job context required")
    _context(asdict(context))
    integer(global_cuda_bytes, 1, _AuditedLedger._MAX_GLOBAL_CUDA_BYTES)
    integer(max_workers, 1, _AuditedLedger._MAX_WORKERS)
    lock = _Lock()
    try:
        jobs = _scan(lock)
        unresolved = [ticket for ticket, (_, resolved) in jobs.items() if not resolved]
        if unresolved:
            raise RecoveryRequired("RECOVERY_REQUIRED: " + ",".join(sorted(unresolved)))
        ticket = uuid.uuid4().hex
        admission = {"schema_version": 1, "kind": "hpo-gpu-admission", "ticket": ticket,
            "context": asdict(context), "boot_id": _boot_id(), "owner_pid": os.getpid(), "admitted_unix": time.time()}
        directory = lock.root / "jobs" / ticket
        directory.mkdir(mode=0o700)
        _sync_directory(directory.parent)
        _write(directory / "admission.json", admission, immutable=True)
        _write(lock.root / "current.json", {"schema_version": 1, "ticket": ticket,
            "admission_sha256": digest(admission)}, immutable=False)
        lock._admission = admission
        return GpuJob(_ISSUER, lock, admission, global_cuda_bytes, max_workers)
    except BaseException:
        lock.close()
        raise


def require_training_job(job: GpuJob, *, ledger: ReservationLedger, run_id: str,
                          global_cuda_bytes: int, max_workers: int) -> None:
    """The exact caller-owned live capability, never an empty replacement ledger."""
    if type(job) is not GpuJob:
        raise ContractError('actual caller GpuJob required')
    job._require_open()
    context=job._admission['context']
    if (context['kind']!='training' or context['run_id']!=run_id or context['evaluation_id'] is not None
            or type(ledger) is not _AuditedLedger or ledger is not job.ledger
            or ledger.global_cuda_bytes!=global_cuda_bytes or ledger.max_workers!=max_workers):
        raise ContractError('HPO run differs from the SAME caller GPU job/context/ledger/limits')
    try:
        if str(uuid.UUID(run_id))!=run_id: raise ValueError('noncanonical UUID')
    except (ValueError,TypeError,AttributeError) as error:
        raise ContractError('HPO training requires predeclared canonical run UUID') from error


def confirm_environment_restart(*, expected_ticket: str, operator: str, confirmation: str) -> RestartReceipt:
    """Explicit operator recovery only. Boot changes themselves grant nothing."""
    _ticket(expected_ticket)
    _text(operator)
    if confirmation != RESTART_CONFIRMATION:
        raise ContractError("explicit environment restart assertion required")
    lock = _Lock()
    try:
        jobs = _scan(lock, allow_missing_current=True)
        if expected_ticket not in jobs:
            raise ContractError("restart does not identify an admitted job")
        admission, resolved = jobs[expected_ticket]
        path = lock.root / "jobs" / expected_ticket / "restart.json"
        if path.exists():
            receipt = _restart(_read(path), admission)
            if receipt.operator != operator or receipt.confirmation != confirmation or receipt.current_boot_id != _boot_id():
                raise ContractError("restart retry differs from exact receipt")
            _resync(path)
            if not (lock.root / "current.json").exists():
                _write(lock.root / "current.json", {"schema_version": 1, "ticket": expected_ticket,
                    "admission_sha256": digest(admission)}, immutable=False)
            return receipt
        if resolved:
            raise ContractError("GPU admission already resolved by another receipt")
        receipt = RestartReceipt(1, "hpo-gpu-environment-restart", expected_ticket,
            digest(admission), admission["boot_id"], _boot_id(), operator, confirmation, time.time())
        _write(path, asdict(receipt), immutable=True)
        # A crash before initial current publication may leave a valid orphan.
        # Explicit recovery supplies its exact original pointer, never an idle reset.
        if not (lock.root / "current.json").exists():
            _write(lock.root / "current.json", {"schema_version": 1, "ticket": expected_ticket,
                "admission_sha256": digest(admission)}, immutable=False)
        return receipt
    finally:
        lock.close()


def release_unconfirmed_admission(*, expected_ticket: str, operator: str, reason: str,
                                   confirmation: str) -> AdmissionReleaseReceipt:
    """Explicit operator admission decision; does not certify cleanup or data."""
    _ticket(expected_ticket)
    _text(operator)
    _text(reason)
    if not reason.strip():
        raise ContractError("non-whitespace admission release reason required")
    if confirmation != RELEASE_CONFIRMATION:
        raise ContractError("explicit unconfirmed admission release authorization required")
    lock = _Lock()
    try:
        jobs = _scan(lock, allow_missing_current=True)
        if expected_ticket not in jobs:
            raise ContractError("release does not identify an admitted job")
        admission, resolved = jobs[expected_ticket]
        path = lock.root / "jobs" / expected_ticket / "release.json"
        if path.exists():
            receipt = _release(_read(path), admission)
            if (receipt.operator != operator or receipt.reason != reason or receipt.confirmation != confirmation
                    or receipt.current_boot_id != _boot_id()):
                raise ContractError("admission release retry differs from exact receipt")
            _resync(path)
            if not (lock.root / "current.json").exists():
                _write(lock.root / "current.json", {"schema_version": 1, "ticket": expected_ticket,
                    "admission_sha256": digest(admission)}, immutable=False)
            return receipt
        if resolved:
            raise ContractError("GPU admission already resolved by another receipt")
        receipt = AdmissionReleaseReceipt(1, "hpo-gpu-admission-release", expected_ticket,
            digest(admission), admission["boot_id"], _boot_id(), operator, reason, confirmation, time.time())
        _write(path, asdict(receipt), immutable=True)
        if not (lock.root / "current.json").exists():
            _write(lock.root / "current.json", {"schema_version": 1, "ticket": expected_ticket,
                "admission_sha256": digest(admission)}, immutable=False)
        return receipt
    finally:
        lock.close()
