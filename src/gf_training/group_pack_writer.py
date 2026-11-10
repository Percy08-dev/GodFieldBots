"""Synchronous, byte-admitted coordinator for one dedicated SQLite Writer child."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from gf_common.group_pack import (
    FORMAT,
    MAX_JSON,
    MIB,
    Json,
    StorageSettings,
    canonical_json,
)
from gf_training.group_pack_db import (
    descriptor_name,
    fsync_directory,
    identity_text,
    validate_descriptor,
)
from gf_training.group_pack_resources import (
    WRITER_BASE,
    ByteBudget,
    CleanupError,
    close_raw_descriptor,
    close_resources,
    receive_header,
    remaining,
    require_certain_releases,
    retain_cleanup,
    send_frame,
)
from gf_training.group_pack_source import CollectionLocator, OriginalFile, _stat_key, inspect_packed


def _json_bound(value: Json, *, depth: int = 0) -> int:
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
    if isinstance(value, list):
        return 2 + sum(_json_bound(v, depth=depth + 1) + 1 for v in value)
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON object keys must be strings")
        return 2 + sum(
            _json_bound(k, depth=depth + 1) + _json_bound(v, depth=depth + 1) + 2
            for k, v in value.items()
        )
    raise ValueError("invalid JSON value")


class GroupPackWriter:
    def __init__(
        self, directory: Path, run_id: str, *, settings: StorageSettings, deadline: float
    ) -> None:
        remaining(deadline)
        identity_text(run_id)
        self.directory = Path(os.path.abspath(directory))
        self.settings = settings
        self.run_id = run_id
        self.deadline = deadline
        self._lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._budget = ByteBudget(settings.budget_bytes, base=WRITER_BASE)
        self._queue = ByteBudget(settings.queue_bytes)
        self._request_id = 0
        self._pending = threading.Condition()
        self._pending_count = 0
        self._sealing = False
        self._closing = False
        self._retained_reservations: list[tuple[int, int]] = []
        self._stopped = False
        self._sealed = False
        self._poisoned = False
        self._base_released = False
        self._process: subprocess.Popen[bytes] | None = None
        self._published: tuple[Path, tuple[int, int]] | None = None
        self._temporary: tuple[Path, tuple[int, int]] | None = None
        self._publication_fd = -1
        self._uncertain_releases: list[Json] = []
        self._cleanup_children: list[Json] = []
        self._database_original: OriginalFile | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            from gf_training.group_pack_source import _no_symlink_path

            _no_symlink_path(self.directory)
            # Catalog custody cannot share the closed collection namespace.
            if settings.db_root.absolute().is_relative_to(self.directory):
                raise ValueError("catalog must be outside collection directory")
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-m", "gf_training.group_pack_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            assert self._process.stdin is not None and self._process.stdout is not None
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            self._rpc(
                "init",
                {
                    "directory": str(self.directory),
                    "run_id": run_id,
                    "settings": {
                        "db_root": str(settings.db_root.absolute()),
                        "ram_root": str(settings.ram_root.absolute()),
                        "budget_bytes": settings.budget_bytes,
                        "queue_bytes": settings.queue_bytes,
                        "max_pack_bytes": settings.max_pack_bytes,
                    },
                    "deadline": deadline,
                },
            )
            self._database_original = OriginalFile(
                self.directory / "collection.sqlite3", self.deadline, mutable=True
            )
        except BaseException as exc:
            self._fail(exc)
            raise

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("Writer did not start")
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.poll() if self._process is not None else None

    @property
    def owned_bytes(self) -> int:
        return self._budget.used

    @property
    def peak_owned_bytes(self) -> int:
        return self._budget.peak

    @property
    def queued_bytes(self) -> int:
        return self._queue.used

    @contextmanager
    def _operation(self) -> Iterator[None]:
        if not self._lock.acquire(timeout=remaining(self.deadline)):
            raise TimeoutError("Writer serialization deadline expired")
        try:
            if self._stopped or self._poisoned or self._sealed or self._closing:
                raise RuntimeError("Writer no longer admits operations")
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _admission(self, metadata_bound: int, payload_size: int, raw_bytes: int) -> Iterator[None]:
        queued = metadata_bound + payload_size
        charged = 4 * raw_bytes + 4 * payload_size + 32 * metadata_bound
        if metadata_bound > 6 * MAX_JSON + 256 or queued > self.settings.queue_bytes:
            raise ValueError("submission exceeds queue capacity")
        if charged + WRITER_BASE > self.settings.budget_bytes:
            raise ValueError("submission exceeds storage capacity")
        with self._pending:
            if self._sealing or self._closing or self._stopped:
                raise RuntimeError("Writer admission has closed")
            self._pending_count += 1
        queued_acquired = False
        acquired = False
        try:
            self._queue.acquire(queued, self.deadline, wait=True)
            queued_acquired = True
            self._budget.acquire(charged, self.deadline, wait=True)
            acquired = True
            yield
        finally:
            try:
                # Retain admission until the SAME child is confirmed reaped.
                if self._poisoned and self._process is not None and self._process.poll() is None:
                    self._retained_reservations.append(
                        (charged if acquired else 0, queued if queued_acquired else 0)
                    )
                    primary = sys.exception()
                    if primary is None:
                        raise CleanupError("Writer cleanup uncertain; reservation retained", self)
                    primary.__dict__["cleanup_owner"] = self
                    primary.add_note("Writer reservation retained until confirmed reap")
                else:
                    if acquired:
                        self._budget.release(charged)
                    if queued_acquired:
                        self._queue.release(queued)
            finally:
                with self._pending:
                    self._pending_count -= 1
                    self._pending.notify_all()

    def _rpc(self, command: str, extra: Json = None, payload: bytes = b"") -> Json:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Writer pipes unavailable")
        request_id = self._request_id
        self._request_id += 1
        request = {"version": 1, "request_id": request_id, "command": command}
        if extra:
            request.update(extra)
        try:
            if self._database_original is not None:
                self._database_original.check_identity()
            send_frame(process.stdin.fileno(), request, payload, self.deadline)
            response, payload_size = receive_header(process.stdout.fileno(), self.deadline, 0)
            if (
                set(response) != {"version", "request_id", "status", "result", "error"}
                or type(response["version"]) is not int
                or response["version"] != 1
                or type(response["request_id"]) is not int
                or response["request_id"] != request_id
                or payload_size != 0
                or response["status"] not in ("ok", "error")
            ):
                raise RuntimeError("invalid Writer acknowledgement")
            if response["status"] == "error":
                error = response["error"]
                if (
                    not isinstance(error, dict)
                    or set(error) != {"type", "message"}
                    or not isinstance(error["type"], str)
                    or not isinstance(error["message"], str)
                    or len(error["message"]) > 256
                ):
                    raise RuntimeError("invalid Writer failure acknowledgement")
                raise RuntimeError("Writer rejected storage operation")
            if response["error"] is not None or (
                command != "seal" and response["result"] is not None
            ):
                raise RuntimeError("invalid Writer response fields")
            if self._database_original is not None:
                self._database_original.check_identity()
            return response["result"]
        except BaseException as exc:
            self._fail(exc)
            raise

    def submit(
        self, group_id: str, pack: bytes, *, raw_bytes: int, record: Json, episodes: list[Json]
    ) -> None:
        identity_text(group_id)
        if type(pack) is not bytes or len(pack) > self.settings.max_pack_bytes + MIB:
            raise ValueError("compressed submission exceeds bound")
        if type(raw_bytes) is not int or raw_bytes < 16 or raw_bytes > self.settings.max_pack_bytes:
            raise ValueError("raw submission exceeds bound")
        extra = {
            "group_id": group_id,
            "raw_bytes": raw_bytes,
            "record": record,
            "episodes": episodes,
        }
        bound = _json_bound(extra) + 256
        with self._admission(bound, len(pack), raw_bytes), self._operation():
            self._rpc("submit", extra, pack)

    def store_control(self, name: str, data: bytes) -> None:
        from gf_common.group_pack import safe_name

        safe_name(name)
        if type(data) is not bytes or len(data) > min(self.settings.max_pack_bytes, 64 * MIB):
            raise ValueError("control exceeds bound")
        with self._admission(_json_bound(name) + 256, len(data), len(data)), self._operation():
            self._rpc("control", {"name": name}, data)

    def _retire(self, *, terminate: bool = False) -> None:
        with self._cleanup_lock:
            self._retire_locked(terminate=terminate)

    def _retire_locked(self, *, terminate: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if terminate and process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5.0 if terminate else remaining(self.deadline))
        except subprocess.TimeoutExpired as exc:
            raise CleanupError("Writer exit not confirmed", self) from exc
        if not terminate and process.stdout is not None and os.read(process.stdout.fileno(), 1):
            raise RuntimeError("unexpected trailing Writer response bytes")
        # The pipes are retired only after exit, so a partial frame cannot survive.
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()
        for charged, queued in self._retained_reservations:
            self._budget.release(charged)
            self._queue.release(queued)
        self._retained_reservations.clear()

    def _release_base(self) -> None:
        require_certain_releases(self)
        if not self._base_released:
            self._budget.release(WRITER_BASE)
            self._base_released = True

    def _cleanup_physical(self) -> None:
        self._invalidate_publication()
        self._published = None
        self._unlink_same(self._temporary)
        self._temporary = None
        close_raw_descriptor(self, "_publication_fd")
        close_resources(self, [self._database_original])
        require_certain_releases(self)

    def _fail(self, cause: BaseException) -> None:
        retain_cleanup(self, cause)
        self._poisoned = True
        self._stopped = True
        self._queue.stop()
        self._budget.stop()
        try:
            self._retire(terminate=True)
            self._cleanup_physical()
            self._release_base()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve primary and cleanup owner
            cause.__dict__["cleanup_owner"] = self
            cause.add_note(f"Writer cleanup remains owned ({type(cleanup).__name__})")

    def _unlink_same(self, item: tuple[Path, tuple[int, int]] | None) -> None:
        if item is None:
            return
        path, identity = item
        try:
            value = path.lstat()
        except FileNotFoundError:
            return
        if (value.st_dev, value.st_ino) != identity:
            raise CleanupError("foreign descriptor replacement; cleanup owner retained", self)
        path.unlink()
        fsync_directory(self.directory)

    def _invalidate_publication(self) -> None:
        """Invalidate only the SAME owned inode through a verified O_RDWR handle.

        The public pathname is never conditionally unlinked: a foreign replacement
        after verification cannot redirect writes through the held descriptor.
        """
        if self._published is None:
            return
        path, identity = self._published
        if self._publication_fd < 0:
            self._publication_fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        info = os.fstat(self._publication_fd)
        if (info.st_dev, info.st_ino) != identity:
            raise CleanupError("owned descriptor handle identity changed", self)
        marker = b"GFB-QUARANTINED-DESCRIPTOR\n"
        os.ftruncate(self._publication_fd, 0)
        if os.pwrite(self._publication_fd, marker, 0) != len(marker):
            raise CleanupError("descriptor invalidation incomplete", self)
        os.fsync(self._publication_fd)
        fsync_directory(self.directory)
        try:
            current = path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) != identity:
            raise CleanupError("foreign descriptor replacement; cleanup owner retained", self)

    def _publish(self, descriptor: Json) -> CollectionLocator:
        target = self.directory / descriptor_name(descriptor["publication"])
        data = canonical_json(descriptor, limit=MAX_JSON)
        temporary = self.directory / f".descriptor-{uuid4().hex}.tmp"
        self._publication_fd = os.open(
            temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(self._publication_fd)
            identity = (info.st_dev, info.st_ino)
            self._temporary = (temporary, identity)
            assert self._database_original is not None
            self._database_original.recheck()
            cursor = 0
            while cursor < len(data):
                remaining(self.deadline)
                count = os.write(self._publication_fd, data[cursor : cursor + 64 * 1024])
                if count <= 0:
                    raise RuntimeError("descriptor write incomplete")
                cursor += count
            os.fsync(self._publication_fd)
            os.link(temporary, target, follow_symlinks=False)
            self._published = (target, identity)
            self._unlink_same(self._temporary)
            self._temporary = None
            fsync_directory(self.directory)
            current = target.lstat()
            if (current.st_dev, current.st_ino) != identity or os.pread(
                self._publication_fd, len(data) + 1, 0
            ) != data:
                raise RuntimeError("published descriptor identity changed")
            validated = inspect_packed(
                target,
                deadline=self.deadline,
                max_owned_bytes=self.settings.budget_bytes - self.owned_bytes,
                allow_pending=True,
                database_original=self._database_original,
            )
            if validated["descriptor"] != descriptor:
                raise RuntimeError("published descriptor changed")
            # Verify the SAME opened inode after the complete DB/hash sweep.
            current = target.lstat()
            if (current.st_dev, current.st_ino) != identity or os.pread(
                self._publication_fd, len(data) + 1, 0
            ) != data:
                raise RuntimeError("published descriptor identity changed")
            self._database_original.recheck()
            completion_stat = os.fstat(self._publication_fd)
            self._database_original.close()
            close_raw_descriptor(self, "_publication_fd")
            require_certain_releases(self)
            self._database_original.check_path_identity()
            if _stat_key(target.lstat()) != _stat_key(completion_stat):
                raise ValueError("descriptor changed during terminal retirement")
            self._published = None
            return CollectionLocator(FORMAT, target)
        except BaseException as exc:
            retain_cleanup(self, exc)
            try:
                self._invalidate_publication()
                self._published = None
                self._unlink_same(self._temporary)
                self._temporary = None
                close_raw_descriptor(self, "_publication_fd")
                close_resources(self, [self._database_original])
                require_certain_releases(self)
            except BaseException as cleanup:  # noqa: BLE001 -- preserve primary and cleanup owner
                exc.__dict__["cleanup_owner"] = self
                exc.add_note(f"descriptor cleanup remains owned ({type(cleanup).__name__})")
            raise

    def seal(self, *, publication: str = "published") -> CollectionLocator:
        descriptor_name(publication)
        with self._pending:
            if self._sealing or self._closing or self._stopped:
                raise RuntimeError("Writer no longer admits sealing")
            self._sealing = True
            try:
                while self._pending_count:
                    self._pending.wait(remaining(self.deadline))
            except BaseException as exc:
                self._fail(exc)
                raise
        with self._operation():
            try:
                descriptor = validate_descriptor(self._rpc("seal", {"publication": publication}))
                assert self._database_original is not None
                self._database_original.freeze()
                self._retire()
                self._database_original.recheck()
                if self.returncode != 0:
                    raise RuntimeError("Writer exit was not healthy")
                if (
                    self._database_original.sha256 != descriptor["database_sha256"]
                    or self._database_original.identity.st_size != descriptor["database_bytes"]
                    or descriptor["run_id"] != self.run_id
                ):
                    raise ValueError("sealed descriptor differs from held original DB")
                locator = self._publish(descriptor)
                self._release_base()
                self._sealed = True
                self._stopped = True
                self._queue.stop()
                self._budget.stop()
                return locator
            except BaseException as exc:
                self._fail(exc)
                raise

    def close(self) -> None:
        # Stop callers waiting on byte admission before waiting on the operation lock.
        with self._pending:
            self._closing = True
            inflight = bool(self._pending_count)
        self._queue.stop()
        self._budget.stop()
        if inflight:
            # Cancellation interrupts the child's pipe operation instead of waiting
            # behind a stopped child for the full collection deadline.
            self._retire(terminate=True)
        if not self._lock.acquire(timeout=max(0.01, self.deadline - time.monotonic())):
            raise CleanupError("Writer operation still owns cleanup", self)
        try:
            if self._process is not None and self._process.poll() is None:
                if self._poisoned or self._stopped:
                    self._retire(terminate=True)
                else:
                    try:
                        self._rpc("close")
                        self._retire()
                        if self.returncode != 0:
                            raise RuntimeError("Writer close exit was not healthy")
                    except BaseException as exc:
                        self._fail(exc)
                        raise
            else:
                self._retire(terminate=True)
            self._stopped = True
            self._cleanup_physical()
            self._release_base()
        except BaseException as error:
            retain_cleanup(self, error)
            raise
        finally:
            self._lock.release()
        with self._pending:
            while self._pending_count:
                self._pending.wait(remaining(self.deadline))
