"""Collection-owned memory/RAM groups and one retained synchronous Writer worker."""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import GroupResult, canonical
from gf_common.group_pack import (
    MIB,
    PackMember,
    StorageSettings,
    canonical_json,
    digest,
    encode_pack,
    member_index,
    parse_json,
)
from gf_training.group_pack_memory import MemoryArtifactSnapshot, MemoryGroupArtifacts
from gf_training.group_pack_resources import (
    CleanupError,
    StorageBudget,
    StorageReservation,
    close_resources,
    json_bound,
    remaining,
    retain_cleanup,
)
from gf_training.group_pack_source import OriginalFile, _no_symlink_path
from gf_training.group_pack_writer import GroupPackWriter, PendingDescriptorOwner

TERMINAL_CONTROL_POLICY = "terminal-control-ro-handoff-v1"


@dataclass
class MemoryGroup:
    sink: MemoryGroupArtifacts
    server_root: Path
    snapshot: MemoryArtifactSnapshot | None = None
    archived: bool = False
    runtime_owner: object | None = None


def _kind(name: str) -> str:
    if name.startswith("server-"):
        return "http-log" if "http" in name else "simulator"
    if name.endswith(".columns"):
        return "candidates"
    if name.endswith(".jsonl"):
        return "journal"
    if "/replay-player-" in name:
        return "replay"
    if name.endswith("/result.json"):
        return "result"
    return "diagnostic"


class GroupPackCollection:
    def __init__(self, settings: StorageSettings, run_id: str, *, deadline: float,
                 owned_directory: Path | None = None) -> None:
        from gf_common.candidate_columns import safe_basename
        from gf_training.group_pack_db import identity_text
        identity_text(run_id)
        safe_basename(run_id)
        remaining(deadline)
        if not settings.db_root.is_absolute() or not settings.ram_root.is_absolute():
            raise ValueError("storage roots must be absolute")
        self.settings, self.run_id, self.deadline = settings, run_id, deadline
        self.budget = StorageBudget(settings, deadline)
        self.directory, self.ram_root = settings.db_root / run_id, settings.ram_root / run_id
        if owned_directory is not None:
            if not owned_directory.is_absolute() or not owned_directory.is_relative_to(settings.db_root):
                raise ValueError('foreign packed persistent collection root')
            self.directory = owned_directory
        self.owned_roots = (self.directory, settings.db_root / 'catalog.sqlite3',
            settings.db_root / 'catalog.sqlite3-wal', settings.db_root / 'catalog.sqlite3-shm', settings.db_root / 'catalog.sqlite3-journal')
        self.writer: GroupPackWriter | None = None
        self.pending: PendingDescriptorOwner | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="group-pack")
        self._futures: set[concurrent.futures.Future[Any]] = set()
        self._future_lock = threading.Lock()
        self._groups: list[MemoryGroup] = []
        self._monitor: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._failure: BaseException | None = None
        self._closed = False
        self._admission_closed = False
        self.inventory: list[dict[str, Any]] = []
        self._metadata_reservations: list[StorageReservation] = []
        self._storage_metrics_bytes: bytes | None = None
        self._storage_metrics_row: dict[str, Any] | None = None
        self._storage_metrics_lock = threading.RLock()
        self._operation_lock = asyncio.Lock()
        self._cleanup_children: list[Any] = []
        self._validation_failures: list[BaseException] = []
        self._operation_errors: list[BaseException] = []
        self._retained_workspaces: list[StorageReservation] = []
        self._archival_tasks: set[asyncio.Task[None]] = set()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float | int | None] = {
            "worker_wait_seconds": 0.0, "write_seconds": 0.0,
            "pack_encode_seconds": 0.0, "checkpoint_and_seal_seconds": None,
            "checkpoint_seconds": None, "delete_mode_seconds": None,
            "peak_ram_bytes": 0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {**self._metrics, "peak_storage_bytes": self.budget.peak_bytes,
                    "terminal_control_policy": TERMINAL_CONTROL_POLICY}

    def _metric_add(self, name: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[name] = (self._metrics[name] or 0) + value

    def _record_error(self, error: BaseException) -> None:
        with self._future_lock:
            self._admission_closed = True
            self._failure = self._failure or error
            if all(item is not error for item in self._operation_errors):
                self._operation_errors.append(error)
        retain_cleanup(self, error)

    def _check_submission(self, *, recovery: bool, prior: bool = True) -> None:
        if self._closed:
            raise RuntimeError("collection worker retired")
        remaining(self.deadline)
        with self._future_lock:
            if prior and any(not future.done() for future in self._futures):
                raise CleanupError("prior actual storage operation has not retired", self)
        if self._failure is not None and not recovery:
            raise RuntimeError("ordinary storage admission closed after failure") from self._failure

    async def run_sync(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._admission_closed:
            raise CleanupError("ordinary storage admission closed", self) from self._failure
        return await self._serialized(operation, args, kwargs, recovery=False)

    async def _serialized(self, operation: Callable[..., Any], args: tuple[Any, ...],
                          kwargs: dict[str, Any], *, recovery: bool) -> Any:
        started = time.monotonic()
        try:
            self._check_submission(recovery=recovery, prior=False)
            async with asyncio.timeout_at(self.deadline):
                await self._operation_lock.acquire()
            try:
                self._check_submission(recovery=recovery)
                self._metric_add("worker_wait_seconds", time.monotonic() - started)
                return await self._run_sync(operation, *args, **kwargs)
            finally:
                self._operation_lock.release()
        except BaseException as error:
            self._record_error(error)
            raise

    async def _run_sync(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        # All checks precede submit; no cleanup is queued behind uncertain work.
        remaining(self.deadline)
        future = self._executor.submit(functools.partial(operation, *args, **kwargs))
        with self._future_lock:
            self._futures.add(future)
        try:
            while not future.done():
                remaining(self.deadline)
                await asyncio.sleep(0.01)
            return future.result()
        except asyncio.CancelledError as initiating:
            self._record_error(initiating)
            try:
                while not future.done():
                    remaining(self.deadline)
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        continue
                future.result()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve primary and actual work owner.
                initiating.add_note("actual storage operation cleanup: " + type(cleanup).__name__)
            raise
        except BaseException as error:
            self._record_error(error)
            raise
        finally:
            if future.done():
                with self._future_lock:
                    self._futures.discard(future)

    async def _join_owned_task(self, task: asyncio.Task[Any]) -> None:
        while not task.done():
            remaining(self.deadline)
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError as error:
                self._record_error(error)
        task.result()

    async def start(self) -> None:
        def initialize() -> None:
            for root in (self.settings.db_root, self.settings.ram_root):
                ancestor = root
                while not ancestor.exists():
                    ancestor = ancestor.parent
                _no_symlink_path(ancestor)
                root.mkdir(parents=True, exist_ok=True)
                _no_symlink_path(root)
            self.ram_root.mkdir(mode=0o700, exist_ok=False)
            self.writer = GroupPackWriter(self.directory, self.run_id, settings=self.settings,
                deadline=self.deadline, budget=self.budget)
            self.measure_ram()
        try:
            await self.run_sync(initialize)
            self._monitor = asyncio.create_task(self._monitor_ram())
        except BaseException as error:
            error.__dict__["cleanup_owner"] = self
            raise

    def measure_ram(self) -> None:
        total = 0
        if self.ram_root.exists():
            for directory, dirs, files in os.walk(self.ram_root, followlinks=False):
                for name in (*dirs, *files):
                    try:
                        info = (Path(directory) / name).lstat()
                    except FileNotFoundError:
                        # Active Simulator temporary renames and our ACK-retired
                        # group removals may race a monitoring pass.
                        continue
                    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                        raise ValueError("unsafe owned Simulator RAM artifact")
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
        self.budget.measure_ram(total)
        with self._metrics_lock:
            self._metrics["peak_ram_bytes"] = max(total, self._metrics["peak_ram_bytes"] or 0)
        space = os.statvfs(self.settings.ram_root)
        # Existing accounted RAM is already allocated; all other reservations need headroom.
        if space.f_bavail * space.f_frsize < self.budget.used - self.budget.ram_bytes:
            raise ValueError("shared RAM storage headroom exhausted")

    async def _monitor_ram(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
                self.measure_ram()
        except asyncio.CancelledError:
            return
        except BaseException as error:  # noqa: BLE001 -- all evidence and cleanup ownership remain reachable.
            self._record_error(error)
            for task in tuple(self._active_tasks):
                task.cancel()

    def track_group_task(self, task: asyncio.Task[Any]) -> None:
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    def new_group(self, first_episode: str) -> MemoryGroup:
        from gf_common.candidate_columns import safe_basename
        safe_basename(first_episode)
        if self._admission_closed or self._closed or self.writer is None:
            raise RuntimeError("collection storage admission closed") from self._failure
        self.measure_ram()
        self._metadata_reservations.append(self.budget.reserve(65536))
        root = self.ram_root / first_episode
        root.mkdir(mode=0o700, exist_ok=False)
        group = MemoryGroup(MemoryGroupArtifacts(self.budget), root)
        self._groups.append(group)
        return group

    def reserve_metadata(self, size: int) -> None:
        self._metadata_reservations.append(self.budget.reserve(size))

    def freeze_storage_metrics(self) -> dict[str, Any]:
        """Declare the exact pre-checkpoint control bytes before inventory closes."""
        with self._storage_metrics_lock:
            if self._closed:
                raise RuntimeError("storage metrics owner already retired")
            match = None
            for entry in self.inventory:
                if entry.get("path") == "storage-metrics.json":
                    if match is not None:
                        raise ValueError("storage metrics inventory duplicated")
                    match = entry
            if self._storage_metrics_bytes is not None:
                row = self._storage_metrics_row
                if (row is None or match is None or match != row or type(match.get("bytes")) is not int
                        or row["sha256"] != digest(self._storage_metrics_bytes)
                        or row["bytes"] != len(self._storage_metrics_bytes)):
                    raise ValueError("frozen storage metrics inventory changed or duplicated")
                return dict(row)
            if match is not None:
                raise ValueError("storage metrics inventory name collision")
            self.reserve_metadata(65536)  # Snapshot/JSON/copies/row admitted BEFORE allocation.
            data = (canonical({"schema_version": 1, "measured_before_checkpoint": True,
                               **self.metrics}) + "\n").encode()
            row = {"path": "storage-metrics.json", "sha256": digest(data), "bytes": len(data)}
            self._storage_metrics_bytes, self._storage_metrics_row = data, row
            self.inventory.append(dict(row))
            return dict(row)

    def _metrics_for_prepare(self, controls: dict[str, bytes], *, diagnostic: bool) -> bytes:
        if "storage-metrics.json" in controls:
            raise ValueError("storage metrics control name collision")
        if not diagnostic and self._storage_metrics_bytes is None:
            raise ValueError("healthy storage metrics must freeze before manifest serialization")
        row = self.freeze_storage_metrics()
        if not diagnostic:
            raw = controls.get("manifest.json")
            if type(raw) is not bytes:
                raise ValueError("healthy storage metrics require original manifest bytes")
            self.reserve_metadata(32 * len(raw) + 65536)
            manifest = parse_json(raw, canonical=False,
                limit=min(self.settings.max_pack_bytes, 64 * MIB))
            files = manifest.get("files") if isinstance(manifest, dict) else None
            entries = ([entry for entry in files if isinstance(entry, dict)
                        and entry.get("path") == "storage-metrics.json"]
                       if isinstance(files, list) else [])
            if len(entries) != 1 or entries[0] != row or type(entries[0].get("bytes")) is not int:
                raise ValueError("healthy manifest storage metrics row is missing, duplicated or changed")
        assert self._storage_metrics_bytes is not None
        return self._storage_metrics_bytes

    def _archive(self, group: MemoryGroup, group_id: str, result: GroupResult,
                 validate: Callable[[MemoryArtifactSnapshot], None] | None) -> None:
        if group.archived or self.writer is None:
            raise RuntimeError("group archival already consumed")
        self.measure_ram()
        # This boundary is called only after the existing environment confirms reap/drain.
        for directory, dirs, files in os.walk(group.server_root, followlinks=False):
            for name in (*dirs, *files):
                path = Path(directory) / name
                before = path.lstat()
                if stat.S_ISDIR(before.st_mode):
                    continue
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("nonregular Simulator artifact")
                temporary = self.budget.reserve(2 * before.st_size + 65536)
                source: OriginalFile | None = None
                try:
                    source = OriginalFile(path, self.deadline)
                    data = source.read(self.settings.max_pack_bytes)
                    group.sink.write(path.relative_to(group.server_root).as_posix(), data)
                    source.recheck()
                    source.close()
                except BaseException as error:
                    if source is not None:
                        self._cleanup_children.append(source)
                    retain_cleanup(self, error)
                    group.sink._adopt(temporary)
                    raise
                else:
                    del data
                    temporary.close()
        snapshot = group.snapshot = group.sink.snapshot()
        validation_failure: BaseException | None = None
        if validate is not None:
            try:
                validate(snapshot)
            except BaseException as error:  # noqa: BLE001 -- all evidence and cleanup ownership remain reachable.
                validation_failure = error
                result.health["storage_validation_failure"] = type(error).__name__
            else:
                snapshot.retire_validation()
        names = snapshot.names()
        inventory_reservation = self.budget.reserve(4096 * len(names)
            + 32 * sum(len(name) for name in names)
            + 32 * sum(len(snapshot.read(name)) for name in names if name.endswith("/result.json"))
            + 32 * json_bound(result.health) + 4096 * len(result.episodes) + 65536)
        self._metadata_reservations.append(inventory_reservation)
        try:
            inventory = self._encode_submit(snapshot, group_id, result)
        except BaseException as archive_error:
            if validation_failure is not None:
                validation_failure.add_note("diagnostic archival failed: " + str(archive_error))
                raise validation_failure from archive_error
            raise
        self.inventory.extend(inventory)
        # The encoder has dropped member/Pack/index aliases BEFORE original bytes retire.
        if validation_failure is None:
            snapshot.close()
        group.sink.close()
        shutil.rmtree(group.server_root)
        group.archived = True
        self.measure_ram()
        if validation_failure is not None:
            self._validation_failures.append(validation_failure)
            raise validation_failure

    def _encode_submit(self, snapshot: MemoryArtifactSnapshot, group_id: str,
                       result: GroupResult) -> list[dict[str, Any]]:
        assert self.writer is not None
        names = snapshot.names()
        index_bound = sum(6 * len(name) + 512 for name in names) + 65536
        payload_size = sum(len(snapshot.read(name)) for name in names)
        workspace = self.budget.reserve(32 * index_bound + 4 * payload_size + 2 * MIB)
        try:
            members = tuple(PackMember(_kind(name), name, snapshot.read(name)) for name in names)
            index = canonical_json({"format": "group-pack-v1", "files": member_index(members)})
            raw_size = 16 + len(index) + payload_size
            if raw_size > self.settings.max_pack_bytes:
                raise ValueError("group Pack exceeds configured bound")
            encode_started = time.monotonic()
            pack = encode_pack(members, self.settings.max_pack_bytes)
            self._metric_add("pack_encode_seconds", time.monotonic() - encode_started)
            write_started = time.monotonic()
            self.writer.submit(group_id, pack, raw_bytes=raw_size, record=result.health,
                episodes=[{"episode_id": episode.episode_id, "game_id": episode.game_id,
                    "record": {"end_kind": episode.end_kind, "failure": episode.failure}}
                    for episode in result.episodes])
            self._metric_add("write_seconds", time.monotonic() - write_started)
            inventory = [{"path": name, "sha256": digest(snapshot.read(name)),
                "bytes": len(snapshot.read(name))} for name in names]
        except BaseException:
            # Completed internal traceback frames still own the actual Pack/index.
            self._retained_workspaces.append(workspace)
            raise
        else:
            del members, index, pack
            workspace.close()
            return inventory

    async def archive(self, group: MemoryGroup, group_id: str, result: GroupResult, *,
                      validate: Callable[[MemoryArtifactSnapshot], None] | None = None) -> None:
        # Establish actual asyncio ownership BEFORE waiting for the serialization lock.
        task = asyncio.create_task(self._serialized(self._archive,
            (group, group_id, result, validate), {}, recovery=True))
        self._archival_tasks.add(task)
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as initiating:
            self._record_error(initiating)
            try:
                await self._join_owned_task(task)
            except BaseException as cleanup:  # noqa: BLE001 -- obtainable archival never replaces original cancellation.
                initiating.add_note("owned queued archival: " + str(cleanup))
            raise
        finally:
            if task.done():
                self._archival_tasks.discard(task)
                if not task.cancelled():
                    task.exception()

    async def prepare(self, controls: dict[str, bytes], *, diagnostic: bool = False) -> PendingDescriptorOwner:
        self._admission_closed = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        if self._failure is not None and not diagnostic:
            raise self._failure
        try:
            metrics_bytes = self._metrics_for_prepare(controls, diagnostic=diagnostic)
        except BaseException as error:
            self._record_error(error)  # Retire completed validation frames with their admitted graph charge.
            raise
        def seal() -> PendingDescriptorOwner:
            assert self.writer is not None
            for name, data in controls.items():
                self.writer.store_control(name, data)
            # Immutable controls record pre-checkpoint measurements. The final
            # seal duration is exposed by this owner after the child has retired.
            self.writer.store_control("storage-metrics.json", metrics_bytes)
            seal_started = time.monotonic()
            self.writer.seal(publication="pending", retain_pending=True, collect_metrics=True)
            assert self.writer.checkpoint_metrics is not None
            with self._metrics_lock:
                self._metrics.update(self.writer.checkpoint_metrics)
            self._metric_add("checkpoint_and_seal_seconds", time.monotonic() - seal_started)
            # Store the actual transfer before a cancelled asyncio caller can
            # lose delivery of this synchronous operation's result.
            self.pending = self.writer.take_pending_owner()
            return self.pending
        pending = self.pending = await self._serialized(seal, (), {}, recovery=diagnostic)
        await self._retire_executor()
        return pending

    async def prepare_diagnostic(self, name: str, value: Any) -> PendingDescriptorOwner:
        self.reserve_metadata(32 * json_bound(value) + 65536)
        return await self.prepare({name: (canonical(value) + "\n").encode()}, diagnostic=True)

    async def _retire_executor(self) -> None:
        with self._future_lock:
            if any(not future.done() for future in self._futures):
                raise CleanupError("actual storage worker still running", self)
        self._executor.shutdown(wait=True, cancel_futures=False)
        for future in self._futures:
            if not future.cancelled() and (error := future.exception()) is not None:
                self._record_error(error)
        self._futures.clear()
        self._closed = True

    def retire_metadata(self) -> None:
        if not self._closed:
            raise CleanupError("storage metadata still has a live worker", self)
        self.inventory.clear()
        self._storage_metrics_bytes = self._storage_metrics_row = None
        for reservation in self._metadata_reservations:
            reservation.close()
        self._metadata_reservations.clear()

    def _retire_internal_graphs(self) -> None:
        seen: set[int] = set()
        def clear(error: BaseException) -> None:
            if id(error) in seen:
                return
            seen.add(id(error))
            tb = error.__traceback__
            while tb is not None:
                frame = tb.tb_frame
                module = frame.f_globals.get("__name__", "")
                internal = (module.startswith("gf_training.group_pack_")
                    or module in {"gf_training.structured_journal", "gf_training.card_instance_journal",
                        "gf_training.journal", "gf_common.group_pack", "gf_common.candidate_columns", "gf_common.structured",
                        "gf_common.structured_wire", "gf_common.card_instance_validation", "gf_common.validation",
                        "gf_models.structured_binding", "gf_models.card_instance_artifacts",
                        "gf_models.card_instance_game_tree_runtime", "gf_models.structured_runtime"}
                    or module == "gf_cli.card_instance" and frame.f_code.co_name in
                        {"validate", "_validate_recorded_episodes"})
                if internal:
                    try:
                        frame.clear()
                    except RuntimeError:
                        pass  # Active/suspended frames are never cleared.
                tb = tb.tb_next
            for nested in (error.__cause__, error.__context__):
                if nested is not None:
                    clear(nested)
        for error in self._operation_errors:
            clear(error)
        for workspace in self._retained_workspaces:
            workspace.close()
        self._retained_workspaces.clear()
        for group in self._groups:
            if group.archived and group.snapshot is not None:
                group.snapshot.close()
                group.sink.close()
        self._validation_failures.clear()
        self._operation_errors.clear()

    async def close(self) -> None:
        self._admission_closed = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        try:
            for task in tuple(self._archival_tasks):
                try:
                    await self._join_owned_task(task)
                except BaseException as error:
                    if not task.done():
                        raise CleanupError("queued archival retirement unconfirmed", self) from error
                self._archival_tasks.discard(task)
            while any(not future.done() for future in self._futures):
                try:
                    remaining(self.deadline)
                except TimeoutError as error:
                    raise CleanupError("actual storage worker still running", self) from error
                await asyncio.sleep(0.01)
            if not self._closed:
                # No post-expiry executor submit: prior work is ACTUALLY finished.
                await self._retire_executor()
                if self.writer is not None:
                    self.writer.close(_abort=time.monotonic() >= self.deadline)
            if self.pending is not None:
                self.pending.close()
            close_resources(self, [])
            self._retire_internal_graphs()
            if any(not group.archived for group in self._groups):
                raise CleanupError("unarchived group diagnostics retained", self)
            self.retire_metadata()
        except BaseException as error:
            retain_cleanup(self, error)
            raise
