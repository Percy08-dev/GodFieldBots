"""Retained original-file custody and bounded group loans, without learning authority."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gf_common.group_pack import (
    FORMAT,
    MAX_JSON,
    MIB,
    Json,
    canonical_json,
    decode_pack,
    parse_json,
    safe_name,
)
from gf_training.group_pack_db import (
    DATABASE,
    descriptor_name,
    validate_database,
    validate_descriptor,
)
from gf_training.group_pack_resources import (
    SOURCE_BASE,
    ByteBudget,
    OwnedDescriptor,
    close_resources,
    remaining,
    reserve,
    retain_cleanup,
)


@dataclass(frozen=True)
class CollectionLocator:
    kind: str
    path: Path

    def __post_init__(self) -> None:
        if self.kind not in ("directory", FORMAT) or not isinstance(self.path, Path):
            raise ValueError("unsupported collection locator")


class ArtifactSource(Protocol):
    def names(self) -> tuple[str, ...]: ...
    def read(self, name: str) -> bytes: ...
    def inventory(self) -> tuple[Json, ...]: ...
    def recheck(self) -> None: ...
    def borrow_group(self, group_id: str) -> AbstractContextManager[GroupLoan]: ...
    def close(self) -> None: ...


def _stat_key(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _no_symlink_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("symlink in original source path")
    return absolute


class OriginalFile(OwnedDescriptor):
    def __init__(
        self, path: Path, deadline: float, *, mutable: bool = False, create: bool = False
    ) -> None:
        super().__init__(-1)
        self.deadline = deadline
        self.mutable = mutable
        self.sha256 = ""
        try:
            self.path = _no_symlink_path(path.parent) / path.name
            flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_RDWR if create else os.O_RDONLY)
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            self.fd = os.open(self.path, flags, 0o600)
            self.identity = os.fstat(self.fd)
            if not stat.S_ISREG(self.identity.st_mode):
                raise ValueError("original source is not a regular file")
            if not mutable:
                self.sha256 = self.hash()
            self.recheck()
        except BaseException as error:
            retain_cleanup(self, error)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve admission error
                retain_cleanup(self, cleanup)
                error.add_note(f"original cleanup remains owned ({type(cleanup).__name__})")
            raise

    def freeze(self) -> None:
        self.check_identity()
        self.identity = os.fstat(self.fd)
        self.mutable = False
        self.sha256 = self.hash()
        self.check_identity()

    def hash(self) -> str:
        result = hashlib.sha256()
        offset = 0
        while True:
            remaining(self.deadline)
            chunk = os.pread(self.fd, 64 * 1024, offset)
            if not chunk:
                break
            offset += len(chunk)
            result.update(chunk)
        return result.hexdigest()

    def check_path_identity(self) -> None:
        """A pathname-only final check never uses a terminally retired raw number."""
        _no_symlink_path(self.path)
        current = self.path.lstat()
        if self.mutable:
            if (current.st_dev, current.st_ino) != (self.identity.st_dev, self.identity.st_ino):
                raise ValueError("writing database inode changed")
        elif _stat_key(current) != _stat_key(self.identity):
            raise ValueError("original file path identity changed")

    def check_identity(self) -> None:
        if self.fd < 0:
            raise RuntimeError("original file closed")
        self.check_path_identity()
        held = os.fstat(self.fd)
        if self.mutable:
            if (held.st_dev, held.st_ino) != (self.identity.st_dev, self.identity.st_ino):
                raise ValueError("writing database inode changed")
        elif _stat_key(held) != _stat_key(self.identity):
            raise ValueError("original file identity changed")

    def recheck(self) -> None:
        self.check_identity()
        if not self.mutable and self.hash() != self.sha256:
            raise ValueError("original file content changed")
        self.check_identity()

    def read(self, limit: int) -> bytes:
        if self.identity.st_size > limit:
            raise ValueError("original file exceeds admission")
        self.recheck()
        parts = []
        offset = 0
        while offset < self.identity.st_size:
            remaining(self.deadline)
            chunk = os.pread(self.fd, min(64 * 1024, self.identity.st_size - offset), offset)
            if not chunk:
                raise ValueError("original file truncated")
            parts.append(chunk)
            offset += len(chunk)
        result = b"".join(parts)
        self.recheck()
        return result


class PackedOriginals:
    def __init__(
        self,
        path: Path,
        *,
        deadline: float,
        max_owned_bytes: int,
        allow_pending: bool = False,
        without_descriptor: bool = False,
        database_original: OriginalFile | None = None,
    ) -> None:
        self.deadline = deadline
        self._cleanup_children: list[Json] = []
        self._owns_database = database_original is None
        self.budget = ByteBudget(max_owned_bytes, base=SOURCE_BASE)
        self.directory = _no_symlink_path(path.parent)
        self.directory_identity = self.directory.stat()
        self.descriptor_file: OriginalFile | None = None
        self.database_file: OriginalFile | None = None
        self.db: sqlite3.Connection | None = None
        self.descriptor: Json = None
        self.validated: Json = None
        self.namespace = {DATABASE} if without_descriptor else {DATABASE, path.name}
        self.metadata_charge = 0
        try:
            self.check_namespace()
            if not without_descriptor:
                self.descriptor_file = OriginalFile(path, deadline)
                # Descriptor metadata and decoded controls index have a 32x charge.
                self.metadata_charge = 32 * self.descriptor_file.identity.st_size + 65536
                self.budget.acquire(self.metadata_charge, deadline)
                self.descriptor = validate_descriptor(
                    parse_json(self.descriptor_file.read(MAX_JSON), limit=MAX_JSON)
                )
                if path.name != descriptor_name(self.descriptor["publication"]):
                    raise ValueError("descriptor filename mismatch")
                if self.descriptor["publication"] == "pending" and not allow_pending:
                    raise ValueError("pending physical descriptor has no public Source authority")
            if database_original is None:
                self.database_file = OriginalFile(self.directory / DATABASE, deadline)
            else:
                self.database_file = database_original
                if database_original.mutable:
                    raise ValueError("readonly inspection requires frozen original DB")
                database_original.recheck()
            if self.descriptor is not None and (
                self.database_file.identity.st_size != self.descriptor["database_bytes"]
                or self.database_file.sha256 != self.descriptor["database_sha256"]
            ):
                raise ValueError("descriptor database identity mismatch")
            header = os.pread(self.database_file.fd, 100, 0)
            if (
                header[:16] != b"SQLite format 3\x00"
                or len(header) < 100
                or header[18:20] != b"\x01\x01"
            ):
                raise ValueError("database is not a sidecar-free DELETE seal")
            # Open the SAME retained inode, even if its pathname races replacement.
            uri = f"file:/proc/self/fd/{self.database_file.fd}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, timeout=min(5.0, remaining(deadline)))
            self.db.execute("PRAGMA query_only=ON")
            self.db.execute("PRAGMA cache_size=-32768")
            self.validated = validate_database(
                self.db,
                deadline=deadline,
                max_pack_bytes=256 * MIB,
                max_owned_bytes=self.budget.limit - self.budget.used,
            )
            extra_charge = self.validated["metadata_charge"]
            self.budget.acquire(extra_charge, deadline)
            self.metadata_charge += extra_charge
            if self.descriptor is not None:
                seal = self.validated["seal"]
                if (
                    seal["seal_id"] != self.descriptor["seal_id"]
                    or seal["run_id"] != self.descriptor["run_id"]
                    or self.validated["index"]["controls"] != self.descriptor["controls"]
                ):
                    raise ValueError("descriptor logical identity mismatch")
            self.recheck()
        except BaseException as exc:
            retain_cleanup(self, exc)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve original admission error
                exc.__dict__["cleanup_owner"] = self
                exc.add_note(f"originals cleanup remains owned ({type(cleanup).__name__})")
            raise

    def check_namespace(self) -> None:
        _no_symlink_path(self.directory)
        current = self.directory.stat()
        if (current.st_dev, current.st_ino) != (
            self.directory_identity.st_dev,
            self.directory_identity.st_ino,
        ):
            raise ValueError("physical collection directory changed")
        with reserve(self.budget, 65536, self.deadline):
            seen: set[str] = set()
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    remaining(self.deadline)
                    if entry.name not in self.namespace or entry.name in seen:
                        raise ValueError("physical collection namespace changed")
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError("unexpected physical namespace object")
                    seen.add(entry.name)
            if seen != self.namespace:
                raise ValueError("physical collection namespace changed")

    def recheck(self) -> None:
        self.check_namespace()
        if self.database_file is not None:
            self.database_file.recheck()
        if self.descriptor_file is not None:
            self.descriptor_file.recheck()
        self.check_namespace()
        if self.database_file is not None:
            self.database_file.check_identity()
        if self.descriptor_file is not None:
            self.descriptor_file.check_identity()

    def close(self) -> None:
        close_resources(
            self,
            [self.db, self.database_file if self._owns_database else None, self.descriptor_file],
        )
        self.db = None
        self.budget.release(self.budget.used)


def inspect_packed(
    path: Path,
    *,
    deadline: float,
    max_owned_bytes: int,
    allow_pending: bool = False,
    without_descriptor: bool = False,
    database_original: OriginalFile | None = None,
) -> Json:
    originals = PackedOriginals(
        path,
        deadline=deadline,
        max_owned_bytes=max_owned_bytes,
        allow_pending=allow_pending,
        without_descriptor=without_descriptor,
        database_original=database_original,
    )
    try:
        result = {
            "descriptor": originals.descriptor,
            "validated": originals.validated,
            "database_sha256": originals.database_file.sha256 if originals.database_file else None,
            "database_bytes": originals.database_file.identity.st_size
            if originals.database_file
            else None,
        }
    except BaseException as error:
        retain_cleanup(originals, error)
        try:
            originals.close()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve inspection failure
            error.add_note(f"inspection cleanup remains owned ({type(cleanup).__name__})")
        raise
    else:
        originals.close()
        return result


class GroupLoan:
    def __init__(
        self,
        source: PackedSource | DirectorySource,
        group_id: str,
        generation: int,
        names: tuple[str, ...],
    ) -> None:
        self._source = source
        self._group_id = group_id
        self._generation = generation
        self._names = names
        self._active = True

    def _check(self) -> None:
        if (
            not self._active
            or self._source._closed
            or self._source._generation != self._generation
            or self._source._group_id != self._group_id
        ):
            raise RuntimeError("stale or retired group loan")
        self._source.recheck()

    def names(self) -> tuple[str, ...]:
        self._check()
        return self._names

    def read(self, name: str) -> bytes:
        self._check()
        if name not in self._names:
            raise KeyError(name)
        return self._source._loan_read(name)


class PackedSource:
    def __init__(self, path: Path, *, deadline: float, max_owned_bytes: int) -> None:
        self._originals = PackedOriginals(path, deadline=deadline, max_owned_bytes=max_owned_bytes)
        self._deadline = deadline
        self._budget = self._originals.budget
        self._closed = False
        self._generation = 0
        self._group_id: str | None = None
        self._loans = 0
        self._cache: dict[str, bytes] = {}
        self._cache_charge = 0
        self._detached_charge = 0
        self._index = self._originals.validated["index"]
        self._names = tuple(
            sorted(
                [m["name"] for m in self._index["members"]]
                + [c["name"] for c in self._index["controls"]]
            )
        )
        self._member_groups = {m["name"]: m["group_id"] for m in self._index["members"]}
        self._groups = {g["group_id"]: g for g in self._index["groups"]}
        self._controls = {c["name"]: c for c in self._index["controls"]}

    def _live(self) -> None:
        remaining(self._deadline)
        if self._closed:
            raise RuntimeError("artifact Source closed")

    @property
    def logical_index(self) -> Json:
        self.recheck()
        charge = self._originals.metadata_charge
        self._budget.acquire(charge, self._deadline)
        try:
            result = parse_json(canonical_json(self._index, limit=charge), limit=charge)
        except BaseException:
            self._budget.release(charge)
            raise
        self._detached_charge += charge
        return result

    @property
    def owned_bytes(self) -> int:
        return self._budget.used

    def names(self) -> tuple[str, ...]:
        self.recheck()
        return self._names

    def inventory(self) -> tuple[Json, ...]:
        self.recheck()
        charge = self._originals.metadata_charge
        self._budget.acquire(charge, self._deadline)
        try:
            values = [
                {
                    "name": m["name"],
                    "bytes": m["length"],
                    "sha256": m["sha256"],
                    "group_id": m["group_id"],
                    "kind": m["kind"],
                }
                for m in self._index["members"]
            ]
            values += [dict(c) for c in self._index["controls"]]
            result = tuple(sorted(values, key=lambda m: m["name"]))

        except BaseException:
            self._budget.release(charge)
            raise
        self._detached_charge += charge
        return result

    def recheck(self) -> None:
        self._live()
        self._originals.recheck()

    def evict_group(self) -> None:
        self._live()
        if self._loans:
            raise RuntimeError("cannot evict a group with live loans")
        self._cache.clear()
        self._group_id = None
        self._generation += 1
        self._budget.release(self._cache_charge)
        self._cache_charge = 0

    def _load(self, group_id: str) -> None:
        self.recheck()
        if self._group_id == group_id:
            return
        if group_id not in self._groups:
            raise KeyError(group_id)
        self.evict_group()
        db = self._originals.db
        assert db is not None
        group = self._groups[group_id]
        size = db.execute(
            "SELECT length(pack) FROM groups WHERE group_id=?", (group_id,)
        ).fetchone()[0]
        charge = 4 * group["raw_bytes"] + 2 * size
        self._budget.acquire(charge, self._deadline)
        try:
            pack = db.execute("SELECT pack FROM groups WHERE group_id=?", (group_id,)).fetchone()[0]
            self._cache = {m.name: m.data for m in decode_pack(pack, group["raw_bytes"])}
            self.recheck()
        except BaseException:
            self._cache.clear()
            self._budget.release(charge)
            raise
        self._cache_charge = charge
        self._group_id = group_id
        self._generation += 1

    @contextmanager
    def borrow_group(self, group_id: str) -> Iterator[GroupLoan]:
        self._load(group_id)
        loan = GroupLoan(self, group_id, self._generation, tuple(self._cache))
        self._loans += 1
        try:
            yield loan
        finally:
            loan._active = False
            self._loans -= 1

    def _loan_read(self, name: str) -> bytes:
        return self._cache[name]

    def read(self, name: str) -> bytes:
        self.recheck()
        if name in self._controls:
            size = self._controls[name]["bytes"]
        elif name in self._member_groups:
            size = next(m["length"] for m in self._index["members"] if m["name"] == name)
        else:
            raise KeyError(name)
        # Returned immutable bytes are conservatively retained until Source close.
        charge = 2 * size
        self._budget.acquire(charge, self._deadline)
        try:
            if name in self._controls:
                db = self._originals.db
                assert db is not None
                result = db.execute("SELECT data FROM controls WHERE name=?", (name,)).fetchone()[0]
            else:
                with self.borrow_group(self._member_groups[name]) as loan:
                    result = loan.read(name)
            self.recheck()
        except BaseException:
            self._budget.release(charge)
            raise
        self._detached_charge += charge
        return result

    def close(self) -> None:
        if self._closed:
            return
        if self._loans:
            raise RuntimeError("cannot close a Source with live group loans")
        try:
            self._originals.close()
        except BaseException as exc:
            exc.__dict__["cleanup_owner"] = self
            raise
        self._cache.clear()
        self._budget.release(self._budget.used)
        self._closed = True
        self._generation += 1


class DirectorySource:
    """Original bytes for legacy validators; does not invent group metadata."""

    def __init__(self, path: Path, *, deadline: float, max_owned_bytes: int) -> None:
        self.directory = _no_symlink_path(path)
        self._deadline = deadline
        self._budget = ByteBudget(max_owned_bytes)
        self._closed = False
        self._generation = 0
        self._group_id: str | None = None
        self._loans = 0
        self._cache: dict[str, bytes] = {}
        self._cache_charge = 0
        self._cleanup_children: list[Json] = []
        self._files: dict[str, OriginalFile] = {}
        self._directories: dict[Path, tuple[int, int]] = {}
        try:
            self._capture(self.directory)
            self.recheck()
        except BaseException as error:
            retain_cleanup(self, error)
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve constructor failure
                error.add_note(f"directory cleanup remains owned ({type(cleanup).__name__})")
            raise

    def _capture(self, directory: Path) -> None:
        remaining(self._deadline)
        self._budget.acquire(65536, self._deadline)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("unexpected directory source entry")
        self._directories[directory] = (info.st_dev, info.st_ino)
        with os.scandir(directory) as entries:
            for entry in entries:
                remaining(self._deadline)
                # One transient DirEntry is unavoidable; retain no Path/name until admission.
                self._budget.acquire(
                    65536 + 128 * (len(str(directory)) + len(entry.name)), self._deadline
                )
                path = directory / entry.name
                if entry.is_symlink():
                    raise ValueError("symlink directory member")
                if entry.is_dir(follow_symlinks=False):
                    self._capture(path)
                else:
                    name = safe_name(path.relative_to(self.directory).as_posix())
                    self._files[name] = OriginalFile(path, self._deadline)

    def _live(self) -> None:
        remaining(self._deadline)
        if self._closed:
            raise RuntimeError("artifact Source closed")

    def _check_namespace(self) -> None:
        workspace = 65536 + 128 * (len(self._files) + len(self._directories))
        with reserve(self._budget, workspace, self._deadline):
            for directory, identity in self._directories.items():
                remaining(self._deadline)
                _no_symlink_path(directory)
                current = directory.stat()
                if (current.st_dev, current.st_ino) != identity:
                    raise ValueError("original directory identity changed")
                expected = {
                    file.path.name for file in self._files.values() if file.path.parent == directory
                }
                expected.update(
                    child.name for child in self._directories if child.parent == directory
                )
                seen: set[str] = set()
                with os.scandir(directory) as entries:
                    for entry in entries:
                        remaining(self._deadline)
                        if entry.name not in expected or entry.name in seen:
                            raise ValueError("directory source namespace changed")
                        seen.add(entry.name)
                if seen != expected:
                    raise ValueError("directory source namespace changed")

    def recheck(self) -> None:
        self._live()
        self._check_namespace()
        for file in self._files.values():
            file.recheck()
        self._check_namespace()
        for file in self._files.values():
            file.check_identity()

    def names(self) -> tuple[str, ...]:
        self.recheck()
        return tuple(sorted(self._files))

    def inventory(self) -> tuple[Json, ...]:
        self.recheck()
        charge = sum(65536 + 32 * len(name) for name in self._files)
        self._budget.acquire(charge, self._deadline)
        try:
            return tuple(
                {"name": name, "bytes": file.identity.st_size, "sha256": file.sha256}
                for name, file in sorted(self._files.items())
            )
        except BaseException:
            self._budget.release(charge)
            raise

    def read(self, name: str) -> bytes:
        self.recheck()
        file = self._files[name]
        charge = 2 * file.identity.st_size
        self._budget.acquire(charge, self._deadline)
        try:
            result = file.read(file.identity.st_size)
            self.recheck()
            return result
        except BaseException:
            self._budget.release(charge)
            raise

    @contextmanager
    def borrow_group(self, group_id: str) -> Iterator[GroupLoan]:
        # A directory's explicit group is its POSIX prefix, not a policy inference.
        safe_name(group_id)
        self.recheck()
        names = tuple(n for n in self._files if n.startswith(group_id + "/"))
        if not names:
            raise KeyError(group_id)
        if self._loans and self._group_id != group_id:
            raise RuntimeError("cannot load another group with live loans")
        if self._group_id != group_id:
            self.evict_group()
            self._group_id = group_id
        loan = GroupLoan(self, group_id, self._generation, names)
        self._loans += 1
        try:
            yield loan
        finally:
            loan._active = False
            self._loans -= 1

    def _loan_read(self, name: str) -> bytes:
        if name not in self._cache:
            file = self._files[name]
            charge = 2 * file.identity.st_size
            self._budget.acquire(charge, self._deadline)
            try:
                data = file.read(file.identity.st_size)
                self.recheck()
            except BaseException:
                self._budget.release(charge)
                raise
            self._cache[name] = data
            self._cache_charge += charge
        return self._cache[name]

    def evict_group(self) -> None:
        self._live()
        if self._loans:
            raise RuntimeError("cannot evict a group with live loans")
        self._generation += 1
        self._group_id = None
        self._cache.clear()
        self._budget.release(self._cache_charge)
        self._cache_charge = 0

    def close(self) -> None:
        if self._closed:
            return
        if self._loans:
            raise RuntimeError("cannot close a Source with live group loans")
        try:
            close_resources(self, list(self._files.values()))
        except BaseException as exc:
            exc.__dict__["cleanup_owner"] = self
            raise
        self._cache.clear()
        self._budget.release(self._budget.used)
        self._closed = True
        self._generation += 1


def open_artifact_source(
    locator: CollectionLocator, *, deadline: float, max_owned_bytes: int
) -> PackedSource | DirectorySource:
    remaining(deadline)
    if locator.kind == FORMAT:
        return PackedSource(locator.path, deadline=deadline, max_owned_bytes=max_owned_bytes)
    if locator.kind == "directory":
        return DirectorySource(locator.path, deadline=deadline, max_owned_bytes=max_owned_bytes)
    raise ValueError("unsupported collection locator")
