"""Synchronously verify and establish durable custody of a closed native collection."""

from __future__ import annotations

import os
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from uuid import uuid4

from gf_common.group_pack import MIB, Json, StorageSettings
from gf_common.hpo_storage import safe_storage_path
from gf_training.group_pack_db import (
    DATABASE,
    CatalogConflictError,
    catalog_update,
    descriptor_name,
    fsync_directory,
    validate_database,
)
from gf_training.group_pack_resources import OwnedDescriptor, remaining
from gf_training.group_pack_source import OriginalFile
from gf_training.group_pack_writer import PendingDescriptorOwner

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_CREATE = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


def _location(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _checked_directory(path: Path, owner: OwnedDescriptor, device: int) -> None:
    actual = os.fstat(owner.fd)
    if actual.st_dev != device or _location(path.lstat()) != _location(actual):
        raise ValueError("admitted durable collection directory changed")


def _mkdirs(path: Path, *, mount_root: Path, device: int) -> None:
    """Create only beneath the admitted mount and sync each new directory entry."""
    safe_storage_path(mount_root)
    relative = path.relative_to(mount_root)
    with ExitStack() as roots:
        parent = OwnedDescriptor(os.open(mount_root, _DIRECTORY))
        roots.callback(parent.close)
        current = mount_root
        _checked_directory(current, parent, device)
        for name in relative.parts:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent.fd)
            except FileExistsError:
                pass
            else:
                os.fsync(parent.fd)
            child = OwnedDescriptor(os.open(name, _DIRECTORY, dir_fd=parent.fd))
            roots.callback(child.close)
            parent, current = child, current / name
            _checked_directory(current, parent, device)


def _write(fd: int, data: bytes, deadline: float) -> None:
    rest = memoryview(data)
    while rest:
        remaining(deadline)
        count = os.write(fd, rest)
        if count <= 0:
            raise OSError("durable collection write incomplete")
        rest = rest[count:]


def _copy(source: OriginalFile, fd: int, deadline: float) -> None:
    offset = 0
    while offset < source.identity.st_size:
        remaining(deadline)
        data = os.pread(source.fd, min(4 * MIB, source.identity.st_size - offset), offset)
        if not data:
            raise OSError("sealed collection copy incomplete")
        _write(fd, data, deadline)
        offset += len(data)
    source.recheck()


def _remove_owned_directory(parent_fd: int, name: str, staged: OwnedDescriptor,
                            files: dict[str, OwnedDescriptor]) -> None:
    """Retire only the directory and files this attempt still holds."""
    path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _location(path) != _location(os.fstat(staged.fd)):
        raise ValueError("private HDD collection staging directory changed")
    for filename, owner in files.items():
        try:
            current = os.stat(filename, dir_fd=staged.fd, follow_symlinks=False)
        except FileNotFoundError:  # Already moved to the exclusive final batch.
            continue
        if _location(current) != _location(os.fstat(owner.fd)):
            raise ValueError("foreign file in private HDD collection staging")
        os.unlink(filename, dir_fd=staged.fd)
    os.rmdir(name, dir_fd=parent_fd)


def promote_collection(
    source: OriginalFile, *, directory: Path, descriptor: Json, raw: bytes,
    settings: StorageSettings, deadline: float, max_owned_bytes: int,
    mount_root: Path, mount_device: int | None = None,
) -> PendingDescriptorOwner:
    """Return the existing pending owner bound to the verified HDD inodes."""
    if source.mutable or source.identity.st_size != descriptor["database_bytes"] or source.sha256 != descriptor["database_sha256"]:
        raise ValueError("detached collection does not match its original sealed DB")
    source.recheck()
    if mount_device is None:
        mount_device = mount_root.stat().st_dev
    device = mount_device
    _mkdirs(directory.parent, mount_root=mount_root, device=device)
    marker = descriptor_name("pending")
    stage_name = f".{directory.name}-{uuid4().hex}.staging"
    stage_path = directory.parent / stage_name
    target_path = directory / marker
    with ExitStack() as outgoing, ExitStack() as pins:
        parent = OwnedDescriptor(os.open(directory.parent, _DIRECTORY))
        pins.callback(parent.close)
        _checked_directory(directory.parent, parent, device)
        os.mkdir(stage_name, mode=0o700, dir_fd=parent.fd)
        staged = OwnedDescriptor(os.open(stage_name, _DIRECTORY, dir_fd=parent.fd))
        pins.callback(staged.close)
        files: dict[str, OwnedDescriptor] = {}
        # Stage retirement needs the same live descriptors; all close attempts
        # still run independently on any transfer, rename or sync error.
        with ExitStack() as staged_names:
            staged_names.callback(_remove_owned_directory, parent.fd, stage_name, staged, files)
            db_file = files[DATABASE] = OwnedDescriptor(os.open(DATABASE, _CREATE, 0o600, dir_fd=staged.fd))
            pins.callback(db_file.close)
            desc_file = files[marker] = OwnedDescriptor(os.open(marker, _CREATE, 0o600, dir_fd=staged.fd))
            outgoing.callback(desc_file.close)
            _copy(source, db_file.fd, deadline)
            _write(desc_file.fd, raw, deadline)

            expected = descriptor["database_sha256"]
            audit = OriginalFile(stage_path / DATABASE, deadline, database_expected_sha256=expected,
                                 max_bytes=source.identity.st_size)
            pins.callback(audit.close)
            if (_location(audit.identity) != _location(os.fstat(db_file.fd))
                    or audit.identity.st_size != source.identity.st_size or audit.hash() != expected):
                raise ValueError("staged HDD database SHA-256 differs from sealed original")
            with closing(sqlite3.connect(f"file:/proc/self/fd/{audit.fd}?mode=ro", uri=True)) as db:
                db.execute("PRAGMA query_only=ON")
                db.execute("PRAGMA cache_size=-32768")
                facts = validate_database(db, deadline=deadline,
                    max_pack_bytes=settings.max_pack_bytes, max_owned_bytes=max_owned_bytes)
            if (facts["seal"]["run_id"] != descriptor["run_id"] or facts["seal"]["seal_id"] != descriptor["seal_id"]
                    or facts["index"]["controls"] != descriptor["controls"]
                    or os.pread(desc_file.fd, len(raw) + 1, 0) != raw):
                raise ValueError("HDD collection does not match the exact pending seal")
            audit.recheck()
            os.fsync(db_file.fd)
            os.fsync(desc_file.fd)
            os.fsync(staged.fd)
            _checked_directory(directory.parent, parent, device)
            _checked_directory(stage_path, staged, device)
            remaining(deadline)
            os.mkdir(directory.name, mode=0o700, dir_fd=parent.fd)
            final = OwnedDescriptor(os.open(directory.name, _DIRECTORY, dir_fd=parent.fd))
            pins.callback(final.close)
            _checked_directory(directory, final, device)
            for name in (DATABASE, marker):  # Publish the descriptor last.
                os.rename(name, name, src_dir_fd=staged.fd, dst_dir_fd=final.fd)
            os.fsync(final.fd)
            os.fsync(staged.fd)

            durable = OriginalFile(directory / DATABASE, deadline, database_expected_sha256=expected,
                                   max_bytes=source.identity.st_size)
            outgoing.callback(durable.close)
            if (_location(durable.identity) != _location(audit.identity)
                    or durable.identity.st_size != audit.identity.st_size
                    or durable.identity.st_mtime_ns != audit.identity.st_mtime_ns):
                raise ValueError("durable database differs from verified HDD inode")
        os.fsync(parent.fd)
        pending = PendingDescriptorOwner(directory, target_path, _location(os.fstat(desc_file.fd)),
                                         raw, descriptor, desc_file.fd, durable)
        pending.recheck()
        try:
            catalog_update(settings, directory / DATABASE, descriptor["run_id"], "sealed",
                           deadline, descriptor, create_only=True)
        except CatalogConflictError as conflict:
            # SQLite refused this exact insertion. Retire only our transferred
            # inode pair, so the unchanged catalog cannot resolve to our seal.
            try:
                _remove_owned_directory(parent.fd, directory.name, final, files)
                os.fsync(parent.fd)
            except OSError as cleanup:
                conflict.add_note("conflicting durable collection cleanup unconfirmed: " + str(cleanup))
            raise
        fsync_directory(settings.db_root)
        pending.recheck()
        remaining(deadline)
        pins.close()
        outgoing.pop_all()
        return pending
