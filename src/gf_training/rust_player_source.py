"""Actual retained Rust-writer originals and canonical data correspondence.

Opening data never grants collection, update, actor or final-parent authority.
The online issuer must additionally own the actual native/service/plan lifetime.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Any

from gf_common.artifacts import json_bound
from gf_common.domain import ContractError, Json, canonical
from gf_training.group_pack_pool import OriginalArtifactReader, PackedArtifactPool
from gf_training.group_pack_source import PackedSource
from gf_training.rust_player_storage import RustPendingDescriptor
from gf_training.structured_source import _json


class RustPlayerOriginals:
    """Borrow SAME pending-descriptor custody; share the actual Source account.

    Decoded records/metadata returned here are borrowed until close. A caller
    retires all such views before closing; canonical numeric banks have their
    own explicit owners in this same account.
    """

    def __init__(self, pending: RustPendingDescriptor, *, pool: PackedArtifactPool,
                 settings: Any, deadline: float) -> None:
        if type(pending) is not RustPendingDescriptor or type(pool) is not PackedArtifactPool:
            raise ContractError("actual Rust pending descriptor and shared original pool required")
        pending.recheck()
        self.pending, self.pool, self.budget = pending, pool, pool.budget
        self.root, self.settings, self.deadline = pending.directory, settings, deadline
        self.source: PackedSource | None = None
        self.files: Json = {}
        self._entries: Json = {}
        self._reservations: list[Any] = []
        self._closed = False
        try:
            self.source = PackedSource(pending.path, deadline=deadline,
                max_owned_bytes=pool.budget.limit, pool=pool, _allow_pending=True)
            original = self.source._originals
            if (original.descriptor_file.sha256 != pending.descriptor_sha256
                    or original.descriptor_file.path != pending.path):
                raise ContractError("reopened originals differ from SAME retained writer descriptor")
            self._entries = {row["name"]: row for row in
                (*self.source._index["members"], *self.source._index["controls"])}
            self.charge(sys.getsizeof(self._entries))
            self.reader = OriginalArtifactReader(self.source, self.read, self.charge, self.root)
            self.finalize()
        except BaseException as error:
            vars(error)["cleanup_owner"] = self
            raise

    def check(self) -> None:
        if self._closed or self.source is None:
            raise ContractError("Rust original data owner is closed")
        self.pending.recheck()
        self.source.recheck()

    def charge(self, size: int) -> None:
        self._reservations.append(self.budget.reserve(size))

    def _name(self, path: Path) -> str:
        try:
            name = path.absolute().relative_to(self.root).as_posix()
        except ValueError as error:
            raise ContractError("original logical member belongs to another collection") from error
        if name not in self._entries:
            raise ContractError("unregistered original logical member")
        return name

    def member_identity(self, name: str) -> Json:
        self.check()
        row = self._entries[name]
        assert self.source is not None
        original = self.source._originals
        return {"recipe": "rust-player-packed-member-v1",
            "descriptor_sha256": self.pending.descriptor_sha256,
            "database_sha256": original.descriptor["database_sha256"],
            "name": name, "length": row.get("length", row.get("bytes")),
            "sha256": row["sha256"]}

    def _remember(self, name: str) -> None:
        if name not in self.files:
            identity = self.member_identity(name)
            self.charge(32 * json_bound(identity) + 65536)
            self.files[name] = {"sha256": identity["sha256"],
                "bytes": identity["length"], "identity": identity}

    def read(self, path: Path) -> bytes:
        self.check()
        name = self._name(path)
        assert self.source is not None
        row = self._entries[name]
        if row.get("length", row.get("bytes")) > self.settings.max_source_file_mib * 1024**2:
            raise ContractError("bounded original logical file required")
        raw = self.source.read(name)
        self._remember(name)
        return raw

    def object(self, path: Path) -> Json:
        name = self._name(path)
        row = self._entries[name]
        self.charge(32 * row.get("length", row.get("bytes")) + 65536)
        return _json(self.read(path))

    def contains(self, path: Path) -> bool:
        self.check()
        return path.absolute().is_relative_to(self.root) and path.relative_to(self.root).as_posix() in self._entries

    def inventory(self, pattern: str) -> tuple[str, ...]:
        self.check()
        return tuple(name for name in sorted(self._entries)
            if pattern == "**/*" or fnmatch.fnmatchcase(name, pattern))

    def remember_inventory_entry(self, name: str, size: int, checksum: str) -> None:
        self.check()
        row = self._entries.get(name)
        if row is None or row.get("length", row.get("bytes")) != size or row["sha256"] != checksum:
            raise ContractError("inventory differs from actual sealed original index")
        self._remember(name)

    def remember(self, path: Path, identity: Json, checksum: str) -> None:
        name = self._name(path)
        if canonical(identity) != canonical(self.member_identity(name)) or self._entries[name]["sha256"] != checksum:
            raise ContractError("logical member identity differs from actual original")
        self._remember(name)

    def finalize(self) -> None:
        self.check()

    def close(self) -> None:
        if self._closed:
            return
        if self.source is not None:
            self.source.close()
        self._entries.clear()
        self.files.clear()
        for reservation in self._reservations:
            reservation.close()
        self._reservations.clear()
        self._closed = True
