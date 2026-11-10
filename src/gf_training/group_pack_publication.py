"""Concrete pending DB owner and physical-descriptor/envelope commit composition."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone
from gf_common.group_pack import FORMAT, MAX_JSON, canonical_json, digest
from gf_training.collection import PackedRunResult
from gf_training.group_pack_collection import GroupPackCollection
from gf_training.group_pack_resources import (
    OwnedDescriptor,
    StorageReservation,
    close_resources,
    remaining,
)
from gf_training.group_pack_resources import json_bound as _json_bound
from gf_training.group_pack_source import (
    CollectionLocator,
    _no_symlink_path,
    _stat_key,
    inspect_packed,
)
from gf_training.group_pack_writer import PendingDescriptorOwner


class _TerminalOriginal(OwnedDescriptor):
    """A fresh readonly capability bound to an ALREADY owned original.

    Only this composite's controlled write-close handoff may change ctime.
    This object neither admits a fresh inode nor changes Source custody rules.
    """

    def __init__(self, path: Path, expected: os.stat_result, sha256: str,
                 deadline: float, *, write_handoff: bool) -> None:
        super().__init__(-1)
        self.path, self.expected, self.sha256 = path, expected, sha256
        self.deadline, self.write_handoff = deadline, write_handoff
        self.identity: os.stat_result | None = None

    def open_verified(self) -> None:
        remaining(self.deadline)
        _no_symlink_path(self.path.parent)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        actual = os.fstat(self.fd)
        expected_key, actual_key = _stat_key(self.expected), _stat_key(actual)
        if (not stat.S_ISREG(actual.st_mode) or actual_key[:5] != expected_key[:5]
                or not self.write_handoff and actual_key != expected_key):
            raise ContractError("terminal original identity changed before readonly hash")
        self.identity = actual
        self.recheck()

    def check_path_identity(self) -> None:
        assert self.identity is not None
        remaining(self.deadline)
        _no_symlink_path(self.path.parent)
        if _stat_key(self.path.lstat()) != _stat_key(self.identity):
            raise ContractError("terminal readonly original path identity changed")

    def recheck(self) -> None:
        assert self.identity is not None
        self.check_path_identity()
        if _stat_key(os.fstat(self.fd)) != _stat_key(self.identity):
            raise ContractError("terminal readonly handle identity changed")
        value = hashlib.sha256()
        offset = 0
        while offset < self.identity.st_size:
            remaining(self.deadline)
            chunk = os.pread(self.fd, min(65536, self.identity.st_size - offset), offset)
            if not chunk:
                raise ContractError("terminal readonly original shortened")
            value.update(chunk)
            offset += len(chunk)
            del chunk
        if os.pread(self.fd, 1, offset) or value.hexdigest() != self.sha256:
            raise ContractError("terminal readonly original content changed")
        if _stat_key(os.fstat(self.fd)) != _stat_key(self.identity):
            raise ContractError("terminal original changed during readonly hash")
        self.check_path_identity()


class GroupPackCollectionPublication:
    def __init__(self, collection: GroupPackCollection, envelope: Path) -> None:
        self.collection, self.envelope = collection, envelope
        self.pending: PendingDescriptorOwner | None = None
        self._physical: Any = None
        self._pointer: Any = None
        self._controls: dict[str, bytes] | None = None
        self._reservation: StorageReservation | None = None
        self.committed = False
        self._result: PackedRunResult | None = None
        self._finished_diagnostic = False
        self._terminal_originals: list[_TerminalOriginal] = []
        self._handoff_reservation: StorageReservation | None = None
        self._cleanup_children: list[Any] = []

    def _verify_readonly_handoff(self, expected: list[tuple[Path, os.stat_result, str, bool]]) -> None:
        # One bounded hash buffer, retained handle/object allowances, SAME account.
        self._handoff_reservation = self.collection.budget.reserve(65536 + 4096 * len(expected))
        for path, identity, sha256, write_handoff in expected:
            original = _TerminalOriginal(path, identity, sha256, self.collection.deadline,
                                         write_handoff=write_handoff)
            self._terminal_originals.append(original)  # Own FD before open/hash can fail.
            original.open_verified()
        for original in self._terminal_originals:
            original.recheck()

    def _check_terminal_namespaces(self, *, diagnostic: bool) -> None:
        for owner, expected in ((self._physical, {"collection.sqlite3", "collection.pending.json"}
                | (set() if diagnostic else {"collection.published.json"})),
                (self._pointer, set() if diagnostic else {"collection.locator.json"})):
            assert owner is not None
            _no_symlink_path(owner.directory)
            current = owner.directory.lstat()
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != owner.directory_identity:
                raise ContractError("terminal original publication directory changed")
            seen = set()
            with os.scandir(owner.directory) as entries:
                for entry in entries:
                    remaining(self.collection.deadline)
                    if entry.name not in expected or not entry.is_file(follow_symlinks=False):
                        raise ContractError("terminal publication namespace changed")
                    seen.add(entry.name)
            if seen != expected:
                raise ContractError("terminal publication namespace changed")

    def _finish_readonly_handoff(self) -> None:
        close_resources(self, self._terminal_originals)
        assert self._physical is not None and self._pointer is not None
        self._physical._release()
        self._pointer._release()

    def _final_readonly_checks(self, *, diagnostic: bool) -> None:
        for original in self._terminal_originals:
            original.check_path_identity()
        self._check_terminal_namespaces(diagnostic=diagnostic)
        for original in self._terminal_originals:
            original.check_path_identity()
        assert self._handoff_reservation is not None
        self._handoff_reservation.close()

    @property
    def result(self) -> PackedRunResult:
        if self._result is None or not (self.committed or self._finished_diagnostic):
            raise ContractError("packed result has not completed its actual publication owner")
        return self._result

    def _result_metrics(self, scope: str) -> Json:
        return {**self.collection.metrics, "publication_scope": scope,
            "checkpoint_measurement": "actual-writer-child-checked-WAL-and-DELETE-operations",
            "seal_measurement": "combined-child-seal-inspection-and-pending-publication",
            "memory_measurement": "managed-reservations-and-observed-RAM-not-RSS"}

    async def diagnostic(self, report: Json, manifest: Json) -> PackedRunResult:
        if self.pending is not None or self.committed or self._finished_diagnostic:
            raise ContractError("packed diagnostic owner already consumed")
        self._reservation = self.collection.budget.reserve(32 * _json_bound(report) + 65536)
        try:
            safe_report = clone(report)
            self.pending = await self.collection.prepare_diagnostic("incomplete.json",
                {"manifest": manifest, "report": safe_report})
            pending = self.pending
            pending.recheck()
            expected = [(pending.path, pending.snapshot, digest(pending.raw), True),
                (pending.database.path, pending.database.identity, pending.database.sha256, False)]
            self._physical = self._open(pending.directory)
            self._pointer = self._open(self.envelope)
            pending.close()
            self._verify_readonly_handoff(expected)
            result = PackedRunResult(self.envelope, safe_report, CollectionLocator(FORMAT, pending.path),
                None, self._result_metrics("diagnostic-pending"))
            self._finish_readonly_handoff()
            await self.collection.close()
            self._final_readonly_checks(diagnostic=True)
            self._reservation.close()
        except BaseException as error:
            error.__dict__["_group_pack_publication_cleanup_owner"] = self
            try:
                await self.collection.close()
                self.pending = self.collection.pending
                self.revoke()
            except BaseException as cleanup:  # noqa: BLE001 -- retain actual pending/worker owners with original failure.
                error.add_note("diagnostic publication cleanup remains owned: " + str(cleanup))
            raise
        self._result = result
        self._finished_diagnostic = True
        return result

    @property
    def cleanup_complete(self) -> bool:
        return (self.collection._closed
            and not self.collection._cleanup_children
            and (self.collection.writer is None or (self.collection.writer.returncode is not None
                and self.collection.writer._base_released))
            and (self._reservation is None or self._reservation.closed)
            and (self._handoff_reservation is None or self._handoff_reservation.closed)
            and not self._cleanup_children
            and all(original.fd < 0 and not original._uncertain_releases
                    for original in self._terminal_originals)
            and (self.pending is None or (self.pending.fd < 0 and self.pending.database.fd < 0
                and not self.pending._uncertain_releases
                and not self.pending.database._uncertain_releases))
            and all(owner is None or owner.cleanup_complete
                    for owner in (self._pointer, self._physical)))

    async def prepare(self, report: Json, manifest: Json) -> None:
        if self.pending is not None or self._controls is not None:
            raise RuntimeError("packed controls already prepared")
        self._reservation = self.collection.budget.reserve(
            32 * (_json_bound(report) + _json_bound(manifest)) + 4 * MAX_JSON)
        self._controls = {"report.json": (canonical(report) + "\n").encode(),
                          "manifest.json": (canonical(manifest) + "\n").encode()}
        try:
            self.pending = await self.collection.prepare(self._controls)
        except BaseException as error:
            error.__dict__["_group_pack_publication_cleanup_owner"] = self
            try:
                await self.collection.close()
                self.pending = self.collection.pending
                self.revoke()
            except BaseException as cleanup:  # noqa: BLE001 -- concrete publication/worker owners remain reachable.
                error.add_note("packed preparation cleanup remains owned: " + str(cleanup))
            raise

    def _open(self, directory: Path) -> Any:
        from gf_training.card_instance_collection import CollectionPublication, _inode
        owner = CollectionPublication(directory, work_deadline=self.collection.deadline,
                                      cleanup_deadline=self.collection.deadline)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        owner._directory_handle = owner._own_handle(descriptor)
        owner.directory_identity = _inode(os.fstat(descriptor))
        return owner

    def _link(self, owner: Any) -> None:
        output = owner.outputs[0]
        owner._check()
        assert owner.directory_fd is not None and output.handle.descriptor is not None
        os.link(output.name + ".tmp", output.name, src_dir_fd=owner.directory_fd,
            dst_dir_fd=owner.directory_fd, follow_symlinks=False)
        output.published = True
        owner._verify(output.name, output.handle.descriptor, output.raw)
        owner._verify(output.name + ".tmp", output.handle.descriptor, output.raw)
        os.unlink(output.name + ".tmp", dir_fd=owner.directory_fd)
        os.fsync(owner.directory_fd)

    def revoke(self) -> None:
        first: BaseException | None = None
        # Consumer-visible locator FIRST. Foreign replacement paths are preserved.
        for owner in (self._pointer, self._physical):
            if owner is None:
                continue
            try:
                owner.revoke()
            except BaseException as error:  # noqa: BLE001 -- all evidence and cleanup ownership remain reachable.
                first = first or error
        if self.pending is not None:
            try:
                self.pending.close()
            except BaseException as error:  # noqa: BLE001 -- all evidence and cleanup ownership remain reachable.
                first = first or error
        try:
            close_resources(self, self._terminal_originals)
        except BaseException as error:  # noqa: BLE001 -- retire all independent readonly capabilities once.
            first = first or error
        if first is not None:
            first.__dict__["cleanup_owner"] = self
            raise first
        if self._reservation is not None:
            self._controls = None
            self._reservation.close()
        if self._handoff_reservation is not None:
            self._handoff_reservation.close()

    def commit(self, report: Json, manifest: Json, *,
               retained_input_guard: Callable[[], None] | None = None) -> PackedRunResult:
        if (self.committed or self.pending is None or self._controls is None
                or self._physical is not None or self._pointer is not None or self._terminal_originals):
            raise ContractError("packed publication requires its actual prepared pending owner")
        if (self._controls != {"report.json": (canonical(report) + "\n").encode(),
                "manifest.json": (canonical(manifest) + "\n").encode()}
                or report.get("healthy") is not True or manifest.get("schema_version") != 5
                or manifest.get("collection_role") == "ppo-final-validation"):
            raise ContractError("packed controls changed or are not ordinary healthy collection5")
        pending = self.pending
        try:
            if retained_input_guard is not None:
                retained_input_guard()
            pending.recheck()
            descriptor = {**pending.descriptor, "publication": "published"}
            raw = canonical_json(descriptor, limit=MAX_JSON)
            target = pending.directory / "collection.published.json"
            locator = canonical_json({"format": "sqlite-group-pack-locator-v1", "schema_version": 1,
                "descriptor": str(target), "descriptor_sha256": digest(raw)}, limit=MAX_JSON)
            self._physical = self._open(pending.directory)
            self._physical._write(target.name, None, _stage_manifest=True, _raw=raw)
            self._pointer = self._open(self.envelope)
            self._pointer._write("collection.locator.json", None, _stage_manifest=True, _raw=locator)
            for owner in (self._physical, self._pointer):
                output = owner.outputs[0]
                owner._verify(output.recorded_name, output.handle.descriptor, output.raw)
            if retained_input_guard is not None:
                retained_input_guard()
            self._link(self._physical)
            self._link(self._pointer)
            pending.recheck()
            inspect_packed(target, deadline=self.collection.deadline,
                max_owned_bytes=self.collection.budget.limit - self.collection.budget.used,
                database_original=pending.database, budget=self.collection.budget)
            pending.recheck()
            expected = [(owner.directory / owner.outputs[0].name,
                os.fstat(owner.outputs[0].handle.descriptor), digest(owner.outputs[0].raw), True)
                for owner in (self._physical, self._pointer)]
            expected.extend([(pending.path, pending.snapshot, digest(pending.raw), True),
                (pending.database.path, pending.database.identity, pending.database.sha256, False)])
            for owner in (self._physical, self._pointer):
                output = owner.outputs[0]
                owner._verify(output.name, output.handle.descriptor, output.raw)
            pending.close()
            for owner in (self._physical, self._pointer):
                owner._release_output(owner.outputs[0])
            self._verify_readonly_handoff(expected)
            result = PackedRunResult(self.envelope, json.loads(self._controls["report.json"]), CollectionLocator(FORMAT, target),
                self._terminal_originals[1].sha256, self._result_metrics("published"))
            self._finish_readonly_handoff()
            if retained_input_guard is not None:
                retained_input_guard()
            self._final_readonly_checks(diagnostic=False)
            if self._reservation is not None:
                self._controls = None
                self._reservation.close()
            self.collection.retire_metadata()
        except BaseException as error:
            try:
                self.revoke()
            except BaseException as cleanup:  # noqa: BLE001 -- preserve initiating error and concrete cleanup owner.
                error.__dict__["cleanup_owner"] = self
                error.add_note("packed composite cleanup remains owned: " + str(cleanup))
            raise
        self.committed = True
        self._physical.committed = self._pointer.committed = True
        self._result = result
        return result
