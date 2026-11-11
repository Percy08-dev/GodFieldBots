"""Concrete pending DB owner and physical-descriptor/envelope commit composition."""
from __future__ import annotations

import json
import math
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone
from gf_common.group_pack import FORMAT, MAX_JSON, canonical_json, digest
from gf_training.collection import PackedRunResult
from gf_training.group_pack_collection import GroupPackCollection
from gf_training.group_pack_resources import (
    StorageReservation,
    close_resources,
    remaining,
)
from gf_training.group_pack_resources import json_bound as _json_bound
from gf_training.group_pack_source import (
    CollectionLocator,
    _no_symlink_path,
    inspect_packed,
)
from gf_training.group_pack_terminal import _TerminalOriginal
from gf_training.group_pack_writer import PendingDescriptorOwner


class GroupPackCollectionPublication:
    def __init__(self, collection: GroupPackCollection, envelope: Path) -> None:
        self.collection, self.envelope = collection, envelope
        self.pending: PendingDescriptorOwner | None = None
        self._physical: Any = None
        self._pointer: Any = None
        self._pending_pointer: Any = None
        self._pending_pointer_retired = False
        self._pending_recovery: Any = None
        self._pending_invalidated = False
        self._final_staged = False
        self._pending_facts = b''
        self._public_facts: Json | None = None
        self._facts_reservation: StorageReservation | None = None
        self._facts_uncertain = False
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
                                         write_handoff=write_handoff, database_expected=path.name == 'collection.sqlite3')
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
            and (not self._final_staged or self.collection.budget.used == 0)
            and not self.collection._cleanup_children
            and (self.collection.writer is None or (self.collection.writer.returncode is not None
                and self.collection.writer._base_released))
            and (self._reservation is None or self._reservation.closed)
            and (self._facts_reservation is None or self._facts_reservation.closed)
            and not self._facts_uncertain
            and (self._handoff_reservation is None or self._handoff_reservation.closed)
            and not self._cleanup_children
            and all(original.fd < 0 and not original._uncertain_releases
                    for original in self._terminal_originals)
            and (self.pending is None or (self.pending.fd < 0 and self.pending.database.fd < 0
                and not self.pending._uncertain_releases
                and not self.pending.database._uncertain_releases))
            and all(owner is None or owner.cleanup_complete
                    for owner in (self._pointer, self._physical, self._pending_pointer, self._pending_recovery)))

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
        self.committed = False
        self._result = None
        first: BaseException | None = None
        # Consumer-visible locator FIRST. Foreign replacement paths are preserved.
        for owner in (self._pointer, None if self._pending_pointer_retired else self._pending_pointer, self._physical):
            if owner is None:
                continue
            try:
                owner.revoke()
            except BaseException as error:  # noqa: BLE001 -- all evidence and cleanup ownership remain reachable.
                first = first or error
        if self._final_staged:
            try:
                self._invalidate_pending_descriptor()
            except BaseException as error:  # noqa: BLE001 - locator invalidation already preceded this independent owner.
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
        if self._pending_recovery is not None:
            try:
                self._pending_recovery._release()
            except BaseException as error:  # noqa: BLE001 - no uncertain raw descriptor retry.
                first = first or error
        if self._facts_uncertain:
            first = first or ContractError('final facts reservation acquisition remains uncertain')
        if first is not None:
            first.__dict__["cleanup_owner"] = self
            raise first
        if self._facts_reservation is not None:
            self._pending_facts = b''
            self._public_facts = None
            self._facts_reservation.close()
        if self._reservation is not None:
            self._controls = None
            self._reservation.close()
        if self._handoff_reservation is not None:
            self._handoff_reservation.close()
        if self._final_staged:
            self.collection.retire_metadata()

    @property
    def resident_bytes(self) -> int:
        return self.collection.budget.used

    @property
    def final_pending(self) -> bool:
        return (self._final_staged and not self.committed and not self._pending_invalidated
            and self.pending is not None and self._pending_pointer is not None
            and not self._pending_pointer.quarantined)

    def begin_final_validation(self, report: Json, manifest: Json) -> None:
        if (self._final_staged or self.pending is None or self._controls is None
                or self._controls != {'report.json': (canonical(report)+'\n').encode(),
                    'manifest.json': (canonical(manifest)+'\n').encode()}
                or manifest.get('collection_role') != 'ppo-final-validation' or report.get('healthy') is not True):
            raise ContractError('one actual prepared packed final collection required')
        self._final_staged = True
        self.pending.recheck()
        clock = manifest['validation_invocation']
        elapsed = time.monotonic()-clock['started_monotonic']
        if (not math.isfinite(elapsed) or not report['wall_seconds'] <= manifest['elapsed_seconds'] <= elapsed
                <= clock['wall_deadline_monotonic']-clock['started_monotonic']):
            raise ContractError('packed final immutable prewrite cost/invocation order differs')
        self._pending_pointer = self._open(self.envelope)
        raw = canonical_json({'format': 'sqlite-group-pack-locator-v1', 'schema_version': 1,
            'descriptor': str(self.pending.path), 'descriptor_sha256': digest(self.pending.raw)}, limit=MAX_JSON)
        self._pending_pointer._write('collection.locator.json', None, _stage_manifest=True, _raw=raw)
        self.pending_manifest()

    def pending_manifest(self) -> str:
        if not self.final_pending or self._pending_pointer_retired:
            raise ContractError('actual packed pending final controls required')
        pending = self.pending
        assert pending is not None
        pending.recheck()
        owner = self._pending_pointer
        output = owner.outputs[0]
        owner._check()
        owner._verify('collection.locator.json.tmp', output.handle.descriptor, output.raw)
        for path in (self.envelope/'collection.locator.json', pending.directory/'collection.published.json'):
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise ContractError('pending packed final public outputs must remain absent')
        return 'manifest.json'

    def retain_final_facts(self, facts: Json) -> None:
        from gf_common.card_instance_collection import (
            FINAL_FACTS_FIELDS,
            PACKED_FINAL_FACTS_FIELDS,
            packed_final_facts,
        )
        from gf_common.domain import closed
        self.pending_manifest()
        pending = self.pending
        assert pending is not None
        if self._facts_reservation is not None:
            raise ContractError('one admitted final semantic snapshot required')
        closed(facts, PACKED_FINAL_FACTS_FIELDS)
        amount = 8*(32*_json_bound(facts)+65536)
        self._facts_reservation = StorageReservation(self.collection.budget, 0)
        before = self.collection.budget.used
        try:
            self.collection.budget.acquire(amount, self.collection.deadline)
        finally:
            if self.collection.budget.used == before+amount:
                self._facts_reservation.size = amount
            elif self.collection.budget.used != before:
                self._facts_uncertain = True
        if self._facts_reservation.size != amount:
            raise ContractError('actual final facts reservation was not confirmed')
        expected = packed_final_facts({key: facts[key] for key in FINAL_FACTS_FIELDS},
            storage=facts['storage'], custody=facts['custody'], source_set_sha256=facts['source_set_sha256'])
        if (canonical(expected) != canonical(facts) or facts['storage']['database_sha256'] != pending.database.sha256
                or facts['storage']['descriptor_sha256'] != digest(pending.raw)
                or facts['storage']['locator_sha256'] != digest(self._pending_pointer.outputs[0].raw)
                or facts['storage']['seal_id'] != pending.descriptor['seal_id']):
            raise ContractError('pending semantic facts differ from SAME owned packed originals')
        self._pending_facts = canonical(expected).encode()

    def _invalidate_pending_descriptor(self) -> None:
        if self.pending is None or self._pending_invalidated:
            return
        pending = self.pending
        descriptor = pending.fd
        if descriptor < 0:
            self._pending_recovery = self._open(pending.directory)
            descriptor = os.open(pending.path.name, os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,
                dir_fd=self._pending_recovery.directory_fd)
            self._pending_recovery._own_handle(descriptor)
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != pending.identity:
            raise ContractError('pending final descriptor recovery differs from original inode')
        if os.pwrite(descriptor, b'!', 0) != 1 or os.pread(descriptor, 1, 0) != b'!':
            raise ContractError('pending final descriptor invalidation unconfirmed')
        os.fsync(descriptor)
        self._pending_invalidated = True

    def finish_final_validation(self, publication: Any, context: Any) -> Json:
        from gf_training.card_instance_final_validation import GameFinalValidationContext
        if (type(context) is not GameFinalValidationContext or context._issued is None
                or context._issued.publication is not publication or publication.group_pack is not self
                or context._failure is not None or not context._accepted or not context._loan_retired
                or not context._parent.cleanup_complete or not self._pending_facts):
            raise ContractError('SAME accepted final context and whole actual parent retirement required')
        self.pending_manifest()
        assert self._controls is not None
        self._commit_prepared(json.loads(self._controls['report.json']), json.loads(self._controls['manifest.json']),
            final_context=context)
        assert self._public_facts is not None
        return self._public_facts

    def _retire_pending_pointer(self) -> None:
        owner = self._pending_pointer
        output = owner.outputs[0]
        owner._verify(output.recorded_name, output.handle.descriptor, output.raw)
        os.unlink(output.recorded_name, dir_fd=owner.directory_fd)
        os.fsync(owner.directory_fd)
        owner._release()
        owner.committed = True
        self._pending_pointer_retired = True

    def _prepare_public_facts(self) -> None:
        from gf_common.card_instance_collection import FINAL_FACTS_FIELDS, packed_final_facts
        from gf_common.domain import digest as object_digest
        from gf_training.group_pack_source import _stat_key
        pending_facts = json.loads(self._pending_facts)
        assert self.pending is not None
        descriptor, locator, predecessor, database = self._terminal_originals
        if (database.sha256 != pending_facts['storage']['database_sha256']
                or predecessor.sha256 != digest(self.pending.raw)):
            raise ContractError('public final facts do not preserve SAME DB/pending predecessor')
        storage = {**pending_facts['storage'], 'locator_sha256': locator.sha256,
            'descriptor_sha256': descriptor.sha256}
        files: list[Json] = []
        for role, original in (('descriptor', descriptor), ('locator', locator), ('predecessor', predecessor), ('database', database)):
            assert original.identity is not None
            files.append({'role': role, 'path': str(original.path), 'sha256': original.sha256,
                'identity': list(_stat_key(original.identity))})
        directories: list[Json] = [{'path': str(owner.directory), 'device': owner.directory_identity[0], 'inode': owner.directory_identity[1]}
            for owner in (self._physical, self._pointer)]
        custody = {'recipe': 'packed-original-custody-v2', 'files': sorted(files, key=lambda row: row['path']),
            'directories': sorted(directories, key=lambda row: row['path'])}
        custody['fingerprint'] = object_digest(custody)
        self._public_facts = packed_final_facts({key: pending_facts[key] for key in FINAL_FACTS_FIELDS},
            storage=storage, custody=custody, source_set_sha256=pending_facts['source_set_sha256'])

    def commit(self, report: Json, manifest: Json, *,
               retained_input_guard: Callable[[], None] | None = None) -> PackedRunResult:
        if manifest.get('collection_role') == 'ppo-final-validation':
            raise ContractError('final packed publication requires SAME retained context')
        return self._commit_prepared(report, manifest, retained_input_guard=retained_input_guard)

    def _commit_prepared(self, report: Json, manifest: Json, *,
               retained_input_guard: Callable[[], None] | None = None, final_context: Any = None) -> PackedRunResult:
        if (self.committed or self.pending is None or self._controls is None
                or self._physical is not None or self._pointer is not None or self._terminal_originals):
            raise ContractError("packed publication requires its actual prepared pending owner")
        if (self._controls != {"report.json": (canonical(report) + "\n").encode(),
                "manifest.json": (canonical(manifest) + "\n").encode()}
                or report.get("healthy") is not True or manifest.get("schema_version") != 5
                or (manifest.get("collection_role") == "ppo-final-validation") != (final_context is not None)):
            raise ContractError("packed controls changed or are not ordinary healthy collection5")
        pending = self.pending
        try:
            if retained_input_guard is not None:
                retained_input_guard()
            pending.recheck()
            if final_context is not None:
                if canonical_json(pending.descriptor, limit=MAX_JSON) != pending.raw:
                    raise ContractError('final pending descriptor value differs from SAME original bytes')
                self._retire_pending_pointer()
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
            inspected = inspect_packed(target, deadline=self.collection.deadline,
                max_owned_bytes=self.collection.budget.limit - self.collection.budget.used,
                database_original=pending.database, budget=self.collection.budget)
            if final_context is not None:
                facts = json.loads(self._pending_facts)
                if inspected['validated']['seal']['logical_sha256'] != facts['storage']['logical_sha256']:
                    raise ContractError('public final original logical DB seal differs from accepted pending Source')
                del facts
            del inspected
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
            if final_context is not None:
                self._prepare_public_facts()
            result = PackedRunResult(self.envelope, json.loads(self._controls["report.json"]), CollectionLocator(FORMAT, target),
                self._terminal_originals[1].sha256, self._result_metrics("published"))
            self._finish_readonly_handoff()
            if retained_input_guard is not None:
                retained_input_guard()
            self._final_readonly_checks(diagnostic=False)
            if self._reservation is not None:
                self._controls = None
                self._reservation.close()
            if self._facts_reservation is not None:
                self._pending_facts = b''
                self._facts_reservation.close()
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
