"""Group Pack database command owner and original clean-child transport."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from gf_common.group_pack import FORMAT, MIB, Json, StorageSettings
from gf_training.group_pack_db import (
    DATABASE,
    catalog_update,
    configure,
    finish_wal,
    fsync_directory,
    identity_text,
    new_database,
    seal_transaction,
    store_control,
    store_controls,
    submit_group,
)
from gf_training.group_pack_resources import (
    WRITER_BASE,
    CleanupError,
    close_resources,
    read_bytes,
    receive_header,
    remaining,
    retain_cleanup,
    send_frame,
)
from gf_training.group_pack_source import OriginalFile, inspect_packed

COMMON = {"version", "request_id", "command"}
COMMANDS = {
    "init": {"directory", "run_id", "settings", "deadline"},
    "submit": {"group_id", "raw_bytes", "record", "episodes"},
    "control": {"name"},
    "seal": {"publication"},
    "seal-metrics": {"publication"},
    "seal-producer": {"publication"},
    "close": set(),
}


def validate_request(value: Json, request_id: int) -> str:
    command = value.get("command")
    if (
        not isinstance(command, str)
        or command not in COMMANDS
        or set(value) != COMMON | COMMANDS[command]
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["request_id"]) is not int
        or value["request_id"] != request_id
    ):
        raise ValueError("invalid Writer request")
    return command


def _settings(value: Json) -> StorageSettings:
    if not isinstance(value, dict) or set(value) != {
        "db_root",
        "ram_root",
        "budget_bytes",
        "queue_bytes",
        "max_pack_bytes",
    }:
        raise ValueError("invalid Writer settings")
    return StorageSettings(
        db_root=Path(value["db_root"]),
        ram_root=Path(value["ram_root"]),
        budget_bytes=value["budget_bytes"],
        queue_bytes=value["queue_bytes"],
        max_pack_bytes=value["max_pack_bytes"],
    )


def _retain_failure(owner: Any, error: BaseException, *, clear_frames: bool = False) -> None:
    """Adopt real nested owners across BOTH exception links before dropping frames."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)
        retain_cleanup(owner, failure)
        if clear_frames:
            # clear_frames deliberately skips executing frames. Their owners
            # must retire current locals before refunding their own byte lease.
            traceback.clear_frames(failure.__traceback__)
            failure.__traceback__ = None


class GroupPackDatabase:
    """One command state machine; construct and call only on the DB owner thread.

    The transport supplies its real memory allowance, without an invented process
    baseline. This owner retains the actual SQLite and original-file handles.
    """

    def __init__(self, *, max_owned_bytes: int | None = None) -> None:
        self.deadline = time.monotonic() + 30
        self.request_id = 0
        self.settings: StorageSettings | None = None
        self.directory: Path | None = None
        self.run_id: str | None = None
        self.db: sqlite3.Connection | None = None
        self.sealed = False
        self._sealed_logical_sha256: str | None = None
        self.database_original: OriginalFile | None = None
        self.originals: list[OriginalFile] = []
        self.max_owned_bytes = max_owned_bytes
        self.finished = False
        self._cleanup_children: list[Any] = []

    @property
    def max_payload_bytes(self) -> int:
        return self.settings.max_pack_bytes + MIB if self.settings else 0

    def _prepare_existing(self) -> None:
        assert self.settings is not None and self.directory is not None and self.run_id is not None
        assert self.db is not None and self.database_original is not None
        self.database_original.check_identity()
        configure(self.db, self.deadline)

    def prepare(self, request: Json, payload_size: int) -> str:
        """Validate before a transport retains/reads its bounded payload."""
        if self.finished:
            raise ValueError("Writer already finished")
        command = validate_request(request, self.request_id)
        if self.request_id == 0 and command != "init":
            raise ValueError("Writer must initialize first")
        if self.request_id != 0 and command == "init":
            raise ValueError("Writer already initialized")
        if command not in ("submit", "control") and payload_size:
            raise ValueError("unexpected Writer payload")
        if command != "init":
            self._prepare_existing()
        if command == "submit":
            assert self.settings is not None
            if (
                type(request["raw_bytes"]) is not int
                or request["raw_bytes"] < 16
                or request["raw_bytes"] > self.settings.max_pack_bytes
            ):
                raise ValueError("invalid raw size")
        elif command == "control":
            assert self.settings is not None
            if payload_size > min(self.settings.max_pack_bytes, 64 * MIB):
                raise ValueError("control exceeds bound")
        return command

    def execute(self, request: Json, payload: bytes | None = None) -> Any:
        command = self.prepare(request, len(payload) if payload is not None else 0)
        return self._execute(request, payload, command)

    def execute_controls(self, request_id: int, values: tuple[tuple[str, bytes], ...]) -> None:
        """Batch controls retained by the same-process writer on its original thread."""
        if self.finished:
            raise ValueError("Writer already finished")
        if type(request_id) is not int or request_id != self.request_id:
            raise ValueError("invalid Writer request")
        if request_id == 0:
            raise ValueError("Writer must initialize first")
        self._prepare_existing()
        assert self.settings is not None and self.max_owned_bytes is not None
        assert self.db is not None and self.database_original is not None
        limit = min(self.settings.max_pack_bytes, 64 * MIB)
        if any(type(data) is not bytes or len(data) > limit for _, data in values):
            raise ValueError("control exceeds bound")
        try:
            store_controls(self.db, values, deadline=self.deadline)
            self.database_original.recheck()
            remaining(self.deadline)
            self.request_id += 1
        except BaseException as error:
            _retain_failure(self, error)
            raise

    def _execute(self, request: Json, payload: bytes | None, command: str) -> Any:
        deadline, settings, directory, run_id = (
            self.deadline,
            self.settings,
            self.directory,
            self.run_id,
        )
        db, sealed, database_original = self.db, self.sealed, self.database_original
        try:
            if command == "init":
                if (
                    type(request["deadline"]) not in (int, float)
                    or request["deadline"] != request["deadline"]
                ):
                    raise ValueError("invalid deadline")
                deadline = request["deadline"]
                remaining(deadline)
                settings = _settings(request["settings"])
                if self.max_owned_bytes is None:
                    raise ValueError("database memory allowance required")
                directory = Path(request["directory"])
                run_id = identity_text(request["run_id"])
                database_original = self._open_original(directory / DATABASE, deadline, create=True)
                try:
                    db = new_database(directory / DATABASE, run_id, deadline, database_original.fd)
                except BaseException as error:
                    # new_database already attempts close, and exposes the SAME
                    # connection only if that close remains unconfirmed.
                    db = error.__dict__.get("cleanup_owner")
                    raise
                database_original.check_identity()
                catalog_update(settings, directory / DATABASE, run_id, "writing", deadline)
                result = None
            else:
                assert settings is not None and directory is not None and run_id is not None
                assert self.max_owned_bytes is not None
                assert db is not None
                assert database_original is not None
                if command == "submit":
                    # Coordinator owns matching reservation through this receive/commit.
                    assert payload is not None
                    submit_group(
                        db,
                        run_id,
                        request["group_id"],
                        payload,
                        raw_bytes=request["raw_bytes"],
                        record=request["record"],
                        episodes=request["episodes"],
                        max_pack_bytes=settings.max_pack_bytes,
                    )
                    result = None
                elif command == "control":
                    assert payload is not None
                    store_control(db, request["name"], payload)
                    result = None
                elif command in ("seal", "seal-metrics", "seal-producer"):
                    publication = request["publication"]
                    if publication not in ("published", "pending"):
                        raise ValueError("invalid publication")
                    original_seal = seal_transaction(
                        db, run_id, max_owned_bytes=self.max_owned_bytes
                    )
                    controls = [
                        {"name": name, "bytes": size, "sha256": sha}
                        for name, size, sha in db.execute(
                            "SELECT name,bytes,sha256 FROM controls ORDER BY name"
                        )
                    ]
                    checkpoint_metrics = finish_wal(db)
                    db.close()
                    db = None
                    database_original.check_identity()
                    os.fsync(database_original.fd)
                    fsync_directory(directory)
                    if command != "seal-producer":
                        database_original.freeze_database()
                    if command == "seal-producer":
                        # Mutable SQLite/WAL handles and the creation O_RDWR FD
                        # retire before the single final physical DB hash.
                        original = database_original
                        readonly = self._open_original(directory / DATABASE, deadline)
                        if (readonly.identity.st_dev, readonly.identity.st_ino) != (
                            original.identity.st_dev,
                            original.identity.st_ino,
                        ):
                            self._close_original(readonly)
                            raise ValueError("producer database original changed")
                        self._close_original(original)
                        database_original = readonly
                        database_original.freeze_database()
                        validated: Any = {
                            "validated": {"seal": original_seal, "index": {"controls": controls}},
                            "database_bytes": database_original.identity.st_size,
                            "database_sha256": database_original.sha256,
                        }
                    else:
                        validated = inspect_packed(
                            directory / DATABASE,
                            deadline=deadline,
                            max_owned_bytes=self.max_owned_bytes,
                            allow_pending=True,
                            without_descriptor=True,
                            database_original=database_original,
                        )
                        if validated["validated"]["seal"] != original_seal:
                            raise ValueError("original Writer seal identity changed")
                    database_original.recheck()
                    result = {
                        "format": FORMAT,
                        "schema_version": 1,
                        "run_id": run_id,
                        "seal_id": validated["validated"]["seal"]["seal_id"],
                        "database": DATABASE,
                        "database_bytes": validated["database_bytes"],
                        "database_sha256": validated["database_sha256"],
                        "controls": validated["validated"]["index"]["controls"],
                        "publication": publication,
                    }
                    catalog_update(
                        settings, directory / DATABASE, run_id, "sealed", deadline, result
                    )
                    database_original.recheck()
                    if command in ("seal-metrics", "seal-producer"):
                        result = {"descriptor": result, "checkpoint_metrics": checkpoint_metrics}
                    self._sealed_logical_sha256 = original_seal["logical_sha256"]
                    sealed = True
                elif command == "close":
                    db.close()
                    db = None
                    result = None
                else:
                    raise ValueError("invalid Writer command")
            assert database_original is not None
            database_original.recheck()
            if command in ("seal", "seal-metrics", "seal-producer", "close"):
                self._close_original(database_original)
            remaining(deadline)
            self.request_id += 1
            self.finished = command in ("seal", "seal-metrics", "seal-producer", "close")
            return result
        except BaseException as error:
            _retain_failure(self, error)
            raise
        finally:
            self.deadline, self.settings, self.directory, self.run_id = (
                deadline,
                settings,
                directory,
                run_id,
            )
            self.db, self.sealed, self.database_original = db, sealed, database_original

    def _open_original(self, path: Path, deadline: float, *, create: bool = False) -> OriginalFile:
        try:
            original = OriginalFile(path, deadline, mutable=True, create=create)
        except BaseException as error:
            nested = error.__dict__.get("cleanup_owner")
            if nested is not None:
                self.originals.append(nested)
            _retain_failure(self, error)
            raise
        self.originals.append(original)
        return original

    def _close_original(self, original: OriginalFile) -> None:
        original.close()
        # Positive close only; an uncertain release remains strongly reachable.
        self.originals[:] = [item for item in self.originals if item is not original]
        self._cleanup_children[:] = [
            item for item in self._cleanup_children if item is not original
        ]

    def fail(self) -> list[str]:
        """Retire actual handles; retain every unconfirmed owner for safe retry."""
        cleanup_errors: list[str] = []
        if self.db is not None:
            database = self.db
            try:
                database.rollback()
            except BaseException as cleanup:  # noqa: BLE001 -- original owner cleanup
                _retain_failure(self, cleanup, clear_frames=True)
                cleanup_errors.append(type(cleanup).__name__)
            try:
                database.close()
            except BaseException as cleanup:  # noqa: BLE001 -- original owner cleanup
                _retain_failure(self, cleanup, clear_frames=True)
                cleanup_errors.append(type(cleanup).__name__)
            else:
                self.db = None
                self._cleanup_children[:] = [
                    item for item in self._cleanup_children if item is not database
                ]
        for original in tuple(self.originals):
            try:
                self._close_original(original)
            except BaseException as cleanup:  # noqa: BLE001 -- original owner cleanup
                _retain_failure(self, cleanup, clear_frames=True)
                cleanup_errors.append(type(cleanup).__name__)
        try:
            close_resources(self, [])
        except BaseException as cleanup:  # noqa: BLE001 -- actual nested owner cleanup
            _retain_failure(self, cleanup, clear_frames=True)
            cleanup_errors.append(type(cleanup).__name__)
        if (
            self.settings is not None
            and self.directory is not None
            and self.run_id is not None
            and not self.sealed
        ):
            try:
                catalog_update(
                    self.settings,
                    self.directory / DATABASE,
                    self.run_id,
                    "failed",
                    min(time.monotonic() + 2, self.deadline),
                )
            except BaseException as cleanup:  # noqa: BLE001 -- original owner cleanup
                _retain_failure(self, cleanup, clear_frames=True)
                cleanup_errors.append(type(cleanup).__name__)
        self.finished = True
        return cleanup_errors

    def close(self) -> None:
        errors = self.fail()
        if errors:
            raise CleanupError("database cleanup remains unconfirmed: " + ",".join(errors), self)


def main() -> int:
    input_fd, output_fd = sys.stdin.fileno(), sys.stdout.fileno()
    os.set_blocking(input_fd, False)
    os.set_blocking(output_fd, False)
    owner = GroupPackDatabase()
    request_id = 0
    try:
        while True:
            request, payload_size = receive_header(
                input_fd, owner.deadline, owner.max_payload_bytes
            )
            command = owner.prepare(request, payload_size)
            if command == "init":
                owner.max_owned_bytes = _settings(request["settings"]).budget_bytes - WRITER_BASE
            payload = (
                read_bytes(input_fd, payload_size, owner.deadline)
                if command in ("submit", "control")
                else None
            )
            result = owner._execute(request, payload, command)
            del payload, request
            send_frame(
                output_fd,
                {
                    "version": 1,
                    "request_id": request_id,
                    "status": "ok",
                    "result": result,
                    "error": None,
                },
                b"",
                owner.deadline,
            )
            request_id += 1
            if owner.finished:
                return 0
    except BaseException as exc:  # noqa: BLE001 -- child reports generic failure and exits
        cleanup_errors = owner.fail()
        try:
            send_frame(
                output_fd,
                {
                    "version": 1,
                    "request_id": request_id,
                    "status": "error",
                    "result": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": (
                            "storage operation failed; cleanup uncertain"
                            if cleanup_errors
                            else "storage operation failed"
                        ),
                    },
                },
                b"",
                time.monotonic() + 1,
            )
        except BaseException as cleanup:  # noqa: BLE001 -- nonzero exit is authoritative
            cleanup_errors.append(type(cleanup).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
