"""Fixed clean-child entry point for group Pack SQLite writes. No policy inputs."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

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
    submit_group,
)
from gf_training.group_pack_resources import (
    WRITER_BASE,
    read_bytes,
    receive_header,
    remaining,
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


def main() -> int:
    input_fd, output_fd = sys.stdin.fileno(), sys.stdout.fileno()
    os.set_blocking(input_fd, False)
    os.set_blocking(output_fd, False)
    deadline = time.monotonic() + 30
    request_id = 0
    settings: StorageSettings | None = None
    directory: Path | None = None
    run_id: str | None = None
    db: sqlite3.Connection | None = None
    sealed = False
    database_original: OriginalFile | None = None
    try:
        while True:
            request, payload_size = receive_header(
                input_fd, deadline, settings.max_pack_bytes + MIB if settings else 0
            )
            command = validate_request(request, request_id)
            if request_id == 0 and command != "init":
                raise ValueError("Writer must initialize first")
            if request_id != 0 and command == "init":
                raise ValueError("Writer already initialized")
            if command not in ("submit", "control") and payload_size:
                raise ValueError("unexpected Writer payload")
            if command == "init":
                if (
                    type(request["deadline"]) not in (int, float)
                    or request["deadline"] != request["deadline"]
                ):
                    raise ValueError("invalid deadline")
                deadline = request["deadline"]
                remaining(deadline)
                settings = _settings(request["settings"])
                directory = Path(request["directory"])
                run_id = identity_text(request["run_id"])
                database_original = OriginalFile(
                    directory / DATABASE, deadline, mutable=True, create=True
                )
                db = new_database(directory / DATABASE, run_id, deadline, database_original.fd)
                database_original.check_identity()
                catalog_update(settings, directory / DATABASE, run_id, "writing", deadline)
                result = None
            else:
                assert settings is not None and directory is not None and run_id is not None
                assert db is not None
                assert database_original is not None
                database_original.check_identity()
                configure(db, deadline)
                if command == "submit":
                    if (
                        type(request["raw_bytes"]) is not int
                        or request["raw_bytes"] < 16
                        or request["raw_bytes"] > settings.max_pack_bytes
                    ):
                        raise ValueError("invalid raw size")
                    # Coordinator owns matching reservation through this receive/commit.
                    payload = read_bytes(input_fd, payload_size, deadline)
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
                    if payload_size > min(settings.max_pack_bytes, 64 * MIB):
                        raise ValueError("control exceeds bound")
                    payload = read_bytes(input_fd, payload_size, deadline)
                    store_control(db, request["name"], payload)
                    result = None
                elif command in ("seal", "seal-metrics"):
                    publication = request["publication"]
                    if publication not in ("published", "pending"):
                        raise ValueError("invalid publication")
                    original_seal = seal_transaction(
                        db, run_id, max_owned_bytes=settings.budget_bytes - WRITER_BASE
                    )
                    checkpoint_metrics = finish_wal(db)
                    db.close()
                    db = None
                    database_original.check_identity()
                    os.fsync(database_original.fd)
                    fsync_directory(directory)
                    database_original.freeze()
                    validated = inspect_packed(
                        directory / DATABASE,
                        deadline=deadline,
                        max_owned_bytes=settings.budget_bytes - WRITER_BASE,
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
                    if command == "seal-metrics":
                        result = {"descriptor": result, "checkpoint_metrics": checkpoint_metrics}
                    sealed = True
                elif command == "close":
                    db.close()
                    db = None
                    result = None
                else:
                    raise ValueError("invalid Writer command")
            assert database_original is not None
            database_original.recheck()
            if command in ("seal", "seal-metrics", "close"):
                database_original.close()
            if command in ("submit", "control"):
                del payload
            del request
            remaining(deadline)
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
                deadline,
            )
            request_id += 1
            if command in ("seal", "seal-metrics", "close"):
                return 0
    except BaseException as exc:  # noqa: BLE001 -- child reports generic failure and exits
        cleanup_errors: list[str] = []
        # Only bounded, generic diagnostics cross IPC. Never echo metadata/private bytes.
        if db is not None:
            try:
                db.rollback()
                db.close()
            except BaseException as cleanup:  # noqa: BLE001 -- process exits after cleanup
                cleanup_errors.append(type(cleanup).__name__)
            db = None
        if database_original is not None:
            try:
                database_original.close()
            except BaseException as cleanup:  # noqa: BLE001 -- nonzero process exit retires handles
                cleanup_errors.append(type(cleanup).__name__)
        if settings is not None and directory is not None and run_id is not None and not sealed:
            try:
                catalog_update(
                    settings,
                    directory / DATABASE,
                    run_id,
                    "failed",
                    min(time.monotonic() + 2, deadline),
                )
            except BaseException as cleanup:  # noqa: BLE001 -- process exits after cleanup
                cleanup_errors.append(type(cleanup).__name__)
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
        except BaseException as cleanup:  # noqa: BLE001 -- failed pipe; nonzero exit is authoritative
            cleanup_errors.append(type(cleanup).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
