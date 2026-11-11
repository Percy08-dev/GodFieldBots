"""Actual subprocess/SQLite failure windows and retained-source mutation checks."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gf_common.group_pack import FORMAT, MIB, PackMember, canonical_json, encode_pack, unpack_raw
from gf_training.group_pack_db import reconcile_collection
from gf_training.group_pack_resources import receive_header, send_frame
from gf_training.group_pack_source import CollectionLocator, open_artifact_source
from gf_training.group_pack_writer import GroupPackWriter
from tests.test_group_pack_storage import settings, submit, writer


def sealed(root: Path) -> CollectionLocator:
    w = writer(root)
    submit(w, "g")
    w.store_control("manifest", b"original")
    return w.seal()


@pytest.mark.parametrize(
    "mutation",
    [
        "db-replace",
        "db-edit",
        "descriptor-replace",
        "descriptor-edit",
        "symlink",
        "sidecar",
        "extra",
    ],
)
def test_retained_source_refuses_same_size_mutation(tmp_path: Path, mutation: str) -> None:
    locator = sealed(tmp_path)
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 30, max_owned_bytes=512 * MIB
    )
    path = locator.path.parent / "collection.sqlite3"
    if mutation.startswith("descriptor"):
        path = locator.path
    if mutation.endswith("replace"):
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
    elif mutation.endswith("edit"):
        original = path.read_bytes()
        with path.open("r+b") as file:
            file.seek(len(original) - 2)
            file.write(b"XY")
        assert path.stat().st_size == len(original)
    elif mutation == "symlink":
        target = tmp_path / "original.db"
        path.rename(target)
        path.symlink_to(target)
    elif mutation == "sidecar":
        (path.parent / "collection.sqlite3-wal").write_bytes(b"")
    else:
        (path.parent / "foreign").write_bytes(b"")
    with pytest.raises(ValueError):
        source.recheck()
    with pytest.raises(ValueError):
        source.read("manifest")
    source.close()


@pytest.mark.parametrize("mutation", ["file", "replace", "extra", "directory", "symlink"])
def test_directory_originals_and_namespace(tmp_path: Path, mutation: str) -> None:
    (tmp_path / "g").mkdir()
    file = tmp_path / "g/data"
    file.write_bytes(b"\x00\xff")
    source = open_artifact_source(
        CollectionLocator("directory", tmp_path),
        deadline=time.monotonic() + 30,
        max_owned_bytes=MIB,
    )
    assert source.read("g/data") == b"\x00\xff"
    with source.borrow_group("g") as loan:
        assert loan.read("g/data") == b"\x00\xff"
        with pytest.raises(RuntimeError):
            source.close()
    with pytest.raises(RuntimeError):
        loan.read("g/data")
    if mutation == "file":
        file.write_bytes(b"ab")
    elif mutation == "replace":
        replacement = tmp_path / "new"
        replacement.write_bytes(file.read_bytes())
        os.replace(replacement, file)
    elif mutation == "extra":
        (tmp_path / "extra").write_bytes(b"")
    elif mutation == "directory":
        (tmp_path / "g").rename(tmp_path / "old")
        (tmp_path / "g").mkdir()
        (tmp_path / "g/data").write_bytes(b"\x00\xff")
    else:
        file.unlink()
        file.symlink_to("/dev/null")
    with pytest.raises(ValueError):
        source.recheck()
    source.close()


def inject_child(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """Patch only child launch in this test, keep actual OS process/SQLite/IPC."""
    original = subprocess.Popen

    def launch(args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        return original([args[0], "-I", "-c", code], **kwargs)

    monkeypatch.setattr("gf_training.group_pack_writer.subprocess.Popen", launch)


@pytest.mark.parametrize("failure", ["checkpoint", "catalog"])
def test_child_seal_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "checkpoint":
        patch = """
def fail(db):
    raise OSError('checkpoint failed')
w.finish_wal = fail
"""
    else:
        patch = """
original = w.catalog_update
def fail(settings, path, run, state, deadline, descriptor=None):
    if state == 'sealed':
        raise OSError('catalog failed')
    return original(settings, path, run, state, deadline, descriptor)
w.catalog_update = fail
"""
    inject_child(
        monkeypatch,
        "import gf_training.group_pack_worker as w\n" + patch + "\nraise SystemExit(w.main())",
    )
    w = writer(tmp_path)
    submit(w, "g")
    with pytest.raises(RuntimeError):
        w.seal()
    assert w.returncode is not None
    assert not list(w.directory.glob("collection.*.json"))
    assert (w.directory / "collection.sqlite3").exists()
    if failure == "catalog":
        assert (
            reconcile_collection(
                w.directory, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
            )["state"]
            == "sealed-without-descriptor"
        )


def test_real_sqlite_full_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject_child(
        monkeypatch,
        """
import gf_training.group_pack_worker as w
original = w.new_database
def small(*args):
    db = original(*args)
    pages = db.execute('pragma page_count').fetchone()[0]
    db.execute(f'pragma max_page_count={pages}')
    return db
w.new_database = small
raise SystemExit(w.main())
""",
    )
    w = writer(tmp_path)
    pack = encode_pack((PackMember("journal", "large", os.urandom(MIB)),))
    with pytest.raises(RuntimeError):
        w.submit("g", pack, raw_bytes=len(unpack_raw(pack)), record={}, episodes=[])
    assert w.returncode is not None
    with sqlite3.connect(w.directory / "collection.sqlite3") as db:
        assert db.execute("select count(*) from groups").fetchone() == (0,)
        assert db.execute("select count(*) from members").fetchone() == (0,)


def test_open_transaction_kill_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "inside-transaction"
    code = """
import gf_training.group_pack_worker as w
import time
from pathlib import Path
def hold(db, run_id, group_id, pack, **kwargs):
    db.execute('BEGIN')
    db.execute('INSERT INTO groups VALUES (?,?,?,?,?,?)', (run_id, group_id, pack, 'x', 1, b'{}'))
    Path(MARKER).write_text('ready')
    time.sleep(30)
w.submit_group = hold
raise SystemExit(w.main())
""".replace("MARKER", repr(str(marker)))
    inject_child(monkeypatch, code)
    w = writer(tmp_path)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            submit(w, "g")
        except Exception as exc:  # noqa: BLE001 -- return thread failure to test owner
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    limit = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < limit:
        time.sleep(0.01)
    assert marker.exists()
    assert w.owned_bytes > 128 * MIB
    os.kill(w.pid, signal.SIGKILL)
    thread.join(5)
    assert not thread.is_alive() and errors
    assert w.returncode is not None and w.owned_bytes == 0
    with sqlite3.connect(w.directory / "collection.sqlite3") as db:
        assert db.execute("select count(*) from groups").fetchone() == (0,)


def test_stopped_child_times_out_reaps_and_releases(tmp_path: Path) -> None:
    w = GroupPackWriter(
        tmp_path / "collection", "run", settings=settings(tmp_path), deadline=time.monotonic() + 0.7
    )
    os.kill(w.pid, signal.SIGSTOP)
    with pytest.raises(TimeoutError):
        submit(w, "g")
    assert w.returncode is not None
    assert w.owned_bytes == w.queued_bytes == 0
    w.close()


@pytest.mark.parametrize("window", ["link", "fsync", "final-validation", "foreign-replacement"])
def test_descriptor_failure_revokes_only_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    import gf_training.group_pack_writer as module

    w = writer(tmp_path)
    submit(w, "g")
    original_link = os.link
    original_fsync = module.fsync_directory
    if window == "link":

        def bad_link(*args: Any, **kwargs: Any) -> None:
            raise OSError("publication link failed")

        monkeypatch.setattr(module.os, "link", bad_link)
    elif window == "fsync":
        calls = 0

        def bad_fsync(path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("publication sync failed")
            original_fsync(path)

        monkeypatch.setattr(module, "fsync_directory", bad_fsync)
    else:

        def bad_validation(path: Path, **kwargs: Any) -> Any:
            if window == "foreign-replacement":
                path.unlink()
                path.write_bytes(b"foreign bytes")
            raise ValueError("late validation failed")

        monkeypatch.setattr(module, "inspect_packed", bad_validation)
    with pytest.raises((OSError, ValueError)) as failure:
        w.seal()
    assert w.returncode == 0
    descriptor = w.directory / "collection.published.json"
    if window == "foreign-replacement":
        assert descriptor.read_bytes() == b"foreign bytes"
        assert failure.value.cleanup_owner is w
        descriptor.unlink()
        w.close()
    else:
        if window == "link":
            assert not descriptor.exists()
        else:
            assert descriptor.read_bytes() == b"GFB-QUARANTINED-DESCRIPTOR\n"
            with pytest.raises(ValueError):
                open_artifact_source(
                    CollectionLocator(FORMAT, descriptor),
                    deadline=time.monotonic() + 10,
                    max_owned_bytes=512 * MIB,
                )
        assert not list(w.directory.glob("*.tmp"))
    monkeypatch.setattr(module.os, "link", original_link)


def test_publication_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gf_training.group_pack_writer as module

    w = writer(tmp_path)
    submit(w, "g")
    original_publish = w._publish
    target = w.directory / "collection.published.json"

    def race(value: Any) -> Any:
        target.write_bytes(b"foreign")
        return original_publish(value)

    monkeypatch.setattr(w, "_publish", race)
    with pytest.raises(FileExistsError):
        w.seal()
    assert target.read_bytes() == b"foreign"
    assert w.returncode == 0
    assert not list(w.directory.glob("*.tmp"))
    assert module is not None


def test_catalog_is_not_authority_and_original_db_not_reopened(tmp_path: Path) -> None:
    locator = sealed(tmp_path)
    original = (locator.path.parent / "collection.sqlite3").read_bytes()
    catalog = tmp_path / "catalog/catalog.sqlite3"
    with sqlite3.connect(catalog) as db:
        assert db.execute("pragma synchronous").fetchone() == (2,)
        row = db.execute("select state,descriptor_sha256 from collections").fetchone()
        assert row is not None and row[0] == "sealed"
        db.execute("update collections set state='writing',descriptor_sha256='wrong'")
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
    )
    source.recheck()
    source.close()
    assert (locator.path.parent / "collection.sqlite3").read_bytes() == original
    assert sorted(p.name for p in locator.path.parent.iterdir()) == [
        "collection.published.json",
        "collection.sqlite3",
    ]
    locator.path.unlink()
    assert (
        reconcile_collection(
            locator.path.parent, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
        )["state"]
        == "sealed-without-descriptor"
    )
    with pytest.raises(ValueError):
        open_artifact_source(
            CollectionLocator(FORMAT, locator.path.parent / "collection.sqlite3"),
            deadline=time.monotonic() + 10,
            max_owned_bytes=512 * MIB,
        )


def test_truncated_ipc_and_oversized_header_actual_child(tmp_path: Path) -> None:
    import sys

    for index, oversized in enumerate((False, True)):
        directory = tmp_path / str(index)
        directory.mkdir()
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "gf_training.group_pack_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        deadline = time.monotonic() + 5
        init = {
            "version": 1,
            "request_id": 0,
            "command": "init",
            "directory": str(directory),
            "run_id": "run",
            "settings": {
                "db_root": str(tmp_path / "catalog"),
                "ram_root": str(tmp_path / "ram"),
                "budget_bytes": 512 * MIB,
                "queue_bytes": 64 * MIB,
                "max_pack_bytes": 16 * MIB,
            },
            "deadline": deadline,
        }
        send_frame(process.stdin.fileno(), init, b"", deadline)
        response, size = receive_header(process.stdout.fileno(), deadline, 0)
        assert response["status"] == "ok" and size == 0
        if oversized:
            process.stdin.write(struct.pack("<QQ", MIB + 1, 0))
        else:
            metadata = canonical_json(
                {"version": 1, "request_id": 1, "command": "control", "name": "PRIVATE"}
            )
            process.stdin.write(struct.pack("<QQ", len(metadata), 10) + metadata + b"12345")
        process.stdin.close()
        response, _ = receive_header(process.stdout.fileno(), deadline, 0)
        assert response["status"] == "error"
        assert b"PRIVATE" not in canonical_json(response)
        assert process.wait(timeout=5) == 1
        process.stdout.close()
        assert process.stderr is not None
        assert process.stderr.read() == b""
        process.stderr.close()
        with sqlite3.connect(directory / "collection.sqlite3") as db:
            assert db.execute("select count(*) from controls").fetchone() == (0,)


def test_source_capacity_and_detached_bytes_accounting(tmp_path: Path) -> None:
    locator = sealed(tmp_path)
    with pytest.raises(ValueError):
        open_artifact_source(locator, deadline=time.monotonic() + 10, max_owned_bytes=64 * MIB)
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
    )
    before = source.owned_bytes
    source.read("manifest")
    assert source.owned_bytes == before + 2 * len(b"original")
    with source.borrow_group("g") as loan:
        loan.read("g/e0")
        resident = source.owned_bytes
    source.evict_group()
    assert before < source.owned_bytes < resident
    source.close()
    assert source.owned_bytes == 0


def test_seal_drains_every_already_admitted_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    w = writer(tmp_path)
    admitted = threading.Event()
    release = threading.Event()
    sealed_event = threading.Event()
    original_admission = w._admission
    errors: list[BaseException] = []
    locators: list[CollectionLocator] = []

    @contextmanager
    def hold(*args: Any) -> Iterator[None]:
        with original_admission(*args):
            admitted.set()
            assert release.wait(5)
            yield

    monkeypatch.setattr(w, "_admission", hold)

    def run_submit() -> None:
        try:
            submit(w, "g")
        except Exception as exc:  # noqa: BLE001 -- return thread failure to test owner
            errors.append(exc)

    def run_seal() -> None:
        try:
            locators.append(w.seal())
        except Exception as exc:  # noqa: BLE001 -- return thread failure to test owner
            errors.append(exc)
        finally:
            sealed_event.set()

    submitting = threading.Thread(target=run_submit)
    submitting.start()
    assert admitted.wait(5)
    sealing = threading.Thread(target=run_seal)
    sealing.start()
    try:
        assert not sealed_event.wait(0.1)
    finally:
        release.set()
        submitting.join(5)
        sealing.join(5)
        w.close()
    assert not errors and len(locators) == 1
    with sqlite3.connect(w.directory / "collection.sqlite3") as db:
        assert db.execute("select count(*) from groups").fetchone() == (1,)


def test_recovery_reports_unsealed_without_repair(tmp_path: Path) -> None:
    w = writer(tmp_path)
    submit(w, "g")
    w.close()
    original = (w.directory / "collection.sqlite3").read_bytes()
    assert (
        reconcile_collection(
            w.directory, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
        )["state"]
        == "failed/unsealed"
    )
    assert (w.directory / "collection.sqlite3").read_bytes() == original


def test_publication_replacement_during_fd_invalidation_leaves_foreign_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gf_training.group_pack_writer as module

    w = writer(tmp_path)
    submit(w, "g")
    target = w.directory / "collection.published.json"
    captured: list[bytes] = []
    original_fstat = os.fstat
    original_truncate = os.ftruncate
    invalidating = False

    def fail_validation(*args: Any, **kwargs: Any) -> Any:
        nonlocal invalidating
        invalidating = True
        raise ValueError("publication failure")

    def race_stat(fd: int) -> os.stat_result:
        result = original_fstat(fd)
        if invalidating and fd == w._publication_fd and target.exists():
            target.unlink()
            target.write_bytes(b"foreign replacement")
        return result

    def capture_truncate(fd: int, length: int) -> None:
        captured.append(os.pread(fd, 32, 0))
        original_truncate(fd, length)

    monkeypatch.setattr(module, "inspect_packed", fail_validation)
    monkeypatch.setattr(module.os, "fstat", race_stat)
    monkeypatch.setattr(module.os, "ftruncate", capture_truncate)
    with pytest.raises(ValueError):
        w.seal()
    assert captured
    assert target.read_bytes() == b"foreign replacement"
    target.unlink()
    w.close()


@pytest.mark.parametrize("mode", ["lost-ack", "truncated-ack", "nonzero-exit", "extra-frame"])
def test_seal_ack_and_exit_must_both_be_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    code = """
import os
import gf_training.group_pack_worker as w
original = w.send_frame
def send(fd, metadata, payload, deadline):
    if isinstance(metadata.get('result'), dict):
        if MODE == 'lost-ack':
            os._exit(0)
        if MODE == 'truncated-ack':
            os.write(fd, b'12345678')
            os._exit(0)
        original(fd, metadata, payload, deadline)
        if MODE == 'extra-frame':
            os.write(fd, b'extra')
            os._exit(0)
        os._exit(7)
    original(fd, metadata, payload, deadline)
w.send_frame = send
raise SystemExit(w.main())
""".replace("MODE", repr(mode))
    inject_child(monkeypatch, code)
    w = writer(tmp_path)
    submit(w, "g")
    with pytest.raises(RuntimeError):
        w.seal()
    assert w.returncode is not None
    assert not list(w.directory.glob("collection.*.json"))
    assert (
        reconcile_collection(
            w.directory, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
        )["state"]
        == "sealed-without-descriptor"
    )


def test_directory_rechecks_namespace_after_hash_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "x").write_bytes(b"original")
    source = open_artifact_source(
        CollectionLocator("directory", tmp_path),
        deadline=time.monotonic() + 10,
        max_owned_bytes=MIB,
    )
    original = source._files["x"].recheck

    def changed() -> None:
        original()
        (tmp_path / "late").write_bytes(b"late")

    monkeypatch.setattr(source._files["x"], "recheck", changed)
    with pytest.raises(ValueError):
        source.recheck()
    source.close()


def test_uncertain_reap_retains_primary_and_reservations_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gf_training.group_pack_resources import CleanupError

    w = writer(tmp_path)
    original_retire = w._retire
    original_rpc = w._rpc
    primary = TimeoutError("initiating timeout")

    def uncertain(*args: Any, **kwargs: Any) -> None:
        raise CleanupError("injected unconfirmed reap", w)

    def fail_rpc(*args: Any, **kwargs: Any) -> Any:
        w._fail(primary)
        raise primary

    monkeypatch.setattr(w, "_retire", uncertain)
    monkeypatch.setattr(w, "_rpc", fail_rpc)
    try:
        with pytest.raises(TimeoutError) as raised:
            submit(w, "g")
        assert raised.value is primary and primary.cleanup_owner is w
        assert w.owned_bytes > 128 * MIB and w.queued_bytes > 0
    finally:
        monkeypatch.setattr(w, "_retire", original_retire)
        monkeypatch.setattr(w, "_rpc", original_rpc)
        w.close()
    assert w.returncode is not None
    assert w.owned_bytes == w.queued_bytes == 0


def test_close_cancels_inflight_writer_and_wakes_waiters(tmp_path: Path) -> None:
    w = writer(tmp_path)
    os.kill(w.pid, signal.SIGSTOP)
    errors: list[BaseException] = []
    finished = threading.Event()

    def run() -> None:
        try:
            submit(w, "g")
        except Exception as exc:  # noqa: BLE001 -- report thread failure
            errors.append(exc)

    def stop() -> None:
        try:
            w.close()
        except Exception as exc:  # noqa: BLE001 -- report thread failure
            errors.append(exc)
        finally:
            finished.set()

    submitting = threading.Thread(target=run)
    submitting.start()
    limit = time.monotonic() + 5
    while not w.queued_bytes and time.monotonic() < limit:
        time.sleep(0.01)
    closing = threading.Thread(target=stop)
    closing.start()
    try:
        assert finished.wait(1)
    finally:
        if w.returncode is None:
            os.kill(w.pid, signal.SIGCONT)
        submitting.join(5)
        closing.join(5)
        w.close()
    assert w.returncode is not None
    assert w.owned_bytes == w.queued_bytes == 0
    assert not submitting.is_alive() and not closing.is_alive()


def test_directory_loans_reuse_and_release_group_reservation(tmp_path: Path) -> None:
    (tmp_path / "g").mkdir()
    (tmp_path / "g/x").write_bytes(b"bytes")
    source = open_artifact_source(
        CollectionLocator("directory", tmp_path),
        deadline=time.monotonic() + 10,
        max_owned_bytes=MIB,
    )
    before = source._budget.used
    with source.borrow_group("g") as loan:
        assert loan.read("g/x") == b"bytes"
        resident = source._budget.used
        assert loan.read("g/x") == b"bytes"
        assert source._budget.used == resident
    source.evict_group()
    assert source._budget.used == before
    source.close()


@pytest.mark.parametrize("kind", ["directory", "packed"])
def test_source_close_failure_retains_retry_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "packed":
        locator = sealed(tmp_path)
    else:
        (tmp_path / "x").write_bytes(b"x")
        locator = CollectionLocator("directory", tmp_path)
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
    )
    file = source._originals.database_file if kind == "packed" else source._files["x"]
    original = file.close

    def failed() -> None:
        raise OSError("close uncertain")

    monkeypatch.setattr(file, "close", failed)
    with pytest.raises(OSError) as raised:
        source.close()
    assert raised.value.cleanup_owner is source
    monkeypatch.setattr(file, "close", original)
    source.close()


@pytest.mark.parametrize(
    "corruption", ["schema", "metadata", "pack", "member", "control", "raw-size"]
)
def test_shared_readonly_validation_rejects_internal_corruption(
    tmp_path: Path, corruption: str
) -> None:
    from gf_common.group_pack import digest
    from gf_training.group_pack_db import logical_index, validate_database

    locator = sealed(tmp_path)
    locator.path.unlink()
    path = locator.path.parent / "collection.sqlite3"
    with sqlite3.connect(path) as db:
        if corruption == "schema":
            db.execute("create table foreign_table(x)")
        elif corruption == "metadata":
            db.execute("insert into metadata values (?,?)", ("extra", b"{}"))
        elif corruption == "pack":
            data = db.execute("select pack from groups").fetchone()[0]
            db.execute("update groups set pack=?", (data[:-1] + bytes([data[-1] ^ 1]),))
        elif corruption == "member":
            db.execute("update members set length=length+1")
        elif corruption == "control":
            db.execute("update controls set data=?", (b"changed!",))
        else:
            db.execute("update groups set raw_bytes=raw_bytes+1")
        if corruption in ("member", "raw-size"):
            seal = json.loads(
                db.execute("select value from metadata where key='seal'").fetchone()[0]
            )
            seal["logical_sha256"] = digest(canonical_json(logical_index(db, "run-1")))
            db.execute("update metadata set value=? where key='seal'", (canonical_json(seal),))
    from contextlib import closing
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db, pytest.raises(ValueError):
        validate_database(db, deadline=time.monotonic()+10, max_pack_bytes=256*MIB,
                          max_owned_bytes=512*MIB)


def test_terminal_original_close_never_retries_recycled_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gf_training.group_pack_source import OriginalFile

    path = tmp_path / "original"
    path.write_bytes(b"original")
    owner = OriginalFile(path, time.monotonic() + 10)
    attempted = owner.fd
    real_close = os.close
    foreign = -1

    def released_then_error(fd: int) -> None:
        nonlocal foreign
        if fd == attempted:
            real_close(fd)
            foreign = os.open("/dev/null", os.O_RDONLY)
            assert foreign == attempted
            raise OSError("after actual release")
        real_close(fd)

    monkeypatch.setattr(os, "close", released_then_error)
    try:
        with pytest.raises(OSError) as raised:
            owner.close()
        assert raised.value.cleanup_owner is owner
        assert owner.fd == -1
        monkeypatch.setattr(os, "close", real_close)
        with pytest.raises(RuntimeError):
            owner.close()
        assert os.fstat(foreign)
    finally:
        monkeypatch.setattr(os, "close", real_close)
        if foreign >= 0:
            real_close(foreign)


@pytest.mark.parametrize("kind", ["original", "directory", "inspection"])
def test_failed_construction_keeps_primary_and_actual_cleanup_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from gf_training.group_pack_source import DirectorySource, OriginalFile, inspect_packed

    (tmp_path / "x").write_bytes(b"x")
    if kind == "directory":
        (tmp_path / "y").write_bytes(b"y")
    primary = ValueError("initiating failure")
    original_close = OriginalFile.close
    captured: list[Any] = []

    def close_failed(file: Any) -> None:
        captured.append(file)
        raise OSError("injected close failure")

    def reject(*args: Any) -> None:
        raise primary

    if kind == "inspection":
        (tmp_path / "x").unlink()
        locator = sealed(tmp_path)
        monkeypatch.setattr(OriginalFile, "close", close_failed)
        with pytest.raises(OSError) as raised:
            inspect_packed(locator.path, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB)
    else:
        monkeypatch.setattr(OriginalFile, "close", close_failed)
        monkeypatch.setattr(
            OriginalFile if kind == "original" else DirectorySource, "recheck", reject
        )
        with pytest.raises(ValueError) as raised:
            if kind == "original":
                OriginalFile(tmp_path / "x", time.monotonic() + 10)
            else:
                DirectorySource(tmp_path, deadline=time.monotonic() + 10, max_owned_bytes=MIB)
        assert raised.value is primary
    owner = raised.value.cleanup_owner
    assert owner is not None and captured
    if kind == "directory":
        assert owner._budget.used > 0 and len(owner._files) == 2
        assert len(captured) == 2
    monkeypatch.setattr(OriginalFile, "close", original_close)
    owner.close()
    assert all(file.fd == -1 for file in captured)


@pytest.mark.parametrize("kind", ["capture", "directory-recheck", "packed-recheck"])
def test_namespace_enumeration_stops_before_unadmitted_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from gf_training.group_pack_source import DirectorySource

    source: Any = None
    if kind == "packed-recheck":
        locator = sealed(tmp_path)
        source = open_artifact_source(
            locator, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB
        )
        directory = locator.path.parent
        unexpected = directory / "unexpected"
    else:
        directory = tmp_path
        unexpected = directory / "unexpected"
        if kind == "directory-recheck":
            source = DirectorySource(directory, deadline=time.monotonic() + 10, max_owned_bytes=MIB)
    unexpected.write_bytes(b"")
    real_scandir = os.scandir
    with real_scandir(directory) as iterator:
        entry = next(item for item in iterator if item.name == "unexpected")
    consumed = 0

    class BoundedScan:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def __iter__(self) -> Any:
            nonlocal consumed
            consumed += 1
            yield entry
            raise AssertionError("enumerated beyond first unadmitted/unexpected entry")

    def scan(path: Any) -> Any:
        return BoundedScan() if Path(path) == directory else real_scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    try:
        with pytest.raises(ValueError):
            if kind == "capture":
                DirectorySource(directory, deadline=time.monotonic() + 10, max_owned_bytes=65536)
            else:
                source.recheck()
        assert consumed == 1
    finally:
        monkeypatch.setattr(os, "scandir", real_scandir)
        if source is not None:
            source.close()


def test_catalog_first_use_is_one_immediate_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import gf_training.group_pack_db as module

    original_connect = sqlite3.connect
    barrier = threading.Barrier(2)

    class Cursor:
        def __init__(self, cursor: Any, synchronize: bool) -> None:
            self.cursor = cursor
            self.synchronize = synchronize

        def fetchone(self) -> Any:
            row = self.cursor.fetchone()
            if self.synchronize:
                barrier.wait(5)
            return row

        def __getattr__(self, name: str) -> Any:
            return getattr(self.cursor, name)

    class Connection:
        def __init__(self, db: Any) -> None:
            self.db = db

        def execute(self, sql: str, *args: Any) -> Any:
            cursor = self.db.execute(sql, *args)
            return Cursor(cursor, sql == "PRAGMA user_version" and not self.db.in_transaction)

        def __enter__(self) -> Any:
            self.db.__enter__()
            return self

        def __exit__(self, *args: object) -> Any:
            return self.db.__exit__(*args)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.db, name)

    monkeypatch.setattr(
        module.sqlite3, "connect", lambda *a, **k: Connection(original_connect(*a, **k))
    )

    def insert(index: int) -> None:
        module.catalog_update(
            settings(tmp_path),
            tmp_path / f"collection-{index}.sqlite3",
            str(index),
            "writing",
            time.monotonic() + 10,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(insert, (1, 2)))
    with original_connect(tmp_path / "catalog/catalog.sqlite3") as db:
        assert db.execute("select count(*) from collections").fetchone() == (2,)


@pytest.mark.parametrize("replacement", ["different", "identical"])
def test_child_post_close_db_replacement_refuses_original_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    import shutil

    if replacement == "different":
        foreign = GroupPackWriter(
            tmp_path / "foreign",
            "run-1",
            settings=settings(tmp_path),
            deadline=time.monotonic() + 20,
        )
        submit(foreign, "evil")
        foreign_locator = foreign.seal()
        foreign.close()
        replacement_path = tmp_path / "replacement.sqlite3"
        shutil.copyfile(foreign_locator.path.parent / "collection.sqlite3", replacement_path)
    else:
        replacement_path = tmp_path / "replacement.sqlite3"
    marker = tmp_path / "replacement-observed"
    code = """
import os, shutil
from pathlib import Path
import gf_training.group_pack_worker as w
original = w.fsync_directory
replaced = False
def swap(directory):
    global replaced
    if not replaced:
        replaced = True
        target = directory / 'collection.sqlite3'
        replacement = Path(REPLACEMENT)
        if IDENTICAL:
            shutil.copyfile(target, replacement)
        assert target.stat().st_size == replacement.stat().st_size
        os.replace(replacement, target)
        Path(MARKER).write_text('replaced')
    original(directory)
w.fsync_directory = swap
raise SystemExit(w.main())
""".replace("REPLACEMENT", repr(str(replacement_path))).replace(
        "IDENTICAL", repr(replacement == "identical")
    )
    code = code.replace("MARKER", repr(str(marker)))
    inject_child(monkeypatch, code)
    w = writer(tmp_path)
    submit(w, "good")
    with pytest.raises((RuntimeError, ValueError)):
        w.seal()
    assert w.returncode is not None and marker.read_text() == "replaced"
    assert not list(w.directory.glob("collection.*.json"))
    preserved = (w.directory / "collection.sqlite3").read_bytes()
    w.close()
    assert (w.directory / "collection.sqlite3").read_bytes() == preserved


@pytest.mark.parametrize("phase", ["after-init", "after-reap", "during-publication", "final-close"])
def test_parent_retains_db_inode_across_every_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import shutil

    import gf_training.group_pack_writer as module

    w = writer(tmp_path)
    submit(w, "good")
    target = w.directory / "collection.sqlite3"
    copied = tmp_path / "copy.sqlite3"
    replaced = False

    def replace() -> None:
        nonlocal replaced
        if not replaced:
            shutil.copyfile(target, copied)
            os.replace(copied, target)
            replaced = True

    if phase == "after-init":
        replace()
    elif phase == "after-reap":
        original_retire = w._retire

        def retire(*, terminate: bool = False) -> None:
            original_retire(terminate=terminate)
            if not terminate:
                replace()

        monkeypatch.setattr(w, "_retire", retire)
    elif phase == "final-close":
        original_close = w._database_original.close

        def close_original() -> None:
            original_close()
            replace()

        monkeypatch.setattr(w._database_original, "close", close_original)
    else:
        original_inspect = module.inspect_packed

        def inspect(*args: Any, **kwargs: Any) -> Any:
            replace()
            return original_inspect(*args, **kwargs)

        monkeypatch.setattr(module, "inspect_packed", inspect)
    with pytest.raises((RuntimeError, ValueError)):
        w.seal()
    assert w.returncode is not None and replaced
    preserved = target.read_bytes()
    w.close()
    assert target.read_bytes() == preserved
    descriptor = w.directory / "collection.published.json"
    if descriptor.exists():
        assert descriptor.read_bytes() == b"GFB-QUARANTINED-DESCRIPTOR\n"


def test_child_validation_compares_its_exact_original_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = """
from uuid import uuid4
import gf_training.group_pack_worker as w
from gf_common.group_pack import canonical_json
original = w.seal_transaction
def changed(db, *args, **kwargs):
    seal = original(db, *args, **kwargs)
    altered = dict(seal, seal_id=str(uuid4()))
    with db:
        db.execute("update metadata set value=? where key='seal'", (canonical_json(altered),))
    return seal
w.seal_transaction = changed
raise SystemExit(w.main())
"""
    inject_child(monkeypatch, code)
    w = writer(tmp_path)
    submit(w, "g")
    with pytest.raises(RuntimeError):
        w.seal()
    assert not list(w.directory.glob("collection.*.json"))


def test_writer_publication_close_error_never_uses_recycled_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = writer(tmp_path)
    submit(w, "g")
    foreign_path = tmp_path / "foreign-file"
    foreign_path.write_bytes(b"foreign contents")
    real_close = os.close
    real_sync = os.fsync
    candidate = -1
    foreign_fd = -1
    attempted = -1

    def tracked_sync(fd: int) -> None:
        nonlocal candidate
        if fd == w._publication_fd and fd >= 0:
            candidate = fd
        real_sync(fd)

    def released_then_error(fd: int) -> None:
        nonlocal foreign_fd, attempted
        if attempted < 0 and fd == candidate:
            attempted = fd
            real_close(fd)
            foreign_fd = os.open(foreign_path, os.O_RDWR)
            if foreign_fd != fd:
                os.dup2(foreign_fd, fd)
                real_close(foreign_fd)
                foreign_fd = fd
            raise OSError("descriptor released before reported error")
        real_close(fd)

    monkeypatch.setattr(os, "close", released_then_error)
    monkeypatch.setattr(os, "fsync", tracked_sync)
    try:
        with pytest.raises(OSError) as raised:
            w.seal()
        assert raised.value.cleanup_owner is w
        assert w._publication_fd == -1 and w._uncertain_releases
        assert w.owned_bytes >= 128 * MIB
        assert foreign_path.read_bytes() == b"foreign contents"
        assert os.fstat(foreign_fd)
        assert (
            w.directory / "collection.published.json"
        ).read_bytes() == b"GFB-QUARANTINED-DESCRIPTOR\n"
        monkeypatch.setattr(os, "close", real_close)
        with pytest.raises(RuntimeError):
            w.close()
        assert os.fstat(foreign_fd)
    finally:
        monkeypatch.setattr(os, "close", real_close)
        if foreign_fd >= 0:
            real_close(foreign_fd)


def test_directory_sync_cleanup_consumes_fd_and_preserves_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gf_training.group_pack_db import fsync_directory

    primary = OSError("initiating directory sync failure")
    real_close = os.close
    attempted = -1
    foreign = -1

    def fail_sync(fd: int) -> None:
        nonlocal attempted
        attempted = fd
        raise primary

    def fail_close(fd: int) -> None:
        nonlocal foreign
        real_close(fd)
        if fd == attempted:
            foreign = os.open("/dev/null", os.O_RDONLY)
            assert foreign == attempted
            raise OSError("close after release")

    monkeypatch.setattr(os, "fsync", fail_sync)
    monkeypatch.setattr(os, "close", fail_close)
    try:
        with pytest.raises(OSError) as raised:
            fsync_directory(tmp_path)
        assert raised.value is primary
        owner = primary.cleanup_owner
        assert owner.fd == -1 and owner._uncertain_releases
        monkeypatch.setattr(os, "close", real_close)
        with pytest.raises(RuntimeError):
            owner.close()
        assert os.fstat(foreign)
    finally:
        monkeypatch.setattr(os, "close", real_close)
        if foreign >= 0:
            real_close(foreign)


@pytest.mark.parametrize("finish", ["wake", "cancel"])
def test_real_second_caller_parks_on_saturated_byte_capacity(tmp_path: Path, finish: str) -> None:
    w = writer(tmp_path, queue_bytes=400)
    os.kill(w.pid, signal.SIGSTOP)
    errors: list[BaseException] = []
    completed: list[str] = []

    def run(name: str) -> None:
        try:
            w.store_control(name, b"x" * 100)
            completed.append(name)
        except Exception as error:  # noqa: BLE001 -- report bounded caller failure
            errors.append(error)

    first = threading.Thread(target=run, args=("x",))
    second = threading.Thread(target=run, args=("y",))
    first.start()
    limit = time.monotonic() + 5
    while not w.queued_bytes and time.monotonic() < limit:
        time.sleep(0.005)
    first_charge, queued = w.owned_bytes, w.queued_bytes
    assert queued > 0
    second.start()
    while w._pending_count < 2 and time.monotonic() < limit:
        time.sleep(0.005)
    try:
        assert w._pending_count == 2 and second.is_alive()
        assert w.queued_bytes == queued and w.owned_bytes == first_charge
        if finish == "wake":
            os.kill(w.pid, signal.SIGCONT)
        else:
            w.close()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        if finish == "wake":
            assert sorted(completed) == ["x", "y"] and not errors
        else:
            assert not completed and len(errors) == 2
    finally:
        if w.returncode is None:
            os.kill(w.pid, signal.SIGCONT)
        first.join(5)
        second.join(5)
        w.close()
    assert w.returncode is not None
    assert w.owned_bytes == w.queued_bytes == 0


def test_inspection_closes_its_actual_owner_inside_callers_except(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gf_training.group_pack_source import PackedOriginals, inspect_packed

    locator = sealed(tmp_path)
    original_close = PackedOriginals.close
    owners: list[Any] = []
    original_init = PackedOriginals.__init__

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        owners.append(self)

    monkeypatch.setattr(PackedOriginals, "__init__", initialize)
    try:
        try:
            raise ValueError("outer handled error")
        except ValueError:
            inspect_packed(locator.path, deadline=time.monotonic() + 10, max_owned_bytes=512 * MIB)
        assert owners and owners[0].database_file.fd == -1
    finally:
        for owner in owners:
            original_close(owner)


@pytest.mark.parametrize("headroom,refuse", [(20_000, True), (40_000, False)])
def test_parent_publication_uses_remaining_joint_writer_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, headroom: int, refuse: bool
) -> None:
    import gf_training.group_pack_source as source_module
    import gf_training.group_pack_writer as writer_module
    from gf_training.group_pack_db import index_admission

    configured = settings(tmp_path, budget_bytes=256 * MIB, queue_bytes=64 * MIB)
    w = GroupPackWriter(
        tmp_path / "collection", "run", settings=configured, deadline=time.monotonic() + 30
    )
    pack = encode_pack([PackMember("diagnostic", "x", b"x" * (16 * MIB - headroom))])
    w.submit("g", pack, raw_bytes=len(unpack_raw(pack)), record={}, episodes=[])
    actual_inspect = writer_module.inspect_packed
    actual_validate = source_module.admit_database
    allowances: list[tuple[int, int]] = []
    requested: list[int] = []
    accepted: list[int] = []

    def inspect(*args: Any, **kwargs: Any) -> Any:
        allowances.append((w.owned_bytes, kwargs["max_owned_bytes"]))
        return actual_inspect(*args, **kwargs)

    def validate(db: sqlite3.Connection, **kwargs: Any) -> Any:
        _, metadata_charge = index_admission(db, kwargs["max_owned_bytes"])
        packed_size, raw_size = db.execute("select length(pack),raw_bytes from groups").fetchone()
        still_owned, inspection_allowance = allowances[-1]
        source_already_owned = inspection_allowance - kwargs["max_owned_bytes"]
        required = (
            still_owned + source_already_owned + metadata_charge + 4 * raw_size + 2 * packed_size
        )
        requested.append(required)
        result = actual_validate(db, **kwargs)
        accepted.append(required)
        return result

    monkeypatch.setattr(writer_module, "inspect_packed", inspect)
    monkeypatch.setattr(source_module, "admit_database", validate)
    try:
        if refuse:
            with pytest.raises(ValueError, match="group exceeds admission"):
                w.seal()
            descriptor = w.directory / "collection.published.json"
            assert descriptor.read_bytes() == b"GFB-QUARANTINED-DESCRIPTOR\n"
            with pytest.raises(ValueError):
                open_artifact_source(
                    CollectionLocator(FORMAT, descriptor),
                    deadline=time.monotonic() + 10,
                    max_owned_bytes=256 * MIB,
                )
            assert requested and requested[0] > configured.budget_bytes and not accepted
        else:
            locator = w.seal()
            assert locator.path.exists() and accepted
            assert max(accepted) <= configured.budget_bytes
        assert allowances == [(128 * MIB, 128 * MIB)]
        assert w.returncode == 0
        original = (w.directory / "collection.sqlite3").read_bytes()
        with sqlite3.connect(f"file:{w.directory / 'collection.sqlite3'}?mode=ro", uri=True) as db:
            assert db.execute("pragma integrity_check").fetchone() == ("ok",)
            assert db.execute("select pack from groups").fetchone() == (pack,)
    finally:
        w.close()
    assert (w.directory / "collection.sqlite3").read_bytes() == original
    assert w.owned_bytes == w.queued_bytes == 0
