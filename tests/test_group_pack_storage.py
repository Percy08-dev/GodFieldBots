from __future__ import annotations

import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gf_common.group_pack import PackMember, StorageSettings, encode_pack
from gf_training.group_pack_db import reconcile_collection
from gf_training.group_pack_source import open_artifact_source
from gf_training.group_pack_writer import GroupPackWriter


def settings(root: Path, **kwargs: int) -> StorageSettings:
    return StorageSettings(db_root=root / "catalog", ram_root=root / "ram", **kwargs)


def writer(root: Path, **kwargs: int) -> GroupPackWriter:
    return GroupPackWriter(
        root / "collection",
        "run-1",
        settings=settings(root, **kwargs),
        deadline=time.monotonic() + 30,
    )


def submit(w: GroupPackWriter, group: str, size: int = 2) -> None:
    members = tuple(PackMember("journal", f"{group}/e{i}", bytes([255, i])) for i in range(size))
    pack = encode_pack(members)
    from gf_common.group_pack import unpack_raw

    w.submit(
        group,
        pack,
        raw_bytes=len(unpack_raw(pack)),
        record={"size": size},
        episodes=[
            {
                "episode_id": f"{group}-e{i}",
                "game_id": None if i == 0 else str(i),
                "record": {"status": "obtained"},
            }
            for i in range(size)
        ],
    )


def test_actual_child_atomic_idempotency_and_seal(tmp_path: Path) -> None:
    w = writer(tmp_path)
    assert w.pid != os.getpid()
    for size in (2, 6, 3):
        submit(w, str(size), size)
        submit(w, str(size), size)
    w.store_control("manifest.json", b"\x00\xff")
    w.store_control("manifest.json", b"\x00\xff")
    locator = w.seal()
    assert w.returncode == 0
    with sqlite3.connect(
        f"file:{locator.path.parent / 'collection.sqlite3'}?mode=ro", uri=True
    ) as db:
        assert db.execute("pragma journal_mode").fetchone() == ("delete",)
        assert db.execute("select count(*) from episodes").fetchone() == (11,)
        assert db.execute("select count(*) from episodes where game_id is null").fetchone() == (3,)
    assert sorted(p.name for p in locator.path.parent.iterdir()) == [
        "collection.published.json",
        "collection.sqlite3",
    ]
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 30, max_owned_bytes=512 * 1024 * 1024
    )
    assert source.read("manifest.json") == b"\x00\xff"
    with source.borrow_group("2") as loan:
        assert loan.names() == ("2/e0", "2/e1")
        assert loan.read("2/e0") == b"\xff\x00"
        with pytest.raises(RuntimeError):
            source.close()
        with pytest.raises(RuntimeError):
            source.evict_group()
        with pytest.raises(RuntimeError), source.borrow_group("6"):
            pass
    source.evict_group()
    with source.borrow_group("2") as current:
        assert current.read("2/e1") == b"\xff\x01"
        with pytest.raises(RuntimeError):
            loan.read("2/e0")
    source.recheck()
    source.close()
    assert (
        reconcile_collection(
            locator.path.parent, deadline=time.monotonic() + 30, max_owned_bytes=512 * 1024 * 1024
        )["state"]
        == "published"
    )


@pytest.mark.parametrize("conflict", ["group", "episode", "control", "member"])
def test_conflict_rolls_back_and_poison_reaps(tmp_path: Path, conflict: str) -> None:
    w = writer(tmp_path)
    submit(w, "first")
    w.store_control("control", b"original")
    with pytest.raises((ValueError, RuntimeError)):
        if conflict == "control":
            w.store_control("control", b"changed")
        else:
            group = "first" if conflict == "group" else "second"
            name = "first/e0" if conflict == "member" else "new-member"
            episode_id = "first-e0" if conflict == "episode" else "new-episode"
            pack = encode_pack((PackMember("result", name, b"new"),))
            from gf_common.group_pack import unpack_raw

            w.submit(
                group,
                pack,
                raw_bytes=len(unpack_raw(pack)),
                record={},
                episodes=[{"episode_id": episode_id, "game_id": None, "record": {}}],
            )
    assert w.returncode is not None
    with pytest.raises(RuntimeError):
        w.seal()
    with sqlite3.connect(tmp_path / "collection/collection.sqlite3") as db:
        assert db.execute("select group_id from groups").fetchall() == [("first",)]
        assert db.execute("select count(*) from episodes").fetchone() == (2,)
        assert db.execute("select data from controls").fetchone() == (b"original",)


def test_concurrent_submission_and_capacity_refusal(tmp_path: Path) -> None:
    w = writer(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: submit(w, str(i)), range(8)))
    w.close()
    assert w.returncode == 0
    tiny = GroupPackWriter(
        tmp_path / "tiny",
        "tiny",
        settings=settings(tmp_path, queue_bytes=1),
        deadline=time.monotonic() + 30,
    )
    with pytest.raises(ValueError):
        submit(tiny, "g")
    tiny.close()


def test_child_kill_timeout_and_pending_no_authority(tmp_path: Path) -> None:
    w = writer(tmp_path)
    submit(w, "g")
    os.kill(w.pid, 9)
    with pytest.raises(RuntimeError):
        w.store_control("x", b"")
    assert w.returncode is not None
    w.close()
    other = GroupPackWriter(
        tmp_path / "pending", "pending", settings=settings(tmp_path), deadline=time.monotonic() + 30
    )
    submit(other, "g")
    locator = other.seal(publication="pending")
    with pytest.raises(ValueError):
        open_artifact_source(
            locator, deadline=time.monotonic() + 30, max_owned_bytes=512 * 1024 * 1024
        )
    with pytest.raises(TimeoutError):
        GroupPackWriter(
            tmp_path / "late", "late", settings=settings(tmp_path), deadline=time.monotonic() - 1
        )


def test_successful_seal_survives_idempotent_writer_close(tmp_path: Path) -> None:
    w = writer(tmp_path)
    submit(w, "g")
    locator = w.seal()
    original = locator.path.read_bytes()
    w.close()
    w.close()
    assert locator.path.read_bytes() == original
    source = open_artifact_source(
        locator, deadline=time.monotonic() + 10, max_owned_bytes=512 * 1024 * 1024
    )
    with source.borrow_group("g") as loan:
        assert loan.read("g/e0") == b"\xff\x00"
    source.close()


@pytest.mark.parametrize("operation", ["submit", "audit"])
def test_metadata_verification_checks_one_decoded_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from contextlib import closing

    from gf_common import group_pack
    from gf_training import group_pack_db

    group_members = {
        "one": (
            PackMember("journal", "a/journal", b"\xff\x00\x89"),
            PackMember("replay", "c/replay", b""),
        ),
        "two": (
            PackMember("candidates", "b/candidates", b"\x00"),
            PackMember("result", "d/result", b"\xfe"),
        ),
    }
    groups = {}
    for group_id, members in group_members.items():
        pack = encode_pack(members)
        groups[group_id] = (pack, len(group_pack.unpack_raw(pack)))
    deadline = time.monotonic() + 5
    database = tmp_path / "collection.sqlite3"
    with (
        database.open("x+b") as original,
        closing(group_pack_db.new_database(database, "run", deadline, original.fileno())) as db,
    ):
        def submit() -> None:
            for group_id, (pack, raw_bytes) in groups.items():
                group_pack_db.submit_group(
                    db, "run", group_id, pack, raw_bytes=raw_bytes,
                    record={}, episodes=[], max_pack_bytes=group_pack.MIB,
                )

        if operation == "audit":
            submit()
            group_pack_db.seal_transaction(db, "run", max_owned_bytes=256 * group_pack.MIB)
            group_pack_db.finish_wal(db)

        original_decoder = group_pack.zstd.ZstdDecompressor
        original_digest = group_pack.digest
        decoder_count = 0
        hashes = []

        def counted_decoder(*args: object, **kwargs: object) -> group_pack.zstd.ZstdDecompressor:
            nonlocal decoder_count
            decoder_count += 1
            return original_decoder(*args, **kwargs)

        def counted_digest(data: bytes | memoryview) -> str:
            hashes.append((type(data), bytes(data)))
            return original_digest(data)

        monkeypatch.setattr(group_pack.zstd, "ZstdDecompressor", counted_decoder)
        monkeypatch.setattr(group_pack, "digest", counted_digest)
        if operation == "submit":
            submit()
        else:
            group_pack_db.validate_database(db, deadline=deadline, max_pack_bytes=group_pack.MIB,
                                           max_owned_bytes=256 * group_pack.MIB)

        assert decoder_count == len(groups)
        assert hashes == [
            (memoryview, member.data)
            for members in group_members.values()
            for member in members
        ]
        assert db.execute("SELECT group_id,pack,raw_bytes FROM groups ORDER BY group_id").fetchall() == [
            (group_id, *contents) for group_id, contents in groups.items()
        ]


def test_submit_rejects_wrong_exact_size_and_trailing_zstd_frame(tmp_path: Path) -> None:
    from contextlib import closing

    from gf_common.group_pack import MIB, unpack_raw
    from gf_training.group_pack_db import new_database, submit_group

    pack = encode_pack((PackMember("candidates", "group/candidates", b"\x00\xff"),))
    raw_bytes = len(unpack_raw(pack))
    database = tmp_path / "collection.sqlite3"
    with (
        database.open("x+b") as original,
        closing(new_database(database, "run", time.monotonic() + 5, original.fileno())) as db,
    ):
        for blob, size, reason in (
            (pack, raw_bytes + 1, "Pack size mismatch"),
            (pack + pack, raw_bytes, "trailing Pack frame"),
        ):
            with pytest.raises(ValueError, match=reason):
                submit_group(db, "run", "group", blob, raw_bytes=size, record={},
                             episodes=[], max_pack_bytes=MIB)
        assert db.execute("SELECT count(*) FROM groups").fetchone() == (0,)


@pytest.mark.parametrize("conflict", ["control", "member"])
def test_in_process_result_controls_commit_as_one_command(tmp_path: Path, conflict: str) -> None:
    from dataclasses import asdict

    from gf_common.group_pack import MIB, digest, unpack_raw
    from gf_training.group_pack_worker import GroupPackDatabase

    owner = GroupPackDatabase(max_owned_bytes=256 * MIB)

    def execute(command: str, payload: bytes | None = None, **fields: object) -> object:
        return owner.execute(dict(version=1, request_id=owner.request_id, command=command, **fields), payload)

    db_settings = settings(tmp_path)
    directory = tmp_path / "collection"
    directory.mkdir()
    execute("init", directory=str(directory), run_id="run",
            settings=asdict(db_settings) | {"db_root": str(db_settings.db_root),
                                         "ram_root": str(db_settings.ram_root)},
            deadline=time.monotonic() + 10)
    try:
        occupied = "second/result.json"
        if conflict == "control":
            execute("control", b"original", name=occupied)
        else:
            blob = encode_pack((PackMember("result", occupied, b"original"),))
            execute("submit", blob, group_id="group", raw_bytes=len(unpack_raw(blob)),
                    record={}, episodes=[])
        assert owner.db is not None
        original = owner.db.execute("SELECT * FROM controls ORDER BY name").fetchall()
        request = owner.request_id
        with pytest.raises(ValueError, match="conflicting control resend|duplicate logical name"):
            owner.execute_controls(request, (("first/result.json", b"first"), (occupied, b"new")))
        assert owner.request_id == request
        assert owner.db.execute("SELECT * FROM controls ORDER BY name").fetchall() == original
        values = (("first/result.json", b"first"), ("third/result.json", b"third"))
        for _ in range(2):
            owner.execute_controls(owner.request_id, values)
        expected = [(name, data, digest(data), len(data)) for name, data in values]
        assert owner.db.execute("SELECT * FROM controls ORDER BY name").fetchall() == sorted(original + expected)
        assert owner.request_id == request + 2
        with pytest.raises(ValueError, match="invalid Writer request"):
            execute("controls", b"untrusted wire payload")
    finally:
        execute("close")
