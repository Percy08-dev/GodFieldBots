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
