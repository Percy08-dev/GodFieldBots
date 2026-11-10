"""Real Writer archival, composite publication and cancelled worker custody."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from gf_common.domain import GroupResult
from gf_common.group_pack import StorageSettings, digest


def test_group_pack_archives_all_owned_server_files_and_retires_group(tmp_path):
    from gf_training.card_instance_collection import CollectionPublication
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    from gf_training.group_pack_source import CollectionLocator, open_artifact_source

    async def run():
        settings = StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram")
        collection = GroupPackCollection(settings, "run-1", deadline=time.monotonic() + 30)
        await collection.start()
        group = collection.new_group("episode")
        group.sink.write("episode/player-1.partial.jsonl", b"original partial\n")
        server = group.server_root / "server-episode"
        server.mkdir()
        (server / "unknown.tmp").write_bytes(b"all simulator evidence")
        await collection.archive(group, "g1", GroupResult([], {"failure": "startup"}))
        assert not group.server_root.exists()
        assert 128 * 1024**2 < collection.budget.used < 129 * 1024**2
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        packed = GroupPackCollectionPublication(collection, envelope)
        manifest = {"schema_version": 5, "files": [collection.freeze_storage_metrics()]}
        await packed.prepare({"healthy": True}, manifest)
        pending = packed.pending
        assert pending is not None and pending.fd >= 0
        publication = CollectionPublication(envelope, work_deadline=collection.deadline,
            cleanup_deadline=collection.deadline, group_pack=packed)
        publication.commit({"healthy": True}, manifest)
        assert publication.committed
        assert pending.fd == -1
        pointer = json.loads((envelope / "collection.locator.json").read_bytes())
        path = Path(pointer["descriptor"])
        assert digest(path.read_bytes()) == pointer["descriptor_sha256"]
        source = open_artifact_source(CollectionLocator("sqlite-group-pack-v1", path),
            deadline=collection.deadline, max_owned_bytes=512 * 1024**2)
        try:
            with source.borrow_group("g1") as loan:
                assert loan.read("episode/player-1.partial.jsonl") == b"original partial\n"
                assert loan.read("server-episode/unknown.tmp") == b"all simulator evidence"
            source.recheck()
            (path.parent / "collection.pending.json").write_bytes(b"changed")
            with pytest.raises(ValueError):
                source.recheck()
        finally:
            source.close()
        assert collection.budget.used == 0
    asyncio.run(run())


def test_cancellation_waits_for_actual_worker_future(tmp_path):
    from gf_training.group_pack_collection import GroupPackCollection

    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await owner.start()
        started, release, completed = threading.Event(), threading.Event(), threading.Event()
        def operation():
            started.set()
            release.wait(5)
            completed.set()
        task = asyncio.create_task(owner.run_sync(operation))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert owner.budget.used > 0
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()
        await owner.close()
        assert owner.budget.used == 0
    asyncio.run(run())


@pytest.mark.parametrize("size", [2, 6, 3])
def test_group_sizes_preserve_explicit_unstarted_games(tmp_path, size):
    import sqlite3

    from gf_common.domain import EpisodeResult
    from gf_training.group_pack_collection import GroupPackCollection

    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await owner.start()
        group = owner.new_group("episode")
        group.sink.write("episode/diagnostic", b"obtained")
        episodes = [EpisodeResult(f"episode-{i}", "pair", "seed", ("random", "heuristic"),
            failure="not_admitted") for i in range(size)]
        await owner.archive(group, "group", GroupResult(episodes, {"failure": "startup"}))
        pending = await owner.prepare({"incomplete.json": b"unsuccessful"}, diagnostic=True)
        pending.close()
        with sqlite3.connect(owner.directory / "collection.sqlite3") as db:
            assert db.execute("select count(*) from episodes where game_id is null").fetchone() == (size,)
            assert db.execute("select count(*) from members").fetchone() == (1,)
        assert not (owner.directory / "collection.published.json").exists()
    asyncio.run(run())


def test_writer_failure_retains_snapshot_ram_and_cleanup_owner(tmp_path, monkeypatch):
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_resources import CleanupError

    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await owner.start()
        group = owner.new_group("episode")
        group.sink.write("episode/player-1.partial.jsonl", b"original")
        def fail(*args, **kwargs):
            raise OSError("injected Writer failure")
        monkeypatch.setattr(owner.writer, "submit", fail)
        with pytest.raises(OSError, match="injected"):
            await owner.archive(group, "group", GroupResult([], {}))
        assert group.snapshot.read("episode/player-1.partial.jsonl") == b"original"
        assert group.server_root.exists()
        assert owner.budget.used > 128 * 1024**2
        with pytest.raises(CleanupError) as caught:
            await owner.close()
        assert caught.value.cleanup_owner is owner
        assert owner.writer.returncode == 0
        assert owner.budget.used > 0
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["policy", "locator", "descriptor", "fd-close"])
def test_composite_failure_invalidates_locator_before_descriptor(tmp_path, monkeypatch, mutation):
    from gf_training.card_instance_collection import CollectionPublication
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_publication import GroupPackCollectionPublication

    async def run():
        collection = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await collection.start()
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        packed = GroupPackCollectionPublication(collection, envelope)
        manifest = {"schema_version": 5, "files": [collection.freeze_storage_metrics()]}
        await packed.prepare({"healthy": True}, manifest)
        outer = CollectionPublication(envelope, work_deadline=collection.deadline,
            cleanup_deadline=collection.deadline, group_pack=packed)
        invalidated = []
        original_invalidate = CollectionPublication._invalidate_output
        def invalidate(self, output):
            invalidated.append(output.name)
            return original_invalidate(self, output)
        monkeypatch.setattr(CollectionPublication, "_invalidate_output", invalidate)
        calls = 0
        foreign = b"foreign replacement"
        def guard():
            nonlocal calls
            calls += 1
            if calls != 3:
                return
            if mutation == "policy":
                raise ValueError("late retained policy mutation")
            if mutation in ("locator", "descriptor"):
                path = (envelope / "collection.locator.json" if mutation == "locator"
                        else collection.directory / "collection.published.json")
                displaced = path.with_name(path.name + ".displaced")
                path.rename(displaced)
                path.write_bytes(foreign)
                raise ValueError("foreign replacement")
        if mutation == "fd-close":
            original_release = CollectionPublication._release
            def release(self):
                original_release(self)
                if self.directory == envelope:
                    raise OSError("post-close failure")
            monkeypatch.setattr(CollectionPublication, "_release", release)
        with pytest.raises((ValueError, OSError)):
            outer.commit({"healthy": True}, manifest, retained_input_guard=guard)
        assert not outer.committed
        assert invalidated[:2] == ["collection.locator.json", "collection.published.json"]
        if mutation in ("locator", "descriptor"):
            path = (envelope / "collection.locator.json" if mutation == "locator"
                    else collection.directory / "collection.published.json")
            assert path.read_bytes() == foreign
        else:
            assert (envelope / "collection.locator.json").read_bytes().startswith(b"!")
            assert (collection.directory / "collection.published.json").read_bytes().startswith(b"!")
    asyncio.run(run())


def test_shared_borrower_wait_allows_another_submission_to_release():
    from concurrent.futures import ThreadPoolExecutor

    from gf_training.group_pack_resources import BorrowedBudget, StorageBudget

    budget = StorageBudget(StorageSettings(), time.monotonic() + 3)
    borrowed = BorrowedBudget(budget)
    borrowed.acquire(budget.limit, budget.deadline)
    entered = threading.Event()
    def wait():
        entered.set()
        borrowed.acquire(1024, budget.deadline, wait=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait)
        assert entered.wait(1)
        borrowed.release(budget.limit)
        future.result(timeout=1)
    assert budget.used == borrowed.used == 1024
    borrowed.release(1024)
    borrowed.stop()
    reservation = budget.reserve(2048)
    reservation.close()
    assert budget.used == 0


def test_ram_monitor_closes_admission_cancels_active_group_and_keeps_bytes(tmp_path):
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_resources import CleanupError

    async def run():
        collection = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await collection.start()
        group = collection.new_group("episode")
        (group.server_root / "growing.log").write_bytes(b"obtained log")
        held = collection.budget.reserve(collection.budget.limit - collection.budget.used)
        active = asyncio.create_task(asyncio.sleep(30))
        collection.track_group_task(active)
        with pytest.raises(asyncio.CancelledError):
            await active
        assert collection._admission_closed
        assert collection.budget.ram_bytes == len(b"obtained log")
        assert collection._failure is not None
        assert group.server_root.exists()
        held.close()
        with pytest.raises(CleanupError):
            await collection.close()
    asyncio.run(run())


def test_validation_failure_archives_bytes_and_retires_parse_graphs_after_cleanup(tmp_path):
    from gf_training.group_pack_collection import GroupPackCollection

    async def run():
        collection = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await collection.start()
        group = collection.new_group("episode")
        group.sink.write("episode/obtained", b"original")
        def refuse(snapshot):
            snapshot.admit_validation(1024**2)
            raise ValueError("invalid Replay closure")
        with pytest.raises(ValueError, match="invalid Replay") as caught:
            await collection.archive(group, "group", GroupResult([], {}), validate=refuse)
        assert caught.value.cleanup_owner is collection
        assert group.archived and not group.server_root.exists()
        assert group.snapshot.read("episode/obtained") == b"original"
        pending = await collection.prepare_diagnostic("failure.json", {"healthy": False})
        pending.close()
        await collection.close()
        assert collection.budget.used == 0
        assert group.snapshot._closed
        assert str(caught.value) == "invalid Replay closure"
    asyncio.run(run())


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "run/nested"])
def test_collection_run_id_cannot_escape_owned_storage_roots(tmp_path, run_id):
    from gf_training.group_pack_collection import GroupPackCollection
    with pytest.raises(ValueError):
        GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            run_id, deadline=time.monotonic() + 30)
    assert not (tmp_path / "db").exists()


def test_cancelled_seal_transfers_actual_pending_owner_before_async_delivery(tmp_path, monkeypatch):
    from gf_training.group_pack_collection import GroupPackCollection

    async def run():
        collection = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await collection.start()
        entered, release = threading.Event(), threading.Event()
        original = collection.writer.take_pending_owner
        issued = []
        def pending():
            value = original()
            issued.append(value)
            entered.set()
            release.wait(5)
            return value
        monkeypatch.setattr(collection.writer, "take_pending_owner", pending)
        task = asyncio.create_task(collection.prepare({"failure.json": b"diagnostic"}, diagnostic=True))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert collection.pending is issued[0]
        assert collection.pending.fd >= 0
        await collection.close()
        assert collection.pending.fd == collection.pending.database.fd == -1
        assert collection.budget.used == 0
    asyncio.run(run())


def test_prepare_failure_keeps_composite_owner_and_retires_writer(tmp_path, monkeypatch):
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_publication import GroupPackCollectionPublication

    async def run():
        collection = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 30)
        await collection.start()
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        composite = GroupPackCollectionPublication(collection, envelope)
        async def failure(*args, **kwargs):
            raise RuntimeError("control preparation failed")
        monkeypatch.setattr(collection, "prepare", failure)
        with pytest.raises(RuntimeError, match="control preparation") as caught:
            await composite.prepare({"healthy": True}, {"schema_version": 5})
        assert caught.value._group_pack_publication_cleanup_owner is composite
        assert collection.writer.returncode == 0
        assert collection._closed
        assert collection.budget.used == 0
    asyncio.run(run())
