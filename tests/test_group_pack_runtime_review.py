"""Blocking Task2 review regressions: custody, retirement, admission and results."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from gf_common.domain import GroupResult
from gf_common.group_pack import MIB, StorageSettings
from gf_training.group_pack_collection import GroupPackCollection
from gf_training.group_pack_publication import GroupPackCollectionPublication


async def prepared(root, report=None):
    owner = GroupPackCollection(StorageSettings(db_root=root / "db", ram_root=root / "ram"),
        "run", deadline=time.monotonic() + 20)
    await owner.start()
    envelope = root / "envelope"
    envelope.mkdir()
    publication = GroupPackCollectionPublication(owner, envelope)
    owner.freeze_storage_metrics()
    await publication.prepare({"healthy": True} if report is None else report,
        {"schema_version": 5, "files": [dict(row) for row in owner.inventory]})
    return owner, publication


def prepared_manifest(publication):
    return json.loads(publication._controls["manifest.json"])


@pytest.mark.parametrize("member", ["collection.sqlite3", "collection.pending.json"])
@pytest.mark.parametrize("stage", ["inspection", "close"])
def test_composite_rejects_late_original_replacement(tmp_path, monkeypatch, member, stage):
    from gf_training import group_pack_publication as module
    owner, publication = asyncio.run(prepared(tmp_path))
    original = owner.directory / member
    replacement = tmp_path / "foreign"
    shutil.copyfile(original, replacement)
    if member.endswith("sqlite3"):
        with replacement.open("r+b") as out:
            out.seek(-1, 2)
            out.write(b"!")
    foreign_bytes = replacement.read_bytes()
    def swap():
        os.replace(replacement, original)
    if stage == "inspection":
        actual = module.inspect_packed
        def inspect(*args, **kwargs):
            result = actual(*args, **kwargs)
            swap()
            return result
        monkeypatch.setattr(module, "inspect_packed", inspect)
    else:
        actual = publication.pending.close
        swapped = False
        def close():
            nonlocal swapped
            if not swapped:
                swapped = True
                swap()
            return actual()
        monkeypatch.setattr(publication.pending, "close", close)
    with pytest.raises((ValueError, OSError)):
        publication.commit({"healthy": True}, prepared_manifest(publication))
    assert not publication.committed
    assert original.read_bytes() == foreign_bytes
    assert (publication.envelope / "collection.locator.json").read_bytes().startswith(b"!")


def test_predecessor_final_namespace_sweep_checks_same_inode(tmp_path, monkeypatch):
    from gf_training.group_pack_source import PackedOriginals
    owner, publication = asyncio.run(prepared(tmp_path))
    publication.commit({"healthy": True}, prepared_manifest(publication))
    originals = PackedOriginals(owner.directory / "collection.published.json",
        deadline=time.monotonic() + 20, max_owned_bytes=512 * MIB)
    actual = originals.check_namespace
    calls = 0
    def namespace():
        nonlocal calls
        actual()
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(originals.predecessor_file.path.read_bytes())
            os.replace(replacement, originals.predecessor_file.path)
    monkeypatch.setattr(originals, "check_namespace", namespace)
    try:
        with pytest.raises(ValueError):
            originals.recheck()
    finally:
        originals.close()


def test_minimum_budget_rejected_before_start_and_actual193_tiny_seal(tmp_path):
    from gf_cli.config import GroupPackStorageConfig
    for size in (129, 192):
        with pytest.raises(ValueError, match="193"):
            StorageSettings(budget_bytes=size * MIB, queue_bytes=MIB, max_pack_bytes=MIB)
        with pytest.raises(ValueError, match="193"):
            GroupPackStorageConfig(budget_mib=size, queue_mib=1, max_pack_mib=1)
    assert not list(tmp_path.iterdir())
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram",
            budget_bytes=193 * MIB, queue_bytes=MIB, max_pack_bytes=MIB), "run", deadline=time.monotonic() + 20)
        await owner.start()
        pending = await owner.prepare_diagnostic("incomplete.json", {})
        pending.close()
        await owner.close()
        assert owner.budget.used == 0
    asyncio.run(run())


def test_client_uses_only_common_storage_contract():
    import ast

    from gf_common import artifacts
    source = Path(__file__).parents[1] / "src/gf_simulator_client/card_instance_environment.py"
    tree = ast.parse(source.read_text())
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gf_training")
        for node in ast.walk(tree))
    assert hasattr(artifacts, "ArtifactAdmission")
    assert hasattr(artifacts, "ArtifactReservation")
    assert callable(artifacts.json_bound)


def test_timeout_never_submits_more_work_and_cleanup_waits_actual_future(tmp_path):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        release = threading.Event()
        owner.deadline = time.monotonic() + 0.05
        task = asyncio.create_task(owner.run_sync(release.wait, 5))
        with pytest.raises(TimeoutError):
            await task
        later = []
        try:
            for label in ("one", "two"):
                with pytest.raises((TimeoutError, RuntimeError)):
                    await owner.run_sync(later.append, label)
            assert len(owner._futures) == 1
            assert owner._admission_closed
            with pytest.raises(RuntimeError):
                await owner.close()
            assert len(owner._futures) == 1 and owner.budget.used > 0
        finally:
            release.set()
            for _ in range(100):
                if all(f.done() for f in owner._futures):
                    break
                await asyncio.sleep(0.01)
            await owner.close()
        assert later == []
        assert owner.budget.used == 0
    asyncio.run(run())


def test_cancelled_queued_archive_is_still_owned_and_committed(tmp_path):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        groups = [owner.new_group(name) for name in ("first", "second")]
        for i, group in enumerate(groups):
            group.sink.write(f"e{i}/original", b"obtained")
        entered, release = threading.Event(), threading.Event()
        def hold(snapshot):
            entered.set()
            release.wait(5)
        first = asyncio.create_task(owner.archive(groups[0], "first", GroupResult([], {}), validate=hold))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        second = asyncio.create_task(owner.archive(groups[1], "second", GroupResult([], {})))
        await asyncio.sleep(0.03)
        second.cancel()
        await asyncio.sleep(0.02)
        assert not second.done()
        release.set()
        await first
        with pytest.raises(asyncio.CancelledError) as caught:
            await second
        assert caught.value.cleanup_owner is owner
        assert all(group.archived for group in groups)
        pending = await owner.prepare_diagnostic("failure.json", {"cancelled": True})
        pending.close()
        await owner.close()
        with sqlite3.connect(owner.directory / "collection.sqlite3") as db:
            assert db.execute("select count(*) from groups").fetchone() == (2,)
    asyncio.run(run())


def test_raw_aliases_are_retired_before_snapshot_charge_release(tmp_path, monkeypatch):
    import sys

    import gf_training.group_pack_collection as module
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        group = owner.new_group("episode")
        group.sink.write("episode/original", os.urandom(2 * MIB))
        entered, release = threading.Event(), threading.Event()
        actual = module.shutil.rmtree
        def pause(path):
            entered.set()
            release.wait(5)
            actual(path)
        monkeypatch.setattr(module.shutil, "rmtree", pause)
        task = asyncio.create_task(owner.archive(group, "group", GroupResult([], {})))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        try:
            for frame in sys._current_frames().values():
                while frame:
                    if frame.f_code.co_name == "_archive":
                        aliases = [frame.f_locals.get(key) for key in ("pack", "index", "members", "data")]
                        live = any(aliases)
                        assert not group.snapshot._closed or not live
                    frame = frame.f_back
        finally:
            release.set()
            await task
            pending = await owner.prepare_diagnostic("incomplete.json", {})
            pending.close()
            await owner.close()
    asyncio.run(run())


def test_failed_validation_has_finite_retirement_without_clearing_user_frame(tmp_path):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        group = owner.new_group("episode")
        group.sink.write("episode/original", b"obtained")
        sentinel = object()
        def refuse(snapshot):
            user_local = sentinel
            snapshot.admit_validation(4 * MIB)
            assert user_local is sentinel
            raise ValueError("original failure facts")
        with pytest.raises(ValueError) as caught:
            await owner.archive(group, "group", GroupResult([], {}), validate=refuse)
        failure = caught.value
        pending = await owner.prepare_diagnostic("failure.json", {"error": str(failure)})
        pending.close()
        await owner.close()
        assert owner.budget.used == 0 and group.snapshot._closed
        frames = []
        tb = failure.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name == "refuse":
                frames.append(tb.tb_frame)
            tb = tb.tb_next
        assert frames and frames[0].f_locals["user_local"] is sentinel
        assert str(failure) == "original failure facts"
    asyncio.run(run())


def test_failed_writer_pack_workspace_stays_charged_until_completed_frames_retire(tmp_path):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        group = owner.new_group("episode")
        group.sink.write("episode/original", os.urandom(2 * MIB))
        owner.writer._process.kill()
        owner.writer._process.wait()
        with pytest.raises((OSError, RuntimeError)) as caught:
            await owner.archive(group, "group", GroupResult([], {}))
        failure = caught.value
        assert owner._retained_workspaces
        charged = sum(item.size for item in owner._retained_workspaces)
        assert charged >= 8 * MIB and owner.budget.used >= charged
        with pytest.raises(RuntimeError, match="unarchived"):
            await owner.close()
        assert owner._retained_workspaces == []
        assert owner.budget.used > 0  # obtained originals remain unarchived diagnostics.
        tb = failure.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_name in ("_encode_submit", "submit", "_rpc"):
                assert not tb.tb_frame.f_locals
            tb = tb.tb_next
    asyncio.run(run())


def test_optional_seal_metrics_are_actual_child_result_and_default_stays_exact(tmp_path):
    from gf_training.group_pack_worker import validate_request
    from gf_training.group_pack_writer import GroupPackWriter
    assert validate_request({"version": 1, "request_id": 1, "command": "seal-metrics",
        "publication": "pending"}, 1) == "seal-metrics"
    for metrics in (False, True):
        root = tmp_path / str(metrics)
        writer = GroupPackWriter(root / "collection", "run",
            settings=StorageSettings(db_root=root / "db", ram_root=root / "ram"),
            deadline=time.monotonic() + 20)
        try:
            writer.store_control("report.json", b"unchanged")
            locator = writer.seal(collect_metrics=metrics)
            assert locator.path.name == "collection.published.json"
            if metrics:
                assert set(writer.checkpoint_metrics) == {"checkpoint_seconds", "delete_mode_seconds"}
                assert all(type(value) is float and value >= 0 for value in writer.checkpoint_metrics.values())
            else:
                assert writer.checkpoint_metrics is None
            with sqlite3.connect(root / "collection/collection.sqlite3") as db:
                assert db.execute("select data from controls").fetchone() == (b"unchanged",)
        finally:
            writer.close()


def test_finish_wal_times_checkpoint_and_delete_transition_separately(monkeypatch):
    from gf_training import group_pack_db as module
    clock = iter([10.0, 11.5, 20.0, 20.25])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    statements = []
    class Connection:
        def execute(self, statement):
            statements.append(statement)
            return self
        def fetchone(self):
            return (0, 3, 3) if len(statements) == 1 else ("delete",)
    metrics = module.finish_wal(Connection())
    assert statements == ["PRAGMA wal_checkpoint(TRUNCATE)", "PRAGMA journal_mode=DELETE"]
    assert metrics == {"checkpoint_seconds": 1.5, "delete_mode_seconds": 0.25}


def test_packed_result_prepared_by_publication_preserves_sealed_report_and_metrics(tmp_path, monkeypatch):
    import hashlib
    import json

    from gf_training.collection import PackedRunResult
    owner, publication = asyncio.run(prepared(tmp_path))
    result = publication.commit({"healthy": True}, prepared_manifest(publication))
    assert type(result) is PackedRunResult
    assert result.directory == publication.envelope
    assert result.report == {"healthy": True}
    assert result.locator.path.name == "collection.published.json"
    assert result.envelope_sha256 == hashlib.sha256(
        (result.directory / "collection.locator.json").read_bytes()).hexdigest()
    assert result.storage_metrics["publication_scope"] == "published"
    assert result.storage_metrics["checkpoint_seconds"] >= 0
    assert result.storage_metrics["delete_mode_seconds"] >= 0
    assert result.storage_metrics["memory_measurement"] == "managed-reservations-and-observed-RAM-not-RSS"
    with sqlite3.connect(owner.directory / "collection.sqlite3") as db:
        raw = db.execute("select data from controls where name='report.json'").fetchone()[0]
        assert json.loads(raw) == result.report
        assert "storage_metrics" not in result.report
    # Result retrieval is a prepared value; it performs no reopen/parse.
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("postcommit reopen"))
    assert publication.result is result


def test_diagnostic_packed_result_has_pending_original_and_no_envelope(tmp_path):
    from gf_training.collection import PackedRunResult
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        publication = GroupPackCollectionPublication(owner, envelope)
        result = await publication.diagnostic({"healthy": False, "comparison_valid": False}, {"schema_version": 5})
        assert type(result) is PackedRunResult
        assert result.envelope_sha256 is None
        assert result.locator.path.name == "collection.pending.json"
        assert result.storage_metrics["publication_scope"] == "diagnostic-pending"
        assert not list(envelope.iterdir())
        assert owner.budget.used == 0
        with pytest.raises(ValueError):
            PackedRunResult(result.directory, {"healthy": True}, result.locator, None, result.storage_metrics)
    asyncio.run(run())


@pytest.mark.parametrize("published", [True, False])
def test_cli_explicitly_serializes_packed_result_scope_and_final_metrics(tmp_path, monkeypatch, capsys, published):
    import json

    import gf_cli
    from tests.test_group_pack_runtime_config import packed_config
    async def create():
        if published:
            _owner, publication = await prepared(tmp_path / "archive", {"healthy": True, "comparison_valid": True})
            return publication.commit({"healthy": True, "comparison_valid": True}, prepared_manifest(publication))
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        return await GroupPackCollectionPublication(owner, envelope).diagnostic(
            {"healthy": False, "comparison_valid": False}, {"schema_version": 5})
    result = asyncio.run(create())
    async def run(*args, **kwargs):
        return result
    monkeypatch.setattr(gf_cli, "run_experiment", run)
    assert gf_cli.main(["collect", "--config", str(packed_config(tmp_path))]) == (0 if published else 1)
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"directory", "report", "locator", "envelope_sha256", "storage_metrics"}
    assert output["report"] == result.report
    assert output["locator"]["path"] == str(result.locator.path)
    assert output["envelope_sha256"] == result.envelope_sha256
    assert output["storage_metrics"]["publication_scope"] == ("published" if published else "diagnostic-pending")
    assert output["storage_metrics"]["checkpoint_seconds"] >= 0


def test_real_replay_validation_traceback_graph_retires_after_archival(tmp_path):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace

    import gf_cli.card_instance as cli
    from gf_common.diverse_actions import NumericCosts
    from gf_common.domain import EpisodeSpec
    from gf_models.structured_runtime import WorkerCaps
    from gf_training.group_pack_memory import MemoryCardInstanceJournal
    from tests.test_card_instance_baseline_runtime import ACTIONS, receipt
    from tests.test_group_pack_runtime_environment import PublicClient

    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        group = owner.new_group("episode")
        client = PublicClient("natural")
        client.terminal = True
        policies = ("heuristic-card-instance-v2", "heuristic-card-instance-v2")
        for viewer in (1, 2):
            journal = MemoryCardInstanceJournal(tmp_path / "episode", "episode", viewer, policies[viewer - 1],
                run_id="run", game_id="game", actions=ACTIONS, bank_episode=f"bank-{viewer}",
                caps=asdict(WorkerCaps(max_positions=4, max_candidates=50)), numeric_costs=asdict(NumericCosts()),
                model_id="heuristic", runtime_evidence=receipt(), artifact_sink=group.sink)
            journal.finish("natural", "draw", await client.observe("game", viewer))
            replay = await client.replay("game", viewer)
            replay["invalid-extra-graph"] = ["x" * 1024 for _ in range(1024)]
            group.sink.write(f"episode/replay-player-{viewer}.json", json.dumps(replay).encode())
        episode = SimpleNamespace(episode_id="episode", game_id="game", policies=policies,
            end_kind="natural", outcome="draw")
        expected = {"episode": (EpisodeSpec("episode", policies, (0, 0), "pair"), receipt())}
        def validate(snapshot):
            cli._validate_recorded_episodes(tmp_path, [episode], ACTIONS, "run", owner.deadline,
                expected, artifacts=snapshot)
        with pytest.raises(ValueError) as caught:
            await owner.archive(group, "group", GroupResult([], {}), validate=validate)
        pending = await owner.prepare_diagnostic("failure.json", {"error": str(caught.value)})
        pending.close()
        await owner.close()
        assert owner.budget.used == 0
        saw_replay = False
        tb = caught.value.__traceback__
        while tb:
            if tb.tb_frame.f_globals.get("__name__") == "gf_common.card_instance_validation":
                saw_replay = True
                assert not tb.tb_frame.f_locals
            tb = tb.tb_next
        assert saw_replay
    asyncio.run(run())


@pytest.mark.parametrize("member", ["collection.pending.json", "collection.published.json", "collection.locator.json"])
def test_terminal_write_close_ctime_only_handoff_accepts_same_original(tmp_path, monkeypatch, member):
    import json
    import stat

    from gf_training.card_instance_collection import CollectionPublication
    from gf_training.group_pack_source import PackedOriginals
    from gf_training.group_pack_writer import PendingDescriptorOwner

    owner, publication = asyncio.run(prepared(tmp_path))
    pending_close = PendingDescriptorOwner.close
    release = CollectionPublication._release_output
    changes = []
    def touch(path):
        before = path.stat()
        path.chmod(stat.S_IMODE(before.st_mode))
        after = path.stat()
        assert before.st_ino == after.st_ino and before.st_mtime_ns == after.st_mtime_ns
        changes.append(path)
    def close(self):
        live = self.fd >= 0
        pending_close(self)
        if live and member == "collection.pending.json":
            touch(self.path)
    def output_release(self, output):
        release(self, output)
        if output.name == member:
            touch(self.directory / output.name)
    monkeypatch.setattr(PendingDescriptorOwner, "close", close)
    monkeypatch.setattr(CollectionPublication, "_release_output", output_release)
    result = publication.commit({"healthy": True}, prepared_manifest(publication))
    assert changes and publication.committed
    assert result.storage_metrics["terminal_control_policy"] == "terminal-control-ro-handoff-v1"
    with sqlite3.connect(owner.directory / "collection.sqlite3") as db:
        metrics = json.loads(db.execute("select data from controls where name='storage-metrics.json'").fetchone()[0])
        assert metrics["terminal_control_policy"] == "terminal-control-ro-handoff-v1"
    # Ordinary Source custody still rejects a later ctime-only mutation.
    originals = PackedOriginals(result.locator.path, deadline=time.monotonic() + 20,
        max_owned_bytes=512 * MIB)
    try:
        touch(owner.directory / "collection.pending.json")
        with pytest.raises(ValueError):
            originals.recheck()
    finally:
        originals.close()


@pytest.mark.parametrize("mutation", ["inode", "content", "mode", "mtime", "database-ctime"])
def test_terminal_control_handoff_rejects_changed_original(tmp_path, monkeypatch, mutation):
    import stat

    from gf_training.group_pack_writer import PendingDescriptorOwner
    _owner, publication = asyncio.run(prepared(tmp_path))
    actual = PendingDescriptorOwner.close
    changed = False
    def close(self):
        nonlocal changed
        actual(self)
        if changed:
            return
        changed = True
        path = self.database.path if mutation == "database-ctime" else self.path
        before = path.stat()
        if mutation == "inode":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        elif mutation == "content":
            raw = path.read_bytes()
            path.write_bytes(b"!" + raw[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif mutation == "mode":
            path.chmod(stat.S_IMODE(before.st_mode) ^ stat.S_IXUSR)
        elif mutation == "mtime":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        else:
            path.chmod(stat.S_IMODE(before.st_mode))
    monkeypatch.setattr(PendingDescriptorOwner, "close", close)
    with pytest.raises((ValueError, OSError)):
        publication.commit({"healthy": True}, prepared_manifest(publication))
    assert not publication.committed
    assert (publication.envelope / "collection.locator.json").read_bytes().startswith(b"!")


@pytest.mark.parametrize("failure", ["during-read", "readonly-close-ctime", "readonly-close-uncertain"])
def test_terminal_readonly_verification_and_close_remain_strict(tmp_path, monkeypatch, failure):
    import stat

    from gf_training import group_pack_publication as module
    owner, publication = asyncio.run(prepared(tmp_path))
    actual_pread, actual_close = os.pread, os.close
    close_method = module._TerminalOriginal.close
    changed = False
    attempts = []
    target_fd = None
    def during_read(fd, size, offset):
        nonlocal changed
        data = actual_pread(fd, size, offset)
        for original in publication._terminal_originals:
            if (not changed and original.fd == fd and original.path.name == "collection.pending.json"):
                changed = True
                info = original.path.stat()
                raw = original.path.read_bytes()
                original.path.write_bytes(b"!" + raw[1:])
                os.utime(original.path, ns=(info.st_atime_ns, info.st_mtime_ns))
        return data
    def close_ctime(self):
        live = self.fd >= 0
        close_method(self)
        if live and self.path.name == "collection.pending.json":
            self.path.chmod(stat.S_IMODE(self.identity.st_mode))
    def remember_close(self):
        nonlocal target_fd
        if self.fd >= 0 and self.path.name == "collection.pending.json":
            target_fd = self.fd
        close_method(self)
    def uncertain_close(fd):
        nonlocal changed
        inject = fd == target_fd and not changed
        if inject:
            attempts.append(fd)
        actual_close(fd)
        if inject:
            changed = True
            raise OSError("injected readonly post-release close failure")
    if failure == "during-read":
        monkeypatch.setattr(os, "pread", during_read)
    elif failure == "readonly-close-ctime":
        monkeypatch.setattr(module._TerminalOriginal, "close", close_ctime)
    else:
        monkeypatch.setattr(module._TerminalOriginal, "close", remember_close)
        monkeypatch.setattr(os, "close", uncertain_close)
    with pytest.raises((ValueError, OSError)) as caught:
        publication.commit({"healthy": True}, prepared_manifest(publication))
    assert not publication.committed
    assert (publication.envelope / "collection.locator.json").read_bytes().startswith(b"!")
    if failure == "readonly-close-uncertain":
        assert len(attempts) == 1
        assert caught.value.cleanup_owner is publication
        assert not publication.cleanup_complete
        assert owner.budget.used > 0
