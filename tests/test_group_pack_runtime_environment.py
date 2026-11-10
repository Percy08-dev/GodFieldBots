"""Scripted public boundaries with real corrected workers and real Pack Writer."""
from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import load_config
from gf_common.group_pack import decode_pack
from tests.card_instance_action_support import inputs
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_contract import fixture
from tests.test_group_pack_runtime_config import BASE, packed_config


class Evidence:
    def __init__(self, binary, source_root, **kwargs):
        self.binary, self.source_root = Path(binary).absolute(), Path(source_root).absolute()
    def preflight(self):
        return receipt()
    def receipt(self, **kwargs):
        return receipt()
    def verify_executed(self, process):
        assert process.returncode is None
    def finalize(self):
        return receipt()


class PublicClient:
    poisoned = False
    def __init__(self, mode):
        self.mode, self.joined, self.terminal = mode, 0, False
        self.game = "game"
        self.value = inputs([], can_pray=True)
        self.attempts = []
        self.entered = asyncio.Event()
        self.hold = asyncio.Event()
        self.closed = False
    async def attest(self):
        pass
    async def public_catalog(self):
        return {row["definition_id"]: row for row in fixture("catalog")["definitions"]}
    async def create(self):
        self.joined, self.terminal, self.attempts = 0, False, []
        return {"game_id": self.game}
    async def join(self, *args):
        self.joined += 1
        return {"player_id": self.joined}
    async def observe(self, game, viewer):
        obs = copy.deepcopy(self.value.observation)
        obs["viewer_player_id"] = viewer
        obs["own"]["player_id"], obs["opponent"]["player_id"] = viewer, 3 - viewer
        if self.terminal:
            obs.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None,
                current_player=None, phase="terminal", result="draw")
        return obs
    async def events(self, *args):
        return [], None
    async def actor_options(self, *args):
        return {key: value for key, value in self.value.action_options.items() if key != "viewer_player_id"}
    async def act(self, game, actor, payload):
        self.entered.set()
        if self.mode == "cancel":
            await self.hold.wait()
        self.attempts.append((actor, payload["action"]))
        self.terminal = True
        return {"command_id": payload["command_id"], "previous_revision": 0,
            "state_revision": 1, "status": "terminal", "result": "draw"}
    async def replay(self, game, viewer):
        decisions = [{"actor": actor, "observation": self.value.observation,
            "action_options": self.value.action_options["action_options"], "visible_events": [],
            "aborted": False, "visible_action": {"type": "action", "action": action}}
            for actor, action in self.attempts if actor == viewer]
        return {"game_id": game, "viewer_player_id": viewer, "requested_turn": None,
            "rule_set_version": "true-match-1v1-v2", "observation_schema_version": "2",
            "public_event_schema_version": "2", "card_ref_scheme": "aes256-card-ref-v1",
            "episode_end": {"type": "terminal", "result": "draw"} if self.terminal else None,
            "turns": [{"turn": 1, "active_player": 1, "decisions": decisions}] if decisions else []}
    async def delete(self, *args):
        pass
    async def close(self):
        self.closed = True


@pytest.mark.parametrize("command", ["collect", "evaluate"])
@pytest.mark.parametrize("mode", ["natural", "cancel", "startup", "drain"])
def test_managed_memory_runtime_evidence_and_incomplete_boundaries(tmp_path, monkeypatch, command, mode):
    import gf_cli.card_instance as cli
    import gf_simulator_client.card_instance_environment as environment

    path = packed_config(tmp_path, 'db_root="db"\nram_root="ram"\n')
    loaded = load_config(path, {"run": {"output_dir": str(tmp_path / "runs"), "concurrency": 1},
        "server": {"config": str(BASE.parent.parent / "test-server.toml")},
        "actions": {"max_sequence_length": 4, "max_candidates": 50}})
    clients, servers = [], []
    class Server:
        def __init__(self, *args, **kwargs):
            self.root = args[4]
            self.process = SimpleNamespace(returncode=None)
            self.drainers = []
            self.expected = {}
            servers.append(self)
        async def start(self):
            self.root.mkdir(parents=True)
            (self.root / "stdout.log").write_bytes(b"initial log")
            if mode == "startup":
                raise OSError("injected startup failure")
            return {"base_url": "http://private-script", "test_server_info": {}}
        async def stop(self, *args):
            self.process.returncode = 0
            (self.root / "late-unknown.tmp").write_bytes(b"written during drain")
            return {"exit_code": 0, "shutdown": "stdin_eof", "log_error": "injected drain failure" if mode == "drain" else None}
    def client(*args, **kwargs):
        value = PublicClient(mode)
        clients.append(value)
        return value
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", Evidence)
    monkeypatch.setattr(environment, "CardInstanceRuntimeEvidence", Evidence)
    monkeypatch.setattr(environment, "AttestedClient", client)
    monkeypatch.setattr(environment.CardInstanceSimulatorEnvironment, "_new_server", lambda self, *a, **kw: Server(*a, **kw))
    async def run():
        task = asyncio.create_task(cli.run_baseline(loaded, command))
        if mode == "cancel":
            while not clients:
                await asyncio.sleep(0.01)
            await clients[0].entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return next((tmp_path / "runs").iterdir()), None
        from gf_training.collection import PackedRunResult
        completed = await task
        assert type(completed) is PackedRunResult
        assert completed.storage_metrics["checkpoint_seconds"] >= 0
        assert completed.storage_metrics["delete_mode_seconds"] >= 0
        assert completed.storage_metrics["publication_scope"] == ("published" if mode == "natural" else "diagnostic-pending")
        return completed.directory, completed.report
    envelope, summary = asyncio.run(run())
    db_path = next((tmp_path / "db").glob("*/collection.sqlite3"))
    with sqlite3.connect(db_path) as db:
        groups = db.execute("select pack from groups").fetchall()
        members = {m.name: m.data for row in groups for m in decode_pack(row[0])}
        controls = dict(db.execute("select name,data from controls"))
        assert groups
        assert any(name.endswith("late-unknown.tmp") for name in members)
        if mode == "natural":
            assert summary["comparison_valid"]
            assert sorted(path.name for path in envelope.iterdir()) == ["collection.locator.json"]
            manifest = json.loads(controls["manifest.json"])
            assert manifest["status"] == "complete"
            assert len(manifest["episodes"]) == 2
            assert all(row["end_kind"] == "natural" for row in manifest["episodes"])
            assert any(name.endswith(".columns") for name in members)
            from gf_common.group_pack import FORMAT
            from gf_training.group_pack_source import CollectionLocator, open_artifact_source
            from tests.test_group_pack_runtime_inventory import audit
            source = open_artifact_source(CollectionLocator(FORMAT, db_path.parent / "collection.published.json"),
                deadline=time.monotonic() + 20, max_owned_bytes=512 * 1024**2)
            try:
                audit(source)
            finally:
                source.close()
        else:
            assert not (envelope / "collection.locator.json").exists()
            assert "failure.json" in controls or "incomplete.json" in controls
            if mode == "startup":
                assert db.execute("select count(*) from episodes where game_id is null").fetchone() == (2,)
            if summary is not None:
                assert not summary["comparison_valid"]
    assert all(server.process.returncode == 0 for server in servers)
    assert all(client.closed for client in clients)
    assert not any((tmp_path / "ram").glob("*/*"))


def test_unconfirmed_simulator_reap_retains_same_group_owner_and_ram(tmp_path, monkeypatch):
    import gf_cli.card_instance as cli
    import gf_simulator_client.card_instance_environment as environment
    from gf_training.group_pack_resources import CleanupError

    loaded = load_config(packed_config(tmp_path, 'db_root="db"\nram_root="ram"\n'),
        {"run": {"output_dir": str(tmp_path / "runs"), "concurrency": 1},
         "actions": {"max_sequence_length": 4, "max_candidates": 50}})
    class Server:
        def __init__(self, *args, **kwargs):
            self.root = args[4]
            self.process, self.drainers, self.expected = SimpleNamespace(returncode=None), [], {}
        async def start(self):
            self.root.mkdir(parents=True)
            (self.root / "stdout.log").write_bytes(b"retained while unreaped")
            return {"base_url": "http://script", "test_server_info": {}}
        async def stop(self, *args):
            return {"exit_code": None, "shutdown": "kill", "reap_failure": "injected", "log_error": None}
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", Evidence)
    monkeypatch.setattr(environment, "CardInstanceRuntimeEvidence", Evidence)
    monkeypatch.setattr(environment, "AttestedClient", lambda *a, **kw: PublicClient("natural"))
    monkeypatch.setattr(environment.CardInstanceSimulatorEnvironment, "_new_server", lambda self, *a, **kw: Server(*a, **kw))
    with pytest.raises(CleanupError, match="reap and drain") as caught:
        asyncio.run(cli.run_baseline(loaded, "collect"))
    owner = caught.value.cleanup_owner
    group = owner._groups[0]
    assert not group.archived and group.runtime_owner is not None
    retained = group.runtime_owner._storage_group_owner._managed_server
    assert retained.process.returncode is None
    assert (retained.root / "stdout.log").read_bytes() == b"retained while unreaped"
    assert owner.writer.returncode == 0
    assert owner.budget.used > 0
    assert not list((tmp_path / "runs").glob("*/collection.locator.json"))


def test_startup_without_allocated_process_is_cleanup_complete():
    from gf_common.domain import GroupResult
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment

    env = object.__new__(CardInstanceSimulatorEnvironment)
    env._storage_group_owner = SimpleNamespace(_managed_server=SimpleNamespace(
        process=None, stopped=True, drainers=[]))
    result = GroupResult([], {"failure": "startup", "process": {"exit_code": None}})
    assert env.storage_cleanup_complete(result)
