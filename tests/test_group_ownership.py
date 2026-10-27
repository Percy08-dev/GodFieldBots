"""Event barriers exercise the actual group/episode owners under CLI scheduling."""
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import gf_cli
import gf_simulator_client.environment as module
from gf_cli.config import load_config
from gf_simulator_client.http import EFFECTIVE_KEYS, AttestedClient, seed_fingerprint
from gf_training.journal import read_journal
from tests.episode_support import AuditedWorkers, PublicScript

ROOT = Path(__file__).parents[1]


class GroupBoundaryHarness:
    def __init__(self, monkeypatch, hold_groups=(0,), cancel_act=False, pause_first_replay=False):
        self.servers = []
        self.active = 0
        self.maximum = 0
        self.started = [asyncio.Event() for _ in range(4)]
        self.stopping = [asyncio.Event() for _ in range(4)]
        self.release = [asyncio.Event() for _ in range(4)]
        self.replay_release = asyncio.Event()
        self.workers = AuditedWorkers(monkeypatch)
        self.environment = None
        owner = self
        original_environment = module.SimulatorEnvironment
        class Server:
            def __init__(self, binary, config, seed, count, directory, *args, **kwargs):
                self.index = len(owner.servers)
                self.directory = directory
                self.stopped = False
                self.client_closed = False
                self.script = PublicScript(mode="cancel_act" if cancel_act and self.index == 0 else "natural", actors=(2,))
                if pause_first_replay and self.index == 0:
                    self.script.replay_release = owner.replay_release
                self.script.info.update(seed_fingerprint=seed_fingerprint(seed), seed_use_limit=count)
                self.expected = {key: self.script.info[key] for key in EFFECTIVE_KEYS}
                owner.servers.append(self)
            async def start(self):
                owner.active += 1
                owner.maximum = max(owner.maximum, owner.active)
                owner.started[self.index].set()
                return {"base_url": f"http://group-{self.index}", "test_server_info": self.script.info}
            async def stop(self, remaining=None):
                if not self.stopped:
                    # Episode journal/replay ownership is released before the
                    # server's log-drain/reap phase starts.
                    episodes = list(self.directory.parent.glob(f"s{self.index}-*-seat*"))
                    assert episodes
                    for path in episodes:
                        if (path / "result.json").exists():
                            assert all((path / f"player-{p}.jsonl").exists() for p in (1, 2))
                            assert all((path / f"replay-player-{p}.json").exists() for p in (1, 2))
                            for p in (1, 2):
                                assert read_journal(path / f"player-{p}.jsonl")[-1]["kind"] == "end"
                    assert self.client_closed
                    owner.stopping[self.index].set()
                    if self.index in hold_groups:
                        await owner.release[self.index].wait()
                    self.stopped = True
                    owner.active -= 1
                return {"exit_code": 0, "shutdown": "stdin_eof", "log_error": None}
        def client(base_url, **kwargs):
            index = int(base_url.rsplit("-", 1)[1])
            server = owner.servers[index]
            real = AttestedClient(base_url, transport=httpx.MockTransport(server.script.handle), **kwargs)
            original_close = real.close
            async def close():
                await original_close()
                server.client_closed = True
            real.close = close
            return real
        def environment(settings, *args):
            owner.environment = original_environment(settings, *args)
            return owner.environment
        monkeypatch.setattr(module, "ManagedServer", Server)
        monkeypatch.setattr(module, "AttestedClient", client)
        monkeypatch.setattr(gf_cli, "SimulatorEnvironment", environment)

    def unblock_all(self):
        self.replay_release.set()
        for event in self.release:
            event.set()


def configured(tmp_path, groups=2, concurrency=1):
    binary = tmp_path / "unused-prebuilt"
    binary.write_bytes(b"fake managed process boundary")
    return load_config(ROOT / "configs/experiments/smoke.toml", {
        "run": {"output_dir": str(tmp_path / "runs"), "concurrency": concurrency,
                "seeds": [f"00000000-0000-0000-0000-{index + 1:012d}" for index in range(groups)]},
        "server": {"binary": str(binary), "source_dir": None}})


def test_group_slot_is_held_until_journals_replays_client_close_logs_and_reap(tmp_path, monkeypatch):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch)
        task = asyncio.create_task(gf_cli.run_experiment(configured(tmp_path), "evaluate"))
        try:
            await asyncio.wait_for(harness.stopping[0].wait(), 3)
            assert len(harness.servers) == harness.active == harness.maximum == 1
            assert not harness.servers[0].stopped
            assert not harness.started[1].is_set()
            harness.release[0].set()
            directory, summary = await asyncio.wait_for(task, 3)
            assert summary["natural_denominator"] == summary["planned_games"] == 4
            assert summary["comparison_valid"]
            assert harness.active == 0 and harness.maximum == 1
            assert all(server.stopped and server.client_closed for server in harness.servers)
            assert all(worker.closed for worker in harness.workers.instances)
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["status"] == "complete" and len(manifest["groups"]) == 2
            assert len(list(directory.glob("s*-seat*/player-*.jsonl"))) == 8
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_two_active_group_slots_bound_concurrency_and_queue_third_until_reap(tmp_path, monkeypatch):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch, hold_groups=(0, 1))
        task = asyncio.create_task(gf_cli.run_experiment(configured(tmp_path, groups=3, concurrency=2), "evaluate"))
        try:
            await asyncio.wait_for(asyncio.gather(harness.stopping[0].wait(), harness.stopping[1].wait()), 3)
            assert harness.active == harness.maximum == len(harness.servers) == 2
            assert not harness.started[2].is_set()
            harness.release[0].set()
            await asyncio.wait_for(harness.started[2].wait(), 3)
            assert harness.maximum == 2 and not harness.servers[1].stopped
            harness.release[1].set()
            _, summary = await asyncio.wait_for(task, 3)
            assert summary["planned_games"] == summary["natural_denominator"] == 6
            assert summary["comparison_valid"] and harness.active == 0
            game_ids = [game for server in harness.servers for game in server.script.created_games]
            assert len(game_ids) == len(set(game_ids)) == 6
            for worker in harness.workers.instances:
                for value in worker.inputs:
                    game = value["history_end"].split("-viewer-", 1)[0]
                    script = next(server.script for server in harness.servers if game in server.script.created_games)
                    viewer = value["observation"]["viewer_player_id"]
                    assert value["observation"]["own"]["hand"][0]["instance_id"] == script.hand_ids_by_game[game][viewer]
                    assert len(value["events"]) == 3
            assert all(worker.closed for worker in harness.workers.instances)
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["act", "reap"])
def test_cancel_active_and_queued_groups_prevents_queued_resource_creation(tmp_path, monkeypatch, stage):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch, cancel_act=stage == "act")
        loaded = configured(tmp_path)
        task = asyncio.create_task(gf_cli.run_experiment(loaded, "collect"))
        try:
            await asyncio.wait_for(harness.started[0].wait(), 3)
            if stage == "act":
                await asyncio.wait_for(harness.servers[0].script.act_entered.wait(), 3)
            else:
                await asyncio.wait_for(harness.stopping[0].wait(), 3)
            task.cancel("initial cancellation")
            # Keep the simulated log/reap boundary closed until the active
            # finalizer reaches it. Parent cancellation must still be pending.
            await asyncio.wait_for(harness.stopping[0].wait(), 3)
            parent_returned_before_reap = task.done()
            harness.release[0].set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert len(harness.servers) == 1 and harness.active == 0
            assert not harness.started[1].is_set()
            assert not parent_returned_before_reap, "CLI cancellation returned before active group cleanup/reap"
            assert all(worker.closed for worker in harness.workers.instances)
            manifest_path = next(Path(loaded.config.run.output_dir).glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "interrupted"
            if stage == "act":
                assert all(episode["outcome"] is None for episode in manifest["episodes"])
                journal = read_journal(manifest_path.parent / "s0-m0-r0-seat0/player-2.jsonl")
                assert journal[-1]["pending_decisions"] == [0] and journal[-1]["terminal_reward"] is None
            else:
                assert len(manifest["episodes"]) == 2
                assert all(episode["outcome"] == "draw" for episode in manifest["episodes"])
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_queued_group_rechecks_admission_deadline_before_creating_resources(tmp_path, monkeypatch):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch)
        task = asyncio.create_task(gf_cli.run_experiment(configured(tmp_path), "evaluate"))
        try:
            await asyncio.wait_for(harness.stopping[0].wait(), 3)
            # Deterministically exhaust admission while the second group is
            # queued; no timing sleep or altered event-loop clock is needed.
            harness.environment.settings = replace(harness.environment.settings, admission_deadline=0)
            harness.release[0].set()
            directory, summary = await asyncio.wait_for(task, 3)
            assert len(harness.servers) == 1 and harness.active == 0
            assert summary["natural_denominator"] == 2 and not summary["comparison_valid"]
            manifest = json.loads((directory / "manifest.json").read_text())
            queued = [e for e in manifest["episodes"] if e["episode_id"].startswith("s1-")]
            assert len(queued) == 2 and all(e["failure"] == "deadline" and e["outcome"] is None for e in queued)
            assert manifest["status"] == "incomplete"
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_external_group_owns_its_games_but_never_stops_external_server(tmp_path, monkeypatch, definitions):
    from gf_common.domain import EpisodeGroup, EpisodeSpec
    from tests.episode_support import SEED, environment
    async def run():
        workers = AuditedWorkers(monkeypatch)
        script = PublicScript(actors=(2,))
        clients = []
        def client(base_url, **kwargs):
            assert base_url == "http://external-owner"
            value = AttestedClient(base_url, transport=httpx.MockTransport(script.handle), **kwargs)
            clients.append(value)
            return value
        monkeypatch.setattr(module, "AttestedClient", client)
        monkeypatch.setattr(module, "ManagedServer", lambda *a, **k: pytest.fail("external owner must not spawn/stop a server"))
        env = environment(tmp_path, external_url="http://external-owner", expected_fingerprint=script.info["seed_fingerprint"])
        spec = EpisodeSpec("s0-m0-r0-seat0", ("random", "random"), (1, 2), "pair")
        result = await env.run_group(EpisodeGroup(SEED, (spec,)))
        assert result.episodes[0].end_kind == "natural" and result.episodes[0].outcome == "draw"
        assert "process" not in result.health and "startup" not in result.health
        assert script.deleted and clients[0].http.is_closed
        assert all(worker.closed for worker in workers.instances)
        assert read_journal(tmp_path / spec.episode_id / "player-2.jsonl")[-1]["terminal_reward"] == 0
    asyncio.run(run())


def test_same_seed_next_game_waits_for_replay_and_journal_finalization(tmp_path, monkeypatch):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch, pause_first_replay=True)
        task = asyncio.create_task(gf_cli.run_experiment(configured(tmp_path, groups=1), "collect"))
        try:
            await asyncio.wait_for(harness.started[0].wait(), 3)
            script = harness.servers[0].script
            await asyncio.wait_for(script.replay_entered.wait(), 3)
            assert script.requests.count(("POST", "/v1/games")) == 1
            run_dir = harness.servers[0].directory.parent
            assert len(list(run_dir.glob("s*-seat*"))) == 1
            assert len(list(run_dir.glob("s*-seat*/player-*.partial.jsonl"))) == 2
            assert not harness.stopping[0].is_set()
            harness.replay_release.set()
            await asyncio.wait_for(harness.stopping[0].wait(), 3)
            assert script.requests.count(("POST", "/v1/games")) == 2
            assert len(list(run_dir.glob("s*-seat*/player-*.jsonl"))) == 4
            harness.release[0].set()
            _, summary = await asyncio.wait_for(task, 3)
            assert summary["comparison_valid"] and harness.active == 0
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_repeated_cancellation_preserves_original_cancel_and_joins_active_finalizer(tmp_path, monkeypatch):
    async def run():
        harness = GroupBoundaryHarness(monkeypatch, cancel_act=True)
        task = asyncio.create_task(gf_cli.run_experiment(configured(tmp_path), "collect"))
        try:
            await asyncio.wait_for(harness.started[0].wait(), 3)
            await asyncio.wait_for(harness.servers[0].script.act_entered.wait(), 3)
            task.cancel("original cancellation")
            await asyncio.wait_for(harness.stopping[0].wait(), 3)
            assert not task.done() and not harness.servers[0].stopped
            task.cancel("repeated cancellation")
            harness.release[0].set()
            with pytest.raises(asyncio.CancelledError) as error:
                await asyncio.wait_for(task, 3)
            assert error.value.args == ("original cancellation",)
            assert harness.active == 0 and harness.servers[0].stopped and harness.servers[0].client_closed
            assert len(harness.servers) == 1 and all(w.closed for w in harness.workers.instances)
        finally:
            harness.unblock_all()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("failure_type", [TimeoutError, RuntimeError])
def test_peer_exception_including_timeout_joins_active_group_before_propagation(tmp_path, monkeypatch, failure_type):
    from gf_common.domain import EpisodeResult, GroupResult
    async def run():
        first_waiting = asyncio.Event()
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()
        resource_closed = False
        class Environment:
            def __init__(self, *args):
                pass
            async def run_group(self, group):
                nonlocal resource_closed
                if group.seed.endswith("0001"):
                    first_waiting.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cleanup_entered.set()
                        await cleanup_release.wait()
                        resource_closed = True
                        spec = group.episodes[0]
                        known = EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                              end_kind="natural", outcome="draw")
                        return GroupResult([known], {"finalization_cancelled": "peer exception"})
                await first_waiting.wait()
                raise failure_type("original group failure")
        monkeypatch.setattr(gf_cli, "SimulatorEnvironment", Environment)
        loaded = configured(tmp_path, concurrency=2)
        task = asyncio.create_task(gf_cli.run_experiment(loaded, "collect"))
        try:
            await asyncio.wait_for(cleanup_entered.wait(), 3)
            assert not task.done() and not resource_closed
            cleanup_release.set()
            with pytest.raises(failure_type, match="original group failure"):
                await asyncio.wait_for(task, 3)
            assert resource_closed
            manifest_path = next(Path(loaded.config.run.output_dir).glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "interrupted"
            assert manifest["episodes"][0]["outcome"] == "draw"
            assert manifest["groups"][0]["finalization_cancelled"]
        finally:
            cleanup_release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_expired_drain_threshold_is_recorded_once_and_does_not_orphan_or_spin(monkeypatch):
    from contextlib import asynccontextmanager

    from gf_common.domain import GroupResult
    async def run():
        entered = asyncio.Event()
        cleanup = asyncio.Event()
        release = asyncio.Event()
        threshold_seen = asyncio.Event()
        closed = False
        timeout_entries = 0
        async def owner():
            nonlocal closed
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await release.wait()
                closed = True
                return GroupResult([])
        @asynccontextmanager
        async def expired_threshold(deadline):
            nonlocal timeout_entries
            timeout_entries += 1
            assert timeout_entries == 1, "expired drain timeout was re-entered"
            threshold_seen.set()
            raise TimeoutError("controlled emergency threshold")
            yield  # pragma: no cover - makes an async context manager
        task = asyncio.create_task(owner())
        await entered.wait()
        diagnostics = {}
        monkeypatch.setattr(gf_cli.asyncio, "timeout_at", expired_threshold)
        drain = asyncio.create_task(gf_cli._join_owned_groups([task], 0, diagnostics))
        try:
            await cleanup.wait()
            await threshold_seen.wait()
            assert not drain.done() and not closed
            assert diagnostics["task_cleanup_failure"]
            drain.cancel("repeat cancellation during overrun join")
            release.set()
            groups = await drain
            assert closed and task.done() and timeout_entries == 1
            assert groups[0].health["cleanup_failure"]
            assert diagnostics["task_cleanup_emergency"]
        finally:
            release.set()
            await asyncio.gather(task, drain, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError])
def test_queued_group_is_never_admitted_after_peer_exception_at_semaphore_release(tmp_path, monkeypatch, failure_type):
    async def run():
        first_entered = asyncio.Event()
        fail_first = asyncio.Event()
        original = failure_type("original active group exception")
        first_resource_closed = False
        queued_resource_created = False
        queued_resource_closed = False
        class Environment:
            def __init__(self, *args):
                pass
            async def run_group(self, group):
                nonlocal first_resource_closed, queued_resource_created, queued_resource_closed
                if group.seed.endswith("0001"):
                    first_entered.set()
                    try:
                        await fail_first.wait()
                    finally:
                        first_resource_closed = True
                    raise original
                queued_resource_created = True
                try:
                    await asyncio.Event().wait()
                finally:
                    queued_resource_closed = True
        monkeypatch.setattr(gf_cli, "SimulatorEnvironment", Environment)
        loaded = configured(tmp_path, groups=2, concurrency=1)
        task = asyncio.create_task(gf_cli.run_experiment(loaded, "collect"))
        try:
            await asyncio.wait_for(first_entered.wait(), 3)
            assert not queued_resource_created
            fail_first.set()
            with pytest.raises(failure_type) as captured:
                await asyncio.wait_for(task, 3)
            assert captured.value is original
            assert first_resource_closed
            assert not queued_resource_created and not queued_resource_closed
            manifest_path = next(Path(loaded.config.run.output_dir).glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            assert manifest["status"] == "interrupted" and manifest["episodes"] == []
        finally:
            fail_first.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_child_timeout_without_peer_is_propagated_instead_of_budget_fallback(tmp_path, monkeypatch):
    async def run():
        entered = asyncio.Event()
        fail = asyncio.Event()
        original = TimeoutError("child timeout with no peers")
        class Environment:
            def __init__(self, *args):
                pass
            async def run_group(self, group):
                entered.set()
                await fail.wait()
                raise original
        monkeypatch.setattr(gf_cli, "SimulatorEnvironment", Environment)
        loaded = configured(tmp_path, groups=1, concurrency=1)
        task = asyncio.create_task(gf_cli.run_experiment(loaded, "collect"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            fail.set()
            with pytest.raises(TimeoutError) as captured:
                await asyncio.wait_for(task, 3)
            assert captured.value is original
            manifest_path = next(Path(loaded.config.run.output_dir).glob("*/manifest.json"))
            assert json.loads(manifest_path.read_text())["status"] == "interrupted"
            assert not (manifest_path.parent / "report.json").exists()
        finally:
            fail.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
