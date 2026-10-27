import asyncio
import time

import pytest

from gf_common.domain import ActionConfig, EpisodeResult
from gf_models.baselines import make_policy
from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment


class FakeJournal:
    def __init__(self):
        self.finished = False
        self.incomplete = False
    def finish(self, *args):
        self.finished = True
    def close_incomplete(self):
        self.incomplete = True


class FakeWorker:
    def __init__(self):
        self.closed = False
    async def close(self, timeout=2):
        self.closed = True


def settings(tmp_path, allowance=1):
    return EnvironmentSettings("unused", "unused", tmp_path, "run", ActionConfig(), wall_deadline=time.monotonic() + allowance)


def test_delete_respects_remaining_wall_and_preserves_outcome(tmp_path):
    async def run():
        class Client:
            poisoned = False
            async def replay(self, game, viewer):
                return {"game_id": game, "viewer_player_id": viewer, "observation_schema_version": "1", "episode_end": None, "turns": []}
            async def delete(self, game):
                await asyncio.sleep(0.25)
        result = EpisodeResult("episode", "pair", "seed", ("random", "random"))
        worker = FakeWorker()
        env = SimulatorEnvironment(settings(tmp_path, 0.05), make_policy, lambda *a: None)
        started = time.monotonic()
        await env._finalize_episode(Client(), "game", tmp_path, result, {}, {1: worker}, {1: [], 2: []}, {})
        assert time.monotonic() - started < 0.15
        assert worker.closed
        assert result.health["delete_failure"]
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["observe", "replay", "delete"])
def test_cancellation_shield_retains_known_outcome_and_cleans_every_stage(tmp_path, make_input, stage):
    async def run():
        reached = asyncio.Event()
        release = asyncio.Event()
        outcome = {"win": {"winner": 1}}
        class Client:
            poisoned = False
            async def checkpoint(self, name):
                if stage == name:
                    reached.set()
                    await release.wait()
            async def observe(self, game, viewer):
                await self.checkpoint("observe")
                return {"result": outcome}
            async def replay(self, game, viewer):
                await self.checkpoint("replay")
                return {"game_id": game, "viewer_player_id": viewer, "observation_schema_version": "1", "episode_end": {"type": "terminal", "result": outcome}, "turns": []}
            async def delete(self, game):
                await self.checkpoint("delete")
        result = EpisodeResult("episode", "pair", "seed", ("random", "random"), outcome=outcome, end_kind="natural")
        journals = {1: FakeJournal(), 2: FakeJournal()}
        workers = {1: FakeWorker(), 2: FakeWorker()}
        env = SimulatorEnvironment(settings(tmp_path), make_policy, lambda *a: None)
        async def owner():
            await env._shield_cleanup(env._finalize_episode(Client(), "game", tmp_path, result, journals,
                                                            workers, {1: [], 2: []}, {}), result.health)
        task = asyncio.create_task(owner())
        await reached.wait()
        task.cancel()
        release.set()
        await task
        assert result.outcome == outcome and result.end_kind == "natural"
        assert all(worker.closed for worker in workers.values())
        assert all(journal.finished for journal in journals.values())
        assert result.health["finalization_cancelled"]
    asyncio.run(run())


def test_double_journal_flush_failure_does_not_skip_worker_or_delete(tmp_path):
    async def run():
        class BrokenJournal(FakeJournal):
            def finish(self, *args):
                raise OSError("disk failed during finish")
            def close_incomplete(self):
                raise OSError("disk failed again during close")
        class Client:
            poisoned = False
            deleted = False
            async def observe(self, game, viewer):
                return {"result": "draw"}
            async def replay(self, game, viewer):
                return {"game_id": game, "viewer_player_id": viewer, "observation_schema_version": "1", "episode_end": {"type": "terminal", "result": "draw"}, "turns": []}
            async def delete(self, game):
                self.deleted = True
        client = Client()
        worker = FakeWorker()
        good = FakeJournal()
        result = EpisodeResult("episode", "pair", "seed", ("random", "random"), outcome="draw", end_kind="natural")
        env = SimulatorEnvironment(settings(tmp_path), make_policy, lambda *a: None)
        await env._shield_cleanup(env._finalize_episode(client, "game", tmp_path, result, {1: BrokenJournal(), 2: good},
                                                       {1: worker}, {1: [], 2: []}, {}), result.health)
        assert result.outcome == "draw" and result.end_kind == "natural"
        assert worker.closed and client.deleted and good.finished
        assert result.health["save_failure"] and result.health["journal_close_failure"]
    asyncio.run(run())


def test_exhausted_work_budget_reserves_report_and_emergency_cleanup_is_unhealthy(tmp_path):
    async def run():
        class Worker(FakeWorker):
            def __init__(self):
                super().__init__()
                self.health = {}
            async def close(self, timeout=2):
                assert timeout == 0
                self.closed = True
                self.health = {"emergency_grace": "cleanup budget exhausted"}
        class Client:
            poisoned = False
            async def replay(self, *args):
                pytest.fail("no replay may start after replay/work budget")
            async def delete(self, *args):
                pytest.fail("no delete may start after work budget")
        result = EpisodeResult("episode", "pair", "seed", ("random", "random"), outcome="draw", end_kind="natural")
        now = time.monotonic()
        config = EnvironmentSettings("unused", "unused", tmp_path, "run", ActionConfig(), wall_deadline=now + 1,
                                     work_deadline=now - 0.01, replay_deadline=now - 0.02)
        env = SimulatorEnvironment(config, make_policy, lambda *a: None)
        worker = Worker()
        journal = FakeJournal()
        await env._finalize_episode(Client(), "game", tmp_path, result, {1: journal}, {1: worker}, {1: [], 2: []}, {})
        assert result.outcome == "draw" and worker.closed and journal.incomplete
        assert result.health["worker_cleanup_failure"]
        assert time.monotonic() < config.wall_deadline
    asyncio.run(run())
