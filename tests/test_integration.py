"""Opt in: GFB_RUN_INTEGRATION=1 uv run pytest -m integration.

These tests start the prebuilt pinned external TestServer; they do not build or
modify the simulator. A local-listener sandbox approval may be required.
"""
import asyncio
import os
import uuid
from pathlib import Path

import pytest

from gf_cli import run_experiment
from gf_cli.config import load_config
from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig
from gf_common.validation import decision_input
from gf_simulator_client.http import ApiError, AttestedClient, seed_fingerprint
from gf_simulator_client.process import ManagedServer
from gf_training.journal import read_journal

ROOT = Path(__file__).parents[1]
SEED = "00000000-0000-0000-0000-000000000001"
pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get("GFB_RUN_INTEGRATION") != "1", reason="explicit binary integration opt-in")]


def test_repeated_creates_independent_reservations_and_cleanup(tmp_path):
    async def run():
        loaded = load_config(ROOT / "configs/experiments/smoke.toml")
        config = loaded.config.server
        server = ManagedServer(config.binary, config.config, SEED, 2, tmp_path / "server")
        client = None
        try:
            startup = await server.start()
            client = AttestedClient(startup["base_url"], expected_fingerprint=seed_fingerprint(SEED), fixed_seed_experiment=True,
                                    expected_effective=server.expected, startup_info=startup["test_server_info"])
            await client.attest()
            definitions = await client.public_catalog()
            first = (await client.create())["game_id"]
            for _ in range(2):
                await client.join(first, str(uuid.uuid4()))
            initial = await client.observe(first, 1)
            actor = initial["current_player"]
            obs = await client.observe(first, actor)
            value = decision_input(obs, [], None, definitions, await client.actor_options(first, actor, obs))
            candidates = PublicCandidateGenerator(ActionConfig(max_sequence_length=1)).generate(value)
            action = next(a for a in candidates.actions if a["type"] == "play_sequence" and a["target"] != actor)
            await client.act(first, actor, {"command_id": str(uuid.uuid4()), "expected_revision": 0, "action": action})
            events, first_cursor = await client.events(first, actor, None)
            assert events and first_cursor
            second = (await client.create())["game_id"]
            assert first != second
            for _ in range(2):
                await client.join(second, str(uuid.uuid4()))
            assert await client.observe(second, 1) == initial
            assert (await client.observe(first, 1))["state_revision"] > 0
            with pytest.raises(ApiError) as exhausted:
                await client.create()
            assert exhausted.value.status == 503
            with pytest.raises(ApiError) as foreign_cursor:
                await client.events(second, actor, first_cursor)
            assert foreign_cursor.value.status == 400
        finally:
            if client:
                await client.close()
            await server.stop()
        assert server.process.returncode == 0
        assert server.health["shutdown"] == "stdin_eof"
    asyncio.run(run())


def test_baseline_natural_end_journals_replay_and_shutdown(tmp_path):
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {"run": {"output_dir": str(tmp_path)}})
    directory, summary = asyncio.run(run_experiment(loaded, "smoke"))
    assert summary["planned_games"] == summary["natural_denominator"] == 2
    assert summary["comparison_valid"]
    journals = sorted(directory.glob("s*-seat*/player-*.jsonl"))
    assert len(journals) == 4
    for path in journals:
        records = read_journal(path)
        assert records[-1]["end_kind"] == "natural"
        assert not records[-1]["pending_decisions"]
    assert len(list(directory.glob("s*-seat*/replay-player-*.json"))) == 4
    import json
    manifest = json.loads((directory / "manifest.json").read_text())
    assert all(group["process"]["exit_code"] == 0 and group["process"]["shutdown"] == "stdin_eof" for group in manifest["groups"])


def test_bounded_real_episode_decision_cutoff_keeps_accepted_action_without_terminal_label(tmp_path):
    """Exercise an actual nonnatural owner cutoff, independent of game outcomes."""
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {
        "run": {"output_dir": str(tmp_path), "max_decisions": 1},
        "opponent": {"model_id": "heuristic"}})
    directory, summary = asyncio.run(run_experiment(loaded, "collect"))
    assert summary["planned_games"] == 2 and summary["natural_denominator"] == 0
    assert not summary["comparison_valid"]
    assert summary["end_counts"] == {"truncated": 2}
    assert summary["failures"] == {"decision_limit": 2}
    for path in directory.glob("s*-seat*/player-*.jsonl"):
        records = read_journal(path)
        assert records[-1]["end_kind"] == "truncated"
        assert records[-1]["terminal_reward"] is None and records[-1]["final_observation"] is None
        assert not any(r["kind"] == "transition" and r["terminal"] for r in records)
    accepted = [r for path in directory.glob("s*-seat*/player-*.jsonl") for r in read_journal(path) if r["kind"] == "acknowledgement"]
    assert len(accepted) == 2 and all(r["response"]["status"] == "active" for r in accepted)
