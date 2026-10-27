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


def test_staged_buy_allow_and_both_decisions_use_strict_client_workers_and_journals(tmp_path):
    """ADR0007 fixed control prefix; its seed/requests stay in this driver only."""
    import json
    from dataclasses import asdict

    from gf_common.domain import PolicyDecision, digest
    from gf_common.validation import reconcile_replay
    from gf_simulator_client.workers import ParticipantWorker
    from gf_training.journal import ParticipantJournal
    from tests.worker_fixtures import fixed_buy_choice_factory

    async def run():
        control = json.loads((ROOT / "docs/issues/cases/0003-buy-choice-control.json").read_text())
        seed = control["experiment_control"]["seed"]
        server_config = load_config(ROOT / "configs/experiments/smoke.toml").config.server
        server = ManagedServer(server_config.binary, server_config.config, seed, 2, tmp_path / "buy-server")
        client = None
        settings = ActionConfig(strategy="enumerate", max_candidates=4096, max_expansions=20000)
        total_accepted = 0
        try:
            startup = await server.start()
            client = AttestedClient(startup["base_url"], expected_fingerprint=seed_fingerprint(seed), fixed_seed_experiment=True,
                                    expected_effective=server.expected, startup_info=startup["test_server_info"])
            await client.attest()
            definitions = await client.public_catalog()
            for accept in (True, False):
                game = (await client.create())["game_id"]
                for _ in range(2):
                    await client.join(game, str(uuid.uuid4()))
                directory = tmp_path / ("accept" if accept else "reject")
                directory.mkdir()
                run_id = str(uuid.uuid4())
                journals = {viewer: ParticipantJournal(directory, "buy-component", viewer, "fixed-buy-control", run_id=run_id,
                             game_id=game, action_config_hash=digest(asdict(settings))) for viewer in (1, 2)}
                workers = {viewer: ParticipantWorker(fixed_buy_choice_factory, "accept" if accept else "reject", 17,
                                                     settings, 3) for viewer in (1, 2)}
                histories, cursors, accepted = {1: [], 2: []}, {1: None, 2: None}, {1: [], 2: []}
                async def current(viewer, game=game, cursors=cursors, histories=histories):
                    obs = await client.observe(game, viewer)
                    assert obs["actionable"] and obs["current_player"] == viewer
                    options = await client.actor_options(game, viewer, obs)
                    events, cursors[viewer] = await client.events(game, viewer, cursors[viewer])
                    histories[viewer].extend(events)
                    return decision_input(obs, histories[viewer], cursors[viewer], definitions, options)
                async def submit(viewer, value, candidates, selection, game=game, journals=journals, accepted=accepted):
                    nonlocal total_accepted
                    command = {"command_id": str(uuid.uuid4()), "expected_revision": value.observation["state_revision"],
                               "action": candidates.actions[selection.index]}
                    index = journals[viewer].begin(value, candidates, selection, command, 0)
                    ack = await client.act(game, viewer, command)
                    journals[viewer].acknowledge(index, ack)
                    accepted[viewer].append(command["action"])
                    total_accepted += 1
                    return ack
                try:
                    for step in control["actions_before_observation"]:
                        viewer = step["player_id"]
                        value = await current(viewer)
                        assert value.observation["state_revision"] == step["expected_revision"]
                        candidates = PublicCandidateGenerator(settings).generate(value)
                        selected = PolicyDecision(candidates.actions.index(step["typed_action"]), "saved-public-prefix-component-v1")
                        await submit(viewer, value, candidates, selected)
                    target = await current(1)
                    assert target.observation["state_revision"] == 6
                    assert (target.observation["phase"], target.observation["required_choice"]) == ("awaiting_response", "defense")
                    assert target.action_options["action_options"]["can_allow"] is True
                    candidates, selected = await workers[1].select(target)
                    assert candidates.actions[selected.index] == {"type": "respond", "ordered_defenses": []}
                    await submit(1, target, candidates, selected)
                    buyer = await current(2)
                    assert buyer.observation["state_revision"] == 7
                    assert (buyer.observation["phase"], buyer.observation["required_choice"]) == ("resolving", "buy_accept_reject")
                    assert buyer.action_options["action_options"]["can_allow"] is False
                    nonactor = await client.observe(game, 1)
                    assert not nonactor["actionable"] and nonactor["typed_action_schema"] == []
                    assert nonactor["current_player"] == 2
                    with pytest.raises(ValueError):
                        await client.actor_options(game, 1, nonactor)
                    seller_before = nonactor["own"]
                    buyer_before = buyer.observation["own"]
                    offer = next(event for event in reversed(buyer.events) if event["type"] == "buy_offered")
                    assert offer["buyer"] == 2 and offer["seller"] == 1 and offer["definition_id"] == "wood-shield" and offer["price"] == 4
                    candidates, selected = await workers[2].select(buyer)
                    assert candidates.actions == ({"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True})
                    assert candidates.actions[selected.index] == {"type": "buy_decision", "accept": accept}
                    await submit(2, buyer, candidates, selected)
                    after = {viewer: await client.observe(game, viewer) for viewer in (1, 2)}
                    assert after[2]["own"]["money"] == buyer_before["money"] - (4 if accept else 0)
                    assert after[1]["own"]["money"] == seller_before["money"] + (4 if accept else 0)
                    assert (any(card["definition_id"] == "wood-shield" for card in after[2]["own"]["hand"])) == accept
                    assert (any(card["instance_id"] == 7 for card in after[1]["own"]["hand"])) == (not accept)
                    events, _ = await client.events(game, 2, cursors[2])
                    transfers = [event for event in events if event["type"] == "artifact_transferred"]
                    assert len(transfers) == int(accept)
                    for viewer in (1, 2):
                        replay = await client.replay(game, viewer)
                        reconcile_replay(replay, viewer, accepted[viewer], None)
                        journals[viewer].finish("truncated", None)
                        records = read_journal(journals[viewer].final_path)
                        assert records[-1]["terminal_reward"] is None and records[-1]["pending_decisions"] == []
                    assert read_journal(journals[2].final_path)[-2]["response"]["previous_revision"] == 7
                finally:
                    for journal in journals.values():
                        journal.close_incomplete()
                    for worker in workers.values():
                        await worker.close()
                await client.delete(game)
        finally:
            if client:
                await client.close()
            await server.stop()
        assert total_accepted == 16  # six-prefix + Allow + purchase, both branches
        assert server.process.returncode == 0 and server.health["shutdown"] == "stdin_eof"
    asyncio.run(run())
