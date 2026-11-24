"""Diagnostic replay of recorded HPO6 actions against pinned Simulator releases."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from random import Random

import pytest

from gf_common.card_instance import (
    ActionConfigV3,
    CardInstanceContract,
    dream_compat_card_instance_contract,
    moon_dream_card_instance_contract,
    pestle_mirage_card_instance_contract,
    shared_card_instance_contract,
    strength_powder_card_instance_contract,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_validation import decision_input, reconcile_replay
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import DecisionInput, Json
from gf_common.structured import DecisionFrame, EventBank, Selection, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import frame_dict, parse_game_generated_v3
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.http import ApiError, AttestedClient, seed_fingerprint
from gf_simulator_client.process import ManagedServer
from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal

_ROOT = Path(__file__).parents[1]


def _public_frame(value: DecisionInput, episode: str) -> tuple[DecisionFrame, EventBank]:
    actor = value.observation["viewer_player_id"]
    bank = EventBank(actor, episode, layout=LAYOUT)
    bank.append(event_delta(actor, bank.episode, 0, bank.boundary.prefix_hash,
                            value.events, layout=LAYOUT))
    return DecisionFrame.from_input(value, bank, 0, layout=LAYOUT), bank


def _native_search_actions(
    value: DecisionInput,
    search: Callable[[bytes], bytes],
    *,
    contract: CardInstanceContract,
    clock_calls: int,
) -> list[Json]:
    """Generate from the participant's public inputs with a deterministic clock."""
    frame, _ = _public_frame(value, "historical-hpo-diagnostic")
    actions = ActionConfigV3(action_contract=contract)
    rules = card_instance_rules()
    request = {
        "frame": frame_dict(frame, layout=LAYOUT),
        "events": value.events,
        "actions": asdict(actions),
        "rules": asdict(rules),
        "numeric_costs": asdict(NumericCosts()),
        "rng_state": Random(17).getstate(),
        "clock": [0.0] * clock_calls,
    }
    result = json.loads(search(json.dumps(request).encode()))
    generated = parse_game_generated_v3(
        result["generated"], action_contract=actions.action_contract, rules=rules
    )
    return [
        generated.table.to_action(index, frame) for index in range(len(generated.table.kinds))
    ]


def _assert_native_search_admits_action(
    value: DecisionInput,
    submitted: Json,
    search: Callable[[bytes], bytes],
    *,
    contract: CardInstanceContract,
    clock_calls: int,
) -> None:
    assert submitted in _native_search_actions(
        value, search, contract=contract, clock_calls=clock_calls,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture,release,env,visible_id,actual_id,clock_calls",
    [
        pytest.param(
            "hpo_dream_discard_p11.json", shared_card_instance_contract(),
            "GFB_HTTP_DIAGNOSTICS_TEST", "strength-powder", "wand-of-mystic-water", 1024,
            id="p11-discard",
        ),
        pytest.param(
            "hpo_dream_costcut_p41.json", strength_powder_card_instance_contract(),
            "GFB_P41_TEST", "spiritual-staff", "saw-boom-boom", 1_000_000,
            id="p41-costcut",
        ),
    ],
)
def test_hpo_dream_matches_recorded_rejection(
    tmp_path: Path,
    fixture: str,
    release: CardInstanceContract,
    env: str,
    visible_id: str,
    actual_id: str,
    clock_calls: int,
) -> None:
    binary = os.environ.get(f"{env}_SERVER")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    source = os.environ.get(f"{env}_SOURCE")
    if not binary or not config or not source:
        pytest.skip("explicit managed Simulator binary/config/approved source required")
    native = pytest.importorskip("gf_native")

    case = json.loads((_ROOT / "tests" / "fixtures" / fixture).read_text())
    assert release.simulator_revision == case["simulator_revision"]
    evidence = CardInstanceRuntimeEvidence(Path(binary), Path(source), contract=release)
    evidence.preflight()
    server = ManagedServer(binary, config, case["seed"], 1, tmp_path / "server")

    async def run() -> None:
        startup = await server.start()
        try:
            evidence.verify_executed(server.process)
            client = AttestedClient(
                startup["base_url"],
                expected_fingerprint=seed_fingerprint(case["seed"]),
                fixed_seed_experiment=True,
                expected_effective=startup["test_server_config"],
                startup_info=startup["test_server_info"],
                contract=release,
                retries=0,
            )
            try:
                game = (await client.create())["game_id"]
                for player in (1, 2):
                    assert (await client.join(game, str(uuid.uuid4())))["player_id"] == player
                revision = case["revision"]
                assert len(case["prefix"]) == revision
                for previous, step in enumerate(case["prefix"]):
                    receipt = await client.act(
                        game,
                        step["player"],
                        {
                            "command_id": str(uuid.uuid4()),
                            "expected_revision": previous,
                            "action": step["action"],
                        },
                    )
                    assert receipt["state_revision"] == previous + 1
                    assert receipt["status"] == "active"

                actor = case["actor"]
                observation = await client.observe(game, actor)
                assert observation["state_revision"] == revision
                assert observation["turn"] == case["turn"]
                assert observation["end_times"] == case["end_times"]
                assert observation["required_choice"] == case["required_choice"]
                assert observation["actionable"] is True
                assert observation["current_player"] == actor
                assert observation["own"] == case["own"]
                options = await client.actor_options(game, actor, observation)
                assert options == {
                    "state_revision": revision,
                    "action_options": case["hints"],
                }
                events, history_end = await client.events(game, actor, None)
                value = decision_input(
                    observation, events, history_end, await client.public_catalog(), options
                )
                submitted = case["submitted"]
                _assert_native_search_admits_action(
                    value, submitted, native.player_search_v5,
                    contract=release, clock_calls=clock_calls,
                )

                # The private dump supplies diagnostic evidence only.
                response = await client.http.get(f"/debug/v1/games/{game}/dump")
                assert response.status_code == 200
                diagnostic = response.json()
                assert diagnostic["state_revision"] == revision
                cards = diagnostic["full_state"]["players"][actor - 1]["cards"]["cards"]
                visible = observation["own"]["cards"]
                descriptors = [row["card"].get("instance", row["card"]) for row in cards]
                assert [row["perceived_definition_id"] for row in descriptors] == [
                    row["definition_id"] for row in visible
                ]
                selected = submitted.get("card_refs") or submitted["ordered_card_refs"]
                slot, = (
                    index for index, row in enumerate(visible)
                    if row["card_ref"] in selected and row["definition_id"] == visible_id
                )
                assert descriptors[slot]["actual_definition_id"] == actual_id

                with pytest.raises(ApiError) as caught:
                    await client.act(
                        game,
                        actor,
                        {
                            "command_id": str(uuid.uuid4()),
                            "expected_revision": revision,
                            "action": submitted,
                        },
                    )
                error = caught.value.body["error"]
                details = error["details"]
                assert {
                    "http_status": caught.value.status,
                    "code": error["code"],
                    "rejection_code": details["rejection_code"],
                    "revision": details["state_revision"],
                    "invalid_attempt_count": details["invalid_attempt_count"],
                } == case["response"]
                assert await client.observe(game, actor) == observation
            finally:
                await client.close()
        finally:
            health = await server.stop()
        assert health["exit_code"] == 0
        assert health["log_error"] is None

    asyncio.run(run())


@pytest.mark.integration
def test_hpo_dream_concealed_draws_keep_journal_replay_and_next_action(tmp_path: Path) -> None:
    binary = os.environ.get("GFB_P65_TEST_SERVER")
    source = os.environ.get("GFB_P65_TEST_SOURCE")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    if not binary or not source or not config:
        pytest.skip("explicit managed Simulator binary/config/approved source required")
    native = pytest.importorskip("gf_native")
    case = json.loads((_ROOT / "docs/issues/cases/0015-concealed-dream-draw.json").read_text())
    release = dream_compat_card_instance_contract()
    assert (release.simulator_revision, release.profile_id, release.catalog_version) == (
        case["simulator_revision"], case["profile_id"], case["catalog_version"],
    )
    expected = case["expected"]
    viewer = expected["viewer"]
    opponent = 3 - viewer
    evidence = CardInstanceRuntimeEvidence(Path(binary), Path(source), contract=release)
    evidence.preflight()
    server = ManagedServer(binary, config, case["game_seed"], 1, tmp_path / "server")
    actions = ActionConfigV3(action_contract=release)
    rules = card_instance_rules()
    caps, costs, _, _ = own_game_worker_settings_v3("random", actions, None, rules=rules)

    def own_draws(events):
        return [row for row in events if row["type"] == "artifact_drawn" and row["player"] == viewer]

    def replay_decision(replay, revision):
        decision, = (decision for turn in replay["turns"] for decision in turn["decisions"]
                     if decision["observation"]["state_revision"] == revision)
        return decision

    async def run() -> None:
        startup = await server.start()
        journals = {}
        try:
            runtime_receipt = evidence.verify_executed(server.process)
            client = AttestedClient(
                startup["base_url"],
                expected_fingerprint=seed_fingerprint(case["game_seed"]),
                fixed_seed_experiment=True,
                expected_effective=startup["test_server_config"],
                startup_info=startup["test_server_info"],
                contract=release,
                retries=0,
            )
            act_responses = []

            async def capture_ack(response):
                if response.request.method == "POST" and response.request.url.path.endswith(("/1/actions", "/2/actions")):
                    await response.aread()
                    act_responses.append((response.status_code, response.json()))

            client.http.event_hooks["response"].append(capture_ack)
            try:
                game = (await client.create())["game_id"]
                for actor in (1, 2):
                    assert (await client.join(game, str(uuid.uuid4())))["player_id"] == actor
                    journals[actor] = CardInstanceJournal(
                        tmp_path, "p65", actor, "random-card-instance-v2",
                        run_id="p65-diagnostic", game_id=game, actions=actions,
                        bank_episode="p65", caps=asdict(caps), numeric_costs=asdict(costs),
                        model_id="random", runtime_evidence=runtime_receipt, rules=rules,
                    )
                catalog = await client.public_catalog()
                banks = {actor: EventBank(actor, "p65", layout=LAYOUT) for actor in (1, 2)}
                cursors = dict.fromkeys(banks)

                for step in [*case["steps"], case["next_step"]]:
                    actor, revision = step["actor"], step["expected_revision"]
                    bank, journal = banks[actor], journals[actor]
                    obs = await client.observe(game, actor)
                    assert obs["state_revision"] == revision
                    hints = await client.actor_options(game, actor, obs)
                    new_events, cursors[actor] = await client.events(game, actor, cursors[actor])
                    start = bank.boundary
                    delta = event_delta(actor, bank.episode, start.end, start.prefix_hash, new_events, layout=LAYOUT)
                    bank.append(delta)
                    visible_events = bank.view().events()
                    value = decision_input(obs, list(visible_events), cursors[actor], catalog, hints)
                    frame = DecisionFrame.from_input(value, bank, start.end, layout=LAYOUT)
                    request = {
                        "frame": frame_dict(frame, layout=LAYOUT), "events": visible_events,
                        "actions": asdict(actions), "rules": asdict(rules), "numeric_costs": asdict(costs),
                        "rng_state": journal.last_search_state, "clock": [0.0] * 1_000_000,
                    }
                    generated = parse_game_generated_v3(
                        json.loads(native.player_search_v5(json.dumps(request).encode()))["generated"],
                        action_contract=release, rules=rules,
                    )
                    action = step["action"]
                    index, = (index for index in range(len(generated.table.kinds))
                              if generated.table.to_action(index, frame) == action)
                    selection = Selection(generated.table.table_hash, index, journal.policy)
                    command = {"command_id": str(uuid.uuid4()), "expected_revision": revision, "action": action}
                    ordinal = journal.begin_frame(frame, delta, generated, selection, command, {})
                    receipt = await client.act(game, actor, command)
                    journal.acknowledge(ordinal, receipt)
                    assert receipt["state_revision"] == revision + 1
                    assert receipt["status"] == "active"

                    if revision == expected["revision_before"]:
                        status, raw_ack = act_responses[-1]
                        assert status == expected["ack_http_status"]
                        ack_events = raw_ack["events"]
                        draws = own_draws(ack_events)
                        assert len(draws) == expected["own_draws"]
                        assert draws.count(expected["event"]) == expected["own_concealed_draws"]
                        own_feed, _ = await client.events(game, viewer, None)
                        other_feed, _ = await client.events(game, opponent, None)
                        assert own_feed[-len(ack_events):] == ack_events
                        assert not own_draws(other_feed)
                        own_replay, other_replay = await client.replay(game, viewer), await client.replay(game, opponent)
                        assert replay_decision(own_replay, revision)["visible_events"] == ack_events
                        assert not own_draws(replay_decision(other_replay, revision)["visible_events"])

                assert receipt["state_revision"] == 8
                for actor, journal in journals.items():
                    journal.finish("truncated", None)
                    stored = read_card_instance_journal(journal.final_path, actions=actions, rules=rules)
                    replay = await client.replay(game, actor)
                    assert replay["episode_end"] is None
                    reconcile_replay(replay, actor, [stored.decisions[i].command["action"] for i in stored.accepted], None)
                    assert stored.bank.events() == banks[actor].view().events()
                    markers = stored.bank.events().count(expected["event"])
                    assert markers == expected["own_concealed_draws" if actor == viewer else "opponent_concealed_draws"]
            finally:
                for journal in journals.values():
                    journal.output.close()
                await client.close()
        finally:
            health = await server.stop()
        assert health["exit_code"] == 0
        assert health["log_error"] is None

    asyncio.run(run())


@pytest.mark.integration
def test_hpo_moon_god_mirage_reaches_weapon_defense(tmp_path: Path) -> None:
    binary = os.environ.get("GFB_MOON_DREAM_TEST_SERVER")
    source = os.environ.get("GFB_MOON_DREAM_TEST_SOURCE")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    if not binary or not source or not config:
        pytest.skip("explicit managed Simulator binary/config/approved source required")
    native = pytest.importorskip("gf_native")
    release = moon_dream_card_instance_contract()
    path = _ROOT / "docs/issues/cases/0016-moon-god-standalone-mirage.json"
    case = json.loads(path.read_text())
    expected = case["expected_repaired"]
    original = case["rejected"]
    evidence = CardInstanceRuntimeEvidence(Path(binary), Path(source), contract=release)
    evidence.preflight()
    server = ManagedServer(binary, config, case["game_seed"], 1, tmp_path / "server")

    async def run() -> None:
        startup = await server.start()
        try:
            evidence.verify_executed(server.process)
            client = AttestedClient(
                startup["base_url"],
                expected_fingerprint=seed_fingerprint(case["game_seed"]),
                fixed_seed_experiment=True,
                expected_effective=startup["test_server_config"],
                startup_info=startup["test_server_info"],
                contract=release,
                retries=0,
            )
            ack = None

            async def capture_act(response):
                nonlocal ack
                if response.request.method == "POST" and response.request.url.path.endswith("/actions"):
                    await response.aread()
                    ack = response.status_code, response.json()

            client.http.event_hooks["response"].append(capture_act)
            try:
                game = (await client.create())["game_id"]
                for actor in (1, 2):
                    assert (await client.join(game, str(uuid.uuid4())))["player_id"] == actor
                assert len(case["accepted"]) == original["revision"] == 70
                for revision, step in enumerate([*case["accepted"], original]):
                    assert step["revision"] == revision
                    receipt = await client.act(game, step["actor"], {
                        "command_id": str(uuid.uuid4()),
                        "expected_revision": revision,
                        "action": step["action"],
                    })
                    assert receipt["state_revision"] == revision + 1
                    assert receipt["status"] == "active"
                assert ack is not None
                status, raw = ack
                assert status == expected["ack_http_status"]
                assert raw["state_revision"] == expected["state_revision"]
                assert raw["events"].count(expected["guardian_acted"]) == 1
                assert raw["events"].count(expected["attack_declared"]) == 1

                defender = expected["defender"]
                observation = await client.observe(game, defender)
                assert observation["state_revision"] == expected["state_revision"]
                assert observation["current_player"] == defender and observation["actionable"] is True
                assert observation["required_choice"] == "defense"
                hints = await client.actor_options(game, defender, observation)
                events, cursor = await client.events(game, defender, None)
                assert expected["guardian_acted"] in events
                assert expected["attack_declared"] in events
                replay = await client.replay(game, defender)
                replay_events, = (
                    decision["visible_events"]
                    for turn in replay["turns"] for decision in turn["decisions"]
                    if decision["observation"]["state_revision"] == original["revision"]
                )
                assert replay_events == raw["events"]
                value = decision_input(observation, events, cursor, await client.public_catalog(), hints)
                _assert_native_search_admits_action(
                    value, expected["response"], native.player_search_v5,
                    contract=release, clock_calls=1_000_000,
                )
                receipt = await client.act(game, defender, {
                    "command_id": str(uuid.uuid4()),
                    "expected_revision": expected["state_revision"],
                    "action": expected["response"],
                })
                assert receipt["state_revision"] == expected["response_revision"]
                assert receipt["status"] == "active"
            finally:
                await client.close()
        finally:
            health = await server.stop()
        assert health["exit_code"] == 0
        assert health["log_error"] is None

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize(
    "release,env,repaired",
    [
        pytest.param(moon_dream_card_instance_contract(), "GFB_MOON_DREAM_TEST", False, id="archived"),
        pytest.param(pestle_mirage_card_instance_contract(), "GFB_PESTLE_MIRAGE_TEST", True, id="repaired"),
    ],
)
def test_hpo_pestle_mirage_replays_recorded_target(
    tmp_path: Path, release: CardInstanceContract, env: str, repaired: bool,
) -> None:
    binary = os.environ.get(f"{env}_SERVER")
    source = os.environ.get(f"{env}_SOURCE")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    if not binary or not source or not config:
        pytest.skip("explicit managed Simulator binary/config/approved source required")
    native = pytest.importorskip("gf_native")
    case = json.loads((_ROOT / "docs/issues/cases/0017-pestle-mirage-target.json").read_text())
    old = case["rejected"]
    expected = case["expected_repaired"]
    actor, revision = old["actor"], old["revision"]
    opponent = expected["submitted_target"]
    assert old["action"]["target"] == actor and opponent == 3 - actor
    if not repaired:
        assert case["original_revision"] == release.simulator_revision
    evidence = CardInstanceRuntimeEvidence(Path(binary), Path(source), contract=release)
    evidence.preflight()
    server = ManagedServer(binary, config, case["game_seed"], 1, tmp_path / "server")

    async def run() -> None:
        startup = await server.start()
        try:
            evidence.verify_executed(server.process)
            client = AttestedClient(
                startup["base_url"],
                expected_fingerprint=seed_fingerprint(case["game_seed"]),
                fixed_seed_experiment=True,
                expected_effective=startup["test_server_config"],
                startup_info=startup["test_server_info"],
                contract=release,
                retries=0,
            )
            try:
                game = (await client.create())["game_id"]
                for player in (1, 2):
                    assert (await client.join(game, str(uuid.uuid4())))["player_id"] == player
                assert len(case["accepted"]) == revision
                for previous, step in enumerate(case["accepted"]):
                    assert step["revision"] == previous
                    receipt = await client.act(game, step["actor"], {
                        "command_id": str(uuid.uuid4()),
                        "expected_revision": previous,
                        "action": step["action"],
                    })
                    assert receipt["state_revision"] == previous + 1 and receipt["status"] == "active"

                observation = await client.observe(game, actor)
                assert observation["state_revision"] == revision and observation["actionable"] is True
                assert observation["current_player"] == actor
                refs = old["action"]["ordered_card_refs"]
                visible = {row["card_ref"]: row["definition_id"] for row in observation["own"]["cards"]}
                assert [visible[ref] for ref in refs] == ["dangerous-pestle", "mirage"]
                hints = await client.actor_options(game, actor, observation)
                assert set(hints["action_options"]["valid_targets"]) == {actor, opponent}
                events, cursor = await client.events(game, actor, None)
                value = decision_input(observation, events, cursor, await client.public_catalog(), hints)

                frame, bank = _public_frame(value, "pestle-mirage-diagnostic")
                generated = PublicCandidateGeneratorV3(
                    ActionConfigV3(action_contract=release), rules=card_instance_rules(), clock=lambda: 0.0,
                ).generate_game(frame, bank.view(), Random(17), NumericCosts())
                py_actions = [generated.table.to_action(i, frame) for i in range(len(generated.table.kinds))]
                native_actions = _native_search_actions(
                    value, native.player_search_v5, contract=release, clock_calls=1_000_000,
                )
                for actions in (py_actions, native_actions):
                    assert {action["target"] for action in actions
                            if action["type"] == "play_sequence" and action["ordered_card_refs"] == refs
                            } == ({opponent} if repaired else {actor, opponent})

                submitted = old["action"] | {"target": opponent} if repaired else old["action"]
                request = {"command_id": str(uuid.uuid4()), "expected_revision": revision, "action": submitted}
                if not repaired:
                    with pytest.raises(ApiError) as caught:
                        await client.act(game, actor, request)
                    error = caught.value.body["error"]
                    assert (caught.value.status, error["details"]["rejection_code"]) == (
                        old["http_status"], old["rejection_code"],
                    )
                    assert await client.observe(game, actor) == observation
                    return

                receipt = await client.act(game, actor, request)
                assert receipt["state_revision"] == expected["revision"] and receipt["status"] == "active"
                declared = {"type": "attack_declared", "attacker": actor,
                            "target": expected["drawn_target"], "attack": expected["attack"]}
                defender = await client.observe(game, expected["drawn_target"])
                assert defender["current_player"] == expected["drawn_target"]
                assert defender["required_choice"] == "defense"
                visible_events, _ = await client.events(game, expected["drawn_target"], None)
                assert declared in visible_events

                # Read the authoritative packet only after the policy and submission are finished.
                response = await client.http.get(f"/debug/v1/games/{game}/dump")
                response.raise_for_status()
                diagnostic = response.json()
                assert diagnostic["state_revision"] == expected["revision"]
                pending = diagnostic["full_state"]["pending_attack"]
                assert {key: pending[key] for key in ("target", "attack", "attribute")} == {
                    "target": expected["drawn_target"], "attack": expected["attack"],
                    "attribute": expected["attribute"],
                }
            finally:
                await client.close()
        finally:
            health = await server.stop()
            assert health["exit_code"] == 0 and health["log_error"] is None

    asyncio.run(run())
