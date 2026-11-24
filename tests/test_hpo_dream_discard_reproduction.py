"""Diagnostic replay of recorded HPO6 Dream rejections under their original releases."""

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
    shared_card_instance_contract,
    strength_powder_card_instance_contract,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_validation import decision_input
from gf_common.diverse_actions import NumericCosts
from gf_common.domain import DecisionInput, Json
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import frame_dict, parse_game_generated_v3
from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
from gf_simulator_client.http import ApiError, AttestedClient, seed_fingerprint
from gf_simulator_client.process import ManagedServer

_ROOT = Path(__file__).parents[1]


def _assert_native_search_admits_action(
    value: DecisionInput,
    submitted: Json,
    search: Callable[[bytes], bytes],
    *,
    contract: CardInstanceContract,
    clock_calls: int,
) -> None:
    """Check public-only candidate generation with a deterministic diagnostic clock."""
    actor = value.observation["viewer_player_id"]
    bank = EventBank(actor, "historical-dream-diagnostic", layout=LAYOUT)
    bank.append(event_delta(actor, bank.episode, 0, bank.boundary.prefix_hash,
                            value.events, layout=LAYOUT))
    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
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
    assert submitted in [
        generated.table.to_action(index, frame) for index in range(len(generated.table.kinds))
    ]


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
