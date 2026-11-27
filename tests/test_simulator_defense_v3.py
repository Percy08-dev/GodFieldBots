"""Original Simulator v3 participant DTOs and raw identity regressions."""

import asyncio
import copy
import json
import random
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from gf_common import card_instance, card_instance_validation
from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_runtime import approved_runtime_build
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import ContractError, digest
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_layout import EVENT_ACTORS, layout_for_contract
from gf_common.structured_tokens import MISSING, tokenize_frame
from gf_common.structured_wire import delta_dict, frame_dict, parse_delta, parse_frame
from gf_simulator_client.http import EFFECTIVE_KEYS, AttestedClient
from tests.card_instance_action_support import inputs
from tests.test_card_instance_contract import fixture
from tests.test_card_instance_contract import replay as replay_fixture

V3_RULE = "true-match-1v1-v3"
DIRECT = "defense:direct"
DECLARED = {
    "type": "attack_declared", "target": 1, "attack": 30,
    "attribute": "fire", "definition_ids": ["fireball", "fireball"],
}
MISSED = {"type": "attack_missed", "target": 1}


def contract():
    factory = getattr(card_instance, "defense_v3_card_instance_contract", None)
    assert factory is not None, "Simulator v3 needs an exact selectable Action contract"
    return factory()


def original_v3(value, direct=()):
    value.observation["rule_set_version"] = V3_RULE
    hints = value.action_options["action_options"]
    hints["rule_set_version"] = V3_RULE
    if value.observation["required_choice"] == "defense":
        hints["candidate_groups"][DIRECT] = [hints["usable_cards"][index] for index in direct]
    return value


def frame_for(value):
    profile = contract()
    actor = value.observation["viewer_player_id"]
    bank = EventBank(actor, "defense-v3-python", layout=LAYOUT, action_contract=profile)
    start = bank.boundary
    delta = event_delta(
        actor, bank.episode, start.end, start.prefix_hash, value.events,
        layout=LAYOUT, action_contract=profile,
    )
    bank.append(delta)
    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
    return frame, bank, delta


def generate(value):
    frame, bank, _ = frame_for(value)
    config = ActionConfigV3(action_contract=contract())
    result = PublicCandidateGeneratorV3(config, rules=card_instance_rules(), clock=lambda: 0.0).generate(
        frame, bank.view(), random.Random(17), NumericCosts()
    )
    return {tuple(result.table.to_action(i, frame)["ordered_card_refs"]) for i in range(len(result.table.kinds))}


def test_release_profile_selects_exact_pins_and_existing_numerical_layout():
    profile = contract()
    assert profile.profile_id == "card-instance-simulator-v3-shared-20261127-defense"
    assert profile.simulator_revision == "997a0de59a456dd84699605c178f12ec74addf84"
    assert profile.gameplay_api == "v3"
    assert profile.catalog_version == "public-ja-v4"
    assert card_instance.parse_card_instance_contract(asdict(profile)) == profile
    assert layout_for_contract(profile) is LAYOUT
    assert card_instance.is_shared_contract(profile)
    assert profile.requires_powder_weapon and profile.mirage_danger_requires_opponent and profile.protects_sacrifice_artifacts


def test_p46_cannot_select_spiritual_hood_as_direct_but_can_waive_turbulence_mp():
    case = json.loads(Path("docs/issues/cases/0019-defense-costcut-role.json").read_text())
    public = case["actor_public"]
    obs, hints = public["observation"], public["action_options"]
    obs["rule_set_version"] = hints["action_options"]["rule_set_version"] = V3_RULE
    rows = hints["action_options"]["usable_cards"]
    hints["action_options"]["candidate_groups"][DIRECT] = [rows[1]]
    definitions = {row["definition_id"]: row for row in fixture("catalog")["definitions"]}
    value = card_instance_validation.decision_input(
        obs, public["events"], public["history_end"], definitions, hints, action_contract=contract(),
    )
    assert generate(value) == {(), tuple(case["accepted_response"]["action"]["ordered_card_refs"])}


def test_neutral_cutcost_armor_remains_a_direct_choice_when_listed():
    value = original_v3(inputs([("spiritual-hood", None)], choice="defense"), direct=(0,))
    assert generate(value) == {(), (value.observation["own"]["cards"][0]["card_ref"],)}


@pytest.mark.parametrize("mutation", ["missing", "reordered", "foreign", "definition", "revision", "unusable", "actor"])
def test_defense_direct_is_required_and_uses_ordered_same_revision_own_usable_rows(mutation):
    value = original_v3(inputs([("spiritual-hood", None), ("turbulence", "learned")], choice="defense"), direct=(0, 1))
    options = value.action_options["action_options"]
    groups = options["candidate_groups"]
    if mutation == "missing":
        del groups[DIRECT]
    elif mutation == "reordered":
        groups[DIRECT].reverse()
    elif mutation == "foreign":
        groups[DIRECT][1] = {"card_ref": "c1_" + "f" * 32, "definition_id": "turbulence"}
    elif mutation == "definition":
        groups[DIRECT][1] = {**groups[DIRECT][1], "definition_id": "spiritual-hood"}
    elif mutation == "revision":
        options["state_revision"] += 1
    elif mutation == "unusable":
        options["usable_cards"] = options["usable_cards"][:1]
        groups["uninterpreted"] = options["usable_cards"]
    elif mutation == "actor":
        options["actor"] = 2
    with pytest.raises(ContractError):
        card_instance_validation.actor_action_options(value.action_options, value.observation, raw=False, action_contract=contract())


def test_an_empty_direct_group_is_known_and_turn_options_omit_it():
    assert generate(original_v3(inputs([("spiritual-hood", None)], choice="defense"))) == {()}
    value = original_v3(inputs([("spiritual-hood", None)]))
    value.action_options["action_options"]["candidate_groups"][DIRECT] = []
    with pytest.raises(ContractError):
        card_instance_validation.actor_action_options(value.action_options, value.observation, raw=False, action_contract=contract())


def test_fog_attack_raw_hashes_and_missing_attacker_use_the_existing_numerical_layout():
    value = replace(original_v3(inputs([("spiritual-hood", None)])), events=(DECLARED, MISSED))
    frame, bank, delta = frame_for(value)
    start = digest([LAYOUT.input_recipe, "event-bank", 1, bank.episode, asdict(contract())])
    assert delta.previous_hash == start
    raw = frame_dict(frame, layout=LAYOUT, action_contract=contract())
    assert raw["frame_hash"] == digest({
        **{key: value for key, value in raw.items() if key != "frame_hash"},
        "action_contract": asdict(contract()),
        "input_recipe": LAYOUT.input_recipe,
        "policy_input_version": LAYOUT.policy_input_version,
    })
    assert bank.view().events() == (DECLARED, MISSED)
    assert parse_frame(raw, layout=LAYOUT, action_contract=contract()) == frame
    with pytest.raises(ContractError):
        parse_frame(raw, layout=LAYOUT)
    raw_delta = delta_dict(delta, layout=LAYOUT, action_contract=contract())
    with pytest.raises(ContractError):
        parse_delta(raw_delta, layout=LAYOUT)
    with pytest.raises(ContractError):
        frame.bank_view(EventBank(1, bank.episode, layout=LAYOUT).view())
    named_value = replace(value, events=(
        {"type": "attack_declared", "attacker": 2, "target": 1, "attack": 30},
        {"type": "attack_missed", "attacker": 2, "target": 1},
    ))
    named, named_bank, _ = frame_for(named_value)
    first = tokenize_frame(frame, bank.view(), layout=LAYOUT)
    second = tokenize_frame(named, named_bank.view(), layout=LAYOUT)
    assert first.recipe_signature == second.recipe_signature
    assert first.interval_numerics == second.interval_numerics
    attacker_column = 4 + EVENT_ACTORS.index("attacker")
    target_column = 4 + EVENT_ACTORS.index("target")
    for row, named_row in zip(first.interval_categories, second.interval_categories, strict=True):
        assert row[attacker_column] == MISSING < named_row[attacker_column]
        assert row[target_column] == named_row[target_column] > MISSING
    empty = EventBank(1, bank.episode, layout=LAYOUT, action_contract=contract())
    foreign = event_delta(1, bank.episode, 0, empty.boundary.prefix_hash, named_value.events, layout=LAYOUT)
    with pytest.raises(ContractError, match="owner/continuity"):
        empty.append(foreign)


@pytest.mark.parametrize("row", [
    {**DECLARED, "attacker": 2}, {**DECLARED, "target": 2}, {**DECLARED, "attribute": "unknown"},
    {**DECLARED, "definition_ids": ["concealed"]}, {**DECLARED, "definition_ids": "fireball"},
    {k: v for k, v in DECLARED.items() if k != "attribute"},
    {**MISSED, "attack": 30},
])
def test_fog_projection_rejects_hybrid_foreign_or_invalid_rows(row):
    value = replace(original_v3(inputs([("spiritual-hood", None)])), events=(copy.deepcopy(row),))
    with pytest.raises(ContractError):
        frame_for(value)


def test_replay_binds_anonymous_event_to_viewer_not_decision_actor():
    original = replay_fixture(concealed=True)
    original.update(rule_set_version=V3_RULE, public_event_schema_version="3")
    decision = original["turns"][0]["decisions"][0]
    assert decision["actor"] == 2 and original["viewer_player_id"] == 1
    decision["observation"]["rule_set_version"] = V3_RULE
    decision["visible_events"] = [DECLARED, MISSED]
    assert card_instance_validation.replay(original, "game", 1, action_contract=contract()) == original
    decision["visible_events"] = [{**DECLARED, "target": 2}]
    with pytest.raises(ContractError):
        card_instance_validation.replay(original, "game", 1, action_contract=contract())


def test_attested_http_uses_v3_route_and_catalog_and_fog_feed():
    info = fixture("info")
    info.update(api_version="v3", public_event_schema_version="3", rule_set_version=V3_RULE)
    catalog = fixture("catalog")
    catalog.update(
        catalog_version="public-ja-v4", rule_set_version=V3_RULE,
        content_sha256="c27c3341dcb0dfdf998322071641c32e9b5f22c8651427ab81a1a49fcb15cb27",
    )
    info["card_catalog"] = {key: val for key, val in catalog.items() if key != "definitions"}
    attestation = {
        "schema_version": 1, "server_kind": "test_server", "server_instance_id": "16d978ce-84f7-4609-be25-2bc520564205",
        "seed_mode": "repeated", "seed_fingerprint": "v1-" + "a" * 64, "seed_use_limit": 1,
        "max_invalid_attempts_per_decision": 3, "max_game_turns": 150,
        "waiting_for_players_timeout_secs": 2, "player_action_timeout_secs": 30,
        "graceful_shutdown_timeout_secs": 2, "server_idle_timeout_secs": 0,
    }
    responses = {
        "/debug/v1/test-server/info": attestation,
        "/v3/info": info,
        "/v3/card-catalogs/public-ja-v4": catalog,
        "/v3/games/game/events/1": {
            "public_event_schema_version": "3", "rule_set_version": V3_RULE,
            "events": [DECLARED, MISSED], "next_cursor": None, "has_more": False,
        },
    }

    def respond(request):
        assert request.url.path in responses, request.url.path
        return httpx.Response(200, json=responses[request.url.path], headers={"Cache-Control": "no-store"})

    async def run():
        client = AttestedClient(
            "http://test", expected_fingerprint=attestation["seed_fingerprint"],
            fixed_seed_experiment=True, expected_effective={key: attestation[key] for key in EFFECTIVE_KEYS},
            transport=httpx.MockTransport(respond), contract=contract(),
        )
        try:
            await client.attest()
            assert len(await client.public_catalog()) == 296
            assert await client.events("game", 1, None) == ([DECLARED, MISSED], None)
        finally:
            await client.close()

    asyncio.run(run())


def test_first_eligible_rainbow_can_neutralize_before_cutcost_armor():
    value = original_v3(
        inputs([("rainbow-curtain", None), ("spiritual-hood", None)], choice="defense"), direct=(0,),
    )
    curtain, hood = (row["card_ref"] for row in value.observation["own"]["cards"])
    assert generate(value) == {(), (curtain,), (curtain, hood)}


def test_flash_does_not_allow_cutcost_pair_even_when_it_waives_mp():
    value = original_v3(
        inputs([("turbulence", "unused"), ("spiritual-hood", None)], choice="defense", flash=True, mp=0),
        direct=(0,),
    )
    assert generate(value) == {()}


def test_v3_policy_worker_and_journal_round_trip_anonymous_originals(tmp_path):
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.card_instance_journal import CardInstanceJournal, read_card_instance_journal
    from tests.test_card_instance_baseline_runtime import shared_runtime_receipt

    value = replace(original_v3(inputs([("spiritual-hood", None)], choice="defense")), events=(DECLARED, MISSED))
    frame, bank, delta = frame_for(value)
    actions = ActionConfigV3(action_contract=contract(), max_sequence_length=4, max_candidates=50)
    rules = card_instance_rules()
    label = "random-card-instance-v2"
    seed = 17
    caps, costs, _, _ = own_game_worker_settings_v3("random", actions, None, rules=rules)
    receipt = shared_runtime_receipt(actions.action_contract)
    assert receipt["build"]["binary_sha256"] == "808f0641bb6467997bb20a952777aee0c3253629f366159ae7def95e710ff6e6"
    assert len(receipt["build"]["spec_files"]) == 13
    assert approved_runtime_build(actions.action_contract) == receipt["build"]

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("random", seed, actions, ledger=ledger)
        try:
            return await worker.select_frame(frame, delta)
        finally:
            await worker.close()

    generated, selection = asyncio.run(run())
    command = {
        "command_id": str(uuid.uuid4()), "expected_revision": frame.revision,
        "action": selection.validate(generated.table, frame),
    }
    assert command["action"] == {"type": "respond", "ordered_card_refs": []}
    journal = CardInstanceJournal(
        tmp_path, "episode", 1, label, run_id="run", game_id="game", actions=actions,
        bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs), model_id="random",
        runtime_evidence=receipt, sampling_seed=seed, rules=rules,
    )
    journal.begin_frame(frame, delta, generated, selection, command, {})
    journal.finish("incomplete", None)
    saved = read_card_instance_journal(journal.final_path, actions=actions)
    assert saved.header["catalog"]["catalog_version"] == "public-ja-v4"
    assert saved.bank.events() == (DECLARED, MISSED)
    assert saved.decisions[0].frame == frame
    old_actions = replace(actions, action_contract=card_instance.sacrifice_card_instance_contract())
    with pytest.raises(ContractError):
        read_card_instance_journal(journal.final_path, actions=old_actions)


def test_hpo6_rejects_native_wheel_without_the_exact_v3_capability():
    from gf_cli.rust_player_entry import require_native_hpo_action_profile
    from gf_training.native_backend import CARD_INSTANCE_CONCEALED_DRAW

    actions = ActionConfigV3(action_contract=contract())
    build = {"card_instance_concealed_draw": CARD_INSTANCE_CONCEALED_DRAW}
    with pytest.raises(ContractError, match="matching native runtime wheel"):
        require_native_hpo_action_profile(actions, build)
    require_native_hpo_action_profile(actions, {**build, "card_instance_defense_v3": actions.action_contract.profile_id})


def test_hpo_model4_checkpoint_uses_v3_catalog_and_shared_token_signature():
    from gf_models.hpo_config import model_defaults_v4
    from gf_models.structured_sealed import static_semantics_v4

    model, rules = model_defaults_v4("mlp"), card_instance_rules()
    semantics = static_semantics_v4(model, layout=LAYOUT, action_contract=contract(), rules=rules)
    historical = static_semantics_v4(
        model, layout=LAYOUT, action_contract=card_instance.sacrifice_card_instance_contract(), rules=rules,
    )
    assert semantics["catalog"] == {
        "catalog_schema_version": 2, "catalog_version": "public-ja-v4",
        "rule_set_version": V3_RULE, "locale": "ja", "definition_count": 296,
        "content_sha256": "c27c3341dcb0dfdf998322071641c32e9b5f22c8651427ab81a1a49fcb15cb27",
    }
    assert historical["catalog"]["catalog_version"] == "public-ja-v3"
    assert semantics["token_signature"] == historical["token_signature"]
