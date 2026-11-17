"""Owner response priority and rejection of superseded rule identities."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.card_instance import card_instance_contract, corrected_card_instance_contract
from gf_common.card_instance_rules import (
    CardInstanceRules,
    DefenseContext,
    card_instance_rules,
    defense_context,
    parse_card_instance_rules,
    rule_signature,
)
from gf_common.domain import ContractError
from tests.card_instance_action_support import facts, inputs, run
from tests.test_card_instance_structured import frame_for

LEGACY = {
    "version": "1",
    "profile_id": "owner-card-instance-candidates-20261103-v1",
    "owner_confirmation_sha256": "3f3090f521041293bcb6de2a3cbac7e74ffc635b4acc0f7f02c957bbcf0e44ee",
}
MIRACLE = (
    {"type": "action_accepted", "player": 2},
    {"type": "artifact_used", "player": 2, "definition_id": "mudflow"},
    {"type": "miracle_used", "player": 2, "definition_id": "mudflow", "mp_spent": 6},
    {"type": "attack_declared", "attacker": 2, "target": 1, "attack": 12},
)
WEAPON = (
    {"type": "action_accepted", "player": 2},
    {"type": "artifact_used", "player": 2, "definition_id": "bronze-club"},
    {"type": "attack_declared", "attacker": 2, "target": 1, "attack": 12},
)


def response(events=MIRACLE):
    value = inputs([("angel-armor", None), ("rainbow-curtain", None)], choice="defense", mp=10)
    value.observation["state_revision"] = 2
    value.observation["own"]["hp"] = 40
    value.action_options["state_revision"] = 2
    value.action_options["action_options"]["state_revision"] = 2
    return replace(value, events=events)


def test_rules_v1_descriptor_and_old_game_binding_are_explicitly_rejected():
    from gf_common.card_instance_runtime import game_binding

    with pytest.raises(ContractError):
        parse_card_instance_rules(LEGACY)
    with pytest.raises(ContractError):
        game_binding(corrected_card_instance_contract(), "3f55f044225fb20c63d683145cdceb397c36f0b7161c7b16793f72210db3e86d")


def test_current_rules_bind_supplemental_authority_without_layout_change():
    current = card_instance_rules()
    assert asdict(current) == {
        "version": "2", "profile_id": "owner-card-instance-candidates-20261117-v2",
        "owner_confirmation_sha256": "bdd1792cde50eae2338b5f068436e40ca87cfbdf0c276ad096651bc0ff27375c",
    }
    assert parse_card_instance_rules(asdict(current)) == current
    with pytest.raises(ContractError):
        CardInstanceRules(**{**LEGACY, "version": "2"})


@pytest.mark.parametrize("name", ["angel-armor", "sky-armor", "moonlight-armor"])
def test_positive_defense_special_prioritizes_miracle_but_keeps_weapon_numeric_role(name):
    cards = [(name, None), ("steel-armor", None)]
    assert not facts(cards, family="defense", rules=card_instance_rules(), context=DefenseContext("miracle")).viable
    assert facts(cards, family="defense", rules=card_instance_rules(), context=DefenseContext("weapon")).complete
    unknown = facts(cards, family="defense", rules=card_instance_rules(), context=DefenseContext())
    assert unknown.viable and not unknown.complete and unknown.context_unknown
    assert facts(cards, refs=(0,), family="defense", rules=card_instance_rules(), context=DefenseContext()).complete


def test_rainbow_is_first_physical_card_and_costcut_support_does_not_hide_order():
    rules = card_instance_rules()
    context = DefenseContext("weapon")
    assert facts([("rainbow-curtain", None), ("steel-armor", None)], family="defense", rules=rules, context=context).complete
    assert not facts([("steel-armor", None), ("rainbow-curtain", None)], family="defense", rules=rules, context=context).viable
    assert facts([("wall", "learned"), ("spiritual-socks", None)], mp=0, family="defense", rules=rules, context=context).complete
    assert not facts([("wall", "learned"), ("spiritual-socks", None)], mp=0, family="defense", flash=True, rules=rules, context=context).viable
    assert not facts([("wall", "learned"), ("spiritual-socks", None), ("rainbow-curtain", None)], mp=0, family="defense", rules=rules, context=context).viable


def test_recorded_mudflow_generator_retains_atoms_excludes_both_pair_orders():
    result, actions, _ = run(response())
    assert result.report.rule_signature == rule_signature(card_instance_contract(), card_instance_rules())
    assert [len(row["ordered_card_refs"]) for row in actions] == [0, 1, 1]


@pytest.mark.parametrize("events,kind", [(MIRACLE, "miracle"), (WEAPON, "weapon"), ((), "unknown")])
def test_supported_public_causes(events, kind):
    value = response(events)
    if kind == "weapon":
        value = replace(value, catalog={**value.catalog, "bronze-club": inputs([("bronze-club", None)]).catalog["bronze-club"]})
        value.observation["own"]["cards"].append({"kind": "artifact", "card_ref": "c1_" + "3".zfill(32), "definition_id": "bronze-club"})
    frame, bank, _ = frame_for(value)
    assert defense_context(frame, bank.view()).incoming == kind


@pytest.mark.parametrize("events", [
    MIRACLE[:-1], MIRACLE + ({"type": "safe", "player": 1},),
    MIRACLE + WEAPON[-1:],
    (WEAPON[0], MIRACLE[1], MIRACLE[2], WEAPON[1], WEAPON[-1]),
    (WEAPON[0], {"type": "guardian_acted", "player": 2, "definition_id": "bronze-club"}, WEAPON[-1]),
    (WEAPON[0], {"type": "artifact_used", "player": 2, "definition_id": "smile-dew"}, WEAPON[-1]),
    (WEAPON[0], WEAPON[1], {**WEAPON[-1], "target": 2}),
    (WEAPON[0], WEAPON[1], {**WEAPON[-1], "attacker": 1}),
    (WEAPON[0], WEAPON[1], {"type": "miracle_used", "player": 2, "definition_id": "mudflow", "mp_spent": 6}, WEAPON[-1]),
])
def test_unsupported_mixed_stale_targeted_and_reflection_histories_are_unknown(events):
    frame, bank, _ = frame_for(response(events))
    assert defense_context(frame, bank.view()) == DefenseContext()


def test_context_uses_verified_own_prefix_ignores_future_and_opponent_learned_list():
    from gf_common.structured import event_delta
    from gf_common.structured_layout import CARD_INSTANCE_LAYOUT

    frame, bank, _ = frame_for(response())
    bank.append(event_delta(1, bank.episode, bank.boundary.end, bank.boundary.prefix_hash,
        [{"type": "safe", "player": 1}], layout=CARD_INSTANCE_LAYOUT))
    assert defense_context(frame, bank.view()).incoming == "miracle"
    _, other, _ = frame_for(response(), episode="foreign")
    with pytest.raises(ContractError):
        defense_context(frame, other.view())
    value = response(())
    value.observation["opponent"]["learned_miracles"] = ["mudflow"]
    frame, bank, _ = frame_for(value)
    assert defense_context(frame, bank.view()) == DefenseContext()


def test_saved_actor_frame_generator_baseline_and_refresh_share_priority():
    """The actor's saved rev2 public prefix is sufficient; no rejected reply enters input."""
    import json
    import random

    from gf_common.card_instance import ActionConfigV3
    from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
    from gf_common.domain import DecisionInput
    from gf_common.refresh_heuristic import RefreshOptions
    from gf_common.structured import DecisionFrame, EventBank, event_delta
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.card_instance_baselines import CardInstanceBaseline
    from gf_models.refresh_heuristic import RefreshHeuristicPolicy

    raw = json.loads((Path(__file__).parent / "fixtures/defense_priority_recorded_own.json").read_text())
    value = DecisionInput(raw["observation"], tuple(raw["events"]), raw["history_end"],
        tuple(raw["typed_metadata"]), raw["catalog"], raw["action_options"])
    bank = EventBank(1, "saved-actor-defense", layout=layout)
    bank.append(event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, value.events, layout=layout))
    frame = DecisionFrame.from_input(value, bank, 0, layout=layout)
    assert frame.revision == 2 and defense_context(frame, bank.view()).incoming == "miracle"
    actions = ActionConfigV3(action_contract=corrected_card_instance_contract())
    rules = card_instance_rules()
    generated = PublicCandidateGeneratorV3(actions, rules=rules, clock=lambda: 0).generate_game(
        frame, bank.view(), random.Random(17), NumericCosts())
    own = value.observation["own"]["cards"]
    ids = {row["definition_id"]: row["card_ref"] for row in own}
    choices = [generated.table.to_action(index, frame) for index in range(len(generated.table.kinds))]
    assert {ids["angel-armor"]} in [set(row["ordered_card_refs"]) for row in choices]
    assert {ids["rainbow-curtain"]} in [set(row["ordered_card_refs"]) for row in choices]
    assert all(not {ids["angel-armor"], ids["rainbow-curtain"]} <= set(row["ordered_card_refs"]) for row in choices)
    for policy in (CardInstanceBaseline("heuristic", actions, rules=rules),
                   RefreshHeuristicPolicy(actions, rules=rules, options=RefreshOptions())):
        selected = policy.select_frame(frame, bank.view(), generated)
        assert generated.table.to_action(selected.index, frame)["ordered_card_refs"] == [ids["angel-armor"]]


def test_weapon_public_identity_without_own_definition_is_unknown():
    frame, bank, _ = frame_for(response(WEAPON))
    assert defense_context(frame, bank.view()) == DefenseContext()


@pytest.mark.parametrize("name", ["angel-armor", "sky-armor", "moonlight-armor"])
def test_generator_keeps_numeric_pairs_only_when_own_public_weapon_overlap_proves_cause(name):
    value = inputs([(name, None), ("steel-armor", None), ("bronze-club", None)], choice="defense")
    _, choices, frame = run(replace(value, events=WEAPON))
    pairs = {(frame.references[0], frame.references[1]), (frame.references[1], frame.references[0])}
    assert pairs <= {tuple(row["ordered_card_refs"]) for row in choices}
    _, unknown, _ = run(value)
    assert all(len(row["ordered_card_refs"]) < 2 for row in unknown)


def test_fog_and_foreign_player_view_cannot_supply_known_context():
    from gf_common.structured import EventBank

    value = response()
    value.observation["own"]["status"]["fog"] = True
    value.observation["opponent"] = {key: value.observation["opponent"][key]
        for key in ("player_id", "learned_miracles", "ascended")}
    frame, bank, _ = frame_for(value)
    assert defense_context(frame, bank.view()) == DefenseContext()
    with pytest.raises(ContractError, match="foreign"):
        defense_context(frame, EventBank(2, frame.episode, layout=frame.layout).view())


def test_current_report_rejects_old_signature_at_nested_search_boundary():
    result, _, _ = run(response())
    with pytest.raises(ContractError, match="rule signature"):
        replace(result.report, rule_signature="1480f832b98c0b78b894a823681848b1dc11093b3a6a2aa462dd2069ed579a36")


def test_context_uses_complete_prefix_when_current_interval_is_empty():
    from gf_common.structured import DecisionFrame

    frame, bank, _ = frame_for(response())
    value = response()
    value.observation["state_revision"] = 3
    value.action_options["state_revision"] = 3
    value.action_options["action_options"]["state_revision"] = 3
    next_frame = DecisionFrame.from_input(value, bank, frame.boundary.end, layout=frame.layout)
    assert next_frame.interval_start == next_frame.boundary.end
    assert defense_context(next_frame, bank.view()).incoming == "miracle"
