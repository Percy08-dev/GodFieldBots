"""Declared public role/payment matrix, independent of bounded search."""

from dataclasses import asdict

import pytest

from gf_common.card_instance import card_instance_contract
from gf_common.card_instance_rules import (
    OWNER_CONFIRMATION_SHA256,
    UTILITY_IDS,
    CardInstanceRules,
    card_instance_rules,
    parse_card_instance_rules,
    rule_signature,
)
from gf_common.domain import ContractError
from tests.card_instance_action_support import facts


def test_closed_owner_rules_and_separate_compatibility_signature():
    rules = card_instance_rules()
    assert asdict(rules) == {
        "version": "1",
        "profile_id": "owner-card-instance-candidates-20261103-v1",
        "owner_confirmation_sha256": OWNER_CONFIRMATION_SHA256,
    }
    signature = rule_signature(card_instance_contract(), rules)
    assert len(signature) == 64
    for name, value in (
        ("version", 1),
        ("version", "2"),
        ("profile_id", "unknown"),
        ("owner_confirmation_sha256", "0" * 64),
    ):
        with pytest.raises(ContractError):
            CardInstanceRules(**{**asdict(rules), name: value})
    with pytest.raises(ContractError):
        rule_signature(card_instance_contract(), asdict(rules))
    with pytest.raises(ContractError):
        rule_signature(asdict(card_instance_contract()), rules)
    assert parse_card_instance_rules(asdict(rules)) == rules
    for raw in ({}, {**asdict(rules), "eligibility": "qualified"}):
        with pytest.raises(ContractError):
            parse_card_instance_rules(raw)


@pytest.mark.parametrize(
    "state1,state2", [("unused", "unused"), ("unused", "learned"), ("learned", "learned")]
)
def test_same_definition_normal_miracles_are_distinct_physical_attack_instances(state1, state2):
    result = facts([("flame", state1), ("flame", state2)], mp=50)
    assert result.viable and result.complete and result.attack_per_round == 20
    assert result.effective_slots == (0, 1)
    assert not facts([("flame", state1), ("ice", state2)]).viable
    assert not facts([("flame", state1)], refs=(0, 0)).viable


@pytest.mark.parametrize("artifact", ["bronze-club", "abyss-dart", "ogre-s-shoes"])
def test_normal_miracle_excludes_all_effective_artifact_attack_roles(artifact):
    assert not facts([("flame", "learned"), (artifact, None)]).viable
    assert not facts([(artifact, None), ("flame", "unused")]).viable
    assert facts([("flame", "unused"), ("fireball", "learned")]).complete


def test_adjacent_costcut_role_suppression_payment_and_shared_attack_facts():
    ordinary = facts([("spiritual-staff", None), ("fireball", "unused")], mp=2)
    support = facts([("fireball", "unused"), ("spiritual-staff", None)], mp=0)
    assert ordinary.complete and ordinary.payment == 2 and ordinary.attack == 14
    assert support.complete and support.payment == 0 and support.attack == 2
    assert support.effective_slots == (0,) and support.cost_cut_pairs == ((0, 1),)
    assert "cutCost" not in support.effects
    normal = facts([("flame", "learned"), ("spiritual-staff", None)], mp=0)
    assert normal.complete and normal.attack == 10  # Bound Staff is no artifact attack.
    assert not facts(
        [("fireball", "unused"), ("boomerang", None), ("spiritual-doll", None)], mp=0
    ).viable
    assert not facts([("spiritual-doll", None)], family="utility").viable


def test_unpaid_prefix_only_last_miracle_can_be_rescued_with_a_distinct_available_cut():
    cards = [("fireball", "unused"), ("meteor", "learned"), ("spiritual-doll", None)]
    prefix = facts(cards, refs=(0,), mp=0)
    assert prefix.viable and not prefix.complete and prefix.payment == 2
    assert facts(cards, refs=(0, 2), mp=0).complete
    assert not facts(
        cards, refs=(0, 1), mp=0
    ).viable  # Earlier unpaid Fireball cannot be fixed later.
    assert facts(cards, refs=(0, 1), mp=2).viable
    assert facts(cards, refs=(0, 1, 2), mp=2).complete
    assert not facts(cards, refs=(0,), mp=0, maximum=1).viable
    assert not facts(cards, refs=(0,), mp=0, usable=(0, 1)).viable


def test_wand_order_aura_position_and_light_composition():
    assert facts([("wand-of-ignition", None), ("boomerang", None)]).attribute == "fire"
    assert facts([("boomerang", None), ("wand-of-ignition", None)]).attribute == "neutral"
    assert facts([("piece-of-brightness", None), ("fireball", "unused")]).attribute == "fire"
    before = facts([("boomerang", None), ("aura", "learned"), ("fireball", "unused")])
    after = facts([("boomerang", None), ("fireball", "unused"), ("aura", "learned")])
    assert before.complete and after.complete
    assert after.attack == before.attack + 2


@pytest.mark.parametrize("count,rounds", [(0, 2), (1, 2), (2, 4), (3, 6)])
def test_mirage_rounds_two_hit_and_attribute_reset(count, rounds):
    result = facts([("saw-boom-boom", None)] + [("mirage", "learned")] * count)
    assert result.complete and result.rounds == rounds
    assert result.attack == result.attack_per_round * rounds
    if count:
        assert result.opponent_only and result.attribute == "neutral"
    reset = facts([("wand-of-ignition", None), ("mirage", "unused"), ("wand-of-ignition", None)])
    assert reset.complete and reset.attribute == "fire"


def test_dynamic_stick_pays_miracles_first_and_includes_zero_mp():
    assert facts([("magical-stick", None)], mp=0).complete
    assert facts([("magical-stick", None)], mp=0).attack == 0
    paid = facts([("magical-stick", None), ("fireball", "unused")], mp=5)
    assert paid.complete and paid.payment == 2 and paid.spent_mp == 5 and paid.attack == 8
    cut = facts([("magical-stick", None), ("fireball", "unused"), ("spiritual-staff", None)], mp=5)
    assert cut.complete and cut.payment == 0 and cut.attack == 12


@pytest.mark.parametrize("name", sorted(UTILITY_IDS))
def test_all_scoped_utility_singletons_complete_without_empty_effect_restrictions(name):
    state = "learned" if name in ("spring", "treasure", "tone", "song", "release") else None
    assert facts([(name, state)], family="utility", mp=50).complete
    if state is not None:
        assert facts([(name, state), ("spiritual-doll", None)], family="utility", mp=0).complete
    assert not facts([(name, state), ("smile-dew", None)], family="utility").viable


def test_status_singleton_costcut_and_no_unrelated_mixture():
    assert facts([("wind", "learned"), ("spiritual-doll", None)], family="status", mp=0).complete
    assert not facts([("wind", "unused"), ("dream", "learned")], family="status").viable


def test_response_numeric_ring_exclusive_costcut_and_physical_flash():
    assert facts([("steel-armor", None), ("mars-ring", None)], family="defense").complete
    assert not facts([("super-mirror", None), ("mars-ring", None)], family="defense").viable
    assert facts([("wall", "learned"), ("spiritual-staff", None)], family="defense", mp=0).complete
    assert facts([("wall", "unused"), ("spiritual-socks", None)], family="defense", mp=0).complete
    assert not facts(
        [("wall", "learned"), ("spiritual-staff", None)], family="defense", mp=0, flash=True
    ).viable
    assert not facts(
        [("wall", "learned"), ("spiritual-staff", None)],
        refs=(0,),
        family="defense",
        mp=0,
        flash=True,
    ).viable
    assert facts([("wall", "learned")], family="defense", mp=50, flash=True).complete
    assert facts([("abyss-dart", None)] * 2, flash=True).complete  # Flash is response-specific.
