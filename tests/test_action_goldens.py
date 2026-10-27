"""Hand-authored public-subset-v1 sets, pinned to simulator 44a26e3.

Expected sets use spec09 §§9.1/9.4,10,11,14,16.3,17.1/17.2 and explicit
public-subset omissions. They are not computed by PublicActionSpace or a scorer.
"""

import pytest

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, canonical


def hand(instance):
    return {"source": "hand_artifact", "instance_id": instance}


def learned(definition):
    return {"source": "learned_miracle", "definition_id": definition}


def attack(elements, target=2):
    return {"type": "play_sequence", "ordered_elements": elements, "target": target}


def assert_each_search_matches(value, expected, **settings):
    expected = {canonical(action) for action in expected}
    for strategy in ("enumerate", "beam"):
        config = ActionConfig(strategy=strategy, beam_width=1000, max_candidates=1000,
                              targets=("opponent",), **settings)
        result = PublicCandidateGenerator(config).generate(value)
        actual = [canonical(action) for action in result.actions]
        assert len(actual) == len(set(actual))
        assert set(actual) == expected, (strategy, set(actual) - expected, expected - set(actual))


def test_duplicate_physical_normal_weapons_additional_order_and_targets(make_input):
    value = make_input(("bronze-club", "bronze-club", "warrior-s-bow"))
    sequences = [[hand(1)], [hand(2)], [hand(3)], [hand(1), hand(3)], [hand(2), hand(3)]]
    expected = [attack(sequence, target) for sequence in sequences for target in (1, 2)]
    for strategy in ("enumerate", "beam"):
        result = PublicCandidateGenerator(ActionConfig(strategy=strategy, beam_width=1000, max_candidates=1000,
                                                       max_sequence_length=2)).generate(value)
        assert {canonical(a) for a in result.actions} == {canonical(a) for a in expected}
        assert len(result.actions) == 10


@pytest.mark.parametrize("mp,sequences", [
    (13, [[hand(1)], [learned("meteor")], [hand(1), learned("meteor")]]),
    (14, [[hand(1)], [learned("meteor")], [hand(1), learned("meteor")],
          [learned("meteor"), learned("meteor")], [hand(1), learned("meteor"), learned("meteor")]]),
])
def test_learned_additional_mp_boundary_and_repeated_payment(make_input, mp, sequences):
    value = make_input(("bronze-club",), ("meteor",), mp=mp)
    assert value.catalog["meteor"]["mp_cost"] == 7
    assert_each_search_matches(value, [attack(s) for s in sequences], max_sequence_length=3)


@pytest.mark.parametrize("flash", [False, True])
def test_flash_hand_artifact_limit_and_explicit_learned_defense_omissions(make_input, flash):
    value = make_input(("steel-armor", "iron-shield"), ("wall", "turbulence"), choice="defense", flash=flash)
    expected = [{"type": "respond", "ordered_defenses": []},
                {"type": "respond_sequence", "ordered_defenses": [hand(1)]},
                {"type": "respond_sequence", "ordered_defenses": [hand(2)]}]
    if not flash:
        expected += [{"type": "respond_sequence", "ordered_defenses": [hand(1), hand(2)]},
                     {"type": "respond_sequence", "ordered_defenses": [hand(2), hand(1)]}]
    # Wall/Turbulence depend on incoming kind, absent from this DTO, so v1 omits
    # them. This is not a claim that Flash limits learned miracles in the game.
    assert_each_search_matches(value, expected, max_sequence_length=2)


@pytest.mark.parametrize("end_times,tag", [(False, "discard"), (True, "sacrifice")])
def test_end_times_replaces_discard_and_preserves_exact_instance_order(make_input, end_times, tag):
    value = make_input(("steel-armor", "sun-amulet"))
    value.observation["end_times"] = end_times
    expected = [{"type": "pray"}, {"type": tag, "instance_ids": [1]}, {"type": tag, "instance_ids": [2]},
                {"type": tag, "instance_ids": [1, 2]}, {"type": tag, "instance_ids": [2, 1]}]
    if not end_times:
        expected = [{"type": "pray"}, {"type": tag, "instance_ids": [1]}]
    assert_each_search_matches(value, expected, max_sequence_length=2)


def test_buy_decision_boolean_domain_is_exact(make_input):
    assert_each_search_matches(make_input(choice="buy_accept_reject"),
                               [{"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True}])


def test_exchange_bounded_public_allocations_and_sell_distinct_instances(make_input):
    value = make_input(("exchange", "sell"), mp=2)
    value.observation["own"].update(hp=3, money=1)
    expected = [{"type": "pray"},
                {"type": "exchange", "artifact": 1, "hp": 2, "mp": 2, "money": 2},
                {"type": "exchange", "artifact": 1, "hp": 6, "mp": 0, "money": 0},
                {"type": "sell", "artifact": 2, "sold_artifact": 1, "target": 2},
                {"type": "discard", "instance_ids": [1]}, {"type": "discard", "instance_ids": [2]}]
    assert_each_search_matches(value, expected, max_sequence_length=1, exchange_allocations=("balanced", "hp"))


def test_dream_uses_visible_definition_without_identity_reconstruction(make_input):
    value = make_input(("bronze-club", "warrior-s-bow"))
    value.observation["own"]["status"]["dream"] = True
    expected = [attack([hand(1)]), attack([hand(2)]), attack([hand(1), hand(2)])]
    assert_each_search_matches(value, expected, max_sequence_length=2)
    # The policy input contains only these perceived public definitions; no
    # alternate actual definition is supplied to construct the expected actions.
    assert set(value.catalog) == {"bronze-club", "warrior-s-bow"}
