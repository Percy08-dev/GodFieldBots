from dataclasses import replace

import pytest

from gf_common.actions import (
    GENERATOR_VERSION,
    PublicActionSpace,
    PublicCandidateGenerator,
    content_score,
)
from gf_common.domain import ActionConfig, ContractError, EmptyCandidates, canonical
from gf_models.baselines import HeuristicPolicy, RandomPolicy


def generate(value, **kwargs):
    return PublicCandidateGenerator(ActionConfig(**kwargs)).generate(value)


def sequence_names(value, action):
    hand = {c["instance_id"]: c["definition_id"] for c in value.observation["own"]["hand"]}
    return [hand[e["instance_id"]] if e["source"] == "hand_artifact" else e["definition_id"] for e in action.get("ordered_elements", [])]


def test_enum_wide_beam_agree_preserve_order_and_identity(make_input):
    value = make_input(("bronze-club", "bronze-club", "warrior-s-bow"))
    reference = generate(value, strategy="enumerate", max_sequence_length=3)
    beam = generate(value, beam_width=1000, max_sequence_length=3)
    assert reference.actions == beam.actions
    singles = [a for a in reference.actions if a["type"] == "play_sequence" and len(a["ordered_elements"]) == 1]
    assert any(a["ordered_elements"][0].get("instance_id") == 1 for a in singles)
    assert any(a["ordered_elements"][0].get("instance_id") == 2 for a in singles)
    assert all(sequence_names(value, a) != ["warrior-s-bow", "bronze-club"] for a in reference.actions)
    assert any(sequence_names(value, a) == ["bronze-club", "warrior-s-bow"] for a in reference.actions)
    for a in reference.actions:
        ids = [e["instance_id"] for e in a.get("ordered_elements", []) if "instance_id" in e]
        assert len(ids) == len(set(ids))


def test_aura_pinned_catalog_order_repeat_and_payment(make_input):
    value = make_input(("bronze-club", "warrior-s-bow"), ("aura",), mp=12)
    assert value.catalog["aura"]["ability"] == "doubleAtk"
    result = generate(value, strategy="enumerate", max_sequence_length=3, targets=("opponent",))
    names = [sequence_names(value, a) for a in result.actions]
    assert ["bronze-club", "aura"] in names
    assert ["bronze-club", "aura", "aura"] in names
    assert ["bronze-club", "aura", "warrior-s-bow"] in names
    assert ["aura", "bronze-club"] not in names
    low = make_input(("bronze-club",), ("aura",), mp=6)
    assert ["bronze-club", "aura", "aura"] not in [sequence_names(low, a) for a in generate(low).actions]
    assert any(content_score(value, a) > 4 for a in result.actions if sequence_names(value, a) == ["bronze-club", "aura", "warrior-s-bow"])


def test_learned_additional_can_repeat_but_hand_cannot(make_input):
    value = make_input(("fireball",), ("meteor",), mp=14)
    result = generate(value, strategy="enumerate", max_sequence_length=3)
    names = [sequence_names(value, a) for a in result.actions]
    assert ["meteor", "meteor"] in names
    assert ["meteor", "meteor", "meteor"] not in names
    assert ["fireball", "fireball"] not in names


def test_basics_reserved_and_budget_errors(make_input):
    value = make_input(("steel-armor",), choice="defense")
    result = generate(value, max_candidates=1, max_expansions=2)
    assert result.actions == ({"type": "respond", "ordered_defenses": []},)
    assert "max_candidates" in result.cutoff_reasons
    buy = make_input(choice="buy_accept_reject")
    with pytest.raises(ContractError):
        generate(buy, max_candidates=1)
    assert {a["accept"] for a in generate(buy, max_candidates=2, max_expansions=2).actions} == {True, False}


def test_budget_counts_prunes_unbounded_length_is_finite(make_input):
    value = make_input(("bronze-club",), ("meteor",), mp=10000)
    result = generate(value, max_expansions=7, max_sequence_length=0)
    assert result.attempts == 7
    assert result.public_prunes > 0
    assert "max_expansions" in result.cutoff_reasons
    assert len(result.actions) > 0


def test_public_prayer_discard_trade_and_no_empty_fallback(make_input):
    value = make_input(("steel-armor", "exchange", "sell", "buy"))
    result = generate(value, max_sequence_length=1)
    assert any(a["type"] == "pray" for a in result.actions)
    assert any(a["type"] == "discard" for a in result.actions)
    assert any(a["type"] == "exchange" for a in result.actions)
    assert any(a["type"] == "sell" for a in result.actions)
    assert any(a["type"] == "buy" for a in result.actions)
    for a in result.actions:
        if a["type"] == "exchange":
            assert sum(a[k] for k in ("hp", "mp", "money")) == 110
            assert all(0 <= a[k] <= 99 for k in ("hp", "mp", "money"))
    assert not any(a["type"] == "pray" for a in generate(make_input()).actions)
    with pytest.raises(EmptyCandidates):
        generate(replace(make_input(("magical-stick",)), action_options=None), use_action_options=False)


def test_empty_prefix_extends_and_cutoff_reasons(make_input):
    value = make_input()
    space = PublicActionSpace(value, ActionConfig())
    assert list(space.completions("offense", ())) == []
    assert list(space.children("offense", ())) == [(0,)]
    result = generate(value, max_sequence_length=1)
    assert "max_sequence_length" in result.cutoff_reasons
    result = generate(make_input(choice="buy_accept_reject"))
    assert result.cutoff_reasons == ("exhausted",)
    assert not result.cutoff


@pytest.mark.parametrize("setting", [{"max_expansions": True}, {"scorer": "hidden"}, {"targets": ("unknown",)}, {"beam_width": 0}])
def test_programmatic_settings_fail_closed(setting):
    with pytest.raises(ContractError):
        PublicCandidateGenerator(ActionConfig(**setting))


def test_policy_versions_random_logprob_and_content_scoring(make_input):
    value = make_input(("bronze-club", "god-sword"))
    candidates = generate(value)
    assert candidates.generator_version == GENERATOR_VERSION
    assert len({canonical(a) for a in candidates.actions}) == len(candidates.actions)
    policy = HeuristicPolicy()
    policy.reset(123)
    chosen = policy.select(value, candidates)
    assert sequence_names(value, candidates.actions[chosen.index]) == ["god-sword"]
    reordered = replace(candidates, actions=tuple(reversed(candidates.actions)))
    assert sequence_names(value, reordered.actions[policy.select(value, reordered).index]) == ["god-sword"]
    policy = RandomPolicy()
    policy.reset(3)
    a = policy.select(value, candidates)
    policy.reset(3)
    assert policy.select(value, candidates) == a
    assert a.logprob < 0
