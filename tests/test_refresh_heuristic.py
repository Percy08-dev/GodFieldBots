"""Behavioral requirements for the distinct participant-local refresh baseline."""

from dataclasses import asdict, replace

import pytest

from gf_common.card_instance import card_instance_contract
from gf_common.domain import ContractError
from tests.refresh_heuristic_support import (
    ACTIONS,
    RULES,
    Episode,
    damage,
    next_turn,
    play,
    remove,
    trade,
)


def api():
    # Lazy imports make the first RED run an assertion about the missing feature.
    import importlib.util

    assert importlib.util.find_spec("gf_models.refresh_heuristic") is not None, (
        "participant-local refresh policy has not been implemented"
    )
    from gf_common.refresh_heuristic import RefreshOptions
    from gf_models.refresh_heuristic import RefreshHeuristicPolicy

    return RefreshOptions, RefreshHeuristicPolicy


def policy(threshold=12):
    options, implementation = api()
    return implementation(ACTIONS, rules=RULES, options=options(attack_threshold=threshold))


def test_closed_options_config_and_complete_policy_identity():
    options, _ = api()
    from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
    from gf_models.refresh_heuristic_config import RefreshHeuristicConfig, parse_refresh_model

    assert asdict(options()) == {"version": "1", "attack_threshold": 12}
    assert parse_refresh_options(asdict(options(attack_threshold=9))) == options(attack_threshold=9)
    assert parse_refresh_model(asdict(RefreshHeuristicConfig())) == RefreshHeuristicConfig()
    for bad in (True, False, 0, -1, 1.5, "12"):
        with pytest.raises(ContractError):
            options(attack_threshold=bad)
        with pytest.raises(ContractError):
            RefreshHeuristicConfig(attack_threshold=bad)
    for raw in ({}, {"version": "1"}, {**asdict(options()), "extra": 1}):
        with pytest.raises(ContractError):
            parse_refresh_options(raw)
    with pytest.raises(ContractError):
        options(version="2")
    with pytest.raises(ContractError):
        RefreshHeuristicConfig(model_id="heuristic")
    with pytest.raises(ContractError):
        parse_refresh_model({"model_id": "heuristic_refresh", "version": "1"})
    label = refresh_policy_label(ACTIONS, RULES, options())
    assert (
        label.startswith("heuristic_refresh-card-instance-v2-")
        and len(label.rsplit("-", 1)[1]) == 64
    )
    assert label != refresh_policy_label(ACTIONS, RULES, options(attack_threshold=13))
    assert label != refresh_policy_label(replace(ACTIONS, max_expansions=129), RULES, options())
    with pytest.raises(ContractError):
        refresh_policy_label(
            replace(ACTIONS, action_contract=card_instance_contract()), RULES, options()
        )


@pytest.mark.parametrize(
    "attack,threshold,want",
    [
        ("gravity-mace", 12, 1),
        ("direct-smash-axe", 12, 0),
        ("gravity-mace", 11, 0),
        ("direct-smash-axe", 13, 1),
    ],
)
def test_initial_threshold_has_inclusive_configurable_boundary(attack, threshold, want):
    sample = Episode().frame([(attack, None)], [play(0), {"type": "pray"}])
    assert policy(threshold).select_frame(*sample).index == want


def test_eligible_attacks_choose_max_total_then_mp_artifacts_and_table_order():
    sample = Episode().frame(
        [
            ("mudflow", "learned"),
            ("direct-smash-axe", None),
            ("spiritual-staff", None),
            ("saw-boom-boom", None),
            ("angel-bow", None),
        ],
        [play(0), play(1), play(2), play(3, 4), play(4)],
    )
    assert policy().select_frame(*sample).index == 3  # total36, not per-round18.
    tied = Episode().frame(
        [("mudflow", "learned"), ("direct-smash-axe", None), ("spiritual-staff", None)],
        [play(0), play(1), play(2)],
    )
    assert policy().select_frame(*tied).index == 1  # MP0, first one-artifact tie.


@pytest.mark.parametrize("amount,want", [(4, 0), (3, 1), (0, 1)])
def test_history_half_boundary_uses_closed_interval_and_all_secondary_damage(amount, want):
    # A12 D4 -> B8; public A16 has predicted8 and qualifies exactly.
    # No declaration is required. Own damage and post-boundary damage are excluded.
    episode2, bot2 = Episode(), policy()
    bot2.select_frame(*episode2.frame([("direct-smash-axe", None)], [play(0)]))
    sample2 = episode2.frame(
        [("direct-smash-axe", None), ("battle-ball", None), ("smile-dew", None)],
        [play(0, 1), play(2, target=1)],
        turn=3,
        events=(damage(amount), damage(99, 1), next_turn(2), damage(99)),
    )
    assert bot2.select_frame(*sample2).index == want


def test_history_can_qualify_subthreshold_and_duplicate_partial_calls_do_not_age():
    episode, bot = Episode(), policy()
    first = episode.frame([("direct-smash-axe", None)], [play(0)])
    assert bot.select_frame(*first) == bot.select_frame(*first)
    response = episode.frame(
        [("steel-armor", None)],
        [{"type": "respond", "ordered_card_refs": []}],
        choice="defense",
        events=(damage(5),),
    )
    bot.select_frame(*response)
    bot.select_frame(*response)
    response2 = episode.frame(
        [("steel-armor", None)],
        [{"type": "respond", "ordered_card_refs": []}],
        choice="defense",
        events=(damage(7), next_turn(2)),
    )
    bot.select_frame(*response2)
    own = episode.frame([("bronze-club", None)], [play(0), {"type": "pray"}], turn=3)
    assert bot.select_frame(*own).index == 0  # B0 permits even A1.
    assert bot.select_frame(*own) == bot.select_frame(*own)


def test_history_expires_after_three_distinct_own_decisions_and_reset():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    for age in (1, 2, 3):
        sample = episode.frame(
            [("smile-dew", None)],
            [play(0, target=1)],
            turn=1 + 2 * age,
            events=(damage(12), next_turn(2)) if age == 1 else (),
        )
        bot.select_frame(*sample)
    sample = episode.frame([("bronze-club", None)], [play(0), {"type": "pray"}], turn=9)
    assert bot.select_frame(*sample).index == 1
    bot.reset(17)
    assert (
        bot.select_frame(
            *Episode("fresh").frame([("bronze-club", None)], [play(0), {"type": "pray"}])
        ).index
        == 1
    )


def test_unclosed_and_zero_attack_intervals_do_not_supply_samples():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    assert (
        bot.select_frame(
            *episode.frame(
                [("bronze-club", None)], [play(0), {"type": "pray"}], turn=3, events=(damage(12),)
            )
        ).index
        == 1
    )
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("magical-stick", None)], [play(0)], mp=0))
    assert (
        bot.select_frame(
            *episode.frame(
                [("bronze-club", None)], [play(0), {"type": "pray"}], turn=3, events=(next_turn(2),)
            )
        ).index
        == 1
    )


def test_cross_episode_truncated_frontier_and_mismatched_rules_fail_closed():
    bot, episode = policy(), Episode()
    sample = episode.frame([("direct-smash-axe", None)], [play(0)], events=(damage(0),))
    bot.select_frame(*sample)
    with pytest.raises(ContractError):
        bot.select_frame(*Episode("foreign").frame([("direct-smash-axe", None)], [play(0)]))
    with pytest.raises(ContractError):
        bot.select_frame(*Episode().frame([("direct-smash-axe", None)], [play(0)]))
    frame, view, generated = sample
    with pytest.raises(ContractError):
        policy().select_frame(
            frame, view, replace(generated, report=replace(generated.report, config_hash="f" * 64))
        )


def test_retention_preserves_multi_card_witness_before_acquisition_count():
    sample = Episode().frame(
        [("bronze-club", None), ("battle-ball", None), ("smile-dew", None), ("heart-dew", None)],
        [play(0, 1), remove(0, 2, kind="sacrifice"), remove(2, 3, kind="sacrifice")],
        end_times=True,
    )
    assert policy().select_frame(*sample).index == 2


def test_renewal_uses_capacity_and_retains_miracles_with_costcut():
    spec = [("fireball", "learned"), ("tone", "learned"), ("spiritual-staff", None)]
    spec += [("steel-armor", None)] * 15
    sample = Episode().frame(spec, [play(0), play(1, target=1), play(1, 2, target=1)], mp=2)
    assert policy().select_frame(*sample).index == 2  # CostCut consumes staff; tone stays.


def test_full18_discard_preparation_preserves_attack_and_minimizes_removals():
    spec = [("bronze-club", None), ("battle-ball", None)] + [("steel-armor", None)] * 16
    sample = Episode().frame(spec, [play(0, 1), remove(0), remove(2, 3), remove(2)])
    assert policy().select_frame(*sample).index == 3


@pytest.mark.parametrize("carrier", ["goddess-s-soap", "nocturnal-broom", "string-of-fate"])
def test_uncertain_automatic_removals_receive_zero_retention(carrier):
    candidates = [play(0), play(1, target=1), {"type": "pray"}]
    if carrier == "string-of-fate":
        candidates[1] = trade("invoke_phenomenon", 1)
    sample = Episode().frame([("bronze-club", None), (carrier, None)], candidates)
    assert policy().select_frame(*sample).index == 2


def test_only_unchanged_exchange_is_renewal():
    sample = Episode().frame(
        [("bronze-club", None), ("exchange", None)],
        [
            play(0),
            trade("exchange", 1, allocation=(41, 49, 20)),
            trade("exchange", 1, allocation=(40, 50, 20)),
        ],
    )
    assert policy().select_frame(*sample).index == 2


@pytest.mark.parametrize("money,mp,want", [(0, 0, 0), (0, 1, 1), (1, 0, 1)])
def test_self_sell_static_price_guard(money, mp, want):
    sample = Episode().frame(
        [("bronze-club", None), ("sell", None)],
        [play(0), trade("sell", 1, sold=0)],
        money=money,
        mp=mp,
    )
    assert policy().select_frame(*sample).index == want


def test_zero_offense_is_last_attack_fallback_and_own_offense_is_not_renewal():
    sample = Episode().frame(
        [("bronze-club", None), ("magical-stick", None)],
        [play(0, target=1), play(0), play(1)],
        mp=0,
    )
    assert policy().select_frame(*sample).index == 2


def test_defense_and_buy_reuse_canonical_scoring_with_new_label():
    from gf_models.card_instance_baselines import CardInstanceBaseline

    for choice, candidates, spec in (
        (
            "defense",
            [
                {"type": "respond", "ordered_card_refs": []},
                lambda f: {"type": "respond", "ordered_card_refs": [f.references[0]]},
            ],
            [("steel-armor", None)],
        ),
        (
            "buy_accept_reject",
            [{"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True}],
            [],
        ),
    ):
        sample = Episode().frame(spec, candidates, choice=choice)
        old = CardInstanceBaseline("heuristic", ACTIONS, rules=RULES).select_frame(*sample)
        bot = policy()
        selected = bot.select_frame(*sample)
        assert selected.index == old.index and selected.policy_version == bot.version == bot.label
        assert selected.policy_version != old.policy_version


def test_equal_attack_and_payment_prefer_fewer_consumed_artifacts():
    sample = Episode().frame(
        [("crossbow", None), ("fireball", "learned"), ("ice", "learned")],
        [play(0, 1), play(2)],
        mp=2,
    )
    assert policy(4).select_frame(*sample).index == 1  # A4, MP2; retained learned miracle wins.


@pytest.mark.parametrize("attack,want", [("abyss-dart", 0), ("battle-ball", 1)])
def test_two_sample_mean_uses_exact_half_boundary(attack, want):
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    bot.select_frame(
        *episode.frame(
            [("direct-smash-axe", None)], [play(0)], turn=3, events=(damage(10), next_turn(2))
        )
    )
    sample = episode.frame(
        [(attack, None), ("smile-dew", None)],
        [play(0), play(1, target=1)],
        turn=5,
        events=(damage(9), next_turn(4)),
    )
    # B2 and B3: A5 predicts2.5 (eligible), A4 predicts1.5 (ineligible).
    assert bot.select_frame(*sample).index == want


def test_sample_usable_next_three_own_decisions_but_not_fourth():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    for age in (1, 2, 3, 4):
        sample = episode.frame(
            [("bronze-club", None)],
            [play(0), {"type": "pray"}],
            turn=2 * age + 1,
            events=(damage(12), next_turn(2)) if age == 1 else (),
        )
        assert bot.select_frame(*sample).index == (0 if age <= 3 else 1)


def test_buy_and_repeated_same_turn_frames_do_not_age_history():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    for turn in (3, 5, 7):
        bot.select_frame(
            *episode.frame(
                [("smile-dew", None)],
                [play(0, target=1)],
                turn=turn,
                events=(damage(12), next_turn(2)) if turn == 3 else (),
            )
        )
    for _ in range(4):
        bot.select_frame(
            *episode.frame(
                [], [{"type": "buy_decision", "accept": True}], choice="buy_accept_reject", turn=7
            )
        )
        bot.select_frame(*episode.frame([("smile-dew", None)], [play(0, target=1)], turn=7))
    sample = episode.frame([("bronze-club", None)], [play(0), {"type": "pray"}], turn=7)
    assert bot.select_frame(*sample).index == 0


def test_closed_window_without_damage_is_zero_not_missing():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("direct-smash-axe", None)], [play(0)]))
    sample = episode.frame(
        [("direct-smash-axe", None), ("smile-dew", None)],
        [play(0), play(1, target=1)],
        turn=3,
        events=(next_turn(2),),
    )
    assert bot.select_frame(*sample).index == 1  # B12 makes A12 ineligible.


def test_full_prayer_is_positive_renewal_with_unknown_retention():
    spec = [("bronze-club", None)] + [("steel-armor", None)] * 17
    sample = Episode().frame(spec, [play(0), remove(1), {"type": "pray"}])
    assert policy().select_frame(*sample).index == 2


def test_renewal_caps_opportunities_by_known_freed_slots():
    spec = [
        ("bronze-club", None),
        ("tone", "learned"),
        ("spiritual-doll", None),
        ("smile-dew", None),
    ] + [("steel-armor", None)] * 14
    sample = Episode().frame(spec, [play(0), play(1, 2, target=1), play(3, target=1)], mp=0)
    # Tone+CostCut selects2 but frees1. Both renew1 and preserve A1; the single slot wins.
    assert policy().select_frame(*sample).index == 2


@pytest.mark.parametrize("kind,target,want", [("sell", 1, 1), ("sell", 2, 2), ("buy", 2, 1)])
def test_trade_retention_uses_only_known_removed_instances(kind, target, want):
    spec = [("bronze-club", None), (kind, None)]
    operation = (
        trade(kind, 1, sold=0, target=target) if kind == "sell" else trade(kind, 1, target=target)
    )
    sample = Episode().frame(
        spec, [play(0), operation, remove(0, 1, kind="sacrifice")], end_times=True
    )
    # Self-Sell and Buy retain the weapon. Away-Sell loses it; Sacrifice renews2.
    assert policy().select_frame(*sample).index == want


def test_own_status_and_opponent_utility_are_not_renewal():
    sample = Episode().frame(
        [("bronze-club", None), ("fog", "learned"), ("smile-dew", None)],
        [play(1, target=1), play(2, target=2), play(0)],
    )
    assert policy().select_frame(*sample).index == 2


def test_no_offense_or_renewal_composes_old_fallback_identity():
    from gf_models.card_instance_baselines import CardInstanceBaseline

    sample = Episode().frame([("fog", "learned")], [play(0, target=1), play(0)])
    old = CardInstanceBaseline("heuristic", ACTIONS, rules=RULES).select_frame(*sample)
    bot = policy()
    selected = bot.select_frame(*sample)
    assert selected.index == old.index and selected.policy_version == bot.label


def test_own_perceived_definitions_are_sufficient_without_catalog_expansion(monkeypatch):
    import gf_common.card_instance_vocab as vocabulary

    sample = Episode().frame([("direct-smash-axe", None)], [play(0), {"type": "pray"}])
    monkeypatch.setattr(vocabulary, "DEFINITION_SHA256", {})
    assert set(sample[0].raw["catalog"]) == {"direct-smash-axe"}
    assert policy().select_frame(*sample).index == 0


def test_same_episode_divergent_prefix_and_changed_duplicate_candidates_are_refused():
    bot, episode = policy(), Episode()
    first = episode.frame([("direct-smash-axe", None)], [play(0)], events=(damage(1),))
    bot.select_frame(*first)
    foreign = Episode().frame([("direct-smash-axe", None)], [play(0)], events=(damage(2),))
    with pytest.raises(ContractError):
        bot.select_frame(*foreign)
    from gf_common.diverse_actions import GameGeneratedCandidatesV3
    from gf_common.structured import CandidateTable

    frame, view, generated = first
    changed = CandidateTable.from_actions(
        frame,
        [{"type": "pray"}],
        action_contract=ACTIONS.action_contract,
        purpose="qualified_game",
        synthetic_capacity=False,
    )
    changed_generated = GameGeneratedCandidatesV3(
        changed, replace(generated.report, table_hash=changed.table_hash, selected_elements=0)
    )
    with pytest.raises(ContractError):
        bot.select_frame(frame, view, changed_generated)


def test_options_and_model_imports_do_not_load_numeric_or_learning_libraries():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import gf_models.refresh_heuristic_config; "
                "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_renewal_retention_is_capped_before_acquisition_ranking():
    episode, bot = Episode(), policy()
    bot.select_frame(*episode.frame([("god-sword", None)], [play(0)]))
    sample = episode.frame(
        [
            ("spear-in-fine-view", None),
            ("petit-saturn", None),
            ("steel-armor", None),
            ("steel-armor", None),
        ],
        [play(0), play(1), remove(0, kind="sacrifice"), remove(1, 2, 3, kind="sacrifice")],
        turn=3,
        events=(next_turn(2),),
        end_times=True,
    )
    # Both retained13/20 cap to12; three renewals beat one.
    assert bot.select_frame(*sample).index == 3


def test_subset_retention_matches_independent_multi_card_containment():
    from gf_models.refresh_heuristic import _Fact, _retention

    facts = (
        _Fact(0b00101, 2, 2, attack=7, offense=True),
        _Fact(0b01000, 1, 1, attack=9, offense=True),
        _Fact(0b10100, 2, 2, attack=13, offense=True),
        _Fact(0b00010, 1, 1, attack=90, offense=False),
    )
    table = _retention(5, facts)
    assert [int(table[mask]) for mask in (0, 1, 2, 4, 5, 8, 13, 20, 31)] == [
        0,
        0,
        0,
        0,
        7,
        9,
        9,
        13,
        13,
    ]


def test_subset_retention_full18_has_exact_int64_principal_capacity():
    from gf_models.refresh_heuristic import _Fact, _retention

    table = _retention(18, (_Fact((1 << 18) - 1, 18, 18, attack=2**63 - 1, offense=True),))
    assert table.nbytes == 2 * 1024**2 and table.dtype.name == "int64"
    assert int(table[-1]) == 2**63 - 1 and int(table[-2]) == 0
