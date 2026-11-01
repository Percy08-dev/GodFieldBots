"""Independent small public domains and bounded v4 discovery evidence."""
from __future__ import annotations

import itertools
import random
from dataclasses import asdict, replace

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV2, estimate_costs
from gf_common.domain import ContractError, canonical
from gf_common.structured import ActionConfigV2
from tests.test_structured_records import frame_for


def generate(value, **settings):
    frame, bank = frame_for(value)
    result = PublicCandidateGeneratorV2(ActionConfigV2(**settings, action_contract=engineering_contract()), clock=lambda: 0.).generate(
        frame, bank.view(), random.Random(17), NumericCosts())
    result.table.validate(frame)
    actions = [result.table.to_action(index, frame) for index in range(len(result.table.kinds))]
    return result, actions


def test_two_ordered_additional_instances_match_independent_reference(make_input):
    value = make_input(hand=("abyss-dart", "abyss-dart"))
    # Independent itertools oracle: both additional instances may stand alone
    # or occur in either order, each with both configured target proposals.
    expected = set()
    for length in (1, 2):
        for sequence in itertools.permutations((1, 2), length):
            for target in (2, 1):
                expected.add(canonical({"type": "play_sequence", "target": target,
                    "ordered_elements": [{"source": "hand_artifact", "instance_id": item} for item in sequence]}))
    result, actions = generate(value, max_sequence_length=2)
    assert {canonical(action) for action in actions} == expected
    assert result.report.precount.mode == "exact"
    assert result.report.precount.complete == 8
    assert result.report.traversal_complete and result.report.presentation_complete


def test_normal_base_only_first_and_normal_miracle_cannot_follow(make_input):
    result, actions = generate(make_input(hand=("bronze-club", "flame", "fireball")), max_sequence_length=3)
    sequences = {tuple(item["instance_id"] for item in action["ordered_elements"])
                 for action in actions if action["type"] == "play_sequence"}
    assert sequences == {(1,), (2,), (3,), (1, 3), (2, 3)}
    assert result.report.precount.complete >= len(actions)


def test_zero_stick_support_completion_and_payment(make_input):
    _, actions = generate(make_input(hand=("magical-stick",), learned=("aura", "mirage"), mp=6), max_sequence_length=2)
    plays = [action for action in actions if action["type"] == "play_sequence"]
    assert any(len(action["ordered_elements"]) == 2 and action["ordered_elements"][1].get("definition_id") == "aura" for action in plays)
    assert any(len(action["ordered_elements"]) == 2 and action["ordered_elements"][1].get("definition_id") == "mirage" and action["target"] == 2 for action in plays)
    assert not any(len(action["ordered_elements"]) == 1 and action["ordered_elements"][0].get("definition_id") in ("aura", "mirage") for action in plays)
    _, zero = generate(make_input(hand=("magical-stick",), mp=0), max_sequence_length=1)
    assert len(zero) == 2


@pytest.mark.parametrize("length", [18, 128])
def test_unique_gameplay_reaches_eighteen_refs_even_with_structural_capacity_128(make_input, length):
    result, actions = generate(make_input(hand=("abyss-dart",) * 18, mp=0),
                               max_sequence_length=length, count_states=2,
                               max_expansions=10000, max_candidates=80)
    assert result.report.precount.mode == "upper_bound"
    assert max(len(action.get("ordered_elements", [])) for action in actions) == 18
    assert result.report.attempts <= 10000
    assert not result.report.traversal_complete
    assert len(actions) <= 80


def test_mandatory_target_variants_cannot_be_truncated(make_input):
    with pytest.raises(ContractError, match="mandatory"):
        generate(make_input(hand=("wand-of-ignition",)), max_candidates=1)


def test_unique_learned_additional_and_aura_match_handwritten_oracle(make_input):
    _, actions = generate(make_input(hand=("bronze-club",), learned=("fireball",), mp=4), max_sequence_length=3)
    actual = {tuple((item.get("instance_id"), item.get("definition_id")) for item in action["ordered_elements"])
              for action in actions if action["type"] == "play_sequence"}
    expected = set()
    for length in range(1, 4):
        for symbols in itertools.product(("base", "extra"), repeat=length):
            if symbols.count("base") > 1 or "base" in symbols[1:] or symbols.count("extra") > 1:
                continue
            expected.add(tuple((1, None) if symbol == "base" else (None, "fireball") for symbol in symbols))
    assert actual == expected
    _, aura = generate(make_input(hand=("bronze-club",), learned=("aura",), mp=12), max_sequence_length=4)
    assert {len(action["ordered_elements"]) for action in aura if action["type"] == "play_sequence"} == {1, 2}


def test_unique_paid_mirage_does_not_expand_to_structural_128_and_attack_twice_exclusion(make_input):
    result, actions = generate(make_input(hand=("magical-stick",), learned=("mirage",), mp=635),
                               max_sequence_length=128, count_states=1, max_expansions=10000, max_candidates=30)
    longest = max((action for action in actions if action["type"] == "play_sequence"),
                  key=lambda action: len(action["ordered_elements"]))
    assert len(longest["ordered_elements"]) == 2 and longest["target"] == 2
    assert result.report.attempts <= 10000
    _, excluded = generate(make_input(hand=("saw-boom-boom",), learned=("mirage",), mp=100), max_sequence_length=3)
    assert not any(any(item.get("definition_id") == "mirage" for item in action.get("ordered_elements", [])) for action in excluded)


def test_defense_numeric_ring_filter_order_matches_independent_permutations(make_input):
    _, actions = generate(make_input(hand=("steel-armor", "mars-ring", "rainbow-curtain"), choice="defense"), max_sequence_length=3)
    actual = {tuple(item["instance_id"] for item in action["ordered_defenses"])
              for action in actions if action["type"] == "respond_sequence"}
    expected = {sequence for length in range(1, 4) for sequence in itertools.permutations((1, 2, 3), length)}
    assert actual == expected
    assert {action["type"] for action in actions} == {"respond", "respond_sequence"}


def test_defense_exclusive_cutcost_flash_and_learned_payment(make_input):
    _, actions = generate(make_input(hand=("spiritual-hood", "steel-armor", "super-mirror", "bouncing-sword"),
                                      learned=("wall", "turbulence"), choice="defense", mp=6), max_sequence_length=3)
    sequences = [action["ordered_defenses"] for action in actions if action["type"] == "respond_sequence"]
    assert any(len(sequence) == 2 and {item.get("instance_id") for item in sequence} == {1, 2} for sequence in sequences)
    assert all(len(sequence) == 1 for sequence in sequences if any(item.get("instance_id") in (3, 4) or item.get("definition_id") in ("wall", "turbulence") for item in sequence))
    _, flash = generate(make_input(hand=("steel-armor", "mars-ring"), choice="defense", flash=True), max_sequence_length=3)
    assert all(len(action.get("ordered_defenses", [])) <= 1 for action in flash)
    _, unpaid = generate(make_input(hand=("steel-armor",), learned=("wall",), choice="defense", mp=0), max_sequence_length=2)
    assert not any(item.get("definition_id") == "wall" for action in unpaid for item in action.get("ordered_defenses", []))


def test_protected_discard_and_sacrifice_are_separate(make_input):
    value = make_input(hand=("sun-amulet", "dangerous-mortar", "steel-armor", "bronze-club"))
    _, normal = generate(value, max_sequence_length=4)
    assert [action["instance_ids"] for action in normal if action["type"] == "discard"] == [[3]]
    obs = value.to_dict()["observation"]
    obs["end_times"] = True
    _, end_times = generate(replace(value, observation=obs), max_sequence_length=4)
    actual = {tuple(action["instance_ids"]) for action in end_times if action["type"] == "sacrifice"}
    expected = {subset for length in range(1, 4) for subset in itertools.combinations((1, 2, 3), length)}
    assert actual == expected


def test_status_and_adopted_utility_complete_targets_with_pending_coverage_report(make_input):
    result, actions = generate(make_input(hand=("fog", "smile-dew")), max_sequence_length=2)
    status = [action for action in actions if action["type"] == "play_sequence"]
    assert {action["target"] for action in status} == {1, 2}
    assert {tuple(item["instance_id"] for item in action["ordered_elements"]) for action in status} == {(1,), (2,)}
    from gf_common.action_contract import PENDING_COVERAGE
    assert result.report.omissions == PENDING_COVERAGE


def test_trade_exact_two_references_and_buy_choices(make_input):
    _, actions = generate(make_input(hand=("sell", "sell", "buy", "bronze-club")), max_sequence_length=2)
    actual = {(action["artifact"], action["sold_artifact"], action["target"])
              for action in actions if action["type"] == "sell"}
    assert actual == {(carrier, sold, target) for carrier in (1, 2) for sold in range(1, 5)
                      if sold != carrier for target in (1, 2)}
    assert {(action["artifact"], action["target"]) for action in actions if action["type"] == "buy"} == {(3, 1), (3, 2)}
    result, choices = generate(make_input(choice="buy_accept_reject"))
    assert {action["accept"] for action in choices} == {True, False}
    assert result.report.mandatory == 2


def test_exchange_conditional_grid_conservation_dedup_supplements_and_mandatory(make_input):
    result, actions = generate(make_input(hand=("exchange",), learned=("fireball",), mp=10), max_sequence_length=1)
    allocations = {(action["hp"], action["mp"], action["money"]) for action in actions if action["type"] == "exchange"}
    total = 70
    assert (40, 10, 20) in allocations and any(hp == 0 for hp, mp, money in allocations)
    assert any(hp == 1 for hp, mp, money in allocations)
    assert all(0 <= number <= 99 for allocation in allocations for number in allocation)
    assert all(sum(allocation) == total for allocation in allocations)
    assert {1, 2, 3} <= {mp for hp, mp, money in allocations}  # visible MP2-cost neighborhood
    # Independently check each public conditional axis grid under all3orders;
    # additional documented supplements may extend this grid.
    for first_axis, second_axis, third_axis in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for index in range(26):
            first = max(0, total - 198) + (min(99, total) - max(0, total - 198)) * index // 25
            remainder = total - first
            lower, upper = max(0, remainder - 99), min(99, remainder)
            for second_index in range(26):
                row = [0, 0, 0]
                row[first_axis], row[second_axis], row[third_axis] = first, lower + (upper - lower) * second_index // 25, remainder - (lower + (upper - lower) * second_index // 25)
                assert tuple(row) in allocations
    assert len(allocations) == sum(action["type"] == "exchange" for action in actions)
    assert result.report.precount.numeric_variants == len(allocations)
    # A tiny expansion budget still reserves one complete numeric Action before
    # discretionary supplements; a carrier-only row is never returned.
    small, small_actions = generate(make_input(hand=("exchange",)), max_sequence_length=1,
                                    max_candidates=3, max_expansions=4)
    assert any(action["type"] == "exchange" and set(action) == {"type", "artifact", "hp", "mp", "money"} for action in small_actions)
    assert small.report.mandatory <= 3


def test_discovered_channels_get_quota_and_no_action_is_duplicated(make_input):
    result, actions = generate(make_input(hand=("wand-of-ignition", "abyss-dart", "steel-armor"),
                                          learned=("aura", "fireball"), mp=40),
                               max_sequence_length=4, max_candidates=40)
    assert len({canonical(action) for action in actions}) == len(actions)
    assert all(count > 0 for count in result.report.channel_counts)
    assert sum(result.report.channel_counts) == len(actions) - result.report.mandatory


def test_actor_hint_target_intersection_and_search_sampling_rng_separation(make_input):
    value = make_input(hand=("bronze-club", "fireball"))
    options = value.to_dict()["action_options"]
    options["action_options"]["valid_targets"] = [1]
    options["action_options"]["usable_hand_elements"] = [2]
    value = replace(value, action_options=options)
    result, actions = generate(value, max_sequence_length=2)
    assert all(action.get("target", 1) == 1 for action in actions)
    assert all(item.get("instance_id") != 1 for action in actions for item in action.get("ordered_elements", []))
    sampling_rng = random.Random(1234)
    before = sampling_rng.getstate()
    generate(value, max_sequence_length=2)
    assert sampling_rng.getstate() == before
    assert result.report.rng_before != result.report.rng_after or result.report.discovered == result.report.mandatory


def test_order_sensitive_public_summaries_wand_mirage_aura_and_dynamic_payment(make_input):
    from gf_common.diverse_actions import _Proposal, _Space, _summary
    value = make_input(hand=("wand-of-ignition", "abyss-dart", "fireball"), learned=("mirage", "aura"), mp=50)
    frame, _ = frame_for(value)
    space = _Space(frame, ActionConfigV2(action_contract=engineering_contract()))
    assert _summary(space, _Proposal("play_sequence", refs=(0, 1))).attribute == "fire"
    assert _summary(space, _Proposal("play_sequence", refs=(1, 0))).attribute == "neutral"
    assert _summary(space, _Proposal("play_sequence", refs=(0, 3))).attribute == "neutral"
    assert _summary(space, _Proposal("play_sequence", refs=(1, 3, 0))).attribute == "fire"
    assert _summary(space, _Proposal("play_sequence", refs=(1, 4, 2))).attack == 12  # 5*2+2
    dynamic, _ = frame_for(make_input(hand=("magical-stick", "fireball"), learned=("aura",), mp=10))
    stick = _Space(dynamic, ActionConfigV2(action_contract=engineering_contract()))
    assert _summary(stick, _Proposal("play_sequence", refs=(0, 1))).attack == 18
    assert _summary(stick, _Proposal("play_sequence", refs=(0, 2))).attack == 16
    zero, _ = frame_for(make_input(hand=("magical-stick",), learned=("aura",), mp=6))
    assert _summary(_Space(zero, ActionConfigV2(action_contract=engineering_contract())), _Proposal("play_sequence", refs=(0, 1))).attack == 0


def test_saturated_precount_has_no_proposal_or_action_construction(make_input, monkeypatch):
    import gf_common.diverse_actions as module
    frame, _ = frame_for(make_input(hand=("abyss-dart",) * 18))
    space = module._Space(frame, ActionConfigV2(max_sequence_length=128, count_states=1, action_contract=engineering_contract()))
    def forbidden(*args, **kwargs):
        pytest.fail("pre-count constructed a proposal/Action")
    monkeypatch.setattr(module, "_Proposal", forbidden)
    report, reasons = module._precount(space, NumericCosts(), lambda: 0.)
    assert report.mode == "upper_bound" and report.saturated
    assert report.complete < module.SATURATION and report.preparation_bytes == module.SATURATION and report.count_states == 1
    assert not report.full_enumeration_admitted and reasons == {"count_states"}


@pytest.mark.parametrize("field", [key for key in asdict(NumericCosts()) if key != "version"])
def test_every_numeric_cost_field_rejects_boolean(field):
    with pytest.raises(ContractError):
        NumericCosts(**{field: True})


def test_numeric_dimensions_and_each_budget_are_consumed_and_saturate():
    costs = NumericCosts(context_width=8, card_width=16, actor_input_width=322,
                         actor_hidden_width=16, position_capacity=18, candidate_chunk=8, fixed_bytes=100,
                         tail_operations_per_candidate=100, tail_working_bytes_per_candidate=8192,
                         tail_tensor_bytes_per_candidate=4096)
    arguments = {"candidates": 16, "elements": 50, "positions": 18, "temporary": 32,
                 "temporary_elements": 100, "retained_prefix_elements": 100}
    baseline = estimate_costs(costs, **arguments)
    for field in ("context_width", "card_width", "actor_input_width", "actor_hidden_width", "position_capacity", "candidate_chunk", "fixed_bytes",
                  "tail_operations_per_candidate", "tail_working_bytes_per_candidate", "tail_tensor_bytes_per_candidate"):
        changed = replace(costs, **{field: getattr(costs, field) + 1})
        assert estimate_costs(changed, **arguments) != baseline, field
    for field, limit in (("max_preparation_bytes", baseline.preparation_bytes),
                         ("max_scoring_operations", baseline.scoring_operations),
                         ("max_storage_bytes", baseline.storage_bytes),
                         ("max_tensor_bytes", baseline.largest_tensor_bytes)):
        assert not baseline.fits(replace(costs, **{field: limit - 1})), field
    with pytest.raises(ContractError, match="capacity"):
        estimate_costs(costs, 1, 1, positions=19)
    from gf_common.diverse_actions import SATURATION
    saturated = estimate_costs(NumericCosts(), SATURATION, SATURATION, 128, SATURATION, SATURATION, SATURATION)
    assert saturated.preparation_bytes == saturated.storage_bytes == SATURATION
    assert saturated.scoring_operations == 0
    # Model128/actions18 uses model128 capacity for factorization, rather than
    # interpreting unused card-position dimensions as categorical columns.
    wide = replace(costs, position_capacity=128, actor_input_width=8 + 128 * 16 + 26)
    assert estimate_costs(wide, **arguments).largest_tensor_bytes < 64 * 1024**2


def test_count_attempt_temp_presentation_and_cost_cutoffs_are_distinct(make_input):
    value = make_input(hand=("spiritual-staff",), learned=("fireball", "aura", "mirage"), mp=0)
    attempt, _ = generate(value, max_sequence_length=8, count_states=1, max_expansions=20)
    assert attempt.report.attempts <= 20 and "max_expansions" in attempt.report.cutoff_reasons
    temporary, _ = generate(value, max_sequence_length=8, count_states=1, max_temporary_candidates=6)
    assert temporary.report.discovered <= 6 and "max_temporary_candidates" in temporary.report.cutoff_reasons
    subset, actions = generate(value, max_sequence_length=3, max_candidates=5)
    assert subset.report.traversal_complete and not subset.report.presentation_complete
    assert len(actions) == 5 and "max_candidates" in subset.report.cutoff_reasons
    frame, bank = frame_for(value)
    with pytest.raises(ContractError, match="mandatory.*numeric_costs"):
        PublicCandidateGeneratorV2(ActionConfigV2(action_contract=engineering_contract()), clock=lambda: 0.).generate(
            frame, bank.view(), random.Random(1), NumericCosts(max_storage_bytes=1))


class StepClock:
    def __init__(self, step):
        self.now, self.step = 0., step
    def __call__(self):
        self.now += self.step
        return self.now


def test_cooperative_wall_clock_reserves_selection_and_records_actual_work(make_input):
    frame, bank = frame_for(make_input(hand=("abyss-dart",) * 18, mp=0))
    clock = StepClock(.00001)
    config = ActionConfigV2(max_sequence_length=128, count_states=1, search_milliseconds=100,
                             max_candidates=60, max_expansions=1000000, action_contract=engineering_contract())
    result = PublicCandidateGeneratorV2(config, clock=clock).generate(frame, bank.view(), random.Random(8), NumericCosts())
    actions = [result.table.to_action(index, frame) for index in range(len(result.table.kinds))]
    assert result.report.selected > result.report.mandatory
    assert max(len(action.get("ordered_elements", [])) for action in actions) == 18
    assert "search_time" in result.report.cutoff_reasons
    assert result.report.search_ms < 80 and result.report.selection_ms > 0
    assert result.report.search_ms + result.report.selection_ms <= 100.1
    count_clock = StepClock(.001)
    counted = PublicCandidateGeneratorV2(replace(config, count_milliseconds=1), clock=count_clock)
    with pytest.raises(ContractError, match="mandatory"):
        # A1ms fake increment per cooperative check and1ms total cannot establish
        # all mandatory variants; it must fail rather than publish a subset.
        counted = PublicCandidateGeneratorV2(replace(config, count_milliseconds=1, search_milliseconds=1), clock=count_clock)
        counted.generate(frame, bank.view(), random.Random(8), NumericCosts())


def test_generated_report_binding_counts_flags_and_rng_fail_closed(make_input):
    from gf_common.diverse_actions import GeneratedCandidates
    result, _ = generate(make_input(hand=("bronze-club",)), max_sequence_length=1)
    for changes in ({"table_hash": "0" * 64}, {"selected": 1}, {"selected_elements": 1}):
        try:
            changed = replace(result.report, **changes)
        except ContractError:
            continue
        with pytest.raises(ContractError):
            GeneratedCandidates(result.table, changed)
    with pytest.raises(ContractError):
        replace(result.report, channel_counts=(1,) * 7)
    with pytest.raises(ContractError):
        replace(result.report, numeric_costs_hash="0" * 64)
    with pytest.raises(ContractError):
        replace(result.report, rng_after=(3, (1,), None))
    with pytest.raises(ContractError):
        replace(result.report, presentation_complete=False)


@pytest.mark.parametrize("hand,learned,choice,length", [
    (("abyss-dart", "abyss-dart"), (), "turn_action", 2),
    (("bronze-club",), ("fireball",), "turn_action", 3),
    (("steel-armor", "mars-ring", "super-mirror"), ("wall",), "defense", 3),
    (("buy", "sell", "exchange"), (), "turn_action", 2),
])
def test_exact_precount_matches_completed_search_and_every_attempt(make_input, hand, learned, choice, length):
    result, actions = generate(make_input(hand=hand, learned=learned, choice=choice), max_sequence_length=length)
    assert result.report.precount.mode == "exact"
    assert result.report.precount.full_enumeration_admitted
    assert result.report.precount.complete == result.report.discovered == len(actions)
    assert result.report.precount.attempts == result.report.attempts
    assert result.report.precount.excluded == result.report.prunes
    assert result.report.precount.elements == len(result.table.slots)


def test_repeat_normal_and_defensive_miracles_are_not_generalized(make_input):
    _, ordinary = generate(make_input(hand=(), learned=("flame",), mp=100), max_sequence_length=4)
    assert all(len(action.get("ordered_elements", [])) <= 1 for action in ordinary)
    _, statuses = generate(make_input(hand=(), learned=("fog",), mp=100), max_sequence_length=4)
    assert all(len(action.get("ordered_elements", [])) <= 1 for action in statuses)
    _, defenses = generate(make_input(hand=(), learned=("wall",), choice="defense", mp=100), max_sequence_length=4)
    assert all(len(action.get("ordered_defenses", [])) <= 1 for action in defenses)


def test_unresolved_pure_cutcost_is_not_promoted_by_hints(make_input):
    _, actions = generate(make_input(hand=("spiritual-doll", "fireball")), max_sequence_length=3)
    assert not any(item.get("instance_id") == 1 for action in actions for item in action.get("ordered_elements", []))
    # It remains an ordinary nonweapon discard candidate; omission of direct
    # utility completion does not invent a general prohibition on its reference.
    assert any(1 in action.get("instance_ids", []) for action in actions)


def test_generator_rejects_foreign_bank_and_does_not_read_future_events(make_input):
    from gf_common.structured import EventBank, EventDelta
    frame, bank = frame_for(make_input(hand=("bronze-club",)))
    generator = PublicCandidateGeneratorV2(ActionConfigV2(max_sequence_length=1, action_contract=engineering_contract()), clock=lambda: 0.)
    first = generator.generate(frame, bank.view(), random.Random(5), NumericCosts())
    bank.append(EventDelta(1, "one", 0, bank.boundary.prefix_hash, [{"type": "player_defeated", "player": 2}]))
    later = generator.generate(frame, bank.view(), random.Random(5), NumericCosts())
    assert first == later
    with pytest.raises(ContractError):
        generator.generate(frame, EventBank(2, "one").view(), random.Random(5), NumericCosts())


def test_fixed_sell_pair_respects_one_reference_configuration_and_cost_capacity(make_input):
    frame, bank = frame_for(make_input(hand=("sell", "bronze-club")))
    config = ActionConfigV2(max_sequence_length=1, action_contract=engineering_contract())
    result = PublicCandidateGeneratorV2(config, clock=lambda: 0.).generate(
        frame, bank.view(), random.Random(3), NumericCosts(position_capacity=1))
    result.table.validate(frame)
    actions = [result.table.to_action(index, frame) for index in range(len(result.table.kinds))]
    assert not any(action["type"] == "sell" for action in actions)
    assert max(b - a for a, b in zip(result.table.offsets[:-1], result.table.offsets[1:], strict=True)) <= 1
    assert "max_sequence_length" in result.report.cutoff_reasons
    assert result.report.precount.complete == len(actions)
    assert result.report.precount.attempts == result.report.attempts


@pytest.mark.parametrize("additional", ["ogre-s-armor", "ogre-s-gauntlet", "ogre-s-helm", "ogre-s-shoes", "strength-powder"])
def test_positive_additional_capability_is_independent_of_broad_kind(make_input, additional):
    result, actions = generate(make_input(hand=("bronze-club", additional)), max_sequence_length=2)
    plays = [action for action in actions if action["type"] == "play_sequence"]
    actual = {tuple(item["instance_id"] for item in action["ordered_elements"]) for action in plays}
    assert actual == {(1,), (2,), (1, 2)}
    assert result.report.mandatory >= 4
    assert all(len(set(sequence)) == len(sequence) for sequence in actual)
    value = make_input(hand=("bronze-club", additional))
    options = value.to_dict()["action_options"]
    options["action_options"]["usable_hand_elements"] = [1]
    _, masked = generate(replace(value, action_options=options), max_sequence_length=2)
    assert not any(item["instance_id"] == 2 for action in masked for item in action.get("ordered_elements", []))


@pytest.mark.parametrize("weight", [100, 2**31 - 1])
def test_skew_channel_weights_preserve_minimum_unique_opportunities(make_input, weight):
    result, actions = generate(make_input(hand=("wand-of-ignition", "abyss-dart", "steel-armor"),
                                          learned=("aura", "fireball"), mp=40),
                               max_sequence_length=4, max_candidates=40,
                               channel_weights=(weight, 1, 1, 1, 1, 1, 1))
    assert result.report.selected == 40
    assert all(count >= 1 for count in result.report.channel_counts)
    assert result.report.channel_counts[0] > result.report.channel_counts[1]
    assert len({canonical(action) for action in actions}) == 40


def test_attempt_budget_precedes_extension_validity_and_numeric_row_construction(make_input, monkeypatch):
    import gf_common.diverse_actions as module
    frame, _ = frame_for(make_input(hand=("spiritual-staff",), learned=("fireball",)))
    space = module._Space(frame, ActionConfigV2(max_sequence_length=3, action_contract=engineering_contract()))
    validity_calls = []
    original_valid = space.valid
    def counted_valid(*args):
        validity_calls.append(1)
        return original_valid(*args)
    monkeypatch.setattr(space, "valid", counted_valid)
    def denied():
        raise module._SearchLimit
    with pytest.raises(module._SearchLimit):
        next(module._walk(space, "offense", (0,), None, tuple(space.offense), denied))
    assert validity_calls == []
    trade_frame, _ = frame_for(make_input(hand=("exchange", "sell", "bronze-club")))
    trade = module._Space(trade_frame, ActionConfigV2(action_contract=engineering_contract()))
    def forbidden_row(*args, **kwargs):
        pytest.fail("compact proposal was constructed before budget admission")
    monkeypatch.setattr(module, "_Proposal", forbidden_row)
    with pytest.raises(module._SearchLimit):
        next(trade.fixed(denied))


def test_sell_only_domain_can_have_zero_mandatory_proposals(make_input):
    value = make_input(hand=("sell", "bronze-club"))
    options = value.to_dict()["action_options"]
    options["action_options"].update(can_pray=False, can_discard=False, can_sacrifice=False,
                                     usable_hand_elements=[])
    result, actions = generate(replace(value, action_options=options), max_sequence_length=2)
    assert result.report.mandatory == 0 and result.report.selected == 2
    assert {(action["artifact"], action["sold_artifact"], action["target"]) for action in actions} == {(1, 2, 1), (1, 2, 2)}
    assert result.report.presentation_complete


def test_exchange_supplements_share_budget_with_long_family_jobs(make_input):
    result, actions = generate(make_input(hand=("exchange", "spiritual-staff"), learned=("fireball", "aura", "mirage"), mp=0),
                               max_sequence_length=128, count_states=1, max_expansions=1000, max_candidates=30)
    assert result.report.attempts <= 1000
    assert max(len(action.get("ordered_elements", [])) for action in actions) == 4
    assert sum(action["type"] == "exchange" for action in actions) > 1
    assert not result.report.traversal_complete


def test_numeric_costs_admit_padded_projection_and_transient_overlap():
    capacity, hidden = 128, 1024
    costs = NumericCosts(context_width=4, card_width=10, actor_input_width=4 + capacity * 10 + 26,
                         actor_hidden_width=hidden, position_capacity=capacity, candidate_chunk=1,
                         tail_operations_per_candidate=100, tail_working_bytes_per_candidate=128,
                         tail_tensor_bytes_per_candidate=64)
    estimate = estimate_costs(costs, candidates=1, elements=1, positions=capacity)
    unpadded, padded = capacity * 18 * hidden * 4, capacity * 19 * hidden * 4
    assert estimate.largest_tensor_bytes == padded
    assert estimate.preparation_bytes >= unpadded + padded
    assert not estimate.fits(replace(costs, max_tensor_bytes=(unpadded + padded) // 2))


def test_head_tail_costs_cover_small_first_width_and_large_later_layers(make_input):
    # The first actor width is1, followed by1024,1024 and1. Independent dense
    # arithmetic for the tail exceeds one million multiply/add pairs per row;
    # it cannot be bounded by a first-width-only coefficient or fixed encoder RAM.
    operations = 2 * (1 * 1024 + 1024 * 1024 + 1024 * 1) + 1024 + 1024
    working = (1024 + 1024 + 1) * 4 * 8
    largest = 1024 * 4
    costs = NumericCosts(context_width=4, card_width=4, actor_input_width=4 + 18 * 4 + 26,
                         actor_hidden_width=1, position_capacity=18, candidate_chunk=4,
                         tail_operations_per_candidate=operations,
                         tail_working_bytes_per_candidate=working,
                         tail_tensor_bytes_per_candidate=largest)
    estimate = estimate_costs(costs, candidates=16, elements=20, positions=18)
    assert estimate.scoring_operations >= 16 * operations
    assert estimate.preparation_bytes >= 4 * working
    assert estimate.largest_tensor_bytes == 4 * largest
    assert not estimate.fits(replace(costs, max_scoring_operations=operations - 1))
    frame, bank = frame_for(make_input(hand=("bronze-club",)))
    with pytest.raises(ContractError, match="mandatory.*numeric_costs"):
        PublicCandidateGeneratorV2(ActionConfigV2(action_contract=engineering_contract()), clock=lambda: 0.).generate(
            frame, bank.view(), random.Random(3), replace(costs, max_scoring_operations=operations - 1))
    for changes in ({"tail_operations_per_candidate": 0}, {"tail_working_bytes_per_candidate": 0},
                    {"tail_tensor_bytes_per_candidate": 0},
                    {"tail_working_bytes_per_candidate": 1, "tail_tensor_bytes_per_candidate": 2}):
        with pytest.raises(ContractError):
            replace(costs, **changes)
    from gf_common.diverse_actions import SATURATION
    huge = replace(costs, tail_operations_per_candidate=SATURATION,
                   tail_working_bytes_per_candidate=SATURATION,
                   tail_tensor_bytes_per_candidate=SATURATION)
    saturated = estimate_costs(huge, 2, 2, 18)
    assert saturated.scoring_operations == saturated.preparation_bytes == saturated.largest_tensor_bytes == SATURATION


@pytest.mark.parametrize("learned", [False, True])
def test_refresh_potential_includes_miracle_opportunities_without_freeing_held_slot(make_input, learned):
    from gf_common.diverse_actions import _Proposal, _Space, _summary
    for possession, expected in ((2, 1), (18, 0)):
        if learned:
            value = make_input(hand=("steel-armor",) * (possession - 1), learned=("fireball",))
            miracle_slot = possession - 1
        else:
            value = make_input(hand=("fireball",) + ("steel-armor",) * (possession - 1))
            miracle_slot = 0
        frame, _ = frame_for(value)
        summary = _summary(_Space(frame, ActionConfigV2(action_contract=engineering_contract())), _Proposal("play_sequence", refs=(miracle_slot,)))
        assert summary.refresh == expected


def test_refresh_potential_staff_distinct_copies_capacity_sacrifice_and_self_sell(make_input):
    from gf_common.diverse_actions import _Proposal, _Space, _summary
    for possession, expected in ((4, 4), (18, 1)):
        value = make_input(hand=("spiritual-staff", "fireball", "fireball", "fireball") + ("steel-armor",) * (possession - 4), mp=0)
        frame, _ = frame_for(value)
        space = _Space(frame, ActionConfigV2(action_contract=engineering_contract()))
        # One ordinary replacement +3distinct miracle acquisitions, capped by
        # capacity freed by Staff; held miracles move to learned, freeing no slots.
        summary = _summary(space, _Proposal("play_sequence", refs=(0, 1, 2, 3)))
        assert summary.refresh == expected
    frame, _ = frame_for(make_input(hand=("sell", "steel-armor") + ("steel-armor",) * 16))
    space = _Space(frame, ActionConfigV2(action_contract=engineering_contract()))
    own_sale = _summary(space, _Proposal("sell", "self", (0, 1)))
    assert own_sale.refresh == 1 and 1 in own_sale.remaining and 0 not in own_sale.remaining
    other_sale = _summary(space, _Proposal("sell", "opponent", (0, 1)))
    assert other_sale.refresh == 1 and 1 not in other_sale.remaining
    assert _summary(space, _Proposal("discard", refs=(0, 1))).refresh == 0
    assert _summary(space, _Proposal("sacrifice", refs=(0, 1))).refresh == 2


def test_scoring_operation_bound_counts_every_padded_position_sum():
    costs = NumericCosts(context_width=4, card_width=4, actor_input_width=4 + 128 * 4 + 26,
                         actor_hidden_width=16, position_capacity=128, candidate_chunk=1,
                         tail_operations_per_candidate=1, tail_working_bytes_per_candidate=1,
                         tail_tensor_bytes_per_candidate=1)
    count = 100
    value = estimate_costs(costs, count, elements=count, positions=128)
    expected_lower_bound = 2 * 128 * 18 * 4 * 16 + 2 * count * 26 * 16 + count * 128 * 16 + 2 * 4 * 16 + count
    assert value.scoring_operations == expected_lower_bound


def test_saturating_product_preserves_zero_even_after_huge_factors():
    from gf_common.diverse_actions import SATURATION, _sat_mul
    assert _sat_mul(SATURATION, 128, 0) == 0
    assert _sat_mul(2, SATURATION, 0, SATURATION) == 0
    assert _sat_mul(2, SATURATION, 1) == SATURATION
