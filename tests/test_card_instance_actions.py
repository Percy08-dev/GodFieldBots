"""Independent public-domain enumeration and shared bounded-search contracts."""

import copy
import itertools
import json
import random
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.card_instance import (
    ActionConfigV3,
    card_instance_contract,
    corrected_card_instance_contract,
)
from gf_common.card_instance_rules import UTILITY_IDS, card_instance_rules, rule_signature
from gf_common.diverse_actions import (
    GeneratedCandidates,
    GeneratedCandidatesV3,
    NumericCosts,
    PublicCandidateGeneratorV2,
    PublicCandidateGeneratorV3,
    SearchReport,
    SearchReportV3,
)
from gf_common.domain import ContractError, DecisionInput, EmptyCandidates, canonical, digest
from gf_common.structured import (
    DecisionFrame,
    EventBank,
    EventDelta,
    action_config_v2,
    event_delta,
)
from gf_common.structured_layout import layout_for_contract
from gf_common.structured_wire import (
    generated_dict,
    generated_dict_v3,
    parse_generated,
    parse_generated_v3,
    parse_report,
)
from tests.card_instance_action_support import inputs, run
from tests.test_card_instance_structured import frame_for


def plays(actions):
    return [action for action in actions if action["type"] == "play_sequence"]


class FakeClock:
    def __init__(self, step=0.0):
        self.now, self.step, self.calls = 0.0, step, 0

    def __call__(self):
        self.now += self.step
        self.calls += 1
        return self.now


@pytest.mark.parametrize("state", ["unused", "learned"])
@pytest.mark.parametrize("name", ["flame", "fireball", "aura", "wall", "wind", "spring"])
@pytest.mark.parametrize("end_times", [False, True])
def test_all_miracle_subclasses_and_states_discard_at_mp0_without_direct_use_mask(
    name, state, end_times
):
    result, actions, frame = run(inputs([(name, state)], mp=0, usable=(), end_times=end_times))
    expected = {
        "type": "sacrifice" if end_times else "discard",
        "card_refs": list(frame.references),
    }
    assert actions == [expected]
    assert result.report.precount.complete == 1 and result.report.discovered == 1
    assert not any("learned" in omission for omission in result.report.omissions)


def test_discard_independent_subsets_include_both_states_but_exclude_artifact_weapons():
    spec = [
        ("flame", "learned"),
        ("fireball", "unused"),
        ("steel-armor", None),
        ("bronze-club", None),
    ]
    result, actions, frame = run(inputs(spec, mp=0, usable=()))
    expected = {
        canonical({"type": "discard", "card_refs": [frame.references[i] for i in subset]})
        for length in range(1, 4)
        for subset in itertools.combinations(range(3), length)
    }
    assert {canonical(action) for action in actions} == expected
    assert result.report.precount.complete == result.report.discovered == 7


@pytest.mark.parametrize(
    "contract", [card_instance_contract(), corrected_card_instance_contract()],
    ids=["07e", "91d5926"],
)
@pytest.mark.parametrize("end_times", [False, True], ids=["discard", "sacrifice"])
@pytest.mark.parametrize("dream", [False, True], ids=["ordinary-view", "dream-view"])
def test_protected_artifact_exact_subsets_use_perceived_definitions(contract, end_times, dream):
    artifacts = ["sun-amulet", "dangerous-mortar", "steel-armor"]
    if dream:
        # Same opaque references, different supplied perceptions. No canonical
        # identity is available to the generator, including for protected cards.
        artifacts = ["steel-armor", "sun-amulet", "dangerous-mortar"]
    value = inputs(
        [(name, None) for name in artifacts]
        + [("bronze-club", None), ("flame", "unused"), ("fireball", "learned")],
        mp=0, usable=(), end_times=end_times,
    )
    value.observation["own"]["status"]["dream"] = dream
    layout = layout_for_contract(contract)
    bank = EventBank(1, "protected-discard", layout=layout)
    bank.append(event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, (), layout=layout))
    frame = DecisionFrame.from_input(value, bank, 0, layout=layout)
    result = PublicCandidateGeneratorV3(
        ActionConfigV3(action_contract=contract), rules=card_instance_rules(), clock=lambda: 0.0,
    ).generate(frame, bank.view(), random.Random(17), NumericCosts())
    result.table.validate(frame)
    actions = [result.table.to_action(index, frame) for index in range(len(result.table.kinds))]
    allowed = (0, 1, 2, 4, 5) if end_times else (0 if dream else 2, 4, 5)
    expected = {
        canonical({
            "type": "sacrifice" if end_times else "discard",
            "card_refs": [frame.references[index] for index in subset],
        })
        for length in range(1, len(allowed) + 1)
        for subset in itertools.combinations(allowed, length)
    }
    assert {canonical(action) for action in actions} == expected
    count, elements = (31, 80) if end_times else (7, 12)
    assert result.report.precount.mode == "exact"
    assert result.report.precount.full_enumeration_admitted
    assert result.report.precount.complete == result.report.discovered == len(actions) == count
    assert result.report.precount.elements == len(result.table.slots) == elements
    assert set(frame.raw["catalog"]) == set(artifacts) | {"bronze-club", "flame", "fireball"}


@pytest.mark.parametrize("name", sorted(UTILITY_IDS))
def test_every_scoped_utility_both_wire_targets_even_with_empty_effect(name):
    state = "learned" if name in ("spring", "treasure", "tone", "song", "release") else None
    result, actions, frame = run(inputs([(name, state)], discard=False))
    assert {canonical(action) for action in actions} == {
        canonical(
            {"type": "play_sequence", "ordered_card_refs": list(frame.references), "target": target}
        )
        for target in (1, 2)
    }
    assert result.report.mandatory == 2 and result.report.precount.complete == 2


def test_independent_additional_instance_permutations_and_exact_search_accounting():
    result, actions, frame = run(
        inputs([("abyss-dart", None)] * 3, discard=False), max_sequence_length=3
    )
    expected = {
        canonical({"type": "play_sequence", "ordered_card_refs": list(sequence), "target": target})
        for length in (1, 2, 3)
        for sequence in itertools.permutations(frame.references, length)
        for target in (1, 2)
    }
    assert {canonical(action) for action in actions} == expected
    assert (
        result.report.precount.mode == "exact" and result.report.precount.full_enumeration_admitted
    )
    assert result.report.precount.complete == result.report.discovered == 30
    assert result.report.precount.attempts == result.report.attempts
    assert result.report.precount.excluded == result.report.prunes
    assert result.report.precount.elements == len(result.table.slots)


@pytest.mark.parametrize(
    "spec,mp",
    [
        ([("fireball", "unused"), ("spiritual-doll", None)], 0),
        ([("flame", "unused"), ("flame", "learned"), ("fireball", "learned")], 50),
        ([("spring", "learned"), ("spiritual-staff", None)], 0),
        ([("wind", "unused"), ("spiritual-doll", None)], 0),
    ],
)
def test_exact_precount_agrees_with_zero_completion_prefix_and_discovery(spec, mp):
    result, actions, frame = run(inputs(spec, mp=mp), max_sequence_length=3)
    assert result.report.precount.complete == result.report.discovered == len(actions)
    assert result.report.precount.attempts == result.report.attempts
    assert result.report.precount.excluded == result.report.prunes
    assert result.report.precount.elements == len(result.table.slots)
    if mp == 0:
        root = frame.references[0]
        assert not any(action["ordered_card_refs"] == [root] for action in plays(actions))
        assert any(
            len(action["ordered_card_refs"]) == 2 and action["ordered_card_refs"][0] == root
            for action in plays(actions)
        )


def test_independent_costcut_small_space_expected_actions():
    result, actions, frame = run(inputs([("fireball", "unused"), ("spiritual-staff", None)], mp=0))
    miracle, staff = frame.references
    expected = [{"type": "discard", "card_refs": [miracle]}]
    expected += [
        {"type": "play_sequence", "ordered_card_refs": sequence, "target": target}
        for sequence in ([staff], [miracle, staff])
        for target in (1, 2)
    ]
    assert {canonical(action) for action in actions} == {canonical(action) for action in expected}
    assert result.report.discovered == 5


def test_summary_uses_same_role_payment_facts_and_keeps_physical_remaining_slots():
    from gf_common.diverse_actions import _CardInstanceSpace, _Proposal

    config = ActionConfigV3(action_contract=card_instance_contract())
    frame, bank, _ = frame_for(inputs([("fireball", "learned"), ("spiritual-staff", None)], mp=0))
    space = _CardInstanceSpace(frame, config, card_instance_rules(), bank.view())
    analysis = space.analysis("offense", (0, 1))
    summary = space.summary(_Proposal("play_sequence", "opponent", (0, 1)))
    assert summary.attack == analysis.attack == 2
    assert summary.resources[0] == analysis.spent_mp == 0
    assert summary.effects == analysis.effects and "cutCost" not in summary.effects
    assert summary.remaining == (0,)  # Learned miracle retains its physical slot.
    full_frame, full_bank, _ = frame_for(inputs([("fireball", "learned")] * 18, mp=50, can_pray=True))
    full = _CardInstanceSpace(full_frame, config, card_instance_rules(), full_bank.view())
    assert full.summary(_Proposal("play_sequence", "opponent", (0,))).refresh == 0
    assert full.summary(_Proposal("pray")).refresh == 1
    assert full.summary(_Proposal("pray")).remaining == tuple(range(18))  # Random victim unknown.
    assert full.summary(_Proposal("discard", refs=(0, 1))).refresh == 0
    assert full.summary(_Proposal("sacrifice", refs=(0, 1))).refresh == 2
    assert full.summary(_Proposal("sacrifice", refs=(0, 1))).remaining == tuple(range(2, 18))
    sell_frame, sell_bank, _ = frame_for(
        inputs([("sell", None), ("steel-armor", None), ("fireball", "learned")])
    )
    sell = _CardInstanceSpace(sell_frame, config, card_instance_rules(), sell_bank.view())
    assert sell.summary(_Proposal("sell", "self", (0, 1))).remaining == (1, 2)
    assert sell.summary(_Proposal("sell", "opponent", (0, 1))).remaining == (2,)


@pytest.mark.parametrize("flash", [False, True])
def test_defense_flash_counts_every_physical_ref_and_offense_control(flash):
    result, actions, frame = run(
        inputs(
            [("wall", "learned"), ("spiritual-socks", None)], mp=0, choice="defense", flash=flash
        )
    )
    wall, socks = frame.references
    expected = [
        {"type": "respond", "ordered_card_refs": []},
        {"type": "respond", "ordered_card_refs": [socks]},
    ]
    if not flash:
        expected.append({"type": "respond", "ordered_card_refs": [wall, socks]})
    assert {canonical(action) for action in actions} == {canonical(action) for action in expected}
    assert result.report.precount.complete == len(expected)
    _, offensive, _ = run(inputs([("abyss-dart", None)] * 2, discard=False, flash=flash))
    assert any(len(action["ordered_card_refs"]) == 2 for action in offensive)


def test_trade_excludes_learned_sold_items_and_exchange_grid_supports_hp0():
    value = inputs(
        [
            ("sell", None),
            ("buy", None),
            ("exchange", None),
            ("fireball", "unused"),
            ("fireball", "learned"),
            ("string-of-fate", None),
        ]
    )
    value.observation["own"].update(hp=1, mp=0, money=0)
    result, actions, frame = run(value, max_sequence_length=2, max_candidates=300)
    sells = [action for action in actions if action["type"] == "sell"]
    assert sells and all(action["sold_card_ref"] != frame.references[4] for action in sells)
    assert {action["target"] for action in actions if action["type"] == "buy"} == {1, 2}
    exchanges = [action for action in actions if action["type"] == "exchange"]
    assert {(a["hp"], a["mp"], a["money"]) for a in exchanges} == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    assert all("target" not in action for action in exchanges)
    assert {"type": "invoke_phenomenon", "card_ref": frame.references[5]} in actions
    assert result.report.precount.complete == result.report.discovered
    assert result.report.precount.attempts == result.report.attempts
    assert result.report.precount.excluded == result.report.prunes


def test_full_exchange_25_interval_grid_is_conserved_and_unique():
    from gf_common.diverse_actions import _grid

    assert len(_grid(0, 99)) == 26 and {0, 99}.issubset(_grid(0, 99))
    value = inputs([("exchange", None)], discard=False)
    value.observation["own"].update(hp=40, mp=10, money=20)
    _, actions, _ = run(value)
    allocations = {(a["hp"], a["mp"], a["money"]) for a in actions}
    assert len(allocations) == len(actions)
    assert all(sum(a) == 70 and all(0 <= v <= 99 for v in a) for a in allocations)
    assert any(a[0] == 0 for a in allocations)


def test_basic_prayer_buy_decisions_and_mandatory_failure():
    result, actions, _ = run(inputs([], can_pray=True))
    assert actions == [{"type": "pray"}] and result.report.mandatory == 1
    result, actions, _ = run(inputs([], choice="buy_accept_reject"))
    assert actions == [
        {"type": "buy_decision", "accept": False},
        {"type": "buy_decision", "accept": True},
    ]
    assert result.report.mandatory == 2
    with pytest.raises(ContractError, match="mandatory"):
        run(inputs([("smile-dew", None)], discard=False), max_candidates=1)
    with pytest.raises(ContractError, match="mandatory.*numeric_costs"):
        run(inputs([("abyss-dart", None)]), costs=NumericCosts(max_storage_bytes=1))
    with pytest.raises(EmptyCandidates, match="v5"):
        run(inputs([], discard=False))


def test_rule_bound_new_wire_and_old_dto_rejection():
    result, _, _ = run(inputs([("bronze-club", None)]))
    binding = {"action_contract": card_instance_contract(), "rules": card_instance_rules()}
    raw = json.loads(json.dumps(generated_dict_v3(result, **binding)))
    assert parse_generated_v3(raw, **binding) == result
    assert result.report.rule_signature == rule_signature(
        contract=binding["action_contract"], rules=binding["rules"]
    )
    assert type(result) is GeneratedCandidatesV3 and type(result.report) is SearchReportV3
    assert {field.name for field in fields(SearchReportV3)} == {
        field.name for field in fields(SearchReport)
    } | {"rule_signature"}
    with pytest.raises(ContractError):
        generated_dict(result)
    with pytest.raises(ContractError):
        GeneratedCandidates(result.table, result.report)
    with pytest.raises(ContractError):
        parse_generated(raw, action_contract=engineering_contract())
    with pytest.raises(ContractError):
        parse_report(raw["report"], action_contract=engineering_contract())
    for change in (
        "missing_rule",
        "wrong_rule",
        "bad_generator",
        "bad_table_hash",
        "foreign_slot",
        "bad_report_count",
    ):
        bad = copy.deepcopy(raw)
        if change == "missing_rule":
            bad["report"].pop("rule_signature")
        elif change == "wrong_rule":
            bad["report"]["rule_signature"] = "0" * 64
        elif change == "bad_generator":
            bad["report"]["generator_version"] = "public-atomic-diverse-v4"
        elif change == "bad_table_hash":
            bad["table"]["table_hash"] = "0" * 64
        elif change == "foreign_slot":
            bad["table"]["slots"] = [18]
        else:
            bad["report"]["selected"] += 1
        with pytest.raises(ContractError):
            parse_generated_v3(bad, **binding)
    with pytest.raises(TypeError):
        PublicCandidateGeneratorV3(ActionConfigV3(action_contract=card_instance_contract()))
    with pytest.raises(ContractError):
        PublicCandidateGeneratorV3(
            ActionConfigV3(action_contract=card_instance_contract()), rules=None
        )
    with pytest.raises(ContractError):
        replace(result.report, rule_signature="0" * 64)


def test_long_band_budgets_deadlines_saturation_and_search_rng_separation():
    value = inputs([("abyss-dart", None)] * 18, discard=False, mp=0)
    result, actions, _ = run(value, count_states=1, max_expansions=3000, max_candidates=50)
    assert result.report.precount.mode == "upper_bound"
    assert result.report.attempts <= 3000 and not result.report.traversal_complete
    assert max(len(action["ordered_card_refs"]) for action in actions) == 18
    assert len(actions) <= 50 and result.report.rng_before != result.report.rng_after
    sampling = random.Random(123)
    before = sampling.getstate()
    clocked, _, _ = run(
        value,
        seed=8,
        clock=FakeClock(0.00001),
        count_states=1,
        search_milliseconds=100,
        max_candidates=60,
        max_expansions=1000000,
    )
    assert (
        "search_time" in clocked.report.cutoff_reasons
        and clocked.report.search_ms + clocked.report.selection_ms <= 100.1
    )
    assert sampling.getstate() == before
    from gf_common.diverse_actions import _CardInstanceSpace, _precount

    frame, bank, _ = frame_for(value)
    config = ActionConfigV3(action_contract=card_instance_contract(), count_states=1)
    space = _CardInstanceSpace(frame, config, card_instance_rules(), bank.view())
    huge = NumericCosts(
        context_width=8,
        card_width=16,
        actor_input_width=4096,
        actor_hidden_width=16,
        tail_operations_per_candidate=2**60,
        tail_working_bytes_per_candidate=1,
        tail_tensor_bytes_per_candidate=1,
    )
    count, reasons = _precount(space, huge, lambda: 0.0)
    assert count.saturated and not count.full_enumeration_admitted and reasons == {"count_states"}
    assert bank.layout is frame.layout


def test_unknown_group_keys_do_not_change_candidates_or_report_and_foreign_views_reject():
    value = inputs([("abyss-dart", None)] * 3, discard=False)
    before, _, _ = run(value, count_states=1, max_candidates=12)
    changed = copy.deepcopy(value)
    changed.action_options["action_options"]["candidate_groups"] = {
        "not-a-role": changed.action_options["action_options"]["usable_cards"]
    }
    after, _, _ = run(changed, count_states=1, max_candidates=12)
    # Frame/table hashes bind the original raw data; row semantics and RNG do not.
    assert before.table.kinds == after.table.kinds and before.table.slots == after.table.slots
    assert before.report.rng_after == after.report.rng_after
    assert before.report.attempts == after.report.attempts
    frame, bank, _ = frame_for(value)
    other = EventBank(2, bank.episode, layout=bank.layout)
    generator = PublicCandidateGeneratorV3(
        ActionConfigV3(action_contract=card_instance_contract()),
        rules=card_instance_rules(),
        clock=lambda: 0.0,
    )
    with pytest.raises(ContractError):
        generator.generate(frame, other.view(), random.Random(17), NumericCosts())


def test_original_214b1fd_generator_goldens_table_report_rng_and_clock_calls():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/card_instance_legacy_generator_golden.json").read_text()
    )
    assert (
        golden["content_sha256"]
        == "40d67782e419c8d7ce6208adee4d0a37a63ea33972f34e84696865848c5f5317"
    )
    assert (
        digest({key: value for key, value in golden.items() if key != "content_sha256"})
        == golden["content_sha256"]
    )
    assert golden["provenance"]["bots_revision"] == "214b1fd556cb93b54298a96f4c43f2999de796c0"
    for case in golden["cases"]:
        raw = case["input"]
        value = DecisionInput(
            raw["observation"],
            tuple(raw["events"]),
            raw["history_end"],
            tuple(raw["typed_metadata"]),
            raw["catalog"],
            raw["action_options"],
        )
        bank = EventBank(case["bank"]["actor"], case["bank"]["episode"], case["bank"]["max_events"])
        bank.append(
            EventDelta(bank.actor, bank.episode, 0, bank.boundary.prefix_hash, value.events)
        )
        frame = DecisionFrame.from_input(value, bank, 0)
        clock = FakeClock(case["clock"]["step_seconds"])
        result = PublicCandidateGeneratorV2(
            action_config_v2(case["actions_config"]), clock=clock
        ).generate(
            frame,
            bank.view(),
            random.Random(case["search_seed"]),
            NumericCosts(**case["numeric_costs"]),
        )
        report = asdict(result.report)
        for name in golden["provenance"]["excluded_report_fields"]:
            report.pop(name)
        assert canonical(asdict(result.table)) == canonical(case["expected_table"]), case["name"]
        assert canonical(report) == canonical(case["expected_report"]), case["name"]
        assert clock.calls == case["clock"]["calls"]
        assert [result.table.to_action(i, frame) for i in range(len(result.table.kinds))] == case[
            "expected_actions"
        ]
