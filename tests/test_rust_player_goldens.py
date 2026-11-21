"""Pinned Python-v5 originals for later native runtime parity (never regenerated here)."""

import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import parse_card_instance_rules
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from gf_common.domain import canonical, digest
from gf_common.refresh_heuristic import RefreshOptions
from gf_common.structured import EventBank
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_wire import (
    game_generated_dict_v3,
    parse_delta,
    parse_frame,
    parse_game_generated_v3,
    parse_selection,
)
from gf_models.card_instance_baselines import CardInstanceBaseline
from gf_models.refresh_heuristic import RefreshHeuristicPolicy

FIXTURE = Path(__file__).parent / "fixtures/rust_player_v5/originals.json"


def originals():
    assert FIXTURE.exists(), "explicit original-v5 capture has not run"
    return json.loads(FIXTURE.read_text())


def state(raw):
    return raw[0], tuple(raw[1]), raw[2]


class ReplayClock:
    def __init__(self, trace):
        self.trace, self.index = trace, 0

    def __call__(self):
        assert self.index < len(self.trace), "generator requested an extra clock reading"
        value = self.trace[self.index]["seconds"]
        self.index += 1
        return value


def test_originals_present_with_source_and_contract_pins():
    raw = originals()
    assert raw["schema"] == "rust-player-python-v5-originals-v1"
    assert raw["source_revision"] == "9680f06ad8ba61d65d142863742908678638b02b"
    assert raw["simulator_revision"] == "072dabbca1458c9526897781d2fd64edeeb4f9ca"
    assert (
        raw["spec07_sha256"] == "8e5223b45a6e8c4c80e272736ed97b9e271136c939eca74427a035ddfdbfcac2"
    )
    assert raw["report_milliseconds_normalization"] == "none; exact clock transcript"
    assert "src/gf_common/diverse_actions.py" in raw["source_sha256"]
    assert all(len(value) == 64 for value in raw["source_sha256"].values())
    assert raw["episodes_sha256"] == digest(raw["episodes"])
    assert {episode["name"] for episode in raw["episodes"]} == {
        "defense-miracle-priority",
        "defense-rainbow-first",
        "duplicate-instances",
        "empty-events",
        "dynamic-weapon-zero-mp",
        "dynamic-weapon-positive-mp",
        "large-candidate-table",
        "clock-cutoffs",
        "refresh-history",
    }


@pytest.mark.parametrize("episode_index", range(9))
def test_current_v5_replays_original_table_report_rng_and_baseline_selections(episode_index):
    episode = originals()["episodes"][episode_index]
    actions = action_config_v3(episode["actions"])
    rules = parse_card_instance_rules(episode["rules"])
    bank = EventBank(episode["actor"], episode["name"], layout=LAYOUT)
    search = random.Random(episode["search_seed"])
    policies = {
        kind: CardInstanceBaseline(kind, actions, rules=rules) for kind in ("random", "heuristic")
    }
    policies["refresh"] = RefreshHeuristicPolicy(
        actions, rules=rules, options=RefreshOptions(**episode["refresh_options"])
    )
    for policy in policies.values():
        policy.reset(episode["sampling_seed"])
    for decision in episode["decisions"]:
        bank.append(parse_delta(decision["delta"], layout=LAYOUT))
        frame = parse_frame(decision["frame"], layout=LAYOUT)
        frame.bank_view(bank.view())
        assert list(frame.references) == decision["references"]
        assert search.getstate() == state(decision["search_rng_before"])
        trace = decision["clock_trace"]
        assert trace[0]["caller"] == "_run_search"
        assert trace[1]["caller"] == "_precount"
        assert trace[-1]["caller"] == "_run_search"
        clock = ReplayClock(trace)
        generated = PublicCandidateGeneratorV3(actions, rules=rules, clock=clock).generate_game(
            frame, bank.view(), search, NumericCosts(**episode["numeric_costs"])
        )
        assert clock.index == len(trace)
        assert search.getstate() == state(decision["search_rng_after"])
        assert canonical(
            game_generated_dict_v3(generated, action_contract=actions.action_contract, rules=rules)
        ) == canonical(decision["generated"])
        saved = parse_game_generated_v3(
            decision["generated"], action_contract=actions.action_contract, rules=rules
        )
        saved.table.validate(frame)
        assert saved == generated
        assert [saved.table.to_action(i, frame) for i in range(len(saved.table.kinds))] == decision[
            "ordered_actions"
        ]
        for kind, policy in policies.items():
            expected = decision["policies"][kind]
            if kind == "refresh":
                assert expected["rng_before"] is expected["rng_after"] is None
            else:
                assert policy.rng.getstate() == state(expected["rng_before"])
            selected = policy.select_frame(frame, bank.view(), generated)
            assert canonical(asdict(selected)) == canonical(expected["selection"])
            assert selected.validate(saved.table, frame) == expected["action"]
            assert parse_selection(expected["selection"]) == selected
            if kind != "refresh":
                assert policy.rng.getstate() == state(expected["rng_after"])
            if kind == "refresh":
                for repeated in expected["repeated_selections"]:
                    assert canonical(
                        asdict(policy.select_frame(frame, bank.view(), generated))
                    ) == canonical(repeated)


def test_original_cases_exercise_required_public_behavior():
    episodes = {episode["name"]: episode for episode in originals()["episodes"]}

    def decision(name):
        return episodes[name]["decisions"][0]

    rainbow = decision("defense-rainbow-first")
    refs = rainbow["references"]
    pairs = [
        row["ordered_card_refs"]
        for row in rainbow["ordered_actions"]
        if len(row["ordered_card_refs"]) == 2
    ]
    assert [refs[0], refs[1]] in pairs and [refs[1], refs[0]] not in pairs
    priority = decision("defense-miracle-priority")
    assert all(len(row["ordered_card_refs"]) <= 1 for row in priority["ordered_actions"])
    duplicate = decision("duplicate-instances")
    cards = duplicate["frame"]["payload"]["observation"]["own"]["cards"]
    assert cards[0]["definition_id"] == cards[1]["definition_id"]
    assert cards[0]["card_ref"] != cards[1]["card_ref"]
    for card in cards[:2]:
        assert any(
            row.get("ordered_card_refs") == [card["card_ref"]]
            for row in duplicate["ordered_actions"]
        )
    empty = decision("empty-events")
    assert empty["delta"]["events"] == []
    assert empty["frame"]["interval_start"] == empty["frame"]["boundary"]["end"] == 0
    for name, mp in (("dynamic-weapon-zero-mp", 0), ("dynamic-weapon-positive-mp", 10)):
        row = decision(name)
        assert row["frame"]["payload"]["observation"]["own"]["mp"] == mp
        assert any(
            action.get("ordered_card_refs") == row["references"]
            for action in row["ordered_actions"]
        )
    large = decision("large-candidate-table")
    assert len(large["generated"]["table"]["kinds"]) >= 1024
    assert large["search_rng_before"] != large["search_rng_after"]
    cutoffs = decision("clock-cutoffs")["generated"]["report"]["cutoff_reasons"]
    assert "count_time" in cutoffs and "search_time" in cutoffs
    history = episodes["refresh-history"]["decisions"]
    assert len(history) >= 4
    assert all(len(row["policies"]["refresh"]["repeated_selections"]) == 2 for row in history)
    assert history[0]["policies"]["refresh"]["action"]["type"] == "play_sequence"
    assert history[-1]["policies"]["refresh"]["action"]["type"] == "play_sequence"
    assert any(
        event["type"] == "damage_applied" for row in history for event in row["delta"]["events"]
    )
    assert any(
        event["type"] == "turn_started" for row in history for event in row["delta"]["events"]
    )


def test_refresh_original_history_changes_a_subthreshold_decision():
    episode = next(row for row in originals()["episodes"] if row["name"] == "refresh-history")
    actions = action_config_v3(episode["actions"])
    rules = parse_card_instance_rules(episode["rules"])
    bank = EventBank(episode["actor"], episode["name"], layout=LAYOUT)
    for row in episode["decisions"]:
        bank.append(parse_delta(row["delta"], layout=LAYOUT))
    final = episode["decisions"][-1]
    frame = parse_frame(final["frame"], layout=LAYOUT)
    generated = parse_game_generated_v3(
        final["generated"], action_contract=actions.action_contract, rules=rules
    )
    fresh = RefreshHeuristicPolicy(actions, rules=rules, options=RefreshOptions())
    fresh.reset(episode["sampling_seed"])
    assert fresh.select_frame(frame, bank.view(), generated).validate(generated.table, frame) == {
        "type": "pray"
    }
    assert final["policies"]["refresh"]["action"] == {
        "type": "play_sequence",
        "ordered_card_refs": final["references"],
        "target": 2,
    }
