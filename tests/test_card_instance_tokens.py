"""Exact new tokens, public-time facts and strict old-consumer permission."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT as V2
from gf_common.structured_layout import LEGACY_LAYOUT, USE_ROLES
from gf_common.structured_tokens import (
    MISSING,
    STATE_NUMERIC_FIELDS,
    category,
    recorded_aggregate_recipe,
    signature,
    signature_payload,
    tokenize_frame,
    tokenize_table,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from tests.test_card_instance_structured import frame_for, make_input, table_for


def features(tokens):
    return {
        key: value
        for key, value in asdict(tokens).items()
        if key not in ("frame_hash", "actor", "episode", "revision", "event_start", "event_end")
    }


def test_signature_closed_layout_and_entire_declared_recipe():
    payload = signature_payload(layout=V2)
    assert payload["action_contract"] == asdict(V2.action_contract)
    assert len(payload["history_fields"]) == 377
    assert payload["card_numeric"] == V2.card_numeric_fields
    assert payload["card_categories"] == V2.card_categories
    assert payload["format_versions"] == V2.format_versions
    assert V2.card_numeric_fields[8] == "possession_position"
    assert V2.card_numeric_fields[13:] == tuple("role." + name for name in sorted(USE_ROLES))
    assert signature() != signature(layout=V2)
    assert recorded_aggregate_recipe(signature(layout=V2), 128, layout=V2) == V2.aggregate_recipe
    with pytest.raises(ContractError):
        recorded_aggregate_recipe(signature(layout=V2), 128)
    for layout, aggregate in (
        (V2, LEGACY_LAYOUT.aggregate_recipe),
        (LEGACY_LAYOUT, V2.aggregate_recipe),
    ):
        with pytest.raises(ContractError):
            signature(layout=layout, aggregate_recipe=aggregate)
    with pytest.raises(ContractError):
        signature(layout=replace(V2, card_numeric_fields=()))


def test_hand_authored_numeric_kind_state_membership_and_static_role_features():
    value = make_input([("bronze-club", None), ("fireball", "unused"), ("fireball", "learned")])
    frame, bank, _ = frame_for(value)
    tokens = tokenize_frame(frame, bank.view(), layout=V2)
    assert tokens.card_numerics[0][7:13] == (1, 1, 0, 0, 0, 0)
    assert tokens.card_numerics[1][7:13] == (1, 2, 0, 0, 0, 0)
    assert tokens.card_numerics[2][7:13] == (1, 3, 0, 0, 0, 0)
    assert tokens.card_present[0][7:] == (True,) * 19
    assert tokens.card_categories[0][6:] == (2, MISSING, 2, MISSING, 2)
    assert tokens.card_categories[1][6:] == (3, 2, 2, 3, 3)
    assert tokens.card_categories[2][6:] == (3, 3, 2, 3, 4)
    assert tokens.card_numerics[0][13:] == tuple(int(role == "weapon") for role in USE_ROLES)
    assert tokens.card_numerics[1][13:] == tuple(
        int(role in ("additional_attack", "miracle:additional_weapon")) for role in USE_ROLES
    )
    assert tokens.card_categories[1][:6] == tokens.card_categories[2][:6]
    assert all(row == (0,) * 26 for row in tokens.card_numerics[3:])
    assert all(row == (0,) * 11 for row in tokens.card_categories[3:])
    validate_frame_tokens(tokens, layout=V2)
    with pytest.raises(ContractError):
        validate_frame_tokens(tokens)
    with pytest.raises(ContractError):
        tokenize_frame(frame, bank.view())


def test_ref_bytes_and_unknown_group_strings_are_never_features():
    value = make_input([("bronze-club", None), ("fireball", "learned")])
    first, bank, _ = frame_for(value)
    original = tokenize_frame(first, bank.view(), layout=V2)
    changed = copy.deepcopy(value)
    mapping = {
        card["card_ref"]: "c1_" + format(2**128 - index - 1, "032x")
        for index, card in enumerate(changed.observation["own"]["cards"])
    }
    for card in changed.observation["own"]["cards"]:
        card["card_ref"] = mapping[card["card_ref"]]
    body = changed.action_options["action_options"]
    for row in body["usable_cards"]:
        row["card_ref"] = mapping[row["card_ref"]]
    body["candidate_groups"] = {"unrelated-new-group": copy.deepcopy(body["usable_cards"])}
    second, second_bank, _ = frame_for(changed, episode="another")
    assert second.frame_hash != first.frame_hash
    assert features(tokenize_frame(second, second_bank.view(), layout=V2)) == features(original)


def test_fog_missingness_dream_and_past_definitions_keep_original_time():
    value = make_input([("angel-axe", None)])
    value.observation["own"]["status"].update(fog=True, dream=True)
    value.observation["opponent"] = {
        "player_id": 2,
        "learned_miracles": ["aura", "aura"],
        "ascended": False,
    }
    value = replace(
        value, events=({"type": "artifact_used", "player": 2, "definition_id": "bronze-club"},)
    )
    frame, bank, _ = frame_for(value)
    tokens = tokenize_frame(frame, bank.view(), layout=V2)
    assert tokens.state_present[STATE_NUMERIC_FIELDS.index("opponent.hp")] is False
    assert tokens.state_categories[-2:] == (MISSING, MISSING)
    assert tokens.state_category_present[-2:] == (False, False)
    assert tokens.interval_categories[0][1] == category("bronze-club", V2.card_categories[0])
    assert tokens.card_categories[0][0] == category("angel-axe", V2.card_categories[0])
    assert set(frame.raw["catalog"]) == {"angel-axe"}


def test_safe_history_exact_tail_signed_values_and_bounded_window():
    events = (
        {"type": "artifact_used", "player": 2, "definition_id": "aura"},
        {"type": "safe", "player": 1},
        {"type": "resource_changed", "player": 2, "hp_delta": -3, "mp_delta": 2, "money_delta": 0},
    )
    frame, bank, _ = frame_for(make_input(events=events))
    tokens = tokenize_frame(frame, bank.view(), aggregate_window=2, layout=V2)
    assert len(tokens.history) == 377 and tokens.history[24] == 1
    assert tokens.history[14] == 1 and tokens.history[2] == 0
    # 25 event counts + 16 actor counts; hp_delta is numeric field7.
    assert tokens.history[41 + 7 * 2 : 41 + 7 * 2 + 2] == (-3, 1)
    # Last314 fields preserve old definition/window/actor semantics exactly.
    assert tokens.history[63 + 296 : 63 + 298] == (2, 1)
    assert tokens.history[63 + 298 :] == (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, -3, 2, 0, 0, 0, 0)
    validate_frame_tokens(tokens, layout=V2)
    with pytest.raises(ContractError):
        validate_frame_tokens(replace(tokens, card_numerics=((True,) * 26,) * 18), layout=V2)


def test_candidate_order_equivariance_codebooks_and_old_consumer_guards():
    frame, bank, _ = frame_for(make_input([("bronze-club", None), ("fireball", "learned")]))
    actions = [
        {"type": "pray"},
        {"type": "play_sequence", "ordered_card_refs": list(frame.references[::-1]), "target": 2},
        {"type": "invoke_phenomenon", "card_ref": frame.references[0]},
    ]
    tokens = tokenize_frame(frame, bank.view(), layout=V2)
    table = table_for(frame, actions)
    candidates = tokenize_table(table, frame, layout=V2)
    validate_candidate_tokens(candidates, tokens, 18, layout=V2)
    assert candidates.kinds == (2, 3, 11)
    assert candidates.slots == (1, 0, 0)
    with pytest.raises(ContractError):
        validate_candidate_tokens(candidates, tokens, 18)
    with pytest.raises(ContractError):
        tokenize_table(table, frame)
    reversed_table = table_for(frame, actions[::-1])
    reversed_tokens = tokenize_table(reversed_table, frame, layout=V2)
    assert reversed_tokens.kinds == candidates.kinds[::-1]
    assert [reversed_table.to_action(i, frame) for i in range(3)] == actions[::-1]


def test_exact_new_token_golden():
    events = (
        {"type": "safe", "player": 1},
        {"type": "miracle_used", "player": 2, "definition_id": "aura", "mp_spent": 3},
    )
    frame, bank, _ = frame_for(
        make_input(
            [("bronze-club", None), ("fireball", "unused"), ("fireball", "learned")], events=events
        )
    )
    tokens = tokenize_frame(frame, bank.view(), layout=V2, aggregate_window=2)
    expected = json.loads(
        (Path(__file__).parent / "fixtures/card_instance_v2_tokens_golden.json").read_text()
    )
    assert canonical(asdict(tokens)) == canonical(expected["frame_tokens"])
    assert digest(signature_payload(layout=V2)) == expected["default_signature"]
