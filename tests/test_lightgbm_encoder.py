from dataclasses import replace

import numpy as np
import pytest

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, ContractError
from gf_models.lightgbm.config import EncoderConfig
from gf_models.lightgbm.encoder import encode, feature_names, signature


def features(value, action, config=None):
    config = config or EncoderConfig()
    return dict(zip(feature_names(config), encode(value, action, config), strict=True))


def test_handcomputed_order_missingness_multiplicity_and_no_numeric_ids(make_input):
    value = make_input(("bronze-club", "god-sword", "bronze-club"))
    action = {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1},
              {"source": "hand_artifact", "instance_id": 2}], "target": 2}
    f = features(value, action)
    assert f["own.hp"] == value.observation["own"]["hp"]
    assert f["hand.kind.weapons"] == 3
    assert f["hand.attack"] == sum(value.catalog[c["definition_id"]]["attack"] for c in value.observation["own"]["hand"])
    assert f["element.0.attack"] == value.catalog["bronze-club"]["attack"]
    assert f["element.1.attack"] == value.catalog["god-sword"]["attack"]
    assert f["element.2.present"] == 0
    assert f["target.opponent"] == 1
    assert f["element.0.mp_cost.missing"] == 1
    renamed = make_input(("bronze-club", "god-sword", "bronze-club"))
    for card in renamed.observation["own"]["hand"]:
        card["instance_id"] += 10000
    renamed.action_options["action_options"]["usable_hand_elements"] = [10001, 10002, 10003]
    renamed.observation["state_revision"] = 987
    renamed.action_options["state_revision"] = 987
    renamed = replace(renamed, history_end="different-control-cursor")
    equivalent = {**action, "ordered_elements": [{"source": "hand_artifact", "instance_id": 10001}, {"source": "hand_artifact", "instance_id": 10002}]}
    np.testing.assert_array_equal(encode(value, action, EncoderConfig()), encode(renamed, equivalent, EncoderConfig()))
    reversed_action = {**action, "ordered_elements": list(reversed(action["ordered_elements"]))}
    assert features(value, reversed_action)["element.0.attack"] == f["element.1.attack"]


def test_hidden_missing_opponent_values_are_distinct_from_observed_zero(make_input):
    value = make_input()
    value.observation["opponent"].pop("hp")
    missing = features(value, {"type": "pray"})
    value.observation["opponent"]["hp"] = 0
    zero = features(value, {"type": "pray"})
    assert missing["opponent.hp"] == zero["opponent.hp"] == 0
    assert missing["opponent.hp.missing"] == 1 and zero["opponent.hp.missing"] == 0


def test_history_window_relative_roles_and_signature(make_input):
    value = make_input()
    events = ({"type": "turn_started", "turn": 2, "active_player": 2},
              {"type": "damage_applied", "player": 1, "amount": 4},
              {"type": "damage_applied", "player": 2, "amount": 7})
    value = replace(value, events=events)
    config = EncoderConfig(history="public_aggregate", history_window=2)
    f = features(value, {"type": "pray"}, config)
    assert f["history.window_truncated"] == 1 and f["history.count"] == 2
    assert f["history.own.amount"] == 4 and f["history.opponent.amount"] == 7
    assert f["history.type.turn_started"] == 0
    assert signature(config) != signature(EncoderConfig())
    assert not any("history." in name for name in feature_names(EncoderConfig()))


@pytest.mark.parametrize("mutation", ["oversized", "foreign", "extra", "hidden"])
def test_encoder_rejects_unsupported_or_leaking_inputs(make_input, mutation):
    value = make_input()
    action = {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}], "target": 2}
    if mutation == "oversized":
        action["ordered_elements"] *= 4
    elif mutation == "foreign":
        action["ordered_elements"][0]["instance_id"] = 999
    elif mutation == "extra":
        action["game_seed"] = "forbidden"
    else:
        value.observation["opponent"]["hand"] = []
    with pytest.raises(ContractError):
        encode(value, action, EncoderConfig())


def test_dream_own_perceived_only_hints_and_candidate_permutation(make_input):
    value = make_input(("bronze-club", "god-sword"))
    value.observation["own"]["status"]["dream"] = True
    candidates = PublicCandidateGenerator(ActionConfig(max_sequence_length=1)).generate(value)
    matrix = np.stack([encode(value, action, EncoderConfig()) for action in candidates.actions])
    reversed_matrix = np.stack([encode(value, action, EncoderConfig()) for action in reversed(candidates.actions)])
    np.testing.assert_array_equal(matrix[::-1], reversed_matrix)
    assert features(value, candidates.actions[0])["own.dream"] == 1
    assert features(value, candidates.actions[0])["hints.missing"] == 0
    no_hints = replace(value, action_options=None)
    assert features(no_hints, candidates.actions[0])["hints.missing"] == 1
    assert not any(token in name for name in feature_names(EncoderConfig()) for token in ("revision", "cursor", "seed", "instance_id", "candidate_index", "game_id"))
