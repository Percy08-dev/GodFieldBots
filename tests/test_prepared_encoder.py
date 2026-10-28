"""Prepared public features retain frozen legacy bytes and isolate caller data."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, ContractError
from gf_models import encoder
from gf_models.config import EncoderConfig
from gf_models.mlp.features import critic_indices, encode_decision, signed_log


def reference_input(make_input, sparse):
    value = make_input(("bronze-club", "god-sword", "bronze-club"), learned=("absorption",))
    value.observation["own"]["status"]["dream"] = True
    value = replace(value, events=(
        {"type": "resource_changed", "player": 1, "hp_delta": -7, "mp_delta": 0, "money_delta": 9},
        {"type": "artifact_used", "player": 2, "definition_id": "日本語・公開"},
        {"type": "miracle_used", "player": 1, "definition_id": "absorption", "mp_spent": 10},
    ))
    if sparse:
        for key in ("hp", "mp", "money", "status", "guardian"):
            value.observation["opponent"].pop(key, None)
        value = replace(value, action_options=None)
    return value


def reference_actions():
    hand = {"source": "hand_artifact", "instance_id": 1}
    learned = {"source": "learned_miracle", "definition_id": "absorption"}
    return (
        {"type": "pray"},
        {"type": "play_sequence", "ordered_elements": [hand, learned], "target": 2},
        {"type": "play_sequence", "ordered_elements": [learned, hand], "target": 1},
        {"type": "respond", "ordered_defenses": [1, 1]},
        {"type": "respond_sequence", "ordered_defenses": [hand, learned]},
        {"type": "discard", "instance_ids": [3, 1]},
        {"type": "sacrifice", "instance_ids": [1]},
        {"type": "exchange", "artifact": 1, "hp": -2, "mp": 0, "money": 4294967295},
        {"type": "sell", "artifact": 1, "sold_artifact": 2, "target": 2},
        {"type": "buy", "artifact": 2, "target": 1},
        {"type": "buy_decision", "accept": True},
        {"type": "buy_decision", "accept": False},
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("config", [EncoderConfig(hash_buckets=2, max_elements=2),
                                    EncoderConfig(history="public_aggregate", history_window=2),
                                    EncoderConfig(history="public_aggregate", history_window=1, hash_buckets=2)])
def test_legacy_feature_and_mlp_bytes(make_input, sparse, config):
    # Generated using the archived pre-ADR0010 encoder, never the prepared path.
    references = json.loads((Path(__file__).parent / "fixtures" / "encoder_legacy_bytes.json").read_text())
    key = f"{sparse}:{config}"
    expected = references[key]
    value = reference_input(make_input, sparse)
    actions = reference_actions()
    prepared = encoder.PreparedDecision(value, config)
    matrix = prepared.encode_many(actions)
    actor, state = encode_decision(value, actions, config)
    assert hashlib.sha256(matrix.tobytes()).hexdigest() == expected["float64"]
    assert hashlib.sha256(actor.tobytes()).hexdigest() == expected["float32"]
    assert hashlib.sha256(state.tobytes()).hexdigest() == expected["state"]
    assert encoder.signature(config) == expected["signature"]
    assert hashlib.sha256("\n".join(encoder.feature_names(config)).encode()).hexdigest() == expected["names"]
    np.testing.assert_array_equal(matrix, np.stack([encoder.encode(value, action, config) for action in actions]))
    np.testing.assert_array_equal(actor, signed_log(matrix))
    np.testing.assert_array_equal(prepared.encode_many(actions[::-1]), matrix[::-1])
    np.testing.assert_array_equal(state, signed_log(prepared.encode({"type": "pray"}))[list(critic_indices(config))])


def test_prepared_snapshot_and_outputs_have_independent_ownership(make_input):
    value = reference_input(make_input, False)
    actions = reference_actions()
    config = EncoderConfig(history="public_aggregate")
    prepared = encoder.PreparedDecision(value, config)
    expected = prepared.encode_many(actions)
    value.observation["own"]["hand"][0]["instance_id"] = 999
    value.observation["own"]["hp"] = 1
    value.events[0]["hp_delta"] = -100
    value.catalog["bronze-club"]["attack"] = 9999
    value.action_options["action_options"]["usable_hand_elements"].clear()
    value.typed_metadata[0]["type"] = "invalid"
    np.testing.assert_array_equal(prepared.encode_many(actions), expected)
    changed = prepared.encode_many(actions)
    changed[:] = 123
    np.testing.assert_array_equal(prepared.encode_many(actions), expected)
    # Original input no longer passes validation; the snapshot cannot bless it.
    with pytest.raises(ContractError):
        encoder.PreparedDecision(value, config)
    with pytest.raises(ContractError, match="foreign"):
        prepared.encode({"type": "discard", "instance_ids": [999]})


@pytest.mark.parametrize("field,invalid", [("history_window", True), ("max_elements", 2.0),
                                            ("hash_buckets", 2.0), ("history_window", 2.0),
                                            ("history", "unsupported")])
def test_config_validation_precedes_template_cache(make_input, field, invalid):
    valid = EncoderConfig(max_elements=2, hash_buckets=2, history_window=1)
    encoder.feature_names(valid)
    invalid_config = replace(valid, **{field: invalid})
    for operation in (lambda: encoder.feature_names(invalid_config),
                      lambda: encoder.signature(invalid_config),
                      lambda: encoder.PreparedDecision(make_input(), invalid_config)):
        with pytest.raises(ContractError):
            operation()


def test_inference_reuses_one_validation_and_common_computation(make_input, monkeypatch):
    value = reference_input(make_input, False)
    config = EncoderConfig(history="public_aggregate")
    encoder.feature_names(config)  # Template generation is independent of decision work.
    validation = encoder.validate_worker_input
    features = encoder._features
    counts = {"validate": 0, "common": 0}
    def validate(value):
        counts["validate"] += 1
        return validation(value)
    def calculate(*args, **kwargs):
        counts["common"] += kwargs.get("common", True)
        return features(*args, **kwargs)
    monkeypatch.setattr(encoder, "validate_worker_input", validate)
    monkeypatch.setattr(encoder, "_features", calculate)
    encode_decision(value, reference_actions(), config)
    assert counts == {"validate": 1, "common": 1}

    from gf_models.lightgbm.policy import LightGBMPolicy
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    class Booster:
        def predict(self, features, *, num_threads):
            assert num_threads == 1
            return features[:, 0]
    policy = object.__new__(LightGBMPolicy)
    policy.booster = Booster()
    policy.encoder = config
    policy.manifest = {"action_config_hash": candidates.config_hash,
                       "generator_version": candidates.generator_version,
                       "scorer_version": candidates.scorer_version}
    counts.update(validate=0, common=0)
    policy.scores(value, candidates)
    assert counts == {"validate": 1, "common": 1}


@pytest.mark.parametrize("action", [
    {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": True}]},
    {"type": "discard", "instance_ids": [999]},
    {"type": "buy_decision", "accept": 1},
    {"type": "pray", "hidden": 0},
    {"type": "play_sequence", "ordered_elements": [], "target": True},
    {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}] * 4},
    {"type": "respond_sequence", "ordered_defenses": [{"source": "learned_miracle", "definition_id": "foreign"}]},
])
def test_prepared_actions_preserve_rejection_checks(make_input, action):
    prepared = encoder.PreparedDecision(make_input(), EncoderConfig())
    with pytest.raises(ContractError):
        prepared.encode(action)


@pytest.mark.parametrize("mutation", ["revision", "viewer", "metadata", "catalog", "events", "opponent"])
def test_preparation_revalidates_the_exact_boundary(make_input, mutation):
    value = make_input()
    if mutation == "revision":
        value.action_options["state_revision"] += 1
    elif mutation == "viewer":
        value.action_options["viewer_player_id"] = 2
    elif mutation == "metadata":
        value = replace(value, typed_metadata=())
    elif mutation == "catalog":
        value.catalog["foreign"] = value.catalog["bronze-club"]
    elif mutation == "events":
        value = replace(value, events=({"type": "action_accepted", "player": 1, "hidden": True},))
    else:
        value.observation["opponent"]["hand"] = []
    with pytest.raises(ContractError):
        encoder.PreparedDecision(value, EncoderConfig())


def test_distinct_catalog_entries_are_never_conflated_by_definition_id(make_input):
    # The existing worker validator accepts this mapping. Optimization must keep
    # its current feature values rather than invent a stricter catalog contract.
    value = make_input(("bronze-club", "god-sword"))
    value.catalog["god-sword"]["definition_id"] = "bronze-club"
    actions = ({"type": "discard", "instance_ids": [1]}, {"type": "discard", "instance_ids": [2]})
    prepared = encoder.PreparedDecision(value, EncoderConfig())
    np.testing.assert_array_equal(prepared.encode_many(actions),
                                  np.stack([encoder.encode(value, action, EncoderConfig()) for action in actions]))


def test_owned_validated_sink_preserves_public_checks(make_input, monkeypatch):
    value = encoder.validate_worker_input(make_input())
    config = EncoderConfig(hash_buckets=2)
    expected = encoder.encode(value, {"type": "pray"}, config)
    def forbidden(value):
        pytest.fail('trusted sink must not rebuild its owned validated input')
    monkeypatch.setattr(encoder, 'validate_worker_input', forbidden)
    np.testing.assert_array_equal(encoder._encode_validated(value, {"type": "pray"}, config), expected)
    with pytest.raises(ContractError):
        encoder._encode_validated(value, {"type": "pray"}, replace(config, hash_buckets=2.0))
    with pytest.raises(ContractError, match='foreign'):
        encoder._encode_validated(value, {"type": "discard", "instance_ids": [999]}, config)
