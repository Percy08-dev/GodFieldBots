"""Rust recipe differential checks against frozen pre-native feature bytes."""
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models import encoder
from gf_models.config import EncoderConfig
from tests.test_prepared_encoder import reference_actions, reference_input

native = pytest.importorskip("gf_native")
BUDGET = 16 * 1024 * 1024


def packed(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()


def native_encode(value, actions, config):
    return native.encode_features(packed(value.to_dict()), packed(actions), packed(asdict(config)),
                                  BUDGET, time.monotonic() + 10)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("config", [EncoderConfig(hash_buckets=2, max_elements=2),
                                    EncoderConfig(history="public_aggregate", history_window=2),
                                    EncoderConfig(history="public_aggregate", history_window=1, hash_buckets=2)])
def test_native_legacy_feature_bytes(make_input, sparse, config):
    references = json.loads((Path(__file__).parent / "fixtures" / "encoder_legacy_bytes.json").read_text())
    expected = references[f"{sparse}:{config}"]
    value = reference_input(make_input, sparse)
    matrix = native_encode(value, reference_actions(), config)
    assert matrix.dtype == np.float64
    assert not matrix.flags.writeable
    assert hashlib.sha256(matrix.tobytes()).hexdigest() == expected["float64"]
    names = native.feature_names(packed(asdict(config)), BUDGET, time.monotonic() + 10)
    assert tuple(names) == encoder.feature_names(config)
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == expected["names"]
    np.testing.assert_array_equal(matrix, encoder.PreparedDecision(value, config).encode_many(reference_actions()))


@pytest.mark.parametrize("action", [
    {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": True}]},
    {"type": "discard", "instance_ids": [999]},
    {"type": "buy_decision", "accept": 1},
    {"type": "pray", "hidden": 0},
    {"type": "play_sequence", "ordered_elements": [], "target": True},
    {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}] * 4},
    {"type": "respond_sequence", "ordered_defenses": [{"source": "learned_miracle", "definition_id": "foreign"}]},
])
def test_native_action_errors_match_reference(make_input, action):
    value = make_input()
    with pytest.raises(ContractError):
        encoder.encode(value, action, EncoderConfig())
    with pytest.raises(ValueError):
        native_encode(value, (action,), EncoderConfig())


def test_native_distinct_catalog_entries_and_legacy_nullable_target(make_input):
    value = make_input(("bronze-club", "god-sword"))
    value.catalog["god-sword"]["definition_id"] = "bronze-club"
    # Existing encoder allows explicit null even on the required buy target.
    actions = ({"type": "discard", "instance_ids": [1]},
               {"type": "discard", "instance_ids": [2]},
               {"type": "buy", "artifact": 1, "target": None})
    config = EncoderConfig()
    expected = encoder.PreparedDecision(value, config).encode_many(actions)
    actual = native_encode(value, actions, config)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[0], actual[1])


def test_native_catalog_lookup_uses_perceived_keys_for_hand_and_learned(make_input):
    value = make_input(("bronze-club", "god-sword"), learned=("absorption",))
    value.catalog["god-sword"]["definition_id"] = "bronze-club"
    value.catalog["absorption"]["definition_id"] = "bronze-club"
    # Deliberately reverse insertion order; only mapping keys identify a card.
    value = replace(value, catalog=dict(reversed(tuple(value.catalog.items()))))
    actions = (
        {"type": "play_sequence", "ordered_elements": [
            {"source": "hand_artifact", "instance_id": 2},
            {"source": "learned_miracle", "definition_id": "absorption"}], "target": 2},
    )
    config = EncoderConfig(history="public_aggregate")
    np.testing.assert_array_equal(native_encode(value, actions, config),
                                  encoder.PreparedDecision(value, config).encode_many(actions))


@pytest.mark.parametrize("field,invalid", [("max_elements", True), ("max_elements", 2.0),
                                            ("hash_buckets", 2.0), ("history_window", 0),
                                            ("history", "unsupported")])
def test_native_config_type_and_range_checks(field, invalid):
    config = asdict(EncoderConfig())
    config[field] = invalid
    with pytest.raises(ValueError):
        native.feature_names(packed(config), BUDGET, time.monotonic() + 10)


@pytest.mark.parametrize("events", ["", {}])
def test_native_features_preserve_empty_event_iterable(make_input, events):
    value = replace(make_input(), events=events)
    config = EncoderConfig(history="public_aggregate")
    actions = ({"type": "pray"},)
    expected = encoder.encode(value, actions[0], config)
    np.testing.assert_array_equal(native_encode(value, actions, config)[0], expected)


def test_native_feature_metadata_budget_and_deadline_admission():
    config = packed(asdict(EncoderConfig(history="public_aggregate", hash_buckets=4096)))
    with pytest.raises(MemoryError):
        native.feature_names(config, 65536, time.monotonic() + 10)
    with pytest.raises(TimeoutError):
        native.feature_names(config, BUDGET, time.monotonic() - 1)
