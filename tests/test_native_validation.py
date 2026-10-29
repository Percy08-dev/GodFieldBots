"""Native actor-boundary differential tests; no simulator or live source work."""
import copy
import time
from dataclasses import replace

import pytest

from gf_common.domain import ContractError, DecisionInput
from gf_common.validation import validate_worker_input
from tests.test_native_features import BUDGET, packed

native = pytest.importorskip("gf_native")


def reference_validate(raw, mode=None):
    value = DecisionInput(raw["observation"], raw["events"], raw["history_end"],
                          raw["typed_metadata"], raw["catalog"], raw["action_options"])
    validate_worker_input(value, mode)


def check(raw, mode=None):
    try:
        reference_validate(raw, mode)
    except (ContractError, TypeError, KeyError, ValueError):
        with pytest.raises(ValueError):
            native.validate_input(packed(raw), mode, BUDGET, time.monotonic() + 10)
        return False
    native.validate_input(packed(raw), mode, BUDGET, time.monotonic() + 10)
    return True


@pytest.mark.parametrize("path,item", [
    (("observation", "state_revision"), True),
    (("observation", "turn"), 2**32),
    (("observation", "own", "hp"), 2**31),
    (("observation", "own", "hp"), None),
    (("observation", "own", "money"), True),
    (("observation", "own", "money"), 1.0),
    (("observation", "opponent", "hp"), -2**31 - 1),
    (("observation", "opponent", "hand"), []),
    (("observation", "SequenceOptions"), {}),
    (("observation", "game_seed"), 1),
    (("action_options", "state_revision"), 1234),
    (("action_options", "viewer_player_id"), 2),
    (("action_options", "action_options", "usable_hand_elements"), [999]),
    (("action_options", "action_options", "usable_hand_elements"), [1, 1]),
    (("action_options", "action_options", "usable_hand_elements"), [True]),
    (("action_options", "action_options", "usable_learned_miracles"), ["foreign"]),
    (("action_options", "action_options", "valid_targets"), [3]),
    (("action_options", "action_options", "can_pray"), 0),
    (("typed_metadata",), []),
    (("events",), [{"type": "rejection", "legal": True}]),
    (("events",), [{"type": "action_accepted", "player": 1, "hidden": True}]),
    (("events",), [{"type": "resource_changed", "player": 1, "hp_delta": 0,
                    "mp_delta": 2**31, "money_delta": 0}]),
])
def test_native_rejects_same_actor_boundary_violations(make_input, path, item):
    raw = make_input().to_dict()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = item
    assert not check(raw)


def test_native_catalog_perceived_identity_and_missing_fog(make_input):
    value = make_input(("bronze-club", "god-sword"))
    raw = value.to_dict()
    raw["catalog"]["foreign"] = copy.deepcopy(raw["catalog"]["bronze-club"])
    assert not check(raw)
    raw = value.to_dict()
    raw["catalog"].pop("god-sword")
    assert not check(raw)
    raw = value.to_dict()
    raw["catalog"]["god-sword"]["definition_id"] = "bronze-club"
    raw["observation"]["own"]["status"]["dream"] = True
    for key in ("hp", "mp", "money", "status", "guardian"):
        raw["observation"]["opponent"].pop(key, None)
    assert check(raw)


@pytest.mark.parametrize("mode", [None, False, True])
@pytest.mark.parametrize("present", [False, True])
def test_native_atomic_presence_mode(make_input, mode, present):
    value = make_input()
    if not present:
        value = replace(value, action_options=None)
    assert check(value.to_dict(), mode) == (mode is None or mode == present)


def test_serde_contract_rejects_unused_surrogates_without_changing_python_reader(make_input):
    raw = make_input().to_dict()
    raw["catalog"]["bronze-club"]["display_name"] = "\ud800"
    raw["catalog"]["bronze-club"]["effect_text"] = "\udc00"
    raw["events"] = [{"type": "status_changed", "player": 1, "status": "\ud800", "active": False}]
    reference_validate(raw)
    with pytest.raises(ValueError):
        native.validate_input(packed(raw), None, BUDGET, time.monotonic() + 10)


def test_native_duplicate_json_keys_keep_last_value(make_input):
    raw = packed(make_input().to_dict())
    encoded = raw.replace(b'"state_revision":0', b'"state_revision":false,"state_revision":0')
    native.validate_input(encoded, None, BUDGET, time.monotonic() + 10)


@pytest.mark.parametrize("events,expected", [("", True), ({}, True), ([], True), ("x", False), ({"type": "action_accepted"}, False)])
def test_native_worker_event_iterable_coercion(make_input, events, expected):
    raw = make_input().to_dict()
    raw["events"] = events
    assert check(raw) == expected


@pytest.mark.parametrize("metadata,expected", [("", False), ({}, False), ([], True)])
def test_native_worker_metadata_is_not_journal_tuple_coercion(make_input, metadata, expected):
    raw = make_input().to_dict()
    raw["observation"]["phase"] = "terminal"
    raw["observation"]["actionable"] = False
    raw["observation"]["typed_action_schema"] = []
    raw["typed_metadata"] = metadata
    raw["action_options"] = None
    assert check(raw) == expected
