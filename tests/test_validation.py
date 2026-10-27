import copy

import pytest

from gf_common.domain import ContractError, DecisionInput, UnsupportedContract
from gf_common.validation import catalog, decision_input, observation, validate_worker_input
from tests.conftest import fixture


def test_current_rust_metadata_matches_and_private_fields_forbidden():
    valid = fixture("observation_p1_public.json")
    assert observation(valid) == valid
    for field in ("action_options", "SequenceOptions", "game_seed", "snapshot"):
        value = copy.deepcopy(valid)
        value[field] = {}
        with pytest.raises(ContractError):
            observation(value)
    value = copy.deepcopy(valid)
    value["typed_action_schema"][0]["required_fields"] = {"hidden": {"type": "boolean"}}
    with pytest.raises(UnsupportedContract):
        observation(value)


@pytest.mark.parametrize("path,value", [
    (("turn",), 2**32), (("turns_until_end_of_world",), 2**32),
    (("state_revision",), True), (("own", "hp"), 2**31),
    (("own", "hp"), -(2**31) - 1), (("opponent", "hp"), 2**31),
    (("own", "mp"), 2**32), (("own", "money"), True),
])
def test_numeric_widths(path, value):
    obs = fixture("observation_p1_public.json")
    target = obs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ContractError):
        observation(obs)


def test_missing_fog_values_preserved(definitions):
    obs = fixture("observation_p1_public.json")
    for key in ("hp", "mp", "money", "status"):
        obs["opponent"].pop(key, None)
    value = decision_input(obs, [], None, definitions)
    assert "hp" not in value.observation["opponent"]
    assert set(value.catalog) == {c["definition_id"] for c in obs["own"]["hand"]}
    assert "bronze-club" not in value.catalog


def test_worker_rejects_foreign_catalog_and_rejection_events(make_input, definitions):
    value = make_input()
    raw = value.to_dict()
    raw["catalog"]["ice"] = definitions["ice"]
    leaked = DecisionInput(raw["observation"], (), None, value.typed_metadata, raw["catalog"])
    with pytest.raises(ContractError):
        validate_worker_input(leaked)
    with pytest.raises(ContractError):
        decision_input(value.observation, [{"type": "rejection", "legal": True}], None, definitions)


def test_dream_perceived_id_is_preserved(make_input, definitions):
    value = make_input(("bronze-club",))
    value.observation["own"]["status"]["dream"] = True
    rebuilt = decision_input(value.observation, [], None, definitions)
    assert rebuilt.catalog == {"bronze-club": definitions["bronze-club"]}


def test_catalog_hash_and_integer_types():
    value = fixture("catalog_public.json")
    assert len(catalog(value)) == 296
    for field, item in (("attack", 2**31), ("price", -1), ("attack", True)):
        mutated = copy.deepcopy(value)
        mutated["definitions"][0][field] = item
        with pytest.raises(ContractError):
            catalog(mutated)


def test_unsupported_additional_choice(make_input):
    value = make_input()
    value.observation["required_choice"] = "additional_choice"
    value.observation["actionable"] = False
    value.observation["typed_action_schema"] = []
    with pytest.raises(UnsupportedContract):
        observation(value.observation)
