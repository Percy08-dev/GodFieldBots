from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gf_common.validation import action_descriptors, catalog, decision_input

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def public_options(obs, definitions):
    """Synthetic atomic script, independently overridden in mask tests."""
    hand = obs["own"]["hand"]
    turn = obs["required_choice"] == "turn_action"
    return {"viewer_player_id": obs["viewer_player_id"], "state_revision": obs["state_revision"],
            "action_options": {
                "required_choice": obs["required_choice"],
                "usable_hand_elements": [c["instance_id"] for c in hand],
                "usable_learned_miracles": list(obs["own"]["learned_miracles"]),
                **{ability + "_artifacts": [c["instance_id"] for c in hand if definitions[c["definition_id"]].get("ability") == ability]
                   for ability in ("exchange", "sell", "buy")},
                "valid_targets": [1, 2],
                "can_pray": turn and not any(definitions[c["definition_id"]]["kind"] == "weapons" for c in hand),
                "can_discard": turn, "can_sacrifice": turn,
                "can_allow": obs["required_choice"] == "defense"}}


@pytest.fixture
def definitions():
    return catalog(fixture("catalog_public.json"))


@pytest.fixture
def make_input(definitions):
    def make(hand=("bronze-club",), learned=(), choice="turn_action", mp=50, flash=False):
        obs = fixture("observation_p1_public.json")
        obs["own"]["hand"] = [{"instance_id": i + 1, "definition_id": name} for i, name in enumerate(hand)]
        obs["own"]["learned_miracles"] = list(learned)
        obs["own"]["mp"] = mp
        obs["own"]["status"]["flash"] = flash
        obs["required_choice"] = choice
        obs["typed_action_schema"] = action_descriptors(choice)
        obs["phase"] = "awaiting_response" if choice == "defense" else "awaiting_action"
        return decision_input(obs, [], None, definitions, public_options(obs, definitions))
    return make
