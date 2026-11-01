"""Exact categories, perceived-time facts and management identity exclusion."""
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError
from gf_common.structured import CandidateTable, EventDelta
from gf_common.structured_tokens import (
    CARD_CATEGORIES,
    MISSING,
    PAD,
    STATE_NUMERIC_FIELDS,
    category,
    signature,
    tokenize_frame,
    tokenize_table,
    validate_frame_tokens,
)
from gf_common.validation import decision_input
from tests.conftest import public_options
from tests.test_structured_records import frame_for


def features(tokens):
    return {key: value for key, value in asdict(tokens).items()
            if key not in ("frame_hash", "actor", "episode", "revision", "event_start", "event_end")}


def test_instance_numbers_and_episode_revision_are_not_features(make_input):
    value = make_input(hand=("bronze-club", "steel-armor"), learned=("fireball", "aura"))
    frame, bank = frame_for(value)
    first = tokenize_frame(frame, bank.view())
    obs = value.to_dict()["observation"]
    obs["state_revision"] = 999
    for index, card in enumerate(obs["own"]["hand"]):
        card["instance_id"] = 2**64 - 1 - index
    altered = decision_input(obs, [], None, value.catalog, public_options(obs, value.catalog))
    second_frame, second_bank = frame_for(altered, "other")
    second = tokenize_frame(second_frame, second_bank.view())
    assert features(first) == features(second)
    assert first.card_numerics[2][8] == 1 and first.card_numerics[3][8] == 2
    assert first.card_categories[0][0] == category("bronze-club", CARD_CATEGORIES[0])
    assert first.card_mask == (True,) * 4 + (False,) * 14
    validate_frame_tokens(first)


def test_fog_missingness_and_opponent_ids_without_extra_catalog(make_input):
    value = make_input()
    obs = value.to_dict()["observation"]
    obs["own"]["status"]["fog"] = True
    obs["opponent"] = {"player_id": 2, "learned_miracles": ["aura"], "ascended": False}
    fogged = decision_input(obs, [], None, value.catalog, public_options(obs, value.catalog))
    frame, bank = frame_for(fogged)
    tokens = tokenize_frame(frame, bank.view())
    hp = STATE_NUMERIC_FIELDS.index("opponent.hp")
    assert tokens.state_present[hp] is False
    assert tokens.state_categories[-2:] == (MISSING, MISSING)
    assert tokens.state_category_present[-2:] == (False, False)
    assert tokens.opponent_learned[0] == category("aura", CARD_CATEGORIES[0])
    assert "aura" not in frame.raw["catalog"]
    assert "opponent.hand" not in STATE_NUMERIC_FIELDS
    assert "opponent.hand_count" not in STATE_NUMERIC_FIELDS


def test_dream_and_observed_history_do_not_rewrite(make_input):
    value = make_input(hand=("bronze-club",))
    obs = value.to_dict()["observation"]
    obs["own"]["status"]["dream"] = True
    # Observed perceived value is used even when different from another frame.
    obs["own"]["hand"][0]["definition_id"] = "angel-axe"
    dream = make_input(hand=("angel-axe",))
    dream = replace(dream, observation=obs)
    frame, bank = frame_for(dream)
    before = tokenize_frame(frame, bank.view())
    event = {"type": "artifact_used", "player": 2, "definition_id": "bronze-club"}
    bank.append(EventDelta(1, "one", 0, bank.boundary.prefix_hash, [event]))
    assert tokenize_frame(frame, bank.view()) == before
    assert before.card_categories[0][0] == category("angel-axe", CARD_CATEGORIES[0])


def test_signed_events_missingness_and_history_128(make_input):
    events = tuple({"type": "resource_changed", "player": 1,
                    "hp_delta": -3, "mp_delta": 2, "money_delta": 0} for _ in range(129))
    frame, bank = frame_for(replace(make_input(), events=events))
    tokens = tokenize_frame(frame, bank.view())
    assert len(tokens.interval_numerics) == 129
    assert -3 in tokens.interval_numerics[0]
    assert tokens.history[14] == 128  # resource_changed exact event code index
    assert PAD != MISSING
    with pytest.raises(ContractError):
        category("unknown-category", CARD_CATEGORIES[0])
    validate_frame_tokens(tokens)
    with pytest.raises(ContractError):
        validate_frame_tokens(replace(tokens, state_numerics=(True,) * len(tokens.state_numerics)))


def test_candidate_slots_and_recipe_signature_are_deterministic(make_input):
    frame, bank = frame_for(make_input(learned=("aura",)))
    table = CandidateTable.from_actions(frame, [{"type": "play_sequence", "target": None,
                                               "ordered_elements": [frame.references[1]] * 18}])
    tokens = tokenize_table(table, frame, 18)
    assert tokens.offsets == (0, 18) and tokens.slots == (1,) * 18
    assert tokens.recipe_signature == tokenize_frame(frame, bank.view()).recipe_signature == signature()
    with pytest.raises(ContractError):
        tokenize_table(table, frame, 17)


def test_exact_public_vocab_categories_and_nondefault_window(make_input, definitions):
    from gf_common import structured_vocab as vocab
    from gf_common.structured_tokens import STATUSES
    assert vocab.DEFINITIONS == tuple(sorted(definitions))
    assert vocab.ABILITIES == tuple(sorted({d["ability"] for d in definitions.values() if "ability" in d}))
    assert category("dark_cloud", STATUSES) != category("darkcloud", STATUSES)
    events = tuple({"type": "action_accepted", "player": 1} for _ in range(9))
    frame, bank = frame_for(replace(make_input(), events=events))
    narrow = tokenize_frame(frame, bank.view(), aggregate_window=7)
    wide = tokenize_frame(frame, bank.view(), aggregate_window=9)
    assert narrow.history[1] == 7 and wide.history[1] == 9
    assert narrow.recipe_signature != wide.recipe_signature


def test_model_neutral_layout_signature_is_pinned():
    # Native reproduction must bind the exact ordered categorical and numeric
    # recipe, including distinct darkcloud/dark_cloud status codes.
    assert signature() == "c08f6301b4feae6f8c63d1f278874a3ddfac40cb94f9981479733ba02d63f798"
