"""Corrected own-visible Guardian events preserve their public namespace/time."""

import hashlib
from dataclasses import replace

import pytest

from gf_common import structured_vocab as vocab
from gf_common.domain import ContractError, canonical
from gf_common.structured import DecisionFrame, EventBank, event_delta
from gf_common.structured_history import exact_history_tail
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT as OLD
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as NEW
from gf_common.structured_tokens import (
    signature_payload,
    tokenize_frame,
    validate_frame_tokens,
)
from tests.card_instance_action_support import inputs
from tests.test_card_instance_contract import fixture

GUARDIAN_EVENTS = ("guardian_set", "guardian_left")
GUARDIAN_ACTIONS = tuple(row["definition_id"] for row in fixture("catalog")["definitions"]
    if row["kind"] == "guardians")


def own_frame(events, actor=1, *, window=128):
    value = inputs([("steel-armor", None)], mp=0, usable=())
    obs = value.observation
    obs.update(viewer_player_id=actor, current_player=actor, active_player=actor)
    obs["own"]["player_id"], obs["opponent"]["player_id"] = actor, 3 - actor
    value.action_options["viewer_player_id"] = actor
    value.action_options["action_options"]["actor"] = actor
    value = replace(value, events=events)
    bank = EventBank(actor, "guardian-event-namespace", layout=NEW)
    bank.append(event_delta(actor, bank.episode, 0, bank.boundary.prefix_hash, events, layout=NEW))
    frame = DecisionFrame.from_input(value, bank, 0, layout=NEW)
    return frame, bank, tokenize_frame(frame, bank.view(), aggregate_window=window, layout=NEW)


def test_guardian_vocabulary_matches_pinned_public_catalog_and_preserves_card_indices():
    guardian_ids = {row["guardian"]["id"] for row in fixture("catalog")["definitions"]
        if "guardian" in row}
    assert tuple(sorted(guardian_ids)) == vocab.GUARDIANS == (
        "earth", "jupiter", "mars", "mercury", "moon", "neptune", "pluto", "saturn", "uranus", "venus")
    assert not guardian_ids.intersection(vocab.DEFINITIONS)
    assert len(vocab.DEFINITIONS) == 296 and len(guardian_ids) == 10
    assert len(GUARDIAN_ACTIONS) == 42
    assert len([row for row in fixture("catalog")["definitions"] if "guardian" in row]) == 54
    assert OLD.event_categories[1] == vocab.DEFINITIONS
    assert NEW.event_categories[1] == (*vocab.DEFINITIONS, *vocab.GUARDIANS)
    old_fields = signature_payload(layout=OLD)["history_fields"]
    new_fields = signature_payload(layout=NEW)["history_fields"]
    assert new_fields[:359] == old_fields[:359]  # 63 base +296 card counts.
    assert new_fields[359:369] == tuple("definition." + name for name in vocab.GUARDIANS)
    assert new_fields[369:] == old_fields[359:]


def assert_event_time_tokens(name, family, actor, category_index, history_index):
    events = (
        {"type": "artifact_used", "player": 1, "definition_id": "fireball"},
        {"type": family, "player": 1, "definition_id": name},
        {"type": family, "player": 2, "definition_id": name},
        {"type": "resource_changed", "player": 1, "hp_delta": 7, "mp_delta": -2, "money_delta": 3},
    )
    frame, bank, tokens = own_frame(events, actor)
    validate_frame_tokens(tokens, layout=NEW)
    assert bank.view().events() == events
    previous = bank.view()._prefixes[0]
    for event, expected in zip(events, bank.view()._prefixes[1:], strict=True):
        previous = hashlib.sha256(bytes.fromhex(previous) + canonical(event).encode()).hexdigest()
        assert previous == expected
    assert tokens.interval_categories[0][1] == 2 + vocab.DEFINITIONS.index("fireball")
    assert tuple(row[1] for row in tokens.interval_categories[1:3]) == (category_index,) * 2
    assert tuple(row[4] for row in tokens.interval_categories[1:3]) == (
        (2, 3) if actor == 1 else (3, 2))
    expected_history = [0] * 387
    for family_name, count in (("artifact_used", 1), (family, 2), ("resource_changed", 1)):
        expected_history[NEW.event_types.index(family_name)] = count
    expected_history[25:27] = [3, 1] if actor == 1 else [1, 3]  # player role.
    expected_history[41 + 7 * 2:41 + 10 * 2] = [7, 1, -2, 1, 3, 1]
    expected_history[63 + vocab.DEFINITIONS.index("fireball")] = 1
    expected_history[history_index] += 2
    expected_history[369:371] = [4, 0]
    player1, player2 = [3, 0, 7, -2, 3, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]
    expected_history[371:] = player1 + player2 if actor == 1 else player2 + player1
    assert tokens.history == tuple(expected_history)
    assert set(frame.raw["catalog"]) == {"steel-armor"}  # No event catalog values fetched.
    bank.append(event_delta(actor, bank.episode, bank.boundary.end, bank.boundary.prefix_hash,
        [{"type": "guardian_left", "player": 1, "definition_id": "earth"}], layout=NEW))
    assert tokenize_frame(frame, bank.view(), layout=NEW) == tokens


@pytest.mark.parametrize("guardian", vocab.GUARDIANS)
@pytest.mark.parametrize("family", GUARDIAN_EVENTS)
@pytest.mark.parametrize("actor", [1, 2])
def test_every_guardian_identity_family_and_viewer_exact_tokens_counts_order_and_prefix(
    guardian, family, actor
):
    index = vocab.GUARDIANS.index(guardian)
    assert_event_time_tokens(guardian, family, actor, 298 + index, 359 + index)


@pytest.mark.parametrize("action", GUARDIAN_ACTIONS)
@pytest.mark.parametrize("actor", [1, 2])
def test_all42_guardian_actions_and_both_viewers_preserve_card_id_tokens_counts_order_prefix(
    action, actor
):
    index = vocab.DEFINITIONS.index(action)
    assert_event_time_tokens(action, "guardian_acted", actor, 2 + index, 63 + index)


@pytest.mark.parametrize("name", ["bronze-club", "steel-armor", "fireball", "guardian-pot"])
@pytest.mark.parametrize("actor", [1, 2])
def test_guardian_acted_retains_generic_known_card_domain_without_invented_kind_constraint(name, actor):
    index = vocab.DEFINITIONS.index(name)
    assert_event_time_tokens(name, "guardian_acted", actor, 2 + index, 63 + index)


def test_guardian_history_window_and_default_card_only_history_remain_distinct():
    events = (
        {"type": "guardian_set", "player": 1, "definition_id": "earth"},
        {"type": "guardian_left", "player": 2, "definition_id": "earth"},
    )
    _, _, tokens = own_frame(events, window=1)
    assert tokens.history[359] == 1
    assert tokens.history[369:371] == (1, 1)
    assert tokens.history[371:] == (0,) * 8 + (1,) + (0,) * 7
    with pytest.raises(ContractError, match="unknown event-time"):
        exact_history_tail(events, 1, prefix_count=2, window=128)
    bank = EventBank(1, "old-card-only", layout=OLD)
    with pytest.raises(ContractError, match="definition"):
        event_delta(1, bank.episode, 0, bank.boundary.prefix_hash, events, layout=OLD)


@pytest.mark.parametrize("family", GUARDIAN_EVENTS)
@pytest.mark.parametrize("name", ["unknown-guardian", "fireball", "guardian-pot", "seafood-soup"])
def test_guardian_events_refuse_unknown_and_card_namespace_ids(family, name):
    with pytest.raises(ContractError, match="namespace"):
        own_frame(({"type": family, "player": 1, "definition_id": name},))


@pytest.mark.parametrize("event", [
    {"type": kind, "player": 1} for kind in
    ("artifact_used", "artifact_drawn", "devil_appeared", "artifact_discarded")
] + [
    {"type": "miracle_used", "player": 1, "mp_spent": 0},
    {"type": "guardian_acted", "player": 1},
    {"type": "buy_offered", "buyer": 1, "seller": 2, "price": 0},
    {"type": "sale_offered", "seller": 1, "target": 2, "price": 0},
    {"type": "artifact_transferred", "from": 1, "to": 2, "price": 0},
    {"type": "phenomenon_invoked"},
])
@pytest.mark.parametrize("name", ["earth", "unknown-card"])
def test_card_events_refuse_guardian_namespace_and_unknown_ids(event, name):
    with pytest.raises(ContractError, match="namespace"):
        own_frame(({**event, "definition_id": name},))


@pytest.mark.parametrize("guardian", vocab.GUARDIANS)
def test_guardian_acted_refuses_every_guardian_identity(guardian):
    with pytest.raises(ContractError, match="namespace"):
        own_frame(({"type": "guardian_acted", "player": 1, "definition_id": guardian},))


def test_stale_corrected_signature_refuses_even_current_equal_width_tokens():
    _, _, tokens = own_frame(())
    old_signature = "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"
    with pytest.raises(ContractError, match="signature"):
        validate_frame_tokens(replace(tokens, recipe_signature=old_signature), layout=NEW)
