"""Pure versioned participant/candidate features; IDs are references, never numbers."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np
import numpy.typing as npt

from gf_common.domain import ContractError, DecisionInput, Json, closed, digest, integer
from gf_common.validation import boolean, validate_worker_input
from gf_models.lightgbm.config import EncoderConfig, encoder_config

ENCODER_VERSION = "participant-action-v1"
KINDS = ("weapons", "armor", "miracles", "sundries", "trade", "guardians", "phenomena", "devils")
ATTRIBUTES = ("fire", "water", "wood", "stone", "light", "darkness")
NUMERIC = ("attack", "defense", "ability_value", "price", "mp_cost", "hit_rate")
ACTIONS = ("pray", "play_sequence", "respond", "respond_sequence", "discard", "sacrifice", "exchange", "sell", "buy", "buy_decision")
EVENTS = ("turn_started", "action_accepted", "artifact_used", "artifact_drawn", "devil_appeared", "artifact_discarded", "guardian_set", "guardian_left", "guardian_acted", "miracle_used", "attack_declared", "attack_missed", "damage_applied", "healed", "resource_changed", "buy_offered", "sale_offered", "artifact_transferred", "sun_amulet_activated", "status_changed", "phenomenon_invoked", "player_ascended", "player_defeated", "game_ended")
Array = npt.NDArray[np.float64]


def _bucket(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def _references(value: DecisionInput, action: Json, config: EncoderConfig) -> list[tuple[str, Json, Any]]:
    tag = action.get("type")
    schemas = {
        "pray": ({"type"}, set()), "play_sequence": ({"type", "ordered_elements"}, {"target"}),
        "respond": ({"type", "ordered_defenses"}, set()), "respond_sequence": ({"type", "ordered_defenses"}, set()),
        "discard": ({"type", "instance_ids"}, set()), "sacrifice": ({"type", "instance_ids"}, set()),
        "exchange": ({"type", "artifact", "hp", "mp", "money"}, set()),
        "sell": ({"type", "artifact", "sold_artifact", "target"}, set()),
        "buy": ({"type", "artifact", "target"}, set()), "buy_decision": ({"type", "accept"}, set())}
    if tag not in schemas:
        raise ContractError("unsupported encoded action")
    closed(action, *schemas[tag])
    if "target" in action and action["target"] is not None:
        integer(action["target"], 1, 2)
    if "accept" in action:
        boolean(action["accept"])
    for resource_key in ("hp", "mp", "money"):
        if resource_key in action:
            integer(action[resource_key], -(2**31) if resource_key == "hp" else 0, 2**32 - 1)
    if tag in ("play_sequence", "respond_sequence"):
        entries = action["ordered_elements" if tag == "play_sequence" else "ordered_defenses"]
    elif tag in ("respond", "discard", "sacrifice"):
        ids = action["ordered_defenses" if tag == "respond" else "instance_ids"]
        if not isinstance(ids, list):
            raise ContractError("ordered references required")
        entries = [{"source": "hand_artifact", "instance_id": instance} for instance in ids]
    elif tag in ("exchange", "sell", "buy"):
        entries = [{"source": "hand_artifact", "instance_id": action[key]} for key in
                   (("artifact", "sold_artifact") if tag == "sell" else ("artifact",))]
    else:
        entries = []
    if not isinstance(entries, list) or len(entries) > config.max_elements:
        raise ContractError("candidate exceeds encoder capacity")
    hand = {card["instance_id"]: card["definition_id"] for card in value.observation["own"]["hand"]}
    learned = value.observation["own"]["learned_miracles"]
    resolved = []
    key: Any
    for element in entries:
        if not isinstance(element, dict):
            raise ContractError("typed element required")
        source = element.get("source")
        if source == "hand_artifact":
            closed(element, {"source", "instance_id"})
            key = integer(element["instance_id"])
            if key not in hand:
                raise ContractError("foreign encoded hand reference")
            definition = hand[key]
        elif source == "learned_miracle":
            closed(element, {"source", "definition_id"})
            key = element["definition_id"]
            if not isinstance(key, str) or key not in learned:
                raise ContractError("foreign encoded learned reference")
            definition = key
        else:
            raise ContractError("unknown encoded source")
        resolved.append((source, value.catalog[definition], key))
    return resolved


def _features(value: DecisionInput | None, action: Json, config: EncoderConfig) -> dict[str, float]:
    obs = value.observation if value is not None else {}
    own = obs.get("own", {})
    opponent = obs.get("opponent", {})
    viewer = own.get("player_id", 1)
    output: dict[str, float] = {}
    def put(name: str, number: Any) -> None:
        output[name] = float(number)
    def optional(name: str, source: Json, key: str) -> None:
        put(name, source.get(key, 0))
        put(name + ".missing", key not in source)
    def category(name: str, item: Any, categories: tuple[str, ...]) -> None:
        for category in categories:
            put(name + "." + category, item == category)
        put(name + ".missing", item is None)
    def hashed(name: str, items: list[str]) -> None:
        buckets = [0] * config.hash_buckets
        for item in items:
            buckets[_bucket(item, config.hash_buckets)] += 1
        for index, count in enumerate(buckets):
            put(f"{name}.{index}", count)
    def definition(name: str, card: Json) -> None:
        category(name + ".kind", card.get("kind"), KINDS)
        category(name + ".attribute", card.get("attribute"), ATTRIBUTES)
        for key in NUMERIC:
            optional(name + "." + key, card, key)
        put(name + ".additional_attack", card.get("additional_attack", False))
        put(name + ".additional_attack.missing", "additional_attack" not in card)
        hashed(name + ".definition", [card["definition_id"]] if card else [])
        hashed(name + ".ability", [card["ability"]] if "ability" in card else [])
    for name, participant in (("own", own), ("opponent", opponent)):
        for key in ("hp", "mp", "money"):
            optional(name + "." + key, participant, key)
        put(name + ".ascended", participant.get("ascended", False))
        status = participant.get("status", {})
        put(name + ".status.missing", "status" not in participant)
        for key in ("fog", "flash", "dream", "dark_cloud"):
            put(name + "." + key, status.get(key, False))
        category(name + ".disease", status.get("disease"), ("cold", "fever", "hell", "heaven"))
        put(name + ".guardian.missing", "guardian" not in participant)
        hashed(name + ".guardian", [participant["guardian"]] if participant.get("guardian") is not None else [])
        put(name + ".learned_count", len(participant.get("learned_miracles", [])))
    for key in ("turn", "turns_until_end_of_world", "end_times"):
        put(key, obs.get(key, 0))
    for key in ("current_player", "active_player", "first_player"):
        put(key + ".own", obs.get(key) == viewer)
        put(key + ".opponent", obs.get(key) == 3 - viewer)
        put(key + ".missing", obs.get(key) is None)
    category("choice", obs.get("required_choice"), ("turn_action", "defense", "buy_accept_reject"))
    for name, entries in (("hand", own.get("hand", [])), ("learned", own.get("learned_miracles", []))):
        cards = [value.catalog[c["definition_id"] if isinstance(c, dict) else c] for c in entries] if value is not None else []
        put(name + ".count", len(cards))
        for kind in KINDS:
            put(name + ".kind." + kind, sum(card["kind"] == kind for card in cards))
        for key in NUMERIC:
            put(name + "." + key, sum(card.get(key, 0) for card in cards))
            put(name + "." + key + ".missing", sum(key not in card for card in cards))
        hashed(name + ".definitions", [card["definition_id"] for card in cards])
        hashed(name + ".abilities", [card["ability"] for card in cards if "ability" in card])
    options = value.action_options["action_options"] if value is not None and value.action_options is not None else None
    put("hints.missing", options is None)
    for key in ("can_pray", "can_discard", "can_sacrifice", "can_allow"):
        put("hints." + key, options[key] if options is not None else 0)
    for key in ("usable_hand_elements", "usable_learned_miracles", "exchange_artifacts", "sell_artifacts", "buy_artifacts", "valid_targets"):
        put("hints." + key + ".count", len(options[key]) if options is not None else 0)
    category("action", action["type"], ACTIONS)
    put("target.own", action.get("target") == viewer)
    put("target.opponent", action.get("target") == 3 - viewer)
    put("target.unspecified", action.get("target") is None)
    for key in ("hp", "mp", "money", "accept"):
        optional("action." + key, action, key)
    refs = _references(value, action, config) if value is not None else []
    put("action.elements", len(refs))
    for position in range(config.max_elements):
        source, card, reference = refs[position] if position < len(refs) else (None, {}, None)
        name = f"element.{position}"
        put(name + ".present", source is not None)
        put(name + ".hand", source == "hand_artifact")
        put(name + ".learned", source == "learned_miracle")
        definition(name, card)
        put(name + ".usable", options is not None and source is not None and reference in options["usable_hand_elements" if source == "hand_artifact" else "usable_learned_miracles"])
    if config.history == "public_aggregate":
        all_events = value.events if value is not None else ()
        events = all_events[-config.history_window:]
        put("history.window_truncated", len(all_events) > config.history_window)
        put("history.count", len(events))
        for tag in EVENTS:
            put("history.type." + tag, sum(event["type"] == tag for event in events))
        for role, participant in (("own", viewer), ("opponent", 3 - viewer)):
            selected = [event for event in events if event.get("player", event.get("attacker", event.get("active_player"))) == participant]
            put("history." + role + ".count", len(selected))
            for key in ("amount", "hp_delta", "mp_delta", "money_delta", "mp_spent", "price", "attack"):
                put("history." + role + "." + key, sum(event.get(key, 0) for event in selected))
        hashed("history.definitions", [event["definition_id"] for event in events if "definition_id" in event])
    return output


def feature_names(config: EncoderConfig) -> tuple[str, ...]:
    encoder_config(asdict(config))
    return tuple(_features(None, {"type": "pray"}, config))


def signature(config: EncoderConfig) -> str:
    return digest({"version": ENCODER_VERSION, "config": asdict(config), "features": feature_names(config)})


def encode(value: DecisionInput, action: Json, config: EncoderConfig) -> Array:
    value = validate_worker_input(value)
    encoder_config(asdict(config))
    result = np.asarray(list(_features(value, action, config).values()), dtype=np.float64)
    if result.shape != (len(feature_names(config)),) or not np.isfinite(result).all():
        raise ContractError("nonfinite/invalid encoder vector")
    return result
