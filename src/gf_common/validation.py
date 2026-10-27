"""Closed Rust-v1 participant DTO validation (spec 10 §3.1, 07 §21.1)."""
from __future__ import annotations

from typing import Any

from gf_common.domain import (
    CATALOG_META,
    ContractError,
    DecisionInput,
    Json,
    UnsupportedContract,
    canonical,
    clone,
    closed,
    digest,
    integer,
)


def action_descriptors(choice: str | None) -> list[Json]:
    instance = {"type": "integer", "minimum": 0, "maximum": 2**64 - 1,
                "sources": ["own.hand[].instance_id"]}
    player = {"type": "integer", "enum": [1, 2],
              "sources": ["own.player_id", "opponent.player_id"]}
    element = {"one_of": [
        {"type": "object", "required_fields": {"source": {"type": "string", "const": "hand_artifact"},
          "instance_id": instance}, "optional_fields": {}, "additional_fields": False},
        {"type": "object", "required_fields": {"source": {"type": "string", "const": "learned_miracle"},
          "definition_id": {"type": "string", "sources": ["own.learned_miracles[]"]}},
         "optional_fields": {}, "additional_fields": False},
    ]}

    def array(item: Json) -> Json:
        return {"type": "array", "items": item, "ordered": True}

    def desc(tag: str, fields: Json, optional: Json | None = None) -> Json:
        return {"type": tag, "required_fields": fields, "optional_fields": optional or {},
                "additional_fields": False}

    if choice == "turn_action":
        return [desc("pray", {}), desc("play_sequence", {"ordered_elements": array(element)},
                    {"target": {"one_of": [player, {"type": "null"}]}}),
                desc("discard", {"instance_ids": array(instance)}),
                desc("sacrifice", {"instance_ids": array(instance)}),
                desc("exchange", {"artifact": instance, "hp": {"type": "integer", "minimum": -(2**31),
                     "maximum": 2**31 - 1}, "mp": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
                     "money": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1}}),
                desc("sell", {"artifact": instance, "sold_artifact": instance, "target": player}),
                desc("buy", {"artifact": instance, "target": player})]
    if choice == "defense":
        return [desc("respond", {"ordered_defenses": array(instance)}),
                desc("respond_sequence", {"ordered_defenses": array(element)})]
    if choice == "buy_accept_reject":
        return [desc("buy_decision", {"accept": {"type": "boolean"}})]
    return []


def boolean(value: Any) -> None:
    if type(value) is not bool:
        raise ContractError("expected boolean")


def text_value(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError("expected nonempty string")


def player(value: Any) -> None:
    integer(value, 1, 2)


def result(value: Any) -> None:
    if value is None or value == "draw":
        return
    closed(value, {"win"})
    closed(value["win"], {"winner"})
    player(value["win"]["winner"])


def status(value: Any) -> None:
    closed(value, {"disease", "fog", "flash", "dream", "dark_cloud"})
    if value["disease"] not in (None, "cold", "fever", "hell", "heaven"):
        raise ContractError("unknown disease")
    for key in ("fog", "flash", "dream", "dark_cloud"):
        boolean(value[key])


def observation(value: Any, viewer: int | None = None) -> Json:
    required = {"observation_schema_version", "action_schema_version", "actionable", "typed_action_schema",
                "state_revision", "viewer_player_id", "turn", "turns_until_end_of_world", "first_player",
                "active_player", "current_player", "phase", "required_choice", "end_times", "own",
                "opponent", "result"}
    closed(value, required)
    if value["observation_schema_version"] != "1" or type(value["action_schema_version"]) is not int or value["action_schema_version"] != 1:
        raise UnsupportedContract("unsupported observation/action version")
    for key in ("viewer_player_id", "first_player", "active_player"):
        player(value[key])
    if viewer is not None and value["viewer_player_id"] != viewer:
        raise ContractError("viewer mismatch")
    if value["current_player"] is not None:
        player(value["current_player"])
    for key in ("state_revision", "turn", "turns_until_end_of_world"):
        integer(value[key], 0, 2**64 - 1 if key == "state_revision" else 2**32 - 1)
    for key in ("actionable", "end_times"):
        boolean(value[key])
    if value["phase"] not in ("awaiting_action", "awaiting_response", "resolving", "terminal"):
        raise UnsupportedContract("unknown phase")
    if value["required_choice"] not in (None, "turn_action", "defense", "buy_accept_reject", "additional_choice"):
        raise UnsupportedContract("unknown choice")
    own = closed(value["own"], {"player_id", "hp", "mp", "money", "hand", "learned_miracles", "status", "guardian", "ascended"})
    opponent = closed(value["opponent"], {"player_id", "learned_miracles", "ascended"}, {"hp", "mp", "money", "status", "guardian"})
    if own["player_id"] != value["viewer_player_id"] or opponent["player_id"] != 3 - own["player_id"]:
        raise ContractError("participant mismatch")
    for participant in (own, opponent):
        player(participant["player_id"])
        boolean(participant["ascended"])
        for key in ("hp", "mp", "money"):
            if key in participant:
                integer(participant[key], -(2**31) if key == "hp" else 0, 2**31 - 1 if key == "hp" else 2**32 - 1)
        if "status" in participant:
            status(participant["status"])
        if "guardian" in participant and participant["guardian"] is not None:
            text_value(participant["guardian"])
        if not isinstance(participant["learned_miracles"], list):
            raise ContractError("learned list required")
        for definition_id in participant["learned_miracles"]:
            text_value(definition_id)
    if not isinstance(own["hand"], list):
        raise ContractError("hand list required")
    ids = set()
    for artifact in own["hand"]:
        closed(artifact, {"instance_id", "definition_id"})
        integer(artifact["instance_id"])
        text_value(artifact["definition_id"])
        if artifact["instance_id"] in ids:
            raise ContractError("duplicate hand instance")
        ids.add(artifact["instance_id"])
    result(value["result"])
    input_phase = value["phase"] in ("awaiting_action", "awaiting_response") or (
        value["phase"] == "resolving" and value["required_choice"] == "buy_accept_reject")
    can_act = value["current_player"] == value["viewer_player_id"] and value["result"] is None and input_phase
    if can_act and value["required_choice"] == "additional_choice":
        raise UnsupportedContract("additional_choice has no v1 action contract")
    expected = action_descriptors(value["required_choice"]) if can_act else []
    if value["actionable"] != bool(expected) or canonical(value["typed_action_schema"]) != canonical(expected):
        raise UnsupportedContract("typed metadata mismatch")
    return clone(value)


def catalog(value: Any) -> Json:
    closed(value, set(CATALOG_META) | {"definitions"})
    for key, expected in CATALOG_META.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise UnsupportedContract("catalog metadata mismatch")
    if not isinstance(value["definitions"], list) or len(value["definitions"]) != 296:
        raise ContractError("catalog count mismatch")
    ids = []
    for definition in value["definitions"]:
        validate_definition(definition)
        ids.append(definition["definition_id"])
    if ids != sorted(set(ids)):
        raise ContractError("catalog ordering/uniqueness mismatch")
    unhashed = {k: v for k, v in value.items() if k != "content_sha256"}
    if digest(unhashed) != value["content_sha256"]:
        raise ContractError("catalog hash mismatch")
    return {d["definition_id"]: clone(d) for d in value["definitions"]}


def validate_definition(value: Any) -> None:
    closed(value, {"definition_id", "display_name", "kind"}, {"attack", "defense", "ability_value", "attribute", "price", "mp_cost", "additional_attack", "hit_rate", "ability", "effect_text", "curse", "guardian"})
    for key in ("definition_id", "display_name"):
        text_value(value[key])
    if value["kind"] not in ("weapons", "armor", "miracles", "sundries", "trade", "guardians", "phenomena", "devils"):
        raise ContractError("unknown catalog kind")
    for key in ("attack", "defense", "ability_value", "price", "mp_cost", "hit_rate"):
        if key in value:
            integer(value[key], 0 if key in ("price", "mp_cost", "hit_rate") else -(2**31), 100 if key == "hit_rate" else 2**32 - 1 if key in ("price", "mp_cost") else 2**31 - 1)
    if "attribute" in value and value["attribute"] not in ("fire", "water", "wood", "stone", "light", "darkness"):
        raise ContractError("unknown attribute")
    if "additional_attack" in value:
        boolean(value["additional_attack"])
    for key in ("ability", "effect_text"):
        if key in value:
            text_value(value[key])
    for key in ("curse", "guardian"):
        if key in value:
            closed(value[key], {"id", "name", "description"})
            for item in value[key].values():
                text_value(item)


def actor_action_options(value: Any, obs: Json, *, raw: bool = False) -> Json:
    """Bind the closed atomic DTO to its actionable actor and exact revision."""
    obs = observation(obs)
    if not obs["actionable"] or obs["current_player"] != obs["viewer_player_id"]:
        raise ContractError("ActionOptions require current actionable actor")
    fields = {"state_revision", "action_options"}
    closed(value, fields if raw else fields | {"viewer_player_id"})
    integer(value["state_revision"])
    if value["state_revision"] != obs["state_revision"]:
        raise ContractError("ActionOptions revision mismatch")
    if not raw:
        player(value["viewer_player_id"])
        if value["viewer_player_id"] != obs["viewer_player_id"]:
            raise ContractError("ActionOptions actor mismatch")
    lists = {"usable_hand_elements", "usable_learned_miracles", "exchange_artifacts",
             "sell_artifacts", "buy_artifacts", "valid_targets"}
    flags = {"can_pray", "can_discard", "can_sacrifice", "can_allow"}
    body = closed(value["action_options"], lists | flags | {"required_choice"})
    if body["required_choice"] != obs["required_choice"]:
        raise ContractError("ActionOptions choice mismatch")
    hand = {c["instance_id"] for c in obs["own"]["hand"]}
    learned = set(obs["own"]["learned_miracles"])
    targets = {obs["own"]["player_id"], obs["opponent"]["player_id"]}
    for key in lists:
        entries = body[key]
        if not isinstance(entries, list):
            raise ContractError("ActionOptions array required")
        for entry in entries:
            if key == "usable_learned_miracles":
                text_value(entry)
            elif key == "valid_targets":
                player(entry)
            else:
                integer(entry)
        allowed = learned if key == "usable_learned_miracles" else targets if key == "valid_targets" else hand
        if len(entries) != len(set(entries)) or not set(entries) <= allowed:
            raise ContractError("duplicate or foreign ActionOptions reference")
    for key in flags:
        boolean(body[key])
    return {"viewer_player_id": obs["viewer_player_id"], "state_revision": value["state_revision"],
            "action_options": clone(body)}


def decision_input(obs: Json, events: list[Json], history_end: str | None, definitions: Json,
                   action_options: Json | None = None) -> DecisionInput:
    obs = observation(obs)
    validate_events(events)
    if history_end is not None:
        text_value(history_end)
    visible = {a["definition_id"] for a in obs["own"]["hand"]} | set(obs["own"]["learned_miracles"])
    try:
        own_catalog = {key: clone(definitions[key]) for key in sorted(visible)}
    except KeyError as exc:
        raise ContractError("unknown own-visible definition") from exc
    for definition in own_catalog.values():
        validate_definition(definition)
    return DecisionInput(obs, tuple(clone(events)), history_end, tuple(clone(obs["typed_action_schema"])), own_catalog,
                         actor_action_options(action_options, obs) if action_options is not None else None)


def validate_worker_input(value: DecisionInput, use_action_options: bool | None = None) -> DecisionInput:
    if use_action_options is not None:
        boolean(use_action_options)
        if (value.action_options is not None) != use_action_options:
            raise ContractError("ActionOptions presence/mode mismatch")
    rebuilt = decision_input(value.observation, list(value.events), value.history_end, value.catalog, value.action_options)
    if set(rebuilt.catalog) != set(value.catalog) or canonical(value.typed_metadata) != canonical(rebuilt.typed_metadata):
        raise ContractError("worker payload boundary mismatch")
    return rebuilt

def validate_events(value: Any) -> None:
    """Closed public event DTOs, including omitted fog values (DTO spec 05)."""
    required: dict[str, set[str]] = {
        "turn_started": {"turn", "active_player"}, "action_accepted": {"player"},
        **{tag: {"player", "definition_id"} for tag in ("artifact_used", "artifact_drawn", "devil_appeared", "artifact_discarded", "guardian_set", "guardian_left", "guardian_acted")},
        "miracle_used": {"player", "definition_id", "mp_spent"},
        "attack_declared": {"attacker", "target", "attack"}, "attack_missed": {"attacker", "target"},
        "damage_applied": {"player", "amount"}, "healed": {"player", "amount"},
        "resource_changed": {"player", "hp_delta", "mp_delta", "money_delta"},
        "buy_offered": {"buyer", "seller", "definition_id", "price"},
        "sale_offered": {"seller", "target", "definition_id", "price"},
        "artifact_transferred": {"from", "to", "definition_id", "price"},
        "sun_amulet_activated": {"player"}, "status_changed": {"player", "status", "active"},
        "phenomenon_invoked": {"definition_id"}, "player_ascended": {"player"},
        "player_defeated": {"player"}, "game_ended": {"result"},
    }
    if not isinstance(value, list):
        raise ContractError("event array required")
    for event in value:
        if not isinstance(event, dict) or event.get("type") not in required:
            raise UnsupportedContract("unknown player event")
        tag = event["type"]
        optional = {"remaining_hp"} if tag == "damage_applied" else {"hp"} if tag == "healed" else set()
        closed(event, required[tag] | {"type"}, optional)
        for key, item in event.items():
            if key in ("player", "attacker", "target", "active_player", "buyer", "seller", "from", "to"):
                integer(item, 1, 2)
            elif key in ("turn", "price", "mp_spent"):
                integer(item, 0, 2**32 - 1)
            elif key in ("attack", "amount", "remaining_hp", "hp", "hp_delta", "mp_delta", "money_delta"):
                integer(item, -(2**31), 2**31 - 1)
            elif key == "active":
                if type(item) is not bool:
                    raise ContractError("event boolean required")
            elif key == "result":
                result(item)
            elif not isinstance(item, str) or not item:
                raise ContractError("event string required")


def reconcile_replay(replay: Json, viewer: int, accepted: list[Json], outcome: Any) -> None:
    """Only positive actor actions and terminal outcome; replay never enters policies."""
    if replay.get("viewer_player_id") != viewer or replay.get("observation_schema_version") != "1":
        raise ContractError("replay perspective/version mismatch")
    actual = []
    for turn in replay.get("turns", []):
        for decision in turn.get("decisions", []):
            if decision.get("actor") != viewer:
                continue
            visible = decision.get("visible_action")
            if visible and visible.get("type") == "action":
                action = visible["action"]
                # Rust serializes omitted optional target as null in replay.
                if action.get("type") == "play_sequence":
                    action = {**action, "target": action.get("target")}
                actual.append(action)
    normalized = [{**a, "target": a.get("target")} if a.get("type") == "play_sequence" else a for a in accepted]
    if actual != normalized:
        raise ContractError("accepted actions/replay mismatch")
    if outcome is not None and replay.get("episode_end") != {"type": "terminal", "result": outcome}:
        raise ContractError("terminal outcome/replay mismatch")



def replay_closure(replay: Any, game: str, viewer: int) -> Json | None:
    """Validate public identity and the entire closed EpisodeEnd DTO (ADR0003)."""
    if not isinstance(replay, dict) or replay.get("game_id") != game or type(replay.get("viewer_player_id")) is not int or replay["viewer_player_id"] != viewer or replay.get("observation_schema_version") != "1" or "episode_end" not in replay:
        raise ContractError("replay game/perspective/version/closure mismatch")
    value = replay["episode_end"]
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError("invalid episode closure")
    if value.get("type") == "terminal":
        closed(value, {"type", "result"})
        result(value["result"])
        if value["result"] is None:
            raise ContractError("terminal closure requires result")
    elif value.get("type") == "aborted":
        closed(value, {"type", "reason", "player", "decision"})
        text_value(value["reason"])
        if value["player"] is not None:
            player(value["player"])
        decision = closed(value["decision"], {"turn", "decision_index", "invalid_attempt_count"})
        for field in ("turn", "decision_index"):
            if decision[field] is not None:
                integer(decision[field], 0, 2**32 - 1)
        integer(decision["invalid_attempt_count"], 0, 2**32 - 1)
    else:
        raise ContractError("unknown episode closure type")
    return clone(value)


def projected_abort_kind(reason: Any) -> str:
    text_value(reason)
    return "truncated" if reason == "turn_limit_exceeded" else "aborted"
