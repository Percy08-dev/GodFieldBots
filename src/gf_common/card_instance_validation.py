"""Closed original public v2 DTOs. Syntax and visibility checks, never legality.

HTTP, replay control suppression, and Dataset observations have distinct entry
points. No missing wire field is synthesized and opaque refs are never features.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from gf_common.card_instance import CATALOG_META, RULE_SET
from gf_common.domain import (
    ContractError,
    Json,
    UnsupportedContract,
    canonical,
    clone,
    closed,
    digest,
    integer,
)
from gf_common.validation import boolean, player, result, status, text_value, validate_definition
from gf_common.validation import validate_events as _legacy_events

CARD_REF_SCHEME = "aes256-card-ref-v1"
_BASE_OBSERVATION = {
    "observation_schema_version",
    "state_revision",
    "viewer_player_id",
    "turn",
    "first_player",
    "active_player",
    "current_player",
    "phase",
    "required_choice",
    "end_times",
    "own",
    "opponent",
    "result",
}
_HTTP_FIELDS = {
    "rule_set_version",
    "card_ref_scheme",
    "action_schema_version",
    "actionable",
    "typed_action_schema",
    "turns_until_end_of_world",
}
_FLAGS = {"can_pray", "can_discard", "can_sacrifice", "can_allow"}
_CARD_LISTS = {"usable_cards", "exchange_cards", "sell_cards", "buy_cards", "phenomenon_cards"}
_SUBCLASSES = {"weapon", "additional_weapon", "attack_support", "defense", "status", "utility"}
_ROLES = {
    "weapon",
    "armor",
    "numeric_defense",
    "additional_attack",
    "cost_cut",
    "trade",
    "utility",
} | {"miracle:" + name for name in _SUBCLASSES}


def _pin(value: Json, name: str, expected: Any) -> None:
    if type(value[name]) is not type(expected) or value[name] != expected:
        raise UnsupportedContract("unsupported public v2 " + name)


def card_ref(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"c1_[0-9a-f]{32}", value) is None:
        raise ContractError("invalid opaque card reference")


def action_descriptors(choice: str | None) -> list[Json]:
    ref = {"type": "string", "pattern": "^c1_[0-9a-f]{32}$", "sources": ["own.cards[].card_ref"]}
    refs = {"type": "array", "items": ref, "ordered": True, "unique": True}
    target = {"type": "integer", "enum": [1, 2], "sources": ["own.player_id", "opponent.player_id"]}

    def desc(kind: str, fields: Json) -> Json:
        return {
            "type": kind,
            "required_fields": fields,
            "optional_fields": {},
            "additional_fields": False,
        }

    if choice == "turn_action":
        return [
            desc("pray", {}),
            desc("play_sequence", {"ordered_card_refs": refs, "target": target}),
            desc("discard", {"card_refs": refs}),
            desc("sacrifice", {"card_refs": refs}),
            desc(
                "exchange",
                {
                    "card_ref": ref,
                    "hp": {"type": "integer", "minimum": -(2**31), "maximum": 2**31 - 1},
                    "mp": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
                    "money": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
                },
            ),
            desc("sell", {"card_ref": ref, "sold_card_ref": ref, "target": target}),
            desc("buy", {"card_ref": ref, "target": target}),
            desc("invoke_phenomenon", {"card_ref": ref}),
        ]
    if choice == "defense":
        return [desc("respond", {"ordered_card_refs": refs})]
    if choice == "buy_accept_reject":
        return [desc("buy_decision", {"accept": {"type": "boolean"}})]
    return []


def _observation(
    value: Any, viewer: int | None, *, representation: str, disabled: bool = False
) -> Json:
    closed(value, _BASE_OBSERVATION | (_HTTP_FIELDS if representation != "dataset" else set()))
    _pin(value, "observation_schema_version", "2")
    for name in ("viewer_player_id", "first_player", "active_player"):
        player(value[name])
    if viewer is not None:
        player(viewer)
        if viewer != value["viewer_player_id"]:
            raise ContractError("observation viewer mismatch")
    if value["current_player"] is not None:
        player(value["current_player"])
    integer(value["state_revision"])
    integer(value["turn"], 0, 2**32 - 1)
    boolean(value["end_times"])
    if value["phase"] not in ("awaiting_action", "awaiting_response", "resolving", "terminal"):
        raise UnsupportedContract("unknown phase")
    if value["required_choice"] not in (
        None,
        "turn_action",
        "defense",
        "buy_accept_reject",
        "additional_choice",
    ):
        raise UnsupportedContract("unknown required choice")
    own = closed(
        value["own"], {"player_id", "hp", "mp", "money", "cards", "status", "guardian", "ascended"}
    )
    opponent = closed(
        value["opponent"],
        {"player_id", "learned_miracles", "ascended"},
        {"hp", "mp", "money", "status", "guardian"},
    )
    for participant in (own, opponent):
        player(participant["player_id"])
        boolean(participant["ascended"])
        for name in ("hp", "mp", "money"):
            if name in participant:
                integer(
                    participant[name],
                    -(2**31) if name == "hp" else 0,
                    2**31 - 1 if name == "hp" else 2**32 - 1,
                )
        if "status" in participant:
            status(participant["status"])
        if "guardian" in participant and participant["guardian"] is not None:
            text_value(participant["guardian"])
    if (
        own["player_id"] != value["viewer_player_id"]
        or opponent["player_id"] != 3 - own["player_id"]
    ):
        raise ContractError("observation participant mismatch")
    if not isinstance(own["cards"], list) or len(own["cards"]) > 18:
        raise ContractError("owned cards array/capacity mismatch")
    seen = set()
    for card in own["cards"]:
        closed(card, {"kind", "card_ref", "definition_id"}, {"state"})
        card_ref(card["card_ref"])
        text_value(card["definition_id"])
        if card["kind"] == "artifact":
            if "state" in card:
                raise ContractError("artifact cannot carry miracle state")
        elif card["kind"] == "miracle":
            if card.get("state") not in ("unused", "learned"):
                raise ContractError("miracle requires original state")
        else:
            raise ContractError("unknown observed card kind")
        if card["card_ref"] in seen:
            raise ContractError("duplicate owned card reference")
        seen.add(card["card_ref"])
    if not isinstance(opponent["learned_miracles"], list) or len(opponent["learned_miracles"]) > 18:
        raise ContractError("opponent learned list/capacity mismatch")
    for name in opponent["learned_miracles"]:
        text_value(name)
    if own["status"]["fog"] and set(opponent) - {"player_id", "learned_miracles", "ascended"}:
        raise ContractError("Fog opponent visibility mismatch")
    result(value["result"])
    if representation != "dataset":
        _pin(value, "rule_set_version", RULE_SET)
        _pin(value, "card_ref_scheme", CARD_REF_SCHEME)
        _pin(value, "action_schema_version", 2)
        integer(value["turns_until_end_of_world"], 0, 2**32 - 1)
        boolean(value["actionable"])
        input_phase = value["phase"] in ("awaiting_action", "awaiting_response") or (
            value["phase"] == "resolving" and value["required_choice"] == "buy_accept_reject"
        )
        can_act = (
            value["current_player"] == own["player_id"] and value["result"] is None and input_phase
        )
        if can_act and value["required_choice"] == "additional_choice":
            raise UnsupportedContract("additional_choice has no public v2 Action syntax")
        expected = action_descriptors(value["required_choice"]) if can_act and not disabled else []
        if value["actionable"] != bool(expected) or canonical(
            value["typed_action_schema"]
        ) != canonical(expected):
            raise UnsupportedContract("typed v2 Action metadata mismatch")
    return clone(value)


def observation(value: Any, viewer: int | None = None) -> Json:
    return _observation(value, viewer, representation="http")


def replay_observation(value: Any, viewer: int, *, aborted: bool) -> Json:
    boolean(aborted)
    return _observation(value, viewer, representation="replay", disabled=aborted)


def dataset_observation(value: Any, viewer: int | None = None) -> Json:
    return _observation(value, viewer, representation="dataset")


def _validate_action(value: Any, obs: Json | None, *, submission: bool) -> Json:
    if not isinstance(value, dict):
        raise ContractError("typed Action object required")
    descriptors = {
        item["type"]: item
        for choice in ("turn_action", "defense", "buy_accept_reject")
        for item in action_descriptors(choice)
    }
    kind = value.get("type")
    if type(kind) is not str or kind not in descriptors:
        raise UnsupportedContract("unknown public v2 Action")
    fields = descriptors[kind]["required_fields"]
    closed(value, {"type"} | set(fields))
    refs: list[str] = []
    for name in fields:
        item = value[name]
        if name in ("ordered_card_refs", "card_refs"):
            if not isinstance(item, list):
                raise ContractError("ordered card refs array required")
            for ref in item:
                card_ref(ref)
            refs.extend(item)
        elif name in ("card_ref", "sold_card_ref"):
            card_ref(item)
            refs.append(item)
        elif name == "target":
            player(item)
        elif name == "accept":
            boolean(item)
        else:
            integer(item, -(2**31) if name == "hp" else 0, 2**31 - 1 if name == "hp" else 2**32 - 1)
    if submission and (len(refs) != len(set(refs)) or len(refs) > 18):
        raise ContractError("repeated card ref or capacity exceeded")
    if obs is not None:
        # Callers pass an already representation-validated original view.
        if kind not in {item["type"] for item in action_descriptors(obs["required_choice"])}:
            raise ContractError("Action kind differs from required choice")
        if not set(refs) <= {card["card_ref"] for card in obs["own"]["cards"]}:
            raise ContractError("Action contains foreign card ref")
    return clone(value)


def validate_action(value: Any, obs: Json | None = None) -> Json:
    """Submission/accepted-Action syntax with distinct bounded own references."""
    return _validate_action(value, obs, submission=True)


def _action_options(value: Any, obs: Json) -> Json:
    fields = (
        {
            "action_options_schema_version",
            "rule_set_version",
            "card_ref_scheme",
            "actor",
            "state_revision",
            "required_choice",
            "candidate_groups",
            "valid_targets",
        }
        | _FLAGS
        | _CARD_LISTS
    )
    closed(value, fields)
    for name, expected in (
        ("action_options_schema_version", "2"),
        ("rule_set_version", RULE_SET),
        ("card_ref_scheme", CARD_REF_SCHEME),
    ):
        _pin(value, name, expected)
    player(value["actor"])
    integer(value["state_revision"])
    if (
        value["actor"] != obs["viewer_player_id"]
        or value["actor"] != obs["current_player"]
        or value["state_revision"] != obs["state_revision"]
        or value["required_choice"] != obs["required_choice"]
    ):
        raise ContractError("ActionOptions actor/revision/choice mismatch")
    own = {card["card_ref"]: card["definition_id"] for card in obs["own"]["cards"]}

    def entries(rows: Any) -> set[str]:
        if not isinstance(rows, list):
            raise ContractError("candidate array required")
        seen: set[str] = set()
        for row in rows:
            closed(row, {"card_ref", "definition_id"})
            card_ref(row["card_ref"])
            text_value(row["definition_id"])
            if row["card_ref"] in seen or own.get(row["card_ref"]) != row["definition_id"]:
                raise ContractError("duplicate/foreign/perception-mismatched atomic hint")
            seen.add(row["card_ref"])
        return seen

    lists = {name: entries(value[name]) for name in _CARD_LISTS}
    if not isinstance(value["candidate_groups"], dict):
        raise ContractError("candidate groups object required")
    for name, rows in value["candidate_groups"].items():
        text_value(name)
        if not entries(rows) <= lists["usable_cards"]:
            raise ContractError("candidate group is not a usable-card subset")
    targets = value["valid_targets"]
    if not isinstance(targets, list):
        raise ContractError("target array required")
    for target in targets:
        player(target)
    if len(targets) != len(set(targets)):
        raise ContractError("duplicate target hint")
    for name in _FLAGS:
        boolean(value[name])
    return clone(value)


def action_options(value: Any, obs: Json) -> Json:
    obs = observation(obs)
    if not obs["actionable"]:
        raise ContractError("ActionOptions require the actionable actor")
    return _action_options(value, obs)


def actor_action_options(value: Any, obs: Json, *, raw: bool = True) -> Json:
    obs = observation(obs)
    closed(value, {"state_revision", "action_options"} | (set() if raw else {"viewer_player_id"}))
    integer(value["state_revision"])
    if value["state_revision"] != obs["state_revision"]:
        raise ContractError("ActionOptions wrapper revision mismatch")
    if not raw:
        player(value["viewer_player_id"])
        if value["viewer_player_id"] != obs["viewer_player_id"]:
            raise ContractError("ActionOptions wrapper actor mismatch")
    action_options(value["action_options"], obs)
    return clone(value)


def catalog(value: Any) -> Json:
    closed(value, set(CATALOG_META) | {"definitions"})
    for name, expected in CATALOG_META.items():
        _pin(value, name, expected)
    if not isinstance(value["definitions"], list) or len(value["definitions"]) != 296:
        raise ContractError("catalog count mismatch")
    ids = []
    for definition in value["definitions"]:
        if not isinstance(definition, dict) or "classification" not in definition:
            raise ContractError("public v2 classification required")
        validate_definition(
            {key: item for key, item in definition.items() if key != "classification"}
        )
        classification = closed(
            definition["classification"], {"definition_class", "use_roles"}, {"miracle_subclass"}
        )
        if classification["definition_class"] not in ("owned_card", "operation", "automatic"):
            raise ContractError("unknown definition class")
        roles = classification["use_roles"]
        if (
            not isinstance(roles, list)
            or any(type(role) is not str or role not in _ROLES for role in roles)
            or len(roles) != len(set(roles))
        ):
            raise ContractError("unknown or repeated use role")
        if "miracle_subclass" in classification and (
            type(classification["miracle_subclass"]) is not str
            or classification["miracle_subclass"] not in _SUBCLASSES
        ):
            raise ContractError("unknown miracle subclass")
        ids.append(definition["definition_id"])
    if (
        ids != sorted(set(ids))
        or digest({key: item for key, item in value.items() if key != "content_sha256"})
        != value["content_sha256"]
    ):
        raise ContractError("catalog ordering/hash mismatch")
    return {definition["definition_id"]: clone(definition) for definition in value["definitions"]}


def public_info(value: Any) -> Json:
    closed(
        value,
        {
            "name",
            "version",
            "api_version",
            "observation_schema_version",
            "action_schema_version",
            "action_options_schema_version",
            "public_event_schema_version",
            "card_ref_scheme",
            "rule_set_version",
            "random_algorithm",
            "random_algorithm_version",
            "card_catalog",
        },
    )
    for name, expected in (
        ("name", "gf-simulator"),
        ("api_version", "v2"),
        ("observation_schema_version", "2"),
        ("action_schema_version", 2),
        ("action_options_schema_version", "2"),
        ("public_event_schema_version", "2"),
        ("card_ref_scheme", CARD_REF_SCHEME),
        ("rule_set_version", RULE_SET),
        ("random_algorithm", "xoshiro256pp"),
        ("random_algorithm_version", "1"),
    ):
        _pin(value, name, expected)
    text_value(value["version"])
    closed(value["card_catalog"], set(CATALOG_META))
    for name, expected in CATALOG_META.items():
        _pin(value["card_catalog"], name, expected)
    return clone(value)


def validate_events(value: Any) -> None:
    if not isinstance(value, list):
        raise ContractError("event array required")
    for event in value:
        if isinstance(event, dict) and event.get("type") == "safe":
            closed(event, {"type", "player"})
            player(event["player"])
        else:
            _legacy_events([event])
        if (
            isinstance(event, dict)
            and event.get("type") == "status_changed"
            and event["status"]
            not in ("cold", "fever", "hell", "heaven", "fog", "flash", "dream", "dark_cloud")
        ):
            raise ContractError("unknown public status")


def event_feed(value: Any) -> Json:
    closed(
        value,
        {"public_event_schema_version", "rule_set_version", "events", "next_cursor", "has_more"},
    )
    _pin(value, "public_event_schema_version", "2")
    _pin(value, "rule_set_version", RULE_SET)
    validate_events(value["events"])
    boolean(value["has_more"])
    if value["next_cursor"] is not None:
        text_value(value["next_cursor"])
    return clone(value)


def command_id(value: Any) -> None:
    if type(value) is not str:
        raise ContractError("canonical command UUID required")
    try:
        valid = str(uuid.UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ContractError("canonical command UUID required")


def action_command(value: Any) -> Json:
    closed(value, {"command_id", "expected_revision", "action"})
    command_id(value["command_id"])
    integer(value["expected_revision"])
    validate_action(value["action"])
    return clone(value)


def join_response(value: Any, game: str) -> Json:
    closed(
        value,
        {"game_id", "player_id", "status", "joined_players", "required_players", "state_revision"},
    )
    player(value["player_id"])
    integer(value["joined_players"], 1, 2)
    _pin(value, "required_players", 2)
    if value["game_id"] != game or value["status"] not in ("waiting_for_players", "active"):
        raise ContractError("invalid Join acknowledgement")
    if value["state_revision"] is not None:
        integer(value["state_revision"])
    if (value["status"] == "active") != (value["joined_players"] == 2):
        raise ContractError("Join status/count mismatch")
    if (value["state_revision"] is not None) != (value["status"] == "active"):
        raise ContractError("Join status/revision mismatch")
    return clone(value)


def act_response(value: Any, payload: Json, viewer: int) -> Json:
    closed(
        value,
        {
            "public_event_schema_version",
            "rule_set_version",
            "card_ref_scheme",
            "command_id",
            "previous_revision",
            "state_revision",
            "status",
            "events",
            "result",
            "next_observation",
            "action_options",
        },
    )
    _pin(value, "public_event_schema_version", "2")
    _pin(value, "rule_set_version", RULE_SET)
    _pin(value, "card_ref_scheme", CARD_REF_SCHEME)
    integer(value["previous_revision"])
    integer(value["state_revision"], value["previous_revision"] + 1)
    if (
        value["command_id"] != payload["command_id"]
        or value["previous_revision"] != payload["expected_revision"]
    ):
        raise ContractError("invalid action acknowledgement identity")
    if value["status"] not in ("active", "terminal", "aborted"):
        raise ContractError("unknown action acknowledgement status")
    result(value["result"])
    if (value["status"] == "terminal") != (value["result"] is not None):
        raise ContractError("acknowledgement status/result mismatch")
    validate_events(value["events"])
    obs = value["next_observation"]
    if obs is not None:
        _observation(obs, viewer, representation="http", disabled=value["status"] != "active")
        if obs["state_revision"] != value["state_revision"]:
            raise ContractError("acknowledgement observation revision mismatch")
        if obs["result"] != value["result"]:
            raise ContractError("acknowledgement observation result mismatch")
    if value["action_options"] is not None:
        if obs is None:
            raise ContractError("ActionOptions lack corresponding acknowledgement observation")
        action_options(value["action_options"], obs)
    return clone(value)


def _closure(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ContractError("episode closure object required")
    if value.get("type") == "terminal":
        closed(value, {"type", "result"})
        result(value["result"])
        if value["result"] is None:
            raise ContractError("terminal result required")
    elif value.get("type") == "aborted":
        closed(value, {"type", "reason", "player", "decision"})
        text_value(value["reason"])
        if value["player"] is not None:
            player(value["player"])
        decision = closed(value["decision"], {"turn", "decision_index", "invalid_attempt_count"})
        for name, item in decision.items():
            if item is not None or name == "invalid_attempt_count":
                integer(item, 0, 2**32 - 1)
    else:
        raise UnsupportedContract("unknown closure kind")


def replay(value: Any, game: str, viewer: int) -> Json:
    closed(
        value,
        {
            "game_id",
            "viewer_player_id",
            "requested_turn",
            "rule_set_version",
            "observation_schema_version",
            "public_event_schema_version",
            "card_ref_scheme",
            "episode_end",
            "turns",
        },
    )
    player(value["viewer_player_id"])
    if value["game_id"] != game or value["viewer_player_id"] != viewer:
        raise ContractError("replay identity mismatch")
    for name, expected in (
        ("rule_set_version", RULE_SET),
        ("observation_schema_version", "2"),
        ("public_event_schema_version", "2"),
        ("card_ref_scheme", CARD_REF_SCHEME),
    ):
        _pin(value, name, expected)
    if value["requested_turn"] is not None:
        integer(value["requested_turn"], 0, 2**32 - 1)
    _closure(value["episode_end"])
    if not isinstance(value["turns"], list):
        raise ContractError("replay turn array required")
    for turn in value["turns"]:
        closed(turn, {"turn", "active_player", "decisions"})
        integer(turn["turn"], 0, 2**32 - 1)
        player(turn["active_player"])
        if not isinstance(turn["decisions"], list):
            raise ContractError("replay decision array required")
        for decision in turn["decisions"]:
            closed(
                decision,
                {
                    "actor",
                    "observation",
                    "action_options",
                    "visible_action",
                    "visible_events",
                    "aborted",
                },
            )
            player(decision["actor"])
            obs = replay_observation(decision["observation"], viewer, aborted=decision["aborted"])
            if decision["actor"] != obs["current_player"]:
                raise ContractError("replay actor differs from original observation")
            if decision["action_options"] is not None:
                _action_options(decision["action_options"], obs)
            visible = decision["visible_action"]
            if visible is not None:
                if not isinstance(visible, dict):
                    raise ContractError("visible Action object required")
                if visible.get("type") == "concealed":
                    closed(visible, {"type"})
                elif visible.get("type") == "action":
                    closed(visible, {"type", "action"})
                    validate_action(visible["action"], obs if decision["actor"] == viewer else None)
                else:
                    raise ContractError("unknown visible Action kind")
            validate_events(decision["visible_events"])
    return clone(value)


def dataset_action(value: Any, *, action_schema_version: Any) -> Json:
    """Preserve original typed attempts, including rejected duplicate/long rows.

    This validates representation only. It grants no candidate, label, accepted
    transition, or Source eligibility authority.
    """
    if type(action_schema_version) is not str or action_schema_version != "2":
        raise UnsupportedContract("Dataset Action schema must be string 2")
    return _validate_action(value, None, submission=False)
