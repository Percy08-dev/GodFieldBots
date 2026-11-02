"""Model-neutral exact token layout, shared by Python models and future native code.

All tuples are immutable owned raw tokens. No learned representation is retained.
Unknown categories reject. PAD0 and MISSING1 are distinct from observed values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gf_common import structured_vocab as vocab
from gf_common.domain import CATALOG_HASH, SIMULATOR_REVISION, ContractError, digest, integer
from gf_common.structured import (
    ACTION_KINDS,
    INPUT_RECIPE,
    TARGETS,
    CandidateTable,
    DecisionFrame,
    EventBankView,
)
from gf_common.structured_history import (
    ACTIVE_AGGREGATE,
    APPENDED_HISTORY_FIELDS,
    HISTORICAL_AGGREGATE,
    HISTORY_WIDTH,
    LEGACY_HISTORY_WIDTH,
    exact_history_tail,
)

PAD, MISSING = 0, 1
EVENT_TYPES = ("turn_started", "action_accepted", "artifact_used", "artifact_drawn", "devil_appeared",
               "artifact_discarded", "guardian_set", "guardian_left", "guardian_acted", "miracle_used",
               "attack_declared", "attack_missed", "damage_applied", "healed", "resource_changed",
               "buy_offered", "sale_offered", "artifact_transferred", "sun_amulet_activated",
               "status_changed", "phenomenon_invoked", "player_ascended", "player_defeated", "game_ended")
# Both spellings are public identifiers: Observation.status uses dark_cloud,
# public-ja-v2 curse.id uses darkcloud. These are separate exact categories;
# this supported vocabulary does not claim an unpublished exhaustive Event enum.
STATUSES = ("cold", "fever", "hell", "heaven", "fog", "flash", "dream", "dark_cloud", "darkcloud")
PHASES = ("awaiting_action", "awaiting_response", "resolving", "terminal")
CHOICES = ("turn_action", "defense", "buy_accept_reject", "additional_choice")
RELATIVE = ("self", "opponent")
RESULTS = ("self_win", "opponent_win", "draw")
SOURCES = ("hand_artifact", "learned_miracle")
STATE_NUMERIC_FIELDS = ("turn", "turns_until_end_of_world", "end_times", "actionable",
                        "own.hp", "own.mp", "own.money", "own.ascended", "own.fog", "own.flash",
                        "own.dream", "own.dark_cloud", "opponent.hp", "opponent.mp", "opponent.money",
                        "opponent.ascended", "opponent.fog", "opponent.flash", "opponent.dream",
                        "opponent.dark_cloud", "can_pray", "can_discard", "can_sacrifice", "can_allow",
                        "target_self", "target_opponent")
STATE_CATEGORIES = (PHASES, CHOICES, RELATIVE, RELATIVE, RELATIVE, RESULTS,
                    vocab.DISEASES, vocab.GUARDIANS, vocab.DISEASES, vocab.GUARDIANS)
CARD_NUMERIC_FIELDS = ("attack", "defense", "ability_value", "price", "mp_cost", "hit_rate",
                       "additional_attack", "usable", "learned_order", "exchange", "sell", "buy")
CARD_CATEGORIES = (vocab.DEFINITIONS, vocab.ABILITIES, vocab.KINDS, vocab.ATTRIBUTES,
                   vocab.DISEASES, vocab.GUARDIANS, SOURCES)
EVENT_ACTORS = ("player", "attacker", "target", "active_player", "buyer", "seller", "from", "to")
EVENT_NUMERIC_FIELDS = ("turn", "price", "mp_spent", "attack", "amount", "remaining_hp", "hp",
                        "hp_delta", "mp_delta", "money_delta", "active")
EVENT_CATEGORIES = (EVENT_TYPES, vocab.DEFINITIONS, STATUSES, RESULTS, *(RELATIVE for _ in EVENT_ACTORS))
STATE_WIDTH = 2 * len(STATE_NUMERIC_FIELDS) + len(STATE_CATEGORIES) + sum(len(c) + 2 for c in STATE_CATEGORIES) + 18 * (len(vocab.DEFINITIONS) + 2)
CARD_DIRECT_WIDTH = 2 * len(CARD_NUMERIC_FIELDS) + sum(len(c) + 2 for c in CARD_CATEGORIES)
EVENT_WIDTH = 2 * len(EVENT_NUMERIC_FIELDS) + sum(len(c) + 2 for c in EVENT_CATEGORIES)
CANDIDATE_WIDTH = len(ACTION_KINDS) + 2 + len(TARGETS) + 2 + 8


def category(value: Any, values: tuple[str, ...]) -> int:
    if value is None:
        return MISSING
    try:
        return values.index(value) + 2
    except ValueError as exc:
        raise ContractError(f"unknown structured category: {value!r}") from exc


def _relative(value: Any, actor: int) -> str | None:
    return None if value is None else "self" if value == actor else "opponent"


def _result(value: Any, actor: int) -> str | None:
    if value is None or value == "draw":
        return value
    return "self_win" if value["win"]["winner"] == actor else "opponent_win"


def signature_payload(aggregate_window: int = 128, *, aggregate_recipe: str = ACTIVE_AGGREGATE) -> dict[str, Any]:
    integer(aggregate_window, 1, 65536)
    payload = {"recipe": INPUT_RECIPE, "revision": SIMULATOR_REVISION, "catalog": CATALOG_HASH,
                   "pad": PAD, "missing": MISSING, "state_numeric": STATE_NUMERIC_FIELDS,
                   "state_categories": STATE_CATEGORIES, "state_category_presence": True, "card_numeric": CARD_NUMERIC_FIELDS,
                   "card_categories": CARD_CATEGORIES, "event_numeric": EVENT_NUMERIC_FIELDS,
                   "event_categories": EVENT_CATEGORIES, "history_window": aggregate_window,
                   "history": "event-counts/actor-role-counts/numeric-sums-and-presence-v1",
                   "candidate_categories": (ACTION_KINDS, TARGETS),
                   "candidate_numeric": ("hp", "mp", "money", "accept"), "slots": 18,
                   "numeric_transform": "signed-log1p-fp32-v1"}
    if aggregate_recipe == ACTIVE_AGGREGATE:
        payload.update(aggregate_recipe=ACTIVE_AGGREGATE, history_append=APPENDED_HISTORY_FIELDS,
            history_actor_priority=('player', 'attacker', 'active_player'), history_actor_fallback='key-presence')
    elif aggregate_recipe != HISTORICAL_AGGREGATE:
        raise ContractError('unsupported structured aggregate recipe')
    return payload


def signature(aggregate_window: int = 128, *, aggregate_recipe: str = ACTIVE_AGGREGATE) -> str:
    return digest(signature_payload(aggregate_window, aggregate_recipe=aggregate_recipe))


def recorded_aggregate_recipe(recorded_signature: str, aggregate_window: int) -> str:
    for recipe in (ACTIVE_AGGREGATE, HISTORICAL_AGGREGATE):
        if recorded_signature == signature(aggregate_window, aggregate_recipe=recipe):
            return recipe
    raise ContractError('unknown recorded structured aggregate signature')


@dataclass(frozen=True)
class FrameTokens:
    frame_hash: str
    recipe_signature: str
    aggregate_window: int
    actor: int
    episode: str
    revision: int
    event_start: int
    event_end: int
    state_numerics: tuple[int, ...]
    state_present: tuple[bool, ...]
    state_categories: tuple[int, ...]
    state_category_present: tuple[bool, ...]
    card_numerics: tuple[tuple[int, ...], ...]
    card_present: tuple[tuple[bool, ...], ...]
    card_categories: tuple[tuple[int, ...], ...]
    card_mask: tuple[bool, ...]
    opponent_learned: tuple[int, ...]
    interval_numerics: tuple[tuple[int, ...], ...]
    interval_present: tuple[tuple[bool, ...], ...]
    interval_categories: tuple[tuple[int, ...], ...]
    history: tuple[int, ...]


@dataclass(frozen=True)
class CandidateTokens:
    table_hash: str
    frame_hash: str
    recipe_signature: str
    kinds: tuple[int, ...]
    targets: tuple[int, ...]
    numerics: tuple[tuple[int, ...], ...]
    present: tuple[tuple[bool, ...], ...]
    offsets: tuple[int, ...]
    slots: tuple[int, ...]


def _numbers(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    return tuple(int(row.get(key, 0)) for key in fields), tuple(key in row for key in fields)


def _event(event: dict[str, Any], actor: int) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]]:
    numeric, present = _numbers(event, EVENT_NUMERIC_FIELDS)
    categories = (category(event["type"], EVENT_TYPES), category(event.get("definition_id"), vocab.DEFINITIONS),
                  category(event.get("status"), STATUSES), category(_result(event.get("result"), actor), RESULTS),
                  *(category(_relative(event.get(key), actor), RELATIVE) for key in EVENT_ACTORS))
    return numeric, present, categories


def tokenize_frame(frame: DecisionFrame, bank_view: EventBankView, *, aggregate_window: int = 128,
                   aggregate_recipe: str = ACTIVE_AGGREGATE) -> FrameTokens:
    integer(aggregate_window, 1, 65536)
    view = frame.bank_view(bank_view)
    obs, payload = frame.observation, frame.raw
    options = payload["action_options"]["action_options"]
    values: dict[str, Any] = {key: obs[key] for key in STATE_NUMERIC_FIELDS[:4]}
    for side in ("own", "opponent"):
        participant = obs[side]
        for key in ("hp", "mp", "money", "ascended"):
            if key in participant:
                values[f"{side}.{key}"] = participant[key]
        for key in ("fog", "flash", "dream", "dark_cloud"):
            if key in participant.get("status", {}):
                values[f"{side}.{key}"] = participant["status"][key]
    for key in ("can_pray", "can_discard", "can_sacrifice", "can_allow"):
        values[key] = options[key]
    for name, player in (("self", frame.actor), ("opponent", 3 - frame.actor)):
        values[f"target_{name}"] = player in options["valid_targets"]
    state, state_present = _numbers(values, STATE_NUMERIC_FIELDS)
    cats = (category(obs["phase"], PHASES), category(obs["required_choice"], CHOICES),
            *(category(_relative(obs[key], frame.actor), RELATIVE) for key in ("first_player", "active_player", "current_player")),
            category(_result(obs["result"], frame.actor), RESULTS),
            *(category(obs[side].get("status", {}).get("disease"), vocab.DISEASES) if key == "disease"
              else category(obs[side].get("guardian"), vocab.GUARDIANS)
              for side in ("own", "opponent") for key in ("disease", "guardian")))
    category_present = (True,) * 6 + tuple(
        ("status" in obs[side]) if key == "disease" else ("guardian" in obs[side])
        for side in ("own", "opponent") for key in ("disease", "guardian"))
    card_numbers: list[tuple[int, ...]] = []
    card_presence: list[tuple[bool, ...]] = []
    card_cats: list[tuple[int, ...]] = []
    own = obs["own"]
    cards = [(card["definition_id"], "hand_artifact", card["instance_id"], 0) for card in own["hand"]]
    cards += [(name, "learned_miracle", name, order + 1) for order, name in enumerate(own["learned_miracles"])]
    for name, source, ref, order in cards:
        definition = payload["catalog"][name]
        numbers = {key: definition[key] for key in CARD_NUMERIC_FIELDS if key in definition}
        numbers.update(usable=ref in options["usable_hand_elements" if source == "hand_artifact" else "usable_learned_miracles"],
                       learned_order=order)
        for ability in ("exchange", "sell", "buy"):
            numbers[ability] = source == "hand_artifact" and ref in options[ability + "_artifacts"]
        numeric, present = _numbers(numbers, CARD_NUMERIC_FIELDS)
        card_numbers.append(numeric)
        card_presence.append(present)
        card_cats.append((category(name, vocab.DEFINITIONS), category(definition.get("ability"), vocab.ABILITIES),
                          category(definition["kind"], vocab.KINDS), category(definition.get("attribute"), vocab.ATTRIBUTES),
                          category(definition.get("curse", {}).get("id"), vocab.DISEASES),
                          category(definition.get("guardian", {}).get("id"), vocab.GUARDIANS), category(source, SOURCES)))
    count = len(cards)
    while len(card_numbers) < 18:
        card_numbers.append((0,) * len(CARD_NUMERIC_FIELDS))
        card_presence.append((False,) * len(CARD_NUMERIC_FIELDS))
        card_cats.append((PAD,) * len(CARD_CATEGORIES))
    opponent = tuple(category(name, vocab.DEFINITIONS) for name in obs["opponent"]["learned_miracles"])
    opponent += (PAD,) * (18 - len(opponent))
    interval = [_event(event, frame.actor) for event in view.events(frame.interval_start)]
    history = [0] * LEGACY_HISTORY_WIDTH
    events = view.events(max(0, view.boundary.end - aggregate_window))
    for event in events:
        event_numbers, event_presence, categories = _event(event, frame.actor)
        history[categories[0] - 2] += 1
        for role, value in enumerate(categories[4:]):
            if value >= 2:
                history[len(EVENT_TYPES) + 2 * role + value - 2] += 1
        offset = len(EVENT_TYPES) + 2 * len(EVENT_ACTORS)
        for index, (number, event_exists) in enumerate(zip(event_numbers, event_presence, strict=True)):
            history[offset + index * 2] += number
            history[offset + index * 2 + 1] += event_exists
    if aggregate_recipe == ACTIVE_AGGREGATE:
        history.extend(exact_history_tail(events, frame.actor, prefix_count=view.boundary.end, window=aggregate_window))
    elif aggregate_recipe != HISTORICAL_AGGREGATE:
        raise ContractError('unsupported structured aggregate recipe')
    return FrameTokens(frame.frame_hash, signature(aggregate_window, aggregate_recipe=aggregate_recipe), aggregate_window, frame.actor, frame.episode, frame.revision,
                       frame.interval_start, frame.boundary.end, state, state_present, cats, category_present, tuple(card_numbers),
                       tuple(card_presence), tuple(card_cats), tuple(i < count for i in range(18)), opponent,
                       tuple(item[0] for item in interval), tuple(item[1] for item in interval),
                       tuple(item[2] for item in interval), tuple(history))


def tokenize_table(table: CandidateTable, frame: DecisionFrame, max_positions: int = 128, *,
                   aggregate_window: int = 128, aggregate_recipe: str = ACTIVE_AGGREGATE) -> CandidateTokens:
    integer(max_positions, 1, 128)
    table.validate(frame)
    if any(b - a > max_positions for a, b in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
        raise ContractError("candidate token position capacity exceeded")
    return CandidateTokens(table.table_hash, frame.frame_hash, signature(aggregate_window, aggregate_recipe=aggregate_recipe),
                           tuple(category(kind, ACTION_KINDS) for kind in table.kinds),
                           tuple(category(target, TARGETS) for target in table.targets), table.numerics,
                           table.presence, table.offsets, table.slots)


def validate_frame_tokens(tokens: FrameTokens) -> None:
    """Validate every shape/category/mask before array allocation."""
    if type(tokens) is not FrameTokens:
        raise ContractError("structured frame token recipe mismatch")
    recipe = recorded_aggregate_recipe(tokens.recipe_signature, tokens.aggregate_window)

    for name in ("state_numerics", "state_present", "state_categories", "state_category_present",
                 "card_numerics", "card_present", "card_categories", "card_mask", "opponent_learned",
                 "interval_numerics", "interval_present", "interval_categories", "history"):
        if type(getattr(tokens, name)) is not tuple:
            raise ContractError("raw token buffers must be immutable tuples")
    integer(tokens.actor, 1, 2)
    integer(tokens.revision)
    integer(tokens.event_start, 0, tokens.event_end)
    integer(tokens.event_end, tokens.event_start, 2**31 - 1)
    if (not isinstance(tokens.episode, str) or not tokens.episode
            or tokens.event_end - tokens.event_start != len(tokens.interval_numerics)):
        raise ContractError("frame token owner/interval mismatch")

    def numeric(row: tuple[int, ...], present: tuple[bool, ...], width: int) -> None:
        if type(row) is not tuple or type(present) is not tuple or len(row) != width or len(present) != width:
            raise ContractError("structured numeric token shape mismatch")
        for number, exists in zip(row, present, strict=True):
            integer(number, -(2**63), 2**63 - 1)
            if type(exists) is not bool or not exists and number != 0:
                raise ContractError("structured token missingness mismatch")

    def categories(row: tuple[int, ...], values: tuple[tuple[str, ...], ...]) -> None:
        if type(row) is not tuple or len(row) != len(values):
            raise ContractError("structured categorical token shape mismatch")
        for value, vocabulary in zip(row, values, strict=True):
            integer(value, 0, len(vocabulary) + 1)

    numeric(tokens.state_numerics, tokens.state_present, len(STATE_NUMERIC_FIELDS))
    categories(tokens.state_categories, STATE_CATEGORIES)
    if any(value == PAD for value in tokens.state_categories):
        raise ContractError("state category cannot be padding")
    if len(tokens.state_category_present) != len(STATE_CATEGORIES) or any(type(v) is not bool for v in tokens.state_category_present):
        raise ContractError("state category missingness shape mismatch")
    if any(len(value) != 18 for value in (tokens.card_numerics, tokens.card_present, tokens.card_categories,
                                          tokens.card_mask, tokens.opponent_learned)):
        raise ContractError("18 card slots required")
    for numbers, present, cats, valid in zip(tokens.card_numerics, tokens.card_present,
                                           tokens.card_categories, tokens.card_mask, strict=True):
        numeric(numbers, present, len(CARD_NUMERIC_FIELDS))
        categories(cats, CARD_CATEGORIES)
        if type(valid) is not bool or not valid and (any(numbers) or any(present) or any(cats)):
            raise ContractError("invalid card padding")
        if valid and any(value == PAD for value in cats):
            raise ContractError("present card category cannot be padding")
    for value in tokens.opponent_learned:
        integer(value, 0, len(vocab.DEFINITIONS) + 1)
        if value == MISSING:
            raise ContractError("opponent learned ID cannot be missing")
    if not (len(tokens.interval_numerics) == len(tokens.interval_present) == len(tokens.interval_categories) <= 65536):
        raise ContractError("interval token shape/cap mismatch")
    for numbers, present, cats in zip(tokens.interval_numerics, tokens.interval_present,
                                     tokens.interval_categories, strict=True):
        numeric(numbers, present, len(EVENT_NUMERIC_FIELDS))
        categories(cats, EVENT_CATEGORIES)
        if any(value == PAD for value in cats):
            raise ContractError("event category cannot be padding")
    if len(tokens.history) != (HISTORY_WIDTH if recipe == ACTIVE_AGGREGATE else LEGACY_HISTORY_WIDTH):
        raise ContractError("history token shape mismatch")
    for value in tokens.history:
        integer(value, -(2**63), 2**63 - 1)


def validate_candidate_tokens(tokens: CandidateTokens, frame: FrameTokens, max_positions: int) -> None:
    if (type(tokens) is not CandidateTokens or tokens.recipe_signature != frame.recipe_signature
            or tokens.frame_hash != frame.frame_hash):
        raise ContractError("candidate token binding mismatch")
    for name in ("kinds", "targets", "numerics", "present", "offsets", "slots"):
        if type(getattr(tokens, name)) is not tuple:
            raise ContractError("candidate token buffers must be immutable tuples")
    count = len(tokens.kinds)
    if not (1 <= count <= 65536 and len(tokens.targets) == len(tokens.numerics) == len(tokens.present) == count
            and len(tokens.offsets) == count + 1 and tokens.offsets[0] == 0
            and tokens.offsets[-1] == len(tokens.slots)):
        raise ContractError("candidate token column mismatch")
    for index in range(count):
        integer(tokens.kinds[index], 2, len(ACTION_KINDS) + 1)
        integer(tokens.targets[index], 2, len(TARGETS) + 1)
        integer(tokens.offsets[index], 0, len(tokens.slots))
        integer(tokens.offsets[index + 1], tokens.offsets[index],
                min(len(tokens.slots), tokens.offsets[index] + max_positions))
        if (type(tokens.numerics[index]) is not tuple or type(tokens.present[index]) is not tuple
                or len(tokens.numerics[index]) != 4 or len(tokens.present[index]) != 4):
            raise ContractError("candidate token numeric shape mismatch")
        for number, exists in zip(tokens.numerics[index], tokens.present[index], strict=True):
            integer(number, -(2**31), 2**32 - 1)
            if type(exists) is not bool or not exists and number != 0:
                raise ContractError("candidate token numeric presence mismatch")
    for slot in tokens.slots:
        integer(slot, 0, 17)
        if not frame.card_mask[slot]:
            raise ContractError("candidate token references padded card")
