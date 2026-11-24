"""Model-neutral exact token layout, shared by Python models and future native code.

All tuples are immutable owned raw tokens. No learned representation is retained.
Unknown categories reject. PAD0 and MISSING1 are distinct from observed values.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from gf_common import structured_layout as layouts
from gf_common import structured_vocab as vocab
from gf_common.card_instance_validation import is_concealed_draw
from gf_common.domain import ContractError, Json, digest, integer
from gf_common.structured import (
    CandidateTable,
    DecisionFrame,
    EventBankView,
)
from gf_common.structured_history import (
    ACTIVE_AGGREGATE,
    APPENDED_HISTORY_FIELDS,
    HISTORICAL_AGGREGATE,
    LEGACY_HISTORY_WIDTH,
    appended_history_fields,
    exact_history_tail,
)
from gf_common.structured_layout import (
    LEGACY_LAYOUT,
    StructuredLayout,
    is_card_instance_layout,
    require_layout,
)

ACTION_KINDS = layouts.ACTION_KINDS
INPUT_RECIPE = layouts.INPUT_RECIPE
TARGETS = layouts.TARGETS
CATALOG_HASH = layouts.CATALOG_HASH
SIMULATOR_REVISION = layouts.SIMULATOR_REVISION
HISTORY_WIDTH = layouts.HISTORY_WIDTH

PAD = layouts.PAD
MISSING = layouts.MISSING
EVENT_TYPES = layouts.EVENT_TYPES
STATUSES = layouts.STATUSES
PHASES = layouts.PHASES
CHOICES = layouts.CHOICES
RELATIVE = layouts.RELATIVE
RESULTS = layouts.RESULTS
SOURCES = layouts.SOURCES
STATE_NUMERIC_FIELDS = layouts.STATE_NUMERIC_FIELDS
STATE_CATEGORIES = layouts.STATE_CATEGORIES
CARD_NUMERIC_FIELDS = layouts.CARD_NUMERIC_FIELDS
CARD_CATEGORIES = layouts.CARD_CATEGORIES
EVENT_ACTORS = layouts.EVENT_ACTORS
EVENT_NUMERIC_FIELDS = layouts.EVENT_NUMERIC_FIELDS
EVENT_CATEGORIES = layouts.EVENT_CATEGORIES
STATE_WIDTH = layouts.STATE_WIDTH
CARD_DIRECT_WIDTH = layouts.CARD_DIRECT_WIDTH
EVENT_WIDTH = layouts.EVENT_WIDTH
CANDIDATE_WIDTH = layouts.CANDIDATE_WIDTH

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


def signature_payload(aggregate_window: int = 128, *, aggregate_recipe: str | None = None,
                      layout: StructuredLayout = LEGACY_LAYOUT) -> dict[str, Any]:
    require_layout(layout)
    aggregate_recipe = layout.aggregate_recipe if aggregate_recipe is None else aggregate_recipe
    integer(aggregate_window, 1, 65536)
    payload = {"recipe": layout.input_recipe, "revision": layout.simulator_revision, "catalog": layout.catalog_sha256,
                   "pad": PAD, "missing": MISSING, "state_numeric": STATE_NUMERIC_FIELDS,
                   "state_categories": STATE_CATEGORIES, "state_category_presence": True, "card_numeric": layout.card_numeric_fields,
                   "card_categories": layout.card_categories, "event_numeric": EVENT_NUMERIC_FIELDS,
                   "event_categories": layout.event_categories, "history_window": aggregate_window,
                   "history": "event-counts/actor-role-counts/numeric-sums-and-presence-v1",
                   "candidate_categories": (layout.action_kinds, layout.targets),
                   "candidate_numeric": ("hp", "mp", "money", "accept"), "slots": 18,
                   "numeric_transform": "signed-log1p-fp32-v1"}
    if is_card_instance_layout(layout):
        if aggregate_recipe != layout.aggregate_recipe:
            raise ContractError("unsupported card-instance history recipe")
        history_fields = (tuple("event." + kind for kind in layout.event_types)
            + tuple("actor." + role + "." + side for role in EVENT_ACTORS for side in RELATIVE)
            + tuple(prefix + name for name in EVENT_NUMERIC_FIELDS for prefix in ("sum.", "present."))
            + appended_history_fields(layout.event_categories[1]))
        payload.update(action_contract=asdict(layout.action_contract), policy_input_version=layout.policy_input_version,
            generator_version=layout.generator, aggregate_recipe=aggregate_recipe, history_fields=history_fields,
            history_actor_priority=("player", "attacker", "active_player"), history_actor_fallback="key-presence",
            state_presence=True, card_presence=True, card_mask=True, event_presence=True,
            candidate_presence=True, format_versions=layout.format_versions)
    elif aggregate_recipe == ACTIVE_AGGREGATE:
        payload.update(aggregate_recipe=ACTIVE_AGGREGATE, history_append=APPENDED_HISTORY_FIELDS,
            history_actor_priority=('player', 'attacker', 'active_player'), history_actor_fallback='key-presence')
    elif aggregate_recipe != HISTORICAL_AGGREGATE:
        raise ContractError('unsupported structured aggregate recipe')
    return payload


def signature(aggregate_window: int = 128, *, aggregate_recipe: str | None = None,
              layout: StructuredLayout = LEGACY_LAYOUT) -> str:
    return digest(signature_payload(aggregate_window, aggregate_recipe=aggregate_recipe, layout=layout))


def recorded_aggregate_recipe(recorded_signature: str, aggregate_window: int, *,
                              layout: StructuredLayout = LEGACY_LAYOUT) -> str:
    require_layout(layout)
    for recipe in ((ACTIVE_AGGREGATE, HISTORICAL_AGGREGATE) if layout is LEGACY_LAYOUT else (layout.aggregate_recipe,)):
        if recorded_signature == signature(aggregate_window, aggregate_recipe=recipe, layout=layout):
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


def _event(event: dict[str, Any], actor: int, layout: StructuredLayout = LEGACY_LAYOUT) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]]:
    numeric, present = _numbers(event, EVENT_NUMERIC_FIELDS)
    categories = (category(event["type"], layout.event_types), category(event.get("definition_id"), layout.event_categories[1]),
                  category(event.get("status"), STATUSES), category(_result(event.get("result"), actor), RESULTS),
                  *(category(_relative(event.get(key), actor), RELATIVE) for key in EVENT_ACTORS))
    return numeric, present, categories


def tokenize_frame(frame: DecisionFrame, bank_view: EventBankView, *, aggregate_window: int = 128,
                   aggregate_recipe: str | None = None, layout: StructuredLayout = LEGACY_LAYOUT) -> FrameTokens:
    require_layout(layout)
    if frame.layout is not layout or bank_view.layout is not layout:
        raise ContractError("raw frame tokenizer layout mismatch")
    aggregate_recipe = layout.aggregate_recipe if aggregate_recipe is None else aggregate_recipe
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
    if layout is LEGACY_LAYOUT:
        cards = [(card["definition_id"], "hand_artifact", card["instance_id"], 0, None) for card in own["hand"]]
        cards += [(name, "learned_miracle", name, order + 1, None) for order, name in enumerate(own["learned_miracles"])]
    else:
        cards = [(card["definition_id"], card["kind"], card["card_ref"], index + 1, card.get("state"))
                 for index, card in enumerate(own["cards"])]
    for name, source, ref, order, miracle_state in cards:
        definition = payload["catalog"][name]
        numbers = {key: definition[key] for key in layout.card_numeric_fields if key in definition}
        if layout is LEGACY_LAYOUT:
            numbers.update(usable=ref in options["usable_hand_elements" if source == "hand_artifact" else "usable_learned_miracles"], learned_order=order)
            for ability in ("exchange", "sell", "buy"):
                numbers[ability] = source == "hand_artifact" and ref in options[ability + "_artifacts"]
        else:
            numbers.update(usable=ref in {row["card_ref"] for row in options["usable_cards"]}, possession_position=order)
            for ability in ("exchange", "sell", "buy", "phenomenon"):
                numbers[ability] = ref in {row["card_ref"] for row in options[ability + "_cards"]}
            for role in layouts.USE_ROLES:
                numbers["role." + role] = role in definition["classification"]["use_roles"]
        numeric, present = _numbers(numbers, layout.card_numeric_fields)
        card_numbers.append(numeric)
        card_presence.append(present)
        categories: tuple[int, ...] = (category(name, vocab.DEFINITIONS), category(definition.get("ability"), vocab.ABILITIES),
                          category(definition["kind"], vocab.KINDS), category(definition.get("attribute"), vocab.ATTRIBUTES),
                          category(definition.get("curse", {}).get("id"), vocab.DISEASES),
                          category(definition.get("guardian", {}).get("id"), vocab.GUARDIANS))
        if layout is LEGACY_LAYOUT:
            categories += (category(source, SOURCES),)
        else:
            classification = definition["classification"]
            membership = "artifact" if source == "artifact" else "miracle_" + str(miracle_state)
            categories += (category(source, layouts.INSTANCE_KINDS), category(miracle_state, layouts.MIRACLE_STATES),
                category(classification["definition_class"], layouts.DEFINITION_CLASSES),
                category(classification.get("miracle_subclass"), layouts.MIRACLE_SUBCLASSES), category(membership, layouts.MEMBERSHIPS))
        card_cats.append(categories)
    count = len(cards)
    while len(card_numbers) < 18:
        card_numbers.append((0,) * len(layout.card_numeric_fields))
        card_presence.append((False,) * len(layout.card_numeric_fields))
        card_cats.append((PAD,) * len(layout.card_categories))
    opponent = tuple(category(name, vocab.DEFINITIONS) for name in obs["opponent"]["learned_miracles"])
    opponent += (PAD,) * (18 - len(opponent))
    corrected = layout is layouts.CORRECTED_CARD_INSTANCE_LAYOUT
    interval = [_event(event, frame.actor, layout) for event in view.events(frame.interval_start)
                if not (corrected and is_concealed_draw(event))]
    history = [0] * (len(layout.event_types) + 2 * len(EVENT_ACTORS) + 2 * len(EVENT_NUMERIC_FIELDS))
    if corrected:
        newest: list[Json] = []
        truncated = False
        for raw_event in reversed(view._events):
            event = json.loads(raw_event)
            if is_concealed_draw(event):
                continue
            if len(newest) == aggregate_window:
                truncated = True
                break
            newest.append(event)
        events = tuple(reversed(newest))
        # exact_history_tail only needs the bounded count and whether one more exists.
        prefix_count = len(events) + truncated
    else:
        events = view.events(max(0, view.boundary.end - aggregate_window))
        prefix_count = view.boundary.end
    for event in events:
        event_numbers, event_presence, categories = _event(event, frame.actor, layout)
        history[categories[0] - 2] += 1
        for role_index, value in enumerate(categories[4:]):
            if value >= 2:
                history[len(layout.event_types) + 2 * role_index + value - 2] += 1
        offset = len(layout.event_types) + 2 * len(EVENT_ACTORS)
        for index, (number, event_exists) in enumerate(zip(event_numbers, event_presence, strict=True)):
            history[offset + index * 2] += number
            history[offset + index * 2 + 1] += event_exists
    if aggregate_recipe == layout.aggregate_recipe:
        history.extend(exact_history_tail(events, frame.actor, prefix_count=prefix_count,
            window=aggregate_window, definition_ids=layout.event_categories[1]))
    elif aggregate_recipe != HISTORICAL_AGGREGATE:
        raise ContractError('unsupported structured aggregate recipe')
    return FrameTokens(frame.frame_hash, signature(aggregate_window, aggregate_recipe=aggregate_recipe, layout=layout), aggregate_window, frame.actor, frame.episode, frame.revision,
                       frame.interval_start, frame.boundary.end, state, state_present, cats, category_present, tuple(card_numbers),
                       tuple(card_presence), tuple(card_cats), tuple(i < count for i in range(18)), opponent,
                       tuple(item[0] for item in interval), tuple(item[1] for item in interval),
                       tuple(item[2] for item in interval), tuple(history))


def tokenize_table(table: CandidateTable, frame: DecisionFrame, max_positions: int = 128, *,
                   aggregate_window: int = 128, aggregate_recipe: str | None = None,
                   layout: StructuredLayout = LEGACY_LAYOUT) -> CandidateTokens:
    require_layout(layout)
    if table.layout is not layout or frame.layout is not layout:
        raise ContractError("raw candidate tokenizer layout mismatch")
    integer(max_positions, 1, 128)
    table.validate(frame)
    return _encode_candidate_tokens(table, frame.frame_hash, max_positions,
        aggregate_window=aggregate_window, aggregate_recipe=aggregate_recipe, layout=layout)


def _encode_candidate_tokens(table: CandidateTable, frame_hash: str, max_positions: int, *,
                             aggregate_window: int, aggregate_recipe: str | None,
                             layout: StructuredLayout) -> CandidateTokens:
    """Canonical encoding body; callers establish full original pair validation."""
    if any(b - a > max_positions for a, b in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
        raise ContractError("candidate token position capacity exceeded")
    return CandidateTokens(table.table_hash, frame_hash, signature(aggregate_window, aggregate_recipe=aggregate_recipe, layout=layout),
                           tuple(category(kind, layout.action_kinds) for kind in table.kinds),
                           tuple(category(target, layout.targets) for target in table.targets), table.numerics,
                           table.presence, table.offsets, table.slots)


def validate_frame_tokens(tokens: FrameTokens, *, layout: StructuredLayout = LEGACY_LAYOUT) -> None:
    """Validate every shape/category/mask before array allocation."""
    if type(tokens) is not FrameTokens:
        raise ContractError("structured frame token recipe mismatch")
    require_layout(layout)
    recipe = recorded_aggregate_recipe(tokens.recipe_signature, tokens.aggregate_window, layout=layout)

    for name in ("state_numerics", "state_present", "state_categories", "state_category_present",
                 "card_numerics", "card_present", "card_categories", "card_mask", "opponent_learned",
                 "interval_numerics", "interval_present", "interval_categories", "history"):
        if type(getattr(tokens, name)) is not tuple:
            raise ContractError("raw token buffers must be immutable tuples")
    integer(tokens.actor, 1, 2)
    integer(tokens.revision)
    integer(tokens.event_start, 0, tokens.event_end)
    integer(tokens.event_end, tokens.event_start, min(2**31 - 1, tokens.event_start + 65536))
    interval = tokens.event_end - tokens.event_start
    rows = len(tokens.interval_numerics)
    if (not isinstance(tokens.episode, str) or not tokens.episode or rows > interval
            or (layout is not layouts.CORRECTED_CARD_INSTANCE_LAYOUT and rows != interval)):
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
        numeric(numbers, present, len(layout.card_numeric_fields))
        categories(cats, layout.card_categories)
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
        categories(cats, layout.event_categories)
        if any(value == PAD for value in cats):
            raise ContractError("event category cannot be padding")
    if len(tokens.history) != (layout.history_width if recipe == layout.aggregate_recipe else LEGACY_HISTORY_WIDTH):
        raise ContractError("history token shape mismatch")
    for value in tokens.history:
        integer(value, -(2**63), 2**63 - 1)


def validate_candidate_tokens(tokens: CandidateTokens, frame: FrameTokens, max_positions: int, *,
                              layout: StructuredLayout = LEGACY_LAYOUT) -> None:
    require_layout(layout)
    recorded_aggregate_recipe(frame.recipe_signature, frame.aggregate_window, layout=layout)
    integer(max_positions, 1, 128)
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
        integer(tokens.kinds[index], 2, len(layout.action_kinds) + 1)
        integer(tokens.targets[index], 2, len(layout.targets) + 1)
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
