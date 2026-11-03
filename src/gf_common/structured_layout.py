"""Closed immutable raw-token layouts, below records and numerical consumers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from gf_common import structured_vocab as vocab
from gf_common.action_contract import ActionContract, engineering_contract, parse_action_contract
from gf_common.card_instance import (
    CardInstanceContract,
    card_instance_contract,
    corrected_card_instance_contract,
    parse_card_instance_contract,
)
from gf_common.domain import CATALOG_HASH, POLICY_INPUT_VERSION, SIMULATOR_REVISION, ContractError
from gf_common.structured_history import ACTIVE_AGGREGATE, HISTORY_WIDTH

INPUT_RECIPE = "participant-structured-v1"
GENERATOR_V2 = "public-atomic-diverse-v4"
ACTION_KINDS = (
    "pray",
    "play_sequence",
    "respond",
    "respond_sequence",
    "discard",
    "sacrifice",
    "exchange",
    "sell",
    "buy",
    "buy_decision",
)
TARGETS = ("omitted", "null", "self", "opponent")
PAD, MISSING = 0, 1
EVENT_TYPES = (
    "turn_started",
    "action_accepted",
    "artifact_used",
    "artifact_drawn",
    "devil_appeared",
    "artifact_discarded",
    "guardian_set",
    "guardian_left",
    "guardian_acted",
    "miracle_used",
    "attack_declared",
    "attack_missed",
    "damage_applied",
    "healed",
    "resource_changed",
    "buy_offered",
    "sale_offered",
    "artifact_transferred",
    "sun_amulet_activated",
    "status_changed",
    "phenomenon_invoked",
    "player_ascended",
    "player_defeated",
    "game_ended",
)
# Both spellings are public identifiers: Observation.status uses dark_cloud,
# public-ja-v2 curse.id uses darkcloud. These are separate exact categories;
# this supported vocabulary does not claim an unpublished exhaustive Event enum.
STATUSES = ("cold", "fever", "hell", "heaven", "fog", "flash", "dream", "dark_cloud", "darkcloud")
PHASES = ("awaiting_action", "awaiting_response", "resolving", "terminal")
CHOICES = ("turn_action", "defense", "buy_accept_reject", "additional_choice")
RELATIVE = ("self", "opponent")
RESULTS = ("self_win", "opponent_win", "draw")
SOURCES = ("hand_artifact", "learned_miracle")
STATE_NUMERIC_FIELDS = (
    "turn",
    "turns_until_end_of_world",
    "end_times",
    "actionable",
    "own.hp",
    "own.mp",
    "own.money",
    "own.ascended",
    "own.fog",
    "own.flash",
    "own.dream",
    "own.dark_cloud",
    "opponent.hp",
    "opponent.mp",
    "opponent.money",
    "opponent.ascended",
    "opponent.fog",
    "opponent.flash",
    "opponent.dream",
    "opponent.dark_cloud",
    "can_pray",
    "can_discard",
    "can_sacrifice",
    "can_allow",
    "target_self",
    "target_opponent",
)
STATE_CATEGORIES = (
    PHASES,
    CHOICES,
    RELATIVE,
    RELATIVE,
    RELATIVE,
    RESULTS,
    vocab.DISEASES,
    vocab.GUARDIANS,
    vocab.DISEASES,
    vocab.GUARDIANS,
)
CARD_NUMERIC_FIELDS = (
    "attack",
    "defense",
    "ability_value",
    "price",
    "mp_cost",
    "hit_rate",
    "additional_attack",
    "usable",
    "learned_order",
    "exchange",
    "sell",
    "buy",
)
CARD_CATEGORIES = (
    vocab.DEFINITIONS,
    vocab.ABILITIES,
    vocab.KINDS,
    vocab.ATTRIBUTES,
    vocab.DISEASES,
    vocab.GUARDIANS,
    SOURCES,
)
EVENT_ACTORS = ("player", "attacker", "target", "active_player", "buyer", "seller", "from", "to")
EVENT_NUMERIC_FIELDS = (
    "turn",
    "price",
    "mp_spent",
    "attack",
    "amount",
    "remaining_hp",
    "hp",
    "hp_delta",
    "mp_delta",
    "money_delta",
    "active",
)
EVENT_CATEGORIES = (
    EVENT_TYPES,
    vocab.DEFINITIONS,
    STATUSES,
    RESULTS,
    *(RELATIVE for _ in EVENT_ACTORS),
)
STATE_WIDTH = (
    2 * len(STATE_NUMERIC_FIELDS)
    + len(STATE_CATEGORIES)
    + sum(len(c) + 2 for c in STATE_CATEGORIES)
    + 18 * (len(vocab.DEFINITIONS) + 2)
)
CARD_DIRECT_WIDTH = 2 * len(CARD_NUMERIC_FIELDS) + sum(len(c) + 2 for c in CARD_CATEGORIES)
EVENT_WIDTH = 2 * len(EVENT_NUMERIC_FIELDS) + sum(len(c) + 2 for c in EVENT_CATEGORIES)
CANDIDATE_WIDTH = len(ACTION_KINDS) + 2 + len(TARGETS) + 2 + 8

CARD_INSTANCE_AGGREGATE = "public-exact-card-refs-v3"
CARD_INSTANCE_ACTION_KINDS = (
    "pray",
    "play_sequence",
    "respond",
    "discard",
    "sacrifice",
    "exchange",
    "sell",
    "buy",
    "buy_decision",
    "invoke_phenomenon",
)
USE_ROLES = (
    "additional_attack",
    "armor",
    "cost_cut",
    "miracle:additional_weapon",
    "miracle:attack_support",
    "miracle:defense",
    "miracle:status",
    "miracle:utility",
    "miracle:weapon",
    "numeric_defense",
    "trade",
    "utility",
    "weapon",
)
INSTANCE_KINDS = ("artifact", "miracle")
MIRACLE_STATES = ("unused", "learned")
DEFINITION_CLASSES = ("owned_card", "automatic", "operation")
MIRACLE_SUBCLASSES = (
    "weapon",
    "additional_weapon",
    "attack_support",
    "defense",
    "status",
    "utility",
)
MEMBERSHIPS = ("artifact", "miracle_unused", "miracle_learned")


@dataclass(frozen=True)
class StructuredLayout:
    input_recipe: str
    policy_input_version: str
    generator: str
    simulator_revision: str
    catalog_sha256: str
    action_contract: ActionContract | CardInstanceContract
    state_numeric_fields: tuple[str, ...]
    state_categories: tuple[tuple[str, ...], ...]
    card_numeric_fields: tuple[str, ...]
    card_categories: tuple[tuple[str, ...], ...]
    event_types: tuple[str, ...]
    event_numeric_fields: tuple[str, ...]
    event_categories: tuple[tuple[str, ...], ...]
    action_kinds: tuple[str, ...]
    targets: tuple[str, ...]
    aggregate_recipe: str
    history_width: int
    order_field: str
    membership_index: int
    format_versions: tuple[tuple[str, int], ...]

    @property
    def state_width(self) -> int:
        return (
            2 * len(self.state_numeric_fields)
            + len(self.state_categories)
            + sum(len(c) + 2 for c in self.state_categories)
            + 18 * (len(vocab.DEFINITIONS) + 2)
        )

    @property
    def card_direct_width(self) -> int:
        return 2 * len(self.card_numeric_fields) + sum(len(c) + 2 for c in self.card_categories)

    @property
    def event_width(self) -> int:
        return 2 * len(self.event_numeric_fields) + sum(len(c) + 2 for c in self.event_categories)

    @property
    def candidate_width(self) -> int:
        return len(self.action_kinds) + 2 + len(self.targets) + 2 + 8


LEGACY_LAYOUT = StructuredLayout(
    INPUT_RECIPE,
    POLICY_INPUT_VERSION,
    GENERATOR_V2,
    SIMULATOR_REVISION,
    CATALOG_HASH,
    engineering_contract(),
    STATE_NUMERIC_FIELDS,
    STATE_CATEGORIES,
    CARD_NUMERIC_FIELDS,
    CARD_CATEGORIES,
    EVENT_TYPES,
    EVENT_NUMERIC_FIELDS,
    EVENT_CATEGORIES,
    ACTION_KINDS,
    TARGETS,
    ACTIVE_AGGREGATE,
    HISTORY_WIDTH,
    "learned_order",
    6,
    (
        ("actions", 2),
        ("model", 2),
        ("protocol", 2),
        ("columns", 1),
        ("journal", 5),
        ("collection", 4),
        ("lineage", 2),
        ("numeric", 3),
        ("learner_wire", 3),
        ("artifact", 3),
        ("dataset", 2),
    ),
)
_CARD_CONTRACT = card_instance_contract()
_CARD_NUMERICS = (
    *CARD_NUMERIC_FIELDS[:8],
    "possession_position",
    *CARD_NUMERIC_FIELDS[9:],
    "phenomenon",
    *("role." + role for role in USE_ROLES),
)
_CARD_CATEGORIES = (
    *CARD_CATEGORIES[:6],
    INSTANCE_KINDS,
    MIRACLE_STATES,
    DEFINITION_CLASSES,
    MIRACLE_SUBCLASSES,
    MEMBERSHIPS,
)
_EVENT_TYPES = (*EVENT_TYPES, "safe")
CARD_INSTANCE_LAYOUT = StructuredLayout(
    _CARD_CONTRACT.input_recipe,
    _CARD_CONTRACT.policy_input_version,
    "public-card-instance-diverse-v5",
    _CARD_CONTRACT.simulator_revision,
    _CARD_CONTRACT.catalog_sha256,
    _CARD_CONTRACT,
    STATE_NUMERIC_FIELDS,
    STATE_CATEGORIES,
    _CARD_NUMERICS,
    _CARD_CATEGORIES,
    _EVENT_TYPES,
    EVENT_NUMERIC_FIELDS,
    (_EVENT_TYPES, *EVENT_CATEGORIES[1:]),
    CARD_INSTANCE_ACTION_KINDS,
    TARGETS,
    CARD_INSTANCE_AGGREGATE,
    HISTORY_WIDTH + 1,
    "possession_position",
    10,
    (
        ("actions", 3),
        ("model", 3),
        ("protocol", 3),
        ("columns", 2),
        ("journal", 6),
        ("collection", 5),
        ("lineage", 3),
        ("numeric", 4),
        ("learner_wire", 4),
        ("artifact", 4),
        ("dataset", 3),
    ),
)


_CORRECTED_CONTRACT = corrected_card_instance_contract()
CORRECTED_CARD_INSTANCE_LAYOUT = replace(CARD_INSTANCE_LAYOUT,
    action_contract=_CORRECTED_CONTRACT, simulator_revision=_CORRECTED_CONTRACT.simulator_revision,
    event_categories=(_EVENT_TYPES, vocab.DEFINITIONS + vocab.GUARDIANS, *EVENT_CATEGORIES[2:]),
    history_width=CARD_INSTANCE_LAYOUT.history_width + len(vocab.GUARDIANS))


def require_layout(layout: StructuredLayout) -> StructuredLayout:
    if (layout is not LEGACY_LAYOUT and layout is not CARD_INSTANCE_LAYOUT
            and layout is not CORRECTED_CARD_INSTANCE_LAYOUT):
        raise ContractError("explicit known immutable structured layout required")
    return layout


def is_card_instance_layout(layout: StructuredLayout) -> bool:
    return require_layout(layout) is not LEGACY_LAYOUT


def layout_for_contract(contract: ActionContract | CardInstanceContract) -> StructuredLayout:
    from dataclasses import asdict

    if type(contract) is ActionContract:
        parse_action_contract(asdict(contract))
        return LEGACY_LAYOUT
    if type(contract) is CardInstanceContract:
        parse_card_instance_contract(asdict(contract))
        return (CARD_INSTANCE_LAYOUT if contract == CARD_INSTANCE_LAYOUT.action_contract
                else CORRECTED_CARD_INSTANCE_LAYOUT)
    raise ContractError("closed structured action contract required")


def layout_for_model_identity(
    version: str, input_recipe: str, contract: ActionContract | CardInstanceContract
) -> StructuredLayout:
    layout = layout_for_contract(contract)
    expected = "2" if layout is LEGACY_LAYOUT else "3"
    if type(version) is not str or version != expected or input_recipe != layout.input_recipe:
        raise ContractError("mixed structured model/input/contract identity")
    return layout
