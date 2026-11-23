"""Bounded public v4 proposals, never an engine legality oracle.

Baseline public inputs remain pinned; ADR0023 §17's owner profile authorizes
the engineering-only proposal subset. No runtime conformance is asserted.
Only acting-participant perceived definitions and same-revision atomic hints are
used; every pending coverage item remains explicit in the report.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.action_contract import PENDING_COVERAGE, ActionContract, validate_game_references
from gf_common.card_instance import (
    GENERATOR_VERSION,
    ActionConfigV3,
    CardInstanceContract,
    is_corrected_contract,
)
from gf_common.card_instance_rules import (
    UTILITY_IDS,
    CardInstanceRules,
    PublicCard,
    SequenceAnalysis,
    analyze_sequence,
    card_instance_rules,
    defense_context,
    defense_kind,
    rule_signature,
)
from gf_common.domain import ContractError, EmptyCandidates, Json, digest, integer
from gf_common.structured import (
    GENERATOR_V2,
    ActionConfigV2,
    CandidateTable,
    DecisionFrame,
    EventBankView,
)
from gf_common.structured_layout import LEGACY_LAYOUT, layout_for_contract

SATURATION = 2**63 - 1
CHANNELS = ("attribute_effect", "attack_breadth", "resources", "refresh", "remaining_hand", "length", "exploration")
OMISSIONS = PENDING_COVERAGE
RINGS = frozenset(("counterAtk", "counterCurse", "counterBoost2xMP", "counter2xAtk", "counterTakeCP"))
EXCLUSIVE = frozenset(("reflectAnything", "reflectWeapon", "reflectMiracle", "bounceWeapon", "bounceMiracle"))
STATUS_MIRACLES = frozenset(("wind", "heaven-wind", "fog", "dream", "dark-cloud"))
UTILITY_BOTH = frozenset(("smile-dew", "heart-dew", "romance-water", "galaxy-geyser", "smile-flower", "heart-flower",
                          "romance-fragrance", "spring", "treasure", "tone", "song", "smile-shell", "heart-shell"))
UTILITY_SELF = frozenset(("thump-thump-tear", "heaven-herb"))


def _sat_add(*values: int) -> int:
    total = 0
    for value in values:
        if value >= SATURATION - total:
            return SATURATION
        total += value
    return total


def _sat_mul(*values: int) -> int:
    if 0 in values:
        return 0
    total = 1
    for value in values:
        if total > SATURATION // value:
            return SATURATION
        total *= value
    return total


def _sha(value: str) -> None:
    if type(value) is not str or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ContractError("invalid report digest")


@dataclass(frozen=True)
class NumericCosts:
    version: str = "proposal-cost-v1"
    context_width: int = 0
    card_width: int = 0
    actor_input_width: int = 0
    actor_hidden_width: int = 0
    position_capacity: int = 128
    candidate_chunk: int = 256
    tail_operations_per_candidate: int = 0
    tail_working_bytes_per_candidate: int = 0
    tail_tensor_bytes_per_candidate: int = 0
    fixed_bytes: int = 0
    max_preparation_bytes: int = 12 * 1024**3
    max_scoring_operations: int = SATURATION
    max_storage_bytes: int = 64 * 1024**2
    max_tensor_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        if self.version != "proposal-cost-v1":
            raise ContractError("unsupported numeric cost version")
        dimensions = (self.context_width, self.card_width, self.actor_input_width, self.actor_hidden_width)
        for value in dimensions:
            integer(value, 0, 2**20)
        if any(dimensions) and not all(dimensions):
            raise ContractError("baseline dimensions must all be zero; neural dimensions positive")
        tail = (self.tail_operations_per_candidate, self.tail_working_bytes_per_candidate,
                self.tail_tensor_bytes_per_candidate)
        for value in tail:
            integer(value, 0, SATURATION)
        if self.tail_working_bytes_per_candidate < self.tail_tensor_bytes_per_candidate:
            raise ContractError("tail working bytes must cover largest tail tensor")
        if self.actor_hidden_width > 0 and not all(tail):
            raise ContractError("neural scoring requires complete positive tail coefficients")
        integer(self.position_capacity, 1, 128)
        integer(self.candidate_chunk, 1, 65536)
        integer(self.fixed_bytes, 0, 12 * 1024**3)
        integer(self.max_preparation_bytes, 1, 12 * 1024**3)
        integer(self.max_scoring_operations, 1, SATURATION)
        integer(self.max_storage_bytes, 1, 12 * 1024**3)
        integer(self.max_tensor_bytes, 1, 64 * 1024**2)


@dataclass(frozen=True)
class CostEstimate:
    preparation_bytes: int
    scoring_operations: int
    storage_bytes: int
    largest_tensor_bytes: int

    def fits(self, costs: NumericCosts) -> bool:
        return (self.preparation_bytes <= costs.max_preparation_bytes
                and self.scoring_operations <= costs.max_scoring_operations
                and self.storage_bytes <= costs.max_storage_bytes
                and self.largest_tensor_bytes <= costs.max_tensor_bytes)


def estimate_costs(costs: NumericCosts, candidates: int, elements: int, positions: int,
                   temporary: int = 0, temporary_elements: int = 0,
                   retained_prefix_elements: int = 0) -> CostEstimate:
    """Saturating compact storage + factorized scoring upper envelope.

    The worker supplies fixed encoder/model/cache bytes. These local increments
    cover owned proposals/prefixes, compact columns/refs, float packing, position
    projections, chunk gather/replay and retained global logits/probability data.
    Tail coefficients describe every configured nonlinear head stage, including
    later hidden widths; fixed_bytes cannot hide candidate-dependent head work.
    No full dense candidate-by-actor-input matrix is assumed or allocated.
    """
    for value in (candidates, elements, temporary, temporary_elements, retained_prefix_elements):
        integer(value, 0, SATURATION)
    integer(positions, 1, 128)
    if positions > costs.position_capacity:
        raise ContractError("generator exceeds numeric position capacity")
    if costs.actor_input_width:
        categorical = costs.actor_input_width - costs.context_width - costs.position_capacity * costs.card_width
        if categorical <= 0:
            raise ContractError("numeric actor dimensions do not cover configured positions")
    else:
        categorical = 0
    storage = _sat_add(_sat_mul(candidates, 64), _sat_mul(elements, 8), 8)
    compact_rows = _sat_mul(candidates, 64)
    compact_refs = _sat_mul(elements, 8)
    chunk = min(candidates, costs.candidate_chunk)
    projection = _sat_mul(costs.position_capacity, 19, costs.actor_hidden_width, 4)
    unpadded_projection = _sat_mul(costs.position_capacity, 18, costs.actor_hidden_width, 4)
    projection_padding = _sat_mul(costs.position_capacity, costs.actor_hidden_width, 4)
    gather = _sat_mul(chunk, costs.position_capacity, costs.actor_hidden_width, 4)
    columns = _sat_mul(candidates, categorical, 4)
    logits = _sat_mul(candidates, 20)
    chunk_head = _sat_mul(chunk, costs.actor_hidden_width, 32)
    tail_working = _sat_mul(chunk, costs.tail_working_bytes_per_candidate)
    tail_tensor = _sat_mul(chunk, costs.tail_tensor_bytes_per_candidate)
    encoder_pack = _sat_mul(_sat_add(costs.context_width, 18 * costs.card_width), 4)
    temporary_bytes = _sat_add(_sat_mul(temporary, 2048), _sat_mul(temporary_elements, 8),
                               _sat_mul(retained_prefix_elements, 32))
    preparation = _sat_add(costs.fixed_bytes, temporary_bytes, storage, encoder_pack,
                           projection, unpadded_projection, projection_padding, gather, columns, logits, chunk_head, tail_working)
    operations = _sat_add(_sat_mul(2, costs.position_capacity, 18, costs.card_width, costs.actor_hidden_width),
                          _sat_mul(2, candidates, categorical, costs.actor_hidden_width),
                          # Production gathers/sums every configured position,
                          # including PAD, even for one-reference candidates.
                          _sat_mul(candidates, costs.position_capacity, costs.actor_hidden_width),
                          _sat_mul(2, costs.context_width, costs.actor_hidden_width),
                          _sat_mul(candidates, costs.tail_operations_per_candidate))
    largest = max(compact_rows, compact_refs, projection, gather, columns, logits, chunk_head, encoder_pack, tail_tensor)
    return CostEstimate(preparation, operations, storage, largest)


@dataclass(frozen=True)
class PrecountReport:
    mode: str
    saturated: bool
    count_states: int
    complete: int
    prefixes: int
    attempts: int
    excluded: int
    targets: int
    numeric_variants: int
    elements: int
    preparation_bytes: int
    scoring_operations: int
    storage_bytes: int
    full_enumeration_admitted: bool

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "upper_bound", "unknown"):
            raise ContractError("invalid pre-count mode")
        for key, value in asdict(self).items():
            if key in ("saturated", "full_enumeration_admitted"):
                if type(value) is not bool:
                    raise ContractError("pre-count boolean required")
            elif key != "mode":
                integer(value, 0, SATURATION)
        if self.full_enumeration_admitted and (self.mode != "exact" or self.saturated):
            raise ContractError("only exact unsaturated pre-count admits full enumeration")


def _rng_state(value: Any) -> None:
    if type(value) is not tuple or len(value) != 3 or value[0] != 3 or type(value[1]) is not tuple or len(value[1]) != 625:
        raise ContractError("closed Python MT19937 search RNG state required")
    for number in value[1][:-1]:
        integer(number, 0, 2**32 - 1)
    integer(value[1][-1], 0, 624)
    if value[2] is not None and (type(value[2]) is not float or not math.isfinite(value[2])):
        raise ContractError("invalid Gaussian RNG cache")


@dataclass(frozen=True)
class _SearchReportFields:
    action_contract: ActionContract | CardInstanceContract = field(kw_only=True)
    generator_version: str
    config_hash: str
    numeric_costs_hash: str
    table_hash: str
    numeric_costs: NumericCosts
    precount: PrecountReport
    attempts: int
    prunes: int
    discovered: int
    mandatory: int
    selected: int
    selected_elements: int
    channel_counts: tuple[int, ...]
    traversal_complete: bool
    presentation_complete: bool
    cutoff_reasons: tuple[str, ...]
    omissions: tuple[str, ...]
    rng_before: tuple[Any, ...]
    rng_after: tuple[Any, ...]
    count_ms: float
    search_ms: float
    selection_ms: float



def _validate_report(report: _SearchReportFields, omissions: tuple[str, ...]) -> None:
    for value in (report.config_hash, report.numeric_costs_hash, report.table_hash):
        _sha(value)
    if digest(asdict(report.numeric_costs)) != report.numeric_costs_hash:
        raise ContractError("report numeric cost binding mismatch")
    for amount in (report.attempts, report.prunes, report.discovered, report.mandatory, report.selected, report.selected_elements):
        integer(amount, 0, SATURATION)
    if (report.prunes > report.attempts or not (0 <= report.mandatory <= report.selected <= report.discovered and report.selected >= 1)
            or type(report.channel_counts) is not tuple or len(report.channel_counts) != 7):
        raise ContractError("search report counts inconsistent")
    for channel_count in report.channel_counts:
        integer(channel_count, 0, report.selected)
    if sum(report.channel_counts) != report.selected - report.mandatory:
        raise ContractError("channel assignments do not cover nonmandatory selections")
    if type(report.traversal_complete) is not bool or type(report.presentation_complete) is not bool:
        raise ContractError("completion boolean required")
    if report.presentation_complete != (report.traversal_complete and report.selected == report.discovered):
        raise ContractError("presentation/traversal completeness mismatch")
    allowed = {"count_time", "count_states", "search_time", "max_expansions", "max_temporary_candidates",
               "max_candidates", "numeric_costs", "max_sequence_length"}
    if (type(report.cutoff_reasons) is not tuple or len(set(report.cutoff_reasons)) != len(report.cutoff_reasons)
            or set(report.cutoff_reasons) - allowed or report.omissions != omissions):
        raise ContractError("unknown search cutoff/omission")
    for timing in (report.count_ms, report.search_ms, report.selection_ms):
        if type(timing) not in (int, float) or not math.isfinite(timing) or timing < 0:
            raise ContractError("invalid search stage timing")
    _rng_state(report.rng_before)
    _rng_state(report.rng_after)


@dataclass(frozen=True)
class SearchReport(_SearchReportFields):
    action_contract: ActionContract = field(kw_only=True)

    def __post_init__(self) -> None:
        if type(self.action_contract) is not ActionContract:
            raise ContractError("explicit report action contract required")
        if self.generator_version != GENERATOR_V2 or type(self.numeric_costs) is not NumericCosts or type(self.precount) is not PrecountReport:
            raise ContractError("invalid search report recipe")
        _validate_report(self, OMISSIONS)


OMISSIONS_V3 = ("unlisted direct-use utility definitions",)


@dataclass(frozen=True)
class SearchReportV3(_SearchReportFields):
    action_contract: CardInstanceContract = field(kw_only=True)
    rule_signature: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if (type(self.action_contract) is not CardInstanceContract or self.generator_version != GENERATOR_VERSION
                or type(self.numeric_costs) is not NumericCosts or type(self.precount) is not PrecountReport):
            raise ContractError("invalid card-instance search report recipe")
        if self.rule_signature != rule_signature(self.action_contract, card_instance_rules()):
            raise ContractError("search report rule signature mismatch")
        _validate_report(self, OMISSIONS_V3)


@dataclass(frozen=True)
class GeneratedCandidates:
    table: CandidateTable
    report: SearchReport

    def __post_init__(self) -> None:
        _validate_generated(self.table, self.report, SearchReport)


@dataclass(frozen=True)
class GeneratedCandidatesV3:
    table: CandidateTable
    report: SearchReportV3

    def __post_init__(self) -> None:
        _validate_generated(self.table, self.report, SearchReportV3)


@dataclass(frozen=True)
class GameGeneratedCandidatesV3:
    """Corrected game-use context, never evidence of Source eligibility."""
    table: CandidateTable
    report: SearchReportV3

    def __post_init__(self) -> None:
        if type(self.table) is not CandidateTable or not is_corrected_contract(self.table.action_contract):
            raise ContractError("game candidates require the corrected contract")
        _validate_generated(self.table, self.report, SearchReportV3, purpose="qualified_game")


def _validate_generated(table: CandidateTable, report: _SearchReportFields,
                        report_type: type[_SearchReportFields], *, purpose: str = "component") -> None:
    if (type(table) is not CandidateTable or type(report) is not report_type
            or table.table_hash != report.table_hash
            or table.action_contract != report.action_contract or table.purpose != purpose
            or table.synthetic_capacity
            or len(table.kinds) != report.selected
            or len(table.slots) != report.selected_elements):
        raise ContractError("generated table/report mismatch")
    validate_game_references(table.slots[start:end] for start, end in zip(table.offsets[:-1], table.offsets[1:], strict=True))


@dataclass(frozen=True)
class _Card:
    slot: int
    hand: bool
    definition: Json = field(compare=False, repr=False)

    @property
    def ability(self) -> str | None:
        return self.definition.get("ability")

    @property
    def miracle(self) -> bool:
        return self.definition["kind"] == "miracles"

    @property
    def additional(self) -> bool:
        return bool(self.definition.get("additional_attack", False))

    @property
    def support(self) -> bool:
        return self.miracle and self.definition["definition_id"] in ("aura", "mirage")


@dataclass(frozen=True)
class _Proposal:
    kind: str
    target: str = "omitted"
    refs: tuple[int, ...] = ()
    numerics: tuple[int, ...] = (0, 0, 0, 0)
    presence: tuple[bool, ...] = (False, False, False, False)


class _Space:
    def __init__(self, frame: DecisionFrame, config: ActionConfigV2) -> None:
        self.frame, self.config = frame, config
        self.obs, self.raw = frame.observation, frame.raw
        self.options = self.raw["action_options"]["action_options"]
        own = self.obs["own"]
        self.cards = [_Card(index, True, self.raw["catalog"][card["definition_id"]])
                      for index, card in enumerate(own["hand"])]
        self.cards += [_Card(len(own["hand"]) + index, False, self.raw["catalog"][name])
                       for index, name in enumerate(own["learned_miracles"])]
        self.by_slot = {card.slot: card for card in self.cards}
        self.hand = [card for card in self.cards if card.hand]
        usable = self.options["usable_hand_elements"]
        learned = self.options["usable_learned_miracles"]
        hand_ids = {card["instance_id"]: index for index, card in enumerate(own["hand"])}
        self.usable = {hand_ids[item] for item in usable}
        self.usable.update(len(own["hand"]) + index for index, name in enumerate(own["learned_miracles"]) if name in learned)
        self.targets = tuple(name for name in config.targets if (frame.actor if name == "self" else 3 - frame.actor) in self.options["valid_targets"])
        self.offense = [card.slot for card in self.cards if card.slot in self.usable and
                        (card.support or (card.definition.get("attack", 0) > 0 and
                         (card.definition["kind"] in ("weapons", "miracles") or card.additional))
                         or (card.definition["kind"] == "weapons" and card.ability == "atkBy2xMP"))]
        self.defense = [card.slot for card in self.cards if card.slot in self.usable and self.defense_kind(card) is not None]
        self.discard = [card.slot for card in self.hand if card.definition["kind"] != "weapons"
                        and (self.obs["end_times"] or card.definition["definition_id"] not in ("sun-amulet", "dangerous-mortar"))]
        self.trade = {ability: [hand_ids[item] for item in self.options[ability + "_artifacts"]
                               if self.by_slot[hand_ids[item]].definition.get("ability") == ability]
                      for ability in ("exchange", "sell", "buy")}
        self.total = sum(own[key] for key in ("hp", "mp", "money"))
        self.maximum = min(config.max_sequence_length, len(self.cards), 18)

    def singleton_targets(self, card: _Card) -> tuple[str, ...]:
        """Explicit adopted targets/predicates, never inferred effect recipients."""
        if card.slot not in self.usable or card.definition.get("mp_cost", 0) > self.obs["own"]["mp"]:
            return ()
        name = card.definition["definition_id"]
        if name in STATUS_MIRACLES or name in UTILITY_BOTH:
            return self.targets
        if name in UTILITY_SELF:
            return tuple(target for target in self.targets if target == "self")
        if name in ("guardian-pot", "release"):
            return tuple(target for target in self.targets if "guardian" in self.obs["own" if target == "self" else "opponent"]
                         and self.obs["own" if target == "self" else "opponent"]["guardian"] is None)
        if name == "goddess-s-soap":
            return tuple(target for target in self.targets if len(self.obs["own" if target == "self" else "opponent"]["learned_miracles"]) >= 2)
        if name == "nocturnal-broom" and len(self.hand) - int(card.hand) >= 3:
            return tuple(target for target in self.targets if target == "self")
        return ()

    @staticmethod
    def defense_kind(card: _Card) -> str | None:
        if card.ability in EXCLUSIVE:
            return "exclusive"
        if card.ability in ("blockWeapon", "blockMiracle") and card.definition.get("defense", 0) <= 0:
            return "exclusive"
        if card.definition.get("defense", 0) > 0 or card.ability in RINGS or card.ability == "filterAtkElement":
            return "ordinary"
        return None

    @property
    def families(self) -> tuple[str, ...]:
        choice = self.obs["required_choice"]
        if choice == "defense":
            return ("defense",) if self.defense else ()
        if choice != "turn_action":
            return ()
        result = ["offense"] if self.offense and self.targets else []
        flag = "can_sacrifice" if self.obs["end_times"] else "can_discard"
        if self.discard and self.options[flag]:
            result.append("discard")
        return tuple(result)

    def choices(self, family: str) -> list[int]:
        return getattr(self, family)

    def valid(self, family: str, refs: tuple[int, ...]) -> bool:
        cards = [self.by_slot[slot] for slot in refs]
        # Owner profile: every physical hand/learned reference occurs at most
        # once within an Action. Cross-decision reuse is a separate rule.
        if len(set(refs)) != len(refs) or len(refs) > self.maximum:
            return False
        if family == "discard":
            return tuple(sorted(refs)) == refs
        if family == "defense":
            if self.obs["own"]["status"]["flash"] and len(refs) > 1:
                return False
            if len(refs) > 1 and any(self.defense_kind(card) == "exclusive" for card in cards):
                return False
        else:
            for index, card in enumerate(cards):
                if card.support:
                    if not any(not other.support for other in cards[:index]):
                        return False
                elif not card.additional and index != 0:
                    return False
            if any(card.ability == "attackEveryEnemy" for card in cards) and any(card.ability == "attackTwice" for card in cards):
                return False
        free = family == "offense" and any(card.definition["kind"] == "weapons" and card.ability == "cutCost" for card in cards)
        return free or sum(card.definition.get("mp_cost", 0) for card in cards if card.miracle) <= self.obs["own"]["mp"]

    def completions(self, family: str, refs: tuple[int, ...],
                    attempt: Callable[[], None]) -> Iterator[_Proposal]:
        if family == "discard":
            attempt()
            yield _Proposal("sacrifice" if self.obs["end_times"] else "discard", refs=refs)
            return
        if family == "defense":
            attempt()
            yield _Proposal("respond_sequence", refs=refs)
            return
        if not any(not self.by_slot[slot].support for slot in refs):
            return
        mirage = any(self.by_slot[slot].ability == "attackEveryEnemy" for slot in refs)
        danger = any(self.by_slot[slot].ability == "danger" for slot in refs)
        targets = self.targets if not mirage or danger else tuple(target for target in self.targets if target == "opponent")
        for target in targets:
            attempt()
            yield _Proposal("play_sequence", target, refs)

    @property
    def basic_count(self) -> int:
        choice = self.obs["required_choice"]
        if choice == "buy_accept_reject":
            return 2
        return int((choice == "defense" and self.options["can_allow"])
                   or (choice == "turn_action" and self.options["can_pray"]))

    def basics(self, attempt: Callable[[], None]) -> Iterator[_Proposal]:
        choice = self.obs["required_choice"]
        if choice == "buy_accept_reject":
            for value in (0, 1):
                attempt()
                yield _Proposal("buy_decision", numerics=(0, 0, 0, value), presence=(False, False, False, True))
        elif choice == "defense" and self.options["can_allow"]:
            attempt()
            yield _Proposal("respond")
        elif choice == "turn_action" and self.options["can_pray"]:
            attempt()
            yield _Proposal("pray")

    def fixed(self, attempt: Callable[[], None]) -> Iterator[_Proposal]:
        """Discretionary sell pairs/numeric supplements, budgeted before rows."""
        if self.obs["required_choice"] != "turn_action":
            return
        for slot in self.trade["sell"] if self.config.max_sequence_length >= 2 else ():
            for other in self.hand:
                if other.slot != slot:
                    for target in self.targets:
                        attempt()
                        yield _Proposal("sell", target, (slot, other.slot))
        for slot in self.trade["exchange"]:
            for index, allocation in enumerate(self.allocations()):
                if index == 0:
                    continue  # the concrete carrier is already mandatory.
                attempt()
                yield _Proposal("exchange", refs=(slot,), numerics=(*allocation, 0),
                                presence=(True, True, True, False))

    def allocations(self) -> Iterator[tuple[int, int, int]]:
        return _allocations(self.obs["own"], self.total,
            tuple(card.definition.get("mp_cost", 0) for card in self.cards if card.miracle))

    def generated(self, table: CandidateTable, fields: tuple[Any, ...]) -> GeneratedCandidates:
        return GeneratedCandidates(table, SearchReport(*fields, action_contract=self.config.action_contract))

    def completion_stats(self, family: str, refs: tuple[int, ...]) -> tuple[int, int, int]:
        if family == "offense":
            mirage = any(self.by_slot[ref].ability == "attackEveryEnemy" for ref in refs)
            danger = any(self.by_slot[ref].ability == "danger" for ref in refs)
            count = len(self.targets) if not mirage or danger else int("opponent" in self.targets)
            return count, len(refs) * count, count
        return 1, len(refs), 0

    def upper_counts(self) -> dict[str, int]:
        return _legacy_upper_counts(self)

    def summary(self, proposal: _Proposal) -> _Summary:
        return _summary(self, proposal)

    def count_fixed(self, tick: Callable[[], None], add: Callable[..., None]) -> None:
        basics = self.basic_count
        add(basics, numeric=basics if self.obs["required_choice"] == "buy_accept_reject" else 0, attempts=basics)
        if self.obs["required_choice"] == "turn_action":
            for card in self.cards:
                tick()
                count = len(self.singleton_targets(card))
                add(count, count, count, attempts=count)
            buy = len(self.trade["buy"]) * len(self.targets)
            sell = (len(self.trade["sell"]) * max(0, len(self.hand) - 1) * len(self.targets)
                    if self.config.max_sequence_length >= 2 else 0)
            add(buy + sell, buy + sell * 2, buy + sell, attempts=buy + sell)
            # Count only the bounded numeric axes as integer triples, shared
            # across carriers. No typed Action or _Proposal is constructed here.
            if self.trade["exchange"]:
                allocations: set[tuple[int, int, int]] = set()
                visits = 0
                for allocation in self.allocations():
                    tick()
                    visits += 1
                    allocations.add(allocation)
                count = len(allocations) * len(self.trade["exchange"])
                attempts = visits * len(self.trade["exchange"])
                add(count, count, numeric=count, attempts=attempts, excluded=attempts - count)


    def mandatory_rows(self, attempt: Callable[[], None]) -> Iterator[_Proposal]:
        yield from self.basics(attempt)
        if self.obs["required_choice"] == "turn_action":
            for card in self.cards:
                for target in self.singleton_targets(card):
                    attempt()
                    yield _Proposal("play_sequence", target, (card.slot,))
            for slot in self.trade["buy"]:
                for target in self.targets:
                    attempt()
                    yield _Proposal("buy", target, (slot,))
            for slot in self.trade["exchange"]:
                attempt()
                allocation = next(self.allocations())
                yield _Proposal("exchange", refs=(slot,), numerics=(*allocation, 0),
                                    presence=(True, True, True, False))



class _CardInstanceSpace:
    def __init__(self, frame: DecisionFrame, config: ActionConfigV3, rules: CardInstanceRules,
                 bank_view: EventBankView) -> None:
        if frame.layout is not layout_for_contract(config.action_contract):
            raise ContractError("candidate configuration/frame contract mismatch")
        self.frame, self.config, self.rules = frame, config, rules
        self.requires_powder_weapon = config.action_contract.requires_powder_weapon
        self.rule_signature = rule_signature(config.action_contract, rules)
        self.obs, self.raw = frame.observation, frame.raw
        self.context = defense_context(frame, bank_view)
        self.options = self.raw["action_options"]["action_options"]
        self.cards = tuple(PublicCard(index, card["kind"], card.get("state"), self.raw["catalog"][card["definition_id"]])
                           for index, card in enumerate(self.obs["own"]["cards"]))
        self.by_slot = {card.slot: card for card in self.cards}
        refs = {card["card_ref"]: index for index, card in enumerate(self.obs["own"]["cards"])}
        self.usable = frozenset(refs[row["card_ref"]] for row in self.options["usable_cards"])
        self.targets = tuple(name for name in config.targets if (frame.actor if name == "self" else 3 - frame.actor) in self.options["valid_targets"])
        self.hand = [card for card in self.cards if card.hand]
        self.trade = {name: [refs[row["card_ref"]] for row in self.options[name + "_cards"]]
                      for name in ("exchange", "sell", "buy")}
        self.phenomena = [refs[row["card_ref"]] for row in self.options["phenomenon_cards"]]
        self.offense = [card.slot for card in self.cards if card.slot in self.usable and (
            "weapon" in card.definition["classification"]["use_roles"] or card.additional
            or card.subclass == "weapon" or card.ability == "cutCost")]
        self.utility = [card.slot for card in self.cards if card.slot in self.usable and (
            card.definition["definition_id"] in UTILITY_IDS or card.ability == "cutCost")]
        self.status = [card.slot for card in self.cards if card.slot in self.usable and (
            card.subclass == "status" or card.ability == "cutCost")]
        self.defense = [card.slot for card in self.cards if card.slot in self.usable and (
            defense_kind(card) is not None or card.ability == "cutCost")]
        # ADR0005's protected-artifact rule applies only to normal Discard;
        # the owner-confirmed permission for every miracle remains independent.
        self.discard = [card.slot for card in self.cards if card.miracle or (
            card.definition["kind"] != "weapons" and (self.obs["end_times"]
                or card.definition["definition_id"] not in ("sun-amulet", "dangerous-mortar")))]
        self.total = sum(self.obs["own"][name] for name in ("hp", "mp", "money"))
        self.maximum = min(config.max_sequence_length, len(self.cards), 18)

    @property
    def families(self) -> tuple[str, ...]:
        if self.obs["required_choice"] == "defense":
            return ("defense",) if self.defense else ()
        if self.obs["required_choice"] != "turn_action":
            return ()
        result = [name for name in ("offense", "utility", "status") if self.choices(name) and self.targets]
        if self.discard and self.options["can_sacrifice" if self.obs["end_times"] else "can_discard"]:
            result.append("discard")
        return tuple(result)

    def choices(self, family: str) -> list[int]:
        return getattr(self, family)

    def analysis(self, family: str, refs: tuple[int, ...]) -> SequenceAnalysis:
        return analyze_sequence(self.cards, refs, family=family, mp=self.obs["own"]["mp"],
            flash=self.obs["own"]["status"]["flash"], maximum=self.config.max_sequence_length, usable=self.usable,
            rules=self.rules, context=self.context,
            requires_powder_weapon=self.requires_powder_weapon)

    def valid(self, family: str, refs: tuple[int, ...]) -> bool:
        if family == "discard":
            return len(refs) <= self.maximum and len(set(refs)) == len(refs) and tuple(sorted(refs)) == refs
        return self.analysis(family, refs).viable

    def completed_targets(self, family: str, refs: tuple[int, ...]) -> tuple[str, ...]:
        if family == "discard":
            return ("omitted",)
        facts = self.analysis(family, refs)
        if not facts.complete:
            return ()
        if family == "defense":
            return ("omitted",)
        return tuple(target for target in self.targets if not facts.opponent_only or target == "opponent")

    def completion_stats(self, family: str, refs: tuple[int, ...]) -> tuple[int, int, int]:
        count = len(self.completed_targets(family, refs))
        return count, count * len(refs), count if family in ("offense", "utility", "status") else 0

    def completions(self, family: str, refs: tuple[int, ...], attempt: Callable[[], None]) -> Iterator[_Proposal]:
        kind = ("sacrifice" if self.obs["end_times"] else "discard") if family == "discard" else (
            "respond" if family == "defense" else "play_sequence")
        for target in self.completed_targets(family, refs):
            attempt()
            yield _Proposal(kind, target, refs)

    @property
    def basic_count(self) -> int:
        choice = self.obs["required_choice"]
        if choice == "buy_accept_reject":
            return 2
        return int(choice == "defense" and self.options["can_allow"] or choice == "turn_action" and self.options["can_pray"])

    def mandatory_rows(self, attempt: Callable[[], None]) -> Iterator[_Proposal]:
        choice = self.obs["required_choice"]
        if choice == "buy_accept_reject":
            for accept in (0, 1):
                attempt()
                yield _Proposal("buy_decision", numerics=(0, 0, 0, accept), presence=(False, False, False, True))
        elif choice == "defense" and self.options["can_allow"]:
            attempt()
            yield _Proposal("respond")
        elif choice == "turn_action":
            if self.options["can_pray"]:
                attempt()
                yield _Proposal("pray")
            for slot in self.trade["buy"]:
                for target in self.targets:
                    attempt()
                    yield _Proposal("buy", target, (slot,))
            for slot in self.trade["exchange"]:
                attempt()
                yield _Proposal("exchange", refs=(slot,), numerics=(*next(self.allocations()), 0), presence=(True, True, True, False))
            for slot in self.phenomena:
                attempt()
                yield _Proposal("invoke_phenomenon", refs=(slot,))

    def allocations(self) -> Iterator[tuple[int, int, int]]:
        return _allocations(self.obs["own"], self.total,
            tuple(card.definition.get("mp_cost", 0) for card in self.cards if card.miracle))

    def fixed(self, attempt: Callable[[], None]) -> Iterator[_Proposal]:
        if self.obs["required_choice"] != "turn_action":
            return
        for slot in self.trade["sell"] if self.config.max_sequence_length >= 2 else ():
            for card in self.hand:
                if card.slot != slot:
                    for target in self.targets:
                        attempt()
                        yield _Proposal("sell", target, (slot, card.slot))
        for slot in self.trade["exchange"]:
            for index, allocation in enumerate(self.allocations()):
                if index:
                    attempt()
                    yield _Proposal("exchange", refs=(slot,), numerics=(*allocation, 0), presence=(True, True, True, False))

    def count_fixed(self, tick: Callable[[], None], add: Callable[..., None]) -> None:
        basics = self.basic_count
        add(basics, numeric=basics if self.obs["required_choice"] == "buy_accept_reject" else 0, attempts=basics)
        if self.obs["required_choice"] != "turn_action":
            return
        buy = len(self.trade["buy"]) * len(self.targets)
        sell = sum(sum(card.slot != slot for card in self.hand) for slot in self.trade["sell"]) * len(self.targets) if self.config.max_sequence_length >= 2 else 0
        phenomena = len(self.phenomena)
        add(buy + sell + phenomena, buy + sell * 2 + phenomena, buy + sell, attempts=buy + sell + phenomena)
        if self.trade["exchange"]:
            unique: set[tuple[int, int, int]] = set()
            visits = 0
            for allocation in self.allocations():
                tick()
                visits += 1
                unique.add(allocation)
            count = len(unique) * len(self.trade["exchange"])
            attempts = visits * len(self.trade["exchange"])
            add(count, count, numeric=count, attempts=attempts, excluded=attempts - count)

    def upper_counts(self) -> dict[str, int]:
        totals = dict.fromkeys(("complete", "prefixes", "attempts", "excluded", "targets", "numeric_variants", "elements"), 0)
        if self.obs["required_choice"] == "turn_action":
            buy = len(self.trade["buy"]) * len(self.targets)
            sell = len(self.trade["sell"]) * max(0, len(self.hand) - 1) * len(self.targets) if self.config.max_sequence_length >= 2 else 0
            exchange = len(self.trade["exchange"]) * (3 * 26**2 + 3 + len(self.cards) * 6)
            fixed = self.basic_count + buy + sell + exchange + len(self.phenomena)
            totals.update(complete=fixed, attempts=fixed, targets=buy + sell, numeric_variants=exchange,
                          elements=buy + 2 * sell + exchange + len(self.phenomena))
        else:
            totals.update(complete=self.basic_count, attempts=self.basic_count,
                          numeric_variants=self.basic_count if self.obs["required_choice"] == "buy_accept_reject" else 0)
        for family in self.families:
            size, previous = len(self.choices(family)), 1
            targets = len(self.targets) if family in ("offense", "utility", "status") else 1
            for depth in range(1, self.maximum + 1):
                count = (math.comb(size, depth) if family == "discard" else math.perm(size, depth)) if depth <= size else 0
                complete = _sat_mul(count, targets)
                attempts = _sat_mul(previous, max(0, size - depth + 1))
                for key, amount in (("complete", complete), ("prefixes", count), ("attempts", _sat_add(attempts, complete)),
                                    ("excluded", attempts), ("targets", complete if family in ("offense", "utility", "status") else 0),
                                    ("elements", _sat_mul(complete, depth))):
                    totals[key] = _sat_add(totals[key], amount)
                previous = count
        return totals

    def summary(self, proposal: _Proposal) -> _Summary:
        cards = [self.by_slot[slot] for slot in proposal.refs]
        facts = SequenceAnalysis()
        if proposal.kind == "play_sequence":
            root = cards[0]
            family = "utility" if root.definition["definition_id"] in UTILITY_IDS else (
                "status" if root.subclass == "status" else "offense")
            facts = self.analysis(family, proposal.refs)
        elif proposal.kind == "respond" and cards:
            facts = self.analysis("defense", proposal.refs)
        if proposal.kind in ("play_sequence", "respond"):
            freed = sum(not card.miracle for card in cards)
            opportunities = len(cards)
            removed = {card.slot for card in cards if not card.miracle}
        elif proposal.kind in ("discard", "sacrifice"):
            freed = len(cards)
            opportunities = freed if proposal.kind == "sacrifice" else 0
            removed = set(proposal.refs)
        elif proposal.kind == "pray":
            freed = int(len(self.cards) == 18)
            opportunities, removed = 1, set()
        else:
            removed = set(proposal.refs[:1] if proposal.kind == "sell" and proposal.target == "self" else proposal.refs)
            freed, opportunities = len(removed), 1
        refresh = min(18, opportunities, max(0, 18 - len(self.cards) + freed))
        resources = proposal.numerics[:3] if proposal.kind == "exchange" else (
            facts.spent_mp, len(cards), sum(card.definition.get("price", 0) for card in cards))
        return _Summary(facts.attribute, facts.effects, facts.attack, resources, refresh,
            tuple(card.slot for card in self.cards if card.slot not in removed), len(cards))

    def generated(self, table: CandidateTable, fields: tuple[Any, ...]) -> GeneratedCandidatesV3:
        return GeneratedCandidatesV3(table, SearchReportV3(*fields, action_contract=self.config.action_contract,
                                                         rule_signature=self.rule_signature))


def _grid(lower: int, upper: int) -> tuple[int, ...]:
    # Endpoints/median first improves bounded numeric breadth. All 25 intervals
    # remain present when the axis has enough distinct integer values.
    values = sorted({lower + (upper - lower) * index // 25 for index in range(26)})
    middle = len(values) // 2
    return tuple(dict.fromkeys((values[0], values[-1], values[middle], *values)))


def _allocations(own: Json, total: int, miracle_costs: tuple[int, ...]) -> Iterator[tuple[int, int, int]]:
    if not 0 <= total <= 297:
        return
    # Current allocation is reserved if valid; otherwise a deterministic
    # conserved allocation completes the mandatory carrier.
    current = tuple(own[key] for key in ("hp", "mp", "money"))
    if all(0 <= value <= 99 for value in current):
        yield current[0], current[1], current[2]
    else:
        hp = min(99, total)
        mp = min(99, total - hp)
        yield hp, mp, total - hp - mp
    # Three resource orderings, conditional two-axis grids. Do not form a
    # Cartesian product then drop out-of-range third-axis values.
    for order in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        lower, upper = max(0, total - 198), min(99, total)
        for first in _grid(lower, upper):
            remaining = total - first
            for second in _grid(max(0, remaining - 99), min(99, remaining)):
                values = [0, 0, 0]
                values[order[0]], values[order[1]], values[order[2]] = first, second, remaining - second
                yield values[0], values[1], values[2]
    balanced = total // 3
    yield balanced, (total - balanced) // 2, total - balanced - (total - balanced) // 2
    if 1 <= total <= 199:
        yield 1, min(99, total - 1), max(0, total - 100)
    for cost in sorted(set(miracle_costs)):
        for mp in (cost - 1, cost, cost + 1):
            if max(0, total - 198) <= mp <= min(99, total):
                for hp in (max(0, total - mp - 99), min(99, total - mp)):
                    yield hp, mp, total - hp - mp


class _CountLimit(Exception):
    pass


def _prefix_capacity(space: _Space | _CardInstanceSpace) -> int:
    # Bounded mode has three length-band iterators per start/family. Each DFS
    # retains at most one stack whose prefixes sum to L(L+1)/2 elements.
    jobs = sum(len(space.choices(family)) for family in space.families) * 3
    return jobs * space.maximum * (space.maximum + 1) // 2


def _legacy_upper_counts(space: _Space) -> dict[str, int]:
    length = space.maximum
    totals = dict.fromkeys(("complete", "prefixes", "attempts", "excluded", "targets", "numeric_variants", "elements"), 0)
    basics = space.basic_count
    target_count = len(space.targets)
    statuses = sum(len(space.singleton_targets(card)) for card in space.cards)
    buy = len(space.trade["buy"]) * target_count
    sell = (len(space.trade["sell"]) * max(0, len(space.hand) - 1) * target_count
            if space.config.max_sequence_length >= 2 else 0)
    # Grid visits, balanced/HP1 and up to six cost-neighbor supplements per
    # visible own miracle. This is a public upper bound, not a claimed count.
    exchange = len(space.trade["exchange"]) * (3 * 26**2 + 3 + len(space.cards) * 6)
    fixed = basics + statuses + buy + sell + exchange
    totals.update(complete=fixed, attempts=fixed, targets=statuses + buy + sell,
                  numeric_variants=exchange + (2 if space.obs["required_choice"] == "buy_accept_reject" else 0),
                  elements=statuses + buy + 2 * sell + exchange)
    for family in space.families:
        choices = space.choices(family)
        size = len(choices)
        previous = 1
        for depth in range(1, length + 1):
            if family == "discard":
                count = math.comb(size, depth) if depth <= size else 0
            else:
                count = math.perm(size, depth) if depth <= size else 0
            attempts = _sat_mul(previous, max(0, size - depth + 1))
            targets = target_count if family == "offense" else 1
            complete = _sat_mul(count, targets)
            totals["complete"] = _sat_add(totals["complete"], complete)
            totals["prefixes"] = _sat_add(totals["prefixes"], count)
            totals["attempts"] = _sat_add(totals["attempts"], attempts, complete)
            totals["excluded"] = _sat_add(totals["excluded"], attempts)
            totals["targets"] = _sat_add(totals["targets"], complete if family == "offense" else 0)
            totals["elements"] = _sat_add(totals["elements"], _sat_mul(complete, depth))
            previous = count
    return totals


def _upper_counts(space: _Space | _CardInstanceSpace) -> dict[str, int]:
    return space.upper_counts()


def _precount(space: _Space | _CardInstanceSpace, costs: NumericCosts, clock: Callable[[], float]) -> tuple[PrecountReport, set[str]]:
    started = clock()
    deadline = started + space.config.count_milliseconds / 1000
    states = 0
    reasons: set[str] = set()
    totals = dict.fromkeys(("complete", "prefixes", "attempts", "excluded", "targets", "numeric_variants", "elements"), 0)

    def tick() -> None:
        nonlocal states
        if states >= space.config.count_states:
            reasons.add("count_states")
            raise _CountLimit
        if clock() >= deadline:
            reasons.add("count_time")
            raise _CountLimit
        states += 1

    def add(complete: int, elements: int = 0, targets: int = 0, numeric: int = 0,
            attempts: int = 1, excluded: int = 0) -> None:
        totals["complete"] += complete
        totals["elements"] += elements
        totals["targets"] += targets
        totals["numeric_variants"] += numeric
        totals["attempts"] += attempts
        totals["excluded"] += excluded

    def tree(family: str, prefix: tuple[int, ...]) -> None:
        if len(prefix) >= space.maximum:
            return
        for slot in space.choices(family):
            if slot in prefix:
                continue
            tick()
            refs = (*prefix, slot)
            if not space.valid(family, refs):
                add(0, excluded=1)
                continue
            totals["prefixes"] += 1
            count, elements, targets = space.completion_stats(family, refs)
            add(count, elements, targets, attempts=1 + count)
            tree(family, refs)

    mode = "exact"
    try:
        tick()
        space.count_fixed(tick, add)
        for family in space.families:
            tree(family, ())
    except _CountLimit:
        mode = "upper_bound"
        totals = _upper_counts(space)
    saturated = any(value >= SATURATION for value in totals.values())
    estimate = estimate_costs(costs, totals["complete"], totals["elements"], space.config.max_sequence_length,
                              totals["complete"], totals["elements"], _prefix_capacity(space))
    saturated = saturated or SATURATION in (estimate.preparation_bytes, estimate.scoring_operations, estimate.storage_bytes, estimate.largest_tensor_bytes)
    admitted = (mode == "exact" and not saturated
                and 0 < totals["complete"] <= min(space.config.max_candidates, space.config.max_temporary_candidates)
                and totals["attempts"] <= space.config.max_expansions and estimate.fits(costs))
    return PrecountReport(mode, saturated, states, **totals,
                          preparation_bytes=estimate.preparation_bytes,
                          scoring_operations=estimate.scoring_operations,
                          storage_bytes=estimate.storage_bytes,
                          full_enumeration_admitted=admitted), reasons


class _SearchLimit(Exception):
    pass


class _Collector:
    def __init__(self, space: _Space | _CardInstanceSpace, costs: NumericCosts, clock: Callable[[], float], deadline: float) -> None:
        self.space, self.costs, self.clock, self.deadline = space, costs, clock, deadline
        self.discovery_deadline = deadline - space.config.search_milliseconds / 1000 * .25
        self.summaries: dict[_Proposal, _Summary] = {}
        self.assigned: dict[_Proposal, int] = {}
        self.proposals: dict[_Proposal, None] = {}
        self.mandatory: dict[_Proposal, None] = {}
        self.attempts = self.prunes = self.elements = 0
        self.reasons: set[str] = set()

    def tick(self, mandatory: bool = False) -> None:
        reason = ("max_expansions" if self.attempts >= self.space.config.max_expansions else
                  "search_time" if self.clock() >= self.discovery_deadline else None)
        if reason is not None:
            self.reasons.add(reason)
            if mandatory:
                raise ContractError(f"mandatory coverage cannot fit {reason}")
            raise _SearchLimit
        self.attempts += 1

    def offer(self, proposal: _Proposal, mandatory: bool = False) -> None:
        if proposal in self.proposals:
            self.prunes += 1  # duplicate proposal, not an engine legality claim.
            if mandatory:
                self.mandatory[proposal] = None
            return
        count, elements = len(self.proposals) + 1, self.elements + len(proposal.refs)
        reason = None
        if count > self.space.config.max_temporary_candidates:
            reason = "max_temporary_candidates"
        elif mandatory and len(self.mandatory) + 1 > self.space.config.max_candidates:
            reason = "max_candidates"
        else:
            estimate = estimate_costs(self.costs, min(count, self.space.config.max_candidates),
                                      min(elements, self.space.config.max_candidates * self.space.config.max_sequence_length),
                                      self.space.config.max_sequence_length, count, elements, _prefix_capacity(self.space))
            if not estimate.fits(self.costs):
                reason = "numeric_costs"
        if reason is not None:
            self.reasons.add(reason)
            if mandatory:
                raise ContractError(f"mandatory coverage cannot fit {reason}")
            raise _SearchLimit
        self.summaries[proposal] = self.space.summary(proposal)
        self.proposals[proposal] = None
        self.elements = elements
        if mandatory:
            self.mandatory[proposal] = None


def _walk(space: _Space | _CardInstanceSpace, family: str, prefix: tuple[int, ...], length: int | None,
          order: tuple[int, ...], attempt: Callable[[], None]) -> Iterator[tuple[int, ...] | None]:
    """Yield every attempted extension, including public-domain exclusions.

    length=None traverses every prefix once; bounded length-band walkers emit
    intermediate attempts too, so work cannot disappear from expansion budgets.
    """
    if len(prefix) >= space.maximum or length is not None and len(prefix) >= length:
        return
    for slot in order:
        if slot in prefix:
            continue
        attempt()
        refs = (*prefix, slot)
        if not space.valid(family, refs):
            yield None
            continue
        yield refs
        yield from _walk(space, family, refs, length, order, attempt)


def _length_order(maximum: int, band: int) -> tuple[int, ...]:
    lengths = list(range(2, maximum + 1))
    if not lengths:
        return ()
    # Disjoint bands cover every length, with maximum/short/median starts.
    ordered = list(dict.fromkeys((maximum, 2, (maximum + 2) // 2, *lengths)))
    return tuple(ordered[band::3])


def _bounded_walk(space: _Space | _CardInstanceSpace, family: str, root: tuple[int, ...], band: int,
                  order: tuple[int, ...], attempt: Callable[[], None]) -> Iterator[tuple[int, ...] | None]:
    for length in _length_order(space.maximum, band):
        for refs in _walk(space, family, root, length, order, attempt):
            # Intermediate states still consume attempts but only target lengths
            # are presented by this job. None means excluded, () means a valid
            # prefix extension not yet at this length band's completion depth.
            yield refs if refs is None or len(refs) == length else ()


def _attribute(left: str | None, right: str) -> str:
    if left is None or left == right:
        return right
    if left == "light" and right in ("fire", "water", "wood", "stone"):
        return right
    if right == "light" and left in ("fire", "water", "wood", "stone"):
        return left
    return "neutral"


@dataclass(frozen=True)
class _Summary:
    attribute: str
    effects: tuple[str, ...]
    attack: int
    resources: tuple[int, ...]
    refresh: int
    remaining: tuple[int, ...]
    length: int


def _refresh_potential(space: _Space, proposal: _Proposal) -> int:
    """Public potential acquisition/refill score, never a predicted draw count.

    Miracle occurrences each offer one acquisition (held miracles merely move
    to learned and do not free a possession slot). Ordinary consumed artifacts
    and Sacrifice provide replacement opportunities. We cap by known free
    possession capacity after consumption/transfer. Ignoring resolution ordering
    makes this a public upper proxy within that known-capacity envelope; later
    reactions, forgetting and RNG-dependent draws/removals are not simulated.
    Self-Sell retains the sold reference and only consumes its carrier. Selling
    away a card frees capacity but does not invent a replacement for that sale.
    """
    cards = [space.by_slot[slot] for slot in proposal.refs]
    possession = len(space.cards)
    removed = [card for card in cards if card.hand]
    if proposal.kind == "sell" and proposal.target == "self":
        removed = removed[:1]
    if proposal.kind in ("discard", "sacrifice"):
        freed = len(removed)
        opportunities = len(removed) if proposal.kind == "sacrifice" else 0
    elif proposal.kind in ("sell", "buy", "exchange"):
        freed = len(removed)
        opportunities = 1
    elif proposal.kind == "pray":
        freed = int(possession >= 18 and bool(space.hand))
        opportunities = 1
    else:
        freed = sum(not card.miracle for card in removed)
        opportunities = freed + sum(card.miracle for card in cards)
    return min(18, opportunities, max(0, 18 - possession + freed))


def _summary(space: _Space, proposal: _Proposal) -> _Summary:
    cards = [space.by_slot[slot] for slot in proposal.refs]
    free = any(card.definition["kind"] == "weapons" and card.ability == "cutCost" for card in cards)
    payment = 0 if free else sum(card.definition.get("mp_cost", 0) for card in cards if card.miracle)
    attack, attribute, wand, reset = 0, None, None, False
    for card in cards:
        if card.ability == "doubleAtk":
            attack = min(SATURATION, attack * max(1, card.definition.get("ability_value", 2)))
        elif card.ability == "attackEveryEnemy":
            attribute, wand, reset = "neutral", None, True
        else:
            amount = (2 * max(0, space.obs["own"]["mp"] - payment) if card.ability == "atkBy2xMP"
                      else card.definition.get("attack", 0))
            attack = min(SATURATION, attack + max(0, amount))
            if card.definition.get("attack", 0) > 0 or card.ability == "atkBy2xMP":
                own_attribute = card.definition.get("attribute", "neutral")
                if card.ability == "setElement":
                    attribute = own_attribute if reset else _attribute(attribute, own_attribute)
                    wand = own_attribute
                else:
                    attribute = _attribute(attribute, wand if card.additional and wand is not None else own_attribute)
    spent = space.obs["own"]["mp"] if any(card.ability == "atkBy2xMP" for card in cards) else payment
    refresh = _refresh_potential(space, proposal)
    removed_refs = proposal.refs[:1] if proposal.kind == "sell" and proposal.target == "self" else proposal.refs
    remaining = tuple(card.slot for card in space.hand if card.slot not in removed_refs)
    numeric_resources = proposal.numerics[:3] if proposal.kind == "exchange" else (spent, len(cards), sum(card.definition.get("price", 0) for card in cards))
    return _Summary(attribute or "neutral", tuple(sorted({card.ability for card in cards if card.ability})),
                    attack, numeric_resources, refresh, remaining, len(cards))


def _breadth(values: list[_Proposal], key: Callable[[_Proposal], Any]) -> deque[_Proposal]:
    ordered = sorted(values, key=key)
    result: deque[_Proposal] = deque()
    # Recursively visit low/high/median ranks; no fixed score top-k dominates.
    intervals = deque([(0, len(ordered))])
    while intervals:
        lower, upper = intervals.popleft()
        if lower >= upper:
            continue
        middle = (lower + upper) // 2
        result.append(ordered[lower])
        if upper - lower > 1:
            result.append(ordered[upper - 1])
        if middle not in (lower, upper - 1):
            result.append(ordered[middle])
        intervals.extend(((lower + 1, middle), (middle + 1, upper - 1)))
    return result


def _select(collector: _Collector, rng: random.Random) -> tuple[list[_Proposal], tuple[int, ...]]:
    selected = list(collector.mandatory)
    used = set(selected)
    others = [proposal for proposal in collector.proposals if proposal not in used]
    if not others:
        return selected, (0,) * 7
    summaries = collector.summaries
    preparation_deadline = collector.clock() + max(0., collector.deadline - collector.clock()) * .5
    prepared = []
    for index, proposal in enumerate(others):
        if index % 64 == 0 and collector.clock() >= preparation_deadline:
            collector.reasons.add("search_time")
            break
        prepared.append(proposal)
    others = prepared
    groups: dict[Any, deque[_Proposal]] = {}
    for proposal in others:
        summary = summaries[proposal]
        groups.setdefault((summary.attribute, summary.effects, proposal.kind, proposal.target), deque()).append(proposal)
    categorical: deque[_Proposal] = deque()
    while groups:
        if collector.clock() >= preparation_deadline:
            collector.reasons.add("search_time")
            break
        for group in tuple(groups):
            categorical.append(groups[group].popleft())
            if not groups[group]:
                del groups[group]
    exploration = list(others)
    rng.shuffle(exploration)
    remaining_groups: dict[tuple[int, ...], deque[_Proposal]] = {}
    for proposal in others:
        remaining_groups.setdefault(summaries[proposal].remaining, deque()).append(proposal)
    remaining_order: deque[_Proposal] = deque()
    while remaining_groups:
        if collector.clock() >= preparation_deadline:
            collector.reasons.add("search_time")
            break
        for group in tuple(remaining_groups):
            remaining_order.append(remaining_groups[group].popleft())
            if not remaining_groups[group]:
                del remaining_groups[group]
    channels = [categorical, _breadth(others, lambda item: summaries[item].attack),
                _breadth(others, lambda item: summaries[item].resources),
                _breadth(others, lambda item: summaries[item].refresh), remaining_order,
                _breadth(others, lambda item: summaries[item].length), deque(exploration)]
    counts = [0] * 7
    limit = collector.space.config.max_candidates
    def take(index: int) -> bool:
        if collector.clock() >= collector.deadline:
            collector.reasons.add("search_time")
            raise _SearchLimit
        skipped = 0
        while channels[index] and channels[index][0] in used:
            channels[index].popleft()
            skipped += 1
            if skipped % 64 == 0 and collector.clock() >= collector.deadline:
                collector.reasons.add("search_time")
                raise _SearchLimit
        if not channels[index] or len(selected) >= limit:
            return False
        proposal = channels[index].popleft()
        selected.append(proposal)
        collector.assigned[proposal] = index
        used.add(proposal)
        counts[index] += 1
        return True

    try:
        # Minimum unique opportunity first. A large first-channel weight must
        # never consume the capacity needed by other discovered nonempty queues.
        for index in range(7):
            if len(selected) >= limit:
                break
            take(index)
        while len(selected) < limit and any(channels):
            progress = False
            for index, weight in enumerate(collector.space.config.channel_weights):
                for _ in range(weight):
                    if not take(index):
                        break
                    progress = True
            if not progress:
                break
    except _SearchLimit:
        return selected, tuple(counts)
    if len(selected) >= limit and len(selected) < len(collector.proposals):
        collector.reasons.add("max_candidates")
    return selected, tuple(counts)


def _table(frame: DecisionFrame, proposals: list[_Proposal], collector: _Collector, *,
           purpose: str = "component") -> CandidateTable:
    offsets = [0]
    slots: list[int] = []
    for index, proposal in enumerate(proposals):
        if collector.clock() >= collector.deadline:
            collector.reasons.add("search_time")
            if index < len(collector.mandatory):
                raise ContractError("mandatory table construction exceeds search deadline")
            del proposals[index:]
            break
        slots.extend(proposal.refs)
        offsets.append(len(slots))
    content: Json = {"frame_hash": frame.frame_hash, "kinds": tuple(proposal.kind for proposal in proposals),
               "targets": tuple(proposal.target for proposal in proposals),
               "numerics": tuple(proposal.numerics for proposal in proposals),
               "presence": tuple(proposal.presence for proposal in proposals),
               "offsets": tuple(offsets), "slots": tuple(slots)}
    contract = collector.space.config.action_contract
    table = CandidateTable(**content, action_contract=contract, purpose=purpose, synthetic_capacity=False,
                           table_hash=digest({**content, "action_contract": asdict(contract), "purpose": purpose,
                                              "synthetic_capacity": False}))
    return table


class PublicCandidateGeneratorV2:
    def __init__(self, config: ActionConfigV2, *, clock: Callable[[], float] = time.monotonic) -> None:
        if type(config) is not ActionConfigV2:
            raise ContractError("closed v2 action configuration required")
        self.config, self.clock = config, clock

    def generate(self, frame: DecisionFrame, bank_view: EventBankView,
                 search_rng: random.Random, numeric_costs: NumericCosts) -> GeneratedCandidates:
        if frame.layout is not LEGACY_LAYOUT:
            raise ContractError("generator2 requires the legacy structured layout")
        if type(search_rng) is not random.Random or type(numeric_costs) is not NumericCosts:
            raise ContractError("independent Random and closed NumericCosts required")
        return _run_search(lambda: _Space(frame, self.config), frame, bank_view, search_rng, numeric_costs, self.clock)


class PublicCandidateGeneratorV3:
    def __init__(self, config: ActionConfigV3, *, rules: CardInstanceRules,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(config) is not ActionConfigV3:
            raise ContractError("closed v3 action configuration required")
        self.rule_signature = rule_signature(config.action_contract, rules)
        self.config, self.rules, self.clock = config, rules, clock

    def generate(self, frame: DecisionFrame, bank_view: EventBankView,
                 search_rng: random.Random, numeric_costs: NumericCosts) -> GeneratedCandidatesV3:
        expected_layout = layout_for_contract(self.config.action_contract)
        if frame.layout is not expected_layout or bank_view.layout is not expected_layout:
            raise ContractError("generator3 requires the card-instance layout")
        if type(search_rng) is not random.Random or type(numeric_costs) is not NumericCosts:
            raise ContractError("independent Random and closed NumericCosts required")
        return _run_search(lambda: _CardInstanceSpace(frame, self.config, self.rules, bank_view), frame, bank_view,
                           search_rng, numeric_costs, self.clock)

    def generate_game(self, frame: DecisionFrame, bank_view: EventBankView,
                      search_rng: random.Random, numeric_costs: NumericCosts) -> GameGeneratedCandidatesV3:
        if not is_corrected_contract(self.config.action_contract):
            raise ContractError("game generation requires the corrected contract")
        expected = layout_for_contract(self.config.action_contract)
        if frame.layout is not expected or bank_view.layout is not expected:
            raise ContractError("game generator frame/bank profile mismatch")
        if type(search_rng) is not random.Random or type(numeric_costs) is not NumericCosts:
            raise ContractError("independent Random and closed NumericCosts required")
        return _run_search(lambda: _CardInstanceSpace(frame, self.config, self.rules, bank_view), frame, bank_view,
                           search_rng, numeric_costs, self.clock, game=True)


def _run_search(space_factory: Callable[[], _Space | _CardInstanceSpace], frame: DecisionFrame, bank_view: EventBankView,
                search_rng: random.Random, numeric_costs: NumericCosts, clock: Callable[[], float], *, game: bool = False) -> Any:
    count_started = clock()
    frame.bank_view(bank_view)  # same participant/episode, no future events.
    space = space_factory()
    if space.trade["exchange"] and not 0 <= space.total <= 297:
        raise ContractError("hinted exchange carrier has no conserved range-valid allocation")
    # Validate all numeric dimensions/costs even for empty proposal spaces.
    estimate_costs(numeric_costs, 0, 0, space.config.max_sequence_length)
    rng_before = search_rng.getstate()
    precount, reasons = _precount(space, numeric_costs, clock)
    search_started = clock()
    collector = _Collector(space, numeric_costs, clock,
                           search_started + space.config.search_milliseconds / 1000)
    collector.reasons.update(reasons)
    if (space.config.max_sequence_length < 2 and space.trade["sell"]
            and len(space.hand) >= 2 and space.targets):
        collector.reasons.add("max_sequence_length")
    roots: dict[str, list[tuple[int, ...]]] = {}
    # Mandatory coverage is attempted before discretionary search. Filtering
    # fixed domains avoids numeric supplements/sell rows delaying a carrier.
    mandatory_attempt = lambda: collector.tick(True)
    for proposal in space.mandatory_rows(mandatory_attempt):
        collector.offer(proposal, True)
    for family in space.families:
        roots[family] = []
        for slot in space.choices(family):
            collector.tick(True)
            root_refs = (slot,)
            if space.valid(family, root_refs):
                roots[family].append(root_refs)
                for proposal in space.completions(family, root_refs, mandatory_attempt):
                    collector.offer(proposal, True)
            else:
                collector.prunes += 1
    complete = False
    try:
        # One shared round-robin work queue: numeric/sell supplements
        # cannot monopolize the attempt/time budget ahead of tree lengths.
        jobs: deque[tuple[str, Iterator[Any]]] = deque([(                "fixed", space.fixed(collector.tick))])
        for family in space.families:
            if precount.full_enumeration_admitted:
                order = tuple(space.choices(family))
                for root in roots[family]:
                    jobs.append((family, _walk(space, family, root, None, order, collector.tick)))
            else:
                shuffled_order = list(space.choices(family))
                search_rng.shuffle(shuffled_order)
                for root in roots[family]:
                    for band in range(3):
                        jobs.append((family, _bounded_walk(space, family, root, band,
                                                          tuple(shuffled_order), collector.tick)))
        while jobs:
            family, job = jobs.popleft()
            try:
                item = next(job)
            except StopIteration:
                continue
            if family == "fixed":
                collector.offer(item)
            elif item is None:
                collector.prunes += 1
            elif item:
                for proposal in space.completions(family, item, collector.tick):
                    collector.offer(proposal)
            jobs.append((family, job))
        complete = True
    except _SearchLimit:
        pass
    if not collector.proposals:
        version, suffix = (("v4", "; owner-profile pending coverage remains") if type(space.config) is ActionConfigV2
                           else ("v5", "; component proposals grant no Simulator qualification"))
        raise EmptyCandidates("no completed " + version + " proposal found; cutoffs="
                              + ",".join(sorted(collector.reasons)) + suffix)
    selection_started = clock()
    selected, channel_counts = _select(collector, search_rng)
    selected_elements = sum(len(proposal.refs) for proposal in selected)
    if not estimate_costs(numeric_costs, len(selected), selected_elements, space.config.max_sequence_length,
                          len(collector.proposals), collector.elements, _prefix_capacity(space)).fits(numeric_costs):
        raise ContractError("selected table exceeds supplied numeric cost envelope")
    table = _table(frame, selected, collector, purpose="qualified_game" if game else "component")
    selected_elements = len(table.slots)
    channel_counts = tuple(sum(collector.assigned.get(proposal) == index for proposal in selected) for index in range(7))
    finished = clock()
    if finished >= collector.deadline:
        collector.reasons.add("search_time")
    fields = (space.config.generator, digest(asdict(space.config)), digest(asdict(numeric_costs)), table.table_hash,
        numeric_costs, precount, collector.attempts, collector.prunes, len(collector.proposals),
        len(collector.mandatory), len(selected), selected_elements, channel_counts,
        complete, complete and len(selected) == len(collector.proposals), tuple(sorted(collector.reasons)),
        OMISSIONS if type(space.config) is ActionConfigV2 else OMISSIONS_V3, rng_before, search_rng.getstate(),
        (search_started - count_started) * 1000, (selection_started - search_started) * 1000,
        (finished - selection_started) * 1000)
    if game:
        assert isinstance(space, _CardInstanceSpace)
        return GameGeneratedCandidatesV3(table, SearchReportV3(*fields, action_contract=space.config.action_contract,
                                                              rule_signature=space.rule_signature))
    return space.generated(table, fields)
