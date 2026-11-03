"""Owner-confirmed public candidate analysis, never Simulator legality authority.

Only own perceived definitions, public state and explicit atomic availability
enter this module. Results are immutable primitive facts; there is no I/O,
search DTO dependency, rejection probing, future draw or hidden-state inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from gf_common.card_instance import (
    GENERATOR_VERSION,
    CardInstanceContract,
    parse_card_instance_contract,
)
from gf_common.domain import ContractError, Json, closed, digest, integer

OWNER_CONFIRMATION_SHA256 = "3f3090f521041293bcb6de2a3cbac7e74ffc635b4acc0f7f02c957bbcf0e44ee"
UTILITY_IDS = frozenset(
    (
        "smile-dew",
        "heart-dew",
        "romance-water",
        "galaxy-geyser",
        "smile-flower",
        "heart-flower",
        "romance-fragrance",
        "spring",
        "treasure",
        "tone",
        "song",
        "smile-shell",
        "heart-shell",
        "thump-thump-tear",
        "heaven-herb",
        "goddess-s-soap",
        "nocturnal-broom",
        "guardian-pot",
        "release",
    )
)
EXCLUSIVE = frozenset(
    ("reflectAnything", "reflectWeapon", "reflectMiracle", "bounceWeapon", "bounceMiracle")
)
RINGS = frozenset(
    ("counterAtk", "counterCurse", "counterBoost2xMP", "counter2xAtk", "counterTakeCP")
)
SATURATION = 2**63 - 1


@dataclass(frozen=True)
class CardInstanceRules:
    version: str = "1"
    profile_id: str = "owner-card-instance-candidates-20261103-v1"
    owner_confirmation_sha256: str = OWNER_CONFIRMATION_SHA256

    def __post_init__(self) -> None:
        for name, descriptor in self.__dataclass_fields__.items():
            if type(getattr(self, name)) is not str or getattr(self, name) != descriptor.default:
                raise ContractError("unknown owner-confirmed card-instance rules")


def card_instance_rules() -> CardInstanceRules:
    return CardInstanceRules()


def parse_card_instance_rules(raw: Json) -> CardInstanceRules:
    return CardInstanceRules(**closed(raw, set(CardInstanceRules.__dataclass_fields__)))


def rule_signature(contract: CardInstanceContract, rules: CardInstanceRules) -> str:
    if type(contract) is not CardInstanceContract or type(rules) is not CardInstanceRules:
        raise ContractError("explicit closed card-instance contract and rules required")
    parse_card_instance_contract(asdict(contract))
    parse_card_instance_rules(asdict(rules))
    return digest(
        {
            "generator_version": GENERATOR_VERSION,
            "input_recipe": contract.input_recipe,
            "action_contract": asdict(contract),
            "rules": asdict(rules),
        }
    )


@dataclass(frozen=True)
class PublicCard:
    slot: int
    kind: str
    state: str | None
    definition: Json = field(compare=False, repr=False)

    @property
    def ability(self) -> str | None:
        return self.definition.get("ability")

    @property
    def miracle(self) -> bool:
        return self.kind == "miracle"

    @property
    def hand(self) -> bool:
        """Unused trade-visible possession, never a fabricated v1 source."""
        return not self.miracle or self.state == "unused"

    @property
    def additional(self) -> bool:
        return "additional_attack" in self.definition["classification"]["use_roles"]

    @property
    def subclass(self) -> str | None:
        return self.definition["classification"].get("miracle_subclass")


@dataclass(frozen=True)
class SequenceAnalysis:
    viable: bool = False
    complete: bool = False
    payment: int = 0
    spent_mp: int = 0
    effective_slots: tuple[int, ...] = ()
    cost_cut_pairs: tuple[tuple[int, int], ...] = ()
    attack_per_round: int = 0
    rounds: int = 1
    attack: int = 0
    attribute: str = "neutral"
    effects: tuple[str, ...] = ()
    opponent_only: bool = False


def _attribute(left: str | None, right: str) -> str:
    if left is None or left == right:
        return right
    if left == "light" and right in ("fire", "water", "wood", "stone"):
        return right
    if right == "light" and left in ("fire", "water", "wood", "stone"):
        return left
    return "neutral"


def defense_kind(card: PublicCard) -> str | None:
    if card.ability in EXCLUSIVE or (
        card.ability in ("blockWeapon", "blockMiracle") and card.definition.get("defense", 0) <= 0
    ):
        return "exclusive"
    if (
        "numeric_defense" in card.definition["classification"]["use_roles"]
        or card.definition.get("defense", 0) > 0
        or card.ability in RINGS
        or card.ability == "filterAtkElement"
    ):
        return "ordinary"
    return None


def analyze_sequence(
    cards: tuple[PublicCard, ...],
    refs: tuple[int, ...],
    *,
    family: str,
    mp: int,
    flash: bool,
    maximum: int,
    usable: frozenset[int],
) -> SequenceAnalysis:
    """Separate extendable prefixes from completed affordable public proposals.

    Adjacent CostCut role binding happens before role compatibility, MP payment,
    attack/effect summaries or Flash counting. All inputs come from a validated
    original frame; slots are local indexes, never opaque reference features.
    """
    integer(mp, 0, 2**32 - 1)
    integer(maximum, 1, 128)
    if type(flash) is not bool or family not in ("offense", "defense", "utility", "status"):
        raise ContractError("closed sequence family and public Flash state required")
    by_slot = {card.slot: card for card in cards}
    if (
        not refs
        or len(refs) > min(maximum, 18)
        or len(set(refs)) != len(refs)
        or any(slot not in by_slot or slot not in usable for slot in refs)
        or family == "defense"
        and flash
        and len(refs) > 1
    ):
        return SequenceAnalysis()
    selected = [by_slot[slot] for slot in refs]
    pairs = tuple(
        (selected[index - 1].slot, card.slot)
        for index, card in enumerate(selected)
        if index and card.ability == "cutCost" and selected[index - 1].miracle
    )
    suppressed = {support for _, support in pairs}
    waived = {miracle for miracle, _ in pairs}
    effective = [card for card in selected if card.slot not in suppressed]
    if family in ("utility", "status"):
        if len(effective) != 1:
            return SequenceAnalysis()
        root = effective[0]
        if (
            family == "utility"
            and root.definition["definition_id"] not in UTILITY_IDS
            or family == "status"
            and root.subclass != "status"
        ):
            return SequenceAnalysis()
    elif family == "defense":
        kinds = [defense_kind(card) for card in effective]
        if any(kind is None for kind in kinds) or "exclusive" in kinds and len(kinds) != 1:
            return SequenceAnalysis()
    else:
        normal_miracles = [card for card in effective if card.miracle and card.subclass == "weapon"]
        artifact_attacks = [
            card
            for card in effective
            if not card.miracle
            and ("weapon" in card.definition["classification"]["use_roles"] or card.additional)
        ]
        if normal_miracles and (
            artifact_attacks
            or effective[0] is not normal_miracles[0]
            or len({card.definition["definition_id"] for card in normal_miracles}) != 1
        ):
            return SequenceAnalysis()
        for index, card in enumerate(effective):
            if card.miracle:
                if card.subclass not in ("weapon", "additional_weapon", "attack_support"):
                    return SequenceAnalysis()
                if card.subclass == "attack_support" and index == 0:
                    return SequenceAnalysis()
            elif card not in artifact_attacks or not card.additional and index != 0:
                return SequenceAnalysis()
    payment = sum(
        card.definition.get("mp_cost", 0)
        for card in selected
        if card.miracle and card.slot not in waived
    )
    affordable = payment <= mp
    salvage = False
    if not affordable and selected[-1].miracle and len(refs) < min(maximum, 18):
        last = selected[-1]
        earlier = payment - (0 if last.slot in waived else last.definition.get("mp_cost", 0))
        salvage = (
            earlier <= mp
            and not (family == "defense" and flash)
            and any(
                card.slot not in refs and card.slot in usable and card.ability == "cutCost"
                for card in cards
            )
        )
    attack, attribute, wand, reset, mirages = 0, None, None, False, 0
    if family == "offense":
        for card in effective:
            if card.ability == "doubleAtk":
                attack = min(SATURATION, attack * max(1, card.definition.get("ability_value", 2)))
            elif card.ability == "attackEveryEnemy":
                attribute, wand, reset = "neutral", None, True
                mirages += 1
            else:
                amount = (
                    2 * max(0, mp - payment)
                    if card.ability == "atkBy2xMP"
                    else max(0, card.definition.get("attack", 0))
                )
                attack = min(SATURATION, attack + amount)
                if amount or card.ability == "atkBy2xMP":
                    own_attribute = card.definition.get("attribute", "neutral")
                    if card.ability == "setElement":
                        attribute = own_attribute if reset else _attribute(attribute, own_attribute)
                        wand = own_attribute
                    else:
                        attribute = _attribute(
                            attribute,
                            wand if card.additional and wand is not None else own_attribute,
                        )
    rounds = (2 if any(card.ability == "attackTwice" for card in effective) else 1) * max(
        1, mirages
    )
    spent = (
        mp
        if family == "offense" and any(card.ability == "atkBy2xMP" for card in effective)
        else payment
    )
    return SequenceAnalysis(
        affordable or salvage,
        affordable,
        payment,
        spent,
        tuple(card.slot for card in effective),
        pairs,
        attack,
        rounds,
        min(SATURATION, attack * rounds),
        attribute or "neutral",
        tuple(sorted({card.ability for card in effective if card.ability})),
        bool(mirages) and not any(card.ability == "danger" for card in effective),
    )
