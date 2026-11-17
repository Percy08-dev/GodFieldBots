"""Owner-confirmed public candidate analysis, never Simulator legality authority.

Only own perceived definitions, public state and explicit atomic availability
enter this module. Results are immutable primitive facts; there is no I/O,
search DTO dependency, rejection probing, future draw or hidden-state inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gf_common.structured import DecisionFrame, EventBankView

from gf_common.card_instance import (
    GENERATOR_VERSION,
    CardInstanceContract,
    parse_card_instance_contract,
)
from gf_common.domain import ContractError, Json, closed, digest, integer

OWNER_CONFIRMATION_SHA256 = "bdd1792cde50eae2338b5f068436e40ca87cfbdf0c276ad096651bc0ff27375c"
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
    version: str = "2"
    profile_id: str = "owner-card-instance-candidates-20261117-v2"
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
class DefenseContext:
    incoming: str = "unknown"

    def __post_init__(self) -> None:
        if self.incoming not in ("unknown", "miracle", "weapon"):
            raise ContractError("closed public defense context required")


UNKNOWN_DEFENSE_CONTEXT = DefenseContext()


def defense_context(frame: DecisionFrame, bank_view: EventBankView) -> DefenseContext:
    """Recognize only one complete direct cause in the verified own prefix.

    No missing miracle event proves a weapon. Weapon classification requires a
    matching definition already in this participant's perceived own catalog.
    Unsupported events anywhere in the current action block leave it unknown.
    """
    view = frame.bank_view(bank_view)
    obs = frame.observation
    unknown = DefenseContext()
    if obs["required_choice"] != "defense" or obs["own"]["status"]["fog"]:
        return unknown
    events = view.events()
    start = next((i for i in range(len(events) - 1, -1, -1)
                  if events[i]["type"] == "action_accepted"), None)
    if start is None:
        return unknown
    block = events[start:]
    if len(block) not in (3, 4) or block[0]["player"] != 3 - frame.actor:
        return unknown
    use, attack = block[1], block[-1]
    if (use["type"] != "artifact_used" or use["player"] != 3 - frame.actor
            or attack["type"] != "attack_declared" or attack["attacker"] != 3 - frame.actor
            or attack["target"] != frame.actor):
        return unknown
    if len(block) == 4:
        miracle = block[2]
        if (miracle["type"] == "miracle_used" and miracle["player"] == use["player"]
                and miracle["definition_id"] == use["definition_id"]):
            return DefenseContext("miracle")
        return unknown
    own_ids = {row["definition_id"] for row in obs["own"]["cards"] if row["kind"] == "artifact"}
    definition = frame.raw["catalog"].get(use["definition_id"]) if use["definition_id"] in own_ids else None
    if definition is not None and "weapon" in definition["classification"]["use_roles"]:
        return DefenseContext("weapon")
    return unknown


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
    context_unknown: bool = False


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
    rules: CardInstanceRules,
    context: DefenseContext = UNKNOWN_DEFENSE_CONTEXT,
) -> SequenceAnalysis:
    """Separate extendable prefixes from completed affordable public proposals.

    Adjacent CostCut role binding happens before role compatibility, MP payment,
    attack/effect summaries or Flash counting. All inputs come from a validated
    original frame; slots are local indexes, never opaque reference features.
    """
    integer(mp, 0, 2**32 - 1)
    integer(maximum, 1, 128)
    if type(rules) is not CardInstanceRules or type(context) is not DefenseContext:
        raise ContractError("explicit rules and public defense context required")
    uncertain = False
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
        if any(card.ability == "filterAtkElement" and index != 0
               for index, card in enumerate(selected)):
            return SequenceAnalysis()
        for index, card in enumerate(effective):
            if card.definition.get("defense", 0) > 0 and card.ability in (
                "blockMiracle", "bounceMiracle", "reflectMiracle"
            ):
                if context.incoming == "miracle":
                    kinds[index] = "exclusive"
                elif context.incoming == "weapon":
                    kinds[index] = "ordinary"
                elif len(effective) > 1:
                    kinds[index] = "ordinary"
                    uncertain = True
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
        affordable and not uncertain,
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
        uncertain,
    )
