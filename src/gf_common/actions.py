"""Versioned public subset, using gameplay spec 09 §§9–17, authorized actor atomic hints (ADR0005).

public-atomic-subset-v3 coverage:
* Actor atomic masks apply to the supported references below. Pray/allow and
  discard/sacrifice families use their dedicated flags (ADR0005). Explicit
  observation-only mode preserves the conservative visible weapon prayer rule.
* Ordered offensive positive-attack weapons, armor/sundry additional attacks,
  attack miracles (no ability), and Aura after an attack; ordinary weapons / normal
  attack miracles first, additional attacks thereafter (§9.4). Learned additional
  attacks and Aura may repeat, paying MP for each occurrence (§16); normal attack
  miracles may occur once as the base, hand instances never repeat.
* Hinted own atkBy2xMP weapons as target-specific singleton plays only
  (ADR0005 bounded amendment); never expanded into combined sequences.
* Ordered positive-defense armor/weapons with no special ability and empty allow
  (§14). Single stop/reflect/bounce cards/Wall/Turbulence are supported but omitted
  when incoming kind is unknown: v1 Observation lacks pending attack metadata.
* Ordered nonweapon discard/sacrifice, trade Exchange/Sell/Buy with bounded named
  allocation presets and configured targets (§§11,17). Normal discard excludes
  Sun Amulet and Dangerous Mortar under the public developer announcement;
  Sacrifice retains its separate public rules (ADR0005).
Omissions: defensive rings/reflect/bounce/stop, spirit cost reduction, Mirage,
zero-attack utility/curse miracles, sundry utility, ability-only weapons, and
special sequence effects beyond offensive card-local effects. No full legality
claim: perceived identities and uncertain public conditions may be rejected.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict

from gf_common.domain import (
    ActionConfig,
    CandidateSet,
    ContractError,
    DecisionInput,
    EmptyCandidates,
    Json,
    canonical,
    digest,
    integer,
)
from gf_common.validation import validate_worker_input

GENERATOR_VERSION = "public-atomic-subset-v3"
SCORER_VERSION = "public-content-v1"


def _element(card: Json) -> Json:
    if "instance_id" in card:
        return {"source": "hand_artifact", "instance_id": card["instance_id"]}
    return {"source": "learned_miracle", "definition_id": card["definition_id"]}


def _cards(value: DecisionInput) -> list[Json]:
    return [{**c, "definition": value.catalog[c["definition_id"]]} for c in value.observation["own"]["hand"]] + [
        {"definition_id": key, "definition": value.catalog[key]} for key in value.observation["own"]["learned_miracles"]]


def action_cards(value: DecisionInput, action: Json) -> list[Json]:
    hand = {c["instance_id"]: value.catalog[c["definition_id"]] for c in value.observation["own"]["hand"]}
    elements = action.get("ordered_elements", action.get("ordered_defenses", []))
    if action["type"] in ("discard", "sacrifice"):
        elements = action["instance_ids"]
    result = []
    for element in elements or []:
        if isinstance(element, int):
            result.append(hand[element])
        elif element["source"] == "hand_artifact":
            result.append(hand[element["instance_id"]])
        else:
            result.append(value.catalog[element["definition_id"]])
    return result


def content_score(value: DecisionInput, action: Json) -> float:
    """Fixed ordering heuristic; IDs and candidate index never become features."""
    tag = action["type"]
    cards = action_cards(value, action)
    if tag == "play_sequence":
        score = 0.0
        for card in cards:
            if card.get("ability") == "doubleAtk":
                score *= max(1, card.get("ability_value", 2))
            else:
                score += max(0, card.get("attack", 0)) * card.get("hit_rate", 100) / 100
            if card.get("attribute") == "darkness":
                score += 6
        if action.get("target") == value.observation["own"]["player_id"]:
            score = -score
        return score - 0.05 * len(cards)
    if tag in ("respond", "respond_sequence"):
        return sum(max(0, c.get("defense", 0)) for c in cards) - 0.05 * len(cards)
    if tag == "pray":
        return 2.0
    if tag == "sacrifice":
        return 1 + len(cards) * 0.1
    if tag == "discard":
        return -1 - sum(c.get("price", 0) for c in cards) * 0.05
    if tag == "buy_decision":
        return 1.0 if action["accept"] else 0.0
    if tag == "exchange":
        return (action["hp"] - value.observation["own"]["hp"]) * 0.2
    return 0.0


class PublicActionSpace:
    """Pure lazy expansion/completion, separate from ordering and approximation."""
    def __init__(self, value: DecisionInput, config: ActionConfig) -> None:
        self.value = value
        self.config = config
        self.obs = value.observation
        self.choice = self.obs["required_choice"]
        self.options = value.action_options["action_options"] if value.action_options is not None else None
        self.cards = _cards(value)
        self.targets = [self.obs["own" if name == "self" else name]["player_id"] for name in config.targets]
        if self.options is not None:
            self.targets = [target for target in self.targets if target in self.options["valid_targets"]]
        self.hand = [c for c in self.cards if "instance_id" in c]
        self.offense = [c for c in self.cards if self._usable(c) and self._offensive(c["definition"])]
        self.defense = [c for c in self.cards if self._usable(c) and c["definition"].get("defense", 0) > 0 and not c["definition"].get("ability")]
        self.discard = [c for c in self.hand if c["definition"]["kind"] != "weapons"
                        and (self.obs["end_times"] or c["definition_id"] not in {"sun-amulet", "dangerous-mortar"})]

    def _usable(self, card: Json) -> bool:
        if self.options is None:
            return True
        return (card["instance_id"] in self.options["usable_hand_elements"] if "instance_id" in card
                else card["definition_id"] in self.options["usable_learned_miracles"])

    @staticmethod
    def _offensive(definition: Json) -> bool:
        if definition.get("ability") == "doubleAtk":
            return definition["kind"] == "miracles"
        if definition.get("attack", 0) <= 0:
            return False
        if definition["kind"] == "miracles":
            return "ability" not in definition
        return definition["kind"] == "weapons" or bool(definition.get("additional_attack"))

    def basics(self) -> list[Json]:
        if self.choice == "buy_accept_reject":
            return [{"type": "buy_decision", "accept": False}, {"type": "buy_decision", "accept": True}]
        if self.choice == "defense":
            return [{"type": "respond", "ordered_defenses": []}] if self.options is None or self.options["can_allow"] else []
        if self.choice == "turn_action" and (self.options["can_pray"] if self.options is not None else not any(c["definition"]["kind"] == "weapons" for c in self.hand)):
            return [{"type": "pray"}]
        return []

    def standalone(self) -> Iterator[Json]:
        if self.choice != "turn_action" or not self.obs["actionable"] or not any(
                descriptor["type"] == "play_sequence" for descriptor in self.value.typed_metadata):
            return
        total = sum(self.obs["own"][k] for k in ("hp", "mp", "money"))
        for card in self.hand:
            ability = card["definition"].get("ability")
            if self.options is not None and card["definition"]["kind"] == "weapons" and ability == "atkBy2xMP" and self._usable(card):
                for target in self.targets:
                    yield {"type": "play_sequence", "ordered_elements": [_element(card)], "target": target}
                continue
            if card["definition"]["kind"] != "trade" or ability not in ("exchange", "sell", "buy"):
                continue
            if self.options is not None and card["instance_id"] not in self.options.get(ability + "_artifacts", []):
                continue
            if ability == "exchange":
                for allocation in self.config.exchange_allocations:
                    if allocation == "balanced":
                        hp = total // 3
                        mp = (total - hp) // 2
                        money = total - hp - mp
                    else:
                        amounts = {k: 0 for k in ("hp", "mp", "money")}
                        remaining = total
                        for key in (allocation, *(k for k in amounts if k != allocation)):
                            amounts[key] = min(99, remaining)
                            remaining -= amounts[key]
                        hp, mp, money = (amounts[k] for k in ("hp", "mp", "money"))
                    yield {"type": "exchange", "artifact": card["instance_id"], "hp": hp, "mp": mp, "money": money}
            elif ability == "sell":
                for sold in self.hand:
                    for target in self.targets:
                        yield {"type": "sell", "artifact": card["instance_id"], "sold_artifact": sold["instance_id"], "target": target}
            elif ability == "buy":
                for target in self.targets:
                    yield {"type": "buy", "artifact": card["instance_id"], "target": target}

    def standalone_pruned(self, action: Json) -> bool:
        if action["type"] == "sell":  # §17.2 distinct instances
            return action["artifact"] == action["sold_artifact"]
        if action["type"] == "exchange":  # §17.1 resource conservation and range
            values = [action[k] for k in ("hp", "mp", "money")]
            return any(not 0 <= n <= 99 for n in values) or sum(values) != sum(self.obs["own"][k] for k in ("hp", "mp", "money"))
        return False

    def families(self) -> list[str]:
        if self.choice == "turn_action":
            discard_allowed = self.options is None or self.options["can_sacrifice" if self.obs["end_times"] else "can_discard"]
            return ["offense", "discard"] if discard_allowed else ["offense"]
        if self.choice == "defense":
            return ["defense"]
        return []

    def children(self, family: str, prefix: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        cards = getattr(self, family)
        for index in range(len(cards)):
            yield (*prefix, index)

    def pruned(self, family: str, prefix: tuple[int, ...]) -> bool:
        cards = [getattr(self, family)[i] for i in prefix]
        hand_ids = [c["instance_id"] for c in cards if "instance_id" in c]
        if len(hand_ids) != len(set(hand_ids)):  # §9 ownership/instance uniqueness
            return True
        if family == "discard":
            return False
        # §16.3: each miracle occurrence pays the visible printed MP cost.
        if sum(c["definition"].get("mp_cost", 0) for c in cards if c["definition"]["kind"] == "miracles") > self.obs["own"]["mp"]:
            return True
        if family == "defense":
            return bool(self.obs["own"]["status"]["flash"] and len(cards) > 1)  # §14
        # §9.4 normal base before additional; support Aura affects preceding attack only.
        for position, card in enumerate(cards):
            definition = card["definition"]
            if definition.get("ability") == "doubleAtk":
                if not any(c["definition"].get("attack", 0) > 0 for c in cards[:position]):
                    return True
            elif not definition.get("additional_attack", False) and position > 0:
                return True
        return False

    def completions(self, family: str, prefix: tuple[int, ...]) -> Iterator[Json]:
        if not prefix:
            return  # Incomplete prefixes remain expandable.
        cards = [getattr(self, family)[i] for i in prefix]
        if family == "discard":
            yield {"type": "sacrifice" if self.obs["end_times"] else "discard", "instance_ids": [c["instance_id"] for c in cards]}
        elif family == "defense":
            yield {"type": "respond_sequence", "ordered_defenses": [_element(c) for c in cards]}
        elif any(c["definition"].get("attack", 0) > 0 for c in cards):
            for target in self.targets:
                yield {"type": "play_sequence", "ordered_elements": [_element(c) for c in cards], "target": target}

    def prefix_score(self, family: str, prefix: tuple[int, ...]) -> float:
        return max((content_score(self.value, a) for a in self.completions(family, prefix)), default=0.0)


class PublicCandidateGenerator:
    def __init__(self, config: ActionConfig) -> None:
        if type(config.use_action_options) is not bool:
            raise ContractError("use_action_options must be boolean")
        if config.scorer != "public_heuristic_v1":
            raise ContractError("unsupported fixed scorer")
        for number in (config.max_candidates, config.max_expansions, config.beam_width):
            integer(number, 1)
        integer(config.max_sequence_length)
        if not config.targets or set(config.targets) - {"self", "opponent"} or not config.exchange_allocations or set(config.exchange_allocations) - {"hp", "mp", "money", "balanced"}:
            raise ContractError("unsupported action settings")
        if config.strategy not in ("enumerate", "beam"):
            raise ContractError("unsupported search strategy")
        if min(config.max_candidates, config.max_expansions, config.beam_width) < 1 or config.max_sequence_length < 0:
            raise ContractError("invalid search bounds")
        self.config = config

    def generate(self, decision_input: DecisionInput) -> CandidateSet:
        started = time.perf_counter()
        value = validate_worker_input(decision_input, self.config.use_action_options)
        if not value.observation["actionable"]:
            raise ContractError("candidate generation requires actionable input")
        space = PublicActionSpace(value, self.config)
        basics = space.basics()
        if len(basics) > min(self.config.max_expansions, self.config.max_candidates):
            raise ContractError("budget cannot reserve all supported basic actions")
        attempts = len(basics)
        public_prunes = 0
        approximate_drops = 0
        cutoff = False
        cutoff_reasons: set[str] = set()
        selected: dict[str, Json] = {canonical(a): a for a in basics}
        reserved = set(selected)

        def offer(action: Json) -> None:
            nonlocal approximate_drops
            key = canonical(action)
            if key in selected:
                return
            selected[key] = action
            if len(selected) > self.config.max_candidates:
                worst = min((k for k in selected if k not in reserved), key=lambda k: (content_score(value, selected[k]), k))
                del selected[worst]
                approximate_drops += 1
                cutoff_reasons.add("max_candidates")

        def attempt() -> bool:
            nonlocal attempts, cutoff
            if attempts >= self.config.max_expansions:
                cutoff = True
                cutoff_reasons.add("max_expansions")
                return False
            attempts += 1
            return True

        for action in space.standalone():
            if not attempt():
                break
            if space.standalone_pruned(action):
                public_prunes += 1
            else:
                offer(action)
        length_cutoff = False
        for family in space.families():
            frontier: list[tuple[int, ...]] = [()]
            depth = 0
            while frontier and not cutoff and (not self.config.max_sequence_length or depth < self.config.max_sequence_length):
                next_frontier: list[tuple[int, ...]] = []
                for prefix in frontier:
                    for child in space.children(family, prefix):
                        if not attempt():
                            break
                        if space.pruned(family, child):
                            public_prunes += 1
                            continue
                        for action in space.completions(family, child):
                            offer(action)
                        next_frontier.append(child)
                        # Bound frontier storage too, without materializing the full product.
                        if self.config.strategy == "beam" and len(next_frontier) > self.config.beam_width:
                            next_frontier.sort(key=lambda p: (-space.prefix_score(family, p), p))
                            next_frontier.pop()
                            approximate_drops += 1
                            cutoff_reasons.add("beam_width")
                    if cutoff:
                        break
                frontier = next_frontier
                depth += 1
            if frontier and self.config.max_sequence_length and depth == self.config.max_sequence_length:
                length_cutoff = True
                cutoff_reasons.add("max_sequence_length")
        cutoff = cutoff or length_cutoff
        if not selected:
            raise EmptyCandidates("public subset produced no completed candidate")
        return CandidateSet(tuple(selected.values()), GENERATOR_VERSION, SCORER_VERSION,
                            digest(asdict(self.config)), attempts, public_prunes, approximate_drops,
                            (time.perf_counter() - started) * 1000, cutoff or bool(approximate_drops),
                            tuple(sorted(cutoff_reasons)) or ("exhausted",))
