"""Participant-local refresh scoring over supplied corrected public candidates.

Outcome and retention values are deliberately bounded public proxies. They never
promise legality, causal attribution, future resources or unknown draw outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import numpy.typing as npt

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import (
    UTILITY_IDS,
    CardInstanceRules,
    PublicCard,
    analyze_sequence,
)
from gf_common.diverse_actions import GameGeneratedCandidatesV3
from gf_common.domain import ContractError, digest, integer
from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
from gf_common.structured import DecisionFrame, EventBankView, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.card_instance_baselines import CardInstanceBaseline


@dataclass(frozen=True)
class _Fact:
    mask: int
    selected: int
    consumed: int
    spent: int = 0
    attack: int = 0
    offense: bool = False
    removed: int = 0
    opportunities: int = 0
    uncertain: bool = False


@dataclass(frozen=True)
class _Pending:
    turn: int
    attack: int
    decision: int
    damage: int = 0


def _facts(
    frame: DecisionFrame, generated: GameGeneratedCandidatesV3, actions: ActionConfigV3,
    rules: CardInstanceRules,
) -> tuple[_Fact, ...]:
    raw, own, table = frame.raw, frame.observation["own"], generated.table
    cards = tuple(
        PublicCard(i, row["kind"], row.get("state"), raw["catalog"][row["definition_id"]])
        for i, row in enumerate(own["cards"])
    )
    refs = {row["card_ref"]: i for i, row in enumerate(own["cards"])}
    usable = frozenset(
        refs[row["card_ref"]] for row in raw["action_options"]["action_options"]["usable_cards"]
    )
    requires_powder_weapon = actions.action_contract.requires_powder_weapon
    result = []
    for row, kind in enumerate(table.kinds):
        slots = table.slots[table.offsets[row] : table.offsets[row + 1]]
        mask = sum(1 << slot for slot in slots)
        artifacts = sum(1 << slot for slot in slots if not cards[slot].miracle)
        fact = _Fact(mask, len(slots), artifacts.bit_count())
        target = table.targets[row]
        if kind == "play_sequence":
            root = cards[slots[0]]
            family = (
                "utility"
                if root.definition["definition_id"] in UTILITY_IDS
                else "status"
                if root.subclass == "status"
                else "offense"
            )
            analysis = analyze_sequence(
                cards,
                slots,
                family=family,
                mp=own["mp"],
                flash=own["status"]["flash"],
                maximum=actions.max_sequence_length,
                usable=usable,
                rules=rules,
                requires_powder_weapon=requires_powder_weapon,
            )
            if not analysis.complete:
                raise ContractError("refresh sequence lacks complete public analysis")
            renewal = family == "utility" and target == "self"
            fact = replace(
                fact,
                spent=analysis.spent_mp,
                attack=analysis.attack,
                offense=family == "offense" and target == "opponent",
                removed=artifacts if renewal else 0,
                opportunities=len(slots) if renewal else 0,
                uncertain=renewal
                and bool(set(analysis.effects) & {"removeUsedMiracles", "removeItems"}),
            )
        elif kind == "pray":
            fact = replace(fact, opportunities=1, uncertain=len(cards) == 18)
        elif kind == "sacrifice":
            fact = replace(fact, removed=mask, opportunities=len(slots))
        elif kind == "exchange":
            if table.numerics[row][:3] == tuple(own[key] for key in ("hp", "mp", "money")):
                fact = replace(fact, removed=mask, opportunities=1)
        elif kind == "sell":
            price = cards[slots[1]].definition.get("price", 0)
            if target != "self" or price <= own["money"] + own["mp"]:
                fact = replace(
                    fact,
                    removed=(1 << slots[0]) if target == "self" else mask,
                    opportunities=1,
                    spent=max(0, price - own["money"]) if target == "self" else 0,
                )
        elif kind in ("buy", "invoke_phenomenon"):
            fact = replace(
                fact, removed=mask, opportunities=1, uncertain=kind == "invoke_phenomenon"
            )
        result.append(fact)
    return tuple(result)


def _retention(slots: int, facts: tuple[_Fact, ...]) -> npt.NDArray[np.int64]:
    """One <=2MiB principal array; strided views make maxima in place."""
    integer(slots, 0, 18)
    values = np.zeros(1 << slots, dtype=np.int64)
    for fact in facts:
        if fact.offense and fact.attack > 0:
            values[fact.mask] = max(int(values[fact.mask]), fact.attack)
    for bit in range(slots):
        halves = values.reshape(-1, 2, 1 << bit)
        np.maximum(halves[:, 1, :], halves[:, 0, :], out=halves[:, 1, :])
    return values


class RefreshHeuristicPolicy:
    def __init__(
        self, actions: ActionConfigV3, *, rules: CardInstanceRules, options: RefreshOptions
    ) -> None:
        self.label = self.version = refresh_policy_label(actions, rules, options)
        self.actions, self.rules, self.options = actions, rules, options
        self._fallback = CardInstanceBaseline("heuristic", actions, rules=rules)
        self.reset(0)

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed)
        self._last: DecisionFrame | None = None
        self._selection: Selection | None = None
        self._own_turn: int | None = None
        self._decision = 0
        self._samples: tuple[tuple[int, int], ...] = ()  # (prevented, origin decision)
        self._pending: _Pending | None = None

    def select_frame(
        self, frame: DecisionFrame, bank_view: EventBankView, generated: GameGeneratedCandidatesV3
    ) -> Selection:
        if (
            type(generated) is not GameGeneratedCandidatesV3
            or frame.layout is not CORRECTED_CARD_INSTANCE_LAYOUT
            or generated.report.config_hash != digest(asdict(self.actions))
            or generated.report.rule_signature != self._fallback.binding["rule_signature"]
        ):
            raise ContractError("refresh requires the exact corrected game input/config/rules")
        view = frame.bank_view(bank_view)
        generated.table.validate(frame)
        previous = self._last
        if previous is not None:
            if (frame.actor, frame.episode) != (previous.actor, previous.episode):
                raise ContractError("refresh history changed participant/episode without reset")
            view.prefix(previous.boundary)
            if frame.frame_hash == previous.frame_hash:
                if (
                    self._selection is None
                    or generated.table.table_hash != self._selection.table_hash
                ):
                    raise ContractError("repeated refresh frame changed supplied candidates")
                return self._selection
            if (
                frame.revision <= previous.revision
                or frame.interval_start != previous.boundary.end
                or frame.observation["turn"] < previous.observation["turn"]
            ):
                raise ContractError("refresh history frontier/revision/turn did not advance")
        own_turn, decision = self._own_turn, self._decision
        turn, choice = frame.observation["turn"], frame.observation["required_choice"]
        if choice == "turn_action" and turn != own_turn:
            own_turn, decision = turn, decision + 1
        samples, pending = list(self._samples), self._pending
        if previous is not None:
            for event in view.events(previous.boundary.end, frame.boundary.end):
                if pending is None:
                    break
                if event["type"] == "turn_started" and event["turn"] > pending.turn:
                    samples.append((max(0, pending.attack - pending.damage), pending.decision))
                    pending = None
                elif event["type"] == "damage_applied" and event["player"] != frame.actor:
                    pending = replace(pending, damage=pending.damage + max(0, event["amount"]))
        samples = [(blocked, origin) for blocked, origin in samples if decision - origin <= 3]
        if choice == "turn_action":
            facts = _facts(frame, generated, self.actions, self.rules)
            usable = [blocked for blocked, origin in samples if 1 <= decision - origin <= 3]
            count, blocked_sum = len(usable), sum(usable)
            eligible = [
                i
                for i, fact in enumerate(facts)
                if fact.offense
                and fact.attack > 0
                and (
                    fact.attack >= self.options.attack_threshold
                    if not count
                    else count * fact.attack >= 2 * blocked_sum
                )
            ]
            if eligible:
                index = max(
                    eligible,
                    key=lambda i: (
                        count * facts[i].attack - blocked_sum if count else facts[i].attack,
                        -facts[i].spent,
                        -facts[i].consumed,
                        -i,
                    ),
                )
            else:
                index = self._renew_or_fallback(frame, bank_view, generated, facts)
            if facts[index].offense and facts[index].attack > 0:
                pending = _Pending(turn, facts[index].attack, decision)
        else:
            index = self._fallback.select_frame(frame, bank_view, generated).index
        selected = Selection(generated.table.table_hash, index, self.version)
        # Commit only after validation and scoring succeed.
        self._last, self._selection = frame, selected
        self._own_turn, self._decision = own_turn, decision
        self._samples, self._pending = tuple(samples), pending
        return selected

    def _renew_or_fallback(
        self,
        frame: DecisionFrame,
        bank_view: EventBankView,
        generated: GameGeneratedCandidatesV3,
        facts: tuple[_Fact, ...],
    ) -> int:
        possession = len(frame.observation["own"]["cards"])
        remaining = (1 << possession) - 1
        retention = _retention(possession, facts)
        renewals = []
        for i, fact in enumerate(facts):
            freed = fact.removed.bit_count()
            if generated.table.kinds[i] == "pray" and possession == 18:
                freed = 1
            opportunities = min(fact.opportunities, max(0, 18 - possession + freed))
            if opportunities > 0:
                retained = 0 if fact.uncertain else int(retention[remaining ^ fact.removed])
                renewals.append(
                    (
                        min(retained, self.options.attack_threshold),
                        opportunities,
                        -fact.spent,
                        -fact.selected,
                        -i,
                    )
                )
        if renewals:
            return -max(renewals)[-1]
        if possession == 18:
            discards = [i for i, kind in enumerate(generated.table.kinds) if kind == "discard"]
            if discards:
                return max(
                    discards,
                    key=lambda i: (
                        min(
                            int(retention[remaining ^ facts[i].mask]), self.options.attack_threshold
                        ),
                        -facts[i].selected,
                        -i,
                    ),
                )
        offense = [i for i, fact in enumerate(facts) if fact.offense]
        if offense:
            return min(
                offense, key=lambda i: (facts[i].attack, facts[i].spent, facts[i].selected, i)
            )
        return self._fallback.select_frame(frame, bank_view, generated).index
