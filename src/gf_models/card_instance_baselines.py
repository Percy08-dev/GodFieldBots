"""Corrected baseline policies; input contains only one participant's public facts."""

from __future__ import annotations

import random
from dataclasses import asdict

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import (
    UTILITY_IDS,
    CardInstanceRules,
    PublicCard,
    analyze_sequence,
    rule_signature,
)
from gf_common.card_instance_runtime import game_binding
from gf_common.diverse_actions import GameGeneratedCandidatesV3
from gf_common.domain import ContractError, digest, integer
from gf_common.structured import DecisionFrame, EventBankView, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT


class CardInstanceBaseline:
    def __init__(self, kind: str, actions: ActionConfigV3, *, rules: CardInstanceRules) -> None:
        self.binding = game_binding(
            actions.action_contract, rule_signature(actions.action_contract, rules)
        )
        if kind not in ("random", "heuristic") or actions.max_sequence_length > 18:
            raise ContractError("corrected baseline kind/game capacity required")
        self.actions, self.rules, self.kind = actions, rules, kind
        self.version = kind + "-card-instance-v2"
        self.rng = random.Random(0)

    def reset(self, sampling_seed: int) -> None:
        integer(sampling_seed)
        self.rng.seed(sampling_seed)

    def select_frame(
        self, frame: DecisionFrame, bank_view: EventBankView, generated: GameGeneratedCandidatesV3
    ) -> Selection:
        if (
            type(generated) is not GameGeneratedCandidatesV3
            or frame.layout is not CORRECTED_CARD_INSTANCE_LAYOUT
            or generated.report.config_hash != digest(asdict(self.actions))
            or generated.report.rule_signature != self.binding["rule_signature"]
        ):
            raise ContractError("corrected baseline requires the exact game input/config/rules")
        table = generated.table
        frame.bank_view(bank_view)
        table.validate(frame)
        if self.kind == "random":
            index = self.rng.randrange(len(table.kinds))
        else:
            raw, own = frame.raw, frame.observation["own"]
            cards = tuple(
                PublicCard(
                    index, card["kind"], card.get("state"), raw["catalog"][card["definition_id"]]
                )
                for index, card in enumerate(own["cards"])
            )
            refs = {card["card_ref"]: index for index, card in enumerate(own["cards"])}
            hints = raw["action_options"]["action_options"]
            usable = frozenset(refs[row["card_ref"]] for row in hints["usable_cards"])
            scores = []
            for row, kind in enumerate(table.kinds):
                selected = table.slots[table.offsets[row] : table.offsets[row + 1]]
                chosen = [cards[slot] for slot in selected]
                if kind == "play_sequence" or kind == "respond" and chosen:
                    root = chosen[0]
                    family = (
                        "defense"
                        if kind == "respond"
                        else (
                            "utility"
                            if root.definition["definition_id"] in UTILITY_IDS
                            else ("status" if root.subclass == "status" else "offense")
                        )
                    )
                    facts = analyze_sequence(
                        cards,
                        selected,
                        family=family,
                        mp=own["mp"],
                        flash=own["status"].get("flash", False),
                        maximum=self.actions.max_sequence_length,
                        usable=usable,
                    )
                    if not facts.complete:
                        raise ContractError(
                            "generated baseline sequence lacks complete public analysis"
                        )
                    effective = [cards[slot] for slot in facts.effective_slots]
                    if kind == "play_sequence":
                        score = float(facts.attack)
                        if facts.attribute == "darkness":
                            score += 6
                        if table.targets[row] == "self":
                            score = -score
                    else:
                        score = float(
                            sum(max(0, card.definition.get("defense", 0)) for card in effective)
                        )
                    score -= 0.05 * len(selected)
                elif kind == "pray":
                    score = 2.0
                elif kind == "sacrifice":
                    score = 1 + 0.1 * len(selected)
                elif kind == "discard":
                    score = -1 - 0.05 * sum(card.definition.get("price", 0) for card in chosen)
                elif kind == "buy_decision":
                    score = float(table.numerics[row][3])
                elif kind == "exchange":
                    score = 0.2 * (table.numerics[row][0] - own["hp"])
                else:
                    score = 0.0
                scores.append(score)
            index = max(range(len(scores)), key=scores.__getitem__)  # fixed first-in-table ties.
        return Selection(table.table_hash, index, self.version)
