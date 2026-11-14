"""Corrected public-only frames and explicitly supplied candidate tables."""

import random
from dataclasses import replace

from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import (
    GameGeneratedCandidatesV3,
    NumericCosts,
    PublicCandidateGeneratorV3,
)
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from tests.card_instance_action_support import inputs

ACTIONS = ActionConfigV3(
    action_contract=corrected_card_instance_contract(),
    max_candidates=64,
    max_expansions=128,
    max_temporary_candidates=128,
)
RULES = card_instance_rules()


def play(*slots, target=2):
    return lambda frame: {
        "type": "play_sequence",
        "ordered_card_refs": [frame.references[slot] for slot in slots],
        "target": target,
    }


def remove(*slots, kind="discard"):
    return lambda frame: {"type": kind, "card_refs": [frame.references[i] for i in slots]}


def trade(kind, carrier, *, sold=None, target=1, allocation=None):
    def action(frame):
        row = {"type": kind, "card_ref": frame.references[carrier]}
        if kind in ("buy", "sell"):
            row["target"] = target
        if sold is not None:
            row["sold_card_ref"] = frame.references[sold]
        if allocation is not None:
            row.update(zip(("hp", "mp", "money"), allocation, strict=True))
        return row

    return action


class Episode:
    def __init__(self, name="refresh-test"):
        self.bank = EventBank(1, name, layout=LAYOUT)
        # Obtain an authentic report/config binding without running another
        # generator for any candidate-selection or retention assertion.
        value = inputs([], can_pray=True)
        frame = DecisionFrame.from_input(value, self.bank, 0, layout=LAYOUT)
        self.report = (
            PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0)
            .generate_game(frame, self.bank.view(), random.Random(0), NumericCosts())
            .report
        )
        self.bank = EventBank(1, name, layout=LAYOUT)
        self.revision = -1
        self.end = 0

    def frame(
        self,
        spec,
        candidates,
        *,
        turn=1,
        events=(),
        mp=50,
        money=20,
        choice="turn_action",
        end_times=False,
        usable=None,
    ):
        value = inputs(
            spec, mp=mp, choice=choice, end_times=end_times, usable=usable, can_pray=True
        )
        self.revision += 1
        value.observation.update(state_revision=self.revision, turn=turn)
        value.observation["own"]["money"] = money
        value.action_options["state_revision"] = self.revision
        value.action_options["action_options"]["state_revision"] = self.revision
        self.bank.append(
            event_delta(
                1,
                self.bank.episode,
                self.bank.boundary.end,
                self.bank.boundary.prefix_hash,
                events,
                layout=LAYOUT,
            )
        )
        value = replace(value, events=self.bank.view().events())
        frame = DecisionFrame.from_input(value, self.bank, self.end, layout=LAYOUT)
        self.end = frame.boundary.end
        table = CandidateTable.from_actions(
            frame,
            [row(frame) if callable(row) else row for row in candidates],
            action_contract=ACTIONS.action_contract,
            purpose="qualified_game",
            synthetic_capacity=False,
        )
        report = replace(
            self.report,
            table_hash=table.table_hash,
            mandatory=0,
            selected=len(table.kinds),
            selected_elements=len(table.slots),
            discovered=len(table.kinds),
            channel_counts=(len(table.kinds), 0, 0, 0, 0, 0, 0),
            traversal_complete=True,
            presentation_complete=True,
        )
        return frame, self.bank.view(), GameGeneratedCandidatesV3(table, report)


def next_turn(turn):
    return {"type": "turn_started", "turn": turn, "active_player": 2}


def damage(amount, player=2):
    return {"type": "damage_applied", "player": player, "amount": amount}
