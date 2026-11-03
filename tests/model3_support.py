"""Small original-card-reference numerical fixtures; no Source/learning authority."""

import copy
from dataclasses import asdict, replace

from gf_common.card_instance_rules import card_instance_rules
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.structured_vocab import GUARDIANS
from gf_models.structured_config import model_config_v3
from tests.card_instance_action_support import inputs
from tests.test_structured_network import small_model


def model3(hand="attention", temporal="mlp", *, layout=LAYOUT):
    raw = asdict(small_model(hand=hand, temporal=temporal))
    raw.update(version="3", input_recipe=layout.input_recipe)
    raw["event"].update(aggregate_recipe=layout.aggregate_recipe, max_interval=8, max_episode=128)
    raw["temporal"].update(max_decisions=8, sequence_length=2)
    return model_config_v3(raw, layout=layout)


def raw_inputs(*, count=3, empty=False, events=True, layout=LAYOUT):
    value = inputs(
        []
        if empty
        else [
            ("bronze-club", None),
            ("steel-armor", None),
            ("fireball", "unused"),
            ("aura", "learned"),
            ("fireball", "learned"),
        ],
        can_pray=empty,
    )
    bank = EventBank(1, "model3-component", layout=layout)
    frames, tokens, history = [], [], []
    for index in range(count):
        start = bank.boundary.end
        if events:
            event = {
                "type": "resource_changed",
                "player": 1,
                "hp_delta": -index,
                "mp_delta": 1,
                "money_delta": 0,
            }
            incoming = [event]
            if layout is LAYOUT:
                family = ("guardian_set", "guardian_acted", "guardian_left")[index % 3]
                incoming.append({"type": family, "player": 1 + index % 2,
                    "definition_id": "seafood-soup" if family == "guardian_acted"
                    else GUARDIANS[index % len(GUARDIANS)]})
            bank.append(
                event_delta(
                    1, bank.episode, start, bank.boundary.prefix_hash, incoming, layout=layout
                )
            )
            history.extend(incoming)
        current = copy.deepcopy(value)
        current.observation["state_revision"] = index
        current.observation["own"]["hp"] -= index
        current.action_options["state_revision"] = index
        current.action_options["action_options"]["state_revision"] = index
        current = replace(current, events=tuple(history))
        frame = DecisionFrame.from_input(current, bank, start, layout=layout)
        frames.append(frame)
        tokens.append(tokenize_frame(frame, bank.view(), layout=layout))
    refs = frames[-1].references
    actions = (
        [{"type": "pray"}]
        if empty
        else [
            {"type": "play_sequence", "ordered_card_refs": [refs[0]], "target": 2},
            {"type": "play_sequence", "ordered_card_refs": [refs[2], refs[4]], "target": 1},
            {
                "type": "play_sequence",
                "ordered_card_refs": [refs[0], refs[2], refs[3]],
                "target": 2,
            },
            {"type": "discard", "card_refs": [refs[1]]},
        ]
    )
    table = CandidateTable.from_actions(
        frames[-1],
        actions,
        action_contract=layout.action_contract,
        purpose="component",
        synthetic_capacity=False,
    )
    candidates = tokenize_table(table, frames[-1], 18, layout=layout)
    return tuple(tokens), candidates, frames, bank, table


def context(*, layout=LAYOUT, action_hash=None, synthetic=False):
    from gf_common.domain import digest

    return {
        "layout": layout,
        "action_contract": layout.action_contract,
        "rules": card_instance_rules(),
        "purpose": "component",
        "synthetic_capacity": synthetic,
        "action_config_hash": digest({"component": "model3-actions"})
        if action_hash is None
        else action_hash,
    }
