"""Participant-only public fixtures for the owner-confirmed pure generator."""

import random
from dataclasses import replace

from gf_common.card_instance import ActionConfigV3, card_instance_contract
from gf_common.card_instance_rules import PublicCard, analyze_sequence, card_instance_rules
from gf_common.card_instance_validation import action_descriptors
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV3
from tests.test_card_instance_structured import frame_for, make_input


def inputs(
    spec,
    *,
    mp=50,
    choice="turn_action",
    flash=False,
    end_times=False,
    usable=None,
    targets=(1, 2),
    can_pray=False,
    discard=True,
):
    value = make_input(spec)
    obs = value.observation
    obs["own"]["mp"] = mp
    obs["own"]["status"]["flash"] = flash
    obs.update(
        required_choice=choice,
        end_times=end_times,
        typed_action_schema=action_descriptors(choice),
        phase="awaiting_response"
        if choice == "defense"
        else "resolving"
        if choice == "buy_accept_reject"
        else "awaiting_action",
    )
    body = value.action_options["action_options"]
    body.update(
        required_choice=choice,
        valid_targets=list(targets),
        can_pray=can_pray,
        can_allow=choice == "defense",
        can_discard=discard and choice == "turn_action" and not end_times,
        can_sacrifice=discard and choice == "turn_action" and end_times,
    )
    # Explicit operation fields are independent of the flat direct-use mask.
    for operation in ("buy", "sell", "exchange", "phenomenon"):
        ability = "callPhenomenon" if operation == "phenomenon" else operation
        body[operation + "_cards"] = [
            {"card_ref": card["card_ref"], "definition_id": card["definition_id"]}
            for card in obs["own"]["cards"]
            if value.catalog[card["definition_id"]].get("ability") == ability
        ]
    if usable is not None:
        body["usable_cards"] = [
            row for index, row in enumerate(body["usable_cards"]) if index in usable
        ]
    body["candidate_groups"] = {"uninterpreted": list(body["usable_cards"])}
    return replace(value, typed_metadata=tuple(obs["typed_action_schema"]))


def run(value, *, seed=17, clock=lambda: 0.0, costs=None, **settings):
    frame, bank, _ = frame_for(value)
    config = ActionConfigV3(action_contract=card_instance_contract(), **settings)
    result = PublicCandidateGeneratorV3(config, rules=card_instance_rules(), clock=clock).generate(
        frame, bank.view(), random.Random(seed), NumericCosts() if costs is None else costs
    )
    result.table.validate(frame)
    actions = [result.table.to_action(index, frame) for index in range(len(result.table.kinds))]
    return result, actions, frame


def facts(spec, refs=None, *, family="offense", mp=50, flash=False, maximum=18, usable=None):
    value = inputs(spec, mp=mp, flash=flash)
    cards = tuple(
        PublicCard(index, card["kind"], card.get("state"), value.catalog[card["definition_id"]])
        for index, card in enumerate(value.observation["own"]["cards"])
    )
    return analyze_sequence(
        cards,
        tuple(range(len(cards))) if refs is None else refs,
        family=family,
        mp=mp,
        flash=flash,
        maximum=maximum,
        usable=frozenset(range(len(cards))) if usable is None else frozenset(usable),
    )
