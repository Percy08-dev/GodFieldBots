"""Raw ownership, prefix integrity, exact typed reconstruction and visibility."""
from dataclasses import FrozenInstanceError, replace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, EventDelta, Selection


def frame_for(value, episode="one"):
    bank = EventBank(value.observation["viewer_player_id"], episode)
    bank.append(EventDelta(bank.actor, episode, 0, bank.boundary.prefix_hash, value.events))
    return DecisionFrame.from_input(value, bank, 0), bank


def test_frame_owned_snapshot_and_no_future_events(make_input):
    value = make_input(hand=("bronze-club",), learned=("fireball",))
    event = {"type": "artifact_used", "player": 2, "definition_id": "bronze-club"}
    value = replace(value, events=(event,))
    frame, bank = frame_for(value)
    frozen = frame.raw
    event["definition_id"] = "angel-armor"
    value.observation["own"]["hand"][0]["definition_id"] = "angel-armor"
    assert frame.raw == frozen
    exposed = frame.observation
    exposed["own"]["hp"] = -100
    assert frame.observation["own"]["hp"] == 40
    view = frame.bank_view(bank.view())
    bank.append(EventDelta(1, "one", 1, bank.boundary.prefix_hash, [{"type": "player_defeated", "player": 2}]))
    assert len(view.events()) == 1
    assert frame.bank_view(bank.view()).events() == view.events()
    with pytest.raises(FrozenInstanceError):
        frame.revision = 4


def test_delta_chain_continuity_owner_and_caps(make_input):
    bank = EventBank(1, "one", max_events=1)
    event = {"type": "action_accepted", "player": 1}
    for delta in (EventDelta(2, "one", 0, bank.boundary.prefix_hash, [event]),
                  EventDelta(1, "other", 0, bank.boundary.prefix_hash, [event]),
                  EventDelta(1, "one", 1, bank.boundary.prefix_hash, [event]),
                  EventDelta(1, "one", 0, "0" * 64, [event])):
        with pytest.raises(ContractError):
            bank.append(delta)
    bank.append(EventDelta(1, "one", 0, bank.boundary.prefix_hash, [event]))
    with pytest.raises(ContractError):
        bank.append(EventDelta(1, "one", 1, bank.boundary.prefix_hash, [event]))
    with pytest.raises(ContractError):
        replace(bank.view(), _prefixes=(bank.view()._prefixes[0], "0" * 64))


def test_decision_intervals_and_history_are_exact(make_input):
    value = make_input()
    frame, bank = frame_for(value)
    with pytest.raises(ContractError):
        DecisionFrame.from_input(value, bank, 0)
    event = {"type": "action_accepted", "player": 1}
    bank.append(EventDelta(1, "one", 0, bank.boundary.prefix_hash, [event]))
    obs = frame.observation
    obs["state_revision"] = 1
    options = value.to_dict()["action_options"]
    options["state_revision"] = 1
    current = replace(value, observation=obs, events=(event,), action_options=options)
    second = DecisionFrame.from_input(current, bank, frame.boundary.end)
    assert second.interval_start == 0 and second.boundary.end == 1
    with pytest.raises(ContractError):
        second.bank_view(EventBank(2, "one").view())
    with pytest.raises(ContractError):
        replace(second, payload=frame.payload)
    with pytest.raises(ContractError):
        replace(second, frame_hash="0" * 64)


@pytest.mark.parametrize("target", ["omitted", None, 1, 2])
def test_ordered_reference_target_presence_and_selection(make_input, target):
    frame, _ = frame_for(make_input(hand=("bronze-club", "bronze-club"), learned=("fireball",)))
    action = {"type": "play_sequence", "ordered_elements": [frame.references[1], frame.references[2], frame.references[0]]}
    if target != "omitted":
        action["target"] = target
    table = CandidateTable.from_actions(frame, [action])
    assert table.slots == (1, 2, 0)
    assert table.to_action(0, frame) == action
    selection = Selection(table.table_hash, 0, "policy-v2", 0., .5)
    assert selection.validate(table, frame) == action
    with pytest.raises(ContractError):
        replace(selection, table_hash="0" * 64).validate(table, frame)
    with pytest.raises(ContractError):
        replace(table, table_hash="0" * 64)
    other, _ = frame_for(make_input(), "other")
    with pytest.raises(ContractError):
        table.to_action(0, other)


def test_compact_carriers_and_numeric_presence(make_input):
    frame, _ = frame_for(make_input(hand=("exchange", "sell", "bronze-club")))
    actions = [{"type": "exchange", "artifact": 1, "hp": 0, "mp": 10, "money": 60},
               {"type": "sell", "artifact": 2, "sold_artifact": 3, "target": 1}]
    table = CandidateTable.from_actions(frame, actions)
    assert table.numerics == ((0, 10, 60, 0), (0, 0, 0, 0))
    assert table.presence[0] == (True, True, True, False)
    assert [table.to_action(i, frame) for i in range(2)] == actions
    assert table.offsets == (0, 1, 3)
    foreign = {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 999}]}
    with pytest.raises(ContractError):
        CandidateTable.from_actions(frame, [foreign])
    fields = table._content()
    fields["slots"] = (17, 1, 2)
    forged = CandidateTable(**fields, table_hash=digest(fields))
    with pytest.raises(ContractError):
        forged.validate(frame)


def test_position_128_is_syntax_capacity_not_legality_oracle(make_input):
    frame, _ = frame_for(make_input(learned=("fireball",)))
    action = {"type": "play_sequence", "ordered_elements": [frame.references[-1]] * 128}
    table = CandidateTable.from_actions(frame, [action])
    assert len(table.slots) == 128
    assert table.to_action(0, frame) == action
    with pytest.raises(ContractError):
        CandidateTable.from_actions(frame, [action], max_positions=18)


@pytest.mark.parametrize("action", [
    {"type": "play_sequence", "ordered_elements": None},
    {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": True}]},
    {"type": "pray", "target": 2},
    {"type": "discard", "instance_ids": [False]},
    {"type": "exchange", "artifact": 1, "hp": 1, "mp": -1, "money": 1},
])
def test_malformed_typed_actions_reject(make_input, action):
    frame, _ = frame_for(make_input())
    with pytest.raises(ContractError):
        CandidateTable.from_actions(frame, [action])


def test_fog_payload_and_buy_numeric_encoding_fail_closed(make_input):
    value = make_input()
    obs = value.to_dict()["observation"]
    obs["own"]["status"]["fog"] = True
    with pytest.raises(ContractError, match="Fog"):
        frame_for(replace(value, observation=obs))
    frame, _ = frame_for(make_input(choice="buy_accept_reject"))
    table = CandidateTable.from_actions(frame, [{"type": "buy_decision", "accept": True}])
    fields = table._content()
    fields["numerics"] = ((0, 0, 0, 2),)
    forged = CandidateTable(**fields, table_hash=digest(fields))
    with pytest.raises(ContractError):
        forged.to_action(0, frame)


def test_all_18_own_slots_and_overflow_are_explicit(make_input):
    frame, _ = frame_for(make_input(hand=("bronze-club",) * 18))
    table = CandidateTable.from_actions(frame, [{"type": "play_sequence", "ordered_elements": list(frame.references)}])
    assert table.slots == tuple(range(18))
    with pytest.raises(ContractError, match="shape"):
        frame_for(make_input(hand=("bronze-club",) * 19))
