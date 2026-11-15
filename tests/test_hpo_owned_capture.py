"""Capture accepts live ACKs and complete own histories, never public replay."""
from __future__ import annotations

import copy
import gc
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.card_instance import corrected_card_instance_contract
from gf_common.domain import ContractError, EpisodeGroup, EpisodeResult, EpisodeSpec, GroupResult
from gf_common.group_pack import StorageSettings
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, Selection, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_training.collection import CollectionPlan
from gf_training.group_pack_resources import StorageBudget
from gf_training.hpo_owned_capture import _new_capture
from tests.card_instance_action_support import inputs
from tests.test_corrected_card_instance import records


def setup_capture(*, names=("first",), seats=None):
    episodes = tuple(EpisodeSpec(name, ("same", "same"), (1, 2), "pair", (seats or {}).get(name, 1)) for name in names)
    group = EpisodeGroup("seed", episodes)
    plan = CollectionPlan((group,), {}, Path("unused"), {}, float("inf"), float("inf"))
    owner, snapshot = object(), object()
    account = StorageBudget(StorageSettings(), time.monotonic() + 30)
    capture = _new_capture(owner, run_id="run", plan=plan, generation=3,
        snapshot_owner=snapshot, learner_seats=seats or {name: 1 for name in names},
        learner_policy_version="actual-learner", budget=account)
    return capture, owner, snapshot, plan, group, account


def decision(*, revision=0, events=()):
    value = replace(inputs([], can_pray=True), events=tuple(events))
    value.observation["state_revision"] = revision
    value.action_options["state_revision"] = revision
    value.action_options["action_options"]["state_revision"] = revision
    frame, _bank, delta = records(value)
    table = CandidateTable.from_actions(frame, [{"type": "pray"}],
        action_contract=corrected_card_instance_contract(), purpose="qualified_game", synthetic_capacity=False)
    selection = Selection(table.table_hash, 0, "actual-learner", -0.123456789123, 0.234567891234)
    command = {"command_id": str(uuid.uuid4()), "expected_revision": revision, "action": {"type": "pray"}}
    ack = {"command_id": command["command_id"], "previous_revision": revision,
        "state_revision": revision + 1, "status": "terminal", "result": "draw"}
    return frame, delta, table, selection, command, ack


def terminal(viewer, revision=1):
    obs = inputs([], can_pray=True).observation
    obs.update(viewer_player_id=viewer, state_revision=revision, current_player=None,
        actionable=False, typed_action_schema=[], phase="terminal", required_choice=None, result="draw")
    obs["own"]["player_id"], obs["opponent"]["player_id"] = viewer, 3 - viewer
    return obs


def finish(capture, name, *, decisions=1, health=None, outcome="draw", end_kind="natural"):
    for viewer in (1, 2):
        capture.terminal_observation(name, viewer, terminal(viewer, decisions))
    result = EpisodeResult(name, "pair", "seed", ("same", "same"), game_id="game-" + name,
        outcome=outcome, end_kind=end_kind, decisions=decisions, health=health or {})
    capture.close_episode(name, result)
    return result


def test_capture_uses_original_selection_and_ack_once_preserving_immutable_aliases():
    capture, owner, snapshot, plan, group, account = setup_capture()
    capture.open_episode("first", "game-first")
    frame, delta, table, selection, command, ack = decision()
    capture.begin_decision("first", 1, frame, delta, table, selection, command)
    capture.accept_decision("first", 1, command["command_id"], ack)
    capture.accept_decision("first", 1, command["command_id"], copy.deepcopy(ack))
    command["action"]["type"] = "changed-after-act"
    ack["result"] = "changed-after-ack"
    result = finish(capture, "first")
    capture.close_group(group, GroupResult([result]))
    batch = capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    own = batch.episodes[0].learner
    assert len(own.decisions) == 1
    assert own.decisions[0].selection is selection
    assert own.decisions[0].selection.logprob == -0.123456789123
    assert own.decisions[0].command["action"] == {"type": "pray"}
    assert own.decisions[0].acknowledgement["result"] == "draw"
    assert own.history.actor == 1
    assert batch.generation == 3 and batch.origin == "producer-owned-v1"
    assert not batch.episodes[0].replay_reconciled
    with pytest.raises(ContractError):
        capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    batch.close()
    assert account.used > 0  # exported own histories still own retention
    del own, frame, delta, table, selection
    gc.collect()
    assert account.used == 0


def test_zero_act_population_keeps_plan_order_and_explicit_learner_seat():
    capture, owner, snapshot, plan, group, _ = setup_capture(names=("first", "second"), seats={"first": 2, "second": 1})
    results = {}
    for name in ("second", "first"):
        capture.open_episode(name, "game-" + name)
        results[name] = finish(capture, name, decisions=0)
    with pytest.raises(ContractError):
        capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    capture.close_group(group, GroupResult([results["first"], results["second"]]))
    batch = capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    assert [e.episode_id for e in batch.episodes] == ["first", "second"]
    assert [e.learner.viewer for e in batch.episodes] == [2, 1]
    assert [len(e.learner.decisions) for e in batch.episodes] == [0, 0]
    batch.close()


@pytest.mark.parametrize("fault", ["pending", "rejected", "wrong_command", "stale_revision", "bool_revision", "conflicting_duplicate", "wrong_viewer", "wrong_model"])
def test_bad_or_unresolved_command_never_seals(fault):
    capture, owner, snapshot, plan, group, _ = setup_capture()
    capture.open_episode("first", "game-first")
    frame, delta, table, selection, command, ack = decision()
    if fault == "wrong_model":
        selection = replace(selection, policy_version="opponent-snapshot")
    if fault in ("wrong_viewer", "wrong_model"):
        with pytest.raises(ContractError):
            capture.begin_decision("first", 2 if fault == "wrong_viewer" else 1,
                frame, delta, table, selection, command)
    else:
        capture.begin_decision("first", 1, frame, delta, table, selection, command)
        if fault != "pending":
            if fault == "wrong_command":
                ack["command_id"] = str(uuid.uuid4())
            if fault == "stale_revision":
                ack["previous_revision"] = 1
            if fault == "bool_revision":
                ack["previous_revision"] = False
            if fault == "rejected":
                ack["status"] = "rejected"
            if fault == "conflicting_duplicate":
                capture.accept_decision("first", 1, command["command_id"], ack)
                ack["state_revision"] = 2
            with pytest.raises(ContractError):
                capture.accept_decision("first", 1, command["command_id"], ack)
    try:
        result = finish(capture, "first")
        capture.close_group(group, GroupResult([result]))
    except ContractError:
        pass
    with pytest.raises(ContractError):
        capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    capture.close()


@pytest.mark.parametrize("fault", ["missing_terminal", "terminal_mismatch", "worker_cleanup", "group_cleanup", "truncated", "wrong_generation_owner", "wrong_snapshot", "wrong_plan"])
def test_terminal_and_actual_cleanup_gate_seal(fault):
    capture, owner, snapshot, plan, group, _ = setup_capture()
    capture.open_episode("first", "game-first")
    capture.terminal_observation("first", 1, terminal(1, 0))
    if fault != "missing_terminal":
        capture.terminal_observation("first", 2, terminal(2, 4 if fault == "terminal_mismatch" else 0))
    result = EpisodeResult("first", "pair", "seed", ("same", "same"), game_id="game-first",
        end_kind="truncated" if fault == "truncated" else "natural", outcome="draw",
        health={"worker_cleanup_failure": "failed"} if fault == "worker_cleanup" else {})
    try:
        capture.close_episode("first", result)
        capture.close_group(group, GroupResult([result], {"client_close_failure": "failed"} if fault == "group_cleanup" else {}))
        capture.seal(object() if fault == "wrong_generation_owner" else owner,
            plan=replace(plan) if fault == "wrong_plan" else plan,
            snapshot_owner=object() if fault == "wrong_snapshot" else snapshot)
    except ContractError:
        pass
    else:
        pytest.fail("bad terminal/cleanup/owner accepted")
    capture.close()


def test_full_own_event_prefixes_remain_complete_without_future_events():
    capture, owner, snapshot, plan, group, _ = setup_capture()
    capture.open_episode("first", "game-first")
    bank = EventBank(1, "actual-bank", layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    accepted = []
    for revision in (0, 1):
        value = inputs([], can_pray=True)
        value.observation["state_revision"] = revision
        value.action_options["state_revision"] = revision
        value.action_options["action_options"]["state_revision"] = revision
        before = bank.boundary
        event = {"type": "artifact_used", "player": 1, "definition_id": "bronze-club"}
        delta = event_delta(1, bank.episode, before.end, before.prefix_hash, [event], layout=bank.layout)
        bank.append(delta)
        value = replace(value, events=tuple(bank.view().events()))
        frame = DecisionFrame.from_input(value, bank, before.end, layout=bank.layout)
        table = CandidateTable.from_actions(frame, [{"type": "pray"}], action_contract=corrected_card_instance_contract(),
            purpose="qualified_game", synthetic_capacity=False)
        selection = Selection(table.table_hash, 0, "actual-learner", -0.1, 0.25)
        command = {"command_id": str(uuid.uuid4()), "expected_revision": revision, "action": {"type": "pray"}}
        ack = {"command_id": command["command_id"], "previous_revision": revision,
            "state_revision": revision + 1, "status": "terminal" if revision else "active", "result": "draw" if revision else None}
        capture.begin_decision("first", 1, frame, delta, table, selection, command)
        capture.accept_decision("first", 1, command["command_id"], ack)
        accepted.append(frame)
    result = finish(capture, "first", decisions=2)
    capture.close_group(group, GroupResult([result]))
    batch = capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    learner = batch.episodes[0].learner
    assert len(learner.history.events()) == 2
    assert len(accepted[0].bank_view(learner.history.prefix(accepted[0].boundary)).events()) == 1
    assert len(accepted[1].bank_view(learner.history).events()) == 2
    assert not batch.episodes[0].participants[1].decisions
    batch.close()


def test_unexplained_revision_gap_cannot_enter_capture():
    capture, *_ = setup_capture()
    capture.open_episode("first", "game-first")
    frame, delta, table, selection, command, ack = decision()
    ack.update(status="active", result=None)
    capture.begin_decision("first", 1, frame, delta, table, selection, command)
    capture.accept_decision("first", 1, command["command_id"], ack)
    frame, delta, table, selection, command, _ = decision(revision=2)
    with pytest.raises(ContractError):
        capture.begin_decision("first", 1, frame, delta, table, selection, command)
    capture.close()


def test_win_control_preserves_original_natural_outcome_and_learner_perspective():
    capture, owner, snapshot, plan, group, _ = setup_capture()
    capture.open_episode("first", "game-first")
    outcome = {"win": {"winner": 2}}
    for viewer in (1, 2):
        obs = terminal(viewer, 0)
        obs["result"] = copy.deepcopy(outcome)
        capture.terminal_observation("first", viewer, obs)
    result = EpisodeResult("first", "pair", "seed", ("same", "same"), game_id="game-first",
        outcome=outcome, end_kind="natural")
    capture.close_episode("first", result)
    capture.close_group(group, GroupResult([result]))
    batch = capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    outcome["win"]["winner"] = 1
    assert batch.episodes[0].outcome == {"win": {"winner": 2}}
    assert batch.episodes[0].learner.terminal_observation["result"] == {"win": {"winner": 2}}
    batch.close()


def test_late_group_failure_after_callback_still_prevents_final_admission():
    capture, owner, snapshot, plan, group, _ = setup_capture()
    capture.open_episode("first", "game-first")
    result = finish(capture, "first", decisions=0)
    group_result = GroupResult([result])
    capture.close_group(group, group_result)
    group_result.health["client_close_failure"] = "late failure"
    with pytest.raises(ContractError):
        capture.seal(owner, plan=plan, snapshot_owner=snapshot)
    capture.close()


@pytest.mark.parametrize("depth", [1, 2])
def test_derived_prefix_alone_retains_capture_charge_until_last_use(depth):
    event = {"type": "artifact_used", "player": 1, "definition_id": "bronze-club"}
    def prefix_only():
        capture, owner, snapshot, plan, group, account = setup_capture()
        capture.open_episode("first", "game-first")
        frame, delta, table, selection, command, ack = decision(events=(event,))
        capture.begin_decision("first", 1, frame, delta, table, selection, command)
        capture.accept_decision("first", 1, command["command_id"], ack)
        result = finish(capture, "first")
        capture.close_group(group, GroupResult([result]))
        batch = capture.seal(owner, plan=plan, snapshot_owner=snapshot)
        prefix = batch.episodes[0].learner.history
        for _ in range(depth):
            prefix = prefix.prefix(prefix.boundary)
        batch.close()
        return account, prefix
    account, prefix = prefix_only()
    gc.collect()
    assert account.used > 0
    assert prefix.events() == (event,)
    assert prefix.layout is CORRECTED_CARD_INSTANCE_LAYOUT
    del prefix
    gc.collect()
    assert account.used == 0
