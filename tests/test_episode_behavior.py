import asyncio
import json

import pytest

from gf_training.journal import ParticipantJournal
from tests.episode_support import (
    AuditedWorkers,
    PublicScript,
    check_persisted_result,
    persisted,
    play,
)


@pytest.mark.parametrize("outcome", ["draw", {"win": {"winner": 2}}])
def test_whole_episode_natural_end_two_private_routes_and_accepted_transitions(tmp_path, monkeypatch, definitions, outcome):
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(outcome=outcome)
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    saved, journals = persisted(tmp_path, spec)
    check_persisted_result(result, saved)
    assert result.end_kind == "natural" and result.outcome == outcome and result.decisions == 4
    assert result.first_player == 2 and result.replay_reconciled and not result.health
    assert script.observations[:3] == [(1, 0), (2, 0), (2, 1)]  # nonactor read then actor's own view
    assert script.event_calls == [(2, None), (1, None), (2, script.cursor(2, 1)), (1, script.cursor(1, 1))]
    assert script.deleted and all(worker.closed for worker in workers.instances)
    for viewer, worker in enumerate(workers.instances, 1):
        own_definition = "bronze-club" if viewer == 1 else "god-sword"
        assert len(worker.inputs) == 2
        for index, value in enumerate(worker.inputs):
            assert value["observation"]["viewer_player_id"] == viewer
            assert value["observation"]["own"]["hand"] == [{"instance_id": script.hand_id(viewer), "definition_id": own_definition}]
            assert "hand" not in value["observation"]["opponent"]
            assert set(value["catalog"]) == {own_definition}
            assert value["history_end"] == script.cursor(viewer, index + 1)
            turns = [event["turn"] for event in value["events"] if event["type"] == "turn_started"]
            assert len(turns) == index + 1 and len(set(turns)) == len(turns)
            own_private_events = [event for event in value["events"] if event["type"] == "artifact_drawn"]
            public_opponent_events = [event for event in value["events"] if event["type"] == "artifact_used"]
            assert len(value["events"]) == 3 * (index + 1)
            assert len(own_private_events) == index + 1 and all(event["player"] == viewer for event in own_private_events)
            assert public_opponent_events == [{"type": "artifact_used", "player": 3 - viewer, "definition_id": "sun-amulet"}] * (index + 1)
            assert not any(marker in json.dumps(value) for marker in ("forbidden_ack", "forbidden_options", "secret_rejection", "seed_fingerprint", "game_seed", "simulator_abort_reason"))
        records = journals[viewer]
        assert [r["kind"] for r in records] == ["header", "decision", "acknowledgement", "transition", "decision", "acknowledgement", "transition", "end"]
        assert records[3] == {"kind": "transition", "decision": 0, "next_decision": 1, "reward": 0, "terminal": False}
        expected = 0 if outcome == "draw" else 1 if viewer == 2 else -1
        assert records[-2]["terminal"] and records[-2]["reward"] == expected
        assert records[-1]["terminal_reward"] == expected and records[-1]["pending_decisions"] == []


@pytest.mark.parametrize("mode,end,failure,decisions,pending", [
    ("rejection", "incomplete", "rejection", 0, [0]),
    ("unsupported", "incomplete", "unsupported_contract", 0, []),
    ("api_aborted", "aborted", "abort", 0, []),
    ("server_abort", "aborted", None, 1, []),
    ("unknown", "incomplete", "transport_failure", 0, [0]),
])
def test_whole_episode_nonnatural_never_reproposes_or_fabricates_labels(tmp_path, monkeypatch, definitions, mode, end, failure, decisions, pending):
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(mode=mode)
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    saved, journals = persisted(tmp_path, spec)
    check_persisted_result(result, saved)
    assert result.end_kind == end and result.failure == failure and result.decisions == decisions
    assert result.outcome is None and len(script.attempts) <= 1
    assert script.deleted and all(worker.closed for worker in workers.instances)
    for viewer, records in journals.items():
        assert not any(r["kind"] == "transition" and r["terminal"] for r in records)
        assert records[-1]["terminal_reward"] is None and records[-1]["outcome"] is None
        assert records[-1]["final_observation"] is None
        assert records[-1]["pending_decisions"] == (pending if viewer == 2 else [])
    if mode == "unknown":
        action_position = next(i for i, (_, path) in enumerate(script.requests) if path.endswith("/actions"))
        assert script.requests[action_position + 1] == ("GET", "/debug/v1/test-server/info")
        assert result.health["replay_or_finalization_failure"]  # server commit remains diagnostic, not a label
        assert len([value for worker in workers.instances for value in worker.inputs]) == 1
    if mode == "server_abort":
        assert journals[2][2]["response"]["status"] == "aborted"
        assert result.simulator_abort_reason == "server_shutdown"
        assert all(records[-1]["simulator_abort_reason"] == "server_shutdown" for records in journals.values())


def test_whole_episode_decision_limit_preserves_accepted_intermediate_action(tmp_path, monkeypatch, definitions):
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript()
    result, _, spec = asyncio.run(play(tmp_path, definitions, script, max_decisions=1))
    _, journals = persisted(tmp_path, spec)
    assert result.end_kind == "truncated" and result.failure == "decision_limit" and result.outcome is None
    assert result.decisions == len(script.attempts) == 1
    assert [r["kind"] for r in journals[2]] == ["header", "decision", "acknowledgement", "end"]
    assert journals[2][-1]["pending_decisions"] == [] and journals[2][-1]["terminal_reward"] is None
    assert script.deleted and all(worker.closed for worker in workers.instances)


@pytest.mark.parametrize("stage", ["decision", "act"])
def test_whole_episode_cancellation_keeps_pending_ledger_and_closes_resources(tmp_path, monkeypatch, definitions, stage):
    async def run():
        workers = AuditedWorkers(monkeypatch, pause=stage == "decision")
        script = PublicScript(mode="cancel_act" if stage == "act" else "natural")
        task = asyncio.create_task(play(tmp_path, definitions, script))
        await asyncio.wait_for((workers.entered if stage == "decision" else script.act_entered).wait(), 2)
        task.cancel()
        result, _, spec = await asyncio.wait_for(task, 2)
        saved, journals = persisted(tmp_path, spec)
        check_persisted_result(result, saved)
        assert result.end_kind == result.failure == "cancelled" and result.outcome is None
        assert result.decisions == 0 and script.deleted
        assert journals[2][-1]["pending_decisions"] == ([0] if stage == "act" else [])
        assert all(journal[-1]["terminal_reward"] is None for journal in journals.values())
        assert all(worker.closed for worker in workers.instances)
    asyncio.run(run())


@pytest.mark.parametrize("stage", ["begin", "acknowledge"])
def test_whole_episode_journal_save_failure_preserves_known_outcome(tmp_path, monkeypatch, definitions, stage):
    import gf_simulator_client.environment as module
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(actors=(2,))
    class BrokenJournal(ParticipantJournal):
        def begin(self, *args, **kwargs):
            if self.viewer == 2 and stage == "begin":
                raise OSError("journal begin failed")
            return super().begin(*args, **kwargs)
        def acknowledge(self, *args, **kwargs):
            if self.viewer == 2 and stage == "acknowledge":
                raise OSError("journal acknowledgement failed")
            return super().acknowledge(*args, **kwargs)
    original = module.SimulatorEnvironment.__init__
    def initialize(self, settings, factory, journal_factory):
        original(self, settings, factory, BrokenJournal)
    monkeypatch.setattr(module.SimulatorEnvironment, "__init__", initialize)
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    saved, journals = persisted(tmp_path, spec)
    check_persisted_result(result, saved)
    assert result.failure == "save_failure" and result.health["save_failure"]
    assert result.outcome == ("draw" if stage == "acknowledge" else None)
    assert len(script.attempts) == (1 if stage == "acknowledge" else 0)
    assert all(worker.closed for worker in workers.instances) and script.deleted
    if stage == "acknowledge":
        assert journals[2][-1]["kind"] == "decision"  # incomplete file retains uncertain recording
        assert not any(r["kind"] == "transition" for r in journals[2])
        assert journals[1][-1]["terminal_reward"] == 0
    else:
        assert journals[2][-1]["terminal_reward"] is None


def test_turn_limit_replay_reason_is_dataset_truncation_without_terminal_label(tmp_path, monkeypatch, definitions):
    """Simulator06 §28.11: lifecycle Aborted, RL truncation, no invented transition."""
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(mode="turn_limit")
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    _, journals = persisted(tmp_path, spec)
    assert journals[2][2]["response"]["status"] == "aborted"
    assert script.replay(2)["episode_end"]["reason"] == "turn_limit_exceeded"
    assert result.outcome is None and result.decisions == 1
    assert journals[2][-1]["terminal_reward"] is None
    assert not any(r["kind"] == "transition" for r in journals[2])
    assert all(worker.closed for worker in workers.instances)
    assert result.end_kind == "truncated"
    assert result.simulator_abort_reason == "turn_limit_exceeded"
    assert all(records[-1]["end_kind"] == "truncated" and records[-1]["simulator_abort_reason"] == "turn_limit_exceeded" for records in journals.values())
