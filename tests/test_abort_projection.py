"""ADR0003 public closure evidence, schema3 compatibility and diagnostic-only cause."""
import asyncio
import copy
import json
import uuid

import pytest

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import END_PROJECTION_VERSION, ActionConfig, ContractError, PolicyDecision
from gf_training.journal import export_parquet, read_journal
from tests.episode_support import AuditedWorkers, PublicScript, persisted, play
from tests.test_journal import journal_decision, link_manifest, new_journal


def closure(reason="turn_limit_exceeded"):
    return {"type": "aborted", "reason": reason, "player": None,
            "decision": {"turn": 1, "decision_index": None, "invalid_attempt_count": 0}}


@pytest.mark.parametrize("change", ["reason", "player", "turn", "index", "attempt", "type", "missing", "extra", "bad_integer", "unreconciled"])
def test_full_matching_reconciled_closure_identity_required_before_projection(tmp_path, monkeypatch, definitions, change):
    workers = AuditedWorkers(monkeypatch)
    script = PublicScript(mode="turn_limit")
    original = script.replay
    def replay(viewer):
        value = original(viewer)
        if viewer == 2:
            end = value["episode_end"]
            if change == "reason":
                end["reason"] = "server_shutdown"
            elif change == "player":
                end["player"] = 1
            elif change == "turn":
                end["decision"]["turn"] = 2
            elif change == "index":
                end["decision"]["decision_index"] = 0
            elif change == "attempt":
                end["decision"]["invalid_attempt_count"] = 1
            elif change == "type":
                value["episode_end"] = {"type": "terminal", "result": "draw"}
            elif change == "missing":
                value["episode_end"] = None
            elif change == "extra":
                end["unrecognized_identity"] = True
            elif change == "bad_integer":
                end["decision"]["invalid_attempt_count"] = True
            else:
                value["turns"][0]["decisions"] = []
        return value
    script.replay = replay
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    saved, journals = persisted(tmp_path, spec)
    assert result.end_kind == "aborted" and result.simulator_abort_reason is None
    assert not result.replay_reconciled and result.health["replay_or_finalization_failure"]
    assert saved["simulator_abort_reason"] is None
    assert all(records[-1]["simulator_abort_reason"] is None for records in journals.values())
    assert all(records[-1]["terminal_reward"] is None for records in journals.values())
    assert all(worker.closed for worker in workers.instances)


def test_missing_closure_cannot_guess_cause_from_aborted_ack(tmp_path, monkeypatch, definitions):
    AuditedWorkers(monkeypatch)
    script = PublicScript(mode="turn_limit")
    original = script.replay
    def replay(viewer):
        value = original(viewer)
        value["episode_end"] = None
        return value
    script.replay = replay
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    _, journals = persisted(tmp_path, spec)
    assert result.end_kind == "aborted" and result.simulator_abort_reason is None
    assert result.health["replay_or_finalization_failure"]
    assert journals[2][2]["response"]["status"] == "aborted"
    assert journals[2][-1]["simulator_abort_reason"] is None


@pytest.mark.parametrize("reason,expected", [("turn_limit_exceeded", "truncated"), ("server_shutdown", "aborted")])
def test_reconciled_abort_projection_preserves_existing_local_failure(tmp_path, monkeypatch, definitions, reason, expected):
    AuditedWorkers(monkeypatch)
    script = PublicScript(actors=(2, 1))
    original = script.replay
    def replay(viewer):
        value = original(viewer)
        value["episode_end"] = closure(reason)
        return value
    script.replay = replay
    result, _, spec = asyncio.run(play(tmp_path, definitions, script, max_decisions=1))
    _, journals = persisted(tmp_path, spec)
    assert result.end_kind == expected and result.simulator_abort_reason == reason
    assert result.failure == "decision_limit" and result.outcome is None
    assert result.decisions == 1 and result.replay_reconciled
    assert journals[2][2]["response"]["status"] == "active"
    assert journals[2][-1]["simulator_abort_reason"] == reason
    assert not any(r["kind"] == "transition" for r in journals[2])


def test_replay_abort_cannot_override_known_natural_outcome(tmp_path, monkeypatch, definitions):
    AuditedWorkers(monkeypatch)
    outcome = {"win": {"winner": 2}}
    script = PublicScript(outcome=outcome, actors=(2,))
    original = script.replay
    def replay(viewer):
        value = original(viewer)
        value["episode_end"] = closure()
        return value
    script.replay = replay
    result, _, spec = asyncio.run(play(tmp_path, definitions, script))
    _, journals = persisted(tmp_path, spec)
    assert result.end_kind == "natural" and result.outcome == outcome
    assert result.simulator_abort_reason is None and result.health["replay_or_finalization_failure"]
    assert journals[2][-1]["terminal_reward"] == 1 and journals[1][-1]["terminal_reward"] == -1


def aborted_journal(directory, value):
    journal = new_journal(directory)
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    command = {"command_id": str(uuid.uuid4()), "expected_revision": 0, "action": candidates.actions[0]}
    index = journal.begin(value, candidates, PolicyDecision(0, "random-v1"), command, 0)
    journal.acknowledge(index, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1,
                                "status": "aborted", "result": None})
    return journal


def test_schema3_aborted_ack_allows_truncation_only_with_exact_reason(tmp_path, make_input):
    journal = aborted_journal(tmp_path, make_input())
    with pytest.raises(ContractError):
        journal.finish("truncated", None)
    with pytest.raises(ContractError):
        journal.finish("truncated", None, simulator_abort_reason="server_shutdown")
    journal.finish("truncated", None, simulator_abort_reason="turn_limit_exceeded")
    records = read_journal(journal.final_path)
    assert records[0]["schema_version"] == 4 and records[0]["end_projection_version"] == END_PROJECTION_VERSION
    assert records[-1]["simulator_abort_reason"] == "turn_limit_exceeded"
    assert records[-1]["terminal_reward"] is None and records[-1]["final_observation"] is None
    assert [r["kind"] for r in records] == ["header", "decision", "acknowledgement", "end"]
    for reason in (None, "server_shutdown"):
        broken = copy.deepcopy(records)
        broken[-1]["simulator_abort_reason"] = reason
        journal.final_path.write_text("\n".join(json.dumps(r) for r in broken))
        with pytest.raises(ContractError):
            read_journal(journal.final_path)


@pytest.mark.parametrize("old_version", [1, 2, 3])
def test_older_journals_are_rejected_without_reinterpretation(tmp_path, make_input, old_version):
    journal = new_journal(tmp_path)
    journal_decision(journal, make_input())
    journal.finish("aborted", None)
    records = read_journal(journal.final_path)
    records[0]["schema_version"] = old_version
    encoded = "\n".join(json.dumps(r) for r in records)
    journal.final_path.write_text(encoded)
    with pytest.raises(ContractError, match="schema 1, schema 2 and schema 3"):
        read_journal(journal.final_path)
    assert journal.final_path.read_text() == encoded


def test_schema3_export_cause_sidecar_only_and_crosschecks_manifest_and_both_replays(tmp_path, make_input):
    parquet = pytest.importorskip("pyarrow.parquet")
    directory = tmp_path / "episode"
    directory.mkdir()
    value = make_input()
    journal = aborted_journal(directory, value)
    journal.finish("truncated", None, simulator_abort_reason="turn_limit_exceeded")
    manifest = link_manifest(tmp_path, journal)
    end = closure()
    for viewer in (1, 2):
        path = directory / f"replay-player-{viewer}.json"
        if path.exists():
            replay = json.loads(path.read_text())
        else:
            replay = {"game_id": manifest["episodes"][0]["game_id"], "viewer_player_id": viewer,
                      "observation_schema_version": "1", "turns": []}
        replay["episode_end"] = end
        path.write_text(json.dumps(replay))
    output = tmp_path / "derived.parquet"
    metadata = export_parquet([journal.final_path], output)
    assert metadata["derived_schema_version"] == metadata["journal_schema_version"] == 4
    assert metadata["end_projection_version"] == END_PROJECTION_VERSION
    assert metadata["eligible_rows"] == 0
    assert metadata["sources"][0]["end"]["simulator_abort_reason"] == "turn_limit_exceeded"
    assert "replay_mismatch_or_missing" not in metadata["sources"][0]["ineligible_reasons"]
    table = parquet.read_table(output)
    assert json.loads(table.to_pylist()[0]["acknowledgement_json"])["response"]["status"] == "aborted"
    assert table.to_pylist()[0]["transition_json"] is None
    encoded_features = json.dumps(table.to_pylist()) + repr(table.schema.metadata)
    for forbidden in ("turn_limit_exceeded", "simulator_abort_reason", "seed_fingerprint", "http://", "seed-a"):
        assert forbidden not in encoded_features
    schema_metadata = json.loads(table.schema.metadata[b"godfieldbots"])
    assert set(schema_metadata) == {"derived_schema_version", "end_projection_version", "feature_version", "reward_version", "policy_input_version"}
    other = json.loads((directory / "replay-player-2.json").read_text())
    other["episode_end"]["decision"]["turn"] = 99
    (directory / "replay-player-2.json").write_text(json.dumps(other))
    rejected = export_parquet([journal.final_path], tmp_path / "inconsistent.parquet")
    assert "replay_mismatch_or_missing" in rejected["sources"][0]["ineligible_reasons"]
    manifest["episodes"][0]["simulator_abort_reason"] = "server_shutdown"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="projection mismatch"):
        export_parquet([journal.final_path], tmp_path / "bad-manifest.parquet")
