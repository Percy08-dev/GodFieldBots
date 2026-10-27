import hashlib
import json
import uuid
from dataclasses import asdict, replace

import pytest

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    ActionConfig,
    ContractError,
    PolicyDecision,
    digest,
)
from gf_training.journal import ParticipantJournal, assign_splits, export_parquet, read_journal


def new_journal(directory, episode="episode", viewer=1, policy="random", run_id=None):
    return ParticipantJournal(directory, episode, viewer, policy, run_id=run_id or str(uuid.uuid4()),
                              game_id="01M5WN0YQY49FDW5YQ7W4J74H2", action_config_hash=digest(asdict(ActionConfig())))


def final_state(value, outcome):
    return dict(value.observation, result=outcome, phase="terminal", actionable=False,
                typed_action_schema=[], current_player=None, required_choice=None)


def link_manifest(directory, journal, split="train", ancestry=None, seed="seed-a"):
    records = read_journal(journal.final_path)
    header = records[0]
    end = records[-1]
    config = {"actions": asdict(replace(ActionConfig(), use_action_options=header["use_action_options"])), "run": {"split": split},
              "model": {"model_id": "random", "version": "1"}}
    accepted = [r["command"]["action"] for r in records if r["kind"] == "decision"]
    natural = end["end_kind"] == "natural"
    outcome = end["outcome"]
    episode = {"episode_id": header["episode_id"], "game_id": header["game_id"], "pair_id": "pair", "seed": seed,
               "policies": ["random", "random"], "end_kind": end["end_kind"], "outcome": outcome,
               "failure": None, "health": {}, "replay_reconciled": True, "simulator_abort_reason": end["simulator_abort_reason"]}
    spec = {"episode_id": header["episode_id"], "pair_id": "pair", "policies": ["random", "random"]}
    manifest = {"schema_version": 3, "run_id": header["run_id"], "status": "complete" if natural else "incomplete",
                "journal_schema_version": 4, "end_projection_version": END_PROJECTION_VERSION, "feature_version": "participant-json-v2", "reward_policy_version": "natural-terminal-v1",
                "policy_input_version": POLICY_INPUT_VERSION, "reviewed_simulator_spec_sha256": REVIEWED_SPEC_HASHES,
                "catalog": CATALOG_META, "effective_config": config, "config_sha256": digest(config), "split": split,
                "simulator_contract_revision": SIMULATOR_REVISION, "observation_schema_version": "1", "action_schema_version": 1,
                "episodes": [episode], "schedule": [{"seed": seed, "ancestry": ancestry, "episodes": [spec]}],
                "groups": [{"episode_ids": [header["episode_id"]]}]}
    report = {"schema_version": 1, "comparison_valid": natural, "failures": {}, "health_failures": {}, "group_health_failures": {}}
    encoded_report = json.dumps(report).encode()
    (directory / "report.json").write_bytes(encoded_report)
    manifest["report_sha256"] = hashlib.sha256(encoded_report).hexdigest()
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    replay = {"game_id": header["game_id"], "viewer_player_id": header["viewer"], "observation_schema_version": "1",
              "episode_end": {"type": "terminal", "result": outcome} if natural else None,
              "turns": [{"decisions": [{"actor": header["viewer"], "visible_action": {"type": "action", "action": a}} for a in accepted]}]}
    (journal.final_path.parent / f"replay-player-{header['viewer']}.json").write_text(json.dumps(replay))
    return manifest


def journal_decision(journal, value, command_id=None):
    command_id = str(uuid.uuid4())
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    selection = PolicyDecision(0, "random-v1", -1.0)
    command = {"command_id": command_id, "expected_revision": value.observation["state_revision"], "action": candidates.actions[0]}
    index = journal.begin(value, candidates, selection, command, 1.5)
    acknowledgement = {"command_id": command_id, "previous_revision": value.observation["state_revision"], "state_revision": value.observation["state_revision"] + 1, "status": "active", "result": None}
    journal.acknowledge(index, acknowledgement)
    return index


def test_roundtrip_exact_candidates_pending_and_next_decision(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    next_value = make_input()
    next_value.observation["state_revision"] = 1
    next_value.action_options["state_revision"] = 1
    journal_decision(journal, next_value)
    journal.finish("truncated", None)
    records = read_journal(journal.final_path)
    decisions = [r for r in records if r["kind"] == "decision"]
    assert decisions[0]["input"] == value.to_dict()
    assert decisions[0]["selection"] == asdict(PolicyDecision(0, "random-v1", -1.0))
    assert [r for r in records if r["kind"] == "transition"] == [{"kind": "transition", "decision": 0, "next_decision": 1, "reward": 0, "terminal": False}]
    assert records[-1]["terminal_reward"] is None
    assert records[-1]["final_observation"] is None
    assert not journal.path.exists()


def test_natural_terminal_reward_requires_observed_own_state(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    outcome = {"win": {"winner": 1}}
    with pytest.raises(ContractError):
        journal.finish("natural", outcome)
    final = final_state(value, outcome)
    journal.finish("natural", outcome, final)
    records = read_journal(journal.final_path)
    assert records[-1]["terminal_reward"] == 1
    assert records[-2]["reward"] == 1


def test_unacknowledged_delivery_stays_pending(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    journal.begin(value, candidates, PolicyDecision(0, "random-v1"), {"command_id": str(uuid.uuid4()), "expected_revision": value.observation["state_revision"], "action": candidates.actions[0]}, 0)
    journal.finish("incomplete", None)
    records = read_journal(journal.final_path)
    assert records[-1]["pending_decisions"] == [0]
    assert not any(r["kind"] == "transition" for r in records)


def test_export_roundtrip_determinism_and_incompatible_versions(tmp_path, make_input):
    pa = pytest.importorskip("pyarrow.parquet")
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    journal = new_journal(episode_dir)
    value = make_input()
    journal_decision(journal, value)
    journal.finish("incomplete", None)
    link_manifest(tmp_path, journal)
    output = tmp_path / "one.parquet"
    metadata = export_parquet([journal.final_path], output)
    assert metadata["rows"] == 1
    row = pa.read_table(output).to_pylist()[0]
    assert json.loads(row["input_json"]) == value.to_dict()
    other = tmp_path / "two.parquet"
    export_parquet([journal.final_path], other)
    assert output.read_bytes() == other.read_bytes()
    records = read_journal(journal.final_path)
    records[1]["candidates"]["generator_version"] = "future"
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError):
        export_parquet([journal.final_path, bad], tmp_path / "bad.parquet")


def test_seed_and_ancestry_splits_both_views_together():
    episodes = [
        {"episode_id": "one-p1", "seed": "a"}, {"episode_id": "one-p2", "seed": "a"},
        {"episode_id": "repeat", "seed": "a", "ancestry": "fork"},
        {"episode_id": "other", "seed": "b", "ancestry": "fork"},
        {"episode_id": "unrelated", "seed": "c"},
    ]
    splits = assign_splits(episodes)
    assert len({splits[key] for key in ("one-p1", "one-p2", "repeat", "other")}) == 1
    assert splits == assign_splits(list(reversed(episodes)))


@pytest.mark.parametrize("mutate", [
    lambda rs: rs[-2].update(reward=-1),
    lambda rs: rs[-1].update(terminal_reward=-1),
    lambda rs: rs[-1].update(end_kind="truncated", outcome=None, terminal_reward=None),
    lambda rs: rs[-1]["final_observation"]["result"].update(win={"winner": 2}),
    lambda rs: rs[2]["response"].update(status="rejected"),
    lambda rs: rs[2]["response"].update(previous_revision=-1),
    lambda rs: rs[2]["response"].update(state_revision=200),
    lambda rs: rs[1]["command"].update(expected_revision=-1),
    lambda rs: rs.append(dict(rs[-1])),
    lambda rs: rs.insert(2, dict(rs[2])),
])
def test_corrupt_reward_ack_order_rejected(tmp_path, make_input, mutate):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    outcome = {"win": {"winner": 1}}
    journal.finish("natural", outcome, final_state(value, outcome))
    records = read_journal(journal.final_path)
    mutate(records)
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError):
        read_journal(journal.final_path)


def test_terminal_transition_without_end_and_schema1_rejected(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    journal.finish("natural", "draw", final_state(value, "draw"))
    records = read_journal(journal.final_path)
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records[:-1]))
    with pytest.raises(ContractError):
        read_journal(journal.final_path)
    records[0]["schema_version"] = 1
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError, match="schema 1"):
        read_journal(journal.final_path)


def test_export_linkage_declared_split_conflicts_and_zero_decision_end(tmp_path, make_input):
    pytest.importorskip("pyarrow")
    sources = []
    value = make_input()
    for number, split in enumerate(("train", "test")):
        directory = tmp_path / str(number)
        episode = directory / "episode"
        episode.mkdir(parents=True)
        journal = new_journal(episode)
        journal.finish("natural", "draw", final_state(value, "draw"))
        link_manifest(directory, journal, split=split)
        sources.append(journal.final_path)
    output = tmp_path / "derived.parquet"
    with pytest.raises(ContractError, match="conflicting declared splits"):
        export_parquet(sources, output)
    metadata = export_parquet([sources[0]], output)
    assert metadata["rows"] == 0
    assert metadata["sources"][0]["decision_count"] == 0
    assert metadata["sources"][0]["end"]["outcome"] == "draw"
    assert metadata["sources"][0]["eligible"]
    with pytest.raises(ContractError, match="duplicate"):
        export_parquet([sources[0], sources[0]], output)
    manifest_path = sources[0].parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="finalized"):
        export_parquet([sources[0]], output)
    manifest["status"] = "complete"
    manifest["episodes"][0]["health"] = {"save_failure": "disk"}
    manifest_path.write_text(json.dumps(manifest))
    metadata = export_parquet([sources[0]], output)
    assert not metadata["sources"][0]["eligible"]
    assert metadata["sources"][0]["episode_result"]["outcome"] == "draw"
    manifest_path.unlink()
    with pytest.raises(ContractError, match="linked run manifest"):
        export_parquet([sources[0]], output)


def test_terminal_ack_result_must_match_end_and_forbids_later_decision(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    journal.finish("natural", "draw", final_state(value, "draw"))
    records = read_journal(journal.final_path)
    records[2]["response"].update(status="terminal", result={"win": {"winner": 2}})
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError, match="acknowledgement/end"):
        read_journal(journal.final_path)
    records[2]["response"]["result"] = "draw"
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    assert read_journal(journal.final_path)[-1]["outcome"] == "draw"
    records.insert(3, {"kind": "transition", "decision": 0, "next_decision": 1, "reward": 0, "terminal": False})
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError, match="continuing transition"):
        read_journal(journal.final_path)


def test_positive_aborted_ack_only_allows_aborted_unlabelled_end(tmp_path, make_input):
    value = make_input()
    journal = new_journal(tmp_path)
    journal_decision(journal, value)
    journal.finish("aborted", None)
    records = read_journal(journal.final_path)
    records[2]["response"]["status"] = "aborted"
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    assert read_journal(journal.final_path)[-1]["terminal_reward"] is None
    records[-1]["end_kind"] = "incomplete"
    journal.final_path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ContractError, match="acknowledgement/end"):
        read_journal(journal.final_path)


@pytest.mark.parametrize("status", ["incomplete", "interrupted"])
def test_export_failed_run_clean_participant_stays_ineligible(tmp_path, make_input, status):
    pytest.importorskip("pyarrow")
    episode = tmp_path / "episode"
    episode.mkdir()
    value = make_input()
    journal = new_journal(episode)
    journal.finish("natural", "draw", final_state(value, "draw"))
    manifest = link_manifest(tmp_path, journal)
    manifest["status"] = status
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    metadata = export_parquet([journal.final_path], tmp_path / "derived.parquet")
    source = metadata["sources"][0]
    assert not source["eligible"]
    assert "run_not_complete" in source["ineligible_reasons"]
    assert source["episode_result"]["outcome"] == "draw"


@pytest.mark.parametrize("case", ["missing", "malformed", "mismatch", "failed", "missing_health", "health_failure"])
def test_complete_manifest_requires_consistent_successful_linked_report(tmp_path, make_input, case):
    pytest.importorskip("pyarrow")
    episode = tmp_path / "episode"
    episode.mkdir()
    value = make_input()
    journal = new_journal(episode)
    journal_decision(journal, value)
    journal.finish("natural", "draw", final_state(value, "draw"))
    manifest = link_manifest(tmp_path, journal)
    path = tmp_path / "report.json"
    if case == "missing":
        path.unlink()
    elif case == "malformed":
        path.write_text("not-json")
    elif case == "mismatch":
        path.write_text(path.read_text() + " ")
    else:
        report = json.loads(path.read_text())
        if case == "failed":
            report["comparison_valid"] = False
        elif case == "missing_health":
            report.pop("health_failures")
        else:
            report["group_health_failures"] = {"client_close_failure": 1}
        encoded = json.dumps(report).encode()
        path.write_bytes(encoded)
        manifest["report_sha256"] = hashlib.sha256(encoded).hexdigest()
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    metadata = export_parquet([journal.final_path], tmp_path / "derived.parquet")
    assert metadata["eligible_rows"] == 0
    assert not metadata["sources"][0]["eligible"]
    assert metadata["sources"][0]["end"]["outcome"] == "draw"
    assert any(reason.startswith("report_") for reason in metadata["sources"][0]["ineligible_reasons"])


def test_unchanged_successful_report_keeps_natural_rows_eligible(tmp_path, make_input):
    parquet = pytest.importorskip("pyarrow.parquet")
    episode = tmp_path / "episode"
    episode.mkdir()
    value = make_input()
    journal = new_journal(episode)
    journal_decision(journal, value)
    journal.finish("natural", "draw", final_state(value, "draw"))
    link_manifest(tmp_path, journal)
    output = tmp_path / "derived.parquet"
    metadata = export_parquet([journal.final_path], output)
    assert metadata["rows"] == metadata["eligible_rows"] == 1
    assert metadata["sources"][0]["eligible"]
    assert parquet.read_table(output).to_pylist()[0]["eligible"]
