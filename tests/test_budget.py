import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import gf_cli
from gf_cli.config import load_config
from gf_common.domain import EpisodeResult, GroupResult, digest

ROOT = Path(__file__).parents[1]


def loaded_config(tmp_path):
    moved = tmp_path / "moved-server"
    moved.write_bytes(b"prebuilt fixture")
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {"run": {"output_dir": str(tmp_path / "runs")}, "server": {"source_dir": None, "binary": str(moved)}})
    config = replace(loaded.config,
                     run=replace(loaded.config.run, wall_seconds=0.35, admission_seconds=0.1, execution_seconds=0.15,
                                 finalization_seconds=0.2, report_reserve_seconds=0.03),
                     server=replace(loaded.config.server, cleanup_reserve_seconds=0.03))
    return replace(loaded, config=config, effective=asdict(config), sha256=digest(asdict(config)))


def fake_environment(monkeypatch, health=None):
    constructed = []
    class Environment:
        def __init__(self, settings, *args):
            constructed.append(settings)
        async def run_group(self, group):
            return GroupResult([EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                               outcome="draw", end_kind="natural", first_player=2) for spec in group.episodes], health or {})
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", Environment)
    return constructed


def test_delayed_preflight_exhausts_budget_before_environment_or_statistics(tmp_path, monkeypatch):
    loaded = loaded_config(tmp_path)
    constructed = fake_environment(monkeypatch)
    real_hash = gf_cli._file_hash
    def delayed_hash(path, deadline=float("inf")):
        if str(path) == loaded.config.server.binary:
            time.sleep(0.2)
        return real_hash(path, deadline)
    monkeypatch.setattr(gf_cli, "_file_hash", delayed_hash)
    monkeypatch.setattr(gf_cli, "report", lambda *a, **k: pytest.fail("statistics must not start"))
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "smoke"))
    assert not constructed
    assert not summary["comparison_valid"] and summary["report_status"] == "minimal"
    assert json.loads((directory / "manifest.json").read_text())["status"] == "incomplete"


def test_delayed_statistics_falls_back_without_claiming_completed_comparison(tmp_path, monkeypatch):
    loaded = loaded_config(tmp_path)
    fake_environment(monkeypatch)
    real_report = gf_cli.report
    def delayed_report(*args, **kwargs):
        value = real_report(*args, **kwargs)
        time.sleep(0.4)
        return value
    monkeypatch.setattr(gf_cli, "report", delayed_report)
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert summary["draws"] == summary["natural_denominator"] == 2
    assert not summary["comparison_valid"] and summary["report_status"] == "minimal"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert all(not group.get("cleanup_failure") for group in manifest["groups"])


def test_delayed_final_manifest_write_is_corrected_to_unsuccessful(tmp_path, monkeypatch):
    loaded = loaded_config(tmp_path)
    fake_environment(monkeypatch)
    real_atomic = gf_cli.atomic_json
    delayed = False
    def delayed_final_write(path, value):
        nonlocal delayed
        real_atomic(path, value)
        if path.name == "manifest.json" and value["status"] == "complete" and not delayed:
            delayed = True
            time.sleep(0.4)
    monkeypatch.setattr(gf_cli, "atomic_json", delayed_final_write)
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert delayed
    assert not summary["comparison_valid"] and summary["natural_denominator"] == 2
    assert manifest["status"] == "incomplete" and manifest["finalization_failure"] == "budget_exhausted"
    assert manifest["budget_overrun_seconds"] > 0
    assert manifest["persistence_duration_seconds"] >= 0.4
    assert not json.loads((directory / "report.json").read_text())["comparison_valid"]


@pytest.mark.parametrize("health", [{"client_close_failure": "failure"}, {"finalization_cancelled": "cancelled"}])
def test_group_health_natural_pair_persists_failed_run(tmp_path, monkeypatch, health):
    loaded = loaded_config(tmp_path)
    fake_environment(monkeypatch, health)
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert summary["natural_denominator"] == 2 and not summary["comparison_valid"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert all(e["outcome"] == "draw" for e in manifest["episodes"])


@pytest.mark.parametrize("failed_correction", ["report.json", "manifest.json", "both"])
def test_independent_failure_corrections_and_export_report_qualification(tmp_path, monkeypatch, make_input, failed_correction):
    pytest.importorskip("pyarrow")
    import uuid

    from gf_common.actions import PublicCandidateGenerator
    from gf_common.domain import PolicyDecision
    from gf_training.journal import export_parquet
    from tests.test_journal import final_state

    loaded = loaded_config(tmp_path)
    class JournalEnvironment:
        def __init__(self, settings, policy_factory, journal_factory):
            self.settings = settings
            self.journal_factory = journal_factory
        async def run_group(self, group):
            episodes = []
            for index, spec in enumerate(group.episodes):
                directory = self.settings.output_dir / spec.episode_id
                directory.mkdir()
                game = "01M5WN0YQY49FDW5YQ7W4J74H" + str(index)
                value = make_input()
                candidates = PublicCandidateGenerator(self.settings.actions).generate(value)
                journal = self.journal_factory(directory, spec.episode_id, 1, spec.policies[0], run_id=self.settings.run_id,
                                               game_id=game, action_config_hash=digest(asdict(self.settings.actions)))
                command = {"command_id": str(uuid.uuid4()), "expected_revision": 0, "action": candidates.actions[0]}
                decision = journal.begin(value, candidates, PolicyDecision(0, spec.policies[0] + "-v1"), command, 0)
                journal.acknowledge(decision, {"command_id": command["command_id"], "previous_revision": 0, "state_revision": 1, "status": "terminal", "result": "draw"})
                journal.finish("natural", "draw", final_state(value, "draw"))
                replay = {"game_id": game, "viewer_player_id": 1, "observation_schema_version": "1",
                          "episode_end": {"type": "terminal", "result": "draw"},
                          "turns": [{"decisions": [{"actor": 1, "visible_action": {"type": "action", "action": command["action"]}}]}]}
                (directory / "replay-player-1.json").write_text(json.dumps(replay))
                episodes.append(EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                              game_id=game, outcome="draw", end_kind="natural", first_player=2, replay_reconciled=True))
            return GroupResult(episodes)
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", JournalEnvironment)
    real_atomic = gf_cli.atomic_json
    delayed = False
    correction_attempts = []
    def partially_failing_correction(path, value):
        nonlocal delayed
        if delayed and ((path.name == "manifest.json" and value.get("status") == "incomplete") or
                        (path.name == "report.json" and value.get("comparison_valid") is False)):
            correction_attempts.append(path.name)
            if failed_correction in (path.name, "both"):
                raise OSError("intentional correction failure for " + path.name)
        real_atomic(path, value)
        if path.name == "manifest.json" and value.get("status") == "complete" and not delayed:
            delayed = True
            time.sleep(0.4)
    monkeypatch.setattr(gf_cli, "atomic_json", partially_failing_correction)
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert not summary["comparison_valid"] and summary["natural_denominator"] == 2
    assert correction_attempts == ["report.json", "manifest.json"]
    expected_failures = {"report.json", "manifest.json"} if failed_correction == "both" else {failed_correction}
    assert set(summary["correction_save_failures"]) == expected_failures
    manifest = json.loads((directory / "manifest.json").read_text())
    if failed_correction == "both":
        # Unavoidable filesystem debt: neither invalidation can be saved. The
        # caller must quarantine the stale artifacts using the returned failure.
        assert manifest["status"] == "complete"
        assert json.loads((directory / "report.json").read_text())["comparison_valid"]
        return
    metadata = export_parquet(sorted(directory.glob("s*-seat*/player-1.jsonl")), tmp_path / "derived.parquet")
    assert metadata["rows"] == 2 and metadata["eligible_rows"] == 0
    assert all(source["end"]["outcome"] == "draw" for source in metadata["sources"])
    if failed_correction == "manifest.json":
        assert manifest["status"] == "complete"
        assert not json.loads((directory / "report.json").read_text())["comparison_valid"]
        assert all("report_hash_mismatch" in source["ineligible_reasons"] for source in metadata["sources"])
    else:
        assert manifest["status"] == "incomplete"
        assert all("run_not_complete" in source["ineligible_reasons"] for source in metadata["sources"])
