"""Fake HTTP application dispatch with actual owned participant subprocesses."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import gf_cli
from gf_cli.config import load_config
from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, EpisodeGroup, EpisodeSpec
from gf_common.preprocessing import PreparationConfig
from gf_common.structured_tokens import signature
from gf_simulator_client.http import seed_fingerprint
from gf_training.collection import CollectionPlan
from gf_training.native_backend import STRUCTURED_CONTRACT, reader_settings
from gf_training.structured_journal import read_structured_journal
from tests.episode_support import PublicScript

ROOT = Path(__file__).parents[1]


def structured_config(tmp_path, *, backend="python", extra=None):
    overrides = {"actions": {"action_contract": asdict(engineering_contract()), "version": "2", "max_sequence_length": 2, "max_candidates": 30, "max_expansions": 2000},
        "model": {"model_id": "heuristic"}, "opponent": {"model_id": "random"},
        "preprocessing": {"reader": "structured-v1", "backend": backend},
        "run": {"output_dir": str(tmp_path / "runs"), "pairs_per_seed": 1, "sampling_seeds": [17],
                "concurrency": 1},
        "server": {"source_dir": None, "external_url": "http://test", "expected_fingerprint": "test",
                   "config": str(ROOT / "configs/test-server.toml"), "fixed_seed_experiment": True}}
    overrides["server"]["expected_fingerprint"] = seed_fingerprint("00000000-0000-0000-0000-000000000001")
    if extra:
        for key, values in extra.items():
            overrides.setdefault(key, {}).update(values)
    preset = tmp_path / "structured-toy.toml"
    preset.write_text("schema_version=1\nkind='experiment'\n[actions]\nversion='2'\n")
    return load_config(preset, overrides)


def instrument(monkeypatch, mode="natural", *, audit_startup=True):
    import gf_cli.structured as composition
    import gf_simulator_client.structured_environment as module
    from gf_simulator_client import environment
    def component_only(contract):
        assert contract == engineering_contract()
    # Private FakeHTTP double only. The common source-qualification gate stays
    # intact; there is no public CLI/profile permission to upgrade eligibility.
    monkeypatch.setattr(gf_cli, "require_qualified_contract", component_only)
    monkeypatch.setattr(environment, "require_qualified_contract", component_only)
    scripts, workers, ledgers = [], [], []
    actual_worker = module.StructuredParticipantWorker
    actual_compose = composition.structured_environment
    class AuditedWorker(actual_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen_worker_count = len(self.lease.ledger.workers)
            workers.append(self)
        async def initialize(self):
            if audit_startup and not self.initialized:
                assert not any(f"/v1/games/{scripts[-1].game_id}/events/" in path for _, path in scripts[-1].requests)
            return await super().initialize()
    def client(*args, **kwargs):
        script = PublicScript(mode=mode)
        scripts.append(script)
        return script.client()
    def compose(settings, ledger):
        ledgers.append(ledger)
        return actual_compose(settings, ledger)
    monkeypatch.setattr(environment, "AttestedClient", client)
    monkeypatch.setattr(module, "StructuredParticipantWorker", AuditedWorker)
    monkeypatch.setattr(composition, "structured_environment", compose)
    return scripts, workers, ledgers


@pytest.mark.parametrize("owned_toy_collection", [False, True])
def test_actions2_baseline_cli_two_seats_closed_journal_inventory_and_identity(tmp_path, monkeypatch, owned_toy_collection):
    scripts, workers, ledgers = instrument(monkeypatch)
    loaded = structured_config(tmp_path)
    plan = None
    if owned_toy_collection:
        from gf_cli.structured import structured_bindings
        registry, labels = structured_bindings(loaded.config.model, loaded.config.opponent)
        episodes = (EpisodeSpec("toy-seat0", labels, (17, 19), "toy-pair", 1),
                    EpisodeSpec("toy-seat1", (labels[1], labels[0]), (19, 17), "toy-pair", 2))
        plan = CollectionPlan((EpisodeGroup(loaded.config.run.seeds[0], episodes),), registry,
            tmp_path / "owned-toy-batch", {"explicit_toy_component": True}, time.monotonic() + 60, time.monotonic() + 65)
    directory, summary = asyncio.run(gf_cli.run_experiment(loaded, "collect", collection=plan))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema_version"] == (4 if owned_toy_collection else 3)
    assert manifest["status"] == "complete" and summary["comparison_valid"]
    assert manifest["journal_schema_version"] == 5 and manifest["feature_version"] == "participant-structured-v1"
    assert manifest["participant_runtime"] == "python-reference-v2"
    assert manifest["preprocessing_implementation"] == "python-reference-structured-v1"
    expected = {"heuristic-structured-v1", "random-structured-v1"}
    assert set(manifest["policy_bindings"]) == expected
    assert {row["model_id"] for row in manifest["policy_bindings"].values()} == {"heuristic", "random"}
    specs = [spec for group in manifest["schedule"] for spec in group["episodes"]]
    assert len(specs) == 2 and specs[0]["policies"] == list(reversed(specs[1]["policies"]))
    assert {tuple(spec["policies"]) for spec in specs} == {tuple(expected), tuple(reversed(tuple(expected)))}
    assert summary["natural_denominator"] == 2
    opaque_banks = set()
    for spec in specs:
        episode = directory / spec["episode_id"]
        saved = json.loads((episode / "result.json").read_text())
        assert saved["outcome"] == "draw" and saved["replay_reconciled"] and not saved["health"]
        assert saved["participant_runtime"] == "python-reference-v2"
        assert set(saved["model_startup_ms"]) == {"1", "2"}
        assert len(saved["decision_timings"]) == 4
        for viewer in (1, 2):
            data = read_structured_journal(episode / f"player-{viewer}.jsonl", action_contract=engineering_contract(), purpose="component", synthetic_capacity=False)
            assert data.header["policy"] == spec["policies"][viewer - 1]
            assert data.header["model_id"] in ("heuristic", "random") and data.header["model_version"] == "1"
            assert data.end["end_kind"] == "natural" and data.end["terminal_reward"] == 0
            assert data.accepted == (0, 1) and len(data.sidecars) == 2 and data.bank.boundary.end == 6
            assert all(decision.selection.logprob is None and decision.selection.value is None for decision in data.decisions)
            opaque_banks.add(data.header["bank_episode"])
            for row in data.records:
                if row["kind"] == "decision":
                    assert "kinds" not in row["table"] and "slots" not in row["table"]
                    assert "events" not in row["frame"]["payload"]
                    assert row["command"]["action"] == data.decisions[row["decision"]].selection.validate(
                        data.decisions[row["decision"]].generated.table, data.decisions[row["decision"]].frame)
    assert len(opaque_banks) == 4
    assert all(worker.closed and worker.process.poll() is not None and worker.lease.released for worker in workers)
    assert len(workers) == 4 and len(ledgers) == 1 and not ledgers[0].workers
    assert all(script.deleted for script in scripts) and sum(len(script.attempts) for script in scripts) == 8


@pytest.mark.parametrize("failure", ["sidecar", "row"])
def test_actions2_owned_write_failure_blocks_act_and_keeps_orphan_evidence(tmp_path, monkeypatch, failure):
    import gf_training.structured_journal as module
    scripts, workers, ledgers = instrument(monkeypatch)
    if failure == "sidecar":
        def failed(*args, **kwargs):
            raise OSError("explicit sidecar publication failure")
        monkeypatch.setattr(module, "publish_candidates", failed)
    else:
        actual = module.StructuredJournal.write
        def write(self, row):
            if row["kind"] == "decision":
                raise OSError("explicit journal row failure")
            return actual(self, row)
        monkeypatch.setattr(module.StructuredJournal, "write", write)
    directory, summary = asyncio.run(gf_cli.run_experiment(structured_config(tmp_path), "collect"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and not summary["comparison_valid"]
    assert all(not script.attempts and script.deleted for script in scripts)
    assert workers and all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert not ledgers[0].workers
    assert all(row["outcome"] is None and row["failure"] == "save_failure" for row in manifest["episodes"])
    columns = list(directory.glob("*/*.columns"))
    assert bool(columns) is (failure == "row")
    for path in directory.glob("*/player-*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert not any(row["kind"] in ("acknowledgement", "transition") for row in rows)
        assert rows[-1]["terminal_reward"] is None


@pytest.mark.parametrize("metadata", [{}, {"structured_preprocessing_contract": STRUCTURED_CONTRACT},
    {"structured_preprocessing_contract": STRUCTURED_CONTRACT, "structured_default_signature": "0" * 64}])
def test_missing_native_structured_capability_fails_before_any_simulator_io(tmp_path, monkeypatch, metadata):
    import gf_training.native_backend as backend
    from gf_simulator_client import environment
    monkeypatch.setattr(gf_cli, "require_qualified_contract", lambda contract: None)  # private metadata-only test double
    monkeypatch.setattr(backend, "load_native", lambda: SimpleNamespace(build_info=lambda: metadata))
    def forbidden(*args, **kwargs):
        pytest.fail("Simulator IO began despite absent structured native capability")
    monkeypatch.setattr(environment, "AttestedClient", forbidden)
    monkeypatch.setattr(environment, "ManagedServer", forbidden)
    with pytest.raises(ContractError, match="structured-v1"):
        asyncio.run(gf_cli.run_experiment(structured_config(tmp_path, backend="rust"), "collect"))
    assert not (tmp_path / "runs").exists()


def test_native_structured_reader_metadata_is_closed_and_additive():
    config = PreparationConfig(backend="rust", reader="structured-v1")
    native = SimpleNamespace(build_info=lambda: {"structured_preprocessing_contract": STRUCTURED_CONTRACT,
                                               "structured_default_signature": signature(128)})
    assert reader_settings(config, native) == asdict(config)
    old = PreparationConfig(backend="rust")
    assert reader_settings(old, SimpleNamespace(build_info=dict)) == {key: value for key, value in asdict(old).items() if key != "reader"}


def test_formal_neural2_component_checkpoint_not_authorized_by_dispatch(tmp_path, monkeypatch):
    import gf_cli.structured as composition
    from gf_models.structured_binding import ConfiguredNeuralModelV2
    def forbidden(*args, **kwargs):
        pytest.fail("bare model component reached Environment construction")
    monkeypatch.setattr(composition, "structured_environment", forbidden)
    loaded = structured_config(tmp_path)
    loaded = replace(loaded, config=replace(loaded.config, model=ConfiguredNeuralModelV2(checkpoint=str(tmp_path / "bare-component"))))
    with pytest.raises(ContractError, match="engineering_only"):
        asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert not (tmp_path / "runs").exists()


def test_observation_only_production_mode_fails_before_backend_or_simulator_io(tmp_path, monkeypatch):
    import gf_cli.structured as composition
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported production input mode passed participant preflight")
    monkeypatch.setattr(composition, "structured_backend", forbidden)
    monkeypatch.setattr(composition, "structured_environment", forbidden)
    with pytest.raises(ContractError, match="requires acting-player atomic hints"):
        structured_config(tmp_path, extra={"actions": {"use_action_options": False}})
    assert not (tmp_path / "runs").exists()


def test_optional_free_baseline_dispatch_does_not_import_torch(tmp_path):
    import subprocess
    import sys
    loaded = structured_config(tmp_path)
    script = """
import importlib.abc
import json
import sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('coordinator imported Torch')
sys.meta_path.insert(0, RejectTorch())
from gf_cli.config import load_config
from gf_cli.structured import structured_backend, structured_bindings
config = load_config(sys.argv[1], json.loads(sys.argv[2])).config
assert structured_backend(config.preprocessing) == 'python-reference-structured-v1'
structured_bindings(config.model, config.opponent)
assert 'torch' not in sys.modules
"""
    overrides = {key: value for key, value in loaded.effective.items() if key not in ("schema_version", "kind", "training")}
    result = subprocess.run([sys.executable, "-c", script, loaded.inputs[0]["path"], json.dumps(overrides)], check=False,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_two_concurrent_games_share_one_four_worker_ledger_and_reap_before_return(tmp_path, monkeypatch):
    from gf_cli.structured import structured_bindings
    scripts, workers, ledgers = instrument(monkeypatch, audit_startup=False)
    loaded = structured_config(tmp_path, extra={"run": {"concurrency": 2}})
    registry, labels = structured_bindings(loaded.config.model, loaded.config.opponent)
    groups = tuple(EpisodeGroup(loaded.config.run.seeds[0], (EpisodeSpec(f"concurrent-{seat}",
        labels if seat == 1 else (labels[1], labels[0]), (17, 19), "toy-concurrent-pair", seat),)) for seat in (1, 2))
    plan = CollectionPlan(groups, registry, tmp_path / "owned-concurrent-toy", {"explicit_toy_component": True},
                          time.monotonic() + 60, time.monotonic() + 65)
    _, summary = asyncio.run(gf_cli.run_experiment(loaded, "collect", collection=plan))
    assert summary["comparison_valid"] and summary["natural_denominator"] == 2
    assert len(ledgers) == 1 and not ledgers[0].workers
    assert max(worker.seen_worker_count for worker in workers) == 4
    assert all(worker.lease.ledger is ledgers[0] and worker.closed and worker.process.poll() is not None for worker in workers)
    assert len(scripts) == 2 and all(script.deleted and len(script.attempts) == 4 for script in scripts)
