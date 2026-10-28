import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gf_cli.config import load_config
from gf_common.domain import ContractError, canonical, digest
from gf_training.gates import feasibility
from gf_training.lightgbm_run import train_experiments
from gf_training.q_data import labeled_samples, load_run, validate_bundles
from tests.lightgbm_support import synthetic_run


@pytest.mark.parametrize("outcome,target", [("draw", 0), ({"win": {"winner": 1}}, 1), ({"win": {"winner": 2}}, -1)])
def test_positive_executed_only_final_return_and_original_candidates(tmp_path, make_input, outcome, target):
    manifest = synthetic_run(tmp_path / "train", make_input, outcomes=outcome)
    bundle = load_run(tmp_path / "train", "train")
    rows, metadata = validate_bundles([bundle])
    samples = labeled_samples(rows, metadata, "train")
    assert len(samples) == 2 and all(sample.target == target for sample in samples)
    assert all(sample.identity[2] == 1 for sample in samples)  # other viewer made no decisions
    for sample, row in zip(samples, rows, strict=True):
        original = json.loads(row["candidate_set_json"])
        selection = json.loads(row["selection_json"])
        assert sample.action == original["actions"][selection["index"]]
    corrupted = copy.deepcopy(rows)
    corrupted[0]["acknowledgement_json"] = None
    with pytest.raises(ContractError, match="unexecuted"):
        labeled_samples(corrupted, metadata, "train")
    assert metadata["manifests"][manifest["run_id"]]["split"] == "train"


@pytest.mark.parametrize("damage", ["missing_journal", "report_hash", "replay", "status", "budget", "health", "test"])
def test_whole_original_run_guards(tmp_path, make_input, damage):
    directory = tmp_path / "run"
    manifest = synthetic_run(directory, make_input)
    if damage == "missing_journal":
        next(directory.glob("*/player-2.jsonl")).unlink()
    elif damage == "report_hash":
        (directory / "report.json").write_text((directory / "report.json").read_text() + " ")
    elif damage == "replay":
        next(directory.glob("*/replay-player-1.json")).write_text("{}")
    elif damage == "status":
        manifest["status"] = "incomplete"
    elif damage == "budget":
        manifest["elapsed_seconds"] = 7201
    elif damage == "health":
        manifest["episodes"][0]["health"]["save_failure"] = "broken"
    else:
        manifest["split"] = "test"
        manifest["effective_config"]["run"]["split"] = "test"
        manifest["config_sha256"] = digest(manifest["effective_config"])
    (directory / "manifest.json").write_text(canonical(manifest))
    with pytest.raises(ContractError):
        validate_bundles([load_run(directory)])


def test_cross_role_equal_seed_and_common_ancestry_rejected_including_pilot(tmp_path, make_input):
    synthetic_run(tmp_path / "train", make_input)
    synthetic_run(tmp_path / "validation", make_input, split="validation")
    with pytest.raises(ContractError, match="conflicting declared splits"):
        validate_bundles([load_run(tmp_path / "train"), load_run(tmp_path / "validation")])
    with pytest.raises(ContractError, match="duplicate source run"):
        validate_bundles([load_run(tmp_path / "train"), load_run(tmp_path / "train")])
    synthetic_run(tmp_path / "fork_train", make_input, ancestry="ancestor")
    synthetic_run(tmp_path / "fork_validation", make_input, split="validation", seeds=("00000000-0000-0000-0000-000000000002",), ancestry="ancestor")
    with pytest.raises(ContractError, match="conflicting declared splits"):
        validate_bundles([load_run(tmp_path / "fork_train"), load_run(tmp_path / "fork_validation")])


def test_small_or_malformed_pilot_never_passes(tmp_path, make_input):
    directory = tmp_path / "pilot"
    manifest = synthetic_run(directory, make_input, split="validation")
    with pytest.raises(ContractError, match="three distinct"):
        feasibility(directory)
    manifest["effective_config"]["run"]["sampling_seeds"] = [17, 29, 43]
    manifest["config_sha256"] = digest(manifest["effective_config"])
    (directory / "manifest.json").write_text(canonical(manifest))
    with pytest.raises(ContractError, match="64 games"):
        feasibility(directory)


def test_explicit_synthetic_full_pilot_checks_every_original_journal(tmp_path, make_input):
    directory = tmp_path / "synthetic-gate-fixture"
    synthetic_run(directory, make_input, split="validation", seeds=tuple(f"00000000-0000-0000-0000-{index:012d}" for index in range(1, 33)),
                  sampling=(17, 29, 43), decisions=False)
    _, proof = feasibility(directory)
    assert proof["passed"] and proof["games"] == 192
    assert proof["split"] == "validation"
    next(directory.glob("*/player-2.jsonl")).unlink()
    with pytest.raises(ContractError):
        feasibility(directory)


def test_current_failed_pilot_rejects_before_native_fit(tmp_path, monkeypatch):
    import asyncio

    import gf_training.lightgbm_run as module
    pilot = Path("runs/pilot/baseline-feasibility-pilot-354f64ce-747").resolve()
    if not pilot.exists():
        pytest.skip("historical local failed-pilot bundle not installed")
    loaded = load_config("configs/experiments/smoke.toml", {"model": {"model_id": "lightgbm"},
                         "training": {"pilot_run": str(pilot)}, "run": {"output_dir": str(tmp_path / "output")}})
    monkeypatch.setattr(module, "_run_numeric", lambda *a, **k: pytest.fail("fit must not start"))
    with pytest.raises(ContractError, match="complete healthy"):
        asyncio.run(train_experiments(loaded))
    assert not (tmp_path / "output").exists()


def test_synthetic_gate_controller_deduplicates_pilot_validation_cost_and_rows(tmp_path, make_input, monkeypatch):
    import asyncio

    import gf_training.lightgbm_run as module
    pilot_path = tmp_path / "synthetic-pilot"
    seeds = tuple(f"00000000-0000-0000-0000-{index:012d}" for index in range(1, 33))
    synthetic_run(pilot_path, make_input, split="validation", seeds=seeds, sampling=(17, 29, 43), decisions=False)
    train_path = tmp_path / "train"
    val_path = tmp_path / "validation"
    train_manifest = synthetic_run(train_path, make_input, seeds=("00000000-0000-0000-0000-000000000100",))
    train_manifest["code"] = {"revision": "collection-code-old", "dirty": False}
    (train_path / "manifest.json").write_text(canonical(train_manifest))
    synthetic_run(val_path, make_input, split="validation", seeds=("00000000-0000-0000-0000-000000000101",))
    loaded = load_config("configs/experiments/smoke.toml", {"model": {"model_id": "lightgbm"}, "training": {
        "pilot_run": str(pilot_path), "train_runs": [str(train_path)], "validation_runs": [str(val_path), str(pilot_path)],
        "accepted_decision_target": 2, "seeds": [17, 29, 43]}})
    calls = []
    async def fake_fit(directory, x, y, vx, vy, **options):
        calls.append((x, y, vx, vy, options))
        return {"status": "complete"}
    # This is a scheduling/gate unit test only: no trained artifact is produced.
    monkeypatch.setattr(module, "_run_numeric", fake_fit)
    import time
    original_provenance = module.provenance
    provenance_seconds = []
    def delayed_provenance(bundle, *, deadline):
        started = time.monotonic()
        time.sleep(.01)
        result = original_provenance(bundle, deadline=deadline)
        provenance_seconds.append(time.monotonic() - started)
        return result
    monkeypatch.setattr(module, "provenance", delayed_provenance)
    provider_calls = []
    def fit_code_provider(deadline):
        provider_calls.append(deadline)
        time.sleep(.02)
        return {"revision": "fit-code-current", "dirty": True, "installed_build_revision": None}
    result = asyncio.run(train_experiments(loaded, code_provenance_provider=fit_code_provider))
    assert len(provider_calls) == 1
    assert len(result["training_runs"]) == len(calls) == 3
    assert len({options["charged_seconds"] for _, _, _, _, options in calls}) == 1
    for x, y, vx, vy, options in calls:
        assert len(x) == len(y) == 2 and len(vx) == len(vy) == 2
        proof = options["provenance_data"]
        assert proof["historical_collection_seconds"] == 3  # pilot/validation charged once
        assert options["fit_code_provenance"]["revision"] == "fit-code-current"
        assert options["fit_code_measurement_seconds"] >= .02
        assert options["charged_seconds"] >= 3 + options["fit_code_measurement_seconds"] + sum(provenance_seconds)
        sources = proof["source_projection"]["manifests"]
        assert sources[train_manifest["run_id"]]["code"]["revision"] == "collection-code-old"
        assert proof["gate"]["split"] == "validation"
    too_small = replace(loaded, config=replace(loaded.config, training=replace(loaded.config.training, accepted_decision_target=1)))
    with pytest.raises(ContractError, match="target interval"):
        asyncio.run(train_experiments(too_small))
    assert len(calls) == 3


def test_training_projection_obeys_expired_original_deadline(tmp_path, make_input):
    synthetic_run(tmp_path / "run", make_input)
    bundle = load_run(tmp_path / "run")
    with pytest.raises(TimeoutError):
        validate_bundles([bundle], deadline=0)


def test_delayed_source_provenance_exhausts_original_deadline_before_any_fit(tmp_path, make_input, monkeypatch):
    import asyncio
    import time
    from dataclasses import asdict

    import gf_training.lightgbm_run as owner
    from gf_training.q_data import RunBundle, Sample

    loaded = load_config("configs/experiments/smoke.toml", {"model": {"model_id": "lightgbm"}, "training": {
        "pilot_run": str(tmp_path / "pilot"), "train_runs": [str(tmp_path / "train")],
        "validation_runs": [str(tmp_path / "validation")], "accepted_decision_target": 1,
        "wall_seconds": .1, "cleanup_seconds": .01}, "run": {"output_dir": str(tmp_path / "output")}})
    bundles = {name: RunBundle(tmp_path / name, {"run_id": name, "schema_version": 3}, (), 0) for name in ("pilot", "train", "validation")}
    config_hash = digest(asdict(loaded.config.actions))
    # Explicit scheduling/budget unit fixture; no real gate or native fit is run.
    monkeypatch.setattr(owner, "feasibility", lambda *a, **k: (bundles["pilot"], {"action_config_hash": config_hash}))
    monkeypatch.setattr(owner, "load_run", lambda path, *a, **k: bundles[path.name])
    monkeypatch.setattr(owner, "validate_bundles", lambda *a, **k: ([], {"sources": [{"header": {"action_config_hash": config_hash}}]}))
    value = make_input()
    action = {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}], "target": 2}
    monkeypatch.setattr(owner, "labeled_samples", lambda rows, evidence, split, **k: [Sample(value, action, 0., (split, "game", 1, 0))])
    deadlines = []
    def delayed(bundle, *, deadline):
        deadlines.append(deadline)
        if bundle.manifest["run_id"] == "pilot":
            time.sleep(.11)
        return {"run_id": bundle.manifest["run_id"]}
    monkeypatch.setattr(owner, "provenance", delayed)
    monkeypatch.setattr(owner, "_run_numeric", lambda *a, **k: pytest.fail("exhausted preparation must never start native fit"))
    with pytest.raises(TimeoutError, match="training ingestion deadline"):
        asyncio.run(train_experiments(loaded))
    assert len(deadlines) == 3 and len(set(deadlines)) == 1
    assert not (tmp_path / "output").exists()
