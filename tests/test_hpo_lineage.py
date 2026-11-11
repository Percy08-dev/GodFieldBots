"""Exact HPO structural lineage, without any live original Source capability."""
import time
from dataclasses import asdict

import pytest

pytest.importorskip("torch")

from gf_common.domain import ContractError, canonical, digest
from gf_models.arrays import atomic_json
from gf_training.structured_lineage import _ChainScan
from tests.hpo_lineage_support import (
    advanced_hpo_chain,
    hpo_state_fixture,
    initial_hpo_chain,
    prune_fixture_files,
    rehash_chain,
)


def test_exact_run5_parses_recipe_and_deadline_projection(tmp_path):
    from gf_training.hpo_lineage import hpo_spec, hpo_state
    state, model, recipe, _ = hpo_state_fixture(tmp_path / "parent")
    actions, actual, training = hpo_state(state)
    assert canonical(asdict(actual)) == canonical(asdict(model))
    assert hpo_spec(state)["training_recipe"] == asdict(recipe)
    assert training.seeds == (17,) and actions.max_sequence_length <= 18
    assert state["effective_config"]["run"]["wall_seconds"] > 7200


@pytest.mark.parametrize("mutation", ["old_schema", "unknown_spec", "spec_hash", "recipe", "settings_hash",
    "runtime_seed", "runtime_updates", "runtime_capacity", "deadline_duration", "absolute_deadline",
    "collected_alias", "used_alias", "formal_field", "failed", "bool_integer", "deadline_ratio",
    "action_capacity", "external_server", "runtime_bool_timer", "closed_bool_updates", "paused_zero", "originals_projection"])
def test_run5_tamper_rejected_after_rehashing_config(tmp_path, mutation):
    from gf_training.hpo_lineage import hpo_state
    state, *_ = hpo_state_fixture(tmp_path / "parent")
    config = state["effective_config"]
    spec = config["hpo_spec"]
    if mutation == "old_schema": state["schema_version"] = 4
    elif mutation == "unknown_spec": spec["surprise"] = 1
    elif mutation == "spec_hash": state["hpo_spec_sha256"] = "f" * 64
    elif mutation == "recipe": spec["training_recipe"]["gamma"] = .9
    elif mutation == "settings_hash": spec["effective_settings_sha256"] = "f" * 64
    elif mutation == "runtime_seed": config["training"]["seeds"] = [18]
    elif mutation == "runtime_updates": config["training"]["max_updates"] += 1
    elif mutation == "runtime_capacity": config["training"]["max_owned_bytes"] += 1
    elif mutation == "deadline_duration": config["run"]["admission_seconds"] += 1
    elif mutation == "absolute_deadline": state["wall_deadline_unix"] += 100
    elif mutation == "collected_alias": state.update(collected_decisions=0, used_decisions=1, accepted_decisions=1)
    elif mutation == "used_alias": state["accepted_decisions"] = 1
    elif mutation == "formal_field": state["formal_recipe"] = "fake"
    elif mutation == "failed": state.update(status="failed", quarantine=True)
    elif mutation == "bool_integer": spec["decision_budget"] = True
    elif mutation == "deadline_ratio": spec["deadline_inputs"]["base_admission_seconds"] = 7140
    elif mutation == "action_capacity": config["actions"]["max_candidates"] = config["model"]["inference"]["max_candidates"] + 1
    elif mutation == "external_server": config["server"]["external_url"] = "http://external"
    elif mutation == "runtime_bool_timer": config["run"]["report_reserve_seconds"] = True
    elif mutation == "closed_bool_updates": state.update(status="complete", stop_reason=None, updates=True,
        completed_updates=True, used_decisions=6, collected_decisions=6, accepted_decisions=6)
    elif mutation == "originals_projection": spec["storage"]["originals_root"] = "/different-originals"
    else: state.update(status="paused", stop_reason="pause", elapsed_seconds=0, artifact_bytes=0,
        invocations=[{"mode": "fresh", "elapsed_seconds": 0, "child_health": {"exit_code": 0, "stderr_bytes": 0}}])
    if mutation != "spec_hash": state["hpo_spec_sha256"] = digest(spec)
    state["config_sha256"] = digest(config)
    # Rebind identity too: these cases must fail the actual semantic guard.
    classification = {key: state[key] for key in ("action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature")}
    state["resume_identity"] = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
        "identity": state["identity"], "execution": state["execution"], "classification": classification, "storage_binding": None})
    with pytest.raises(ContractError): hpo_state(state)


def test_actual_saved_g0_chain_and_same_scan(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    expected, records, _ = initial_hpo_chain(directory)
    with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30) as scan:
        actual, chain = validate_hpo_chain(directory, max_owned_bytes=1024**3,
            deadline=scan.deadline, scan=scan)
        assert canonical(actual) == canonical(expected) and canonical(chain) == canonical(records)
        assert directory / "run.json" in scan.files
        assert not scan._closed
    with pytest.raises(ContractError):
        validate_hpo_chain(directory, require_complete=True, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


@pytest.mark.parametrize("mutation", ["missing_retained", "foreign_file", "registry_hash", "protected_deleted", "evaluation_ref"])
def test_registry4_per_file_and_protected_references_reject(tmp_path, mutation):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    initial_hpo_chain(directory)
    registry = read_json(directory / "artifact-registry.json")
    entry = registry["generations"]["0"]["files"]["resume.json"]
    if mutation == "missing_retained": (directory / "artifacts" / "000000" / "resume.json").unlink()
    elif mutation == "foreign_file": (directory / "artifacts" / "000000" / "foreign").write_text("foreign")
    elif mutation == "registry_hash": entry["sha256"] = "f" * 64
    elif mutation == "protected_deleted": entry["status"] = "deleted"
    else: registry["evaluation_references"]["evaluation-1"] = {"generation": 1, "inference_sha256": "f" * 64}
    atomic_json(directory / "artifact-registry.json", registry)
    with pytest.raises((ContractError, FileNotFoundError)):
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_readers_do_not_import_source_or_torch_in_clean_process():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", ("import sys; import gf_training.hpo_lineage; "
        "assert 'torch' not in sys.modules; "
        "assert 'gf_training.card_instance_source' not in sys.modules; "
        "assert 'gf_training.group_pack_game_source' not in sys.modules")], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_complete_actual_saved_cpu_chain_is_only_structural_evidence(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    expected, _, _ = advanced_hpo_chain(directory)
    state, records = validate_hpo_chain(directory, require_complete=True, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    assert state["used_decisions"] == state["accepted_decisions"] == 6
    assert state["collected_decisions"] == 6 and state["optimizer_steps"] == 8
    assert canonical(state) == canonical(expected)
    assert records[-1]["parent_controls"]["snapshot_pool"] == [1, 2]
    # Honest CPU metadata and absent original collections are observable. This
    # API does not establish a production CUDA run or complete-original Source.
    assert state["device"] == "cpu" and not list((directory / "collections").iterdir())


@pytest.mark.parametrize("budget,updates,collected", [(1, 1, 3), (4, 2, 6)])
def test_one_and_partial_final_budget_preserve_complete_collected_counts(tmp_path, budget, updates, collected):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory, updates=updates, budget=budget)
    state, _ = validate_hpo_chain(directory, require_complete=True, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    assert state["used_decisions"] == budget and state["collected_decisions"] == collected


@pytest.mark.parametrize("mutation", ["schema", "unknown", "used", "collected", "overshoot", "delta", "steps",
    "metrics_delta", "epochs", "endpoint_cost", "context_cost", "parity_count", "proof_qualified", "proof_schema",
    "proof_hash", "selection_order", "selection_seed", "selection_hash", "selection_context", "numeric_recipe",
    "numeric_schema", "numeric_decisions", "numeric_episodes", "numeric_selection", "numeric_policy", "numeric_outcome",
    "pool_duplicate", "generation_counts", "foreign_numeric", "numeric_corrupt"])
def test_update5_rehashed_semantic_tampering_rejects(tmp_path, mutation):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory)
    path = directory / "updates" / "000001.json"
    update = read_json(path)
    numeric_path = directory / update["numeric"] / "rollout.json"
    numeric = read_json(numeric_path)
    if mutation == "schema": update["schema_version"] = 4
    elif mutation == "unknown": update["synthetic_authority"] = True
    elif mutation == "used": update["used_decisions"] += 1
    elif mutation == "collected": update["collected_decisions"] += 1
    elif mutation == "overshoot": update["overshoot_decisions"] += 1
    elif mutation == "delta": update["expected_optimizer_step_delta"] += 1
    elif mutation == "steps": update["metrics"]["optimizer_steps"] += 1
    elif mutation == "metrics_delta": update["metrics"]["optimizer_step_delta"] += 1
    elif mutation == "epochs": update["metrics"]["epochs"] += 1
    elif mutation == "endpoint_cost": update["metrics"]["endpoint_evaluations"] += 1
    elif mutation == "context_cost": update["metrics"]["context_decisions"] = 0
    elif mutation == "parity_count": update["metrics"]["parity"]["decisions"] += 1
    elif mutation == "proof_qualified": update["source_proof"]["qualified"] = True
    elif mutation == "proof_schema": update["source_proof"]["schema_version"] = 2
    elif mutation == "proof_hash": update["source_proof"]["source_set_sha256"] = "f" * 64
    elif mutation == "selection_order": update["source_proof"]["selection"]["indices"].reverse()
    elif mutation == "selection_seed": update["source_proof"]["selection"]["derived_seed"] += 1
    elif mutation == "selection_hash": update["source_proof"]["selection"]["indices_sha256"] = "f" * 64
    elif mutation == "selection_context": update["source_proof"]["selection"]["context"]["used_before"] += 1
    elif mutation == "numeric_recipe": numeric["training_recipe"]["epochs"] += 1
    elif mutation == "numeric_schema": numeric["schema_version"] = 4
    elif mutation == "numeric_decisions": numeric["decision_count"] += 1
    elif mutation == "numeric_episodes": numeric["episodes"].pop()
    elif mutation == "numeric_selection": numeric["selection"]["used_decisions"] += 1
    elif mutation == "numeric_policy": numeric["policy_version"] = "other"
    elif mutation == "numeric_outcome": numeric["episodes"][0]["outcome"] = None
    elif mutation in ("pool_duplicate", "generation_counts"):
        generation_path = directory / "generations" / "000001.json"
        record = read_json(generation_path)
        if mutation == "pool_duplicate": record["parent_controls"]["snapshot_pool"] = [0, 1, 1]
        else: record["collected_decisions"] += 1
        atomic_json(generation_path, record)
    elif mutation == "foreign_numeric": (numeric_path.parent / "foreign.npz").write_bytes(b"foreign")
    else:
        name = next(name for name in numeric["data"]["files"] if name.endswith(".npz"))
        (numeric_path.parent / name).write_bytes(b"corrupt")
    atomic_json(path, update)
    atomic_json(numeric_path, numeric)
    rehash_chain(directory)
    with pytest.raises((ContractError, FileNotFoundError)):
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_real_natural_win_dto_is_accepted_structurally(tmp_path):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory)
    path = directory / "numeric" / "batch-000000" / "rollout.json"
    numeric = read_json(path)
    numeric["episodes"][0]["outcome"] = {"win": {"winner": 1}}
    atomic_json(path, numeric)
    rehash_chain(directory)
    validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_independent_inference_and_resume_pruning_keeps_historical_chain(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory, updates=4, budget=12)
    prune_fixture_files(directory, 1)
    prune_fixture_files(directory, 2, inference=False)
    state, records = validate_hpo_chain(directory, require_complete=True, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    assert state["updates"] == 4 and records[-1]["parent_controls"]["snapshot_pool"] == [3, 4]
    assert not list((directory / "artifacts" / "000001").iterdir())
    assert not (directory / "artifacts" / "000002" / "resume.json").exists()


@pytest.mark.parametrize("mutation", ["latest_resume", "g0", "pool", "evaluation", "receipt", "identity", "pending", "foreign_deleted"])
def test_retention_dependency_closure_and_per_file_receipts(tmp_path, mutation):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory, updates=4, budget=12)
    prune_fixture_files(directory, 1)
    if mutation == "latest_resume": prune_fixture_files(directory, 3, inference=False)
    elif mutation == "g0": prune_fixture_files(directory, 0)
    elif mutation == "pool": prune_fixture_files(directory, 3)
    registry = read_json(directory / "artifact-registry.json")
    entry = registry["generations"]["1"]["files"]["resume.json"]
    if mutation == "evaluation": registry["evaluation_references"]["eval-1"] = {"generation": 1,
        "inference_sha256": registry["generations"]["1"]["files"]["inference.json"]["sha256"]}
    elif mutation == "receipt": entry["deletion_receipt"]["sha256"] = "f" * 64
    elif mutation == "identity": entry["deletion_intent"]["original_identity"].pop()
    elif mutation == "pending": entry.update(status="deleting", deletion_receipt=None)
    elif mutation == "foreign_deleted": (directory / "artifacts" / "000001" / "resume.json").write_text("replacement")
    atomic_json(directory / "artifact-registry.json", registry)
    with pytest.raises((ContractError, FileNotFoundError)):
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_base_deadline_can_preserve_an_extra_reserve_gap(tmp_path):
    from gf_training.hpo_lineage import hpo_state
    state, *_ = hpo_state_fixture(tmp_path / "parent")
    config = state["effective_config"]
    config["hpo_spec"]["deadline_inputs"]["base_execution_seconds"] = 7000
    config["run"]["admission_seconds"] = 14340 * 5400 / 7000
    state["admission_deadline_unix"] = state["created_unix"] + config["run"]["admission_seconds"]
    state["hpo_spec_sha256"] = digest(config["hpo_spec"])
    state["config_sha256"] = digest(config)
    classification = {key: state[key] for key in ("action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature")}
    state["resume_identity"] = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
        "identity": state["identity"], "execution": state["execution"], "classification": classification, "storage_binding": None})
    hpo_state(state)


def test_completion_counter_rejects_bool_even_when_equal_to_one(tmp_path):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import hpo_state
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory, updates=1, budget=1)
    state = read_json(directory / "run.json")
    state["completed_updates"] = True
    with pytest.raises(ContractError): hpo_state(state)


@pytest.mark.parametrize("mutation", ["resource_model", "binding", "resume_identity", "steps", "rng_count", "recipe"])
def test_rehashed_saved_resume_must_match_full_run_contract(tmp_path, mutation):
    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    initial_hpo_chain(directory)
    path = directory / "artifacts" / "000000" / "resume.json"
    resume = read_json(path)
    if mutation == "resource_model": resume["model"]["resources"]["workspace_bytes"] += 1
    elif mutation == "binding": resume["binding"]["candidate_chunk"] += 1
    elif mutation == "resume_identity": resume["resume_identity"] = "f" * 64
    elif mutation == "steps": resume["optimizer_steps"] = 1
    elif mutation == "rng_count": resume["cuda_rng_count"] = 1
    else: resume["training_recipe"]["epochs"] += 1
    atomic_json(path, resume)
    generation_path = directory / "generations" / "000000.json"
    record = read_json(generation_path)
    record["resume_sha256"] = sha256(path)
    record["artifact_files"]["resume.json"] = sha256(path)
    atomic_json(generation_path, record)
    registry = read_json(directory / "artifact-registry.json")
    registry["generations"]["0"]["files"]["resume.json"]["sha256"] = sha256(path)
    atomic_json(directory / "artifact-registry.json", registry)
    rehash_chain(directory)
    with pytest.raises(ContractError):
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_numeric_ancestor_symlink_cannot_adopt_external_files(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    advanced_hpo_chain(directory)
    outside = tmp_path / "external-numeric"
    (directory / "numeric").rename(outside)
    (directory / "numeric").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError):
        validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


def test_cpu_structural_observation_does_not_enable_gpu_campaign(tmp_path):
    from gf_cli.hpo_config import campaign_config
    from gf_training.hpo_lineage import hpo_state
    from tests.test_hpo_config import campaign_raw
    state, *_ = hpo_state_fixture(tmp_path / "parent")
    assert hpo_state(state)[2].device == "cpu"
    raw = campaign_raw()
    raw["inference"] = state["execution"]["inference"]["binding"]
    with pytest.raises(ContractError, match="requires GPU"):
        campaign_config(raw, origin=tmp_path)


def test_old_game_lineage_cannot_accept_hpo_versions(tmp_path):
    from gf_training.structured_lineage import _validate_component_chain
    directory = tmp_path / "parent"
    initial_hpo_chain(directory)
    with (_ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30) as scan,
            pytest.raises(ContractError)):
        _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=True)


def test_state_timer_arithmetic_never_admits_unrepresentable_absolute_clocks(tmp_path):
    from gf_training.hpo_lineage import hpo_state
    state, *_ = hpo_state_fixture(tmp_path / "parent")
    state["created_unix"] = state["last_unix"] = 1e18
    for key, duration in (("wall_deadline_unix", 14400), ("execution_deadline_unix", 14340),
            ("admission_deadline_unix", 14340 * 5400 / 7140)):
        state[key] = state["created_unix"] + duration
    with pytest.raises(ContractError): hpo_state(state)


def test_explicit_packed_run5_is_structural_without_resolving_originals(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    directory = tmp_path / "parent"
    initial_hpo_chain(directory, packed=True)
    with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 30) as scan:
        state, _ = validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=scan.deadline, scan=scan)
        assert state["schema_version"] == 5 and "storage_binding" in state
        assert scan._packed_state is None and not scan._packed_owners


@pytest.mark.parametrize("marker", [None, "wrong", "model4;collection6;proof3;numeric5;selection1;provider1"])
def test_recorded_native_hpo_requires_its_own_marker_without_loading_binary(tmp_path, marker, monkeypatch):
    from gf_training import native_backend
    from gf_training.hpo_lineage import hpo_state
    state, *_ = hpo_state_fixture(tmp_path / "parent")
    state["effective_config"]["preprocessing"]["backend"] = "rust"
    classification = {key: state[key] for key in ("action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature")}
    build = {"card_instance_contract": native_backend.CARD_INSTANCE_CONTRACT,
        "card_instance_signature": state["layout_signature"], "card_instance_models": "mlp,gru,transformer",
        "abi": "cp314-gil", "native_source_sha256": "a" * 64}
    if marker is not None: build["card_instance_hpo"] = marker
    state["identity"]["preprocessing"] = {"backend": "rust", "reader_contract": "participant-structured-v2",
        "implementation": "rust-player-card-refs-v2", **classification, "binary": {"backend": "native",
            "extension_path": "/recorded/native.so", "extension_sha256": "a" * 64, "build": build,
            "python_abi": "cpython-314", "gil_disabled": False}}
    state["config_sha256"] = digest(state["effective_config"])
    state["resume_identity"] = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
        "identity": state["identity"], "execution": state["execution"], "classification": classification, "storage_binding": None})
    monkeypatch.setattr(native_backend, "load_native", lambda: pytest.fail("recorded metadata loaded native binary"))
    if marker in (None, "wrong"):
        with pytest.raises(ContractError): hpo_state(state)
    else: hpo_state(state)
