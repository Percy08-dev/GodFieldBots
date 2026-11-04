"""Explicit PPO reader and frozen collection linkage, never Source authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli import card_instance as cli
from gf_cli.config import load_config
from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import game_binding
from gf_common.domain import ContractError, EpisodeGroup, EpisodeSpec, RuntimePolicyBinding, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.structured_binding import policy_label_game_v3
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training import card_instance_collection as records
from gf_training.collection import CollectionPlan

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mutation", [None, "server", "model", "concurrency", "inputs", "effective", "sha"])
def test_online_collection_binds_caller_config_before_runtime(monkeypatch, mutation):
    import asyncio

    from gf_training import card_instance_source

    original = load_config(ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml")
    owner = SimpleNamespace(state={"config_sha256": original.sha256,
        "effective_config": deepcopy(original.effective), "config_inputs": list(original.inputs)})
    monkeypatch.setattr(card_instance_source, "validate_owned_collection", lambda *args, **kwargs: owner)
    plan = SimpleNamespace(execution_deadline=float("inf"))
    ledger = ReservationLedger(original.config.run.global_cuda_budget_bytes,
                               original.config.run.max_participant_workers)
    if mutation is None:
        assert cli._validate_online_collection(original, plan, ledger, float("inf")) is owner
        return
    loaded = original
    if mutation in ("server", "model", "concurrency"):
        config = original.config
        if mutation == "server":
            config = replace(config, server=replace(config.server, config="different-server.toml"))
        elif mutation == "model":
            config = replace(config, model=replace(config.model,
                event=replace(config.model.event, aggregate_window=1)))
        else:
            config = replace(config, run=replace(config.run, concurrency=1))
        effective = asdict(config)
        loaded = replace(original, config=config, effective=effective, sha256=digest(effective))
    elif mutation == "inputs":
        loaded = replace(original, inputs=original.inputs[:-1])
    elif mutation == "effective":
        effective = deepcopy(original.effective)
        effective["run"]["concurrency"] = 1
        loaded = replace(original, effective=effective)
    elif mutation == "sha":
        loaded = replace(original, sha256="0" * 64)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime allocation preceded frozen Config refusal")

    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", forbidden)
    with pytest.raises(ContractError, match="caller differs from SAME parent"):
        asyncio.run(cli.collect_ppo(loaded, plan, ledger))


def test_native_collection_refuses_old_capabilities_before_runtime(monkeypatch):
    import asyncio

    from gf_training import native_backend

    loaded = load_config(ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml",
                         {"preprocessing": {"backend": "rust"}})
    # Isolate the next failure boundary after the separately tested live-owner
    # guard: an old extension cannot reach runtime or artifact allocation.
    monkeypatch.setattr(cli, "_validate_online_collection", lambda *args: object())
    old = SimpleNamespace(build_info=lambda: {"structured_contract": "structured-v1"})
    monkeypatch.setattr(native_backend, "load_native", lambda: old)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime allocation preceded native capability refusal")

    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", forbidden)
    plan = SimpleNamespace(execution_deadline=float("inf"))
    ledger = ReservationLedger(loaded.config.run.global_cuda_budget_bytes,
                               loaded.config.run.max_participant_workers)
    with pytest.raises(ContractError, match="native binary lacks accepted card-instance"):
        asyncio.run(cli.collect_ppo(loaded, plan, ledger))


@pytest.mark.parametrize("name", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("backend", ["python", "rust"])
def test_explicit_game_reader_acceptance_config(name, backend):
    loaded = load_config(ROOT / f"configs/experiments/card-instance-{name}-ppo-acceptance.toml",
                         {"preprocessing": {"backend": backend}})
    assert loaded.config.preprocessing.backend == backend
    assert loaded.config.preprocessing.reader == "participant-structured-v2"
    assert loaded.config.model.checkpoint is None
    assert loaded.config.training.max_updates == 2
    assert loaded.config.run.split == "train"
    assert (loaded.config.run.wall_seconds, loaded.config.run.admission_seconds,
            loaded.config.run.execution_seconds) == (7200, 5400, 7140)


@pytest.mark.parametrize("preset,override", [
    ("card-instance-mlp-ppo-acceptance.toml", {"preprocessing": {"backend": "rust", "reader": "legacy"}}),
    ("card-instance-v2-corrected-contract.toml", {"preprocessing": {"reader": "participant-structured-v2"}}),
    ("ppo-cpu.toml", {"preprocessing": {"reader": "participant-structured-v2"}}),
])
def test_game_reader_does_not_relax_old_or_unsupported_routes(preset, override):
    with pytest.raises(ContractError):
        load_config(ROOT / "configs/experiments" / preset, override)


@pytest.mark.parametrize("mutation", [None, "missing", "extra", "registry", "schedule", "seats",
                                      "relative", "batch", "boolean_generation", "duplicate_episode"])
def test_ppo_linkage_matches_exact_frozen_schedule_registry_and_seats(tmp_path, monkeypatch, mutation):
    schedule = [{"seed": "seed", "episodes": [
        {"episode_id": "seat0", "learner_viewer": 1},
        {"episode_id": "seat1", "learner_viewer": 2},
    ]}]
    registry = {"learner": {"model_id": "mlp", "policy_options": {"manifest": "frozen"}}}
    base = {"schedule": schedule, "policy_bindings": registry,
            "source_eligibility": "not-authorized-by-collection-metadata"}
    linkage = {"parent_run_id": "parent", "parent_run_path": str(tmp_path.resolve()),
        "batch_id": "batch-000000", "collection_generation": 0,
        "learner_seats": {"seat0": 1, "seat1": 2},
        "schedule_digest": digest(schedule), "policy_registry_digest": digest(registry)}
    if mutation == "missing":
        linkage.pop("policy_registry_digest")
    elif mutation == "extra":
        linkage["qualified"] = True
    elif mutation in ("registry", "schedule"):
        linkage["policy_registry_digest" if mutation == "registry" else "schedule_digest"] = "0" * 64
    elif mutation == "seats":
        linkage["learner_seats"] = {"seat0": 2, "seat1": 1}
    elif mutation == "relative":
        linkage["parent_run_path"] = "relative-parent"
    elif mutation == "batch":
        linkage["batch_id"] = "batch-000001"
    elif mutation == "boolean_generation":
        linkage["collection_generation"] = False
    elif mutation == "duplicate_episode":
        schedule[0]["episodes"][1]["episode_id"] = "seat0"
        linkage["schedule_digest"] = digest(schedule)
    # The shared neural/runtime producer is independently tested; this test
    # isolates the extra original-linkage validation, without faking a Source.
    monkeypatch.setattr(records, "collection_metadata", lambda **kwargs: deepcopy(base))
    args = {"actions": ActionConfigV3(action_contract=corrected_card_instance_contract()),
        "config": {}, "runtime_evidence": [], "schedule": schedule, "inventory": [],
        "policies": ["learner"], "bots_revision": "0" * 40,
        "policy_bindings": {}, "neural_inputs": {}, "linkage": linkage}
    if mutation is None:
        value = records.ppo_collection_metadata(**args)
        assert value["collection_role"] == "ppo-batch" and value["linkage"] == linkage
        assert value["source_eligibility"] == "not-authorized-by-collection-metadata"
        linkage["learner_seats"]["seat0"] = 2
        assert value["linkage"]["learner_seats"]["seat0"] == 1
    else:
        with pytest.raises(ContractError):
            records.ppo_collection_metadata(**args)


def _collection_fixture(tmp_path):
    loaded = load_config(ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml")
    actions, configured = loaded.config.actions, loaded.config.model
    model = configured.architecture(layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    rules = card_instance_rules()
    options = {"version": "3", "checkpoint": str(tmp_path / "checkpoint"),
        "expected_manifest_digest": "a" * 64, "model": asdict(model),
        "inference": asdict(configured.inference), "cuda_visible_devices": "",
        "action_config_hash": digest(asdict(actions)),
        **game_binding(actions.action_contract, rule_signature(actions.action_contract, rules))}
    label = policy_label_game_v3(model, configured.inference, "a" * 64,
        action_contract=actions.action_contract, rules=rules,
        action_config_hash=digest(asdict(actions)), cuda_visible_devices="")
    bindings = {label: RuntimePolicyBinding("mlp", options),
                "random-card-instance-v2": RuntimePolicyBinding("random")}
    groups = (EpisodeGroup("00000000-0000-0000-0000-000000000001",
        (EpisodeSpec("episode", (label, "random-card-instance-v2"), (17, 29), "pair", 1),)),)
    plan = CollectionPlan(groups, bindings, tmp_path / "owned-batch", {}, float("inf"), float("inf"))
    return loaded, plan, ReservationLedger(12 * 1024**3, 4), label


def test_copied_plan_cannot_reach_runtime_or_output_without_live_owner(tmp_path, monkeypatch):
    import asyncio

    loaded, plan, ledger, _ = _collection_fixture(tmp_path)

    def reject(*args):
        raise ContractError("same live owner required")

    monkeypatch.setattr(cli, "_validate_online_collection", reject)
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: pytest.fail("runtime reached"))
    with pytest.raises(ContractError, match="same live owner"):
        asyncio.run(cli.collect_ppo(loaded, plan, ledger))
    assert not plan.output.exists()


@pytest.mark.parametrize("neural_count", [1, 3, 6])
@pytest.mark.parametrize("rebind", [False, True])
def test_collector_preserves_frozen_registry_plan_and_borrowed_ledger(tmp_path, monkeypatch, rebind, neural_count):
    import asyncio

    import gf_models.card_instance_artifacts as artifacts
    import gf_models.structured_runtime as runtime

    loaded, plan, ledger, label = _collection_fixture(tmp_path)
    for index in range(1, neural_count):
        plan.bindings[f"frozen-snapshot-{index}"] = deepcopy(plan.bindings[label])
    released, checked = [], []
    owner = SimpleNamespace(release=lambda: released.append(True))
    monkeypatch.setattr(cli, "_validate_online_collection", lambda *args: SimpleNamespace(live=True))
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: SimpleNamespace(preflight=lambda: None))
    def from_registry(bindings, actions, **kwargs):
        assert bindings is plan.bindings
        assert sum(value.policy_options is not None for value in bindings.values()) == neural_count
        checked.append(bindings)
        if rebind:
            raise ContractError("current artifact cannot rebind frozen registry")
        return owner

    monkeypatch.setattr(artifacts, "RetainedGameArtifacts", SimpleNamespace(for_registry=from_registry))

    def admit(actions, roles, **kwargs):
        assert kwargs["learner_cuda_bytes"] == ledger.cuda_bytes
        assert roles == (plan.bindings[label], plan.bindings["random-card-instance-v2"])
        return {"peak_cuda_bytes": 0}

    async def execute(actual_loaded, command, bindings, labels, **kwargs):
        assert actual_loaded is loaded and command == "collect" and bindings is plan.bindings
        assert kwargs["collection"] is plan and kwargs["owned_ledger"] is ledger
        assert kwargs["artifact_inputs"] is owner
        return plan.output, {"comparison_valid": True}

    monkeypatch.setattr(runtime, "admit_game_workers_v3", admit)
    monkeypatch.setattr(cli, "_run_corrected", execute)
    if rebind:
        with pytest.raises(ContractError, match="cannot rebind"):
            asyncio.run(cli.collect_ppo(loaded, plan, ledger))
        assert checked == [plan.bindings]
    else:
        result = asyncio.run(cli.collect_ppo(loaded, plan, ledger))
        assert result.directory == plan.output and result.report["comparison_valid"]
        assert checked == [plan.bindings]
    assert released == ([] if rebind else [True]) and not plan.output.exists()
