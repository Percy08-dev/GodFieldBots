"""Controlled saved HPO chains; these fixtures grant no original Source authority."""
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gf_cli.config import CorrectedServerConfigV2, ExperimentConfig, RunConfigV2
from gf_cli.hpo_config import HPOOperatingConfig, HPOStorageConfig
from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
from gf_common.domain import digest
from gf_common.preprocessing import PreparationConfig
from gf_models.arrays import atomic_json, sha256
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV4, input_binding_game_v3
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_runtime import PROCESS_ENVELOPE_RECIPE
from gf_models.torch_runtime import hardware
from gf_training.hpo_config import HPOTrainingConfig, effective_settings
from gf_training.hpo_numeric import HpoLearner
from gf_training.lineage import checkpoint_path, generation_path
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES
from tests.test_hpo_network import small_model4


def hpo_state_fixture(directory: Path, *, decision_budget: int = 6, packed=False):
    model = small_model4()
    binding = InferenceBindingV2(device="cpu", max_candidates=64, candidate_chunk=2)
    recipe = replace(HPOTrainingConfig(), epochs=2, minibatch_size=2, collection_games=8)
    operating = replace(HPOOperatingConfig(), max_updates=4, source_budget_bytes=1024**3,
        inference_latest=1, pool_every_decisions=2, opponent_pool_size=2)
    storage = replace(HPOStorageConfig(), parent_root=str(directory.parent),
        numeric_root=str(directory / "numeric"), optuna_path=str(directory.parent / "optuna.sqlite3"))
    spec = {"schema_version": 1, "kind": "ppo-hpo-run", "decision_budget": decision_budget,
        "training_recipe": asdict(recipe),
        "effective_settings_sha256": effective_settings(model, recipe, {}, layout=LAYOUT).identity,
        "base_config_sha256": "b" * 64, "campaign_config_sha256": "c" * 64,
        "study_identity": "d" * 64, "operating": asdict(operating), "storage": asdict(storage),
        "selection_version": "hpo-endpoint-selection-v1", "deadline_version": "hpo-deadline-v1",
        "deadline_inputs": {"run_wall_seconds": 14400, "base_wall_seconds": 7200,
            "base_admission_seconds": 5400, "base_execution_seconds": 7140,
            "base_finalization_seconds": 60}}
    run = RunConfigV2(split="train", wall_seconds=14400, execution_seconds=14340,
        admission_seconds=14340 * 5400 / 7140)
    configured = ConfiguredNeuralModelV4(**{name: getattr(model, name) for name in model.__dataclass_fields__},
        inference=binding)
    config = asdict(ExperimentConfig())
    config.update(run=asdict(run), server=asdict(CorrectedServerConfigV2()),
        actions=asdict(ACTIONS), model=asdict(configured),
        training=asdict(PPOTrainingConfig(device="cpu", max_updates=4)),
        preprocessing=asdict(PreparationConfig(reader="participant-structured-v2", memory_budget_mib=1024)),
        hpo_spec=spec)
    storage_binding = None
    if packed:
        from gf_cli.config import GroupPackStorageConfig
        from gf_common.group_pack_game import storage_binding as make_storage_binding
        storage_binding = make_storage_binding()
        config["storage"] = asdict(GroupPackStorageConfig())
    server_config = directory.parent / "hpo-server.toml"
    server_config.write_text("[controlled_fixture]\n")
    config["server"]["config"] = str(server_config)
    classification = input_binding_game_v3(ACTIONS.action_contract, RULES)
    execution = {role: {"binding": asdict(binding), "cuda_visible_devices": ""}
        for role in ("learner", "inference")}
    identity = {"code": {"revision": None, "dirty": True},
        "process_envelope_recipe": PROCESS_ENVELOPE_RECIPE,
        "preprocessing": {"backend": "python", "reader_contract": "participant-structured-v2",
            "implementation": "python-reference-player-card-refs-v2", **classification},
        "integration": {"binary_sha256": receipt()["binary"]["sha256"],
            "server_config_sha256": sha256(server_config),
            "declared_simulator_revision": ACTIONS.action_contract.simulator_revision,
            "source_revision": ACTIONS.action_contract.simulator_revision,
            "source_spec_sha256": dict(CORRECTED_REVIEWED_SPEC_HASHES)}}
    resume = digest({"version": "hpo-ppo-resume-v1", "config_sha256": digest(config),
        "identity": identity, "execution": execution, "classification": classification, "storage_binding": storage_binding})
    now = time.time()
    state = {"schema_version": 5, "algorithm": "ppo", "run_id": "controlled-hpo-fixture",
        "status": "running", "run_variant": "hpo-ppo-v1", "hpo_spec_sha256": digest(spec),
        "effective_config": config, "config_sha256": digest(config), "config_inputs": [],
        "identity": identity, "seed": 17, "device": "cpu", "created_unix": now, "last_unix": now,
        "wall_deadline_unix": now + run.wall_seconds,
        "admission_deadline_unix": now + run.admission_seconds,
        "execution_deadline_unix": now + run.execution_seconds,
        "updates": 0, "accepted_decisions": 0, "collected_decisions": 0, "used_decisions": 0,
        "optimizer_steps": 0, "generation_sha256": None, "collection_seconds": 0., "update_seconds": 0.,
        "invocations": [], "completed_updates": None, "model_id": "mlp", "model_version": "4",
        "input_recipe": LAYOUT.input_recipe, "resume_identity": resume,
        "execution": execution, "hardware": hardware("cpu"), "stop_reason": None, **classification}
    if packed:
        state["storage_binding"] = storage_binding
    return state, model, recipe, binding


def initial_hpo_chain(directory: Path, *, decision_budget=6, packed=False):
    directory.mkdir()
    for name in ("artifacts", "generations", "updates", "collections", "numeric"):
        (directory / name).mkdir()
    state, model, recipe, binding = hpo_state_fixture(directory, decision_budget=decision_budget, packed=packed)
    learner = HpoLearner(model, recipe, 17, binding, state["resume_identity"], actions=ACTIONS, rules=RULES)
    saved = learner.save(checkpoint_path(directory, 0), shard_bytes=524288)
    classification = input_binding_game_v3(ACTIONS.action_contract, RULES)
    rng = np.random.Generator(np.random.PCG64(17))
    record = {"schema_version": 4, "run_id": state["run_id"], "generation": 0,
        "parent_sha256": None, "model_id": "mlp", "model_version": "4", "input_recipe": LAYOUT.input_recipe,
        "inference_sha256": saved["inference_digest"], "resume_sha256": saved["resume_digest"],
        "artifact_files": saved["files"], "accepted_decisions": 0, "collected_decisions": 0,
        "used_decisions": 0, "optimizer_steps": 0, "update_sha256": None,
        "hpo_spec_sha256": state["hpo_spec_sha256"],
        "parent_controls": {"schedule_opponent_rng": rng.bit_generator.state,
            "snapshot_pool": [0], "collection_position": 0}, **classification}
    atomic_json(generation_path(directory, 0), record)
    state["generation_sha256"] = sha256(generation_path(directory, 0))
    atomic_json(directory / "run.json", state)
    atomic_json(directory / "latest.json", {"generation": 0, "sha256": state["generation_sha256"]})
    registry = {"schema_version": 4, "generations": {"0": {"files": {name: {
        "sha256": checksum, "status": "retained", "deletion_intent": None, "deletion_receipt": None}
        for name, checksum in saved["files"].items()}}}, "evaluation_references": {}}
    atomic_json(directory / "artifact-registry.json", registry)
    return state, [record], learner


def advanced_hpo_chain(directory: Path, *, updates=2, budget=6, complete=True):
    """Actual learner saves with skeletal numeric facts, deliberately no originals."""
    import copy

    from gf_models.sharded_arrays import read_json, write_bundle
    from gf_models.structured_binding import policy_label_game_v4, safe_options_game_v4
    from gf_models.structured_config import architecture_identity
    from tests.test_hpo_numeric import rollout_for
    state, records, learner = initial_hpo_chain(directory, decision_budget=budget)
    spec = state["effective_config"]["hpo_spec"]
    recipe = learner.training
    classification = learner.input_binding
    registry = read_json(directory / "artifact-registry.json")
    pool = [0]
    for generation in range(1, updates + 1):
        previous = records[-1]
        used = min(3, budget - previous["used_decisions"])
        context = {"selection_version": spec["selection_version"], "decision_budget": budget,
            "used_before": previous["used_decisions"], "remaining": budget - previous["used_decisions"],
            "learning_seed": 17, "batch": generation - 1}
        seed = int(digest({"recipe": "hpo-ppo-selection-seed-v1", "context": context})[:32], 16)
        indices = list(range(3)) if used == 3 else sorted(np.random.Generator(np.random.PCG64(seed)).choice(3, size=used, replace=False).tolist())
        selection = {"context": context, "derived_seed": seed, "episode_counts": [3, *([0] * 7)],
            "collected_decisions": 3, "used_decisions": used, "indices": indices, "indices_sha256": digest(indices)}
        rollout = rollout_for(learner, selected=tuple((0, index) for index in indices))
        metrics = learner.update(rollout)
        saved = learner.save(checkpoint_path(directory, generation), shard_bytes=524288)
        batch = f"batch-{generation - 1:06}"
        numeric = directory / "numeric" / batch
        numeric.mkdir()
        inference = read_json(checkpoint_path(directory, generation - 1) / "inference.json")
        semantics = {key: value for key, value in inference.items() if key in (
            "input_recipe", "generator_version", "policy_input_version", "end_projection_version", "reviewed_spec_hashes",
            "simulator_revision", "catalog", "token_signature", "numeric_transform", "runtime", "numerics", "rule_signature", "spec_inventory_sha256")}
        label = policy_label_game_v4(learner.model, learner.binding, previous["inference_sha256"],
            action_contract=ACTIONS.action_contract, rules=RULES, action_config_hash=learner.action_config_hash,
            cuda_visible_devices="")
        policies = {"random-card-instance-v2": {"model_id": "random", "policy_options": None},
            "heuristic-card-instance-v2": {"model_id": "heuristic", "policy_options": None}}
        for number in [generation - 1, *previous["parent_controls"]["snapshot_pool"]]:
            sha = records[number]["inference_sha256"]
            policy_label = policy_label_game_v4(learner.model, learner.binding, sha,
                action_contract=ACTIONS.action_contract, rules=RULES, action_config_hash=learner.action_config_hash,
                cuda_visible_devices="")
            policies[policy_label] = {"model_id": "mlp", "policy_options": safe_options_game_v4({
                "version": "4", "checkpoint": str(checkpoint_path(directory, number)),
                "expected_manifest_digest": sha, "model": asdict(learner.model), "inference": asdict(learner.binding),
                "cuda_visible_devices": "", "action_config_hash": learner.action_config_hash, **classification},
                actions=ACTIONS, rules=RULES)}
        # Typed bundle bytes are real; full token/raw-array correspondence is
        # intentionally absent and only a real original Source may qualify it.
        data = write_bundle(numeric, "rollout", {"controlled": np.asarray(indices, dtype=np.int64)})
        episodes = [{"actor": 1, "episode": f"episode-{index}", "frames": [{}] * count,
            "selections": [{}] * count, "end_kind": "natural", "outcome": "draw", "reason": None,
            "has_targets": bool(count), **classification} for index, count in enumerate(selection["episode_counts"])]
        manifest = {"schema_version": 5, "kind": "structured-rollout", **classification, **semantics,
            "architecture_identity": architecture_identity(learner.model, layout=LAYOUT),
            "action_config_hash": learner.action_config_hash, "policy_version": label,
            "decision_count": used, "episodes": episodes, "data": data, "training_recipe": asdict(recipe),
            "selection": selection, "expected_optimizer_step_delta": metrics["optimizer_step_delta"]}
        atomic_json(numeric / "rollout.json", manifest)
        numeric_sha = sha256(numeric / "rollout.json")
        proof = {"schema_version": 3, "kind": "card-instance-hpo-original-source-proof", **classification,
            "qualified": False, "parent_qualification": "not_performed", "manifest_sha256": "e" * 64,
            "source_set_sha256": digest({}), "files": {}, "games": 8, "accepted_decisions": 3,
            "closure_counts": {"natural": 8}, "missing_replays": [], "selection": selection,
            "expected_optimizer_step_delta": metrics["optimizer_step_delta"]}
        update = {"schema_version": 5, "status": "complete", "run_id": state["run_id"], "generation": generation,
            "parent_sha256": state["generation_sha256"], "batch_id": batch, "collection": "collections/" + batch,
            "collection_manifest_sha256": proof["manifest_sha256"], "numeric": "numeric/" + batch,
            "numeric_sha256": numeric_sha, "accepted_decisions": used, "collected_decisions": 3,
            "used_decisions": used, "overshoot_decisions": 3 - used,
            "expected_optimizer_step_delta": metrics["optimizer_step_delta"], "source_proof": proof,
            "metrics": metrics, "planned_games": 8, "actual_games": 8, "health": {},
            "policy_bindings": policies, "learner_seats": {f"episode-{index}": 1 for index in range(8)},
            "collection_seconds": .1, "update_seconds": .2, "model_id": "mlp", "model_version": "4",
            "input_recipe": LAYOUT.input_recipe, "numeric_files": {"rollout.json": numeric_sha, **data["files"]},
            "hpo_spec_sha256": state["hpo_spec_sha256"], **classification}
        atomic_json(directory / "updates" / f"{generation:06}.json", update)
        state["used_decisions"] += used
        state["accepted_decisions"] += used
        state["collected_decisions"] += 3
        state["optimizer_steps"] = learner.steps
        state["collection_seconds"] += .1
        state["update_seconds"] += .2
        if state["used_decisions"] // spec["operating"]["pool_every_decisions"] > previous["used_decisions"] // spec["operating"]["pool_every_decisions"]:
            pool = (pool + [generation])[-spec["operating"]["opponent_pool_size"]:]
        record = {**copy.deepcopy(previous), "generation": generation, "parent_sha256": state["generation_sha256"],
            "inference_sha256": saved["inference_digest"], "resume_sha256": saved["resume_digest"],
            "artifact_files": saved["files"], "accepted_decisions": state["used_decisions"],
            "used_decisions": state["used_decisions"], "collected_decisions": state["collected_decisions"],
            "optimizer_steps": learner.steps, "update_sha256": sha256(directory / "updates" / f"{generation:06}.json"),
            "parent_controls": {**copy.deepcopy(previous["parent_controls"]), "snapshot_pool": pool[:], "collection_position": generation}}
        records.append(record)
        atomic_json(generation_path(directory, generation), record)
        state.update(updates=generation, generation_sha256=sha256(generation_path(directory, generation)))
        registry["generations"][str(generation)] = {"files": {name: {"sha256": checksum,
            "status": "retained", "deletion_intent": None, "deletion_receipt": None}
            for name, checksum in saved["files"].items()}}
    if complete:
        state.update(status="complete", completed_updates=updates, elapsed_seconds=0., artifact_bytes=0,
            invocations=[{"mode": "fresh", "elapsed_seconds": .3, "child_health": {"exit_code": 0, "stderr_bytes": 0}}])
    atomic_json(directory / "run.json", state)
    atomic_json(directory / "latest.json", {"generation": updates, "sha256": state["generation_sha256"]})
    atomic_json(directory / "artifact-registry.json", registry)
    return state, records, learner


def rehash_chain(directory: Path):
    """Rehash all links after semantic tampering, so tests reach real guards."""
    from gf_models.sharded_arrays import read_json
    state = read_json(directory / "run.json")
    parent = None
    for number in range(state["updates"] + 1):
        path = generation_path(directory, number)
        record = read_json(path)
        record["parent_sha256"] = parent
        if number:
            update_path = directory / "updates" / f"{number:06}.json"
            update = read_json(update_path)
            update["parent_sha256"] = parent
            numeric = directory / update["numeric"] / "rollout.json"
            update["numeric_sha256"] = sha256(numeric)
            update["numeric_files"]["rollout.json"] = update["numeric_sha256"]
            atomic_json(update_path, update)
            record["update_sha256"] = sha256(update_path)
        atomic_json(path, record)
        parent = sha256(path)
    state["generation_sha256"] = parent
    atomic_json(directory / "run.json", state)
    atomic_json(directory / "latest.json", {"generation": state["updates"], "sha256": parent})


def prune_fixture_files(directory: Path, generation: int, *, inference=True):
    """Simulate already completed owned deletion; never used by production."""
    from gf_models.sharded_arrays import read_json
    from gf_training.structured_lineage import _stat_identity
    registry = read_json(directory / "artifact-registry.json")
    files = registry["generations"][str(generation)]["files"]
    for name, entry in files.items():
        if not inference and (name == "inference.json" or name.startswith("weights-")):
            continue
        path = checkpoint_path(directory, generation) / name
        identity = list(_stat_identity(path.stat()))
        intent = {"operation_id": "fixture-delete", "staging_basename": "prune-" + "a" * 32,
            "sha256": entry["sha256"], "original_identity": identity}
        entry.update(status="deleted", deletion_intent=intent,
            deletion_receipt={key: value for key, value in intent.items() if key != "staging_basename"}
                | {"completed_unix": time.time()})
        path.unlink()
    atomic_json(directory / "artifact-registry.json", registry)
