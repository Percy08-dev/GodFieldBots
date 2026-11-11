"""Inference5 and recipe-sensitive resume5, with rehashed corruption tests."""
import copy
import subprocess
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError, canonical
from gf_models.arrays import atomic_json, sha256
from gf_models.sharded_arrays import read_bundle, read_json, write_bundle
from gf_models.structured_artifacts import (
    inspect_game_inference_v3,
    inspect_game_inference_v4,
    load_game_inference_v4,
)
from gf_models.structured_binding import inference_model, safe_options_game_v3, safe_options_game_v4
from gf_models.structured_sealed import SealedScan, inspect_retained_game_inference_v4
from gf_training.hpo_artifacts import read_resume
from gf_training.hpo_config import HPOTrainingConfig
from tests.test_card_instance_learning import ACTIONS, RULES
from tests.test_hpo_numeric import learner_for, rollout_for
from tests.test_structured_learning import assert_same_learners


def saved(tmp_path, *, recipe=None, trained=True, device="cpu"):
    learner = learner_for(recipe=recipe, device=device)
    if trained:
        learner.update(rollout_for(learner))
    directory = tmp_path / "artifact5"
    receipt = learner.save(directory, shard_bytes=524288)
    return learner, directory, receipt


def inference_args(learner, receipt):
    return {"expected_digest": receipt["inference_digest"], "model": learner.model,
        "binding": learner.binding, "action_config_hash": learner.action_config_hash,
        "action_contract": ACTIONS.action_contract, "rules": RULES}


def test_inference5_is_architecture_only_with_independent_loader(tmp_path):
    learner, directory, receipt = saved(tmp_path, trained=False)
    manifest = read_json(directory / "inference.json")
    assert manifest["schema_version"] == 5
    assert canonical(manifest["model"]) == canonical(inference_model(learner.model))
    assert "resources" not in manifest["model"]
    assert "sequence_length" not in manifest["model"]["temporal"]
    assert not ({"training_recipe", "optimizer_recipe", "binding", "reward_policy_version",
                 "tbptt_length", "evaluation", "campaign"} & manifest.keys())
    model = replace(learner.model, resources=replace(learner.model.resources, workspace_bytes=1000000))
    loaded, _ = load_game_inference_v4(directory, **{**inference_args(learner, receipt), "model": model})
    assert loaded.config.resources == model.resources
    for name, value in learner.network.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    code = """
import json,sys
from pathlib import Path
from gf_models.structured_artifacts import load_game_inference_v4
from gf_models.hpo_config import model_config_v4
from gf_models.structured_config import InferenceBindingV2
from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
data=json.loads(sys.argv[1]); actions=action_config_v3(data['actions'])
from gf_common.structured_layout import layout_for_contract
model=model_config_v4(data['model'],layout=layout_for_contract(actions.action_contract))
load_game_inference_v4(Path(data['directory']),expected_digest=data['digest'],model=model,
 binding=InferenceBindingV2(**data['binding']),action_config_hash=data['action_hash'],
 action_contract=actions.action_contract,rules=card_instance_rules())
assert not any(name=='gf_training' or name.startswith('gf_training.') for name in sys.modules)
"""
    payload = canonical({"directory": str(directory), "digest": receipt["inference_digest"],
        "model": asdict(learner.model), "binding": asdict(learner.binding),
        "actions": asdict(ACTIONS), "action_hash": learner.action_config_hash})
    result = subprocess.run([sys.executable, "-c", code, payload], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    scan = SealedScan(max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 20)
    inspected = inspect_retained_game_inference_v4(directory,
        expected_inference_digest=receipt["inference_digest"], actions=ACTIONS, rules=RULES, scan=scan)
    assert inspected.inference == manifest


def test_configured_adam_recipe_and_exact_resume_continue(tmp_path):
    recipe = replace(HPOTrainingConfig(), adam_learning_rate=.001, adam_beta1=.8,
        adam_beta2=.97, adam_epsilon=1e-6, epochs=2, minibatch_size=2, tbptt_length=1)
    continuous = learner_for(recipe=recipe)
    first = rollout_for(continuous)
    continuous.update(first)
    directory = tmp_path / "resume"
    receipt = continuous.save(directory, shard_bytes=524288)
    manifest = read_json(directory / "resume.json")
    assert manifest["schema_version"] == 5 and manifest["training_recipe"] == asdict(recipe)
    assert canonical(manifest["model"]) == canonical(asdict(continuous.model))
    assert manifest["optimizer_recipe"]["lr"] == .001
    assert manifest["optimizer_recipe"]["betas"] == [.8, .97]
    resumed = learner_for(recipe=recipe)
    resumed.restore(directory)
    assert_same_learners(continuous, resumed)
    next_rollout = rollout_for(continuous)
    continuous.update(next_rollout)
    resumed.update(next_rollout)
    assert_same_learners(continuous, resumed)
    wrong = learner_for(recipe=replace(recipe, tbptt_length=2))
    before = {name: value.clone() for name, value in wrong.network.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ContractError):
        wrong.restore(directory)
    assert torch.equal(rng, torch.get_rng_state())
    assert not wrong.optimizer.state and wrong.steps == 0
    assert all(torch.equal(value, wrong.network.state_dict()[name]) for name, value in before.items())
    assert receipt["resume_digest"] == sha256(directory / "resume.json")
    from gf_models.structured_artifacts import adam_recipe
    from gf_models.structured_sealed import inspect_retained_game_artifact_v4
    scanned = inspect_retained_game_artifact_v4(directory,
        expected_inference_digest=receipt["inference_digest"], actions=ACTIONS, rules=RULES,
        training_recipe=asdict(recipe), optimizer_recipe=adam_recipe(.001, .8, .97, 1e-6),
        scan=SealedScan(max_owned_bytes=512 * 1024**2, deadline=time.monotonic() + 20))
    assert scanned.resume == manifest


@pytest.mark.parametrize("corruption", ["negative_moment", "step", "missing_tensor", "cpu_rng", "pcg", "recipe", "adam"])
def test_rehashed_resume_corruption_atomic_rejection(tmp_path, corruption):
    _learner, directory, _ = saved(tmp_path)
    manifest = read_json(directory / "resume.json")
    arrays = {key: value.copy() for key, value in read_bundle(directory, manifest["state"]).items()}
    if corruption == "pcg":
        manifest["shuffle_rng"]["state"]["inc"] = 2
    elif corruption == "recipe":
        manifest["training_recipe"]["epochs"] += 1
    elif corruption == "adam":
        manifest["optimizer_recipe"]["lr"] = .001
    else:
        key = next(name for name in arrays if name.endswith("exp_avg_sq"))
        if corruption == "negative_moment":
            arrays[key].flat[0] = -1
        elif corruption == "step":
            arrays[next(name for name in arrays if name.endswith(".step"))] = np.asarray(0, np.int64)
        elif corruption == "missing_tensor":
            del arrays[key]
        else:
            arrays["rng.torch_cpu"] = np.ones(3, np.uint8)
        for name in manifest["state"]["files"]:
            (directory / name).unlink()
        manifest["state"] = write_bundle(directory, "resume", arrays, shard_bytes=524288)
    atomic_json(directory / "resume.json", manifest)
    pristine = learner_for()
    before = {name: value.clone() for name, value in pristine.network.state_dict().items()}
    shuffle = copy.deepcopy(pristine.shuffle.bit_generator.state)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ContractError):
        pristine.restore(directory)
    assert torch.equal(rng, torch.get_rng_state())
    assert canonical(shuffle) == canonical(pristine.shuffle.bit_generator.state)
    assert pristine.steps == 0 and not pristine.optimizer.state
    assert all(torch.equal(value, pristine.network.state_dict()[name]) for name, value in before.items())
    with pytest.raises(ContractError):
        read_resume(directory, pristine.model, pristine.binding, pristine.resume_identity,
            pristine.training, actions=ACTIONS, rules=RULES)


def test_missing_weight_and_legacy_format_not_reinterpreted(tmp_path):
    learner, directory, receipt = saved(tmp_path, trained=False)
    manifest = read_json(directory / "inference.json")
    manifest["schema_version"] = 4
    atomic_json(directory / "inference.json", manifest)
    with pytest.raises(ContractError):
        inspect_game_inference_v4(directory, **{**inference_args(learner, receipt),
            "expected_digest": sha256(directory / "inference.json")})
    with pytest.raises(ContractError):
        inspect_game_inference_v3(directory, **{**inference_args(learner, receipt),
            "expected_digest": sha256(directory / "inference.json")})
    manifest["schema_version"] = 5
    arrays = {key: value for key, value in read_bundle(directory, manifest["weights"]).items()}
    arrays.pop(next(iter(arrays)))
    for name in manifest["weights"]["files"]:
        (directory / name).unlink()
    manifest["weights"] = write_bundle(directory, "weights", arrays, shard_bytes=524288)
    atomic_json(directory / "inference.json", manifest)
    with pytest.raises(ContractError):
        inspect_game_inference_v4(directory, **{**inference_args(learner, receipt),
            "expected_digest": sha256(directory / "inference.json")})


def test_safe_model4_options_closed_and_old_parser_unchanged(tmp_path):
    learner, directory, receipt = saved(tmp_path, trained=False)
    options = {"version": "4", "checkpoint": str(directory),
        "expected_manifest_digest": receipt["inference_digest"], "model": asdict(learner.model),
        "inference": asdict(learner.binding), "cuda_visible_devices": None,
        "action_config_hash": learner.action_config_hash, **learner.input_binding}
    assert canonical(safe_options_game_v4(options, actions=ACTIONS, rules=RULES)) == canonical(options)
    with pytest.raises(ContractError):
        safe_options_game_v3(options, actions=ACTIONS, rules=RULES)
    for field, value in (("version", "3"), ("training_recipe", asdict(learner.training)),
                         ("action_config_hash", "0" * 64)):
        with pytest.raises(ContractError):
            safe_options_game_v4({**options, field: value}, actions=ACTIONS, rules=RULES)


def test_save_invalid_actual_optimizer_creates_no_directory(tmp_path):
    learner = learner_for()
    learner.optimizer.param_groups[0]["lr"] = .4
    with pytest.raises(ContractError):
        learner.save(tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_resource_free_metadata_accepts_custom_model_above_default_parameter_limit():
    from math import prod

    from gf_models.hpo_config import model_config_v4
    from gf_models.structured_artifacts import _game_context_v4
    from gf_models.structured_config import parameter_shapes
    from gf_models.structured_sealed import _game_sealed_context_v4
    from tests.model3_support import LAYOUT
    from tests.test_hpo_network import small_model4
    raw = asdict(small_model4())
    raw["state"].update(hidden_sizes=[1024, 1024, 1024], output_size=1024)
    raw["card"].update(output_size=512)
    raw["heads"].update(actor_hidden_sizes=[1024, 1024, 1024])
    raw["resources"]["max_parameters"] = 100000000
    model = model_config_v4(raw, layout=LAYOUT)
    assert sum(prod(shape) for shape in parameter_shapes(model, layout=LAYOUT).values()) > 16000000
    descriptor = inference_model(model)
    for context in (_game_context_v4(ACTIONS.action_contract, RULES, "a" * 64),
                    _game_sealed_context_v4(ACTIONS, RULES)):
        recorded = context.model(descriptor)
        assert inference_model(recorded) == descriptor


def configured_for(learner, directory):
    from gf_models.structured_binding import configured_model_v4
    from tests.model3_support import LAYOUT
    return configured_model_v4({**asdict(learner.model), "checkpoint": str(directory),
        "inference": asdict(learner.binding), "cuda_visible_devices": None}, layout=LAYOUT)


def retained(configured):
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    return RetainedGameArtifacts((configured,), ACTIONS, rules=RULES,
        max_owned_bytes=256 * 1024**2, preflight_deadline=time.monotonic() + 30,
        finalization_deadline=time.monotonic() + 60)


def test_model4_retained_owner_recorded_metadata_and_registry(tmp_path):
    from gf_models.card_instance_artifacts import RetainedGameArtifacts, validate_neural_metadata
    learner, directory, _ = saved(tmp_path, trained=False)
    configured = configured_for(learner, directory)
    owner = retained(configured)
    try:
        label, execution = owner.binding(configured)
        assert "-ppo-v4-" in label
        metadata = owner.journal_metadata(label)
        safe, recorded_label = validate_neural_metadata(metadata, actions=ACTIONS, rules=RULES)
        assert safe == execution.policy_options and label == recorded_label
        assert "resources" not in metadata["inference_manifest"]["model"]
        registry = RetainedGameArtifacts.for_registry({label: execution}, ACTIONS,
            rules=RULES, max_owned_bytes=256 * 1024**2, preflight_deadline=time.monotonic() + 20,
            finalization_deadline=time.monotonic() + 30)
        try:
            assert registry.binding(replace(configured, cuda_visible_devices="")) == (label, execution)
            registry.finalize(deadline=time.monotonic() + 15)
        finally:
            registry.release()
        bad = copy.deepcopy(metadata)
        bad["policy_options"]["version"] = "3"
        with pytest.raises(ContractError):
            validate_neural_metadata(bad, actions=ACTIONS, rules=RULES)
    finally:
        owner.release()


def test_actual_model4_worker_bootstrap_own_frame_and_metadata(tmp_path):
    import asyncio

    from gf_models.card_instance_artifacts import validate_neural_metadata
    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.test_card_instance_baseline_runtime import game
    learner, directory, _ = saved(tmp_path, trained=False)
    configured = configured_for(learner, directory)
    owner = retained(configured)
    label, execution = owner.binding(configured)
    frame, _, delta, _ = game()

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
            policy_options=execution.policy_options, artifact_inputs=owner)
        try:
            await worker.initialize()
            generated, selection = await worker.select_frame(frame, delta)
            assert selection.policy_version == label
            assert selection.logprob is not None and selection.value is not None
            selection.validate(generated.table, frame)
            assert validate_neural_metadata(worker.neural_metadata, actions=ACTIONS, rules=RULES)[1] == label
            assert worker.initial["policy_options"]["version"] == "4"
        finally:
            await worker.close()
        assert worker.process.poll() is not None and worker.lease.released
    try:
        asyncio.run(run())
    finally:
        owner.release()


@pytest.mark.parametrize("mutation", ["protocol", "options_version", "model_version"])
def test_actual_model4_bootstrap_rejects_wrong_header_types_and_old_versions(tmp_path, mutation):
    import asyncio

    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    learner, directory, _ = saved(tmp_path, trained=False)
    configured = configured_for(learner, directory)
    owner = retained(configured)
    _, execution = owner.binding(configured)

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
            policy_options=execution.policy_options, artifact_inputs=owner)
        if mutation == "protocol":
            worker.initial["protocol_version"] = True
        elif mutation == "options_version":
            worker.initial["policy_options"]["version"] = "3"
        else:
            worker.initial["policy_options"]["model"]["version"] = "3"
        try:
            with pytest.raises(RuntimeError, match="ContractError"):
                await worker.initialize()
        finally:
            await worker.close()
        assert worker.process.poll() is not None and worker.lease.released
    try:
        asyncio.run(run())
    finally:
        owner.release()


@pytest.mark.parametrize("mutation", ["parameter_order", "foreign_parameter", "group_option", "defaults_option"])
def test_adam_single_group_owned_order_and_execution_cannot_be_relabelled(tmp_path, mutation):
    learner = learner_for()
    if mutation == "parameter_order":
        learner.optimizer.param_groups[0]["params"].reverse()
    elif mutation == "foreign_parameter":
        learner.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(torch.ones(1))
    elif mutation == "group_option":
        learner.optimizer.param_groups[0]["foreach"] = True
    else:
        learner.optimizer.defaults["lr"] = .2
    with pytest.raises(ContractError):
        learner.save(tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
