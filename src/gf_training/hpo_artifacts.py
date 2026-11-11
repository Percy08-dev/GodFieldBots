"""Recipe-aware resume5 ownership; shared safe model/shard/Adam/RNG mechanics."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, digest
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.structured_artifacts import (
    _game_context_v4,
    _read_resume,
    _restore_resume,
    _save_resume,
    adam_recipe,
    inspect_game_inference_v4,
    load_game_inference_v4,
    save_game_inference_v4,
    validate_save_context,
)
from gf_models.structured_config import InferenceBindingV2, ModelConfigV4, resource_estimate
from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config

if TYPE_CHECKING:
    from gf_training.hpo_numeric import HpoLearner


def _recipe_options(training: HPOTrainingConfig) -> dict[str, Any]:
    if type(training) is not HPOTrainingConfig:
        raise ContractError("resume5 requires the complete closed HPO recipe")
    training = hpo_training_config(asdict(training))
    return {"_optimizer_recipe": adam_recipe(training.adam_learning_rate,
        training.adam_beta1, training.adam_beta2, training.adam_epsilon),
        "_resume_metadata": {"training_recipe": asdict(training)}}


def read_resume(directory: Path, model: ModelConfigV4, binding: InferenceBindingV2,
                resume_identity: str, training: HPOTrainingConfig, *, actions: ActionConfigV3,
                rules: CardInstanceRules) -> tuple[Json, dict[str, Any]]:
    return _read_resume(directory, model, binding, resume_identity,
        action_contract=actions.action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v4(actions.action_contract, rules, digest(asdict(actions))),
        **_recipe_options(training))


def save_learner(learner: HpoLearner, directory: Path, *, code_revision: str | None = None,
                 shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    options = _recipe_options(learner.training)
    learner._context.bind(learner.network)
    learner._context.check(learner.input_binding)
    # Validate actual optimizer and network BEFORE inference creates any file.
    validate_save_context(learner.network, learner.model, learner.binding, learner.optimizer,
        _optimizer_recipe=options["_optimizer_recipe"], **learner._options)
    learner.synchronize()
    assert learner._context.rules is not None
    manifest = save_game_inference_v4(directory, learner.network, learner.model, learner.action_config_hash,
        action_contract=learner.action_contract, rules=learner._context.rules,
        code_revision=code_revision, shard_bytes=shard_bytes)
    resume = _save_resume(directory, learner.network, learner.optimizer, learner.shuffle, learner.steps,
        learner.model, learner.binding, learner.resume_identity, shard_bytes=shard_bytes,
        action_contract=learner.action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=learner._context, **options)
    return {"inference_digest": sha256(directory / "inference.json"),
        "resume_digest": sha256(directory / "resume.json"),
        "files": {**manifest["weights"]["files"], **resume["state"]["files"],
            "inference.json": sha256(directory / "inference.json"), "resume.json": sha256(directory / "resume.json")}}


def restore_learner(learner: HpoLearner, directory: Path) -> None:
    learner._context.bind(learner.network)
    learner._context.check(learner.input_binding)
    options = _recipe_options(learner.training)
    # Retain the live owner until a full staged network/optimizer/RNG validates.
    tensors = [*learner.network.parameters(), *learner.network.buffers()]
    tensors.extend(parameter.grad for parameter in learner.network.parameters() if parameter.grad is not None)
    tensors.extend(value for state in learner.optimizer.state.values() for value in state.values()
        if isinstance(value, learner.torch.Tensor))
    existing = sum(value.numel() * value.element_size() for value in tensors)
    staged = resource_estimate(learner.model, learner.binding,
        suffix_decisions=learner.training.tbptt_length, **learner._options)["total_bytes"]
    if existing + staged > min(learner.model.resources.max_owned_bytes, learner.binding.managed_budget_bytes):
        raise ContractError("staged resume5 plus live learner exceeds managed reservation")
    assert learner._context.rules is not None
    inference_options: dict[str, Any] = {"model": learner.model, "binding": learner.binding,
        "action_config_hash": learner.action_config_hash, "action_contract": learner.action_contract,
        "rules": learner._context.rules, "expected_digest": sha256(directory / "inference.json")}
    inspect_game_inference_v4(directory, **inference_options)
    _read_resume(directory, learner.model, learner.binding, learner.resume_identity,
        action_contract=learner.action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=learner._context, **options)
    network, _ = load_game_inference_v4(directory, **inference_options)
    optimizer = learner._optimizer(network)
    # Shuffle is staged as well; _restore_resume publishes global RNG only after
    # every array and all temporary Adam state has passed validation/allocation.
    import numpy as np
    shuffle = np.random.Generator(np.random.PCG64(0))
    steps = _restore_resume(directory, network, optimizer, shuffle, learner.model, learner.binding,
        learner.resume_identity, action_contract=learner.action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=learner._context, **options)
    learner.network, learner.optimizer, learner.shuffle, learner.steps = network, optimizer, shuffle, steps
