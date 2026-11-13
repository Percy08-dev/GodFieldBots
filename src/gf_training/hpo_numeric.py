"""Configurable endpoint PPO over the shared actor/critic and frozen parity path."""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, canonical, digest, integer
from gf_models.arrays import MAX_ARRAY_BYTES
from gf_models.hpo_config import validate_model_v4
from gf_models.structured_artifacts import _game_context_v4
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV4,
    architecture_identity,
    resource_estimate,
)
from gf_models.structured_network import make_network
from gf_models.torch_runtime import torch_runtime
from gf_training.hpo_config import HPOTrainingConfig, hpo_training_config
from gf_training.hpo_rollout import HpoRollout
from gf_training.structured_numeric import Learner
from gf_training.structured_rollout import StructuredRollout


def hpo_losses(logprobs: Any, values: Any, mask: Any, selected: Any, old: Any,
               advantage: Any, target: Any, *, training: HPOTrainingConfig) -> tuple[Any, Json]:
    import torch
    chosen = logprobs.gather(1, selected[:, None]).squeeze(1)
    ratio = (chosen - old).exp()
    policy = -torch.minimum(ratio * advantage,
        ratio.clamp(1 - training.clip_range, 1 + training.clip_range) * advantage).mean()
    mse = ((values - target) ** 2).mean()
    entropy = -(logprobs.exp() * logprobs.masked_fill(~mask, 0)).sum(-1).mean()
    loss = policy + training.value_coefficient * mse - training.entropy_coefficient * entropy
    return loss, {"policy_loss": float(policy.detach()), "value_mse": float(mse.detach()),
        "entropy": float(entropy.detach()), "approx_kl": float((old - chosen).mean().detach()),
        "total_loss": float(loss.detach())}


class HpoLearner(Learner):
    def __init__(self, model: ModelConfigV4, training: HPOTrainingConfig, seed: int,
                 binding: InferenceBindingV2, resume_identity: str, *, actions: ActionConfigV3,
                 rules: CardInstanceRules) -> None:
        if type(actions) is not ActionConfigV3 or type(training) is not HPOTrainingConfig:
            raise ContractError("closed HPO actions/training recipe required")
        training = hpo_training_config(asdict(training))
        self.action_config_hash = digest(asdict(actions))
        self._context = _game_context_v4(actions.action_contract, rules, self.action_config_hash)
        self._options = self._context.options
        validate_model_v4(model, **self._options)
        if (actions.max_sequence_length > model.max_positions
                or type(resume_identity) is not str or len(resume_identity) != 64
                or set(resume_identity) - set("0123456789abcdef")):
            raise ContractError("closed HPO learner recipe/actions/resume identity required")
        integer(seed, 0, 2**63 - 1)
        self.input_binding = self._context.classification()
        self.action_contract, self.purpose, self.synthetic_capacity = actions.action_contract, "qualified_game", False
        self.model, self.training, self.binding = model, training, binding
        self.resume_identity = resume_identity
        resource_estimate(model, binding, suffix_decisions=training.tbptt_length, **self._options)
        self.torch = torch_runtime(binding.device)
        self.torch.random.default_generator.manual_seed(seed)
        if binding.device != "cpu":
            self.torch.cuda.manual_seed_all(seed)
        self.network = make_network(model, seed, binding, **self._options)
        self._context.bind(self.network, fresh=True)
        self.optimizer = self._optimizer(self.network)
        self.shuffle = np.random.Generator(np.random.PCG64(seed))
        self.steps = 0

    def _optimizer(self, network: Any) -> Any:
        return self.torch.optim.Adam(network.parameters(), lr=self.training.adam_learning_rate,
            betas=(self.training.adam_beta1, self.training.adam_beta2), eps=self.training.adam_epsilon)

    def _suffix_length(self, decisions: int) -> int:
        return min(decisions, self.training.tbptt_length)

    def admit(self, rollout: StructuredRollout) -> None:
        if type(rollout) is not HpoRollout or type(rollout.model) is not ModelConfigV4:
            raise ContractError("HPO learner requires exact numeric5 component")
        self._context.bind(self.network)
        self._context.check(self.input_binding)
        self._context.check(rollout.input_binding)
        rollout.require_learning()
        if (architecture_identity(rollout.model, **self._options) != architecture_identity(self.model, **self._options)
                or rollout.action_config_hash != self.action_config_hash
                or canonical(asdict(rollout.training)) != canonical(asdict(self.training))):
            raise ContractError("HPO rollout model/actions/training recipe mismatch")
        # Whole ORIGINAL parity also needs unselected endpoints. Admit them all
        # before input packing, shuffle advancement or optimizer mutation.
        for episode in rollout.episodes:
            for position, candidates in enumerate(episode.tables):
                frames = episode.frames[:position + 1]
                size = resource_estimate(self.model, self.binding, decisions=len(frames),
                    interval_events=max(len(frame.interval_numerics) for frame in frames),
                    candidates=len(candidates.kinds), suffix_decisions=self._suffix_length(len(frames)), **self._options)["total_bytes"]
                if rollout.resident_bytes + size > min(self.model.resources.max_owned_bytes, self.binding.managed_budget_bytes):
                    raise ContractError("complete rollout plus endpoint exceeds learner reservation")

    def update(self, rollout: StructuredRollout) -> Json:
        started = time.perf_counter()
        parity = self.parity(rollout)
        assert type(rollout) is HpoRollout
        torch, recipe = self.torch, self.training
        totals: dict[str, float] = {}
        denominator = 0
        previous_steps = self.steps
        self.network.train()
        context_decisions = suffix_decisions = 0
        for _ in range(recipe.epochs):
            order = self.shuffle.permutation(len(rollout)).tolist()
            for start in range(0, len(order), recipe.minibatch_size):
                logical = order[start:start + recipe.minibatch_size]
                self.optimizer.zero_grad(set_to_none=True)
                for index in logical:
                    frames, candidates = rollout.endpoint(index)
                    suffix = self._suffix_length(len(frames))
                    logs, value, _ = self.network(frames, candidates, suffix_length=suffix)
                    selected, old, _, advantage, target = rollout.vector(index)
                    self.synchronize()
                    mask = torch.ones((1, len(candidates.kinds)), dtype=torch.bool, device=self.binding.device)
                    selected_tensor = torch.tensor([selected], dtype=torch.int64, device=self.binding.device)
                    vectors = [torch.tensor([number], dtype=torch.float32, device=self.binding.device)
                        for number in (old, advantage, target)]
                    loss, measured = hpo_losses(logs[None], value[None], mask, selected_tensor, *vectors, training=recipe)
                    if not torch.isfinite(loss):
                        raise ContractError("nonfinite HPO PPO loss")
                    (loss / len(logical)).backward()
                    for name, number in measured.items():
                        totals[name] = totals.get(name, 0.) + number
                    denominator += 1
                    context_decisions += len(frames)
                    suffix_decisions += suffix
                    del logs, value, loss
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), recipe.max_grad_norm, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ContractError("nonfinite HPO PPO gradients")
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        metrics = {name: value / denominator for name, value in totals.items()}
        expected = recipe.epochs * math.ceil(rollout.used_decisions / recipe.minibatch_size)
        if (self.steps - previous_steps != expected or not all(math.isfinite(value) for value in metrics.values())
                or any(not torch.isfinite(parameter).all() for parameter in self.network.parameters())):
            raise ContractError("invalid HPO optimizer step count/nonfinite output")
        return {**metrics, "optimizer_steps": self.steps, "optimizer_step_delta": expected,
            "collected_decisions": rollout.collected_decisions, "used_decisions": rollout.used_decisions,
            "parity": parity, "tbptt_length": recipe.tbptt_length, "epochs": recipe.epochs,
            "endpoint_evaluations": denominator, "context_decisions": context_decisions,
            "suffix_decisions": suffix_decisions, "update_seconds": time.perf_counter() - started}

    def save(self, directory: Path, *, code_revision: str | None = None,
             shard_bytes: int = MAX_ARRAY_BYTES, save_budget_bytes: int = 512*1024**2) -> Json:
        from gf_training.hpo_artifacts import save_learner
        return save_learner(self, directory, code_revision=code_revision, shard_bytes=shard_bytes,save_budget_bytes=save_budget_bytes)

    def restore(self, directory: Path) -> None:
        from gf_training.hpo_artifacts import restore_learner
        restore_learner(self, directory)
