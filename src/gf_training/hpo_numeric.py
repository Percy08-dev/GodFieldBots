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
               advantage: Any, target: Any, *, training: HPOTrainingConfig,
               _endpoint_terms: bool = False, _measure_clip: bool = False) -> tuple[Any, Any]:
    import torch
    reduce = (lambda value: value) if _endpoint_terms else (lambda value: value.mean())
    chosen = logprobs.gather(1, selected[:, None]).squeeze(1)
    ratio = (chosen - old).exp()
    policy = -reduce(torch.minimum(ratio * advantage,
        ratio.clamp(1 - training.clip_range, 1 + training.clip_range) * advantage))
    mse = reduce((values - target) ** 2)
    entropy = -reduce((logprobs.exp() * logprobs.masked_fill(~mask, 0)).sum(-1))
    loss = policy + training.value_coefficient * mse - training.entropy_coefficient * entropy
    if _endpoint_terms:
        clipped = (ratio.detach() - 1).abs() > training.clip_range
        return loss, torch.stack((policy, mse, entropy, old - chosen, loss, clipped), -1).detach()
    metrics = {"policy_loss": float(policy.detach()), "value_mse": float(mse.detach()),
        "entropy": float(entropy.detach()), "approx_kl": float((old - chosen).mean().detach()),
        "total_loss": float(loss.detach())}
    if _measure_clip:
        metrics["clip_fraction"] = float(((ratio.detach() - 1).abs() > training.clip_range).float().mean())
    return loss, metrics


def _explained_variance(rollout: Any) -> float | None:
    """Population moments of the frozen Critic predictions and exact GAE targets."""
    target_mean = residual_mean = target_m2 = residual_m2 = 0.
    count = len(rollout)
    for index in range(count):
        _, _, value, _, target = rollout.vector(index)
        residual = target - value
        target_delta = target - target_mean
        target_mean += target_delta / (index + 1)
        target_m2 += target_delta * (target - target_mean)
        residual_delta = residual - residual_mean
        residual_mean += residual_delta / (index + 1)
        residual_m2 += residual_delta * (residual - residual_mean)
    if target_m2 <= count * 1e-12:
        return None
    explained = 1. - residual_m2 / target_m2
    return explained if math.isfinite(explained) else None


def _diagnostics(clip_fraction: float, explained_variance: float | None, grad_mean: float,
                 grad_max: float, learning_rate: float) -> Json:
    return {"schema_version": 1, "clip_fraction": clip_fraction, "explained_variance": explained_variance,
            "pre_clip_grad_norm_mean": grad_mean, "pre_clip_grad_norm_max": grad_max,
            "learning_rate": learning_rate}


class HpoLearner(Learner):
    def __init__(self, model: ModelConfigV4, training: HPOTrainingConfig, seed: int,
                 binding: InferenceBindingV2, resume_identity: str, *, actions: ActionConfigV3,
                 rules: CardInstanceRules, _owned_device_admission: Json | None = None,
                 _same_host_device_bytes: int | None = None) -> None:
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
        configured_envelope=resource_estimate(model, binding, suffix_decisions=training.tbptt_length, **self._options)["total_bytes"]
        device_bytes = None
        if _same_host_device_bytes is not None:
            from gf_models.learner_cuda import learner_cuda_minima
            integer(_same_host_device_bytes, 1, 2**63 - 1)
            if _owned_device_admission is not None or binding.device != 'cuda:0':
                raise ContractError('distinct same-host CUDA initialization route required')
            if learner_cuda_minima(model, binding, **self._options)['phases']['initialize'] > _same_host_device_bytes:
                raise ContractError('same-host CUDA initialization exceeds device allowance')
        if _owned_device_admission is not None:
            from gf_models.learner_cuda import owned_learner_cuda_admission
            expected = owned_learner_cuda_admission(model,binding,
                logical_bytes=configured_envelope,global_cuda_bytes=_owned_device_admission['global_cuda_bytes'],
                pool_cuda_bytes=_owned_device_admission['pool_cuda_bytes'],
                overlap_scratch_bytes=_owned_device_admission['overlap_scratch_bytes'],**self._options)
            if canonical(expected)!=canonical(_owned_device_admission):
                raise ContractError('owned learner device admission differs from checked constructor')
            device_bytes = expected['device_bytes']
        self.torch = torch_runtime(binding.device)
        self.torch.random.default_generator.manual_seed(seed)
        if binding.device != "cpu":
            self.torch.cuda.manual_seed_all(seed)
        self.network = make_network(model, seed, binding, **self._options)
        self._context.bind(self.network, fresh=True)
        self.optimizer = self._optimizer(self.network)
        self.shuffle = np.random.Generator(np.random.PCG64(seed))
        self.steps = 0
        self._last_update_execution: str | None=None
        self._training_execution: Any = None
        self._training_reservation_bytes: int = configured_envelope
        self._training_device_bytes: int | None = device_bytes
        self._owned_snapshot: Any = None
        self._owned_generation = 0
        self._owned_failed = False
        self._owned_legacy = False
        self._owned_source_budget: Any = None

    def _optimizer(self, network: Any) -> Any:
        return self.torch.optim.Adam(network.parameters(), lr=self.training.adam_learning_rate,
            betas=(self.training.adam_beta1, self.training.adam_beta2), eps=self.training.adam_epsilon)

    def _suffix_length(self, decisions: int) -> int:
        return min(decisions, self.training.tbptt_length)

    def admit(self, rollout: StructuredRollout, *, _device_bytes: int | None = None) -> None:
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
        if _device_bytes is not None:
            integer(_device_bytes, 1, 2**63 - 1)
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
                positions = 4 * self.model.temporal.max_decisions * self.model.temporal.hidden_size
                if _device_bytes is not None and size + positions > _device_bytes:
                    raise ContractError('HPO endpoint exceeds operation device allowance')

    def update(self, rollout: StructuredRollout, *, _device_bytes: int | None = None) -> Json:
        if _device_bytes is not None:
            self.admit(rollout, _device_bytes=_device_bytes)
        self._last_update_execution=None
        self._owned_legacy=True
        self._owned_snapshot=None
        started = time.perf_counter()
        parity_started = time.perf_counter()
        parity = self.parity(rollout) if _device_bytes is None else self.parity(rollout, _device_bytes=_device_bytes)
        self._last_parity_seconds = time.perf_counter() - parity_started
        assert type(rollout) is HpoRollout
        return self._update_numeric(rollout, parity, started, self.network, _device_bytes=_device_bytes)

    def update_owned(self, rollout: Any, *, _device_bytes: int | None = None) -> Json:
        self._last_update_execution=None
        from gf_training.hpo_owned_rollout import OwnedNumericBatch
        from gf_training.rust_ppo_rollout import _RustTrainingRollout
        device_bytes = self._training_device_bytes if _device_bytes is None else _device_bytes
        if device_bytes is not None:
            integer(device_bytes, 1, 2**63 - 1)
        if self._owned_legacy:
            raise ContractError('legacy update/restore cannot enter producer-owned numerical route')
        if type(rollout) not in (OwnedNumericBatch, _RustTrainingRollout) or getattr(self, '_owned_failed', False):
            raise ContractError('owned learner is terminal or batch owner invalid')
        if rollout._budget is not self._owned_source_budget:
            raise ContractError('owned learner and batch require SAME invocation Source account')
        self._context.bind(self.network)
        self._context.check(rollout.input_binding)
        if rollout._state != 'PERSISTED':
            raise ContractError('owned update requires persisted batch, single use')
        if rollout.generation != getattr(self, '_owned_generation', 0):
            raise ContractError('owned batch generation differs from actual learner')
        if (canonical(asdict(rollout.model)) != canonical(asdict(self.model))
                or canonical(asdict(rollout.training)) != canonical(asdict(self.training))
                or rollout.action_config_hash != self.action_config_hash):
            raise ContractError('owned rollout model/actions/training mismatch')
        for index in range(len(rollout)):
            frames, candidates = rollout.endpoint(index)
            size = resource_estimate(self.model, self.binding, decisions=len(frames),
                interval_events=max(len(f.interval_numerics) for f in frames),
                candidates=len(candidates.kinds), suffix_decisions=self._suffix_length(len(frames)), **self._options)['total_bytes']
            if rollout.resident_bytes + size > min(self.model.resources.max_owned_bytes, self.binding.managed_budget_bytes):
                raise ContractError('owned rollout plus endpoint exceeds learner reservation')
        snapshot_owner = rollout._snapshot_owner
        if snapshot_owner is not None and getattr(snapshot_owner, '_learner', None) is not self:
            raise ContractError('owned batch frozen snapshot differs from actual learner')
        singletons_checked = device_bytes is not None or type(rollout) is _RustTrainingRollout
        if singletons_checked:
            self._validate_singleton_capacity(rollout, _device_bytes=device_bytes)
        rollout._begin_update()
        try:
            if not singletons_checked:
                result = self._update_numeric(rollout, {'status': 'not_run'}, time.perf_counter(), self.network._forward_owned)
            else:
                result = self._update_numeric(rollout, {'status': 'not_run'}, time.perf_counter(), self.network._forward_owned,
                    _singletons_checked=True, _device_bytes=device_bytes)
            rollout._state = 'UPDATED'
            self._owned_generation = getattr(self, '_owned_generation', 0) + 1
            self._owned_snapshot = None
            return result
        except BaseException:
            rollout._state = 'FAILED'
            self._owned_failed = True
            raise

    def _update_numeric(self, rollout: Any, parity: Json, started: float, forward: Any, *,
                        _singletons_checked: bool = False, _device_bytes: int | None = None) -> Json:
        self._last_update_execution=None
        from gf_training.hpo_training_execution import TrainingExecutionSpec
        from gf_training.rust_ppo_rollout import _RustTrainingRollout
        spec = getattr(self, '_training_execution', None)
        if spec is None:
            spec=TrainingExecutionSpec(variant='microbatch' if self.model.model_id=='mlp' else 'shared-prefix')
        if spec is not None:
            if type(spec) is not TrainingExecutionSpec:
                raise ContractError('exact experimental execution settings required')
            if spec.variant == 'shared-prefix' and self.model.model_id == 'mlp':
                raise ContractError('shared-prefix is sequence-only')
            if spec.variant != 'baseline':
                if _singletons_checked:
                    return self._update_experimental(rollout, parity, started, forward, spec,
                        _singletons_checked=True, _device_bytes=_device_bytes)
                return self._update_experimental(rollout, parity, started, forward, spec, _device_bytes=_device_bytes)
        torch, recipe = self.torch, self.training
        explained_variance = _explained_variance(rollout)
        totals: dict[str, float] = {}
        denominator = 0
        previous_steps = self.steps
        expected = recipe.epochs * math.ceil(rollout.used_decisions / recipe.minibatch_size)
        grad_sum = grad_max = 0.
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
                    if type(rollout) is _RustTrainingRollout:
                        window = rollout.pack(self.network, (index,))
                        logs, value = self.network._learning_serial(window, frames, candidates, suffix)
                        raw = window.operands
                        self.synchronize()
                        loss, measured = hpo_losses(logs, value, raw['mask'], raw['selected'], *raw['vectors'],
                            training=recipe, _measure_clip=True)
                        del window, raw
                    else:
                        logs, value, _ = forward(frames, candidates, suffix_length=suffix)
                        selected, old, _, advantage, target = rollout.vector(index)
                        self.synchronize()
                        mask = torch.ones((1, len(candidates.kinds)), dtype=torch.bool, device=self.binding.device)
                        selected_tensor = torch.tensor([selected], dtype=torch.int64, device=self.binding.device)
                        vectors = [torch.tensor([number], dtype=torch.float32, device=self.binding.device)
                            for number in (old, advantage, target)]
                        loss, measured = hpo_losses(logs[None], value[None], mask, selected_tensor, *vectors,
                            training=recipe, _measure_clip=True)
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
                grad_norm = float(norm)
                if not math.isfinite(grad_norm):
                    raise ContractError("nonfinite HPO PPO gradients")
                grad_sum += grad_norm
                grad_max = max(grad_max, grad_norm)
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        metrics = {name: value / denominator for name, value in totals.items()}
        if (self.steps - previous_steps != expected or not all(math.isfinite(value) for value in metrics.values())
                or any(not torch.isfinite(parameter).all() for parameter in self.network.parameters())):
            raise ContractError("invalid HPO optimizer step count/nonfinite output")
        self._last_update_execution='baseline'
        diagnostics = _diagnostics(metrics.pop("clip_fraction"), explained_variance,
            grad_sum / expected, grad_max, float(learning_rate))
        return {**metrics, "optimizer_steps": self.steps, "optimizer_step_delta": expected,
            "collected_decisions": rollout.collected_decisions, "used_decisions": rollout.used_decisions,
            "parity": parity, "tbptt_length": recipe.tbptt_length, "epochs": recipe.epochs,
            "endpoint_evaluations": denominator, "context_decisions": context_decisions,
            "suffix_decisions": suffix_decisions, "update_seconds": time.perf_counter() - started,
            "diagnostics": diagnostics}

    def _validate_singleton_capacity(self, rollout: Any, *, _device_bytes: int | None = None) -> None:
        from gf_training.hpo_update_planning import admit_group, execution_plan
        from gf_training.rust_ppo_rollout import _RustTrainingRollout

        native = type(rollout) is _RustTrainingRollout
        variant = getattr(getattr(self, '_training_execution', None), 'variant', 'microbatch')
        staged = self.binding.device != 'cpu' and variant == 'microbatch'
        device_bytes = self._training_device_bytes if _device_bytes is None else _device_bytes
        for index in range(len(rollout)):
            plan = execution_plan([rollout.endpoint(index)],self.model,self.binding,self.training.tbptt_length,
                layout=self.network.layout,reservation_bytes=self._training_reservation_bytes,
                resident_bytes=rollout.resident_bytes,
                device_bytes=device_bytes)
            if native:
                if plan.staging is not None and not staged:
                    admit_group(rollout, (index,), self.model, self.binding, self.training.tbptt_length,
                        layout=self.network.layout, reservation_bytes=self._training_reservation_bytes,
                        device_bytes=device_bytes)
                rollout.require_singleton_packing(index, plan.staging if staged else None)

    def _update_experimental(self, rollout: Any, parity: Json, started: float, forward: Any, spec: Any, *,
                            _singletons_checked: bool = False, _device_bytes: int | None = None) -> Json:
        from gf_training.hpo_update_execution import METRICS, execution_groups, group_forward
        # Validate singleton capacity BEFORE shuffle, gradients or optimizer
        # mutation. Group planning only refines capacity, never runtime failure.
        if spec.variant in ('microbatch','shared-prefix') and not _singletons_checked:
            self._validate_singleton_capacity(rollout, _device_bytes=_device_bytes)
        torch, recipe = self.torch, self.training
        explained_variance = _explained_variance(rollout)
        totals = torch.zeros(len(METRICS) + 1, dtype=torch.float64, device=self.binding.device)
        grad_sum = totals.new_zeros(())
        grad_max = totals.new_zeros(())
        denominator = context_decisions = suffix_decisions = 0
        encoded_frames = groups_count = transfers = 0
        previous_steps = self.steps
        expected = recipe.epochs * math.ceil(rollout.used_decisions / recipe.minibatch_size)
        self.network.train()
        for _ in range(recipe.epochs):
            order = self.shuffle.permutation(len(rollout)).tolist()
            for start in range(0,len(order),recipe.minibatch_size):
                logical = order[start:start+recipe.minibatch_size]
                groups = execution_groups(self,rollout,logical,spec, _device_bytes=_device_bytes)
                self.optimizer.zero_grad(set_to_none=True)
                for group in groups:
                    loss,measured,window,extra,encoded,copies = group_forward(self,rollout,group,spec,forward,
                        _device_bytes=_device_bytes)
                    # Finite real endpoint losses were checked before reduction.
                    (loss.mean()*(len(group.indices)/len(logical))).backward()
                    totals += measured.to(torch.float64).sum(0)
                    denominator += len(group.indices)
                    for index in group.indices:
                        frames,_ = rollout.endpoint(index)
                        context_decisions += len(frames)
                        suffix_decisions += self._suffix_length(len(frames))
                    encoded_frames += encoded
                    transfers += copies
                    groups_count += 1
                    # Last backward (including checkpoint recomputation) precedes
                    # destruction; next packing cannot alias a retained graph.
                    del loss,measured,window,extra
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),recipe.max_grad_norm,error_if_nonfinite=True)
                grad_sum += norm
                grad_max = torch.maximum(grad_max, norm)
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        *numbers, clip_fraction, grad_mean, grad_max_value = torch.cat((
            totals / denominator, torch.stack((grad_sum / expected, grad_max)),
        )).cpu().tolist()
        metrics = dict(zip(METRICS,numbers,strict=True))
        if (self.steps-previous_steps != expected or not all(math.isfinite(value) for value in metrics.values())
                or any(not torch.isfinite(parameter).all() for parameter in self.network.parameters())):
            raise ContractError('invalid HPO optimizer step count/nonfinite output')
        self._last_update_execution=spec.variant
        self._training_update_diagnostics = {'variant':spec.variant,'frame_encodes':encoded_frames,
            'microbatches':groups_count,'transfers':transfers}
        diagnostics = _diagnostics(clip_fraction, explained_variance, grad_mean, grad_max_value,
            float(learning_rate))
        return {**metrics,'optimizer_steps':self.steps,'optimizer_step_delta':expected,
            'collected_decisions':rollout.collected_decisions,'used_decisions':rollout.used_decisions,
            'parity':parity,'tbptt_length':recipe.tbptt_length,'epochs':recipe.epochs,
            'endpoint_evaluations':denominator,'context_decisions':context_decisions,
            'suffix_decisions':suffix_decisions,'update_seconds':time.perf_counter()-started,
            'diagnostics':diagnostics}

    def snapshot_owned(self, *, budget: Any, save_budget_bytes: int = 512*1024**2,
                       code_revision: str | None = None) -> Any:
        from gf_training.hpo_owned_snapshot import export_learner_snapshot
        return export_learner_snapshot(self, budget=budget, save_budget_bytes=save_budget_bytes,
            code_revision=code_revision)

    def save(self, directory: Path, *, code_revision: str | None = None,
             shard_bytes: int = MAX_ARRAY_BYTES, save_budget_bytes: int = 512*1024**2) -> Json:
        from gf_training.hpo_artifacts import save_learner
        return save_learner(self, directory, code_revision=code_revision, shard_bytes=shard_bytes,save_budget_bytes=save_budget_bytes)

    def restore(self, directory: Path, *, expected_resume_sha256: str | None = None) -> None:
        if expected_resume_sha256 is not None:
            raise ContractError('Model4 digest-bound restore requires the local held-policy/restore_snapshot route')
        self._owned_legacy=True
        self._owned_snapshot=None
        from gf_training.hpo_artifacts import restore_learner
        restore_learner(self, directory)
