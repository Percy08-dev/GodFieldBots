"""E2E structured endpoint PPO using the shared fixed loss/GAE conventions.

No collection, source authorization, publication or independent trainer loop.
Scalar endpoint accumulation preserves logical128-endpoint Adam/clip/shuffle
semantics while keeping all-candidate normalization and suffix gradients intact.
For strict physical CPU ownership, the caller starts a clean process with CUDA
visibility disabled before any Torch import. Standard pinned Adam may probe an
available accelerator on a GPU-visible host even when its parameters are on CPU.
This module never mutates process visibility or overrides Adam health checks.
"""
from __future__ import annotations

import math
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import ContractError, Json, digest, integer
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.config import LEGACY_PPO_UPDATE, PPOUpdateConfig
from gf_models.structured_artifacts import (
    _ArtifactContext,
    _game_context_v3,
    validate_save_context,
)
from gf_models.structured_artifacts import (
    _inspect_inference as inspect_inference,
)
from gf_models.structured_artifacts import (
    _load_inference as load_inference,
)
from gf_models.structured_artifacts import (
    _read_resume as read_resume,
)
from gf_models.structured_artifacts import (
    _restore_resume as restore_resume,
)
from gf_models.structured_artifacts import (
    _save_inference as save_inference,
)
from gf_models.structured_artifacts import (
    _save_resume as save_resume,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    architecture_identity,
    resource_estimate,
)
from gf_models.structured_network import make_network
from gf_models.torch_runtime import torch_runtime
from gf_training.ppo_math import losses
from gf_training.structured_rollout import StructuredRollout


def _ppo_episode_runs(rollout: StructuredRollout) -> tuple[range, ...]:
    runs = []
    start = 0
    for _, locations in groupby(rollout.locations, itemgetter(0)):
        end = start + sum(1 for _ in locations)
        runs.append(range(start, end))
        start = end
    return tuple(runs)


def _ppo_epoch_order(rng: np.random.Generator, decisions: int, episode_runs: tuple[range, ...],
                     strategy: str) -> list[int]:
    if strategy == 'all':
        return rng.permutation(decisions).tolist()
    episodes = (rng.permutation(len(episode_runs)).tolist()
        if strategy in ('episode_order', 'episode_interleave') else range(len(episode_runs)))
    if strategy == 'episode_order':
        return list(chain.from_iterable(episode_runs[index] for index in episodes))
    orders = [(rng.permutation(len(run)) + run.start).tolist() for run in
        (episode_runs[index] for index in episodes)]
    if strategy == 'within_episode':
        return list(chain.from_iterable(orders))
    active = deque(iter(order) for order in orders)
    interleaved = []
    while active:
        episode = active.popleft()
        index = next(episode, None)
        if index is not None:
            interleaved.append(index)
            active.append(episode)
    return interleaved


@dataclass(frozen=True)
class _OperationDeviceLimit:
    device_bytes: int | None
    owned_rollout: StructuredRollout | None = None


class Learner:
    def __init__(self, model: Any, seed: int, binding: InferenceBindingV2,
                 action_config_hash: str, resume_identity: str, *, action_contract: Any,
                 purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None) -> None:
        self._context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
        self._options = self._context.options
        self.input_binding = self._context.classification()
        self.action_contract, self.purpose, self.synthetic_capacity = action_contract, purpose, synthetic_capacity
        for value in (action_config_hash, resume_identity):
            if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ContractError("closed action/resume identity digests required")
        integer(seed, 0, 2**63 - 1)
        resource_estimate(model, binding, **self._options)
        self.torch = torch_runtime(binding.device)
        self.torch.random.default_generator.manual_seed(seed)
        if binding.device != "cpu":
            self.torch.cuda.manual_seed_all(seed)
        self.model, self.binding = model, binding
        self.action_config_hash, self.resume_identity = action_config_hash, resume_identity
        self.network = make_network(model, seed, binding, **self._options)
        self._context.bind(self.network, fresh=True)
        self.optimizer = self.torch.optim.Adam(self.network.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
        self.shuffle = np.random.Generator(np.random.PCG64(seed))
        self.steps = 0

    def synchronize(self) -> None:
        if self.binding.device != "cpu":
            self.torch.cuda.synchronize(self.binding.device)

    def admit(self, rollout: StructuredRollout, *, _device_bytes: int | None = None) -> None:
        _device_bytes = self._current_device_limit(_device_bytes)
        if type(rollout.model) is not (ModelConfigV2 if self._context.schema == 3 else ModelConfigV3):
            raise ContractError('historical history descriptor cannot enter active learning')
        self._context.bind(self.network)
        self._context.check(self.input_binding)
        self._context.check(rollout.input_binding)
        rollout.require_learning()
        if architecture_identity(rollout.model, **self._options) != architecture_identity(self.model, **self._options) or rollout.action_config_hash != self.action_config_hash:
            raise ContractError("learner rollout architecture/action mismatch")
        if _device_bytes is not None:
            integer(_device_bytes, 1, 2**63 - 1)
        # Complete batch preflight before reference construction, packing or any
        # optimizer step; no legacy4096 candidate total admission is inherited.
        for index in range(len(rollout)):
            frames, candidates = rollout.endpoint(index)
            endpoint_bytes = resource_estimate(self.model, self.binding, decisions=len(frames),
                              interval_events=max(len(frame.interval_numerics) for frame in frames),
                              candidates=len(candidates.kinds), suffix_decisions=min(len(frames), self.model.temporal.sequence_length), **self._options)["total_bytes"]
            if rollout.resident_bytes + endpoint_bytes > min(self.model.resources.max_owned_bytes, self.binding.managed_budget_bytes):
                raise ContractError("whole rollout plus endpoint exceeds learner reservation")
            positions = 4 * self.model.temporal.max_decisions * self.model.temporal.hidden_size
            if _device_bytes is not None and endpoint_bytes + positions > _device_bytes:
                raise ContractError('actual endpoint plus position storage exceeds operation device allowance')

    def parity(self, rollout: StructuredRollout, *, _device_bytes: int | None = None) -> Json:
        current = getattr(self, '_rust_operation_limit', None)
        if current is not None and current.owned_rollout is not None and current.owned_rollout is rollout:
            return {'status': 'not_run'}
        _device_bytes = self._current_device_limit(_device_bytes)
        if _device_bytes is not None:
            # Refuse before RNG capture/reference construction; do not perform
            # the same complete logical preflight twice on this new route.
            self.admit(rollout, _device_bytes=_device_bytes)
        # Component2 preserves its original stream behavior. Game parity is read-only.
        devices = list(range(self.torch.cuda.device_count())) if self.binding.device != "cpu" else []
        scope = self.torch.random.fork_rng(devices=devices) if self._context.schema >= 4 else nullcontext()
        with scope:
            return (self._parity(rollout) if _device_bytes is None else
                self._parity(rollout, _device_bytes=_device_bytes, _preflight_done=True))

    def _suffix_length(self, decisions: int) -> int:
        return min(decisions, self.model.temporal.sequence_length)

    def _parity(self, rollout: StructuredRollout, *, _device_bytes: int | None = None,
                _preflight_done: bool = False) -> Json:
        _device_bytes = self._current_device_limit(_device_bytes)
        if not _preflight_done:
            if _device_bytes is None:
                self.admit(rollout)
            else:
                self.admit(rollout, _device_bytes=_device_bytes)
        torch = self.torch
        cpu_binding = replace(self.binding, device="cpu")
        cpu = make_network(self.model, 0, cpu_binding, **self._options)
        cpu.load_state_dict({name: value.detach().cpu() for name, value in self.network.state_dict().items()})
        maximum = 0.
        for episode in rollout.episodes:
            cache = cpu.new_cache()
            for position, (frame, candidates, selection) in enumerate(zip(episode.frames, episode.tables, episode.records.selections, strict=True)):
                frames = episode.frames[:position + 1]
                with torch.no_grad():
                    context, cards = cpu.step(frame, cache)
                    cached_logs, _ = cpu.distribution(context, cards, candidates)
                    cached_value = cpu.value(context)
                    cpu_logs, cpu_value, _ = cpu(frames, candidates)
                    device_logs, device_value, _ = self.network(frames, candidates)
                # This actual learner-device suffix graph stays differentiable;
                # no persisted or previous-update latent is used as a shortcut.
                split_logs, split_value, _ = self.network(frames, candidates,
                    suffix_length=self._suffix_length(len(frames)))
                if not split_logs.requires_grad or not split_value.requires_grad:
                    raise ContractError("learner split lost suffix gradients")
                self.synchronize()
                checks = ((cpu_logs[selection.index], torch.tensor(selection.logprob, dtype=torch.float32)),
                          (cpu_value, torch.tensor(selection.value, dtype=torch.float32)),
                          (cpu_logs, cached_logs), (cpu_value, cached_value),
                          (cpu_logs, device_logs.detach().cpu()), (cpu_value, device_value.detach().cpu()),
                          (cpu_logs, split_logs.detach().cpu()), (cpu_value, split_value.detach().cpu()))
                for expected, actual in checks:
                    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                        raise ContractError("whole-batch frozen saved/cache/full/device/split parity failed before update")
                    maximum = max(maximum, float((actual - expected).abs().max()))
                del split_logs, split_value, checks
        return {"atol": 1e-5, "rtol": 1e-5, "max_abs_error": maximum,
                "decisions": sum(len(episode.frames) for episode in rollout.episodes),
                "checks": ["saved", "cpu-cache", "cpu-full", "learner-full", "learner-split"]}

    def _current_device_limit(self, explicit: int | None) -> int | None:
        current = getattr(self, '_rust_operation_limit', None)
        return current.device_bytes if current is not None else explicit

    def update_with_device_limit(self, rollout: StructuredRollout, *, device_bytes: int) -> Json:
        """Rust-only context around the exact frozen update, not another loop.

        Caller holds serialized_torch. Removing this numerical context does NOT
        settle a failed device-account operation or release its actual graphs.
        """
        integer(device_bytes, 1, 2**63 - 1)
        if self.model.version != '3' or getattr(self, '_rust_operation_limit', None) is not None:
            raise ContractError('non-nested Model3 operation device limit required')
        self._rust_operation_limit = _OperationDeviceLimit(device_bytes)
        try:
            return self.update(rollout)
        finally:
            del self._rust_operation_limit

    def _update_owned_model3(self, rollout: StructuredRollout, *, device_bytes: int | None = None,
                             recipe: PPOUpdateConfig = LEGACY_PPO_UPDATE) -> Json:
        """Actual Source handoff with one preflight around Model3 optimization."""
        if device_bytes is not None:
            integer(device_bytes, 1, 2**63 - 1)
        if self.model.version != '3' or getattr(self, '_rust_operation_limit', None) is not None:
            raise ContractError('non-nested owned Model3 operation required')
        self._rust_operation_limit = _OperationDeviceLimit(device_bytes, rollout)
        try:
            self.admit(rollout)
            cuda_width = recipe.gpu_microbatch_size if self.binding.device != 'cpu' else 1
            if (cuda_width == 1 and recipe.minibatch_size == LEGACY_PPO_UPDATE.minibatch_size
                    and recipe.epochs == LEGACY_PPO_UPDATE.epochs and recipe.shuffle_strategy == 'all'):
                return self.update(rollout)
            if self.model.model_id != 'mlp':
                raise ContractError('configurable ordinary PPO requires Model3 MLP')
            return self._update_ordinary_model3(rollout, recipe, cuda_width=cuda_width)
        finally:
            del self._rust_operation_limit

    def _packed_model3_loss(self, rollout: StructuredRollout, indices: tuple[int, ...]) -> tuple[Any, Json]:
        from gf_models.structured_learning import pack_window
        from gf_training.hpo_update_planning import group_frames

        endpoints = [rollout.endpoint(index) for index in indices]
        window = pack_window(self.network, group_frames(endpoints, self.model, False),
            [candidates for _, candidates in endpoints], [rollout.vector(index) for index in indices])
        logs, values = self.network._learning_forward(window, [1] * len(indices), 1)
        raw = window.operands
        return losses(logs, values, raw['mask'], raw['selected'], *raw['vectors'])

    def _scalar_model3_loss(self, rollout: StructuredRollout, index: int) -> tuple[Any, Json]:
        torch = self.torch
        frames, candidates = rollout.endpoint(index)
        logs, value, _ = self.network(frames, candidates,
            suffix_length=min(len(frames), self.model.temporal.sequence_length))
        selected, old, _, advantage, target = rollout.vector(index)
        self.synchronize()
        mask = torch.ones((1, len(candidates.kinds)), dtype=torch.bool, device=self.binding.device)
        selected_tensor = torch.tensor([selected], dtype=torch.int64, device=self.binding.device)
        vectors = [torch.tensor([number], dtype=torch.float32, device=self.binding.device)
            for number in (old, advantage, target)]
        return losses(logs[None], value[None], mask, selected_tensor, *vectors)

    def _update_ordinary_model3(self, rollout: StructuredRollout, recipe: PPOUpdateConfig, *, cuda_width: int) -> Json:
        from gf_training.hpo_update_planning import plan_groups

        torch = self.torch
        parity = self.parity(rollout)
        totals: dict[str, float] = {}
        denominator = 0
        packed = cuda_width > 1
        device = self.binding.device
        gradient_bytes = (sum(parameter.numel() * parameter.element_size()
            for parameter in self.network.parameters()) if packed else 0)
        runs = _ppo_episode_runs(rollout) if recipe.shuffle_strategy != 'all' else ()
        self.network.train()
        for _ in range(recipe.epochs):
            order = _ppo_epoch_order(self.shuffle, len(rollout), runs, recipe.shuffle_strategy)
            for start in range(0, len(order), recipe.minibatch_size):
                logical = order[start:start + recipe.minibatch_size]
                self.optimizer.zero_grad(set_to_none=True)
                if packed:
                    # After zeroing, the batch needs at most one complete gradient
                    # set in addition to one group's live graph and inputs.
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    memory = torch.cuda.memory_stats(device)
                    idle_bytes = memory['reserved_bytes.all.current'] - memory['active_bytes.all.current']
                    plans = plan_groups(rollout, logical, self.model, self.binding, 1,
                        variant='microbatch', reservation_bytes=self.binding.managed_budget_bytes,
                        layout=self.network.layout, device_bytes=self._current_device_limit(None),
                        mlp_width=cuda_width, packed_device_headroom=free_bytes + idle_bytes - gradient_bytes)
                    groups = (group.indices for group in plans)
                else:
                    groups = ((index,) for index in logical)
                for indices in groups:
                    size = len(indices)
                    loss, measured = (self._packed_model3_loss(rollout, indices) if packed else
                        self._scalar_model3_loss(rollout, indices[0]))
                    if not torch.isfinite(loss):
                        raise ContractError("nonfinite structured PPO loss")
                    weighted = loss * (size / len(logical)) if packed else loss / len(logical)
                    weighted.backward()
                    for name, number in measured.items():
                        totals[name] = totals.get(name, 0.) + size * number
                    denominator += size
                    del loss, weighted
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ContractError("nonfinite structured PPO gradients")
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        metrics = {name: value / denominator for name, value in totals.items()}
        if not all(math.isfinite(value) for value in metrics.values()) or any(not torch.isfinite(parameter).all() for parameter in self.network.parameters()):
            raise ContractError("nonfinite structured PPO update output")
        return {**metrics, "optimizer_steps": self.steps, "parity": parity,
                "sequence_length": self.model.temporal.sequence_length}

    def update(self, rollout: StructuredRollout) -> Json:
        torch = self.torch
        parity = self.parity(rollout)  # ENTIRE frozen batch before any Adam change.
        totals: dict[str, float] = {}
        denominator = 0
        self.network.train()
        for _ in range(4):
            order = self.shuffle.permutation(len(rollout)).tolist()
            for start in range(0, len(order), 128):
                logical = order[start:start + 128]
                # Scalar endpoint graphs are admitted individually before packing;
                # original logical ordering/weights and the one clip/step remain.
                self.optimizer.zero_grad(set_to_none=True)
                for index in logical:
                    frames, candidates = rollout.endpoint(index)
                    logs, value, _ = self.network(frames, candidates,
                        suffix_length=min(len(frames), self.model.temporal.sequence_length))
                    selected, old, _, advantage, target = rollout.vector(index)
                    self.synchronize()
                    mask = torch.ones((1, len(candidates.kinds)), dtype=torch.bool, device=self.binding.device)
                    selected_tensor = torch.tensor([selected], dtype=torch.int64, device=self.binding.device)
                    vectors = [torch.tensor([number], dtype=torch.float32, device=self.binding.device)
                               for number in (old, advantage, target)]
                    loss, measured = losses(logs[None], value[None], mask, selected_tensor, *vectors)
                    if not torch.isfinite(loss):
                        raise ContractError("nonfinite structured PPO loss")
                    (loss / len(logical)).backward()
                    for name, number in measured.items():
                        totals[name] = totals.get(name, 0.) + number
                    denominator += 1
                    del logs, value, loss
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ContractError("nonfinite structured PPO gradients")
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        metrics = {name: value / denominator for name, value in totals.items()}
        if not all(math.isfinite(value) for value in metrics.values()) or any(not torch.isfinite(parameter).all() for parameter in self.network.parameters()):
            raise ContractError("nonfinite structured PPO update output")
        return {**metrics, "optimizer_steps": self.steps, "parity": parity,
                "sequence_length": self.model.temporal.sequence_length}

    def save(self, directory: Path, *, code_revision: str | None = None,
             shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
        self._context.bind(self.network)
        self._context.check(self.input_binding)
        validate_save_context(self.network, self.model, self.binding, self.optimizer, **self._options)
        self.synchronize()
        manifest = save_inference(directory, self.network, self.model, self.action_config_hash,
                                  code_revision=code_revision, shard_bytes=shard_bytes, action_contract=self.action_contract,
                                  purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context)
        resume = save_resume(directory, self.network, self.optimizer, self.shuffle, self.steps,
                             self.model, self.binding, self.resume_identity, shard_bytes=shard_bytes, action_contract=self.action_contract,
                             purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context)
        return {"inference_digest": sha256(directory / "inference.json"),
                "resume_digest": sha256(directory / "resume.json"),
                "files": {**manifest["weights"]["files"], **resume["state"]["files"],
                          "inference.json": sha256(directory / "inference.json"),
                          "resume.json": sha256(directory / "resume.json")}}

    def restore(self, directory: Path, *, expected_resume_sha256: str | None = None) -> None:
        self._context.bind(self.network)
        self._context.check(self.input_binding)
        # Stage complete weights/optimizer validation in fresh owners. A failed
        # ordinary resume never partially overwrites this learner's parameters.
        existing_bytes = sum(buffer.numel() * buffer.element_size() for buffer in self.network.buffers())
        existing_bytes += sum(parameter.numel() * parameter.element_size() for parameter in self.network.parameters())
        existing_bytes += sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in self.network.parameters() if parameter.grad is not None)
        existing_bytes += sum(value.numel() * value.element_size() for state in self.optimizer.state.values()
                              for value in state.values() if isinstance(value, self.torch.Tensor))
        staged = resource_estimate(self.model, self.binding, **self._options)["total_bytes"]
        if existing_bytes + staged > min(self.model.resources.max_owned_bytes, self.binding.managed_budget_bytes):
            raise ContractError("staged resume plus live learner exceeds managed reservation")
        inspect_inference(directory, model=self.model, binding=self.binding,
                          action_config_hash=self.action_config_hash, action_contract=self.action_contract,
                          purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context)
        # Stream/validate every resume shard and RNG/header before constructing
        # a staged network. The final publisher revalidates its own fresh data.
        read_resume(directory, self.model, self.binding, self.resume_identity, action_contract=self.action_contract,
                    purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context,
                    expected_resume_sha256=expected_resume_sha256)
        network, _ = load_inference(directory, model=self.model, binding=self.binding,
                                     action_config_hash=self.action_config_hash, action_contract=self.action_contract,
                          purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context)
        optimizer = self.torch.optim.Adam(network.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
        steps = restore_resume(directory, network, optimizer, self.shuffle, self.model,
                                self.binding, self.resume_identity, action_contract=self.action_contract,
                                purpose=self.purpose, synthetic_capacity=self.synthetic_capacity, _context=self._context,
                                expected_resume_sha256=expected_resume_sha256)
        self.network, self.optimizer, self.steps = network, optimizer, steps


class GameLearner(Learner):
    """Closed Model3 game wrapper; numerical update is inherited unchanged."""
    def __init__(self, model: ModelConfigV3, seed: int, binding: InferenceBindingV2,
                 resume_identity: str, *, actions: ActionConfigV3, rules: CardInstanceRules) -> None:
        action_hash = digest(asdict(actions))
        context = _game_context_v3(actions.action_contract, rules, action_hash)
        if type(model) is not ModelConfigV3 or model.max_positions > 18:
            raise ContractError('game learner requires exact real18 Model3')
        super().__init__(model, seed, binding, action_hash, resume_identity,
            action_contract=actions.action_contract, purpose='qualified_game', synthetic_capacity=False,
            _context=context)
