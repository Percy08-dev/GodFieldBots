"""Owned-child PPO FP32 math, parity, microbatch gradients and typed resume."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json
from gf_models.arrays import load_arrays
from gf_models.config import MLPModelConfig
from gf_models.mlp.model import make_network, restore_weights, save_artifact, torch_runtime
from gf_training.ppo import RolloutDecision
from gf_training.ppo_math import losses
from gf_training.ppo_state import restore_resume, save_resume


def microbatches(indices: list[int], decisions: list[RolloutDecision], slots: int = 4096) -> list[list[int]]:
    output: list[list[int]] = []
    current: list[int] = []
    maximum = 0
    for index in indices:
        count = len(decisions[index].actor)
        if count > slots:
            raise ContractError('single decision exceeds padded microbatch slots')
        if current and max(maximum, count) * (len(current) + 1) > slots:
            output.append(current)
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, count)
    if current:
        output.append(current)
    return output


def padded(decisions: list[RolloutDecision], indices: list[int], device: str) -> tuple[Any, ...]:
    import torch
    selected = [decisions[index] for index in indices]
    count = max(len(item.actor) for item in selected)
    actor = np.zeros((len(selected), count, selected[0].actor.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(selected), count), dtype=np.bool_)
    for index, item in enumerate(selected):
        actor[index, :len(item.actor)] = item.actor
        mask[index, :len(item.actor)] = True
    arrays = [actor, np.stack([item.state for item in selected]), mask,
              np.asarray([item.selected for item in selected], dtype=np.int64)]
    return tuple(torch.from_numpy(value).to(device) for value in arrays)




class Learner:
    def __init__(self, model: MLPModelConfig, seed: int, device: str) -> None:
        self.torch = torch_runtime(device)
        self.torch.random.default_generator.manual_seed(seed)
        if device != "cpu":
            self.torch.cuda.manual_seed_all(seed)
        self.model, self.device = model, device
        self.network = make_network(model, seed).to(device)
        self.optimizer = self.torch.optim.Adam(self.network.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
        self.shuffle = np.random.Generator(np.random.PCG64(seed))
        self.steps = 0

    def hardware(self) -> Json:
        from gf_models.torch_runtime import hardware
        return hardware(self.device)

    def synchronize(self) -> None:
        if self.device != 'cpu':
            self.torch.cuda.synchronize(self.device)

    def parity(self, decisions: list[RolloutDecision]) -> Json:
        torch = self.torch
        cpu = make_network(self.model, 0)
        cpu.load_state_dict({key: value.detach().cpu() for key, value in self.network.state_dict().items()})
        maximum = 0.
        # Validate every decision and every valid candidate before any step.
        with torch.no_grad():
            for indices in microbatches(list(range(len(decisions))), decisions):
                actor, state, mask, selected = padded(decisions, indices, 'cpu')
                logprobs, values = cpu(actor, state, mask)
                device_logprobs, device_values = self.network(actor.to(self.device), state.to(self.device), mask.to(self.device))
                self.synchronize()
                device_logprobs, device_values = device_logprobs.cpu(), device_values.cpu()
                saved_logprobs = torch.tensor([decisions[index].old_logprob for index in indices], dtype=torch.float32)
                saved_values = torch.tensor([decisions[index].old_value for index in indices], dtype=torch.float32)
                checks = ((logprobs.gather(1, selected[:, None]).squeeze(1), saved_logprobs), (values, saved_values),
                          (logprobs[mask], device_logprobs[mask]), (values, device_values))
                for actual, expected in checks:
                    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                        raise ContractError('frozen CPU/saved/device parity failed before update')
                    maximum = max(maximum, float((actual - expected).abs().max()))
        return {'atol': 1e-5, 'rtol': 1e-5, 'max_abs_error': maximum, 'decisions': len(decisions)}

    def update(self, decisions: list[RolloutDecision]) -> Json:
        torch = self.torch
        parity = self.parity(decisions)
        if self.device != 'cpu':
            torch.cuda.reset_peak_memory_stats(self.device)
        metrics = []
        self.network.train()
        for _ in range(4):
            order = self.shuffle.permutation(len(decisions)).tolist()
            for start in range(0, len(order), 128):
                logical = order[start:start + 128]
                self.optimizer.zero_grad(set_to_none=True)
                for indices in microbatches(logical, decisions):
                    actor, state, mask, selected = padded(decisions, indices, self.device)
                    logprobs, values = self.network(actor, state, mask)
                    def vector(field: str, indices: list[int] = indices) -> Any:
                        return torch.tensor([getattr(decisions[index], field) for index in indices], device=self.device, dtype=torch.float32)
                    loss, measured = losses(logprobs, values, mask, selected, vector('old_logprob'), vector('advantage'), vector('target'))
                    (loss * (len(indices) / len(logical))).backward()
                    metrics.append((measured, len(indices)))
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), .5, error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ContractError('nonfinite PPO gradients')
                self.optimizer.step()
                self.steps += 1
        self.synchronize()
        denominator = sum(count for _, count in metrics)
        result = {key: sum(item[key] * count for item, count in metrics) / denominator for key in metrics[0][0]}
        if not all(math.isfinite(value) for value in result.values()):
            raise ContractError('nonfinite PPO metrics')
        return {**result, 'parity': parity, 'optimizer_steps': self.steps,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device != 'cpu' else 0,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device != 'cpu' else 0}

    def save(self, directory: Path, action_hash: str) -> Json:
        self.synchronize()
        inference = save_artifact(self.network, directory, self.model, action_hash)
        save_resume(directory, self.network, self.optimizer, self.shuffle, self.steps, self.device)
        # Verify exact array roundtrip while original tensors remain alive.
        restored = load_arrays(directory / 'weights.npz', inference['weights'])
        for key, value in self.network.state_dict().items():
            if not np.array_equal(value.detach().cpu().numpy(), restored[key]):
                raise ContractError('CPU checkpoint roundtrip failed')
        return {'weights': inference['weights']['sha256'], 'optimizer_steps': self.steps}

    def restore(self, directory: Path) -> None:
        restore_weights(self.network, load_arrays(directory / 'weights.npz'))
        self.steps = restore_resume(directory, self.network, self.optimizer, self.shuffle, self.device)
