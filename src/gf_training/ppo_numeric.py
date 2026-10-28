"""Owned-child PPO FP32 math, parity, microbatch gradients and typed resume."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, closed
from gf_models.config import MLPModelConfig
from gf_models.mlp.arrays import load_arrays, save_arrays
from gf_models.mlp.model import make_network, restore_weights, save_artifact, torch_runtime
from gf_training.journal import atomic_json
from gf_training.ppo import RolloutDecision


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


def losses(logprobs: Any, values: Any, mask: Any, selected: Any, old: Any, advantage: Any, target: Any) -> tuple[Any, Json]:
    import torch
    chosen = logprobs.gather(1, selected[:, None]).squeeze(1)
    ratio = (chosen - old).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
    mse = ((values - target) ** 2).mean()
    safe_logprobs = logprobs.masked_fill(~mask, 0)
    entropy = -(logprobs.exp() * safe_logprobs).sum(-1).mean()
    loss = policy + .5 * mse - .01 * entropy
    return loss, {'policy_loss': float(policy.detach()), 'value_mse': float(mse.detach()),
                  'entropy': float(entropy.detach()), 'approx_kl': float((old - chosen).mean().detach()),
                  'total_loss': float(loss.detach())}


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
        torch = self.torch
        name = torch.cuda.get_device_name(0) if self.device != 'cpu' else 'cpu'
        driver_api = None
        if self.device != 'cpu':
            # CUDA is initialized before this query. This is the supported CUDA
            # API version, not an attested NVIDIA driver build number.
            import ctypes
            driver = ctypes.CDLL('libcuda.so.1')
            query = driver.cuDriverGetVersion
            query.argtypes = [ctypes.POINTER(ctypes.c_int)]
            query.restype = ctypes.c_int
            actual = ctypes.c_int()
            if query(ctypes.byref(actual)) != 0 or actual.value <= 0:
                raise ContractError('CUDA driver API identity query failed')
            driver_api = actual.value
        return {'device': self.device, 'name': name, 'cuda_runtime': torch.version.cuda,
                'device_count': torch.cuda.device_count() if self.device != 'cpu' else 0,
                'cuda_driver_api_version': driver_api, 'exact_driver_build': 'unverified'}

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
        torch = self.torch
        self.synchronize()
        inference = save_artifact(self.network, directory, self.model, action_hash)
        arrays = {'torch_cpu_rng': torch.get_rng_state().numpy().copy()}
        for index, state in enumerate(torch.cuda.get_rng_state_all() if self.device != 'cpu' else []):
            arrays[f'torch_cuda_rng.{index}'] = state.cpu().numpy().copy()
        for index, parameter in enumerate(self.network.parameters()):
            state = self.optimizer.state.get(parameter, {})
            if state:
                arrays[f'adam.{index}.step'] = np.asarray(int(state['step'].item()), dtype=np.int64)
                for key in ('exp_avg', 'exp_avg_sq'):
                    arrays[f'adam.{index}.{key}'] = state[key].detach().cpu().numpy().astype(np.float32, copy=True)
        metadata = save_arrays(directory / 'resume.npz', arrays)
        resume = {'schema_version': 1, 'arrays': metadata, 'shuffle_rng': self.shuffle.bit_generator.state,
                  'optimizer_steps': self.steps, 'device': self.device}
        atomic_json(directory / 'resume.json', resume)
        # Verify exact array roundtrip while original tensors remain alive.
        restored = load_arrays(directory / 'weights.npz', inference['weights'])
        for key, value in self.network.state_dict().items():
            if not np.array_equal(value.detach().cpu().numpy(), restored[key]):
                raise ContractError('CPU checkpoint roundtrip failed')
        load_arrays(directory / 'resume.npz', metadata)
        return {'weights': inference['weights']['sha256'], 'optimizer_steps': self.steps}

    def restore(self, directory: Path) -> None:
        torch = self.torch
        restore_weights(self.network, load_arrays(directory / 'weights.npz'))
        resume = json.loads((directory / 'resume.json').read_text())
        closed(resume, {'schema_version', 'arrays', 'shuffle_rng', 'optimizer_steps', 'device'})
        if resume['schema_version'] != 1 or resume['device'] != self.device:
            raise ContractError('resume device/schema mismatch')
        arrays = load_arrays(directory / 'resume.npz', resume['arrays'])
        expected_keys = {'torch_cpu_rng'}
        cuda_count = torch.cuda.device_count() if self.device != 'cpu' else 0
        expected_keys.update(f'torch_cuda_rng.{index}' for index in range(cuda_count))
        self.steps = resume['optimizer_steps']
        if type(self.steps) is not int or self.steps < 0:
            raise ContractError('invalid optimizer step count')
        if self.steps:
            for index, parameter in enumerate(self.network.parameters()):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    expected_keys.add(f'adam.{index}.{key}')
                step = arrays[f'adam.{index}.step']
                if step.dtype != np.int64 or step.shape != () or int(step) != self.steps:
                    raise ContractError('Adam counter mismatch')
                state = {'step': torch.tensor(float(step), dtype=torch.float32)}
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = arrays[f'adam.{index}.{key}']
                    if value.dtype != np.float32 or tuple(value.shape) != tuple(parameter.shape):
                        raise ContractError('Adam state mismatch')
                    state[key] = torch.from_numpy(value.copy()).to(self.device)
                self.optimizer.state[parameter] = state
        if set(arrays) != expected_keys:
            raise ContractError('resume RNG/optimizer keys mismatch')
        torch.set_rng_state(torch.from_numpy(arrays['torch_cpu_rng'].copy()))
        if cuda_count:
            torch.cuda.set_rng_state_all([torch.from_numpy(arrays[f'torch_cuda_rng.{index}'].copy()) for index in range(cuda_count)])
        self.shuffle.bit_generator.state = resume['shuffle_rng']
