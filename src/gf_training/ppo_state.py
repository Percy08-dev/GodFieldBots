"""Typed, bounded Adam/RNG persistence shared by the closed PPO learners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, canonical, closed
from gf_models.arrays import atomic_json, load_arrays, save_arrays


def save_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int, device: str,
                *, schema_version: int = 1, identity: Json | None = None) -> Json:
    import torch
    identity = {} if identity is None else identity
    arrays = {'torch_cpu_rng': torch.get_rng_state().numpy().copy()}
    for index, state in enumerate(torch.cuda.get_rng_state_all() if device != 'cpu' else []):
        arrays[f'torch_cuda_rng.{index}'] = state.cpu().numpy().copy()
    for index, parameter in enumerate(network.parameters()):
        state = optimizer.state.get(parameter, {})
        if state:
            arrays[f'adam.{index}.step'] = np.asarray(int(state['step'].item()), dtype=np.int64)
            for key in ('exp_avg', 'exp_avg_sq'):
                arrays[f'adam.{index}.{key}'] = state[key].detach().cpu().numpy().astype(np.float32, copy=True)
    metadata = save_arrays(directory / 'resume.npz', arrays)
    resume = {'schema_version': schema_version, 'arrays': metadata, 'shuffle_rng': shuffle.bit_generator.state,
              'optimizer_steps': steps, 'device': device, **identity}
    atomic_json(directory / 'resume.json', resume)
    load_arrays(directory / 'resume.npz', metadata)
    return metadata


def restore_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any, device: str,
                   *, schema_version: int = 1, identity: Json | None = None) -> int:
    import torch
    identity = {} if identity is None else identity
    resume = json.loads((directory / 'resume.json').read_text())
    closed(resume, {'schema_version', 'arrays', 'shuffle_rng', 'optimizer_steps', 'device'} | set(identity))
    if type(resume['schema_version']) is not int or resume['schema_version'] != schema_version or resume['device'] != device or any(canonical(resume[key]) != canonical(value) for key, value in identity.items()):
        raise ContractError('resume device/schema mismatch')
    arrays = load_arrays(directory / 'resume.npz', resume['arrays'])
    expected_keys = {'torch_cpu_rng'}
    cuda_count = torch.cuda.device_count() if device != 'cpu' else 0
    expected_keys.update(f'torch_cuda_rng.{index}' for index in range(cuda_count))
    steps = resume['optimizer_steps']
    if type(steps) is not int or steps < 0:
        raise ContractError('invalid optimizer step count')
    if steps:
        for index, parameter in enumerate(network.parameters()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                expected_keys.add(f'adam.{index}.{key}')
            step = arrays[f'adam.{index}.step']
            if step.dtype != np.int64 or step.shape != () or int(step) != steps:
                raise ContractError('Adam counter mismatch')
            state = {'step': torch.tensor(float(step), dtype=torch.float32)}
            for key in ('exp_avg', 'exp_avg_sq'):
                value = arrays[f'adam.{index}.{key}']
                if value.dtype != np.float32 or tuple(value.shape) != tuple(parameter.shape):
                    raise ContractError('Adam state mismatch')
                state[key] = torch.from_numpy(value.copy()).to(device)
            optimizer.state[parameter] = state
    if set(arrays) != expected_keys:
        raise ContractError('resume RNG/optimizer keys mismatch')
    torch.set_rng_state(torch.from_numpy(arrays['torch_cpu_rng'].copy()))
    if cuda_count:
        torch.cuda.set_rng_state_all([torch.from_numpy(arrays[f'torch_cuda_rng.{index}'].copy()) for index in range(cuda_count)])
    shuffle.bit_generator.state = resume['shuffle_rng']
    return steps
