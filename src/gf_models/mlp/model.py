"""Lazy Torch model, public inference artifacts and exact numerical recipe."""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.actions import GENERATOR_VERSION, SCORER_VERSION
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    SIMULATOR_REVISION,
    ContractError,
    Json,
    canonical,
    closed,
)
from gf_models.config import MLPModelConfig, mlp_model_config
from gf_models.encoder import feature_names, signature
from gf_models.mlp.arrays import atomic_json, load_arrays, save_arrays, sha256
from gf_models.mlp.features import critic_indices

NUMERICS = {"precision": "ieee-fp32", "tf32": False, "amp": False, "deterministic": True,
            "cublas_workspace_config": ":4096:8", "threads": 1}


def runtime_metadata() -> Json:
    versions = {"torch": importlib.metadata.version('torch'), "numpy": np.__version__,
                "python": platform.python_version(), "gil_enabled": sys._is_gil_enabled()}
    if versions['torch'] not in ('2.14.1+cpu', '2.14.1+cu132') or versions['numpy'] != '2.5.4' or sys.version_info[:2] != (3, 14) or not versions['gil_enabled']:
        raise ContractError('unverified MLP native runtime')
    return versions


def torch_runtime(device: str) -> Any:
    versions = runtime_metadata()
    import torch
    if torch.__version__ != versions['torch'] or torch.version.cuda != ('13.2' if versions['torch'].endswith('+cu132') else None):
        raise ContractError('MLP module/distribution/CUDA runtime mismatch')
    if device == 'cuda:0' and versions['torch'] != '2.14.1+cu132':
        raise ContractError('CUDA learner requires the pinned cu132 build')
    if device not in ('cpu', 'cuda:0') or device == 'cuda:0' and not torch.cuda.is_available():
        raise ContractError("configured learner device unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    backends: Any = torch.backends
    backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def make_network(model: MLPModelConfig, seed: int) -> Any:
    torch = torch_runtime('cpu')
    original_rng = torch.get_rng_state()
    torch.random.default_generator.manual_seed(seed)
    nn = torch.nn
    def branch(width: int, gain: float) -> Any:
        layers = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        for index in (0, 2, 4):
            nn.init.orthogonal_(layers[index].weight, math.sqrt(2) if index != 4 else gain)
            nn.init.zeros_(layers[index].bias)
        return layers
    base: Any = nn.Module
    class ActorCritic(base):
        def __init__(self) -> None:
            super().__init__()
            self.actor = branch(len(feature_names(model.encoder)), .01)
            self.critic = branch(len(critic_indices(model.encoder)), 1)
        def forward(self, actor: Any, state: Any, mask: Any) -> tuple[Any, Any]:
            logits = self.actor(actor).squeeze(-1).masked_fill(~mask, -torch.inf)
            return logits.log_softmax(-1), self.critic(state).squeeze(-1)
    network = ActorCritic()
    torch.set_rng_state(original_rng)
    return network


def weights(network: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().numpy().astype(np.float32, copy=True) for name, tensor in network.state_dict().items()}


def restore_weights(network: Any, arrays: dict[str, Any]) -> None:
    import torch
    expected = network.state_dict()
    if set(arrays) != set(expected):
        raise ContractError("MLP weight keys mismatch")
    for key, tensor in expected.items():
        value = arrays[key]
        if value.dtype != np.float32 or tuple(value.shape) != tuple(tensor.shape) or not np.isfinite(value).all():
            raise ContractError("MLP weight shape/type mismatch")
    network.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in arrays.items()}, strict=True)


def save_artifact(network: Any, directory: Path, model: MLPModelConfig, action_hash: str) -> Json:
    directory.mkdir(parents=True, exist_ok=False)
    tensors = save_arrays(directory / 'weights.npz', weights(network))
    result = {"schema_version": 1, "model": {**asdict(model), "checkpoint": None},
              "runtime": runtime_metadata(), "numerics": NUMERICS, "transform": "signed-log1p-fp32-v1",
              "encoder_signature": signature(model.encoder), "features": feature_names(model.encoder),
              "critic_indices": critic_indices(model.encoder), "weights": tensors,
              "simulator_contract_revision": SIMULATOR_REVISION, "policy_input_version": POLICY_INPUT_VERSION,
              "catalog": CATALOG_META, "end_projection_version": END_PROJECTION_VERSION,
              "reward_version": "natural-terminal-v1", "generator_version": GENERATOR_VERSION,
              "scorer_version": SCORER_VERSION, "action_config_hash": action_hash}
    atomic_json(directory / 'inference.json', result)
    return result


def inspect_artifact(directory: Path, *, expected_digest: str | None = None, action_config_hash: str | None = None,
                     model: MLPModelConfig | None = None, deadline: float = float('inf')) -> Json:
    import time
    if time.monotonic() >= deadline:
        raise TimeoutError('MLP artifact deadline')
    if expected_digest is not None and sha256(directory / 'inference.json') != expected_digest:
        raise ContractError('MLP manifest hash mismatch')
    result = json.loads((directory / 'inference.json').read_text())
    closed(result, {'schema_version', 'model', 'runtime', 'numerics', 'transform', 'encoder_signature', 'features',
                    'critic_indices', 'weights', 'simulator_contract_revision', 'policy_input_version', 'catalog',
                    'end_projection_version', 'reward_version', 'generator_version', 'scorer_version', 'action_config_hash'})
    config = mlp_model_config(result['model'])
    constants = {'schema_version': 1, 'numerics': NUMERICS, 'transform': 'signed-log1p-fp32-v1',
                 'runtime': runtime_metadata(), 'encoder_signature': signature(config.encoder),
                 'features': feature_names(config.encoder), 'critic_indices': critic_indices(config.encoder),
                 'simulator_contract_revision': SIMULATOR_REVISION, 'policy_input_version': POLICY_INPUT_VERSION,
                 'catalog': CATALOG_META, 'end_projection_version': END_PROJECTION_VERSION,
                 'reward_version': 'natural-terminal-v1', 'generator_version': GENERATOR_VERSION, 'scorer_version': SCORER_VERSION}
    if any(canonical(result[key]) != canonical(value) for key, value in constants.items()):
        raise ContractError('MLP artifact semantics/runtime mismatch')
    if model is not None and canonical(result['model']) != canonical({**asdict(model), 'checkpoint': None}):
        raise ContractError('MLP model recipe mismatch')
    if action_config_hash is not None and result['action_config_hash'] != action_config_hash:
        raise ContractError('MLP action recipe mismatch')
    load_arrays(directory / 'weights.npz', result['weights'])
    return result


def load_artifact(directory: Path, **options: Any) -> tuple[Any, Json]:
    result = inspect_artifact(directory, **options)
    network = make_network(mlp_model_config(result['model']), 0)
    restore_weights(network, load_arrays(directory / 'weights.npz', result['weights']))
    network.eval()
    return network, result
