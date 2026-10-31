"""Closed sequence recipes, safe v2 artifacts and lazy network dispatch."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from gf_models.arrays import atomic_json, load_arrays, save_arrays, sha256
from gf_models.config import (
    GRUModelConfig,
    SequenceModelConfig,
    TransformerModelConfig,
    model_config,
)
from gf_models.encoder import feature_names, signature
from gf_models.features import critic_indices
from gf_models.torch_runtime import (
    NUMERICS,
    restore_weights,
    runtime_metadata,
    weights,
)

SEQUENCE_RECIPE = 'own-decision-full-history-tbptt-v1'
MAX_HISTORY = 1024


def model_backend(model: SequenceModelConfig) -> Json:
    if isinstance(model, GRUModelConfig):
        return {'gru': 'torch-nn-gru', 'cudnn_enabled': False}
    if isinstance(model, TransformerModelConfig):
        return {'attention': 'torch-sdpa', 'sdpa_backend': 'math',
                'position_encoding': 'sinusoidal-cpu-fp32-v1'}
    raise ContractError('unsupported sequence model')


def make_network(model: SequenceModelConfig, seed: int) -> Any:
    if isinstance(model, GRUModelConfig):
        from gf_models.gru.model import make_network as make_gru
        return make_gru(model, seed)
    if isinstance(model, TransformerModelConfig):
        from gf_models.transformer.model import make_network as make_transformer
        return make_transformer(model, seed)
    raise ContractError('unsupported sequence network model')


def _semantics(model: SequenceModelConfig) -> Json:
    return {'schema_version': 2, 'runtime': runtime_metadata(), 'numerics': NUMERICS,
            'transform': 'signed-log1p-fp32-v1', 'encoder_signature': signature(model.encoder),
            'features': feature_names(model.encoder), 'critic_indices': critic_indices(model.encoder),
            'sequence_recipe': SEQUENCE_RECIPE, 'max_history': MAX_HISTORY, 'backend': model_backend(model),
            'simulator_contract_revision': SIMULATOR_REVISION, 'policy_input_version': POLICY_INPUT_VERSION,
            'catalog': CATALOG_META, 'end_projection_version': END_PROJECTION_VERSION,
            'reward_version': 'natural-terminal-v1', 'generator_version': GENERATOR_VERSION,
            'scorer_version': SCORER_VERSION}


def save_artifact(network: Any, directory: Path, model: SequenceModelConfig, action_hash: str) -> Json:
    directory.mkdir(parents=True, exist_ok=False)
    tensors = save_arrays(directory / 'weights.npz', weights(network))
    result = {**_semantics(model), 'model': {**asdict(model), 'checkpoint': None},
              'weights': tensors, 'action_config_hash': action_hash}
    atomic_json(directory / 'inference.json', result)
    return result


def inspect_artifact(directory: Path, *, expected_digest: str | None = None,
                     action_config_hash: str | None = None, model: SequenceModelConfig | None = None,
                     deadline: float = float('inf')) -> Json:
    if time.monotonic() >= deadline:
        raise TimeoutError('sequence artifact deadline')
    if expected_digest is not None and sha256(directory / 'inference.json', deadline) != expected_digest:
        raise ContractError('sequence manifest hash mismatch')
    result = json.loads((directory / 'inference.json').read_text())
    closed(result, set(_semantics(GRUModelConfig())) | {'model', 'weights', 'action_config_hash'})
    config = model_config(result['model'])
    if not isinstance(config, (GRUModelConfig, TransformerModelConfig)):
        raise ContractError('unsupported sequence artifact model')
    if any(canonical(result[key]) != canonical(value) for key, value in _semantics(config).items()):
        raise ContractError('sequence artifact semantics/runtime mismatch')
    if model is not None and canonical(result['model']) != canonical({**asdict(model), 'checkpoint': None}):
        raise ContractError('sequence model recipe mismatch')
    if action_config_hash is not None and result['action_config_hash'] != action_config_hash:
        raise ContractError('sequence action recipe mismatch')
    load_arrays(directory / 'weights.npz', result['weights'])
    if time.monotonic() >= deadline:
        raise TimeoutError('sequence artifact deadline')
    return result


def load_artifact(directory: Path, **options: Any) -> tuple[Any, Json]:
    result = inspect_artifact(directory, **options)
    config = model_config(result['model'])
    if not isinstance(config, (GRUModelConfig, TransformerModelConfig)):
        raise ContractError('unsupported sequence artifact model')
    network = make_network(config, 0)
    restore_weights(network, load_arrays(directory / 'weights.npz', result['weights']))
    network.eval()
    return network, result
