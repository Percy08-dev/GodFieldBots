"""Closed lazy inference artifact dispatch, without importing Torch in parents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json
from gf_models.config import GRUModelConfig, MLPModelConfig, TransformerModelConfig


def inspect_artifact(directory: Path, *, model: MLPModelConfig | GRUModelConfig | TransformerModelConfig, **options: Any) -> Json:
    if isinstance(model, MLPModelConfig):
        from gf_models.mlp.model import inspect_artifact as inspect_mlp
        return inspect_mlp(directory, model=model, **options)
    elif isinstance(model, (GRUModelConfig, TransformerModelConfig)):
        from gf_models.sequence import inspect_artifact as inspect_sequence
        return inspect_sequence(directory, model=model, **options)
    else:
        raise ContractError('unsupported PPO artifact model')
