"""Pure FP32 participant features and candidate-independent critic selection."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gf_common.domain import ContractError, DecisionInput, Json
from gf_models.config import EncoderConfig
from gf_models.encoder import encode, feature_names

FloatArray = npt.NDArray[np.float32]


def critic_indices(config: EncoderConfig) -> tuple[int, ...]:
    return tuple(i for i, name in enumerate(feature_names(config))
                 if not name.startswith(("action.", "target.", "element.")))


def signed_log(value: npt.NDArray[np.float64]) -> FloatArray:
    result = (np.sign(value) * np.log1p(np.abs(value))).astype(np.float32)
    if not np.isfinite(result).all():
        raise ContractError("nonfinite transformed MLP input")
    return result


def encode_decision(value: DecisionInput, actions: tuple[Json, ...], config: EncoderConfig) -> tuple[FloatArray, FloatArray]:
    if not actions:
        raise ContractError("MLP requires candidates")
    actor = signed_log(np.stack([encode(value, action, config) for action in actions]))
    # Explicit feature-name selection; no aggregation over the candidate list.
    state = signed_log(encode(value, {"type": "pray"}, config))[list(critic_indices(config))]
    return actor, state
