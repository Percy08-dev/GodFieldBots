"""Explicit Python reference ingestion; native production keeps numeric chunks."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np

from gf_common.domain import ContractError, DecisionInput
from gf_models.config import SequenceModelConfig
from gf_models.encoder import feature_names
from gf_models.features import critic_indices, encode_decision
from gf_models.sequence import MAX_HISTORY
from gf_training.native_learning import BulkRollout, RolloutChunk
from gf_training.q_data import check_deadline


def empty_chunk(model: SequenceModelConfig) -> RolloutChunk:
    return RolloutChunk(np.empty((0, len(feature_names(model.encoder))), dtype=np.float32),
                        np.empty((0, len(critic_indices(model.encoder))), dtype=np.float32),
                        np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.zeros(2, dtype=np.int64))


def trajectory_chunk(rows: Any, selections: Any, advantages: Any, targets: Any,
                     model: SequenceModelConfig, deadline: float) -> RolloutChunk:
    if len(rows) > MAX_HISTORY:
        raise ContractError('sequence own-decision history resource limit exceeded')
    states = np.empty((len(rows), len(critic_indices(model.encoder))), dtype=np.float32)
    actors = []
    for index, row in enumerate(rows):
        check_deadline(deadline)
        raw = json.loads(row['input_json'])
        value = DecisionInput(raw['observation'], tuple(raw['events']), raw['history_end'],
                              tuple(raw['typed_metadata']), raw['catalog'], raw['action_options'])
        actor, states[index] = encode_decision(value, tuple(json.loads(row['candidate_set_json'])['actions']), model.encoder)
        actors.append(actor)
    offsets = np.cumsum([0] + [len(actor) for actor in actors], dtype=np.int64)
    return RolloutChunk(np.concatenate(actors), states, offsets, np.asarray([item['index'] for item in selections], dtype=np.int64),
                        np.asarray([item['logprob'] for item in selections], dtype=np.float64),
                        np.asarray([item['value'] for item in selections], dtype=np.float64), advantages, targets,
                        np.asarray([0, len(rows)], dtype=np.int64))


def finalize(chunks: list[RolloutChunk]) -> BulkRollout:
    from gf_training.ppo import normalize_advantages
    raw = np.concatenate([chunk.advantage for chunk in chunks])
    normalized = normalize_advantages(raw)
    start = 0
    output = []
    for chunk in chunks:
        stop = start + len(chunk.selected)
        output.append(replace(chunk, advantage=normalized[start:stop]))
        start = stop
    return BulkRollout(tuple(output), normalized)
