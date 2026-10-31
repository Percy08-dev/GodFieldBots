"""Python-owned targets and PPO transforms for readonly native numeric batches.

A structural proof grants permission to stage learning arrays, not permission to
fit or update a model. The integration owner retains the source/finalization gate.
This module never reconstructs policy inputs or changes native feature recipes.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from gf_common.domain import ContractError
from gf_models.config import EncoderConfig, encoder_config
from gf_models.encoder import feature_names
from gf_models.features import critic_indices, signed_log

if TYPE_CHECKING:
    from gf_training.ppo import RolloutDecision

Array = npt.NDArray[Any]


def owned_array_bytes(arrays: Sequence[Array]) -> int:
    """Count actual NumPy owner buffers once; views and native leases add zero.

    Native-exported buffers are charged by the native invocation. Only NumPy
    allocations owned on this side belong in retained_python_bytes.
    """
    owners: dict[int, Array] = {}
    for array in arrays:
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        if root.flags.owndata:
            owners[id(root)] = root
    return sum(owner.nbytes for owner in owners.values())


def _ready(global_ready: bool) -> None:
    if global_ready is not True:
        raise ContractError('trusted complete-source structural proof required')


def _array(batch: Any, name: str, dtype: Any, shape: tuple[int, ...] | None = None) -> Array:
    value = getattr(batch, name, None)
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ContractError(f'native {name} dtype/array mismatch')
    if value.flags.writeable or shape is not None and value.shape != shape:
        raise ContractError(f'native {name} must be readonly with the prescribed shape')
    if not np.isfinite(value).all():
        raise ContractError(f'nonfinite native {name}')
    return value


@dataclass(frozen=True)
class _Batch:
    features: Array
    state_features: Array
    episode_offsets: Array
    candidate_offsets: Array
    candidate_counts: Array
    selected: Array
    old_logprob: Array
    old_value: Array
    old_logprob_present: Array
    old_value_present: Array
    terminal_rewards: Array
    winner: Array
    learner_viewer: Array
    split: Array
    feature_layout: str

    @property
    def decisions(self) -> int:
        return len(self.selected)

    @property
    def episodes(self) -> int:
        return len(self.winner)


def _checked(batch: Any, width: int | None = None) -> _Batch:
    features = _array(batch, 'features', np.float64)
    state = _array(batch, 'state_features', np.float64)
    selected = _array(batch, 'selected', np.int64)
    winner = _array(batch, 'winner', np.int64)
    if features.ndim != 2 or state.ndim != 2 or selected.ndim != 1 or winner.ndim != 1:
        raise ContractError('native feature/decision/episode dimensions mismatch')
    decisions, episodes = len(selected), len(winner)
    if state.shape != (decisions, features.shape[1]) or width is not None and features.shape[1] != width:
        raise ContractError('native state/encoder feature width mismatch')
    arrays: dict[str, Array] = {}
    for name in ('candidate_counts', 'old_logprob_present', 'old_value_present'):
        arrays[name] = _array(batch, name, np.int64, (decisions,))
    for name in ('old_logprob', 'old_value'):
        arrays[name] = _array(batch, name, np.float64, (decisions,))
    for name in ('learner_viewer', 'split'):
        arrays[name] = _array(batch, name, np.int64, (episodes,))
    episode_offsets = _array(batch, 'episode_offsets', np.int64, (episodes + 1,))
    candidate_offsets = _array(batch, 'candidate_offsets', np.int64, (decisions + 1,))
    rewards = _array(batch, 'terminal_rewards', np.float64, (episodes, 2))
    layout = getattr(batch, 'feature_layout', None)
    if type(layout) is not str or layout not in ('selected', 'all_candidates'):
        raise ContractError('unsupported native feature layout')
    if episode_offsets[0] != 0 or episode_offsets[-1] != decisions or np.any(episode_offsets < 0) or np.any(episode_offsets > decisions) or np.any(np.diff(episode_offsets) < 0):
        raise ContractError('invalid native episode offsets')
    if candidate_offsets[0] != 0 or candidate_offsets[-1] != len(features) or np.any(candidate_offsets < 0) or np.any(candidate_offsets > len(features)) or np.any(np.diff(candidate_offsets) <= 0):
        raise ContractError('invalid native candidate offsets')
    counts = arrays['candidate_counts']
    if np.any(counts <= 0) or np.any(selected < 0) or np.any(selected >= counts):
        raise ContractError('invalid native original candidate count/selection')
    expected_counts = np.diff(candidate_offsets)
    if layout == 'selected':
        if len(features) != decisions or np.any(expected_counts != 1):
            raise ContractError('selected layout must contain one encoded row per decision')
    elif not np.array_equal(expected_counts, counts):
        raise ContractError('all-candidate layout count mismatch')
    for name in ('old_logprob', 'old_value'):
        mask = arrays[name + '_present']
        if np.any((mask != 0) & (mask != 1)) or np.any(arrays[name][mask == 0] != 0):
            raise ContractError('invalid native probability/value presence evidence')
    if np.any((winner < 0) | (winner > 2)) or np.any((arrays['learner_viewer'] < 1) | (arrays['learner_viewer'] > 2)) or np.any((arrays['split'] < 0) | (arrays['split'] > 1)):
        raise ContractError('invalid native terminal/viewer/split facts')
    # Check BOTH participant journals before deriving even the learner's label.
    # Includes zero-decision episodes and never infers a result from old values.
    for episode, result in enumerate(winner):
        first = 0 if result == 0 else 1 if result == 1 else -1
        if rewards[episode, 0] != first or rewards[episode, 1] != -first:
            raise ContractError('both-view natural terminal reward mismatch')
    return _Batch(features, state, episode_offsets, candidate_offsets, counts, selected,
                  arrays['old_logprob'], arrays['old_value'], arrays['old_logprob_present'],
                  arrays['old_value_present'], rewards, winner, arrays['learner_viewer'],
                  arrays['split'], layout)


def fixed_targets(batch: Any, *, global_ready: bool) -> npt.NDArray[np.float64]:
    """Stage natural-return labels after the trusted structural proof.

    Final source/ownership checks and model-fit admission remain the caller's
    responsibility. Original selected indexes are metadata, not row indexes in
    the selected-only feature matrix.
    """
    _ready(global_ready)
    value = _checked(batch)
    if value.feature_layout != 'selected':
        raise ContractError('fixed targets require selected feature layout')
    output = np.empty(value.decisions, dtype=np.float64)
    for episode in range(value.episodes):
        start, stop = value.episode_offsets[episode:episode + 2]
        output[start:stop] = value.terminal_rewards[episode, value.learner_viewer[episode] - 1]
    return output


def fixed_consumer_layout(config: EncoderConfig) -> tuple[int, int, int, int]:
    """Selected-layout labels, finite-check masks and offset/reward temporaries."""
    width = len(feature_names(encoder_config(asdict(config))))
    return (4096, width + 128, 0, 16)


@dataclass(frozen=True)
class RolloutChunk:
    """Owned numeric chunk; candidate slices are views, never action dictionaries."""
    actor: Array
    state: Array
    offsets: Array
    selected: Array
    old_logprob: Array
    old_value: Array
    advantage: Array
    target: Array
    episode_offsets: Array

    def arrays(self) -> tuple[Array, ...]:
        return (self.actor, self.state, self.offsets, self.selected, self.old_logprob,
                self.old_value, self.advantage, self.target, self.episode_offsets)


class BulkRollout(Sequence['RolloutDecision']):
    """Chunked numeric rollout compatible with the existing bounded shard writer."""
    def __init__(self, chunks: tuple[RolloutChunk, ...], advantage: Array) -> None:
        self.chunks = chunks
        self.advantage = advantage
        offsets = [0]
        for chunk in chunks:
            offsets.append(offsets[-1] + len(chunk.selected))
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return self._offsets[-1]

    @overload
    def __getitem__(self, index: int) -> RolloutDecision: ...
    @overload
    def __getitem__(self, index: slice) -> list[RolloutDecision]: ...
    def __getitem__(self, index: int | slice) -> RolloutDecision | list[RolloutDecision]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect_right(self._offsets, index) - 1
        return self._decision(self.chunks[position], index - self._offsets[position])

    @staticmethod
    def _decision(chunk: RolloutChunk, index: int) -> RolloutDecision:
        # Lazy import avoids a cycle when ppo.ingest_collection uses this adapter.
        from gf_training.ppo import RolloutDecision
        start, stop = chunk.offsets[index:index + 2]
        return RolloutDecision(chunk.actor[start:stop], chunk.state[index], int(chunk.selected[index]),
                               float(chunk.old_logprob[index]), float(chunk.old_value[index]),
                               float(chunk.advantage[index]), float(chunk.target[index]))

    def __iter__(self) -> Iterator[RolloutDecision]:
        for chunk in self.chunks:
            for index in range(len(chunk.selected)):
                yield self._decision(chunk, index)

    def retained_python_bytes(self) -> int:
        return owned_array_bytes(tuple(array for chunk in self.chunks for array in chunk.arrays())
                                 + (self.advantage,))


class PPOLearning:
    """Stage complete episodes per chunk; normalize once after the trusted EOF gate."""
    def __init__(self, config: EncoderConfig) -> None:
        self.config = encoder_config(asdict(config))
        self.width = len(feature_names(self.config))
        self.critic = critic_indices(self.config)
        self._chunks: list[RolloutChunk] = []
        self._decisions = 0
        self._finished = False

    def consumer_layout(self) -> tuple[int, int, int, int]:
        """Additional fixed/per-decision/per-candidate/per-episode byte bounds.

        Includes owned output, temporary float64 signed-log ufunc arrays, critic
        selection, validation masks/offsets, and episode GAE. Native raw buffers
        are already charged to their native owners. Bound capacities, not views.
        """
        return (4096, 32 * self.width + 8 * len(self.critic) + 256,
                32 * self.width, 16)

    def consumer_bytes(self, *, decisions: int, candidates: int, episodes: int) -> int:
        for count in (decisions, candidates, episodes):
            if type(count) is not int or count < 0:
                raise ContractError('nonnegative consumer capacity bounds required')
        fixed, per_decision, per_candidate, per_episode = self.consumer_layout()
        return fixed + decisions * per_decision + candidates * per_candidate + episodes * per_episode

    def finalize_bytes(self) -> int:
        """Additional global scalar workspace; actor/state chunks need no final copy."""
        return 4096 + 48 * self._decisions

    def retained_python_bytes(self) -> int:
        return owned_array_bytes(tuple(array for chunk in self._chunks for array in chunk.arrays()))

    def append(self, batch: Any, *, global_ready: bool) -> None:
        _ready(global_ready)
        if self._finished:
            raise ContractError('PPO learning accumulator is finalized')
        value = _checked(batch, self.width)
        if value.feature_layout != 'all_candidates' or np.any(value.split != 0):
            raise ContractError('PPO requires all-candidate train episodes')
        if np.any(value.old_logprob_present != 1) or np.any(value.old_value_present != 1):
            raise ContractError('saved PPO probability/value evidence required')
        from gf_training.ppo import gae
        # Keep the exact NumPy signed_log and existing GAE operation order.
        actor = signed_log(value.features)
        state = signed_log(value.state_features)[:, list(self.critic)]
        advantages = np.empty(value.decisions, dtype=np.float64)
        targets = np.empty(value.decisions, dtype=np.float64)
        for episode in range(value.episodes):
            start, stop = value.episode_offsets[episode:episode + 2]
            if start == stop:
                continue
            reward = value.terminal_rewards[episode, value.learner_viewer[episode] - 1]
            advantages[start:stop], targets[start:stop] = gae(value.old_value[start:stop], reward)
        chunk = RolloutChunk(actor, state, value.candidate_offsets.copy(), value.selected.copy(),
                             value.old_logprob.copy(), value.old_value.copy(), advantages, targets,
                             value.episode_offsets.copy())
        for array in chunk.arrays():
            array.flags.writeable = False
        self._chunks.append(chunk)
        self._decisions += value.decisions

    def finalize(self, *, global_ready: bool) -> BulkRollout:
        """Caller supplies True only after structural proof AND all chunks reached EOF."""
        _ready(global_ready)
        if self._finished:
            raise ContractError('PPO learning accumulator is finalized')
        if not self._decisions:
            raise ContractError('PPO batch contains no accepted learner decisions')
        from gf_training.ppo import normalize_advantages
        raw = np.concatenate([chunk.advantage for chunk in self._chunks])
        normalized = normalize_advantages(raw)
        normalized.flags.writeable = False
        chunks = []
        start = 0
        for chunk in self._chunks:
            stop = start + len(chunk.selected)
            chunks.append(replace(chunk, advantage=normalized[start:stop]))
            start = stop
        result = BulkRollout(tuple(chunks), normalized)
        self._chunks.clear()
        self._finished = True
        return result
