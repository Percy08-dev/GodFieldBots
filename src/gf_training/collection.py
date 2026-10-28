"""Closed collection requests; training depends on this callable contract only."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from gf_common.domain import EpisodeGroup, EpisodeSpec, Json, RuntimePolicyBinding, digest

TRAIN_NAMESPACE = uuid.UUID('2de8b1c1-6db1-5cc9-a2e7-933ce3d10001')
TEST_NAMESPACE = uuid.UUID('2de8b1c1-6db1-5cc9-a2e7-933ce3d10002')


@dataclass(frozen=True)
class CollectionPlan:
    groups: tuple[EpisodeGroup, ...]
    bindings: dict[str, RuntimePolicyBinding]
    output: Path
    linkage: Json
    execution_deadline: float
    wall_deadline: float
    kind: str = 'ppo_batch'
    started_monotonic: float | None = None


@dataclass(frozen=True)
class FinalizedCollection:
    directory: Path
    report: Json


def training_groups(seed: int, batch: int, learner: str, snapshots: tuple[str, ...], rng: object) -> tuple[EpisodeGroup, ...]:
    import numpy as np
    assert isinstance(rng, np.random.Generator)
    groups = []
    opponents = ['random'] * 2 + ['heuristic'] * 2 + [snapshots[int(rng.integers(len(snapshots)))] for _ in range(4)]
    for pair, opponent in enumerate(opponents):
        game_seed = str(uuid.uuid5(TRAIN_NAMESPACE, f'ppo-train-v1/{seed}/{batch}/{pair}'))
        pair_id = f'b{batch:06}-p{pair}'
        streams = tuple(int(digest({'recipe': 'ppo-policy-stream-v1', 'seed': seed, 'batch': batch, 'pair': pair, 'role': role})[:16], 16) for role in ('learner', 'opponent'))
        sampling = (streams[0], streams[1])
        episodes = (EpisodeSpec(pair_id + '-seat0', (learner, opponent), sampling, pair_id, 1),
                    EpisodeSpec(pair_id + '-seat1', (opponent, learner), (sampling[1], sampling[0]), pair_id, 2))
        groups.append(EpisodeGroup(game_seed, episodes))
    return tuple(groups)


def validation_groups(seed: int, learner: str) -> tuple[EpisodeGroup, ...]:
    groups = []
    for index in range(32):
        opponent = 'random' if index < 16 else 'heuristic'
        pair_id = f'validation-p{index:02}'
        streams = [int(digest({'recipe': 'ppo-final-validation-stream-v1', 'learning_seed': seed, 'pair': index, 'role': role})[:16], 16)
                   for role in ('learner', 'opponent')]
        groups.append(EpisodeGroup(str(uuid.UUID(int=index + 1)), (
            EpisodeSpec(pair_id + '-seat0', (learner, opponent), (streams[0], streams[1]), pair_id, 1),
            EpisodeSpec(pair_id + '-seat1', (opponent, learner), (streams[1], streams[0]), pair_id, 2))))
    return tuple(groups)


@dataclass(frozen=True)
class OnlinePPOContext:
    parent_directory: Path
    parent_run_id: str
    batch_id: str
    generation: int
    ownership_token: str
