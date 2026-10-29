"""Pure accepted-trajectory targets and bounded numeric rollout serialization."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, DecisionInput, Json, closed
from gf_models.config import MLPModelConfig
from gf_models.mlp.arrays import MAX_ARRAY_BYTES, load_arrays, save_arrays, sha256
from gf_models.mlp.features import encode_decision
from gf_training.journal import atomic_json
from gf_training.q_data import check_deadline, load_run, validate_bundles


@dataclass(frozen=True)
class RolloutDecision:
    actor: Any
    state: Any
    selected: int
    old_logprob: float
    old_value: float
    advantage: float
    target: float


def gae(values: Any, terminal_reward: float, lam: float = .95) -> tuple[Any, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or terminal_reward not in (-1, 0, 1):
        raise ContractError('natural finite trajectory required')
    advantages = np.zeros_like(values)
    carry, next_value = 0., 0.
    for index in range(len(values) - 1, -1, -1):
        reward = terminal_reward if index == len(values) - 1 else 0.
        carry = reward + next_value - values[index] + lam * carry
        advantages[index] = carry
        next_value = values[index]
    return advantages, values + advantages


def normalize_advantages(values: Any) -> Any:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ContractError('finite batch advantages required')
    return ((values - values.mean()) / (values.std(ddof=0) + 1e-8)).astype(np.float32)


def ingest_collection(directory: Path, model: MLPModelConfig, linkage: Json, *, deadline: float, online_context: Any) -> tuple[list[RolloutDecision], Json]:
    bundle = load_run(directory, 'train', deadline=deadline, online_context=online_context)
    manifest = bundle.manifest
    if manifest.get('schema_version') != 4 or manifest.get('collection_kind') != 'ppo_batch' or manifest.get('ppo') != linkage:
        raise ContractError('PPO exact parent/batch/generation linkage required')
    rows, evidence = validate_bundles([bundle], deadline=deadline, online_context=online_context)
    seats = {spec['episode_id']: spec['learner_viewer'] for group in manifest['schedule'] for spec in group['episodes']}
    returns = {(source['header']['game_id'], source['header']['viewer']): source['end']['terminal_reward'] for source in evidence['sources']}
    episode_games = {episode['game_id']: episode['episode_id'] for episode in manifest['episodes']}
    trajectories: dict[str, list[Json]] = {episode['game_id']: [] for episode in manifest['episodes']}
    for row in rows:
        if row['viewer'] == seats[episode_games[row['game_id']]]:
            trajectories[row['game_id']].append(row)
    decisions = []
    for game_id, trajectory in trajectories.items():
        check_deadline(deadline)
        trajectory.sort(key=lambda row: row['decision'])
        if not trajectory:
            continue  # A learner may naturally finish without taking an action.
        selections = [json.loads(row['selection_json']) for row in trajectory]
        episode_name = episode_games[game_id]
        expected_label = next(spec['policies'][seats[episode_name] - 1] for group in manifest['schedule'] for spec in group['episodes'] if spec['episode_id'] == episode_name)
        if not expected_label.startswith('mlp-') or any(item['policy_version'] != 'mlp-ppo-v1-' + expected_label[4:] for item in selections):
            raise ContractError('learner selection identity differs from frozen generation')
        if any(type(item['logprob']) not in (float, int) or type(item['value']) not in (float, int) or not item['policy_version'].startswith('mlp-ppo-v1-') for item in selections):
            raise ContractError('saved PPO probability/value evidence required')
        advantages, targets = gae([item['value'] for item in selections], returns[(game_id, seats[episode_games[game_id]])])
        for row, selection, advantage, target in zip(trajectory, selections, advantages, targets, strict=True):
            check_deadline(deadline)
            raw = json.loads(row['input_json'])
            value = DecisionInput(raw['observation'], tuple(raw['events']), raw['history_end'], tuple(raw['typed_metadata']), raw['catalog'], raw['action_options'])
            candidates = json.loads(row['candidate_set_json'])['actions']
            actor, state = encode_decision(value, tuple(candidates), model.encoder)
            decisions.append(RolloutDecision(actor, state, selection['index'], float(selection['logprob']), float(selection['value']), float(advantage), float(target)))
    if not decisions:
        raise ContractError('PPO batch contains no accepted learner decisions')
    normalized = normalize_advantages([item.advantage for item in decisions])
    result = [RolloutDecision(item.actor, item.state, item.selected, item.old_logprob, item.old_value, float(adv), item.target)
              for item, adv in zip(decisions, normalized, strict=True)]
    return result, {'source_set_sha256': evidence['source_set_sha256'], 'manifest_sha256': sha256(directory / 'manifest.json'),
                    'decisions': len(result), 'games': len(manifest['episodes'])}


def write_rollout(directory: Path, decisions: Sequence[RolloutDecision], *, deadline: float = float("inf")) -> Json:
    directory.mkdir(parents=True, exist_ok=False)
    shards: list[Json] = []
    current: list[RolloutDecision] = []
    size = 0
    def flush() -> None:
        nonlocal size
        if not current:
            return
        offsets = np.cumsum([0] + [len(item.actor) for item in current], dtype=np.int64)
        arrays = {'actor': np.concatenate([item.actor for item in current]).astype(np.float32),
                  'state': np.stack([item.state for item in current]).astype(np.float32), 'offsets': offsets,
                  'selected': np.asarray([item.selected for item in current], dtype=np.int64),
                  'old_logprob': np.asarray([item.old_logprob for item in current], dtype=np.float32),
                  'old_value': np.asarray([item.old_value for item in current], dtype=np.float32),
                  'advantage': np.asarray([item.advantage for item in current], dtype=np.float32),
                  'target': np.asarray([item.target for item in current], dtype=np.float32)}
        name = f'shard-{len(shards):06}.npz'
        check_deadline(deadline)
        metadata = save_arrays(directory / name, arrays)
        check_deadline(deadline)
        shards.append({'file': name, 'metadata': metadata})
        current.clear()
        size = 0
    for item in decisions:
        check_deadline(deadline)
        item_bytes = item.actor.nbytes + item.state.nbytes + 64
        if item_bytes > MAX_ARRAY_BYTES:
            raise ContractError('single decision exceeds numeric shard bound')
        if size + item_bytes > MAX_ARRAY_BYTES:
            flush()
        current.append(item)
        size += item_bytes
    flush()
    manifest = {'schema_version': 1, 'decision_count': len(decisions), 'shards': shards}
    check_deadline(deadline)
    atomic_json(directory / 'rollout.json', manifest)
    check_deadline(deadline)
    return manifest


def read_rollout(directory: Path, expected_hash: str) -> list[RolloutDecision]:
    if sha256(directory / 'rollout.json') != expected_hash:
        raise ContractError('rollout digest mismatch')
    manifest = json.loads((directory / 'rollout.json').read_text())
    closed(manifest, {'schema_version', 'decision_count', 'shards'})
    if manifest['schema_version'] != 1:
        raise ContractError('unsupported rollout schema')
    output = []
    for shard in manifest['shards']:
        closed(shard, {'file', 'metadata'})
        if Path(shard['file']).name != shard['file']:
            raise ContractError('unsafe shard path')
        arrays = load_arrays(directory / shard['file'], shard['metadata'])
        if set(arrays) != {'actor', 'state', 'offsets', 'selected', 'old_logprob', 'old_value', 'advantage', 'target'}:
            raise ContractError('rollout keys mismatch')
        count = len(arrays['selected'])
        offsets = arrays['offsets']
        if offsets.dtype != np.int64 or offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(arrays['actor']) or np.any(np.diff(offsets) <= 0):
            raise ContractError('invalid rollout offsets')
        for key in ('old_logprob', 'old_value', 'advantage', 'target'):
            if arrays[key].shape != (count,) or arrays[key].dtype != np.float32:
                raise ContractError('invalid rollout scalar array')
        if arrays['selected'].dtype != np.int64 or arrays['state'].ndim != 2 or len(arrays['state']) != count or arrays['actor'].ndim != 2 or arrays['state'].dtype != np.float32 or arrays['actor'].dtype != np.float32:
            raise ContractError('invalid rollout features')
        for index in range(count):
            actor = arrays['actor'][offsets[index]:offsets[index + 1]]
            selected = int(arrays['selected'][index])
            if not 0 <= selected < len(actor):
                raise ContractError('invalid rollout selected index')
            output.append(RolloutDecision(actor, arrays['state'][index], selected, float(arrays['old_logprob'][index]), float(arrays['old_value'][index]), float(arrays['advantage'][index]), float(arrays['target'][index])))
    if not output or len(output) != manifest['decision_count']:
        raise ContractError('rollout count mismatch')
    return output
