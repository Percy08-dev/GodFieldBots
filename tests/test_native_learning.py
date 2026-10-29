"""Synthetic native numeric batches retain the existing Python learning math."""
import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.config import EncoderConfig
from gf_models.encoder import encode, feature_names
from gf_models.mlp.features import encode_decision
from gf_training.native_learning import (
    PPOLearning,
    fixed_consumer_layout,
    fixed_targets,
    owned_array_bytes,
)
from gf_training.ppo import RolloutDecision, gae, normalize_advantages, write_rollout

CONFIG = EncoderConfig(history='public_aggregate', max_elements=2, hash_buckets=2)


def readonly(value, dtype):
    result = np.asarray(value, dtype=dtype)
    result.flags.writeable = False
    return result


def make_batch(episodes, *, layout='all_candidates', config=CONFIG):
    features, states, selected, counts, values, probabilities = [], [], [], [], [], []
    candidate_offsets, episode_offsets = [0], [0]
    rewards, winner, viewers, splits = [], [], [], []
    for result, viewer, decisions in episodes:
        first = 0 if result == 0 else 1 if result == 1 else -1
        rewards.append((first, -first))
        winner.append(result)
        viewers.append(viewer)
        splits.append(0)
        for value, actions, selection, logprob, old_value in decisions:
            chosen = actions if layout == 'all_candidates' else (actions[selection],)
            features.extend(encode(value, action, config) for action in chosen)
            states.append(encode(value, {'type': 'pray'}, config))
            selected.append(selection)
            counts.append(len(actions))
            probabilities.append(logprob)
            values.append(old_value)
            candidate_offsets.append(len(features))
        episode_offsets.append(len(selected))
    width = len(feature_names(config))
    return SimpleNamespace(
        features=readonly(np.asarray(features).reshape(-1, width), np.float64),
        state_features=readonly(np.asarray(states).reshape(-1, width), np.float64),
        selected=readonly(selected, np.int64), candidate_counts=readonly(counts, np.int64),
        candidate_offsets=readonly(candidate_offsets, np.int64),
        episode_offsets=readonly(episode_offsets, np.int64),
        old_logprob=readonly(probabilities, np.float64), old_value=readonly(values, np.float64),
        old_logprob_present=readonly(np.ones(len(selected)), np.int64),
        old_value_present=readonly(np.ones(len(selected)), np.int64),
        terminal_rewards=readonly(np.asarray(rewards).reshape(-1, 2), np.float64),
        winner=readonly(winner, np.int64), learner_viewer=readonly(viewers, np.int64),
        split=readonly(splits, np.int64), feature_layout=layout,
    )


def legacy_decisions(episodes):
    decisions = []
    for winner, viewer, trajectory in episodes:
        if not trajectory:
            continue
        reward = 0 if winner == 0 else 1 if winner == viewer else -1
        advantages, targets = gae([item[4] for item in trajectory], reward)
        for item, advantage, target in zip(trajectory, advantages, targets, strict=True):
            value, actions, selected, old_logprob, old_value = item
            actor, state = encode_decision(value, actions, CONFIG)
            decisions.append(RolloutDecision(actor, state, selected, old_logprob, old_value,
                                             float(advantage), float(target)))
    normalized = normalize_advantages([item.advantage for item in decisions])
    return [replace(item, advantage=float(advantage))
            for item, advantage in zip(decisions, normalized, strict=True)]


@pytest.fixture
def episodes(make_input):
    value = make_input(('bronze-club', 'god-sword'))
    actions = ({'type': 'discard', 'instance_ids': [1]},
               {'type': 'discard', 'instance_ids': [2]}, {'type': 'pray'})
    other = make_input(('god-sword',))
    return [(1, 1, [(value, actions, 2, -1.1, 2.), (other, actions[-1:], 0, 0., -.5)]),
            (2, 1, []),
            (2, 2, [(value, actions[:2], 1, -.8, 3.)])]


def assert_rollouts_equal(actual, expected):
    assert len(actual) == len(expected)
    for item, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(item.actor, reference.actor)
        np.testing.assert_array_equal(item.state, reference.state)
        assert item.actor.dtype == item.state.dtype == np.float32
        for field in ('selected', 'old_logprob', 'old_value', 'advantage', 'target'):
            assert getattr(item, field) == getattr(reference, field)


def test_streamed_ppo_matches_legacy_math_and_global_normalization(episodes):
    learner = PPOLearning(CONFIG)
    first, second = make_batch(episodes[:2]), make_batch(episodes[2:])
    learner.append(first, global_ready=True)
    learner.append(second, global_ready=True)
    assert learner.retained_python_bytes() > 0
    with pytest.raises(ContractError, match='proof'):
        learner.finalize(global_ready=False)
    result = learner.finalize(global_ready=True)
    assert_rollouts_equal(result, legacy_decisions(episodes))
    np.testing.assert_array_equal(result.chunks[0].episode_offsets, [0, 2, 2])
    assert result[-1].advantage != 0  # Normalizing its one-row chunk separately gives zero.
    assert_rollouts_equal(result[:2], legacy_decisions(episodes)[:2])
    assert len(result[::-1]) == 3
    assert learner.retained_python_bytes() == 0
    assert result.retained_python_bytes() > 0
    with pytest.raises(ContractError, match='finalized'):
        learner.append(first, global_ready=True)


def test_fixed_labels_are_natural_returns_and_keep_original_selected_index(episodes):
    batch = make_batch(episodes, layout='selected')
    # First chosen index is 2 although its encoded candidate segment has length 1.
    assert batch.selected[0] == 2
    expected = np.asarray([1., 1., 1.])
    labels = fixed_targets(batch, global_ready=True)
    np.testing.assert_array_equal(labels, expected)
    assert not np.array_equal(labels, batch.old_value)
    assert labels.flags.owndata
    assert labels.dtype == np.float64
    with pytest.raises(ContractError, match='proof'):
        fixed_targets(batch, global_ready=False)


def test_both_view_rewards_and_zero_decision_episodes_are_checked_first(episodes):
    batch = make_batch(episodes)
    rewards = batch.terminal_rewards.copy()
    rewards[1, 1] = -1  # This episode has zero decisions and learner viewer 1.
    batch.terminal_rewards = readonly(rewards, np.float64)
    learner = PPOLearning(CONFIG)
    with pytest.raises(ContractError, match='both-view'):
        learner.append(batch, global_ready=True)
    assert learner.retained_python_bytes() == 0
    batch.feature_layout = 'selected'
    batch.features = readonly(batch.features[[2, 3, 5]], np.float64)
    batch.candidate_offsets = readonly([0, 1, 2, 3], np.int64)
    with pytest.raises(ContractError, match='both-view'):
        fixed_targets(batch, global_ready=True)


@pytest.mark.parametrize('field,invalid', [
    ('selected', np.asarray([3, 0, 1], dtype=np.int64)),
    ('selected', np.asarray([2, 0, 1], dtype=np.float64)),
    ('candidate_counts', np.asarray([0, 1, 2], dtype=np.int64)),
    ('candidate_offsets', np.asarray([0, 3, 2, 6], dtype=np.int64)),
    ('candidate_offsets', np.asarray([0, 3, 4, 2**63 - 1], dtype=np.int64)),
    ('episode_offsets', np.asarray([0, 2, 1, 3], dtype=np.int64)),
    ('old_value', np.asarray([2., np.nan, 3.], dtype=np.float64)),
    ('old_value_present', np.asarray([1, 0, 1], dtype=np.int64)),
    ('old_logprob_present', np.asarray([1, 2, 1], dtype=np.int64)),
    ('winner', np.asarray([1, 3, 2], dtype=np.int64)),
    ('learner_viewer', np.asarray([1, 0, 2], dtype=np.int64)),
    ('split', np.asarray([0, 1, 0], dtype=np.int64)),
])
def test_invalid_numeric_evidence_rejected(episodes, field, invalid):
    batch = make_batch(episodes)
    invalid = invalid.copy()
    invalid.flags.writeable = False
    setattr(batch, field, invalid)
    learner = PPOLearning(CONFIG)
    with pytest.raises(ContractError):
        learner.append(batch, global_ready=True)
    assert learner.retained_python_bytes() == 0


def test_missing_masks_writeable_arrays_and_missing_probability_rejected(episodes):
    learner = PPOLearning(CONFIG)
    batch = make_batch(episodes)
    del batch.old_value_present
    with pytest.raises(ContractError):
        learner.append(batch, global_ready=True)
    batch = make_batch(episodes)
    batch.old_value = batch.old_value.copy()
    with pytest.raises(ContractError):
        learner.append(batch, global_ready=True)
    batch = make_batch(episodes)
    batch.old_logprob_present = readonly([0, 1, 1], np.int64)
    batch.old_logprob = readonly([0., 0., -.8], np.float64)
    with pytest.raises(ContractError, match='saved PPO'):
        learner.append(batch, global_ready=True)


def test_empty_learner_episodes_have_no_invented_decisions(episodes):
    batch = make_batch([episodes[1]])
    learner = PPOLearning(CONFIG)
    learner.append(batch, global_ready=True)
    with pytest.raises(ContractError, match='no accepted'):
        learner.finalize(global_ready=True)
    fixed = make_batch([episodes[1]], layout='selected')
    assert fixed_targets(fixed, global_ready=True).shape == (0,)


def test_native_raw_exports_released_and_owned_capacity_not_views(episodes):
    learner = PPOLearning(CONFIG)
    batch = make_batch(episodes)
    features = weakref.ref(batch.features)
    learner.append(batch, global_ready=True)
    del batch
    gc.collect()
    assert features() is None
    owner = np.arange(100, dtype=np.float64)
    assert owned_array_bytes((owner[:3], owner[3:7], owner)) == owner.nbytes
    borrowed = np.frombuffer(bytes(64), dtype=np.float64)
    assert owned_array_bytes((borrowed,)) == 0
    result = learner.finalize(global_ready=True)
    normalized = result.advantage
    assert owned_array_bytes(tuple(chunk.advantage for chunk in result.chunks) + (normalized,)) == normalized.nbytes


def test_consumer_bounds_cover_owned_arrays_and_existing_shard_order(episodes, tmp_path, monkeypatch):
    learner = PPOLearning(CONFIG)
    batch = make_batch(episodes)
    bound = learner.consumer_bytes(decisions=3, candidates=6, episodes=3)
    assert learner.consumer_layout()[2] > 0
    assert fixed_consumer_layout(CONFIG)[2] == 0
    learner.append(batch, global_ready=True)
    assert learner.retained_python_bytes() < bound
    assert learner.finalize_bytes() >= 48 * 3
    result = learner.finalize(global_ready=True)
    expected = legacy_decisions(episodes)
    # Force multiple shards while retaining the established item-byte rule.
    monkeypatch.setattr('gf_training.ppo.MAX_ARRAY_BYTES', max(item.actor.nbytes + item.state.nbytes + 64 for item in expected))
    reference = write_rollout(tmp_path / 'reference', expected)
    actual = write_rollout(tmp_path / 'native', result)
    assert actual == reference
    assert len(actual['shards']) == 3
    with pytest.raises(ContractError):
        learner.consumer_bytes(decisions=True, candidates=1, episodes=1)
