from collections import Counter

import numpy as np


def test_ordinary_schedule_scales_paired_curriculum_and_preserves_legacy_stream():
    from gf_training.collection import ordinary_rust_training_groups, training_groups

    def schedule(rng, **kwargs):
        return ordinary_rust_training_groups(17, 2, 'learner', ('initial', 'prior'), rng, **kwargs)

    legacy_rng = np.random.default_rng(29)
    legacy = training_groups(17, 2, 'learner', ('initial', 'prior'), legacy_rng, actions_version=3)
    explicit_rng = np.random.default_rng(29)
    assert schedule(explicit_rng, collection_games=16) == legacy
    assert explicit_rng.bit_generator.state == legacy_rng.bit_generator.state

    groups = schedule(np.random.default_rng(29), collection_games=256)
    assert len(groups) == len({group.seed for group in groups}) == 128
    counts = Counter()
    for group in groups:
        first, second = group.episodes
        assert first.learner_viewer == 1 and second.learner_viewer == 2
        assert first.pair_id == second.pair_id
        assert first.policies == second.policies[::-1]
        assert first.sampling_seeds == second.sampling_seeds[::-1]
        counts[first.policies[1]] += 2
    assert counts['random-card-instance-v2'] == counts['heuristic-card-instance-v2'] == 64
    assert counts['initial'] + counts['prior'] == 128


def test_cold_ordinary_audit_binds_game_count_and_step_delta():
    import pytest

    from gf_common.domain import ContractError
    from gf_models.config import PPOUpdateConfig
    from gf_training.rust_player_completion import _audit_ordinary_update_counts

    recipe = PPOUpdateConfig(collection_games=256, minibatch_size=64, epochs=2)
    previous = {'optimizer_steps': 5, 'accepted_decisions': 122}
    current = {'optimizer_steps': 9, 'accepted_decisions': 187}
    update = {'actual_games': 256, 'planned_games': 256, 'metrics': {'optimizer_steps': 9}}
    _audit_ordinary_update_counts(recipe, previous, current, update, decisions=65)
    for changes in ({'actual_games': 16}, {'planned_games': 16}, {'metrics': {'optimizer_steps': 7}}):
        with pytest.raises(ContractError):
            _audit_ordinary_update_counts(recipe, previous, current, update | changes, decisions=65)
    with pytest.raises(ContractError):
        _audit_ordinary_update_counts(recipe, previous, current | {'optimizer_steps': 7}, update, decisions=65)


def test_grouped_model3_planning_accepts_128_and_splits_before_packing():
    import pytest

    from gf_training.hpo_update_planning import UpdateCapacityError, plan_groups
    from tests.test_card_instance_learning import learner_for, rollout_for

    learner = learner_for()
    rollout = rollout_for(learner, count=1)
    endpoint = rollout.endpoint(0)

    class UniformInput:
        resident_bytes = rollout.resident_bytes
        def endpoint(self, index):
            return endpoint

    def plan(headroom):
        return plan_groups(UniformInput(), list(range(128)), learner.model, learner.binding, 1,
            variant='microbatch', reservation_bytes=learner.binding.managed_budget_bytes,
            layout=learner.network.layout, mlp_width=128, packed_device_headroom=headroom)

    wide, = plan(1024**3)
    assert wide.indices == tuple(range(128))
    assert wide.plan.staging is None
    memory = wide.plan.estimate['packed_bytes'] + wide.plan.estimate['activation_bytes']
    split = plan(memory - 1)
    assert 1 < len(split) < 128
    assert tuple(index for group in split for index in group.indices) == tuple(range(128))
    assert all(group.plan.staging is None and
        group.plan.estimate['packed_bytes'] + group.plan.estimate['activation_bytes'] < memory
        for group in split)
    with pytest.raises(UpdateCapacityError, match='headroom'):
        plan(1)


def test_four_ordinary_shuffles_preserve_exact_episode_semantics():
    from gf_training.structured_numeric import _ppo_epoch_order

    runs = (range(3), range(3, 4), range(4, 8))
    for seed in range(3):
        for strategy in ('all', 'episode_order', 'within_episode', 'episode_interleave'):
            rng = np.random.default_rng(seed)
            expected_rng = np.random.default_rng(seed)
            if strategy == 'all':
                expected = expected_rng.permutation(8).tolist()
            else:
                episodes = (expected_rng.permutation(3).tolist()
                    if strategy in ('episode_order', 'episode_interleave') else list(range(3)))
                orders = [list(runs[episode]) if strategy == 'episode_order' else
                    expected_rng.permutation(list(runs[episode])).tolist() for episode in episodes]
                expected = ([rows[position] for position in range(4) for rows in orders if position < len(rows)]
                    if strategy == 'episode_interleave' else [index for rows in orders for index in rows])
            actual = _ppo_epoch_order(rng, 8, runs, strategy)
            assert actual == expected and sorted(actual) == list(range(8))
            assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_ordinary_cpu_batches_advance_once_per_tail_and_resume_exactly(tmp_path):
    from gf_models.config import PPOUpdateConfig
    from tests.test_card_instance_learning import learner_for, rollout_for
    from tests.test_structured_learning import assert_same_learners

    running = learner_for()
    rollout = rollout_for(running, count=3)
    recipe = PPOUpdateConfig(minibatch_size=2, epochs=3, shuffle_strategy='within_episode',
        gpu_microbatch_size=128)
    result = running._update_owned_model3(rollout, recipe=recipe)
    assert result['optimizer_steps'] == running.steps == 6
    assert result['parity'] == {'status': 'not_run'}
    directory = tmp_path / 'saved'
    running.save(directory, shard_bytes=1024**2)
    restored = learner_for()
    restored.restore(directory)
    assert restored._update_owned_model3(rollout, recipe=recipe) == running._update_owned_model3(rollout, recipe=recipe)
    assert restored.steps == running.steps == 12
    assert_same_learners(running, restored)


def test_model3_packed_loss_excludes_padding_and_keeps_gradients():
    import pytest

    torch = pytest.importorskip('torch')

    from tests.test_card_instance_learning import learner_for, rollout_for

    learner = learner_for()
    episodes = (rollout_for(learner, count=1, empty=True), rollout_for(learner, count=1))

    class UnequalInput:
        def endpoint(self, index):
            return episodes[index].endpoint(0)

        def vector(self, index):
            return episodes[index].vector(0)

    rollout = UnequalInput()
    scalar, scalar_metrics = zip(*(learner._scalar_model3_loss(rollout, index) for index in range(2)), strict=True)
    expected = torch.stack(scalar).mean()
    expected.backward()
    scalar_gradients = [None if param.grad is None else param.grad.clone() for param in learner.network.parameters()]
    learner.optimizer.zero_grad(set_to_none=True)
    loss, metrics = learner._packed_model3_loss(rollout, (0, 1))
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
    assert metrics == pytest.approx({key: sum(row[key] for row in scalar_metrics) / 2 for key in metrics}, abs=1e-5)
    loss.backward()
    for param, reference in zip(learner.network.parameters(), scalar_gradients, strict=True):
        if reference is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, reference, atol=1e-5, rtol=1e-5)
