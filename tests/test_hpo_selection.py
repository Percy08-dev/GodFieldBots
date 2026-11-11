"""Endpoint selection is independent of policy/trainer streams."""
import copy
from collections import Counter

import numpy as np
import pytest

from gf_common.domain import ContractError, canonical


def test_selection_all_partial_one_empty_and_zero_budget():
    from gf_training.hpo_collection import selection_for_counts
    for remaining in (0, 1, 3, 7, 20):
        actual = selection_for_counts((2, 0, 5), learning_seed=17, batch=3,
            decision_budget=20, used_before=20 - remaining)
        assert actual == selection_for_counts((2, 0, 5), learning_seed=17, batch=3,
            decision_budget=20, used_before=20 - remaining)
        assert actual['collected_decisions'] == 7
        assert actual['used_decisions'] == min(remaining, 7)
        assert actual['indices'] == sorted(set(actual['indices']))
        if remaining >= 7:
            assert actual['indices'] == list(range(7))
    assert selection_for_counts((0, 0), learning_seed=17, batch=0,
        decision_budget=4, used_before=0)['indices'] == []


def test_replay_rejects_rehashed_indices_seed_count_and_context():
    from gf_common.domain import digest
    from gf_training.hpo_collection import replay_selection, selection_for_counts
    actual = selection_for_counts((4, 3), learning_seed=17, batch=1, decision_budget=3, used_before=0)
    assert len(replay_selection(actual, (4, 3))) == 3
    for field in ('indices', 'derived_seed', 'episode_counts', 'context'):
        changed = copy.deepcopy(actual)
        if field == 'indices':
            changed[field] = next([*value] for value in ((0, 1, 2), (3, 4, 5)) if list(value) != actual[field])
            changed['indices_sha256'] = digest(changed[field])
        elif field == 'episode_counts': changed[field] = [3, 4]
        elif field == 'context': changed[field]['remaining'] += 1
        else: changed[field] += 1
        with pytest.raises(ContractError): replay_selection(changed, (4, 3))


@pytest.mark.parametrize('games', [8, 16, 24, 88])
def test_variable_schedule_ratios_seats_order_and_rng_isolation(games):
    from gf_training.hpo_collection import training_groups_hpo
    rng = np.random.Generator(np.random.PCG64(17))
    groups = training_groups_hpo(17, 0, 'same', ('same',), rng, collection_games=games)
    assert len(groups) == games // 2
    assert Counter(group.episodes[0].policies[1] for group in groups) == {
        'random-card-instance-v2': games // 8, 'heuristic-card-instance-v2': games // 8, 'same': games // 4}
    for pair, group in enumerate(groups):
        assert group.episodes[0].episode_id == f'b000000-p{pair}-seat0'
        assert [episode.learner_viewer for episode in group.episodes] == [1, 2]
        assert group.episodes[0].sampling_seeds == group.episodes[1].sampling_seeds[::-1]
    before = canonical(rng.bit_generator.state)
    from gf_training.hpo_collection import selection_for_counts
    selection_for_counts((10,), learning_seed=17, batch=0, decision_budget=3, used_before=0)
    assert canonical(rng.bit_generator.state) == before


@pytest.mark.parametrize('batch', [1000, 1_000_000])
def test_hpo_linkage_keeps_declared_generation_domain_and_canonical_names(tmp_path, batch):
    from gf_common.card_instance_collection import checked_linkage, hpo_batch_role
    from gf_common.domain import digest
    from gf_common.hpo import selection_context
    from gf_common.schedule import serialize_schedule
    from gf_training.hpo_collection import training_groups_hpo
    groups = training_groups_hpo(17, batch, 'learner', ('learner',),
        np.random.Generator(np.random.PCG64(17)), collection_games=8)
    schedule = serialize_schedule(groups, 4)
    registry = {'learner': {'model_id': 'mlp', 'policy_options': None}}
    link = {'parent_run_id': 'controlled', 'parent_run_path': str(tmp_path),
        'batch_id': f'batch-{batch:06}', 'collection_generation': batch,
        'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest': digest(schedule), 'policy_registry_digest': digest(registry),
        'selection_context': selection_context(decision_budget=3, used_before=0, learning_seed=17, batch=batch)}
    assert checked_linkage(link, role=hpo_batch_role(8), schedule=schedule, registry=registry) == link
    broken = {**link, 'batch_id': f'batch-{batch+1:06}'}
    with pytest.raises(ContractError): checked_linkage(broken, role=hpo_batch_role(8), schedule=schedule, registry=registry)


def test_original_ppo_linkage_retains_generation_999_bound(tmp_path):
    from gf_common.card_instance_collection import PPO_BATCH, checked_linkage
    from gf_common.domain import digest
    schedule = [{'episodes': [{'episode_id': f'e{index}', 'learner_viewer': index%2+1} for index in range(16)]}]
    registry = {}
    link = {'parent_run_id': 'controlled', 'parent_run_path': str(tmp_path), 'batch_id': 'batch-000999',
        'collection_generation': 999, 'learner_seats': {f'e{index}': index%2+1 for index in range(16)},
        'schedule_digest': digest(schedule), 'policy_registry_digest': digest(registry)}
    assert checked_linkage(link, role=PPO_BATCH, schedule=schedule, registry=registry) == link
    with pytest.raises(ContractError):
        checked_linkage({**link, 'collection_generation': 1000, 'batch_id': 'batch-001000'},
            role=PPO_BATCH, schedule=schedule, registry=registry)
