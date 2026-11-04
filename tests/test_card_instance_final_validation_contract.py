"""Pure final64 records/schedules; none of these records grants Source authority."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.card_instance_collection import (
    FINAL_VALIDATION,
    PPO_BATCH,
    GameCollectionRole,
    checked_invocation,
    checked_linkage,
    checked_projection,
    checked_role,
    config_projection,
    validation_invocation,
)
from gf_common.domain import ContractError, RuntimePolicyBinding, canonical, digest
from gf_common.schedule import serialize_schedule
from gf_training.collection import validation_groups
from tests.test_formal_ppo_control import loaded


def contract(tmp_path, generation=1000):
    original = loaded(tmp_path)
    cfg = replace(original.config, run=replace(original.config.run, split='validation'))
    projected = asdict(cfg)
    label = 'UNADMITTED-final-binding-only'
    registry = {label: asdict(RuntimePolicyBinding('mlp', {'expected_manifest_digest': 'a' * 64})),
        'random-card-instance-v2': asdict(RuntimePolicyBinding('random')),
        'heuristic-card-instance-v2': asdict(RuntimePolicyBinding('heuristic'))}
    schedule = serialize_schedule(validation_groups(17, label, actions_version=3), 4)
    link = {'parent_run_id': 'UNADMITTED', 'parent_run_path': str(tmp_path / 'parent'),
        'batch_id': 'validation-final', 'final_generation': generation, 'final_inference_sha256': 'a' * 64,
        'policy_label': label, 'learner_seats': {spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']},
        'schedule_digest': digest(schedule), 'policy_registry_digest': digest(registry)}
    return original, projected, registry, schedule, link


@pytest.mark.parametrize('seed', [17, 29, 43])
def test_final64_preserves_exact_opponents_seats_and_role_sampling_streams(seed):
    groups = validation_groups(seed, 'final', actions_version=3)
    assert len(groups) == 32 and sum(len(group.episodes) for group in groups) == 64
    for index, group in enumerate(groups):
        left, right = group.episodes
        opponent = ('random' if index < 16 else 'heuristic') + '-card-instance-v2'
        assert left.policies == ('final', opponent) and right.policies == (opponent, 'final')
        assert (left.learner_viewer, right.learner_viewer) == (1, 2)
        assert left.sampling_seeds == tuple(reversed(right.sampling_seeds))
        for spec in group.episodes:
            for role, position in [('learner', spec.learner_viewer - 1), ('opponent', 2 - spec.learner_viewer)]:
                assert spec.sampling_seeds[position] == int(digest({'recipe': 'ppo-final-validation-stream-v1',
                    'learning_seed': seed, 'pair': index, 'role': role})[:16], 16)
    assert [group.seed for group in groups] == [str(__import__('uuid').UUID(int=index + 1)) for index in range(32)]


@pytest.mark.parametrize('generation', [0, 1, 999, 1000])
def test_closed_final_linkage_accepts_structural_generation1000_not_update_permission(tmp_path, generation):
    _, _, registry, schedule, link = contract(tmp_path, generation)
    assert checked_linkage(link, role=FINAL_VALIDATION, schedule=schedule, registry=registry) == link
    with pytest.raises(ContractError):
        checked_linkage(link, role=PPO_BATCH, schedule=schedule, registry=registry)


@pytest.mark.parametrize('mutation', ['generation', 'batch', 'sha', 'relative', 'seat', 'missing', 'extra', 'schedule', 'registry', 'label'])
def test_closed_final_linkage_refuses_mixed_batch_fields_and_membership(tmp_path, mutation):
    _, _, registry, schedule, link = contract(tmp_path)
    if mutation == 'generation': link['final_generation'] = 1001
    elif mutation == 'batch': link['batch_id'] = 'final-validation'
    elif mutation == 'sha': link['final_inference_sha256'] = 'g' * 64
    elif mutation == 'relative': link['parent_run_path'] = 'parent'
    elif mutation == 'seat': link['learner_seats'][next(iter(link['learner_seats']))] = 0
    elif mutation == 'missing': link.pop('policy_label')
    elif mutation == 'extra': link['collection_generation'] = 7
    elif mutation == 'schedule': schedule.pop()
    elif mutation == 'registry': registry.pop('random-card-instance-v2')
    else: link['policy_label'] = ''
    with pytest.raises(ContractError):
        checked_linkage(link, role=FINAL_VALIDATION, schedule=schedule, registry=registry)


def test_role_descriptors_cannot_be_reconstructed_or_widened():
    with pytest.raises(ContractError):
        checked_role(GameCollectionRole('ppo-final-validation', 'validation', 64, FINAL_VALIDATION.linkage_keys))
    assert checked_role(FINAL_VALIDATION) is FINAL_VALIDATION


@pytest.mark.parametrize('duration,execution,admission', [(9000, 7140, 5400), (7200, 7140, 5400), (6000, 5940, 5400), (120, 60, 60), (60.5, .5, .5)])
def test_fresh_deadlines_cap_once_and_preserve60_seconds(duration, execution, admission):
    run = {'wall_seconds': 7200, 'admission_seconds': 5400, 'execution_seconds': 7140, 'finalization_seconds': 60}
    raw = validation_invocation(started_unix=123., started_monotonic=1000., caller_deadline_monotonic=1000. + duration, run=run)
    assert raw['execution_deadline_monotonic'] == 1000. + execution
    assert raw['admission_deadline_monotonic'] == 1000. + admission
    assert raw['wall_deadline_monotonic'] - raw['execution_deadline_monotonic'] >= 60
    assert checked_invocation(raw, run=run) == raw
    for key in ('admission_deadline_monotonic', 'execution_deadline_monotonic', 'wall_deadline_monotonic'):
        mutated = {**raw, key: raw[key] + 1}
        with pytest.raises(ContractError):
            checked_invocation(mutated, run=run)


@pytest.mark.parametrize('duration', [0, 59, 60, float('inf'), float('nan')])
def test_no_execution_before_caller_can_preserve_finalization_reserve(duration):
    run = {'wall_seconds': 7200, 'admission_seconds': 5400, 'execution_seconds': 7140, 'finalization_seconds': 60}
    with pytest.raises(ContractError):
        validation_invocation(started_unix=123., started_monotonic=1000., caller_deadline_monotonic=1000. + duration, run=run)


def test_projected_effective_digest_retains_original_transitive_input_byte_origins(tmp_path):
    original, projected, _, _, _ = contract(tmp_path)
    before = canonical(original.inputs)
    result = config_projection(original.effective, list(original.inputs), projected)
    assert result == {'recipe': 'run-split-train-to-validation-v1', 'parent_config_sha256': original.sha256,
        'parent_config_inputs_sha256': digest(list(original.inputs)), 'projected_config_sha256': digest(projected)}
    assert canonical(original.inputs) == before and original.effective['run']['split'] == 'train'
    assert all(Path(row['path']).is_absolute() for row in original.inputs)
    checked_projection(result, parent=original.effective, inputs=list(original.inputs), projected=projected)
    for key in ('model', 'server', 'preprocessing', 'run'):
        changed = deepcopy(projected)
        changed[key]['UNADMITTED-change'] = True
        with pytest.raises(ContractError):
            config_projection(original.effective, list(original.inputs), changed)
    with pytest.raises(ContractError):
        checked_projection(result, parent=original.effective, inputs=[], projected=projected)
