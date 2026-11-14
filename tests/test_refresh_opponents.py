"""Explicit opt-in curriculum; fixture schedules carry no game/learning evidence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from gf_common.domain import ContractError, canonical, digest

ROOT = Path(__file__).parents[1]
DESCRIPTOR = {'version': '1', 'preset': 'mixed-refresh-16-v1',
    'refresh': {'version': '1', 'attack_threshold': 12}}


def ordinary_config(tmp_path, *, packed=False, threshold=12):
    from gf_cli.config import load_config
    source = ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml'
    path = tmp_path / 'mixed.toml'
    text = source.read_text().replace('"../', '"' + str(ROOT / 'configs') + '/')
    text += ('\n[training_opponents]\nversion="1"\npreset="mixed-refresh-16-v1"\n'
        f'[training_opponents.refresh]\nversion="1"\nattack_threshold={threshold}\n')
    if packed:
        text += '\n[storage]\n'
    path.write_text(text)
    return load_config(path)


def test_ordinary_config_explicit_variant_coherent_and_old_defaults_exact(tmp_path):
    from gf_cli.config import ExperimentConfig, GroupPackExperimentConfig, load_config
    old = load_config(ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml')
    assert 'training_opponents' not in asdict(ExperimentConfig())
    assert 'training_opponents' not in asdict(GroupPackExperimentConfig())
    assert 'training_opponents' not in old.effective
    for packed in (False, True):
        loaded = ordinary_config(tmp_path, packed=packed)
        assert loaded.config.training_opponents == DESCRIPTOR
        assert loaded.effective == __import__('json').loads(canonical(asdict(loaded.config)))
        assert loaded.sha256 == digest(asdict(loaded.config))
        assert isinstance(loaded.config, GroupPackExperimentConfig) == packed
    assert ordinary_config(tmp_path, threshold=2**130).config.training_opponents['refresh']['attack_threshold'] == 2**130


@pytest.mark.parametrize('mutation', [
    lambda v: v.update(version='2'), lambda v: v.update(preset='other'),
    lambda v: v.update(extra=1), lambda v: v.pop('refresh'),
    lambda v: v['refresh'].update(extra=1), lambda v: v['refresh'].update(attack_threshold=True),
    lambda v: v['refresh'].update(attack_threshold=0), lambda v: v['refresh'].update(version='2'),
])
def test_closed_descriptor(mutation):
    import gf_common.training_opponents as module
    raw = deepcopy(DESCRIPTOR)
    mutation(raw)
    with pytest.raises(ContractError):
        module.parse_training_opponents(raw)


@pytest.mark.parametrize('hpo', [False, True])
def test_mixed_schedule_replaces_only_pair3_with_identical_rng_and_seats(hpo):
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
    from gf_training import collection, hpo_collection
    from tests.test_card_instance_learning import ACTIONS
    original = hpo_collection.training_groups_hpo if hpo else collection.training_groups
    mixed = getattr(hpo_collection if hpo else collection, 'training_groups_hpo_refresh' if hpo else 'training_groups_refresh', None)
    assert callable(mixed), 'concrete mixed schedule wrapper required'
    first = np.random.Generator(np.random.PCG64(17))
    second = np.random.Generator(np.random.PCG64(17))
    options = {'collection_games': 16} if hpo else {'actions_version': 3}
    old = original(17, 2, 'learner', ('snap0', 'snap1', 'snap2'), first, **options)
    new = mixed(17, 2, 'learner', ('snap0', 'snap1', 'snap2'), second,
        actions=ACTIONS, training_opponents=DESCRIPTOR, **({'collection_games': 16} if hpo else {}))
    label = refresh_policy_label(ACTIONS, card_instance_rules(), RefreshOptions())
    assert len(new) == 8 and sum(len(g.episodes) for g in new) == 16
    for pair, (a, b) in enumerate(zip(old, new, strict=True)):
        assert a.seed == b.seed
        for x, y in zip(a.episodes, b.episodes, strict=True):
            assert x.episode_id == y.episode_id and x.pair_id == y.pair_id
            assert x.sampling_seeds == y.sampling_seeds and x.learner_viewer == y.learner_viewer
            assert y.policies == tuple(label if p == 'heuristic-card-instance-v2' and pair == 3 else p for p in x.policies)
    assert first.bit_generator.state == second.bit_generator.state


def test_hpo_schema2_descriptor_study_identity_and_materialization(tmp_path):
    from gf_cli.hpo_config import HPOCampaignConfig, campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_lineage import hpo_spec
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw['storage'] = {'parent_root': str(tmp_path / 'parents'), 'numeric_root': str(tmp_path / 'parents')}
    old = campaign_config(raw, origin=tmp_path)
    assert 'training_opponents' not in asdict(old.config)
    raw.update(schema_version=2, training_opponents=deepcopy(DESCRIPTOR))
    loaded = campaign_config(raw, origin=tmp_path)
    assert type(loaded.config) is not HPOCampaignConfig
    assert loaded.config.study_identity != old.config.study_identity
    changed = deepcopy(raw)
    changed['training_opponents']['refresh']['attack_threshold'] = 13
    assert campaign_config(changed, origin=tmp_path).config.study_identity != loaded.config.study_identity
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id='00000000-0000-0000-0000-000000000001')
    spec = hpo_spec({'effective_config': run.effective, 'hpo_spec_sha256': digest(run.config.hpo_spec)})
    assert spec['schema_version'] == 2 and spec['training_opponents'] == DESCRIPTOR
    assert 'training_opponents' not in run.effective
    assert settings.identity == effective_settings(old.config.model, old.config.training, {}, layout=LAYOUT).identity


@pytest.mark.parametrize('schema,descriptor', [(1, DESCRIPTOR), (2, None)])
def test_hpo_schema_option_pair_is_explicit(tmp_path, schema, descriptor):
    from gf_cli.hpo_config import campaign_config
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw['schema_version'] = schema
    if descriptor is not None:
        raw['training_opponents'] = descriptor
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)


@pytest.mark.parametrize('games', [8, 24, 32])
def test_hpo_mixed_rejects_other_collection_counts_before_admission(tmp_path, games):
    from gf_cli.hpo_config import campaign_config
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw.update(schema_version=2, training_opponents=DESCRIPTOR, training={'collection_games': games})
    with pytest.raises(ContractError, match='16'):
        campaign_config(raw, origin=tmp_path)


def test_ordinary_reconstructed_registry_is_full_refresh_and_tamper_refuses(tmp_path):
    import time

    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
    from gf_training.card_instance_source import recorded_plan
    from tests.card_instance_learning_support import initial_parent
    from tests.test_card_instance_learning import ACTIONS
    directory = tmp_path / 'parent'
    from gf_training.lineage import run_lock
    with run_lock(directory, create=True):
        _, state, records, _ = initial_parent(directory, training_opponents=DESCRIPTOR)
        plan, _, _ = recorded_plan(directory, state, records, 0, time.monotonic()+60)
    label = refresh_policy_label(ACTIONS, card_instance_rules(), RefreshOptions())
    assert plan.bindings[label].model_id == 'heuristic_refresh'
    assert plan.bindings[label].policy_options == DESCRIPTOR['refresh']
    assert plan.groups[3].episodes[0].policies[1] == label
    broken = deepcopy(state)
    broken['effective_config']['training_opponents']['refresh']['attack_threshold'] = 13
    with pytest.raises(ContractError):
        recorded_plan(directory, broken, records, 0, time.monotonic()+60)


def test_hpo_reconstructed_registry_and_independent_lineage_agree(tmp_path):
    import time

    from gf_training.hpo_lineage import _policies
    from gf_training.hpo_source import recorded_hpo_plan
    from tests.hpo_support import variable_initial_hpo_chain
    directory = tmp_path / 'parent'
    state, records, learner = variable_initial_hpo_chain(directory, games=16, training_opponents=DESCRIPTOR)
    plan, learner_label, _ = recorded_hpo_plan(directory, state, records, 0, time.monotonic()+60)
    registry, expected_label = _policies(state, records, directory, learner.model)
    assert canonical(registry) == canonical({key: asdict(value) for key, value in plan.bindings.items()})
    assert learner_label == expected_label
    refresh = [(label, value) for label, value in plan.bindings.items() if value.model_id == 'heuristic_refresh']
    assert len(refresh) == 1 and plan.groups[3].episodes[0].policies[1] == refresh[0][0]


def test_native_mixed_capability_is_additive_and_requires_baseline():
    from types import SimpleNamespace

    import gf_training.native_backend as backend
    check = getattr(backend, 'require_native_refresh_opponents', None)
    assert callable(check), 'new parent capability gate required'
    for info in ({}, {'card_instance_refresh_opponents': 'mixed16-v1;ordinary1;hpo2'},
                 {'card_instance_refresh': backend.CARD_INSTANCE_REFRESH_BASELINE}):
        with pytest.raises(ContractError):
            check(SimpleNamespace(build_info=lambda info=info: info.items()))
    check(SimpleNamespace(build_info=lambda: {'card_instance_refresh': backend.CARD_INSTANCE_REFRESH_BASELINE,
        'card_instance_refresh_opponents': 'mixed16-v1;ordinary1;hpo2',
        'card_instance_guardian_acted': backend.CARD_INSTANCE_GUARDIAN_ACTED}.items()))


@pytest.mark.parametrize('space', [
    {'parameters': [{'path': 'training.collection_games', 'kind': 'categorical', 'choices': [16, 24]}], 'coupled': []},
    {'parameters': [{'path': 'training.collection_games', 'kind': 'int', 'low': 8, 'high': 16, 'step': 8}], 'coupled': []},
    {'parameters': [], 'coupled': [{'name': 'counts', 'paths': ['training.collection_games', 'training.epochs'], 'choices': [[16, 1], [8, 2]]}]},
])
def test_hpo_mixed_search_cannot_resize_recipe(tmp_path, space):
    from gf_cli.hpo_config import campaign_config
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw.update(schema_version=2, training_opponents=DESCRIPTOR, space=space)
    with pytest.raises(ContractError, match='16'):
        campaign_config(raw, origin=tmp_path)


def test_mixed_formal_rejection_and_frozen_learning_kernels():
    from gf_common.card_instance_formal import checked_config
    from gf_training.card_instance_formal import load_recipe, verify_kernel_ast
    from tests.test_formal_ppo_contract import config_for
    recipe = load_recipe()
    verify_kernel_ast(recipe)
    raw = config_for()
    checked_config(raw, recipe)
    raw['training_opponents'] = DESCRIPTOR
    with pytest.raises(ContractError):
        checked_config(raw, recipe)


def test_final64_and_hpo_objective_panels_remain_canonical():
    from gf_training.collection import validation_groups
    groups = validation_groups(17, 'learner', actions_version=3)
    assert len(groups) == 32
    assert {p for group in groups for episode in group.episodes for p in episode.policies} == {
        'learner', 'random-card-instance-v2', 'heuristic-card-instance-v2'}
    # Objective schedule retains the original canonical heuristic.
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from tests.test_hpo_evaluation_role import binding
    link, _, _ = binding()
    actual = hpo_evaluation_groups(link)
    assert {p for group in actual for episode in group.episodes for p in episode.policies} == {
        link['left_parent']['policy_label'], 'heuristic-card-instance-v2'}


def test_ordinary_native_missing_mixed_capability_precedes_runtime_admission(tmp_path, monkeypatch):
    import time
    from dataclasses import replace
    from types import SimpleNamespace

    from gf_training.card_instance_ppo import _GameOperations
    from gf_training.native_backend import CARD_INSTANCE_REFRESH_BASELINE
    loaded = ordinary_config(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    module = SimpleNamespace(build_info=lambda: [('card_instance_refresh', CARD_INSTANCE_REFRESH_BASELINE),
        ('card_instance_guardian_acted', 'card-or-guardian-v1')])
    monkeypatch.setattr('gf_training.native_backend.load_native', lambda: module)
    def forbidden(*args, **kwargs):
        pytest.fail('runtime allocation preceded native mixed capability check')
    monkeypatch.setattr('gf_simulator_client.card_instance_runtime.CardInstanceRuntimeEvidence', forbidden)
    with pytest.raises(ContractError, match='training opponents capability'):
        _GameOperations().preflight(config, deadline=time.monotonic()+60)


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_hpo_native_missing_mixed_capability_precedes_gpu_job_admission(tmp_path, monkeypatch, backend):
    import asyncio
    from types import SimpleNamespace

    from gf_cli.hpo import _train_seed
    from gf_cli.hpo_config import campaign_config
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_training.hpo_config import effective_settings
    from gf_training.native_backend import CARD_INSTANCE_REFRESH_BASELINE
    from tests.test_hpo_config import campaign_raw
    source = ROOT/'configs/experiments/card-instance-mlp-ppo-acceptance.toml'
    path = tmp_path/'base.toml'
    text = source.read_text().replace('"../', '"'+str(ROOT/'configs')+'/')
    text += f'\n[preprocessing]\nbackend="{backend}"\n'
    path.write_text(text)
    raw = campaign_raw()
    raw.update(schema_version=2, base_experiment=str(path), training_opponents=DESCRIPTOR)
    raw['storage'] = {'parent_root': str(tmp_path/'parents'), 'numeric_root': str(tmp_path/'parents')}
    loaded = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    module = SimpleNamespace(build_info=lambda: [('card_instance_refresh', CARD_INSTANCE_REFRESH_BASELINE),
        ('card_instance_guardian_acted', 'card-or-guardian-v1')])
    monkeypatch.setattr('gf_training.native_backend.load_native', lambda: module)
    def forbidden(*args, **kwargs):
        pytest.fail('GPU job admission preceded native mixed capability check')
    monkeypatch.setattr('gf_training.hpo_gpu_lease.admit_gpu_job', forbidden)
    store = SimpleNamespace(put=lambda *args: None)
    proposal = SimpleNamespace(settings=settings, number=0, sha256='a'*64)
    with pytest.raises(ContractError, match='training opponents capability'):
        asyncio.run(_train_seed(store, loaded, {'campaign_id': 'fixture'}, proposal, 17))


def test_ordinary_python_mixed_validation_does_not_load_native(tmp_path, monkeypatch):
    from gf_training.card_instance_ppo import _GameOperations
    loaded = ordinary_config(tmp_path)
    def forbidden():
        pytest.fail('ordinary Python mixed path loaded optional native')
    monkeypatch.setattr('gf_training.native_backend.load_native', forbidden)
    assert _GameOperations().validate_config(loaded.config).model_id == 'mlp'
