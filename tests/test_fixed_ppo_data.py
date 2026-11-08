"""Whole-parent frozen data eligibility and reconstructible query indexes."""
import asyncio
from dataclasses import asdict, replace

import pytest

from gf_cli.config import LoadedConfig
from gf_common.domain import ContractError, digest
from gf_training.fixed_data import build_dataset, index_dataset, inspect_dataset, validate_policy
from gf_training.journal import atomic_json
from gf_training.lineage import read_json
from tests.ppo_support import configuration


def test_frozen_dataset_all_sources_roles_and_regenerable_index(tmp_path, frozen_sources):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    manifest = build_dataset(parents, validations, path)
    inspected, train, validation = inspect_dataset(path)
    assert inspected == manifest
    assert len(train) == 48 and len(validation) == 192
    assert manifest['validation_games'] == 192
    assert [parent['seed'] for parent in manifest['parents']] == [17, 29, 43]
    assert {sample.identity[2] for sample in (*train, *validation)} == {1, 2}
    assert all(sample.target == (1 if sample.identity[2] == 1 else -1) for sample in (*train, *validation))
    from gf_models.mlp.arrays import sha256
    from gf_training.ppo import read_rollout
    numeric = parents[0] / "numeric" / "batch-000000"
    targets = [item.target for item in read_rollout(numeric, sha256(numeric / "rollout.json"))]
    assert targets.count(1.) == 8 and targets.count(-1.) == 8
    assert manifest['shared_generation_seconds'] == pytest.approx(sum(read_json(parent / 'run.json')['elapsed_seconds'] for parent in parents) + sum(read_json(value / 'manifest.json')['elapsed_seconds'] for value in validations))
    pytest.importorskip('pyarrow')
    first = index_dataset(path, tmp_path / 'first.parquet')
    second = index_dataset(path, tmp_path / 'second.parquet')
    assert first == second and first['authoritative'] is False
    import pyarrow.parquet as pq
    rows = pq.read_table(tmp_path / 'first.parquet').to_pylist()
    assert len(rows) == 240 and {row['learner_viewer'] for row in rows} == {1, 2}
    assert all(row['return'] == (1 if row['learner_viewer'] == 1 else -1) for row in rows)
    train_opponents = [row['opponent'] for row in rows if row['split'] == 'train']
    assert train_opponents.count('random') == 12 and train_opponents.count('heuristic') == 12
    assert sum(label.startswith('mlp-') for label in train_opponents) == 24
    assert {row['opponent'] for row in rows if row['split'] == 'validation'} == {'random', 'heuristic'}
    state = read_json(parents[1] / 'run.json')
    state['status'] = 'failed'
    atomic_json(parents[1] / 'run.json', state)
    for operation in (lambda: inspect_dataset(path), lambda: index_dataset(path, tmp_path / 'invalid.parquet')):
        with pytest.raises(ContractError):
            operation()
    assert not (tmp_path / 'invalid.parquet').exists()


def test_frozen_dataset_rejects_omission_reordering_and_changed_original(tmp_path, frozen_sources):
    parents, validations = frozen_sources
    with pytest.raises(ContractError):
        build_dataset(parents[:2], validations[:2], tmp_path / 'omitted.json')
    with pytest.raises(ContractError, match='ordered'):
        build_dataset(parents[::-1], validations[::-1], tmp_path / 'reordered.json')
    path = tmp_path / 'dataset.json'
    build_dataset(parents, validations, path)
    journal = next((parents[0] / 'collections').glob('*/*/player-1.jsonl'))
    journal.write_bytes(journal.read_bytes() + b'\n')
    with pytest.raises((ContractError, ValueError)):
        inspect_dataset(path)


@pytest.mark.parametrize('parent_status', ['failed', 'running'])
def test_inclusive_lightgbm_rejects_ppo_sources_before_native_child(tmp_path, frozen_sources, monkeypatch, parent_status):
    import gf_training.lightgbm_run as module
    from gf_cli.config import load_config
    from gf_models.config import TrainingConfig
    from gf_training.q_data import load_run
    parents, validations = frozen_sources
    parent = read_json(parents[0] / 'run.json')
    collection = parents[0] / 'collections' / 'batch-000000'
    bundle = load_run(collection)
    parent['status'] = parent_status
    atomic_json(parents[0] / 'run.json', parent)
    monkeypatch.setattr(module, 'feasibility', lambda *args, **kwargs: (bundle, {'action_config_hash': digest(asdict(configuration(tmp_path).config.actions))}))
    monkeypatch.setattr(module, '_run_numeric', lambda *args, **kwargs: pytest.fail('schema-4 source must fail before native child'))
    loaded = load_config('configs/experiments/lightgbm.toml')
    config = replace(loaded.config, training=TrainingConfig(pilot_run=str(collection), train_runs=(str(collection),), validation_runs=(str(validations[0]),), accepted_decision_target=1))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    with pytest.raises(ContractError):
        asyncio.run(module.train_experiments(loaded))


def test_final_validation_rejects_spoofed_binding_and_selection_identity(tmp_path, frozen_sources):
    import json
    parents, validations = frozen_sources
    source = validations[0] / 'manifest.json'
    original = read_json(source)
    changed = json.loads(json.dumps(original))
    label = next(name for name in changed['policy_bindings'] if name.startswith('mlp-'))
    changed['policy_bindings'][label] = {'model_id': 'random', 'policy_options': None}
    atomic_json(source, changed)
    with pytest.raises(ContractError, match='bindings'):
        build_dataset(parents, validations, tmp_path / 'spoofed.json')
    atomic_json(source, original)
    journal = next(validations[0].glob('*/player-1.jsonl'))
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    next(record for record in records if record['kind'] == 'decision')['selection']['policy_version'] = 'mlp-ppo-v1-' + '0' * 64
    journal.write_text(''.join(json.dumps(record) + '\n' for record in records))
    with pytest.raises(ContractError, match='selection'):
        build_dataset(parents, validations, tmp_path / 'wrong-selection.json')


def test_final_validation_rejects_external_server_before_parent_or_collector(tmp_path):
    loaded = configuration(tmp_path)
    config = replace(loaded.config, server=replace(loaded.config.server, external_url='http://localhost:9999'))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    async def forbidden(plan):
        pytest.fail('external fixed server cannot implement 32 distinct validation seeds')
    with pytest.raises(ContractError, match='owned distinct-seed'):
        asyncio.run(validate_policy(loaded, tmp_path / 'missing-parent', tmp_path / 'output', forbidden))
    assert not (tmp_path / 'output').exists()
