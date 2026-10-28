"""Fixed-data cost semantics stay separate from legacy inclusive training."""
import asyncio
import json
from dataclasses import asdict, replace

import pytest

from gf_cli.bindings import qualify_training_bundle
from gf_cli.config import LoadedConfig, load_config
from gf_common.domain import ActionConfig, ContractError, canonical, digest
from gf_models.config import TrainingConfig
from gf_models.lightgbm.model import inspect_artifact
from gf_models.lightgbm.policy import manifest_digest
from gf_training.lightgbm_run import run_numeric, train_fixed_data
from gf_training.q_data import Sample
from tests.test_lightgbm_native import toy_data as toy_data  # noqa: PLC0414 - pytest fixture export


def test_schema2_native_component_cost_excludes_shared_generation_and_qualification_is_closed(tmp_path, toy_data):
    model, training, x, y = toy_data
    training = replace(training, cost_mode='fixed_data')
    source = {'fixed_dataset_content_sha256': 'a' * 64, 'shared_generation_cost_key': 'b' * 64, 'shared_generation_seconds': 10000.}
    directory = tmp_path / 'fixed-toy'
    result = asyncio.run(run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training,
                                     seed=17, charged_seconds=.1, provenance_data=source))
    assert result['schema_version'] == 2 and result['cost_mode'] == 'fixed_data'
    assert result['charged_prior_seconds'] == .1
    assert result['charged_total_seconds'] == pytest.approx(.1 + result['elapsed_seconds'])
    assert result['charged_total_seconds'] < training.wall_seconds < source['shared_generation_seconds']
    checkpoint = directory / 'checkpoint'
    with pytest.raises(ContractError, match='toy'):
        inspect_artifact(checkpoint)
    # Control-only qualification fixture; inference artifact remains explicitly Toy.
    record = json.loads((directory / 'training.json').read_text())
    record['purpose'] = 'trained'
    (directory / 'training.json').write_text(canonical(record))
    qualify_training_bundle(checkpoint, manifest_digest(str(checkpoint)))
    record['cost_mode'] = 'inclusive_generation'
    (directory / 'training.json').write_text(canonical(record))
    with pytest.raises(ContractError, match='fixed-data'):
        qualify_training_bundle(checkpoint, manifest_digest(str(checkpoint)))


def test_each_fixed_fit_charges_current_validation_transform_but_generation_once(tmp_path, make_input, monkeypatch):
    import gf_training.fixed_data as data
    import gf_training.lightgbm_run as owner
    loaded = load_config('configs/experiments/lightgbm.toml')
    training = TrainingConfig(cost_mode='fixed_data', dataset_manifest=str(tmp_path / 'dataset.json'), seeds=(17, 29, 43))
    config = replace(loaded.config, training=training, run=replace(loaded.config.run, output_dir=str(tmp_path)))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    sample = Sample(make_input(), {'type': 'pray'}, 1., ('run', 'game', 1, 0))
    manifest = {'content_sha256': 'a' * 64, 'shared_generation_cost_key': 'b' * 64,
                'shared_generation_seconds': 10000., 'action_config_hash': digest(asdict(config.actions)),
                'train_decisions': 1, 'validation_decisions': 1}
    (tmp_path / 'dataset.json').write_text(canonical(manifest))
    validations = []
    def prepare(path, encoder, *, preprocessing, deadline, wall_deadline):
        from gf_training.journal import AcceptedDecision
        from gf_training.preparation import PreparationSession
        from gf_training.preparation_numeric import PreparedInputs
        validations.append(True)
        assert wall_deadline - deadline == max(training.cleanup_seconds, preprocessing.cleanup_seconds)
        session = PreparationSession(preprocessing, deadline=deadline, wall_deadline=wall_deadline)
        session.read_json(path)
        owner = PreparedInputs(session, manifest, encoder)
        for split in ('train', 'validation'):
            owner.accept(AcceptedDecision(sample.value, sample.action, sample.target, sample.identity, split), None)
        owner.seal(manifest)
        return owner
    monkeypatch.setattr(data, 'prepare_fixed_inputs', prepare)
    calls = []
    async def fit(*args, **kwargs):
        calls.append(kwargs)
        assert 0 < kwargs['charged_seconds'] < kwargs['training'].wall_seconds
        assert kwargs['provenance_data']['shared_generation_seconds'] == 10000.
        return {'status': 'complete'}
    monkeypatch.setattr(owner, '_run_numeric', fit)
    result = asyncio.run(train_fixed_data(loaded))
    assert len(validations) == 3 and [call['seed'] for call in calls] == [17, 29, 43]
    assert result['shared_generation_seconds'] == 10000. and result['shared_generation_cost_key'] == 'b' * 64
    assert {call['provenance_data']['shared_generation_cost_key'] for call in calls} == {'b' * 64}
