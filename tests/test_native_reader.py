"""Closed reader capability preflight, including historical native wheel settings."""
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_training.native_backend import reader_settings


def binary(capability=None):
    return SimpleNamespace(build_info=lambda: [('typed_reader_contract', capability)]
                           if capability is not None else [])


def test_historical_wheel_receives_only_legacy_settings():
    config = PreparationConfig(backend='rust')
    settings = reader_settings(config, binary())
    assert settings == {key: value for key, value in asdict(config).items() if key != 'reader'}
    assert config.reader == 'legacy'


@pytest.mark.parametrize('reader', ['legacy', 'typed-borrowed-v1'])
def test_capable_wheel_receives_explicit_closed_reader(reader):
    config = PreparationConfig(backend='rust', reader=reader)
    assert reader_settings(config, binary('typed-borrowed-v1'))['reader'] == reader


@pytest.mark.parametrize('capability', [None, '', 'typed-borrowed-v0'])
def test_incapable_typed_wheel_fails_before_cancel_handle_or_source(capability, monkeypatch):
    import gf_training.native_authorization as authorization
    import gf_training.native_ppo as ppo
    native = binary(capability)
    native.CancelHandle = lambda *args: pytest.fail('capability failure must precede CancelHandle')
    monkeypatch.setattr(authorization, 'load_native', lambda: native)
    monkeypatch.setattr(authorization, 'binary_identity', lambda *args: pytest.fail('no identity after failed capability'))
    monkeypatch.setattr(ppo, 'load_native', lambda: native)
    config = PreparationConfig(backend='rust', reader='typed-borrowed-v1')
    with pytest.raises(ContractError, match='does not support'):
        authorization.NativePreparation({'kind': 'offline'}, config, deadline=time.monotonic() + 5)
    with pytest.raises(ContractError, match='does not support'):
        ppo.preprocessing_identity(config, deadline=time.monotonic() + 5)


def test_typed_python_and_unknown_reader_reject_before_source_or_collection(monkeypatch):
    import gf_training.native_ppo as ppo
    monkeypatch.setattr(ppo, 'load_native', lambda: pytest.fail('Python typed reader cannot load native'))
    config = PreparationConfig(backend='python', reader='typed-borrowed-v1')
    with pytest.raises(ContractError, match='requires the Rust'):
        ppo.preprocessing_identity(config, deadline=time.monotonic() + 5)
    with pytest.raises(ContractError):
        preparation_config(asdict(config))
    with pytest.raises(ContractError, match='unsupported preprocessing reader'):
        reader_settings(PreparationConfig(backend='rust', reader='future'), binary())


def test_new_python_ppo_identity_records_reader_without_rewriting_history():
    from gf_training.native_ppo import preprocessing_identity
    assert preprocessing_identity(PreparationConfig(), deadline=time.monotonic() + 5) == {
        'backend': 'python', 'reader_contract': 'legacy'}


def test_operation_proof_requires_selected_reader_and_never_silently_falls_back():
    import json

    from gf_training.native_authorization import NativePreparation
    owner=NativePreparation.__new__(NativePreparation)
    owner.config=PreparationConfig(backend='rust',reader='typed-borrowed-v1')
    for proof in ({'structural_valid':True},{'reader_contract':'legacy'}):
        with pytest.raises(ContractError,match='reader differs'):
            owner._decode_proof(json.dumps(proof))
    assert owner._decode_proof('{"reader_contract":"typed-borrowed-v1"}')['reader_contract']=='typed-borrowed-v1'
    owner.config=PreparationConfig(backend='rust')
    assert owner._decode_proof('{"source_set_sha256":"unchanged"}')=={
        'source_set_sha256':'unchanged','reader_contract':'legacy'}


def test_historical_wheel_admits_mlp_and_refuses_gru_before_handle_or_collection(tmp_path, monkeypatch):
    import asyncio

    import gf_training.native_ppo as ppo
    from gf_cli.config import load_config
    from gf_training.native_backend import require_ppo_model
    from gf_training.ppo_run import train_ppo
    old = binary('typed-borrowed-v1')
    require_ppo_model('mlp', old)
    old.CancelHandle = lambda *args: pytest.fail('GRU capability failure must precede handle creation')
    monkeypatch.setattr(ppo, 'load_native', lambda: old)
    with pytest.raises(ContractError, match='does not support PPO model gru'):
        ppo.preprocessing_identity(PreparationConfig(backend='rust', reader='typed-borrowed-v1'),
                                   deadline=time.monotonic() + 5, model_id='gru')
    async def forbidden(plan):
        pytest.fail('an old native wheel must not collect a GRU batch')
    loaded = load_config('configs/experiments/gru-ppo-cpu.toml')
    output = tmp_path / 'not-created'
    with pytest.raises(ContractError, match='does not support PPO model gru'):
        asyncio.run(train_ppo(loaded, forbidden, output))
    assert not output.exists()


@pytest.mark.parametrize('marker,model', [('mlp', 'mlp'), ('gru', 'gru'), ('mlp,gru', 'gru'), ('gru,mlp', 'mlp'), ('mlp,gru,transformer', 'transformer')])
def test_closed_native_model_capabilities(marker, model):
    from gf_training.native_backend import require_ppo_model
    native = SimpleNamespace(build_info=lambda: [('ppo_policy_models', marker)])
    require_ppo_model(model, native)


@pytest.mark.parametrize('marker', ['', 'mlp,mlp', 'mlp,,gru', 'mlp, gru', 'future', 'mlp,future', 'mlp,transformer,transformer', None, [], 1])
def test_malformed_native_model_capabilities_never_fall_back(marker):
    from gf_training.native_backend import require_ppo_model
    native = SimpleNamespace(build_info=lambda: [('ppo_policy_models', marker)])
    with pytest.raises(ContractError, match='malformed'):
        require_ppo_model('mlp', native)


def test_native_capability_unknown_selected_model_and_bare_policy_identity_refuse():
    from gf_models.ppo_identity import model_id, selection_version
    from gf_training.native_backend import require_ppo_model
    with pytest.raises(ContractError):
        require_ppo_model('future', binary())
    with pytest.raises(ContractError):
        model_id({'model_id': 'future'})
    assert selection_version('mlp-p') == 'mlp-ppo-v1-p'
    assert selection_version('gru-p') == 'gru-ppo-v1-p'
    assert selection_version('transformer-p') == 'transformer-ppo-v1-p'
    for label in ('p', '', 'mlp-', 'gru-', 'transformer-', 'future-p'):
        with pytest.raises(ContractError):
            selection_version(label)


@pytest.mark.parametrize('identity', ['mlp', 'gru', 'transformer'])
def test_compact_native_parent_identity_drives_closed_authorization(identity):
    import numpy as np

    from gf_training.native_authorization import _expected_collection_items
    parent = {'directory': '/tmp/parent', 'state': {'model_id': identity, 'seed': 17},
              'generations': [{'inference_sha256': 'a' * 64, 'parent_controls': {
                  'schedule_opponent_rng': np.random.default_rng(17).bit_generator.state, 'snapshot_pool': [0]}}]}
    summary = {'parents': [parent], 'collections': [{'parent_index': 0, 'collection_generation': 0,
                                                    'collection_kind': 'ppo_batch', 'directory': '/tmp/batch'}]}
    authorized = list(_expected_collection_items(summary))
    assert authorized[0]['bindings'][identity + '-' + 'a' * 64]['model_id'] == identity
    del parent['state']['model_id']
    with pytest.raises(ContractError, match='model identity'):
        list(_expected_collection_items(summary))
    legacy = list(_expected_collection_items(summary, legacy_mlp=True))
    assert legacy[0]['bindings']['mlp-' + 'a' * 64]['model_id'] == 'mlp'
    for malformed in ('future', '', None, True):
        parent['state']['model_id'] = malformed
        with pytest.raises(ContractError, match='model identity'):
            list(_expected_collection_items(summary))
    parent['state']['model_id'] = 'gru'
    with pytest.raises(ContractError, match='historical'):
        list(_expected_collection_items(summary, legacy_mlp=True))


def test_gru_capable_wheel_cannot_collect_transformer_before_its_own_capability(tmp_path, monkeypatch):
    import asyncio

    import gf_training.native_ppo as ppo
    from gf_cli.config import load_config
    from gf_training.ppo_run import train_ppo
    native = SimpleNamespace(build_info=lambda: [('typed_reader_contract', 'typed-borrowed-v1'),
                                                 ('ppo_policy_models', 'mlp,gru')])
    native.CancelHandle = lambda *args: pytest.fail('capability failure must precede handle creation')
    monkeypatch.setattr(ppo, 'load_native', lambda: native)
    async def forbidden(plan):
        pytest.fail('a GRU-only capable wheel must not collect Transformer games')
    loaded = load_config('configs/experiments/transformer-ppo-cpu.toml')
    output = tmp_path / 'not-created'
    with pytest.raises(ContractError, match='does not support PPO model transformer'):
        asyncio.run(train_ppo(loaded, forbidden, output))
    assert not output.exists()
