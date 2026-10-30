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
