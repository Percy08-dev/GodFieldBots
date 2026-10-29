"""Explicit serde parser policy and loaded-extension compatibility identity."""
import importlib
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_training.native_backend import load_native


def test_preflight_rejects_old_native_parser_identity(monkeypatch):
    old = SimpleNamespace(build_info=lambda: [('abi', 'cp314-gil'),
                                               ('recipe_identity', 'native-preprocessing-v1')])
    original_import = importlib.import_module
    monkeypatch.setattr('gf_training.native_backend.importlib.import_module',
                        lambda name: old if name == 'gf_native' else original_import(name))
    with pytest.raises(ContractError, match='parser/resource contract'):
        load_native()


@pytest.mark.parametrize('encoded', [b'NaN', b'Infinity', b'-Infinity', b'"\\ud800"',
                                   b'{"x":0,"$serde_json::private::Number":"1"}',
                                   b'{"\\u0024serde_json::private::Number":"1"}'])
def test_standard_json_policy_rejects_extensions_and_reserved_keys(encoded):
    native = pytest.importorskip('gf_native')
    with pytest.raises(ValueError):
        native.canonical_json(encoded, 1024 * 1024, time.monotonic() + 10)
