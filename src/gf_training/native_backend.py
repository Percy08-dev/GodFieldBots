"""Explicit optional native loading and identity; imports no learning runtime."""
from __future__ import annotations

import hashlib
import importlib
import sys
import sysconfig
from dataclasses import asdict
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any

from gf_common.domain import ContractError, Json
from gf_common.preprocessing import PreparationConfig

TYPED_READER = 'typed-borrowed-v1'


def reader_settings(config: PreparationConfig, native: ModuleType | None = None) -> Json:
    """Closed reader preflight before handles/source owners; old wheels keep old DTOs."""
    if config.reader not in ('legacy', TYPED_READER):
        raise ContractError('unsupported preprocessing reader')
    if config.backend != 'rust':
        if config.reader != 'legacy':
            raise ContractError('typed preprocessing reader requires the Rust backend')
        raise ContractError('native preprocessing settings require the Rust backend')
    module = native or load_native()
    information = dict(getattr(module, 'build_info', lambda: ())())
    capable = information.get('typed_reader_contract') == TYPED_READER
    if config.reader == TYPED_READER and not capable:
        raise ContractError('native binary does not support typed-borrowed-v1')
    settings = asdict(config)
    if not capable and config.reader == 'legacy':
        settings.pop('reader')
    return settings


def load_native() -> ModuleType:
    """Fail before source/collection work when the selected backend is absent."""
    if (sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 14)
            or sysconfig.get_config_var('Py_GIL_DISABLED')):
        raise ContractError('native preprocessing requires CPython 3.14 with the GIL')
    try:
        native = importlib.import_module('gf_native')
    except ImportError as exc:
        raise ContractError('native preprocessing requires the optional native package') from exc
    import numpy as np

    if np.__version__ != '2.5.4':
        raise ContractError('native preprocessing requires NumPy 2.5.4')
    information = dict(native.build_info())
    if information.get('abi') != 'cp314-gil' or information.get('recipe_identity') != 'native-preprocessing-v1':
        raise ContractError('incompatible native preprocessing ABI/recipe')
    if (information.get('parser_contract') != 'serde-json-v2'
            or information.get('json_memory_policy') != 'json-document-32x-plus64kib-v1'):
        raise ContractError('incompatible native JSON parser/resource contract')
    return native


def binary_identity(native: ModuleType | None = None) -> Json:
    """Hash the extension that was actually loaded, independently of source Git."""
    module = native or load_native()
    module_name: Any = getattr(module.build_info, '__module__', None)
    binary_module = importlib.import_module(module_name) if isinstance(module_name, str) else module
    source: Any = binary_module.__file__
    if not isinstance(source, str):
        raise ContractError('native extension has no loaded binary path')
    path = Path(source).resolve(strict=True)
    if not any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise ContractError('native identity must identify the loaded extension binary')
    before = path.stat()
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ContractError('native extension changed while computing its identity')
    return {'backend': 'native', 'extension_path': str(path), 'extension_sha256': checksum,
            'build': dict(module.build_info()), 'python_abi': sys.implementation.cache_tag,
            'gil_disabled': bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}
