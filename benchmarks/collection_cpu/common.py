"""Closed execution options and bounded measurement identities."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.inference.common import (
    MAX_JSON,
    MAX_OUTPUT,
    REPO,
    check_files,
    immutable_json,
    read_json,
    sha256,
    tree_bytes,
)

__all__ = ['MAX_JSON', 'MAX_OUTPUT', 'REPO', 'check_files', 'immutable_json',
           'read_json', 'sha256', 'tree_bytes']

ROOT = REPO/'runs/collection-optimization-v2'
MODELS = ('mlp', 'gru', 'transformer')
TRANSFERS = ('T0','reuse','T1','T2','T3')


@dataclass(frozen=True)
class Options:
    parent: bool = False
    transfer: str = 'T0'
    http: bool = False
    diagnostic: bool = False

    def __post_init__(self) -> None:
        if (type(self.parent) is not bool or type(self.http) is not bool
                or type(self.diagnostic) is not bool or self.transfer not in TRANSFERS):
            raise ValueError('closed collection experiment options required')
        if self.diagnostic and (self.parent or self.http or self.transfer != 'T0'):
            raise ValueError('passive diagnostic must use unchanged control')

    # Original inference admission helper uses these immutable fixed invariants.
    chunk = 256
    alignment = 0
    compile = 'none'

    def identity(self) -> dict[str, Any]:
        return asdict(self)


def options(path: Path) -> Options:
    value = read_json(path)
    if not isinstance(value, dict) or set(value) != {'parent','transfer','http','diagnostic'}:
        raise ValueError('all four closed experiment options required')
    return Options(**value)


def code_files() -> dict[str, str]:
    paths = [*(REPO/'src').rglob('*.py'), *(REPO/'benchmarks/inference').glob('*.py'),
             *(REPO/'benchmarks/collection_cpu').glob('*.py')]
    return {str(path.relative_to(REPO)):sha256(path) for path in sorted(paths)}


def inference_helper(name: str) -> Any:
    path = REPO/'benchmarks/inference'/f'{name}.py'
    key = 'collection_original_inference_'+name
    if key in sys.modules:
        return sys.modules[key]
    # Historical scripts use their sibling `common`; keep those exact helpers.
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def runtime_identity() -> dict[str,Any]:
    import importlib.metadata
    import platform
    import sysconfig

    from gf_training.native_backend import binary_identity
    return {'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),
            'gil_disabled':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
            'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','httpx')},
            'native':binary_identity()}


def namespace(path: Path) -> Path:
    path = path.absolute()
    if not path.is_relative_to(ROOT) or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('experiment outputs must remain under runs/collection-optimization-v2')
    if path.exists() or path.is_symlink():
        raise ValueError('fresh experimental output namespace required')
    return path
