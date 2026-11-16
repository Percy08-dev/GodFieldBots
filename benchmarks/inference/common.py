"""Bounded file identities shared by disposable inference measurement commands."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MAX_JSON = 64 * 1024**2
MAX_OUTPUT = 512 * 1024**2


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise ValueError('bounded original regular JSON required: '+str(path))
    return json.loads(path.read_text())


def immutable_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > MAX_JSON:
        raise ValueError('experimental JSON output cap exceeded')
    with path.open('xb') as stream:
        stream.write(encoded)


def options(path: Path) -> Any:
    from gf_models.inference_experiment import ExperimentOptions
    return ExperimentOptions(**read_json(path))


def code_files() -> dict[str, str]:
    paths = [*(REPO/'src').rglob('*.py'), *Path(__file__).parent.glob('*.py')]
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(paths)}


def check_files(files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = Path(name)
        if not path.is_absolute():
            path = REPO/path
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError('frozen input changed: '+str(path))


def tree_bytes(path: Path) -> int:
    size = 0
    for entry in path.rglob('*'):
        if entry.is_symlink():
            raise ValueError('experimental output/cache symlinks unsupported')
        if entry.is_file():
            size += entry.stat().st_size
    return size
