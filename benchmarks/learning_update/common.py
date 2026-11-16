"""Bounded identities for the update comparison, without training authority."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[2]
PRIOR=REPO/'runs/hpo-producer-owned/assessment/device-io-detail-v1'
INVENTORY=REPO/'runs/learning-update-preparation-20261117/input-state-inventory.json'
PRODUCER='8b08e56e821c8ff72fabf892383c144021b0e8885dfdbe38218e21f38e25f59a'
MAX_OUTPUT=512*1024**2


def sha256(path: Path) -> str:
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink() or path.stat().st_size>64*1024**2:
        raise ValueError('bounded regular JSON required: '+str(path))
    return json.loads(path.read_bytes())


def immutable_json(path: Path,value: Any) -> None:
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>64*1024**2:
        raise ValueError('diagnostic JSON exceeds bound')
    with path.open('xb') as stream:
        stream.write(raw)


def check_files(files: dict[str,str]) -> None:
    for name,expected in files.items():
        path=Path(name)
        if not path.is_absolute():
            path=REPO/path
        if not path.is_file() or path.is_symlink() or sha256(path)!=expected:
            raise ValueError('frozen file changed: '+str(path))


def code_files() -> dict[str,str]:
    paths=[*(REPO/'src').rglob('*.py'),*Path(__file__).parent.glob('*.py'),
        REPO/'pyproject.toml',REPO/'uv.lock']
    return {str(path.relative_to(REPO)):sha256(path) for path in sorted(paths)}


def revision(path: Path=REPO) -> str:
    return subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,
        capture_output=True,text=True).stdout.strip()


def prior_runner() -> Any:
    for name in ('runner.py','measure.py','analyze.py'):
        path=PRIOR/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('required historical observer missing: '+str(path))
    sys.path.insert(0,str(PRIOR))
    spec=importlib.util.spec_from_file_location('update_prior_runner',PRIOR/'runner.py')
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_bytes(path: Path) -> int:
    total=0
    for entry in path.rglob('*'):
        if entry.is_symlink():
            raise ValueError('diagnostic output symlink unsupported')
        if entry.is_file():
            total+=entry.stat().st_size
    return total
