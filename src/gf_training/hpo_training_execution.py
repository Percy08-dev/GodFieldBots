"""Closed, opt-in update experiment settings; never checkpoint/config authority."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from gf_common.domain import ContractError, canonical, closed, integer

VARIANTS = ('baseline', 'metrics', 'packed', 'microbatch', 'shared-prefix')


@dataclass(frozen=True)
class TrainingExecutionSpec:
    schema_version: int = 1
    variant: str = 'baseline'
    diagnostic_directory: str | None = None
    profile_start: int | None = None
    profile_groups: int = 1
    profile_shapes: bool = False

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1 or self.variant not in VARIANTS:
            raise ContractError('unknown training execution schema/variant')
        if self.diagnostic_directory is not None and (type(self.diagnostic_directory) is not str
                or not Path(self.diagnostic_directory).is_absolute()):
            raise ContractError('absolute diagnostic destination required')
        if self.profile_start is not None:
            integer(self.profile_start, 0, 1_000_000)
            if self.diagnostic_directory is None:
                raise ContractError('profiling requires diagnostic destination')
        integer(self.profile_groups, 1, 16)
        if type(self.profile_shapes) is not bool:
            raise ContractError('closed profiling flag required')


def write_execution_spec(path: Path, spec: TrainingExecutionSpec) -> str:
    if type(spec) is not TrainingExecutionSpec or not path.is_absolute():
        raise ContractError('exact settings and absolute fresh path required')
    raw = canonical(asdict(spec)).encode()
    if len(raw) > 4096:
        raise ContractError('execution settings exceed 4096 bytes')
    try:
        with path.open('xb') as stream:
            stream.write(raw)
    except FileExistsError as error:
        raise ContractError('fresh execution settings required') from error
    return hashlib.sha256(raw).hexdigest()


def read_execution_spec(path: Path, expected_sha256: str) -> TrainingExecutionSpec:
    if (not path.is_absolute() or type(expected_sha256) is not str or len(expected_sha256) != 64
            or set(expected_sha256)-set('0123456789abcdef')):
        raise ContractError('absolute execution settings and SHA256 required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 4096:
            raise ContractError('bounded regular execution settings required')
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    if len(raw) > 4096 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ContractError('execution settings SHA256 mismatch')
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ContractError('duplicate execution setting')
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        closed(value, set(TrainingExecutionSpec.__dataclass_fields__))
        result = TrainingExecutionSpec(**value)
        if canonical(asdict(result)).encode() != raw:
            raise ContractError('canonical execution settings required')
        return result
    except (ValueError, TypeError) as error:
        raise ContractError('invalid execution settings') from error
