"""Bounded, typed, pickle-free NumPy tensor persistence."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import time
import zipfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from gf_common.domain import ContractError, Json, canonical, closed


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.replace(path)

MAX_ARRAY_BYTES = 64 * 1024 * 1024


def _source_guard(path: Path, source: BinaryIO, identity: tuple[int, ...], check: Callable[[], None] | None) -> Callable[[], None]:
    def guard() -> None:
        if check is not None:
            check()
        held, current = os.fstat(source.fileno()), path.lstat()
        if (not stat.S_ISREG(held.st_mode) or not stat.S_ISREG(current.st_mode)
                or any(tuple(getattr(value, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) != identity
                       for value in (held, current))):
            raise ContractError('array source differs from original admitted snapshot')
    return guard


def sha256(path: Path, deadline: float = float("inf"), *, _check: Callable[[], None] | None = None,
           _source: BinaryIO | None = None) -> str:
    if _check is not None:
        _check()
    value = hashlib.sha256()
    position = None if _source is None else _source.tell()
    with path.open('rb') if _source is None else nullcontext(_source) as source:
        if _source is not None:
            source.seek(0)
        try:
            while chunk := source.read(1024 * 1024):
                if _check is not None:
                    _check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("artifact hashing deadline")
                value.update(chunk)
        finally:
            if position is not None:
                source.seek(position)
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact hashing deadline")
    if _check is not None:
        _check()
    return value.hexdigest()


def save_arrays(path: Path, arrays: dict[str, Any], *, _check: Callable[[], None] | None = None) -> Json:
    if _check is not None:
        _check()
    result: Json = {}
    for key, raw in arrays.items():
        if _check is not None:
            _check()
        if not key or not all(c.isalnum() or c in '._-' for c in key):
            raise ContractError("invalid array key")
        value = np.asarray(raw)
        if value.dtype not in (np.dtype('float32'), np.dtype('int64'), np.dtype('uint8')) or value.nbytes > MAX_ARRAY_BYTES or not np.isfinite(value).all():
            raise ContractError("invalid checkpoint tensor")
        result[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes}
    if sum(item["bytes"] for item in result.values()) > MAX_ARRAY_BYTES:
        raise ContractError("checkpoint/shard decoded byte limit exceeded")
    if _check is not None:
        _check()
    np.savez(path, **arrays)
    if _check is not None:
        _check()
    manifest = {"sha256": sha256(path, _check=_check) if _check is not None else sha256(path), "arrays": result}
    if _check is not None:
        _check()
    atomic_json(path.with_suffix('.json'), manifest)
    if _check is not None:
        _check()
    return manifest


def _load_arrays(path: Path, expected: Json | None = None, *, _check: Callable[[], None] | None = None,
                 _source: BinaryIO | None = None) -> dict[str, Any]:
    if _check is not None:
        _check()
    manifest = json.loads(path.with_suffix('.json').read_text()) if expected is None else expected
    closed(manifest, {"sha256", "arrays"})
    actual_hash = sha256(path, _check=_check, _source=_source) if _check is not None or _source is not None else sha256(path)
    if actual_hash != manifest['sha256'] or not isinstance(manifest['arrays'], dict):
        raise ContractError("array hash mismatch")
    output = {}
    decoded_bytes = 0
    with zipfile.ZipFile(path if _source is None else _source) as archive:
        if _check is not None:
            _check()
        members = archive.infolist()
        names = {key + '.npy' for key in manifest['arrays']}
        if len(members) != len(names) or {entry.filename for entry in members} != names:
            raise ContractError("checkpoint tensor keys mismatch")
        # Read bounded bytes and validate *all* NPY headers before NumPy can allocate.
        validated = []
        for entry in members:
            if _check is not None:
                _check()
            if entry.file_size > MAX_ARRAY_BYTES + 4096:
                raise ContractError("unbounded checkpoint archive")
            key = entry.filename[:-4]
            metadata = manifest['arrays'][key]
            closed(metadata, {"dtype", "shape", "bytes"})
            shape = metadata['shape']
            if not isinstance(shape, list) or len(shape) > 8 or any(type(size) is not int or size < 0 or size > MAX_ARRAY_BYTES for size in shape):
                raise ContractError("invalid checkpoint dimensions")
            dtype = np.dtype(metadata['dtype'])
            if dtype not in (np.dtype('float32'), np.dtype('int64'), np.dtype('uint8')):
                raise ContractError("unsupported checkpoint dtype")
            nbytes = math.prod(shape) * dtype.itemsize
            if type(metadata['bytes']) is not int or metadata['bytes'] != nbytes or nbytes > MAX_ARRAY_BYTES:
                raise ContractError("invalid checkpoint byte count")
            decoded_bytes += nbytes
            if decoded_bytes > MAX_ARRAY_BYTES:
                raise ContractError("checkpoint/shard decoded byte limit exceeded")
            encoded = archive.read(entry)
            if _check is not None:
                _check()
            source = io.BytesIO(encoded)
            version = np.lib.format.read_magic(source)
            if version == (1, 0):
                actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(source, max_header_size=4096)
            elif version == (2, 0):
                actual_shape, _, actual_dtype = np.lib.format.read_array_header_2_0(source, max_header_size=4096)
            else:
                raise ContractError("unsupported checkpoint NPY version")
            if actual_shape != tuple(shape) or actual_dtype != dtype or actual_dtype.hasobject or source.tell() + nbytes != len(encoded):
                raise ContractError("checkpoint NPY header/payload mismatch")
            validated.append((key, encoded, metadata))
        for key, encoded, metadata in validated:
            if _check is not None:
                _check()
            value = np.load(io.BytesIO(encoded), allow_pickle=False)
            if _check is not None:
                _check()
            if not np.isfinite(value).all():
                raise ContractError("nonfinite checkpoint tensor")
            output[key] = value
    if _check is not None:
        _check()
    return output


def load_arrays(path: Path, expected: Json | None = None, *, _check: Callable[[], None] | None = None,
                _source: BinaryIO | None = None, _snapshot: tuple[int, ...] | None = None) -> dict[str, Any]:
    position = None
    if _source is not None:
        if _snapshot is None:
            raise ContractError('caller-owned array FD requires original admitted snapshot')
        _check = _source_guard(path, _source, _snapshot, _check)
        position = _source.tell()
    try:
        return _load_arrays(path, expected, _check=_check, _source=_source)
    except TimeoutError as error:
        if _check is not None:
            raise
        raise ContractError('invalid typed-array artifact') from error
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise ContractError('invalid typed-array artifact') from error
    finally:
        if position is not None and _source is not None:
            _source.seek(position)
