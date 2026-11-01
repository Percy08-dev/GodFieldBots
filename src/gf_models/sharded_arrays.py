"""Closed, hashed bundles of bounded typed NumPy shards (no pickle)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_models.arrays import MAX_ARRAY_BYTES, load_arrays, save_arrays, sha256

MAX_MANIFEST_BYTES = 64 * 1024**2
DTYPES = {"float32": 4, "int64": 8, "uint8": 1}


def _pairs(pairs: list[tuple[str, Any]]) -> Json:
    result: Json = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate manifest key")
        result[key] = value
    return result


def read_json(path: Path, *, expected_digest: str | None = None,
              max_owned_bytes: int = 12 * 1024**3) -> Json:
    try:
        integer(max_owned_bytes, 1, 12 * 1024**3)
        document_limit = min(MAX_MANIFEST_BYTES, max(0, (max_owned_bytes - 65536) // 32))
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError("manifest must be an owned regular file")
            raw = source.read(document_limit + 1)
            after = os.fstat(source.fileno())
            if any(getattr(before, key) != getattr(after, key) for key in
                   ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                raise ContractError("manifest changed while reading")
        if len(raw) > document_limit:
            raise ContractError("JSON document reservation exceeded before parse")
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ContractError("unbounded manifest")
        if expected_digest is not None:
            _hash(expected_digest)
            if hashlib.sha256(raw).hexdigest() != expected_digest:
                raise ContractError("manifest hash mismatch")
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite manifest")))
        if not isinstance(value, dict):
            raise ContractError("manifest object required")
        return value
    except ContractError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        raise ContractError("invalid bounded manifest") from exc


def _hash(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("invalid file digest")


def _key(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ContractError("invalid stable tensor key")


def json_bytes(value: Any, *, limit: int | None = None) -> int:
    """Exact canonical UTF-8 byte count including newline, without a full copy."""
    limit = MAX_MANIFEST_BYTES if limit is None else limit
    integer(limit, 1)
    size = 1
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        size += len(chunk.encode())
        if size > limit:
            raise ContractError("complete JSON manifest exceeds byte bound")
    return size


def plan_bundle(prefix: str, descriptors: Json, *, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    """Pure closed shard/sidecar plan; fixed-length hash placeholders are exact."""
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    if not isinstance(prefix, str) or re.fullmatch(r"[a-z][a-z0-9-]*", prefix) is None or not descriptors:
        raise ContractError("invalid typed bundle plan")
    integer(len(descriptors), 1, 1_000_000)
    batches: list[Json] = []
    current: Json = {}
    size = 0
    for key in sorted(descriptors):
        _key(key)
        item = closed(descriptors[key], {"dtype", "shape", "bytes"})
        shape, dtype = item["shape"], item["dtype"]
        if (not isinstance(dtype, str) or dtype not in DTYPES or not isinstance(shape, list)
                or len(shape) > 8 or any(type(axis) is not int or not 0 <= axis <= MAX_ARRAY_BYTES for axis in shape)):
            raise ContractError("invalid planned tensor dtype/shape")
        integer(item["bytes"], 0, shard_bytes)
        if math.prod(shape) * DTYPES[dtype] != item["bytes"]:
            raise ContractError("planned tensor byte count mismatch")
        if current and size + item["bytes"] > shard_bytes:
            batches.append(current)
            current, size = {}, 0
        current[key] = item
        size += item["bytes"]
    if current:
        batches.append(current)
    integer(len(batches), 1, 65536)
    placeholder = "0" * 64
    shards, files = [], {}
    for index, batch in enumerate(batches):
        name = f"{prefix}-{index:06}.npz"
        metadata = {"sha256": placeholder, "arrays": batch}
        json_bytes(metadata)
        shards.append({"file": name, "metadata": metadata})
        files[name] = files[name[:-4] + ".json"] = placeholder
    plan = {"schema_version": 1, "prefix": prefix, "shards": shards, "files": files,
            "tensor_count": len(descriptors), "decoded_bytes": sum(item["bytes"] for item in descriptors.values())}
    json_bytes(plan)
    return plan


def preflight_outputs(directory: Path, names: list[str]) -> None:
    """Reject every existing final/temp entry, including dangling symlinks."""
    if directory.is_symlink():
        raise ContractError("refusing symlink output directory")
    for name in names:
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ContractError("unsafe owned output basename")
        path = directory / name
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise ContractError("refusing preexisting owned output entry")


def create_output_directory(directory: Path, manifest_name: str) -> None:
    preflight_outputs(directory, [manifest_name, manifest_name + ".tmp"])
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ContractError("refusing existing artifact output directory") from exc


def write_owned_json(path: Path, value: Any) -> None:
    json_bytes(value)
    preflight_outputs(path.parent, [path.name, path.name + ".tmp"])
    from gf_models.arrays import atomic_json
    atomic_json(path, value)


def inspect_bundle(directory: Path, manifest: Json, *, max_decoded_bytes: int = 12 * 1024**3) -> Json:
    closed(manifest, {"schema_version", "prefix", "shards", "files", "tensor_count", "decoded_bytes"})
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ContractError("unsupported typed bundle schema")
    prefix = manifest["prefix"]
    if not isinstance(prefix, str) or re.fullmatch(r"[a-z][a-z0-9-]*", prefix) is None:
        raise ContractError("invalid shard prefix")
    shards, files = manifest["shards"], manifest["files"]
    if not isinstance(shards, list) or not 1 <= len(shards) <= 65536 or not isinstance(files, dict):
        raise ContractError("invalid shard/file map")
    integer(max_decoded_bytes, 0, 12 * 1024**3)
    integer(manifest["decoded_bytes"], 0, max_decoded_bytes)
    integer(manifest["tensor_count"], 1, 1_000_000)
    expected_files: set[str] = set()
    tensors: Json = {}
    total = 0
    for index, shard in enumerate(shards):
        closed(shard, {"file", "metadata"})
        name = f"{prefix}-{index:06}.npz"
        if shard["file"] != name:
            raise ContractError("unsafe/reordered shard path")
        metadata = closed(shard["metadata"], {"sha256", "arrays"})
        _hash(metadata["sha256"])
        if not isinstance(metadata["arrays"], dict) or not metadata["arrays"]:
            raise ContractError("nonempty typed shard required")
        shard_bytes = 0
        for key, descriptor in metadata["arrays"].items():
            _key(key)
            if key in tensors:
                raise ContractError("duplicate tensor across shards")
            closed(descriptor, {"dtype", "shape", "bytes"})
            shape, dtype = descriptor["shape"], descriptor["dtype"]
            if (not isinstance(dtype, str) or dtype not in DTYPES or not isinstance(shape, list) or len(shape) > 8
                    or any(type(size) is not int or not 0 <= size <= MAX_ARRAY_BYTES for size in shape)):
                raise ContractError("invalid tensor dtype/shape")
            size = math.prod(shape) * DTYPES[dtype]
            integer(descriptor["bytes"], 0, MAX_ARRAY_BYTES)
            if size != descriptor["bytes"]:
                raise ContractError("tensor byte count mismatch")
            shard_bytes += size
            if shard_bytes > MAX_ARRAY_BYTES:
                raise ContractError("decoded shard exceeds64MiB")
            tensors[key] = descriptor
        total += shard_bytes
        if total > max_decoded_bytes:
            raise ContractError("bundle decoded reservation exceeded")
        expected_files.update((name, name[:-4] + ".json"))
        if files.get(name) != metadata["sha256"]:
            raise ContractError("shard/file hash binding mismatch")
    if set(files) != expected_files or total != manifest["decoded_bytes"] or len(tensors) != manifest["tensor_count"]:
        raise ContractError("bundle tensor totals/file ownership mismatch")
    # Validate the entire closed inventory before decoding any tensor.
    for name, expected in files.items():
        _hash(expected)
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ContractError("bundle file hash/ownership mismatch")
    for shard in shards:
        if canonical(read_json((directory / shard["file"]).with_suffix(".json"),
                                    expected_digest=files[shard["file"][:-4] + ".json"])) != canonical(shard["metadata"]):
            raise ContractError("shard sidecar mismatch")
    return tensors


def write_bundle(directory: Path, prefix: str, arrays: dict[str, Any], *,
                 shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    if not isinstance(prefix, str) or re.fullmatch(r"[a-z][a-z0-9-]*", prefix) is None or not arrays:
        raise ContractError("invalid typed bundle request")
    # No np.asarray(list) or dtype conversion before size validation. Callers
    # supply already-owned typed arrays; a huge declared tensor cannot allocate.
    for key, array in arrays.items():
        _key(key)
        if (not isinstance(array, np.ndarray) or str(array.dtype) not in DTYPES
                or array.dtype.hasobject or array.nbytes > shard_bytes or not np.isfinite(array).all()):
            raise ContractError("invalid/oversized typed tensor before shard write")
    if directory.is_symlink():
        raise ContractError("refusing symlink output directory")
    directory.mkdir(parents=True, exist_ok=True)
    plan = plan_bundle(prefix, {key: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes}
                                for key, value in arrays.items()}, shard_bytes=shard_bytes)
    destinations = list(plan["files"]) + [shard["file"][:-4] + ".json.tmp" for shard in plan["shards"]]
    preflight_outputs(directory, destinations)
    shards, files = [], {}
    for shard in plan["shards"]:
        name = shard["file"]
        batch = {key: arrays[key] for key in shard["metadata"]["arrays"]}
        path = directory / name
        metadata = save_arrays(path, batch)
        if canonical(metadata["arrays"]) != canonical(shard["metadata"]["arrays"]):
            raise ContractError("written tensor metadata differs from admitted plan")
        shards.append({"file": name, "metadata": metadata})
        files[name] = metadata["sha256"]
        files[name[:-4] + ".json"] = sha256(path.with_suffix(".json"))
    manifest = {"schema_version": 1, "prefix": prefix, "shards": shards, "files": files,
                "tensor_count": len(arrays), "decoded_bytes": sum(array.nbytes for array in arrays.values())}
    inspect_bundle(directory, manifest)
    return manifest


def read_bundle(directory: Path, manifest: Json, *, max_decoded_bytes: int = 12 * 1024**3) -> dict[str, Any]:
    inspect_bundle(directory, manifest, max_decoded_bytes=max_decoded_bytes)
    result: dict[str, Any] = {}
    for shard in manifest["shards"]:
        result.update(load_arrays(directory / shard["file"], shard["metadata"]))
    for value in result.values():
        value.flags.writeable = False
    return result
