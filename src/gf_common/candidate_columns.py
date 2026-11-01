"""Pure candidate-columns-v1 byte layout. No filesystem/model dependencies.

Raw codes are zero-based ACTION_KINDS/TARGETS indices; token PAD/MISSING codes
never enter this codec. All byte counts/offsets are checked before allocation.
"""
from __future__ import annotations

import hashlib
import re
import struct
from itertools import pairwise
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.structured import ACTION_KINDS, TARGETS, CandidateTable

CODEC = "candidate-columns-v1"
MAX_COLUMN_BYTES = 64 * 1024**2
I64 = struct.Struct("<q")


def _sha(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("candidate column digest required")


def safe_basename(value: Any) -> str:
    if (type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", value) is None
            or value in (".", "..")):
        raise ContractError("unsafe candidate column basename")
    return value


def column_layout(count: int, elements: int, *, max_bytes: int = MAX_COLUMN_BYTES) -> tuple[list[Json], int]:
    integer(count, 1, 65536)
    integer(elements, 0, count * 128)
    integer(max_bytes, 1, MAX_COLUMN_BYTES)
    offset = 0
    columns = []
    for name, dtype, shape in (("kinds", "uint8", [count]), ("targets", "uint8", [count]),
                                ("numerics", "int64", [count, 4]), ("presence", "uint8", [count, 4]),
                                ("offsets", "int64", [count + 1]), ("slots", "uint8", [elements])):
        width = 8 if dtype == "int64" else 1
        if width == 8:
            offset = (offset + 7) // 8 * 8
        size = width
        for axis in shape:
            size *= axis
        columns.append({"name": name, "dtype": dtype, "shape": shape, "offset": offset, "bytes": size})
        offset += size
        if offset > max_bytes:
            raise ContractError("candidate column decoded byte limit exceeded")
    return columns, offset


def validate_descriptor(descriptor: Json, *, expected_file: str | None = None,
                        expected_frame_hash: str | None = None, max_bytes: int = MAX_COLUMN_BYTES) -> tuple[list[Json], int]:
    closed(descriptor, {"schema_version", "codec", "file", "count", "elements", "bytes", "sha256",
                        "frame_hash", "table_hash", "columns"})
    if type(descriptor["schema_version"]) is not int or descriptor["schema_version"] != 1 or descriptor["codec"] != CODEC:
        raise ContractError("unsupported candidate column codec/schema")
    name = safe_basename(descriptor["file"])
    if expected_file is not None and name != expected_file:
        raise ContractError("foreign actor/decision candidate column file")
    for key in ("sha256", "frame_hash", "table_hash"):
        _sha(descriptor[key])
    if expected_frame_hash is not None and descriptor["frame_hash"] != expected_frame_hash:
        raise ContractError("candidate column frame binding mismatch")
    columns, size = column_layout(descriptor["count"], descriptor["elements"], max_bytes=max_bytes)
    integer(descriptor["bytes"], 1, max_bytes)
    if descriptor["bytes"] != size or canonical(descriptor["columns"]) != canonical(columns):
        raise ContractError("candidate column layout/size mismatch")
    return columns, size


def encode_candidate_columns(table: CandidateTable, file: str, *, max_positions: int = 128,
                             max_bytes: int = MAX_COLUMN_BYTES) -> tuple[bytes, Json]:
    safe_basename(file)
    integer(max_positions, 1, 128)
    if any(stop - start > max_positions for start, stop in zip(table.offsets[:-1], table.offsets[1:], strict=True)):
        raise ContractError("candidate column row exceeds active position capacity")
    columns, size = column_layout(len(table.kinds), len(table.slots), max_bytes=max_bytes)
    payload = bytearray(size)  # zero initialization also establishes canonical alignment padding.
    by_name = {column["name"]: column for column in columns}
    start = by_name["kinds"]["offset"]
    payload[start:start + len(table.kinds)] = bytes(ACTION_KINDS.index(kind) for kind in table.kinds)
    start = by_name["targets"]["offset"]
    payload[start:start + len(table.targets)] = bytes(TARGETS.index(target) for target in table.targets)
    start = by_name["numerics"]["offset"]
    for row in table.numerics:
        for value in row:
            I64.pack_into(payload, start, value)
            start += 8
    start = by_name["presence"]["offset"]
    payload[start:start + 4 * len(table.presence)] = bytes(int(value) for row in table.presence for value in row)
    start = by_name["offsets"]["offset"]
    for value in table.offsets:
        I64.pack_into(payload, start, value)
        start += 8
    start = by_name["slots"]["offset"]
    payload[start:start + len(table.slots)] = bytes(table.slots)
    owned = bytes(payload)
    return owned, {"schema_version": 1, "codec": CODEC, "file": file, "count": len(table.kinds),
                   "elements": len(table.slots), "bytes": size, "sha256": hashlib.sha256(owned).hexdigest(),
                   "frame_hash": table.frame_hash, "table_hash": table.table_hash, "columns": columns}


def decode_candidate_columns(payload: bytes, descriptor: Json, *, expected_file: str | None = None,
                             expected_frame_hash: str | None = None, max_positions: int = 128,
                             max_bytes: int = MAX_COLUMN_BYTES) -> CandidateTable:
    columns, size = validate_descriptor(descriptor, expected_file=expected_file,
                                       expected_frame_hash=expected_frame_hash, max_bytes=max_bytes)
    integer(max_positions, 1, 128)
    if type(payload) is not bytes or len(payload) != size or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ContractError("candidate column payload length/hash mismatch")
    previous = 0
    parts: dict[str, memoryview] = {}
    view = memoryview(payload)
    for column in columns:
        offset, length = column["offset"], column["bytes"]
        if any(view[previous:offset]):
            raise ContractError("nonzero candidate alignment padding")
        parts[column["name"]] = view[offset:offset + length]
        previous = offset + length
    if any(code >= len(ACTION_KINDS) for code in parts["kinds"]) or any(code >= len(TARGETS) for code in parts["targets"]):
        raise ContractError("unknown raw candidate kind/target code")
    if any(value not in (0, 1) for value in parts["presence"]) or any(slot > 17 for slot in parts["slots"]):
        raise ContractError("invalid candidate presence/own slot code")
    count = descriptor["count"]
    numerics = tuple(tuple(I64.unpack_from(parts["numerics"], (row * 4 + column) * 8)[0] for column in range(4)) for row in range(count))
    presence = tuple(tuple(bool(parts["presence"][row * 4 + column]) for column in range(4)) for row in range(count))
    offsets = tuple(I64.unpack_from(parts["offsets"], index * 8)[0] for index in range(count + 1))
    if (offsets[0] != 0 or offsets[-1] != descriptor["elements"]
            or any(not 0 <= stop - start <= max_positions for start, stop in pairwise(offsets))):
        raise ContractError("candidate column offsets/active capacity mismatch")
    return CandidateTable(descriptor["frame_hash"], tuple(ACTION_KINDS[code] for code in parts["kinds"]),
                          tuple(TARGETS[code] for code in parts["targets"]), numerics, presence,
                          offsets, tuple(parts["slots"]), descriptor["table_hash"])
