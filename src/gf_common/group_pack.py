"""Closed, bounded physical group Pack contract. No learning authority is conveyed."""

from __future__ import annotations

import hashlib
import json
import struct
import unicodedata
from compression import zstd
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "sqlite-group-pack-v1"
PACK_FORMAT = "group-pack-v1"
MAGIC = b"GFBPACK1"
MIB = 1024 * 1024
WRITER_BASE = 128 * MIB
SOURCE_BASE = 64 * MIB
MIN_STORAGE_BYTES = WRITER_BASE + SOURCE_BASE + MIB
MAX_INDEX = 8 * MIB
MAX_ENTRIES = 100_000
MAX_JSON = MIB
KINDS = frozenset(
    ("journal", "replay", "candidates", "result", "http-log", "simulator", "diagnostic")
)
type Json = Any


@dataclass(frozen=True)
class PackMember:
    kind: str
    name: str
    data: bytes


@dataclass(frozen=True)
class StorageSettings:
    db_root: Path = Path("/mnt/d/godfield")
    ram_root: Path = Path("/dev/shm/godfieldbots")
    budget_bytes: int = 2048 * MIB
    queue_bytes: int = 512 * MIB
    max_pack_bytes: int = 256 * MIB

    def __post_init__(self) -> None:
        for value in (self.budget_bytes, self.queue_bytes, self.max_pack_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("storage limits must be positive integers")
        if self.budget_bytes < MIN_STORAGE_BYTES:
            raise ValueError("storage budget requires at least 193 MiB for Writer/inspection/runtime")
        if self.queue_bytes > self.budget_bytes:
            raise ValueError("impossible storage budget")
        if self.max_pack_bytes > 256 * MIB:
            raise ValueError("sqlite-group-pack-v1 supports at most 256 MiB per Pack")
        if not isinstance(self.db_root, Path) or not isinstance(self.ram_root, Path):
            raise TypeError("storage roots must be Paths")


def safe_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or any(c in ("", ".", "..") for c in name.split("/"))
        or any(unicodedata.category(c).startswith("C") for c in name)
    ):
        raise ValueError("unsafe logical name")
    try:
        name.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("invalid logical name") from exc
    return name


def canonical_json(value: Json, *, limit: int = MAX_INDEX) -> bytes:
    """Bound UTF-8 output while encoding; reject non-JSON and non-finite values."""
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    parts: list[bytes] = []
    size = 0
    try:
        for part in encoder.iterencode(value):
            data = part.encode("utf-8")
            size += len(data)
            if size > limit:
                raise ValueError("JSON exceeds byte bound")
            parts.append(data)
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc
    return b"".join(parts)


def _object(pairs: list[tuple[str, Json]]) -> dict[str, Json]:
    result: dict[str, Json] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_json(data: bytes, *, canonical: bool = True, limit: int = MAX_INDEX) -> Json:
    if len(data) > limit:
        raise ValueError("JSON exceeds byte bound")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
        if canonical and canonical_json(value, limit=limit) != data:
            raise ValueError("noncanonical JSON")
        return value
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid JSON") from exc


def digest(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def member_index(members: tuple[PackMember, ...]) -> list[dict[str, Json]]:
    entries = []
    offset = 0
    previous = ""
    if len(members) > MAX_ENTRIES:
        raise ValueError("too many Pack members")
    for member in members:
        safe_name(member.name)
        if member.kind not in KINDS or type(member.data) is not bytes or member.name <= previous:
            raise ValueError("invalid or duplicate Pack member")
        entries.append(
            {
                "kind": member.kind,
                "name": member.name,
                "offset": offset,
                "length": len(member.data),
                "sha256": digest(member.data),
            }
        )
        offset += len(member.data)
        previous = member.name
    return entries


def encode_pack(
    members: tuple[PackMember, ...] | list[PackMember], max_uncompressed_bytes: int = 256 * MIB
) -> bytes:
    _limit(max_uncompressed_bytes)
    if len(members) > MAX_ENTRIES:
        raise ValueError("too many Pack members")
    ordered = tuple(sorted(members, key=lambda m: m.name))
    payload_size = sum(len(m.data) for m in ordered)
    if payload_size + 16 > max_uncompressed_bytes:
        raise ValueError("Pack exceeds byte bound")
    index = canonical_json({"format": PACK_FORMAT, "files": member_index(ordered)})
    if 16 + len(index) + payload_size > max_uncompressed_bytes:
        raise ValueError("Pack exceeds byte bound")
    compressor = zstd.ZstdCompressor(level=3)
    chunks = [compressor.compress(MAGIC + struct.pack("<Q", len(index)) + index)]
    chunks.extend(compressor.compress(member.data) for member in ordered)
    chunks.append(compressor.flush())
    result = b"".join(chunks)
    if len(result) > max_uncompressed_bytes + MIB:
        raise ValueError("compressed Pack exceeds bound")
    return result


def _limit(value: int) -> None:
    if type(value) is not int or value < 16:
        raise ValueError("invalid Pack bound")


def unpack_raw(compressed: bytes, max_uncompressed_bytes: int = 256 * MIB) -> bytes:
    _limit(max_uncompressed_bytes)
    if type(compressed) is not bytes or len(compressed) > max_uncompressed_bytes + MIB:
        raise ValueError("compressed Pack exceeds bound")
    decoder = zstd.ZstdDecompressor(
        options={
            zstd.DecompressionParameter.window_log_max: max(
                21, (max_uncompressed_bytes - 1).bit_length()
            )
        }
    )
    try:
        # max_length prevents an oversized expansion being allocated first.
        raw = decoder.decompress(compressed, max_length=max_uncompressed_bytes + 1)
        if len(raw) > max_uncompressed_bytes or not decoder.eof or decoder.unused_data:
            raise ValueError("invalid, oversized or trailing Pack frame")
        return raw
    except zstd.ZstdError as exc:
        raise ValueError("invalid Pack frame") from exc


def _checked_pack(
    compressed: bytes, max_uncompressed_bytes: int
) -> tuple[bytes, int, list[dict[str, Json]]]:
    raw = unpack_raw(compressed, max_uncompressed_bytes)
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise ValueError("invalid Pack header")
    size = struct.unpack("<Q", raw[8:16])[0]
    if size > MAX_INDEX or size > len(raw) - 16:
        raise ValueError("invalid Pack index length")
    index = parse_json(raw[16 : 16 + size])
    if (
        not isinstance(index, dict)
        or set(index) != {"format", "files"}
        or index["format"] != PACK_FORMAT
        or not isinstance(index["files"], list)
        or len(index["files"]) > MAX_ENTRIES
    ):
        raise ValueError("invalid Pack index")
    cursor = 0
    previous = ""
    payload_start = 16 + size
    view = memoryview(raw)
    files: list[dict[str, Json]] = index["files"]
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "kind",
            "name",
            "offset",
            "length",
            "sha256",
        }:
            raise ValueError("invalid Pack entry")
        name = safe_name(entry["name"])
        offset, length = entry["offset"], entry["length"]
        if (
            name <= previous
            or type(offset) is not int
            or type(length) is not int
            or offset != cursor
            or length < 0
            or length > len(raw) - payload_start - cursor
            or not isinstance(entry["kind"], str)
            or entry["kind"] not in KINDS
        ):
            raise ValueError("invalid Pack range or order")
        start = payload_start + cursor
        if digest(view[start : start + length]) != entry["sha256"]:
            raise ValueError("Pack member hash mismatch")
        cursor += length
        previous = name
    if payload_start + cursor != len(raw):
        raise ValueError("Pack payload coverage mismatch")
    return raw, payload_start, files


def inspect_pack(compressed: bytes, max_uncompressed_bytes: int) -> tuple[int, list[dict[str, Json]]]:
    """Fully check a Pack and return its byte length and verified member index."""
    raw, _, files = _checked_pack(compressed, max_uncompressed_bytes)
    return len(raw), files


def decode_pack(
    compressed: bytes, max_uncompressed_bytes: int = 256 * MIB
) -> tuple[PackMember, ...]:
    raw, payload_start, files = _checked_pack(compressed, max_uncompressed_bytes)
    members = []
    for entry in files:
        start = payload_start + entry["offset"]
        members.append(PackMember(entry["kind"], entry["name"], raw[start : start + entry["length"]]))
    return tuple(members)
