"""Version-1 private worker transport: bounded length-framed UTF-8 JSON."""
from __future__ import annotations

import json
from typing import IO

from gf_common.domain import ContractError, Json, canonical

MAX_FRAME_BYTES = 64 * 1024 * 1024


def _read_exact(stream: IO[bytes], length: int) -> bytes:
    pieces = bytearray()
    while len(pieces) < length:
        piece = stream.read(length - len(pieces))
        if not piece:
            raise EOFError("truncated worker frame")
        pieces.extend(piece)
    return bytes(pieces)


def _frame_length(stream: IO[bytes], maximum: int = MAX_FRAME_BYTES) -> int:
    size = int.from_bytes(_read_exact(stream, 8), "big")
    if not 0 < size <= min(maximum, MAX_FRAME_BYTES):
        raise ContractError("invalid worker frame length")
    return size


def _frame_object(stream: IO[bytes], size: int) -> Json:
    value = json.loads(_read_exact(stream, size))
    if not isinstance(value, dict):
        raise ContractError("worker frame object required")
    return value


def read_frame(stream: IO[bytes]) -> Json:
    return _frame_object(stream, _frame_length(stream))


def frame_decode_bytes(size: int) -> int:
    from gf_common.domain import integer
    integer(size, 1, MAX_FRAME_BYTES)
    return 34 * size + 65536


def read_frame_admitted(stream: IO[bytes], *, max_owned_bytes: int, retained_bytes: int = 0,
                        max_frame_bytes: int = MAX_FRAME_BYTES) -> Json:
    from gf_common.domain import integer
    integer(max_owned_bytes, 1, 2**63-1)
    integer(retained_bytes, 0, max_owned_bytes)
    integer(max_frame_bytes, 1, MAX_FRAME_BYTES)
    size = _frame_length(stream, max_frame_bytes)
    if retained_bytes + frame_decode_bytes(size) > max_owned_bytes:
        raise ContractError('worker frame decoded/copy admission exceeded before body read')
    return _frame_object(stream, size)


def write_frame(stream: IO[bytes], value: Json) -> None:
    encoded = canonical(value).encode()
    if not 0 < len(encoded) <= MAX_FRAME_BYTES:
        raise ContractError("worker frame exceeds protocol bound")
    framed = len(encoded).to_bytes(8, "big") + encoded
    position = 0
    while position < len(framed):
        written = stream.write(framed[position:])
        if written is None or written <= 0:
            raise OSError("worker frame write made no progress")
        position += written
    stream.flush()
