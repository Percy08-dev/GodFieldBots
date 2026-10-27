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


def read_frame(stream: IO[bytes]) -> Json:
    size = int.from_bytes(_read_exact(stream, 8), "big")
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ContractError("invalid worker frame length")
    value = json.loads(_read_exact(stream, size))
    if not isinstance(value, dict):
        raise ContractError("worker frame object required")
    return value


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
