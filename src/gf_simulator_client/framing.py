"""Version-1 private worker transport: bounded length-framed UTF-8 JSON."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from gf_common.domain import ContractError, Json, canonical


class _Reader(Protocol):
    def read(self,size: int) -> bytes: ...


class _Writer(Protocol):
    def write(self,value: bytes) -> int | None: ...
    def flush(self) -> None: ...


MAX_FRAME_BYTES = 64 * 1024 * 1024


def _read_exact(stream: _Reader, length: int) -> bytes:
    pieces = bytearray()
    while len(pieces) < length:
        piece = stream.read(length - len(pieces))
        if not piece:
            raise EOFError("truncated worker frame")
        pieces.extend(piece)
    return bytes(pieces)


def _frame_length(stream: _Reader, maximum: int = MAX_FRAME_BYTES) -> int:
    size = int.from_bytes(_read_exact(stream, 8), "big")
    if not 0 < size <= min(maximum, MAX_FRAME_BYTES):
        raise ContractError("invalid worker frame length")
    return size


def _frame_object(stream: _Reader, size: int) -> Json:
    value = json.loads(_read_exact(stream, size))
    if not isinstance(value, dict):
        raise ContractError("worker frame object required")
    return value


def read_frame(stream: _Reader) -> Json:
    return _frame_object(stream, _frame_length(stream))


def frame_decode_bytes(size: int) -> int:
    from gf_common.domain import integer
    integer(size, 1, MAX_FRAME_BYTES)
    return 34 * size + 65536


def read_frame_admitted(stream: _Reader, *, max_owned_bytes: int, retained_bytes: int = 0,
                        max_frame_bytes: int = MAX_FRAME_BYTES, on_frame: Callable[[], None] | None = None) -> Json:
    from gf_common.domain import integer
    integer(max_owned_bytes, 1, 2**63-1)
    integer(retained_bytes, 0, max_owned_bytes)
    integer(max_frame_bytes, 1, MAX_FRAME_BYTES)
    size = _frame_length(stream, max_frame_bytes)
    if on_frame is not None:on_frame()
    if retained_bytes + frame_decode_bytes(size) > max_owned_bytes:
        raise ContractError('worker frame decoded/copy admission exceeded before body read')
    return _frame_object(stream, size)


def write_frame(stream: _Writer, value: Json) -> None:
    encoded = canonical(value).encode()
    _write_encoded(stream,encoded)


def _write_encoded(stream: _Writer,encoded: bytes) -> None:
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


def write_frame_profiled(stream: _Writer,value: Json) -> Json:
    """One original encoding/write; timings remain coordinator diagnostics."""
    import time
    started=time.perf_counter()
    encoded=canonical(value).encode()
    packed=time.perf_counter()
    _write_encoded(stream,encoded)
    return {'worker_driver_request_encode_ms':(packed-started)*1000,
        'worker_driver_request_write_ms':(time.perf_counter()-packed)*1000}


def read_frame_profiled(stream: _Reader) -> tuple[Json,Json]:
    """Body time starts after the header arrives; header wait includes computation."""
    import time
    started=time.perf_counter()
    size=_frame_length(stream)
    header=time.perf_counter()
    raw=_read_exact(stream,size)
    transferred=time.perf_counter()
    value=json.loads(raw)
    if not isinstance(value,dict):raise ContractError('worker frame object required')
    return value,{'worker_driver_reply_header_wait_ms':(header-started)*1000,
        'worker_driver_reply_body_ms':(transferred-header)*1000,
        'worker_driver_reply_json_decode_ms':(time.perf_counter()-transferred)*1000,
        'worker_driver_reply_bytes':size}
