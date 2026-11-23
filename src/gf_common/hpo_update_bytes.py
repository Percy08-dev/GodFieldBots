"""Bounded encoding of canonical HPO update objects."""
from __future__ import annotations

from collections.abc import Callable

from gf_common.domain import ContractError, Json, canonical
from gf_common.hpo_storage import require_zstd

UPDATE_MAX_BYTES = 64 * 1024**2  # Match the existing sealed JSON envelope.
WINDOW_LOG_MAX = 23
ENCODER_BYTES = 4 * 1024**2  # Level-3, 2 MiB-window stream workspace (~3.5 MiB).
_BLOCK = 128 * 1024  # Zstandard maximum block size.


def encode_update(value: Json, *, compressed: bool) -> bytes:
    raw = (canonical(value) + '\n').encode()
    if len(raw) > UPDATE_MAX_BYTES:
        raise ContractError('HPO update JSON size exceeds bound')
    if compressed:
        zstd = require_zstd()
        raw = zstd.compress(raw, options={zstd.CompressionParameter.compression_level: 3,
            zstd.CompressionParameter.window_log: 21})
        if len(raw) > UPDATE_MAX_BYTES:
            raise ContractError('stored HPO update size exceeds bound')
    return raw


def decode_zstd_update(stored: bytes, *, charge: Callable[[int], None],
                       check: Callable[[], object]) -> bytes | bytearray:
    """One complete frame. The caller has bounded and authenticated stored bytes."""
    check()
    zstd = require_zstd()
    try:
        size = zstd.get_frame_info(stored).decompressed_size
        if not stored.startswith(b'\x28\xb5\x2f\xfd'):
            raise ContractError('HPO update requires a data frame')
        if stored[4] & 0x20:
            assert size is not None
            window = size
        else:
            descriptor = stored[5]
            base = 1 << (10 + (descriptor >> 3))
            window = base + (base >> 3) * (descriptor & 7)
        # The library's single-pass stream shortcut can skip its window check.
        if window > 1 << WINDOW_LOG_MAX:
            raise ContractError('HPO update zstd window exceeds bound')
        block_bytes = min(_BLOCK, max(1024, window))
        # Window, input block and output wrap/literal scratch, plus the DCtx.
        workspace = len(stored) + window + 3 * block_bytes + _BLOCK
        if size is not None and size > UPDATE_MAX_BYTES:
            raise ContractError('expanded HPO update exceeds JSON bound')
        charge(workspace + (size + 1 if size is not None else 0))
        decoder = zstd.ZstdDecompressor(options={zstd.DecompressionParameter.window_log_max: WINDOW_LOG_MAX})
        if size is not None:
            raw: bytes | bytearray = decoder.decompress(stored, size + 1)
            if len(raw) != size:
                raise ContractError('HPO zstd frame size differs')
        else:
            raw = bytearray()
            incoming = stored
            while True:
                check()
                block = min(_BLOCK, UPDATE_MAX_BYTES + 1 - len(raw))
                # The growing bytearray, its replacement and the new block may overlap.
                charge(workspace + 3 * (len(raw) + block))
                raw.extend(decoder.decompress(incoming, block))
                if len(raw) > UPDATE_MAX_BYTES:
                    raise ContractError('expanded HPO update exceeds JSON bound')
                if decoder.eof or decoder.needs_input:
                    break
                incoming = b''
        if not decoder.eof or decoder.unused_data:
            raise ContractError('HPO update requires exactly one complete zstd frame')
    except zstd.ZstdError as error:
        raise ContractError('invalid HPO update zstd frame') from error
    check()
    return raw
