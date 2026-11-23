"""Resolve HPO update bytes under the enclosing sealed run configuration."""
from __future__ import annotations

import os
import time
from pathlib import Path

from gf_common.domain import ContractError, Json
from gf_common.hpo_storage import hpo_diagnostics
from gf_models.structured_sealed import MAX_SEALED_SCAN_BYTES, SealedScan, checksum


def compressed_updates(state: Json) -> bool:
    spec = state.get('effective_config', {}).get('hpo_spec')
    diagnostics = hpo_diagnostics(spec['storage']) if spec is not None else None
    return diagnostics is not None and diagnostics.update_compression == 'zstd'


def update_name(generation: int, *, compressed: bool) -> str:
    return f'{generation:06}.json' + ('.zst' if compressed else '')


def read_hpo_update(parent: Path, generation: int, state: Json, expected_sha256: str,
                    *, scan: SealedScan | None = None) -> Json:
    """Inspect the stored update and verify the digest from its generation record."""
    if scan is None:
        scan = SealedScan(max_owned_bytes=MAX_SEALED_SCAN_BYTES, deadline=time.monotonic() + 60)
    compressed = compressed_updates(state)
    path = parent / 'updates' / update_name(generation, compressed=compressed)
    expected = checksum(expected_sha256)
    if compressed:
        return scan.hpo_zstd_object(path, expected)
    return scan.object(path, expected)


def check_update_inventory(parent: Path, count: int, state: Json, scan: SealedScan) -> None:
    compressed = compressed_updates(state)
    scan.inventory(parent / 'updates', {update_name(number, compressed=compressed) for number in range(1, count + 1)})


def check_cold_update_inventory(parent: Path, count: int, *, compressed: bool, deadline: float) -> None:
    """Validate a Rust-player cold namespace under its original parent lock."""
    directory = parent / 'updates'
    if directory.is_symlink():
        raise ContractError('symlink HPO update directory')
    seen = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if time.monotonic() >= deadline:
                raise TimeoutError('HPO update inventory deadline')
            prefix = entry.name.partition('.')[0]
            if (not prefix.isascii() or not prefix.isdecimal() or not 1 <= (number := int(prefix)) <= count
                    or entry.name != update_name(number, compressed=compressed) or not entry.is_file(follow_symlinks=False)):
                raise ContractError('HPO update inventory has foreign entry')
            seen += 1
    if seen != count:
        raise ContractError('HPO update inventory has missing entries')
