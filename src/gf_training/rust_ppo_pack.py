"""Physical verification of a sealed Rust PPO Group Pack, without opening SQLite."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer
from gf_common.group_pack import MAX_JSON, parse_json
from gf_training.group_pack_db import DATABASE, validate_descriptor, validate_seal_object
from gf_training.group_pack_source import OriginalFile


def validate_pack_verification(value: Json, *, run_id: str) -> Json:
    from gf_common.rust_player import hex_identity

    row = closed(value, {'descriptor_sha256', 'database_sha256', 'database_bytes',
        'seal_id', 'logical_sha256'})
    hex_identity(row['descriptor_sha256'])
    hex_identity(row['database_sha256'])
    integer(row['database_bytes'], 1)
    validate_seal_object({'schema_version': 1, 'run_id': run_id, 'seal_id': row['seal_id'],
        'logical_sha256': row['logical_sha256']}, run_id)
    return row


def check_sealed_pack(descriptor_raw: bytes, database: OriginalFile, *,
                      descriptor_sha256: str, manifest_sha256: str, logical_sha256: str,
                      run_id: str, expected: Json | None = None) -> Json:
    """Confirm the held physical originals match the trusted seal and manifest claims."""
    observed_descriptor = hashlib.sha256(descriptor_raw).hexdigest()
    if observed_descriptor != descriptor_sha256:
        raise ContractError('sealed PPO descriptor checksum differs')
    descriptor = validate_descriptor(parse_json(descriptor_raw, limit=MAX_JSON))
    if (descriptor['run_id'] != run_id or descriptor['publication'] != 'pending'
            or database.path.name != DATABASE or database.mutable
            or database.identity.st_size != descriptor['database_bytes']
            or not any(row['name'] == 'manifest.json' and row['sha256'] == manifest_sha256
                for row in descriptor['controls'])):
        raise ContractError('sealed PPO descriptor differs from the actual collection')
    database.check_identity()
    observed_database = database.hash()
    database.check_identity()
    if observed_database != descriptor['database_sha256']:
        raise ContractError('sealed PPO database checksum differs')
    observed = validate_pack_verification({'descriptor_sha256': observed_descriptor,
        'database_sha256': observed_database, 'database_bytes': database.identity.st_size,
        'seal_id': descriptor['seal_id'], 'logical_sha256': logical_sha256}, run_id=run_id)
    if expected is not None and observed != expected:
        raise ContractError('sealed PPO database verification differs from saved evidence')
    return observed


def verify_saved_pack(path: Path, expected: Json, *, run_id: str, manifest_sha256: str,
                      deadline: float, budget: Any) -> Json:
    """Verify committed saved evidence under the caller's actual workspace."""
    witness = validate_pack_verification(expected, run_id=run_id)
    with budget.reserve(2 * 64 * 1024):
        descriptor = OriginalFile(path, deadline)
        try:
            if descriptor.sha256 != witness['descriptor_sha256']:
                raise ContractError('sealed PPO descriptor checksum differs')
            size = descriptor.identity.st_size
            if size > MAX_JSON:
                raise ContractError('sealed PPO descriptor exceeds its format bound')
            with budget.reserve(32 * size):
                raw = descriptor.read(MAX_JSON)
                database = OriginalFile(path.parent / DATABASE, deadline,
                    database_expected_sha256=witness['database_sha256'])
                try:
                    observed = check_sealed_pack(raw, database,
                        descriptor_sha256=witness['descriptor_sha256'], manifest_sha256=manifest_sha256,
                        logical_sha256=witness['logical_sha256'], run_id=run_id, expected=witness)
                    descriptor.check_identity()
                    return observed
                finally:
                    database.close()
        finally:
            descriptor.close()


def verify_numeric_files(directory: Path, files: Json, *, deadline: float, budget: Any) -> None:
    """Stream the committed shard checksums without invoking the numeric reader."""
    from gf_models.structured_sealed import basename

    with budget.reserve(2 * 64 * 1024):
        for name, expected in files.items():
            # Open in the held-expected-hash mode and observe each payload once.
            original = OriginalFile(directory / basename(name), deadline, database_expected_sha256=expected)
            try:
                if original.hash() != expected:
                    raise ContractError('sealed PPO numeric shard checksum differs')
                original.check_identity()
            finally:
                original.close()
