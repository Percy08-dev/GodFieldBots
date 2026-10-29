"""Typed native index columns into existing Arrow/Parquet, with no Python rows."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig
from gf_training.journal import atomic_json
from gf_training.native_authorization import NativePreparation
from gf_training.native_learning import fixed_targets


def _column(pa: Any, field: Any, value: Any, rows: int) -> Any:
    if pa.types.is_string(field.type):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ContractError('native index UTF8 buffer pair required')
        data, offsets = value
        if (not isinstance(data, np.ndarray) or data.dtype != np.uint8 or data.ndim != 1
                or not isinstance(offsets, np.ndarray) or offsets.dtype != np.int32
                or offsets.shape != (rows + 1,) or data.flags.writeable or offsets.flags.writeable
                or not data.flags.c_contiguous or not offsets.flags.c_contiguous
                or offsets[0] != 0 or offsets[-1] != len(data) or np.any(offsets < 0)
                or np.any(offsets > len(data)) or np.any(np.diff(offsets) < 0)):
            raise ContractError('invalid native index UTF8 offsets/data')
        result = pa.Array.from_buffers(field.type, rows, [None, pa.py_buffer(offsets), pa.py_buffer(data)])
        result.validate(full=True)
        return result
    dtype = np.float64 if pa.types.is_float64(field.type) else np.int64
    if (not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != (rows,)
            or value.flags.writeable or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ContractError('invalid native index numeric column')
    return pa.Array.from_buffers(field.type, rows, [None, pa.py_buffer(value)])


def index_dataset(path: Path, output: Path, settings: PreparationConfig, *, deadline: float) -> Json:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gf_training.fixed_data import _publish_owned

    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists(follow_symlinks=False) or sidecar.exists(follow_symlinks=False):
        raise ContractError('index output and sidecar must both be fresh paths')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary_sidecar = temporary.with_suffix(temporary.suffix + '.json')
    schema = pa.schema([('run_id', pa.string()), ('parent_run_id', pa.string()), ('batch_id', pa.string()),
                        ('generation', pa.int64()), ('split', pa.string()), ('learner_viewer', pa.int64()),
                        ('opponent', pa.string()), ('source_manifest_sha256', pa.string()),
                        ('game_id', pa.string()), ('decision', pa.int64()), ('return', pa.float64()),
                        ('dataset_content_sha256', pa.string())])
    started = time.monotonic()
    wall = min(deadline, started + settings.wall_seconds)
    owner: NativePreparation | None = None
    writer: Any = None
    try:
        owner = NativePreparation({'kind': 'offline', 'operation': 'index', 'encoder': asdict(EncoderConfig()),
                                   'dataset_path': str(path.resolve())}, settings,
                                  deadline=wall - settings.cleanup_seconds, wall_deadline=wall)
        count = 0
        while (batch := owner.next_batch(0, (4096, 128, 0, 16))) is not None:
            labels = fixed_targets(batch, global_ready=owner.proof()['structural_valid'])
            rows = len(labels)
            if batch.index_columns is None or len(batch.index_columns) != len(schema):
                raise ContractError('native index column count mismatch')
            if not np.array_equal(batch.index_columns[10], labels):
                raise ContractError('native index recorded return differs from Python label')
            if rows:
                arrays = [_column(pa, field, value, rows)
                          for field, value in zip(schema, batch.index_columns, strict=True)]
                # The reference ledger serializes canonical JSON before Arrow
                # reads each row, so its column insertion order is alphabetical.
                table = pa.Table.from_arrays(arrays, schema=schema).select(sorted(schema.names))
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(temporary, table.schema)
                owner.check()
                writer.write_table(table)
                count += rows
                del table, arrays
            del labels, batch
        if writer is None:
            raise ContractError('fixed index requires rows')
        writer.close()
        writer = None
        owner.check()
        checksum = hashlib.sha256()
        with temporary.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                owner.check()
                checksum.update(block)
        proof = owner.proof()
        dataset = proof['dataset_manifest']
        if count != dataset['train_decisions'] + dataset['validation_decisions']:
            raise ContractError('native index row count differs from validated dataset')
        result = {'schema_version': 1, 'authoritative': False,
                  'dataset_content_sha256': dataset['content_sha256'], 'rows': count,
                  'row_content_sha256': proof['index_row_content_sha256'],
                  'parquet_sha256': checksum.hexdigest()}
        atomic_json(temporary_sidecar, result)
        _publish_owned(owner, ((temporary, output), (temporary_sidecar, sidecar)))
        return result
    except BaseException as error:
        for close in (lambda: writer.close() if writer is not None else None,
                      lambda: owner.abort() if owner is not None else None):
            try:
                close()
            except BaseException as cleanup_error:  # noqa: BLE001 - drain every owned writer/session
                error.add_note(f'native index cleanup failed: {cleanup_error}')
        raise
    finally:
        for artifact in (temporary, temporary_sidecar,
                         temporary_sidecar.with_name(temporary_sidecar.name + '.tmp')):
            artifact.unlink(missing_ok=True)
