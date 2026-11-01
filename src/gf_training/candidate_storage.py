"""Parent-owned typed candidate sidecar publication, bounded reads/inventory."""
from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from gf_common.action_contract import ActionContract, parse_action_contract, validate_record_purpose
from gf_common.candidate_columns import (
    MAX_COLUMN_BYTES,
    decode_candidate_columns,
    encode_candidate_columns,
    validate_descriptor,
)
from gf_common.domain import ContractError, Json, integer
from gf_common.structured import CandidateTable, DecisionFrame


def candidate_basename(actor: int, decision: int) -> str:
    integer(actor, 1, 2)
    integer(decision, 0, 2**32 - 1)
    return f"player-{actor}-decision-{decision:010}.columns"


def _identity(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _bounded_hash(source: BinaryIO, size: int) -> str:
    remaining = size
    offset = 0
    value = hashlib.sha256()
    while remaining:
        # Bypass BufferedReader caches: same-size external writes can otherwise
        # leave both stat timestamps and a seek/read buffer apparently unchanged.
        chunk = os.pread(source.fileno(), min(remaining, 1024**2), offset)
        if not chunk:
            raise ContractError("candidate file shortened during verification")
        remaining -= len(chunk)
        offset += len(chunk)
        value.update(chunk)
    if os.pread(source.fileno(), 1, size):
        raise ContractError("candidate file grew during verification")
    return value.hexdigest()


def _open_regular(path: Path, size: int) -> BinaryIO:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ContractError("candidate sidecar missing/unsafe") from exc
    source = os.fdopen(descriptor, "rb")
    value = os.fstat(source.fileno())
    if not stat.S_ISREG(value.st_mode) or value.st_size != size:
        source.close()
        raise ContractError("candidate sidecar file size/type mismatch")
    return source


def publish_candidates(directory: Path, frame: DecisionFrame, table: CandidateTable, decision: int, *,
                       max_positions: int = 128, max_bytes: int = MAX_COLUMN_BYTES,
                       action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Json:
    validate_record_purpose(purpose, action_contract)
    parse_action_contract(asdict(table.action_contract), expected=action_contract)
    if (table.purpose != purpose or type(synthetic_capacity) is not bool
            or table.synthetic_capacity != synthetic_capacity):
        raise ContractError("candidate publication purpose/synthetic classification mismatch")
    table.validate(frame)
    name = candidate_basename(frame.actor, decision)
    final, temporary = directory / name, directory / (name + ".tmp")
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError("candidate sidecar directory must be owned and regular")
    for path in (final, temporary):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise ContractError("candidate sidecar destination already exists")
    payload, descriptor = encode_candidate_columns(table, name, max_positions=max_positions, max_bytes=max_bytes)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK
    owned = os.open(temporary, flags, 0o600)
    # Keep the actual written object open through publication. Failed temporary
    # and final objects remain orphan evidence; no descriptor is returned on error.
    with os.fdopen(owned, "w+b") as output:
        for start in range(0, len(payload), 1024**2):
            chunk = payload[start:start + 1024**2]
            if output.write(chunk) != len(chunk):
                raise OSError("candidate sidecar write made incomplete progress")
        output.flush()
        os.fsync(output.fileno())
        verified = os.fstat(output.fileno())
        if not stat.S_ISREG(verified.st_mode) or verified.st_size != len(payload) or _bounded_hash(output, len(payload)) != descriptor["sha256"]:
            raise ContractError("candidate sidecar write/hash mismatch")
        # Atomic no-overwrite publication. Linking legitimately changes ctime and
        # nlink, so compare immutable object identity + size/mtime and actual bytes.
        os.link(temporary, final, follow_symlinks=False)
        with _open_regular(final, len(payload)) as published:
            actual = os.fstat(published.fileno())
            same_object = (actual.st_dev, actual.st_ino) == (verified.st_dev, verified.st_ino)
            if (not same_object or actual.st_mtime_ns != verified.st_mtime_ns
                    or _bounded_hash(published, len(payload)) != descriptor["sha256"]):
                raise ContractError("published candidate object/content differs from verified owned temporary")
            after_hash = os.fstat(published.fileno())
            current = final.lstat()
            # Never refresh the baseline after hashing: even an EOF-time
            # truncation must differ from the captured PRE-hash snapshot.
            if (_identity(after_hash) != _identity(actual) or after_hash.st_size != len(payload)
                    or not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(actual)):
                raise ContractError("published candidate path/content changed during verification")
        temp_now = temporary.lstat()
        if not stat.S_ISREG(temp_now.st_mode) or (temp_now.st_dev, temp_now.st_ino) != (verified.st_dev, verified.st_ino):
            raise ContractError("owned candidate temporary path replaced during publication")
        temporary.unlink()
    return descriptor


@dataclass(frozen=True)
class _ReadSnapshot:
    table: CandidateTable
    identity: tuple[int, ...]
    sha256: str
    source: BinaryIO
    path: Path
    size: int


def _still_current(snapshot: _ReadSnapshot) -> None:
    # Stat granularity cannot alone detect a same-size in-place overwrite. Rehash
    # this SAME opened object, then bind the path to its original verified identity.
    if (_bounded_hash(snapshot.source, snapshot.size) != snapshot.sha256
            or _identity(os.fstat(snapshot.source.fileno())) != snapshot.identity):
        raise ContractError("candidate content changed after verified read")
    try:
        current = snapshot.path.lstat()
    except OSError as exc:
        raise ContractError("candidate inventory entry disappeared after verified read") from exc
    if not stat.S_ISREG(current.st_mode) or _identity(current) != snapshot.identity:
        raise ContractError("candidate inventory path changed after verified read")


@contextmanager
def _verified_read(directory: Path, descriptor: Json, frame: DecisionFrame, decision: int, *,
                   max_positions: int = 128, max_bytes: int = MAX_COLUMN_BYTES,
                   action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Iterator[_ReadSnapshot]:
    name = candidate_basename(frame.actor, decision)
    _, size = validate_descriptor(descriptor, expected_file=name, expected_frame_hash=frame.frame_hash, max_bytes=max_bytes,
                                  action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    path = directory / name
    if directory.is_symlink():
        raise ContractError("symlink candidate sidecar directory")
    with _open_regular(path, size) as source:
        before = os.fstat(source.fileno())
        payload = source.read(size + 1)
        after = os.fstat(source.fileno())
        if _identity(before) != _identity(after):
            raise ContractError("candidate sidecar changed while reading")
        table = decode_candidate_columns(payload, descriptor, expected_file=name, expected_frame_hash=frame.frame_hash,
                                         max_positions=max_positions, max_bytes=max_bytes, action_contract=action_contract,
                                         purpose=purpose, synthetic_capacity=synthetic_capacity)
        table.validate(frame)
        snapshot = _ReadSnapshot(table, _identity(after), hashlib.sha256(payload).hexdigest(), source, path, size)
        yield snapshot
        _still_current(snapshot)


def read_candidates(directory: Path, descriptor: Json, frame: DecisionFrame, decision: int, *,
                    max_positions: int = 128, max_bytes: int = MAX_COLUMN_BYTES,
                    action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> CandidateTable:
    with _verified_read(directory, descriptor, frame, decision, max_positions=max_positions, max_bytes=max_bytes,
                         action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity) as snapshot:
        return snapshot.table


def sidecar_inventory(directory: Path, descriptor: Json, frame: DecisionFrame, decision: int, *,
                      max_positions: int = 128, max_bytes: int = MAX_COLUMN_BYTES,
                      action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Json:
    with _verified_read(directory, descriptor, frame, decision, max_positions=max_positions, max_bytes=max_bytes,
                         action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity) as snapshot:
        _still_current(snapshot)
        return {"path": str(snapshot.path.absolute()), "sha256": snapshot.sha256, "bytes": snapshot.size,
                "identity": list(snapshot.identity), "frame_hash": descriptor["frame_hash"], "table_hash": descriptor["table_hash"],
                "action_contract": asdict(snapshot.table.action_contract), "purpose": snapshot.table.purpose,
                "synthetic_capacity": snapshot.table.synthetic_capacity}
