"""Invocation-owned source reads, memory admission, evidence and scratch lifetime."""
from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from gf_common.domain import ContractError, Json
from gf_common.preprocessing import PreparationConfig, preparation_config

MIB = 1024 * 1024


class ResourceLimit(ContractError):
    """The operation cannot retain complete evidence within its declared budget."""


def deep_size(value: Any) -> int:
    """Charge the same ownership graph once per identity without recursive calls."""
    seen: set[int] = set()
    pending = [value]
    total = 0
    sizeof = sys.getsizeof
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sizeof(item)
        kind = type(item)
        if kind is str or kind is int or kind is bool or kind is float or item is None:
            continue
        if kind is dict:
            pending.extend(item)
            pending.extend(item.values())
        elif kind is list or kind is tuple or kind is set or kind is frozenset:
            pending.extend(item)
        elif isinstance(item, dict):
            # Preserve the existing items() contract for dictionary subclasses.
            for key, child in item.items():
                pending.append(key)
                pending.append(child)
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
        elif isinstance(item, Path):
            total += sizeof(str(item)) + 256
        elif hasattr(item, '__dict__'):
            pending.append(vars(item))
    return total


def canonical_digest(value: Any, *, deadline: float = float('inf')) -> str:
    """The existing canonical digest without allocating a second whole JSON tree."""
    hasher = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    for piece in encoder.iterencode(value):
        if time.monotonic() >= deadline:
            raise TimeoutError('canonical evidence hashing deadline')
        hasher.update(piece.encode())
    return hasher.hexdigest()


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def of(cls, stat: os.stat_result) -> FileIdentity:
        return cls(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class PreparationSession:
    """Serial preprocessing owner; accounting is an admission policy, not an RSS cap."""

    def __init__(self, config: PreparationConfig | None = None, *, deadline: float = float('inf'),
                 wall_deadline: float | None = None) -> None:
        self.config = preparation_config(asdict(config or PreparationConfig()))
        if self.config.backend != 'python':
            raise ContractError('Rust preparation requires the explicit native dispatch path')
        self.started = time.monotonic()
        self.wall_deadline = (wall_deadline if wall_deadline is not None else
                              min(deadline, self.started + self.config.wall_seconds))
        self.deadline = min(deadline, self.wall_deadline - self.config.cleanup_seconds)
        self.locks = ExitStack()
        self.parent_cache: dict[str, Any] = {}
        self._json: dict[Path, Any] = {}
        self._identities: dict[Path, tuple[FileIdentity, str]] = {}
        self._inventories: list[tuple[Path, str, tuple[str, ...]]] = []
        self._collections: set[tuple[Path, str]] = set()
        self._managed = 0
        self._scratch = 0
        self._closed = False
        self.scratch_directory: Path | None = None
        self.diagnostics: Json = {'schema_version': 1, 'backend': 'python', 'settings': asdict(self.config),
                                 'accounting_policy': '64x-raw-plus32MiB-decode; identity-deduplicated-retained-v1',
                                 'managed_peak_bytes': 0, 'scratch_peak_bytes': 0,
                                 'cache_peak_bytes': 0, 'source_reads': 0, 'journal_validations': 0,
                                 'parent_chain_validations': 0, 'episode_count': 0, 'cleanup': 'pending', 'read_bytes': 0,
                                 'phase_seconds': {'read_hash': 0., 'decode': 0., 'feature_encoding': 0., 'numeric_staging': 0.}}

    def __enter__(self) -> Self:
        self.check()
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        if self._closed:
            return
        if kind is None:
            self.finalize()
        else:
            try:
                self.abort()
            except BaseException as error:  # noqa: BLE001 - preserve original operation failure
                self.diagnostics['cleanup_error'] = str(error)

    def check(self, *, cleanup: bool = False) -> None:
        if self._closed:
            raise ContractError('preparation ownership is closed')
        if time.monotonic() >= (self.wall_deadline if cleanup else self.deadline):
            raise TimeoutError('preparation deadline')

    def charge(self, amount: int) -> None:
        self.check()
        if amount < 0 or self._managed + amount > self.config.memory_budget_mib * MIB:
            raise ResourceLimit('preprocessing managed memory budget exhausted')
        self._managed += amount
        self.diagnostics['managed_peak_bytes'] = max(self.diagnostics['managed_peak_bytes'], self._managed)

    def release(self, amount: int) -> None:
        if amount < 0 or amount > self._managed:
            raise RuntimeError('invalid preprocessing memory release')
        self._managed -= amount

    def retain(self, value: Any) -> int:
        amount = deep_size(value) + 256  # Per-entry dict/set/list growth headroom.
        self.charge(amount)
        return amount

    def reserve_scratch(self, amount: int) -> None:
        self.check()
        if amount < 0 or self._scratch + amount > self.config.scratch_budget_mib * MIB:
            raise ResourceLimit('preprocessing scratch budget exhausted')
        self._scratch += amount
        self.diagnostics['scratch_peak_bytes'] = max(self.diagnostics['scratch_peak_bytes'], self._scratch)

    def release_scratch(self, amount: int) -> None:
        if amount < 0 or amount > self._scratch:
            raise RuntimeError('invalid preprocessing scratch release')
        self._scratch -= amount

    def scratch(self) -> Path:
        self.check()
        if self.scratch_directory is None:
            self.scratch_directory = Path(tempfile.mkdtemp(prefix='gf-preparation-'))
            self.diagnostics['scratch_directory'] = str(self.scratch_directory)
        return self.scratch_directory

    def inventory(self, directory: Path, pattern: str) -> None:
        names = tuple(sorted(str(path.resolve()) for path in directory.glob(pattern)))
        entry = (directory, pattern, names)
        self.retain(entry)
        self._inventories.append(entry)

    def _remember(self, path: Path, identity: FileIdentity, checksum: str) -> None:
        previous = self._identities.get(path)
        if previous is not None and previous != (identity, checksum):
            raise ContractError('source changed during preparation')
        if previous is None:
            self.retain((path, identity, checksum))
            self._identities[path] = (identity, checksum)

    def read_bytes(self, path: Path, *, admitted: FileIdentity | None = None) -> bytes:
        started = time.monotonic()
        self.check()
        path = path.resolve()
        before = FileIdentity.of(path.stat())
        if admitted is not None and before != admitted:
            raise ContractError('source changed after memory admission')
        limit = self.config.max_source_file_mib * MIB
        if before.size > limit:
            raise ResourceLimit(f'source exceeds max_source_file_mib: {path}')
        with path.open('rb') as stream:
            if FileIdentity.of(os.fstat(stream.fileno())) != before:
                raise ContractError('source replaced before bounded read')
            encoded = stream.read(before.size + 1)
            if len(encoded) > limit:
                raise ResourceLimit('source grew beyond max_source_file_mib')
            after = FileIdentity.of(os.fstat(stream.fileno()))
        if before != after or before != FileIdentity.of(path.stat()) or len(encoded) != before.size:
            raise ContractError('source changed during bounded read')
        self._remember(path, before, hashlib.sha256(encoded).hexdigest())
        self.diagnostics['source_reads'] += 1
        self.diagnostics['read_bytes'] += len(encoded)
        self.diagnostics['phase_seconds']['read_hash'] += time.monotonic() - started
        self.check()
        return encoded

    def hash_file(self, path: Path) -> str:
        """Hash bytes read, reusing an already-read exact identity within this owner."""
        self.check()
        path = path.resolve()
        prior = self._identities.get(path)
        before = FileIdentity.of(path.stat())
        if prior is not None:
            if prior[0] != before:
                raise ContractError('source identity changed during preparation')
            return prior[1]
        if before.size > self.config.max_source_file_mib * MIB:
            raise ResourceLimit(f'source exceeds max_source_file_mib: {path}')
        if path.suffix == '.json':
            self.read_json(path)
            return self._identities[path][1]
        started = time.monotonic()
        hasher = hashlib.sha256()
        buffer_charge = min(MIB, before.size + 1)
        self.charge(buffer_charge)
        try:
            with path.open('rb') as stream:
                if FileIdentity.of(os.fstat(stream.fileno())) != before:
                    raise ContractError('source replaced before hashing')
                length = 0
                while block := stream.read(min(MIB, before.size - length + 1)):
                    self.check()
                    length += len(block)
                    if length > before.size:
                        raise ContractError('source grew beyond admitted hash length')
                    hasher.update(block)
                after = FileIdentity.of(os.fstat(stream.fileno()))
            if before != after or before != FileIdentity.of(path.stat()):
                raise ContractError('source changed while hashing')
            self._remember(path, before, hasher.hexdigest())
            self.diagnostics['source_reads'] += 1
            self.diagnostics['read_bytes'] += before.size
            self.diagnostics['phase_seconds']['read_hash'] += time.monotonic() - started
        finally:
            self.release(buffer_charge)
        return hasher.hexdigest()

    def read_json(self, path: Path) -> Json:
        self.check()
        path = path.resolve()
        if path in self._json:
            self.hash_file(path)  # Check identity even while returning retained evidence.
            return self._json[path]
        admitted = FileIdentity.of(path.stat())
        size = admitted.size
        if size > self.config.max_source_file_mib * MIB:
            raise ResourceLimit(f'source exceeds max_source_file_mib: {path}')
        reservation = 64 * size + MIB
        self.charge(reservation)
        try:
            encoded = self.read_bytes(path, admitted=admitted)
            decode_started = time.monotonic()
            value = json.loads(encoded.decode('utf-8'))
            self.diagnostics['phase_seconds']['decode'] += time.monotonic() - decode_started
            if not isinstance(value, dict):
                raise ContractError('object record required')
            actual = deep_size(value)
            if actual > reservation:
                self.charge(actual - reservation)
                reservation = actual
        except BaseException:
            self.release(reservation)
            raise
        self.release(reservation - actual)
        self._json[path] = value  # Essential live evidence, charged as working memory.
        return value

    def validate_parent(self, directory: Path) -> tuple[Json, list[Json]]:
        from gf_training.lineage import run_lock, validate_chain
        directory = directory.resolve()
        key = str(directory)
        if key not in self.parent_cache:
            self.locks.enter_context(run_lock(directory))
            for pattern in ('generations/*.json', 'updates/*.json', 'artifacts/*/*'):
                self.inventory(directory, pattern)
            self.parent_cache[key] = validate_chain(directory, deadline=self.deadline, session=self)
            self.diagnostics['parent_chain_validations'] += 1
            # Parsed objects are separately charged by read_json; only container ownership here.
            self.charge(sys.getsizeof(self.parent_cache[key][1]) + sys.getsizeof(self.parent_cache[key]))
        return self.parent_cache[key]

    def validate_collection(self, manifest: Json, directory: Path) -> None:
        from gf_training.lineage import validate_collection_parent
        path = (directory / 'manifest.json').resolve()
        key = (path, self.hash_file(path))
        if key in self._collections:
            return
        parent = Path(manifest['ppo']['parent_run_path']).resolve()
        self.validate_parent(parent)
        validate_collection_parent(manifest, directory, parent_cache=self.parent_cache, locks=self.locks,
                                   deadline=self.deadline, session=self)
        self.retain(key)
        self._collections.add(key)

    @contextmanager
    def episode(self, paths: tuple[Path, ...]) -> Iterator[dict[Path, Any]]:
        """Reserve both journals/Replays before decoding a complete episode."""
        self.check()
        admitted = [(path.resolve(), FileIdentity.of(path.stat())) for path in paths]
        if any(identity.size > self.config.max_source_file_mib * MIB for _, identity in admitted):
            raise ResourceLimit('episode source exceeds max_source_file_mib')
        reservation = 64 * sum(identity.size for _, identity in admitted) + 32 * MIB
        self.charge(reservation)
        self._episode_charge = reservation
        payload: dict[Path, Any] = {}
        try:
            for path, identity in admitted:
                encoded = self.read_bytes(path, admitted=identity)
                decode_started = time.monotonic()
                if path.suffix == '.jsonl':
                    records = []
                    # Match open(encoding='utf-8') universal-newline iteration;
                    # splitlines() would also split valid Unicode in JSON strings.
                    with io.StringIO(encoded.decode('utf-8'), newline=None) as lines:
                        for line in lines:
                            self.check()
                            record = line.removesuffix('\n')
                            if len(record.encode('utf-8')) > self.config.max_record_mib * MIB:
                                raise ResourceLimit('journal exceeds max_record_mib')
                            records.append(json.loads(line))
                    payload[path.resolve()] = records
                else:
                    payload[path.resolve()] = json.loads(encoded)
                self.diagnostics['phase_seconds']['decode'] += time.monotonic() - decode_started
            self.diagnostics['episode_count'] += 1
            yield payload
        finally:
            payload.clear()
            self.release(self._episode_charge)
            self._episode_charge = 0

    def settle_episode(self, evidence: Any) -> None:
        actual = deep_size(evidence)
        self.diagnostics['episode_deep_size_passes'] = self.diagnostics.get('episode_deep_size_passes', 0) + 1
        if actual > self._episode_charge:
            self.charge(actual - self._episode_charge)
        else:
            self.release(self._episode_charge - actual)
        self._episode_charge = actual

    def verify_sources(self) -> None:
        self.check(cleanup=True)
        for path, (identity, _) in self._identities.items():
            self.check(cleanup=True)
            if FileIdentity.of(path.stat()) != identity:
                raise ContractError('source invalidated before preparation publication')
        for directory, pattern, names in self._inventories:
            if tuple(sorted(str(path.resolve()) for path in directory.glob(pattern))) != names:
                raise ContractError('source inventory changed before publication')
        # Mutable control identities are rehashed under the still-owned source locks.
        for parent_directory in self.parent_cache:
            for name in ('run.json', 'latest.json', 'artifact-registry.json'):
                path = Path(parent_directory) / name
                expected = self._identities[path.resolve()][1]
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    while block := stream.read(MIB):
                        self.check(cleanup=True)
                        hasher.update(block)
                if hasher.hexdigest() != expected:
                    raise ContractError('mutable parent control changed before publication')

    def _close(self) -> None:
        error: BaseException | None = None
        try:
            if self.scratch_directory is not None:
                shutil.rmtree(self.scratch_directory)
        except BaseException as exc:  # noqa: BLE001 - locks still close on scratch failure
            error = exc
        finally:
            self._json.clear()
            self.parent_cache.clear()
            self._collections.clear()
            self._identities.clear()
            self._inventories.clear()
            try:
                self.locks.close()
            except BaseException as exc:  # noqa: BLE001 - preserve first cleanup failure
                error = error or exc
            self._managed = self._scratch = 0
            self._closed = True
            self.diagnostics['cleanup'] = 'complete' if error is None else 'failed'
            self.diagnostics['elapsed_seconds'] = time.monotonic() - self.started
            self.diagnostics['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if error is not None:
            raise error

    def finalize(self) -> None:
        if self._closed:
            return
        try:
            self.verify_sources()
        except BaseException:
            try:
                self._close()
            except BaseException as error:  # noqa: BLE001 - retain original verification failure
                self.diagnostics['cleanup_error'] = str(error)
            raise
        self._close()
        if time.monotonic() >= self.wall_deadline:
            raise TimeoutError('preparation cleanup exceeded original deadline')

    def abort(self) -> None:
        if not self._closed:
            self._close()
