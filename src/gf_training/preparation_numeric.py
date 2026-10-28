"""Ordered typed staging, exact legacy NPZ serialization and explicit lifetimes."""
from __future__ import annotations

import hashlib
import math
import mmap
import os
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
import numpy.typing as npt

from gf_common.domain import ContractError, Json, canonical, integer
from gf_models.config import EncoderConfig, encoder_config
from gf_models.encoder import _encode_validated, feature_names
from gf_training.journal import AcceptedDecision
from gf_training.preparation import MIB, PreparationSession
from gf_training.q_data import RunBundle

Array = npt.NDArray[np.float64]


class PreparedInputs:
    """Own staged numeric inputs until an explicit pre-promotion finalization.

    Optional array properties lend readonly mappings. Callers must discard all
    borrowed views before finalizing. Production serialization uses bounded file
    windows directly, so a full matrix never becomes a resident Python array.
    """

    def __init__(self, session: PreparationSession, frozen: Json, config: EncoderConfig) -> None:
        self.session = session
        self.config = encoder_config(asdict(config))
        self.diagnostics = session.diagnostics
        self.dataset = frozen
        self._width = len(feature_names(self.config))
        session.retain(feature_names(self.config))  # Retained schema is essential working memory.
        train = integer(frozen.get('train_decisions'), 1)
        validation = integer(frozen.get('validation_decisions'), 1)
        self.shapes = {'x': (train, self._width), 'y': (train,),
                       'validation_x': (validation, self._width), 'validation_y': (validation,)}
        size = sum(math.prod(shape) * 8 for shape in self.shapes.values())
        # Simultaneous .npy staging, destination legacy NPZ and both header sets.
        session.reserve_scratch(2 * size + 32768)
        self._window = min(session.config.chunk_target_mib * MIB,
                           max(4096, session.config.memory_budget_mib * MIB // 8))
        session.charge(2 * self._window + self._width * 32 + 65536)
        self.diagnostics['numeric_window_bytes'] = self._window
        self.diagnostics['numeric_staging_bytes'] = size
        self.diagnostics['chunk_granularity'] = 'serial complete episode; numeric/index copy windows bounded by chunk_target'
        self._paths = {name: session.scratch() / f'{name}.npy' for name in self.shapes}
        self._outputs: dict[str, BinaryIO] = {}
        self._identity_paths = {split: session.scratch() / f'{split}-identities.jsonl' for split in ('train', 'validation')}
        self._identity_outputs: dict[str, BinaryIO] = {}
        self._counts = {'train': 0, 'validation': 0}
        self._mappings: dict[str, Any] = {}
        self._sealed = False
        self._finalized = False
        try:
            for name, path in self._paths.items():
                stream = path.open('xb')
                self._outputs[name] = stream
                np.lib.format.write_array_header_1_0(stream, {'descr': np.dtype('float64').str,
                                                            'fortran_order': False, 'shape': self.shapes[name]})
            for split, path in self._identity_paths.items():
                self._identity_outputs[split] = path.open('xb')
        except BaseException:
            self.abort()
            raise

    def __enter__(self) -> Self:
        if not self._sealed or self._finalized:
            raise ContractError('numeric preparation is not usable')
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        if self._finalized:
            return
        if kind is None:
            self.finalize()
        else:
            try:
                self.abort()
            except BaseException as error:  # noqa: BLE001 - preserve the primary fit failure
                self.diagnostics['cleanup_error'] = str(error)

    def accept(self, decision: AcceptedDecision, bundle: RunBundle) -> None:
        self.session.check()
        split = decision.split
        prefix = '' if split == 'train' else 'validation_'
        count = self._counts[split]
        if count >= self.shapes[prefix + 'x'][0]:
            raise ContractError('validated decisions exceed frozen numeric shape')
        started = time.monotonic()
        vector = _encode_validated(decision.value, decision.action, self.config)
        self.diagnostics['phase_seconds']['feature_encoding'] += time.monotonic() - started
        started = time.monotonic()
        self._outputs[prefix + 'x'].write(vector.tobytes())
        self._outputs[prefix + 'y'].write(np.float64(decision.target).tobytes())
        encoded = (canonical([split, *decision.identity]) + '\n').encode()
        self.session.reserve_scratch(len(encoded))
        self._identity_outputs[split].write(encoded)
        self._counts[split] += 1
        self.diagnostics['phase_seconds']['numeric_staging'] += time.monotonic() - started

    def seal(self, dataset: Json) -> None:
        if self._counts != {'train': self.shapes['x'][0], 'validation': self.shapes['validation_x'][0]}:
            raise ContractError('validated numeric row count differs from frozen shape')
        self.dataset = dataset
        self._close_writers()
        identity = hashlib.sha256()
        for split in ('train', 'validation'):
            with self._identity_paths[split].open('rb') as stream:
                while block := stream.read(self._window):
                    self.session.check()
                    identity.update(block)
        self.diagnostics['sample_identity_sha256'] = identity.hexdigest()
        self._sealed = True

    def _borrow(self, name: str) -> Array:
        if not self._sealed or self._finalized:
            raise ContractError('numeric preparation is not usable')
        if name not in self._mappings:
            size = math.prod(self.shapes[name]) * 8
            self.session.charge(size)
            value = np.load(self._paths[name], mmap_mode='r', allow_pickle=False)
            value.flags.writeable = False
            self._mappings[name] = value
        return self._mappings[name]

    @property
    def x(self) -> Array:
        return self._borrow('x')

    @property
    def y(self) -> Array:
        return self._borrow('y')

    @property
    def validation_x(self) -> Array:
        return self._borrow('validation_x')

    @property
    def validation_y(self) -> Array:
        return self._borrow('validation_y')

    def write_numeric(self, path: Path) -> None:
        """Copy exact NPY bytes into NumPy's stored/ZIP64 legacy container layout."""
        if not self._sealed or self._finalized:
            raise ContractError('numeric preparation is not usable')
        started = time.monotonic()
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in self._paths.items():
                with source.open('rb') as stream, archive.open(name + '.npy', 'w', force_zip64=True) as destination:
                    while block := stream.read(self._window):
                        self.session.check()
                        destination.write(block)
        self.session.check()
        self.diagnostics['phase_seconds']['numeric_staging'] += time.monotonic() - started

    def _close_writers(self) -> None:
        error: BaseException | None = None
        for stream in (*self._outputs.values(), *self._identity_outputs.values()):
            if not stream.closed:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                except BaseException as exc:  # noqa: BLE001 - close all owned streams
                    error = error or exc
                    try:
                        stream.close()
                    except BaseException as close_error:  # noqa: BLE001 - report secondary close failure
                        self.diagnostics['stream_close_error'] = str(close_error)
        if error is not None:
            raise error

    def _drop_mappings(self) -> None:
        error: BaseException | None = None
        for value in self._mappings.values():
            mapping = getattr(value, '_mmap', None)
            if isinstance(mapping, mmap.mmap):
                try:
                    mapping.close()
                except BaseException as exc:  # noqa: BLE001 - attempt every owned mapping
                    error = error or exc
        self._mappings.clear()
        if error is not None:
            raise error

    def finalize(self) -> None:
        if self._finalized:
            return
        started = time.monotonic()
        try:
            self._close_writers()
            self._drop_mappings()
            self.session.finalize()
        except BaseException:
            try:
                self.abort()
            except BaseException as error:  # noqa: BLE001 - preserve finalization failure
                self.diagnostics['cleanup_error'] = str(error)
            raise
        finally:
            self.diagnostics.setdefault('phase_seconds', {})['finalization'] = time.monotonic() - started
        self._finalized = True

    def abort(self) -> None:
        if self._finalized:
            return
        error: BaseException | None = None
        for close in (self._close_writers, self._drop_mappings, self.session.abort):
            try:
                close()
            except BaseException as exc:  # noqa: BLE001 - always release source ownership
                error = error or exc
        self._finalized = True
        if error is not None:
            raise error


def prepare_inputs(path: Path, config: EncoderConfig, session: PreparationSession,
                   validate: Callable[[Path, PreparationSession, Callable[[AcceptedDecision, RunBundle], None]], Json]) -> PreparedInputs:
    owner: PreparedInputs | None = None
    try:
        frozen = session.read_json(path)
        owner = PreparedInputs(session, frozen, config)
        actual = validate(path, session, owner.accept)
        owner.seal(actual)
        phases = session.diagnostics['phase_seconds']
        phases['source_validation_and_bookkeeping'] = max(0., time.monotonic() - session.started - sum(phases.values()))
        session.diagnostics['phase_scope'] = 'disjoint measured read/hash, decode, feature encoding, numeric staging, finalization; remaining preparation wall is validation/bookkeeping'
        return owner
    except BaseException:
        try:
            owner.abort() if owner is not None else session.abort()
        except BaseException as error:  # noqa: BLE001 - preserve source/validation error
            session.diagnostics['cleanup_error'] = str(error)
        raise
