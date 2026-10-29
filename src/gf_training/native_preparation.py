"""Fixed-data native dispatch and Python-owned labels/legacy numeric containers."""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np

from gf_common.domain import ContractError, Json, integer
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig, encoder_config
from gf_training.native_authorization import NativePreparation
from gf_training.native_learning import fixed_consumer_layout, fixed_targets

MIB = 1024 * 1024


class _PreparationWall:
    """Fixed exclusive construction ledger; native worker timings remain nested."""
    phases = ('owner_session_setup', 'next_batch_wait', 'proof_and_initialization',
              'array_reward_validation_labels', 'bounded_npy_write', 'batch_release', 'seal', 'other')

    def __init__(self, clock: Callable[[], int] | None = None) -> None:
        self._clock = clock or time.perf_counter_ns
        self._started = self._since = self._clock()
        self._current = 'other'
        self._ns = dict.fromkeys(self.phases, 0)
        self._result: Json | None = None

    def _switch(self, name: str) -> None:
        now = self._clock()
        self._ns[self._current] += now - self._since
        self._since, self._current = now, name

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._current
        self._switch(name)
        try:
            yield
        finally:
            self._switch(previous)

    def finish(self, *, success: bool) -> Json:
        if self._result is None:
            self._switch('other')
            total = self._since - self._started
            summed = sum(self._ns.values())
            self._result = {'scope': 'NativePreparedInputs construction entry to ledger close; excludes archive, final source checks and cleanup',
                            'child_timing_scope': 'next_batch_wait includes FFI execution/handoff and proof-envelope decoding; native and worker counters are nested',
                            'parent_nanoseconds': total, 'parent_seconds': total / 1e9,
                            'partition_nanoseconds': dict(self._ns),
                            'partition_sum_nanoseconds': summed,
                            'residual_nanoseconds': total - summed, 'success': success}
        return self._result


def _deadlines(settings: PreparationConfig, deadline: float, wall_deadline: float | None) -> tuple[float, float]:
    started = time.monotonic()
    wall = (wall_deadline if wall_deadline is not None else
            min(deadline, started + settings.wall_seconds))
    return min(deadline, wall - settings.cleanup_seconds), wall

def _owner(plan: Json, settings: PreparationConfig, deadline: float,
           wall_deadline: float | None = None) -> NativePreparation:
    work, wall = _deadlines(settings, deadline, wall_deadline)
    owner = NativePreparation(plan, settings, deadline=work, wall_deadline=wall)
    return owner


def summary(plan: Json, settings: PreparationConfig, *, deadline: float) -> tuple[Json, NativePreparation]:
    """Validate facts under ownership; the caller finalizes after staging output."""
    plan = {**plan, 'operation': 'summary', 'encoder': asdict(EncoderConfig())}
    owner = _owner(plan, settings, deadline)
    try:
        while (batch := owner.next_batch(0, (4096, 128, 0, 16))) is not None:
            labels = fixed_targets(batch, global_ready=owner.proof()['structural_valid'])
            del labels, batch
        return owner.proof()['dataset_manifest'], owner
    except BaseException as error:
        try:
            owner.abort()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve source failure after owned cleanup
            error.add_note(f'native preparation cleanup failed: {cleanup_error}')
        raise


class NativePreparedInputs:
    """Keep source locks through learner reap and final validation/cleanup.

    Native exports are released after each episode batch. Only Python-derived
    labels enter these legacy NPY/NPZ files; native buffers never receive labels.
    Compatibility properties export separately leased readonly native arrays.
    """

    def __init__(self, path: Path, config: EncoderConfig, settings: PreparationConfig,
                 *, deadline: float, wall_deadline: float | None = None) -> None:
        self._wall = _PreparationWall()
        with self._wall.phase('owner_session_setup'):
            self.config = encoder_config(asdict(config))
            self.session = _owner({'kind': 'offline', 'operation': 'prepare',
                                   'dataset_path': str(path.resolve()), 'encoder': asdict(config)},
                                  settings, deadline, wall_deadline)
            self._setup(settings)
        try:
            while True:
                with self._wall.phase('next_batch_wait'):
                    batch = self.session.next_batch(0, fixed_consumer_layout(config))
                if batch is None:
                    break
                with self._wall.phase('proof_and_initialization'):
                    proof = self.session.proof()
                    if not self.shapes:
                        self.dataset = proof['dataset_manifest']
                        self._initialize(batch.features.shape[1])
                with self._wall.phase('array_reward_validation_labels'):
                    labels = fixed_targets(batch, global_ready=proof['structural_valid'])
                with self._wall.phase('bounded_npy_write'):
                    self._consume(batch, labels)
                with self._wall.phase('batch_release'):
                    del labels, batch
            with self._wall.phase('seal'):
                self._seal()
                capture = getattr(self.session, 'data_stage_snapshot', None)
                self.diagnostics['native_data_stage_snapshot'] = capture() if callable(capture) else None
            self.diagnostics['native_prepared_inputs_wall'] = self._wall.finish(success=True)
        except BaseException as error:
            self.diagnostics['native_prepared_inputs_wall'] = self._wall.finish(success=False)
            try:
                self.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve original initialization failure
                error.add_note(f'native numeric cleanup failed: {cleanup_error}')
            raise

    def _setup(self, settings: PreparationConfig) -> None:
        self.diagnostics = self.session.diagnostics
        self.diagnostics['accounting_policy'] = 'numeric-capacity-and-coarse-json-document-admission-v2'
        self.diagnostics['json_heap_scope'] = '32*raw_input_bytes+64KiB document/file estimates; not exact serde heap or RSS'
        self._directory: Path | None = None
        self._lease: Any = None
        self._outputs: dict[str, BinaryIO] = {}
        self._paths: dict[str, Path] = {}
        self._offsets: dict[str, int] = {}
        self._mappings: dict[str, Any] = {}
        self._closed = False
        self.shapes: dict[str, tuple[int, ...]] = {}
        self._counts = {'train': 0, 'validation': 0}
        self._window = min(settings.chunk_target_mib * MIB,
                           max(4096, settings.memory_budget_mib * MIB // 8))

    @classmethod
    async def create_async(cls, path: Path, config: EncoderConfig, settings: PreparationConfig,
                           *, deadline: float, wall_deadline: float | None = None) -> Self:
        self = cls.__new__(cls)
        self._wall = _PreparationWall()
        with self._wall.phase('owner_session_setup'):
            self.config = encoder_config(asdict(config))
            work, wall = _deadlines(settings, deadline, wall_deadline)
            self.session = await NativePreparation.create_async(
                {'kind': 'offline', 'operation': 'prepare', 'dataset_path': str(path.resolve()),
                 'encoder': asdict(config)}, settings, deadline=work, wall_deadline=wall)
            self._setup(settings)
        try:
            while True:
                with self._wall.phase('next_batch_wait'):
                    batch = await self.session.anext_batch(0, fixed_consumer_layout(config))
                if batch is None:
                    break
                with self._wall.phase('proof_and_initialization'):
                    proof = self.session.proof()
                    if not self.shapes:
                        self.dataset = proof['dataset_manifest']
                        size = self._set_shapes(batch.features.shape[1])
                        self._lease = await self.session.areserve_staging(2 * self._window + 65536,
                                                                         2 * size + 32768)
                        self._open_writers()
                with self._wall.phase('array_reward_validation_labels'):
                    labels = fixed_targets(batch, global_ready=proof['structural_valid'])
                with self._wall.phase('bounded_npy_write'):
                    await self._aconsume(batch, labels)
                with self._wall.phase('batch_release'):
                    del labels, batch
            with self._wall.phase('seal'):
                self._seal()
                capture = getattr(self.session, 'adata_stage_snapshot', None)
                self.diagnostics['native_data_stage_snapshot'] = await capture() if callable(capture) else None
                await asyncio.sleep(0)
            self.diagnostics['native_prepared_inputs_wall'] = self._wall.finish(success=True)
        except BaseException as error:
            self.diagnostics['native_prepared_inputs_wall'] = self._wall.finish(success=False)
            try:
                await self.aabort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve original initialization failure
                error.add_note(f'native numeric cleanup failed: {cleanup_error}')
            raise
        return self

    def _seal(self) -> None:
        if not self.shapes or self._counts != {'train': self.shapes['x'][0],
                                               'validation': self.shapes['validation_x'][0]}:
            raise ContractError('native numeric counts differ from validated dataset')
        self._close_writers()
        self.diagnostics.update(sample_identity_sha256=self.session.proof()['sample_identity_sha256'],
                                numeric_window_bytes=self._window,
                                numeric_staging_bytes=sum(math.prod(shape) * 8 for shape in self.shapes.values()))

    def _buffers(self, batch: Any, labels: Any) -> Any:
        # fixed_targets has checked both terminal perspectives, including empty
        # episodes, before any zero-decision episode can be skipped here.
        for episode, split in enumerate(batch.split):
            start, stop = batch.episode_offsets[episode:episode + 2]
            if start == stop:
                continue
            prefix = '' if split == 0 else 'validation_'
            yield self._outputs[prefix + 'x'], memoryview(batch.features[start:stop]).cast('B')
            yield self._outputs[prefix + 'y'], memoryview(labels[start:stop]).cast('B')
            self._counts['train' if split == 0 else 'validation'] += int(stop - start)

    def _consume(self, batch: Any, labels: Any) -> None:
        for output, buffer in self._buffers(batch, labels):
            for offset in range(0, len(buffer), self._window):
                self.session.check()
                output.write(buffer[offset:offset + self._window])

    async def _aconsume(self, batch: Any, labels: Any) -> None:
        for output, buffer in self._buffers(batch, labels):
            for offset in range(0, len(buffer), self._window):
                self.session.check()
                output.write(buffer[offset:offset + self._window])
                await asyncio.sleep(0)

    def _set_shapes(self, width: int) -> int:
        train = integer(self.dataset['train_decisions'], 1)
        validation = integer(self.dataset['validation_decisions'], 1)
        self.shapes = {'x': (train, width), 'y': (train,),
                       'validation_x': (validation, width), 'validation_y': (validation,)}
        return sum(math.prod(shape) * 8 for shape in self.shapes.values())

    def _initialize(self, width: int) -> None:
        size = self._set_shapes(width)
        self._lease = self.session.reserve_staging(2 * self._window + 65536, 2 * size + 32768)
        self._open_writers()

    def _open_writers(self) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix='gf-native-numeric-'))
        for name, shape in self.shapes.items():
            path = self._directory / (name + '.npy')
            self._paths[name] = path
            output = path.open('xb')
            self._outputs[name] = output
            np.lib.format.write_array_header_1_0(output, {'descr': np.dtype('float64').str,
                                                         'fortran_order': False, 'shape': shape})
            self._offsets[name] = output.tell()

    def _borrow(self, name: str) -> Any:
        self.session.check()
        if name not in self._mappings:
            shape = self.shapes[name]
            rows, columns = shape if len(shape) == 2 else (shape[0], 1)
            value = self.session.export_staging(str(self._paths[name]), self._offsets[name], rows, columns)
            self._mappings[name] = value.reshape(shape)
        return self._mappings[name]

    @property
    def x(self) -> Any:
        return self._borrow('x')

    @property
    def y(self) -> Any:
        return self._borrow('y')

    @property
    def validation_x(self) -> Any:
        return self._borrow('validation_x')

    @property
    def validation_y(self) -> Any:
        return self._borrow('validation_y')

    def write_numeric(self, path: Path) -> None:
        self.session.check()
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in self._paths.items():
                with source.open('rb') as stream, archive.open(name + '.npy', 'w', force_zip64=True) as destination:
                    while block := stream.read(self._window):
                        self.session.check()
                        destination.write(block)
        self.session.check()

    async def awrite_numeric(self, path: Path) -> None:
        self.session.check()
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in self._paths.items():
                with source.open('rb') as stream, archive.open(name + '.npy', 'w', force_zip64=True) as destination:
                    while block := stream.read(self._window):
                        self.session.check()
                        destination.write(block)
                        await asyncio.sleep(0)
        self.session.check()
        await asyncio.sleep(0)

    def _close_writers(self) -> None:
        first: BaseException | None = None
        for output in self._outputs.values():
            if not output.closed:
                try:
                    output.flush()
                    os.fsync(output.fileno())
                    output.close()
                except BaseException as error:  # noqa: BLE001 - close every owned stream
                    first = first or error
                    try:
                        output.close()
                    except BaseException as close_error:  # noqa: BLE001 - retain secondary cleanup evidence
                        error.add_note(str(close_error))
        if first is not None:
            raise first

    def _cleanup(self) -> None:
        self._close_writers()
        self._mappings.clear()  # Export owners retain buffers and quota for surviving views.
        if self._directory is not None:
            shutil.rmtree(self._directory)
            self._directory = None
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def finalize(self) -> None:
        if self._closed:
            return
        try:
            self._cleanup()
            self.session.finalize()
        except BaseException as error:
            try:
                self.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve finalization failure
                error.add_note(f'native numeric cleanup failed: {cleanup_error}')
            raise
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        first: BaseException | None = None
        for close in (self._cleanup, self.session.abort):
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - release all owned resources before returning
                first = first or error
        self._closed = True
        if first is not None:
            raise first

    async def afinalize(self) -> None:
        if self._closed:
            return
        try:
            self._cleanup()
            await self.session.afinalize()
        except BaseException as error:
            try:
                await self.aabort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve finalization failure
                error.add_note(f'native numeric cleanup failed: {cleanup_error}')
            raise
        self._closed = True

    async def aabort(self) -> None:
        if self._closed:
            return
        first: BaseException | None = None
        try:
            self._cleanup()
        except BaseException as error:  # noqa: BLE001 - still close native resources
            first = error
        try:
            await self.session.aabort()
        except BaseException as error:  # noqa: BLE001 - release ownership despite output cleanup failure
            first = first or error
        self._closed = True
        if first is not None:
            raise first

    async def __aenter__(self) -> Self:
        self.session.check()
        return self

    async def __aexit__(self, kind: object, value: object, traceback: object) -> None:
        if not self._closed:
            try:
                if kind is None:
                    await self.afinalize()
                else:
                    await self.aabort()
            except BaseException as error:
                if not isinstance(value, BaseException):
                    raise
                value.add_note(f'native numeric cleanup failed: {error}')

    def __enter__(self) -> Self:
        self.session.check()
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        if not self._closed:
            if kind is None:
                self.finalize()
            else:
                try:
                    self.abort()
                except BaseException as error:
                    if isinstance(value, BaseException):
                        value.add_note(f'native numeric cleanup failed: {error}')
                    else:
                        raise


def build(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], output: Path,
          settings: PreparationConfig, *, deadline: float) -> Json:
    import uuid

    from gf_common.domain import canonical
    from gf_training.fixed_data import _publish_owned
    from gf_training.journal import atomic_json

    if output.exists(follow_symlinks=False):
        raise ContractError('frozen dataset output already exists')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    owner: NativePreparation | None = None
    lease: Any = None
    try:
        manifest, owner = summary({'kind': 'build', 'ppo_runs': [str(path.resolve()) for path in ppo_runs],
                                   'validation_runs': [str(path.resolve()) for path in validation_runs]},
                                  settings, deadline=deadline)
        size = 2 * len(canonical(manifest).encode()) + 4096
        lease = owner.reserve_staging(size, size)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(temporary, manifest)
        _publish_owned(owner, ((temporary, output),))
        return manifest
    except BaseException as error:
        if owner is not None:
            try:
                owner.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve publication/source failure
                error.add_note(f'native dataset cleanup failed: {cleanup_error}')
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_name(temporary.name + '.tmp').unlink(missing_ok=True)
        if lease is not None:
            lease.release()


def inspect(path: Path, settings: PreparationConfig, *, deadline: float) -> Json:
    manifest, owner = summary({'kind': 'offline', 'dataset_path': str(path.resolve())},
                              settings, deadline=deadline)
    with owner:
        return manifest
