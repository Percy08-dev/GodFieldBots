"""Owned controls-only native diagnostics; not a structured reader capability.

No Torch, Journal5 projection, NPZ verification or source qualification. The
full reader remains unavailable until its separately reviewed implementation.
"""
from __future__ import annotations

import asyncio
import json
import stat
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Self

from gf_common.action_contract import parse_action_contract
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.preprocessing import PreparationConfig, preparation_config
from gf_models.structured_config import recorded_model_config_v2
from gf_models.structured_sealed import checksum, validate_recorded_semantics
from gf_training.lineage import run_lock
from gf_training.native_backend import binary_identity, load_native
from gf_training.native_execution import NativeExecution
from gf_training.preparation import ResourceLimit


def _invoke(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except MemoryError as error:
        raise ResourceLimit(str(error)) from error
    except ValueError as error:
        raise ContractError(str(error)) from error


class NativeStructuredControls:
    """One locked diagnostic owner and one joined native FFI worker.

    The admitted summary lease stays in the native owner through recheck/close;
    retained_python_bytes means ADDITIONAL caller ownership, not that summary.
    No control result or purpose flag authorizes collection, export or fitting.
    """
    def __init__(self, directory: Path, config: PreparationConfig, *, deadline: float,
                 native: ModuleType | None = None, retained_python_bytes: int = 0) -> None:
        started = time.monotonic()
        self.config = preparation_config(asdict(config))
        if self.config.backend != 'rust' or self.config.reader != 'structured-v1':
            raise ContractError('explicit native structured diagnostic settings required')
        integer(retained_python_bytes, 0)
        self.directory = Path(directory).absolute()
        self.deadline = min(deadline, started + self.config.wall_seconds - self.config.cleanup_seconds)
        self.retained_python_bytes = retained_python_bytes
        self.native = native
        self._session: Any = None
        self._executor: NativeExecution | None = None
        self._locks = ExitStack()
        self._encoded: bytes | None = None
        self._entered = self._closed = False
        self._closing_requested = False
        self._operation = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self.identity: Json | None = None
        self._base_bytes = 65536

    def _check(self) -> None:
        if self._closed or self._closing_requested:
            raise ContractError('native structured diagnostics are closed')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('native structured diagnostic deadline')

    def _prepare(self) -> bytes:
        self._check()
        if self._entered:
            raise ContractError('native structured diagnostic owner already entered')
        for parent in (self.directory, *self.directory.parents):
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ContractError('native structured diagnostic root must be nonsymlink directories')
        self.native = self.native or load_native()
        if not callable(getattr(self.native, 'StructuredParentControls', None)):
            raise ContractError('loaded native binary lacks controls-only diagnostic class')
        self.identity = binary_identity(self.native)
        self._base_bytes += 32 * len(canonical(self.identity).encode())
        available = self.config.memory_budget_mib * 1024**2 - self._base_bytes - self.retained_python_bytes
        if available <= 0:
            raise ResourceLimit('native structured caller/control ownership exceeds budget')
        encoded = canonical({'memory_bytes': available, 'max_json_bytes': self.config.max_source_file_mib * 1024**2}).encode()
        self._locks.enter_context(run_lock(self.directory))
        cancel = self.native.CancelHandle(self.deadline)
        self._executor = NativeExecution(cancel, thread_name_prefix='gf-structured-controls')
        self._entered = True
        return encoded

    def _construct(self, encoded: bytes) -> None:
        assert self.native is not None
        assert self._executor is not None
        # Publish the returned owner before the submitted function completes,
        # including when Python cancellation wins the awaiting Future's race.
        self._session = self.native.StructuredParentControls(str(self.directory), encoded, self.deadline,
                                                              self._executor._cancel_handle)

    @staticmethod
    def _validate(encoded: bytes) -> None:
        value = json.loads(encoded)
        closed(value, {'schema_version', 'kind', 'validation_scope', 'npz_validated', 'journal5_validated', 'qualified',
            'parent_qualification', 'directory', 'state', 'clocks', 'model', 'execution', 'semantics', 'actions_sha256',
            'action_contract', 'purpose', 'synthetic_capacity', 'identity_sha256', 'generations', 'control_fingerprint'})
        if (type(value['schema_version']) is not int or value['schema_version'] != 2 or value['kind'] != 'structured-parent-controls'
                or value['validation_scope'] != 'metadata-maps-filehashes-only'
                or any(value[key] is not False for key in ('npz_validated', 'journal5_validated', 'qualified', 'synthetic_capacity'))
                or value['parent_qualification'] != 'not_performed' or value['purpose'] != 'component'):
            raise ContractError('controls-only result cannot claim numeric/Source authority')
        parse_action_contract(value['action_contract'])
        model = recorded_model_config_v2(value['model'], value['semantics']['token_signature'])
        validate_recorded_semantics(value['semantics'], model)
        for key in ('control_fingerprint', 'actions_sha256', 'identity_sha256'):
            checksum(value[key])
        if not isinstance(value['generations'], list):
            raise ContractError('bounded generation summaries required')
        integer(len(value['generations']), 1, 1001)

    @property
    def result(self) -> Json:
        self._check()
        if self._encoded is None:
            raise ContractError('native diagnostic summary not delivered')
        return json.loads(self._encoded)

    def __enter__(self) -> Self:
        try:
            encoded = self._prepare()
            assert self._executor is not None
            _invoke(self._executor.call, self._construct, encoded)
            self._encoded = _invoke(self._executor.call, self._session.summary)
            if type(self._encoded) is not bytes:
                raise ContractError('native diagnostic bytes required')
            self._validate(self._encoded)
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - initiating failure survives joined cleanup
                error.add_note('structured diagnostic cleanup failed: ' + type(cleanup).__name__)
            raise

    async def __aenter__(self) -> Self:
        try:
            async with self._operation:
                encoded = self._prepare()
                assert self._executor is not None
                await self._executor.acall(self._construct, encoded)
                self._encoded = await self._executor.acall(self._session.summary)
                if type(self._encoded) is not bytes:
                    raise ContractError('native diagnostic bytes required')
                self._validate(self._encoded)
                self._check()
            return self
        except BaseException as error:
            try:
                await self.aclose()
            except BaseException as cleanup:  # noqa: BLE001 - initiating cancellation survives joined cleanup
                error.add_note('structured diagnostic cleanup failed: ' + type(cleanup).__name__)
            raise

    def recheck(self, retained_python_bytes: int = 0) -> None:
        self._check()
        integer(retained_python_bytes, 0)
        assert self._executor is not None
        _invoke(self._executor.call, self._session.recheck, retained_python_bytes)

    async def arecheck(self, retained_python_bytes: int = 0) -> None:
        async with self._operation:
            self._check()
            integer(retained_python_bytes, 0)
            assert self._executor is not None
            try:
                await self._executor.acall(self._session.recheck, retained_python_bytes)
            except (ValueError, MemoryError) as error:
                raise (ResourceLimit(str(error)) if isinstance(error, MemoryError) else ContractError(str(error))) from error
            self._check()

    def close(self) -> None:
        if self._closed:
            return
        if self._operation.locked() or self._executor is not None and self._executor._active is not None:
            raise ContractError('active asynchronous diagnostic requires awaited aclose; ownership retained')
        self._closing_requested = True
        try:
            if self._executor is not None and self._session is not None:
                _invoke(self._executor.call, self._session.close)
        finally:
            try:
                if self._executor is not None:
                    self._executor.close()
            finally:
                self._encoded = None
                self._locks.close()
                self._closed = True

    async def _aclose_owned(self) -> None:
        if self._operation.locked() and self._executor is not None:
            self._executor.cancel()
        async with self._operation:
            try:
                if self._executor is not None and self._session is not None:
                    await self._executor.acall(self._session.close)
            finally:
                try:
                    if self._executor is not None:
                        await self._executor.aclose()
                finally:
                    self._encoded = None
                    self._locks.close()
                    self._closed = True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closing_requested = True
        task = self._close_task
        if task is None or task.done() and not self._closed:
            task = self._close_task = asyncio.create_task(self._aclose_owned())
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                if self._executor is not None:
                    self._executor.cancel()
            except BaseException:  # noqa: BLE001 - retrieve the exact joined cleanup result
                break
        try:
            task.result()
        except BaseException as error:
            if cancellation is not None:
                cancellation.add_note('structured close ended during cancellation: ' + type(error).__name__)
                raise cancellation
            raise
        if cancellation is not None:
            raise cancellation

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException as error:
            if not isinstance(value, BaseException):
                raise
            value.add_note('structured diagnostic final recheck failed: ' + type(error).__name__)

    async def __aexit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            await self.aclose()
        except BaseException as error:
            if not isinstance(value, BaseException):
                raise
            value.add_note('structured diagnostic final recheck failed: ' + type(error).__name__)
