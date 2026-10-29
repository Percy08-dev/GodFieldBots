"""One owned native FFI thread, with cancellable waits and mandatory joins.

Locks, authorization, NumPy learning and publication remain on the caller's
thread. Submit only native operations here. The independently callable cancel
handle must exist before the first submission, including session construction.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Self, TypeVar

from gf_common.domain import ContractError

T = TypeVar('T')


class NativeExecution:
    def __init__(self, cancel_handle: Any, *, thread_name_prefix: str = 'gf-native-ffi') -> None:
        if not callable(getattr(cancel_handle, 'cancel', None)):
            raise ContractError('independent native cancel handle required')
        self._cancel_handle = cancel_handle
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._active: Future[Any] | None = None
        self._closed = False

    def cancel(self) -> None:
        if not self._closed:
            self._cancel_handle.cancel()

    def _submit(self, function: Callable[..., T], *arguments: Any) -> Future[T]:
        if self._closed:
            raise ContractError('native execution is closed')
        if self._active is not None:
            raise ContractError('native execution already has an active operation')
        future = self._executor.submit(function, *arguments)
        self._active = future
        return future

    @staticmethod
    def _note(original: BaseException, label: str, error: BaseException) -> None:
        if error is not original:
            original.add_note(f'{label}: {type(error).__name__}: {error}')

    def _cancel_for(self, original: BaseException) -> None:
        try:
            self.cancel()
        except BaseException as error:  # noqa: BLE001 - join even if cancellation itself fails
            self._note(original, 'native cancellation request failed', error)

    def _join_sync(self, future: Future[Any], original: BaseException) -> None:
        while not future.done():
            try:
                future.result(timeout=.05)
            except KeyboardInterrupt:
                self._cancel_for(original)
            except TimeoutError:
                continue
            except BaseException:  # noqa: BLE001 - worker completion, original cancellation wins
                break  # The submitted operation has finished with an error.
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - original cancellation wins
            self._note(original, 'native work ended during cancellation', error)

    async def _join_async(self, completed: asyncio.Future[Any], future: Future[Any],
                          original: BaseException) -> None:
        while not completed.done():
            try:
                await asyncio.shield(completed)
            except asyncio.CancelledError:
                self._cancel_for(original)
            except BaseException:  # noqa: BLE001 - retrieve worker error after joining
                break
        try:
            future.result()
        except BaseException as error:  # noqa: BLE001 - retrieve failure, preserve cancellation
            self._note(original, 'native work ended during cancellation', error)
        finally:
            if completed.done() and not completed.cancelled():
                completed.exception()  # Retrieve an error even if cancellation won before await.

    def call(self, function: Callable[..., T], *arguments: Any) -> T:
        """Wait in short intervals so the main thread can serve KeyboardInterrupt."""
        future = self._submit(function, *arguments)
        try:
            while True:
                try:
                    return future.result(timeout=.05)
                except TimeoutError:
                    if future.done():
                        # A native deadline exception is also TimeoutError. Do
                        # not confuse it with a pending Future's polling timeout.
                        return future.result()
        except KeyboardInterrupt as cancellation:
            self._cancel_for(cancellation)
            self._join_sync(future, cancellation)
            raise
        finally:
            self._active = None

    async def acall(self, function: Callable[..., T], *arguments: Any) -> T:
        future = self._submit(function, *arguments)
        completed = asyncio.wrap_future(future)
        try:
            await asyncio.shield(completed)
            await asyncio.sleep(0)  # Deliver a cancellation queued beside native success.
            return future.result()
        except asyncio.CancelledError as cancellation:
            self._cancel_for(cancellation)
            await self._join_async(completed, future, cancellation)
            # Even when native work completed concurrently, cancellation remains
            # the outcome; its result must never authorize a subsequent update.
            raise
        finally:
            self._active = None

    def _shutdown(self) -> None:
        # Every submitted function is already joined. Only the idle executor
        # thread remains; repeat shutdown if a further SIGINT interrupts its join.
        cancellation: KeyboardInterrupt | None = None
        while True:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
                break
            except KeyboardInterrupt as error:
                cancellation = cancellation or error
                self._cancel_for(cancellation)
        self._closed = True
        self._cancel_handle = None
        if cancellation is not None:
            raise cancellation

    def close(self) -> None:
        if self._closed:
            return
        if self._active is not None:
            error = ContractError('native execution closed during an active operation')
            self._cancel_for(error)
            self._join_sync(self._active, error)
            self._active = None
        self._shutdown()

    async def aclose(self) -> None:
        if self._closed:
            return
        cancellation: asyncio.CancelledError | None = None
        failure: BaseException | None = None
        if self._active is not None:
            future = self._active
            completed = asyncio.wrap_future(future)
            try:
                await asyncio.shield(completed)
            except asyncio.CancelledError as error:
                cancellation = error
                self._cancel_for(error)
                await self._join_async(completed, future, error)
            except BaseException as error:  # noqa: BLE001 - rethrow after owned shutdown
                failure = error
            self._active = None
        self._shutdown()
        if cancellation is not None:
            raise cancellation
        if failure is not None:
            raise failure

    def __enter__(self) -> Self:
        if self._closed:
            raise ContractError('native execution is closed')
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException as error:
            if isinstance(value, (KeyboardInterrupt, asyncio.CancelledError)):
                self._note(value, 'native executor cleanup failed during cancellation', error)
                raise value
            raise

    async def __aenter__(self) -> Self:
        if self._closed:
            raise ContractError('native execution is closed')
        return self

    async def __aexit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            await self.aclose()
        except BaseException as error:
            if isinstance(value, (KeyboardInterrupt, asyncio.CancelledError)):
                self._note(value, 'native executor cleanup failed during cancellation', error)
                raise value
            raise
