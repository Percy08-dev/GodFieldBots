"""Separate, bounded worker-thread profiler population; never primary timing."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gf_simulator_client.performance_diagnostics import clock_identity, owner_identity


class ProfileStatus:
    """Bounded shared BEGIN/END/result holder. The first collection is terminal."""
    def __init__(self, block: Any, lock: Any, terminal: Any = None) -> None:
        self.block, self.lock = block, lock
        self.terminal = SimpleNamespace(value=0) if terminal is None else terminal

    @classmethod
    def local(cls) -> ProfileStatus:
        return cls(bytearray(16384), threading.RLock())

    @classmethod
    def shared(cls, context: Any) -> ProfileStatus:
        return cls(context.RawArray('B', 16384), context.RLock(), context.RawValue('b', 0))

    def _read(self) -> dict[str, Any]:
        data = bytes(self.block).split(b'\0', 1)[0]
        return json.loads(data) if data else {'state': 'waiting', 'scope': None, 'result': None}

    def _write(self, data: dict[str, Any]) -> None:
        encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
        if len(encoded) >= len(self.block):
            raise ValueError('bounded profile status exhausted')
        self.block[:] = encoded + bytes(len(self.block) - len(encoded))

    def read(self) -> dict[str, Any]:
        if not self.lock.acquire(False):
            return {'state': 'ended' if self.terminal.value else 'unavailable', 'scope': None, 'result': None}
        try:
            data = self._read()
            if self.terminal.value and data['state'] != 'done':
                data['state'] = 'ended'
            return data
        except Exception:  # noqa: BLE001 - optional diagnostic loss must not affect policy.
            return {'state': 'ended' if self.terminal.value else 'unavailable', 'scope': None, 'result': None}
        finally:
            self.lock.release()

    def begin(self, scope: dict[str, Any]) -> None:
        if self.terminal.value:
            return
        if not self.lock.acquire(False):
            self.terminal.value = 1
            return
        try:
            data = self._read()
            if data['state'] == 'waiting':
                self._write({'state': 'active', 'scope': dict(scope), 'result': None})
        except Exception:  # noqa: BLE001 - optional diagnostic loss must not affect policy.
            self.terminal.value = 1
        finally:
            self.lock.release()

    def end(self) -> None:
        # Coordinator-only terminal store does not depend on the worker's lock.
        self.terminal.value = 1
        if not self.lock.acquire(False):
            return
        try:
            data = self._read()
            if data['state'] == 'active':
                data['state'] = 'ended'
                if data['result'] is not None and not data['result']['complete']:
                    data['result']['errors'] = list(dict.fromkeys(data['result']['errors'] + ['scope_ended_before_post3']))
                self._write(data)
        except Exception:  # noqa: BLE001 - optional diagnostic loss must not affect policy.
            self.terminal.value = 1
        finally:
            self.lock.release()

    def publish(self, result: dict[str, Any], *, done: bool = False) -> None:
        if not self.lock.acquire(False):
            raise ValueError('diagnostic profile status unavailable')
        try:
            data = self._read()
            if self.terminal.value and data['state'] != 'done' and result['complete']:
                result = result | {'complete': False, 'errors': result['errors'] + ['scope_ended_before_post3']}
            data['result'] = result
            if done and not self.terminal.value:
                data['state'] = 'done'
            self._write(data)
        finally:
            self.lock.release()


class ForwardProfile:
    def __init__(self, directory: Path, observation_id: str, status: ProfileStatus, *, role: str,
            profiler_factory: Any = None, torch_version: str | None = None) -> None:
        self.directory, self.observation_id, self.status, self.role = directory, observation_id, status, role
        self.factory, self.torch_version = profiler_factory, torch_version
        self.profiler: Any = None
        self.record: dict[str, Any] | None = None
        self.finished = False
        self.invalid = False
        self._trace_fd: int | None = None
        self._directory_fd: int | None = None
        self._execution_errors: tuple[type[BaseException], ...] = ()

    def _fail(self, reason: str) -> None:
        self.invalid = True
        if self.record is not None:
            if reason not in self.record['errors']:
                self.record['errors'].append(reason)
            self.record['complete'] = False
            try:
                self.status.publish(dict(self.record))
            except Exception:  # noqa: BLE001 - optional diagnostic loss must not affect policy.
                self.invalid = True  # Lost status remains incomplete, never a forward failure.

    def _cleanup(self) -> None:
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception as error:
                if isinstance(error, self._execution_errors):
                    self._fail('accelerator_execution_error')
                    raise  # Preserve the normal owner's genuine accelerator failure/poison path.
                self._fail('profiler_stop_failed')
            self.profiler = None
        self.finished = True

    def after_forward(self, *, scored_rows: int, boundary: str) -> None:
        """Called on the real successful forward thread, never with policy data."""
        if self.finished:
            return
        try:
            state = self.status.read()
            if state['state'] == 'waiting':
                return
            if state['state'] == 'unavailable':
                self._fail('profile_status_unavailable')
                if self.record is not None and threading.get_native_id() == self.record['worker_tid']:
                    self._cleanup()
                return
            tid = threading.get_native_id()
            if self.record is not None and tid != self.record['worker_tid']:
                self._fail('execution_thread_changed')
                return  # Thread-local profiler must not be stopped here.
            if state['state'] == 'ended' or self.invalid:
                self._fail('scope_ended_before_post3' if state['state'] == 'ended' else 'invalid_profile_window')
                self._cleanup()
                return
            if boundary not in ('legacy-post-transport-forward', 'native-post-row-forward') or scored_rows <= 0:
                self._fail('invalid_forward_boundary')
                return
            if self.record is None:
                if self.factory is None:
                    import torch
                    self.torch_version = torch.__version__
                    self._execution_errors = (torch.cuda.AcceleratorError, torch.cuda.CudaError, torch.cuda.OutOfMemoryError)
                    self.factory = lambda: torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
                self.record = {'schema_version': 1, 'kind': 'rust-player-forward-profile-v1',
                    'observation_id': self.observation_id, 'owner': owner_identity(self.role),
                    'clock': clock_identity(), 'scope': state['scope'], 'recipe': 'forward-window-v1',
                    'enabled': True, 'boundary': boundary, 'worker_tid': tid,
                    'start_after_forward': 1, 'stop_after_forward': 3, 'callbacks_seen': 0,
                    'scored_rows': [None, None, None], 'start_monotonic_ns': None,
                    'end_monotonic_ns': None, 'complete': False, 'errors': [],
                    'trace_path': None, 'trace_sha256': None, 'trace_bytes': None,
                    'trace_clock': 'torch-profiler-native', 'torch_version': self.torch_version}
            if boundary != self.record['boundary']:
                self._fail('forward_boundary_changed')
                self._cleanup()
                return
            ordinal = self.record['callbacks_seen'] + 1
            self.record['callbacks_seen'] = ordinal
            self.record['scored_rows'][ordinal - 1] = scored_rows
            if ordinal == 1:
                self.profiler = self.factory()
                self.profiler.start()
                self.record['start_monotonic_ns'] = time.monotonic_ns()
            elif ordinal == 3:
                self.profiler.stop()
                self.record['end_monotonic_ns'] = time.monotonic_ns()
                self._export()
                self.profiler = None
                self.record['complete'] = True
                self.finished = True
            self.status.publish(dict(self.record), done=self.finished)
        except Exception as exc:
            if isinstance(exc, self._execution_errors):
                self._fail('accelerator_execution_error')
                raise  # Actual execution errors retain existing runtime failure authority.
            self._fail('profile_callback:' + type(exc).__name__)
            # Cleanup is on this proven worker only.
            if self.record is not None and threading.get_native_id() == self.record['worker_tid']:
                self._cleanup()

    def _export(self) -> None:
        name = f'forward-{os.getpid()}-{uuid.uuid4().hex}.trace.json'
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self._directory_fd = directory_fd
        fd = None
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory_fd)
            self._trace_fd = fd
            original = os.fstat(fd)
            # Export retains the ORIGINAL open description even if the public name changes.
            self.profiler.export_chrome_trace(f'/proc/self/fd/{fd}')
            os.fsync(fd)
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (not stat.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)):
                raise OSError('profile trace pathname replaced')
            os.lseek(fd, 0, os.SEEK_SET)
            hasher, size = hashlib.sha256(), 0
            while data := os.read(fd, 1024 * 1024):
                hasher.update(data)
                size += len(data)
            final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(final.st_mode) or (final.st_dev, final.st_ino) != (original.st_dev, original.st_ino):
                raise OSError('profile trace pathname replaced during hash')
            assert self.record is not None
            self.record.update(trace_path=name, trace_sha256=hasher.hexdigest(), trace_bytes=size)
        finally:
            close_error = None
            if self._trace_fd is not None:
                try:
                    os.close(self._trace_fd)
                    self._trace_fd = None
                except OSError as error:
                    close_error = error  # Original descriptor remains reachable on callback.
            if self._directory_fd is not None:
                try:
                    os.close(self._directory_fd)
                    self._directory_fd = None
                except OSError as error:
                    close_error = close_error or error
            if close_error is not None:
                raise close_error


def missing_profile_result(*, observation_id: str, owner: dict[str, Any], clock: dict[str, Any],
        scope: dict[str, Any], boundary: str, torch_version: str) -> dict[str, Any]:
    """No callbacks is explicitly incomplete; no worker TID/window/trace is invented."""
    return {'schema_version': 1, 'kind': 'rust-player-forward-profile-v1',
        'observation_id': observation_id, 'owner': owner, 'clock': clock, 'scope': scope,
        'recipe': 'forward-window-v1', 'enabled': True, 'boundary': boundary,
        'worker_tid': None, 'start_after_forward': 1, 'stop_after_forward': 3,
        'callbacks_seen': 0, 'scored_rows': [None, None, None], 'start_monotonic_ns': None,
        'end_monotonic_ns': None, 'complete': False, 'errors': ['scope_ended_before_post3'],
        'trace_path': None, 'trace_sha256': None, 'trace_bytes': None,
        'trace_clock': 'torch-profiler-native', 'torch_version': torch_version}
