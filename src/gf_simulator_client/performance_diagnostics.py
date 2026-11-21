"""Opt-in diagnostic I/O edge. Files are evidence only, never authority."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from gf_common.performance_metrics import PhaseMetrics

DIRECTORY_ENV = 'GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR'
OBSERVATION_ENV = 'GF_BOTS_PERFORMANCE_OBSERVATION_ID'
PROFILE_ENV = 'GF_BOTS_PERFORMANCE_PROFILE'
TIMING_ENV = 'GF_BOTS_PERFORMANCE_TIMING'
ROLES = ('host', 'native-runtime', 'legacy-coordinator', 'legacy-shared-broker',
    'legacy-gpu-worker', 'legacy-participant', 'native-supervisor', 'native-inference', 'writer')
PHASES = ('startup', 'collection', 'update', 'save', 'retirement', 'cold', 'evaluation', 'idle')


def owner_identity(role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError('unsupported diagnostic owner role')
    # proc stat comm can contain spaces/parentheses; fields start after final ).
    return process_identity(os.getpid(), role)


def process_identity(pid: int, role: str) -> dict[str, Any]:
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'role': role}


def clock_identity() -> dict[str, str]:
    return {'name': 'CLOCK_MONOTONIC', 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


class Diagnostics:
    def __init__(self, directory: Path, observation_id: str, *, role: str = 'host', profile_recipe: str = 'off') -> None:
        if not directory.is_absolute() or not observation_id or '\0' in observation_id:
            raise ValueError('absolute directory and nonempty UTF-8 observation ID required')
        observation_id.encode('utf-8')  # Refuse surrogate-escaped invalid environment text, without normalization.
        self.directory, self.observation_id = directory, observation_id
        self.owner, self.clock = owner_identity(role), clock_identity()
        self.profile_recipe = profile_recipe
        self._phase_scope: dict[str, Any] | None = None
        self.current_scope: dict[str, Any] | None = None
        self.errors: list[str] = []
        self._retained_fds: list[int] = []
        self._fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        if os.fstat(self._fd).st_uid != os.getuid():
            self.close()
            raise ValueError('diagnostic directory must belong to current user')

    @staticmethod
    def scope(*, run_id: str | None = None, operation_id: str | None = None,
            invocation_id: str | None = None, batch_id: str | None = None,
            phase: str) -> dict[str, Any]:
        if phase not in PHASES:
            raise ValueError('unsupported diagnostic phase')
        return {'run_id': run_id, 'operation_id': operation_id, 'invocation_id': invocation_id,
            'batch_id': batch_id if phase == 'collection' else None, 'phase': phase}

    def envelope(self, kind: str, *, role: str | None = None) -> dict[str, Any]:
        return {'schema_version': 1, 'kind': kind, 'observation_id': self.observation_id,
            'owner': self.owner if role is None else self.owner | {'role': role}, 'clock': self.clock}

    def write(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Exclusive original FD, no replaceable pathname reopened for hashing."""
        if self._retained_fds:
            return None  # One uncertain original FD disables further files; no growing leak.
        fd = None
        try:
            name = f'{record["kind"]}-{os.getpid()}-{uuid.uuid4().hex}.json'
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600, dir_fd=self._fd)
            original = os.fstat(fd)
            data = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
            with os.fdopen(fd, 'wb', closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(fd)
            published = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
            if (not stat.S_ISREG(published.st_mode)
                    or (published.st_dev, published.st_ino) != (original.st_dev, original.st_ino)):
                raise OSError('diagnostic output pathname replaced')
            return {'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        except Exception as exc:  # noqa: BLE001 - diagnostic loss grants no policy/cleanup authority.
            error = 'diagnostic_write:' + type(exc).__name__
            if error not in self.errors:
                self.errors.append(error)
            return None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    self._retained_fds.append(fd)
                    self.error("diagnostic_close_failed")

    def phase(self, event: str, scope: dict[str, Any], *, monotonic_ns: int | None = None,
            links: dict[str, Any] | None = None, role: str | None = None) -> dict[str, Any] | None:
        if event not in ('phase_start', 'phase_end', 'host_reaped', 'prepared_written', 'completion_written'):
            raise ValueError('unsupported diagnostic phase event')
        return self.write(self.envelope('rust-player-phase-event-v1', role=role) | {
            'scope': dict(scope), 'monotonic_ns': time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
            'event': event, 'links': dict.fromkeys(('operation_sha256', 'prepared_sha256',
                'exit_sha256', 'completion_sha256')) | (links or {})})

    def metrics(self, metrics: PhaseMetrics | dict[str, Any], *, role: str | None = None,
            complete: bool = True) -> dict[str, Any] | None:
        row = metrics.snapshot(end_ns=time.monotonic_ns(), complete=complete) if isinstance(metrics, PhaseMetrics) else metrics
        if self.errors:
            row = row | {'complete': False, 'errors': row['errors'] + self.errors}
        return self.write(self.envelope('rust-player-phase-metrics-v1', role=role) | row)

    def error(self, error: str) -> None:
        if error not in self.errors:
            self.errors.append(error)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def diagnostics_from_environment(env: Mapping[str, str], *, role: str = 'host') -> Diagnostics | None:
    directory, observation = env.get(DIRECTORY_ENV), env.get(OBSERVATION_ENV)
    recipe = env.get(PROFILE_ENV, 'off')
    if recipe not in ('off', 'forward-window-v1'):
        raise ValueError('unsupported performance profile recipe')
    if directory is None and observation is None and recipe == 'off':
        return None
    if not directory or not observation:
        raise ValueError('performance diagnostics require directory and observation ID pair')
    return Diagnostics(Path(directory), observation, role=role, profile_recipe=recipe)


@cache
def host_diagnostics() -> Diagnostics | None:
    # Consume profile control at the host edge. Explicit callback carries it to workers.
    result = diagnostics_from_environment(os.environ)
    os.environ.pop(PROFILE_ENV, None)
    return result


def external_environment(env: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in env.items()
        if key not in (DIRECTORY_ENV, OBSERVATION_ENV, PROFILE_ENV, TIMING_ENV)}


def record_accepted_reply(metrics: PhaseMetrics, timings: dict[str, Any], *, end_ns: int) -> None:
    """Only the parent calls this after its complete, existing reply validation."""
    metrics.increment('inference_successes')
    ready = timings.get('shared_numeric_ready_ns')
    if type(ready) is not int or ready < 0 or ready > end_ns:
        metrics.error('missing_or_invalid_numeric_ready_clock')
        return
    metrics.interval('inference_roundtrip', ready, end_ns)


def transition_phase(diagnostics: Diagnostics | None, phase: str, **identity: Any) -> None:
    if diagnostics is None:
        return
    previous = getattr(diagnostics, '_phase_scope', None)
    if previous is not None:
        diagnostics.phase('phase_end', previous)
    scope = diagnostics.scope(**({key: value for key, value in (previous or {}).items()
        if key not in ('phase', 'batch_id')} | identity), phase=phase)
    diagnostics.phase('phase_start', scope)
    diagnostics.current_scope = scope
    diagnostics._phase_scope = scope


def finish_phase(diagnostics: Diagnostics | None) -> None:
    if diagnostics is not None:
        scope = getattr(diagnostics, '_phase_scope', None)
        if scope is not None:
            diagnostics.phase('phase_end', scope)
            diagnostics._phase_scope = None
            diagnostics.current_scope = None
