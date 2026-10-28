"""Standalone Linux observer; no application or numerical imports.

Run with the project interpreter, never ``uv --active``::

  env -u VIRTUAL_ENV .venv/bin/python tools/benchmark_performance.py \
    --output /tmp/measurement-unique --seconds 7200 -- \
    .venv/bin/python tools/benchmark_workload.py --help

The output and scratch directories must not exist. They are never deleted by
this tool. Only its own scratch tree is scanned, never immutable source trees.
Use a fresh monitor process for every trial (subreaper ownership is process-wide).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HZ = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')
COUNTERS = ('user_seconds', 'system_seconds', 'minor_faults', 'major_faults',
            'rchar', 'wchar', 'read_bytes', 'write_bytes', 'cancelled_write_bytes')
LIMITATIONS = [
    '100ms samples can miss short processes and memory peaks; PSS/scratch sampled at 1s.',
    'Summed simultaneous RSS can count shared pages more than once; PSS may be unavailable.',
    'Final wait4 CPU/fault totals include waited descendants once; samples are not added.',
    'Process storage I/O is OS accounting, not physical device traffic; WSL host I/O is unseen.',
    'Memory, scratch and deadline observations are soft limits, not OS resource quotas.',
    'Exit rusage has no byte I/O counters; short/exiting process I/O can be missed.',
    'Observer process and optional device sampler are excluded from workload accounting.',
]


def proc_stat(raw: str) -> dict[str, Any]:
    fields = raw[raw.rindex(')') + 2:].split()
    return {'pid': int(raw.split(' ', 1)[0]), 'ppid': int(fields[1]),
            'state': fields[0], 'start': int(fields[19]),
            'user_seconds': int(fields[11]) / HZ,
            'system_seconds': int(fields[12]) / HZ,
            'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
            'rss_bytes': int(fields[21]) * PAGE}


def key_values(path: Path, *, kib: bool = False) -> tuple[dict[str, int] | None, str | None]:
    try:
        result = {}
        for line in path.read_text().splitlines():
            key, value = line.split(':', 1)
            if ' ' in key:  # smaps_rollup begins with a memory-map header.
                continue
            result[key] = int(value.split()[0]) * (1024 if kib else 1)
        return result, None
    except (OSError, ValueError, IndexError) as error:
        return None, f'{type(error).__name__}: {error}'


def process_table() -> dict[int, dict[str, Any]]:
    table = {}
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            try:
                item = proc_stat((path / 'stat').read_text())
                table[item['pid']] = item
            except (OSError, ValueError, IndexError):
                pass  # An unowned process may disappear during enumeration.
    return table


def discover_owned(table: dict[int, dict[str, Any]], owned: dict[int, int],
                   baseline: set[tuple[int, int]], observer_pid: int) -> dict[int, dict[str, Any]]:
    """Authorize current identities; recycled PIDs never inherit old ownership."""
    changed = True
    while changed:
        changed = False
        for pid, item in table.items():
            identity = (pid, item['start'])
            parent = table.get(item['ppid'])
            descendant = parent is not None and owned.get(item['ppid']) == parent['start']
            adopted = item['ppid'] == observer_pid and identity not in baseline
            if owned.get(pid) != item['start'] and (descendant or adopted):
                owned[pid] = item['start']
                changed = True
    return {pid: item for pid, item in table.items() if owned.get(pid) == item['start']}


def directory_bytes(path: Path) -> dict[str, Any]:
    logical = allocated = 0
    try:
        for base, dirs, files in os.walk(path, followlinks=False):
            # lstat avoids following a workload-created link into a source tree.
            for name in [*dirs, *files]:
                stat = (Path(base) / name).lstat()
                logical += stat.st_size
                allocated += stat.st_blocks * 512
        return {'logical_bytes': logical, 'allocated_bytes': allocated, 'unavailable': None}
    except OSError as error:
        return {'logical_bytes': None, 'allocated_bytes': None, 'unavailable': str(error)}


class BoundedWriter:
    def __init__(self, path: Path, limit: int) -> None:
        self.file = path.open('xb')
        self.limit = limit
        self.written = 0
        self.discarded = 0

    def write(self, data: bytes, *, whole: bool = False) -> None:
        allowed = max(0, self.limit - self.written)
        kept = data if len(data) <= allowed else (b'' if whole else data[:allowed])
        self.file.write(kept)
        self.file.flush()
        self.written += len(kept)
        self.discarded += len(data) - len(kept)

    def close(self) -> dict[str, int]:
        self.file.close()
        return {'written_bytes': self.written, 'discarded_bytes': self.discarded}


def subreaper(value: int | None = None) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER')
    if value is not None and libc.prctl(36, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    return previous.value


def gpu_sample() -> dict[str, Any]:
    """Device scope only, independently owned and bounded; never imports Torch."""
    try:
        with subprocess.Popen(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True,
        ) as child:
            assert child.stdout is not None
            os.set_blocking(child.stdout.fileno(), False)
            raw = bytearray()
            deadline = time.monotonic() + .5
            try:
                while time.monotonic() < deadline and len(raw) <= 65536:
                    try:
                        block = os.read(child.stdout.fileno(), min(4096, 65537 - len(raw)))
                    except BlockingIOError:
                        time.sleep(.005)
                        continue
                    if not block:
                        break
                    raw.extend(block)
                else:
                    raise TimeoutError('nvidia-smi timeout/oversize')
                child.wait(timeout=max(.001, deadline - time.monotonic()))
            except (TimeoutError, subprocess.TimeoutExpired):
                child.kill()
                child.wait()
                return {'values': None, 'unavailable': 'nvidia-smi timeout/oversize'}
        if child.returncode or len(raw) > 65536:
            return {'values': None, 'unavailable': 'nvidia-smi failed/oversize'}
        rows = [[int(field.strip()) for field in line.split(',')]
                for line in raw.decode().splitlines()]
        return {'values': rows, 'unavailable': None, 'scope': 'device, not workload attribution'}
    except (OSError, ValueError, UnicodeError) as error:
        return {'values': None, 'unavailable': str(error)}


def observe(command: list[str], output: Path, *, seconds: float = 7200,
            sampling: bool = True, log_bytes: int = 1048576,
            timeline_bytes: int = 16 * 1048576, device_gpu: bool = False) -> dict[str, Any]:
    if (not command or not math.isfinite(seconds) or seconds <= 0
            or log_bytes <= 0 or timeline_bytes <= 0):
        raise ValueError('command and finite positive bounds required')
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'scratch'
    scratch.mkdir()
    logs = {name: BoundedWriter(output / f'{name}.log', log_bytes)
            for name in ('stdout', 'stderr')}
    timeline = BoundedWriter(output / 'timeline.jsonl', timeline_bytes)
    previous_subreaper = subreaper(1)
    baseline = {(pid, item['start']) for pid, item in process_table().items()}
    owned: dict[int, int] = {}
    exited: list[dict[str, Any]] = []
    sampled: dict[tuple[int, int], dict[str, int | float]] = {}
    peaks: dict[str, int | None] = dict.fromkeys(('rss_bytes', 'pss_bytes', 'swap_bytes',
                                               'scratch_logical_bytes', 'scratch_allocated_bytes'))
    errors: set[str] = set()
    cancelled: str | None = None
    child: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    observer_before = resource.getrusage(resource.RUSAGE_SELF)
    root_status: int | None = None
    root_identity: tuple[int, int] | None = None
    previous_handlers = {}

    def cancel(signum: int, frame: Any) -> None:
        nonlocal cancelled
        cancelled = signal.Signals(signum).name

    for number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[number] = signal.signal(number, cancel)

    def discover() -> dict[int, dict[str, Any]]:
        return discover_owned(process_table(), owned, baseline, os.getpid())

    def reap(table: dict[int, dict[str, Any]]) -> None:
        nonlocal root_status
        for pid, item in table.items():
            if item['ppid'] != os.getpid():
                continue
            try:
                waited, status, usage = os.wait4(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if waited:
                code = os.waitstatus_to_exitcode(status)
                if child is not None and (pid, item['start']) == root_identity:
                    child.returncode = code  # Popen must never independently wait.
                    root_status = code
                exited.append({'pid': pid, 'start': item['start'], 'returncode': code,
                               'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
                               'minor_faults': usage.ru_minflt, 'major_faults': usage.ru_majflt,
                               'raw_rusage_swaps': usage.ru_nswap,
                               'swaps': None, 'swap_unavailable_reason': 'Linux ru_nswap is not maintained',
                               'maxrss_bytes': usage.ru_maxrss * 1024,
                               'block_input_operations': usage.ru_inblock,
                               'block_output_operations': usage.ru_oublock})

    def terminate(table: dict[int, dict[str, Any]], number: int) -> None:
        for pid, item in table.items():
            try:
                # Refresh identity immediately before signalling a potentially recycled PID.
                if proc_stat(Path(f'/proc/{pid}/stat').read_text())['start'] == item['start']:
                    os.kill(pid, number)
            except (OSError, ValueError, IndexError):
                pass

    next_sample = next_slow = started
    last_sample = started
    prior_sample_cpu = 0.
    stop_started: float | None = None
    try:
        env = dict(os.environ, GFB_BENCHMARK_SCRATCH=str(scratch.resolve()),
                   GFB_BENCHMARK_RESULT=str((output / 'workload.json').resolve()),
                   TMPDIR=str(scratch.resolve()))
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=env, start_new_session=True)
        owned[child.pid] = proc_stat(Path(f'/proc/{child.pid}/stat').read_text())['start']
        root_identity = (child.pid, owned[child.pid])
        for name, stream in (('stdout', child.stdout), ('stderr', child.stderr)):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while True:
            now = time.monotonic()
            table = discover()
            slow = sampling and now >= next_slow
            if sampling and now >= next_sample:
                row: dict[str, Any] = {'seconds': now - started, 'processes': []}
                rss = sum(item['rss_bytes'] for item in table.values())
                pss: int | None = 0 if slow else None
                swap: int | None = 0 if slow else None
                for pid, item in table.items():
                    io, reason = key_values(Path(f'/proc/{pid}/io'))
                    if reason:
                        errors.add('process_io: ' + reason)
                    counters = {key: item[key] for key in COUNTERS if key in item}
                    if io is not None:
                        counters.update({key: io[key] for key in COUNTERS if key in io})
                    identity = (pid, item['start'])
                    prior = sampled.setdefault(identity, {})
                    for key, value in counters.items():
                        prior[key] = max(value, prior.get(key, 0))
                    entry = {**item, 'io': io, 'io_unavailable': reason}
                    if slow:
                        memory, reason = key_values(Path(f'/proc/{pid}/smaps_rollup'), kib=True)
                        entry['memory'] = memory
                        entry['memory_unavailable'] = reason
                        if memory is None or 'Pss' not in memory or 'Swap' not in memory:
                            pss = swap = None
                            errors.add('process_pss_swap: ' + str(reason or 'missing fields'))
                        else:
                            pss = pss + memory['Pss'] if pss is not None else None
                            swap = swap + memory['Swap'] if swap is not None else None
                    row['processes'].append(entry)
                row.update(rss_bytes=rss, pss_bytes=pss, swap_bytes=swap)
                observed_cpu = sum(float(item.get('user_seconds', 0))
                                   + float(item.get('system_seconds', 0))
                                   for item in sampled.values())
                row['sampled_occupied_cores_lower_bound'] = (
                    (observed_cpu - prior_sample_cpu) / (now - last_sample)
                    if now > last_sample else None)
                prior_sample_cpu, last_sample = observed_cpu, now
                values = {'rss_bytes': rss, 'pss_bytes': pss, 'swap_bytes': swap}
                if slow:
                    disk = directory_bytes(scratch)
                    row['scratch'] = disk
                    values.update(scratch_logical_bytes=disk['logical_bytes'],
                                  scratch_allocated_bytes=disk['allocated_bytes'])
                    if device_gpu:
                        row['gpu'] = gpu_sample()
                    next_slow = now + 1
                for key, value in values.items():
                    if value is not None:
                        peaks[key] = max(value, peaks[key] or 0)
                timeline.write((json.dumps(row, separators=(',', ':')) + '\n').encode(), whole=True)
                next_sample = now + .1
            reap(table)
            table = discover()
            if cancelled is None and now - started >= seconds:
                cancelled = 'deadline'
            if (root_status is not None and any(item['state'] != 'Z' for item in table.values())
                    and cancelled is None):
                cancelled = 'descendants outlived command'
            if cancelled and table:
                if stop_started is None:
                    stop_started = now
                terminate(table, signal.SIGTERM if now - stop_started < 1 else signal.SIGKILL)
            if root_status is not None and not table and not selector.get_map():
                break
            for ready, _ in selector.select(.02):
                try:
                    data = os.read(ready.fd, 65536)
                except BlockingIOError:
                    continue
                if data:
                    logs[ready.data].write(data)
                else:
                    selector.unregister(ready.fileobj)
    except BaseException as error:  # noqa: BLE001 - reclaim all owned processes before reporting
        cancelled = f'observer exception: {type(error).__name__}: {error}'
        errors.add(cancelled)
    finally:
        # Exception paths retain the same wait4 owner and drain/kill descendants.
        while True:
            table = discover()
            if not table:
                break
            terminate(table, signal.SIGKILL)
            reap(table)
            time.sleep(.01)
        selector.close()
        for stream in ((child.stdout, child.stderr) if child is not None else ()):
            if stream is not None:
                stream.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        subreaper(previous_subreaper)
        log_result = {name: writer.close() for name, writer in logs.items()}
        timeline_result = timeline.close()
    elapsed = time.monotonic() - started
    totals = {key: (sum(item[key] for item in exited) if exited else None) for key in
              ('user_seconds', 'system_seconds', 'minor_faults', 'major_faults',
               'block_input_operations', 'block_output_operations')}
    totals['swaps'] = None  # Linux exposes this rusage field but does not maintain it.
    cpu = (sum(item['user_seconds'] + item['system_seconds'] for item in exited)
           if exited else None)
    observer_after = resource.getrusage(resource.RUSAGE_SELF)
    # /proc/PID/io can absorb waited-child counters. Count only independently
    # reaped roots, never both a parent and the descendants it already waited.
    reaped_samples = [sampled[(item['pid'], item['start'])] for item in exited
                      if (item['pid'], item['start']) in sampled]
    io_totals = {key: (sum(item[key] for item in reaped_samples if key in item)
                       if any(key in item for item in reaped_samples) else None)
                 for key in COUNTERS[4:]}
    result = {'schema_version': 1, 'command': command, 'returncode': root_status,
              'status': 'completed' if root_status == 0 and cancelled is None else 'failed',
              'cancellation': cancelled, 'wall_seconds': elapsed, 'sampling': sampling,
              'cpu': {**totals, 'aggregate_seconds': cpu,
                      'mean_occupied_cores': cpu / elapsed if cpu is not None else None,
                      'one_core_percent': 100 * cpu / elapsed if cpu is not None else None,
                      'whole_machine_percent': (100 * cpu / elapsed / (os.cpu_count() or 1)
                                                if cpu is not None else None),
                      'unavailable_reason': None if exited else 'no workload exit accounting'},
              'sampled_io_lower_bounds': io_totals, 'peaks': peaks, 'exits': exited,
              'exit_rusage_scope': 'each reaped root plus descendants it waited; no sampled CPU added',
              'per_process_exit_limitation': 'waited-descendant exit usage is aggregated by its parent; '
                                             'only sampled self counters can separate those processes',
              'scratch_residual': directory_bytes(scratch), 'logs': log_result,
              'final_output_bytes_before_measurement_write': directory_bytes(output),
              'timeline': timeline_result, 'metric_unavailable_reasons': sorted(errors),
              'sampling_unavailable_reason': None if sampling else 'optional sampling disabled',
              'gpu': {'status': 'device sampling enabled' if device_gpu else 'not applicable: CPU workload',
                      'torch_allocated_reserved_peaks': None,
                      'allocator_unavailable_reason': 'only the GPU workload owner can supply allocator peaks'},
              'observer_cpu_seconds': observer_after.ru_utime + observer_after.ru_stime
              - observer_before.ru_utime - observer_before.ru_stime,
              'machine': {'platform': platform.platform(), 'python': sys.version,
                          'executable': sys.executable, 'cpu_count': os.cpu_count(),
                          'affinity': sorted(os.sched_getaffinity(0)),
                          'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'limitations': LIMITATIONS}
    (output / 'measurement.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=7200)
    parser.add_argument('--no-sampling', action='store_true')
    parser.add_argument('--device-gpu', action='store_true')
    parser.add_argument('--log-bytes', type=int, default=1048576)
    parser.add_argument('--timeline-bytes', type=int, default=16 * 1048576)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    result = observe(command, args.output, seconds=args.seconds, sampling=not args.no_sampling,
                     log_bytes=args.log_bytes, timeline_bytes=args.timeline_bytes,
                     device_gpu=args.device_gpu)
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'wall_seconds': result['wall_seconds']}))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
