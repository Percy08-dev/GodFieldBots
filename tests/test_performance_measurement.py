"""Behavioral checks for the isolated stdlib measurement process."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = Path(__file__).resolve().parents[1] / 'tools/benchmark_performance.py'
WORKLOAD = TOOL.with_name('benchmark_workload.py')
spec = importlib.util.spec_from_file_location('performance_observer', TOOL)
assert spec is not None and spec.loader is not None
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
workload_spec = importlib.util.spec_from_file_location('performance_workload', WORKLOAD)
assert workload_spec is not None and workload_spec.loader is not None
workload = importlib.util.module_from_spec(workload_spec)
workload_spec.loader.exec_module(workload)


def run_monitor(tmp_path: Path, code: str, *options: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    output = tmp_path / 'measurement'
    completed = subprocess.run(
        [sys.executable, str(TOOL), '--output', str(output), *options,
         '--', sys.executable, '-c', code],
        text=True, capture_output=True, timeout=20, check=False,
    )
    assert (output / 'measurement.json').exists(), completed.stderr
    return completed, json.loads((output / 'measurement.json').read_text())


def test_wait4_counts_short_waited_children_once(tmp_path: Path) -> None:
    code = """
import json,resource,subprocess,sys,time
for _ in range(3):
    subprocess.run([sys.executable,'-c',
        'import time; t=time.process_time(); exec("while time.process_time()-t < .04: pass")'],check=True)
t=time.process_time()
while time.process_time()-t < .06: pass
a=resource.getrusage(resource.RUSAGE_SELF)
b=resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps({'cpu':a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime}))
"""
    completed, result = run_monitor(tmp_path, code)
    assert completed.returncode == 0, completed.stderr
    measured = result['cpu']['aggregate_seconds']
    expected = json.loads((tmp_path / 'measurement/stdout.log').read_text())['cpu']
    assert expected >= .18
    assert expected <= measured <= expected + .06
    assert len(result['exits']) == 1  # The command already waited its children.
    assert result['observer_cpu_seconds'] >= 0
    assert result['scratch_residual']['logical_bytes'] == 0


def test_unwaited_short_child_exit_is_owned_and_reaped(tmp_path: Path) -> None:
    code = """
import subprocess,sys,time
subprocess.Popen([sys.executable,'-c','import time; t=time.process_time(); exec("while time.process_time()-t < .08: pass")'])
time.sleep(.2)
"""
    completed, result = run_monitor(tmp_path, code)
    # The zombie child is adopted even if no sampling point observed its work.
    assert completed.returncode == 0, completed.stderr
    assert len(result['exits']) == 2
    assert result['cpu']['aggregate_seconds'] >= .08
    for item in result['exits']:
        assert not Path(f'/proc/{item["pid"]}').exists()


def test_bounded_logs_and_jsonl_records(tmp_path: Path) -> None:
    completed, result = run_monitor(
        tmp_path,
        'import sys,time; sys.stdout.write("a"*300000); sys.stderr.write("b"*300000); time.sleep(.35)',
        '--log-bytes', '128', '--timeline-bytes', '1500',
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / 'measurement/stdout.log').stat().st_size == 128
    assert (tmp_path / 'measurement/stderr.log').stat().st_size == 128
    assert result['logs']['stdout']['discarded_bytes'] == 300000 - 128
    assert result['timeline']['written_bytes'] <= 1500
    assert result['timeline']['discarded_bytes'] > 0
    for line in (tmp_path / 'measurement/timeline.jsonl').read_text().splitlines():
        json.loads(line)


def test_samples_combined_owned_output_without_following_external_links(tmp_path: Path) -> None:
    protected = tmp_path / 'protected-source'
    protected.mkdir()
    source = protected / 'untouched.bin'
    with source.open('wb') as stream:
        stream.truncate(64 * 1024 * 1024)
    original = source.stat()
    code = f"""
import os,time
from pathlib import Path
scratch=Path(os.environ['GFB_BENCHMARK_SCRATCH'])
artifacts=scratch.parent/'artifacts'
artifacts.mkdir()
(artifacts/'model.bin').write_bytes(b'm'*(512*1024))
temporary=scratch/'numeric.bin'
temporary.write_bytes(b'n'*(1024*1024))
(scratch.parent/'external-link').symlink_to({str(protected)!r},target_is_directory=True)
time.sleep(1.2)
temporary.unlink()
"""
    completed, result = run_monitor(tmp_path, code)
    assert completed.returncode == 0, completed.stderr
    peaks = result['peaks']
    assert peaks['scratch_logical_bytes'] >= 1024 * 1024
    assert peaks['output_logical_bytes'] >= peaks['scratch_logical_bytes'] + 512 * 1024
    assert peaks['output_allocated_bytes'] >= peaks['scratch_allocated_bytes']
    assert peaks['output_logical_bytes'] < original.st_size  # The external tree was not traversed.
    assert result['scratch_residual']['logical_bytes'] == 0
    assert (tmp_path / 'measurement/artifacts/model.bin').stat().st_size == 512 * 1024
    assert source.stat().st_size == original.st_size
    assert source.stat().st_mtime_ns == original.st_mtime_ns
    rows = [json.loads(line) for line in (tmp_path / 'measurement/timeline.jsonl').read_text().splitlines()]
    storage_rows = [row for row in rows if 'output' in row]
    assert len(storage_rows) >= 2
    assert all(row['output']['logical_bytes'] >= row['scratch']['logical_bytes'] for row in storage_rows)
    assert 'scratch is a subset' in result['storage_scope']


def test_deadline_cleans_escaped_descendant(tmp_path: Path) -> None:
    code = """
import subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import os,time; os.setsid(); time.sleep(60)'])
print(child.pid,flush=True)
time.sleep(60)
"""
    completed, result = run_monitor(tmp_path, code, '--seconds', '.3')
    assert completed.returncode == 1
    assert result['status'] == 'failed'
    assert result['cancellation'] == 'deadline'
    pid = int((tmp_path / 'measurement/stdout.log').read_text())
    assert not Path(f'/proc/{pid}').exists()
    assert len(result['exits']) == 2


def test_command_failure_preserves_exit_and_reaps_lingering_child(tmp_path: Path) -> None:
    code = """
import os,subprocess,sys
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
print(child.pid,flush=True)
os._exit(7)
"""
    completed, result = run_monitor(tmp_path, code)
    assert completed.returncode == 1
    assert result['returncode'] == 7
    assert result['cancellation'] == 'descendants outlived command'
    pid = int((tmp_path / 'measurement/stdout.log').read_text())
    assert not Path(f'/proc/{pid}').exists()


def test_malformed_and_unavailable_metrics_are_not_zero(tmp_path: Path) -> None:
    missing, reason = observer.key_values(tmp_path / 'missing')
    assert missing is None and reason
    malformed = tmp_path / 'metric'
    malformed.write_text('Pss: not-a-number kB\n')
    value, reason = observer.key_values(malformed, kib=True)
    assert value is None and reason
    malformed.write_text('abc-def ---p 00000000 00:00 0 [rollup]\nPss: 123 kB\nSwap: 0 kB\n')
    value, reason = observer.key_values(malformed, kib=True)
    assert value == {'Pss': 123 * 1024, 'Swap': 0}
    assert reason is None
    completed, result = run_monitor(tmp_path, 'pass', '--no-sampling')
    assert completed.returncode == 0
    assert all(value is None for value in result['peaks'].values())
    assert all(value is None for value in result['sampled_io_lower_bounds'].values())
    assert result['gpu']['status'].startswith('not applicable')


def test_failed_launch_still_reports_unavailable_exit_metrics(tmp_path: Path) -> None:
    result = observer.observe(['/definitely-missing-gfb-benchmark-command'], tmp_path / 'failed')
    assert result['status'] == 'failed'
    assert result['returncode'] is None
    assert result['cpu']['aggregate_seconds'] is None
    assert result['cpu']['unavailable_reason']
    assert 'FileNotFoundError' in result['cancellation']


def test_archive_payload_proof_matches_array_bytes(tmp_path: Path) -> None:
    import numpy as np

    arrays = {'x': np.arange(1200, dtype=np.float64).reshape(300, 4),
              'y': np.asarray([-.5, 1., 0.], dtype=np.float64)}
    path = tmp_path / 'arrays.npz'
    np.savez(path, x=arrays['x'], y=arrays['y'])
    assert workload.archive_proof(path) == workload.array_proof(arrays)


def test_scanner_read_bytes_adapter_preserves_counter_and_scope() -> None:
    metrics = workload.source_read_metrics({'read_bytes': 3215786181, 'source_bytes': 1}, 2.)
    assert metrics['scanned_source_read_bytes'] == 3215786181
    assert metrics['scanned_source_read_mib_per_second'] == 3215786181 / (1024 * 1024) / 2
    assert metrics['scanned_source_read_unavailable_reason'] is None
    assert 'static unique inventory and kernel rchar' in metrics['scanned_source_read_scope']
    assert workload.source_read_metrics({'read_bytes': 0}, 2.)['scanned_source_read_bytes'] == 0
    unavailable = workload.source_read_metrics({'source_bytes': 123}, 2.)
    assert unavailable['scanned_source_read_bytes'] is None
    assert unavailable['scanned_source_read_mib_per_second'] is None
    assert unavailable['scanned_source_read_unavailable_reason']


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), -float('inf'), 0., -1.])
def test_nonfinite_or_nonpositive_deadlines_fail_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('invalid duration must not launch any process')
    monkeypatch.setattr(observer.subprocess, 'Popen', forbidden)
    output = tmp_path / 'invalid-duration'
    with pytest.raises(ValueError, match='finite positive'):
        observer.observe(['ignored'], output, seconds=seconds)
    assert not output.exists()
    args = SimpleNamespace(seconds=seconds, output=output)
    with pytest.raises(ValueError, match='finite and positive'):
        workload.paired(args)
    assert not output.exists()


def test_discovery_replaces_recycled_baseline_and_owned_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {
        102: {'pid': 102, 'start': 31, 'ppid': 101},  # Child precedes its newly adopted parent.
        100: {'pid': 100, 'start': 10, 'ppid': 42},
        101: {'pid': 101, 'start': 21, 'ppid': 42},
        103: {'pid': 103, 'start': 40, 'ppid': 42},
        104: {'pid': 104, 'start': 50, 'ppid': 400},
        105: {'pid': 105, 'start': 60, 'ppid': 104},
    }
    monkeypatch.setattr(observer, 'process_table', lambda: table)
    owned = {102: 30, 103: 39, 104: 49}
    current = observer.discover_owned(observer.process_table(), owned, {(100, 10), (101, 20)}, 42)
    assert set(current) == {101, 102, 103}
    assert owned[101] == 21 and owned[102] == 31 and owned[103] == 40
    assert 104 not in current and 105 not in current  # Stale ownership cannot authorize a new family.


def test_recycled_root_pid_cannot_replace_original_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {'started': False, 'reaped': 0}
    pid = 999999
    def process_table() -> dict:
        if not state['started'] or state['reaped'] == 2:
            return {}
        return {pid: {'pid': pid, 'start': state['reaped'] + 1,
                      'ppid': os.getpid(), 'state': 'Z'}}
    def popen(*args: object, **kwargs: object) -> SimpleNamespace:
        state['started'] = True
        streams = []
        for _ in range(2):
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            streams.append(os.fdopen(read_fd, 'rb'))
        return SimpleNamespace(pid=pid, stdout=streams[0], stderr=streams[1], returncode=None)
    usage = SimpleNamespace(ru_utime=0., ru_stime=0., ru_minflt=0, ru_majflt=0,
                            ru_nswap=0, ru_maxrss=0, ru_inblock=0, ru_oublock=0)
    def wait4(wait_pid: int, flags: int) -> tuple:
        state['reaped'] += 1
        return wait_pid, (0 if state['reaped'] == 1 else 7 << 8), usage
    original_read = Path.read_text
    fields = ['Z', str(os.getpid()), *['0'] * 22]
    fields[19] = '1'
    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        return f'{pid} (fake) ' + ' '.join(fields) if str(path) == f'/proc/{pid}/stat' else original_read(path)
    monkeypatch.setattr(observer, 'process_table', process_table)
    monkeypatch.setattr(observer.subprocess, 'Popen', popen)
    monkeypatch.setattr(observer.os, 'wait4', wait4)
    monkeypatch.setattr(observer.platform, 'platform', lambda: 'synthetic-process-test')
    monkeypatch.setattr(Path, 'read_text', read_text)
    result = observer.observe(['fake'], tmp_path / 'root-reuse', sampling=False)
    assert result['returncode'] == 0
    assert [(item['start'], item['returncode']) for item in result['exits']] == [(1, 0), (2, 7)]


def make_reference(reference: Path) -> Path:
    source = reference / 'src'
    for name in workload.source_modules('preparation'):
        path = source / Path(*name.split('.'))
        if '.' not in name:
            path = path / '__init__.py'
        else:
            path = path.with_suffix('.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    return source


@pytest.mark.parametrize('failure', ['missing', 'incomplete', 'shadowed'])
def test_reference_preflight_fails_before_trial_or_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    reference = tmp_path / 'reference'
    if failure != 'missing':
        source = make_reference(reference)
        if failure == 'incomplete':
            (source / 'gf_training/fixed_data.py').unlink()
        else:
            shadow = tmp_path / 'shadowed'
            shadow.mkdir()
            (shadow / 'config.py').write_text('')
            (shadow / 'encoder.py').write_text('')
            (source / 'gf_models/__init__.py').write_text(f'__path__ = [{str(shadow)!r}]\n')
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail('failed source preflight must not reach input inventory or trial launch')
    monkeypatch.setattr(workload, 'source_inventory', forbidden)
    args = SimpleNamespace(seconds=10., output=tmp_path / 'output', reference=reference,
                           kind='preparation', execute=True)
    with pytest.raises(ValueError, match='source|shadowed'):
        workload.paired(args)
    assert not args.output.exists()


def test_clean_reference_probe_pins_import_origins_and_hashes(tmp_path: Path) -> None:
    source = make_reference(tmp_path / 'reference')
    proof = workload.preflight_source(source, 'preparation')
    assert proof['expected'] == proof['actual_imports']
    assert all(Path(item['path']).is_relative_to(source) for item in proof['actual_imports'].values())


def test_existing_source_directory_refused_and_symlinks_not_followed(tmp_path: Path) -> None:
    protected = tmp_path / 'protected'
    protected.mkdir()
    sentinel = protected / 'source.json'
    sentinel.write_text('source evidence')
    with pytest.raises(FileExistsError):
        observer.observe([sys.executable, '-c', 'pass'], protected)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (scratch / 'external').symlink_to(protected, target_is_directory=True)
    measured = observer.directory_bytes(scratch)
    assert measured['logical_bytes'] == (scratch / 'external').lstat().st_size
    assert sentinel.read_text() == 'source evidence'


def test_paired_plan_has_warmup_alternation_and_budget_matrix(tmp_path: Path) -> None:
    reference=tmp_path/'reference'
    make_reference(reference)
    completed = subprocess.run(
        [sys.executable, str(WORKLOAD), 'paired', '--kind', 'preparation',
         '--output', str(tmp_path / 'planned'),'--reference',str(reference)],
        text=True, capture_output=True, timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    plan = decoded['plan']
    standard = [row for row in plan if row['cell'] == 'standard']
    assert [(row['trial'], row['variant']) for row in standard] == [
        (-1, 'new'), (-1, 'old'), (0, 'old'), (0, 'new'),
        (1, 'new'), (1, 'old'), (2, 'old'), (2, 'new')]
    assert {row['cell'] for row in plan} == {
        'standard', 'memory2048-chunk16', 'memory2048-chunk64',
        'memory8192-chunk16'}
    assert decoded['aliases']['memory8192-chunk64']['cell'] == 'standard'
    assert len(plan) == 17
    assert len([row for row in plan if row['trial'] < 0]) == 2
    for cell in ('memory2048-chunk16', 'memory2048-chunk64', 'memory8192-chunk16'):
        assert [(row['trial'], row['variant']) for row in plan if row['cell'] == cell] == [
            (0, 'new'), (1, 'new'), (2, 'new')]
    assert not (tmp_path / 'planned').exists()
    assert all('VIRTUAL_ENV' in row['command'] for row in plan)
    assert 'torch' not in observer.__dict__


@pytest.mark.skipif(not hasattr(os, 'kill'), reason='Linux process monitor')
def test_sigterm_cancellation_finishes_report(tmp_path: Path) -> None:
    output = tmp_path / 'signal-run'
    process = subprocess.Popen(
        [sys.executable, str(TOOL), '--output', str(output), '--',
         sys.executable, '-c', 'import time; print("ready",flush=True); time.sleep(60)'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        log = output / 'stdout.log'
        if log.exists() and log.stat().st_size:
            break
        time.sleep(.02)
    process.terminate()
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 1, (stdout, stderr)
    result = json.loads((output / 'measurement.json').read_text())
    assert result['cancellation'] == 'SIGTERM'
    assert all(not Path(f'/proc/{item["pid"]}').exists() for item in result['exits'])
