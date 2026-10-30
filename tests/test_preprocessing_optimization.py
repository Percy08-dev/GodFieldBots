"""Experiment controls must fail closed without touching the main environment."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'optimization_runner', ROOT / 'benchmarks/native_preprocessing/runner.py')
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fixed_budget_rejects_silent_relaxation(tmp_path):
    original = (runner.HERE / 'experiment.toml').read_text()
    changed = tmp_path / 'experiment.toml'
    changed.write_text(original.replace('memory_budget_mib = 2048', 'memory_budget_mib = 4096'))
    with pytest.raises(ValueError, match='memory_budget_mib'):
        runner.config(changed)


def test_environment_cannot_inherit_allocator_or_foreign_pythonpath(monkeypatch):
    for key in ('VIRTUAL_ENV', 'LD_PRELOAD', 'PYTHONPATH', 'RUSTFLAGS'):
        monkeypatch.setenv(key, '/foreign')
    env = runner.environment(Path('/owned/site'))
    assert env['PYTHONPATH'] == f'{ROOT}/src:/owned/site'
    assert all(key not in env for key in ('VIRTUAL_ENV', 'LD_PRELOAD', 'RUSTFLAGS'))
    assert env['OPENBLAS_NUM_THREADS'] == '1'


def test_environment_removes_allocator_and_compiler_tuning_without_logging_values(monkeypatch):
    keys = ('_RJEM_MALLOC_CONF', 'MALLOC_CONF', 'MIMALLOC_VERBOSE', 'MI_OPTION',
            'mimalloc_purge_delay', 'MiMalloc_Allow_Large_Os_Pages', 'mi_option',
            'JEMALLOC_OVERRIDE', 'SYS_WITH_MALLOC_CONF', 'TIKV_JEMALLOC_CONFIG',
            'MALLOC_TRIM_THRESHOLD_', 'GLIBC_TUNABLES', 'CC', 'CXX', 'CFLAGS',
            'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'AR', 'RANLIB', 'CARGO_BUILD_RUSTFLAGS',
            'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS', 'CARGO_PROFILE_RELEASE_LTO')
    for key in keys:
        monkeypatch.setenv(key, 'secret-tuning-value')
    assert not set(keys) & runner.environment().keys()
    proof = runner.sanitization_proof()
    assert set(keys) <= set(proof['removed_key_names'])
    assert 'secret-tuning-value' not in json.dumps(proof)


@pytest.mark.parametrize('proof,key,value', [
    ('measurement', 'status', 'failed'),
    ('measurement', 'returncode', 2),
    ('measurement', 'cancellation', 'deadline'),
    ('owner', 'observer_returncode', 2),
    ('owner', 'cancellation', 'SIGINT'),
    ('owner', 'cleanup_deadline_exceeded', True),
    ('owner', 'cleanup_owner_joined', False),
])
def test_helper_pass_cannot_override_failed_or_cancelled_observer(tmp_path, proof, key, value):
    oracle = tmp_path / 'oracle.json'
    oracle.write_text(json.dumps({'status': 'PASS', 'completed': True}))
    measurement = {'status': 'completed', 'returncode': 0, 'cancellation': None,
                   'command': [sys.executable, str(runner.HERE / 'typed_oracle.py'),
                               '--output', str(oracle)]}
    owner = {'observer_returncode': 0, 'cancellation': None,
             'cleanup_deadline_exceeded': False, 'cleanup_owner_joined': True}
    if proof == 'measurement':
        measurement[key] = value
    else:
        owner[key] = value
    runner.write(tmp_path / 'measurement.json', measurement)
    runner.write(tmp_path / 'owner-lifecycle.json', owner)
    with pytest.raises(ValueError, match='completed, uncancelled enclosing observer'):
        runner.completed_oracle(oracle)


def test_oracle_completion_requires_its_own_successful_observer(tmp_path):
    oracle = tmp_path / 'oracle.json'
    helper = {'status': 'PASS', 'completed': True}
    runner.write(oracle, helper)
    with pytest.raises(ValueError, match='enclosing observer'):
        runner.completed_oracle(oracle)
    measurement = {'status': 'completed', 'returncode': 0, 'cancellation': None,
                   'command': [sys.executable, str(runner.HERE / 'typed_oracle.py'),
                               '--output', str(oracle)]}
    runner.write(tmp_path / 'measurement.json', measurement)
    runner.write(tmp_path / 'owner-lifecycle.json', {
        'observer_returncode': 0, 'cancellation': None,
        'cleanup_deadline_exceeded': False, 'cleanup_owner_joined': True})
    assert runner.completed_oracle(oracle) == helper
    measurement['command'][-1] = str(tmp_path / 'unrelated-oracle.json')
    runner.write(tmp_path / 'measurement.json', measurement)
    with pytest.raises(ValueError, match='enclosing observer'):
        runner.completed_oracle(oracle)


def test_trial_order_and_concurrency_are_fixed(tmp_path):
    variants = []
    for allocator in runner.ALLOCATORS:
        path = tmp_path / f'{allocator}.json'
        path.write_text(json.dumps({'allocator': allocator, 'site': '/owned/site',
                                   'native_identity': {'extension_sha256': allocator}}))
        variants.append(path)
    args = runner.parser().parse_args(['run', '--output', str(tmp_path / 'trials'),
                                      *[part for path in variants for part in
                                        ('--variant', str(path))]])
    plan = runner.trial_plan(args, runner.config(runner.HERE / 'experiment.toml'))
    assert len(plan) == 36
    assert [row['allocator'] for row in plan[6:9]] == ['mimalloc', 'jemalloc', 'system']
    for row in plan:
        command = row['command']
        assert command[command.index('--workers') + 1] == str(row['workers'])
        assert command[command.index('--max-in-flight') + 1] == str(row['workers'])
        assert 'LD_PRELOAD' in command


def test_wrong_binary_is_rejected(tmp_path, monkeypatch):
    wheel = tmp_path / 'test.whl'
    wheel.write_bytes(b'wheel')
    path = tmp_path / 'variant.json'
    path.write_text(json.dumps({'site': str(tmp_path / 'site'), 'wheel': str(wheel),
                               'wheel_sha256': runner.sha(wheel),
                               'native_identity': {'extension_sha256': 'expected'}}))
    monkeypatch.setattr(runner, 'native_probe', lambda site: {'extension_sha256': 'wrong'})
    with pytest.raises(ValueError, match='wrong loaded binary'):
        runner.verify_variant(path)


def test_setup_failure_and_timeout_are_not_success():
    with pytest.raises(RuntimeError, match='command failed'):
        runner.checked([runner.sys.executable, '-c', 'raise SystemExit(3)'], seconds=5)
    with pytest.raises(runner.subprocess.TimeoutExpired):
        runner.checked([runner.sys.executable, '-c', 'import time; time.sleep(5)'], seconds=1)


def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('bounded fixture condition not reached')
        time.sleep(.02)


@pytest.mark.parametrize('failure', ['nonzero', 'timeout'])
def test_setup_reclaims_stdio_closed_term_ignoring_descendant(tmp_path, failure):
    ready = tmp_path / 'descendant.pid'
    descendant = ('import os,signal,time;from pathlib import Path;'
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                  f'Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)')
    root = ('import subprocess,sys,time;from pathlib import Path;'
            f'p=subprocess.Popen([sys.executable,"-c",{descendant!r}],'
            'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);'
            f'path=Path({str(ready)!r});deadline=time.monotonic()+3\n'
            'while not path.exists() and time.monotonic()<deadline: time.sleep(.01)\n'
            + ('raise SystemExit(3)' if failure == 'nonzero' else 'time.sleep(30)'))
    try:
        with pytest.raises(RuntimeError if failure == 'nonzero' else subprocess.TimeoutExpired):
            runner.checked([sys.executable, '-c', root], seconds=1)
        pid = int(ready.read_text())
        until(lambda: not running(pid))
    finally:
        if ready.exists() and running(pid := int(ready.read_text())):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize('cleanup_seconds,expired', [(5, False), (.05, True)])
def test_observer_cancellation_joins_term_ignoring_descendant(tmp_path, cleanup_seconds, expired):
    ready = tmp_path / 'descendant.pid'
    output = tmp_path / 'observed'
    descendant = ('import os,signal,time;from pathlib import Path;'
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                  f'Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)')
    workload = ('import subprocess,sys,time;'
                f'subprocess.Popen([sys.executable,"-c",{descendant!r}]);time.sleep(30)')
    command = [sys.executable, str(ROOT / 'tools/benchmark_performance.py'),
               '--output', str(output), '--seconds', '20', '--', sys.executable, '-c', workload]
    owner_pid = tmp_path / 'owner.pid'
    controller = ('import importlib.util,sys;from pathlib import Path;'
                  f's=importlib.util.spec_from_file_location("runner",{str(runner.HERE / "runner.py")!r});'
                  'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
                  'original=m.subprocess.Popen\n'
                  'def start(*args,**kwargs):\n'
                  ' p=original(*args,**kwargs)\n'
                  f' Path({str(owner_pid)!r}).write_text(str(p.pid))\n'
                  ' return p\n'
                  'm.subprocess.Popen=start\n'
                  f'sys.exit(m.run_observer({command!r},cleanup_seconds={cleanup_seconds!r}).returncode)')
    child = subprocess.Popen([sys.executable, '-c', controller], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        until(ready.exists)
        pid = int(ready.read_text())
        child.send_signal(signal.SIGINT)
        time.sleep(.1)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=8) == 2
        assert not running(pid)
        proof = json.loads((output / 'owner-lifecycle.json').read_text())
        assert proof['signal_count'] == 2
        assert proof['cancellation'] == 'SIGINT'
        assert proof['cleanup_owner_joined']
        assert proof['cleanup_deadline_exceeded'] is expired
        measurement = json.loads((output / 'measurement.json').read_text())
        assert measurement['status'] == 'failed'
        assert any(row['pid'] == pid and row['returncode'] == -signal.SIGKILL
                   for row in measurement['exits'])
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                for path in (ready, owner_pid):
                    if path.exists() and running(pid := int(path.read_text())):
                        os.kill(pid, signal.SIGKILL)
                child.kill()
                child.wait()


def test_default_dry_run_creates_no_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'absent'
    monkeypatch.setattr(runner.sys, 'argv', ['runner', 'build', '--allocator', 'system',
                                          '--output', str(output)])
    runner.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)['execute'] is False


def test_dry_run_still_refuses_production_output(monkeypatch):
    monkeypatch.setattr(runner.sys, 'argv', ['runner', 'build', '--allocator', 'system',
                                          '--output', str(ROOT / 'src/accidental-output')])
    with pytest.raises(ValueError, match='experiment output'):
        runner.main()


def test_typed_primary_and_counter_plans_require_distinct_feature_builds(tmp_path):
    builds = []
    for ap in (True, False):
        path = tmp_path / f'{ap}.json'
        path.write_text(json.dumps({'identity': {'arbitrary_precision': ap,
                                               'allocation_counts': False},
                                    'binary': f'/owned/{ap}/binary', 'binary_sha256': str(ap)}))
        builds.append(path)
    args = runner.parser().parse_args(['typed-run', '--output', str(tmp_path / 'trials'),
                                      '--inventory', '/frozen/journals.json',
                                      *[part for path in builds for part in
                                        ('--typed-build', str(path))]])
    cfg = runner.config(runner.HERE / 'experiment.toml')
    plan = runner.typed_trial_plan(args, cfg)
    assert len(plan) == 24
    assert [row['variant'] for row in plan[8:12]] == ['B', 'C', 'D', 'A']
    for row in plan:
        assert row['command'][row['command'].index('--') + 1] == (
            '/owned/False/binary' if row['variant'] == 'D' else '/owned/True/binary')
    args.command = 'typed-counter'
    with pytest.raises(ValueError, match='counter builds must be separate'):
        runner.typed_trial_plan(args, cfg)


def test_generated_schema_matches_single_schema_source():
    spec = importlib.util.spec_from_file_location('schema_generator',
                                                 runner.HERE / 'generate_schema.py')
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for borrowed, name in ((False, 'owned.rs'), (True, 'borrowed.rs')):
        assert (generator.OUT / name).read_text() == generator.generate(borrowed)


def test_protected_original_fit_manifest_rejects_changed_bytes(tmp_path):
    records = []
    for seed in range(6):
        directory = tmp_path / str(seed)
        directory.mkdir()
        file = directory / 'model.txt'
        file.write_text('original model')
        stat = file.stat()
        records.append({'directory': str(directory), 'files': {'model.txt': {
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino,
            'sha256': runner.sha(file)}}})
    path = tmp_path / 'original-six.json'
    path.write_text(json.dumps(records))
    assert len(runner.protected_fits(path)) == 6
    file.write_text('modified model')
    with pytest.raises(ValueError, match='protected original fit'):
        runner.protected_fits(path)


@pytest.mark.parametrize('mutation', ['duplicate', 'omit', 'swap', 'hash', 'bytes'])
def test_journal_inventory_binds_exact_frozen_authority(tmp_path, mutation):
    entries = [{'path': f'/frozen/{index}/player-1.jsonl', 'logical_bytes': index + 1,
                'sha256': f'{index:064x}'} for index in range(13568)]
    frozen = {'inventory': {'entries': entries}}
    supplied = [dict(entry) for entry in entries]
    if mutation == 'duplicate':
        supplied[-1] = supplied[0]
    elif mutation == 'omit':
        supplied.pop()
    elif mutation == 'swap':
        supplied[0], supplied[1] = supplied[1], supplied[0]
    elif mutation == 'hash':
        supplied[0]['sha256'] = 'f' * 64
    else:
        supplied[0]['logical_bytes'] += 1
    path = tmp_path / 'journals.json'
    path.write_text(json.dumps(supplied))
    with pytest.raises(ValueError, match='exact ordered frozen authority'):
        runner.frozen_journals(frozen, path)


def test_inventory_sibling_tamper_is_not_authority(tmp_path):
    entries = [{'path': f'/frozen/{index}/player-1.jsonl', 'logical_bytes': 123,
                'sha256': 'a' * 64} for index in range(13568)]
    frozen = {'inventory': {'entries': entries}}
    supplied = tmp_path / 'caller.json'
    supplied.write_text(json.dumps(entries))
    (tmp_path / 'journals.json').write_text('[]')
    assert runner.frozen_journals(frozen, supplied) == entries
    with pytest.raises(ValueError):
        runner.frozen_journals(frozen, tmp_path / 'journals.json')


def test_report_preserves_nested_metrics_and_integer_resource_summaries(tmp_path):
    rows = []
    for trial in range(5):
        value = trial + 1
        rows.append({'allocator': 'system', 'workers': 1, 'trial': trial, 'measurement': {
            'wall_seconds': value,
            'cpu': {key: value for key in ('aggregate_seconds', 'user_seconds', 'system_seconds')},
            'peaks': {key: value * 1048576 for key in (
                'rss_bytes', 'pss_bytes', 'scratch_logical_bytes', 'scratch_allocated_bytes',
                'output_logical_bytes', 'output_allocated_bytes')},
            'sampled_io_lower_bounds': {'rchar': None, 'wchar': value,
                                       'read_bytes': value, 'write_bytes': value}}})
    results = tmp_path / 'results.json'
    runner.write(results, rows)
    args = runner.parser().parse_args(['report', '--results', str(results)])
    report = runner.report(args)['groups']['system-1']
    assert report['wall_seconds'] == {'count': 5, 'median': 3, 'min': 1, 'max': 5, 'iqr': 2}
    assert report['user_seconds']['median'] == 3
    assert report['rss_mib']['median'] == 3
    assert report['read_mib']['median'] == 3 / 1048576
    assert report['rchar']['count'] == 0
    assert report['five_trials_complete']
