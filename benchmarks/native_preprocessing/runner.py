"""ADR0016 isolated experiments. Every public command is a dry run by default.

Use the project Python with VIRTUAL_ENV unset. Output directories must be new.
No command installs into the project environment or changes production sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ALLOCATORS = ('system', 'mimalloc', 'jemalloc')
FROZEN = 'ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f'
FIXED = {'schema_version': 1, 'memory_budget_mib': 2048, 'cache_budget_mib': 512,
         'chunk_target_mib': 16, 'scratch_budget_mib': 8192, 'max_source_file_mib': 64,
         'max_record_mib': 8, 'warmups': 1, 'trials': 5, 'worker_counts': [1, 4]}
SANITIZED_KEYS = ('VIRTUAL_ENV', 'LD_PRELOAD', 'PYTHONPATH', 'RUSTFLAGS',
                  'CARGO_ENCODED_RUSTFLAGS', 'CARGO_TARGET_DIR', 'RUSTC', 'RUSTDOC',
                  'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTDOCFLAGS',
                  '_RJEM_MALLOC_CONF', 'SYS_WITH_MALLOC_CONF', 'GLIBC_TUNABLES',
                  'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'AR', 'RANLIB')
SANITIZED_PREFIXES = ('MIMALLOC_', 'MI_', 'JEMALLOC_', 'TIKV_JEMALLOC_', 'MALLOC_',
                      'CARGO_PROFILE_', 'CARGO_BUILD_', 'CARGO_TARGET_',
                      'CC_', 'CXX_', 'CFLAGS_', 'CXXFLAGS_', 'CPPFLAGS_', 'LDFLAGS_',
                      'AR_', 'RANLIB_', 'HOST_', 'TARGET_')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def config(path: Path) -> dict[str, Any]:
    result = tomllib.loads(path.read_text())
    for key, expected in FIXED.items():
        if result.get(key) != expected or type(result.get(key)) is not type(expected):
            raise ValueError(f'fixed experiment condition changed: {key}')
    for key in ('trial_seconds', 'cleanup_seconds'):
        if type(result.get(key)) is not int or result[key] <= 0:
            raise ValueError(f'positive integer required: {key}')
    return result


def workload() -> Any:
    spec = importlib.util.spec_from_file_location('gfb_frozen_workload',
                                                ROOT / 'tools/benchmark_workload.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(site: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.upper() in SANITIZED_KEYS or key.upper().startswith(SANITIZED_PREFIXES):
            env.pop(key)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', CARGO_HOME='/tmp/gfb-opt-cargo')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), *([str(site)] if site else [])])
    env.update(CARGO_PROFILE_RELEASE_OPT_LEVEL='3', CARGO_PROFILE_RELEASE_DEBUG='0',
               CARGO_PROFILE_RELEASE_STRIP='none', CARGO_INCREMENTAL='0')
    return env


def sanitization_proof() -> dict[str, Any]:
    return {'removed_key_names': sorted(key for key in os.environ
                                       if key.upper() in SANITIZED_KEYS
                                       or key.upper().startswith(SANITIZED_PREFIXES)),
            'explicit_keys': SANITIZED_KEYS, 'prefixes': SANITIZED_PREFIXES,
            'policy': 'case-insensitive key matching before every build/probe/workload; '
                      'fixed release/thread controls reapplied; '
                      'default allocator configuration, no background-thread or override tuning; no values logged'}


def checked(command: list[str], *, seconds: int, env: dict[str, str] | None = None) -> str:
    """Bound setup children; observers own timed workload process-tree cleanup."""
    with subprocess.Popen(command, env=env or environment(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=seconds)
        except BaseException:
            # Setup tools own a fresh process group; never signal the caller's group.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The root may have exited and closed stdio while a descendant
                # ignores TERM. Always kill the owned group, even after EOF.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.communicate(timeout=4)
            raise
        if child.returncode:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise RuntimeError(f'command failed ({child.returncode}): {command}\n{stderr[-8192:]}')
        return stdout


def run_observer(command: list[str], *, cleanup_seconds: int) -> subprocess.CompletedProcess:
    """Forward cancellation, then join the existing observer's sole cleanup owner.

    Repeated signals never kill that owner. Cleanup deadline expiry invalidates
    the run but does not abandon a process that still owns live descendants.
    """
    child = None
    cancelled = None
    cancel_started = None
    signal_count = 0
    expired = False
    previous = {}

    def cancel(signum: int, frame: Any) -> None:
        nonlocal cancelled, cancel_started, signal_count
        signal_count += 1
        if cancelled is None:
            cancelled = signal.Signals(signum).name
            cancel_started = time.monotonic()
            if child is not None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass

    def join() -> int:
        nonlocal expired
        assert child is not None
        while True:
            try:
                return child.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                if (cancel_started is not None and not expired
                        and time.monotonic() - cancel_started >= cleanup_seconds):
                    expired = True
                    print('Observer cleanup deadline exceeded; retaining owner until reaped.',
                          file=sys.stderr, flush=True)

    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, cancel)
    try:
        if cancelled is not None:
            raise InterruptedError('cancelled before observer launch')
        child = subprocess.Popen(command, env=environment(), start_new_session=True)
        if cancelled is not None:  # Signal during Popen construction.
            child.send_signal(signal.SIGTERM)
        try:
            code = join()
        except BaseException:
            if cancel_started is None:
                cancelled = 'runner exception'
                cancel_started = time.monotonic()
                child.send_signal(signal.SIGTERM)
            join()
            raise
        directory = Path(command[command.index('--output') + 1])
        if directory.is_dir():
            write(directory / 'owner-lifecycle.json', {
                'schema_version': 1, 'observer_returncode': code, 'cancellation': cancelled,
                'signal_count': signal_count, 'cleanup_deadline_exceeded': expired,
                'cleanup_owner_joined': True,
                'scope': 'runner retained and joined the existing observer; no duplicate watchdog'})
        return subprocess.CompletedProcess(command, 2 if cancelled is not None or expired else code)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def source_hash(directory: Path) -> str:
    h = hashlib.sha256()
    for path in sorted([*directory.joinpath('src').rglob('*.rs'), directory / 'build.rs']):
        h.update(str(path.relative_to(directory)).encode())
        h.update(b'\0')
        h.update(path.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def typed_source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted([*HERE.joinpath('rust/src').glob('*.rs'), HERE / 'generate_schema.py',
                        HERE / 'rust/Cargo.toml', HERE / 'rust/Cargo.lock']):
        h.update(str(path.relative_to(HERE)).encode() + b'\0')
        h.update(path.read_bytes() + b'\0')
    return h.hexdigest()


def code_files(config_path: Path) -> list[dict[str, str]]:
    files = {*ROOT.joinpath('src').rglob('*.py'), *HERE.rglob('*.py'),
             *HERE.joinpath('rust/src').glob('*.rs'), ROOT / 'pyproject.toml', ROOT / 'uv.lock',
             ROOT / 'native/Cargo.toml', ROOT / 'native/Cargo.lock',
             HERE / 'rust/Cargo.toml', HERE / 'rust/Cargo.lock', config_path.resolve(),
             ROOT / 'tools/benchmark_workload.py', ROOT / 'tools/benchmark_performance.py'}
    return [{'path': str(path.resolve()), 'sha256': sha(path)} for path in sorted(files)]


def build_log(command: list[str], output: Path, *, env: dict[str, str], seconds: int) -> None:
    # The outer setup group owns the compiler and all of its descendants on timeout.
    log_command = [sys.executable, '-c',
                   ('import subprocess,sys; from pathlib import Path; '
                    'p=subprocess.run(sys.argv[2:],capture_output=True); '
                    'Path(sys.argv[1]+"/build.stdout.log").write_bytes(p.stdout); '
                    'Path(sys.argv[1]+"/build.stderr.log").write_bytes(p.stderr); '
                    'sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr); '
                    'sys.exit(p.returncode)'), str(output), *command]
    checked(log_command, seconds=seconds, env=env)


def native_probe(site: Path | None = None) -> dict[str, Any]:
    code = ('import json; from gf_training.native_backend import binary_identity; '
            'print(json.dumps(binary_identity()))')
    return json.loads(checked([sys.executable, '-c', code], seconds=30, env=environment(site)))


def protected_fits(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if len(records) != 6:
        raise ValueError('original-six manifest must identify six fits')
    result = []
    for record in records:
        for name, proof in record['files'].items():
            file = Path(record['directory']) / name
            stat = file.stat()
            if any(actual != proof[key] for actual, key in
                   ((stat.st_size, 'size'), (stat.st_mtime_ns, 'mtime_ns'), (stat.st_ino, 'inode'))):
                raise ValueError(f'protected original fit stat changed: {file}')
            expected = proof.get('sha256', proof.get('recorded_sha256'))
            if expected is None or sha(file) != expected:
                raise ValueError(f'protected original fit bytes changed: {file}')
            result.append({'path': str(file), 'sha256': expected})
    return result


def private_recipe(allocator: str, cfg: dict[str, Any]) -> tuple[str, str]:
    manifest = (ROOT / 'native/Cargo.toml').read_text()
    lib = (ROOT / 'native/src/lib.rs').read_text()
    if allocator not in ALLOCATORS:
        raise ValueError('unknown allocator label')
    if allocator != 'system':
        dependency, version, declaration = (
            ('mimalloc', cfg['versions']['mimalloc'],
             'mimalloc::MiMalloc = mimalloc::MiMalloc') if allocator == 'mimalloc' else
            ('tikv-jemallocator', cfg['versions']['tikv_jemallocator'],
             'tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc'))
        features = json.dumps(cfg['allocator_features'][allocator])
        dependency_line = f'{dependency} = {{ version = "={version}", features = {features} }}'
        manifest = manifest.replace('[dependencies]', '[dependencies]\n' + dependency_line, 1)
        lib = '#[global_allocator]\nstatic EXPERIMENT_ALLOCATOR: ' + declaration + ';\n' + lib
    return manifest + '\n[workspace]\n', lib


def verify_variant(path: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    item = json.loads(path.read_text())
    site = Path(item['site'])
    if not site.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError('wheel overlay must belong to build directory')
    if sha(Path(item['wheel'])) != item['wheel_sha256']:
        raise ValueError('wheel changed')
    actual = native_probe(site)
    if actual != item['native_identity']:
        raise ValueError('wrong loaded binary/build identity')
    if not Path(actual['extension_path']).resolve().is_relative_to(site.resolve()):
        raise ValueError('native extension escaped wheel overlay')
    if actual['build']['native_source_sha256'] != source_hash(path.parent / 'native'):
        raise ValueError('private build source identity mismatch')
    private = path.parent / 'native'
    expected_manifest, expected_lib = private_recipe(item['allocator'], cfg or config(HERE / 'experiment.toml'))
    if ((private / 'Cargo.toml').read_text() != expected_manifest
            or (private / 'src/lib.rs').read_text() != expected_lib):
        raise ValueError('allocator label/declaration/pinned dependency mismatch')
    for original in [*ROOT.joinpath('native/src').rglob('*.rs'), ROOT / 'native/build.rs']:
        relative = original.relative_to(ROOT / 'native')
        if relative != Path('src/lib.rs') and (private / relative).read_bytes() != original.read_bytes():
            raise ValueError('non-allocator native source changed')
    if {p.relative_to(private) for p in private.joinpath('src').rglob('*.rs')} != {
            p.relative_to(ROOT / 'native') for p in ROOT.joinpath('native/src').rglob('*.rs')}:
        raise ValueError('private native source file set changed')
    if actual['build']['cargo_lock_sha256'] != sha(private / 'Cargo.lock'):
        raise ValueError('loaded native lock identity mismatch')
    if actual['build']['profile'] != 'release' or actual['build']['rustflags']:
        raise ValueError('uncontrolled native profile/flags')
    return item


def freeze(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    helper = workload()
    dataset = ROOT / cfg['dataset']
    data = json.loads(dataset.read_text())
    if data['content_sha256'] != FROZEN or data['train_decisions'] + data['validation_decisions'] != 83783:
        raise ValueError('wrong frozen dataset')
    inventory = helper.source_inventory(dataset)
    for entry in inventory['entries']:
        entry['sha256'] = sha(Path(entry['path']))
    fits = protected_fits(args.protected_fits)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'freeze.json', {
        'schema_version': 1, 'config': cfg, 'config_sha256': sha(args.config),
        'git_revision': checked(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], seconds=10).strip(),
        'git_status': checked(['git', '-C', str(ROOT), 'status', '--porcelain'], seconds=10),
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'rustc': checked(['rustc', '--version'], seconds=10).strip(),
        'native_identity': native_probe(), 'native_source_sha256': source_hash(ROOT / 'native'),
        'dataset_manifest_sha256': sha(dataset), 'inventory': inventory, 'original_fits': fits,
        'protected_manifest': str(args.protected_fits),
        'protected_manifest_sha256': sha(args.protected_fits),
        'observer_sha256': sha(ROOT / 'tools/benchmark_performance.py'),
        'code_files': code_files(args.config),
        'workload_sha256': sha(ROOT / 'tools/benchmark_workload.py'),
        'thread_environment': {key: environment()[key] for key in
                               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'environment_sanitization': sanitization_proof(),
        'eligibility': 'not inferred from inventory; validate executes full existing preparation',
    })
    journals = [entry for entry in inventory['entries']
                if Path(entry['path']).name in ('player-1.jsonl', 'player-2.jsonl')]
    if len(journals) != 13568:
        raise ValueError('frozen journal inventory count changed')
    write(args.output / 'journals.json', journals)


def verify_freeze(path: Path, cfg: dict[str, Any], *, require_code: bool = False) -> dict[str, Any]:
    frozen = json.loads(path.read_text())
    if frozen['config'] != cfg:
        raise ValueError('frozen configuration changed')
    if require_code and not frozen.get('code_files'):
        raise ValueError('primary/counter matrix requires a final code freeze')
    for item in frozen.get('code_files', []):
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError(f'frozen code changed: {item["path"]}')
    if sha(Path(frozen['protected_manifest'])) != frozen['protected_manifest_sha256']:
        raise ValueError('protected original-six manifest changed')
    protected_fits(Path(frozen['protected_manifest']))
    for item in [*frozen['inventory']['entries'], *frozen['original_fits']]:
        file = Path(item['path'])
        if sha(file) != item['sha256']:
            raise ValueError(f'frozen input changed: {file}')
    for name, key in [('benchmark_performance.py', 'observer_sha256'),
                      ('benchmark_workload.py', 'workload_sha256')]:
        if sha(ROOT / 'tools' / name) != frozen[key]:
            raise ValueError(f'benchmark helper changed: {name}')
    if source_hash(ROOT / 'native') != frozen['native_source_sha256']:
        raise ValueError('production native source changed')
    if native_probe() != frozen['native_identity']:
        raise ValueError('main native installation changed')
    return frozen


def frozen_journals(frozen: dict[str, Any], inventory: Path) -> list[dict[str, Any]]:
    """Only the hashed freeze inventory is authority; its sibling is a convenience."""
    expected = [entry for entry in frozen['inventory']['entries']
                if Path(entry['path']).name in ('player-1.jsonl', 'player-2.jsonl')]
    paths = [entry['path'] for entry in expected]
    if len(expected) != 13568 or len(set(paths)) != len(paths):
        raise ValueError('frozen journal authority count/duplicates invalid')
    supplied = json.loads(inventory.read_text())
    if supplied != expected:
        raise ValueError('journal inventory differs from exact ordered frozen authority')
    return expected


def build(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / 'native'
    shutil.copytree(ROOT / 'native', private,
                    ignore=shutil.ignore_patterns('target', '*.so', '__pycache__'))
    manifest = private / 'Cargo.toml'
    text, lib = private_recipe(args.allocator, cfg)
    manifest.write_text(text)
    (private / 'src/lib.rs').write_text(lib)
    env = environment()
    env['CARGO_TARGET_DIR'] = str(args.output / 'target')
    original_lock = (private / 'Cargo.lock').read_bytes()
    metadata = json.loads(checked(['cargo', 'metadata', '--offline', '--format-version', '1',
                                  '--manifest-path', str(manifest),
                                  *(['--locked'] if args.allocator == 'system' else [])],
                                 seconds=120, env=env))
    current_lock = (private / 'Cargo.lock').read_bytes()
    if args.allocator == 'system' and current_lock != original_lock:
        raise ValueError('System lock bytes changed')
    def package_ids(raw: bytes) -> set[tuple[str, str, str | None, str | None]]:
        return {(p['name'], p['version'], p.get('source'), p.get('checksum'))
                for p in tomllib.loads(raw.decode())['package']}
    original_ids, actual_ids = package_ids(original_lock), package_ids(current_lock)
    old_names = {p[0] for p in original_ids}
    if not original_ids <= actual_ids or any(p not in original_ids and p[0] in old_names
                                            for p in actual_ids):
        raise ValueError('allocator build changed an original dependency identity')
    write(args.output / 'cargo-metadata.json', metadata)
    command = ['/tmp/gfb-opt-build-env/bin/maturin', 'build', '--release', '--locked',
               '--offline', '--manifest-path', str(manifest), '--interpreter', sys.executable,
               '--out', str(args.output / 'wheels')]
    started = time.monotonic()
    build_log(command, args.output, seconds=cfg['trial_seconds'], env=env)
    wheels = list((args.output / 'wheels').glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected exactly one private wheel')
    site = args.output / 'site'
    site.mkdir()
    with zipfile.ZipFile(wheels[0]) as archive:
        for name in archive.namelist():
            if not (site / name).resolve().is_relative_to(site.resolve()):
                raise ValueError('unsafe wheel path')
        archive.extractall(site)
    item = {'schema_version': 1, 'allocator': args.allocator,
            'allocator_proof': 'private source declaration + pinned lock + loaded source/binary identity',
            'wheel': str(wheels[0]), 'wheel_sha256': sha(wheels[0]), 'site': str(site),
            'native_identity': native_probe(site), 'source_sha256': source_hash(private),
            'manifest_sha256': sha(manifest), 'lock_sha256': sha(private / 'Cargo.lock'),
            'allocator_features': cfg['allocator_features'].get(args.allocator, []),
            'default_features': True, 'release_profile': {'opt_level': 3, 'debug': 0},
            'build_seconds': time.monotonic() - started, 'command': command,
            'environment_sanitization': sanitization_proof(),
            'environment': {k: env.get(k) for k in ('CARGO_HOME', 'CARGO_TARGET_DIR', 'RUSTFLAGS',
                                                   'LD_PRELOAD', 'PYTHONPATH')}}
    write(args.output / 'variant.json', item)
    verify_variant(args.output / 'variant.json', cfg)


def typed_build(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment()
    env['CARGO_TARGET_DIR'] = str(args.output / 'target')
    source = typed_source_hash()
    env['GFB_EXPERIMENT_SOURCE_SHA256'] = source
    command = ['cargo', 'build', '--release', '--offline', '--locked', '--manifest-path',
               str(HERE / 'rust/Cargo.toml')]
    if args.no_ap:
        command += ['--no-default-features']
    if args.counter:
        command += ['--features', 'allocation_counts']
    started = time.monotonic()
    build_log(command, args.output, env=env, seconds=cfg['trial_seconds'])
    binary = args.output / 'target/release/gfb-json-bench'
    identity = json.loads(checked([str(binary), '--identity'], seconds=10))
    if typed_source_hash() != source or identity != {
            'allocator': 'System', 'arbitrary_precision': not args.no_ap,
            'allocation_counts': args.counter, 'source_sha256': source,
            'contract': 'typed-journal-experiment-v1'}:
        raise ValueError('typed build source/feature identity mismatch')
    write(args.output / 'typed-build.json', {'schema_version': 1, 'binary': str(binary),
          'binary_sha256': sha(binary), 'identity': identity, 'source_sha256': source,
          'build_seconds': time.monotonic() - started, 'command': command,
          'environment_sanitization': sanitization_proof(),
          'rustc': checked(['rustc', '--version'], seconds=10).strip(),
          'environment': {key: env[key] for key in (
              'CARGO_HOME', 'CARGO_TARGET_DIR', 'CARGO_PROFILE_RELEASE_OPT_LEVEL',
              'CARGO_PROFILE_RELEASE_DEBUG', 'CARGO_PROFILE_RELEASE_STRIP',
              'CARGO_INCREMENTAL', 'GFB_EXPERIMENT_SOURCE_SHA256')}})


def verify_typed_build(path: Path) -> dict[str, Any]:
    build = json.loads(path.read_text())
    if sha(Path(build['binary'])) != build['binary_sha256']:
        raise ValueError('typed binary bytes changed')
    identity = json.loads(checked([build['binary'], '--identity'], seconds=10))
    if identity != build['identity'] or identity['source_sha256'] != typed_source_hash():
        raise ValueError('typed binary/source/feature mismatch')
    return build


def typed_trial_plan(args: argparse.Namespace, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    builds = [json.loads(path.read_text()) for path in args.typed_build]
    wanted_counter = args.command == 'typed-counter'
    lookup = {}
    for build in builds:
        if build['identity']['allocation_counts'] != wanted_counter:
            raise ValueError('counter builds must be separate from primary trials')
        ap = build['identity']['arbitrary_precision']
        if ap in lookup:
            raise ValueError('duplicate typed feature build')
        lookup[ap] = build
    if set(lookup) != {True, False}:
        raise ValueError('both AP and isolated no-AP builds required')
    result = []
    trials = [0] if wanted_counter else range(-cfg['warmups'], cfg['trials'])
    variants = ['A', 'B', 'C', 'D']
    for trial in trials:
        offset = max(0, trial) % 4
        for variant in variants[offset:] + variants[:offset]:
            build = lookup[variant != 'D']
            directory = args.output / f'{trial}-{variant}'
            command = [sys.executable, str(ROOT / 'tools/benchmark_performance.py'),
                       '--output', str(directory), '--seconds', str(cfg['trial_seconds']), '--',
                       build['binary'], '--inventory', str(args.inventory), '--variant', variant,
                       '--mode', 'counter' if wanted_counter else 'measure',
                       '--seconds', str(cfg['trial_seconds']), '--memory-budget-mib',
                       str(cfg['memory_budget_mib'])]
            result.append({'variant': variant, 'trial': trial, 'directory': str(directory),
                           'binary_sha256': build['binary_sha256'], 'command': command})
    return result


def typed_validate(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    frozen = verify_freeze(args.freeze, cfg)
    frozen_journals(frozen, args.inventory)
    build = verify_typed_build(args.typed_build)
    if not build['identity']['arbitrary_precision'] or build['identity']['allocation_counts']:
        raise ValueError('oracle requires a primary AP build')
    command = [sys.executable, str(ROOT / 'tools/benchmark_performance.py'), '--output',
               str(args.output), '--seconds', str(cfg['trial_seconds']), '--',
               sys.executable, str(HERE / 'typed_oracle.py'), '--binary', build['binary'],
               '--inventory', str(args.inventory), '--variant', args.variant,
               '--output', str(args.output / 'oracle.json'), '--seconds', str(cfg['trial_seconds'])]
    completed = run_observer(command, cleanup_seconds=cfg['cleanup_seconds'])
    if completed.returncode:
        raise ValueError('typed correctness oracle failed')
    completed_oracle(args.output / 'oracle.json')
    verify_freeze(args.freeze, cfg)


def completed_oracle(path: Path) -> dict[str, Any]:
    """Helper completion is eligible only after its observer and owner finish."""
    try:
        measurement = json.loads((path.parent / 'measurement.json').read_text())
        owner = json.loads((path.parent / 'owner-lifecycle.json').read_text())
        command = measurement['command']
        output = Path(command[command.index('--output') + 1]).resolve()
        if (measurement['status'] != 'completed' or measurement['returncode'] != 0
                or measurement['cancellation'] is not None
                or owner['observer_returncode'] != 0 or owner['cancellation'] is not None
                or owner['cleanup_deadline_exceeded'] is not False
                or owner['cleanup_owner_joined'] is not True
                or output != path.resolve()):
            raise ValueError('unsuccessful or unrelated enclosing observer')
        return json.loads(path.read_text())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError('oracle requires a completed, uncancelled enclosing observer and owner') from error


def typed_run(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    frozen = verify_freeze(args.freeze, cfg, require_code=True)
    builds = [verify_typed_build(path) for path in args.typed_build]
    inventory_hash = sha(args.inventory)
    frozen_journals(frozen, args.inventory)
    validation = [completed_oracle(path) for path in args.validation]
    if {v['variant'] for v in validation} != {'B', 'C'} or len(validation) != 2:
        raise ValueError('complete B/C correctness oracles required before timed/counter runs')
    for item in validation:
        if (item['status'] != 'PASS' or item['files'] != 13568 or item['records'] != 534640
                or item['new_rejections'] != 0 or item['inventory_sha256'] != inventory_hash):
            raise ValueError('invalid/incomplete correctness oracle')
        if item['source_sha256'] != typed_source_hash():
            raise ValueError('correctness oracle used a different typed source')
        expected_identity = {'allocator': 'System', 'arbitrary_precision': True,
                             'allocation_counts': False, 'source_sha256': typed_source_hash(),
                             'contract': 'typed-journal-experiment-v1'}
        if (not item.get('completed') or item.get('binary_identity') != expected_identity
                or sha(Path(item['binary'])) != item['binary_sha256']
                or item['helper_sha256'] != sha(HERE / 'typed_oracle.py')
                or json.loads(checked([item['binary'], '--identity'], seconds=10)) != expected_identity):
            raise ValueError('correctness oracle binary/features/completion proof mismatch')
        if args.command == 'typed-run' and item['binary_sha256'] not in {
                build['binary_sha256'] for build in builds
                if build['identity']['arbitrary_precision'] and not build['identity']['allocation_counts']}:
            raise ValueError('correctness oracle did not use the primary AP binary')
    if validation[0]['canonical_records_sha256'] != validation[1]['canonical_records_sha256']:
        raise ValueError('B/C canonical correctness identity mismatch')
    plan = typed_trial_plan(args, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'plan.json', plan)
    results = []
    for item in plan:
        completed = run_observer(item['command'], cleanup_seconds=cfg['cleanup_seconds'])
        directory = Path(item['directory'])
        item['measurement'] = json.loads((directory / 'measurement.json').read_text())
        if completed.returncode or item['measurement']['status'] != 'completed':
            item['failure'] = 'typed workload/observer failed; condition invalid'
        else:
            item['workload'] = json.loads((directory / 'stdout.log').read_text())
            if (item['workload']['files'] != 13568 or item['workload']['records'] != 534640
                    or item['workload']['variant'] != item['variant']):
                item['failure'] = 'typed workload count/variant mismatch'
        results.append(item)
        write(args.output / 'results.json', results)
        if 'failure' in item:
            raise ValueError(item['failure'])
    for build in builds:
        if sha(Path(build['binary'])) != build['binary_sha256']:
            raise ValueError('typed binary changed during matrix')
    verify_freeze(args.freeze, cfg, require_code=True)


def output_identity(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in ('dataset_content_sha256', 'dataset_manifest_sha256',
                                      'arrays', 'numeric_archive_sha256', 'sample_identity_sha256')}


def trial_plan(args: argparse.Namespace, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [json.loads(path.read_text()) for path in args.variant]
    if len({v['allocator'] for v in variants}) != len(variants):
        raise ValueError('duplicate allocator variant')
    if args.command in ('run', 'allocator-plan') and {v['allocator'] for v in variants} != set(ALLOCATORS):
        raise ValueError('primary matrix requires all three allocators')
    trials = [-1] if args.command == 'validate' else range(-cfg['warmups'], cfg['trials'])
    plan = []
    for workers in cfg['worker_counts']:
        for trial in trials:
            offset = max(0, trial) % len(variants)
            for variant in variants[offset:] + variants[:offset]:
                directory = args.output / f'{workers}-{trial}-{variant["allocator"]}'
                env = environment(Path(variant['site']))
                command = [sys.executable, str(ROOT / 'tools/benchmark_performance.py'),
                           '--output', str(directory), '--seconds', str(cfg['trial_seconds']), '--',
                           'env', '-u', 'VIRTUAL_ENV', '-u', 'LD_PRELOAD',
                           *[f'{k}={env[k]}' for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                              'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')],
                           sys.executable, str(HERE / 'workload_child.py'), 'preparation',
                           '--variant', 'rust', '--source', str(ROOT / 'src'), '--dataset',
                           str(ROOT / cfg['dataset']), '--encoder', 'history', '--memory-mib', '2048',
                           '--chunk-mib', '16', '--workers', str(workers), '--max-in-flight',
                           str(workers), '--seconds', str(cfg['trial_seconds']), '--native-sha256',
                           variant['native_identity']['extension_sha256']]
                plan.append({'allocator': variant['allocator'], 'workers': workers, 'trial': trial,
                             'directory': str(directory), 'command': command,
                             'native_identity': variant['native_identity']})
    return plan


def run_trials(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    verify_freeze(args.freeze, cfg, require_code=args.command == 'run')
    for path in args.variant:
        verify_variant(path, cfg)
    plan = trial_plan(args, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'plan.json', plan)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    results = []
    for item in plan:
        # Observer owns deadlines, descendants, cancellation and scratch cleanup.
        completed = run_observer(item['command'], cleanup_seconds=cfg['cleanup_seconds'])
        directory = Path(item['directory'])
        item['measurement'] = json.loads((directory / 'measurement.json').read_text())
        if completed.returncode or item['measurement']['status'] != 'completed':
            item['failure'] = 'workload or observer failed; condition invalid'
        else:
            result = json.loads((directory / 'workload.json').read_text())
            item['workload'] = result
            item['post_release'] = json.loads((directory / 'post-release.json').read_text())
            proof = output_identity(result)
            reference = proof if reference is None else reference
            if proof != reference or result['native_identity'] != item['native_identity']:
                item['failure'] = 'output or loaded binary identity mismatch'
        results.append(item)
        write(args.output / 'results.json', results)
        if 'failure' in item:
            raise ValueError(item['failure'])
    verify_freeze(args.freeze, cfg, require_code=args.command == 'run')
    write(args.output / 'output-identity.json', reference)


def summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {'count': 0, 'median': None, 'min': None, 'max': None, 'iqr': None}
    quartiles = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else []
    return {'count': len(values), 'median': statistics.median(values), 'min': min(values),
            'max': max(values), 'iqr': quartiles[2] - quartiles[0] if quartiles else None}


def report(args: argparse.Namespace) -> dict[str, Any]:
    rows = json.loads(args.results.read_text())
    groups = {}
    typed = bool(rows) and 'variant' in rows[0]
    labels = sorted({(row['variant'], None) if typed else (row['allocator'], row['workers'])
                     for row in rows})
    for label, workers in labels:
        selected = [row for row in rows if row.get('variant' if typed else 'allocator') == label
                    and row.get('workers') == workers and row['trial'] >= 0]
        good = [row for row in selected if 'failure' not in row]
        fields: dict[str, tuple[str, ...]] = {
                  'wall_seconds': ('wall_seconds',),
                  **{key: ('cpu', key) for key in
                     ('aggregate_seconds', 'user_seconds', 'system_seconds')},
                  **{key: ('peaks', key) for key in
                     ('rss_bytes', 'pss_bytes', 'scratch_logical_bytes', 'scratch_allocated_bytes',
                      'output_logical_bytes', 'output_allocated_bytes')},
                  **{key: ('sampled_io_lower_bounds', key) for key in
                     ('rchar', 'wchar', 'read_bytes', 'write_bytes')}}
        group: dict[str, Any] = {}
        for name, path in fields.items():
            values = []
            for row in good:
                value = row['measurement']
                for key in path:
                    value = value[key]
                if value is not None:
                    values.append(value)
            group[name] = summary(values)
        group.update(failures=len(selected) - len(good), measured_trials=len(selected),
                     five_trials_complete=len(good) == 5 and len(selected) == 5)
        if typed:
            phase_names = sorted({name for row in good for name in row['workload']['phases']})
            group['phase_wall_seconds'] = {name: summary([
                row['workload']['phases'][name]['wall_seconds'] for row in good
                if name in row['workload']['phases']]) for name in phase_names}
            group['coarse_managed_peak_bytes'] = summary([
                row['workload']['coarse_managed_peak_bytes'] for row in good])
            group['allocation_diagnostics'] = {
                name: {key: summary([row['workload']['phases'][name][key] for row in good
                                     if key in row['workload']['phases'].get(name, {})])
                       for key in ('allocation_requests', 'requested_bytes', 'deallocation_requests',
                                   'reallocation_requests')}
                for name in phase_names
                if any('allocation_requests' in row['workload']['phases'].get(name, {}) for row in good)}
        for key in ('Rss', 'Pss'):
            resident = []
            for row in good:
                if typed:
                    raw = row['workload'].get('post_release_smaps_rollup')
                else:
                    raw = row.get('post_release', {}).get('proc', {}).get('smaps_rollup')
                if isinstance(raw, str):
                    for line in raw.splitlines():
                        if line.startswith(key + ':'):
                            resident.append(int(line.split()[1]) * 1024)
            group['post_release_' + key.lower() + '_bytes'] = summary(resident)
        for name, value in tuple(group.items()):
            if name.endswith('_bytes') and isinstance(value, dict) and 'median' in value:
                group[name.removesuffix('_bytes') + '_mib'] = {
                    key: (number / 1048576 if number is not None and key != 'count' else number)
                    for key, number in value.items()}
        groups[label if typed else f'{label}-{workers}'] = group
    for label, workers in labels:
        name = label if typed else f'{label}-{workers}'
        base = groups.get('A' if typed else f'system-{workers}', {}).get('wall_seconds', {}).get('median')
        median = groups[name]['wall_seconds']['median']
        groups[name]['wall_ratio_to_A' if typed else 'wall_ratio_to_system'] = (
            median / base if median is not None and base else None)
    return {'schema_version': 1, 'groups': groups,
            'units': {'durations': 'seconds', 'bytes': 'bytes', 'mib': '1048576 bytes'},
            'scope': 'whole observer wall; nested worker diagnostics remain separate in raw results',
            'acceptance': 'root decision required; no production adoption'}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, default=HERE / 'experiment.toml')
    result.add_argument('--execute', action='store_true')
    result.add_argument('--protected-fits', type=Path,
                        default=ROOT / 'runs/assessments/native-migration-original-six.json')
    sub = result.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'build', 'validate', 'run', 'allocator-plan', 'typed-plan',
                 'typed-build', 'typed-validate', 'typed-run', 'typed-counter', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path)
        if name == 'build':
            p.add_argument('--allocator', choices=ALLOCATORS, required=True)
        if name in ('validate', 'run', 'allocator-plan'):
            p.add_argument('--variant', type=Path, action='append', required=True)
            p.add_argument('--freeze', type=Path)
            p.add_argument('--reference', type=Path)
        if name == 'report':
            p.add_argument('--results', type=Path, required=True)
        if name == 'typed-build':
            p.add_argument('--no-ap', action='store_true')
            p.add_argument('--counter', action='store_true')
        if name in ('typed-run', 'typed-counter', 'typed-plan'):
            p.add_argument('--typed-build', type=Path, action='append', required=True)
            p.add_argument('--inventory', type=Path, required=True)
            p.add_argument('--freeze', type=Path)
            p.add_argument('--validation', type=Path, action='append', default=[])
        if name == 'typed-validate':
            p.add_argument('--typed-build', type=Path, required=True)
            p.add_argument('--inventory', type=Path, required=True)
            p.add_argument('--freeze', type=Path, required=True)
            p.add_argument('--variant', choices=('B', 'C'), required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    cfg = config(args.config)
    if args.output is not None:
        args.output = args.output.resolve()
        if not (args.output.is_relative_to(Path('/tmp'))
                or args.output.is_relative_to(ROOT / 'runs/assessments')):
            raise ValueError('experiment output must be under /tmp or runs/assessments')
    if args.command == 'report':
        print(json.dumps(report(args), indent=2))
        return
    if args.output is None:
        raise ValueError('--output required')
    if args.command == 'typed-plan':
        print(json.dumps({'execute': False, 'plan': typed_trial_plan(args, cfg)}, indent=2))
        return
    if not args.execute or args.command == 'allocator-plan':
        plan = trial_plan(args, cfg) if args.command in ('validate', 'run', 'allocator-plan') else (
            typed_trial_plan(args, cfg) if args.command in ('typed-run', 'typed-counter') else [])
        print(json.dumps({'execute': False, 'command': args.command, 'config': cfg,
                          'output': str(args.output), 'plan': plan}, indent=2))
        return
    if args.command == 'freeze':
        freeze(args, cfg)
    elif args.command == 'build':
        build(args, cfg)
    elif args.command == 'typed-build':
        typed_build(args, cfg)
    elif args.command == 'typed-validate':
        typed_validate(args, cfg)
    elif args.command in ('typed-run', 'typed-counter'):
        if args.freeze is None:
            raise ValueError('freeze required')
        typed_run(args, cfg)
    else:
        if args.freeze is None or (args.command == 'run' and args.reference is None):
            raise ValueError('freeze required; measured run also requires validated output reference')
        run_trials(args, cfg)


if __name__ == '__main__':
    main()
