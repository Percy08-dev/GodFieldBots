"""Frozen-input workloads and sequential paired driver for ADR0010/ADR0011.

No fits, PPO updates, live collection, or source mutations are performed.
The driver and observer are stdlib-only; numerical imports happen in workload
children. Example (inspect plan first; add --execute to run)::

  env -u VIRTUAL_ENV .venv/bin/python tools/benchmark_workload.py paired \
    --kind preparation --output /tmp/gfb-pairs --encoder history

ADR0011 native comparison (one matching 2GiB/16MiB cell, no budget matrix)::

  env -u VIRTUAL_ENV .venv/bin/python tools/benchmark_workload.py paired \
    --kind preparation --comparison native --output /tmp/gfb-native-pairs

The native package may be installed in this interpreter, or supplied with
--native-pythonpath. The declared Python source always precedes those paths;
every imported gf_common/gf_models/gf_training module must originate there.

The preparation plan includes the full old/new standard comparison and all new
2/8GiB x 16/64MiB cells with cache512MiB. Standard old/new has separately reported
first-startup/warmup then three alternating pairs. Additional budget cells have
three fresh-process measured trials using the already-warm corpus. Fixed/replay use generation
123 and the first eight validation17 schedule groups. They are fixed diagnostics,
not population percentiles. Use --kind overhead for a small sampling on/off test.
All runs inherit the same interpreter, thread settings and warm OS cache; no
cache flush or cold-cache claim is made. Source read/hash and validation overlap
inside the scanner and are never relabeled as disjoint independent timings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN = 'ab5d4913fda45795d1cd5aa02a4a9c4c3464268298c4d3ea49d56d4909df770f'
NATIVE_REFERENCE = Path('/tmp/gfb-native-reference-42ce5e6')


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def source_modules(kind: str, *, native: bool = False) -> tuple[str, ...]:
    common = ('gf_common', 'gf_common.domain', 'gf_models', 'gf_models.config', 'gf_models.encoder')
    if kind == 'preparation':
        modules = (*common, 'gf_training', 'gf_training.fixed_data', 'gf_training.q_data',
                   'gf_training.journal', 'gf_training.lineage')
        return (*modules, 'gf_training.native_backend', 'gf_training.native_preparation') if native else modules
    return (*common, 'gf_models.mlp', 'gf_models.mlp.policy', 'gf_models.mlp.features',
            'gf_models.mlp.model', 'gf_models.mlp.arrays')


def source_probe(source: Path, modules: tuple[str, ...]) -> dict[str, Any]:
    """Run only in a fresh probe child with the same source path as a trial."""
    import importlib

    source = source.resolve()
    for name in modules:
        importlib.import_module(name)
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.split('.', 1)[0] not in ('gf_common', 'gf_models', 'gf_training'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None or not Path(filename).resolve().is_relative_to(source):
            raise ValueError(f'shadowed source import {name}: {filename}; expected {source}')
        path = Path(filename).resolve()
        result[name] = {'path': str(path), 'sha256': file_hash(path)}
    return result


def native_identity() -> dict[str, Any]:
    """Use the backend's actual loaded-binary proof, never Git as a substitute."""
    from gf_training.native_backend import binary_identity

    return dict(binary_identity())


def source_environment(source: Path, native_paths: list[Path] | None = None,
                       *, native: bool = False) -> dict[str, str]:
    paths = [str(source.resolve())]
    if native:
        paths.extend(str(path.resolve()) for path in native_paths or [])
        paths.extend(part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep) if part)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)),
                       PYTHONDONTWRITEBYTECODE='1')
    environment.pop('VIRTUAL_ENV', None)
    return environment


def preflight_source(source: Path, kind: str, *, native: bool = False,
                     native_paths: list[Path] | None = None) -> dict[str, Any]:
    """Fail before expensive work if a reference can fall back to installed code."""
    source = source.resolve()
    modules = source_modules(kind, native=native)
    expected = {}
    for name in modules:
        path = source / Path(*name.split('.'))
        path = path / '__init__.py' if path.is_dir() else path.with_suffix('.py')
        if not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError(f'missing/incomplete source tree: {path}')
        expected[name] = {'path': str(path.resolve()), 'sha256': file_hash(path)}
    environment = source_environment(source, native_paths, native=native)
    command = [sys.executable, str(Path(__file__).resolve()), '_source-probe',
               '--source', str(source), '--kind', kind]
    if native:
        command.append('--native')
    completed = subprocess.run(
        command,
        env=environment, check=False, text=True, capture_output=True, timeout=20,
    )
    if completed.returncode:
        raise ValueError(f'source preflight failed: {source}: {completed.stderr[-8192:]}')
    try:
        decoded = json.loads(completed.stdout)
        actual = decoded['source_imports'] if native else decoded
        if not isinstance(actual, dict):
            raise TypeError('source import proof must be an object')
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f'source preflight returned malformed proof: {source}') from error
    for name, expected_proof in expected.items():
        if actual.get(name) != expected_proof:
            raise ValueError(f'shadowed/changed source import: {name}: {actual.get(name)} != {expected_proof}')
    proof: dict[str, Any] = {'source': str(source), 'expected': expected, 'actual_imports': actual}
    if native:
        identity = decoded.get('native_identity')
        if not isinstance(identity, dict) or not identity.get('extension_sha256') or not identity.get('build'):
            raise ValueError('native source preflight returned no loaded binary/build identity')
        proof['native_identity'] = identity
    return proof


def array_proof(arrays: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, array in arrays.items():
        hasher = hashlib.sha256()
        # Row windows bound contiguous copies for noncontiguous mapping views.
        for offset in range(0, len(array), 128):
            hasher.update(array[offset:offset + 128].tobytes(order='C'))
        result[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                        'sha256': hasher.hexdigest()}
    return result


def archive_proof(path: Path) -> dict[str, Any]:
    """Hash NPY payloads directly, without loading a full matrix from the ZIP."""
    import numpy as np

    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f'unsupported numeric header {version}')
                if fortran or dtype.hasobject:
                    raise ValueError('numeric archive requires plain C-order arrays')
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(block)
                result[name.removesuffix('.npy')] = {
                    'shape': list(shape), 'dtype': str(dtype), 'sha256': hasher.hexdigest()}
    return result


def source_inventory(path: Path) -> dict[str, Any]:
    """One static stat inventory outside trials, not an eligibility validator.

    Only schema-declared evidence files are enumerated. Sizes are unique input
    bytes, not actual read traffic; every measured reader still validates bytes.
    """
    dataset = json.loads(path.read_text())
    files = {path.resolve()}
    for collection in dataset['collections']:
        directory = Path(collection['directory'])
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        files.update((manifest_path.resolve(), (directory / 'report.json').resolve()))
        for episode in manifest['episodes']:
            for name in ('player-1.jsonl', 'player-2.jsonl',
                         'replay-player-1.json', 'replay-player-2.json'):
                files.add((directory / episode['episode_id'] / name).resolve())
    for name in dataset['ppo_runs']:
        parent = Path(name)
        files.update((parent / item).resolve() for item in ('run.json', 'latest.json', 'artifact-registry.json'))
        state = json.loads((parent / 'run.json').read_text())
        registry = json.loads((parent / 'artifact-registry.json').read_text())
        for number in range(state['updates'] + 1):
            files.add((parent / 'generations' / f'{number:06}.json').resolve())
            artifacts = parent / 'artifacts' / f'{number:06}'
            files.add((artifacts / 'inference.json').resolve())
            if registry[str(number)]['status'] == 'retained':
                files.update((artifacts / item).resolve() for item in
                             ('weights.json', 'weights.npz', 'resume.json', 'resume.npz'))
            if number:
                update_path = parent / 'updates' / f'{number:06}.json'
                update = json.loads(update_path.read_text())
                files.add(update_path.resolve())
                files.add((parent / update['numeric'] / 'rollout.json').resolve())
    entries = [{'path': str(file), 'logical_bytes': file.stat().st_size} for file in sorted(files)]
    encoded = json.dumps(entries, ensure_ascii=False, separators=(',', ':')).encode()
    return {'schema_version': 1, 'scope': 'declared frozen collection and parent-chain evidence; stat once outside trials',
            'unique_files': len(entries), 'unique_logical_bytes': sum(item['logical_bytes'] for item in entries),
            'inventory_sha256': hashlib.sha256(encoded).hexdigest(), 'entries': entries,
            'limitations': 'sizes are test conditions, not read traffic or an eligibility proof; source hashes validated every trial'}


def source_read_metrics(diagnostics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """Adapt the scanner's actual byte counter without conflating I/O scopes."""
    count = diagnostics.get('read_bytes')
    return {'scanned_source_read_bytes': count,
            'scanned_source_read_mib_per_second': count / (1024 * 1024) / elapsed if count is not None else None,
            'scanned_source_read_unavailable_reason': (None if count is not None else
                diagnostics.get('read_bytes_unavailable', 'reader does not expose read_bytes')),
            'scanned_source_read_scope': 'initial reader-accounted source reads/hashes; '
                'excludes final mutable-control rehashes; distinct from static unique inventory and kernel rchar'}


def native_transfer_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Only report owner counters; absence is unavailable, never an inferred zero."""
    return {key: {'value': diagnostics.get(key),
                  'unavailable_reason': None if diagnostics.get(key) is not None else
                  f'preparation owner does not expose {key}'}
            for key in ('ffi_data_calls', 'payload_copy_stats')}


def preparation_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Keep the optimized reference compatible with its older config signature."""
    settings: dict[str, Any] = {'memory_budget_mib': args.memory_mib, 'chunk_target_mib': args.chunk_mib,
                              'cache_budget_mib': 512}
    if args.variant == 'rust':
        settings.update(backend='rust', workers=args.workers, max_in_flight=args.max_in_flight)
    return settings


def preparation(args: argparse.Namespace) -> dict[str, Any]:
    entered = time.monotonic()
    import numpy as np

    from gf_models.config import EncoderConfig

    config = EncoderConfig(history='public_aggregate' if args.encoder == 'history' else 'none')
    started = entered
    phases = {'imports': time.monotonic() - entered}
    deadline = args.command_started + args.seconds
    scratch = Path(os.environ['GFB_BENCHMARK_SCRATCH'])
    target = scratch / 'arrays.npz'
    diagnostics: dict[str, Any]
    identities: str | None
    if args.variant == 'old':
        from gf_models.encoder import encode
        from gf_training.fixed_data import inspect_dataset
        tick = time.monotonic()
        dataset, train, validation = inspect_dataset(args.dataset, deadline=deadline)
        phases['read_hash_validation'] = time.monotonic() - tick
        tick = time.monotonic()
        x = np.stack([encode(item.value, item.action, config) for item in train])
        validation_x = np.stack([encode(item.value, item.action, config) for item in validation])
        arrays = {'x': x, 'y': np.asarray([item.target for item in train], dtype=np.float64),
                  'validation_x': validation_x,
                  'validation_y': np.asarray([item.target for item in validation], dtype=np.float64)}
        phases['numeric_encoding'] = time.monotonic() - tick
        tick = time.monotonic()
        identity = hashlib.sha256()
        for split, samples in (('train', train), ('validation', validation)):
            for sample in samples:
                identity.update((json.dumps([split, *sample.identity], ensure_ascii=False,
                                            separators=(',', ':')) + '\n').encode())
        identities = identity.hexdigest()
        phases['identity_proof'] = time.monotonic() - tick
        diagnostics = {'managed_budget': 'unsupported by old baseline',
                       'read_bytes': None, 'read_bytes_unavailable': 'legacy reader does not count bytes'}
        owner = None
    else:
        from gf_common.preprocessing import PreparationConfig
        from gf_training.fixed_data import prepare_fixed_inputs
        tick = time.monotonic()
        owner = prepare_fixed_inputs(args.dataset, config,
                                     preprocessing=PreparationConfig(**preparation_settings(args)),
                                     deadline=deadline, wall_deadline=deadline)
        dataset = owner.dataset
        phases['read_hash_validation_encoding'] = time.monotonic() - tick
        diagnostics = owner.diagnostics
        identities = diagnostics.get('sample_identity_sha256')
    try:
        if (dataset['content_sha256'] != FROZEN or dataset['train_decisions'] != 81523
                or dataset['validation_decisions'] != 2260):
            raise ValueError('workload requires the frozen full 83,783-row dataset')
        tick = time.monotonic()
        if owner is None:
            np.savez(target, x=arrays['x'], y=arrays['y'],
                     validation_x=arrays['validation_x'], validation_y=arrays['validation_y'])
        else:
            owner.write_numeric(target)
        phases['numeric_staging'] = time.monotonic() - tick
        tick = time.monotonic()
        proof = archive_proof(target)
        archive_hash = file_hash(target)
        archive_bytes = target.stat().st_size
        phases['output_proof'] = time.monotonic() - tick
        tick = time.monotonic()
        if owner is not None:
            owner.finalize()
        target.unlink()
        phases['final_check_cleanup'] = time.monotonic() - tick
    except BaseException:
        if owner is not None:
            owner.abort()
        raise
    elapsed = time.monotonic() - started
    # These wall phases are disjoint. Scanner's nested diagnostics remain nested.
    return {'kind': 'preparation', 'variant': args.variant, 'encoder': args.encoder,
            'dataset_content_sha256': dataset['content_sha256'],
            'dataset_manifest_sha256': file_hash(args.dataset), 'arrays': proof,
            'numeric_archive_sha256': archive_hash, 'sample_identity_sha256': identities,
            'numeric_archive_bytes': archive_bytes,
            'accepted_actions': 83783, 'internal_preparation_wall_seconds': elapsed,
            'internal_accepted_actions_per_second': 83783 / elapsed, 'phases': phases,
            'timing_scope': 'internal phases exclude function-return object teardown; '
                            'primary timings use observer total command wall',
            'unattributed_wall_seconds': max(0., elapsed - sum(phases.values())),
            **source_read_metrics(diagnostics, elapsed),
            'native_transfer_metrics': native_transfer_metrics(diagnostics) if args.variant == 'rust' else None,
            'nested_preparation_diagnostics': diagnostics,
            'config': {'memory_mib': args.memory_mib, 'chunk_mib': args.chunk_mib,
                       'cache_mib': 512,
                       **({'backend': 'rust', 'workers': args.workers,
                           'max_in_flight': args.max_in_flight} if args.variant == 'rust' else {})}
                      if args.variant != 'old' else None,
            'unavailable_phases': ['native_fit: not applicable; no fit is run',
                                   'read/hash vs validation: interleaved within scanner']}


def policy_workload(args: argparse.Namespace) -> dict[str, Any]:
    entered = time.monotonic()
    import numpy as np

    from gf_common.domain import CandidateSet, DecisionInput
    from gf_models.encoder import encode
    from gf_models.mlp.features import encode_decision
    from gf_models.mlp.policy import MLPPolicy

    phases: dict[str, float] = {'imports': time.monotonic() - entered}
    tick = time.monotonic()
    parent = args.root / 'train-seed17'
    record = json.loads((parent / 'generations/000123.json').read_text())
    policy = MLPPolicy(str(parent / 'artifacts/000123'), record['inference_sha256'])
    phases['load'] = time.monotonic() - tick
    tick = time.monotonic()
    validation = args.root / 'validation17'
    manifest = json.loads((validation / 'manifest.json').read_text())
    decisions = []
    source_proof = {}
    for group in manifest['schedule'][:8]:
        for spec in group['episodes']:
            path = validation / spec['episode_id'] / f'player-{spec["learner_viewer"]}.jsonl'
            raw = path.read_bytes()
            source_proof[str(path.relative_to(validation))] = hashlib.sha256(raw).hexdigest()
            for line in raw.splitlines():
                row = json.loads(line)
                if row['kind'] == 'decision':
                    decisions.append(row)
    if args.mode == 'fixed':
        decisions.sort(key=lambda row: len(row['candidates']['actions']))
        decisions = [decisions[len(decisions) // 2], decisions[int(.95 * (len(decisions) - 1))]]
        if [len(row['candidates']['actions']) for row in decisions] != [44, 275]:
            raise ValueError('frozen 44/275 case selection changed')
    elif args.replay_limit:
        decisions = decisions[:args.replay_limit]
    cases = []
    for row in decisions:
        raw = row['input']
        value = DecisionInput(raw['observation'], tuple(raw['events']), raw['history_end'],
                              tuple(raw['typed_metadata']), raw['catalog'], raw['action_options'])
        cs = row['candidates']
        candidates = CandidateSet(tuple(cs['actions']), **{k: v for k, v in cs.items() if k != 'actions'})
        cases.append((value, candidates))
    phases['input_read_and_decode'] = time.monotonic() - tick
    tick = time.monotonic()
    outputs: list[dict[str, Any]] = []
    for value, candidates in cases:
        actor, state = encode_decision(value, candidates.actions, policy.config.encoder)
        outputs.append({'candidates': len(candidates.actions), 'events': len(value.events),
                        'arrays': array_proof({'actor': actor, 'critic': state,
                                              'selected_features': encode(value, candidates.actions[0], policy.config.encoder)})})
    phases['encoding_output_proof'] = time.monotonic() - tick
    tick = time.monotonic()
    policy.reset(17)
    for value, candidates in cases:
        policy.select(value, candidates)
    phases['warmup'] = time.monotonic() - tick
    tick = time.monotonic()
    policy.reset(17)
    for (value, candidates), output in zip(cases, outputs, strict=True):
        if args.mode == 'fixed':
            policy.reset(17)
        choices = []
        durations = []
        for _ in range(args.calls if args.mode == 'fixed' else 1):
            before = time.perf_counter()
            choice = policy.select(value, candidates)
            durations.append(time.perf_counter() - before)
            choices.append(repr(choice))
        output.update(select_seconds=durations,
                      select_p50_seconds=float(np.quantile(durations, .5)),
                      select_p95_seconds=float(np.quantile(durations, .95)),
                      selected_sequence_sha256=hashlib.sha256('\n'.join(choices).encode()).hexdigest())
    phases['policy_select'] = time.monotonic() - tick
    # Forward-only time excludes the independently measured encoder/selection.
    tick = time.monotonic()
    for (value, candidates), output in zip(cases, outputs, strict=True):
        actor, state = encode_decision(value, candidates.actions, policy.config.encoder)
        torch = policy.torch
        inputs = (torch.from_numpy(actor).unsqueeze(0), torch.from_numpy(state).unsqueeze(0),
                  torch.ones((1, len(actor)), dtype=torch.bool))
        times = []
        with torch.no_grad():
            for _ in range(args.calls if args.mode == 'fixed' else 1):
                before = time.perf_counter()
                policy.network(*inputs)
                times.append(time.perf_counter() - before)
        output['forward_seconds'] = times
    phases['forward_diagnostics_including_reencoding'] = time.monotonic() - tick
    if policy.torch.cuda.is_initialized():
        raise RuntimeError('CPU benchmark initialized CUDA')
    durations = [duration for output in outputs for duration in output['select_seconds']]
    return {'kind': args.mode, 'generation': 123, 'sampling_seed': 17, 'cases': outputs,
            'source_journal_sha256': source_proof, 'checkpoint_sha256': record['inference_sha256'],
            'phases': phases, 'select_p50_seconds': float(np.quantile(durations, .5)),
            'select_p95_seconds': float(np.quantile(durations, .95)),
            'accepted_action_scope': 'saved inputs only; selected actions are never submitted',
            'source_bytes': sum((validation / name).stat().st_size for name in source_proof),
            'candidate_slots': sum(item['candidates'] for item in outputs),
            'cuda_initialized': False, 'torch_threads': policy.torch.get_num_threads()}


def overhead() -> dict[str, Any]:
    """Small deterministic CPU/memory/I/O workload, identical with/without samples."""
    scratch = Path(os.environ['GFB_BENCHMARK_SCRATCH'])
    data = bytearray(4 * 1024 * 1024)
    started = time.monotonic()
    count = 0
    hasher = hashlib.sha256()
    for index in range(240):
        data[index] = index
        hasher.update(data)
        count += len(data)
    path = scratch / 'temporary.bin'
    path.write_bytes(data)
    path.read_bytes()
    path.unlink()
    return {'kind': 'overhead', 'sha256': hasher.hexdigest(), 'hashed_bytes': count,
            'wall_seconds': time.monotonic() - started}


def paired(args: argparse.Namespace) -> None:
    """Plan/run independent monitor invocations; no application imports here."""
    monitor = ROOT / 'tools/benchmark_performance.py'
    helper = Path(__file__).resolve()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('seconds must be finite and positive')
    if args.output.exists():
        raise ValueError('paired output directory must not already exist')
    native = getattr(args, 'comparison', 'legacy') == 'native'
    if native and args.kind != 'preparation':
        raise ValueError('native comparison requires preparation')
    reference = args.reference or (NATIVE_REFERENCE if native else Path('/tmp/gfb-performance-reference'))
    native_paths = getattr(args, 'native_pythonpath', [])
    if native and (args.workers not in (1, 2, 4) or args.max_in_flight != args.workers):
        raise ValueError('native paired trials require workers/max-in-flight 1/1, 2/2 or 4/4')
    # Both probes run before inventory parsing, output creation, or any trial.
    preflight = ({'python': preflight_source(reference / 'src', args.kind),
                  'rust': preflight_source(ROOT / 'src', args.kind, native=True, native_paths=native_paths)}
                 if native else
                 {'old': preflight_source(reference / 'src', args.kind),
                  'new': preflight_source(ROOT / 'src', args.kind)}
                 if args.kind != 'overhead' else {})
    standard = ('python', 'rust') if native else (
        ('old', 'new') if args.kind != 'overhead' else ('off', 'on'))
    cells: list[tuple[str, tuple[str, ...], int, int]] = [
        ('standard', standard, 2048 if native else 8192, 16 if native else 64)]
    if args.kind == 'preparation' and not native:
        cells.extend((f'memory{memory}-chunk{chunk}', ('new',), memory, chunk)
                     for memory in (2048, 8192) for chunk in (16, 64)
                     if (memory, chunk) != (8192, 64))
    aliases = ({'memory8192-chunk64': {'cell': 'standard', 'variant': 'new',
                                      'reason': 'identical settings; reuse the same three measured trials'}}
               if args.kind == 'preparation' and not native else {})
    plan: list[dict[str, Any]] = []
    for cell, variants, memory, chunk in cells:
        for trial in ((-1, 0, 1, 2) if cell == 'standard' else (0, 1, 2)):
            order = variants if trial % 2 == 0 or (native and trial < 0) else tuple(reversed(variants))
            for variant in order:
                label = f'{cell}-{"warmup" if trial < 0 else trial}-{variant}'
                directory = args.output / label
                command = [sys.executable, str(monitor), '--output', str(directory),
                           '--seconds', str(args.seconds)]
                if variant == 'off':
                    command.append('--no-sampling')
                source = reference / 'src' if variant in ('old', 'python') else ROOT / 'src'
                environment = source_environment(source, native_paths, native=variant == 'rust')
                command += ['--', 'env', '-u', 'VIRTUAL_ENV',
                            f'PYTHONPATH={environment["PYTHONPATH"]}', 'PYTHONDONTWRITEBYTECODE=1',
                            sys.executable, str(helper)]
                if args.kind == 'preparation':
                    command += ['preparation', '--variant', variant, '--dataset', str(args.dataset),
                                '--encoder', args.encoder, '--memory-mib', str(memory),
                                '--chunk-mib', str(chunk), '--seconds', str(args.seconds)]
                    if native:
                        command += ['--source', str(source)]
                    if variant == 'rust':
                        command += ['--workers', str(args.workers),
                                    '--max-in-flight', str(args.max_in_flight),
                                    '--native-sha256', preflight['rust']['native_identity']['extension_sha256']]
                elif args.kind in ('fixed', 'replay'):
                    command += ['policy', '--mode', args.kind, '--root', str(args.root),
                                '--calls', str(args.calls), '--replay-limit', str(args.replay_limit)]
                else:
                    command += ['overhead']
                plan.append({'cell': cell, 'variant': variant, 'trial': trial,
                             'directory': str(directory), 'command': command})
    if not args.execute:
        print(json.dumps({'execute': False, 'plan': plan, 'aliases': aliases,
                          'source_preflight': preflight}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    (args.output / 'aliases.json').write_text(json.dumps(aliases, indent=2))
    (args.output / 'source-preflight.json').write_text(json.dumps(preflight, indent=2))
    inventory = None
    if args.kind == 'preparation':
        inventory_started = time.monotonic()
        inventory = source_inventory(args.dataset)
        inventory['elapsed_seconds_outside_trials'] = time.monotonic() - inventory_started
        (args.output / 'input-inventory.json').write_text(json.dumps(inventory, indent=2))
    measurements: list[dict[str, Any]] = []
    for item in plan:
        completed = subprocess.run(item['command'], check=False)
        directory = Path(item['directory'])
        measurement = json.loads((directory / 'measurement.json').read_text())
        item['measurement'] = measurement
        if inventory is not None:
            item['input_conditions'] = {
                'inventory_sha256': inventory['inventory_sha256'],
                'unique_logical_bytes': inventory['unique_logical_bytes'],
                'unique_input_mib_per_total_command_second': (
                    inventory['unique_logical_bytes'] / (1024 * 1024) / measurement['wall_seconds']),
                'scope': 'static unique inventory, not actual I/O traffic'}
        if completed.returncode:
            item['failure'] = 'workload/monitor failure; never a partial success'
        else:
            item['workload'] = json.loads((directory / 'workload.json').read_text())
            if (native and item['variant'] == 'rust'
                    and item['workload'].get('native_identity') != preflight['rust']['native_identity']):
                item['failure'] = 'native loaded binary identity changed after preflight'
            if args.kind == 'preparation':
                item['primary_accepted_actions_per_second'] = (
                    item['workload']['accepted_actions'] / measurement['wall_seconds'])
        item['primary_timing_scope'] = 'observer total command wall, including object teardown and process exit'
        measurements.append(item)
        (args.output / 'results.json').write_text(json.dumps(measurements, indent=2))
    # Time/resource fields must never enter deterministic output comparisons.
    def output_identity(workload: dict[str, Any]) -> Any:
        if args.kind == 'preparation':
            return {key: workload[key] for key in ('dataset_content_sha256', 'arrays',
                                                   'numeric_archive_sha256', 'sample_identity_sha256')}
        if args.kind in ('fixed', 'replay'):
            return [{key: item[key] for key in ('candidates', 'arrays', 'selected_sequence_sha256')}
                    for item in workload['cases']]
        return workload['sha256']
    identities = [output_identity(item['workload']) for item in measurements if 'workload' in item]
    equivalence = bool(identities) and all(value == identities[0] for value in identities)
    (args.output / 'equivalence.json').write_text(json.dumps(
        {'schema_version': 1, 'exact_output_identity': equivalence,
         'all_trials_completed': all('failure' not in item for item in measurements),
         'not_acceptance': 'independent review and root disposition still required'}, indent=2))
    lines = ['| Cell | Variant | Wall median [min,max] s | CPU median s | RSS peak bytes | Failures |',
             '| --- | --- | --- | --- | --- | --- |']
    for cell, variants, _, _ in cells:
        for variant in variants:
            selected = [item for item in measurements if item['cell'] == cell
                        and item['variant'] == variant and item['trial'] >= 0]
            good = [item['measurement'] for item in selected if 'failure' not in item]
            walls = [item['wall_seconds'] for item in good]
            cpu = [item['cpu']['aggregate_seconds'] for item in good]
            rss = [item['peaks']['rss_bytes'] for item in good if item['peaks']['rss_bytes'] is not None]
            wall_text = (f'{statistics.median(walls):.6f} [{min(walls):.6f},{max(walls):.6f}]'
                         if walls else 'unavailable')
            cpu_text = f'{statistics.median(cpu):.6f}' if cpu else 'unavailable'
            lines.append(f'| {cell} | {variant} | {wall_text} | {cpu_text} | '
                         f'{max(rss) if rss else "unavailable"} | {len(selected) - len(good)} |')
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n\n'
        'Warmup/startup trials are separate in results.json. Cache was not flushed. '
        'Sampling-off retains process ownership and final wait4 accounting. '
        f'Cell aliases: {json.dumps(aliases)}. '
        f'Exact output identity: {equivalence}. Timing trials are not detailed profiling. '
        'See measurement limitations in every trial.\n')
    print('\n'.join(lines))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('preparation')
    prep.add_argument('--variant', choices=('old', 'new', 'python', 'rust'), required=True)
    prep.add_argument('--dataset', type=Path, default=ROOT / 'runs/ppo-comparison-v2/dataset.json')
    prep.add_argument('--encoder', choices=('observation', 'history'), default='history')
    prep.add_argument('--memory-mib', type=int, default=8192)
    prep.add_argument('--chunk-mib', type=int, default=64)
    prep.add_argument('--seconds', type=float, default=7200)
    prep.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    prep.add_argument('--max-in-flight', type=int)
    prep.add_argument('--source', type=Path)
    prep.add_argument('--native-sha256', help=argparse.SUPPRESS)
    policy = sub.add_parser('policy')
    policy.add_argument('--mode', choices=('fixed', 'replay'), default='fixed')
    policy.add_argument('--root', type=Path, default=ROOT / 'runs/ppo-comparison-v2')
    policy.add_argument('--calls', type=int, default=20)
    policy.add_argument('--replay-limit', type=int, default=0)
    sub.add_parser('overhead')
    pairs = sub.add_parser('paired')
    pairs.add_argument('--kind', choices=('preparation', 'fixed', 'replay', 'overhead'), required=True)
    pairs.add_argument('--output', type=Path, required=True)
    pairs.add_argument('--comparison', choices=('legacy', 'native'), default='legacy')
    pairs.add_argument('--reference', type=Path)
    pairs.add_argument('--native-pythonpath', type=Path, action='append', default=[])
    pairs.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    pairs.add_argument('--max-in-flight', type=int)
    pairs.add_argument('--dataset', type=Path, default=ROOT / 'runs/ppo-comparison-v2/dataset.json')
    pairs.add_argument('--root', type=Path, default=ROOT / 'runs/ppo-comparison-v2')
    pairs.add_argument('--encoder', choices=('observation', 'history'), default='history')
    pairs.add_argument('--seconds', type=float, default=7200)
    pairs.add_argument('--calls', type=int, default=20)
    pairs.add_argument('--replay-limit', type=int, default=0)
    pairs.add_argument('--execute', action='store_true')
    probe = sub.add_parser('_source-probe', help=argparse.SUPPRESS)
    probe.add_argument('--source', type=Path, required=True)
    probe.add_argument('--kind', choices=('preparation', 'fixed', 'replay'), required=True)
    probe.add_argument('--native', action='store_true')
    return parser


def workload_source_proof(args: argparse.Namespace) -> dict[str, Any]:
    """Verify actual imports before the workload opens any frozen data."""
    native = args.variant == 'rust'
    source = args.source or (NATIVE_REFERENCE / 'src' if args.variant == 'python' else ROOT / 'src')
    actual = source_probe(source, source_modules('preparation', native=native))
    result = {'source': str(source.resolve()), 'actual_imports': actual}
    if native:
        identity = native_identity()
        if args.native_sha256 is not None and identity['extension_sha256'] != args.native_sha256:
            raise ValueError('native loaded binary changed after preflight')
        result['native_identity'] = identity
    return result


def main() -> None:
    started = time.monotonic()
    args = argument_parser().parse_args()
    args.command_started = started
    if hasattr(args, 'max_in_flight'):
        args.max_in_flight = args.workers if args.max_in_flight is None else args.max_in_flight
        if args.max_in_flight < 1:
            raise ValueError('max-in-flight must be positive')
    if args.command == '_source-probe':
        proof = source_probe(args.source, source_modules(args.kind, native=args.native))
        print(json.dumps({'source_imports': proof, 'native_identity': native_identity()}
                         if args.native else proof))
        return
    if hasattr(args, 'seconds') and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('seconds must be finite and positive')
    if args.command == 'paired':
        paired(args)
        return
    if getattr(args, 'calls', 1) < 1 or getattr(args, 'replay_limit', 0) < 0:
        raise ValueError('positive calls and nonnegative replay limit required')
    declared_source = None
    if args.command == 'preparation':
        if args.variant != 'rust' and (args.workers != 1 or args.max_in_flight != 1):
            raise ValueError('Python preparation is serial')
        if args.variant in ('python', 'rust') or args.source is not None:
            declared_source = workload_source_proof(args)
    launch_seconds = time.monotonic() - started
    result = preparation(args) if args.command == 'preparation' else (
        policy_workload(args) if args.command == 'policy' else overhead())
    provenance_started = time.monotonic()
    module_hashes = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if (name.startswith(('gf_training.', 'gf_models.', 'gf_common.'))
                and filename is not None and Path(filename).is_file()):
            module_hashes[name] = file_hash(Path(filename))
    versions: dict[str, str | None] = {}
    for name in ('numpy', 'torch', 'lightgbm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result.update(schema_version=1, module_startup_seconds=launch_seconds,
                  total_module_seconds=time.monotonic() - started,
                  provenance_seconds=time.monotonic() - provenance_started,
                  imported_source_sha256=module_hashes, dependency_versions=versions,
                  source_helper_sha256=file_hash(Path(__file__)),
                  thread_environment={key: os.environ.get(key) for key in
                                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
    if declared_source is not None:
        # Recheck source origin after all workload imports, too.
        declared_source['actual_imports'] = source_probe(Path(declared_source['source']), ())
        result['declared_source'] = declared_source
        if 'native_identity' in declared_source:
            result['native_identity'] = declared_source['native_identity']
    encoded = json.dumps(result, separators=(',', ':'))
    destination = os.environ.get('GFB_BENCHMARK_RESULT')
    if destination:
        Path(destination).write_text(encoded + '\n')
        print(json.dumps({'kind': result['kind'], 'result': destination}))
    else:
        print(encoded)


if __name__ == '__main__':
    main()
