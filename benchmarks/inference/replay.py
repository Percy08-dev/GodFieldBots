"""Prepare hashed real participant tokens and replay in a fresh owned process.

Preparation uses existing complete-parent and typed Numeric6 readers. Diagnostic
identities stay outside tensor inputs. Profiling is an explicitly separate run.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common import (
    MAX_OUTPUT,
    code_files,
    immutable_json,
    options,
    read_json,
    sha256,
    tree_bytes,
)


def prepare(args: Any) -> int:
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.sharded_arrays import read_json as original_json
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_rollout import read_hpo_rollout
    from gf_training.hpo_source import validate_hpo_parent
    parent = args.source.resolve(strict=True)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('fresh input output directory required')
    state, records = validate_hpo_parent(parent, require_complete=False,
        max_owned_bytes=2*1024**3, deadline=time.monotonic()+600)
    if state['status'] not in ('incomplete', 'complete', 'paused'):
        raise ValueError('healthy accepted original parent required')
    actions, model, _ = hpo_state(state)
    recipe = hpo_training_config(state['effective_config']['hpo_spec']['training_recipe'])
    update = original_json(parent/'updates/000001.json', expected_digest=records[1]['update_sha256'],
                           max_owned_bytes=2*1024**3)
    rollout = read_hpo_rollout(parent/update['numeric'], update['numeric_sha256'], model, recipe,
        actions=actions, rules=card_instance_rules(), expected_selection=update['source_proof']['selection'],
        expected_snapshot_identity=records[0]['snapshot_identity'], numeric_budget_bytes=512*1024**2,
        max_owned_bytes=2*1024**3)
    # Freeze complete prefixes through declared endpoints, including their real
    # candidate populations. No targets/rewards/diagnostics are exported as inputs.
    sequences = []
    for endpoint in args.endpoints.split(','):
        ep, end = map(int, endpoint.split(':'))
        episode = rollout.episodes[ep]
        if not 0 <= end < len(episode.frames):
            raise ValueError('endpoint outside recorded participant sequence')
        sequences.append({'frames': [asdict(frame) for frame in episode.frames[:end+1]],
                          'candidates': [asdict(table) for table in episode.tables[:end+1]],
                          'source_location': {'episode': ep, 'through_decision': end}})
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.is_dir():
        checkpoint = checkpoint/'inference.json'
    manifest = original_json(checkpoint, max_owned_bytes=2*1024**3)
    from gf_common.domain import canonical
    from gf_models.structured_binding import inference_model
    # Model4 artifact architecture deliberately omits caller-owned resources.
    # Compare through its existing exact projection, preserving all parameters.
    if canonical(manifest['model']) != canonical(inference_model(model)):
        raise ValueError('checkpoint/input model recipe differs; use matching model parent')
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'checkpoint'
    target.mkdir()
    files = {str(checkpoint): sha256(checkpoint)}
    for name, expected in manifest['weights']['files'].items():
        path = checkpoint.parent/name
        if sha256(path) != expected:
            raise ValueError('original checkpoint bundle changed')
        files[str(path)] = expected
    for name in ('run.json', 'updates/000001.json'):
        path = parent/name
        files[str(path)] = sha256(path)
    for name in update['numeric_files']:
        path = parent/update['numeric']/name
        files[str(path)] = sha256(path)
    for name in ['inference.json', *manifest['weights']['files']]:
        source = checkpoint if name == 'inference.json' else checkpoint.parent/name
        shutil.copyfile(source, target/name)
    immutable_json(args.output/'inputs.json', {'schema_version': 1, 'population': 'recorded-real-player-visible',
                                             'sequences': sequences})
    local = {'inputs.json': sha256(args.output/'inputs.json')}
    local.update({'checkpoint/'+path.name: sha256(path) for path in target.iterdir()})
    immutable_json(args.output/'manifest.json', {'schema_version': 1, 'kind': 'fixed-inference-replay',
        'parent': str(parent), 'recorded_producer': manifest['producer'], 'simulator': manifest['action_contract'],
        'schema_versions': {'inference': manifest['schema_version'], 'numeric': 6, 'tokens': 1},
        'model': manifest['model'], 'runtime_model': asdict(model),
        'inference_binding': state['effective_config']['model']['inference'], 'action_config_hash': manifest['action_config_hash'], 'actions': asdict(actions),
        'checkpoint_sha256': sha256(checkpoint), 'original_files': files, 'files': local,
        'preparation_code': code_files(), 'source_validation': 'validate_hpo_parent full-original audit',
        'qualification': 'replay measurement only; no new Source or learning authority'})
    if tree_bytes(args.output) > MAX_OUTPUT:
        raise ValueError('prepared output cap exceeded')
    print(json.dumps({'status': 'PREPARED', 'output': str(args.output), 'sequences': len(sequences)}))
    return 0


def _tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple(item) for item in value)
    if isinstance(value, dict):
        return {name: _tuple(item) for name, item in value.items()}
    return value


def child(args: Any) -> int:
    # Explicit cache controls precede Torch/compiler imports; never use default
    # global caches or an environment-supplied cache implicitly.
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(args.cache)
    os.environ['TRITON_CACHE_DIR'] = str(args.cache/'triton')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import CandidateTokens, FrameTokens, validate_candidate_tokens
    from gf_models.hpo_config import model_config_v4
    from gf_models.inference_experiment import install
    from gf_models.structured_artifacts import load_game_inference_v4
    from gf_models.structured_config import InferenceBindingV2
    request = read_json(args.output/'request.json')
    if request['code'] != code_files() or request['input_manifest_sha256'] != sha256(args.input/'manifest.json'):
        raise ValueError('replay code/input identity changed after immutable launch sidecar')
    manifest = read_json(args.input/'manifest.json')
    for name, expected in manifest['files'].items():
        if sha256(args.input/name) != expected:
            raise ValueError('prepared replay bytes changed')
    data = read_json(args.input/'inputs.json')
    actions = action_config_v3(manifest['actions'])
    model = model_config_v4(manifest['runtime_model'], layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    network, _ = load_game_inference_v4(args.input/'checkpoint', expected_digest=manifest['checkpoint_sha256'],
        model=model, binding=InferenceBindingV2(**{**manifest['inference_binding'], 'device': args.device}), action_config_hash=manifest['action_config_hash'],
        action_contract=actions.action_contract, rules=card_instance_rules())
    selected = options(args.options)
    from gf_models.inference_experiment import ExperimentOptions
    installation_started = time.perf_counter()
    experiment = None if selected == ExperimentOptions() else install(network, selected)
    installation_seconds = time.perf_counter()-installation_started if experiment is not None else 0.
    def compiler_facts() -> dict[str, Any]:
        return {'installation': 'original-unmodified', 'backend_graphs': 0, 'backend_compile_seconds': 0.} if experiment is None else experiment.diagnostics()
    import torch
    sequences = []
    for sequence in data['sequences']:
        frames = [FrameTokens(**_tuple(frame)) for frame in sequence['frames']]
        candidates = [CandidateTokens(**_tuple(table)) for table in sequence['candidates']]
        for frame, table in zip(frames, candidates, strict=True):
            validate_candidate_tokens(table, frame, model.max_positions, layout=network.layout)
        sequences.append((frames, candidates))
    rows = []
    first = True
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device != 'cpu':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats()
    from contextlib import nullcontext
    profile_context = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True,
                                             with_stack=False) if args.profile else nullcontext(None)
    with torch.no_grad(), profile_context as profiler:
        for loop in range(args.warm_loops+1):
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            child_cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            before_graphs = compiler_facts()['backend_graphs']
            gpu_start = torch.cuda.Event(enable_timing=True) if args.device != 'cpu' else None
            gpu_end = torch.cuda.Event(enable_timing=True) if args.device != 'cpu' else None
            if gpu_start is not None:
                gpu_start.record()
            count = candidates_count = 0
            for frames, candidates in sequences:
                cache = network.new_cache()
                for frame, table in zip(frames, candidates, strict=True):
                    call_start = time.perf_counter()
                    context, cards = network.step(frame, cache)
                    logs, _ = network.distribution(context, cards, table)
                    value = network.value(context)
                    if not torch.isfinite(logs).all() or not torch.isfinite(value):
                        raise ValueError('nonfinite structured inference output')
                    probabilities = logs.exp().cpu()
                    if not torch.isfinite(probabilities).all():
                        raise ValueError('nonfinite CPU sampling probabilities')
                    if first:
                        if args.device != 'cpu':
                            torch.cuda.synchronize()
                        first_latency = time.perf_counter()-call_start
                        first = False
                    count += 1
                    candidates_count += len(table.kinds)
                cache.reset()
                if experiment is not None:
                    experiment.reset()
            cuda_ms = None
            if gpu_end is not None:
                gpu_end.record()
                torch.cuda.synchronize()
                assert gpu_start is not None
                cuda_ms = gpu_start.elapsed_time(gpu_end)
            child_cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            graph_delta = compiler_facts()['backend_graphs']-before_graphs
            rows.append({'compiler_backend_graph_delta': compiler_facts()['backend_graphs']-before_graphs,
                'waited_compiler_child_cpu_seconds': child_cpu_end.ru_utime+child_cpu_end.ru_stime
                    -child_cpu_start.ru_utime-child_cpu_start.ru_stime, 'population': 'compiling-loop' if graph_delta else 'cold-first-loop' if loop == 0 else 'warm-loop', 'loop': loop,
                'host_wall_seconds': time.perf_counter()-wall_start, 'process_cpu_seconds': time.process_time()-cpu_start,
                'cuda_event_ms': cuda_ms, 'valid_decisions': count, 'valid_candidates': candidates_count,
                'compiler': compiler_facts()})
    profile: dict[str, Any] = {'status': 'NOT_RUN', 'tensor_core_use': 'unresolved; no hardware counters'}
    if profiler is not None:
        profiler.export_chrome_trace(str(args.output/'trace.json'))
        events = profiler.events()
        keys = profiler.key_averages(group_by_input_shape=True)
        profile = {'status': 'CAPTURED', 'kernel_events': sum('cuda' in str(e.device_type).lower() for e in events),
            'kernel_names': sorted({e.name for e in events if 'cuda' in str(e.device_type).lower()})[:2048],
            'copy_events': sum(any(word in e.name.lower() for word in ('memcpy', 'copy_', '_to_copy')) for e in events),
            'operators': [{'name': e.key, 'count': e.count, 'input_shapes': e.input_shapes,
                           'cpu_time_total_us': e.cpu_time_total} for e in list(keys)[:2048]],
            'tensor_core_use': 'unresolved; inspect trace kernel names; hardware counters unavailable'}
    immutable_json(args.output/'result.json', {'schema_version': 1, 'status': 'REPLAYED',
        'first_call_host_seconds': first_latency, 'installation_host_seconds': installation_seconds, 'loops': rows, 'profile': profile,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'device': args.device, 'device_name': 'cpu' if args.device == 'cpu' else torch.cuda.get_device_name(),
        'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'rss_scope': 'direct replay process peak RSS; excludes compiler descendants',
        'cpu_scope': 'process_time excludes compiler children; separate RUSAGE_CHILDREN covers waited children only',
        'cuda_peak_allocated': None if args.device == 'cpu' else torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved': None if args.device == 'cpu' else torch.cuda.max_memory_reserved(),
        'compiler': compiler_facts(), 'profiled': args.profile,
        'precision': selected.precision, 'tensor_core_use': profile['tensor_core_use'],
        'qualification': 'performance only; no output/action equality gate or learning quality claim'})
    return 0


def run(args: Any) -> int:
    args.input = args.input.resolve(strict=True)
    args.cache = args.cache.absolute()
    args.output = args.output.absolute()
    if args.output.exists() or args.output.is_symlink() or args.cache.is_symlink():
        raise ValueError('fresh output and unlinked explicit cache required')
    if not 1 <= args.warm_loops <= 100 or not 1 <= args.timeout <= 3600:
        raise ValueError('bounded warm loops and child deadline required')
    if args.cache_state == 'empty':
        if args.cache.exists():
            raise ValueError('empty-cache population requires a new cache path')
        args.cache.mkdir(parents=True, exist_ok=False)
    elif not args.cache.is_dir() or not any(args.cache.iterdir()):
        raise ValueError('populated-cache population requires an existing populated explicit cache')
    if tree_bytes(args.cache) > MAX_OUTPUT:
        raise ValueError('compiler cache cap exceeded before launch')
    selected = options(args.options)
    args.output.mkdir(parents=True, exist_ok=False)
    immutable_json(args.output/'options.json', selected.identity())
    immutable_json(args.output/'request.json', {'schema_version': 1, 'input': str(args.input),
        'input_manifest_sha256': sha256(args.input/'manifest.json'), 'options': selected.identity(),
        'code': code_files(), 'cache': str(args.cache), 'cache_state': args.cache_state,
        'initial_cache_bytes': tree_bytes(args.cache), 'warm_loops': args.warm_loops,
        'profile': args.profile, 'device': args.device, 'timeout': args.timeout})
    command = [sys.executable, str(Path(__file__).resolve()), '_child', '--input', str(args.input),
        '--options', str(args.output/'options.json'), '--output', str(args.output), '--cache', str(args.cache),
        '--device', args.device, '--warm-loops', str(args.warm_loops)]
    if args.profile:
        command.append('--profile')
    started = time.monotonic()
    failure = None
    with (args.output/'stdout.log').open('xb') as stdout, (args.output/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-started > args.timeout:
                    raise TimeoutError('replay child absolute deadline elapsed')
                if tree_bytes(args.cache)+tree_bytes(args.output) > 2*MAX_OUTPUT:
                    raise ValueError('replay cache/output cap exceeded')
                try:
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:  # noqa: BLE001 — always reap the owned child
            failure = {'type': type(error).__name__, 'message': str(error)[:2048]}
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        code = process.wait()
    immutable_json(args.output/'receipt.json', {'schema_version': 1, 'pid': process.pid,
        'returncode': code, 'actual_child_waited': True, 'failure': failure,
        'controller_host_seconds': time.monotonic()-started, 'cache_bytes': tree_bytes(args.cache),
        'scope': 'direct replay child wait; ordinary compiler runs synchronous compile_threads1'})
    print(json.dumps({'output': str(args.output), 'returncode': code, 'failure': failure}))
    return 2 if failure or code else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--checkpoint', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--endpoints', default='0:3,2:3,12:19')
    for name in ('run', '_child'):
        cmd = commands.add_parser(name)
        for option in ('input', 'options', 'output', 'cache'):
            cmd.add_argument('--'+option, type=Path, required=True)
        cmd.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
        cmd.add_argument('--warm-loops', type=int, default=5)
        cmd.add_argument('--profile', action='store_true')
        if name == 'run':
            cmd.add_argument('--cache-state', choices=('empty', 'populated'), required=True)
            cmd.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    return {'prepare': prepare, 'run': run, '_child': child}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
