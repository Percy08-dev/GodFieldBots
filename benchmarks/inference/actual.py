"""Fresh-only actual comparison; compose original coordinator, Source and receipts.

No resume/campaign/evaluation option exists. PLAN_ONLY performs original config
materialization and charged admission. --execute requires a root-issued freeze.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    REPO,
    check_files,
    code_files,
    immutable_json,
    options,
    read_json,
    sha256,
    tree_bytes,
)

PRIOR = REPO/'runs/hpo-producer-owned/assessment/device-io-detail-v1'


def prior_runner() -> Any:
    if any(not (PRIOR/name).is_file() or (PRIOR/name).is_symlink() for name in ('runner.py', 'measure.py', 'analyze.py')):
        raise ValueError('required historical local measurement helpers missing: '+str(PRIOR)+
                         '; restore the recorded runner.py and measure.py before using actual comparison')
    sys.path.insert(0, str(PRIOR))
    spec = importlib.util.spec_from_file_location('inference_prior_runner', PRIOR/'runner.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def charged_admission(original: Any, selected: Any = None) -> Any:
    from gf_models.inference_experiment import WORKER_STORAGE_BYTES
    def admitted(*args: Any, **kwargs: Any) -> Any:
        if selected is not None:
            from dataclasses import replace
            # Charge the effective experimental chunk/padded bucket everywhere
            # this shared formula is consumed, including actual pool residency.
            model, inference, *rest = args
            chunk = selected.chunk + max(0, selected.alignment-1)
            args = (model, replace(inference, candidate_chunk=chunk), *rest)
        base = original(*args, **kwargs)
        extra = base['workers'] * WORKER_STORAGE_BYTES
        phases = {name: value + extra for name, value in base['ram_phases'].items()}
        ram, cuda = max(phases.values()), base['peak_cuda_bytes'] + extra
        if ram > base['managed_ram_bytes'] or cuda > base['managed_cuda_bytes']:
            raise ValueError('experimental all-cache worker envelope exceeds original managed caps')
        summaries = {}
        if 'worker_host_bytes' in base:
            summaries['worker_host_bytes'] = base['worker_host_bytes'] + extra
        if 'cuda_phases' in base:
            summaries['cuda_phases'] = {name:value + extra for name,value in base['cuda_phases'].items()}
        return {**base, **summaries, 'worker_bytes': base['worker_bytes'] + extra, 'ram_phases': phases,
                'peak_ram_bytes': ram, 'peak_cuda_bytes': cuda,
                'experimental_storage_bytes': extra, 'experimental_storage_per_worker': WORKER_STORAGE_BYTES}
    return admitted


def execution_patches(settings: Any, cache: Path) -> list[Any]:
    from gf_models import structured_runtime
    from gf_simulator_client.resident_pool import ResidentPhysicalWorker
    original = ResidentPhysicalWorker._spawn
    def spawn(self: Any, ledger: Any, resident_bytes: int, arguments: Any, environment: Any, **kwargs: Any) -> Any:
        expected = 'gf_simulator_client.resident_bootstrap'
        if arguments[1:4] != ['-E', '-m', expected]:
            raise ValueError('experimental bootstrap requires original resident argv')
        worker_cache = cache/str(self.viewer)
        worker_cache.mkdir(exist_ok=True)
        args = [arguments[0], '-E', str(Path(__file__).with_name('bootstrap.py')),
                json.dumps(settings.identity(), separators=(',', ':')), str(worker_cache), *arguments[4:]]
        return original(self, ledger, resident_bytes, args, environment, **kwargs)
    return [(structured_runtime, 'hpo_resident_training_admission',
             charged_admission(structured_runtime.hpo_resident_training_admission, settings)), (ResidentPhysicalWorker, '_spawn', spawn)]


def verify_freeze(path: Path, loaded: Any, runtime: Any) -> dict[str, Any]:
    import subprocess

    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_training.native_backend import binary_identity
    freeze = read_json(path)
    revision = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    if freeze['schema_version'] != 1 or freeze['git_revision'] != revision:
        raise ValueError('root source freeze revision differs')
    files = freeze['files']
    covered = {str((REPO/name).resolve()) for name in files}
    required = {str((REPO/name).resolve()) for name in code_files()}
    required.update(str(PRIOR/name) for name in ('runner.py', 'measure.py', 'analyze.py'))
    required.update(row['path'] for row in loaded.inputs)
    required.update((str(runtime.config.server.binary), str(runtime.config.server.config)))
    native = binary_identity()
    required.add(native['extension_path'])
    simulator = Path(runtime.config.server.source_dir)
    required.update(str(simulator/name) for name in CORRECTED_REVIEWED_SPEC_HASHES)
    actual_sim = subprocess.run(['git', '-C', str(simulator), 'rev-parse', 'HEAD'],
                               check=True, capture_output=True, text=True).stdout.strip()
    if actual_sim != runtime.config.server.simulator_revision or not required <= covered:
        raise ValueError('freeze lacks current application/helper/config/native/Simulator coverage')
    check_files(files)
    return {'root_freeze_sha256': sha256(path), 'git_revision': revision, 'files': files,
            'simulator_revision': actual_sim, 'native': native}


async def monitored_execute(runner: Any, args: Any, result: Any, diagnostic: Any, cache: Path) -> None:
    task = asyncio.create_task(runner.execute(args, result, diagnostic))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=1.)
            if tree_bytes(cache) > 512*1024**2 or tree_bytes(args.report_dir) > 1024**3:
                raise ValueError('actual experimental cache/output cap exceeded')
        await task
    finally:
        if not task.done():
            # Original coordinator must observe cancellation and finish its
            # existing custody/receipt cleanup before this wrapper returns.
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'options', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--context-id', required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        raise ValueError('invoke from repository root')
    selected = options(args.options)
    if selected.compile == 'reduce-overhead':
        raise ValueError('resident reduce-overhead private capture cleanup unavailable')
    runner = prior_runner()
    runner.canonical_uuid(args.run_id); runner.canonical_uuid(args.context_id)
    args.arm, args.mode, args.pause_after_update = 'C', 'fresh', None
    args.config = args.config.resolve(strict=True)
    args.report_dir = args.output.absolute()
    if args.report_dir.exists() or args.report_dir.is_symlink():
        raise ValueError('fresh experimental report directory required')
    loaded, settings, runtime = runner.materialize(args.config, args.run_id, 'C')
    parent = Path(loaded.config.storage.parent_root)/args.run_id
    if parent.exists() or parent.is_symlink():
        raise ValueError('fresh parent namespace required')
    from dataclasses import replace

    from gf_cli.hpo_config import joint_resource_admission
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_models import structured_runtime
    with runner.patches([(structured_runtime, 'hpo_resident_training_admission',
                          charged_admission(structured_runtime.hpo_resident_training_admission, selected))]):
        admission = joint_resource_admission(replace(loaded.config, model=settings.model, training=settings.training),
                                            layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY', 'parent': str(parent), 'admission': admission,
                          'options': selected.identity(), 'materialized_sha256': runtime.sha256}))
        return 0
    if args.freeze is None:
        raise ValueError('execution requires root-issued frozen reviewed source')
    identity = verify_freeze(args.freeze, loaded, runtime)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    cache = args.report_dir/'compiler-cache'
    cache.mkdir()
    request = {'schema_version': 1, 'kind': 'experimental-inference-actual', 'mode': 'fresh',
        'options': selected.identity(), 'code': code_files(), 'identity': identity,
        'python': sys.version, 'python_executable': sys.executable, 'gil_disabled': not sys._is_gil_enabled(),
        'platform': platform.platform(), 'arm': 'C', 'pause_after_update': None,
        'production_identity_scope': 'base-only', 'quality_eligible': False,
        'run_id': args.run_id, 'context_id': args.context_id, 'parent': str(parent),
        'config_file_sha256': sha256(args.config), 'admission': admission,
        'scope': 'exploratory inference performance; no objective or ordinary quality claim'}
    immutable_json(args.report_dir/'request.json', request)
    result: dict[str, Any] = {'schema_version': 1, 'request': request, 'train_calls': 0, 'probe_expectation_met': False,
              'gpu_termination': None, 'gpu_termination_context_verified': False, 'raw_training_report': None}
    measure = runner.Measurement()
    diagnostic = runner.BoundedDiagnostics(args.report_dir/'diagnostics.jsonl')
    started = time.monotonic()
    code = 0
    try:
        from gf_training import hpo_ppo
        original = hpo_ppo.train_hpo_ppo
        async def once(*pos: Any, **kw: Any) -> Any:
            result['train_calls'] += 1
            if result['train_calls'] != 1 or kw['mode'] != 'fresh':
                raise ValueError('exactly one fresh original training invocation required')
            return await original(*pos, **kw)
        patches = runner.production_patches(measure, 'C') + execution_patches(selected, cache)
        patches.append((hpo_ppo, 'train_hpo_ppo', once))
        check_files(identity['files'])
        with runner.patches(patches):
            asyncio.run(monitored_execute(runner, args, result, diagnostic, cache))
    except BaseException as error:  # noqa: BLE001 — preserve original failure and cleanup evidence
        result['failure'] = runner.error_value(error)
        code = 130 if isinstance(error, KeyboardInterrupt) else 2
    finally:
        result['helper_started_monotonic'] = started
        result['helper_finished_monotonic'] = time.monotonic()
        result['helper_elapsed_seconds'] = result['helper_finished_monotonic']-started
        from gf_models.torch_runtime import runtime_metadata
        result['torch_runtime'] = runtime_metadata()
        result['diagnostics'] = diagnostic.finish()
        result['measurement'] = measure.payload()
        rows = result['measurement'].pop('rows')
        with (args.report_dir/'timings.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        result['compiler_cache_bytes'] = tree_bytes(cache)
        if result['compiler_cache_bytes'] > 512*1024**2:
            result['cache_limit_exceeded'] = True
            code = code or 2
        result['exit_code'] = code
        immutable_json(args.report_dir/'result.json', result)
    print(json.dumps({'output': str(args.report_dir), 'exit_code': code,
                      'status': result.get('status'), 'updates': result.get('updates')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
