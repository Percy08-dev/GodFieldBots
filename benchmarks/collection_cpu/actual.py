"""One fresh original coordinator invocation; no default/resume/Source changes.

Same CLI as benchmarks/inference/actual.py. Options are the closed four-field
collection options. Root supplies frozen source and immutable trial identities.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import (
    Options,
    code_files,
    immutable_json,
    inference_helper,
    namespace,
    options,
)


def execution_patches(settings: Options, output: Path) -> list[Any]:
    from gf_models import structured_runtime
    from gf_simulator_client.resident_pool import ResidentPhysicalWorker
    original_actual = inference_helper('actual')
    spawn_original = ResidentPhysicalWorker._spawn

    def spawn(self: Any, ledger: Any, resident_bytes: int, arguments: Any,
              environment: Any, **kwargs: Any) -> Any:
        if arguments[1:4] != ['-E','-m','gf_simulator_client.resident_bootstrap']:
            raise ValueError('original clean resident argv required')
        args = [arguments[0], '-E', str(Path(__file__).with_name('bootstrap.py')),
                json.dumps(settings.identity(), separators=(',',':')), str(output), *arguments[4:]]
        return spawn_original(self, ledger, resident_bytes, args, environment, **kwargs)

    result = [(structured_runtime,'hpo_resident_training_admission',
               original_actual.charged_admission(structured_runtime.hpo_resident_training_admission, settings)),
              (ResidentPhysicalWorker,'_spawn',spawn)]
    if settings.parent:
        from benchmarks.collection_cpu.parent import patches
        extra, _proof = patches()
        result.extend(extra)
    if settings.http:
        from benchmarks.collection_cpu.http_overlap import patches as http_patches
        extra, _http = http_patches()
        result.extend(extra)
    if settings.diagnostic:
        from benchmarks.collection_cpu.diagnostics import patches as diagnostic_patches
        result.extend(diagnostic_patches(output))
    return result


def require_launch(freeze: Any, *, run_id: str, context_id: str, materialized_sha256: str,
                   selected: Options, output: Path) -> None:
    matches = [row for row in freeze.get('materializations',[]) if row['run_id']==run_id]
    expected = {'run_id':run_id,'context_id':context_id,'materialized_sha256':materialized_sha256,
                'options':selected.identity(),'output':str(output.absolute())}
    if matches != [expected]:
        raise ValueError('exact root-frozen launch IDs/options/materialization/output required')


def main() -> int:
    # Reuse the already reviewed original composition, including immutable launch
    # sidecars, exact once guard, actual JobContext/ledger and original receipts.
    original = inference_helper('actual')
    if '--output' in sys.argv:
        namespace(Path(sys.argv[sys.argv.index('--output')+1]))
    original_verify = original.verify_freeze

    def verify(path: Path, loaded: Any, runtime: Any) -> Any:
        identity = original_verify(path,loaded,runtime)
        from benchmarks.collection_cpu.common import read_json
        require_launch(read_json(path),
            run_id=sys.argv[sys.argv.index('--run-id')+1],
            context_id=sys.argv[sys.argv.index('--context-id')+1],
            materialized_sha256=runtime.sha256,
            selected=options(Path(sys.argv[sys.argv.index('--options')+1])),
            output=Path(sys.argv[sys.argv.index('--output')+1]))
        from benchmarks.collection_cpu.common import REPO, check_files
        covered = {str((REPO/name).resolve()):value for name,value in identity['files'].items()}
        option_path = Path(sys.argv[sys.argv.index('--options')+1]).resolve(strict=True)
        if str(option_path) not in covered:
            raise ValueError('root freeze must include original closed options file')
        check_files({str(option_path):covered[str(option_path)]})
        op = runtime.config.hpo_spec['operating']
        if op['managed_ram_bytes'] != 24*1024**3 or op['managed_cuda_bytes'] != 12*1024**3:
            raise ValueError('original24GiB RAM/12GiB CUDA caps required')
        return identity

    original.verify_freeze = verify
    original.options = options
    original.code_files = code_files
    original.execution_patches = execution_patches

    def write(path: Path, value: Any) -> None:
        if path.name == 'request.json':
            from benchmarks.collection_cpu.common import sha256
            option_path = Path(sys.argv[sys.argv.index('--options')+1]).resolve(strict=True)
            value.update(options_file={'path':str(option_path),'sha256':sha256(option_path)},
                         kind='experimental-collection-actual',
                         population='passive-diagnostic' if value['options']['diagnostic'] else 'performance',
                         scope='collection investigation only; base identities grant no new qualification')
        if path.name == 'result.json' and value['request']['options']['diagnostic']:
            from benchmarks.collection_cpu.diagnostics import flush
            value['collection_cpu_parent_counters'] = flush()
        immutable_json(path, value)

    original.immutable_json = write
    return int(original.main())


if __name__ == '__main__':
    raise SystemExit(main())
