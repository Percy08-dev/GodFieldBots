"""Call the unchanged frozen workload, then report resident memory after return/GC."""
import gc
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.argv[0] = str(ROOT / 'tools/benchmark_workload.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
    started = time.monotonic()
    collected = gc.collect()
    gc_seconds = time.monotonic() - started
    snapshots: dict[str, str | dict[str, str]] = {}
    for name in ('status', 'smaps_rollup'):
        try:
            snapshots[name] = Path('/proc/self', name).read_text()
        except OSError as error:
            snapshots[name] = {'unavailable': str(error)}
    destination = Path(os.environ['GFB_BENCHMARK_RESULT']).parent / 'post-release.json'
    destination.write_text(json.dumps({
        'schema_version': 1, 'scope': 'workload main returned, owners dropped, explicit GC; '
        'interpreter/imports remain; probe and GC included in whole observer wall',
        'gc_seconds': gc_seconds, 'gc_collected': collected, 'proc': snapshots,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
