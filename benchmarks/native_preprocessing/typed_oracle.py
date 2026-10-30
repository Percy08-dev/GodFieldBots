"""Untimed typed export versus the unchanged Python Journal reader/canonicalizer.

Run under benchmark_performance.py for external deadline/process-tree ownership.
This process is never a primary parse/drop timing trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from gf_common.domain import canonical
from gf_training.journal import read_journal


def run(args: argparse.Namespace) -> dict:
    inputs = json.loads(args.inventory.read_text())
    if args.output.exists():
        raise ValueError('oracle output must not exist')
    command = [str(args.binary), '--inventory', str(args.inventory), '--variant', args.variant,
               '--mode', 'export', '--seconds', str(args.seconds), '--memory-budget-mib', '2048']
    identity = json.loads(subprocess.check_output([str(args.binary), '--identity'], timeout=10))
    if not identity['arbitrary_precision'] or identity['allocation_counts']:
        raise ValueError('oracle requires primary AP binary')
    started = time.monotonic()
    expected = iter(inputs)
    digest = hashlib.sha256()
    files = records = 0
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    try:
        assert child.stdout
        # One complete journal per line, bounded by source limit plus envelope.
        for line in child.stdout:
            if time.monotonic() - started >= args.seconds:
                raise TimeoutError('oracle original deadline')
            declared = next(expected)
            filename, typed = json.loads(line)
            if Path(filename).resolve() != Path(declared['path']).resolve():
                raise ValueError('export inventory/order mismatch')
            original = read_journal(Path(filename), deadline=started + args.seconds)
            if len(original) != len(typed):
                raise ValueError(f'record count changed: {filename}')
            for index, (left, right) in enumerate(zip(original, typed, strict=True)):
                # Dict order is intentionally canonical; array/value/record order is exact.
                if left != right or canonical(left) != canonical(right):
                    raise ValueError(f'legacy semantic/canonical/value/order mismatch: {filename}:{index}')
                digest.update(canonical(right).encode())
                digest.update(b'\n')
            records += len(original)
            files += 1
        if next(expected, None) is not None or child.wait(timeout=5):
            raise ValueError('typed export failed or omitted journals')
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if child.stdout:
            child.stdout.close()
    return {'schema_version': 1, 'contract': 'typed-journal-experiment-v1',
            'variant': args.variant, 'files': files, 'records': records,
            'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
            'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            'binary': str(args.binary.resolve()), 'binary_identity': identity, 'completed': True,
            'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_sha256': identity['source_sha256'],
            'canonical_records_sha256': digest.hexdigest(), 'new_rejections': 0,
            'elapsed_seconds': time.monotonic() - started,
            'status': 'PASS', 'oracle': 'unchanged read_journal plus Python canonical exact equality',
            'limitation': 'no future typed feature/array equivalence or learning eligibility'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--variant', required=True, choices=('B', 'C'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError('positive seconds required')
    result = run(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
