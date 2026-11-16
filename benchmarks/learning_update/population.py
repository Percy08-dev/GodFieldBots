"""Execute the fixed18trial population sequentially; audits are a later command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.learning_update.common import check_files, read_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan=read_json(args.plan)
    if plan.get('kind')!='update-actual-population' or len(plan['trials'])!=18:
        raise ValueError('complete fixed18trial plan required')
    check_files(read_json(args.freeze)['files'])
    for row in plan['trials']:
        command=[sys.executable,'-m','benchmarks.learning_update.actual','--config',row['config'],
            '--output',row['report'],'--run-id',row['run_id'],'--context-id',row['context_id'],
            '--variant',row['variant'],'--freeze',str(args.freeze)]
        if args.execute:
            command.append('--execute')
        # Failure preserves its actual attempt and stops; never skip/rebudget
        # or substitute a later run to manufacture a complete population.
        subprocess.run(command,check=True)


if __name__=='__main__':
    main()
