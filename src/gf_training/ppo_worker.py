"""One clean numerical child for a whole PPO invocation; closed JSON IPC."""
from __future__ import annotations

import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import resource
import time
from pathlib import Path

from gf_common.domain import canonical, closed
from gf_models.config import mlp_model_config
from gf_training.ppo import read_rollout
from gf_training.ppo_numeric import Learner


def main() -> None:
    learner = None
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get('command')
        started = time.monotonic()
        if command == 'initialize':
            closed(request, {'command', 'model', 'seed', 'device', 'output', 'action_hash'}, {'restore'})
            if learner is not None:
                raise ValueError('learner already initialized')
            learner = Learner(mlp_model_config(request['model']), request['seed'], request['device'])
            if request.get('restore'):
                learner.restore(Path(request['restore']))
            result = {} if request.get('restore') else learner.save(Path(request['output']), request['action_hash'])
            result['hardware'] = learner.hardware()
        elif command == 'update':
            closed(request, {'command', 'batch', 'batch_hash', 'output', 'action_hash'})
            if learner is None:
                raise ValueError('learner not initialized')
            result = learner.update(read_rollout(Path(request['batch']), request['batch_hash']))
            result['update_seconds'] = time.monotonic() - started
            save_started = time.monotonic()
            result.update(learner.save(Path(request['output']), request['action_hash']))
            result['save_seconds'] = time.monotonic() - save_started
        elif command == 'close':
            closed(request, {'command'})
            print(canonical({'closed': True}), flush=True)
            return
        else:
            raise ValueError('unsupported learning command')
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_cpu_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(canonical(result), flush=True)


if __name__ == '__main__':
    main()
