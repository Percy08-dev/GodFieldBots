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
from typing import Any

from gf_common.domain import canonical, closed
from gf_models.config import GRUModelConfig, MLPModelConfig, TransformerModelConfig, model_config
from gf_training.ppo import read_rollout


def main() -> None:
    execution_cap=None
    if len(sys.argv)>1:
        from gf_training.hpo_learning_process import parse_execution_cap
        execution_cap=parse_execution_cap(tuple(sys.argv[1:]))
    sys.argv[:]=['gf-ppo-learner']
    learner: Any = None
    model = None
    for line in sys.stdin:
        request = json.loads(line)
        if 'protocol_version' in request:
            from gf_training.structured_numeric_worker import serve
            serve(request, sys.stdin, execution_cap=execution_cap)
            return
        if execution_cap is not None:
            raise ValueError('Model4 cap cannot enter frozen numerical worker')
        command = request.get('command')
        started = time.monotonic()
        if command == 'initialize':
            closed(request, {'command', 'model', 'seed', 'device', 'output', 'action_hash'}, {'restore', 'sequence_length'})
            if learner is not None:
                raise ValueError('learner already initialized')
            model = model_config(request['model'])
            if isinstance(model, (GRUModelConfig, TransformerModelConfig)):
                from gf_training.sequence_numeric import Learner as SequenceLearner
                learner = SequenceLearner(model, request['seed'], request['device'], request['sequence_length'])
            elif isinstance(model, MLPModelConfig) and 'sequence_length' not in request:
                from gf_training.ppo_numeric import Learner as MLPLearner
                learner = MLPLearner(model, request['seed'], request['device'])
            else:
                raise ValueError('unsupported PPO learner model/config')
            if request.get('restore'):
                learner.restore(Path(request['restore']))
            result = {} if request.get('restore') else learner.save(Path(request['output']), request['action_hash'])
            result['hardware'] = learner.hardware()
        elif command == 'update':
            closed(request, {'command', 'batch', 'batch_hash', 'output', 'action_hash'})
            if learner is None:
                raise ValueError('learner not initialized')
            rollout: Any
            if isinstance(model, (GRUModelConfig, TransformerModelConfig)):
                from gf_training.sequence_rollout import read_rollout as read_sequence
                rollout = read_sequence(Path(request['batch']), request['batch_hash'], model)
            else:
                rollout = read_rollout(Path(request['batch']), request['batch_hash'])
            try:
                result = learner.update(rollout)
            finally:
                # Retire all batch-owned arrays before saving/acknowledging or
                # loading another batch, including failed numerical updates.
                del rollout
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
