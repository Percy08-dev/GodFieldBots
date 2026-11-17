"""Clean original resident transport with opt-in transfer helper installation."""
from __future__ import annotations

import json
import os
import sys
import weakref
from pathlib import Path
from typing import Any

# Direct path execution in an -E clean child must still find tools-only modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import Options

_raw, _output = sys.argv[1:3]
sys.argv[:] = [sys.argv[0], *sys.argv[3:]]
_settings = Options(**json.loads(_raw))
_output_path = Path(_output)
if not _output_path.is_absolute() or not _output_path.is_dir() or _output_path.is_symlink():
    raise ValueError('owned actor diagnostic directory required')

import gf_simulator_client.resident_bootstrap as original
from benchmarks.collection_cpu.transfer import install
from gf_models import structured_network
from gf_models.inference_experiment import ExperimentOptions
from gf_models.inference_experiment import install as historical_install

_make = structured_network.make_network
_networks: list[Any] = []
_sync: Any = None
from benchmarks.collection_cpu.diagnostics import Counters, actor_install, actor_patches

_counters = Counters()
if _settings.diagnostic:
    actor_install(_counters)


def make_network(*args: Any, **kwargs: Any) -> Any:
    global _sync
    network = _make(*args, **kwargs)
    if _sync is None:
        import torch
        _sync = torch.cuda.synchronize
        torch.cuda.synchronize = _counters.wrap('cuda.synchronize',synchronize) if _settings.diagnostic else synchronize
    if _settings.transfer == 'reuse':
        historical_install(network, ExperimentOptions(packing='reuse'), resident=True)
    elif _settings.transfer != 'T0':
        install(network, _settings.transfer)
    if _settings.diagnostic:
        actor_patches(network,_counters)
    _networks.append(weakref.ref(network))
    return network


structured_network.make_network = make_network
import gf_models.hpo_owned_inference as owned_inference

owned_inference.make_network = make_network

# The original policy's synchronization is the consumption fence. Observe it
# AFTER successful completion, with no additional synchronization or early reuse.
def synchronize(*args: Any, **kwargs: Any) -> Any:
    result = _sync(*args, **kwargs)
    for ref in _networks:
        network = ref()
        experiment = None if network is None else getattr(network, '_inference_experiment', None)
        if experiment is not None and hasattr(experiment, 'consumed'):
            experiment.consumed()
    return result


_exchange = original.ResidentEpisodeTransport.exchange
_dropped = 0


def exchange(self: Any, envelope: Any) -> Any:
    global _dropped
    response = _exchange(self, envelope)
    if _settings.diagnostic and envelope['request']['command'] in ('END','PARK','SYNC','CLOSE'):
        records = []
        for ref in _networks:
            network = ref()
            experiment = None if network is None else getattr(network, '_inference_experiment', None)
            if experiment is not None:
                records.append(experiment.diagnostics())
        path = _output_path/f'actor-{os.getpid()}.jsonl'
        facts = {'networks':records,'command':envelope['request']['command'],
                 'counters':_counters.payload(),'dropped_snapshots':_dropped}
        line = json.dumps(facts,allow_nan=False)+'\n'
        try:
            if len(line)>65536 or (path.exists() and path.stat().st_size+len(line)>1024**2):
                _dropped += 1
            else:
                with path.open('a') as stream:
                    stream.write(line)
            if envelope['request']['command']=='CLOSE':
                from benchmarks.collection_cpu.common import immutable_json
                facts['dropped_snapshots'] = _dropped
                immutable_json(_output_path/f'actor-{os.getpid()}-final.json',facts)
        except (OSError,ValueError):
            # Passive diagnostics cannot invalidate an original action/cleanup.
            _dropped += 1
    return response


_transport_type: Any = original.ResidentEpisodeTransport
_transport_type.exchange = exchange
original.main()
