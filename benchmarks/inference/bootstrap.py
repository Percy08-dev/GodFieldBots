"""Experiment-only clean resident entry; original transport owns all requests/FDs."""
from __future__ import annotations

import json
import os
import sys
import weakref
from pathlib import Path
from typing import Any

# Extract closed numeric execution controls before original argv/environment scrub.
_raw_options, _cache = sys.argv[1:3]
sys.argv[:] = [sys.argv[0], *sys.argv[3:]]
import gf_simulator_client.resident_bootstrap as original
from gf_models import structured_network
from gf_models.inference_experiment import ExperimentOptions, install

_options = ExperimentOptions(**json.loads(_raw_options))
if _options.compile == 'reduce-overhead':
    raise ValueError('resident reduce-overhead private captured tensors unsupported')
_cache_path = Path(_cache)
if not _cache_path.is_absolute() or not _cache_path.is_dir() or _cache_path.is_symlink():
    raise ValueError('explicit owned compiler cache required')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(_cache_path)
os.environ['TRITON_CACHE_DIR'] = str(_cache_path/'triton')
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
_make_network = structured_network.make_network
_networks: list[Any] = []


def make_network(*args: Any, **kwargs: Any) -> Any:
    # Original make_network resets runtime IEEE settings. Reapply only here in
    # the isolated actor, after every construction and before inference.
    network = _make_network(*args, **kwargs)
    if _options != ExperimentOptions():
        install(network, _options, resident=True)
        _networks.append(weakref.ref(network))
    return network


structured_network.make_network = make_network
# Artifact loaders can hold the original function by direct import before this
# bootstrap patch. Bind the owned inference loader explicitly too.
import gf_models.hpo_owned_inference as owned_inference

owned_inference.make_network = make_network
# _load_inference imports make_network locally, so no artifact alias is changed.
# Keep diagnostic compiler accounting outside policy-visible transport/inputs.
_exchange = original.ResidentEpisodeTransport.exchange
_events = 0

def exchange(self: Any, envelope: Any) -> Any:
    global _events
    response = _exchange(self, envelope)
    if envelope['request']['command'] in ('END', 'PARK', 'SYNC', 'CLOSE') and _networks:
        records = [network._inference_experiment.diagnostics() for ref in _networks if (network := ref()) is not None]
        path = _cache_path.parent.parent/('actor-'+str(os.getpid())+'-compiler.jsonl')
        line = json.dumps({'event': _events, 'compiler_threads': 1, 'networks': records}, allow_nan=False)+'\n'
        if len(line) > 65536 or (path.exists() and path.stat().st_size+len(line) > 1024**2):
            raise ValueError('experimental actor compiler diagnostic cap exceeded')
        with path.open('a') as stream:
            stream.write(line)
        _events += 1
    return response

_transport_type: Any = original.ResidentEpisodeTransport
_transport_type.exchange = exchange
original.main()
