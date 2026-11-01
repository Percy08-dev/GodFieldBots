"""Closed numerical wire3; metadata helpers are safe before Torch import.

This is a component numerical delegate, not a coordinator or Source authority.
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gf_common.action_contract import ActionContract, parse_action_contract
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_models.structured_binding import (
    check_input_binding,
    child_cuda_visibility,
    execution_digest,
    input_binding,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    architecture_identity,
    model_config_v2,
)


def checksum(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError('wire3 SHA256 required')
    return value


def wire_identity(model: ModelConfigV2, binding: InferenceBindingV2, action_hash: str, resume_identity: str,
                  *, action_contract: ActionContract, purpose: str, synthetic_capacity: bool,
                  cuda_visible_devices: str | None) -> Json:
    visibility = child_cuda_visibility(binding.device, cuda_visible_devices)
    return {'model_id': model.model_id, 'model_version': model.version,
            'architecture_identity': architecture_identity(model), 'action_config_hash': checksum(action_hash),
            'resume_identity': checksum(resume_identity),
            **input_binding(action_contract, purpose, synthetic_capacity),
            'execution': {'binding': asdict(binding), 'cuda_visible_devices': visibility},
            'execution_digest': execution_digest(model, binding, cuda_visible_devices=visibility)}


def _number(value: Any) -> None:
    if type(value) not in (int, float):
        raise ContractError('wire3 finite number required')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ContractError('wire3 finite number required')


def validate_reply(reply: Json, command: str, identity: Json) -> None:
    fields = {'protocol_version', 'command', 'identity'}
    closed(reply, fields | ({'closed'} if command == 'close' else {'artifact', 'hardware', 'timings'}
                           | ({'metrics'} if command == 'update' else set())))
    if (type(reply['protocol_version']) is not int or reply['protocol_version'] != 3
            or reply['command'] != command or canonical(reply['identity']) != canonical(identity)):
        raise ContractError('wire3 acknowledgement identity/command mismatch')
    if command == 'close':
        if reply['closed'] is not True:
            raise ContractError('wire3 close acknowledgement required')
        return
    artifact = closed(reply['artifact'], {'inference_digest', 'resume_digest', 'files', 'optimizer_steps'})
    checksum(artifact['inference_digest'])
    checksum(artifact['resume_digest'])
    integer(artifact['optimizer_steps'], 0)
    files = artifact['files']
    if not isinstance(files, dict) or not {'inference.json', 'resume.json'} <= set(files):
        raise ContractError('wire3 full artifact filemap required')
    for name, value in files.items():
        if (type(name) is not str or name not in ('inference.json', 'resume.json')
                and re.fullmatch(r'(weights|resume)-[0-9]{6}\.(npz|json)', name) is None):
            raise ContractError('wire3 foreign artifact file')
        checksum(value)
    if files['inference.json'] != artifact['inference_digest'] or files['resume.json'] != artifact['resume_digest']:
        raise ContractError('wire3 artifact digest/filemap mismatch')
    for prefix in ('weights', 'resume'):
        npz = {name[:-4] for name in files if name.startswith(prefix + '-') and name.endswith('.npz')}
        sidecars = {name[:-5] for name in files if name.startswith(prefix + '-') and name.endswith('.json')}
        if not npz or npz != sidecars or npz != {f'{prefix}-{index:06}' for index in range(len(npz))}:
            raise ContractError('wire3 incomplete artifact shards')
    hardware = closed(reply['hardware'], {'device', 'name', 'cuda_runtime', 'device_count',
                                         'cuda_driver_api_version', 'exact_driver_build'})
    device = identity['execution']['binding']['device']
    if hardware['device'] != device or type(hardware['name']) is not str or not hardware['name']:
        raise ContractError('wire3 hardware device/name mismatch')
    if hardware['cuda_runtime'] is not None and type(hardware['cuda_runtime']) is not str:
        raise ContractError('wire3 CUDA runtime identity invalid')
    integer(hardware['device_count'], 0 if device == 'cpu' else 1, 0 if device == 'cpu' else 2**31 - 1)
    if device == 'cpu':
        if hardware['cuda_driver_api_version'] is not None:
            raise ContractError('wire3 CPU driver identity invalid')
    else:
        integer(hardware['cuda_driver_api_version'], 1)
    if hardware['exact_driver_build'] != 'unverified':
        raise ContractError('wire3 unsupported exact driver identity')
    timings = closed(reply['timings'], {'elapsed_seconds', 'peak_cpu_memory_bytes', 'save_seconds',
                                      'update_seconds' if command == 'update' else 'initialize_seconds'})
    integer(timings['peak_cpu_memory_bytes'], 0)
    for name, value in timings.items():
        _number(value)
        if value < 0:
            raise ContractError('wire3 negative timing')
    if command == 'update':
        metrics = closed(reply['metrics'], {'policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss',
                                           'optimizer_steps', 'sequence_length', 'parity'})
        for name in ('policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss'):
            _number(metrics[name])
        integer(metrics['optimizer_steps'], 1)
        integer(metrics['sequence_length'], 1, 1024)
        if metrics['optimizer_steps'] != artifact['optimizer_steps']:
            raise ContractError('wire3 optimizer counter mismatch')
        parity = closed(metrics['parity'], {'atol', 'rtol', 'max_abs_error', 'decisions', 'checks'})
        for name in ('atol', 'rtol', 'max_abs_error'):
            _number(parity[name])
        integer(parity['decisions'], 1)
        if (parity['atol'] != 1e-5 or parity['rtol'] != 1e-5 or parity['max_abs_error'] < 0
                or parity['checks'] != ['saved', 'cpu-cache', 'cpu-full', 'learner-full', 'learner-split']):
            raise ContractError('wire3 parity recipe mismatch')


def serve(initial: Json, lines: Iterable[str]) -> None:
    closed(initial, {'protocol_version', 'command', 'identity', 'model', 'seed', 'output', 'restore'})
    if type(initial['protocol_version']) is not int or initial['protocol_version'] != 3 or initial['command'] != 'initialize':
        raise ContractError('wire3 initialize required')
    identity = closed(initial['identity'], {'model_id', 'model_version', 'architecture_identity', 'action_config_hash',
        'resume_identity', 'action_contract', 'purpose', 'synthetic_capacity', 'execution', 'execution_digest'})
    model = model_config_v2(initial['model'])
    execution = closed(identity['execution'], {'binding', 'cuda_visible_devices'})
    binding = InferenceBindingV2(**closed(execution['binding'], set(asdict(InferenceBindingV2()))))
    contract = parse_action_contract(identity['action_contract'])
    check_input_binding({key: identity[key] for key in ('action_contract', 'purpose', 'synthetic_capacity')},
        action_contract=contract, purpose='component', synthetic_capacity=False)
    expected = wire_identity(model, binding, identity['action_config_hash'], identity['resume_identity'],
        action_contract=contract, purpose='component', synthetic_capacity=False,
        cuda_visible_devices=execution['cuda_visible_devices'])
    if canonical(identity) != canonical(expected):
        raise ContractError('wire3 initialize binding mismatch')
    if 'torch' in sys.modules:
        raise ContractError('wire3 Torch imported before visibility selection')
    visibility = execution['cuda_visible_devices']
    if visibility is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = visibility
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visibility:
        raise ContractError('wire3 actual visibility mismatch before Torch')
    integer(initial['seed'], 0, 2**63 - 1)
    for key in ('output', 'restore'):
        if initial[key] is not None and (type(initial[key]) is not str or not initial[key]):
            raise ContractError('wire3 artifact path required')
    if initial['output'] is None:
        raise ContractError('wire3 output required')
    started = time.monotonic()
    from gf_models.sharded_arrays import read_json
    from gf_models.torch_runtime import hardware
    from gf_training.structured_numeric import Learner
    from gf_training.structured_rollout import read_rollout
    learner = Learner(model, initial['seed'], binding, identity['action_config_hash'], identity['resume_identity'],
        action_contract=contract, purpose='component', synthetic_capacity=False)
    if initial['restore'] is not None:
        directory = Path(initial['restore'])
        learner.restore(directory)
        inference, resume = read_json(directory / 'inference.json'), read_json(directory / 'resume.json')
        from gf_models.arrays import sha256
        saved = {'inference_digest': sha256(directory / 'inference.json'), 'resume_digest': sha256(directory / 'resume.json'),
            'files': {**inference['weights']['files'], **resume['state']['files'],
                'inference.json': sha256(directory / 'inference.json'), 'resume.json': sha256(directory / 'resume.json')}}
        save_seconds = 0.
    else:
        saving = time.monotonic()
        saved = learner.save(Path(initial['output']))
        save_seconds = time.monotonic() - saving
    actual_hardware = hardware(binding.device)

    def acknowledge(command: str, began: float, artifact: Json | None = None, **extra: Any) -> None:
        reply = {'protocol_version': 3, 'command': command, 'identity': identity, **extra}
        if command != 'close':
            if artifact is None:
                raise ContractError('wire3 numerical artifact missing')
            reply.update(artifact={**artifact, 'optimizer_steps': learner.steps}, hardware=actual_hardware)
            reply['timings'].update(elapsed_seconds=time.monotonic() - began,
                peak_cpu_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        validate_reply(reply, command, identity)
        print(canonical(reply), flush=True)

    acknowledge('initialize', started, saved, timings={'initialize_seconds': time.monotonic() - started - save_seconds,
                                                       'save_seconds': save_seconds})
    for line in lines:
        request = json.loads(line)
        command = request.get('command')
        fields = {'protocol_version', 'command', 'identity'}
        closed(request, fields | ({'batch', 'batch_digest', 'expected_policy_version', 'output'} if command == 'update' else set()))
        if (type(request['protocol_version']) is not int or request['protocol_version'] != 3
                or canonical(request['identity']) != canonical(identity)):
            raise ContractError('wire3 request identity mismatch')
        if command == 'close':
            acknowledge('close', time.monotonic(), closed=True)
            return
        if command != 'update':
            raise ContractError('wire3 unknown numerical command')
        checksum(request['batch_digest'])
        if any(type(request[key]) is not str or not request[key] for key in ('batch', 'output', 'expected_policy_version')):
            raise ContractError('wire3 update path/policy required')
        started = time.monotonic()
        rollout = read_rollout(Path(request['batch']), request['batch_digest'], model,
            action_contract=contract, purpose='component', synthetic_capacity=False)
        try:
            if rollout.policy_version != request['expected_policy_version']:
                raise ContractError('wire3 frozen behavior policy mismatch')
            metrics = learner.update(rollout)
        finally:
            del rollout
        learner.synchronize()
        update_seconds = time.monotonic() - started
        saving = time.monotonic()
        saved = learner.save(Path(request['output']))
        acknowledge('update', started, saved, metrics=metrics,
            timings={'update_seconds': update_seconds, 'save_seconds': time.monotonic() - saving})
    raise ContractError('wire3 EOF before explicit close')
