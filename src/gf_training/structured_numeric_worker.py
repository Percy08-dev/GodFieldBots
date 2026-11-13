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

from gf_common.action_contract import parse_action_contract
from gf_common.card_instance import ActionConfigV3, action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as GAME_LAYOUT
from gf_models.structured_binding import (
    check_input_binding,
    child_cuda_visibility,
    execution_binding_v3,
    execution_digest,
    input_binding,
    input_binding_game_v3,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    architecture_identity,
    model_config_v2,
    model_config_v3,
)


def checksum(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError('wire3 SHA256 required')
    return value


def wire_identity(model: Any, binding: InferenceBindingV2, action_hash: str, resume_identity: str,
                  *, action_contract: Any, purpose: str, synthetic_capacity: bool,
                  cuda_visible_devices: str | None, _actions: ActionConfigV3 | None = None,
                  storage_binding: Json | None = None, _hpo: bool = False) -> Json:
    visibility = child_cuda_visibility(binding.device, cuda_visible_devices)
    storage = {}
    if storage_binding is not None:
        from gf_common.group_pack_game import checked_storage_binding
        if _actions is None:
            raise ContractError('wire5 requires corrected actions')
        storage = {'storage_binding': checked_storage_binding(storage_binding)}
    if _actions is not None:
        from gf_common.domain import digest
        if action_hash != digest(asdict(_actions)):
            raise ContractError("wire4 action Config hash mismatch")
        if _hpo:
            from gf_models.structured_binding import execution_binding_v4
        return {"model_id": model.model_id, "model_version": "4" if _hpo else "3",
            "architecture_identity": architecture_identity(model, layout=GAME_LAYOUT), **storage,
            "action_config_hash": checksum(action_hash), "resume_identity": checksum(resume_identity),
            "actions": asdict(_actions), **input_binding_game_v3(_actions.action_contract, card_instance_rules()),
            "execution": {"binding": asdict(binding), "cuda_visible_devices": visibility},
            "execution_digest": (execution_binding_v4 if _hpo else execution_binding_v3)(model, binding, layout=GAME_LAYOUT, cuda_visible_devices=visibility)}
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
    hpo = identity.get('model_version') == '4'
    game = identity.get('model_version') in ('3', '4')
    fields = {'protocol_version', 'command', 'identity'}
    if game and command == 'update':
        fields |= {'transaction_id', 'collection_generation', 'expected_optimizer_steps'}
        checksum(reply.get('transaction_id'))
        integer(reply.get('collection_generation'), 0, 2**63 - 1 if hpo else 999)
        integer(reply.get('expected_optimizer_steps'), 0)
    closed(reply, fields | ({'closed'} if command == 'close' else {'artifact', 'hardware', 'timings'}
                           | ({'metrics'} if command == 'update' else set())))
    if (type(reply['protocol_version']) is not int or reply['protocol_version'] != (6 if hpo else 5 if 'storage_binding' in identity else 4 if game else 3)
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
        hpo_fields = {'optimizer_step_delta', 'collected_decisions', 'used_decisions', 'tbptt_length',
            'epochs', 'endpoint_evaluations', 'context_decisions', 'suffix_decisions', 'update_seconds'}
        metrics = closed(reply['metrics'], {'policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss',
                                           'optimizer_steps', 'parity'} | (hpo_fields if hpo else {'sequence_length'}))
        for name in ('policy_loss', 'value_mse', 'entropy', 'approx_kl', 'total_loss'):
            _number(metrics[name])
        integer(metrics['optimizer_steps'], 1)
        integer(metrics['tbptt_length'] if hpo else metrics['sequence_length'], 1, 1024)
        if hpo:
            from gf_common.hpo import optimizer_delta
            from gf_training.hpo_config import hpo_training_config
            recipe = hpo_training_config(identity['training_recipe'])
            for name in hpo_fields - {'update_seconds'}:
                integer(metrics[name], 1)
            _number(metrics['update_seconds'])
            if (metrics['epochs'] != recipe.epochs or metrics['tbptt_length'] != recipe.tbptt_length
                    or metrics['collected_decisions'] < metrics['used_decisions']
                    or metrics['endpoint_evaluations'] != recipe.epochs * metrics['used_decisions']
                    or metrics['optimizer_step_delta'] != optimizer_delta(metrics['used_decisions'],
                        epochs=recipe.epochs, minibatch_size=recipe.minibatch_size)
                    or metrics['update_seconds'] < 0):
                raise ContractError('wire6 configured counts/steps mismatch')
        if metrics['optimizer_steps'] != artifact['optimizer_steps']:
            raise ContractError('wire3 optimizer counter mismatch')
        parity = closed(metrics['parity'], {'atol', 'rtol', 'max_abs_error', 'decisions', 'checks'})
        for name in ('atol', 'rtol', 'max_abs_error'):
            _number(parity[name])
        integer(parity['decisions'], 1)
        if (parity['atol'] != 1e-5 or parity['rtol'] != 1e-5 or parity['max_abs_error'] < 0
                or parity['checks'] != ['saved', 'cpu-cache', 'cpu-full', 'learner-full', 'learner-split']):
            raise ContractError('wire3 parity recipe mismatch')
        if hpo and parity['decisions'] != metrics['collected_decisions']:
            raise ContractError('wire6 parity must cover all complete originals')


def _select_visibility(initial: Json, binding: InferenceBindingV2, visibility: str | None) -> None:
    if 'torch' in sys.modules:
        raise ContractError('wire3 Torch imported before visibility selection')
    if visibility is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = visibility
    if os.environ.get('CUDA_VISIBLE_DEVICES') != visibility:
        raise ContractError('wire3 actual visibility mismatch before Torch')


def serve(initial: Json, lines: Iterable[str]) -> None:
    closed(initial, {'protocol_version', 'command', 'identity', 'model', 'seed', 'output', 'restore'})
    hpo = initial.get('protocol_version') == 6
    packed = initial.get('protocol_version') == 5 or hpo and 'storage_binding' in initial.get('identity', {})
    game = initial.get('protocol_version') in (4, 5, 6)
    protocol = 6 if hpo else 5 if packed else 4 if game else 3
    if type(initial['protocol_version']) is not int or initial['protocol_version'] != protocol or initial['command'] != 'initialize':
        raise ContractError('wire3 initialize required')
    identity = closed(initial['identity'], {'model_id', 'model_version', 'architecture_identity', 'action_config_hash',
        'resume_identity', 'action_contract', 'purpose', 'synthetic_capacity', 'execution', 'execution_digest'} | ({'actions', 'rule_signature', 'layout_signature'} if game else set()) | ({'storage_binding'} if packed else set()) | ({'training_recipe','phase_limits'} if hpo else set()))
    training: Any = None
    if hpo:
        from gf_models.hpo_config import model_config_v4
        from gf_training.hpo_config import hpo_training_config
        model: Any = model_config_v4(initial['model'], layout=GAME_LAYOUT)
        training = hpo_training_config(identity['training_recipe'])
        from gf_training.hpo_artifacts import checked_phase_limits, save_estimate
        limits=checked_phase_limits(identity['phase_limits'])
        if save_estimate(model)['additional_bytes']>limits['save_budget_bytes']:
            raise ContractError('wire6 additional save envelope insufficient before learner allocation')
    else:
        model = model_config_v3(initial['model'], layout=GAME_LAYOUT) if game else model_config_v2(initial['model'])
    execution = closed(identity['execution'], {'binding', 'cuda_visible_devices'})
    binding = InferenceBindingV2(**closed(execution['binding'], set(asdict(InferenceBindingV2()))))
    actions = action_config_v3(identity['actions']) if game else None
    contract: Any = actions.action_contract if actions is not None else parse_action_contract(identity['action_contract'])
    if not game:
        check_input_binding({key: identity[key] for key in ('action_contract', 'purpose', 'synthetic_capacity')},
        action_contract=contract, purpose='component', synthetic_capacity=False)
    if hpo:
        from gf_training.hpo_numeric_worker import wire_identity as hpo_identity
        expected = hpo_identity(model, training, binding, identity['action_config_hash'], identity['resume_identity'],
            actions=actions, cuda_visible_devices=execution['cuda_visible_devices'], storage_binding=identity.get('storage_binding'),phase_limits=identity['phase_limits'])
    else:
        expected = wire_identity(model, binding, identity['action_config_hash'], identity['resume_identity'],
            action_contract=contract, purpose='component', synthetic_capacity=False,
            cuda_visible_devices=execution['cuda_visible_devices'], _actions=actions, storage_binding=identity.get('storage_binding'))
    if canonical(identity) != canonical(expected):
        raise ContractError('wire3 initialize binding mismatch')
    _select_visibility(initial, binding, execution['cuda_visible_devices'])
    integer(initial['seed'], 0, 2**63 - 1)
    for key in ('output', 'restore'):
        if initial[key] is not None and (type(initial[key]) is not str or not initial[key]):
            raise ContractError('wire3 artifact path required')
    if initial['output'] is None:
        raise ContractError('wire3 output required')
    started = time.monotonic()
    from gf_models.sharded_arrays import read_json
    from gf_models.torch_runtime import hardware
    from gf_training.structured_numeric import GameLearner, Learner
    from gf_training.structured_rollout import read_game_rollout, read_rollout
    if hpo:
        from gf_training.hpo_numeric import HpoLearner
        assert actions is not None
        learner: Any = HpoLearner(model, training, initial['seed'], binding, identity['resume_identity'], actions=actions, rules=card_instance_rules())
    else:
        learner = (GameLearner(model, initial['seed'], binding, identity['resume_identity'], actions=actions, rules=card_instance_rules()) if actions is not None else
            Learner(model, initial['seed'], binding, identity['action_config_hash'], identity['resume_identity'],
            action_contract=contract, purpose='component', synthetic_capacity=False))
    if initial['restore'] is not None:
        directory = Path(initial['restore'])
        if hpo:
            assert actions is not None
            from gf_models.arrays import sha256
            from gf_models.structured_artifacts import adam_recipe
            from gf_models.structured_sealed import SealedScan, inspect_retained_game_artifact_v4
            inspect_retained_game_artifact_v4(directory,expected_inference_digest=sha256(directory/'inference.json'),
                actions=actions,rules=card_instance_rules(),training_recipe=asdict(training),
                optimizer_recipe=adam_recipe(training.adam_learning_rate,training.adam_beta1,training.adam_beta2,training.adam_epsilon),
                scan=SealedScan(max_owned_bytes=min(limits['save_budget_bytes'],12*1024**3),deadline=float('inf')))
        learner.restore(directory)
        inference, resume = read_json(directory / 'inference.json'), read_json(directory / 'resume.json')
        from gf_models.arrays import sha256
        saved = {'inference_digest': sha256(directory / 'inference.json'), 'resume_digest': sha256(directory / 'resume.json'),
            'files': {**inference['weights']['files'], **resume['state']['files'],
                'inference.json': sha256(directory / 'inference.json'), 'resume.json': sha256(directory / 'resume.json')}}
        save_seconds = 0.
    else:
        saving = time.monotonic()
        saved = learner.save(Path(initial['output']),**({'save_budget_bytes':limits['save_budget_bytes']} if hpo else {}))
        save_seconds = time.monotonic() - saving
    actual_hardware = hardware(binding.device)

    def acknowledge(command: str, began: float, artifact: Json | None = None, **extra: Any) -> None:
        reply = {'protocol_version': protocol, 'command': command, 'identity': identity, **extra}
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
        closed(request, fields | ({'batch', 'batch_digest', 'expected_policy_version', 'output'} | ({'transaction_id', 'projection'} if game else set()) if command == 'update' else set()))
        if (type(request['protocol_version']) is not int or request['protocol_version'] != protocol
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
        rollout: Any
        if game:
            assert actions is not None
            validator: Any = validate_projection
            transaction: Any = digest_projection
            arguments = {}
            if hpo:
                from gf_training.hpo_numeric_worker import digest_projection as hpo_transaction
                from gf_training.hpo_numeric_worker import validate_projection as hpo_projection
                validator, transaction = hpo_projection, hpo_transaction
                arguments = {'training': training}
            projection = validator(request['projection'], expected_steps=learner.steps, policy=request['expected_policy_version'], batch_digest=request['batch_digest'], storage_binding=identity.get('storage_binding'), **arguments)
            if request['transaction_id'] != transaction(projection, storage_binding=identity.get('storage_binding')):
                raise ContractError('wire4 transaction mismatch')
            from gf_models.structured_artifacts import semantics_v3
            declaration = (learner._context.semantics(model) if hpo else semantics_v3(model, layout=GAME_LAYOUT, action_contract=contract, rules=card_instance_rules()))
            from gf_models.arrays import sha256
            manifest = read_json(Path(request['batch']) / 'rollout.json', expected_digest=request['batch_digest'], max_owned_bytes=model.resources.max_owned_bytes)
            from gf_models.sharded_arrays import validate_bundle_metadata
            validate_bundle_metadata(manifest['data'], max_decoded_bytes=model.resources.max_owned_bytes)
            files = {'rollout.json': request['batch_digest'], **manifest['data']['files']}
            if digest_files(files) != projection['numeric_files_sha256']:
                raise ContractError('wire4 numeric original file inventory mismatch')
            for name, expected_hash in files.items():
                if sha256(Path(request['batch']) / name) != expected_hash:
                    raise ContractError('wire4 numeric original file hash mismatch')
            if hpo:
                from gf_training.hpo_rollout import read_hpo_rollout
                rollout = read_hpo_rollout(Path(request['batch']), request['batch_digest'], model, training,
                    actions=actions, rules=card_instance_rules(), expected_selection=projection['selection'],
                    expected_correspondence=projection['source_correspondence'], storage_binding=identity.get('storage_binding'),
                    max_owned_bytes=identity['phase_limits']['numeric_budget_bytes'],
                    numeric_budget_bytes=identity['phase_limits']['numeric_budget_bytes'])
            else:
                rollout = read_game_rollout(Path(request['batch']), request['batch_digest'], model, actions=actions, rules=card_instance_rules(), declaration=declaration, expected_correspondence=projection['source_correspondence'], storage_binding=identity.get('storage_binding'))
        else:
            rollout = read_rollout(Path(request['batch']), request['batch_digest'], model,
                action_contract=contract, purpose='component', synthetic_capacity=False)
        try:
            if rollout.policy_version != request['expected_policy_version']:
                raise ContractError('wire3 frozen behavior policy mismatch')
            metrics = learner.update(rollout)
            if hpo and (metrics['optimizer_step_delta'] != projection['expected_optimizer_step_delta']
                    or learner.steps != projection['expected_optimizer_steps'] + projection['expected_optimizer_step_delta']
                    or metrics['collected_decisions'] != projection['collected_decisions']
                    or metrics['used_decisions'] != projection['used_decisions']):
                raise ContractError('wire6 optimizer acknowledgement differs from frozen Source expectations')
        finally:
            del rollout
        learner.synchronize()
        update_seconds = time.monotonic() - started
        saving = time.monotonic()
        saved = learner.save(Path(request['output']),**({'save_budget_bytes':limits['save_budget_bytes']} if hpo else {}))
        acknowledge('update', started, saved, **({'transaction_id': request['transaction_id'], 'collection_generation': projection['collection_generation'], 'expected_optimizer_steps': projection['expected_optimizer_steps']} if game else {}), metrics=metrics,
            timings={'update_seconds': update_seconds, 'save_seconds': time.monotonic() - saving})
    raise ContractError('wire3 EOF before explicit close')


def digest_files(files: Json) -> str:
    from gf_common.domain import digest
    return digest(dict(sorted(files.items())))


def digest_projection(projection: Json, *, storage_binding: Json | None = None) -> str:
    from gf_common.domain import digest
    if storage_binding is not None:
        from gf_common.group_pack_game import checked_storage_binding
        checked_storage_binding(storage_binding)
    return digest({'recipe': 'card-instance-packed-update-transaction-v1' if storage_binding is not None else 'card-instance-update-transaction-v1', 'projection': projection})


def validate_projection(value: Json, *, expected_steps: int, policy: str, batch_digest: str, storage_binding: Json | None = None) -> Json:
    from gf_training.card_instance_source import PROJECTION_KEYS
    closed(value, PROJECTION_KEYS)
    if storage_binding is not None:
        from gf_common.group_pack_game import checked_storage_binding
        checked_storage_binding(storage_binding)
    if (value['recipe'] != ('card-instance-packed-source-update-v1' if storage_binding is not None else 'card-instance-source-update-v1') or value['expected_optimizer_steps'] != expected_steps
            or type(value['expected_optimizer_steps']) is not int or value['expected_policy_version'] != policy
            or value['numeric_manifest_sha256'] != checksum(batch_digest)):
        raise ContractError('wire4 Source expectation/step/policy mismatch')
    generation = integer(value['collection_generation'], 0, 999)
    if value['batch_id'] != f'batch-{generation:06}' or type(value['parent_run_id']) is not str or not value['parent_run_id']:
        raise ContractError('wire4 Source parent/batch/generation mismatch')
    for key in ('collection_manifest_sha256', 'numeric_files_sha256'):
        checksum(value[key])
    # Full semantics binding is checked by the numeric reader before parity.
    correspondence = value['source_correspondence']
    closed(correspondence, {'recipe', 'control_fingerprint', 'semantics_sha256', 'episode_count', 'episode_digests', 'binding_sha256'})
    if correspondence['recipe'] != ('card-instance-packed-source-correspondence-v1' if storage_binding is not None else 'structured-source-cohort-correspondence-v2') or correspondence['episode_count'] != 16:
        raise ContractError('wire4 Source requires complete16 original correspondence')
    return value
