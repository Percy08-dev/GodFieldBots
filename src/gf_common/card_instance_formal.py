"""Pure closed formal-PPO control contracts; no files, models or Source authority."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer

RECIPE_SHA256 = '871fc8d4591e62d302bb2f79d98d474da15cbdb19c86506e5a70557f4ecd67b2'
ALGORITHM_SHA256 = '318eab51e51ab242c629111634ff25e986c1a0a0ece0d246694fd72a27c90cc9'
RECIPE_ID = 'card-instance-formal-ppo-v1'
RUN_VARIANT = 'formal-parent-v1'
FORMAL_RUN_FIELDS = {'run_variant', 'formal_declaration', 'formal_stop'}
DECLARATION_FIELDS = {'schema_version', 'kind', 'recipe_id', 'recipe_sha256', 'algorithm_recipe_sha256',
    'seed_slot', 'config_sha256', 'config_inputs_sha256', 'input_binding', 'execution_identity_sha256', 'identity_sha256'}
REFERENCE_FIELDS = {'relative_path', 'sha256', 'recipe_id', 'recipe_sha256', 'algorithm_recipe_sha256', 'seed_slot'}
STOP_FIELDS = {'schema_version', 'kind', 'recipe_id', 'recipe_sha256', 'declaration_sha256', 'reason',
    'generation', 'clock', 'elapsed_seconds', 'owned_bytes', 'admission_reservation_bytes', 'max_owned_bytes', 'max_updates'}
CLOCK_FIELDS = {'invocation_started_unix', 'invocation_started_monotonic', 'observed_monotonic',
    'admission_deadline_monotonic', 'execution_deadline_monotonic', 'wall_deadline_monotonic'}
NATIVE_FORMAL_MARKER = 'run3-formal-parent-v1;declaration1;stop1'
PACKED_RECIPE_SHA256 = '27853b108089667e3e5f3e8f9d54c202b2f76583334f40954f1a1ab2a6844a47'
PACKED_RECIPE_ID = 'card-instance-formal-ppo-group-pack-v1'
PACKED_RUN_VARIANT = 'formal-packed-parent-v1'
PACKED_DECLARATION_FIELDS = DECLARATION_FIELDS | {'storage_binding', 'logical_config_sha256'}
NATIVE_PACKED_FORMAL_MARKER = 'run4-formal-packed-parent-v1;declaration2;stop2'


def packed_recipe_bytes(raw: bytes) -> Json:
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != PACKED_RECIPE_SHA256:
        raise ContractError('formal pinned storage variant recipe bytes differ')
    value = json.loads(raw)
    if (value['base_recipe_id'] != RECIPE_ID or value['base_recipe_sha256'] != RECIPE_SHA256
            or value['algorithm_recipe_sha256'] != ALGORITHM_SHA256
            or (canonical(value) + '\n').encode() != raw):
        raise ContractError('formal storage variant base/algorithm/canonical bytes differ')
    return value


def checked_packed_config(value: Json, recipe: Json) -> Json:
    """Check the exact storage table before checking the complete old literal."""
    storage = closed(value.get('storage'), {'format', 'db_root', 'ram_root', 'budget_mib', 'queue_mib', 'max_pack_mib'})
    if storage['format'] != 'sqlite-group-pack-v1':
        raise ContractError('formal exact group Pack storage format required')
    for name, expected in (('budget_mib', 2048), ('queue_mib', 512), ('max_pack_mib', 256)):
        if type(storage[name]) is not int or storage[name] != expected:
            raise ContractError('formal storage sizes differ from frozen variant')
    for name in ('db_root', 'ram_root'):
        path = storage[name]
        if type(path) is not str or not path or not PurePosixPath(path).is_absolute():
            raise ContractError('formal explicit absolute storage root required')
    return checked_config({key: item for key, item in value.items() if key != 'storage'}, recipe)


def _packed_formal(state: Json) -> bool:
    packed = state.get('run_variant') == PACKED_RUN_VARIANT
    if packed:
        from gf_common.group_pack_game import checked_storage_binding
        if type(state.get('schema_version')) is not int or state['schema_version'] != 4:
            raise ContractError('formal packed variant requires explicit Run4')
        checked_storage_binding(state.get('storage_binding'))
    elif 'storage_binding' in state or 'storage' in state.get('effective_config', {}):
        raise ContractError('formal storage requires explicit packed variant')
    return packed



def _sha(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - set('0123456789abcdef'):
        raise ContractError('formal full SHA256 required')
    return value


def recipe_bytes(raw: bytes) -> Json:
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != RECIPE_SHA256:
        raise ContractError('formal pinned literal recipe bytes differ')
    value = json.loads(raw)
    if value['schema_version'] != 1 or value['kind'] != 'card-instance-formal-ppo-recipe' or value['recipe_id'] != RECIPE_ID:
        raise ContractError('formal literal recipe discriminator differs')
    algorithm = value['algorithm_recipe']
    if algorithm['content_sha256'] != ALGORITHM_SHA256 or digest({k: v for k, v in algorithm.items() if k != 'content_sha256'}) != ALGORITHM_SHA256:
        raise ContractError('formal algorithm canonical content differs')
    return value


def checked_config(value: Json, recipe: Json) -> Json:
    """Materialize one literal variant; no recursive overlay or mutable defaults."""
    expected = clone(recipe['base_config'])
    model = value.get('model', {})
    model_id = model.get('model_id')
    device = model.get('inference', {}).get('device')
    backend = value.get('preprocessing', {}).get('backend')
    seeds = value.get('training', {}).get('seeds')
    if (model_id not in recipe['model_profiles'] or device not in recipe['allowed_inference_devices']
            or backend not in recipe['allowed_reader_backends'] or type(seeds) not in (list, tuple) or len(seeds) != 1):
        raise ContractError('formal closed model/device/backend/seed variant required')
    seed = integer(seeds[0], 0)
    if seed not in recipe['allowed_seed_slots']:
        raise ContractError('formal seed must be exactly one of17/29/43')
    expected['model'] = clone(recipe['model_profiles'][model_id])
    expected['model']['inference']['device'] = device
    expected['preprocessing']['backend'] = backend
    expected['training']['seeds'] = [seed]
    for section, name in (('run', 'name'), ('run', 'output_dir'), ('server', 'source_dir'), ('server', 'binary'), ('server', 'config')):
        actual = value.get(section, {}).get(name)
        if type(actual) is not str or not actual:
            raise ContractError('formal explicit nonempty path/name identity required')
        expected[section][name] = actual
    if canonical(value) != canonical(expected):
        raise ContractError('formal effective configuration differs from full frozen literal variant')
    return expected


def planned_execution(config: Json) -> Json:
    model = config['model']
    inference = clone(model['inference'])
    learner = {**inference, 'device': config['training']['device']}
    return {role: {'binding': binding, 'cuda_visible_devices': '' if binding['device'] == 'cpu' else model['cuda_visible_devices']}
        for role, binding in (('learner', learner), ('inference', inference))}


def declaration_reference(value: Json, *, packed: bool = False) -> Json:
    raw = closed(value, REFERENCE_FIELDS)
    for name in ('sha256', 'recipe_sha256', 'algorithm_recipe_sha256'):
        _sha(raw[name])
    integer(raw['seed_slot'], 0)
    if (raw['relative_path'] != 'formal-declaration.json' or raw['recipe_id'] != (PACKED_RECIPE_ID if packed else RECIPE_ID)
            or raw['recipe_sha256'] != (PACKED_RECIPE_SHA256 if packed else RECIPE_SHA256) or raw['algorithm_recipe_sha256'] != ALGORITHM_SHA256
            or raw['seed_slot'] not in (17, 29, 43)):
        raise ContractError('formal declaration reference differs from closed recipe')
    return clone(raw)


def validate_declaration(value: Json, state: Json, recipe: Json) -> None:
    packed = _packed_formal(state)
    raw = closed(value, PACKED_DECLARATION_FIELDS if packed else DECLARATION_FIELDS)
    logical = (checked_packed_config if packed else checked_config)(state['effective_config'], recipe)
    expected = state['effective_config'] if packed else logical
    reference = declaration_reference(state['formal_declaration'], packed=packed)
    identity = state['identity']
    if canonical(identity.get('formal_parent')) != canonical(reference):
        raise ContractError('formal SAME declaration reference must bind original identity')
    base = {key: val for key, val in identity.items() if key != 'formal_parent'}
    wanted = {'schema_version': 2 if packed else 1,
        'kind': 'card-instance-formal-packed-parent-declaration' if packed else 'card-instance-formal-parent-declaration',
        'recipe_id': PACKED_RECIPE_ID if packed else RECIPE_ID,
        'recipe_sha256': PACKED_RECIPE_SHA256 if packed else RECIPE_SHA256, 'algorithm_recipe_sha256': ALGORITHM_SHA256,
        'seed_slot': expected['training']['seeds'][0], 'config_sha256': digest(expected),
        'config_inputs_sha256': digest(state['config_inputs']), 'input_binding': recipe['input_binding'],
        'execution_identity_sha256': digest(planned_execution(expected)), 'identity_sha256': digest(base)}
    if packed:
        wanted.update(storage_binding=state['storage_binding'], logical_config_sha256=digest(logical))
    if (canonical(raw) != canonical(wanted) or reference['seed_slot'] != wanted['seed_slot']
            or canonical(state['execution']) != canonical(planned_execution(expected))
            or state['config_sha256'] != wanted['config_sha256']
            or base.get('configuration_inputs_sha256') != wanted['config_inputs_sha256']
            or base.get('integration', {}).get('server_config_sha256') != recipe['server_config_sha256']):
        raise ContractError('formal declaration/base identity/config/execution differs')


def clock_values(state: Json, invocation_unix: float, invocation_monotonic: float, observed: float) -> Json:
    return {'invocation_started_unix': invocation_unix, 'invocation_started_monotonic': invocation_monotonic,
        'observed_monotonic': observed,
        'admission_deadline_monotonic': (invocation_monotonic + state['admission_deadline_unix']) - invocation_unix,
        'execution_deadline_monotonic': (invocation_monotonic + state['execution_deadline_unix']) - invocation_unix,
        'wall_deadline_monotonic': (invocation_monotonic + state['wall_deadline_unix']) - invocation_unix}


def stop_receipt(state: Json, *, invocation_unix: float, invocation_monotonic: float,
                 observed_monotonic: float, owned_bytes: int) -> Json | None:
    packed = _packed_formal(state)
    clock = clock_values(state, invocation_unix, invocation_monotonic, observed_monotonic)
    for number in clock.values():
        if type(number) not in (float, int) or not math.isfinite(number):
            raise ContractError('formal finite clock anchors required')
    integer(owned_bytes, 0)
    training = state['effective_config']['training']
    if (observed_monotonic >= clock['execution_deadline_monotonic'] or observed_monotonic >= clock['wall_deadline_monotonic']
            or owned_bytes > training['max_owned_bytes']):
        raise ContractError('formal execution/wall/actual storage overrun cannot complete')
    if (training['max_updates'], training['max_owned_bytes'], training['admission_reservation_bytes']) != (1000, 10 * 1024**3, 1024**3):
        raise ContractError('formal stop limits differ from literal recipe')
    if invocation_unix < state['created_unix'] or invocation_monotonic < 0 or observed_monotonic < invocation_monotonic:
        raise ContractError('formal stop original invocation clock invalid')
    generation = integer(state['updates'], 0, 1000)
    reason = ('max_updates' if generation >= training['max_updates'] else
        'admission_deadline' if observed_monotonic >= clock['admission_deadline_monotonic'] else
        'storage_admission' if owned_bytes + training['admission_reservation_bytes'] > training['max_owned_bytes'] else None)
    if reason is None:
        return None
    if generation == 0:
        raise ContractError('formal zero-update completion is ineligible')
    return {'schema_version': 2 if packed else 1,
        'kind': 'card-instance-formal-packed-stop' if packed else 'card-instance-formal-stop',
        'recipe_id': PACKED_RECIPE_ID if packed else RECIPE_ID,
        'recipe_sha256': PACKED_RECIPE_SHA256 if packed else RECIPE_SHA256, 'declaration_sha256': state['formal_declaration']['sha256'],
        'reason': reason, 'generation': generation, 'clock': clock,
        'elapsed_seconds': (invocation_unix + (observed_monotonic - invocation_monotonic)) - state['created_unix'],
        'owned_bytes': owned_bytes, 'admission_reservation_bytes': training['admission_reservation_bytes'],
        'max_owned_bytes': training['max_owned_bytes'], 'max_updates': training['max_updates']}


def validate_stop(state: Json) -> None:
    raw = closed(state['formal_stop'], STOP_FIELDS)
    clock = closed(raw['clock'], CLOCK_FIELDS)
    expected = stop_receipt(state, invocation_unix=clock['invocation_started_unix'],
        invocation_monotonic=clock['invocation_started_monotonic'], observed_monotonic=clock['observed_monotonic'],
        owned_bytes=raw['owned_bytes'])
    if expected is None or canonical(raw) != canonical(expected):
        raise ContractError('formal stop reason/clock/usage arithmetic differs from captured decision')
    if (clock['observed_monotonic'] < clock['invocation_started_monotonic'] or raw['elapsed_seconds'] < 0
            # Match the SAME coordinator's absolute-clock operation order and
            # .1-second rollback allowance; elapsed subtraction can round.
            or state['last_unix'] < (clock['invocation_started_unix']
                + (clock['observed_monotonic'] - clock['invocation_started_monotonic'])) - .1
            or state['completed_updates'] != state['updates'] or state['optimizer_steps'] <= 0):
        raise ContractError('formal stop must bind positive committed prefix and original elapsed budget')


def validate_formal_state(state: Json) -> None:
    packed = _packed_formal(state)
    if state.get('run_variant') != (PACKED_RUN_VARIANT if packed else RUN_VARIANT):
        raise ContractError('explicit original formal variant required')
    declaration_reference(state['formal_declaration'], packed=packed)
    if state['status'] in ('running', 'paused'):
        if state['formal_stop'] is not None:
            raise ContractError('running/paused formal control has no successful stop')
    elif state['status'] == 'complete':
        validate_stop(state)
