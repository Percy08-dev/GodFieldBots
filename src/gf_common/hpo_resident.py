"""Closed resident HPO management facts; never policy inputs or live authority."""
from __future__ import annotations

import re
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer

RESIDENT_MODE = 'resident-worker-v1'
RESIDENT_VARIANT = 'hpo-ppo-resident-v1'
RESIDENT_CAPABILITY = 'collection7;proof4;registry5;pool1'
CACHE_ENTRIES = 6
CACHE_WEIGHT_BYTES = 128 * 1024**2


def resident_runtime(value: Json) -> Json:
    raw = closed(value, {'mode', 'cache_entries', 'cache_weight_bytes'})
    if raw['mode'] != RESIDENT_MODE:
        raise ContractError('explicit resident-worker-v1 runtime required')
    integer(raw['cache_entries'], CACHE_ENTRIES, CACHE_ENTRIES)
    integer(raw['cache_weight_bytes'], CACHE_WEIGHT_BYTES, CACHE_WEIGHT_BYTES)
    return dict(raw)


def is_resident(state: Json) -> bool:
    return state.get('run_variant') == RESIDENT_VARIANT


CACHE_OWNER_KEYS = {'job_ticket', 'pool_id', 'pool_epoch', 'worker_key', 'pid', 'viewer', 'lease_epoch'}


def cache_owner(value: Json) -> Json:
    raw = closed(value, CACHE_OWNER_KEYS)
    for name in ('job_ticket', 'pool_id'):
        if (type(raw[name]) is not str or len(raw[name]) != 32
                or any(char not in '0123456789abcdef' for char in raw[name])):
            raise ContractError('resident job/pool identity must be a lowercase32hex token')
    for name in ('pool_epoch', 'worker_key', 'pid', 'lease_epoch'):
        integer(raw[name], 1, 2**63-1)
    integer(raw['viewer'], 1, 2)
    return dict(raw)


REQUEST_KEYS = CACHE_OWNER_KEYS | {'schema_version', 'protocol', 'run_id', 'generation', 'command',
    'request_epoch', 'checkpoint_sha256', 'cache_identity'}
ACK_KEYS = REQUEST_KEYS | {'clean_reset', 'private_state_empty', 'pending_io_empty', 'cuda_synchronized', 'reset_version'}


def _sha(value: object) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
        raise ContractError('resident digest must be lowercase SHA256')


def checked_ack(value: Json) -> Json:
    raw = closed(value, ACK_KEYS)
    cache_owner({key:raw[key] for key in CACHE_OWNER_KEYS})
    integer(raw['schema_version'], 1, 1)
    integer(raw['generation'], 0, 2**63-1)
    integer(raw['request_epoch'], 1, 2**63-1)
    _sha(raw['checkpoint_sha256'])
    if raw['cache_identity'] is not None:
        _sha(raw['cache_identity'])
    if (raw['protocol'] != RESIDENT_MODE or raw['command'] not in ('END', 'PARK', 'SYNC', 'EVICT')
            or type(raw['run_id']) is not str or not raw['run_id'] or len(raw['run_id']) > 128
            or raw['reset_version'] != 'resident-reset-v1'
            or any(raw[key] is not True for key in ('clean_reset', 'private_state_empty', 'pending_io_empty', 'cuda_synchronized'))):
        raise ContractError('resident original clean reset/CUDA/PARK acknowledgement required')
    if (raw['command'] == 'EVICT') != (raw['cache_identity'] is not None):
        raise ContractError('resident EVICT requires exact cache identity')
    return raw


def checked_retirement(value: Any) -> Json:
    from gf_common.domain import canonical, digest
    raw = closed(value, {'schema_version', 'kind', 'run_id', 'collection_id', 'job_ticket',
        'pool_id', 'pool_epoch', 'generation', 'owners'})
    integer(raw['schema_version'], 1, 1)
    integer(raw['pool_epoch'], 1, 2**63-1)
    integer(raw['generation'], 0, 2**63-1)
    if (raw['kind'] != 'hpo-resident-logical-retirement' or type(raw['collection_id']) is not str
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', raw['collection_id']) is None
            or type(raw['owners']) is not list or not 1 <= len(raw['owners']) <= 4):
        raise ContractError('closed resident logical retirement evidence required')
    seen = set()
    viewers = {1:0, 2:0}
    for row in raw['owners']:
        closed(row, CACHE_OWNER_KEYS | {'end_ack', 'end_ack_sha256', 'park_ack', 'park_ack_sha256'})
        cache_owner({key:row[key] for key in CACHE_OWNER_KEYS})
        if row['worker_key'] in seen:
            raise ContractError('resident retirement repeats physical owner')
        seen.add(row['worker_key'])
        viewers[row['viewer']] += 1
        if viewers[row['viewer']] > 2:
            raise ContractError('resident viewer has more than two physical owners')
        for name, command in (('end_ack', 'END'), ('park_ack', 'PARK')):
            ack = checked_ack(row[name])
            if ack['command'] != command or row[name+'_sha256'] != digest(ack):
                raise ContractError('resident retirement acknowledgement digest/operation differs')
            for key in CACHE_OWNER_KEYS | {'run_id', 'generation'}:
                expected = row[key] if key in CACHE_OWNER_KEYS else raw[key]
                if canonical(ack[key]) != canonical(expected):
                    raise ContractError('resident retirement acknowledgement owner/epoch differs')
            for key in ('job_ticket', 'pool_id', 'pool_epoch'):
                if row[key] != raw[key]:
                    raise ContractError('resident retirement has foreign job/pool')
        if (row['park_ack']['request_epoch'] <= row['end_ack']['request_epoch']
                or row['park_ack']['checkpoint_sha256'] != row['end_ack']['checkpoint_sha256']):
            raise ContractError('resident retirement PARK must follow same-policy END')
    return raw


def checked_cache_release(binding: Json, value: Any) -> Json:
    """Audit persisted release facts; this never issues the live release capability."""
    from gf_common.domain import digest
    raw = closed(value, {'schema_version','kind','binding_sha256'}, {'ack','ack_sha256','resource'})
    integer(raw['schema_version'], 1, 1)
    if raw['binding_sha256'] != digest(binding):
        raise ContractError('resident cache release changed original pin binding')
    if raw['kind'] == 'hpo-resident-cache-eviction':
        closed(raw, {'schema_version','kind','binding_sha256','ack','ack_sha256'})
        ack = checked_ack(raw['ack'])
        if (ack['command'] != 'EVICT' or raw['ack_sha256'] != digest(ack)
                or ack['cache_identity'] != binding['cache_identity']
                or ack['checkpoint_sha256'] != binding['inference_sha256']
                or ack['lease_epoch'] < binding['lease_epoch']
                or ack['run_id'] != binding['run_id'] or ack['generation'] < binding['generation']):
            raise ContractError('resident cache release requires exact original EVICT ACK')
        for key in CACHE_OWNER_KEYS - {'lease_epoch'}:
            if ack[key] != binding[key]:
                raise ContractError('resident EVICT ACK has foreign physical owner')
    elif raw['kind'] == 'hpo-resident-cache-reaped':
        closed(raw, {'schema_version','kind','binding_sha256','resource'})
        resource = closed(raw['resource'], {'kind','key','pid','outcome'})
        if (resource['kind'] != 'worker' or resource['outcome'] != 'reaped'
                or type(resource['key']) is not int or resource['key'] != binding['worker_key']
                or type(resource['pid']) is not int or resource['pid'] != binding['pid']):
            raise ContractError('resident cache release reap differs from original physical owner')
    else:
        raise ContractError('resident cache release evidence kind invalid')
    return raw


def require_hpo_runtime_execution(spec: Json) -> None:
    """Validate explicit execution dispatch; legacy specifications remain fresh."""
    if type(spec.get('schema_version')) is int and spec['schema_version'] == 3:
        if 'runtime' not in spec:
            raise ContractError('resident execution requires explicit runtime')
        resident_runtime(spec['runtime'])


def checked_evaluation_runtime(value: Json, linkage: Json, registry: Json) -> Json:
    """Bind management execution evidence to already-validated panel originals."""
    from gf_common.domain import clone, digest
    raw = closed(value, {'schema_version','kind','runtime','evaluation_id','job_ticket',
        'linkage_sha256','policy_registry_sha256'})
    integer(raw['schema_version'],1,1)
    resident_runtime(raw['runtime'])
    if (raw['kind'] != 'hpo-resident-evaluation-runtime' or raw['evaluation_id'] != linkage['evaluation_id']
            or raw['job_ticket'] != linkage['gpu_ticket'] or raw['linkage_sha256'] != digest(linkage)
            or raw['policy_registry_sha256'] != digest(registry)):
        raise ContractError('resident evaluation runtime differs from actual panel linkage/registry')
    return clone(raw)
