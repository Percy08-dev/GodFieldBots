"""Closed storage variants for corrected original inputs; values grant no authority."""
from __future__ import annotations

from gf_common.domain import ContractError, Json, canonical, closed

CORRESPONDENCE_RECIPE = 'card-instance-packed-source-correspondence-v1'
PROJECTION_RECIPE = 'card-instance-packed-source-update-v1'
PROOF_KIND = 'card-instance-packed-original-source-proof'


def storage_binding() -> Json:
    return {'format': 'sqlite-group-pack-v1', 'pack_format': 'group-pack-v1',
        'database_schema': 1, 'member_identity': 'packed-member-v1',
        'provider': 'python-original-bytes-v1'}


def checked_storage_binding(value: object) -> Json:
    expected = storage_binding()
    closed(value, set(expected))
    if not isinstance(value, dict):
        raise ContractError('storage binding object required')
    if type(value['database_schema']) is not int or canonical(value) != canonical(expected):
        raise ContractError('explicit checked group Pack storage binding required')
    return expected


def packed_state(state: Json) -> bool:
    """Select an explicit variant, never interpreting a high version as corrected."""
    if state.get('schema_version') == 4 and type(state.get('schema_version')) is int:
        checked_storage_binding(state.get('storage_binding'))
        return True
    if 'storage_binding' in state or 'storage' in state.get('effective_config', {}):
        raise ContractError('packed storage requires explicit run4 state')
    return False


def checked_storage_custody(storage: Json, custody: Json) -> tuple[Json, Json]:
    """Closed values only, preserving the supplied objects and existing checks."""
    import re
    from pathlib import Path

    from gf_common.domain import digest, integer

    def checksum(item: object) -> str:
        if type(item) is not str or re.fullmatch('[0-9a-f]{64}', item) is None:
            raise ContractError('packed original SHA256 required')
        return item
    storage = closed(storage, {'format', 'locator_sha256', 'descriptor_sha256',
        'database_sha256', 'seal_id', 'logical_sha256'})
    if storage['format'] != 'sqlite-group-pack-v1' or type(storage['seal_id']) is not str:
        raise ContractError('packed proof storage identity required')
    for key in ('locator_sha256', 'descriptor_sha256', 'database_sha256', 'logical_sha256'):
        checksum(storage[key])
    custody = closed(custody, {'recipe', 'files', 'directories', 'fingerprint'})
    if custody['recipe'] != 'packed-original-custody-v1':
        raise ContractError('packed original custody recipe required')
    roles = set()
    previous = ''
    for entry in custody['files']:
        closed(entry, {'role', 'path', 'sha256', 'identity'})
        path, role = entry['path'], entry['role']
        if not isinstance(path, str) or not Path(path).is_absolute() or path <= previous or role in roles:
            raise ContractError('packed custody requires unique path-sorted absolute originals')
        if role not in ('locator', 'descriptor', 'database', 'predecessor'):
            raise ContractError('unknown packed custody original role')
        roles.add(role)
        previous = path
        checksum(entry['sha256'])
        if type(entry['identity']) is not list or len(entry['identity']) != 6:
            raise ContractError('packed custody requires original stat6 identity')
        for number in entry['identity']:
            integer(number, 0)
    if not {'locator', 'descriptor', 'database'}.issubset(roles):
        raise ContractError('packed custody requires every actual original')
    previous = ''
    for row in custody['directories']:
        closed(row, {'path', 'device', 'inode'})
        if type(row['path']) is not str or not Path(row['path']).is_absolute() or row['path'] <= previous:
            raise ContractError('packed custody requires sorted actual directories')
        previous = row['path']
        integer(row['device'], 0)
        integer(row['inode'], 0)
    if len(custody['directories']) != 2 or checksum(custody['fingerprint']) != digest({key: item for key, item in custody.items() if key != 'fingerprint'}):
        raise ContractError('packed custody directory/fingerprint mismatch')
    return storage, custody


def checked_packed_proof(value: Json) -> Json:
    """Check proof2's closed value variant; actual originals still grant custody."""
    import re

    from gf_common.domain import digest, integer
    from gf_common.group_pack import safe_name

    def checksum(item: object) -> str:
        if type(item) is not str or re.fullmatch('[0-9a-f]{64}', item) is None:
            raise ContractError('packed original SHA256 required')
        return item
    fields = {'schema_version', 'kind', 'action_contract', 'purpose', 'synthetic_capacity',
        'rule_signature', 'layout_signature', 'qualified', 'parent_qualification',
        'manifest_sha256', 'source_set_sha256', 'files', 'games', 'accepted_decisions',
        'closure_counts', 'missing_replays', 'storage', 'custody'}
    closed(value, fields)
    if type(value['schema_version']) is not int or value['schema_version'] != 2 or value['kind'] != PROOF_KIND:
        raise ContractError('explicit original packed proof2 required')
    storage, _ = checked_storage_custody(value['storage'], value['custody'])
    if not isinstance(value['files'], dict):
        raise ContractError('packed logical file map required')
    for name, entry in value['files'].items():
        safe_name(name)
        closed(entry, {'sha256', 'bytes', 'identity'})
        checksum(entry['sha256'])
        integer(entry['bytes'], 0)
        member = closed(entry['identity'], {'format', 'database_sha256', 'group_id', 'pack_sha256',
            'name', 'offset', 'length', 'sha256'})
        if (member['format'] != 'packed-member-v1' or member['database_sha256'] != storage['database_sha256']
                or member['name'] != name or member['length'] != entry['bytes'] or member['sha256'] != entry['sha256']):
            raise ContractError('packed stable logical member identity mismatch')
        if member['group_id'] is None:
            if member['pack_sha256'] is not None or member['offset'] is not None:
                raise ContractError('packed DB control cannot claim a Pack range')
        else:
            safe_name(member['group_id'])
            checksum(member['pack_sha256'])
            integer(member['offset'], 0)
    if checksum(value['source_set_sha256']) != digest(value['files']):
        raise ContractError('packed logical source set hash mismatch')
    return value
