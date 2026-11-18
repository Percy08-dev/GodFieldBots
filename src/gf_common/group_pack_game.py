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
    hpo = (type(state.get('schema_version')) is int and state['schema_version'] == 5
        and state.get('run_variant') in ('hpo-ppo-v1','hpo-ppo-resident-v1','hpo-ppo-owned-v1','hpo-ppo-owned-shared-v1') and state.get('model_version') == '4')
    if (state.get('schema_version') == 4 and type(state.get('schema_version')) is int) or hpo and 'storage_binding' in state:
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
    if custody['recipe'] not in ('packed-original-custody-v1', 'packed-original-custody-v2'):
        raise ContractError('packed original custody recipe required')
    if type(custody['files']) is not list or type(custody['directories']) is not list:
        raise ContractError('packed custody arrays required')
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
    for role, key in (('locator', 'locator_sha256'), ('descriptor', 'descriptor_sha256'),
                      ('database', 'database_sha256')):
        if next(row['sha256'] for row in custody['files'] if row['role'] == role) != storage[key]:
            raise ContractError('packed custody/storage checksum differs')
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
    shared=type(value.get('schema_version')) is int and value['schema_version']==3
    closed(value, fields | ({'service_usage'} if shared else set()))
    if shared:
        from dataclasses import asdict

        from gf_common.card_instance import shared_card_instance_contract
        if value['action_contract']!=asdict(shared_card_instance_contract()):
            raise ContractError('shared packed proof requires adopted contract')
        from gf_common.shared_inference import checked_usage
        checked_usage(value['service_usage'])
    if type(value['schema_version']) is not int or value['schema_version'] != (3 if shared else 2) or value['kind'] != PROOF_KIND:
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


def same_storage_custody(left_storage: Json, left: Json, right_storage: Json, right: Json) -> bool:
    """Validate both closed receipts, then compare only the approved recipe change."""
    checked_storage_custody(left_storage, left)
    checked_storage_custody(right_storage, right)
    if canonical(left_storage) != canonical(right_storage):
        return False
    if left['recipe'] == right['recipe']:
        return canonical(left) == canonical(right)
    return canonical({k: v for k, v in left.items() if k not in ('recipe', 'fingerprint')}) == canonical(
        {k: v for k, v in right.items() if k not in ('recipe', 'fingerprint')})


def same_packed_proof(left: Json, right: Json) -> bool:
    """Proof2 stays exact except validated historical/current custody recipes."""
    checked_packed_proof(left)
    checked_packed_proof(right)
    return (same_storage_custody(left['storage'], left['custody'], right['storage'], right['custody'])
        and canonical({k: v for k, v in left.items() if k != 'custody'}) == canonical(
            {k: v for k, v in right.items() if k != 'custody'}))


def same_packed_dataset(left: Json, right: Json, *, historical_correspondences: tuple[Json, ...] | None = None) -> bool:
    """Bind each saved aggregate before checking every actual row and correspondence."""
    from gf_common.card_instance_game_tree import packed_manifest
    from gf_common.domain import clone, digest

    packed_manifest(left)
    packed_manifest(right)
    if len(left['collections']) != len(right['collections']):
        return False
    if historical_correspondences is not None and len(historical_correspondences) != len(left['collections']):
        return False
    compared = clone(right)
    for index, (saved, actual) in enumerate(zip(left['collections'], compared['collections'], strict=True)):
        if not same_storage_custody(saved['storage'], saved['custody'], actual['storage'], actual['custody']):
            return False
        # Only the independently checked custody evidence can differ. Historical
        # correspondence is compared exactly; episode/role/numeric facts are never omitted.
        if historical_correspondences is not None:
            expected = historical_correspondences[index]
            from gf_common.card_instance_game_tree import (
                checked_final_correspondence,
                checked_training_correspondence,
            )
            if saved['split'] == 'train':
                checked_training_correspondence(expected, packed=True)
            else:
                checked_final_correspondence(expected, manifest_sha256=saved['manifest_sha256'], packed=True)
            if saved['custody']['recipe'] == actual['custody']['recipe'] and canonical(
                    saved['original_correspondence']) != canonical(actual['original_correspondence']):
                return False
            if canonical(saved['original_correspondence']) != canonical(expected):
                return False
            ignored = ('control_fingerprint', 'binding_sha256') if saved['split'] == 'train' else ()
            if canonical({k: v for k, v in actual['original_correspondence'].items() if k not in ignored}) != canonical(
                    {k: v for k, v in expected.items() if k not in ignored}):
                return False
            actual['original_correspondence'] = expected
        actual['custody'] = saved['custody']
    compared['source_set_sha256'] = digest({'recipe': 'card-instance-game-dataset-originals-v1',
        'parents': compared['parents'], 'collections': compared['collections']})
    compared['content_sha256'] = digest({k: v for k, v in compared.items() if k != 'content_sha256'})
    return canonical(left) == canonical(compared)
