"""Dataset4 value intent, without executed-parent, Source or fitting authority."""
from copy import deepcopy
from pathlib import Path

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.domain import ContractError, digest
from gf_common.group_pack_game import storage_binding
from tests.test_game_tree_dataset import dto


def custody(root):
    rows = [{'role': role, 'path': str(Path(root) / name), 'sha256': 'a' * 64, 'identity': [1, i, 33152, 10, 1, 1]}
        for i, (role, name) in enumerate((('locator', 'collection.locator.json'),
            ('descriptor', 'physical/collection.published.json'), ('database', 'physical/collection.sqlite3'),
            ('predecessor', 'physical/collection.pending.json')), 1)]
    value = {'recipe': 'packed-original-custody-v1', 'files': sorted(rows, key=lambda row: row['path']),
        'directories': [{'path': root, 'device': 1, 'inode': 1}, {'path': root + '/physical', 'device': 1, 'inode': 2}]}
    return {**value, 'fingerprint': digest(value)}


def packed_dto():
    value = dto()
    value.update(schema_version=4, source_kind=game.PACKED_SOURCE_KIND, source_origin=game.packed_origin(), storage_binding=storage_binding())
    for row in value['parents']:
        row['storage_binding'] = storage_binding()
    for row in value['collections']:
        row['storage'] = {'format': 'sqlite-group-pack-v1', 'locator_sha256': 'a' * 64,
            'descriptor_sha256': 'a' * 64, 'database_sha256': 'a' * 64, 'seal_id': 'UNADMITTED-value', 'logical_sha256': 'a' * 64}
        row['custody'] = custody(row['directory'])
        if row['split'] == 'validation':
            row['original_correspondence'] = game.final_correspondence('a' * 64, ['e' * 64] * 64, packed=True)
        else:
            relation = row['original_correspondence']
            relation['recipe'] = 'card-instance-packed-source-correspondence-v1'
            relation['binding_sha256'] = digest({key: item for key, item in relation.items() if key != 'binding_sha256'})
    value['source_set_sha256'] = game.source_identity(value['parents'], value['collections'])
    value['content_sha256'] = digest({key: item for key, item in value.items() if key != 'content_sha256'})
    return value


def test_dataset4_closed_provenance_keeps_old_population_features_and_labels():
    raw = packed_dto()
    actual = game.packed_manifest(raw)
    assert actual == raw and actual is not raw
    assert actual['counts'] == dto()['counts']
    assert actual['selection'] == dto()['selection']
    assert actual['label_recipe'] == dto()['label_recipe']
    assert not hasattr(actual, 'selected_inputs')
    with pytest.raises(ContractError):
        game.manifest(raw)
    with pytest.raises(ContractError):
        game.packed_manifest(dto())


@pytest.mark.parametrize('change', ['top-binding', 'parent-binding', 'row-extra', 'custody-hash',
    'custody-role', 'mixed-correspondence', 'final-correspondence', 'fixture-origin', 'schema', 'unknown'])
def test_dataset4_rejects_mixed_or_unbound_storage_subtypes(change):
    raw = packed_dto()
    if change == 'top-binding': raw['storage_binding']['database_schema'] = True
    elif change == 'parent-binding': raw['parents'][0]['storage_binding']['provider'] = 'foreign'
    elif change == 'row-extra': raw['collections'][0]['storage']['claimed_qualified'] = True
    elif change == 'custody-hash': raw['collections'][0]['custody']['fingerprint'] = 'f' * 64
    elif change == 'custody-role': raw['collections'][0]['custody']['files'][0]['role'] = 'foreign'
    elif change == 'mixed-correspondence': raw['collections'][0]['original_correspondence']['recipe'] = 'structured-source-cohort-correspondence-v2'
    elif change == 'final-correspondence': raw['collections'][1]['original_correspondence']['recipe'] = 'card-instance-final-validation-correspondence-v1'
    elif change == 'fixture-origin': raw['source_origin']['outcome_origin'] = 'declared-component-fixture'
    elif change == 'schema': raw['schema_version'] = 3
    else: raw['qualified'] = True
    raw['source_set_sha256'] = game.source_identity(raw['parents'], raw['collections'])
    raw['content_sha256'] = digest({key: item for key, item in raw.items() if key != 'content_sha256'})
    with pytest.raises(ContractError):
        game.packed_manifest(raw)


def test_storage_is_preserved_in_full_formal_homogeneity_comparison():
    from tests.test_game_tree_dataset import comparison_state
    first = comparison_state()
    first['effective_config']['storage'] = {'format': 'sqlite-group-pack-v1', 'db_root': '/db', 'ram_root': '/ram'}
    other = deepcopy(first)
    other['effective_config']['storage']['db_root'] = '/other'
    assert game.normalized_formal_config(first['effective_config']) != game.normalized_formal_config(other['effective_config'])


def test_shared_storage_checker_preserves_same_values_and_rejects_unknown_control_fields():
    from gf_common.group_pack_game import checked_storage_custody
    row = packed_dto()['collections'][0]
    actual_storage, actual_custody = checked_storage_custody(row['storage'], row['custody'])
    assert actual_storage is row['storage'] and actual_custody is row['custody']
    with pytest.raises(ContractError):
        checked_storage_custody({**row['storage'], 'extra': True}, row['custody'])
    with pytest.raises(ContractError):
        checked_storage_custody(row['storage'], {**row['custody'], 'extra': True})
