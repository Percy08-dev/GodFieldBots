"""Pure paired storage context, without Source or fitting authorization."""
from copy import deepcopy

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.domain import ContractError
from gf_common.group_pack_game import storage_binding
from gf_models.structured_tree_pairing import (
    TREE3_GAME_RECORDS,
    TREE3_PACKED_GAME_RECORDS,
    GameTrainedProvenanceV4,
    PackedGameTrainedProvenanceV5,
    _records,
    _TreeRecords,
    provenance_for_request,
    validate_request,
)
from tests.test_game_tree_records import request


def packed_request():
    value = request()
    value.update(schema_version=5, source_kind=game.PACKED_SOURCE_KIND,
        source_origin=game.packed_origin(), storage_binding=storage_binding())
    return value


def test_packed_request_is_closed_distinct_singleton_with_same_numeric_recipe():
    raw = packed_request()
    assert validate_request(raw, _context=TREE3_PACKED_GAME_RECORDS) == raw
    assert TREE3_PACKED_GAME_RECORDS.features is TREE3_GAME_RECORDS.features
    assert TREE3_PACKED_GAME_RECORDS.label == TREE3_GAME_RECORDS.label
    assert TREE3_PACKED_GAME_RECORDS.request_fields == TREE3_GAME_RECORDS.request_fields
    with pytest.raises(ContractError):
        _records(_TreeRecords(True, True, True))
    with pytest.raises(ContractError):
        validate_request(raw, _context=TREE3_GAME_RECORDS)
    with pytest.raises(ContractError):
        validate_request(request(), _context=TREE3_PACKED_GAME_RECORDS)


def test_provenance5_binds_same_immutable_request_and_storage_fields():
    raw = packed_request()
    provenance = provenance_for_request(raw, 'a' * 64, _context=TREE3_PACKED_GAME_RECORDS)
    assert type(provenance) is PackedGameTrainedProvenanceV5
    fields = provenance.to_dict()
    assert fields['schema_version'] == 5
    assert fields['kind'] == 'structured-fixed-dataset4-fit'
    assert fields['storage_binding'] == storage_binding()
    assert PackedGameTrainedProvenanceV5.parse(fields) == provenance
    with pytest.raises(ContractError):
        GameTrainedProvenanceV4.parse(fields)
    raw['storage_binding']['provider'] = 'mutated'
    assert provenance.to_dict()['storage_binding'] == storage_binding()


@pytest.mark.parametrize('change', ['old-kind', 'old-origin', 'binding', 'schema', 'extra', 'missing'])
def test_packed_request_refuses_crossed_identity_or_unknown_fields(change):
    raw = deepcopy(packed_request())
    if change == 'old-kind': raw['source_kind'] = game.SOURCE_KIND
    elif change == 'old-origin': raw['source_origin'] = game.origin()
    elif change == 'binding': raw['storage_binding']['database_schema'] = True
    elif change == 'schema': raw['schema_version'] = 4
    elif change == 'extra': raw['qualified'] = True
    else: del raw['storage_binding']
    with pytest.raises(ContractError):
        validate_request(raw, _context=TREE3_PACKED_GAME_RECORDS)


def test_pair5_keeps_numeric_transport4_exact_old_keysets_and_inference4():
    assert TREE3_PACKED_GAME_RECORDS.schema == 5
    assert TREE3_PACKED_GAME_RECORDS.numeric_schema == 4
    assert TREE3_PACKED_GAME_RECORDS.inference_schema == 4
    assert TREE3_PACKED_GAME_RECORDS.numeric_extra_fields == {'source_kind', 'source_origin'}
    assert 'storage_binding' in TREE3_PACKED_GAME_RECORDS.extra_fields
    assert TREE3_GAME_RECORDS.numeric_schema == TREE3_GAME_RECORDS.schema == 4
