"""Packed own-artifact values, without qualified Source, trained pair or fit claims."""
from copy import deepcopy

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.card_instance import corrected_card_instance_contract
from gf_common.domain import ContractError, canonical
from gf_common.group_pack_game import storage_binding
from gf_models import structured_tree_artifacts as artifacts
from gf_models.structured_tree_pairing import TREE3_PACKED_GAME_RECORDS, provenance_for_request
from tests.test_game_tree_artifact_helpers import declaration, forbid_native
from tests.test_group_pack_tree_records import packed_request


def packed_declaration():
    model, manifest, descriptor = declaration()
    provenance = provenance_for_request(packed_request(), '4' * 64, _context=TREE3_PACKED_GAME_RECORDS)
    origin = artifacts.PackedGameTreeOriginV3('5' * 64, corrected_card_instance_contract(),
        'qualified_game', False, 'trained', provenance, game.PACKED_SOURCE_KIND,
        canonical(game.packed_origin()).encode(), canonical(storage_binding()).encode())
    manifest.update(origin.fields())
    return model, manifest, descriptor, origin


def test_packed_inference4_origin_is_immutable_closed_and_keeps_old_features(monkeypatch):
    model, manifest, _descriptor, origin = packed_declaration()
    forbid_native(monkeypatch)
    result = artifacts.validate_game_inference_metadata_v3(manifest, model=model, action_config_hash='5' * 64)
    assert result == manifest and result['schema_version'] == 4
    assert result['provenance']['schema_version'] == 5
    old = declaration()[1]
    for key in ('model', 'feature_recipe', 'feature_signature', 'feature_layout', 'label_recipe', 'native_parameters'):
        if key in old:
            assert result[key] == old[key]
    result['storage_binding']['provider'] = 'changed'
    assert origin.fields()['storage_binding'] == storage_binding()
    assert manifest['storage_binding'] == storage_binding()


@pytest.mark.parametrize('change', ['old-kind', 'old-origin', 'old-provenance', 'extra', 'binding'])
def test_inference4_packed_subtype_refuses_crossed_records_before_booster(change, monkeypatch):
    model, raw, _, _ = packed_declaration()
    forbid_native(monkeypatch)
    if change == 'old-kind': raw['source_kind'] = game.SOURCE_KIND
    elif change == 'old-origin': raw['source_origin'] = game.origin()
    elif change == 'old-provenance': raw['provenance'] = declaration()[1]['provenance']
    elif change == 'extra': raw['claimed_qualified'] = True
    else: raw['storage_binding']['database_schema'] = True
    with pytest.raises(ContractError):
        artifacts.validate_game_inference_metadata_v3(raw, model=model, action_config_hash='5' * 64)


def test_runtime_packed_options_have_exact_new_binding_without_extra_policy_inputs():
    from gf_models.card_instance_game_tree_runtime import safe_tree_options_game_v3
    from tests.test_card_instance_game_tree_runtime import ACTIONS, RULES, controlled_options
    options = controlled_options()
    options.update(source_kind=game.PACKED_SOURCE_KIND, source_origin=game.packed_origin(), storage_binding=storage_binding())
    actual = safe_tree_options_game_v3(options, actions=ACTIONS, rules=RULES)
    assert actual == options
    old = deepcopy(options)
    old['source_kind'], old['source_origin'] = game.SOURCE_KIND, game.origin()
    with pytest.raises(ContractError):
        safe_tree_options_game_v3(old, actions=ACTIONS, rules=RULES)


def test_game_artifact_subtype_routing_preserves_model_preflight_before_read(tmp_path, monkeypatch):
    from gf_models.structured_tree_config import ConfiguredTreeModelV2
    monkeypatch.setattr(artifacts.SealedScan, 'object', lambda *args: pytest.fail('metadata read before exact game model guard'))
    with pytest.raises(ContractError):
        artifacts.inspect_game_artifact_v3(tmp_path, expected_digest='a'*64, model=ConfiguredTreeModelV2(),
            action_config_hash='b'*64, action_contract=corrected_card_instance_contract(),
            purpose='qualified_game', synthetic_capacity=False)


@pytest.mark.parametrize('entry', [artifacts.inspect_game_artifact_v3, artifacts.load_game_artifact_v3])
@pytest.mark.parametrize('option,value', [('expected_tree_descriptor', {}),
    ('expected_tree_descriptor', False), ('expected_feature_signature', 'malformed'),
    ('expected_feature_signature', 7)])
def test_game_artifact_optional_expected_bindings_refuse_before_metadata_read(tmp_path, monkeypatch, entry, option, value):
    model, _, _, _ = packed_declaration()
    monkeypatch.setattr(artifacts.SealedScan, 'object', lambda *args: pytest.fail('metadata read before optional binding guard'))
    with pytest.raises(ContractError):
        entry(tmp_path, expected_digest='a'*64, model=model, action_config_hash='b'*64,
            action_contract=corrected_card_instance_contract(), purpose='qualified_game',
            synthetic_capacity=False, **{option: value})
