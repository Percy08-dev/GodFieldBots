"""Pure real-record intent/closed contexts; never a verified Source or actual fit."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.card_instance_fixture import input_binding as fixture_binding
from gf_common.card_instance_fixture import origin as fixture_origin
from gf_common.domain import ContractError, canonical
from gf_models.structured_tree_config import ConfiguredTreeModelV3, TreeResourcesV2
from gf_models.structured_tree_pairing import (
    TREE3_GAME_RECORDS,
    TREE3_RECORDS,
    _records,
    _TreeRecords,
    fit_admission,
    game_fit_admission,
    model_envelope,
    validate_request,
)


def request():
    model=ConfiguredTreeModelV3()
    fit={'num_boost_round':500,'early_stopping_rounds':50,'learning_rate':.05,'min_data_in_leaf':20,'num_threads':1}
    rows={'train':8,'validation':4}
    admission=game_fit_admission(model,fit,rows,retained_bytes=1024,total_bytes=8*1024**3)
    return {'schema_version':4,'kind':'structured-tree-fit-request','run_id':'a'*32,'model':asdict(model.architecture()),
        'execution':{'inference':asdict(model.inference),'resources':asdict(model.resources)},'fit_settings':fit,'seed':17,
        'input_binding':game.input_binding(),'feature_signature':'b'*64,'label_recipe':game.LABEL,
        'dataset_manifest_sha256':'c'*64,'dataset_content_sha256':'d'*64,'source_set_sha256':'e'*64,
        'input_files':{'inputs/manifest.json':'f'*64},'row_counts':rows,'producer':{'code_revision':None,'source_digest':'1'*64},
        'runtime':{'numpy_version':'2.5.4','lightgbm_version':'4.7.0','native_library_sha256':'2'*64},
        'model_envelope':model_envelope(model.architecture(),500,_context=TREE3_GAME_RECORDS),
        'source_kind':game.SOURCE_KIND,'source_origin':game.origin(),
        'shared_generation':{'seconds':123.,'cost_key':'3'*64,'scope':game.SHARED_SCOPE},
        'managed_memory':admission['managed_memory']}


def test_game_context_disjoint_fixture_and_not_reconstructible():
    raw=request()
    assert validate_request(raw,_context=TREE3_GAME_RECORDS)==raw
    with pytest.raises(ContractError):validate_request(raw,_context=TREE3_RECORDS)
    with pytest.raises(ContractError):_records(_TreeRecords(True,True))
    with pytest.raises(ContractError):TREE3_GAME_RECORDS.source({'source_kind':None,'source_origin':None},toy=True)
    with pytest.raises(ContractError):TREE3_GAME_RECORDS.binding(fixture_binding())
    with pytest.raises(ContractError):TREE3_RECORDS.binding(game.input_binding())


@pytest.mark.parametrize('change',['fixture-origin','declared-label','construction-cost','cap','peak','extra','source-kind'])
def test_real_request_refuses_crossed_origin_or_resource_fields(change):
    raw=request()
    if change=='fixture-origin':raw['source_origin']=fixture_origin()
    elif change=='declared-label':raw['label_recipe']='card-instance-declared-terminal-v1'
    elif change=='construction-cost':raw['shared_fixture_construction']=raw.pop('shared_generation')
    elif change=='cap':raw['managed_memory']['child_limit_bytes']-=1
    elif change=='peak':raw['managed_memory']['reserved_peak_bytes']-=1
    elif change=='extra':raw['managed_memory']['qualified']=True
    else:raw['source_kind']='card-instance-tree-fixture-originals'
    with pytest.raises(ContractError):validate_request(raw,_context=TREE3_GAME_RECORDS)


def test_child_local_and_full_declared_child_global_checks_are_independent():
    raw=request();model=ConfiguredTreeModelV3();fit=raw['fit_settings']
    # A tiny child cannot be rescued by global8GiB even when very few rows exist.
    tiny=replace(model,resources=TreeResourcesV2(max_owned_bytes=512*1024**2))
    with pytest.raises(ContractError,match='CHILD-LOCAL'):
        game_fit_admission(tiny,fit,raw['row_counts'],retained_bytes=0,total_bytes=8*1024**3)
    local=fit_admission(model,fit,raw['row_counts'],retained_bytes=0,_context=TREE3_GAME_RECORDS)
    assert local['child_local_bytes']<model.resources.max_owned_bytes
    with pytest.raises(ContractError,match='full configured child'):
        game_fit_admission(model,fit,raw['row_counts'],retained_bytes=7*1024**3,total_bytes=8*1024**3)
    actual=game_fit_admission(model,fit,raw['row_counts'],retained_bytes=1024,total_bytes=8*1024**3)
    assert actual['managed_memory']['child_limit_bytes']==model.resources.max_owned_bytes
    assert actual['total_bytes']==actual['controller_reserved_bytes']+model.resources.max_owned_bytes


def test_final64_correspondence_includes_empty_episode_digest_and_closed_hash():
    rows=['a'*64]*64
    value=game.final_correspondence('b'*64,rows)
    assert game.checked_final_correspondence(value,manifest_sha256='b'*64)==value
    before=canonical(value);rows[0]='c'*64
    assert canonical(value)==before
    for field in ('episode_count','recipe','binding_sha256'):
        changed=deepcopy(value);changed[field]=65 if field=='episode_count' else 'wrong'
        with pytest.raises(ContractError):game.checked_final_correspondence(changed,manifest_sha256='b'*64)


def test_immutable_game_provenance_origin_cannot_be_relabelled_fixture_or_toy():
    from gf_common.card_instance import corrected_card_instance_contract
    from gf_models.structured_tree_artifacts import GameTreeOriginV3, TreeOriginV3
    from gf_models.structured_tree_pairing import GameTrainedProvenanceV4, provenance_for_request
    raw=request();provenance=provenance_for_request(raw,'4'*64,_context=TREE3_GAME_RECORDS)
    assert type(provenance) is GameTrainedProvenanceV4
    value=GameTreeOriginV3('5'*64,corrected_card_instance_contract(),'qualified_game',False,'trained',
        provenance,game.SOURCE_KIND,canonical(game.origin()).encode())
    assert value.fields()['provenance']==provenance.to_dict()
    assert value.fields()['source_origin']==game.origin()
    with pytest.raises(ContractError):
        TreeOriginV3('5'*64,corrected_card_instance_contract(),'component',False,'trained',
            provenance,game.SOURCE_KIND,canonical(game.origin()).encode())
    with pytest.raises(ContractError):
        GameTreeOriginV3('5'*64,corrected_card_instance_contract(),'qualified_game',False,'toy',None,None,None)
