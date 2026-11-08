"""Own-artifact primitives, NOT healthy pair/formal Source or native-fit evidence."""
import hashlib
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.card_instance import corrected_card_instance_contract
from gf_common.card_instance_fixture import origin as fixture_origin
from gf_common.domain import ContractError, canonical
from gf_common.structured_tokens import signature
from gf_models import structured_tree_artifacts as artifacts
from gf_models import structured_tree_pairing as pairing
from gf_models.structured_tree_config import ConfiguredTreeModelV3, TreeDescriptor
from tests.test_game_tree_records import request

TEXT=b'UNADMITTED-controlled-own-model-NOT-native-LightGBM-text\n'


def declaration():
    model=ConfiguredTreeModelV3();context=pairing.TREE3_GAME_RECORDS
    layout=context.layout(model.encoder)
    descriptor=TreeDescriptor(1,1,hashlib.sha256(TEXT).hexdigest(),len(TEXT))
    provenance=pairing.provenance_for_request(request(),'4'*64,_context=context)
    origin=artifacts.GameTreeOriginV3('5'*64,corrected_card_instance_contract(),'qualified_game',False,
        'trained',provenance,game.SOURCE_KIND,canonical(game.origin()).encode())
    manifest={'schema_version':4,'kind':'structured-tree-inference','model':asdict(model.architecture()),
        'architecture_identity':context.identity(model.architecture()),'input_recipe':context.features.layout.input_recipe,
        'token_signature':signature(model.encoder.aggregate_window,layout=context.features.layout),
        'feature_recipe':context.features.recipe,'feature_signature':layout['feature_signature'],
        'feature_names':layout['feature_names'],'categorical_indices':layout['categorical_indices'],
        'semantics':artifacts.semantics_v3(model,'trained',_context=context),
        'rule_signature':layout['rule_signature'],'layout_signature':layout['layout_signature'],'feature_layout':layout,
        **origin.fields(),'native_parameters':artifacts.NATIVE_PARAMETERS,
        'runtime':{'numpy_version':'2.5.4','lightgbm_version':'4.7.0','native_library_sha256':'2'*64},
        'model_file':{'file':'model.txt','sha256':descriptor.model_sha256,'bytes':len(TEXT)},
        'tree_descriptor':asdict(descriptor),
        'creation_execution':{'inference':asdict(model.inference),'resources':asdict(model.resources),'tree_descriptor':asdict(descriptor)}}
    return model,manifest,descriptor


def forbid_native(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('native model/pair work before structural refusal')
    monkeypatch.setattr(artifacts,'_native',forbidden)
    monkeypatch.setattr(pairing,'inspect_training',forbidden)


def test_pure_game_inference_validator_clones_and_has_no_pair_or_booster_authority(monkeypatch):
    model,manifest,descriptor=declaration();forbid_native(monkeypatch)
    result=artifacts.validate_game_inference_metadata_v3(manifest,model=model,action_config_hash='5'*64,
        expected_tree_descriptor=descriptor,expected_feature_signature=manifest['feature_signature'])
    assert result==manifest and result is not manifest
    result['source_origin']['outcome_origin']='changed'
    assert manifest['source_origin']==game.origin()
    assert not hasattr(result,'booster') and not hasattr(result,'binding')


@pytest.mark.parametrize('field',['fixture','toy','request','feature','descriptor'])
def test_pure_game_validator_rejects_wrong_context_or_full_binding(field,monkeypatch):
    model,manifest,descriptor=declaration();forbid_native(monkeypatch)
    options={}
    if field=='fixture':manifest['source_origin']=fixture_origin()
    elif field=='toy':manifest['fit_origin']='toy';manifest['provenance']=None
    elif field=='request':manifest['provenance']['source_origin']=fixture_origin()
    elif field=='feature':options['expected_feature_signature']='6'*64
    else:options['expected_tree_descriptor']=replace(descriptor,max_depth=2)
    with pytest.raises(ContractError):
        artifacts.validate_game_inference_metadata_v3(manifest,model=model,action_config_hash='5'*64,**options)


def controlled_native(monkeypatch,manifest,descriptor,*,full_descriptor=None,booster=None):
    constructed=[]
    def create(**options):
        assert options['model_str']==TEXT.decode()
        value=SimpleNamespace() if booster is None else booster
        constructed.append(value)
        return value
    monkeypatch.setattr(artifacts,'_text_descriptor',lambda *args,**kwargs:descriptor)
    monkeypatch.setattr(artifacts,'runtime_metadata',lambda **kwargs:manifest['runtime'])
    monkeypatch.setattr(artifacts,'_native',lambda **kwargs:SimpleNamespace(Booster=create))
    monkeypatch.setattr(artifacts,'_validate_booster',lambda *args,**kwargs:(TEXT.decode(),full_descriptor or descriptor))
    monkeypatch.setattr(pairing,'inspect_training',lambda *args,**kwargs:pytest.fail('child opened completion/pair'))
    return constructed


@pytest.mark.parametrize('metadata_type,text_type',[(bytes,bytes),(str,str),(bytes,str),(str,bytes)])
def test_controlled_own_handle_hashes_exact_metadata_bytes_without_claiming_loaded_pair(monkeypatch,metadata_type,text_type):
    model,manifest,descriptor=declaration()
    # Noncanonical whitespace/newline is deliberately part of the inference SHA.
    raw=(canonical(manifest)+' \n').encode()
    constructed=controlled_native(monkeypatch,manifest,descriptor)
    metadata=raw if metadata_type is bytes else raw.decode()
    text=TEXT if text_type is bytes else TEXT.decode()
    handle=artifacts._load_game_own_model_v3(metadata,text,expected_digest=hashlib.sha256(raw).hexdigest(),
        model=model,action_config_hash='5'*64,retained_bytes=0,deadline=time.monotonic()+60,
        own_pair_reference={'request_sha256':'4'*64,'completion_sha256':'f'*64})
    assert type(handle) is artifacts._OwnGameTreeV3 and len(constructed)==1
    assert not isinstance(handle,artifacts.LoadedGameTreeV3)
    assert set(vars(handle))=={'booster','model','origin','descriptor'}
    assert handle.descriptor==descriptor and handle.origin.fields()['source_kind']==game.SOURCE_KIND
    assert not hasattr(handle,'training_binding') and not hasattr(handle,'pair')
    artifacts._bind_model_v3(handle.booster,model,handle.origin)
    with pytest.raises(ContractError,match='binding differs'):
        artifacts._bind_model_v3(handle.booster,replace(model,encoder=replace(model.encoder,aggregate_window=32)),handle.origin)


@pytest.mark.parametrize('fault',['metadata-sha','text-sha','request-ref','completion-syntax','extra-ref','retained-cap','wrong-type'])
def test_own_child_early_guards_precede_native_or_completion_access(monkeypatch,fault):
    model,manifest,_descriptor=declaration();forbid_native(monkeypatch)
    raw=(canonical(manifest)+'\n').encode();text=TEXT;expected=hashlib.sha256(raw).hexdigest();retained=0
    reference={'request_sha256':'4'*64,'completion_sha256':'f'*64}
    if fault=='metadata-sha':expected='a'*64
    elif fault=='text-sha':text+=b'changed'
    elif fault=='request-ref':reference['request_sha256']='b'*64
    elif fault=='completion-syntax':reference['completion_sha256']='not-a-sha'
    elif fault=='extra-ref':reference['directory']='/forbidden/controller/path'
    elif fault=='retained-cap':retained=model.resources.max_owned_bytes
    else:raw=bytearray(raw)
    with pytest.raises(ContractError):
        artifacts._load_game_own_model_v3(raw,text,expected_digest=expected,model=model,action_config_hash='5'*64,
            retained_bytes=retained,deadline=time.monotonic()+60,own_pair_reference=reference)


def test_actual_full_booster_descriptor_checked_before_binding(monkeypatch):
    model,manifest,descriptor=declaration();raw=canonical(manifest).encode()
    constructed=controlled_native(monkeypatch,manifest,descriptor,full_descriptor=replace(descriptor,max_depth=2))
    with pytest.raises(ContractError,match='full Booster'):
        artifacts._load_game_own_model_v3(raw,TEXT,expected_digest=hashlib.sha256(raw).hexdigest(),
            model=model,action_config_hash='5'*64,retained_bytes=0,deadline=time.monotonic()+60,
            own_pair_reference={'request_sha256':'4'*64,'completion_sha256':'f'*64})
    assert len(constructed)==1 and not hasattr(constructed[0],'_structured_tree_origin')


def test_game_pair_helpers_force_closed_context_and_cannot_return_toy(monkeypatch,tmp_path):
    seen=[]
    class StopBeforeOriginals(RuntimeError):pass
    def inspect(*args,**kwargs):
        seen.append(kwargs);raise StopBeforeOriginals()
    monkeypatch.setattr(pairing,'inspect_training',inspect)
    with pytest.raises(StopBeforeOriginals):pairing.inspect_training_game_v3(tmp_path,deadline=123.)
    assert seen[0]['_context'] is pairing.TREE3_GAME_RECORDS
    binding=pairing.TrainingBinding(str(tmp_path),'a'*64,'b'*64)
    def resolve(*args,**kwargs):seen.append(kwargs);return tmp_path/'artifact','c'*64,binding
    monkeypatch.setattr(pairing,'resolve_checkpoint',resolve)
    assert pairing.resolve_checkpoint_game_v3(tmp_path)==(tmp_path/'artifact','c'*64,binding)
    assert seen[1]['_context'] is pairing.TREE3_GAME_RECORDS
    monkeypatch.setattr(pairing,'resolve_checkpoint',lambda *args,**kwargs:(tmp_path,'c'*64,None))
    with pytest.raises(ContractError):pairing.resolve_checkpoint_game_v3(tmp_path)
