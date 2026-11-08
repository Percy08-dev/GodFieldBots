"""Recipe DTOs only: no fake formal run/runtime/qualified Source fixture."""
from copy import deepcopy

import pytest

from gf_common import card_instance_game_tree as game
from gf_common.domain import ContractError, digest
from gf_training.card_instance_game_tree_data import _cohort_comparison


def dto():
    parents=[];collections=[]
    for i,seed in enumerate((17,29,43)):
        generations=[{'generation':g,'generation_sha256':str(i+1)*64,'update_sha256':None if g==0 else 'a'*64,
            'inference_sha256':'b'*64,'resume_sha256':'c'*64,'artifact_files_sha256':'d'*64} for g in (0,1)]
        parents.append({'parent_index':i,'directory':f'/UNADMITTED-DTO-only/p{i}','run_id':f'UNADMITTED-{i}',
            'seed':seed,'run_sha256':str(i+1)*64,'formal_declaration_sha256':'a'*64,'config_sha256':'b'*64,
            'config_inputs_sha256':'c'*64,'resume_identity':'d'*64,'registry_sha256':'e'*64,'final_generation':1,
            'final_generation_sha256':'f'*64,'final_inference_sha256':'0'*64,'generations':generations,'elapsed_seconds':10.})
        for split,count,g in [('train',16,0),('validation',64,1)]:
            correlation={'recipe':'structured-source-cohort-correspondence-v2','control_fingerprint':'a'*64,
                'semantics_sha256':'b'*64,'episode_count':16,'episode_digests':['c'*64]*16}
            correlation['binding_sha256']=digest(correlation)
            roles=[{'episode_id':f'UNADMITTED-episode-{i}-{split}-{j}','viewer':1+j%2,
                'policy_label':'UNADMITTED-policy-intent','end_kind':'natural','outcome':'draw','accepted_count':0} for j in range(count)]
            collections.append({'parent_index':i,'directory':f'/UNADMITTED-DTO-only/p{i}/{split}',
                'collection_role':'ppo-batch' if split=='train' else 'ppo-final-validation','split':split,'generation':g,
                'manifest_sha256':'a'*64,'source_set_sha256':'b'*64,'schedule_sha256':'c'*64,'bindings_sha256':'d'*64,
                'original_correspondence':correlation if split=='train' else game.final_correspondence('a'*64,['e'*64]*64),
                'numeric':{'directory':f'/UNADMITTED-DTO-only/p{i}/numeric','sha256':'f'*64,'files':{}} if split=='train' else None,
                'selected_roles':roles,'counts':{'games':count,'participants':2*count,'selected_decisions':0,'zero_endpoint_games':count},
                'elapsed_seconds':1.})
    value={'schema_version':3,'kind':game.KIND,'selection':game.SELECTION,'label_recipe':game.LABEL,
        'source_kind':game.SOURCE_KIND,'source_origin':game.origin(),'input_binding':game.input_binding(),
        'input_recipe':game.input_binding()['action_contract']['input_recipe'],
        'tree_projection':{'encoder':{},'feature_layout':{},'feature_signature':'a'*64,'token_signature':'b'*64},
        'neural_cohort':{'model':{},'actions_sha256':'a'*64,'execution':{},'semantics':{},'recipe_id':'UNADMITTED-intent',
            'recipe_sha256':'b'*64,'algorithm_recipe_sha256':'c'*64,'code':{}},
        'parents':parents,'collections':collections,
        'counts':{'train':{'games':48,'participants':96,'selected_decisions':0,'zero_endpoint_games':48},
            'validation':{'games':192,'participants':384,'selected_decisions':0,'zero_endpoint_games':192}},
        'source_set_sha256':game.source_identity(parents,collections),
        'shared_generation':game.shared_generation(parents,collections)}
    return {**value,'content_sha256':digest(value)}


def test_dto_keeps_all_zero_endpoint_populations_without_fitting_permission():
    value=dto();actual=game.manifest(value)
    assert actual==value and actual is not value
    assert actual['counts']['validation']['games']==192
    assert actual['shared_generation']['seconds']==33.
    assert not hasattr(actual,'selected_inputs')


@pytest.mark.parametrize('change',['seed','cohort','zero','unknown-outcome','duplicate','fake-numeric','correspondence','origin','input','cost','extra'])
def test_real_recipe_rejects_missing_membership_mixed_origins_or_fake_targets(change):
    raw=dto()
    if change=='seed':raw['parents'][1]['seed']=17
    elif change=='cohort':raw['collections'].pop(0)
    elif change=='zero':raw['counts']['validation']['zero_endpoint_games']-=1
    elif change=='unknown-outcome':raw['collections'][0]['selected_roles'][0]['outcome']=None
    elif change=='duplicate':raw['parents'][1]['run_id']=raw['parents'][0]['run_id'];raw['parents'][1]['run_sha256']=raw['parents'][0]['run_sha256']
    elif change=='fake-numeric':raw['collections'][1]['numeric']={}
    elif change=='correspondence':raw['collections'][1]['original_correspondence']['episode_digests'].pop()
    elif change=='origin':raw['source_origin']['outcome_origin']='declared-component-fixture'
    elif change=='input':raw['input_binding']['purpose']='component'
    elif change=='cost':raw['shared_generation']['seconds']+=1
    else:raw['qualified']=True
    with pytest.raises(ContractError):game.manifest(raw)


def test_homogeneity_normalizer_changes_only_the_declared_relocatable_fields():
    base={'run':{'name':'one','output_dir':'/one','concurrency':2},'server':{'source_dir':'/source','binary':'/bin','config':'/cfg','simulator_revision':'fixed'},
        'training':{'seeds':[17],'max_updates':1000},'inactive_canonical':{'x':123}}
    other=deepcopy(base)
    other['run'].update(name='two',output_dir='/two');other['server'].update(source_dir='/s2',binary='/b2',config='/c2');other['training']['seeds']=[29]
    assert game.normalized_formal_config(base)==game.normalized_formal_config(other)
    other['inactive_canonical']['x']=124
    assert game.normalized_formal_config(base)!=game.normalized_formal_config(other)
    assert base['training']['seeds']==[17]


def comparison_state(seed=17):
    """Memory-only intent fields; never a formal original or Source constructor."""
    return {'scope':'UNADMITTED-comparison-only','run_id':f'UNADMITTED-{seed}','seed':seed,
        'effective_config':{'run':{'name':f'seed-{seed}','output_dir':f'/intent/{seed}','split':'train','concurrency':2},
            'server':{'source_dir':f'/intent/source/{seed}','binary':f'/intent/bin/{seed}','config':f'/intent/config/{seed}'},
            'training':{'seeds':[seed],'max_updates':1000},'inactive':{'unchanged':True}},
        'hardware':{'device':'cuda:0','name':'UNADMITTED-GPU','cuda_runtime':'13.2','device_count':1,
            'cuda_driver_api_version':13020,'exact_driver_build':'UNADMITTED-build'},
        'identity':{'code':{'revision':'a'*40,'dirty':True,'installed_build_revision':None},
            'module_sha256':{'gf_common/example.py':'b'*64},'cuda_distributions':{'cuda-bindings':'UNADMITTED-version'},
            'runtime':{'torch':'UNADMITTED-version'},'numerics':{'precision':'ieee-fp32'},'package':'UNADMITTED-version',
            'integration':{'binary_sha256':'c'*64,'server_config_sha256':'d'*64},
            'configuration_inputs_sha256':digest({'seed-specific-original-inputs':seed}),
            'preprocessing':{'backend':'rust','reader_contract':'participant-structured-v2','binary':{
                'extension_path':'/intent/fixed-extension.so','extension_sha256':'e'*64,
                'build':{'native_source_sha256':'f'*64}}},'process_envelope_recipe':'UNADMITTED-fixed-recipe',
            'formal_parent':{'relative_path':'formal-declaration.json','sha256':digest({'original-declaration':seed}),
                'seed_slot':seed,'recipe_id':'UNADMITTED-fixed-recipe','recipe_sha256':'0'*64,'algorithm_recipe_sha256':'1'*64}}}


def test_full_cohort_identity_keeps_only_approved_per_seed_and_config_path_exclusions():
    neural=dto()['neural_cohort']
    first=comparison_state(17)
    for seed in (29,43):
        assert _cohort_comparison(first,neural)==_cohort_comparison(comparison_state(seed),neural)
    assert first['effective_config']['training']['seeds']==[17]
    assert first['identity']['formal_parent']['seed_slot']==17


@pytest.mark.parametrize('field',['device','name','cuda_runtime','device_count','cuda_driver_api_version','exact_driver_build'])
def test_cohort_refuses_different_actual_hardware_even_when_execution_and_code_match(field):
    first=comparison_state();other=deepcopy(first)
    old=other['hardware'][field]
    other['hardware'][field]=old+1 if type(old) is int else old+'-changed'
    neural=dto()['neural_cohort']
    assert _cohort_comparison(first,neural)!=_cohort_comparison(other,neural)


@pytest.mark.parametrize('change',['module_sha256','cuda_distributions','runtime','numerics','package','code',
    'integration','process_envelope_recipe','preprocessing-reader','preprocessing-native-source','preprocessing-extension-path'])
def test_cohort_refuses_all_non_run_specific_identity_changes_without_added_path_normalization(change):
    first=comparison_state();other=deepcopy(first);identity=other['identity']
    if change=='module_sha256':identity[change]['gf_common/example.py']='2'*64
    elif change=='cuda_distributions':identity[change]['cuda-bindings']='changed'
    elif change=='runtime':identity[change]['torch']='changed'
    elif change=='numerics':identity[change]['precision']='changed'
    elif change=='code':identity[change]['installed_build_revision']='changed'
    elif change=='integration':identity[change]['binary_sha256']='2'*64
    elif change=='preprocessing-reader':identity['preprocessing']['reader_contract']='changed'
    elif change=='preprocessing-native-source':identity['preprocessing']['binary']['build']['native_source_sha256']='2'*64
    elif change=='preprocessing-extension-path':identity['preprocessing']['binary']['extension_path']='/intent/different-extension.so'
    else:identity[change]='changed'
    neural=dto()['neural_cohort']
    assert _cohort_comparison(first,neural)!=_cohort_comparison(other,neural)


@pytest.mark.parametrize('record',['identity','hardware'])
@pytest.mark.parametrize('change',['unknown','missing'])
def test_cohort_identity_and_hardware_comparison_is_closed(record,change):
    state=comparison_state()
    if change=='unknown':state[record]['unknown-field']='must-not-ignore'
    else:state[record].pop('module_sha256' if record=='identity' else 'exact_driver_build')
    with pytest.raises(ContractError):_cohort_comparison(state,dto()['neural_cohort'])


def test_cohort_still_compares_common_recipe_and_full_inactive_config():
    first=comparison_state();other=deepcopy(first)
    neural=dto()['neural_cohort'];changed=deepcopy(neural);changed['algorithm_recipe_sha256']='2'*64
    assert _cohort_comparison(first,neural)!=_cohort_comparison(first,changed)
    other['effective_config']['inactive']['unchanged']=False
    assert _cohort_comparison(first,neural)!=_cohort_comparison(other,neural)
