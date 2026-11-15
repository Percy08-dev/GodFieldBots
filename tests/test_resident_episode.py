"""Shared original protocol3 episode semantics and complete resident reset."""
from dataclasses import asdict

import pytest

from gf_common.card_instance_rules import rule_signature
from gf_common.card_instance_runtime import game_binding
from gf_common.domain import ContractError
from gf_common.structured_wire import delta_dict, frame_dict
from gf_models.structured_runtime import own_game_worker_settings_v3
from tests.test_card_instance_learning import ACTIONS, RULES


def initial(kind='random', seed=17, options=None):
    caps, costs, _, _ = own_game_worker_settings_v3(kind, ACTIONS, options, rules=RULES)
    return {'protocol_version': 3, 'model_id': kind, 'sampling_seed': seed,
        'actions': asdict(ACTIONS), 'rules': asdict(RULES), 'caps': asdict(caps),
        'numeric_costs': asdict(costs), **game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES)),
        **({} if options is None else {'policy_options': options})}


def message(frame, delta):
    return {'protocol_version': 3, 'kind': 'select', 'frame': frame_dict(frame, layout=frame.layout),
        'delta': delta_dict(delta, layout=frame.layout),
        **game_binding(ACTIONS.action_contract, rule_signature(ACTIONS.action_contract, RULES))}


def test_original_episode_reset_drops_bank_previous_rng_and_denies_use():
    from gf_simulator_client.card_instance_episode import CardInstanceEpisode
    episode = CardInstanceEpisode(initial())
    assert episode.ready['kind'] == 'ready'
    episode.reset()
    assert episode.private_state_empty()
    with pytest.raises(ContractError, match='retired'):
        episode.select({})


def test_cache_reuse_checks_identity_and_resets_inactive_entries():
    from gf_models.resident_cache import ResidentPolicyCache
    cache = ResidentPolicyCache()
    class Policy:
        def __init__(self): self.seeds = []
        def reset(self, seed): self.seeds.append(seed)
    policies = []
    def load():
        value = Policy()
        policies.append(value)
        return value
    a = cache.acquire('a'*64, weight_bytes=16, load=load, validate=lambda: None)
    cache.release()
    b = cache.acquire('b'*64, weight_bytes=16, load=load, validate=lambda: None)
    cache.release()
    assert a is not b
    assert cache.acquire('a'*64, weight_bytes=16, load=load, validate=lambda: None) is a
    assert len(policies) == 2
    cache.reset_all()
    assert a.seeds == b.seeds == [0, 0]
    assert cache.private_state_empty()
    cache.release()
    with pytest.raises(ContractError, match='weight'):
        cache.acquire('a'*64, weight_bytes=17, load=load, validate=lambda: None)
    assert len(policies) == 2


def test_cache_reserves_bound_before_loading_and_never_evicts_active():
    from gf_models.resident_cache import ResidentPolicyCache
    cache = ResidentPolicyCache()
    loaded = []
    class Policy:
        def reset(self, seed): pass
    def load(): loaded.append(1); return Policy()
    with pytest.raises(ContractError, match='capacity'):
        cache.acquire('a'*64, weight_bytes=128*1024**2+1, load=load, validate=lambda: None)
    assert not loaded
    cache.acquire('a'*64, weight_bytes=1, load=load, validate=lambda: None)
    with pytest.raises(ContractError, match='active'):
        cache.evict('a'*64)
    cache.release()
    for index in range(1, 7):
        for identity in cache.eviction_candidates(1):
            cache.evict(identity)
        cache.acquire(f'{index:064x}', weight_bytes=1, load=load, validate=lambda: None)
        cache.release()
    assert len(cache.identities) == 6 and 'a'*64 not in cache.identities


def test_resident_evaluation_runtime_binds_both_actual_linkage_and_registry():
    from gf_common.domain import digest
    from gf_common.hpo_resident import checked_evaluation_runtime
    linkage = {'evaluation_id': 'panel-1', 'gpu_ticket': 'a'*32,
        'left_parent': {'inference_sha256':'b'*64}, 'right_parent': {'inference_sha256':'c'*64}}
    registry = {'left': {'model_id':'mlp'}, 'right': {'model_id':'gru'}}
    value = {'schema_version':1,'kind':'hpo-resident-evaluation-runtime',
        'runtime':{'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728},
        'evaluation_id':'panel-1','job_ticket':'a'*32,'linkage_sha256':digest(linkage),'policy_registry_sha256':digest(registry)}
    assert checked_evaluation_runtime(value,linkage,registry) == value
    with pytest.raises(ContractError):
        checked_evaluation_runtime(value,{**linkage,'right_parent':linkage['left_parent']},registry)
    with pytest.raises(ContractError):
        checked_evaluation_runtime(value,linkage,{})


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_partial_new_weight_load_preserves_original_frozen_network(tmp_path,monkeypatch,kind):
    import gf_models.structured_artifacts as artifacts
    from gf_models.resident_cache import ResidentPolicyCache
    from gf_models.structured_policy import GameStructuredPolicyV4
    from tests.test_hpo_numeric import learner_for
    learner=learner_for(kind)
    first=tmp_path/'first'
    second=tmp_path/'second'
    a=learner.save(first)
    with learner.torch.no_grad(): next(learner.network.parameters()).add_(.125)
    b=learner.save(second)
    def load(path,receipt):
        return GameStructuredPolicyV4.from_artifact(path,receipt['inference_digest'],learner.model,learner.binding,
            learner.action_config_hash,action_contract=ACTIONS.action_contract,rules=RULES)
    cache=ResidentPolicyCache()
    frozen=cache.acquire('a'*64,weight_bytes=1,load=lambda:load(first,a),validate=lambda:None)
    original={key:value.clone() for key,value in frozen.network.state_dict().items()}
    cache.release()
    def partial(network,arrays):
        with learner.torch.no_grad(): next(network.parameters()).add_(100)
        raise RuntimeError('injected partial copy into DISTINCT new network')
    monkeypatch.setattr(artifacts,'restore_weights',partial)
    with pytest.raises(RuntimeError,match='partial copy'):
        cache.acquire('b'*64,weight_bytes=1,load=lambda:load(second,b),validate=lambda:None)
    assert cache.identities==('a'*64,) and cache.active is None
    assert all(learner.torch.equal(value,frozen.network.state_dict()[key]) for key,value in original.items())


@pytest.mark.parametrize('fault',[None,'runtime','receipt','fresh'])
def test_collection7_evaluation_reopens_leaf_against_original_linkage_registry(tmp_path,fault):
    import copy
    import time

    from gf_cli.hpo_evaluation import check_evaluation_execution, check_evaluation_receipt_runtime
    from gf_common.card_instance_collection import (
        checked_hpo_evaluation_manifest,
        hpo_evaluation_role,
    )
    from gf_common.domain import digest
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.structured_source import _SourceOwner
    from tests.hpo_resident_support import retirement_fact
    from tests.test_hpo_evaluation_role import actual_metadata
    directory,manifest,actions=actual_metadata(tmp_path)
    original=copy.deepcopy(manifest)
    runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728}
    linkage=manifest['linkage']
    leaf={'schema_version':1,'kind':'hpo-resident-evaluation-runtime','runtime':runtime,
        'evaluation_id':linkage['evaluation_id'],'job_ticket':linkage['gpu_ticket'],
        'linkage_sha256':digest(linkage),'policy_registry_sha256':digest(manifest['policy_bindings'])}
    retirement=retirement_fact(linkage['evaluation_id'])
    retirement.update(collection_id=linkage['evaluation_id'],job_ticket=linkage['gpu_ticket'])
    for owner in retirement['owners']:
        owner['job_ticket']=linkage['gpu_ticket']
        for key in ('end_ack','park_ack'):
            owner[key]['job_ticket']=linkage['gpu_ticket']
            owner[key+'_sha256']=digest(owner[key])
    manifest.update(schema_version=7,execution_runtime=leaf,logical_retirement=retirement)
    receipt={'schema_version':2,'execution_runtime_sha256':digest(leaf)}
    if fault=='runtime': manifest['execution_runtime']['job_ticket']='f'*32
    elif fault=='receipt': receipt['execution_runtime_sha256']='0'*64
    elif fault=='fresh': manifest['schema_version']=6
    if fault is not None:
        with pytest.raises(ContractError):
            checked_hpo_evaluation_manifest(manifest)
            check_evaluation_receipt_runtime(receipt,manifest)
    else:
        checked_hpo_evaluation_manifest(manifest)
        check_evaluation_execution(manifest,runtime)
        check_evaluation_receipt_runtime(receipt,manifest)
        owner=_SourceOwner(directory,PreparationConfig(),time.monotonic()+30)
        rows,specs=inspect_game_manifest(owner,manifest,actions,'validation',role=hpo_evaluation_role(4))
        assert len(rows)==len(specs)==4
        checked_hpo_evaluation_manifest(original)
        check_evaluation_execution(original,None)
        with pytest.raises(ContractError): check_evaluation_execution(original,runtime)


@pytest.mark.parametrize('left_resident,right_resident,choice,accepted',[
    (False,False,'omitted',True),(True,False,'omitted',False),(False,True,'omitted',False),
    (True,True,'fresh',True),(False,False,'resident',True),(True,False,'resident',True)])
def test_comparison_runtime_choice_is_explicit_and_independent(tmp_path,left_resident,right_resident,choice,accepted):
    from gf_cli.hpo_config import campaign_config
    from gf_cli.hpo_evaluation import comparison_execution_runtime
    from tests.test_hpo_config import campaign_raw
    runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728}
    def config(resident):
        raw=campaign_raw()
        if resident: raw.update(schema_version=3,runtime=runtime)
        return campaign_config(raw,origin=tmp_path).config
    left,right=config(left_resident),config(right_resident)
    arguments={} if choice=='omitted' else {'choice':None if choice=='fresh' else runtime}
    if accepted:
        assert comparison_execution_runtime(left,right,**arguments)==(runtime if choice=='resident' else None)
    else:
        with pytest.raises(ContractError,match='explicit evaluation runtime'):
            comparison_execution_runtime(left,right,**arguments)


def test_resident_rejects_visibility_change_before_allocator_or_environment_mutation(tmp_path):
    import subprocess
    import sys

    from gf_common.domain import canonical
    from gf_models.structured_binding import input_binding_game_v3, safe_options_game_v4
    from tests.test_hpo_numeric import learner_for
    learner=learner_for('mlp')
    path=tmp_path/'artifact'
    receipt=learner.save(path)
    options=safe_options_game_v4({'version':'4','checkpoint':str(path),'expected_manifest_digest':receipt['inference_digest'],
        'model':asdict(learner.model),'inference':{**asdict(learner.binding),'device':'cuda:0'},'cuda_visible_devices':'0',
        'action_config_hash':learner.action_config_hash,**input_binding_game_v3(ACTIONS.action_contract,RULES)},actions=ACTIONS,rules=RULES)
    payload=initial('mlp',17,options)
    script=r'''
import copy,json,os,sys
initial=json.loads(sys.argv[1])
from gf_simulator_client.resident_bootstrap import ResidentEpisodeTransport
from gf_simulator_client.card_instance_episode import CardInstanceEpisode
from gf_models.structured_policy import GameStructuredPolicyV4
from gf_common.domain import ContractError
calls=[]
class Stub:
    version='CPU allocator stub only; no GPU placement claim'
    def reset(self,seed): pass
def allocate(*args,**kwargs):
    calls.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
    return Stub()
GameStructuredPolicyV4.from_artifact=allocate
transport=ResidentEpisodeTransport(1)
transport.checkpoint=initial['policy_options']['expected_manifest_digest']
transport.episode=CardInstanceEpisode(initial,policy_loader=transport._load)
transport._reset()
other=copy.deepcopy(initial)
other['policy_options']['cuda_visible_devices']='1'
try:
    CardInstanceEpisode(other,policy_loader=transport._load)
except ContractError:
    pass
else:
    raise AssertionError('resident accepted conflicting process visibility')
assert calls==['0'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
transport.episode=CardInstanceEpisode(initial,policy_loader=transport._load)
transport._reset()
assert calls==['0']
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
try:
    CardInstanceEpisode(initial,policy_loader=transport._load)
except ContractError:
    pass
else:
    raise AssertionError('resident accepted backend environment mutation on cache hit')
assert calls==['0'] and os.environ['CUBLAS_WORKSPACE_CONFIG']==':16:8'
'''
    result=subprocess.run([sys.executable,'-E','-c',script,canonical(payload)],capture_output=True,text=True,timeout=20,check=False)
    assert result.returncode==0,result.stderr
