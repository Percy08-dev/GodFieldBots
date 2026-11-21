"""Persisted shared receipts are correspondence data, never live capabilities."""
import json
from pathlib import Path

import pytest

from gf_common.domain import ContractError, digest


def test_service_pin_rejects_registration_for_another_config_or_run(tmp_path):
    from gf_common.shared_inference import shared_runtime
    from gf_models.arrays import sha256
    from gf_training.journal import atomic_json
    from gf_training.shared_serving import validate_service_references
    service='a'*32
    registration={'schema_version':1,'kind':'shared-service-registration','service_id':service,
        'execution':'b'*32,'job_id':'b'*32,'run_id':'run-a','config_sha256':'c'*64,
        'libraries':{'torchrl':'0.15.0','tensordict':'0.15.2'},'runtime_sha256':digest(shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1)),
        'epoch':1,'broker_pid':123,'gpu_pid':124,'device':'cuda:0'}
    folder=tmp_path/'services'/service
    folder.mkdir(parents=True)
    atomic_json(folder/'registration.json',registration)
    generation=tmp_path/'generations/000000.json'
    generation.parent.mkdir()
    atomic_json(generation,{'inference_sha256':'d'*64})
    record={'inference_sha256':'d'*64,'artifact_files':{'weights-000000.npz':'e'*64}}
    record['parent_controls']={'snapshot_pool':[0]}
    binding={'run_id':'run-a','token':'f'*32,'generation':0,'inference_sha256':'d'*64,
        'generation_sha256':sha256(generation),'policy_sha256':'1'*64,
        'weights_sha256':digest(record['artifact_files']),'snapshot_identity':None,
        'service_id':service,'service_key':1,'broker_pid':123,'gpu_pid':124,
        'cache_identity':digest({'checkpoint':'d'*64,'service_id':service})}
    row={'schema_version':1,'kind':'shared-service-cache-reference','phase':'acquire','binding':binding}
    reference=folder/'references'/binding['token']
    reference.mkdir(parents=True)
    atomic_json(reference/'acquire.json',row)
    state={'run_id':'run-a','config_sha256':'c'*64,'status':'running',
        'execution':{'service':{'libraries':{'torchrl':'0.15.0','tensordict':'0.15.2'},'runtime':shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1)}}}
    class Scan:
        root=tmp_path
        def object(self,path):return json.loads(Path(path).read_text())
        def verified_digest(self,path):return sha256(path)
    state.update(model_version='3',model_id='mlp',effective_config={'model':{'version':'3','model_id':'mlp'},'actions':{}},
        action_contract={},rule_signature='1'*64,layout_signature='2'*64,purpose='qualified_game',synthetic_capacity=False)
    state['execution']['inference']={'binding':{},'cuda_visible_devices':None}
    options={'version':'3','checkpoint':str(tmp_path/'artifacts/000000'),'expected_manifest_digest':'d'*64,
        'model':{'version':'3','model_id':'mlp'},'inference':{},'cuda_visible_devices':None,'action_config_hash':digest({}),
        **{key:state[key] for key in ('action_contract','rule_signature','layout_signature','purpose','synthetic_capacity')}}
    binding['policy_sha256']=digest({'model_id':'mlp','policy_options':options})
    atomic_json(reference/'acquire.json',row)
    registry={'service_references':{binding['token']:row}}
    from gf_training.shared_inventory import new_invocation
    stamp=new_invocation(0)
    stamp.update(service_id=service,execution=registration['execution'],job_id=registration['job_id'],registration_sha256=digest(registration))
    state['execution']['service']['record_version']=2
    state['invocations']=[{'mode':'fresh','shared_service':stamp}]
    Scan.directories=lambda self,path,names:None  # This focused test audits registration values; real scans test namespace fences.
    validate_service_references(registry,state,[record],Scan())
    for key,value in (('config_sha256','2'*64),('run_id','run-b'),('runtime_sha256','3'*64),('broker_pid',125),('gpu_pid',126)):
        atomic_json(folder/'registration.json',registration|{key:value})
        with pytest.raises(ContractError):validate_service_references(registry,state,[record],Scan())
