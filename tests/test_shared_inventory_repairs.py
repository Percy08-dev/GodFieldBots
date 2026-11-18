"""Integrated review regressions: absent evidence cannot erase required custody."""
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.shared_inference import ServingBudget


def test_empty_service_reference_map_cannot_erase_committed_shared_collection(tmp_path):
    from gf_training.shared_serving import validate_service_references
    state={'status':'complete','updates':1,'run_variant':'hpo-ppo-owned-shared-v1','invocations':[]}
    with pytest.raises(ContractError):
        validate_service_references({'service_references':{}},state,[{},{}],SimpleNamespace(root=tmp_path))


def test_owned_preparation_failure_releases_actual_snapshot_binding_and_budget(tmp_path):
    from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_training.shared_serving import SharedServingController
    budget=ServingBudget(1024**2)
    banks=ReadonlyRamBankSet.build({'w':np.zeros(8,dtype=np.float32)},budget)
    class Origin:pass
    origin=Origin()
    snapshot=_issue_actor_snapshot(origin,inference=banks,metadata={'generation':0,'snapshot_identity':'a'*64,'inference_digest':'b'*64},
        check_origin=lambda value:None,budget=budget)
    origin._actor_capability=snapshot
    registry=SimpleNamespace(snapshots={0:SimpleNamespace(actor_capability=snapshot)})
    service=SimpleNamespace(runtime={'cache_weight_bytes':1024**2,'cache_entries':6})
    ops=SimpleNamespace(ownership=SimpleNamespace(directory=tmp_path))
    controller=SharedServingController(ops,None,{'run_id':'run'},service,owned_registry=registry)
    before=budget.used
    artifacts=SimpleNamespace(journal_metadata=lambda label:(_ for _ in ()).throw(RuntimeError('metadata failure')))
    binding=SimpleNamespace(model_id='mlp',policy_options={'expected_manifest_digest':'b'*64,'checkpoint':str(tmp_path/'artifacts/000000')})
    try:
        with pytest.raises(RuntimeError,match='metadata failure'):controller.ensure({'policy':binding},artifacts)
        assert not snapshot._state.bindings
        assert budget.used==before
        assert all(bank._pins==0 for bank in banks.banks)
    finally:
        for held in tuple(snapshot._state.bindings.values()):held.close(controller)
        snapshot._retire_origin(origin)
        banks.close()
    assert budget.used==0


@pytest.fixture
def inventory(tmp_path):
    import time

    from gf_common.domain import clone, digest
    from gf_models.arrays import sha256
    from gf_models.structured_sealed import SealedScan
    from gf_training.journal import atomic_json
    from gf_training.shared_inventory import new_invocation
    sid='a'*32
    registration={'schema_version':1,'kind':'shared-service-registration','service_id':sid,'execution':'b'*32,
        'job_id':'c'*32,'run_id':'run','config_sha256':'d'*64,'runtime_sha256':digest({}),
        'epoch':1,'broker_pid':123,'gpu_pid':124,'device':'cuda:0','libraries':{}}
    folder=tmp_path/'services'/sid
    folder.mkdir(parents=True)
    atomic_json(folder/'registration.json',registration)
    refs={}
    publications=[]
    records=[{'inference_sha256':'1'*64,'parent_controls':{'snapshot_pool':[0]}},
        {'inference_sha256':'2'*64,'parent_controls':{'snapshot_pool':[0]}}]
    for g,token in enumerate(('e'*32,'f'*32)):
        checkpoint=records[g]['inference_sha256']
        ack={'accepted':True,'checkpoint':checkpoint,'publication':g+1}
        row={'schema_version':1,'ordinal':g+1,'token':token,'checkpoint':checkpoint,'generation':g,'ack_sha256':digest(ack)}
        publications.append(row)
        (folder/'references'/token).mkdir(parents=True)
        atomic_json(folder/'references'/token/'publication.json',row|{'ack':ack})
        refs[token]={'binding':{'service_id':sid,'generation':g,'inference_sha256':checkpoint}}
    usage={'schema_version':1,'service_id':sid,'execution':'b'*32,'job_id':'c'*32,
        'registration_sha256':digest(registration),'config_sha256':'d'*64,'runtime_sha256':digest({}),
        'generation':0,'batch_id':'batch-000000','start':{'requests':0,'publications':0},
        'end':{'requests':3,'publications':1},'bindings':{'mlp':{'checkpoint':'1'*64,'token':'e'*32}}}
    update={'service_usage':usage,'source_proof':{'service_usage':clone(usage)},
        'policy_bindings':{'mlp':{'model_id':'mlp','policy_options':{'expected_manifest_digest':'1'*64}}}}
    (tmp_path/'updates').mkdir()
    atomic_json(tmp_path/'updates/000001.json',update)
    records[1]['update_sha256']=sha256(tmp_path/'updates/000001.json')
    retirement={'schema_version':1,'kind':'shared-service-retirement','accepted':True,'broker_pid':123,
        'service_id':sid,'registration_sha256':digest(registration),'successful':True,'relays_joined':True,
        'gpu_pid':124,'gpu_exitcode':0,'pending':0,'active_relays':0,'failures':0,'requests':3,'publications':2}
    atomic_json(folder/'retirement.json',retirement)
    proof={'key':1,'service_id':sid,'broker_pid':123,'broker_outcome':'reaped','gpu_pid':124,'gpu_outcome':'reaped',
        'gpu_exitcode':0,'registration_sha256':digest(registration),'retirement_sha256':digest(retirement)}
    atomic_json(folder/'proof.json',proof)
    stamp=new_invocation(0)
    stamp.update(finished=True,end_generation=1,service_id=sid,execution='b'*32,job_id='c'*32,
        registration_sha256=digest(registration),proof_sha256=digest(proof),publications=publications)
    state={'run_id':'run','status':'complete','updates':1,'config_sha256':'d'*64,
        'execution':{'service':{'record_version':2,'runtime':{},'libraries':{}}},
        'invocations':[{'mode':'fresh','shared_service':stamp}]}
    def check():
        scan=SealedScan(max_owned_bytes=16*1024**2,deadline=time.monotonic()+10)
        scan.root=tmp_path
        from gf_training.shared_inventory import validate_inventory
        result=validate_inventory({'service_references':refs},state,records,scan)
        scan.finalize()
        return result
    return SimpleNamespace(check=check,state=state,refs=refs,records=records,folder=folder,root=tmp_path)


def test_complete_inventory_accepts_last_commit_publication_tail_and_noop_resume(inventory):
    from gf_training.shared_inventory import new_invocation
    assert len(inventory.check())==1
    noop=new_invocation(1)
    noop.update(finished=True,end_generation=1)
    inventory.state['invocations'].append({'mode':'resume','shared_service':noop})
    assert len(inventory.check())==1


def test_single_uncommitted_drained_tail_reconciles_without_inventing_an_update(inventory):
    import json

    from gf_common.domain import clone, digest
    from gf_training.journal import atomic_json
    usage=clone(json.loads((inventory.root/'updates/000001.json').read_text())['service_usage'])
    usage.update(generation=1,batch_id='batch-000001',start=usage['end'],end={'requests':4,'publications':2},
        bindings={'old':{'checkpoint':'1'*64,'token':'e'*32},'current':{'checkpoint':'2'*64,'token':'f'*32}})
    stamp=inventory.state['invocations'][0]['shared_service']
    stamp['tail_usage']=usage
    retired=json.loads((inventory.folder/'retirement.json').read_text())
    retired['requests']=4
    atomic_json(inventory.folder/'retirement.json',retired)
    proof=json.loads((inventory.folder/'proof.json').read_text())
    proof['retirement_sha256']=digest(retired)
    atomic_json(inventory.folder/'proof.json',proof)
    stamp['proof_sha256']=digest(proof)
    assert len(inventory.check())==1 and len(inventory.records)==2
    stamp['tail_usage']['generation']=0
    stamp['tail_usage']['batch_id']='batch-000000'
    with pytest.raises(ContractError):inventory.check()


@pytest.mark.parametrize('damage',['all_refs','one_ref','all_services','one_service','proof','publication',
    'job','execution','checkpoint','request_count','publication_count','all_invocations','proof_key','retirement_boolean'])
def test_independent_inventory_rejects_omissions_and_wrong_identity(inventory,damage):
    import shutil

    from gf_common.domain import digest
    from gf_training.journal import atomic_json
    if damage=='all_refs':inventory.refs.clear()
    elif damage=='one_ref':inventory.refs.pop('e'*32)
    elif damage=='all_services':shutil.rmtree(inventory.root/'services')
    elif damage=='one_service':shutil.rmtree(inventory.folder)
    elif damage=='proof':(inventory.folder/'proof.json').unlink()
    elif damage=='publication':(inventory.folder/'references'/('f'*32)/'publication.json').unlink()
    elif damage=='all_invocations':inventory.state['invocations'].clear()
    elif damage in ('job','execution'):inventory.state['invocations'][0]['shared_service'][damage+'_id' if damage=='job' else damage]='9'*32
    elif damage=='checkpoint':inventory.state['invocations'][0]['shared_service']['publications'][1]['checkpoint']='8'*64
    else:
        import json
        retirement=json.loads((inventory.folder/'retirement.json').read_text())
        if damage=='retirement_boolean':retirement['pending']=False
        elif damage!='proof_key':retirement['requests' if damage=='request_count' else 'publications']+=1
        atomic_json(inventory.folder/'retirement.json',retirement)
        proof=json.loads((inventory.folder/'proof.json').read_text())
        if damage=='proof_key':proof['key']=2
        proof['retirement_sha256']=digest(retirement)
        atomic_json(inventory.folder/'proof.json',proof)
        inventory.state['invocations'][0]['shared_service']['proof_sha256']=digest(proof)
    with pytest.raises((ContractError,OSError)):inventory.check()


def test_genuine_pre_service_generation_zero_has_no_required_service_namespace(tmp_path):
    import time

    from gf_models.structured_sealed import SealedScan
    from gf_training.shared_inventory import new_invocation, validate_inventory
    scan=SealedScan(max_owned_bytes=1024**2,deadline=time.monotonic()+10)
    scan.root=tmp_path
    state={'status':'running','execution':{'service':{'record_version':2}},'invocations':[
        {'mode':'fresh','shared_service':new_invocation(0)}]}
    assert validate_inventory({'service_references':{}},state,[{}],scan)=={}


def test_failed_service_startup_is_not_relabelled_never_created():
    from gf_training.shared_inventory import finish_invocation, new_invocation
    failure=RuntimeError('original constructor failure')
    service=SimpleNamespace(service_id='a'*32,execution='b'*32,registration=None,proof=None,
        process=object(),closed=False,_no_child_error=None)
    failure._shared_service_owner=service
    operations=SimpleNamespace(shared_controller=None,ledger=SimpleNamespace(service=SimpleNamespace(owner=service)))
    stamp=new_invocation(0)
    finish_invocation(operations,{'updates':0},stamp)
    assert stamp['service_id']==service.service_id and stamp['registration_sha256'] is None
    assert stamp['proof_sha256'] is None and stamp['finished'] is True
    operations.ledger.service=None
    stamp=new_invocation(0)
    finish_invocation(operations,{'updates':0},stamp,error=failure)
    assert stamp['service_id']==service.service_id


@pytest.mark.parametrize('owned',[False,True])
@pytest.mark.parametrize('stage',['registry','digest','row','track','journal','write','publish'])
def test_preparation_and_publication_faults_keep_real_banks_and_bindings_reachable(tmp_path,monkeypatch,owned,stage):
    from gf_common.hpo_actor_snapshot import _issue_actor_snapshot
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_training import shared_serving as module
    from gf_training.journal import atomic_json
    budget=ServingBudget(1024**2)
    banks=ReadonlyRamBankSet.build({'w':np.zeros(8,dtype=np.float32)},budget)
    class Origin:pass
    origin=Origin()
    snapshot=_issue_actor_snapshot(origin,inference=banks,metadata={'generation':0,'snapshot_identity':'a'*64,'inference_digest':'b'*64},
        check_origin=lambda value:None,budget=budget)
    origin._actor_capability=snapshot
    metadata={'inference_manifest':{'weights':{'files':{'weights-0.npz':'c'*64}}}}
    artifact_registry={'service_references':{}}
    atomic_json(tmp_path/'artifact-registry.json',artifact_registry)
    registry=SimpleNamespace(snapshots={0:SimpleNamespace(actor_capability=snapshot)},registry=artifact_registry,
        writer=SimpleNamespace(write=lambda *args,**kwargs:None)) if owned else None
    def fail(*args,**kwargs):raise RuntimeError('fixture '+stage)
    service=SimpleNamespace(runtime={'cache_weight_bytes':1024**2,'cache_entries':6},service_id='d'*32,
        registration={'broker_pid':123,'gpu_pid':124},directory=tmp_path/'services'/('d'*32),deadline=100.,
        _persist=lambda *args:None,close=lambda:{'confirmed':True},failed=False,
        publish=lambda *args,**kwargs:{'accepted':True,'checkpoint':'b'*64,'publication':1})
    controller=module.SharedServingController(SimpleNamespace(ownership=SimpleNamespace(directory=tmp_path)),None,
        {'run_id':'run','updates':1,'generation_sha256':'e'*64},service,owned_registry=registry)
    controller._generation_digest=lambda generation:'f'*64
    artifacts=SimpleNamespace(journal_metadata=lambda label:metadata,_scan=SimpleNamespace(finalize=lambda:None))
    monkeypatch.setattr(module,'read_bundle',lambda *args,**kwargs:{'w':np.zeros(8,dtype=np.float32)})
    if stage=='registry':controller._registry=fail
    elif stage=='digest':controller._generation_digest=fail
    elif stage=='row':
        class BrokenRegistration(dict):
            def __getitem__(self,key):
                if key=='gpu_pid':fail()
                return super().__getitem__(key)
        service.registration=BrokenRegistration(service.registration)
    elif stage=='track':
        class BrokenTracking(dict):
            def __setitem__(self,key,value):fail()
        controller.pins=BrokenTracking()
    elif stage=='journal':controller._record=fail
    elif stage=='write':controller._write=fail
    else:service.publish=fail
    binding=SimpleNamespace(model_id='mlp',policy_options={'expected_manifest_digest':'b'*64,
        'checkpoint':str(tmp_path/'artifacts/000000'),'model':{},'inference':{}})
    before=budget.used
    try:
        with pytest.raises(RuntimeError,match='fixture '+stage):controller.ensure({'mlp':binding},artifacts)
        tracked=stage in ('journal','write','publish')
        assert bool(controller.pins) is tracked
        assert bool(snapshot._state.bindings) is (tracked and owned)
        assert (controller.budget.used>0) is (tracked and not owned)
        if tracked:
            # Persistence can still fail; confirmed service retirement must free
            # every actual binding/staging bank in its independent finalizers.
            try:controller.close()
            except RuntimeError as error:assert str(error)=='fixture '+stage
        assert not controller.pins and not snapshot._state.bindings
        assert controller.budget.used==0 and budget.used==before
        assert all(bank._pins==0 for bank in banks.banks)
    finally:
        for held in tuple(snapshot._state.bindings.values()):held.close(controller)
        snapshot._retire_origin(origin)
        banks.close()
    assert budget.used==0
