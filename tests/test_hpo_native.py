"""HPO capability admission; structural support never grants Source authority."""
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig


def test_missing_hpo_capability_refuses_before_any_native_session():
    from gf_training.native_backend import hpo_reader_settings
    module = SimpleNamespace(build_info=lambda: [('card_instance_contract',
        'participant-structured-v2;journal6;rollout4;generation3;model3')])
    with pytest.raises(ContractError, match='HPO'):
        hpo_reader_settings(PreparationConfig(backend='rust', reader='participant-structured-v2'), module)


@pytest.fixture(scope='module')
def candidate():
    path=os.environ.get('GFB_HPO_NATIVE_CANDIDATE')
    if path is None:
        pytest.skip('explicit private candidate required; installed extension unchanged')
    path=Path(path).resolve(strict=True)
    assert path.is_relative_to('/tmp')
    spec=importlib.util.spec_from_file_location('gf_native',path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    sys.modules['gf_native']=module
    spec.loader.exec_module(module)
    assert callable(getattr(module.CardInstanceNativeSession,'authorize_hpo',None))
    return module


@pytest.fixture(scope='module')
def controlled_parent(tmp_path_factory):
    from tests.hpo_support import original_hpo_chain
    directory=tmp_path_factory.mktemp('hpo-native-held')/'parent'
    original_hpo_chain(directory)
    return directory


def _ffi_open(candidate,directory):
    from gf_common.domain import canonical
    settings={'memory_bytes':1024**3,'scratch_bytes':512*1024**2,
        'max_source_file_bytes':64*1024**2,'max_record_bytes':8*1024**2,'chunk_target_bytes':8*1024**2}
    plan={'schema_version':5,'mode':'historical_parent',
        'parents':[{'directory':str(directory),'require_complete':False}],'collections':[]}
    return candidate.CardInstanceNativeSession(canonical(plan).encode(),canonical(settings).encode(),time.monotonic()+90)


def test_actual_candidate_public_hpo_admission_and_old_markers(candidate):
    from gf_training.native_backend import CARD_INSTANCE_CONTRACT, hpo_reader_settings
    settings=PreparationConfig(backend='rust',reader='participant-structured-v2')
    info=dict(candidate.build_info())
    assert info['card_instance_hpo']=='model4;collection6;proof3;numeric5;selection1;provider1'
    assert info['card_instance_contract']==CARD_INSTANCE_CONTRACT
    assert hpo_reader_settings(settings,candidate)==asdict(settings)
    assert hpo_reader_settings(settings,candidate,packed=True)==asdict(settings)


def _authorization(directory,pending):
    from gf_common.domain import canonical, digest
    from gf_common.schedule import serialize_schedule
    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_json
    from gf_models.structured_sealed import SEMANTIC_KEYS_V3
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_lineage import hpo_state, validate_hpo_chain
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.structured_rollout import _source_correspondence
    state,records=validate_hpo_chain(directory,max_owned_bytes=1024**3,deadline=time.monotonic()+90)
    assert canonical(pending['parent']['state'])==canonical(state)
    assert canonical(pending['parent']['generations'])==canonical(records)
    assert canonical(pending['parent']['model'])==canonical(asdict(hpo_state(state)[1]))
    initial=read_json(directory/'artifacts/000000/inference.json')
    semantics={key:initial[key] for key in SEMANTIC_KEYS_V3-{'reward_policy_version'}}
    collections=[]
    for raw in pending['collections']:
        generation=raw['generation']
        plan,label,_=recorded_hpo_plan(directory,state,records,generation,time.monotonic()+90)
        assert raw['linkage']==plan.linkage and raw['learner_label']==label
        assert raw['schedule_sha256']==digest(serialize_schedule(plan.groups,4))
        assert raw['bindings_sha256']==digest({key:asdict(value) for key,value in plan.bindings.items()})
        assert canonical(raw['semantics'])==canonical(semantics)
        proof=raw['original_proof']
        assert proof['source_set_sha256']==digest(proof['files'])
        context=plan.linkage['selection_context']
        selection=selection_for_counts(tuple(raw['episode_counts']),**{key:context[key]
            for key in ('learning_seed','batch','decision_budget','used_before')})
        control={'run_id':state['run_id'],'config_sha256':state['config_sha256'],
            'identity':state['identity'],'execution':state['execution'],'plan':plan.linkage,
            'files':{f'generations/{generation:06}.json':sha256(directory/'generations'/f'{generation:06}.json'),
                f'artifacts/{generation:06}/inference.json':records[generation]['inference_sha256']},
            'source_set_sha256':proof['source_set_sha256'],'recipe':'card-instance-hpo-source-controls-v1',
            'hpo_spec_sha256':state['hpo_spec_sha256']}
        assert canonical(control)==canonical(raw['control'])
        control['selection']=selection
        correspondence=_source_correspondence(digest(control),semantics,tuple(raw['episode_digests']),game=True,hpo=True)
        recipe=state['effective_config']['hpo_spec']['training_recipe']
        delta=recipe['epochs']*((selection['used_decisions']+recipe['minibatch_size']-1)//recipe['minibatch_size'])
        collections.append({'collection_index':raw['collection_index'],'selection':selection,
            'training_recipe':recipe,'expected_optimizer_step_delta':delta,'expected_correspondence':correspondence})
    files={f'generations/{index:06}.json':sha256(directory/'generations'/f'{index:06}.json') for index in range(len(records))}
    return {'schema_version':5,'pending_sha256':digest(pending),'parent':{'run_sha256':sha256(directory/'run.json'),
        'config_sha256':state['config_sha256'],'resume_identity':state['resume_identity'],
        'generation_files_sha256':digest(files),'registry_sha256':sha256(directory/'artifact-registry.json')},
        'collections':collections}


def test_actual_candidate_pending_and_final_same_session_control_quota(candidate,controlled_parent):
    from gf_common.domain import canonical
    session=_ffi_open(candidate,controlled_parent)
    try:
        pending_bytes=session.control_summary()
        pending=json.loads(pending_bytes)
        assert pending['kind']=='card-instance-hpo-pending-controls'
        assert all('source_proof' not in row and 'correspondence' not in row for row in pending['collections'])
        request=_authorization(controlled_parent,pending)
        raw=canonical(request).encode()
        assert len(raw)<=pending['authorization_max_bytes']
        before=session.control_reserved_bytes()
        final_bytes=session.authorize_hpo(raw)
        final=json.loads(final_bytes)
        assert final['kind']=='card-instance-hpo-source-controls'
        assert session.control_reserved_bytes()>before
        assert session.control_reserved_bytes()>=64*(len(pending_bytes)+len(final_bytes)+pending['authorization_max_bytes'])+32*1024**2
        for actual,expected in zip(final['collections'],request['collections'],strict=True):
            assert actual['selection']==expected['selection']
            assert canonical(actual['correspondence'])==canonical(expected['expected_correspondence'])
            assert actual['source_proof']['schema_version']==3 and actual['source_proof']['qualified'] is False
        packet=session.export_recorded_cohort(0)
        payload=json.loads(packet[0])
        assert payload['schema_version']==3 and payload['kind']=='card-instance-hpo-recorded-cohort-payload'
        assert len(payload['metadata']['episodes'])==8
        assert payload['metadata']['decision_count']==final['collections'][0]['selection']['used_decisions']
        assert payload['metadata']['selection']==final['collections'][0]['selection']
        del packet
        import gc
        gc.collect()
        session.retire_payload()
        with pytest.raises(ValueError): session.authorize_hpo(raw)
    finally: session.close()


def test_actual_candidate_oversize_authorization_refuses_before_parse_and_stays_failed(candidate,controlled_parent):
    session=_ffi_open(candidate,controlled_parent)
    try:
        pending=json.loads(session.control_summary())
        with pytest.raises(MemoryError,match='pre-admitted'): session.authorize_hpo(b'x'*(pending['authorization_max_bytes']+1))
        with pytest.raises(RuntimeError,match='closed'): session.recheck()
    finally: session.close()


def test_actual_candidate_same_native_inputs_reads_hpo_original_numeric(candidate,controlled_parent,tmp_path):
    import numpy as np

    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.test_card_instance_learning import ACTIONS
    plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(controlled_parent),
        'require_complete':False}],'collections':[]}
    settings=PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024)
    with _NativeGameInputs(plan,settings,actions=ACTIONS,max_owned_bytes=1024**3,
            deadline=time.monotonic()+90,native=candidate) as owner:
        assert owner.summary['kind']=='card-instance-hpo-source-controls'
        for index,collection in enumerate(owner.summary['collections']):
            owner.collection_index=index
            directory=controlled_parent/'numeric'/f'batch-{collection["generation"]:06}'
            rollout=owner.read_numeric(directory,sha256(directory/'rollout.json'))
            assert rollout.collected_decisions==collection['learner_decisions']
            assert rollout.used_decisions==collection['selection']['used_decisions']
            del rollout
            target=tmp_path/f'comparison-{index}'
            actual=owner.compare_recorded(index,target)
            original=read_json(directory/'rollout.json')
            expected_arrays=read_bundle(directory,original['data'])
            actual_arrays=read_bundle(target,actual['data'])
            assert set(actual_arrays)==set(expected_arrays)
            for name in actual_arrays:
                np.testing.assert_array_equal(actual_arrays[name],expected_arrays[name],err_msg=name)
    assert owner.cleanup_complete


@pytest.mark.parametrize('numeric_limit',[64*1024**2,1024**3])
def test_actual_native_read_forwards_declared_nondefault_numeric_limit(candidate,tmp_path,monkeypatch,numeric_limit):
    from gf_models.arrays import sha256
    from gf_training import hpo_rollout
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.hpo_support import original_hpo_chain
    from tests.test_card_instance_learning import ACTIONS
    directory=tmp_path/'parent'
    original_hpo_chain(directory,updates=1,numeric_budget_bytes=numeric_limit)
    calls=[]
    original=hpo_rollout.read_hpo_rollout
    def checked(*args,**kwargs):
        calls.append(kwargs.get('numeric_budget_bytes'))
        assert kwargs.get('numeric_budget_bytes')==numeric_limit
        return original(*args,**kwargs)
    monkeypatch.setattr(hpo_rollout,'read_hpo_rollout',checked)
    plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(directory),
        'require_complete':False}],'collections':[]}
    with _NativeGameInputs(plan,PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024),
            actions=ACTIONS,max_owned_bytes=1024**3,deadline=time.monotonic()+120,native=candidate) as owner:
        numeric=directory/'numeric/batch-000000'
        rollout=owner.read_numeric(numeric,sha256(numeric/'rollout.json'))
        assert rollout.used_decisions==rollout.collected_decisions==8
        del rollout
    assert calls==[numeric_limit,numeric_limit]
    assert owner.cleanup_complete


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_actual_candidate_three_models_partial_last_originals_and_arrays(candidate,tmp_path,kind):
    import numpy as np

    from gf_models.arrays import sha256
    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.hpo_support import original_hpo_chain
    from tests.test_card_instance_learning import ACTIONS
    directory=tmp_path/'parent'
    original_hpo_chain(directory,model_id=kind,decision_budget=17)
    plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(directory),
        'require_complete':False}],'collections':[]}
    settings=PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024)
    with _NativeGameInputs(plan,settings,actions=ACTIONS,max_owned_bytes=1024**3,
            deadline=time.monotonic()+120,native=candidate) as owner:
        last=owner.summary['collections'][-1]
        assert last['selection']['collected_decisions']==8 and last['selection']['used_decisions']==1
        actual=owner.compare_recorded(2,tmp_path/'regenerated')
        original=read_json(directory/'numeric/batch-000002/rollout.json')
        assert sum(len(episode['frames']) for episode in actual['episodes'])==8
        assert actual['decision_count']==1
        left=read_bundle(directory/'numeric/batch-000002',original['data'])
        right=read_bundle(tmp_path/'regenerated',actual['data'])
        assert set(left)==set(right)
        for name in left:
            np.testing.assert_array_equal(left[name],right[name],err_msg=name)
        owner.collection_index=2
        rollout=owner.read_numeric(directory/'numeric/batch-000002',sha256(directory/'numeric/batch-000002/rollout.json'))
        del rollout


def test_actual_hpo_packet_view_keeps_numeric_guard_after_close(candidate,controlled_parent):
    import gc

    from gf_common.domain import canonical
    session=_ffi_open(candidate,controlled_parent)
    pending=json.loads(session.control_summary())
    session.authorize_hpo(canonical(_authorization(controlled_parent,pending)).encode())
    packet=session.export_recorded_cohort(0)
    array=packet[1][0]
    del packet
    gc.collect()
    assert session.numeric_used_bytes()>0
    session.close()
    assert session.numeric_used_bytes()>0 and session.used_bytes()>0
    del array
    gc.collect()
    assert session.numeric_used_bytes() is None and session.used_bytes()==0


@pytest.mark.parametrize('fault',['pending','seed','count','recipe','delta','correspondence'])
def test_actual_candidate_bad_authorization_is_terminal(candidate,controlled_parent,fault):
    from gf_common.domain import canonical
    session=_ffi_open(candidate,controlled_parent)
    try:
        pending=json.loads(session.control_summary())
        request=_authorization(controlled_parent,pending)
        row=request['collections'][0]
        if fault=='pending': request['pending_sha256']='f'*64
        elif fault=='seed': row['selection']['derived_seed']+=1
        elif fault=='count': row['selection']['episode_counts'][0]+=1
        elif fault=='recipe': row['training_recipe']={**row['training_recipe'],'epochs':3}
        elif fault=='delta': row['expected_optimizer_step_delta']+=1
        else: row['expected_correspondence']['control_fingerprint']='f'*64
        with pytest.raises(ValueError): session.authorize_hpo(canonical(request).encode())
        with pytest.raises(RuntimeError,match='closed'): session.recheck()
    finally: session.close()


def test_actual_candidate_pruned_historical_manifest_uses_owned_receipt(candidate,tmp_path):
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.hpo_lineage_support import prune_fixture_files
    from tests.hpo_support import original_hpo_chain
    from tests.test_card_instance_learning import ACTIONS
    directory=tmp_path/'parent'
    original_hpo_chain(directory)
    prune_fixture_files(directory,1)
    plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(directory),
        'require_complete':False}],'collections':[]}
    with _NativeGameInputs(plan,PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024),
            actions=ACTIONS,max_owned_bytes=1024**3,deadline=time.monotonic()+120,native=candidate) as owner:
        assert len(owner.summary['collections'])==3
        owner.compare_recorded(1,tmp_path/'regenerated')
    assert owner.cleanup_complete
    # Actual native Run5/Registry4 admission must reject a forged receipt even
    # when the historical manifest has already been physically pruned.
    from gf_training.journal import atomic_json
    registry=json.loads((directory/'artifact-registry.json').read_text())
    registry['generations']['1']['files']['inference.json']['deletion_receipt']['sha256']='f'*64
    atomic_json(directory/'artifact-registry.json',registry)
    with pytest.raises(ValueError):
        _ffi_open(candidate,directory)


def test_explicit_candidate_never_grants_live_hpo_source(candidate,tmp_path):
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.test_card_instance_learning import ACTIONS
    plan={'schema_version':5,'mode':'live_cohort','parents':[{'directory':str(tmp_path/'missing'),
        'require_complete':False}],'collections':[{'directory':str(tmp_path/'missing/collections/batch-000000'),
        'parent_index':0}],'cohort_collection_index':0}
    owner=_NativeGameInputs(plan,PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024),
        actions=ACTIONS,max_owned_bytes=1024**3,deadline=time.monotonic()+60,native=candidate)
    with pytest.raises(ContractError,match='historical diagnostic only'):
        owner.__enter__()


def test_source_one_byte_numeric_budget_refuses_before_publication(tmp_path):
    from gf_models.sharded_arrays import read_json
    from tests.hpo_support import original_hpo_chain
    rejected=tmp_path/'source-refused'
    with pytest.raises(ContractError,match='numeric token envelope exceeded before allocation'):
        original_hpo_chain(rejected,updates=1,numeric_budget_bytes=1)
    assert read_json(rejected/'run.json')['updates']==0
    assert not (rejected/'numeric/batch-000000').exists()
    assert not (rejected/'artifacts/000001').exists()


def test_actual_candidate_unqualified_components_one_byte_budget_refuses_native_payload(candidate,tmp_path):
    from gf_common.domain import canonical
    from tests.hpo_support import original_hpo_chain
    directory=tmp_path/'parent'
    # The native preallocation counterexample needs actual persisted components.
    # These unqualified CPU bytes do not claim a production Source update under
    # the 1-byte limit; the ordinary Source test separately proves early refusal.
    original_hpo_chain(directory,updates=1,numeric_budget_bytes=1,_unqualified_numeric_components=True)
    session=_ffi_open(candidate,directory)
    try:
        pending=json.loads(session.control_summary())
        assert pending['parent']['state']['effective_config']['hpo_spec']['operating']['numeric_budget_bytes']==1
        session.authorize_hpo(canonical(_authorization(directory,pending)).encode())
        before=session.used_bytes()
        with pytest.raises(MemoryError,match='incremental numeric'): session.export_recorded_cohort(0)
        assert session.used_bytes()<=before
        with pytest.raises(RuntimeError,match='closed'): session.recheck()
    finally: session.close()


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_actual_candidate_packed_hpo_originals_and_numeric_arrays(candidate,tmp_path,kind):
    import numpy as np

    from gf_common.preprocessing import preparation_config
    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.native_group_pack import OriginalArtifactProvider
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_support import original_hpo_chain
    from tests.test_card_instance_learning import ACTIONS
    directory=tmp_path/'parent'
    state,_,_=original_hpo_chain(directory,model_id=kind,decision_budget=17,packed=True)
    deadline=time.monotonic()+120
    with _ChainScan(directory=directory,max_owned_bytes=1024**3,deadline=deadline) as scan:
        validate_hpo_chain(directory,max_owned_bytes=1024**3,deadline=deadline,scan=scan)
        scan.configure_packed(state)
        owners=tuple(scan.packed_owner(directory/'collections'/f'batch-{index:06}') for index in range(3))
        with OriginalArtifactProvider(owners) as provider:
            plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(directory),'require_complete':False}],
                'collections':[],'storage_binding':state['storage_binding'],'artifact_registry_sha256':provider.registry_sha256}
            settings=preparation_config({**state['effective_config']['preprocessing'],'backend':'rust'})
            with _NativeGameInputs(plan,settings,actions=ACTIONS,max_owned_bytes=1024**3,deadline=deadline,native=candidate,
                    artifact_provider=provider,shared_controls=scan,extra_owned_bytes=scan.retained) as source:
                actual=source.compare_recorded(2,tmp_path/'regenerated')
                original=read_json(directory/'numeric/batch-000002/rollout.json')
                left=read_bundle(directory/'numeric/batch-000002',original['data'])
                right=read_bundle(tmp_path/'regenerated',actual['data'])
                assert set(left)==set(right)
                for name in left: np.testing.assert_array_equal(left[name],right[name],err_msg=name)


@pytest.mark.parametrize('fault',['journal','control','cancel'])
def test_actual_candidate_boundary_mutation_or_cancel_never_authorizes(candidate,tmp_path,fault):
    from gf_common.domain import canonical
    from tests.hpo_support import original_hpo_chain
    directory=tmp_path/'parent'
    original_hpo_chain(directory,updates=1)
    session=_ffi_open(candidate,directory)
    pending=json.loads(session.control_summary())
    request=_authorization(directory,pending)
    if fault=='cancel': session.cancel_handle().cancel()
    else:
        path=next((directory/'collections/batch-000000').glob('*/player-1.jsonl')) if fault=='journal' else directory/'run.json'
        path.write_bytes(path.read_bytes()+b'\n')
    try:
        with pytest.raises((ValueError,RuntimeError)):
            session.authorize_hpo(canonical(request).encode())
        with pytest.raises(RuntimeError): session.recheck()
    finally:
        try: session.close()
        except (ValueError,RuntimeError): pass
    assert session.used_bytes()==0


@pytest.mark.parametrize('packed',[False,True])
def test_actual_candidate_g0_has_no_synthetic_collection_or_numeric(candidate,tmp_path,packed):
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.native_group_pack import OriginalArtifactProvider
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_support import variable_initial_hpo_chain
    from tests.test_card_instance_learning import ACTIONS
    directory=tmp_path/'parent'
    state,_,_=variable_initial_hpo_chain(directory,games=8,packed=packed)
    deadline=time.monotonic()+90
    with _ChainScan(directory=directory,max_owned_bytes=1024**3,deadline=deadline) as scan:
        validate_hpo_chain(directory,max_owned_bytes=1024**3,deadline=deadline,scan=scan)
        scan.configure_packed(state)
        provider=OriginalArtifactProvider((),pool=scan.ensure_packed_pool(),shared_controls=scan) if packed else None
        try:
            plan={'schema_version':5,'mode':'historical_parent','parents':[{'directory':str(directory),'require_complete':False}],
                'collections':[]}
            if packed: plan.update(storage_binding=state['storage_binding'],artifact_registry_sha256=provider.registry_sha256)
            with _NativeGameInputs(plan,PreparationConfig(backend='rust',reader='participant-structured-v2',memory_budget_mib=1024),
                    actions=ACTIONS,max_owned_bytes=1024**3,deadline=deadline,native=candidate,
                    artifact_provider=provider,shared_controls=scan if packed else None,extra_owned_bytes=scan.retained if packed else 0) as owner:
                assert owner.summary['collections']==[] and owner._correspondence_done
            assert owner.cleanup_complete
        finally:
            if provider is not None: provider.close()
