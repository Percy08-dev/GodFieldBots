"""Installed native Collection7/Proof4 cohorts; explicitly unqualified CPU only."""
import pytest

from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_real_packed_resident_native_cohort_two_updates_all_models(global_root,tmp_path,monkeypatch,kind):
    from gf_training.native_backend import load_native
    from tests.test_resident_training import run_actual_source_case
    run_actual_source_case(global_root,tmp_path,monkeypatch,packed=True,kind=kind,
        native_backend=load_native(),games=16)


@pytest.mark.parametrize('resident',[False,True])
def test_real_native_packed_payload_projection_preserves_full_proof_and_rejects_hybrids(global_root,tmp_path,monkeypatch,resident):
    import copy
    import gc
    import json
    import time

    import numpy as np

    from gf_common.domain import ContractError, canonical, digest
    from gf_common.preprocessing import preparation_config
    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.native_backend import load_native
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.native_group_pack import OriginalArtifactProvider
    from gf_training.native_structured_session import _payload_views
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_resident_support import retirement_fact
    from tests.hpo_support import original_hpo_chain
    from tests.test_resident_episode import ACTIONS
    from tests.test_resident_training import run_actual_source_case
    if resident:
        directory=run_actual_source_case(global_root,tmp_path,monkeypatch,packed=True,
            native_backend=load_native(),games=16)
    else:
        directory=tmp_path/'parent'
        original_hpo_chain(directory,updates=1,decision_budget=8,packed=True)
    deadline=time.monotonic()+90
    with _ChainScan(directory=directory,max_owned_bytes=1024**3,deadline=deadline) as scan:
        state,_=validate_hpo_chain(directory,max_owned_bytes=1024**3,deadline=deadline,scan=scan)
        scan.configure_packed(state)
        owners=tuple(scan.packed_owner(directory/'collections'/f'batch-{index:06}') for index in range(state['updates']))
        with OriginalArtifactProvider(owners) as provider:
            plan={'schema_version':6 if resident else 5,'mode':'historical_parent',
                'parents':[{'directory':str(directory),'require_complete':False}],'collections':[],
                'storage_binding':state['storage_binding'],'artifact_registry_sha256':provider.registry_sha256}
            settings=preparation_config({**state['effective_config']['preprocessing'],'backend':'rust'})
            with _NativeGameInputs(plan,settings,actions=ACTIONS,max_owned_bytes=1024**3,deadline=deadline,
                    artifact_provider=provider,shared_controls=scan,extra_owned_bytes=scan.retained) as owner:
                packet=owner._native_call(owner._session.export_recorded_cohort,lambda held:(0,held))
                raw=json.loads(packet[0])
                before=canonical(raw['source_proof'])
                metadata,arrays=_payload_views(packet,game=True,recorded=True,packed=True,hpo=True)
                assert metadata['source_proof']['schema_version']==(4 if resident else 3)
                assert canonical(metadata['source_proof'])==before==canonical(owner.summary['collections'][0]['source_proof'])
                assert ('logical_retirement' in metadata['source_proof']) is resident
                assert arrays and all(not value.flags.writeable for value in arrays.values())
                del metadata,arrays
                mutations=['version2','version5','unknown','legacy-with-retirement']
                if resident: mutations+=['missing-retirement','dirty-reset','wrong-batch']
                for mutation in mutations:
                    changed=copy.deepcopy(raw)
                    proof=changed['source_proof']
                    if mutation=='version2': proof['schema_version']=2
                    elif mutation=='version5': proof['schema_version']=5
                    elif mutation=='unknown': proof['unexpected_authority']=True
                    elif mutation=='legacy-with-retirement':
                        proof.update(schema_version=3,logical_retirement=retirement_fact(state['run_id']))
                    elif mutation=='missing-retirement': proof.pop('logical_retirement')
                    elif mutation=='dirty-reset': proof['logical_retirement']['owners'][0]['end_ack']['clean_reset']=False
                    else:
                        retirement=proof['logical_retirement']
                        retirement['generation']+=1
                        retirement['collection_id']=f"batch-{retirement['generation']:06}"
                        for worker in retirement['owners']:
                            for key in ('end_ack','park_ack'):
                                worker[key]['generation']=retirement['generation']
                                worker[key+'_sha256']=digest(worker[key])
                    with pytest.raises(ContractError):
                        _payload_views((canonical(changed).encode(),*packet[1:]),game=True,recorded=True,packed=True,hpo=True)
                assert canonical(raw['source_proof'])==before
                del packet
                gc.collect()  # Retire actual exported Python views before native retirement.
                owner._native_call(owner._session.retire_payload)
                output=tmp_path/'assessment'
                manifest=owner.compare_recorded(0,output)
                original=read_json(directory/'numeric/batch-000000/rollout.json')
                left=read_bundle(directory/'numeric/batch-000000',original['data'])
                right=read_bundle(output,manifest['data'])
                assert left.keys()==right.keys()
                for key in left: np.testing.assert_array_equal(left[key],right[key])
                assert canonical(owner.summary['collections'][0]['source_proof'])==before
            assert owner.cleanup_complete
        assert provider._closed
        assert provider._executor_owner is not None and provider._executor_owner._closed
        assert provider._executor_owner._hold is None
