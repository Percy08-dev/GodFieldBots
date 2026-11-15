"""Bounded hand-authored resident originals; never executed-game/live authority.

Generate explicit native component fixtures with:
    uv run python -m tests.hpo_resident_support /tmp/NEW-DIRECTORY
"""
import sys
import time
from pathlib import Path


def retirement_fact(run_id='run-a'):
    from gf_common.domain import digest
    owner = {'job_ticket':'a'*32,'pool_id':'b'*32,'pool_epoch':1,
        'worker_key':1,'pid':123,'viewer':1,'lease_epoch':1}
    base = {**owner,'schema_version':1,'protocol':'resident-worker-v1','run_id':run_id,
        'generation':0,'checkpoint_sha256':'c'*64,'cache_identity':None,
        'clean_reset':True,'private_state_empty':True,'pending_io_empty':True,
        'cuda_synchronized':True,'reset_version':'resident-reset-v1'}
    end = {**base,'command':'END','request_epoch':1}
    park = {**base,'command':'PARK','request_epoch':2}
    return {'schema_version':1,'kind':'hpo-resident-logical-retirement','run_id':run_id,
        'collection_id':'batch-000000','job_ticket':'a'*32,'pool_id':'b'*32,'pool_epoch':1,'generation':0,
        'owners':[{**owner,'end_ack':end,'end_ack_sha256':digest(end),'park_ack':park,'park_ack_sha256':digest(park)}]}

def main():
    from gf_models.arrays import atomic_json
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_retention import pin_hpo_cache
    from gf_training.hpo_source import recorded_hpo_plan
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.hpo_support import original_hpo_collection
    root=Path(sys.argv[1]).resolve()
    root.mkdir()
    owner={'job_ticket':'a'*32,'pool_id':'b'*32,'pool_epoch':1,'worker_key':1,'pid':123,'viewer':1,'lease_epoch':1}
    for name in ('absent','acquired','pending','released','live'):
        folder=root/name
        state,records,learner=initial_hpo_chain(folder,resident=True)
        if name in ('acquired','pending','released'):
            pin=pin_hpo_cache(folder,'cache-1',0,records[0]['inference_sha256'],owner=owner,max_owned_bytes=1024**3,deadline=time.monotonic()+30)
            if name=='pending':
                (folder/'cache-reference-receipts/cache-1/acquire.receipt.json').unlink()
            elif name=='released':
                # Pure persisted diagnostic fixture: no new live release authority is issued.
                import uuid

                from gf_common.domain import digest
                from gf_models.sharded_arrays import read_json
                token=folder/'cache-reference-receipts/cache-1'
                registry=read_json(folder/'artifact-registry.json')
                registry['cache_references']={}
                from gf_models.arrays import sha256
                atomic_json(token/'release.intent.json',{'schema_version':1,'kind':'hpo-cache-reference-release-intent',
                    'operation_id':str(uuid.uuid4()),'binding':pin.binding,'acquire_receipt_sha256':sha256(token/'acquire.receipt.json'),
                    'release_evidence':{'schema_version':1,'kind':'hpo-resident-cache-reaped','binding_sha256':digest(pin.binding),
                        'resource':{'kind':'worker','key':1,'pid':123,'outcome':'reaped'}}})
                atomic_json(folder/'artifact-registry.json',registry)
                atomic_json(token/'release.receipt.json',{'schema_version':1,'kind':'hpo-cache-reference-release-receipt','binding':pin.binding,
                    'intent_sha256':sha256(token/'release.intent.json'),'registry_sha256':sha256(folder/'artifact-registry.json'),'confirmed_unix':time.time()})
        if name=='live':
            plan,_,_=recorded_hpo_plan(folder,state,records,0,time.monotonic()+60)
            original_hpo_collection(plan,state,learner,logical_retirement=retirement_fact(state['run_id']))
    atomic_json(root/'selection.json',selection_for_counts((1,)*8,learning_seed=17,batch=0,decision_budget=6,used_before=0))
    print(root)


if __name__ == "__main__":
    main()
