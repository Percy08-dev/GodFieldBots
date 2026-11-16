"""Cold checkpoint ownership and original staged state application."""
from dataclasses import replace

import numpy as np
import pytest

from gf_common.domain import ContractError


@pytest.mark.parametrize('kind', ['mlp','gru','transformer'])
def test_owned_restore_stages_original_arrays_without_legacy_restore(tmp_path, monkeypatch, kind):
    import torch

    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_artifacts import _restore_owned_learner
    from gf_training.hpo_config import HPOTrainingConfig
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    original=learner_for(kind,recipe=replace(HPOTrainingConfig(),epochs=1))
    account=budget()
    snapshot=original.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'saved')
    target=learner_for(kind,recipe=original.training)
    def forbidden(*args,**kwargs): raise AssertionError('legacy restore/readback')
    monkeypatch.setattr('gf_training.hpo_artifacts.restore_learner',forbidden)
    monkeypatch.setattr('gf_models.structured_artifacts._read_resume',forbidden)
    before=torch.get_rng_state().clone()
    manifest=read_json(tmp_path/'saved/resume.json')
    bad={**manifest,'shuffle_rng':{**manifest['shuffle_rng'],'bit_generator':'wrong'}}
    with pytest.raises(ContractError):
        _restore_owned_learner(target,inference=snapshot.inference,resume=snapshot.resume,
            actor_metadata=snapshot.actor_metadata,resume_manifest=bad,generation=0,budget=account,actions=ACTIONS)
    assert torch.equal(before,torch.get_rng_state())
    _restore_owned_learner(target,inference=snapshot.inference,resume=snapshot.resume,
        actor_metadata=snapshot.actor_metadata,resume_manifest=manifest,generation=0,budget=account,actions=ACTIONS)
    assert target.steps==0 and target._owned_generation==0
    assert target._owned_legacy is False
    for name,value in original.network.state_dict().items():
        torch.testing.assert_close(value,target.network.state_dict()[name],rtol=0,atol=0)
    np.testing.assert_array_equal(torch.get_rng_state().numpy(),snapshot.resume.array('rng.torch_cpu'))
    assert target.shuffle.bit_generator.state==original.shuffle.bit_generator.state
    snapshot.close()


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_pause_resume_original_generation_no_history_replay(tmp_path,monkeypatch,kind):
    import asyncio
    import importlib
    import json

    from gf_models.sharded_arrays import read_bundle
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch,kind)
    directory=tmp_path/'parents'/run_id
    async def run():
        paused,ops=await run_owned(runtime,directory,pause_after_update=1)
        assert paused['status']=='paused' and paused['updates']==1
        before={path:path.read_bytes() for path in (directory/'generations').iterdir()}
        targets=('gf_training.hpo_lineage.validate_hpo_chain','gf_training.hpo_source.validate_hpo_parent',
            'gf_training.card_instance_source._open_batch','gf_training.hpo_rollout.read_hpo_rollout')
        for name in targets: importlib.import_module(name.rsplit('.',1)[0])
        def forbidden(*args,**kwargs): raise AssertionError('historical sample replay during cold restore')
        with monkeypatch.context() as patch:
            for name in targets: patch.setattr(name,forbidden)
            resumed,new_ops=await run_owned(runtime,directory,mode='resume')
        assert resumed['status']=='complete' and resumed['updates']==2 and resumed['used_decisions']==32
        assert new_ops.owned_registry.process.current_snapshot.generation==2
        assert all(path.read_bytes()==value for path,value in before.items())
        assert ops.budget.used==0
        if new_ops.completion is not None: new_ops.completion.close()
        assert new_ops.budget.used==0
        manifest=json.loads((directory/'artifacts/000002/resume.json').read_text())
        state=read_bundle(directory/'artifacts/000002',manifest['state'])
        assert any(name.startswith('adam.') for name in state)
    asyncio.run(run())


def test_external_numeric6_reader_checks_owned_origin_and_saved_arrays(tmp_path):
    from gf_models.arrays import sha256
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_rollout import read_hpo_rollout
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import captured_record
    learner=learner_for()
    batch,owner,plan,snapshot,account,record=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,model=learner.model,
        training=learner.training,actions=ACTIONS,rules=RULES,selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    numeric.persist(tmp_path/'numeric')
    options={'actions':ACTIONS,'rules':RULES,'expected_selection':selection,'expected_records':(record,)}
    with pytest.raises(ContractError):
        read_hpo_rollout(tmp_path/'numeric',sha256(tmp_path/'numeric/rollout.json'),learner.model,learner.training,**options)
    restored=read_hpo_rollout(tmp_path/'numeric',sha256(tmp_path/'numeric/rollout.json'),learner.model,learner.training,
        expected_snapshot_identity=snapshot.identity,**options)
    assert len(restored)==3
    for index in range(3): assert restored.vector(index)==numeric.vector(index)
    numeric.close(); batch.close(); snapshot.close()


def test_owned_external_metadata_import_does_not_replay_samples(tmp_path,monkeypatch):
    import asyncio

    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_campaign,_settings=owned_runtime(tmp_path,monkeypatch)
    async def run():
        _report,ops=await run_owned(runtime,tmp_path/'parents'/run_id)
        ops.completion.close()
        from gf_training.hpo_owned_restore import import_owned_final_policy
        imported=import_owned_final_policy(tmp_path/'parents'/run_id,deadline=__import__('time').monotonic()+30,
            max_owned_bytes=runtime.config.hpo_spec['operating']['source_budget_bytes'],parent_budget_bytes=runtime.config.hpo_spec['operating']['parent_budget_bytes'])
        assert imported.state['status']=='complete'
        assert imported.actor_capability.require() is imported
        assert not hasattr(imported,'resume')
        assert imported.records[-1]['generation']==2
        imported.close()
    asyncio.run(run())


def test_owned_external_evaluation_pins_preserve_metadata_route(tmp_path,monkeypatch):
    import asyncio
    import time

    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        _,ops=await run_owned(runtime,directory)
        sha=ops.completion.files['inference.json']; ops.completion.close()
        def forbidden(*args,**kwargs): raise AssertionError('full chain/sample route during evaluation pin')
        monkeypatch.setattr('gf_training.hpo_retention.validate_hpo_chain',forbidden)
        pin=pin_hpo_evaluation(directory,'evaluation-new-job',2,sha,max_owned_bytes=2*1024**3,deadline=time.monotonic()+30)
        assert pin.phase=='acquire' and pin.binding['generation']==2
        released=release_hpo_evaluation(directory,pin.binding,max_owned_bytes=2*1024**3,deadline=time.monotonic()+30)
        assert released.phase=='release'
    asyncio.run(run())


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_cold_resume_matches_continuous_final_adam_rng_arrays(tmp_path,monkeypatch,kind):
    import asyncio
    import json

    from gf_models.sharded_arrays import read_bundle
    from tests.hpo_owned_support import owned_runtime, run_owned
    async def one(root,pause):
        root.mkdir()
        runtime,run_id,_,_=owned_runtime(root,monkeypatch,kind)
        directory=root/'parents'/run_id
        first,ops=await run_owned(runtime,directory,pause_after_update=1 if pause else None)
        if pause:
            assert first['status']=='paused'
            _,ops=await run_owned(runtime,directory,mode='resume')
        snapshot=ops.completion
        assert snapshot is not None
        snapshot.close()
        inference=json.loads((directory/'artifacts/000002/inference.json').read_text())
        resume=json.loads((directory/'artifacts/000002/resume.json').read_text())
        arrays={**read_bundle(directory/'artifacts/000002',inference['weights']),**read_bundle(directory/'artifacts/000002',resume['state'])}
        return arrays,resume['shuffle_rng']
    async def run():
        continuous,continuous_shuffle=await one(tmp_path/'continuous',False)
        resumed,resumed_shuffle=await one(tmp_path/'resumed',True)
        assert continuous.keys()==resumed.keys()
        for name in continuous: np.testing.assert_array_equal(continuous[name],resumed[name],err_msg=name)
        assert continuous_shuffle==resumed_shuffle
    asyncio.run(run())


@pytest.mark.parametrize('case',['identity','expired','latest','not-paused','collection-extra','cache-extra'])
def test_cold_invalid_parent_refuses_before_new_process_and_refunds(tmp_path,monkeypatch,case):
    import asyncio
    import json
    from dataclasses import replace

    from gf_models.arrays import atomic_json
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    observed=[]
    original=_OwnedHpoOperations.read_cold
    def read(self,*args,**kwargs):
        observed.append(self)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(_OwnedHpoOperations,'read_cold',read)
    async def run():
        _,first=await run_owned(runtime,directory,pause_after_update=1)
        parent=json.loads((directory/'run.json').read_text())
        invocation=runtime
        if case=='identity': invocation=replace(runtime,inputs=(*runtime.inputs,{'changed':'configuration-input'}))
        elif case=='expired':
            for key in ('created_unix','last_unix','wall_deadline_unix','admission_deadline_unix','execution_deadline_unix'): parent[key]-=100000
            atomic_json(directory/'run.json',parent)
        elif case=='latest': atomic_json(directory/'latest.json',{'generation':1,'sha256':'a'*64})
        elif case=='collection-extra': (directory/'collections/foreign').mkdir()
        elif case=='cache-extra': (directory/'cache-references/foreign').write_text('foreign')
        else:
            parent.update(status='incomplete',stop_reason='time_cap')
            atomic_json(directory/'run.json',parent)
        with pytest.raises((ValueError,TimeoutError,RuntimeError)):
            await run_owned(invocation,directory,mode='resume')
        assert observed and observed[-1]._startup_child is None
        assert observed[-1].budget.used==0 and observed[-1].writer.parent_budget.used==0
        assert not (directory/'generations/000002.json').exists()
        assert first.budget.used==0
    asyncio.run(run())


def test_wrong_restore_ack_poison_reaps_and_never_recommits(tmp_path,monkeypatch):
    import asyncio
    import json

    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    seen=[]
    read=HpoOwnedLearningProcess._read_reply
    make=_OwnedHpoOperations.make_child
    def child(self,*args,**kwargs):
        result=make(self,*args,**kwargs); seen.append(self); return result
    monkeypatch.setattr(_OwnedHpoOperations,'make_child',child)
    async def corrupt(self,*args,**kwargs):
        reply=await read(self,*args,**kwargs)
        if reply.get('kind')=='RESTORED': return {**reply,'generation':reply['generation']+1}
        return reply
    async def run():
        await run_owned(runtime,directory,pause_after_update=1)
        original=(directory/'generations/000001.json').read_bytes()
        monkeypatch.setattr(HpoOwnedLearningProcess,'_read_reply',corrupt)
        with pytest.raises(ContractError): await run_owned(runtime,directory,mode='resume')
        ops=seen[-1]
        assert ops.owned_registry.process._owned_failed
        assert ops.owned_registry.process.process.returncode is not None
        assert ops.budget.used==0
        assert (directory/'generations/000001.json').read_bytes()==original
        assert not (directory/'generations/000002.json').exists()
        assert json.loads((directory/'run.json').read_text())['status']=='failed'
    asyncio.run(run())


def test_cold_decodes_each_control_and_only_required_checkpoint_arrays_once(tmp_path,monkeypatch):
    import asyncio
    from collections import Counter

    from gf_models import arrays
    from gf_models.structured_sealed import SealedScan
    from gf_training.hpo_owned_restore import _ColdScan
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    decoded=Counter(); loaded=Counter()
    object_reader=SealedScan.object; array_reader=arrays.load_arrays
    def objects(self,path,*args,**kwargs):
        if type(self) is _ColdScan:
            if path.is_relative_to(directory): key=path.relative_to(directory).as_posix()
            else:
                archive=__import__('pathlib').Path(runtime.config.storage.db_root)/run_id/'collections'
                assert path.is_relative_to(archive)
                key='archive/'+path.relative_to(archive).as_posix()
            decoded[key]+=1
        return object_reader(self,path,*args,**kwargs)
    def load(path,*args,**kwargs):
        loaded[path.relative_to(directory).as_posix()]+=1
        return array_reader(path,*args,**kwargs)
    async def run():
        await run_owned(runtime,directory,pause_after_update=1)
        monkeypatch.setattr(SealedScan,'object',objects)
        monkeypatch.setattr(arrays,'load_arrays',load)
        _,ops=await run_owned(runtime,directory,mode='resume')
        ops.completion.close()
        assert all(count==1 for count in decoded.values())
        assert decoded['generations/000000.json']==decoded['generations/000001.json']==decoded['updates/000001.json']==1
        assert loaded=={'artifacts/000000/weights-000000.npz':1,'artifacts/000001/weights-000000.npz':1,'artifacts/000001/resume-000000.npz':1}
        assert not any(name.startswith('numeric/') for name in decoded)
        assert all(name.endswith('/collection.locator.json') for name in decoded if name.startswith('collections/'))
        assert all(name.endswith(('/collection.published.json','/collection.pending.json')) for name in decoded if name.startswith('archive/'))
        assert ops.budget.used==0 and ops.writer.parent_budget.used==0
    asyncio.run(run())


def test_rejected_resume_confirms_only_actual_prechild_operational_cleanup(tmp_path,monkeypatch):
    import asyncio
    from dataclasses import replace

    from gf_training import hpo_gpu_lease as gpu
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    async def run():
        await run_owned(runtime,directory,pause_after_update=1)
        original=(directory/'run.json').read_bytes()
        changed=replace(runtime,inputs=(*runtime.inputs,{'changed':'input'}))
        with pytest.raises(ContractError) as rejected:
            await run_owned(changed,directory,mode='resume')
        receipt=getattr(rejected.value,'hpo_gpu_termination',None)
        assert type(receipt) is gpu.TerminationReceipt
        assert receipt.resources==()
        admission=gpu._read(tmp_path/'global/jobs'/receipt.ticket/'admission.json')
        assert gpu._termination(gpu._read(tmp_path/'global/jobs'/receipt.ticket/'termination.json'),admission)==receipt
        assert (directory/'run.json').read_bytes()==original
        assert not hasattr(rejected.value,'completion')
        assert not (directory/'generations/000002.json').exists()
    asyncio.run(run())


@pytest.mark.parametrize('revision',[False,123,'bad','a'*41])
def test_owned_restore_rejects_malformed_original_producer_revision(tmp_path,revision):
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_artifacts import _restore_owned_learner
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner=learner_for(); account=budget(); snapshot=learner.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'saved')
    manifest=read_json(tmp_path/'saved/resume.json')
    manifest['producer']['code_revision']=revision
    target=learner_for()
    with pytest.raises(ContractError):
        _restore_owned_learner(target,inference=snapshot.inference,resume=snapshot.resume,
            actor_metadata=snapshot.actor_metadata,resume_manifest=manifest,generation=0,budget=account,actions=ACTIONS)
    snapshot.close()
