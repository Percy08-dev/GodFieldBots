"""Actual Wire7 lifecycle with inherited channel and real capture inputs."""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

from tests.test_hpo_numeric import learner_for
from tests.test_hpo_owned_numeric import budget, captured_record


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_actual_wire7_initialize_owned_update_and_reap(tmp_path, kind):
    from gf_common.domain import ContractError
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_structured_numeric_worker import assert_artifacts_equal
    learner = learner_for(kind, recipe=replace(HPOTrainingConfig(), epochs=1))
    account, ledger = budget(), ReservationLedger()
    facts=[]
    async def run():
        child = HpoOwnedLearningProcess(tmp_path, 1024**2, learner.model, learner.training, learner.binding,
            learner.action_config_hash, learner.resume_identity, actions=ACTIONS, ledger=ledger,
            cuda_visible_devices='', cleanup_deadline=time.monotonic()+60, budget=account,
            diagnostics=True,progress=facts.append)
        deadline = time.monotonic()+60
        await child.start()
        initial = updated = None
        try:
            initial = await child.initialize_owned(17, tmp_path/'g0', deadline=deadline)
            assert initial.generation == child.generation == child.steps == 0
            assert initial.metrics == {}
            initial.require_process(child)
            assert initial.actor_capability.require() is initial
            # Real capture, then bind its owner to the actual child snapshot for this test invocation.
            batch, owner, plan, local_snapshot, capture_budget, record = captured_record(learner)
            # A production converter is deliberately absent. Build the capture with original
            # child snapshot through the fixture's explicit construction seam.
            from tests.test_hpo_owned_wire import recapture
            selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
            foreign=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=local_snapshot,
                model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
                selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=capture_budget)
            foreign.persist(tmp_path/'foreign-numeric')
            with pytest.raises(ContractError,match='Source'):
                await child.update_owned(foreign,tmp_path/'wrong-source-output',deadline=deadline)
            assert not ledger.updating and foreign._state=='PERSISTED'
            foreign.close()
            foreign.require_retired()
            batch.close()
            batch, owner, plan = recapture(record, initial, account)
            selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=3, used_before=0)
            numeric = encode_owned_batch(batch, owner=owner, plan=plan, snapshot_owner=initial,
                model=learner.model, training=learner.training, actions=ACTIONS, rules=RULES,
                selected_locations=((0,0),(0,1),(0,2)), selection=selection, budget=account)
            numeric.persist(tmp_path/'numeric')
            with pytest.raises(ContractError,match='audited'):
                await child.update_owned(numeric,tmp_path/'wrong-resident-authority',deadline=deadline,resident_authority=object())
            assert numeric._state=='PERSISTED' and not ledger.updating
            updated = await child.update_owned(numeric, tmp_path/'g1', deadline=deadline)
            assert updated.generation == child.generation == 1
            assert updated.metrics['parity'] == {'status':'not_run'}
            assert updated.actor_capability.require() is updated
            assert not child.channel.outstanding
            with pytest.raises(ContractError):
                await child.update_owned(numeric, tmp_path/'retry', deadline=deadline)
            from gf_training.hpo_rollout import HpoRollout
            reference = HpoRollout((record,), learner.model, learner.training, actions=ACTIONS, rules=RULES,
                selected_locations=((0,0),(0,1),(0,2)))
            reference_metrics = learner.update(reference)
            assert updated.metrics['diagnostics'] == pytest.approx(reference_metrics['diagnostics'])
            learner.save(tmp_path/'direct')
            assert_artifacts_equal(tmp_path/'direct', tmp_path/'g1')
            numeric.close()
            batch.close()
            local_snapshot.close()
        finally:
            await child.close(deadline)
            if initial is not None: initial.close()
            if updated is not None: updated.close()
        assert child.process.returncode == 0 and child.lease.released and ledger.learner is None
    asyncio.run(run())
    actual=[row for row in facts if row.get('diagnostic_boundary')=='update-synchronized']
    assert len(actual)==1 and actual[0]['diagnostic_update']==1
    assert actual[0]['update_execution']==('microbatch' if kind=='mlp' else 'shared-prefix')


def recapture(record, snapshot, account):
    import uuid
    from pathlib import Path

    from gf_common.domain import EpisodeGroup, EpisodeResult, EpisodeSpec, GroupResult
    from gf_common.structured import EventBank, event_delta
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_owned_capture import _new_capture
    from tests.test_hpo_owned_capture import terminal
    owner = object()
    group = EpisodeGroup('seed', (EpisodeSpec('first', ('same','same'), (1,2), 'pair',1),))
    plan = CollectionPlan((group,), {}, Path('unused'), {}, float('inf'),float('inf'))
    capture = _new_capture(owner,run_id='run',plan=plan,generation=snapshot.generation,
        snapshot_owner=snapshot,learner_seats={'first':1},learner_policy_version=record.selections[0].policy_version,budget=account)
    capture.open_episode('first','game')
    for viewer in (1,2):
        bank=EventBank(viewer,record.bank.episode if viewer==1 else 'actual-opponent-bank',layout=record.bank.layout)
        capture.participant_created('first','game',viewer,bank.view())
    for index,(frame,table,selection) in enumerate(zip(record.frames,record.tables,record.selections,strict=True)):
        delta = event_delta(1,frame.episode,frame.interval_start,frame.interval_start_hash,
            record.bank.events(frame.interval_start,frame.boundary.end),layout=frame.layout)
        command={'command_id':str(uuid.uuid4()),'expected_revision':frame.revision,'action':selection.validate(table,frame)}
        capture.begin_decision('first',1,frame,delta,table,selection,command)
        last=index+1==len(record.frames)
        capture.accept_decision('first',1,command['command_id'],{'command_id':command['command_id'],
            'previous_revision':frame.revision,'state_revision':frame.revision+1 if last else record.frames[index+1].revision,
            'status':'terminal' if last else 'active','result':record.outcome if last else None})
    for viewer in (1,2):
        obs=terminal(viewer,record.frames[-1].revision+1)
        obs['result']=record.outcome
        capture.terminal_observation('first',viewer,obs)
    result=EpisodeResult('first','pair','seed',('same','same'),game_id='game',outcome=record.outcome,end_kind='natural',decisions=len(record.frames))
    capture.close_episode('first',result)
    capture.close_group(group,GroupResult([result]))
    return capture.seal(owner,plan=plan,snapshot_owner=snapshot),owner,plan


@pytest.mark.parametrize('field', ['ready_bool_generation','final_bool_steps','final_identity','final_file_digest'])
def test_wire7_wrong_actual_save_ack_cannot_mint_snapshot_and_reaps(tmp_path,monkeypatch,field):
    from gf_common.domain import ContractError
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.ppo_run import LearningProcess
    from tests.test_card_instance_learning import ACTIONS
    learner=learner_for()
    account,ledger=budget(),ReservationLedger()
    real_request=LearningProcess.request
    async def bad_ready(self,request,deadline):
        reply=await real_request(self,request,deadline)
        if field=='ready_bool_generation' and reply.get('kind')=='SNAPSHOT_READY': reply['generation']=False
        return reply
    monkeypatch.setattr(LearningProcess,'request',bad_ready)
    real_read=HpoOwnedLearningProcess._read_reply
    async def bad_final(self,deadline):
        reply=await real_read(self,deadline)
        if reply.get('kind')=='SAVED':
            if field=='final_bool_steps': reply['optimizer_steps']=False
            if field=='final_identity': reply['snapshot_identity']='0'*64
            if field=='final_file_digest': reply['files_sha256']='0'*64
        return reply
    monkeypatch.setattr(HpoOwnedLearningProcess,'_read_reply',bad_final)
    async def run():
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+20,budget=account)
        await child.start()
        try:
            with pytest.raises(ContractError):
                await child.initialize_owned(17,tmp_path/'g0',deadline=time.monotonic()+10)
            assert child.current_snapshot is None and child.steps is None and child._owned_failed
            with pytest.raises(ContractError):
                await child.initialize_owned(17,tmp_path/'retry',deadline=time.monotonic()+10)
        finally:
            try: await child.close(time.monotonic()+1)
            except (ContractError,RuntimeError,TimeoutError,BrokenPipeError): pass
        assert child.process.returncode is not None and child.lease.released and ledger.learner is None
        assert account.used==0
    asyncio.run(run())


@pytest.mark.parametrize('boundary',['before_fd','after_snapshot_ack'])
def test_wire7_cancel_retains_original_batch_until_actual_reap(tmp_path,monkeypatch,boundary):
    from gf_common.domain import ContractError
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training import hpo_owned_process as module
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    learner=learner_for(recipe=replace(HPOTrainingConfig(),epochs=1))
    account,ledger=budget(),ReservationLedger()
    async def run():
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+30,budget=account)
        await child.start()
        initial=await child.initialize_owned(17,tmp_path/'g0',deadline=time.monotonic()+15)
        local,owner,plan,local_snapshot,_capture_budget,record=captured_record(learner)
        local.close()
        batch,owner,plan=recapture(record,initial,account)
        selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
        numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=initial,
            model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
            selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
        numeric.persist(tmp_path/'numeric')
        entered=asyncio.Event()
        never=asyncio.Event()
        original_ready=module._ready
        async def pause_ready(channel,*,write=False,deadline):
            if boundary=='before_fd' and write and child._inflight_batch is numeric:
                entered.set()
                await never.wait()
            await original_ready(channel,write=write,deadline=deadline)
        monkeypatch.setattr(module,'_ready',pause_ready)
        original_read=child._read_reply
        async def pause_final(deadline):
            if boundary=='after_snapshot_ack' and child._partial_banks:
                entered.set()
                await never.wait()
            return await original_read(deadline)
        monkeypatch.setattr(child,'_read_reply',pause_final)
        task=asyncio.create_task(child.update_owned(numeric,tmp_path/'g1',deadline=time.monotonic()+15))
        await asyncio.wait_for(entered.wait(),15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert child._owned_failed and child.current_snapshot is initial
        assert numeric._state=='DISPATCHED' and child.process.returncode is None
        with pytest.raises(ContractError,match='completion'): numeric.close()
        try: await child.close(time.monotonic()+1)
        except (ContractError,RuntimeError,TimeoutError,BrokenPipeError): pass
        assert child.process.returncode is not None and child.lease.released
        assert numeric._state=='FAILED' and not child.channel.outstanding
        numeric.close()
        batch.close()
        initial.close()
        local_snapshot.close()
        # Original frame/table/Selection aliases deliberately retain capture
        # admission after container retirement, in this SAME Source account.
        del record
        import gc
        gc.collect()
        assert account.used==0
    asyncio.run(run())


def test_wire7_owned_metrics_decoder_preserves_not_run_and_finite_guards():
    from types import SimpleNamespace

    from gf_common.domain import ContractError
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_process import _owned_metrics
    recipe=HPOTrainingConfig()
    batch=SimpleNamespace(collected_decisions=3,used_decisions=2,locations=((0,0),(0,2)))
    fields={'policy_loss':0.1,'value_mse':0.2,'entropy':0.3,'approx_kl':0.0,'total_loss':0.4,
        'optimizer_steps':4,'optimizer_step_delta':4,'collected_decisions':3,'used_decisions':2,
        'parity':{'status':'not_run'},'tbptt_length':32,'epochs':4,'endpoint_evaluations':8,
        'context_decisions':16,'suffix_decisions':16,'update_seconds':0.1}
    with pytest.raises(ContractError, match='diagnostics'):
        _owned_metrics(fields,batch=batch,training=recipe,expected_steps=4)
    _owned_metrics(fields,batch=batch,training=recipe,expected_steps=4,historical=True)
    fields['diagnostics'] = {'schema_version':1,'clip_fraction':.5,'explained_variance':None,
        'pre_clip_grad_norm_mean':.2,'pre_clip_grad_norm_max':.3,'learning_rate':.01}
    _owned_metrics(fields,batch=batch,training=recipe,expected_steps=4)
    broken = dict(fields, diagnostics=dict(fields['diagnostics'], clip_fraction=True))
    with pytest.raises(ContractError, match='diagnostics'):
        _owned_metrics(broken,batch=batch,training=recipe,expected_steps=4,historical=True)
    for key,value in [('total_loss',float('nan')),('optimizer_steps',True),('parity',{'max_error':0}),('context_decisions',0)]:
        invalid={**fields,key:value}
        with pytest.raises(ContractError):
            _owned_metrics(invalid,batch=batch,training=recipe,expected_steps=4)


def test_numeric6_partial_writer_cannot_dispatch_update(tmp_path,monkeypatch):
    from gf_common.domain import ContractError
    from gf_models import sharded_arrays
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    real=sharded_arrays.save_arrays
    calls=[]
    def partial(path,arrays,**kw):
        calls.append(path)
        if len(calls)==2: raise OSError('partial numeric writer')
        return real(path,arrays,**kw)
    monkeypatch.setattr(sharded_arrays,'save_arrays',partial)
    with pytest.raises(OSError,match='partial numeric'):
        numeric.persist(tmp_path/'partial',shard_bytes=16384)
    assert calls[0].exists() and not (tmp_path/'partial'/'rollout.json').exists()
    with pytest.raises(ContractError): learner.update_owned(numeric)
    assert learner.steps==0 and not learner.optimizer.state
    numeric.close()
    numeric.require_retired()
    batch.close()
    snapshot.close()


def test_public_bundle_writer_still_runs_original_strict_inspection(tmp_path,monkeypatch):
    import numpy as np

    from gf_common.domain import ContractError
    from gf_models import sharded_arrays
    def strict(*args,**kw): raise ContractError('original strict inspection')
    monkeypatch.setattr(sharded_arrays,'inspect_bundle',strict)
    with pytest.raises(ContractError,match='original strict inspection'):
        sharded_arrays.write_bundle(tmp_path,'weights',{'weight':np.ones(4,np.float32)})


def test_owned_worker_save_phase_preflight_before_process_or_network_allocation(tmp_path):
    from gf_common.domain import ContractError
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    learner=learner_for()
    ledger=ReservationLedger()
    account=budget()
    with pytest.raises(ContractError,match='save'):
        HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+30,budget=account,
            phase_limits={'numeric_budget_bytes':512*1024**2,'save_budget_bytes':1})
    assert ledger.learner is None and account.used==0
