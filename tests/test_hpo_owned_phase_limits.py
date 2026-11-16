"""Configured save phase is distinct from each immutable bank/numeric bound."""
import asyncio
import time

import pytest

from gf_common.domain import ContractError

MIB=1024**2


def test_owned_export_honors_original_one_gib_save_allowance(tmp_path):
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner=learner_for(); account=budget()
    snapshot=learner.snapshot_owned(budget=account,save_budget_bytes=1024*MIB)
    try:
        snapshot.save(tmp_path/'saved')
        snapshot.save_ack.require(snapshot)
        assert snapshot._save_budget_limit==1024*MIB
        assert all(bank.size<=512*MIB for bank in (*snapshot.inference.banks,*snapshot.resume.banks))
    finally: snapshot.close()
    assert account.used==0


def test_actual_wire7_honors_one_gib_save_and_keeps_numeric_bank_caps(tmp_path):
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner=learner_for(); account=budget(); ledger=ReservationLedger()
    async def run():
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,cuda_visible_devices='',
            cleanup_deadline=time.monotonic()+30,budget=account,
            phase_limits={'numeric_budget_bytes':512*MIB,'save_budget_bytes':1024*MIB})
        snapshot=None
        try:
            await child.start()
            snapshot=await child.initialize_owned(17,tmp_path/'saved',deadline=time.monotonic()+20)
            assert snapshot.generation==0 and snapshot.optimizer_steps==0
            snapshot.save_ack.require(snapshot)
        finally:
            await child.close(time.monotonic()+5)
            if snapshot is not None: snapshot.close()
        assert child.process.returncode==0 and child.lease.released and account.used==0
    asyncio.run(run())


@pytest.mark.parametrize('used,bookkeeping,expected',[(0,0,512),(512,0,512),(768,8,248),(1015,8,1)])
def test_save_receive_uses_remaining_aggregate_before_individual_bank_cap(used,bookkeeping,expected):
    from gf_training.hpo_owned_worker import _remaining_snapshot_bytes
    assert _remaining_snapshot_bytes(1024*MIB,used*MIB,bookkeeping=bookkeeping*MIB)==expected*MIB


def test_oversize_bank_descriptor_remains_refused_without_allocating_payload():
    from gf_common.ram_bank import _layout
    from gf_training.hpo_owned_worker import _remaining_snapshot_bytes
    size=512*MIB+1
    descriptor={'name':'too-large','dtype':'|u1','shape':[size],'offset':0,'nbytes':size}
    with pytest.raises(ContractError):
        _layout([descriptor],size,_remaining_snapshot_bytes(1024*MIB,0,bookkeeping=0))


def test_exhausted_receive_phase_refuses_before_channel_or_metadata_allocation():
    from gf_training.hpo_owned_worker import _receive_control, _remaining_snapshot_bytes
    from tests.test_hpo_owned_numeric import budget
    with pytest.raises(ContractError): _remaining_snapshot_bytes(1024*MIB,1016*MIB,bookkeeping=8*MIB)
    class NoReceive:
        def receive(self,*args,**kwargs): raise AssertionError('physical receive before aggregate admission')
    account=budget()
    with pytest.raises(ContractError):
        _receive_control(NoReceive(),purpose='save',generation=0,identity='never',budget=account,max_admitted_bytes=1)
    assert account.used==0


def test_actual_owned_pause_resume_with_original_one_gib_save_phase(tmp_path,monkeypatch):
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,_,_=owned_runtime(tmp_path,monkeypatch,save_budget_bytes=1024*MIB)
    directory=tmp_path/'parents'/run_id
    async def run():
        paused,old=await run_owned(runtime,directory,pause_after_update=1)
        assert paused['status']=='paused' and old.budget.used==0
        report,resumed=await run_owned(runtime,directory,mode='resume')
        assert report['status']=='complete' and report['updates']==2 and report['used_decisions']==32
        resumed.completion.close()
        assert resumed.budget.used==0 and resumed.writer.parent_budget.used==0
    asyncio.run(run())


def test_numeric_phase_still_refuses_more_than_one_bank_cap_before_allocation(tmp_path):
    import os

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner=learner_for(); account=budget(); ledger=ReservationLedger()
    baseline=len(os.listdir('/proc/self/fd'))
    with pytest.raises(ContractError):
        HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,cuda_visible_devices='',
            cleanup_deadline=time.monotonic()+30,budget=account,
            phase_limits={'numeric_budget_bytes':1024*MIB,'save_budget_bytes':1024*MIB})
    assert len(os.listdir('/proc/self/fd'))==baseline and account.used==0 and ledger.learner is None
