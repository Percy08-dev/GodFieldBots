"""Every actual received FD has a cleanup owner before fallible processing."""
from __future__ import annotations

import asyncio
import errno
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.ram_bank import RamBankBuilder, RamBankChannel, ReceivedBank
from gf_training.hpo_owned_worker import _receive_control
from tests.test_hpo_numeric import learner_for
from tests.test_hpo_owned_numeric import budget


def fd_count():
    return len(tuple(Path('/proc/self/fd').iterdir()))


@pytest.mark.parametrize('fault',['layout','admission','json','ack'])
def test_control_receive_failure_closes_original_without_ack_or_charge_leak(monkeypatch,fault):
    account=budget()
    raw=b'{' if fault=='json' else b'{}'
    bank=RamBankBuilder({'bad' if fault=='layout' else 'control':np.frombuffer(raw,dtype=np.uint8)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    original_charge=account.used
    original_fds=fd_count()
    lease=parent.send(bank,purpose='save',generation=0,identity='control')
    class Admission:
        calls=0
        def reserve(self,size,*,wait=False):
            self.calls+=1
            if fault=='admission' and self.calls==2: raise ContractError('control decode admission')
            return account.reserve(size,wait=wait)
    if fault=='ack':
        def fail_send(*args,**kw): raise OSError(errno.EIO,'completion send uncertainty')
        monkeypatch.setattr(receiver,'_send',fail_send)
    expected=(OSError if fault=='ack' else ValueError if fault=='json' else ContractError)
    try:
        with pytest.raises(expected):
            _receive_control(receiver,purpose='save',generation=0,identity='control',budget=Admission())
        assert account.used==original_charge
        assert fd_count()==original_fds
        assert not lease.released and parent.outstanding==(lease,)
    finally:
        process.wait(timeout=5)
        lease.release_after_reap(process)
        bank.close()
        parent.close()
        receiver.close()
    assert account.used==0


@pytest.mark.parametrize('fault',['retain','admission','ack'])
def test_actual_wire7_retain_emfile_keeps_original_delivery_until_known_cleanup(tmp_path,monkeypatch,fault):
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    learner=learner_for()
    account,ledger=budget(),ReservationLedger()
    failure=(ContractError('actual control admission') if fault=='admission' else
        OSError(errno.EMFILE if fault=='retain' else errno.EIO,'actual received construction failure'))
    if fault=='retain':
        def fail_retain(self,account): raise failure
        monkeypatch.setattr(ReceivedBank,'retain',fail_retain)
    elif fault=='admission':
        real_reserve=account.reserve
        calls=[]
        def reserve(size,*,wait=False):
            calls.append(size)
            if len(calls)==2: raise failure
            return real_reserve(size,wait=wait)
        monkeypatch.setattr(account,'reserve',reserve)
    else:
        real_complete=ReceivedBank.complete
        def complete(self,synchronize):
            if self._envelope['identity'].endswith(':inference:0'): raise failure
            return real_complete(self,synchronize)
        monkeypatch.setattr(ReceivedBank,'complete',complete)
    async def run():
        original_fds=fd_count()
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+20,budget=account)
        await child.start()
        try:
            with pytest.raises(type(failure)) as caught:
                await child.initialize_owned(17,tmp_path/'g0',deadline=time.monotonic()+10)
            assert caught.value is failure
        finally:
            try: await child.close(time.monotonic()+1)
            except (ContractError,RuntimeError,TimeoutError,BrokenPipeError): pass
        await asyncio.sleep(0)
        assert child.process.returncode is not None and child.lease.released
        assert not child._partial_banks and not child._partial_controls
        assert account.used==0
        assert fd_count()==original_fds
    asyncio.run(run())


def test_channel_keeps_direct_received_owner_through_close_and_live_export():
    account=budget()
    bank=RamBankBuilder({'values':np.arange(4,dtype=np.int64)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    source_charge=account.used
    original_fds=fd_count()
    lease=parent.send(bank,purpose='batch',generation=0,identity='direct')
    delivery=receiver.receive(purpose='batch',generation=0,identity='direct',budget=account)
    alias=delivery.bank.array('values').base.base
    assert receiver.received_pending==(delivery,)
    with pytest.raises(BufferError): receiver.retire_received(lambda:None)
    assert account.used>source_charge and not lease.released
    receiver.close()
    assert receiver.received_pending==(delivery,) and not lease.released
    alias.release()
    del alias
    receiver.retire_received(lambda:None)
    assert not receiver.received_pending and account.used==source_charge
    # The receiver socket alone closed, so there is one fewer FD than before
    # transfer; the original sender bank and its lease still survive.
    assert fd_count()==original_fds-1
    with pytest.raises(ContractError): delivery.complete(lambda:None)
    process.wait(timeout=5)
    lease.release_after_reap(process)
    bank.close()
    parent.close()
    assert account.used==0


_UNCERTAIN=[]


def test_received_uncertain_close_keeps_exact_owner_and_never_retries_mapping():
    from gf_common.ram_bank import RamBankCleanupError
    account=budget()
    bank=RamBankBuilder({'values':np.arange(4,dtype=np.int64)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    lease=parent.send(bank,purpose='batch',generation=0,identity='direct')
    delivery=receiver.receive(purpose='batch',generation=0,identity='direct',budget=account)
    class Mapping:
        calls=0
        def __init__(self,real): self.real=real
        def close(self):
            self.calls+=1
            raise OSError(errno.EIO,'uncertain receive mapping close')
    mapping=Mapping(delivery.bank._mapping)
    delivery.bank._mapping=mapping
    charge=account.used
    with pytest.raises(RamBankCleanupError) as caught: receiver.retire_received(lambda:None)
    assert caught.value.cleanup_owner is delivery.bank
    assert receiver.received_pending==(delivery,) and account.used==charge
    with pytest.raises(RamBankCleanupError): receiver.retire_received(lambda:None)
    assert mapping.calls==1
    process.wait(timeout=5)
    lease.release_after_reap(process)
    bank.close()
    parent.close()
    receiver.close()
    assert account.used>0
    _UNCERTAIN.append((receiver,account))


def test_numeric_receive_layout_failure_retires_data_and_decoded_control_without_ack():
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from gf_training.hpo_owned_worker import _control_bank, _receive_numeric
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_owned_numeric import captured_record
    learner=learner_for()
    batch,owner,plan,snapshot,account,_=captured_record(learner)
    selection=selection_for_counts((3,),learning_seed=17,batch=0,decision_budget=3,used_before=0)
    numeric=encode_owned_batch(batch,owner=owner,plan=plan,snapshot_owner=snapshot,
        model=learner.model,training=learner.training,actions=ACTIONS,rules=RULES,
        selected_locations=((0,0),(0,1),(0,2)),selection=selection,budget=account)
    control=_control_bank(numeric.metadata,account)
    malformed=RamBankBuilder({'wrong':np.ones(1,np.int64)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    baseline,descriptors=account.used,fd_count()
    header=parent.send(control,purpose='batch',generation=0,identity=numeric.identity+':metadata')
    data=parent.send(malformed,purpose='batch',generation=0,identity=numeric.identity+':0')
    request={'generation':0,'batch_identity':numeric.identity,'bank_count':1,'policy_version':numeric.policy_version}
    with pytest.raises(ContractError,match='layout'):
        _receive_numeric(receiver,request,learner,account,snapshot.identity,ACTIONS,512*1024**2)
    assert account.used==baseline and fd_count()==descriptors
    assert not receiver.received_pending and not data.released
    parent.accept_completion(header)
    process.wait(timeout=5)
    data.release_after_reap(process)
    receiver.close()
    parent.close()
    control.close()
    malformed.close()
    numeric.close()
    numeric.require_retired()
    batch.close()
    snapshot.close()


def test_actual_parent_reap_cleanup_defers_retained_view_then_retries_without_resending(tmp_path,monkeypatch):
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    learner=learner_for()
    account,ledger=budget(),ReservationLedger()
    aliases=[]
    failure=OSError(errno.EIO,'after retained original before ACK')
    retain,complete=ReceivedBank.retain,ReceivedBank.complete
    def keep_alias(self,admission):
        bank=retain(self,admission)
        aliases.append(bank.array(bank.descriptors[0].name).base.base)
        return bank
    def fail_complete(self,synchronize):
        if self._envelope['identity'].endswith(':inference:0'): raise failure
        return complete(self,synchronize)
    monkeypatch.setattr(ReceivedBank,'retain',keep_alias)
    monkeypatch.setattr(ReceivedBank,'complete',fail_complete)
    async def run():
        original_fds=fd_count()
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+20,budget=account)
        await child.start()
        with pytest.raises(OSError) as caught:
            await child.initialize_owned(17,tmp_path/'g0',deadline=time.monotonic()+10)
        assert caught.value is failure
        try: await child.close(time.monotonic()+1)
        except (ContractError,RuntimeError,TimeoutError,BrokenPipeError): pass
        assert child.process.returncode is not None and child.lease.released
        assert child._partial_banks and account.used>0
        assert all(not bank._close_uncertain for bank in child._partial_banks)
        for alias in aliases: alias.release()
        aliases.clear()
        try: await child.close(time.monotonic()+1)
        except (ContractError,RuntimeError,TimeoutError,BrokenPipeError): pass
        await asyncio.sleep(0)
        assert not child._partial_banks and not child.channel.received_pending
        assert account.used==0 and fd_count()==original_fds
        assert child.current_snapshot is None and child._owned_failed
    asyncio.run(run())
