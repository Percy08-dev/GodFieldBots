"""Immutable single-export learner snapshots and inference-only restore."""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from gf_common.domain import ContractError
from tests.test_hpo_numeric import learner_for
from tests.test_hpo_owned_numeric import budget, captured_record


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('updated', [False, True])
def test_snapshot_same_actual_inference_resume_arrays_and_distinct_actor_backing(tmp_path, monkeypatch, kind, updated):
    import torch

    from gf_models.hpo_owned_inference import load_owned_inference
    from gf_models.sharded_arrays import read_bundle, read_json
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import HPOTrainingConfig
    from gf_training.hpo_owned_rollout import encode_owned_batch
    from tests.test_card_instance_learning import ACTIONS, RULES
    learner = learner_for(kind, recipe=replace(HPOTrainingConfig(), epochs=1))
    account = budget()
    if updated:
        batch, owner, plan, snapshot, account, _ = captured_record(learner)
        selection = selection_for_counts((3,), learning_seed=17, batch=0, decision_budget=3, used_before=0)
        numeric = encode_owned_batch(batch, owner=owner, plan=plan, snapshot_owner=snapshot,
            model=learner.model, training=learner.training, actions=ACTIONS, rules=RULES,
            selected_locations=((0,0),(0,1),(0,2)), selection=selection, budget=account)
        numeric.persist(tmp_path/'numeric')
        learner.update_owned(numeric)
        numeric.close()
        batch.close()
        snapshot.close()
    if updated:
        with pytest.raises(ContractError,match='Source'):
            learner.snapshot_owned(budget=budget())
    snapshot = learner.snapshot_owned(budget=account)
    assert snapshot is learner.snapshot_owned(budget=account)
    assert snapshot.generation == int(updated) and snapshot.optimizer_steps == learner.steps
    infer_inodes = {os.fstat(bank.fileno()).st_ino for bank in snapshot.inference.banks}
    resume_inodes = {os.fstat(bank.fileno()).st_ino for bank in snapshot.resume.banks}
    assert infer_inodes.isdisjoint(resume_inodes)
    assert all(not name.startswith(('adam.', 'rng.')) for name in snapshot.inference.names)
    learner.save(tmp_path/'legacy')
    with monkeypatch.context() as patch:
        def forbidden(*args,**kw): pytest.fail('owned snapshot called a strict reader')
        patch.setattr('gf_models.sharded_arrays.inspect_bundle',forbidden)
        patch.setattr('gf_models.sharded_arrays.read_bundle',forbidden)
        patch.setattr('gf_models.sharded_arrays.load_arrays',forbidden)
        ack = snapshot.save(tmp_path/'owned')
    ack.require(snapshot)
    for manifest, field in [('inference.json','weights'), ('resume.json','state')]:
        actual = read_json(tmp_path/'owned'/manifest)
        expected = read_json(tmp_path/'legacy'/manifest)
        aa = read_bundle(tmp_path/'owned', actual[field])
        ea = read_bundle(tmp_path/'legacy', expected[field])
        assert aa.keys() == ea.keys()
        for name in aa:
            np.testing.assert_array_equal(aa[name], ea[name])
    rng = torch.get_rng_state().clone()
    network = load_owned_inference(snapshot.inference, snapshot.actor_metadata,
        model=learner.model, binding=learner.binding, actions=ACTIONS, rules=RULES)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in learner.network.state_dict().items():
        torch.testing.assert_close(value, network.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ContractError):
        load_owned_inference(snapshot.resume, snapshot.actor_metadata,
            model=learner.model, binding=learner.binding, actions=ACTIONS, rules=RULES)
    snapshot.close()


def test_partial_snapshot_save_never_issues_ack_and_poison_learner(tmp_path, monkeypatch):
    from gf_training import hpo_owned_snapshot
    learner = learner_for()
    snapshot = learner.snapshot_owned(budget=budget())
    real = hpo_owned_snapshot._write_producer_bundle
    def fail_resume(directory, prefix, arrays, **kw):
        if prefix == 'resume':
            raise OSError('partial save')
        return real(directory, prefix, arrays, **kw)
    monkeypatch.setattr(hpo_owned_snapshot, '_write_producer_bundle', fail_resume)
    with pytest.raises(OSError, match='partial'):
        snapshot.save(tmp_path/'partial')
    assert snapshot.save_ack is None and learner._owned_failed
    with pytest.raises(ContractError):
        snapshot.save(tmp_path/'retry')
    snapshot.close()


def test_received_retained_snapshot_survives_transfer_ack(tmp_path):
    import subprocess
    import sys

    from gf_common.ram_bank import RamBankBuilder, RamBankChannel
    account = budget()
    original = RamBankBuilder({'weight': np.ones(4, np.float32)}, account).seal()
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    parent, receiver = RamBankChannel.pair(child=process)
    lease = parent.send(original, purpose='model', generation=0, identity='snapshot')
    delivery = receiver.receive(purpose='model', generation=0, identity='snapshot', budget=account)
    retained = delivery.retain(account)
    assert os.fstat(retained.fileno()).st_ino == os.fstat(original.fileno()).st_ino
    delivery.complete(lambda: None)
    parent.accept_completion(lease)
    assert lease.released
    np.testing.assert_array_equal(retained.array('weight'), np.ones(4, np.float32))
    original.close()
    np.testing.assert_array_equal(retained.array('weight'), np.ones(4, np.float32))
    retained.close()
    process.wait(timeout=5)
    parent.close()
    receiver.close()
    assert account.used == 0


def test_retained_receiver_dup_failure_refunds_only_new_attempt(monkeypatch):
    import errno
    import subprocess
    import sys

    from gf_common.ram_bank import RamBankBuilder, RamBankChannel
    account=budget()
    original=RamBankBuilder({'weight':np.ones(4,np.float32)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    lease=parent.send(original,purpose='model',generation=0,identity='snapshot')
    delivery=receiver.receive(purpose='model',generation=0,identity='snapshot',budget=account)
    charge=account.used
    with monkeypatch.context() as patch:
        def refuse_dup(fd): raise OSError(errno.EMFILE,'dup')
        patch.setattr('gf_common.ram_bank.os.dup',refuse_dup)
        with pytest.raises(OSError): delivery.retain(account)
    assert account.used==charge and not lease.released
    delivery.complete(lambda:None)
    parent.accept_completion(lease)
    original.close()
    process.wait(timeout=5)
    parent.close()
    receiver.close()
    assert account.used==0


def test_cancellation_with_live_retained_view_keeps_real_mapping_charge():
    import subprocess
    import sys

    from gf_common.ram_bank import RamBankBuilder, RamBankChannel
    account=budget()
    original=RamBankBuilder({'weight':np.ones(4,np.float32)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
    parent,receiver=RamBankChannel.pair(child=process)
    lease=parent.send(original,purpose='model',generation=0,identity='snapshot')
    delivery=receiver.receive(purpose='model',generation=0,identity='snapshot',budget=account)
    retained=delivery.retain(account)
    view=retained.array('weight')[1:]
    lease.cancel()
    with pytest.raises(BufferError): original.close()
    with pytest.raises(BufferError): retained.close()
    charge=account.used
    assert charge>0
    process.terminate()
    process.wait(timeout=5)
    lease.release_after_reap(process)
    original.close()
    delivery.bank.close()
    assert account.used>0 and account.used<charge
    np.testing.assert_array_equal(view,np.ones(3,np.float32))
    del view
    retained.close()
    parent.close()
    receiver.close()
    assert account.used==0


def test_snapshot_ack_cannot_be_rebound_to_other_generation_or_snapshot(tmp_path):
    learner=learner_for()
    snapshot=learner.snapshot_owned(budget=budget())
    ack=snapshot.save(tmp_path/'g0')
    other=learner_for().snapshot_owned(budget=budget())
    with pytest.raises(ContractError): ack.require(other)
    snapshot.generation+=1
    with pytest.raises(ContractError): ack.require(snapshot)
    snapshot.generation-=1
    ack.require(snapshot)
    snapshot.close()
    other.close()


# Terminal uncertainty deliberately keeps its actual owner until this bounded
# test process exits; tests must not retry or force-refund historical resources.
_QUARANTINED = []


def test_retained_mapping_close_uncertainty_quarantines_original_owner():
    import errno
    import subprocess
    import sys

    from gf_common.ram_bank import RamBankBuilder, RamBankChannel, RamBankCleanupError
    account=budget()
    original=RamBankBuilder({'weight':np.ones(4,np.float32)},account).seal()
    process=subprocess.Popen([sys.executable,'-c','pass'])
    parent,receiver=RamBankChannel.pair(child=process)
    lease=parent.send(original,purpose='model',generation=0,identity='snapshot')
    delivery=receiver.receive(purpose='model',generation=0,identity='snapshot',budget=account)
    retained=delivery.retain(account)
    real=retained._mapping
    class UncertainClose:
        calls=0
        def close(self):
            self.calls+=1
            raise OSError(errno.EIO,'uncertain retained mapping close')
    mapping=UncertainClose()
    retained._mapping=mapping
    charge=account.used
    with pytest.raises(RamBankCleanupError) as caught: retained.close()
    assert caught.value.cleanup_owner is retained and account.used==charge
    with pytest.raises(RamBankCleanupError): retained.close()
    assert mapping.calls==1
    delivery.complete(lambda:None)
    parent.accept_completion(lease)
    original.close()
    process.wait(timeout=5)
    parent.close()
    receiver.close()
    assert account.used>0
    _QUARANTINED.append((retained,real,account))
