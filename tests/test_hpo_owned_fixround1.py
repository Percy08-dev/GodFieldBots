"""Accepted review regressions: shared admission, original retention and startup."""
from __future__ import annotations

import time

import pytest

from gf_common.domain import ContractError
from tests.test_hpo_owned_numeric import budget


@pytest.mark.parametrize('denial',['parent','source'])
def test_control_denial_precedes_all_canonical_encoding(tmp_path,monkeypatch,denial):
    from gf_training.hpo_owned_lineage import OwnedControlWriter
    from gf_training.lineage import run_lock
    account=budget()
    if denial=='source': account.limit=1
    def forbidden(*args,**kwargs): raise AssertionError('serialized before admission')
    monkeypatch.setattr('gf_training.hpo_owned_lineage.canonical',forbidden)
    with run_lock(tmp_path) as owner:
        writer=OwnedControlWriter(owner,budget=account,limit=1 if denial=='parent' else 1024**2)
        with pytest.raises((ValueError,ContractError),match='capacity|budget|envelope|reservation'):
            writer.write(tmp_path/'value.json',{'text':'before allocation'},deadline=time.monotonic()+5)
        assert writer.failed is False
    assert account.used==0


def test_shared_parent_phase_denies_overlapping_reservations_and_rolls_back():
    from gf_common.artifacts import ArtifactPhaseAdmission
    account=budget()
    phase=ArtifactPhaseAdmission(account,40_900_000)
    reservations=[phase.reserve(size) for size in (6_525_632,277_696,1_628_800)]
    assert phase.used==account.used==8_432_128
    with pytest.raises(ValueError): phase.reserve(32_762_240)
    assert phase.used==account.used==8_432_128
    for item in reservations: item.close()
    assert phase.used==account.used==0
    account.limit=1
    with pytest.raises(ValueError): phase.reserve(2)
    assert phase.used==0


def test_original_retention_protects_g0_and_resume_only_pairs(tmp_path):
    from gf_training.hpo_owned_lineage import OwnedRunRegistry
    from gf_training.lineage import run_lock
    files={'inference.json':'i','weights-000000.npz':'w','resume.json':'r','resume-000000.npz':'a'}
    with run_lock(tmp_path) as owner:
        registry=OwnedRunRegistry(owner,{},budget=budget(),parent_budget_bytes=1024**2)
        registry.records=[{'generation':g,'artifact_files':files,'used_decisions':g,
            'parent_controls':{'snapshot_pool':[4]}} for g in range(5)]
        protected=registry.protected_files({'inference_latest':1,'resume_latest':3,'inference_periodic_decisions':1000})
        assert protected[0]==set(files)
        assert protected[2]==set(files) and protected[3]==set(files) and protected[4]==set(files)
        assert protected[1]==set()
        registry.close()


@pytest.mark.parametrize('swap',['leaf','parent','none'])
def test_owned_retirement_never_deletes_replacement_after_intent(tmp_path,monkeypatch,swap):
    import os
    from types import SimpleNamespace

    from gf_training.hpo_owned_lineage import OwnedRunRegistry, _identity
    from gf_training.lineage import run_lock
    account=budget()
    with run_lock(tmp_path) as owner:
        (tmp_path/'artifacts').mkdir()
        (tmp_path/'retention').mkdir()
        records=[]
        for generation in range(4):
            folder=tmp_path/'artifacts'/f'{generation:06}'
            folder.mkdir()
            (folder/'inference.json').write_bytes(b'original')
            records.append({'artifact_files':{'inference.json':'a'*64},'generation':generation,'used_decisions':generation,
                'parent_controls':{'snapshot_pool':[3]}})
        registry=OwnedRunRegistry(owner,{},budget=account,parent_budget_bytes=16*1024**2)
        registry.records=records
        registry.current=SimpleNamespace(generation=3)
        registry.registry['generations']={str(g):{'files':{'inference.json':{'status':'retained','sha256':'a'*64,
            'deletion_intent':None,'deletion_receipt':None}}} for g in range(4)}
        registry.file_identities={(g,'inference.json'):_identity(tmp_path/'artifacts'/f'{g:06}'/'inference.json') for g in range(4)}
        # These are the actual publication-time directory originals.
        registry.artifact_identity=__import__('gf_training.structured_lineage',fromlist=['_stat_identity'])._stat_identity((tmp_path/'artifacts').lstat())
        registry.generation_directories={g:__import__('gf_training.structured_lineage',fromlist=['_stat_identity'])._stat_identity((tmp_path/'artifacts'/f'{g:06}').lstat()) for g in range(4)}
        victim=tmp_path/'artifacts/000001/inference.json'
        backup=tmp_path/'original-backup'
        foreign=tmp_path/'foreign'
        foreign.mkdir()
        (foreign/'inference.json').write_bytes(b'foreign')
        write=registry.writer.write
        injected=False
        def inject(path,value,**options):
            nonlocal injected
            result=write(path,value,**options)
            if path.name.endswith('.intent.json') and not injected:
                injected=True
                if swap=='leaf':
                    os.rename(victim,backup)
                    os.replace(foreign/'inference.json',victim)
                elif swap=='parent':
                    os.rename(victim.parent,backup)
                    victim.parent.symlink_to(foreign,target_is_directory=True)
            return result
        monkeypatch.setattr(registry.writer,'write',inject)
        if swap=='none':
            registry.retain({'inference_latest':1,'resume_latest':2,'inference_periodic_decisions':1000},
                deadline=time.monotonic()+10,pool=None)
            assert not victim.exists()
            assert list((tmp_path/'retention').glob('*.receipt.json'))
            assert registry.registry['generations']['1']['files']['inference.json']['status']=='deleted'
            assert not registry.failed
            registry.close()
            assert account.used==0
            return
        with pytest.raises((ContractError,OSError)):
            registry.retain({'inference_latest':1,'resume_latest':2,'inference_periodic_decisions':1000},
                deadline=time.monotonic()+10,pool=None)
        assert injected
        assert (victim if swap=='leaf' else foreign/'inference.json').read_bytes()==b'foreign'
        assert (backup if swap=='leaf' else backup/'inference.json').read_bytes()==b'original'
        assert not list((tmp_path/'retention').glob('*.receipt.json'))
        assert registry.registry['generations']['1']['files']['inference.json']['status']!='deleted'
        # A failed mutation is ineligible and keeps any uncertain original custody.
        assert registry.failed
        for retirement in registry._retirement_owners: retirement.close()
        registry.close()
        assert account.used==0


def test_provider_failed_validation_refunds_shared_parent_reservation(tmp_path):
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    from tests.test_hpo_numeric import learner_for
    account=budget()
    learner=learner_for()
    snapshot=learner.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'model')
    cap=snapshot.actor_capability
    before=account.used
    phase=ArtifactPhaseAdmission(account,256*1024**2)
    manifest=snapshot.inference_manifest
    manifest['kind']='wrong'
    with pytest.raises(ContractError):
        OwnedGameArtifacts({'fixture':(cap,RuntimePolicyBinding('mlp',{'expected_manifest_digest':cap.checkpoint}),[],manifest)},
            budget=account,parent_budget=phase)
    assert phase.used==0 and account.used==before
    snapshot.close()


def test_parent_counter_is_atomic_across_threads():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from gf_common.artifacts import ArtifactPhaseAdmission
    account=budget()
    phase=ArtifactPhaseAdmission(account,1024**2)
    gate=Barrier(2)
    def overlap():
        lease=None
        try:
            try: lease=phase.reserve(600_000)
            except ValueError: pass
            gate.wait(timeout=5)
            return lease is not None
        finally:
            if lease is not None: lease.close()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(overlap) for _ in range(2)]
        assert sum(future.result(timeout=10) for future in futures)==1
    assert phase.peak==600_000 and phase.used==account.used==0


def test_failed_phase_lease_construction_refunds_original_source(monkeypatch):
    from gf_common.artifacts import ArtifactPhaseAdmission
    account=budget()
    phase=ArtifactPhaseAdmission(account,1024**2)
    def failed(*args,**kwargs): raise MemoryError('lease construction')
    monkeypatch.setattr('gf_common.artifacts._PhaseReservation',failed)
    with pytest.raises(MemoryError): phase.reserve(1024)
    assert phase.used==account.used==0


@pytest.mark.parametrize('alternate',[False,'factory','bound','marker'])
def test_owned_unstarted_close_requires_exact_original_creator(tmp_path,alternate):
    import asyncio
    import os

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    learner=learner_for()
    account=budget()
    async def run():
        before=len(os.listdir('/proc/self/fd'))
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ReservationLedger(),
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+5,budget=account)
        if alternate=='factory':
            original=child._create_process
            child._create_process=lambda:original()
        elif alternate=='bound':
            from types import MethodType
            child._create_process=MethodType(child._create_process.__func__,object())
        elif alternate=='marker':
            child._owned_factory_request=False
        try:
            if alternate:
                with pytest.raises(ContractError,match='original|factory|creator'):
                    await child.close(time.monotonic()+5)
                assert child.channel.socket.fileno()>=0 and child._inherited.socket.fileno()>=0
            else:
                await child.close(time.monotonic()+5)
                assert len(os.listdir('/proc/self/fd'))==before
        finally:
            child.channel.close()
            child._inherited.close()
        assert len(os.listdir('/proc/self/fd'))==before and account.used==0
    asyncio.run(run())
