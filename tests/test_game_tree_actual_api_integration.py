"""Actual accepted API integration; UNADMITTED resource/DTO controls only.

No formal Parent is successfully admitted, no qualified Dataset is fabricated,
and no Booster, learning, GPU, Simulator or game process is created.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.card_instance import action_config_v3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, canonical, digest
from gf_common.preprocessing import PreparationConfig
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature, tokenize_frame, tokenize_table
from gf_models.structured_sealed import SealedScan
from gf_models.structured_tree_config import ConfiguredTreeModelV3, TreeEncoderConfigV3
from gf_models.structured_tree_encoder import feature_layout_v3
from gf_training.card_instance_final_validation import (
    GameFinalValidationSource,
    _Final64OriginalContract,
    open_parent_game_final_validation,
)
from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
from gf_training.card_instance_game_tree_data import (
    GameTreeBudget,
    VerifiedGameTreeDataset3,
    _GameTreeOriginals,
    _tree_tokens,
    build_game_tree_dataset,
    inspect_game_tree_dataset,
)
from gf_training.card_instance_journal import read_card_instance_journal
from gf_training.card_instance_source_locks import GameSourceCustody
from gf_training.lineage import _OWNED_LOCKS
from gf_training.structured_rollout import game_episode_records
from gf_training.structured_tree_fit import _OwnedTreeFitChild, fit_game_dataset

ROOT=Path(__file__).resolve().parents[1]


def settings():return PreparationConfig(reader='participant-structured-v2')


def roots(tmp_path):
    parents=tuple(tmp_path/f'UNADMITTED-parent-{i}' for i in range(3))
    validations=tuple(tmp_path/f'UNADMITTED-validation-{i}' for i in range(3))
    for path in (*parents,*validations):path.mkdir()
    for path in parents:(path/'run.json').write_bytes(b'{"schema_version":3,"scope":"UNADMITTED-ordinary-NOT-formal"}')
    for path in validations:(path/'manifest.json').write_bytes(b'{"scope":"UNADMITTED-resource-NOT-C5"}')
    return parents,validations


def assert_retired(parents,validations):
    assert all(path.resolve() not in _OWNED_LOCKS for path in parents)
    assert all({p.name for p in path.iterdir()}=={'manifest.json'} for path in validations)


def index(tmp_path,parents,validations):
    # Existing pure intent DTO, not fabricated original controls/runtime. Even
    # resealing these fields cannot pass actual FormalGameParent.__enter__.
    from gf_common import card_instance_game_tree as game
    from tests.test_game_tree_dataset import dto
    raw=dto()
    encoder=TreeEncoderConfigV3();layout=feature_layout_v3(encoder)
    raw['tree_projection']={'encoder':asdict(encoder),'feature_layout':layout,
        'feature_signature':layout['feature_signature'],'token_signature':signature(encoder.aggregate_window,layout=LAYOUT)}
    for i,parent in enumerate(raw['parents']):parent['directory']=str(parents[i])
    for row in raw['collections']:
        i=row['parent_index']
        row['directory']=str(validations[i] if row['split']=='validation' else parents[i]/'unadmitted-cohort')
        if row['numeric'] is not None:row['numeric']['directory']=str(parents[i]/'unadmitted-numeric')
    raw['source_set_sha256']=game.source_identity(raw['parents'],raw['collections'])
    raw['shared_generation']=game.shared_generation(raw['parents'],raw['collections'])
    raw['content_sha256']=digest({key:value for key,value in raw.items() if key!='content_sha256'})
    directory=tmp_path/'intent-index';directory.mkdir()
    path=directory/'manifest.json';path.write_text(canonical(raw)+'\n')
    receipt={'schema_version':1,'kind':'card-instance-game-dataset-preparation',
        'dataset_content_sha256':raw['content_sha256'],'seconds':0.,
        'scope':'actual original inspection and selected Tree projection; excludes shared PPO/final-validation generation and fitting'}
    (directory/'preparation.json').write_text(canonical(receipt)+'\n')
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('route',['build','inspect','fit'])
def test_all_public_routes_reject_actual_ordinary_parent_before_rows_child_or_output(tmp_path,monkeypatch,route):
    parents,validations=roots(tmp_path)
    before=[(path/'run.json').read_bytes() for path in parents]
    def forbidden(*args,**kwargs):pytest.fail('ordinary parent reached row/child allocation')
    monkeypatch.setattr(np,'empty',forbidden)
    monkeypatch.setattr(_OwnedTreeFitChild,'start',forbidden)
    output=tmp_path/'must-not-exist'
    with pytest.raises(ContractError,match='original formal declaration'):
        if route=='build':
            build_game_tree_dataset(parents,validations,output,settings(),encoder=TreeEncoderConfigV3(),deadline=time.monotonic()+60)
        else:
            path,sha=index(tmp_path,parents,validations)
            if route=='inspect':inspect_game_tree_dataset(path,expected_digest=sha,settings=settings(),deadline=time.monotonic()+60)
            else:
                from gf_models.config import TrainingConfig
                asyncio.run(fit_game_dataset(path,expected_digest=sha,directory=output,model=ConfiguredTreeModelV3(),
                    settings=settings(),training=TrainingConfig(),seed=17,deadline=time.monotonic()+60))
    assert not output.exists()
    assert [(path/'run.json').read_bytes() for path in parents]==before
    assert_retired(parents,validations)


def test_dataset_preentry_full_child_and_other_metadata_reach_actual_parent_scan(tmp_path,monkeypatch):
    import gf_training.structured_lineage as lineage
    parents,validations=roots(tmp_path)
    observed=[]
    class StopBeforeDecode(RuntimeError):pass
    def read(scan,path,*args):
        observed.append((scan.limit,path));raise StopBeforeDecode('UNADMITTED pre-entry scan bound')
    monkeypatch.setattr(lineage._ChainScan,'object',read)
    total=64*1024**2;child=32*1024**2;caller=16384
    source=_GameTreeOriginals(parents,validations,settings(),encoder=TreeEncoderConfigV3(),max_owned_bytes=total,
        deadline=time.monotonic()+60,extra_owned_bytes=caller,child_limit_bytes=child)
    with pytest.raises(StopBeforeDecode):source.__enter__()
    assert observed==[(total-child-caller-131072,parents[0]/'run.json')]
    assert source.cleanup_complete
    assert_retired(parents,validations)


def test_actual_rust_original_history_gate_refuses_before_index_when_marker_unpublished(tmp_path):
    from gf_training.native_backend import load_native
    try:native=load_native()
    except (ContractError,ImportError):pytest.skip('optional installed native extension unavailable')
    if dict(native.build_info()).get('card_instance_original_history')=='journal6;codec2;episode-dto1;recorded16':
        pytest.skip('negative requires currently unpublished installed history marker')
    with pytest.raises(ContractError):
        VerifiedGameTreeDataset3(tmp_path/'missing','a'*64,replace(settings(),backend='rust'),deadline=time.monotonic()+60)
    assert not tuple(tmp_path.iterdir())


def test_dataset_closed_before_first_entry_cannot_reopen_or_allocate_actual_parent(tmp_path,monkeypatch):
    dataset=VerifiedGameTreeDataset3(tmp_path/'missing','a'*64,settings(),deadline=time.monotonic()+60)
    dataset.close()
    assert dataset.cleanup_complete
    def forbidden(*args,**kwargs):pytest.fail('closed Dataset attempted actual parent opening')
    monkeypatch.setattr(FormalGameParent,'__enter__',forbidden)
    with pytest.raises(ContractError,match='one lifetime'):dataset.__enter__()


def primitive_parent(keeper,path,limit,*,extra=0):
    # Same actual class/custody/counters, intentionally NO admission/run controls.
    parent=FormalGameParent(path,max_owned_bytes=limit,deadline=keeper.deadline,
        ownership=keeper.ownership_for(path),extra_owned_bytes=extra)
    parent._entered=parent._active=True;parent._token=parent.borrowed.token
    parent.scan=SimpleNamespace(retained=1024,files={},inventories={},finalize=lambda:None)
    parent.recheck=parent.require_active
    return parent


def test_actual_three_parent_loans_and_global_governor_preserve_base_until_physical_retirement(tmp_path):
    parents,validations=roots(tmp_path)
    total=512*1024**2;child=256*1024**2
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        actual=[primitive_parent(keeper,path,total) for path in parents]
        governor=GameTreeBudget(total,dataset_bytes=65536,child_limit_bytes=child)
        for i,parent in enumerate(actual):governor.parent_intrinsic(i,parent.resident_bytes)
        loans=[];physical=[]
        # Exercise Dataset's actual counter propagation kernel with actual
        # accepted resource APIs; this shell has NO state/worklist/qualification.
        manager=object.__new__(_GameTreeOriginals)
        manager._budget=governor;manager._parents=actual;manager._sources=physical
        manager._loans=loans;manager._custody=keeper
        for i,parent in enumerate(actual):
            loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=governor.external_base(i),deadline=keeper.deadline)
            loan.admit_local(16384)
            source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),(),
                {'scope':'UNADMITTED-physical-resource-NOT-Source-admission'},budget_loan=loan,reservation=16384)
            loan._bind_source(source)
            loans.append(loan);physical.append(source)
            manager._sync()
        governor.reserve_sink(32768)
        manager._sync()
        assert governor.fit_memory()['child_limit_bytes']==child
        assert governor.retained_bytes==sum(parent.resident_bytes for parent in actual)+3*16384+65536+32768+child
        bases=[loan.base_bytes for loan in loans]
        for parent,loan,source,base in zip(actual,loans,physical,bases,strict=True):
            with pytest.raises(ContractError):loan.retire_local(source)
            source.close()
            assert source.cleanup_complete and loan.cleanup_complete and parent._external_bytes==base
            source.close()
            assert parent._external_bytes==base  # idempotent close does not erase caller/other parents/sink
        for parent in reversed(actual):parent.__exit__(None,None,None)
        assert all(keeper.ownership_for(path) is parent.borrowed for path,parent in zip(parents,actual,strict=True))
    assert_retired(parents,validations)


def test_third_actual_parent_scan_preentry_charges_two_retained_resource_owners_and_full_child(tmp_path,monkeypatch):
    import gf_training.structured_lineage as lineage
    parents,validations=roots(tmp_path)
    total=64*1024**2;child=32*1024**2
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        retained=[primitive_parent(keeper,path,total) for path in parents[:2]]
        budget=GameTreeBudget(total,dataset_bytes=131072,child_limit_bytes=child)
        for i,parent in enumerate(retained):budget.parent_intrinsic(i,parent.resident_bytes)
        extra=budget.retained_bytes
        observed=[]
        class StopBeforeDecode(RuntimeError):pass
        def read(scan,path,*args):observed.append(scan.limit);raise StopBeforeDecode('UNADMITTED third-owner preflight')
        monkeypatch.setattr(lineage._ChainScan,'object',read)
        third=FormalGameParent(parents[2],max_owned_bytes=total,deadline=keeper.deadline,
            ownership=keeper.ownership_for(parents[2]),extra_owned_bytes=extra)
        with pytest.raises(StopBeforeDecode):third.__enter__()
        assert observed==[total-extra] and extra==child+131072+sum(parent.resident_bytes for parent in retained)
        assert third.cleanup_complete
        for parent in reversed(retained):parent.__exit__(None,None,None)
    assert_retired(parents,validations)


def test_readonly_opener_requires_same_actual_loan_and_manifest_token_before_contract_or_files(tmp_path):
    parents,validations=roots(tmp_path)
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        parent=primitive_parent(keeper,parents[0],64*1024**2)
        loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=4096,deadline=keeper.deadline)
        with pytest.raises(ContractError,match='SAME complete-custody'), open_parent_game_final_validation(
                parent,validations[0],manifest_ownership=keeper.ownership_for(validations[1]),
                budget_loan=loan,max_owned_bytes=parent.limit,deadline=keeper.deadline):
            pytest.fail('foreign manifest token opened readonly Source')
        # No Source construction or qualification occurred; borrowed locks live.
        assert not loan.cleanup_complete
        parent.__exit__(RuntimeError,RuntimeError('UNADMITTED resource cleanup'),None)
        assert loan.cleanup_complete and parent.cleanup_complete
    assert_retired(parents,validations)


@pytest.fixture(scope='module')
def original_records(tmp_path_factory):
    # Frozen accepted structural DTO fixture. It is NEVER formal/runtime evidence.
    fixture=json.loads((ROOT/'native/tests/fixtures/card_instance_final64_originals.json').read_bytes())
    directory=tmp_path_factory.mktemp('UNADMITTED-dataset-DTOs')
    for name,raw in fixture['files'].items():
        path=directory/name;path.parent.mkdir(exist_ok=True);path.write_bytes(bytes.fromhex(raw))
    actions=action_config_v3(fixture['state']['effective_config']['actions'])
    records=[]
    for row in fixture['oracle']:
        data=read_card_instance_journal(directory/row['episode_id']/f"player-{row['viewer']}.jsonl",actions=actions)
        decisions=[data.decisions[i] for i in data.accepted]
        records.append(game_episode_records(data.bank,tuple(r.frame for r in decisions),tuple(r.generated.table for r in decisions),
            tuple(r.selection for r in decisions),'natural',row['outcome'],actions=actions,rules=card_instance_rules()))
    manifest=json.loads((directory/'manifest.json').read_bytes())
    return tuple(records),actions,manifest['linkage']['policy_label']


def test_actual_original_borrow_feeds_dataset_tree_window_and_zero_endpoint_without_nn_cache(original_records,tmp_path):
    records,actions,label=original_records
    parents,validations=roots(tmp_path)
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        parent=primitive_parent(keeper,parents[0],512*1024**2)
        loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=65536,deadline=keeper.deadline)
        loan.admit_local(64*1024**2)
        source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),records,
            {'scope':'UNADMITTED-original-DTOS-NOT-qualified-Source'},budget_loan=loan,reservation=8*1024**2)
        loan._bind_source(source)
        source._original_contract=_Final64OriginalContract(parent,b'{}',b'[]',b'{}',actions,settings(),
            canonical({'seed':17,'final_binding':{'label':label}}).encode())
        encoder=TreeEncoderConfigV3(aggregate_window=7)
        with source.borrow_original_selected_episode(1,budget_loan=loan) as borrow:
            iterator=borrow.iter_decisions();decision=next(iterator)
            tokens,candidates=_tree_tokens(decision,encoder)
            assert tokens.aggregate_window==7 and tokens.recipe_signature==signature(7,layout=LAYOUT)
            assert tokens==tokenize_frame(decision.frame,decision.bank,aggregate_window=7,layout=LAYOUT)
            assert candidates==tokenize_table(decision.table,decision.frame,18,aggregate_window=7,layout=LAYOUT)
            assert tokens.recipe_signature!=signature(2,layout=LAYOUT)
            assert decision.selection==records[1].selections[0]
            assert list(iterator)==[]
            del decision,tokens,candidates
        assert borrow.cleanup_complete
        with source.borrow_original_selected_episode(0,budget_loan=loan) as empty:
            assert empty.decision_count==0 and empty.end_kind=='natural' and list(empty.iter_decisions())==[]
        source.close()
        assert source.cleanup_complete and loan.cleanup_complete and parent._external_bytes==65536
        parent.__exit__(None,None,None)
    assert_retired(parents,validations)


def test_actual_borrow_cancellation_retires_unit_before_source_and_keeps_failed_parent_base(original_records,tmp_path):
    records,actions,label=original_records
    parents,validations=roots(tmp_path)
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        parent=primitive_parent(keeper,parents[0],512*1024**2)
        loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=65536,deadline=keeper.deadline)
        loan.admit_local(64*1024**2)
        source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),records,
            {'scope':'UNADMITTED-cancellation-resource-NOT-Source-admission'},budget_loan=loan,reservation=8*1024**2)
        loan._bind_source(source)
        source._original_contract=_Final64OriginalContract(parent,b'{}',b'[]',b'{}',actions,settings(),
            canonical({'seed':17,'final_binding':{'label':label}}).encode())
        seen=[]
        async def run():
            entered=asyncio.Event()
            async def body():
                with source.borrow_original_selected_episode(1,budget_loan=loan) as borrow:
                    seen.append(borrow)
                    entered.set();await asyncio.Event().wait()
            task=asyncio.create_task(body());await entered.wait()
            task.cancel('original unit cancellation')
            with pytest.raises(asyncio.CancelledError,match='original unit cancellation') as captured:await task
            return captured.value
        primary=asyncio.run(run())
        assert seen[0].cleanup_complete and parent._unit_borrow is None
        before=parent._external_bytes
        with pytest.raises(asyncio.CancelledError) as closed:source.close()
        assert closed.value is primary and source.cleanup_complete and loan.cleanup_complete
        assert parent._external_bytes==before>65536  # failed parent never gains a new nonzero resize
        parent.__exit__(type(primary),primary,primary.__traceback__)
        assert parent.cleanup_complete
    assert_retired(parents,validations)


def test_caller_fact_clone_pre_admission_refuses_before_actual_source_getter(tmp_path,monkeypatch):
    parents,validations=roots(tmp_path)
    total=64*1024**2
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as keeper:
        parent=primitive_parent(keeper,parents[0],total)
        # Protected byte LENGTH only, intentionally not valid Config/admission.
        parent._state_bytes=b' '*1024**2;parent._records_bytes=b'[]'
        parent.scan.retained+=len(parent._state_bytes)
        budget=GameTreeBudget(total,dataset_bytes=65536,child_limit_bytes=32*1024**2)
        budget.parent_intrinsic(0,parent.resident_bytes)
        loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=budget.external_base(0),deadline=keeper.deadline)
        loan.admit_local(65536)
        physical=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),(),
            {'scope':'UNADMITTED-copy-reservation-NOT-Source-admission'},budget_loan=loan,reservation=65536)
        loan._bind_source(physical)
        manager=object.__new__(_GameTreeOriginals)
        manager._custody=keeper;manager._parents=[parent];manager._sources=[physical];manager._loans=[loan]
        manager._budget=budget;manager._scan=SealedScan(max_owned_bytes=total,deadline=keeper.deadline)
        manager.limit=total
        manager._active=True;manager._closed=False;manager._failure=None
        manager._persistent=manager._controls=65536;manager._read_peak=manager._extra=0
        manager._sync()
        budget.reserve_sink(total-budget.retained_bytes-2*1024**2)
        manager._sync()
        def forbidden(self):pytest.fail('fact clone reached actual Source getter after caller reservation exhausted')
        monkeypatch.setattr(GameFinalValidationSource,'facts',property(forbidden))
        with pytest.raises(ContractError,match='joint8GiB'):manager._facts(0)
        physical.close();parent.__exit__(None,None,None)
        assert physical.cleanup_complete and loan.cleanup_complete and parent.cleanup_complete
    assert_retired(parents,validations)


def pre_source_resource_owner(tmp_path,*,max_owned_bytes=64*1024**2):
    parents,validations=roots(tmp_path)
    owner=_GameTreeOriginals(parents,validations,settings(),encoder=TreeEncoderConfigV3(),
        max_owned_bytes=max_owned_bytes,deadline=time.monotonic()+60)
    owner._custody.__enter__();owner._entered=True
    parent=primitive_parent(owner._custody,parents[0],owner.limit)
    owner._parents.append(parent)
    return owner,parent,parents,validations


@pytest.mark.parametrize('failure_point',['after-unbound-loan','loan-admission'])
def test_pre_source_failure_actual_parent_retires_unbound_loan_before_all_custody_releases(tmp_path,monkeypatch,failure_point):
    owner,parent,parents,validations=pre_source_resource_owner(tmp_path)
    exits=[];actual_exit=FormalGameParent.__exit__
    def observed_exit(self,*args):
        exits.append(self)
        return actual_exit(self,*args)
    monkeypatch.setattr(FormalGameParent,'__exit__',observed_exit)
    if failure_point=='after-unbound-loan':
        loan=FinalSourceBudgetLoan(parent,custodian=owner._custody,base_bytes=131072,deadline=parent.deadline)
        owner._loans.append(loan)
        primary=ContractError('UNADMITTED wrong ordered seed before Source construction')
    else:
        with pytest.raises(ContractError) as captured:
            FinalSourceBudgetLoan(parent,custodian=owner._custody,base_bytes=parent.limit-1,deadline=parent.deadline)
        primary=captured.value
        loan=parent._resource_loan  # actual constructor attached its cleanup owner before admission
    owner._failure=primary
    assert not owner._sources and not loan.cleanup_complete
    try:
        with pytest.raises(ContractError) as closed:owner.close()
        assert closed.value is primary
        assert parent.cleanup_complete and loan.cleanup_complete and owner.cleanup_complete
        assert exits==[parent]
        with pytest.raises(ContractError):owner._live()
        owner.close()
        assert exits==[parent]  # no consumed parent exit/empty stack retry
        assert_retired(parents,validations)
    finally:
        # Also permits a red run to release the actual temporary resource locks.
        if not parent.cleanup_complete:actual_exit(parent,type(primary),primary,None)
        if not owner.cleanup_complete:
            try:owner.close()
            except ContractError:pass


def test_uncertain_live_source_stays_retained_before_parent_and_consumed_scope_cannot_confirm_cleanup(original_records,tmp_path,monkeypatch):
    records,actions,label=original_records
    owner,parent,parents,validations=pre_source_resource_owner(tmp_path,max_owned_bytes=512*1024**2)
    loan=FinalSourceBudgetLoan(parent,custodian=owner._custody,base_bytes=65536,deadline=parent.deadline)
    loan.admit_local(64*1024**2);owner._loans.append(loan)
    source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),records,
        {'scope':'UNADMITTED-live-unit-NOT-Source-admission'},budget_loan=loan,reservation=8*1024**2)
    loan._bind_source(source);owner._sources.append(source)
    source._original_contract=_Final64OriginalContract(parent,b'{}',b'[]',b'{}',actions,settings(),
        canonical({'seed':17,'final_binding':{'label':label}}).encode())
    primary=RuntimeError('UNADMITTED initiating cleanup failure')
    owner._failure=primary
    attempts=[];exits=[];actual_exit=FormalGameParent.__exit__
    def close_source():attempts.append(source);source.close()
    def observed_exit(self,*args):exits.append(self);return actual_exit(self,*args)
    monkeypatch.setattr(FormalGameParent,'__exit__',observed_exit)
    scope=ExitStack();scope.callback(close_source)  # cleanup ONLY, never a Source-permission callback
    owner._source_scopes.append(scope)
    with source.borrow_original_selected_episode(0,budget_loan=loan) as borrow:
        for _ in range(2):
            with pytest.raises(RuntimeError) as captured:owner.close()
            assert captured.value is primary and captured.value._game_tree_dataset_cleanup_owner is owner
            assert not owner.cleanup_complete and not source.cleanup_complete and not parent.cleanup_complete
            assert len(owner._custody._entries)==6 and all(path in _OWNED_LOCKS for path in parents)
        assert attempts==[source] and not exits and not scope._exit_callbacks
    assert borrow.cleanup_complete
    # Retire the ACTUAL still-reachable Source; an empty scope is not evidence.
    source.close()
    with pytest.raises(RuntimeError) as captured:owner.close()
    assert captured.value is primary and owner.cleanup_complete and loan.cleanup_complete
    assert exits==[parent] and attempts==[source]
    assert_retired(parents,validations)
