"""Real DTOs under UNADMITTED custody primitives; no formal Source upgrade."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, canonical
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_training.card_instance_final_validation import (
    GameFinalValidationSource,
    _Final64OriginalContract,
)
from gf_training.card_instance_formal import FinalSourceBudgetLoan
from gf_training.card_instance_source import OriginalGameEpisodeBorrow
from gf_training.structured_rollout import _owned_bytes
from tests.test_native_card_instance_final_validation import originals as reference_originals
from tests.test_native_card_instance_final_validation import packet_from_reference
from tests.test_original_source_resources import custody as resource_custody
from tests.test_original_source_resources import primitive_parent


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    return reference_originals.__wrapped__(tmp_path_factory)


@pytest.fixture
def custody(tmp_path):
    yield from resource_custody.__wrapped__(tmp_path)


def test_cached_original_final_borrow_reprojects_arbitrary_window_without_reexport_and_invalidates_pending_iteration(originals,custody,monkeypatch):
    parent=primitive_parent(custody,monkeypatch);parent.limit=1024**3
    keeper,_,_=custody
    loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=4096,deadline=parent.deadline)
    loan.admit_local(128*1024**2)  # entire controlled DTO construction before copies
    records=tuple(packet_from_reference(originals,i)[1]for i in range(64))
    source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),records,
        {'scope':'UNADMITTED-original-cache-NOT-Source-admission'},budget_loan=loan,
        reservation=_owned_bytes(records)+65536)
    loan._bind_source(source)
    label=originals[2]['linkage']['policy_label']
    source._original_contract=_Final64OriginalContract(parent,b'{}',b'[]',b'{}',originals[3],None,
        canonical({'seed':17,'final_binding':{'label':label}}).encode())
    with pytest.raises(ContractError):OriginalGameEpisodeBorrow(parent,source,loan,record=records[1],episode_id='forged')
    with source.borrow_original_selected_episode(1,budget_loan=loan)as borrow:
        assert borrow.episode_id==originals[1]['oracle'][1]['episode_id']
        with pytest.raises(AttributeError):
            borrow.episode_id='renamed'
        assert borrow.records is records[1] and borrow.export_bytes==0
        assert borrow.resident_bytes==1024  # shared original cache is not charged twice
        pending=borrow.iter_decisions()
        with pytest.raises(ContractError):borrow.iter_decisions()
        decision=next(pending)
        expected_frame=tokenize_frame(decision.frame,decision.bank,aggregate_window=7,layout=LAYOUT)
        candidates=tokenize_table(decision.table,decision.frame,18,aggregate_window=7,layout=LAYOUT)
        assert expected_frame.aggregate_window==7 and candidates.table_hash==decision.table.table_hash
        assert decision.selection.index==records[1].selections[0].index
        with pytest.raises(ContractError):source.close()
        with pytest.raises(ContractError), source.borrow_original_selected_episode(2,budget_loan=loan):
            pass
    assert borrow.cleanup_complete
    assert list(pending)==[]
    with pytest.raises(ContractError):_ = borrow.records
    with pytest.raises(ContractError):_ = borrow.episode_id
    with source.borrow_original_selected_episode(0,budget_loan=loan)as empty:
        assert empty.episode_id==originals[1]['oracle'][0]['episode_id']
        with pytest.raises(AttributeError):
            empty.episode_id='renamed-zero'
        assert empty.decision_count==0 and list(empty.iter_decisions())==[] and empty.end_kind=='natural'
        empty.close()
        with pytest.raises(ContractError):_ = empty.episode_id
    with pytest.raises(ContractError):_ = empty.episode_id
    source.close()
    assert source.cleanup_complete and loan.cleanup_complete and parent._external_bytes==4096


@pytest.mark.parametrize('corruption',['bank','digest','source'])
def test_episode_identity_rechecks_actual_bank_digest_and_source_failure(originals,custody,monkeypatch,corruption):
    from dataclasses import replace

    from gf_common.card_instance_rules import card_instance_rules
    from gf_training.structured_rollout import game_episode_records
    parent=primitive_parent(custody,monkeypatch);parent.limit=1024**3
    keeper,_,_=custody
    loan=FinalSourceBudgetLoan(parent,custodian=keeper,base_bytes=4096,deadline=parent.deadline)
    loan.admit_local(128*1024**2)
    records=tuple(packet_from_reference(originals,i)[1]for i in range(64))
    source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),records,
        {'scope':'UNADMITTED-identity-guard-NOT-formal-Source'},budget_loan=loan,
        reservation=_owned_bytes(records)+65536)
    loan._bind_source(source)
    label=originals[2]['linkage']['policy_label']
    source._original_contract=_Final64OriginalContract(parent,b'{}',b'[]',b'{}',originals[3],None,
        canonical({'seed':17,'final_binding':{'label':label}}).encode())
    try:
        with pytest.raises(ContractError), source.borrow_original_selected_episode(1,budget_loan=loan)as borrow:
            assert borrow.episode_id==originals[1]['oracle'][1]['episode_id']
            if corruption=='bank':
                borrow._record=records[2]  # controlled private corruption, never an accepted Source edit
            elif corruption=='digest':
                original=records[1]
                changed=replace(original.selections[0],value=original.selections[0].value+0.25)
                borrow._record=game_episode_records(original.bank,original.frames,original.tables,(changed,),
                    original.end_kind,original.outcome,actions=originals[3],rules=card_instance_rules())
            else:
                source._failure=RuntimeError('controlled actual-source lifetime failure')
            _ = borrow.episode_id
        with pytest.raises(ContractError):_ = borrow.episode_id
        with pytest.raises(ContractError):_ = borrow.records
    finally:
        try:
            source.close()
        except (ContractError,RuntimeError):
            pass
        assert source.cleanup_complete and loan.cleanup_complete
