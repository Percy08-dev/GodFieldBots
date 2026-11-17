"""Actual custody/resource primitives; NO formal-parent admission or fit authority."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, RuntimePolicyBinding, canonical, digest
from gf_training.card_instance_formal import FinalSourceBudgetLoan, FormalGameParent
from gf_training.card_instance_source_locks import GameSourceCustody


@pytest.fixture
def custody(tmp_path):
    parents=tuple(tmp_path/f'UNADMITTED-parent-{i}' for i in range(3))
    validations=tuple(tmp_path/f'UNADMITTED-validation-{i}' for i in range(3))
    for path in (*parents,*validations):
        path.mkdir()
    for path in validations:
        (path/'manifest.json').write_bytes(b'{"scope":"UNADMITTED-custody-only"}')
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as owner:
        yield owner,parents,validations


def primitive_parent(custody,monkeypatch,*,extra=0):
    owner,parents,_=custody
    parent=FormalGameParent(parents[0],max_owned_bytes=16*1024**2,deadline=time.monotonic()+60,
        ownership=owner.ownership_for(parents[0]),extra_owned_bytes=extra)
    parent._entered=parent._active=True
    parent._token=parent.borrowed.token
    parent.scan=SimpleNamespace(retained=1024,files={},inventories={})
    parent.recheck=parent.require_active
    assert not (parents[0]/'run.json').exists()
    return parent


def test_preentry_extra_reaches_scan_before_first_decode_and_never_enters_native(custody,monkeypatch):
    import gf_training.structured_lineage as lineage
    owner,parents,_=custody
    observed=[]
    class Stop(RuntimeError): pass
    def read(scan,path,*args):
        observed.append(scan.limit)
        raise Stop('UNADMITTED before run decoding')
    monkeypatch.setattr(lineage._ChainScan,'object',read)
    parent=FormalGameParent(parents[0],max_owned_bytes=1024**2,extra_owned_bytes=12345,
        deadline=time.monotonic()+60,ownership=owner.ownership_for(parents[0]))
    with pytest.raises(Stop):
        parent.__enter__()
    assert observed==[1024**2-12345] and parent._native_inputs is None
    assert parent.cleanup_complete
    assert owner.ownership_for(parents[0]) is parent.borrowed  # borrowed lock still held


def test_native_initial_external_is_counted_once_and_actual_floor_is_not_intrinsic(custody,monkeypatch):
    parent=primitive_parent(custody,monkeypatch,extra=100)
    parent._native_scan_base=1024
    parent.scan.retained=1064
    calls=[]
    class Native:
        amount=0
        fail=False
        def caller_growth(self,n):
            self.pending=n
        def recheck(self):
            if self.fail: raise OSError('UNADMITTED failed actual lease update')
            self.amount=self.pending
            calls.append(self.amount)
        @property
        def joint_owned_bytes(self):
            return 500+100+self.amount
    native=Native();parent._native_inputs=native
    parent.admit_external(300)
    assert calls[-1]==240 and parent._declared_external_bytes==300 and parent.resident_bytes==540
    parent.admit_external(50)
    assert calls[-1]==40 and parent._declared_external_bytes==100 and parent.resident_bytes==540
    before=len(calls);parent.admit_external(0)
    assert len(calls)==before and parent._declared_external_bytes==100
    native.fail=True
    with pytest.raises(OSError):parent.admit_external(400)
    assert parent._declared_external_bytes==100 and parent._poisoned
    with pytest.raises(ContractError):parent.admit_external(1)
    parent.admit_external(0)  # cleanup only; actual native floor is not fabricated zero
    assert parent._declared_external_bytes==100


def test_concrete_loan_preserves_base_highwater_and_releases_only_physically_retired_source(custody,monkeypatch):
    from gf_training.card_instance_final_validation import GameFinalValidationSource
    parent=primitive_parent(custody,monkeypatch)
    owner,_,_=custody
    loan=FinalSourceBudgetLoan(parent,custodian=owner,base_bytes=4096,deadline=parent.deadline)
    loan.admit_local(2048);loan.admit_local(1)
    assert loan.source_bytes==2048 and parent._external_bytes==6144
    loan.update_base(8192,custodian=owner)
    assert loan.source_bytes==2048 and parent._external_bytes==10240
    with pytest.raises(ContractError):loan.update_base(0,custodian=object())
    source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),(),
        {'scope':'UNADMITTED-resource-only'},budget_loan=loan)
    loan._bind_source(source)
    with pytest.raises(ContractError):loan.retire_local(source)
    source.close()
    assert source.cleanup_complete and loan.cleanup_complete and parent._external_bytes==8192
    source.close();loan.retire_local(source)
    assert parent._external_bytes==8192
    with pytest.raises(ContractError):loan.admit_local(1)


def test_failed_parent_retirement_keeps_declared_reservation_and_expired_loan_has_no_permission(custody,monkeypatch):
    from gf_training.card_instance_final_validation import GameFinalValidationSource
    parent=primitive_parent(custody,monkeypatch)
    owner,_,_=custody
    loan=FinalSourceBudgetLoan(parent,custodian=owner,base_bytes=4096,deadline=parent.deadline)
    loan.admit_local(2048)
    source=GameFinalValidationSource(parent,SimpleNamespace(recheck=lambda:None),(),
        {'scope':'UNADMITTED-resource-only'},budget_loan=loan)
    loan._bind_source(source)
    primary=RuntimeError('UNADMITTED initiating failure')
    parent._poison(primary);source._failure=primary
    with pytest.raises(RuntimeError):source.close()
    assert source.cleanup_complete and loan.cleanup_complete
    assert parent._external_bytes==6144  # no forbidden nonzero failed-session update
    with pytest.raises(ContractError):loan.require(parent)
    parent.admit_external(0)
    assert parent._external_bytes==0


@pytest.mark.parametrize('extra',[True,-1,1024])
def test_invalid_preentry_reservation_refuses_without_read_or_lock(tmp_path,extra):
    with pytest.raises(ContractError):
        FormalGameParent(tmp_path/'absent',max_owned_bytes=1024,extra_owned_bytes=extra,deadline=time.monotonic()+60)
    assert not tuple(tmp_path.iterdir())


def test_parent_derived_contract_requires_full_original_settings_without_cli_defaults(custody,monkeypatch):
    from pathlib import Path

    from gf_common.card_instance import action_config_v3
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_final_validation import _original_contract
    root=Path(__file__).parents[1]
    example=json.loads((root/'native/tests/fixtures/card_instance_models3_rules_v2.json').read_bytes())['cases'][0]
    parent=primitive_parent(custody,monkeypatch)
    config={'actions':example['actions'],'preprocessing':asdict(PreparationConfig()),'run':{'split':'train'}}
    state={'scope':'UNADMITTED-control-parse-NOT-parent-admission','effective_config':config,
        'config_sha256':digest(config),'config_inputs':[],'run_id':'UNADMITTED','seed':17,'updates':1,
        'identity':{'code':{},'integration':{}}}
    parent._state_bytes=canonical(state).encode();parent._records_bytes=b'[]'
    parent.final_binding=lambda: ('UNADMITTED',RuntimePolicyBinding('mlp',{'expected_manifest_digest':'a'*64}))
    owner,_,_=custody
    loan=FinalSourceBudgetLoan(parent,custodian=owner,base_bytes=4096,deadline=parent.deadline)
    contract=_original_contract(parent,budget_loan=loan)
    assert contract.actions==action_config_v3(example['actions'])
    assert json.loads(contract.projected_effective_bytes)=={**config,'run':{'split':'validation'}}
    assert json.loads(contract.input_origin_bytes)==[]
    del config['preprocessing']['chunk_target_mib']
    state['config_sha256']=digest(config);parent._state_bytes=canonical(state).encode()
    with pytest.raises(ContractError):_original_contract(parent,budget_loan=loan)
    loan._retire_failed_opening()
    assert loan.cleanup_complete and parent._external_bytes>=4096
