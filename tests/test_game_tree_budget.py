"""Logical owned-byte accounting, never Source/fit permission or RSS evidence."""
import pytest

from gf_common.domain import ContractError
from gf_training.card_instance_game_tree_data import GameTreeBudget


def test_all_three_owners_and_full_child_share_one_allowance():
    budget=GameTreeBudget(8*1024**3,dataset_bytes=1024,child_limit_bytes=2*1024**3)
    for i in range(3):budget.parent_intrinsic(i,1024**3)
    for i in range(3):budget.source_intrinsic(i,128*1024**2)
    assert budget.retained_bytes==5*1024**3+3*128*1024**2+1024
    assert budget.external_base(1)==budget.retained_bytes-1024**3-128*1024**2
    plan=budget.fit_memory()
    assert plan['reserved_peak_bytes']==budget.retained_bytes
    assert plan['child_limit_bytes']==2*1024**3
    with pytest.raises(ContractError):budget.reserve_sink(4*1024**3)
    with pytest.raises(ContractError):budget.reserve_sink(1)


def test_source_retirement_cannot_erase_other_parent_or_sink_base():
    budget=GameTreeBudget(8*1024**3)
    for i in range(3):budget.parent_intrinsic(i,100)
    budget.source_intrinsic(0,80);budget.reserve_sink(200)
    base=budget.external_base(0)
    with pytest.raises(ContractError):budget.source_intrinsic(0,0)
    budget.source_intrinsic(0,0,retired=True)
    assert budget.external_base(0)==base==400
    assert budget.retained_bytes==500
