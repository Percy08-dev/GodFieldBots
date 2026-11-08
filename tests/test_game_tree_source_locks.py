"""Actual temporary resource locks only; these files are NOT original Sources."""
import os
import time

import pytest

from gf_common.domain import ContractError
from gf_training.card_instance_source_locks import GameSourceCustody, ValidationManifestOwnership
from gf_training.lineage import _OWNED_LOCKS


def roots(tmp_path):
    parents=tuple(tmp_path/f'p{i}' for i in range(3));validations=tuple(tmp_path/f'v{i}' for i in range(3))
    for path in (*parents,*validations):path.mkdir()
    for path in validations:(path/'manifest.json').write_text('{"scope":"UNADMITTED-lock-only"}')
    return parents,validations


def test_readonly_flock_preserves_inventory_and_real_inode_identity(tmp_path):
    _,validations=roots(tmp_path);path=validations[0]
    before=(path/'manifest.json').read_bytes()
    with ValidationManifestOwnership(path,deadline=time.monotonic()+60) as owner:
        info=(path/'manifest.json').stat()
        assert owner.inode==(info.st_dev,info.st_ino)
        with pytest.raises((BlockingIOError,ContractError)), ValidationManifestOwnership(path,deadline=time.monotonic()+60):
            pytest.fail('independent open stole exclusive flock')
        assert {p.name for p in path.iterdir()}=={'manifest.json'}
    assert owner.cleanup_complete and (path/'manifest.json').read_bytes()==before
    with pytest.raises(ContractError):owner.require(path)


def test_whole_ordered_set_borrow_keeps_actual_tokens_and_outer_locks(tmp_path):
    parents,validations=roots(tmp_path)
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as owner:
        with GameSourceCustody(parents,validations,deadline=owner.deadline,ownership=owner) as borrowed:
            assert borrowed.ownership_for(validations[1]) is owner.ownership_for(validations[1])
            assert borrowed.ownership_for(parents[1]) is owner.ownership_for(parents[1])
        assert all(path in _OWNED_LOCKS for path in parents)
    assert owner.cleanup_complete and all(path not in _OWNED_LOCKS for path in parents)
    assert all(not (path/'.lock').exists() for path in validations)


def test_same_bytes_manifest_replacement_poison_cannot_repair_custody(tmp_path):
    _,validations=roots(tmp_path);path=validations[0]
    owner=ValidationManifestOwnership(path,deadline=time.monotonic()+60).__enter__()
    replacement=path/'replacement';replacement.write_bytes((path/'manifest.json').read_bytes())
    os.replace(replacement,path/'manifest.json')
    with pytest.raises(ContractError):owner.require(path)
    with pytest.raises(ContractError):owner.close()
    assert owner.cleanup_complete


def test_borrowed_custody_deadline_cannot_extend_its_own_loan(tmp_path):
    parents,validations=roots(tmp_path)
    with GameSourceCustody(parents,validations,deadline=time.monotonic()+60) as outer:
        borrowed=GameSourceCustody(parents,validations,deadline=time.monotonic()-1,ownership=outer).__enter__()
        with pytest.raises(TimeoutError):borrowed.require()
        borrowed.close()
        outer.require()


def test_manifest_change_reports_failure_but_confirmed_closes_do_not_become_uncertain(tmp_path):
    parents,validations=roots(tmp_path)
    owner=GameSourceCustody(parents,validations,deadline=time.monotonic()+60).__enter__()
    path=validations[0]/'manifest.json'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ContractError):owner.require()
    with pytest.raises(ContractError):owner.close()
    assert owner.cleanup_complete and not owner._uncertain
