"""Every HPO writer/retirer requires the original current-thread flock owner."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_models.arrays import atomic_json, sha256
from gf_training.hpo_lineage import commit_hpo_generation, write_hpo_control
from gf_training.hpo_retention import _transition, retain_hpo_generations
from gf_training.lineage import _OWNED_LOCKS, RunOwnership, generation_path, run_lock
from gf_training.structured_lineage import _ChainScan
from tests.hpo_lineage_support import advanced_hpo_chain, initial_hpo_chain


def rewrite(root,*,ownership=None):
    with _ChainScan(directory=root,max_owned_bytes=1024**2,deadline=time.monotonic()+30) as scan:
        scan.object(root/'run.json')
        options={} if ownership is None else {'ownership':ownership}
        return write_hpo_control(root/'run.json',{'sentinel':'after'},scan=scan,**options)


def test_unowned_publication_cannot_replace_run_with_fresh_scan(tmp_path):
    root=tmp_path/'parent'
    root.mkdir()
    atomic_json(root/'run.json',{'sentinel':'before'})
    before=(root/'run.json').read_bytes()
    with pytest.raises(ContractError): rewrite(root)
    assert (root/'run.json').read_bytes()==before


def test_cross_thread_publication_refuses_before_replacement(tmp_path):
    root=tmp_path/'parent'
    root.mkdir()
    atomic_json(root/'run.json',{'sentinel':'before'})
    before=(root/'run.json').read_bytes()
    with (run_lock(root),ThreadPoolExecutor(max_workers=1) as executor,pytest.raises(ContractError)):
        executor.submit(rewrite,root).result()
    assert (root/'run.json').read_bytes()==before


@pytest.mark.parametrize('mode',['cross-thread','forged-process-map'])
def test_retirement_requires_actual_owner_before_any_unlink(tmp_path,monkeypatch,mode):
    root=tmp_path/'parent'
    advanced_hpo_chain(root,updates=4,budget=12)
    target=root/'artifacts'/'000001'/'resume.json'
    before=target.read_bytes()
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+60}
    if mode=='cross-thread':
        with (run_lock(root),ThreadPoolExecutor(max_workers=1) as executor,pytest.raises(ContractError)):
            executor.submit(retain_hpo_generations,root,4,**options).result()
    else:
        monkeypatch.setitem(_OWNED_LOCKS,root,'not-an-actual-lock')
        with pytest.raises(ContractError): retain_hpo_generations(root,4,**options)
    assert target.read_bytes()==before
    assert not list((root/'artifacts').glob('prune-*'))


@pytest.mark.parametrize('mode',['forged-equal','cross-thread','stale','foreign-root'])
def test_borrowed_reference_transaction_refuses_invalid_owner(tmp_path,monkeypatch,mode):
    root=tmp_path/'parent'
    state,records,_=advanced_hpo_chain(root)
    binding={'run_id':state['run_id'],'token':'ref','generation':1,
        'inference_sha256':records[1]['inference_sha256'],'generation_sha256':sha256(generation_path(root,1))}
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+30}
    before=(root/'artifact-registry.json').read_bytes()
    with run_lock(root) as original:
        if mode=='forged-equal':
            forged=RunOwnership(original.directory,original.token)
            with pytest.raises(ContractError): _transition(root,binding,'acquire',ownership=forged,**options)
        elif mode=='cross-thread':
            with ThreadPoolExecutor(max_workers=1) as executor,pytest.raises(ContractError):
                executor.submit(_transition,root,binding,'acquire',ownership=original,**options).result()
        elif mode=='foreign-root':
            foreign=tmp_path/'foreign'
            foreign.mkdir()
            with run_lock(foreign) as owner,pytest.raises(ContractError):
                _transition(root,binding,'acquire',ownership=owner,**options)
    if mode=='stale':
        monkeypatch.setitem(_OWNED_LOCKS,root,original.token)
        with pytest.raises(ContractError): _transition(root,binding,'acquire',ownership=original,**options)
    assert (root/'artifact-registry.json').read_bytes()==before
    assert not (root/'evaluation-reference-receipts').exists()


def test_same_original_owner_can_publish_and_equal_field_forgery_cannot(tmp_path):
    root=tmp_path/'parent'
    root.mkdir()
    atomic_json(root/'run.json',{'sentinel':'before'})
    with run_lock(root) as owner:
        with pytest.raises(ContractError): rewrite(root,ownership=RunOwnership(owner.directory,owner.token))
        rewrite(root,ownership=owner)


def test_g0_cross_thread_commit_refuses_before_publishing(tmp_path):
    root=tmp_path/'parent'
    state,records,_=initial_hpo_chain(root)
    record=records[0]
    generation_path(root,0).unlink()
    (root/'latest.json').unlink()
    state['generation_sha256']=None
    atomic_json(root/'run.json',state)
    atomic_json(root/'artifact-registry.json',{'schema_version':4,'generations':{},'evaluation_references':{}})
    with run_lock(root),ThreadPoolExecutor(max_workers=1) as executor:
        call=executor.submit(commit_hpo_generation,root,state,generation=0,update=None,
            controls=record['parent_controls'],artifact_files=record['artifact_files'],
            max_owned_bytes=1024**3,deadline=time.monotonic()+30)
        with pytest.raises(ContractError): call.result()
    assert not generation_path(root,0).exists() and not (root/'latest.json').exists()
