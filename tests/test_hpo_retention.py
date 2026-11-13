"""Owned per-file retirement and concrete evaluation-reference transactions."""
import time

import pytest

pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_models.sharded_arrays import read_json
from gf_training.lineage import run_lock
from tests.hpo_lineage_support import advanced_hpo_chain


def test_pin_release_exact_replay_and_terminal_token(tmp_path):
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    root = tmp_path/'parent'
    _, records, _ = advanced_hpo_chain(root)
    options = {'max_owned_bytes': 1024**3, 'deadline': time.monotonic()+30}
    pin = pin_hpo_evaluation(root, 'evaluation-1', 1, records[1]['inference_sha256'], **options)
    assert pin == pin_hpo_evaluation(root, 'evaluation-1', 1, records[1]['inference_sha256'], **options)
    released = release_hpo_evaluation(root, pin.binding, **options)
    assert released == release_hpo_evaluation(root, pin.binding, **options)
    assert read_json(root/'artifact-registry.json')['evaluation_references'] == {}
    assert {p.name for p in (root/'evaluation-reference-receipts'/'evaluation-1').iterdir()} == {
        'acquire.intent.json','acquire.receipt.json','release.intent.json','release.receipt.json'}
    with pytest.raises(ContractError):
        pin_hpo_evaluation(root, 'evaluation-1', 1, records[1]['inference_sha256'], **options)


def test_retention_prunes_only_unprotected_files_and_preserves_pin(tmp_path):
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_retention import pin_hpo_evaluation, retain_hpo_generations
    root = tmp_path/'parent'
    _, records, _ = advanced_hpo_chain(root, updates=4, budget=12)
    options = {'max_owned_bytes': 1024**3, 'deadline': time.monotonic()+60}
    pin_hpo_evaluation(root, 'evaluation-1', 1, records[1]['inference_sha256'], **options)
    with run_lock(root): retain_hpo_generations(root, 4, **options)
    registry = read_json(root/'artifact-registry.json')
    assert registry['generations']['1']['files']['inference.json']['status'] == 'retained'
    assert registry['generations']['1']['files']['resume.json']['status'] == 'deleted'
    assert all(v['status']=='retained' for g in ('0','3','4') for v in registry['generations'][g]['files'].values())
    assert all(v['status']=='deleted' for v in registry['generations']['2']['files'].values())
    validate_hpo_chain(root, require_complete=True, **options)


@pytest.mark.parametrize('stage', ['after_intent','after_registry'])
def test_pin_crash_replays_only_original_transaction(tmp_path, monkeypatch, stage):
    import gf_training.hpo_retention as retention
    root=tmp_path/'parent'
    _, records, _=advanced_hpo_chain(root)
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+30}
    original=retention._write_control
    def fault(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if (stage=='after_intent' and path.name=='acquire.intent.json'
                or stage=='after_registry' and path.name=='artifact-registry.json'):
            raise OSError('injected transaction crash')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(retention,'_write_control',fault)
        with pytest.raises(OSError): retention.pin_hpo_evaluation(root,'evaluation-1',1,records[1]['inference_sha256'],**options)
    result=retention.pin_hpo_evaluation(root,'evaluation-1',1,records[1]['inference_sha256'],**options)
    assert result.binding['generation']==1


def test_release_without_matching_journal_refuses(tmp_path):
    from gf_training.hpo_retention import release_hpo_evaluation
    root=tmp_path/'parent'
    state, records, _=advanced_hpo_chain(root)
    binding={'run_id':state['run_id'],'token':'absent','generation':1,
        'inference_sha256':records[1]['inference_sha256'],'generation_sha256':'f'*64}
    with pytest.raises(ContractError):
        release_hpo_evaluation(root,binding,max_owned_bytes=1024**3,deadline=time.monotonic()+30)


def test_receipt_hash_is_historical_despite_unrelated_reference_mutations(tmp_path):
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    root=tmp_path/'parent'
    _,records,_=advanced_hpo_chain(root)
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+30}
    original=pin_hpo_evaluation(root,'one',1,records[1]['inference_sha256'],**options)
    other=pin_hpo_evaluation(root,'two',0,records[0]['inference_sha256'],**options)
    assert pin_hpo_evaluation(root,'one',1,records[1]['inference_sha256'],**options)==original
    release=release_hpo_evaluation(root,original.binding,**options)
    release_hpo_evaluation(root,other.binding,**options)
    assert release_hpo_evaluation(root,original.binding,**options)==release


def test_release_crash_after_registry_removal_reconciles_matching_intent(tmp_path,monkeypatch):
    import gf_training.hpo_retention as retention
    root=tmp_path/'parent'
    _,records,_=advanced_hpo_chain(root)
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+30}
    pin=retention.pin_hpo_evaluation(root,'one',1,records[1]['inference_sha256'],**options)
    original=retention._write_control
    def fault(path,*args,**kwargs):
        if path.name=='release.receipt.json': raise OSError('after removal before receipt')
        return original(path,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(retention,'_write_control',fault)
        with pytest.raises(OSError): retention.release_hpo_evaluation(root,pin.binding,**options)
    assert read_json(root/'artifact-registry.json')['evaluation_references']=={}
    assert retention.release_hpo_evaluation(root,pin.binding,**options).phase=='release'


@pytest.mark.parametrize('fault_kind',['replacement','completion_write'])
def test_delete_fault_preserves_uncertain_receipts_and_foreign_objects(tmp_path,monkeypatch,fault_kind):
    import gf_training.hpo_retention as retention
    from gf_training.hpo_lineage import validate_hpo_chain
    root=tmp_path/'parent'
    advanced_hpo_chain(root,updates=4,budget=12)
    options={'max_owned_bytes':1024**3,'deadline':time.monotonic()+60}
    foreign=[]
    if fault_kind=='replacement':
        original=retention._delete_staged_file
        def fault(parent,parent_fd,identity,path,*args,**kwargs):
            name=args[3]
            target=path/name
            target.unlink()
            target.write_bytes(b'foreign replacement')
            foreign.append(target)
            return original(parent,parent_fd,identity,path,*args,**kwargs)
        monkeypatch.setattr(retention,'_delete_staged_file',fault)
    else:
        original=retention._write_control
        def fault(path,value,*args,**kwargs):
            if path.name=='artifact-registry.json' and any(e['status']=='deleted' for g in value['generations'].values() for e in g['files'].values()):
                raise OSError('completion publication failed')
            return original(path,value,*args,**kwargs)
        monkeypatch.setattr(retention,'_write_control',fault)
    with run_lock(root),pytest.raises((ContractError,OSError)):
        retention.retain_hpo_generations(root,4,**options)
    for path in foreign: assert path.read_bytes()==b'foreign replacement'
    registry=read_json(root/'artifact-registry.json')
    assert any(e['status']=='deleting' and e['deletion_receipt'] is None for g in registry['generations'].values() for e in g['files'].values())
    with pytest.raises(ContractError): validate_hpo_chain(root,**options)


def test_another_thread_cannot_borrow_the_training_parent_lock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from gf_training.hpo_retention import pin_hpo_evaluation
    root=tmp_path/'parent'
    _,records,_=advanced_hpo_chain(root)
    with run_lock(root),ThreadPoolExecutor(max_workers=1) as executor:
        pending=executor.submit(pin_hpo_evaluation,root,'foreign-thread',1,records[1]['inference_sha256'],
            max_owned_bytes=1024**3,deadline=time.monotonic()+30)
        with pytest.raises(ContractError): pending.result()
