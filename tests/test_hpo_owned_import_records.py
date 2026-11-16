"""Registry6 retirement pointers and bounded archive presence are explicit formats."""
import asyncio
import json
import time
from pathlib import Path

import pytest

from tests.test_hpo_owned_import_custody import paused_parent as paused_parent  # noqa: PLC0414


@pytest.fixture(scope='module')
def pruned_parent(tmp_path_factory):
    from tests.hpo_owned_support import auditable_runtime, owned_runtime, run_owned
    root=tmp_path_factory.mktemp('owned-pruned')
    operating={'inference_latest':2,'resume_latest':2}
    with pytest.MonkeyPatch.context() as patch:
        runtime,run_id,_,_=owned_runtime(root,patch,decision_budget=80,operating=operating)
        auditable_runtime(patch)
        report,ops=asyncio.run(run_owned(runtime,root/'parents'/run_id,pause_after_update=4,production=True))
        assert report['status']=='paused' and report['updates']==4 and ops.budget.used==0
    directory=root/'parents'/run_id
    registry=json.loads((directory/'artifact-registry.json').read_text())
    assert registry['generations']['1']['files']['resume.json']['status']=='deleted'
    assert type(registry['generations']['1']['files']['resume.json']['deletion_receipt']) is str
    return root,directory


def metadata(directory):
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_training.hpo_owned_restore import OwnedColdImport
    from gf_training.lineage import run_lock
    from tests.test_hpo_owned_numeric import budget
    account=budget(); phase=ArtifactPhaseAdmission(account,1024**3)
    with run_lock(directory) as ownership:
        imported=OwnedColdImport(ownership,budget=account,parent_budget=phase,deadline=time.monotonic()+30)
        try: return imported.state,imported.records
        finally: imported.close()


def test_actual_pruned_parent_accepts_owned_metadata_without_sample_reads(pruned_parent,monkeypatch):
    _,directory=pruned_parent
    def forbidden(*args,**kwargs): raise AssertionError('historical member/bundle audit in metadata import')
    monkeypatch.setattr('gf_training.group_pack_source.inspect_packed',forbidden)
    monkeypatch.setattr('gf_models.sharded_arrays.read_bundle',forbidden)
    state,records=metadata(directory)
    assert state['updates']==4 and len(records)==5


@pytest.mark.parametrize('change',['missing','tampered','redirected','staging'])
def test_owned_retirement_rejects_bad_completion_before_archive_or_model_execution(pruned_parent,change):
    _,directory=pruned_parent
    path=directory/'retention/000001-resume.json.receipt.json'
    original=path.read_bytes()
    stage=None
    try:
        if change=='missing': path.unlink()
        elif change=='redirected':
            path.unlink(); path.symlink_to(directory/'retention/000001-resume.json.intent.json')
        elif change=='tampered':
            value=json.loads(original); value['original_identity'][2]+=1
            path.write_text(json.dumps(value))
        else:
            value=json.loads(original); stage=directory/'artifacts'/value['staging_basename']; stage.mkdir()
        with pytest.raises((ValueError,OSError)): metadata(directory)
    finally:
        if stage is not None: stage.rmdir()
        if path.is_symlink(): path.unlink()
        path.write_bytes(original)


@pytest.mark.parametrize('change',['malformed','redirected','missing-db','wrong-size','foreign','missing-descriptor','pending','manifest'])
def test_archive_presence_rejects_without_opening_members(paused_parent,change):
    directory=paused_parent
    locator=directory/'collections/batch-000000/collection.locator.json'
    raw=locator.read_bytes(); value=json.loads(raw)
    descriptor=Path(value['descriptor']); document=json.loads(descriptor.read_text())
    database=descriptor.parent/document['database']
    moved=database.with_name('test-archive-backup')
    extra=descriptor.parent/'foreign'
    descraw=descriptor.read_bytes()
    try:
        if change=='malformed': locator.write_text('{}')
        elif change=='redirected':
            value['descriptor']=str(directory/'foreign/collection.published.json'); locator.write_text(json.dumps(value))
        elif change=='missing-db': database.rename(moved)
        elif change=='missing-descriptor': descriptor.unlink()
        elif change in ('pending','manifest'):
            if change=='pending': document['publication']='pending'
            else: next(row for row in document['controls'] if row['name']=='manifest.json')['sha256']='a'*64
            descriptor.write_text(json.dumps(document))
            from gf_models.arrays import sha256
            value['descriptor_sha256']=sha256(descriptor); locator.write_text(json.dumps(value))
        elif change=='wrong-size':
            document['database_bytes']+=1; descriptor.write_text(json.dumps(document))
            from gf_models.arrays import sha256
            value['descriptor_sha256']=sha256(descriptor); locator.write_text(json.dumps(value))
        else: extra.write_text('foreign')
        before=(directory/'run.json').read_bytes()
        with pytest.raises((ValueError,OSError)): metadata(directory)
        assert (directory/'run.json').read_bytes()==before
    finally:
        locator.write_bytes(raw); descriptor.write_bytes(descraw)
        if moved.exists(): moved.rename(database)
        if extra.exists(): extra.unlink()


def test_actual_pruned_parent_resumes_and_passes_full_audit(pruned_parent,monkeypatch):
    from gf_training.hpo_source import validate_hpo_parent
    from tests.hpo_owned_support import auditable_runtime, owned_runtime, run_owned
    root,directory=pruned_parent
    runtime,_run_id,_,_=owned_runtime(root,monkeypatch,decision_budget=80,operating={'inference_latest':2,'resume_latest':2})
    auditable_runtime(monkeypatch)
    def forbidden(*args,**kwargs): raise AssertionError('pruned resume entered historical sample reader')
    async def run():
        with monkeypatch.context() as patch:
            patch.setattr('gf_training.hpo_source.validate_hpo_parent',forbidden)
            patch.setattr('gf_training.hpo_rollout.read_hpo_rollout',forbidden)
            report,ops=await run_owned(runtime,directory,mode='resume',production=True)
        assert report['updates']==5 and report['used_decisions']==80 and report['status']=='complete'
        ops.completion.close()
        assert ops.budget.used==0 and ops.writer.parent_budget.used==0
    asyncio.run(run())
    state,records=validate_hpo_parent(directory,require_complete=True,max_owned_bytes=2*1024**3,deadline=time.monotonic()+60)
    assert state['status']=='complete' and len(records)==6
    registry=json.loads((directory/'artifact-registry.json').read_text())
    assert registry['generations']['1']['files']['resume.json']['status']=='deleted'
    assert all(row['status']=='retained' for row in registry['generations']['0']['files'].values())
    assert all(row['status']=='retained' for row in registry['generations']['4']['files'].values())
    assert all(row['status']=='retained' for row in registry['generations']['5']['files'].values())
    from gf_training.hpo_owned_restore import import_owned_final_policy
    final=import_owned_final_policy(directory,deadline=time.monotonic()+30,max_owned_bytes=2*1024**3,parent_budget_bytes=1024**3)
    assert final.generation==5 and not hasattr(final,'resume')
    account=final._budget
    final.close()
    assert account.used==0


def test_bad_archive_rejected_before_child_with_original_operational_cleanup(paused_parent,monkeypatch):
    from gf_common.domain import ContractError
    from gf_training.hpo_gpu_lease import TerminationReceipt
    from tests.hpo_owned_support import owned_runtime, run_owned
    directory=paused_parent; root=directory.parent.parent
    runtime,_run_id,_,_=owned_runtime(root,monkeypatch)
    locator=directory/'collections/batch-000000/collection.locator.json'
    raw=locator.read_bytes(); state=(directory/'run.json').read_bytes()
    async def run():
        try:
            locator.write_text('{}')
            with pytest.raises(ContractError) as rejected: await run_owned(runtime,directory,mode='resume')
            assert type(rejected.value.hpo_gpu_termination) is TerminationReceipt
            assert rejected.value.hpo_gpu_termination.resources==()
            assert (directory/'run.json').read_bytes()==state
            assert not (directory/'generations/000002.json').exists()
        finally: locator.write_bytes(raw)
    asyncio.run(run())
