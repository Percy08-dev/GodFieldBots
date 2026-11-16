"""Independent evaluation owners drain once; failures retain exact actual custody."""
import numpy as np
import pytest


class DataBank:
    def __init__(self):
        from gf_common.ram_bank import RamBankBuilder
        from tests.test_hpo_owned_numeric import budget
        self.budget=budget()
        self.bank=RamBankBuilder({'weight':np.arange(4,dtype=np.float32)},self.budget).seal()

    def close(self): self.bank.close()


def test_job_close_failure_still_retires_actual_data_owner(tmp_path,monkeypatch):
    from gf_cli.hpo_evaluation import _drain_final_owners
    from gf_training import hpo_gpu_lease as gpu
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'global')
    data=DataBank()
    job=gpu.admit_gpu_job(gpu.JobContext('fixture','eval','evaluation','run','eval'),global_cuda_bytes=1024,max_workers=1)
    job.confirm_termination()
    close=job.close
    def fail():
        close()
        raise OSError('job close postcondition failure')
    monkeypatch.setattr(job,'close',fail)
    try:
        with pytest.raises(OSError,match='job close postcondition') as failed:
            _drain_final_owners(job,[data])
        assert data.bank._fd==-1 and data.budget.used==0
        assert job in failed.value.hpo_final_cleanup_owners
    finally: data.close()


def test_two_actual_data_owners_second_retires_when_first_has_exported_view():
    from gf_cli.hpo_evaluation import _drain_final_owners
    first,second=DataBank(),DataBank()
    view=first.bank.array('weight')
    try:
        with pytest.raises(BufferError) as failed: _drain_final_owners(None,[first,second])
        assert second.bank._fd==-1 and second.budget.used==0
        assert first.bank._fd>=0 and first.budget.used>0
        assert failed.value.hpo_final_cleanup_owners==(first,)
    finally:
        del view
        first.close(); second.close()
    assert first.budget.used==0


def test_primary_error_survives_cleanup_failure_and_all_other_owners_retire():
    from gf_cli.hpo_evaluation import _drain_final_owners
    first,second=DataBank(),DataBank()
    view=first.bank.array('weight')
    primary=ValueError('original provider failure')
    try:
        _drain_final_owners(None,[first,second],primary=primary)
        assert primary.hpo_final_cleanup_owners==(first,)
        assert second.bank._fd==-1 and second.budget.used==0
        assert first.budget.used>0
        assert any('final inference' in note for note in primary.__notes__)
    finally:
        del view
        first.close(); second.close()


def test_comparison_parent_admission_precedes_manifest_getter(tmp_path,monkeypatch):
    import asyncio
    import time

    from gf_cli.hpo_evaluation import (
        _comparison_artifacts,
        _final_policy_reference,
        completion_inspection,
    )
    from gf_training import hpo_owned_restore as restore
    from gf_training.hpo_owned_completion import OwnedExternalFinalPolicy
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,profile,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    imported=[]
    actual=restore.import_owned_final_policy
    def load(*args,**kwargs):
        data=actual(*args,**kwargs)
        imported.append(data)
        data.parent_budget.limit=data.parent_budget.used
        return data
    async def run():
        _,ops=await run_owned(runtime,directory)
        reference,binding=_final_policy_reference(completion_inspection(ops.completion,directory),'a'*64)
        ops.completion.close()
        monkeypatch.setattr(restore,'import_owned_final_policy',load)
        def forbidden(self): raise AssertionError('manifest getter before preparation admission')
        monkeypatch.setattr(OwnedExternalFinalPolicy,'inference_manifest',property(forbidden))
        with pytest.raises(ValueError):
            _comparison_artifacts(profile,profile,[(directory,reference['policy_label'],binding)],{reference['policy_label']:binding},runtime.config,
                deadline=time.monotonic()+30,work=time.monotonic()+40,max_owned_bytes=256*1024**2)
        assert imported and imported[0]._closed and imported[0]._budget.used==0
    asyncio.run(run())


@pytest.mark.parametrize('failure',['import','provider'])
def test_comparison_primary_survives_actual_policy_retirement_failure(tmp_path,monkeypatch,failure):
    import asyncio
    import time

    from gf_cli.hpo_evaluation import (
        _comparison_artifacts,
        _final_policy_reference,
        completion_inspection,
    )
    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    from gf_training import hpo_owned_restore as restore
    from tests.hpo_owned_support import owned_runtime, run_owned
    runtime,run_id,profile,_=owned_runtime(tmp_path,monkeypatch)
    directory=tmp_path/'parents'/run_id
    actual=restore.import_owned_final_policy
    imported=[]; views=[]
    primary=ValueError('original '+failure+' failure')
    def load(*args,**kwargs):
        if imported and failure=='import': raise primary
        data=actual(*args,**kwargs)
        imported.append(data)
        views.append(data.inference.array(data.inference.names[0]))
        return data
    async def run():
        _,ops=await run_owned(runtime,directory)
        reference,binding=_final_policy_reference(completion_inspection(ops.completion,directory),'a'*64)
        ops.completion.close()
        monkeypatch.setattr(restore,'import_owned_final_policy',load)
        if failure=='provider':
            def deny(*args,**kwargs): raise primary
            monkeypatch.setattr(OwnedGameArtifacts,'__init__',deny)
        sources=[(directory,reference['policy_label'],binding)]
        if failure=='import': sources.append((directory/'different',reference['policy_label'],binding))
        try:
            with pytest.raises(ValueError) as failed:
                _comparison_artifacts(profile,profile,sources,{reference['policy_label']:binding},runtime.config,
                    deadline=time.monotonic()+30,work=time.monotonic()+40,max_owned_bytes=256*1024**2)
            assert failed.value is primary
            assert failed.value.hpo_final_cleanup_owners==(imported[0],)
            assert imported[0]._budget.used>0 and not imported[0]._closed
        finally:
            views.clear()
            for data in imported: data.close()
        assert all(data._budget.used==0 for data in imported)
    asyncio.run(run())
