"""Shared service reap expands only the explicit GPU termination version."""
from dataclasses import asdict

import pytest

from gf_common.shared_inference import shared_runtime
from gf_simulator_client.shared_service import SharedInferenceService
from tests.test_hpo_gpu_lease import admit
from tests.test_hpo_gpu_lease import global_root as global_root  # noqa: PLC0414


def test_actual_shared_service_requires_composite_reap_and_issues_schema2(global_root,tmp_path):
    from gf_common.domain import ContractError
    from gf_training.hpo_gpu_lease import _read, _termination
    with admit() as job:
        service=SharedInferenceService(job.ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
            execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu',job_id=job.ticket)
        try:
            with pytest.raises(ContractError):job.confirm_termination()
            proof=service.close()
            receipt=job.confirm_termination()
            assert receipt.schema_version==2
            assert receipt.services==(proof,)
            assert receipt.resources==()
            assert job.require_confirmed_termination(receipt) is receipt
            assert _termination(__import__("json").loads(__import__("json").dumps(asdict(receipt))),_read(job._lock.root/'jobs'/job.ticket/'admission.json'))==receipt
        finally:
            if not service.closed:service.close()


def test_observed_broker_no_child_failure_releases_only_actual_never_started_service(global_root,tmp_path,monkeypatch):
    from gf_simulator_client import shared_service
    def no_child(*args,**kwargs):raise OSError('actual creator fixture before child start')
    monkeypatch.setattr(shared_service.subprocess,'Popen',no_child)
    with admit() as job:
        with pytest.raises(OSError,match='before child start'):
            SharedInferenceService(job.ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
                execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu',job_id=job.ticket)
        assert job.ledger.service is None
        receipt=job.confirm_termination()
        assert receipt.schema_version==2
        assert receipt.services[0]['broker_outcome']=='never_started'
        assert receipt.services[0]['gpu_outcome']=='never_started'
        assert receipt.services[0]['broker_pid'] is None and receipt.services[0]['gpu_pid'] is None
