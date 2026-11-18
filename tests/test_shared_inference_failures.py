"""Operational cleanup facts remain distinct from failed data and unknown custody."""
import os
import socket
import time
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.shared_inference import shared_runtime
from gf_simulator_client.shared_service import SharedInferenceService
from gf_simulator_client.structured_resources import ReservationLedger
from tests.test_hpo_gpu_lease import admit
from tests.test_hpo_gpu_lease import global_root as global_root  # noqa: PLC0414


def test_metadata_retirement_does_not_retire_decode_buffers_and_abort_reaps(tmp_path):
    ledger=ReservationLedger(1024,2)
    service=SharedInferenceService(ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
        execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu')
    owner=SimpleNamespace(process=SimpleNamespace(returncode=None))
    index,fd=service.acquire_endpoint(owner)
    endpoint=socket.socket(fileno=os.dup(fd))
    try:
        route=service.authorize(index,owner,game='g',participant=1,revision=0,checkpoint='b'*64,deadline=time.monotonic()+10.)
        endpoint.sendall((1024).to_bytes(8,'big'))  # Header accepted; body decode is actually blocked.
        limit=time.monotonic()+5
        while True:
            observed=service._exchange({'command':'QUIESCE'})
            if observed.get('active_relays')==1:break
            assert time.monotonic()<limit
        service.complete(index,route,completed=False)
        observed=service._exchange({'command':'QUIESCE'})
        assert observed['pending']==0 and observed['active_relays']==1 and observed['accepted'] is False
        with pytest.raises(ContractError):service.quiesce()
        proof=service.close()
        assert proof['gpu_outcome']=='reaped' and proof['broker_outcome']=='reaped'
        assert service.failed and service.retirement['successful'] is False
        assert service.retirement['active_relays']==0 and ledger.service is None
    finally:
        endpoint.close()
        if not service.closed:service.close()


def test_broker_loss_cannot_issue_gpu_child_reap_or_release_admission(global_root,tmp_path):
    with admit() as job:
        service=SharedInferenceService(job.ledger,shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),
            execution='a'*32,directory=tmp_path,slots=2,estimate={'cuda_bytes':0},device='cpu',job_id=job.ticket)
        service.process.kill();service.process.wait(5)
        try:
            with pytest.raises(ContractError):service.close()
            with pytest.raises(ContractError):job.confirm_termination()
            assert not service.lease.released and service.proof is None
        finally:
            # Closing local descriptors is not a descendant reap claim.
            for left,_ in service._pairs:left.close()
            service.channel.close()
            service.process.stdin.close();service.process.stdout.close()
