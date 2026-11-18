"""Real clean broker uses official transport and attests its original child reap."""
import socket
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torchrl')

from gf_common.ram_bank import ReadonlyRamBankSet
from gf_common.shared_inference import ServingBudget, shared_runtime
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.structured_resources import ReservationLedger
from tests.model3_support import LAYOUT, model3, raw_inputs


def test_clean_broker_real_numeric_roundtrip_and_composite_reap(tmp_path):
    from gf_models.shared_numeric import numeric_row
    from gf_simulator_client.shared_service import SharedInferenceService
    model=model3()
    binding=InferenceBindingV2(max_candidates=64,candidate_chunk=2)
    network=make_network(model,17,binding,layout=LAYOUT)
    budget=ServingBudget(128*1024**2)
    banks=ReadonlyRamBankSet.build({name:value.detach().numpy() for name,value in network.state_dict().items()},budget)
    runtime=shared_runtime({'mode':'shared-mlp-v1','max_batch_size':2},game_concurrency=2)
    ledger=ReservationLedger(1024,4)
    service=SharedInferenceService(ledger,runtime,execution='a'*32,directory=tmp_path,slots=4,
        estimate={'cuda_bytes':0},device='cpu')
    owner=SimpleNamespace(process=SimpleNamespace(returncode=None))
    index,fd=service.acquire_endpoint(owner)
    endpoint=socket.socket(fileno=__import__('os').dup(fd))
    stream=endpoint.makefile('rwb',buffering=0)
    try:
        service.publish(banks,model=asdict(model),binding=asdict(binding),checkpoint='b'*64,generation=0,identity='c'*64)
        frames,candidates,*_=raw_inputs(count=1)
        routing=service.authorize(index,owner,game='g1',participant=1,revision=0,checkpoint='b'*64,deadline=time.monotonic()+10.)
        # The old request has actual acceptance authority before publication, but
        # its numeric body reaches the service only after the committed new version.
        from gf_common.domain import ContractError
        with pytest.raises(ContractError):service.evict('b'*64)
        replacement=make_network(model,19,binding,layout=LAYOUT)
        newer=ReadonlyRamBankSet.build({name:value.detach().numpy() for name,value in replacement.state_dict().items()},budget)
        try:service.publish(newer,model=asdict(model),binding=asdict(binding),checkpoint='d'*64,generation=1,identity='e'*64)
        finally:newer.close()
        write_frame(stream,{'routing':routing,'numeric':numeric_row(frames[0],candidates,model,layout=LAYOUT)})
        response=read_frame(stream)
        assert response['routing']==routing and len(response['logs'])==len(candidates.kinds)
        from gf_models.shared_numeric import score_numeric_row, tensor_row
        with torch.no_grad():expected,_=score_numeric_row(network,tensor_row(numeric_row(frames[0],candidates,model,layout=LAYOUT)))
        assert response['logs']==expected.tolist()
        service.complete(index,routing,completed=True)
        service.quiesce()
        owner.process.returncode=0
        service.release_endpoint(index,owner)
        proof=service.close()
        assert proof['gpu_exitcode']==0 and service.closed and ledger.service is None
        assert service.close() is proof
    finally:
        stream.close();endpoint.close();banks.close()
