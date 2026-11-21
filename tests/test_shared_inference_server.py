"""Official process service executes the real MLP and immutable registry updates."""
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchrl')

from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, model3, raw_inputs


def registry_with_publications(device,versions,diagnostic_block=None):
    """Test factory preloads committed inference-only versions before serving."""
    from gf_models.shared_registry import make_registry
    if diagnostic_block is None:
        registry=make_registry(device=device,cache_entries=6,cache_weight_bytes=134217728)
    else:
        from gf_models.shared_telemetry import make_served_registry
        registry=make_served_registry(device=device,cache_entries=6,cache_weight_bytes=134217728,diagnostic_block=diagnostic_block)
    for weights in versions:registry.load_policy_weights(weights)
    return registry


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_official_server_ragged_routes_and_keeps_old_checkpoint_after_publication(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from tensordict import NonTensorData, TensorDict
    from torchrl.modules.inference_server import InferenceServer, MPTransport

    from gf_models.shared_numeric import numeric_row, score_numeric_row, tensor_row
    from gf_models.shared_registry import make_registry, publication
    model=model3()
    binding=InferenceBindingV2(device=device,max_candidates=64,candidate_chunk=2)
    network=make_network(model,17,binding,layout=LAYOUT)
    first=publication(network,'a'*64)
    second_network=make_network(model,19,binding,layout=LAYOUT)
    second=publication(second_network,'b'*64)
    ctx=mp.get_context('spawn')
    transport=MPTransport(ctx=ctx)
    clients=[transport.client(),transport.client()]
    server=InferenceServer(policy_factory=partial(make_registry,device=device,cache_entries=6,cache_weight_bytes=134217728),
        transport=transport,service_backend='process',service_backend_options={'mp_context':ctx,'startup_timeout':30.},
        policy_device=device,output_device='cpu',max_batch_size=2,min_batch_size=2,timeout=.001)
    frames,candidates,*_=raw_inputs(count=2)
    other,other_candidates,*_=raw_inputs(count=1,empty=True,events=False)
    rows=[tensor_row(numeric_row(frames[-1],candidates,model,layout=LAYOUT)),tensor_row(numeric_row(other[-1],other_candidates,model,layout=LAYOUT))]
    def request(index,digest='a'*64):
        return clients[index](TensorDict({'inputs':rows[index],'routing':NonTensorData({'checkpoint':digest,'request':index})},batch_size=[]))
    original=None
    try:
        server.start();original=server._process
        assert server.update_model_weights(first,timeout=10.) == {'accepted':True}
        with ThreadPoolExecutor(2) as executor:
            answers=list(executor.map(request,(0,1)))
        with torch.no_grad(): expected=score_numeric_row(network,rows[0].to(device))
        torch.testing.assert_close(answers[0]['logs'],expected[0].cpu(),atol=1e-5,rtol=1e-5)
        assert answers[1]['logs'].shape==(1,)
        assert answers[0].get_non_tensor('routing')=={'checkpoint':'a'*64,'request':0}
        assert server.update_model_weights(second,timeout=10.)=={'accepted':True}
        old=request(0)
        assert torch.equal(old['logs'],answers[0]['logs'])
        new=request(0,'b'*64)
        assert not torch.equal(new['logs'],old['logs'])
        assert old['tensor_batch_size'].item()==1
    finally:
        server.shutdown()
        if original is not None:
            original.join(10)
            assert original.exitcode is not None and not original.is_alive()
        transport.close()


@pytest.mark.parametrize('version',[3,4])
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_official_mixed_checkpoint_batch_matches_each_original_network_in_both_orders(version,device):
    import json
    import os
    if device!='cpu' and os.environ.get('GF_RUN_CUDA')!='1':pytest.skip('separate actual CUDA activation')
    from tensordict import NonTensorData, TensorDict
    from torchrl.modules.inference_server import InferenceServer, MPTransport

    from gf_models.shared_numeric import numeric_row, tensor_row
    from gf_models.shared_registry import publication
    from tests.test_hpo_network import small_model4
    model=model3() if version==3 else small_model4(depth=2)
    binding=InferenceBindingV2(device=device,max_candidates=64,candidate_chunk=2)
    original=[make_network(model,seed,binding,layout=LAYOUT) for seed in (17,19)]
    frames_a,candidates_a,*_=raw_inputs(count=3)
    frames_b,candidates_b,*_=raw_inputs(count=1,empty=True,events=False)
    inputs=[(frames_a,candidates_a),(frames_b,candidates_b)]
    with torch.no_grad():
        expected=[network(frames,candidates) for network,(frames,candidates) in zip(original,inputs,strict=True)]
    checkpoints=['a'*64,'b'*64]
    publications=[publication(network,checkpoint) for network,checkpoint in zip(original,checkpoints,strict=True)]
    rows=[tensor_row(numeric_row(frames[-1],candidates,model,layout=LAYOUT)) for frames,candidates in inputs]
    requests=[TensorDict({'inputs':rows[index],'routing':NonTensorData({'checkpoint':checkpoints[index],'request':index})},batch_size=[])
        for index in (0,1)]
    ctx=mp.get_context('spawn')
    transport=MPTransport(ctx=ctx)
    clients=[transport.client(),transport.client()]
    # Both requests are queued before server startup; the factory loads both
    # immutable checkpoints before the official serving loop can drain them.
    initial=[clients[index].submit(requests[index]) for index in (0,1)]
    formation=.25  # Test rendezvous condition, distinct from production1ms.
    diagnostic_block=ctx.Array('Q',8,lock=ctx.Lock())
    server=InferenceServer(policy_factory=partial(registry_with_publications,device,publications,diagnostic_block),transport=transport,
        service_backend='process',service_backend_options={'mp_context':ctx,'startup_timeout':30.},
        policy_device=device,output_device='cpu',max_batch_size=2,min_batch_size=2,timeout=formation)
    process=None
    try:
        server.start();process=server._process
        for order,futures in (((0,1),initial),((1,0),None)):
            if futures is None:futures=[clients[index].submit(requests[index]) for index in order]
            answers=[future.result(timeout=30.) for future in futures]
            assert all(answer['transport_batch_size'].item()>1 for answer in answers)
            assert {answer.get_non_tensor('routing')['checkpoint'] for answer in answers}==set(checkpoints)
            for index,answer in zip(order,answers,strict=True):
                assert answer.get_non_tensor('routing')=={'checkpoint':checkpoints[index],'request':index}
                torch.testing.assert_close(answer['logs'],expected[index][0].cpu(),atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(answer['probabilities'],expected[index][0].exp().cpu(),atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(answer['value'],expected[index][1].cpu(),atol=1e-5,rtol=1e-5)
                assert 'cuda_allocator' not in answer
            from gf_models.shared_telemetry import read_allocator
            allocator=read_allocator(diagnostic_block)
            if device=='cpu':assert allocator is None
            else:
                assert allocator is not None and allocator['allocated_bytes']>0
                assert allocator['boundary']=='post-forward-host' and allocator['synchronization']=='none-added'
            print(json.dumps({'model_version':version,'device':device,'request_order':order,
                'formation_timeout_seconds':formation,'observed_transport_batches':[int(answer['transport_batch_size'].item()) for answer in answers],
                'wrapped_allocator':allocator}),flush=True)
    finally:
        server.shutdown()
        if process is not None:
            process.join(10.)
            assert process.exitcode is not None and not process.is_alive()
        transport.close()
