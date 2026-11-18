"""Serving allocator observation delegates math and cannot block on a dead writer."""
import multiprocessing as mp

import pytest

torch=pytest.importorskip('torch')


def test_shared_allocator_reader_is_nonblocking_and_cpu_wrapper_never_queries_cuda(monkeypatch):
    from tensordict import NonTensorData, TensorDict, lazy_stack

    from gf_models.shared_numeric import numeric_row, tensor_row
    from gf_models.shared_registry import publication
    from gf_models.shared_telemetry import make_served_registry, read_allocator
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from tests.model3_support import LAYOUT, model3, raw_inputs
    ctx=mp.get_context('spawn')
    block=ctx.Array('Q',8,lock=ctx.Lock())
    assert read_allocator(block) is None
    block.get_lock().acquire()
    try:assert read_allocator(block) is None
    finally:block.get_lock().release()
    def forbidden(*args,**kwargs):raise AssertionError('CPU wrapper queried CUDA')
    monkeypatch.setattr(torch.cuda,'memory_stats',forbidden)
    model=model3()
    network=make_network(model,17,InferenceBindingV2(max_candidates=64),layout=LAYOUT)
    wrapper=make_served_registry(device='cpu',cache_entries=6,cache_weight_bytes=134217728,diagnostic_block=block)
    wrapper.load_policy_weights(publication(network,'a'*64))
    frames,candidates,*_=raw_inputs(count=1)
    row=tensor_row(numeric_row(frames[0],candidates,model,layout=LAYOUT))
    result=wrapper(lazy_stack([TensorDict({'inputs':row,'routing':NonTensorData({'checkpoint':'a'*64})},batch_size=[])]))
    assert result['logs'].shape[-1]==len(candidates.kinds)
    assert read_allocator(block) is None
    assert not any('diagnostic' in name for name in wrapper.state_dict())
