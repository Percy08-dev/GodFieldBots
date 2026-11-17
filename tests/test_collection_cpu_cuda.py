"""Opt-in actual CUDA transfer mechanics; no timing or sampled-action gate."""
import os
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('GF_RUN_CUDA') != '1', reason='actual HOST CUDA opt-in')
torch = pytest.importorskip('torch')

from benchmarks.collection_cpu.transfer import TransferPacker, install
from gf_common.domain import ContractError
from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, raw_inputs
from tests.test_hpo_network import small_model4


@pytest.mark.parametrize('mode',['T1','T2','T3'])
@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_cuda_corrected_model4_views_tail_final_sync_reset(kind,mode,monkeypatch):
    assert torch.cuda.is_available(), 'explicit host CUDA check requires actual GPU'
    model = small_model4(kind,depth=2)
    binding = InferenceBindingV2(device='cuda:0',max_candidates=512,candidate_chunk=256)
    reference = make_network(model,17,binding,layout=LAYOUT)
    candidate = make_network(model,17,binding,layout=LAYOUT)
    experiment = install(candidate,mode)
    frames,tokens,*_ = raw_inputs(count=3)
    count = 257
    tokens = replace(tokens,kinds=(tokens.kinds[0],)*count,targets=(tokens.targets[0],)*count,
        numerics=(tokens.numerics[0],)*count,present=(tokens.present[0],)*count,
        offsets=tuple(range(count+1)),slots=(0,)*count)
    copies = []
    copy_original = torch.Tensor.copy_
    def copy(destination,source,*args,**kwargs):
        if destination.is_cuda and source.device.type=='cpu':
            copies.append(kwargs.get('non_blocking',False))
        return copy_original(destination,source,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    with torch.no_grad():
        left_cache,right_cache = reference.new_cache(),candidate.new_cache()
        for frame in frames:
            left,left_cards = reference.step(frame,left_cache)
            right,right_cards = candidate.step(frame,right_cache)
            torch.cuda.synchronize('cuda:0')
            experiment.consumed()
        expected = reference.distribution(left,left_cards,tokens)
        actual = candidate.distribution(right,right_cards,tokens)
        expected_value,actual_value = reference.value(left),candidate.value(right)
        retained = [pair for pair in experiment.packer.buffers.values()]
        assert experiment.packer.inflight=={'candidates'}
        assert all(cpu.is_pinned() for cpu,device in retained) if mode=='T3' else all(not cpu.is_pinned() for cpu,device in retained)
        with pytest.raises(ContractError,match='in-flight'):
            experiment.features(tokens)
        # Original final-forward synchronization, followed by consumption notice.
        torch.cuda.synchronize('cuda:0')
        experiment.consumed()
        for expected_tensor,actual_tensor in zip(expected,actual,strict=True):
            torch.testing.assert_close(expected_tensor,actual_tensor,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(expected_value,actual_value,atol=1e-5,rtol=1e-5)
        assert copies and all(copies) if mode=='T3' else copies and not any(copies)
        experiment.reset()
        assert experiment.private_empty()
        assert all(not cpu.count_nonzero() and not device.count_nonzero() for cpu,device in retained)


@pytest.mark.parametrize('mode',['T1','T2','T3'])
def test_cuda_dtype_bit_patterns_lifetime_and_failed_sync_retains_storage(mode,monkeypatch):
    assert torch.cuda.is_available(), 'explicit host CUDA check requires actual GPU'
    packer = TransferPacker('cuda:0',mode,max_bytes=4096,max_tensor_bytes=2048)
    floats = np.array([0.,-0.,np.nan,np.inf],np.float32)
    packed = packer.pack({'f':floats,'i':((1,18),(4,2)),'b':(True,False)},
                         {'f':np.float32,'i':np.int64,'b':np.bool_},group='frame')
    assert packer.inflight=={'frame'}
    with pytest.raises(ContractError,match='in-flight'):
        packer.pack({'f':floats},{'f':np.float32},group='frame')
    torch.cuda.synchronize('cuda:0')
    packer.consumed()
    assert packed['f'].cpu().numpy().tobytes()==floats.tobytes()
    assert packed['i'].dtype==torch.int64 and packed['b'].dtype==torch.bool
    assert packed['i'].cpu().tolist()==[[1,18],[4,2]]
    before = dict(packer.buffers)
    sync = torch.cuda.synchronize
    def failed(*args,**kwargs):
        raise RuntimeError('sync failure sentinel')
    monkeypatch.setattr(torch.cuda,'synchronize',failed)
    with pytest.raises(RuntimeError,match='sync failure'):
        packer.reset()
    assert all(packer.buffers[key][0] is pair[0] for key,pair in before.items())
    monkeypatch.setattr(torch.cuda,'synchronize',sync)
    packer.reset()
    assert packer.private_empty()
    assert all(not cpu.count_nonzero() and not device.count_nonzero() for cpu,device in before.values())


def test_cuda_nonblocking_copy_cancellation_retains_owned_capacities(monkeypatch):
    import asyncio
    assert torch.cuda.is_available(), 'explicit host CUDA check requires actual GPU'
    packer = TransferPacker('cuda:0','T3',max_bytes=4096,max_tensor_bytes=2048)
    copy = torch.Tensor.copy_
    def cancelled(destination,source,*args,**kwargs):
        copy(destination,source,*args,**kwargs)
        raise asyncio.CancelledError
    monkeypatch.setattr(torch.Tensor,'copy_',cancelled)
    with pytest.raises(asyncio.CancelledError):
        packer.pack({'refs':tuple(range(18))},{'refs':np.int64},group='candidates')
    assert packer.inflight=={'candidates'} and packer.buffers
    held = list(packer.buffers.values())
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    packer.reset()
    assert packer.private_empty()
    assert all(not cpu.count_nonzero() and not device.count_nonzero() for cpu,device in held)
