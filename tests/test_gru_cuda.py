"""Opt-in CUDA functional gates; run in a fresh explicitly configured process.

GFB_RUN_GRU_CUDA=1 enables small all-batch updates and typed resume checks.
GFB_RUN_GRU_CUDA_LONG=1 additionally enables 1024-token endpoint checks. The
latter do not substitute for an all-batch parity gate or a real-game smoke.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from gf_models.config import GRUModelConfig
from gf_models.torch_runtime import weights
from gf_training.sequence_numeric import Learner, _cached_contexts, microbatches, packed
from tests.test_gru_ppo import persisted

pytestmark = pytest.mark.skipif(os.environ.get('GFB_RUN_GRU_CUDA') != '1', reason='explicit CUDA functional opt-in required')


def configured_process():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set before starting this fresh test process'


@pytest.mark.parametrize('window', [1, 32, 128, 1024])
def test_cuda_cpu_collected_batch_actual_split_update_and_typed_resume(tmp_path, window):
    configured_process()
    cpu = Learner(GRUModelConfig(), 17, 'cpu', window)
    rollout = persisted(tmp_path, cpu)
    gpu = Learner(cpu.model, 17, 'cuda:0', window)
    result = gpu.update(rollout)
    assert result['parity']['decisions'] == len(rollout)
    assert result['parity']['checks'] == ['saved', 'cpu-cache', 'cpu-full', 'learner-full', 'learner-split']
    assert result['cuda_parity_peak_allocated_bytes'] > 0 and result['cuda_peak_allocated_bytes'] > 0
    assert not gpu.torch.backends.cudnn.enabled
    gpu.save(tmp_path / 'generation', 'a' * 64)
    restored = Learner(cpu.model, 999, 'cuda:0', window)
    restored.restore(tmp_path / 'generation')
    assert restored.steps == gpu.steps
    for key, value in weights(gpu.network).items():
        np.testing.assert_array_equal(value, weights(restored.network)[key])
    for parameter, restored_parameter in zip(gpu.network.parameters(), restored.network.parameters(), strict=True):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            gpu.torch.testing.assert_close(gpu.optimizer.state[parameter][key], restored.optimizer.state[restored_parameter][key], atol=0, rtol=0)
    print(json.dumps({'scope': 'small synthetic complete batch all-parity/update/resume; not performance or learning quality',
                      'sequence_length': window, **result}, sort_keys=True))


@pytest.mark.skipif(os.environ.get('GFB_RUN_GRU_CUDA_LONG') != '1', reason='explicit long-endpoint opt-in required')
@pytest.mark.parametrize('window', [1, 32, 128, 1024])
def test_cuda_maximum_history_endpoint_full_cache_split_gradients_and_admission(tmp_path, window):
    configured_process()
    cpu = Learner(GRUModelConfig(), 17, 'cpu', window)
    rollout = persisted(tmp_path, cpu, (1024,), candidates=3)
    gpu = Learner(cpu.model, 17, 'cuda:0', window)
    torch = gpu.torch
    indices = microbatches([1023], rollout, window, retained_cache_bytes=1024 * 128 * 4)
    assert indices == [[1023]]
    tokens_cpu, actor_cpu, mask_cpu = packed(rollout, indices[0], 'cpu')
    cached = _cached_contexts(cpu.network, rollout)
    with torch.no_grad():
        reference = cpu.network(tokens_cpu, actor_cpu, mask_cpu)
        cached_reference = cpu.network.heads(torch.from_numpy(cached[-1:]), tokens_cpu[:, -1], actor_cpu, mask_cpu)
    gpu.torch.cuda.reset_peak_memory_stats(gpu.device)
    tokens = tokens_cpu.to(gpu.device).detach().requires_grad_()
    actor, mask = actor_cpu.to(gpu.device), mask_cpu.to(gpu.device)
    with torch.no_grad():
        full = gpu.network(tokens, actor, mask)
    prefix = max(0, 1024 - window)
    split = gpu.network.split(tokens, actor, mask, prefix)
    for cached_value, full_value, split_value, expected in zip(cached_reference, full, split, reference, strict=True):
        torch.testing.assert_close(cached_value, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(full_value.cpu(), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(split_value.detach().cpu(), expected, atol=1e-5, rtol=1e-5)
    (split[0][:, 0].sum() + split[1].sum()).backward()
    assert torch.count_nonzero(tokens.grad[:, :prefix]) == 0
    assert torch.count_nonzero(tokens.grad[:, prefix:]) > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in gpu.network.parameters())
    gpu.synchronize()
    print(json.dumps({'scope': 'single 1024-token endpoint functional check; not all-batch parity or update',
                      'sequence_length': window, 'prefix_length': prefix,
                      'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(gpu.device),
                      'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(gpu.device)}, sort_keys=True))
