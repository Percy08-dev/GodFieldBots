"""Separate opt-in CUDA numeric coverage; no gameplay or learning-quality claim.

Run in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8 and
GFB_RUN_STRUCTURED_CUDA=1. Unavailable configured CUDA fails explicitly.
"""
from __future__ import annotations

import os
from dataclasses import replace

import pytest

from gf_models.structured_config import InferenceBindingV2, ModelConfigV2, TemporalConfigV2
from gf_models.structured_network import make_network
from tests.test_structured_network import cache_storage, inputs

pytestmark = pytest.mark.skipif(os.environ.get("GFB_RUN_STRUCTURED_CUDA") != "1",
                               reason="explicit structured CUDA numeric opt-in required")


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("positions", [18, 128])
def test_cuda_default_attention_cpu_full_split_cache_and_dense_chunk_gradients(make_input, temporal, positions):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    model = replace(ModelConfigV2(), model_id=temporal, max_positions=positions,
                    temporal=TemporalConfigV2(kind=temporal, num_layers=2 if temporal == "transformer" else 1))
    cpu = make_network(model, 17)
    gpu = make_network(model, 17, InferenceBindingV2(device="cuda:0", candidate_chunk=2))
    dense_gpu = make_network(model, 17, InferenceBindingV2(device="cuda:0", candidate_chunk=2))
    import torch
    frames, candidates = inputs(make_input, positions=positions)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        cpu_full = cpu(frames, candidates)
        gpu_full = gpu(frames, candidates)
        cpu_context, cpu_cards = cpu.encode_history(frames)
        gpu_context, gpu_cards = gpu.encode_history(frames)
        for context, cards, network in ((cpu_context, cpu_cards, cpu), (gpu_context, gpu_cards, gpu)):
            cache = network.new_cache()
            storage = cache_storage(cache) if temporal == "transformer" else None
            for frame in frames:
                cached_context, cached_cards = network.step(frame, cache)
                if temporal == "transformer":
                    assert cache_storage(cache) == storage
            torch.testing.assert_close(context, cached_context, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(cards, cached_cards, atol=1e-5, rtol=1e-5)
            if temporal == "transformer":
                cache.reset()
                assert cache_storage(cache) == storage
                for frame in frames:
                    replay_context, _ = network.step(frame, cache)
                torch.testing.assert_close(context, replay_context, atol=1e-5, rtol=1e-5)
                assert cache_storage(cache) == storage
        torch.testing.assert_close(cpu_context, gpu_context.cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cpu_cards, gpu_cards.cpu(), atol=1e-5, rtol=1e-5)
    gpu_split = gpu(frames, candidates, suffix_length=1)
    dense = dense_gpu(frames, candidates, suffix_length=1, dense=True)
    for expected, full, split, dense_value in zip(cpu_full, gpu_full, gpu_split, dense, strict=True):
        torch.testing.assert_close(expected, full.cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(expected, split.detach().cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(split, dense_value, atol=1e-5, rtol=1e-5)
    (-gpu_split[0][1] + .2 * gpu_split[1].square() - .01 * gpu_split[2]).backward()
    (-dense[0][1] + .2 * dense[1].square() - .01 * dense[2]).backward()
    for (name, factorized), (other_name, reference) in zip(gpu.named_parameters(), dense_gpu.named_parameters(), strict=True):
        assert name == other_name
        assert (factorized.grad is None) == (reference.grad is None), name
        if factorized.grad is not None:
            assert torch.isfinite(factorized.grad).all(), name
            torch.testing.assert_close(factorized.grad, reference.grad, atol=1e-5, rtol=1e-5, msg=name)
    for prefix in ("state_encoder", "definition_embedding", "ability_embedding", "card_encoder",
                   "hand_attention", "hand_encoder", "event_encoder"):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for name, parameter in gpu.named_parameters() if name.startswith(prefix)), prefix
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() > 0
