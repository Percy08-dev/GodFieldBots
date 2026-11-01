"""Opt-in synthetic C1 CUDA parity/state gates; no real-game/quality claim.

Run in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8 and
GFB_RUN_STRUCTURED_LEARNING_CUDA=1. Configured unavailable CUDA fails explicitly.
"""
from __future__ import annotations

import os

import pytest

from tests.test_structured_learning import assert_same_learners, learner_for, rollout_for

pytestmark = pytest.mark.skipif(os.environ.get("GFB_RUN_STRUCTURED_LEARNING_CUDA") != "1",
                               reason="explicit structured learning CUDA opt-in required")


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_cuda_cpu_collection_whole_batch_parity_exact_two_update_resume(make_input, tmp_path, temporal):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    cpu = learner_for(temporal)
    cpu_collected = rollout_for(make_input, cpu)
    gpu = learner_for(temporal, device="cuda:0")
    result = gpu.update(cpu_collected)
    assert result["parity"]["checks"] == ["saved", "cpu-cache", "cpu-full", "learner-full", "learner-split"]
    assert result["parity"]["max_abs_error"] <= 1e-5
    gpu.torch.cuda.synchronize()
    gpu.save(tmp_path / "generation", shard_bytes=400000)
    cpu_rng = gpu.torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in gpu.torch.cuda.get_rng_state_all()]
    expected = gpu.torch.rand(7, device="cuda:0")
    restored = learner_for(temporal, device="cuda:0")
    restored.restore(tmp_path / "generation")
    assert restored.torch.equal(restored.torch.get_rng_state(), cpu_rng)
    assert all(restored.torch.equal(actual, original) for actual, original in zip(restored.torch.cuda.get_rng_state_all(), cuda_rng, strict=True))
    assert restored.torch.equal(restored.torch.rand(7, device="cuda:0"), expected)
    assert_same_learners(gpu, restored)
    next_rollout = rollout_for(make_input, gpu, episode="next")
    gpu.update(next_rollout)
    restored.update(next_rollout)
    assert_same_learners(gpu, restored)
    gpu.torch.cuda.synchronize()


def test_cuda_direct_hand_empty_event_state_absent_until_later_gradient(make_input, tmp_path):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    learner = learner_for(hand="direct", device="cuda:0")
    learner.update(rollout_for(make_input, learner, empty=True, events=False))
    assert all(parameter not in learner.optimizer.state for name, parameter in learner.network.named_parameters()
               if name.startswith("event_encoder"))
    learner.save(tmp_path / "generation")
    resumed = learner_for(hand="direct", device="cuda:0")
    resumed.restore(tmp_path / "generation")
    richer = rollout_for(make_input, learner, episode="next")
    learner.update(richer)
    resumed.update(richer)
    assert_same_learners(learner, resumed)
    assert all(resumed.optimizer.state[parameter]["step"].item() == 4
               for name, parameter in resumed.network.named_parameters() if name.startswith("event_encoder"))
