"""Opt-in actual CUDA learner/CPU-or-CUDA actors, engineering FakeHTTP only."""
from __future__ import annotations

import os

import pytest

from tests.test_structured_ppo import run_continuous_and_resume

pytestmark = pytest.mark.skipif(os.environ.get('GFB_RUN_STRUCTURED_PPO_CUDA') != '1', reason='separate actual coordinator CUDA evidence')


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('inference_device', ['cpu', 'cuda:0'])
def test_default_cuda_learner_three_models_cpu_cuda_inference_resume(tmp_path, monkeypatch, kind, inference_device):
    run_continuous_and_resume(tmp_path, monkeypatch, kind=kind, learning_device='cuda:0', inference_device=inference_device)
