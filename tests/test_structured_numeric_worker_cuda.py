"""Opt-in real CUDA numerical IPC with default model2 architectures."""
from __future__ import annotations

import os

import pytest

from gf_common.action_contract import engineering_contract
from gf_models.structured_binding import model_defaults_v2
from gf_models.structured_config import InferenceBindingV2
from gf_training.structured_numeric import Learner
from tests.test_structured_learning import ACTION_HASH, RESUME_ID
from tests.test_structured_numeric_worker import run_lifecycle

pytestmark = pytest.mark.skipif(os.getenv('GFB_RUN_STRUCTURED_NUMERIC_CUDA') != '1', reason='separate CUDA numerical IPC evidence')


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_default_cuda_numerical_process_two_updates_and_restore(tmp_path, make_input, kind):
    model = model_defaults_v2(kind).architecture()
    assert model.hand.kind == 'attention'
    if kind == 'transformer':
        assert model.temporal.num_layers == 2
    learner = Learner(model, 17, InferenceBindingV2(device='cuda:0'), ACTION_HASH, RESUME_ID,
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)
    run_lifecycle(tmp_path, make_input, learner)
