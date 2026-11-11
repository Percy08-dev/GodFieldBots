"""Opt-in actual-device functional numerics, never experiment/strength evidence."""
import asyncio
import os
from dataclasses import replace

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("GF_RUN_CUDA") != "1", reason="actual HOST CUDA opt-in")
torch = pytest.importorskip("torch")

from gf_models.structured_config import InferenceBindingV2
from gf_models.structured_network import make_network
from gf_training.hpo_config import HPOTrainingConfig
from tests.model3_support import LAYOUT, raw_inputs
from tests.test_hpo_artifacts import configured_for, retained
from tests.test_hpo_network import small_model4
from tests.test_hpo_numeric import learner_for, rollout_for
from tests.test_structured_learning import assert_same_learners


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_cuda_shared_full_split_cache_cpu_and_gradients(kind, activation):
    assert torch.cuda.is_available(), "explicit CUDA run requires actual GPU"
    model = small_model4(kind, activation=activation, depth=2)
    gpu = make_network(model, 17, InferenceBindingV2(device="cuda:0", max_candidates=64), layout=LAYOUT)
    cpu = make_network(model, 17, InferenceBindingV2(max_candidates=64), layout=LAYOUT)
    frames, candidates, *_ = raw_inputs()
    full = gpu(frames, candidates)
    split = gpu(frames, candidates, suffix_length=1)
    reference = cpu(frames, candidates)
    for expected, actual, cpu_value in zip(full, split, reference, strict=True):
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(expected.cpu(), cpu_value, atol=1e-5, rtol=1e-5)
    (-split[0][0] + split[1].square()).backward()
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in gpu.parameters())
    with torch.no_grad():
        cache = gpu.new_cache()
        for frame in frames:
            context, cards = gpu.step(frame, cache)
        logs, _ = gpu.distribution(context, cards, candidates)
        torch.testing.assert_close(logs, full[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gpu.value(context), full[1], atol=1e-5, rtol=1e-5)
        long = torch.randn(1024, model.temporal.hidden_size, device="cuda:0")
        output, _ = gpu.encode_temporal(long)
        prefix, _ = gpu.encode_temporal(long[:3])
        torch.testing.assert_close(output[:3], prefix, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_cuda_selected_update_resume_and_rng(tmp_path, kind):
    recipe = replace(HPOTrainingConfig(), epochs=2, minibatch_size=1, tbptt_length=1,
        adam_learning_rate=.001, adam_beta1=.8, adam_beta2=.97)
    learner = learner_for(kind, recipe=recipe, device="cuda:0", activation="tanh")
    result = learner.update(rollout_for(learner, selected=((0, 0), (0, 2))))
    assert result["optimizer_step_delta"] == 4 and result["collected_decisions"] == 3
    directory = tmp_path / "cuda5"
    learner.save(directory)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    restored = learner_for(kind, recipe=recipe, device="cuda:0", activation="tanh")
    restored.restore(directory)
    assert_same_learners(learner, restored)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    next_rollout = rollout_for(learner, selected=((0, 1),))
    learner.update(next_rollout)
    restored.update(next_rollout)
    assert_same_learners(learner, restored)


def test_cuda_actual_participant_model4_inference5(tmp_path):
    from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.test_card_instance_baseline_runtime import game
    from tests.test_card_instance_learning import ACTIONS
    learner = learner_for(device="cuda:0")
    directory = tmp_path / "inference5"
    learner.save(directory)
    configured = configured_for(learner, directory)
    owner = retained(configured)
    label, execution = owner.binding(configured)
    frame, _, delta, _ = game()

    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker("mlp", 17, ACTIONS, ledger=ledger,
            policy_options=execution.policy_options, artifact_inputs=owner)
        try:
            generated, selected = await worker.select_frame(frame, delta)
            assert selected.policy_version == label and selected.value is not None
            selected.validate(generated.table, frame)
        finally:
            await worker.close()
        assert worker.lease.released and worker.process.poll() is not None
    try:
        asyncio.run(run())
    finally:
        owner.release()
