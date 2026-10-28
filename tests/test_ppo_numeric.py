"""Numerical recipes and information boundaries, using synthetic component data."""
import math
from dataclasses import asdict

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.config import MLPModelConfig
from gf_models.encoder import feature_names
from gf_models.mlp.features import critic_indices, encode_decision
from gf_models.mlp.model import make_network, weights
from gf_training.ppo import RolloutDecision, gae, normalize_advantages, read_rollout, write_rollout


def synthetic_decisions(count=9):
    torch = pytest.importorskip('torch')
    from gf_training.ppo_numeric import Learner
    model = MLPModelConfig()
    learner = Learner(model, 17, 'cpu')
    rng = np.random.default_rng(2)
    output = []
    for index in range(count):
        actor = rng.normal(size=(1 + index % 7, len(feature_names(model.encoder)))).astype(np.float32)
        state = rng.normal(size=(len(critic_indices(model.encoder)),)).astype(np.float32)
        with torch.no_grad():
            logs, value = learner.network(torch.from_numpy(actor)[None], torch.from_numpy(state)[None], torch.ones((1, len(actor)), dtype=torch.bool))
        output.append(RolloutDecision(actor, state, 0, float(logs[0, 0]), float(value[0]), float(index % 3 - 1), float(index % 2)))
    return learner, output


def test_hand_computed_gae_targets_and_population_normalization():
    advantages, targets = gae([.2, .4], 1)
    np.testing.assert_allclose(advantages, [.77, .6])
    np.testing.assert_allclose(targets, [.97, 1.])
    np.testing.assert_allclose(gae([.2, .4], -1)[0], [-1.13, -1.4])
    np.testing.assert_allclose(normalize_advantages([1., 2., 3.]), [-math.sqrt(1.5), 0, math.sqrt(1.5)], rtol=1e-6)
    np.testing.assert_array_equal(normalize_advantages([3, 3]), [0, 0])
    with pytest.raises(ContractError):
        gae([], 1)


def test_critic_independent_of_candidate_order_contents_and_multiplicity(make_input):
    from gf_common.actions import PublicCandidateGenerator
    from gf_common.domain import ActionConfig
    value = make_input()
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value).actions
    model = MLPModelConfig()
    actor, state = encode_decision(value, candidates, model.encoder)
    permuted, other = encode_decision(value, tuple(reversed(candidates)) + candidates[:1], model.encoder)
    np.testing.assert_array_equal(actor[::-1], permuted[:-1])
    np.testing.assert_array_equal(state, other)
    assert all(not feature_names(model.encoder)[index].startswith(('action.', 'element.', 'target.')) for index in critic_indices(model.encoder))


def test_network_initialization_preserves_rng_and_cpu_does_not_initialize_cuda(monkeypatch):
    torch = pytest.importorskip('torch')
    before = torch.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CPU must not inspect CUDA availability'))
    network = make_network(MLPModelConfig(), 17)
    assert torch.equal(before, torch.get_rng_state())
    for name, parameter in network.named_parameters():
        if name.endswith('bias'):
            assert torch.count_nonzero(parameter) == 0


def test_hand_computed_clipping_value_entropy_and_masked_gradient():
    torch = pytest.importorskip('torch')
    from gf_training.ppo_numeric import losses
    logits = torch.tensor([[math.log(.75), math.log(.25), -torch.inf]], requires_grad=True)
    value = torch.tensor([.5], requires_grad=True)
    loss, metrics = losses(logits, value, torch.tensor([[True, True, False]]), torch.tensor([0]),
                           torch.tensor([math.log(.5)]), torch.tensor([2.]), torch.tensor([1.]))
    entropy = -.75 * math.log(.75) - .25 * math.log(.25)
    assert metrics['policy_loss'] == pytest.approx(-2.4)
    assert metrics['value_mse'] == pytest.approx(.25)
    assert metrics['entropy'] == pytest.approx(entropy)
    assert float(loss.detach()) == pytest.approx(-2.4 + .125 - .01 * entropy)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(value.grad).all()
    assert float(value.grad[0]) == pytest.approx(-.5)


def test_full_batch_parity_fails_before_any_parameter_or_adam_change():
    from dataclasses import replace
    learner, decisions = synthetic_decisions()
    before = weights(learner.network)
    decisions[-1] = replace(decisions[-1], old_value=decisions[-1].old_value + .1)
    with pytest.raises(ContractError, match='parity'):
        learner.update(decisions)
    assert learner.steps == 0 and not learner.optimizer.state
    for name, tensor in weights(learner.network).items():
        np.testing.assert_array_equal(tensor, before[name])


def test_rollout_hashes_typed_arrays_and_exact_optimizer_restore(tmp_path):
    from gf_models.mlp.arrays import sha256
    from gf_training.ppo_numeric import Learner
    learner, decisions = synthetic_decisions()
    write_rollout(tmp_path / 'rollout', decisions)
    loaded = read_rollout(tmp_path / 'rollout', sha256(tmp_path / 'rollout' / 'rollout.json'))
    for actual, expected in zip(loaded, decisions, strict=True):
        np.testing.assert_array_equal(actual.actor, expected.actor)
        assert actual.selected == expected.selected
    learner.update(loaded)
    learner.save(tmp_path / 'generation', 'a' * 64)
    restored = Learner(MLPModelConfig(), 17, 'cpu')
    restored.restore(tmp_path / 'generation')
    # Rebase saved rollout evidence to the new frozen generation for next batch.
    torch = learner.torch
    from dataclasses import replace
    rebased = []
    with torch.no_grad():
        for item in decisions:
            logs, value = learner.network(torch.from_numpy(item.actor)[None], torch.from_numpy(item.state)[None], torch.ones((1, len(item.actor)), dtype=torch.bool))
            rebased.append(replace(item, old_logprob=float(logs[0, 0]), old_value=float(value[0])))
    learner.update(rebased)
    restored.update(rebased)
    assert learner.steps == restored.steps
    for name, tensor in weights(learner.network).items():
        np.testing.assert_array_equal(tensor, weights(restored.network)[name])
    path = tmp_path / 'generation' / 'resume.npz'
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ContractError, match='hash'):
        restored.restore(tmp_path / 'generation')


def test_microbatch_padded_slot_limit_and_permutation():
    from gf_training.ppo_numeric import microbatches
    _, decisions = synthetic_decisions(13)
    chunks = microbatches(list(range(13)), decisions, slots=14)
    assert [index for chunk in chunks for index in chunk] == list(range(13))
    assert all(len(chunk) * max(len(decisions[index].actor) for index in chunk) <= 14 for chunk in chunks)


def test_mlp_config_is_closed_and_legacy_feature_signature_stable():
    from gf_models.config import mlp_model_config, ppo_training_config
    from gf_models.encoder import signature
    from gf_models.lightgbm.encoder import signature as legacy
    assert legacy(MLPModelConfig().encoder) == signature(MLPModelConfig().encoder)
    value = asdict(MLPModelConfig())
    value['network']['num_leaves'] = 31
    with pytest.raises(ContractError):
        mlp_model_config(value)
    with pytest.raises(ContractError):
        ppo_training_config({'algorithm': 'ppo', 'learning_rate': .1})


def test_array_header_rejected_before_numpy_materialization(tmp_path, monkeypatch):
    import io
    import zipfile

    from gf_models.mlp.arrays import load_arrays, sha256
    source = io.BytesIO()
    np.lib.format.write_array_header_1_0(source, {'descr': '<f4', 'fortran_order': False, 'shape': (1 << 38,)})
    path = tmp_path / 'tiny-malicious.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('x.npy', source.getvalue())
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('header must fail before array allocation'))
    with pytest.raises(ContractError, match='header'):
        load_arrays(path, {'sha256': sha256(path), 'arrays': {'x': {'dtype': 'float32', 'shape': [1], 'bytes': 4}}})


def test_owned_learning_seed_rng_is_reproducible_and_parity_preserves_it():
    from gf_training.ppo_numeric import Learner
    learner, decisions = synthetic_decisions()
    torch = learner.torch
    expected = torch.get_rng_state().clone()
    learner.parity(decisions)
    assert torch.equal(expected, torch.get_rng_state())
    Learner(MLPModelConfig(), 17, 'cpu')
    assert torch.equal(expected, torch.get_rng_state())


@pytest.mark.parametrize('probability,advantage,policy_gradient', [(.75, -2., 3.), (.25, -2., 0.), (.25, 2., -1.)])
def test_negative_advantage_clipping_and_unclipped_actor_gradient(probability, advantage, policy_gradient):
    torch = pytest.importorskip('torch')
    from gf_training.ppo_numeric import losses
    logs = torch.tensor([[math.log(probability), math.log(1 - probability)]], requires_grad=True)
    loss, _ = losses(logs, torch.tensor([0.]), torch.tensor([[True, True]]), torch.tensor([0]),
                     torch.tensor([math.log(.5)]), torch.tensor([advantage]), torch.tensor([0.]))
    loss.backward()
    entropy_gradient = .01 * probability * (1 + math.log(probability))
    assert float(logs.grad[0, 0]) == pytest.approx(policy_gradient + entropy_gradient, abs=1e-6)


def test_uneven_padded_microbatch_gradients_match_full_logical_batch():
    from gf_training.ppo_numeric import losses, microbatches, padded
    learner, decisions = synthetic_decisions(13)
    torch = learner.torch
    gradients = []
    for slots in (4096, 14):
        learner.optimizer.zero_grad(set_to_none=True)
        for indices in microbatches(list(range(len(decisions))), decisions, slots):
            actor, state, mask, selected = padded(decisions, indices, 'cpu')
            logs, value = learner.network(actor, state, mask)
            vectors = [torch.tensor([getattr(decisions[index], key) for index in indices]) for key in ('old_logprob', 'advantage', 'target')]
            loss, _ = losses(logs, value, mask, selected, *vectors)
            (loss * len(indices) / len(decisions)).backward()
        gradients.append([parameter.grad.clone() for parameter in learner.network.parameters()])
    for full, split in zip(*gradients, strict=True):
        torch.testing.assert_close(full, split, atol=1e-6, rtol=1e-5)


def test_checkpoint_restores_future_owned_rng_draws(tmp_path):
    from gf_training.ppo_numeric import Learner
    learner, _ = synthetic_decisions()
    learner.save(tmp_path / 'rng', 'a' * 64)
    expected_torch = learner.torch.rand(20)
    expected_shuffle = learner.shuffle.integers(2**31, size=20)
    restored = Learner(MLPModelConfig(), 999, 'cpu')
    restored.restore(tmp_path / 'rng')
    assert learner.torch.equal(expected_torch, learner.torch.rand(20))
    np.testing.assert_array_equal(expected_shuffle, restored.shuffle.integers(2**31, size=20))


def test_array_payload_dtype_and_total_decoded_bound(tmp_path, monkeypatch):
    import io
    import zipfile

    from gf_models.mlp import arrays as persistence
    source = io.BytesIO()
    np.save(source, np.asarray([1], dtype=np.int64), allow_pickle=False)
    path = tmp_path / 'dtype.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('x.npy', source.getvalue())
    metadata = {'sha256': persistence.sha256(path), 'arrays': {'x': {'dtype': 'float32', 'shape': [1], 'bytes': 4}}}
    with pytest.raises(ContractError, match='header'):
        persistence.load_arrays(path, metadata)
    monkeypatch.setattr(persistence, 'MAX_ARRAY_BYTES', 8)
    with pytest.raises(ContractError, match='decoded'):
        persistence.save_arrays(tmp_path / 'bound.npz', {'x': np.ones(2, dtype=np.float32), 'y': np.ones(1, dtype=np.float32)})


def test_cpu_hardware_identity_never_queries_cuda_driver(monkeypatch):
    import ctypes
    learner, _ = synthetic_decisions()
    monkeypatch.setattr(ctypes, 'CDLL', lambda *args: pytest.fail('CPU learner must not query CUDA driver'))
    assert learner.hardware()['cuda_driver_api_version'] is None
