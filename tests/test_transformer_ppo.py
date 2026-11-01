"""Transformer causal/cache/TBPTT and owned CPU PPO component proofs."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_cli.config import LoadedConfig, load_config
from gf_common.domain import ContractError, digest
from gf_models.config import TransformerModelConfig, transformer_model_config
from gf_models.encoder import feature_names
from gf_models.features import critic_indices
from gf_models.sequence import save_artifact
from gf_models.torch_runtime import weights
from gf_models.transformer.model import cache_storage_bytes, make_network, model_storage_bytes
from gf_models.transformer.policy import TransformerPolicy
from gf_training.lineage import owned_binding, read_json, validate_chain
from gf_training.ppo_run import train_ppo
from gf_training.sequence_numeric import Learner, microbatches
from tests.ppo_support import synthetic_server_identity
from tests.test_gru_ppo import persisted, synthetic_sequence_collector


def inputs(network, length=35, batch=2):
    import torch
    model = TransformerModelConfig()
    generator = torch.Generator().manual_seed(3)
    return (torch.randn(batch, length, len(critic_indices(model.encoder)), generator=generator),
            torch.randn(batch, 3, len(feature_names(model.encoder)), generator=generator),
            torch.ones((batch, 3), dtype=torch.bool))


def test_closed_transformer_config_initialization_position_bytes_and_storage():
    import torch
    model = TransformerModelConfig()
    before = torch.get_rng_state().clone()
    network = make_network(model, 17)
    assert torch.equal(before, torch.get_rng_state())
    actual = sum(tensor.numel() * tensor.element_size() for tensor in (*network.parameters(), *network.buffers()))
    assert model_storage_bytes(model) == actual
    assert cache_storage_bytes() == 2097152
    assert network.positions.dtype == torch.float32 and 'positions' not in network.state_dict()
    np.testing.assert_array_equal(network.positions[0].numpy()[::2], np.zeros(64, dtype=np.float32))
    np.testing.assert_array_equal(network.positions[0].numpy()[1::2], np.ones(64, dtype=np.float32))
    assert not torch.equal(network.layers[0].query.weight, network.layers[1].query.weight)
    for layer in network.layers:
        for projection in (layer.query, layer.key, layer.value, layer.output):
            torch.testing.assert_close(projection.weight @ projection.weight.T, torch.eye(128), atol=2e-6, rtol=1e-5)
        assert torch.count_nonzero(layer.norm1.bias) == torch.count_nonzero(layer.norm2.bias) == 0
    for key in ('hidden_size', 'num_layers', 'num_heads', 'feedforward_size'):
        malformed = asdict(model)
        malformed['network'][key] = float(malformed['network'][key])
        with pytest.raises(ContractError):
            transformer_model_config(malformed)
    assert load_config('configs/experiments/transformer-ppo-cpu.toml').config.model == model


@pytest.mark.parametrize('window', [1, 31, 32, 33, 128, 1024])
def test_transformer_split_detaches_all_prefix_kv_preserves_suffix_and_full_bptt_gradients(window, monkeypatch):
    import torch
    network = make_network(TransformerModelConfig(), 17)
    length = min(window + 2, 35)
    tokens, actor, mask = inputs(network, length)
    tokens.requires_grad_()
    full = network(tokens, actor, mask)
    prefix = max(0, length - window)
    original = network.encode
    captured = []
    def encode(tokens, past=None, offset=0, valid=None):
        result = original(tokens, past, offset, valid)
        captured.append((offset, result[1]))
        return result
    monkeypatch.setattr(network, 'encode', encode)
    split = network.split(tokens, actor, mask, prefix)
    for actual, expected in zip(split, full, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if prefix:
        assert len(captured[0][1]) == 2
        assert all(not tensor.requires_grad and tensor.grad_fn is None for pair in captured[0][1] for tensor in pair)
    assert all(tensor.requires_grad for pair in captured[-1][1] for tensor in pair)
    (split[0][:, 0].sum() + split[1].sum()).backward()
    assert torch.count_nonzero(tokens.grad[:, :prefix]) == 0
    assert torch.count_nonzero(tokens.grad[:, prefix:]) > 0
    if length - prefix > 1:
        assert torch.count_nonzero(tokens.grad[:, prefix:-1]) > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    if not prefix:
        gradients = [parameter.grad.clone() for parameter in network.parameters()]
        network.zero_grad(set_to_none=True)
        (full[0][:, 0].sum() + full[1].sum()).backward()
        for expected, parameter in zip(gradients, network.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)


def test_preallocated_kv_step_uses_actual_past_and_never_concatenates_history(monkeypatch):
    import torch
    network = make_network(TransformerModelConfig(), 17)
    tokens, actor, mask = inputs(network, length=35, batch=1)
    with torch.no_grad():
        expected = network(tokens, actor, mask)
    cache = network.create_cache()
    pointers = cache.keys.data_ptr(), cache.values.data_ptr()
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention
    def attention(query, key, value, **options):
        calls.append((query.shape[-2], key.shape[-2], options))
        return original(query, key, value, **options)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'cat', lambda *args, **kwargs: pytest.fail('incremental cache must not concatenate history'))
        patch.setattr(torch.nn.functional, 'scaled_dot_product_attention', attention)
        for position in range(35):
            # Unused future storage must never become an attention input.
            cache.keys[:, :, :, position:] = 1e6
            cache.values[:, :, :, position:] = -1e6
            context = network.step(tokens[:, position], cache)
    with torch.no_grad():
        actual = network.heads(context, tokens[:, -1], actor, mask)
    for current, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(current, reference, atol=1e-5, rtol=1e-5)
    assert pointers == (cache.keys.data_ptr(), cache.values.data_ptr())
    assert cache.position == 35 and len(calls) == 70
    assert all(query == 1 and options['is_causal'] is False and options['attn_mask'] is None for query, _, options in calls)
    assert [keys for _, keys, _ in calls] == [position + 1 for position in range(35) for _ in range(2)]
    cache.reset()
    assert cache.position == 0 and not torch.count_nonzero(cache.keys) and not torch.count_nonzero(cache.values)
    cache.position = 1024
    with pytest.raises(ContractError, match='resource'):
        network.step(tokens[:, 0], cache)


def test_transformer_causality_padding_critic_candidate_independence_and_episode_reset():
    import torch
    network = make_network(TransformerModelConfig(), 17)
    tokens, actor, mask = inputs(network, length=9, batch=1)
    with torch.no_grad():
        original, _ = network.encode(tokens)
        changed = tokens.clone()
        changed[:, 5:] = 1e5
        future, _ = network.encode(changed)
        torch.testing.assert_close(original[:, :5], future[:, :5], atol=0, rtol=0)
        valid = torch.tensor([[True] * 5 + [False] * 4])
        padded = network(changed, actor, mask, valid=valid)
        short = network(tokens[:, :5], actor, mask)
        for actual, expected in zip(padded, short, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        logs, value = network(tokens, actor, mask)
        other_logs, other_value = network(tokens, actor.flip(1), mask)
        torch.testing.assert_close(logs.flip(1), other_logs)
        torch.testing.assert_close(value, other_value, atol=0, rtol=0)
        extended = torch.cat((actor, torch.full_like(actor, 1e6)), 1)
        extended_mask = torch.tensor([[True] * 3 + [False] * 3])
        padded_logs, padded_value = network(tokens, extended, extended_mask)
        torch.testing.assert_close(padded_logs[:, :3], logs, atol=0, rtol=0)
        torch.testing.assert_close(padded_value, value, atol=0, rtol=0)
        cache = network.create_cache()
        first = network.step(tokens[:, 0], cache)
        network.step(tokens[:, 1], cache)
        cache.reset()
        torch.testing.assert_close(network.step(tokens[:, 0], cache), first, atol=0, rtol=0)


def test_transformer_maximum_prefix_admission_counts_kv_masks_and_attention(tmp_path):
    learner = Learner(TransformerModelConfig(), 17, 'cpu')
    rollout = persisted(tmp_path, learner, (1024, 1024), candidates=1)
    # 2 layers * 4 heads * 1024**2 is exactly the score-slot ceiling.
    assert microbatches([1023], rollout, 32, attention_multiplier=8) == [[1023]]
    assert microbatches([1023, 2047], rollout, 32, attention_multiplier=8) == [[1023], [2047]]
    with pytest.raises(ContractError, match='resource'):
        microbatches([1023], rollout, 32, attention_multiplier=8, attention_slots=8388607)
    with pytest.raises(ContractError, match='resource'):
        microbatches([1023], rollout, 32, attention_multiplier=8, retained_cache_bytes=64 * 1024 * 1024)


def test_transformer_all_batch_real_cache_split_gate_before_adam_and_exact_resume(tmp_path, monkeypatch):
    import gf_training.sequence_numeric as numeric
    learner = Learner(TransformerModelConfig(), 17, 'cpu', 2)
    rollout = persisted(tmp_path, learner)
    original = numeric.make_network
    cached_positions = []
    def network(model, seed):
        result = original(model, seed)
        step = result.step
        def tracked(state, cache):
            cached_positions.append(cache.position)
            return step(state, cache)
        result.step = tracked
        return result
    monkeypatch.setattr(numeric, 'make_network', network)
    learner.parity(rollout)
    assert cached_positions == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6]
    before = weights(learner.network)
    evidence = rollout.shards[-1].old_value
    evidence.flags.writeable = True
    saved = evidence[-1].copy()
    evidence[-1] += .1
    with pytest.raises(ContractError, match='parity'):
        learner.update(rollout)
    assert learner.steps == 0 and not learner.optimizer.state
    for key, value in before.items():
        np.testing.assert_array_equal(value, weights(learner.network)[key])
    evidence[-1] = saved
    learner.update(rollout)
    learner.save(tmp_path / 'generation', 'a' * 64)
    restored = Learner(learner.model, 999, 'cpu', 2)
    restored.restore(tmp_path / 'generation')
    with pytest.raises(ContractError, match='mismatch'):
        Learner(learner.model, 17, 'cpu', 3).restore(tmp_path / 'generation')
    next_batch = persisted(tmp_path, learner, name='next-batch')
    learner.update(next_batch)
    restored.update(next_batch)
    for key, value in weights(learner.network).items():
        np.testing.assert_array_equal(value, weights(restored.network)[key])


def test_transformer_policy_owns_cache_and_budget_before_allocation(tmp_path, make_input, monkeypatch):
    from gf_common.actions import PublicCandidateGenerator
    from gf_common.domain import ActionConfig
    from gf_models.arrays import sha256
    from gf_models.config import SequenceInferenceConfig
    model = TransformerModelConfig()
    network = make_network(model, 17)
    save_artifact(network, tmp_path / 'model', model, 'a' * 64)
    policy = TransformerPolicy(str(tmp_path / 'model'), sha256(tmp_path / 'model' / 'inference.json'))
    other = TransformerPolicy(str(tmp_path / 'model'), sha256(tmp_path / 'model' / 'inference.json'))
    policy.reset(123)
    other.reset(123)
    value = make_input()
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    pointer = policy.cache.keys.data_ptr()
    first = policy.select(value, candidates)
    policy.select(value, candidates)
    assert policy.cache.position == 2 and other.cache.position == 0
    assert other.select(value, candidates) == first
    policy.reset(123)
    assert policy.select(value, candidates) == first and policy.cache.keys.data_ptr() == pointer
    policy.cache.position = 1024
    with pytest.raises(ContractError, match='resource'):
        policy.select(value, candidates)
    tiny = replace(model, inference=SequenceInferenceConfig(cache_budget_bytes=1))
    save_artifact(network, tmp_path / 'tiny', tiny, 'a' * 64)
    import gf_models.transformer.policy as module
    monkeypatch.setattr(module, 'load_artifact', lambda *args, **kwargs: pytest.fail('budget admission must precede model/cache allocation'))
    with pytest.raises(ContractError, match='budget'):
        TransformerPolicy(str(tmp_path / 'tiny'), sha256(tmp_path / 'tiny' / 'inference.json'))


def test_transformer_real_coordinator_two_updates_pause_resume_and_identity(tmp_path, make_input):
    loaded = load_config('configs/experiments/transformer-ppo-cpu.toml', {
        'preprocessing': {'backend': 'python', 'reader': 'legacy'}, 'server': synthetic_server_identity(tmp_path)})
    config = replace(loaded.config, training=replace(loaded.config.training, max_updates=2, sequence_length=2))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    directory = tmp_path / 'parent'
    collector = synthetic_sequence_collector(make_input, 'transformer')
    result = asyncio.run(train_ppo(loaded, collector, directory, pause_after_update=1))
    assert result['status'] == 'paused' and result['updates'] == 1
    label, binding = owned_binding(directory, 1)
    assert label.startswith('transformer-') and binding.model_id == 'transformer'
    result = asyncio.run(train_ppo(loaded, collector, directory, mode='resume'))
    assert result['status'] == 'complete' and result['updates'] == 2
    state, _ = validate_chain(directory)
    assert state['effective_config']['model']['model_id'] == 'transformer'
    assert read_json(directory / 'updates' / '000002.json')['metrics']['parity']['checks'][-1] == 'learner-split'
