"""Shared actor algebra, E2E gradients and temporal parity on raw tokens."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, EventDelta
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_common.validation import decision_input
from gf_models.structured_config import (
    CardConfigV2,
    EventConfigV2,
    HandConfigV2,
    HeadConfigV2,
    ModelConfigV2,
    StateConfigV2,
    TemporalConfigV2,
    parameter_shapes,
)
from gf_models.structured_network import make_network
from tests.conftest import public_options


def small_model(hand="attention", temporal="mlp", positions=18, window=128):
    hand_config = (HandConfigV2(kind="direct") if hand == "direct" else
                   HandConfigV2(kind=hand, hidden_sizes=(16,), latent_size=16,
                                attention_feedforward=32 if hand == "attention" else 128))
    return ModelConfigV2(model_id=temporal, max_positions=positions,
                         state=StateConfigV2(hidden_sizes=(16,), output_size=12),
                         card=CardConfigV2(definition_embedding=8, ability_embedding=4, output_size=16),
                         hand=hand_config, event=EventConfigV2(hidden_size=8, aggregate_window=window),
                         temporal=TemporalConfigV2(kind=temporal, hidden_size=16,
                             num_layers=2 if temporal == "transformer" else 1,
                             feedforward_size=32 if temporal == "transformer" else 256),
                         heads=HeadConfigV2(actor_hidden_sizes=(16, 12), critic_hidden_sizes=(12,)))


def inputs(make_input, count=3, *, empty=False, positions=18, window=128):
    value = make_input(hand=() if empty else ("bronze-club", "steel-armor"),
                       learned=() if empty else ("fireball", "aura"))
    bank = EventBank(1, "episode")
    frames, tokens, events = [], [], []
    for index in range(count):
        event = {"type": "resource_changed", "player": 1, "hp_delta": -index, "mp_delta": 1, "money_delta": 0}
        bank.append(EventDelta(1, "episode", index, bank.boundary.prefix_hash, [event]))
        events.append(event)
        obs = value.to_dict()["observation"]
        obs["state_revision"] = index
        obs["own"]["hp"] -= index
        current = decision_input(obs, events, None, value.catalog, public_options(obs, value.catalog))
        frame = DecisionFrame.from_input(current, bank, index)
        frames.append(frame)
        tokens.append(tokenize_frame(frame, bank.view(), aggregate_window=window))
    actions = ([{"type": "pray"}] if empty else [
        {"type": "play_sequence", "target": 2, "ordered_elements": [frames[-1].references[0]]},
        {"type": "play_sequence", "target": 1, "ordered_elements": [frames[-1].references[2]] * positions},
        {"type": "play_sequence", "target": None, "ordered_elements": [frames[-1].references[3], frames[-1].references[0]]},
        {"type": "discard", "instance_ids": [2]}])
    table = CandidateTable.from_actions(frames[-1], actions, positions)
    return tokens, tokenize_table(table, frames[-1], positions, aggregate_window=window)


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_all_modes_full_split_cache_and_shapes(make_input, hand, temporal):
    model = small_model(hand, temporal)
    network = make_network(model, 7)
    frames, candidates = inputs(make_input)
    assert parameter_shapes(model) == {name: tuple(value.shape) for name, value in network.named_parameters()}
    logprob, value, entropy = network(frames, candidates)
    assert torch.isfinite(logprob).all() and torch.isfinite(value) and torch.isfinite(entropy)
    torch.testing.assert_close(logprob.exp().sum(), torch.tensor(1.), atol=1e-6, rtol=1e-6)
    context, cards = network.encode_history(frames)
    split_context, split_cards = network.encode_history(frames, suffix_length=1)
    torch.testing.assert_close(context, split_context, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cards, split_cards, atol=1e-5, rtol=1e-5)
    cache = network.new_cache()
    with torch.no_grad():
        for frame in frames:
            cached_context, cached_cards = network.step(frame, cache)
    torch.testing.assert_close(context, cached_context, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cards, cached_cards, atol=1e-5, rtol=1e-5)
    assert cache.position == len(frames)


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("positions", [18, 128])
def test_dense_factorized_chunks_all_parameter_and_encoder_gradients(make_input, positions, hand, temporal):
    model = small_model(hand, temporal, positions=positions)
    reference, factorized = make_network(model, 8), make_network(model, 8)
    frames, candidates = inputs(make_input, positions=positions)
    expected = reference(frames, candidates, dense=True)
    actual = factorized(frames, candidates, chunk_size=1)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    loss1 = -expected[0][1] + .2 * expected[1].square() - .01 * expected[2]
    loss2 = -actual[0][1] + .2 * actual[1].square() - .01 * actual[2]
    loss1.backward()
    loss2.backward()
    for (name, left), (other_name, right) in zip(reference.named_parameters(), factorized.named_parameters(), strict=True):
        assert name == other_name
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-5, msg=name)
    prefixes = ["state_encoder", "definition_embedding", "ability_embedding", "card_encoder",
                "event_encoder", "actor", "critic"]
    if hand != "direct":
        prefixes.append("hand_encoder")
    if hand == "attention":
        prefixes.append("hand_attention")
    if temporal != "mlp":
        prefixes.append("temporal")
    for prefix in prefixes:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in factorized.named_parameters() if n.startswith(prefix)), prefix


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_empty_hands_events_and_masks(make_input, hand, temporal):
    network = make_network(small_model(hand, temporal), 4)
    frames, candidates = inputs(make_input, count=1, empty=True)
    frame = replace(frames[0], event_end=frames[0].event_start,
                    interval_numerics=(), interval_present=(), interval_categories=())
    probabilities, value, entropy = network([frame], candidates)
    assert probabilities.item() == 0 and entropy.item() == 0 and torch.isfinite(value)
    with pytest.raises(ContractError):
        network([frame], candidates, mask=torch.tensor([False]))


def permuted(candidates, order):
    slots, offsets = [], [0]
    for index in order:
        slots.extend(candidates.slots[candidates.offsets[index]:candidates.offsets[index + 1]])
        offsets.append(len(slots))
    return replace(candidates, kinds=tuple(candidates.kinds[i] for i in order),
                   targets=tuple(candidates.targets[i] for i in order),
                   numerics=tuple(candidates.numerics[i] for i in order),
                   present=tuple(candidates.present[i] for i in order), offsets=tuple(offsets), slots=tuple(slots))


def test_candidate_permutation_global_normalization_and_critic_independence(make_input):
    network = make_network(small_model(), 6)
    frames, candidates = inputs(make_input)
    context, cards = network.encode_history(frames)
    original = network(frames, candidates, chunk_size=1)
    order = [3, 1, 0, 2]
    shuffled = network(frames, permuted(candidates, order), chunk_size=3)
    torch.testing.assert_close(original[0][order], shuffled[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(original[1], shuffled[1], atol=0, rtol=0)
    subset = permuted(candidates, [0, 1])
    _, subset_value, _ = network(frames, subset)
    torch.testing.assert_close(original[1], subset_value, atol=0, rtol=0)
    assert torch.equal(network.value(context), original[1])
    masked, entropy = network.distribution(context, cards, candidates, mask=torch.tensor([True, False, True, False]))
    assert torch.isneginf(masked[1]) and torch.isfinite(entropy)
    torch.testing.assert_close(masked.exp().sum(), torch.tensor(1.), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("temporal", ["gru", "transformer"])
def test_prefix_has_no_encoder_gradients_suffix_remains_end_to_end(make_input, temporal):
    network = make_network(small_model(temporal=temporal), 2)
    frames, candidates = inputs(make_input)
    observed = []
    def record(module, args, output):
        observed.append(output.requires_grad)
    hook = network.state_encoder.register_forward_hook(record)
    output = network(frames, candidates, suffix_length=1)
    (-output[0][0] + output[1].square()).backward()
    hook.remove()
    assert observed == [False, False, True]
    for prefix in ("state_encoder", "card_encoder", "event_encoder", "temporal"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in network.named_parameters() if n.startswith(prefix))
    # A cache generated before an optimizer modification is never reused.
    cache = network.new_cache()
    with torch.no_grad():
        next(network.parameters()).add_(.01)
        with pytest.raises(ContractError, match="stale"):
            network.step(frames[0], cache)


def test_nondefault_width_window_and_identity_state(make_input):
    model = small_model(hand="deepsets", temporal="gru", window=7)
    model = replace(model, state=StateConfigV2(kind="identity"),
                    card=replace(model.card, output_size=13),
                    temporal=replace(model.temporal, hidden_size=19))
    network = make_network(model, 9)
    frames, candidates = inputs(make_input, window=7)
    assert torch.isfinite(network(frames, candidates)[0]).all()
    wrong, _ = inputs(make_input, window=128)
    with pytest.raises(ContractError):
        network(wrong, candidates)


@pytest.mark.parametrize("kind", ["deepsets", "attention"])
@pytest.mark.parametrize("pooling", ["sum", "mean", "max"])
def test_shared_hand_pooling_has_no_arbitrary_position_encoding(make_input, kind, pooling):
    model = small_model(hand=kind)
    network = make_network(replace(model, hand=replace(model.hand, pooling=pooling)), 12)
    frames, candidates = inputs(make_input, count=1)
    frame = frames[0]
    permutation = (3, 0, 2, 1, *range(4, 18))
    inverse = tuple(permutation.index(i) for i in range(18))
    shuffled_frame = replace(frame,
        card_numerics=tuple(frame.card_numerics[i] for i in permutation),
        card_present=tuple(frame.card_present[i] for i in permutation),
        card_categories=tuple(frame.card_categories[i] for i in permutation),
        card_mask=tuple(frame.card_mask[i] for i in permutation))
    shuffled_candidates = replace(candidates, slots=tuple(inverse[i] for i in candidates.slots))
    original = network(frames, candidates)
    shuffled = network([shuffled_frame], shuffled_candidates)
    for left, right in zip(original, shuffled, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    empty = replace(frame, card_numerics=((0,) * len(frame.card_numerics[0]),) * 18,
                    card_present=((False,) * len(frame.card_present[0]),) * 18,
                    card_categories=((0,) * len(frame.card_categories[0]),) * 18, card_mask=(False,) * 18)
    output, actor_cards = network.encode_frame(empty)
    assert torch.isfinite(output).all() and torch.count_nonzero(actor_cards) == 0


def test_temporal_history_rejects_other_owner_and_gaps(make_input):
    network = make_network(small_model(temporal="gru"), 2)
    frames, candidates = inputs(make_input)
    for changed in (replace(frames[1], actor=2), replace(frames[1], episode="other"),
                    replace(frames[1], revision=frames[0].revision),
                    replace(frames[1], event_start=0)):
        with pytest.raises(ContractError):
            network([frames[0], changed, frames[2]], candidates)


def test_dense_reference_and_override_chunk_admit_actual_materialized_rows(make_input):
    from gf_models.structured_config import ResourceConfigV2
    model = small_model(positions=128)
    # The largest parameter (~250KiB) and a single chunk fit, the requested
    # full reference/gather tensor does not. No large tensor should be allocated.
    model = replace(model, resources=ResourceConfigV2(max_tensor_bytes=1024 * 1024),
                    temporal=replace(model.temporal, max_decisions=3, sequence_length=3),
                    event=replace(model.event, max_interval=8))
    from gf_models.structured_config import InferenceBindingV2
    binding = InferenceBindingV2(candidate_chunk=1, max_candidates=1000)
    network = make_network(model, 1, binding)
    frames, candidates = inputs(make_input, count=1, positions=128)
    context, cards = network.encode_history(frames)
    count = 1000
    many = replace(candidates, kinds=(candidates.kinds[0],) * count,
                   targets=(candidates.targets[0],) * count, numerics=(candidates.numerics[0],) * count,
                   present=(candidates.present[0],) * count, offsets=tuple(range(count + 1)), slots=(0,) * count)
    with pytest.raises(ContractError, match="tensor"):
        network.dense_logits(context, cards, many)
    with pytest.raises(ContractError, match="tensor"):
        network.factorized_logits(context, cards, many, chunk_size=1000)


def test_cache_first_interval_cannot_skip_observed_history(make_input):
    network = make_network(small_model(temporal="gru"), 2)
    frames, _ = inputs(make_input)
    with torch.no_grad(), pytest.raises(ContractError, match="first"):
        network.step(frames[1], network.new_cache())
    with pytest.raises(ContractError, match="first"):
        network.encode_history(frames[1:])


def test_step_configured_episode_cap_cannot_be_bypassed(make_input):
    model = small_model(temporal="gru", window=4)
    model = replace(model, event=replace(model.event, max_interval=4, max_episode=4))
    network = make_network(model, 1)
    frames, _ = inputs(make_input, count=5, window=4)
    cache = network.new_cache()
    with torch.no_grad():
        for frame in frames[:4]:
            network.step(frame, cache)
        with pytest.raises(ContractError, match="episode"):
            network.step(frames[-1], cache)


@pytest.mark.parametrize("dense", [False, True])
def test_endpoint_admits_effective_scoring_chunk_jointly_with_history(make_input, monkeypatch, dense):
    from gf_models.structured_config import InferenceBindingV2, resource_estimate
    model = small_model(positions=128)
    model = replace(model, temporal=replace(model.temporal, max_decisions=3, sequence_length=3),
                    event=replace(model.event, max_interval=8))
    frames, candidates = inputs(make_input, count=3, positions=128)
    count = 1000
    many = replace(candidates, kinds=(candidates.kinds[0],) * count,
                   targets=(candidates.targets[0],) * count, numerics=(candidates.numerics[0],) * count,
                   present=(candidates.present[0],) * count, offsets=tuple(range(count + 1)), slots=(0,) * count)
    original = InferenceBindingV2(candidate_chunk=1, max_candidates=count)
    effective = replace(original, candidate_chunk=count if dense else 100)
    endpoint = {"decisions": 3, "interval_events": 1, "candidates": count, "suffix_decisions": 3}
    separate_maximum = max(
        resource_estimate(model, original)["total_bytes"],
        resource_estimate(model, original, **endpoint)["total_bytes"],
        resource_estimate(model, effective, decisions=1, interval_events=0, candidates=count)["total_bytes"])
    combined = resource_estimate(model, effective, **endpoint)["total_bytes"]
    assert separate_maximum < combined
    budget = (separate_maximum + combined) // 2
    binding = replace(original, managed_budget_bytes=budget)
    # Reproduce the old gap: every independent admission passes under the same
    # ceiling, while the actual combined endpoint must reject.
    resource_estimate(model, binding)
    resource_estimate(model, binding, **endpoint)
    resource_estimate(model, replace(effective, managed_budget_bytes=budget),
                      decisions=1, interval_events=0, candidates=count)
    with pytest.raises(ContractError, match="managed memory"):
        resource_estimate(model, replace(effective, managed_budget_bytes=budget), **endpoint)
    network = make_network(model, 1, binding)
    def encoder_must_not_run(*args, **kwargs):
        pytest.fail("endpoint admission must reject before encoder allocation")
    monkeypatch.setattr(network, "encode_history", encoder_must_not_run)
    with pytest.raises(ContractError, match="managed memory"):
        network(frames, many, dense=dense, chunk_size=100)


def assert_scaled_isometry(matrix, gain):
    # Check the mathematical initialization property, independently of the RNG
    # algorithm or the initializer helper used by production construction.
    rows, columns = matrix.shape
    gram = matrix @ matrix.T if rows <= columns else matrix.T @ matrix
    expected = torch.eye(min(rows, columns), device=matrix.device) * gain**2
    torch.testing.assert_close(gram, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_initialization_gate_projection_gains_independent_draws_and_seed(make_input, temporal):
    import math
    model = small_model(temporal=temporal)
    model = replace(model, event=replace(model.event, num_layers=2))
    if temporal == "gru":
        model = replace(model, temporal=replace(model.temporal, num_layers=2))
    rng_before = torch.get_rng_state().clone()
    network = make_network(model, 71)
    assert torch.equal(torch.get_rng_state(), rng_before)
    same, different = make_network(model, 71), make_network(model, 72)
    for (name, parameter), (same_name, repeated) in zip(network.named_parameters(), same.named_parameters(), strict=True):
        assert name == same_name and torch.equal(parameter, repeated), name
    assert not torch.equal(network.event_encoder.weight_ih_l0, different.event_encoder.weight_ih_l0)
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.GRU):
            for parameter_name, parameter in module.named_parameters():
                if parameter_name.startswith("weight"):
                    gates = parameter.chunk(3, 0)
                    for gate in gates:
                        assert_scaled_isometry(gate, 1.)
                    assert all(not torch.equal(gates[0], other) for other in gates[1:])
                else:
                    assert torch.count_nonzero(parameter) == 0
        elif isinstance(module, torch.nn.Embedding):
            assert torch.count_nonzero(module.weight[0]) == 0
            assert_scaled_isometry(module.weight[1:], 1.)
        elif isinstance(module, torch.nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, torch.nn.Linear):
            projection = name.rsplit(".", 1)[-1]
            if module is network.actor[-1]:
                expected_gain = .01
            elif module is network.critic[-1] or projection in ("query", "key", "value", "output", "feedforward2"):
                expected_gain = 1.
            else:
                expected_gain = math.sqrt(2)
            assert_scaled_isometry(module.weight, expected_gain)
            assert torch.count_nonzero(module.bias) == 0
    attention_layers = list(network.hand_attention)
    if temporal == "transformer":
        attention_layers += list(network.temporal)
    for layer in attention_layers:
        assert not torch.equal(layer.query.weight, layer.key.weight)
        assert not torch.equal(layer.key.weight, layer.value.weight)
        assert not torch.equal(layer.value.weight, layer.output.weight)
    assert not torch.equal(attention_layers[0].query.weight, attention_layers[1].query.weight)


def cache_storage(cache):
    return tuple((buffer.data_ptr(), buffer.untyped_storage().data_ptr(), buffer.untyped_storage().nbytes())
                 for buffer in (cache.keys, cache.values))


def test_transformer_cache_owns_fixed_admitted_storage_and_reset_reuses_it(make_input, monkeypatch):
    from gf_models.structured_config import InferenceBindingV2, resource_estimate
    model = small_model(temporal="transformer")
    model = replace(model, temporal=replace(model.temporal, max_decisions=3, sequence_length=3))
    network = make_network(model, 5)
    frames, _ = inputs(make_input, count=4)
    cache = network.new_cache()
    reserved = cache_storage(cache)
    assert cache.keys.shape == (model.temporal.num_layers, 1, model.temporal.num_heads,
                                model.temporal.max_decisions,
                                model.temporal.hidden_size // model.temporal.num_heads)
    assert sum(storage[2] for storage in reserved) == resource_estimate(model, InferenceBindingV2())["tensor_bytes"]["temporal_cache"]
    original_cat = torch.cat
    def reject_growing_kv(tensors, dim=0, **kwargs):
        if dim == 2 and all(tensor.ndim == 4 for tensor in tensors):
            pytest.fail("inference must append into bounded KV buffers")
        return original_cat(tensors, dim=dim, **kwargs)
    monkeypatch.setattr(torch, "cat", reject_growing_kv)
    with torch.no_grad():
        outputs = []
        for index, frame in enumerate(frames[:3]):
            old_prefix = cache.keys[..., :index, :].clone()
            context, _ = network.step(frame, cache)
            outputs.append(context.clone())
            assert cache_storage(cache) == reserved
            assert torch.equal(cache.keys[..., :index, :], old_prefix)
            assert torch.count_nonzero(cache.keys[..., index, :]) > 0
            assert torch.count_nonzero(cache.keys[..., index + 1:, :]) == 0
        with pytest.raises(ContractError):
            network.step(frames[3], cache)
        cache.reset()
        assert cache_storage(cache) == reserved
        assert cache.position == 0 and cache.owner is None and cache.revision == -1 and cache.event_end == 0
        assert torch.count_nonzero(cache.keys) == torch.count_nonzero(cache.values) == 0
        for expected, frame in zip(outputs, frames[:3], strict=True):
            actual, _ = network.step(frame, cache)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            assert cache_storage(cache) == reserved


def test_cache_admission_rejects_before_allocating_reserved_buffers(monkeypatch):
    import gf_models.structured_network as module
    model = small_model(temporal="transformer")
    network = make_network(model, 5)
    original_estimate = module.resource_estimate
    def reject_full_reservation(config, binding, **kwargs):
        assert config is model and not kwargs
        assert original_estimate(config, binding)["tensor_bytes"]["temporal_cache"] == (
            2 * config.temporal.num_layers * config.temporal.max_decisions * config.temporal.hidden_size * 4)
        raise ContractError("test reservation denied")
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("KV allocation preceded admission")
    monkeypatch.setattr(module, "resource_estimate", reject_full_reservation)
    monkeypatch.setattr(torch, "zeros", unexpected_allocation)
    with pytest.raises(ContractError, match="reservation denied"):
        network.new_cache()
