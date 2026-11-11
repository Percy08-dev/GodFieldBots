"""Model4 shared kernels; controlled tokens confer no Source authority."""
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError
from gf_models.hpo_config import model_config_v4, model_defaults_v4
from gf_models.structured_config import InferenceBindingV2, ModelConfigV3, parameter_shapes
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, model3, raw_inputs


def small_model4(kind="mlp", hand="attention", activation="relu", depth=1):
    raw = asdict(model3(hand, kind))
    raw.update(version="4", initialization="orthogonal-activation-aware-v1")
    raw["temporal"].pop("sequence_length")
    raw["temporal"].update(max_decisions=1024, activation=activation, num_layers=depth)
    for name in ("state", "card", "hand", "heads"):
        raw[name]["activation"] = activation
    return model_config_v4(raw, layout=LAYOUT)


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_default_ordered_arrays_and_outer_rng_exact_model3(kind):
    model = model_defaults_v4(kind)
    old = replace(ModelConfigV3(), model_id=kind,
        temporal=replace(ModelConfigV3().temporal, kind=kind,
            num_layers=2 if kind == "transformer" else 1))
    torch.manual_seed(73)
    before = torch.get_rng_state().clone()
    left = make_network(old, 17, layout=LAYOUT)
    right = make_network(model, 17, layout=LAYOUT)
    assert torch.equal(before, torch.get_rng_state())
    assert list(left.state_dict()) == list(right.state_dict())
    for name, array in left.state_dict().items():
        assert torch.equal(array, right.state_dict()[name]), name


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_varied_shared_network_dense_split_cache_gradients(kind, hand, activation):
    model = small_model4(kind, hand, activation, depth=3)
    network = make_network(model, 17, InferenceBindingV2(max_candidates=64, candidate_chunk=2), layout=LAYOUT)
    assert parameter_shapes(model, layout=LAYOUT) == {
        name: tuple(value.shape) for name, value in network.state_dict().items()}
    frames, candidates, *_ = raw_inputs()
    full = network(frames, candidates, dense=True)
    split = network(frames, candidates, suffix_length=1, chunk_size=1)
    for left, right in zip(full, split, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    (-split[0][0] + split[1].square()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in network.named_parameters() if name.startswith("card_encoder"))
    if kind == "mlp":
        assert hasattr(network, "temporal_dense")
        assert network.temporal_dense[0].weight.grad is not None
    with torch.no_grad():
        cache = network.new_cache()
        for frame in frames:
            context, cards = network.step(frame, cache)
        logs, _ = network.distribution(context, cards, candidates)
        torch.testing.assert_close(logs, full[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(network.value(context), full[1], atol=1e-5, rtol=1e-5)
        # Candidate order and count cannot enter the critic.
        context2, _ = network.encode_history(frames, suffix_length=1)
        torch.testing.assert_close(network.value(context2), full[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_activation_gain_and_functional_card_frame_activation(kind):
    model = small_model4(kind, activation="tanh", depth=2)
    network = make_network(model, 17, layout=LAYOUT)
    for name in ("card_encoder", "frame_projection"):
        weight = getattr(network, name).weight
        product = weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
        torch.testing.assert_close(product, torch.eye(len(product)) * (5 / 3) ** 2, atol=1e-5, rtol=1e-5)
    frame = raw_inputs(count=1)[0][0]
    with torch.no_grad():
        network.card_encoder.weight.zero_()
        network.card_encoder.bias.fill_(-1)
        network.frame_projection.weight.zero_()
        network.frame_projection.bias.fill_(-1)
        if kind == "mlp":
            for module in network.temporal_dense:
                if isinstance(module, torch.nn.Linear):
                    module.weight.copy_(torch.eye(module.out_features))
                    module.bias.zero_()
        context, cards = network.encode_frame(frame)
    assert (cards[torch.tensor(frame.card_mask), :model.card.output_size] < 0).all()
    assert (context < 0).all()


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_full_forward_capacity_causality_and_cache_limit(kind):
    network = make_network(small_model4(kind, depth=2), 17, layout=LAYOUT)
    width = network.config.temporal.hidden_size
    frames = torch.randn(1024, width)
    with torch.no_grad():
        full, _ = network.encode_temporal(frames)
        prefix, _ = network.encode_temporal(frames[:3])
        torch.testing.assert_close(full[:3], prefix, atol=1e-5, rtol=1e-5)
        assert full.shape == (1024, width)
        with pytest.raises(ContractError):
            network.encode_temporal(torch.randn(1025, width))


def test_all_encoder_dimensions_and_head_depths_are_shared():
    raw = asdict(small_model4("transformer", activation="tanh", depth=2))
    raw["state"].update(hidden_sizes=[13, 11, 9], output_size=7)
    raw["card"].update(definition_embedding=5, ability_embedding=3, output_size=12)
    raw["hand"].update(hidden_sizes=[15, 9], latent_size=7, attention_heads=3,
                       attention_layers=3, attention_feedforward=17)
    raw["event"].update(hidden_size=11, num_layers=3)
    raw["temporal"].update(hidden_size=15, num_heads=3, feedforward_size=19)
    raw["heads"].update(actor_hidden_sizes=[17, 13, 7], critic_hidden_sizes=[11, 5])
    model = model_config_v4(raw, layout=LAYOUT)
    network = make_network(model, 17, layout=LAYOUT)
    frames, candidates, *_ = raw_inputs()
    result = network(frames, candidates)
    assert all(torch.isfinite(value).all() for value in result)
    assert network.actor[-1].in_features == 7 and network.critic[-1].in_features == 5


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_all_linear_initialization_rules_and_hidden_activations(kind, activation):
    model = small_model4(kind, activation=activation, depth=2)
    network = make_network(model, 17, layout=LAYOUT)
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Linear):
            gain = (.01 if module is network.actor[-1] else 1. if module is network.critic[-1]
                or name.rsplit(".", 1)[-1] in ("query", "key", "value", "output", "feedforward2")
                else 5 / 3 if activation == "tanh" else 2**.5)
            weight = module.weight
            gram = weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
            torch.testing.assert_close(gram, torch.eye(len(gram)) * gain**2, atol=1e-5, rtol=1e-5)
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
    hidden = torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU
    assert isinstance(network.actor[1], hidden) and isinstance(network.critic[1], hidden)
    for group in ("hand_attention", "temporal"):
        if hasattr(network, group) and isinstance(getattr(network, group), torch.nn.ModuleList):
            assert all(layer.activation == activation for layer in getattr(network, group))


@pytest.mark.parametrize("pooling", ["sum", "mean", "max"])
@pytest.mark.parametrize("state", ["identity", "mlp"])
def test_pooling_identity_and_empty_inputs(pooling, state):
    raw = asdict(small_model4(activation="tanh"))
    raw["state"]["kind"] = state
    raw["hand"]["pooling"] = pooling
    network = make_network(model_config_v4(raw, layout=LAYOUT), 17, layout=LAYOUT)
    frames, candidates, *_ = raw_inputs(count=1, empty=True, events=False)
    result = network(frames, candidates)
    assert all(torch.isfinite(value).all() for value in result)
    assert result[0].shape == (1,) and float(result[0].detach().exp().sum()) == 1


@pytest.mark.parametrize("kind", ["mlp", "gru", "transformer"])
def test_model4_candidate_permutation_and_count_leave_critic_exact(kind):
    from gf_common.structured import CandidateTable
    from gf_common.structured_tokens import tokenize_table
    network = make_network(small_model4(kind, activation="tanh", depth=2), 17, layout=LAYOUT)
    frames, candidates, originals, _bank, table = raw_inputs()
    frame = originals[-1]
    with torch.no_grad():
        logs, value, _ = network(frames, candidates)
        order = (3, 1, 0, 2)
        permuted = CandidateTable.from_actions(frame, [table.to_action(index, frame) for index in order],
            action_contract=LAYOUT.action_contract, purpose="component", synthetic_capacity=False)
        new_logs, new_value, _ = network(frames, tokenize_table(permuted, frame, layout=LAYOUT))
        torch.testing.assert_close(new_logs, logs[list(order)], atol=1e-5, rtol=1e-5)
        assert torch.equal(new_value, value)
        single = CandidateTable.from_actions(frame, [table.to_action(0, frame)],
            action_contract=LAYOUT.action_contract, purpose="component", synthetic_capacity=False)
        assert torch.equal(network(frames, tokenize_table(single, frame, layout=LAYOUT))[1], value)


def test_component_activations_are_independent_and_event_history_recipe_is_used():
    from gf_common.structured_tokens import tokenize_frame, tokenize_table
    raw = asdict(small_model4("transformer", depth=2))
    for component in ("state", "hand", "heads"):
        raw[component]["activation"] = "tanh"
    raw["event"]["aggregate_window"] = 16
    model = model_config_v4(raw, layout=LAYOUT)
    network = make_network(model, 17, layout=LAYOUT)
    assert isinstance(network.state_encoder[1], torch.nn.Tanh)
    assert isinstance(network.actor[1], torch.nn.Tanh)
    assert network.hand_attention[0].activation == "tanh"
    assert network.temporal[0].activation == "relu"
    _, _, originals, bank, table = raw_inputs()
    frames = tuple(tokenize_frame(frame, bank.view(), aggregate_window=16, layout=LAYOUT) for frame in originals)
    candidates = tokenize_table(table, originals[-1], aggregate_window=16, layout=LAYOUT)
    result = network(frames, candidates)
    assert all(torch.isfinite(value).all() for value in result)
    with torch.no_grad():
        network.card_encoder.weight.zero_()
        network.card_encoder.bias.fill_(-1)
        network.frame_projection.weight.zero_()
        network.frame_projection.bias.fill_(-1)
        context, cards = network.encode_frame(frames[-1])
        assert torch.equal(context, torch.zeros_like(context))
        assert torch.equal(cards[:, :model.card.output_size], torch.zeros_like(cards[:, :model.card.output_size]))
