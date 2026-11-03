"""Shared model3 kernels; synthetic numerics do not qualify a learning run."""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gf_common.domain import ContractError, digest
from gf_common.structured import CandidateTable
from gf_common.structured_layout import CARD_INSTANCE_LAYOUT as OLD_CARD
from gf_common.structured_tokens import tokenize_table
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV3,
    architecture_identity,
    dimensions,
    model_config_v2,
    model_config_v3,
    parameter_shapes,
    resource_estimate,
)
from gf_models.structured_network import make_network
from tests.model3_support import LAYOUT, model3, raw_inputs


def binding_for(device):
    if device != "cpu" and os.environ.get("GF_RUN_CUDA") != "1":
        pytest.skip("actual host CUDA checks use GF_RUN_CUDA=1")
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("requested actual CUDA check has no CUDA device")
    return InferenceBindingV2(device=device, max_candidates=64, candidate_chunk=2)


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_pre_model3_committed_model2_named_weights_and_post_rng_golden(hand, temporal):
    golden = json.loads(
        (Path(__file__).parent / "fixtures/model2_pre_model3_initialization.json").read_text()
    )
    assert golden["bots_revision"] == "0f87d785c0b98d106f97df5670ef338b8322842b"
    assert (
        golden["content_sha256"]
        == "0cab97ac901406d2f5dec60fd356096b2fd59bd5eefe5b51b2627bfdfc3c1414"
    )
    assert (
        digest({key: value for key, value in golden.items() if key != "content_sha256"})
        == golden["content_sha256"]
    )
    case = next(
        case for case in golden["cases"] if case["hand"] == hand and case["model_id"] == temporal
    )
    torch.manual_seed(case["outer_torch_seed"])
    network = make_network(model_config_v2(case["model"]), case["seed"])
    actual = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value_sha256": hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        for name, value in network.state_dict().items()
    }
    assert actual == case["named_state_dict"]
    rng = torch.get_rng_state().numpy().tobytes()
    assert rng.hex() == case["post_torch_cpu_rng_hex"]
    assert hashlib.sha256(rng).hexdigest() == case["post_torch_cpu_rng_sha256"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_model3_all_modes_dense_chunk_gradients_full_split_cache(device, hand, temporal):
    binding = binding_for(device)
    model = model3(hand, temporal)
    dense = make_network(model, 17, binding, layout=LAYOUT)
    chunked = make_network(model, 17, binding, layout=LAYOUT)
    assert parameter_shapes(model, layout=LAYOUT) == {
        name: tuple(value.shape) for name, value in dense.state_dict().items()
    }
    frames, candidates, *_ = raw_inputs()
    expected = dense(frames, candidates, dense=True)
    actual = chunked(frames, candidates, chunk_size=1)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    if device != "cpu":
        cpu = make_network(
            model, 17, InferenceBindingV2(max_candidates=64, candidate_chunk=2), layout=LAYOUT
        )
        cpu_result = cpu(frames, candidates, dense=True)
        for left, right in zip(cpu_result, expected, strict=True):
            torch.testing.assert_close(left, right.cpu(), atol=1e-5, rtol=1e-5)
    (-expected[0][1] + 0.2 * expected[1].square() - 0.01 * expected[2]).backward()
    (-actual[0][1] + 0.2 * actual[1].square() - 0.01 * actual[2]).backward()
    for (name, left), (other, right) in zip(
        dense.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert name == other and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-5, msg=name)
    prefixes = [
        "state_encoder",
        "definition_embedding",
        "ability_embedding",
        "card_encoder",
        "event_encoder",
        "actor",
        "critic",
    ]
    if hand != "direct":
        prefixes.append("hand_encoder")
    if hand == "attention":
        prefixes.append("hand_attention")
    if temporal != "mlp":
        prefixes.append("temporal")
    for prefix in prefixes:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in chunked.named_parameters()
            if n.startswith(prefix)
        ), prefix
    with torch.no_grad():
        context, cards = dense.encode_history(frames)
        split_context, split_cards = dense.encode_history(frames, suffix_length=1)
        torch.testing.assert_close(context, split_context, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cards, split_cards, atol=1e-5, rtol=1e-5)
        cache = dense.new_cache()
        for frame in frames:
            cached_context, cached_cards = dense.step(frame, cache)
        torch.testing.assert_close(context, cached_context, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cards, cached_cards, atol=1e-5, rtol=1e-5)
        next(dense.parameters()).add_(0.1)
        with pytest.raises(ContractError, match="stale"):
            dense.step(frames[-1], cache)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"], ids=["cpu", "cuda"])
@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_model3_empty_hands_events_masks_finite(device, hand, temporal):
    network = make_network(model3(hand, temporal), 17, binding_for(device), layout=LAYOUT)
    frames, candidates, *_ = raw_inputs(count=1, empty=True, events=False)
    result = network(frames, candidates)
    assert all(torch.isfinite(value).all() for value in result)
    assert result[0].shape == (1,) and float(result[0].detach().exp().sum()) == 1.0


@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_model3_candidate_permutation_critic_independence_and_distinct_same_definition_instances(
    temporal,
):
    model = model3(temporal=temporal)
    network = make_network(model, 17, InferenceBindingV2(max_candidates=64), layout=LAYOUT)
    frames, candidates, originals, _bank, table = raw_inputs()
    frame = originals[-1]
    order = (3, 1, 0, 2)
    permuted = CandidateTable.from_actions(
        frame,
        [table.to_action(index, frame) for index in order],
        action_contract=LAYOUT.action_contract,
        purpose="component",
        synthetic_capacity=False,
    )
    new_candidates = tokenize_table(permuted, frame, layout=LAYOUT)
    with torch.no_grad():
        before, value, _ = network(frames, candidates)
        after, other_value, _ = network(frames, new_candidates)
        torch.testing.assert_close(after, before[list(order)], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(value, other_value, atol=0, rtol=0)
        fewer = tokenize_table(
            CandidateTable.from_actions(
                frame,
                [table.to_action(0, frame)],
                action_contract=LAYOUT.action_contract,
                purpose="component",
                synthetic_capacity=False,
            ),
            frame,
            layout=LAYOUT,
        )
        torch.testing.assert_close(network(frames, fewer)[1], value, atol=0, rtol=0)
    assert frame.references[2] != frame.references[4]
    assert frames[-1].card_categories[2][0] == frames[-1].card_categories[4][0]


def test_model3_exact_layout_admission_wrong_profile_and_stale_equal_width_before_tensor_work(
    monkeypatch,
):
    import gf_models.torch_runtime as runtime
    from gf_models.structured_artifacts import semantics

    model = model3()
    assert dimensions(ModelConfigV3(), layout=LAYOUT) == {
        "state": 64,
        "hand": 128,
        "card_input": 166,
        "card_actor": 71,
        "frame": 611,
        "actor": 1432,
    }
    for operation in (
        lambda: make_network(model, 17),
        lambda: dimensions(model),
        lambda: parameter_shapes(model),
        lambda: resource_estimate(model, InferenceBindingV2()),
        lambda: semantics(model),
    ):
        with pytest.raises(ContractError):
            operation()
    old_frames, _, *_ = raw_inputs(layout=OLD_CARD)
    network = make_network(model, 17, layout=LAYOUT)
    assert architecture_identity(model, layout=LAYOUT) != architecture_identity(
        model, layout=OLD_CARD
    )
    monkeypatch.setattr(
        network,
        "_tensor",
        lambda *args, **kwargs: pytest.fail("profile mismatch reached array packing"),
    )
    with pytest.raises(ContractError):
        network.encode_frame(old_frames[-1])
    frames, *_ = raw_inputs()
    with pytest.raises(ContractError):
        network.encode_frame(replace(frames[-1], recipe_signature=
            "72db3a64f6d087a5f889c4e3cf6b4eb46a8595fd334e1249ca2d51f76496e1ad"))
    monkeypatch.setattr(
        runtime,
        "torch_runtime",
        lambda *args: pytest.fail("invalid model reached Torch construction"),
    )
    with pytest.raises(ContractError):
        make_network(
            replace(model, resources=replace(model.resources, max_parameters=1)), 17, layout=LAYOUT
        )
    with pytest.raises(ContractError):
        model_config_v3(asdict(model), layout=replace(LAYOUT))
    fake = replace(LAYOUT, event_categories=(LAYOUT.event_categories[0],
        tuple(reversed(LAYOUT.event_categories[1])), *LAYOUT.event_categories[2:]))
    assert fake.event_width == LAYOUT.event_width and fake.history_width == LAYOUT.history_width
    with pytest.raises(ContractError):
        make_network(model, 17, layout=fake)


@pytest.mark.parametrize("hand", ["direct", "deepsets", "attention"])
@pytest.mark.parametrize("temporal", ["mlp", "gru", "transformer"])
def test_repaired_event_and_history_widths_drive_parameters_and_resource_estimates(hand, temporal):
    model = model3(hand, temporal)
    binding = InferenceBindingV2(max_candidates=64, candidate_chunk=2)
    before, after = (parameter_shapes(model, layout=layout) for layout in (OLD_CARD, LAYOUT))
    assert after["event_encoder.weight_ih_l0"] == (3 * model.event.hidden_size, 405)
    assert after["frame_projection.weight"][1] == before["frame_projection.weight"][1] + 10
    assert before["event_encoder.weight_ih_l0"][1] + 10 == after["event_encoder.weight_ih_l0"][1]
    estimates = [resource_estimate(model, binding, layout=layout) for layout in (OLD_CARD, LAYOUT)]
    old, new = estimates
    assert new["parameters"] - old["parameters"] == 10 * (
        3 * model.event.hidden_size + model.temporal.hidden_size)
    assert new["tensor_bytes"]["interval_tokens"] == model.event.max_interval * 405 * 4
    assert (new["tensor_bytes"]["frame_history"] - old["tensor_bytes"]["frame_history"]
        == model.temporal.max_decisions * 10 * 4)
