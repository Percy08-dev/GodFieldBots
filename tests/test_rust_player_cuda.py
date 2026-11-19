"""Hand-counted native row storage lifetimes; no GPU or Source authority."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING, Any

import pytest

from gf_common.domain import ContractError
from gf_common.structured_layout import (
    CARD_INSTANCE_LAYOUT,
)
from gf_common.structured_layout import (
    CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT,
)
from gf_models.structured_config import (
    CardConfigV2,
    EventConfigV3,
    HandConfigV2,
    HeadConfigV2,
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    ModelConfigV4,
    StateConfigV2,
    TemporalConfigV2,
    TemporalConfigV4,
    validate_config,
)

if TYPE_CHECKING:
    from gf_models.rust_player_cuda import NativeMlpRowCudaProfile


def model4(*, hand: str = 'deepsets', pooling: str = 'mean') -> ModelConfigV4:
    model = ModelConfigV4(
        max_positions=2,
        state=StateConfigV2(hidden_sizes=(3,), output_size=2),
        card=CardConfigV2(definition_embedding=2, ability_embedding=1, output_size=4),
        hand=(HandConfigV2(kind='direct') if hand == 'direct' else HandConfigV2(
            kind=hand, hidden_sizes=(3,), latent_size=2, pooling=pooling,
            attention_heads=2 if hand == 'attention' else 4,
            attention_feedforward=6 if hand == 'attention' else 128)),
        event=EventConfigV3(hidden_size=2, num_layers=1, aggregate_window=8,
            max_interval=8, max_episode=128),
        temporal=TemporalConfigV4(hidden_size=4),
        heads=HeadConfigV2(actor_hidden_sizes=(3, 2), critic_hidden_sizes=(3,)),
    )
    validate_config(model, layout=LAYOUT)
    return model


def profile(model: ModelConfigV3 | ModelConfigV4 | None = None, *, chunk: int = 2,
            count: int = 65536) -> NativeMlpRowCudaProfile:
    from gf_models.rust_player_cuda import NativeMlpRowCudaProfile

    return NativeMlpRowCudaProfile(model4() if model is None else model,
        InferenceBindingV2(candidate_chunk=chunk, max_candidates=count), layout=LAYOUT)


def test_raw_operands_alias_r_and_new_masks_have_additional_storage() -> None:
    # Corrected layout: state numeric=104B, categories=292B, opponent=21456B;
    # completed state=21996B. Card numeric/present=3744B, input=8712B,
    # encoded cards=288B, membership=360B, order=72B, actor cards=792B.
    # All native raw leaves (including separate suffix/order/reference leaves)
    # already belong to R; only transformed outputs and new ~mask are below.
    row = profile().estimate(interval_events=0, candidates=5, input_leaf_bytes=1234)
    expected = {
        'frame_state_float': 312,
        'frame_state_categories': 688,
        'frame_state_opponents': 64764,
        'frame_state_concat': 43848,
        'frame_state_opponent_mask': 43452,
        'frame_state_encoder': 22020,
        'frame_card_numeric': 27620,
        'frame_card_input': 39428,
        'frame_card_encoder_mask': 35054,
        'frame_card_membership': 35828,
        'frame_card_order': 35324,
        'frame_actor_cards': 36044,
    }
    for phase, want in expected.items():
        assert row.additional_phases[phase] == want, phase
    assert row.peak_phase == 'frame_state_opponents'
    assert row.peak_additional_bytes == 64764
    assert row.peak_bytes == 65998
    shifted = profile().estimate(interval_events=0, candidates=5, input_leaf_bytes=2468)
    assert shifted.additional_phases == row.additional_phases
    assert shifted.peak_bytes - row.peak_bytes == 1234


@pytest.mark.parametrize(('pooling', 'pool_bytes'), [('sum', 36124), ('mean', 36140),
    ('max', 36284)])
def test_pooling_retains_latent_and_reduction_temporaries(pooling: str, pool_bytes: int) -> None:
    row = profile(model4(pooling=pooling)).estimate(
        interval_events=0, candidates=1, input_leaf_bytes=0)
    assert row.additional_phases['frame_hand_encoder'] == 36404
    assert row.additional_phases['frame_hand_mask'] == 36278
    assert row.additional_phases['frame_hand_pool'] == pool_bytes


@pytest.mark.parametrize(('events', 'deepsets', 'attention'), [
    (0, (36132, 40776, 39252, 37736), (36996, 41640, 40116, 38600)),
    (2, (43820, 44032, 42508, 40992), (44684, 44320, 42796, 41280)),
])
def test_zero_event_attention_kv_survives_but_positive_gru_retires_it(
    events: int, deepsets: tuple[int, ...], attention: tuple[int, ...]
) -> None:
    for hand, want in [('deepsets', deepsets), ('attention', attention)]:
        row = profile(model4(hand=hand)).estimate(
            interval_events=events, candidates=1, input_leaf_bytes=0)
        names = ('frame_events_zero' if events == 0 else 'frame_event_pack',
            'frame_history', 'frame_concat', 'frame_projection')
        assert tuple(row.additional_phases[name] for name in names) == want
        if events:
            # Full input/sequence/hidden exist on CUDA, no CPU full-input gates.
            assert row.additional_phases['frame_event_gru'] == (
                40356 if hand == 'attention' else 39492)
        if hand == 'attention':
            # Second attention RHS retains old output (288) and old K/V (576).
            assert row.additional_phases['frame_hand_attention'] == 47024


def test_direct_hand_and_identity_state_use_aliases() -> None:
    direct = profile(model4(hand='direct')).estimate(
        interval_events=0, candidates=1, input_leaf_bytes=0)
    # Direct hand=34416B, category expansion=30672B, masked_fill=2H+18B.
    assert direct.additional_phases['frame_hand_direct'] == 104822
    assert direct.additional_phases['frame_events_zero'] == 70396
    identity = replace(model4(), state=StateConfigV2(kind='identity'))
    validate_config(identity, layout=LAYOUT)
    row = profile(identity).estimate(interval_events=0, candidates=1, input_leaf_bytes=0)
    assert row.additional_phases['frame_state_encoder'] == 21996
    assert row.additional_phases['frame_card_numeric'] == 27612
    assert row.additional_phases['frame_hand_pool'] == 36132
    # The frame contains identity's large state, without another state allocation.
    assert row.additional_phases['frame_concat'] == 61232


@pytest.mark.parametrize(('candidates', 'chunk', 'full', 'partial', 'concat'), [
    (1, 2, None, 1444, 1404),
    (4, 2, 1812, None, 1740),
    (5, 2, 1916, 1876, 1852),
    (5, 99, None, 2052, 1852),
])
def test_only_last_full_and_partial_chunks_need_live_logits(
    candidates: int, chunk: int, full: int | None, partial: int | None, concat: int
) -> None:
    row = profile(chunk=chunk).estimate(
        interval_events=0, candidates=candidates, input_leaf_bytes=0)
    assert row.additional_phases.get('candidate_chunk_full') == full
    assert row.additional_phases.get('candidate_chunk_partial') == partial
    assert row.additional_phases['candidate_concat'] == concat
    if candidates == 5:
        # Includes padding output copy, operand copy/einsum, shared projection,
        # full features and int64 position indices; raw references remain in R.
        assert row.additional_phases['candidate_projection'] == 2252
        assert row.additional_phases['candidate_features'] == 1780


def test_native_normalization_critic_finite_and_cpu_returns_have_distinct_lifetimes() -> None:
    row = profile().estimate(interval_events=0, candidates=5, input_leaf_bytes=0)
    # U=808B. Scores+logs=40B survive critic and finite check (5n=25B).
    assert {name: row.additional_phases[name] for name in (
        'output_logsoftmax', 'output_critic', 'output_finite',
        'output_probability_cpu_transfer', 'output_logs_cpu_return', 'output_value_cpu_return'
    )} == {
        'output_logsoftmax': 848,
        'output_critic': 872,
        'output_finite': 877,
        'output_probability_cpu_transfer': 44,
        'output_logs_cpu_return': 24,
        'output_value_cpu_return': 4,
    }


def test_profiles_follow_each_checkpoint_geometry_and_model3_layout() -> None:
    base = model4()
    deep = replace(base, temporal=replace(base.temporal, num_layers=3),
        event=replace(base.event, num_layers=3))
    validate_config(deep, layout=LAYOUT)
    row = profile(deep).estimate(interval_events=2, candidates=5, input_leaf_bytes=0)
    assert row.additional_phases['frame_event_gru'] == 39556
    assert row.additional_phases['frame_temporal_dense'] == 41024
    assert profile(base).estimate(interval_events=2, candidates=5,
        input_leaf_bytes=0).additional_phases['output_critic'] == 872
    from gf_models.rust_player_cuda import NativeMlpRowCudaProfile

    old = ModelConfigV3(max_positions=base.max_positions, state=base.state,
        card=base.card, hand=base.hand, event=base.event, heads=base.heads,
        temporal=TemporalConfigV2(hidden_size=4))
    validate_config(old, layout=CARD_INSTANCE_LAYOUT)
    old_profile = NativeMlpRowCudaProfile(old, InferenceBindingV2(), layout=CARD_INSTANCE_LAYOUT)
    old_row = old_profile.estimate(interval_events=2, candidates=5, input_leaf_bytes=0)
    assert old_row.additional_phases['frame_event_pack'] == 43580
    assert old_row.additional_phases['frame_projection'] == 40872
    assert old_row.additional_phases['frame_history'] == 43832


def test_result_and_profile_are_immutable_and_r_is_added_once() -> None:
    compiled = profile()
    row = compiled.estimate(interval_events=0, candidates=1000, input_leaf_bytes=99)
    assert row.recipe == 'native-mlp-row-cuda-storage-v1'
    assert row.input_leaf_bytes == 99
    assert row.peak_phase == 'candidate_features'
    assert row.peak_additional_bytes == 192820
    assert row.peak_bytes == 192919
    with pytest.raises(FrozenInstanceError):
        compiled._geometry = None  # type: ignore[misc, assignment]
    with pytest.raises(FrozenInstanceError):
        row.peak_bytes = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        row.additional_phases['output_finite'] = 0  # type: ignore[index]


@pytest.mark.parametrize(('name', 'bad'), [
    ('interval_events', -1), ('interval_events', 9), ('interval_events', True),
    ('interval_events', 1.0), ('candidates', 0), ('candidates', 65537),
    ('candidates', False), ('candidates', 1.0), ('input_leaf_bytes', -1),
    ('input_leaf_bytes', False), ('input_leaf_bytes', 1.0),
])
def test_boundary_counts_refuse_invalid_native_geometry(name: str, bad: Any) -> None:
    counts = {'interval_events': 0, 'candidates': 1, 'input_leaf_bytes': 0}
    counts[name] = bad
    with pytest.raises(ContractError):
        profile().estimate(**counts)


def test_existing_binding_caps_apply_without_new_caps() -> None:
    compiled = profile(count=3, chunk=99)
    assert compiled.estimate(interval_events=8, candidates=3,
        input_leaf_bytes=2**80).input_leaf_bytes == 2**80
    with pytest.raises(ContractError):
        compiled.estimate(interval_events=0, candidates=4, input_leaf_bytes=0)


@pytest.mark.parametrize('model', [ModelConfigV2(), replace(ModelConfigV4(), model_id='gru'),
    replace(ModelConfigV4(), model_id='transformer')])
def test_unsupported_networks_are_not_reinterpreted(model: Any) -> None:
    with pytest.raises(ContractError):
        profile(model)


def test_compiled_hot_path_never_recalculates_model_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    import gf_models.rust_player_cuda as native_cuda

    compiled = profile()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('per-row model geometry reconstruction')

    monkeypatch.setattr(native_cuda, 'dimensions', forbidden)
    # Maximal native candidate count still creates a bounded number of phases.
    row = compiled.estimate(interval_events=8, candidates=65536, input_leaf_bytes=0)
    assert len(row.additional_phases) <= 32
    assert row.additional_phases['candidate_features'] == 12583732


def test_single_layer_attention_has_no_previous_layer_outputs() -> None:
    model = model4(hand='attention')
    model = replace(model, hand=replace(model.hand, attention_layers=1))
    validate_config(model, layout=LAYOUT)
    row = profile(model).estimate(interval_events=0, candidates=1, input_leaf_bytes=0)
    assert row.additional_phases['frame_hand_attention'] == 46160
    assert row.additional_phases['frame_events_zero'] == 36996


def test_heterogeneous_widths_and_positions_use_their_own_geometry() -> None:
    model = model4()
    model = replace(model, max_positions=3, state=replace(model.state, output_size=4),
        card=replace(model.card, output_size=8), hand=replace(model.hand, latent_size=3),
        event=replace(model.event, hidden_size=3, num_layers=2),
        temporal=replace(model.temporal, hidden_size=6),
        heads=replace(model.heads, actor_hidden_sizes=(5,)))
    validate_config(model, layout=LAYOUT)
    row = profile(model).estimate(interval_events=1, candidates=5, input_leaf_bytes=123)
    assert row.additional_phases['candidate_features'] == 2084
    assert row.additional_phases['candidate_projection'] == 3924
    assert row.additional_phases['candidate_chunk_full'] == 3016
    assert row.additional_phases['candidate_chunk_partial'] == 2924
    assert row.additional_phases['candidate_concat'] == 2848
    assert row.additional_phases['output_critic'] == 1168
    assert row.additional_phases['frame_event_gru'] == 38632


def test_cpu_binding_still_calculates_cuda_gru_geometry_without_device_work() -> None:
    from gf_models.rust_player_cuda import NativeMlpRowCudaProfile

    model = model4()
    cpu = InferenceBindingV2(candidate_chunk=2, max_candidates=8)
    cuda = replace(cpu, device='cuda:0')
    cpu_row = NativeMlpRowCudaProfile(model, cpu, layout=LAYOUT).estimate(
        interval_events=2, candidates=5, input_leaf_bytes=0)
    cuda_row = NativeMlpRowCudaProfile(model, cuda, layout=LAYOUT).estimate(
        interval_events=2, candidates=5, input_leaf_bytes=0)
    assert cpu_row == cuda_row
    # CPU FullLayer precomputes 3 * E * hidden FP32 gates (48B here).
    # The planned CUDA per-step/fused formula explicitly omits that CPU storage.
    assert cpu_row.additional_phases['frame_event_gru'] == 39492
