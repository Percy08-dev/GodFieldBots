"""Pure tensor-storage estimate for one native, no-grad actor inference row.

R is the caller's sum of distinct final-dtype CUDA input-leaf allocations. Raw
operands, reference slices, reshapes and detach views alias R. Equal-valued
native leaves are still distinct allocations. Named phases add only live
intermediate storage to R, excluding parameters, positions, learner storage,
backend workspace and context. These are estimates, not allocator/physical
bounds or admission authority. No device is needed to calculate CUDA geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType

from gf_common.domain import ContractError, integer
from gf_common.structured_layout import StructuredLayout
from gf_models.actor_cuda import _attention_scratch, _mlp_scratch
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV3,
    ModelConfigV4,
    dimensions,
)

NATIVE_MLP_ROW_CUDA_RECIPE = 'native-mlp-row-cuda-storage-v1'


def _onehot_additional(rows: int, categories: tuple[tuple[str, ...], ...]) -> int:
    """Raw indices are external: prior FP32 columns + current int64/FP32, then cat."""
    floats, conversion_peak = 0, 0
    for vocabulary in categories:
        column = 4 * rows * (len(vocabulary) + 2)
        conversion_peak = max(conversion_peak, floats + 3 * column)
        floats += column
    return max(conversion_peak, 2 * floats)


@dataclass(frozen=True)
class NativeMlpRowCudaEstimate:
    recipe: str
    input_leaf_bytes: int
    additional_phases: Mapping[str, int]
    peak_additional_bytes: int
    peak_phase: str
    peak_bytes: int


@dataclass(frozen=True)
class _Geometry:
    fixed_phases: tuple[tuple[str, int], ...]
    hand_retained: int
    hand_kv: int
    event_input_per_event: int
    event_pack_per_event: int
    event_sequence_per_event: int
    event_previous_layer_per_event: int
    event_hidden_bytes: int
    event_hidden_stack_bytes: int
    history_bytes: int
    frame_bytes: int
    context_bytes: int
    temporal_dense_scratch: int
    shared_bytes: int
    shared_projection_bytes: int
    kind_per_candidate: int
    target_per_candidate: int
    feature_per_candidate: int
    projection_peak: int
    scoring_base: int
    chunk_per_candidate: int
    critic_scratch: int
    max_interval: int
    max_candidates: int
    candidate_chunk: int


def _compile(model: ModelConfigV3 | ModelConfigV4, binding: InferenceBindingV2,
             layout: StructuredLayout) -> _Geometry:
    # The caller owns full model validation. Check only this integration's closed
    # model/binding support and exact layout pairing (through dimensions).
    if (type(model) not in (ModelConfigV3, ModelConfigV4)
            or type(model) is ModelConfigV3 and model.model_id != 'mlp'
            or model.model_id != model.temporal.kind):
        raise ContractError('native row CUDA profile requires Model3 MLP or Model4')
    if type(binding) is not InferenceBindingV2:
        raise ContractError('native row CUDA profile requires InferenceBindingV2')
    d = dimensions(model, layout=layout)
    state_numeric = 4 * len(layout.state_numeric_fields)
    state_categories = 4 * sum(len(values) + 2 for values in layout.state_categories)
    opponent = 4 * 18 * (len(layout.card_categories[0]) + 2)
    state_input = 4 * layout.state_width
    state_output = 0 if model.state.kind == 'identity' else 4 * d['state']
    state_scratch = (0 if model.state.kind == 'identity' else _mlp_scratch(
        (layout.state_width, *model.state.hidden_sizes, model.state.output_size)))
    state = state_input + state_output
    card_numeric = 4 * 18 * len(layout.card_numeric_fields)
    numbers = 2 * card_numeric
    embeddings = 4 * 18 * (model.card.definition_embedding + model.card.ability_embedding)
    suffix = 4 * 18 * sum(len(values) + 2 for values in layout.card_categories[2:])
    card_input, cards = 4 * 18 * d['card_input'], 4 * 18 * model.card.output_size
    membership = 4 * 18 * (len(layout.card_categories[layout.membership_index]) + 2)
    order, inverted_mask = 4 * 18, 18
    actor_cards = 4 * 18 * d['card_actor']
    prefix = state + numbers + card_input + cards + membership + order + actor_cards
    phases = {
        'frame_state_float': 3 * state_numeric,
        'frame_state_categories': state_numeric + _onehot_additional(1, layout.state_categories),
        'frame_state_opponents': state_numeric + state_categories + 3 * opponent,
        'frame_state_concat': state_numeric + state_categories + opponent + state_input,
        'frame_state_opponent_mask': state_input + opponent,
        'frame_state_encoder': state_input + state_scratch,
        'frame_card_numeric': state + 3 * card_numeric,
        'frame_card_input': state + numbers + max(
            embeddings + _onehot_additional(18, layout.card_categories[2:]),
            embeddings + suffix + card_input),
        'frame_card_encoder_mask': state + numbers + card_input + 2 * cards + inverted_mask,
        'frame_card_membership': state + numbers + card_input + cards + max(
            3 * membership, 2 * membership + inverted_mask),
        'frame_card_order': state + numbers + card_input + cards + membership + 3 * order,
        'frame_actor_cards': prefix + 4 * 18,
    }
    attention_output = cards if model.hand.kind == 'attention' else 0
    hand_kv = 2 * cards if model.hand.kind == 'attention' else 0
    if model.hand.kind == 'direct':
        latent, hand = 0, 4 * d['hand']
        phases['frame_hand_direct'] = prefix + max(
            _onehot_additional(18, layout.card_categories), 2 * hand + inverted_mask)
    else:
        latent, hand = 4 * 18 * model.hand.latent_size, 4 * model.hand.latent_size
        hand_base = prefix + attention_output + hand_kv
        phases['frame_hand_encoder'] = hand_base + _mlp_scratch(
            (model.card.output_size, *model.hand.hidden_sizes, model.hand.latent_size), 18)
        phases['frame_hand_mask'] = hand_base + 2 * latent + inverted_mask
        if model.hand.pooling == 'sum':
            pool = latent + hand
        elif model.hand.pooling == 'mean':
            pool = latent + max(hand + 16, 2 * hand + 8)
        else:
            # masked latent, values + int64 indices, then hand/zeros/where.
            pool = max(2 * latent + inverted_mask, 2 * latent + 3 * hand,
                       latent + 3 * hand + 1)
        phases['frame_hand_pool'] = hand_base + pool
        if model.hand.kind == 'attention':
            previous = cards + hand_kv if model.hand.attention_layers > 1 else 0
            phases['frame_hand_attention'] = prefix + previous + _attention_scratch(
                model.card.output_size, model.hand.attention_heads,
                model.hand.attention_feedforward, 18, 18)
    hand_retained = prefix + attention_output + hand_kv + latent + hand
    event_numeric = 4 * len(layout.event_numeric_fields)
    event_categories = 4 * sum(len(values) + 2 for values in layout.event_categories)
    event_input = 2 * event_numeric + event_categories
    event_pack = max(3 * event_numeric,
        event_numeric + _onehot_additional(1, layout.event_categories),
        event_numeric + event_categories + event_input)
    context = 4 * model.temporal.hidden_size
    temporal_scratch = (_mlp_scratch((model.temporal.hidden_size,) * model.temporal.num_layers)
                        if model.model_id == 'mlp' and model.temporal.num_layers > 1 else 0)
    shared = context + actor_cards
    actor = model.heads.actor_hidden_sizes
    first = actor[0]
    projection = 4 * model.max_positions * 19 * first
    unpadded, padding = 4 * model.max_positions * 18 * first, 4 * model.max_positions * first
    operand_copy = 4 * model.max_positions * first * d['card_actor']
    # Same candidate-chunk scratch as actor_cuda, but all refs are views of R.
    x = 4 * first
    tail = x + max(_mlp_scratch((*actor, 1)),
                   4 * (first + (actor[1] if len(actor) > 1 else 1)))
    chunk = max(4 * model.max_positions * first + 2 * x, 3 * x, tail)
    return _Geometry(
        fixed_phases=tuple(phases.items()), hand_retained=hand_retained, hand_kv=hand_kv,
        event_input_per_event=event_input, event_pack_per_event=event_pack,
        event_sequence_per_event=4 * model.event.hidden_size,
        event_previous_layer_per_event=4 * model.event.hidden_size if model.event.num_layers > 1 else 0,
        event_hidden_bytes=4 * model.event.hidden_size,
        event_hidden_stack_bytes=4 * model.event.num_layers * model.event.hidden_size,
        history_bytes=4 * layout.history_width, frame_bytes=4 * d['frame'],
        context_bytes=context, temporal_dense_scratch=temporal_scratch,
        shared_bytes=shared, shared_projection_bytes=4 * first,
        kind_per_candidate=4 * (len(layout.action_kinds) + 2),
        target_per_candidate=4 * (len(layout.targets) + 2),
        feature_per_candidate=4 * layout.candidate_width,
        projection_peak=max(operand_copy + unpadded, unpadded + padding + projection),
        scoring_base=shared + 4 * first + projection + 8 * model.max_positions,
        chunk_per_candidate=chunk,
        critic_scratch=_mlp_scratch((model.temporal.hidden_size, *model.heads.critic_hidden_sizes, 1)),
        max_interval=model.event.max_interval, max_candidates=binding.max_candidates,
        candidate_chunk=binding.candidate_chunk,
    )


@dataclass(frozen=True)
class NativeMlpRowCudaProfile:
    """Compile once per already validated checkpoint; estimate actual E/n/R in O(1).

    Device selection in binding does not change this CUDA geometry. In
    particular, event GRU scratch models CUDA per-step/fused-cell storage even
    when this pure calculation runs on a CPU host. CPU GRU full input gates are
    a different implementation and are not evidence for this CUDA estimate.
    """

    model: InitVar[ModelConfigV3 | ModelConfigV4]
    binding: InitVar[InferenceBindingV2]
    layout: InitVar[StructuredLayout] = field(kw_only=True)
    _geometry: _Geometry = field(init=False, repr=False)

    def __post_init__(self, model: ModelConfigV3 | ModelConfigV4,
                      binding: InferenceBindingV2, layout: StructuredLayout) -> None:
        object.__setattr__(self, '_geometry', _compile(model, binding, layout))

    def estimate(self, *, interval_events: int, candidates: int,
                 input_leaf_bytes: int) -> NativeMlpRowCudaEstimate:
        """Additional live storage by phase, then add R exactly once to the peak."""
        g = self._geometry
        integer(interval_events, 0, g.max_interval)
        integer(candidates, 1, g.max_candidates)
        if type(input_leaf_bytes) is not int or input_leaf_bytes < 0:
            raise ContractError('input leaf bytes must be a nonnegative integer')
        phases = dict(g.fixed_phases)
        if interval_events:
            event_input = interval_events * g.event_input_per_event
            sequence = interval_events * g.event_sequence_per_event
            # Pinned CUDA GRU: no full precomputed CPU input gates. Retain
            # initial/final/stacked hidden and previous layer across cell work.
            gru = (3 * g.event_hidden_stack_bytes
                + interval_events * g.event_previous_layer_per_event
                + max(2 * sequence, sequence + 11 * g.event_hidden_bytes))
            phases['frame_event_pack'] = g.hand_retained + interval_events * g.event_pack_per_event
            phases['frame_event_gru'] = g.hand_retained + event_input + gru
            # `_` retires old attention KV only AFTER the GRU RHS completes.
            frame_base = (g.hand_retained - g.hand_kv + event_input + sequence
                          + g.event_hidden_stack_bytes)
        else:
            # No GRU tuple assignment: `_` still retains the hand attention KV.
            frame_base = g.hand_retained + g.event_hidden_bytes
            phases['frame_events_zero'] = frame_base
        phases['frame_history'] = frame_base + 3 * g.history_bytes
        phases['frame_concat'] = frame_base + g.history_bytes + g.frame_bytes
        phases['frame_projection'] = frame_base + g.frame_bytes + 2 * g.context_bytes
        if g.temporal_dense_scratch:
            phases['frame_temporal_dense'] = (frame_base + g.frame_bytes + g.context_bytes
                                               + g.temporal_dense_scratch)
        # Only context + actor cards survive the frame call (U).
        kind = candidates * g.kind_per_candidate
        target = candidates * g.target_per_candidate
        numeric = 16 * candidates
        features = candidates * g.feature_per_candidate
        shared = g.shared_bytes + g.shared_projection_bytes
        phases['candidate_features'] = shared + max(
            3 * kind, kind + 3 * target, kind + target + 3 * numeric,
            kind + target + numeric + features)
        phases['candidate_projection'] = shared + features + g.projection_peak
        scoring = g.scoring_base + features
        full, partial = divmod(candidates, g.candidate_chunk)
        # Earlier equal-sized chunks retain fewer previous logits. Evaluate
        # only the last full and final partial chunk, independent of n.
        if full:
            previous = (full - 1) * g.candidate_chunk
            phases['candidate_chunk_full'] = (scoring + 4 * previous
                                               + g.candidate_chunk * g.chunk_per_candidate)
        if partial:
            phases['candidate_chunk_partial'] = (scoring + 4 * full * g.candidate_chunk
                                                  + partial * g.chunk_per_candidate)
        phases['candidate_concat'] = scoring + 8 * candidates
        phases['output_logsoftmax'] = g.shared_bytes + 8 * candidates
        phases['output_critic'] = g.shared_bytes + 8 * candidates + g.critic_scratch
        # Scores + logs + value survive; isfinite(logs) peaks at FP32 abs + bool.
        phases['output_finite'] = g.shared_bytes + 8 * candidates + 4 + 5 * candidates
        # Frame context/cards and scores have retired. Probability transfer is
        # synchronous; provisional_sample validates/samples entirely on CPU.
        phases['output_probability_cpu_transfer'] = 8 * candidates + 4
        phases['output_logs_cpu_return'] = 4 * candidates + 4
        phases['output_value_cpu_return'] = 4
        peak_phase = max(phases, key=lambda phase: phases[phase])
        peak = phases[peak_phase]
        return NativeMlpRowCudaEstimate(
            recipe=NATIVE_MLP_ROW_CUDA_RECIPE, input_leaf_bytes=input_leaf_bytes,
            additional_phases=MappingProxyType(phases), peak_additional_bytes=peak,
            peak_phase=peak_phase, peak_bytes=input_leaf_bytes + peak,
        )
