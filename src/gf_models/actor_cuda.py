"""Pure soft device estimate for one normal, no-grad, single-row Model4 actor.

The phases follow structured_network and structured_policy. Views do not get a
second storage charge. Native backend/allocator demand remains the configured
workspace/context allowance; this is not a physical peak guarantee. Loading is
owned separately by the resident BEGIN/SYNC reservation, never this estimate.
"""
from __future__ import annotations

from itertools import pairwise
from math import prod

from gf_common.domain import ContractError, Json
from gf_common.structured_layout import StructuredLayout
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV3,
    ModelConfigV4,
    dimensions,
    parameter_shapes,
    resource_estimate,
)

ACTOR_CUDA_RECIPE = 'model4-single-row-actor-cuda-v1'


def _mlp_scratch(widths: tuple[int, ...], rows: int = 1) -> int:
    """Caller retains input; linear input/output or activation input/output overlap."""
    return 4 * rows * max(max((source if i else 0) + target, 2 * target)
                         for i, (source, target) in enumerate(pairwise(widths)))


def _onehot_peak(rows: int, categories: tuple[tuple[str, ...], ...]) -> int:
    if not categories:
        return 0
    widths = [len(values) + 2 for values in categories]
    raw = 8 * rows * len(widths)
    floats = 4 * rows * sum(widths)
    # Raw int64 indices remain during expansion. At conversion, one int64
    # expansion coexists with accumulated FP32 columns; cat then copies columns.
    return raw + max(floats + 8 * rows * max(widths), 2 * floats)


def _attention_scratch(width: int, heads: int, feedforward: int, queries: int, keys: int) -> int:
    """New Q/K/V, MATH attention, then residual/norm/feedforward; input is external."""
    q = 4 * queries * width
    qkv = 3 * q  # step projects one new key/value; full cached KV is external.
    scores = heads * queries * keys
    # MATH scores, softmax, safe-softmax output; bool validity and row masks.
    attend = qkv + q + 3 * 4 * scores + scores + heads * queries + 2 * q
    # Q/K/V and attended survive the finish call. Residual and norm survive
    # linear/activation overlap; the last residual addition has two new outputs.
    finish = qkv + q + 2 * q + max(2 * 4 * queries * feedforward,
                                  4 * queries * feedforward + 2 * q)
    return max(qkv + q, attend, finish)


def actor_cuda_estimate(model: ModelConfigV4, binding: InferenceBindingV2, *,
                        layout: StructuredLayout) -> Json:
    """Keep all legacy shape/managed guards; change no policy-visible costs."""
    if type(model) not in (ModelConfigV3,ModelConfigV4):
        raise ContractError('actor device estimate requires homogeneous Model4')
    resource_estimate(model, binding, layout=layout, suffix_decisions=1)
    d = dimensions(model, layout=layout)
    h, layers = model.temporal.hidden_size, model.temporal.num_layers
    eh, el, events = model.event.hidden_size, model.event.num_layers, model.event.max_interval
    count, chunk, positions = binding.max_candidates, min(binding.candidate_chunk, binding.max_candidates), model.max_positions
    actor = model.heads.actor_hidden_sizes
    a = actor[0]
    card_width = model.card.output_size
    membership = len(layout.card_categories[layout.membership_index]) + 2
    state_input = 4 * layout.state_width
    state = 0 if model.state.kind == 'identity' else 4 * d['state']
    cats = 8 * 18 * len(layout.card_categories)
    numbers = 4 * 18 * 2 * len(layout.card_numeric_fields)
    card_input, cards = 4 * 18 * d['card_input'], 4 * 18 * card_width
    source, order, mask = 4 * 18 * membership, 4 * 18, 18
    actor_cards = 4 * 18 * d['card_actor']
    parts = {
        'state_input_fp32': state_input, 'state_output_fp32': state,
        'card_categories_int64': cats, 'card_numbers_fp32': numbers,
        'card_input_fp32': card_input, 'cards_fp32': cards,
        'card_source_fp32': source, 'card_order_fp32': order, 'card_mask_bool': mask,
        'actor_cards_fp32': actor_cards,
    }
    prefix = sum(parts.values())
    opponent = 18 * (len(layout.card_categories[0]) + 2)
    state_peak = max(2 * state_input,
        state_input + 8 * opponent + 8 * 18,
        _onehot_peak(1, layout.state_categories) + 4 * len(layout.state_numeric_fields),
        4 * 4 * len(layout.state_numeric_fields),  # raw/sign/abs/log or raw/sign/log/product.
        state_input + (0 if model.state.kind == 'identity' else
            _mlp_scratch((layout.state_width, *model.state.hidden_sizes, model.state.output_size))),
        state_input + 4 * opponent)  # repeated opponent mask, multiplied in place.
    card_common = state_input + state + cats + numbers
    card_peak = max(card_common + 2 * card_input - numbers,
        card_common + 4 * 18 * (model.card.definition_embedding + model.card.ability_embedding)
            + _onehot_peak(18, layout.card_categories[2:]),
        card_common + card_input + mask + 2 * cards,
        prefix + 4 * 18,  # mask.float() while actor_cards cat copies its inputs.
        prefix - source - order - actor_cards + 12 * 18 * membership,
        state_input + state + cats + 4 * 4 * 18 * len(layout.card_numeric_fields))
    hand_extra = cards if model.hand.kind == 'attention' else 0
    hand_kv = 2 * cards if model.hand.kind == 'attention' else 0
    if model.hand.kind == 'direct':
        hand = 4 * d['hand']
        latent = 0
        hand_peak = prefix + max(_onehot_peak(18, layout.card_categories), 2 * hand)
    else:
        latent, hand = 4 * 18 * model.hand.latent_size, 4 * model.hand.latent_size
        encoder = _mlp_scratch((card_width, *model.hand.hidden_sizes, model.hand.latent_size), 18)
        pooling = (2 * latent + hand + 8 * model.hand.latent_size if model.hand.pooling == 'max'
                   else latent + 2 * hand + 16)
        hand_peak = prefix + hand_extra + hand_kv + max(encoder, 2 * latent, pooling)
        if model.hand.kind == 'attention':
            # The loop retains its old output/KV while the next layer runs.
            previous = cards + hand_kv if model.hand.attention_layers > 1 else 0
            hand_peak = max(hand_peak, prefix + previous + _attention_scratch(
                card_width, model.hand.attention_heads, model.hand.attention_feedforward, 18, 18))
    frame_retained = prefix + hand_extra + hand_kv + latent + hand
    event_input = 4 * events * layout.event_width
    event_sequence = 4 * events * eh
    # Pinned ATen/native/RNN.cpp FullLayer precomputes full input gates on CPU.
    # CUDA uses per-step GRUCell input/hidden transforms + fused-cell workspace.
    full_gates = 4 * events * 3 * eh if binding.device == 'cpu' else 0
    previous_layer = event_sequence if el > 1 else 0
    event_gru = 3 * 4 * el * eh + previous_layer + max(
        2 * event_sequence, event_sequence + full_gates + 4 * (3 + 3 + 5) * eh)
    parts.update({
        'event_input_fp32': event_input,
        'event_sequence_outputs_and_stack_fp32': 2 * event_sequence,
        'event_previous_layer_output_fp32': previous_layer,
        'event_full_precomputed_input_gates_fp32': full_gates,
        'event_input_hidden_gates_fused_workspace_fp32': 4 * (3 + 3 + 5) * eh,
        'event_initial_final_stacked_hidden_fp32': 3 * 4 * el * eh,
    })
    event_pack = max(2 * event_input,
        8 * events * len(layout.event_numeric_fields) + _onehot_peak(events, layout.event_categories),
        4 * 4 * events * len(layout.event_numeric_fields))
    event_peak = frame_retained + max(event_pack, event_input + event_gru)
    # `_` keeps the final GRU time output; its assignment retires hand K/V.
    projection_retained = frame_retained - hand_kv + event_input + event_sequence + 4 * el * eh
    frame_projection = projection_retained + 4 * d['frame'] + max(
        4 * 4 * layout.history_width, 2 * 4 * h,
        4 * h + (_mlp_scratch((h,) * layers) if model.model_id == 'mlp' and layers > 1 else 0))
    shared = 4 * h + actor_cards
    if model.model_id == 'transformer':
        temporal = 2 * layers * model.temporal.max_decisions * h * 4
        temporal_scratch = actor_cards + 2 * 4 * h + _attention_scratch(h,
            model.temporal.num_heads, model.temporal.feedforward_size, 1, model.temporal.max_decisions)
    elif model.model_id == 'gru':
        temporal = layers * h * 4
        # Old hidden is persistent. New per-layer hidden, final stack and one
        # sequence output overlap native input/hidden gates and fused workspace.
        temporal_scratch = shared + 4 * (2 * layers * h + h + (3 + 3 + 5) * h)
    else:
        temporal, temporal_scratch = 0, shared
    kind_width, target_width = len(layout.action_kinds) + 2, len(layout.targets) + 2
    features = 4 * count * layout.candidate_width
    feature_peak = max(2 * features,
        8 * count + 12 * count * kind_width,
        4 * count * kind_width + 8 * count + 12 * count * target_width,
        4 * count * (kind_width + target_width) + 4 * 4 * count * 4)
    projection, padding = 4 * positions * 19 * a, 4 * positions * a
    unpadded = 4 * positions * 18 * a
    weight_copy = 4 * positions * a * d['card_actor']
    refs, gather = 8 * chunk * positions, 4 * chunk * positions * a
    x = 4 * chunk * a
    tail = 4 * chunk * a + max(_mlp_scratch((*actor, 1), chunk),
                               4 * chunk * (actor[0] + (actor[1] if len(actor) > 1 else 1)))
    chunk_scratch = max(gather + 2 * x, 3 * x, tail)
    scoring_fixed = shared + 4 * a + features + projection + 8 * positions
    normalization = 5 * 4 * count
    parts.update({
        'shared_context_cards_fp32': shared, 'candidate_features_fp32': features,
        'candidate_onehot_peak_bytes': feature_peak, 'candidate_shared_projection_fp32': 4 * a,
        'position_card_projection_fp32': projection, 'projection_operand_copy_fp32': weight_copy,
        'chunk_references_int64': refs, 'chunk_card_gather_fp32': gather,
        'chunk_actor_tail_peak_fp32': tail,
        'all_chunk_logits_and_cat_fp32': 2 * 4 * count,
        'normalization_logits_logprob_exp_masked_product_fp32': normalization,
        'temporal_attention_scores_fp32': model.temporal.num_heads * model.temporal.max_decisions * 4
            if model.model_id == 'transformer' else 0,
    })
    phases = {
        'frame_state': state_peak, 'frame_cards': card_peak, 'frame_hand': hand_peak,
        'frame_events': event_peak, 'frame_projection': frame_projection,
        'temporal_step': temporal_scratch,
        'candidate_features': shared + 4 * a + feature_peak,
        'candidate_projection': shared + 4 * a + features + max(weight_copy + unpadded,
                                                                unpadded + padding + projection),
        'candidate_references': scoring_fixed + 4 * count + 2 * refs,
        'candidate_chunk': scoring_fixed + 4 * count + refs + chunk_scratch,
        'candidate_concat': scoring_fixed + refs + 2 * 4 * count,
        'normalization': shared + max(normalization, 4 * 4 * count + 2 * count) + 8,
        'critic_return': shared + 4 * count + 4 + max(
            _mlp_scratch((h, *model.heads.critic_hidden_sizes, 1)), 4 * count + 8, count + 8),
    }
    weights = 4 * sum(prod(shape) for shape in parameter_shapes(model, layout=layout).values())
    table = 4 * model.temporal.max_decisions * h
    transient = max(phases.values())
    workspace = model.resources.workspace_bytes
    context = model.resources.cuda_context_bytes if binding.device == 'cuda:0' else 0
    total = weights + table + temporal + workspace + context + transient
    return {'recipe': ACTOR_CUDA_RECIPE, 'parameter_bytes': weights, 'position_bytes': table,
        'temporal_bytes': temporal, 'workspace_bytes': workspace, 'context_bytes': context,
        'transient_bytes': transient, 'transient_phases': phases, 'components': parts,
        'cuda_bytes': total if binding.device == 'cuda:0' else 0}
