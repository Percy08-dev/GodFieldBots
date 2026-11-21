"""Actor-only tensor geometry, independently checked against actual CPU networks."""

from dataclasses import asdict, replace

import pytest

from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.structured_config import InferenceBindingV2


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_persistent_actor_bytes_match_real_network_and_active_temporal_storage(kind):
    pytest.importorskip('torch')
    from gf_models.actor_cuda import actor_cuda_estimate
    from gf_models.structured_network import make_network
    from tests.test_hpo_network import small_model4

    model = small_model4(kind, depth=3)
    cpu = InferenceBindingV2(device='cpu', max_candidates=64, candidate_chunk=2)
    network = make_network(model, 17, cpu, layout=LAYOUT)
    estimate = actor_cuda_estimate(model, replace(cpu, device='cuda:0'), layout=LAYOUT)
    assert estimate['parameter_bytes'] == sum(t.numel() * t.element_size() for t in network.parameters())
    assert estimate['position_bytes'] == network.positions.numel() * network.positions.element_size()
    cache = network.new_cache()
    if kind == 'transformer':
        temporal = sum(t.numel() * t.element_size() for t in (cache.keys, cache.values))
    elif kind == 'gru':
        import torch
        with torch.no_grad():
            _, hidden = network.encode_temporal(torch.zeros(1, model.temporal.hidden_size))
        temporal = hidden.numel() * hidden.element_size()
    else:
        temporal = 0
    assert estimate['temporal_bytes'] == temporal
    assert estimate['workspace_bytes'] == model.resources.workspace_bytes
    assert estimate['context_bytes'] == model.resources.cuda_context_bytes
    assert estimate['transient_bytes'] == max(estimate['transient_phases'].values())
    assert estimate['cuda_bytes'] == sum(estimate[key] for key in (
        'parameter_bytes', 'position_bytes', 'temporal_bytes', 'workspace_bytes', 'context_bytes', 'transient_bytes'))
    assert actor_cuda_estimate(model, cpu, layout=LAYOUT)['cuda_bytes'] == 0


def test_candidate_scoring_keeps_full_features_global_normalization_and_chunked_gathers():
    from gf_models.actor_cuda import actor_cuda_estimate
    from gf_models.structured_config import dimensions
    from tests.test_hpo_network import small_model4

    model = small_model4()
    binding = InferenceBindingV2(device='cuda:0', max_candidates=64, candidate_chunk=2)
    estimate = actor_cuda_estimate(model, binding, layout=LAYOUT)
    parts = estimate['components']
    assert parts['candidate_features_fp32'] == 64 * LAYOUT.candidate_width * 4
    assert parts['chunk_references_int64'] == 2 * model.max_positions * 8
    assert parts['chunk_card_gather_fp32'] == 2 * model.max_positions * model.heads.actor_hidden_sizes[0] * 4
    assert parts['normalization_logits_logprob_exp_masked_product_fp32'] == 64 * 5 * 4
    assert parts['shared_context_cards_fp32'] == 4 * (model.temporal.hidden_size + 18 * dimensions(model, layout=LAYOUT)['card_actor'])
    wide = actor_cuda_estimate(model, replace(binding, candidate_chunk=64), layout=LAYOUT)
    assert wide['components']['candidate_features_fp32'] == parts['candidate_features_fp32']
    assert wide['components']['chunk_card_gather_fp32'] == 64 * model.max_positions * model.heads.actor_hidden_sizes[0] * 4
    assert wide['transient_phases']['candidate_chunk'] > estimate['transient_phases']['candidate_chunk']


def test_event_sequence_and_attention_bounds_follow_maximum_geometry_without_changing_costs():
    from gf_common.domain import digest
    from gf_models.actor_cuda import actor_cuda_estimate
    from gf_models.structured_binding import input_binding_game_v3
    from gf_models.structured_runtime import own_game_worker_settings_v3
    from tests.test_card_instance_learning import ACTIONS, RULES
    from tests.test_hpo_network import small_model4

    model = small_model4('transformer', depth=3)
    model = replace(model, event=replace(model.event, num_layers=3))
    binding = InferenceBindingV2(device='cuda:0', max_candidates=64, candidate_chunk=2)
    estimate = actor_cuda_estimate(model, binding, layout=LAYOUT)
    parts = estimate['components']
    assert parts['event_input_fp32'] == model.event.max_interval * LAYOUT.event_width * 4
    assert parts['event_sequence_outputs_and_stack_fp32'] == 2 * model.event.max_interval * model.event.hidden_size * 4
    assert parts['event_previous_layer_output_fp32'] == model.event.max_interval * model.event.hidden_size * 4
    assert parts['temporal_attention_scores_fp32'] == model.temporal.num_heads * model.temporal.max_decisions * 4
    # Admission must remain separate from policy-visible candidate/search costs.
    options = {'version': '4', 'checkpoint': '/tmp/actor-estimate-fixture',
        'expected_manifest_digest': 'a' * 64, 'model': asdict(model), 'inference': asdict(binding),
        'cuda_visible_devices': '0', 'action_config_hash': digest(asdict(ACTIONS)),
        **input_binding_game_v3(ACTIONS.action_contract, RULES)}
    before = own_game_worker_settings_v3('transformer', ACTIONS, options, rules=RULES)
    saved = asdict(model)
    actor_cuda_estimate(model, binding, layout=LAYOUT)
    assert asdict(model) == saved
    assert own_game_worker_settings_v3('transformer', ACTIONS, options, rules=RULES) == before
