"""Role-aware shared MLP estimates; managed allowances, not physical quotas."""
from __future__ import annotations

from math import prod
from typing import Any

from gf_common.domain import ContractError, Json
from gf_common.shared_inference import SHARED_ADMISSION, shared_runtime
from gf_models.structured_config import parameter_shapes


def shared_service_estimate(model: Any, binding: Any, runtime: Json, *, layout: Any,game_concurrency: int | None=None) -> Json:
    from gf_models.actor_cuda import actor_cuda_estimate
    from gf_models.shared_telemetry import DIAGNOSTIC_PAYLOAD_BYTES
    if model.model_id != 'mlp' or binding.device != 'cuda:0':
        raise ContractError('shared admission requires CUDA MLP')
    runtime = shared_runtime(runtime,game_concurrency=runtime.get('max_batch_size',1))
    actor = actor_cuda_estimate(model,binding,layout=layout)
    weight = 4*sum(prod(shape) for shape in parameter_shapes(model,layout=layout).values())
    positions = 4*model.temporal.max_decisions*model.temporal.hidden_size
    slots = min(runtime['cache_entries'],runtime['cache_weight_bytes']//weight)
    if slots < 1:
        raise ContractError('shared cache cannot admit one immutable checkpoint')
    # All raw leaves of every request cross to CUDA before the first row forward.
    state = 8*len(layout.state_numeric_fields)+12*len(layout.state_categories)+18*12
    cards = (8*18*len(layout.card_categories)+8*18*len(layout.card_numeric_fields)+18
        +8*18*(len(layout.card_categories)-2)+4*18)  # Derived category suffix and card-order leaves.
    history = 4*layout.history_width
    events = model.event.max_interval*(8*len(layout.event_numeric_fields)+8*len(layout.event_categories))
    candidates = binding.max_candidates*(16+32+8*model.max_positions)
    row_inputs = state+cards+history+events+candidates
    games=runtime['max_batch_size'] if game_concurrency is None else game_concurrency
    batch = min(runtime['max_batch_size'],games)
    row_output=8*binding.max_candidates+4+4*8  # Logs/probabilities, value, three adapter scalars and server version.
    outputs = batch*row_output
    raw = batch*row_inputs
    context = model.resources.cuda_context_bytes+model.resources.workspace_bytes
    resident = slots*(weight+positions)+context
    scratch = raw+max(actor['transient_bytes']+(batch-1)*row_output,outputs)
    publication = weight+positions+weight  # fresh network + CPU-to-device weight copy.
    return {'recipe':SHARED_ADMISSION,'cache_slots':slots,'parameter_bytes':weight,
        'position_bytes':positions,'context_workspace_bytes':context,'max_batch_size':runtime['max_batch_size'],
        'effective_max_batch_size':batch,'queued_numeric_input_bytes':games*row_inputs,
        'diagnostic_shared_payload_bytes':DIAGNOSTIC_PAYLOAD_BYTES,'row_input_bytes':row_inputs,'batch_input_bytes':raw,'retained_output_bytes':outputs,
        'row_workspace_bytes':actor['transient_bytes'],'resident_bytes':resident,
        'collection_scratch_bytes':scratch,'publication_scratch_bytes':publication,
        'cuda_bytes':resident+max(scratch,publication),
        'scope':'estimated managed tensor/context allowance; physical peaks require measurement'}


def shared_training_admission(model: Any,binding: Any,tbptt_length: int,operating: Json,
                              runtime: Json,*,layout: Any) -> Json:
    from gf_models.structured_runtime import _hpo_training_components
    base = _hpo_training_components(model,binding,tbptt_length,operating,layout=layout)
    service = shared_service_estimate(model,binding,runtime,layout=layout,game_concurrency=operating['game_concurrency'])
    # Host compatibility charge remains explicit, separate from reduced GPU ownership.
    workers = base['workers']
    host = base['worker_bytes']
    broker = service['queued_numeric_input_bytes']+service['retained_output_bytes']+runtime['cache_weight_bytes']+service['diagnostic_shared_payload_bytes']
    fixed = operating['parent_budget_bytes']+base['learner_bytes']+host+broker
    phases = {'collection':fixed+operating['collector_budget_bytes'],
        'update':fixed+operating['source_budget_bytes']+operating['numeric_budget_bytes'],
        'save':fixed+operating['source_budget_bytes']+operating['numeric_budget_bytes']+operating['save_budget_bytes'],
        'sync':fixed+runtime['cache_weight_bytes']}
    cuda = {phase:base['learner_bytes']+service['resident_bytes']+scratch for phase,scratch in (
        ('collection',service['collection_scratch_bytes']),('update',0),('save',0),('sync',service['publication_scratch_bytes']))}
    for resource,values,limit in (('ram',phases,operating['managed_ram_bytes']),('cuda',cuda,operating['managed_cuda_bytes'])):
        phase=max(values,key=values.__getitem__)
        if values[phase]>limit:
            raise ContractError(f'shared managed budget exceeded: resource={resource} phase={phase} required={values[phase]} limit={limit}')
    if operating.get('opponent_pool_size',5)+1>service['cache_slots']:
        raise ContractError('shared cache cannot retain complete frozen opponent inventory')
    return base | {'recipe':SHARED_ADMISSION,'worker_bytes':0,'worker_host_bytes':host,
        'workers':workers,'service':service,'broker_host_bytes':broker,'ram_phases':phases,
        'cuda_phases':cuda,'peak_ram_bytes':max(phases.values()),'peak_cuda_bytes':max(cuda.values())}
