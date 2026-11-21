"""Role-aware shared MLP estimates; managed allowances, not physical quotas."""
from __future__ import annotations

from math import prod
from typing import Any

from gf_common.domain import ContractError, Json
from gf_common.shared_inference import SHARED_ADMISSION, shared_runtime
from gf_models.structured_config import parameter_shapes

HOST_RECIPE='shared-mlp-host-calibrated-v1'


def _input_leaf_bytes(model: Any,binding: Any,layout: Any) -> tuple[int,...]:
    """Exactly the numeric_row/tensor_row leaves, before device transfer."""
    events,count=model.event.max_interval,binding.max_candidates
    return (4*len(layout.state_numeric_fields),4*len(layout.state_numeric_fields),
        8*len(layout.state_categories),4*len(layout.state_categories),18*8,18*4,
        8*18*len(layout.card_categories),8*18*(len(layout.card_categories)-2),
        4*18*len(layout.card_numeric_fields),4*18*len(layout.card_numeric_fields),18,4*18,
        4*layout.history_width,4*events*len(layout.event_numeric_fields),
        4*events*len(layout.event_numeric_fields),8*events*len(layout.event_categories),
        8*count,8*count,16*count,16*count,8*count*model.max_positions)


def _host_components(model: Any,binding: Any,runtime: Json,service: Json,workers: int,games: int) -> Json:
    # Observed private high-waters and one aggregate empirical shared-PSS term.
    # The latter may overlap Source-backed pages; it is not proven disjoint
    # runtime allocation. File lengths cannot be subtracted from mapped PSS.
    # Their sum is a soft plan, not a simultaneous observed or hard peak.
    roles={'participants_private_bytes':workers*368377856,'shared_runtime_bytes':439175168,
        'service_private_bytes':1410199552,'broker_private_bytes':462737408,
        'coordinator_private_bytes':56020992,'tracker_private_bytes':6672384,
        'simulator_private_bytes':games*12427264}
    backings=games*(service['row_input_bytes']+service['row_output_bytes'])
    input_copy=service['largest_input_leaf_bytes']
    output_copies=games*service['largest_output_leaf_bytes']
    staging={'input_output_backings_bytes':backings,'input_handoff_bytes':input_copy,
        'output_handoff_bytes':output_copies,'shared_backing_count':1,'decoded_json_framing_included':False,
        'scope':'raw tensor increment plus simultaneous MP handoff copies; warm allocator reuse may absorb it; not a Python/framing bound'}
    calibration={'recipe':HOST_RECIPE,'scope':'calibrated soft planning estimate; not a full-process upper bound',
        'sources':['runs/shared-inference-components/v2/topology32-frozen.json',
            'runs/shared-inference-components/v3/controller-preload.json',
            'runs/shared-inference-components/v3/actual-c16-role-attribution.json'],
        'runtime':{'python':'3.14.10','gil_enabled':True,'torch':'2.14.1+cu132','torchrl':'0.15.0','tensordict':'0.15.2'},
        'basis':{'games':16,'participants':32,'max_batch_size':16,'max_candidates':32768,
            'max_interval_events':4096,'max_positions':18,'width':192,'parameter_bytes':6959624,'position_bytes':786432},
        'observed_occupancy':{'component_checkpoints':1,'actual_concurrent_checkpoints':2,
            'actual_updates':2,'configured_cache_entries':runtime['cache_entries'],
            'configured_cache_weight_bytes':runtime['cache_weight_bytes'],'filled_cache_measured':False},
        'extrapolation':{'model_payload_differs':service['parameter_bytes']!=6959624 or service['position_bytes']!=786432,
            'concurrency_differs':games!=16 or workers!=32,'batch_geometry_differs':service['effective_max_batch_size']!=16,
            'input_geometry_differs':binding.max_candidates!=32768 or model.event.max_interval!=4096 or model.max_positions!=18,
            'cache_capacity_exceeds_observed_occupancy':runtime['cache_entries']>2},
        'checkpoint_backing_owner':'source_budget_bytes','gpu_checkpoint_owner':'service.cuda_bytes',
        'shared_pss_source_overlap_possible':True,
        'shared_pss_scope':'aggregate empirical nonlearner PSS minus private; possible Source-backed overlap; not proven disjoint pure-runtime allocation',
        'broker_python_framing_bound':False,'full_history_search_shape_bound':False,
        'allocator_retention_scope':'includes observed startup/two-update retained pages; longer runs/full-cache/larger-model host occupancy unmeasured',
        'publication_scope':'CPU incoming weights and CPU positional construction coexist; separate Source backing and GPU cache excluded',
        'diagnostic_payload_counted_in_shared_runtime':True}
    return {'host_roles':roles,'host_calibration':calibration,'host_staging':staging,
        'runtime_host_bytes':sum(roles.values()),'broker_numeric_staging_bytes':backings+input_copy+output_copies,
        'publication_host_staging_bytes':service['parameter_bytes']+service['position_bytes']}


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
    # Keep the complete raw input batch before the first row-local forward.
    leaves=_input_leaf_bytes(model,binding,layout)
    row_inputs=sum(leaves)
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
        'row_output_bytes':row_output,'largest_input_leaf_bytes':max(leaves),'largest_output_leaf_bytes':max(4*binding.max_candidates,8),
        'row_workspace_bytes':actor['transient_bytes'],'resident_bytes':resident,
        'collection_scratch_bytes':scratch,'publication_scratch_bytes':publication,
        'cuda_bytes':resident+max(scratch,publication),
        'scope':'estimated managed tensor/context allowance; physical peaks require measurement'}


def shared_training_admission(model: Any,binding: Any,tbptt_length: int,operating: Json,
                              runtime: Json,*,layout: Any) -> Json:
    from gf_models.structured_runtime import _hpo_training_components
    base = _hpo_training_components(model,binding,tbptt_length,operating,layout=layout)
    service = shared_service_estimate(model,binding,runtime,layout=layout,game_concurrency=operating['game_concurrency'])
    workers,games=base['workers'],operating['game_concurrency']
    host=_host_components(model,binding,runtime,service,workers,games)
    fixed=base['learner_bytes']+host['runtime_host_bytes']
    source=operating['source_budget_bytes']
    phases={'collection':fixed+source+operating['collector_budget_bytes']+host['broker_numeric_staging_bytes'],
        'update':fixed+source+operating['numeric_budget_bytes'],
        'save':fixed+source+operating['numeric_budget_bytes']+operating['save_budget_bytes'],
        'sync':fixed+source+host['publication_host_staging_bytes']}
    cuda = {phase:base['learner_bytes']+service['resident_bytes']+scratch for phase,scratch in (
        ('collection',service['collection_scratch_bytes']),('update',0),('save',0),('sync',service['publication_scratch_bytes']))}
    for resource,values,limit in (('ram',phases,operating['managed_ram_bytes']),('cuda',cuda,operating['managed_cuda_bytes'])):
        phase=max(values,key=values.__getitem__)
        if values[phase]>limit:
            raise ContractError(f'shared managed budget exceeded: resource={resource} phase={phase} required={values[phase]} limit={limit}')
    if operating.get('opponent_pool_size',5)+1>service['cache_slots']:
        raise ContractError('shared cache cannot retain complete frozen opponent inventory')
    return base | host | {'recipe':SHARED_ADMISSION,'host_recipe':HOST_RECIPE,'worker_bytes':0,
        'worker_host_bytes':host['host_roles']['participants_private_bytes'],
        'workers':workers,'service':service,'broker_host_bytes':host['host_roles']['broker_private_bytes'],
        'persistent_host_bytes':fixed,'learner_host_scope':'unchanged full-learner compatibility allowance, not a calibrated physical prediction',
        'host_limits':{key:operating[key] for key in ('managed_ram_bytes','managed_cuda_bytes','parent_budget_bytes',
            'source_budget_bytes','numeric_budget_bytes','save_budget_bytes','collector_budget_bytes')}|{'parent_nested_in_source':True},
        'ram_phases':phases,'cuda_phases':cuda,'peak_ram_bytes':max(phases.values()),'peak_cuda_bytes':max(cuda.values())}
