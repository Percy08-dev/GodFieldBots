"""Pure non-group CUDA minima and a distinct normal owned learner allowance.

These retain configured context/workspace, and bound the supported canonical
Adam foreach and single-tensor paths. They are managed estimates, not measured
process memory. Logical update geometry remains a separate unchanged ceiling.
"""
from __future__ import annotations

from math import prod

from gf_common.domain import ContractError, Json, integer
from gf_common.structured_layout import StructuredLayout
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV4,
    parameter_shapes,
    resource_estimate,
)

OWNED_LEARNER_RECIPE = 'owned-cuda-learner-admission-v1'


def learner_group_fixed_bytes(model: ModelConfigV4, binding: InferenceBindingV2, *,
                              layout: StructuredLayout) -> int:
    """The unchanged logical planner's fixed components, shared with its floor."""
    fixed = resource_estimate(model,binding,decisions=1,interval_events=0,candidates=1,
        suffix_decisions=1,layout=layout)
    return fixed['parameter_optimizer_cpu_reference_bytes'] + fixed['context_workspace_bytes']


def learner_group_metadata_bytes(frames: int, endpoints: int) -> int:
    return 65536 + frames * 512 + endpoints * 1024


def learner_cuda_minima(model: ModelConfigV4, binding: InferenceBindingV2, *,
                        layout: StructuredLayout) -> Json:
    """Position buffer is absent from parameter_shapes but live in every network."""
    if type(model) is not ModelConfigV4 or binding.device != 'cuda:0':
        raise ContractError('owned CUDA minima require exact Model4 CUDA binding')
    shapes = parameter_shapes(model,layout=layout)
    weights = 4 * sum(prod(shape) for shape in shapes.values())
    largest = 4 * max(prod(shape) for shape in shapes.values())
    positions = 4 * model.temporal.max_decisions * model.temporal.hidden_size
    fixed = model.resources.cuda_context_bytes + model.resources.workspace_bytes
    # Canonical Adam has CPU FP32 steps; CUDA RNG is exported to CPU bytes.
    # Five FP64 metric accumulators, grad norms/list+stack, norm/check scalars.
    scalars = 5 * 8 + 2 * len(shapes) * 4 + 16
    finite = largest + largest // 4  # FP32 abs/finite construction and bool mask.
    optimizer = max(weights,2 * largest)  # foreach sqrt list or sqrt/div denominator pair.
    phases = {
        'initialize': fixed + weights + positions,
        'restore': fixed + 4 * weights + 2 * positions + finite + scalars,
        'optimizer': fixed + 4 * weights + positions + optimizer + scalars,
        'export': fixed + 4 * weights + positions + finite + scalars,
        # Every accepted nonempty training update needs >=1 frame/endpoint.
        # No assumed future events/candidates/raw activations enter this floor.
        'nonempty_update_floor': learner_group_fixed_bytes(model,binding,layout=layout)
            + positions + learner_group_metadata_bytes(1,1),
    }
    return {'parameter_bytes':weights,'position_bytes':positions,
        'context_workspace_bytes':fixed,'optimizer_foreach_bytes':weights,
        'optimizer_single_denominator_bytes':2 * largest,'finite_check_bytes':finite,
        'scalar_cuda_bytes':scalars,'optimizer_step_cpu_bytes':4 * len(shapes),
        'optimizer_step_cuda_bytes':0,'export_rng_cuda_bytes':0,'phases':phases}


def owned_learner_cuda_admission(model: ModelConfigV4, binding: InferenceBindingV2, *,
                                  logical_bytes: int, global_cuda_bytes: int,
                                  pool_cuda_bytes: int, overlap_scratch_bytes: int,
                                  layout: StructuredLayout) -> Json:
    for value in (logical_bytes,global_cuda_bytes):
        integer(value,1,2**63-1)
    for value in (pool_cuda_bytes,overlap_scratch_bytes):
        integer(value,0,2**63-1)
    future = pool_cuda_bytes + overlap_scratch_bytes
    if future >= global_cuda_bytes:
        raise ContractError('owned future pool exceeds CUDA allowance before learner: '
            f'resource=cuda phase=pool required={future} limit={global_cuda_bytes} '
            f'workers={pool_cuda_bytes} overlap_scratch={overlap_scratch_bytes}')
    device = min(logical_bytes,global_cuda_bytes-future)
    minima = learner_cuda_minima(model,binding,layout=layout)
    phase = max(minima['phases'],key=minima['phases'].__getitem__)
    if device < minima['phases'][phase]:
        raise ContractError('owned learner phase cannot fit device allowance: '
            f'resource=cuda phase={phase} required={minima["phases"][phase]} limit={device} '
            f'logical={logical_bytes} workers={pool_cuda_bytes} overlap_scratch={overlap_scratch_bytes}')
    return {'schema_version':1,'recipe':OWNED_LEARNER_RECIPE,'logical_bytes':logical_bytes,
        'device_bytes':device,'global_cuda_bytes':global_cuda_bytes,'pool_cuda_bytes':pool_cuda_bytes,
        'overlap_scratch_bytes':overlap_scratch_bytes,'phase_minima':minima}
