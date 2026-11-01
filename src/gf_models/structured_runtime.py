"""Stdlib-only own-worker limits/costs and conservative resident/decision envelopes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import prod

from gf_common.diverse_actions import NumericCosts
from gf_common.domain import ContractError, Json, integer
from gf_common.structured import ActionConfigV2
from gf_models.structured_binding import safe_options_v2
from gf_models.structured_config import (
    InferenceBindingV2,
    dimensions,
    model_config_v2,
    parameter_shapes,
    resource_estimate,
)


@dataclass(frozen=True)
class WorkerCaps:
    max_interval_events: int = 4096
    max_episode_events: int = 262144
    max_decisions: int | None = None
    max_positions: int = 18
    max_candidates: int = 32768

    def __post_init__(self) -> None:
        integer(self.max_interval_events, 1, 65536)
        integer(self.max_episode_events, self.max_interval_events, 2**31 - 1)
        if self.max_decisions is not None:
            integer(self.max_decisions, 1, 1024)
        integer(self.max_positions, 1, 128)
        integer(self.max_candidates, 1, 65536)


def own_worker_settings(model_id: str, actions: ActionConfigV2, options: Json | None) -> tuple[WorkerCaps, NumericCosts, int, int]:
    if model_id in ("random", "heuristic"):
        if options is not None:
            raise ContractError("baseline has no checkpoint/runtime binding")
        return WorkerCaps(max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates), NumericCosts(), 0, 0
    if model_id not in ("mlp", "gru", "transformer") or options is None:
        raise ContractError("neural2 own settings required")
    safe = safe_options_v2(options, action_contract=actions.action_contract, purpose="component", synthetic_capacity=False)
    model = model_config_v2(safe["model"])
    binding = InferenceBindingV2(**safe["inference"])
    if model.model_id != model_id or actions.max_sequence_length > model.max_positions or actions.max_candidates > binding.max_candidates:
        raise ContractError("own model/actions capacity mismatch")
    caps = WorkerCaps(model.event.max_interval, model.event.max_episode, model.temporal.max_decisions,
                      actions.max_sequence_length, actions.max_candidates)
    dims = dimensions(model)
    widths = (*model.heads.actor_hidden_sizes, 1)
    tail_ops = model.heads.actor_hidden_sizes[0]
    for source, target in pairwise(widths):
        tail_ops += 2 * source * target + target
    tail_ops += sum(model.heads.actor_hidden_sizes[1:])
    sizes = [shape for shape in parameter_shapes(model).values()]
    weights = sum(prod(shape) for shape in sizes) * 4
    cache = (2 * model.temporal.num_layers * model.temporal.max_decisions * model.temporal.hidden_size * 4
             if model_id == "transformer" else model.temporal.num_layers * model.temporal.hidden_size * 4 if model_id == "gru" else 0)
    positions = model.temporal.max_decisions * model.temporal.hidden_size * 4
    resident = weights + cache + positions + model.resources.workspace_bytes + (model.resources.cuda_context_bytes if binding.device != "cpu" else 0)
    peak = resource_estimate(model, binding, suffix_decisions=1)["total_bytes"]
    costs = NumericCosts(context_width=model.temporal.hidden_size, card_width=dims["card_actor"],
                         actor_input_width=dims["actor"], actor_hidden_width=widths[0],
                         position_capacity=model.max_positions, candidate_chunk=binding.candidate_chunk,
                         tail_operations_per_candidate=tail_ops,
                         tail_working_bytes_per_candidate=sum(widths) * 32,
                         tail_tensor_bytes_per_candidate=max(widths) * 4,
                         fixed_bytes=resource_estimate(model, binding, candidates=1, suffix_decisions=1)["total_bytes"],
                         max_preparation_bytes=min(model.resources.max_owned_bytes, binding.managed_budget_bytes),
                         max_tensor_bytes=model.resources.max_tensor_bytes)
    cuda_resident = resident if binding.device != "cpu" else 0
    cuda_temporary = max(0, peak - resident) if binding.device != "cpu" else 0
    return caps, costs, cuda_resident, cuda_temporary


def caps_dict(value: WorkerCaps) -> Json:
    return asdict(value)
