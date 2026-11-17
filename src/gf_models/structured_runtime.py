"""Stdlib-only own-worker limits/costs and conservative resident/decision envelopes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import prod

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.diverse_actions import NumericCosts
from gf_common.domain import ContractError, Json, RuntimePolicyBinding, canonical, integer
from gf_common.structured import ActionConfigV2
from gf_common.structured_layout import LEGACY_LAYOUT, StructuredLayout, layout_for_contract
from gf_models.structured_binding import (
    input_binding_game_v3,
    safe_options_game_v3,
    safe_options_game_v4,
    safe_options_v2,
    safe_options_v3,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    ModelConfigV4,
    dimensions,
    model_config_v2,
    model_config_v3,
    parameter_shapes,
    resource_estimate,
)

PROCESS_ENVELOPE_RECIPE = "structured-process-envelope-v1"


def learner_process_bytes(model: ModelConfigV2 | ModelConfigV3, binding: InferenceBindingV2, *, layout: StructuredLayout = LEGACY_LAYOUT) -> int:
    """Full configured W envelope, charged until the numerical process is reaped."""
    peak = resource_estimate(model, binding, layout=layout)["total_bytes"]
    return peak if binding.device != "cpu" else 0


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
    if model_id == 'lightgbm' and options is not None:
        from gf_models.structured_tree_config import (
            TreeDescriptor,
            configured_tree_v2,
            numeric_costs,
        )
        from gf_models.structured_tree_encoder import safe_tree_options_v2
        safe = safe_tree_options_v2(options, action_contract=actions.action_contract, purpose='component', synthetic_capacity=False)
        tree = configured_tree_v2({**safe['model'], 'checkpoint': safe['checkpoint'],
            'inference': safe['inference'], 'resources': safe['resources']})
        if actions.max_sequence_length > tree.encoder.max_positions or actions.max_candidates > tree.inference.max_candidates:
            raise ContractError('Tree2 own model/actions capacity mismatch')
        return WorkerCaps(max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates), numeric_costs(tree, TreeDescriptor(**safe['tree_descriptor'])), 0, 0
    if model_id not in ("mlp", "gru", "transformer") or options is None:
        raise ContractError("neural2 own settings required")
    safe = safe_options_v2(options, action_contract=actions.action_contract, purpose="component", synthetic_capacity=False)
    model = model_config_v2(safe["model"])
    binding = InferenceBindingV2(**safe["inference"])
    if model.model_id != model_id or actions.max_sequence_length > model.max_positions or actions.max_candidates > binding.max_candidates:
        raise ContractError("own model/actions capacity mismatch")
    return _neural_settings(model, binding, actions)


def own_worker_settings_v3(model_id: str, actions: ActionConfigV3, options: Json, *, rules: CardInstanceRules) -> tuple[WorkerCaps, NumericCosts, int, int]:
    if type(actions) is not ActionConfigV3:
        raise ContractError("explicit actions3 required")
    from gf_common.domain import digest
    safe = safe_options_v3(options, action_contract=actions.action_contract, rules=rules, purpose="component",
        synthetic_capacity=False, action_config_hash=digest(asdict(actions)))
    layout = layout_for_contract(actions.action_contract)
    model = model_config_v3(safe["model"], layout=layout)
    binding = InferenceBindingV2(**safe["inference"])
    if model.model_id != model_id or actions.max_sequence_length > model.max_positions or actions.max_candidates > binding.max_candidates:
        raise ContractError("own model3/actions capacity mismatch")
    return _neural_settings(model, binding, actions, layout=layout)


def _neural_settings(model: ModelConfigV2 | ModelConfigV3 | ModelConfigV4, binding: InferenceBindingV2,
                     actions: ActionConfigV2 | ActionConfigV3, *, layout: StructuredLayout = LEGACY_LAYOUT) -> tuple[WorkerCaps, NumericCosts, int, int]:
    caps = WorkerCaps(model.event.max_interval, model.event.max_episode, model.temporal.max_decisions,
                      actions.max_sequence_length, actions.max_candidates)
    dims = dimensions(model, layout=layout)
    widths = (*model.heads.actor_hidden_sizes, 1)
    tail_ops = model.heads.actor_hidden_sizes[0]
    for source, target in pairwise(widths):
        tail_ops += 2 * source * target + target
    tail_ops += sum(model.heads.actor_hidden_sizes[1:])
    peak = resource_estimate(model, binding, suffix_decisions=1, layout=layout)["total_bytes"]
    costs = NumericCosts(context_width=model.temporal.hidden_size, card_width=dims["card_actor"],
                         actor_input_width=dims["actor"], actor_hidden_width=widths[0],
                         position_capacity=model.max_positions, candidate_chunk=binding.candidate_chunk,
                         tail_operations_per_candidate=tail_ops,
                         tail_working_bytes_per_candidate=sum(widths) * 32,
                         tail_tensor_bytes_per_candidate=max(widths) * 4,
                         fixed_bytes=resource_estimate(model, binding, candidates=1, suffix_decisions=1, layout=layout)["total_bytes"],
                         max_preparation_bytes=min(model.resources.max_owned_bytes, binding.managed_budget_bytes),
                         max_tensor_bytes=model.resources.max_tensor_bytes)
    return caps, costs, peak if binding.device != "cpu" else 0, 0


def caps_dict(value: WorkerCaps) -> Json:
    return asdict(value)


def own_game_worker_settings_v3(model_id: str, actions: ActionConfigV3, options: Json | None, *,
                                rules: CardInstanceRules) -> tuple[WorkerCaps, NumericCosts, int, int]:
    if type(actions) is not ActionConfigV3 or actions.max_sequence_length > 18:
        raise ContractError("game worker requires real18 actions3")
    input_binding_game_v3(actions.action_contract, rules)
    if model_id == "heuristic_refresh":
        from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
        refresh_policy_label(actions, rules, parse_refresh_options(options if options is not None else {}))
        return WorkerCaps(max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates), NumericCosts(), 0, 0
    if model_id in ("random", "heuristic"):
        if options is not None:
            raise ContractError("baseline forbids policy options")
        return WorkerCaps(max_positions=actions.max_sequence_length, max_candidates=actions.max_candidates), NumericCosts(), 0, 0
    if model_id == "lightgbm" and options is not None:
        from gf_models.card_instance_game_tree_runtime import game_tree_settings
        return game_tree_settings(options, actions=actions, rules=rules)
    if model_id not in ("mlp", "gru", "transformer") or options is None:
        raise ContractError("neural game worker requires explicit safe options")
    v4 = type(options) is dict and options.get("version") == "4"
    safe_parser = safe_options_game_v4 if v4 else safe_options_game_v3
    safe = safe_parser(options, actions=actions, rules=rules)
    layout = layout_for_contract(actions.action_contract)
    from gf_models.hpo_config import model_config_v4
    parser = model_config_v4 if v4 else model_config_v3
    model, binding = parser(safe["model"], layout=layout), InferenceBindingV2(**safe["inference"])
    if model.model_id != model_id or actions.max_sequence_length > model.max_positions or actions.max_candidates > binding.max_candidates:
        raise ContractError("game model/actions capacity mismatch")
    return _neural_settings(model, binding, actions, layout=layout)


def admit_game_workers_v3(actions: ActionConfigV3, roles: tuple[RuntimePolicyBinding, RuntimePolicyBinding], *,
                          rules: CardInstanceRules, game_concurrency: int, global_cuda_bytes: int = 12 * 1024**3,
                          max_workers: int = 4, learner_cuda_bytes: int = 0) -> Json:
    integer(game_concurrency, 1, 2)
    integer(max_workers, 1, 4)
    integer(global_cuda_bytes, 1, 12 * 1024**3)
    return _admit_game_workers(actions,roles,rules=rules,game_concurrency=game_concurrency,
        global_cuda_bytes=global_cuda_bytes,max_workers=max_workers,learner_cuda_bytes=learner_cuda_bytes)


def _admit_game_workers(actions: ActionConfigV3,roles: tuple[RuntimePolicyBinding,RuntimePolicyBinding],*,
                        rules: CardInstanceRules,game_concurrency: int,global_cuda_bytes: int,
                        max_workers: int,learner_cuda_bytes: int) -> Json:
    integer(learner_cuda_bytes, 0, global_cuda_bytes)
    if type(roles) is not tuple or len(roles) != 2 or any(type(role) is not RuntimePolicyBinding for role in roles):
        raise ContractError("joint admission requires both configured player roles")
    settings = [own_game_worker_settings_v3(role.model_id, actions, role.policy_options, rules=rules) for role in roles]
    resident = game_concurrency * sum(row[2] for row in settings)
    temporary = game_concurrency * max(row[3] for row in settings)
    peak = learner_cuda_bytes + resident + temporary
    if 2 * game_concurrency > max_workers or peak > global_cuda_bytes:
        raise ContractError("joint game worker demand exceeds shared worker/CUDA budget")
    return {"recipe": PROCESS_ENVELOPE_RECIPE, "game_concurrency": game_concurrency,
        "workers": 2 * game_concurrency, "worker_resident_bytes": resident,
        "decision_temporary_bytes": temporary, "learner_cuda_bytes": learner_cuda_bytes,
        "peak_cuda_bytes": peak, "global_cuda_bytes": global_cuda_bytes, "max_workers": max_workers}


def admit_hpo_game_workers(actions: ActionConfigV3,roles: tuple[RuntimePolicyBinding,RuntimePolicyBinding],*,
                            rules: CardInstanceRules,game_concurrency: int,global_cuda_bytes: int,
                            max_workers: int,learner_cuda_bytes: int=0) -> Json:
    integer(game_concurrency,1,2**63-1)
    integer(max_workers,1,2**63-1)
    integer(global_cuda_bytes,1,2**63-1)
    if type(roles) is not tuple or len(roles)!=2:
        raise ContractError('HPO admission requires both atomic player roles')
    for role in roles:
        if type(role) is not RuntimePolicyBinding or role.model_id != 'heuristic_refresh' and role.policy_options is not None and role.policy_options.get('version')!='4':
            raise ContractError('HPO runtime admits exact model4 or canonical baseline roles')
    return _admit_game_workers(actions,roles,rules=rules,game_concurrency=game_concurrency,
        global_cuda_bytes=global_cuda_bytes,max_workers=max_workers,learner_cuda_bytes=learner_cuda_bytes)


def admit_hpo_resident_game_workers(actions: ActionConfigV3,
                                    roles: tuple[RuntimePolicyBinding,RuntimePolicyBinding], *,
                                    rules: CardInstanceRules, model: ModelConfigV4,
                                    inference: InferenceBindingV2, worker_cuda_bytes: int,
                                    begin_import_bytes: int, game_concurrency: int,
                                    global_cuda_bytes: int, max_workers: int,
                                    learner_cuda_bytes: int) -> Json:
    """Validate real roles, charging the SAME training pool including parked caches.

    Amounts come from the owning runtime pool, so explicit experimental buffers
    survive admission. A baseline in the current game cannot retire that pool's
    other cached neural identities or its next concurrent BEGIN allowance.
    """
    integer(game_concurrency,1,2**63-1)
    integer(max_workers,1,2**63-1)
    integer(global_cuda_bytes,1,2**63-1)
    integer(learner_cuda_bytes,0,global_cuda_bytes)
    integer(worker_cuda_bytes,0,global_cuda_bytes)
    integer(begin_import_bytes,0,global_cuda_bytes)
    if type(model) is not ModelConfigV4 or type(roles) is not tuple or len(roles)!=2:
        raise ContractError('homogeneous resident HPO admission requires both actual Model4 roles')
    for role in roles:
        if (type(role) is not RuntimePolicyBinding or role.model_id != 'heuristic_refresh'
                and role.policy_options is not None and role.policy_options.get('version')!='4'):
            raise ContractError('HPO runtime admits exact model4 or canonical baseline roles')
        # Keep all original syntax, action/model capacities and NumericCosts
        # validation; only the old returned device quantity is superseded.
        own_game_worker_settings_v3(role.model_id,actions,role.policy_options,rules=rules)
        if role.model_id in ('mlp','gru','transformer'):
            options = role.policy_options
            if (options is None or canonical(options['model']) != canonical(asdict(model))
                    or canonical(options['inference']) != canonical(asdict(inference))):
                raise ContractError('resident HPO training requires homogeneous model/inference roles')
    workers = 2 * game_concurrency
    resident, imports = workers * worker_cuda_bytes, workers * begin_import_bytes
    peak = learner_cuda_bytes + resident + imports
    if workers>max_workers or peak>global_cuda_bytes:
        raise ContractError('joint resident HPO game workers exceed managed budget: '
            f'resource=cuda phase=collection required={peak} limit={global_cuda_bytes} '
            f'learner={learner_cuda_bytes} workers={resident} begin_staging={imports}')
    return {'recipe':'hpo-resident-actor-envelope-v3','game_concurrency':game_concurrency,
        'workers':workers,'worker_resident_bytes':resident,'begin_import_bytes':imports,
        'decision_temporary_bytes':0,'learner_cuda_bytes':learner_cuda_bytes,
        'peak_cuda_bytes':peak,'global_cuda_bytes':global_cuda_bytes,'max_workers':max_workers}


def admit_hpo_training(actions: ActionConfigV3,model: ModelConfigV4,inference: InferenceBindingV2,
                        tbptt_length: int,operating: Json,*,rules: CardInstanceRules) -> Json:
    """Action-bound runtime entry to the shared pure HPO admission calculation."""
    input_binding_game_v3(actions.action_contract,rules)
    return hpo_training_admission(model,inference,tbptt_length,operating,
        layout=layout_for_contract(actions.action_contract))


def _hpo_training_components(model: ModelConfigV4,inference: InferenceBindingV2,tbptt_length: int,
                            operating: Json,*,layout: StructuredLayout) -> Json:
    """One conservative phase calculation for both CLI and actual runtime.

    Parent Source remains live through child save and acknowledgement inspection.
    The numeric allowance is retained conservatively too: deleting the worker's
    rollout does not prove all request/selection/array metadata has disappeared.
    """
    if type(model) is not ModelConfigV4:
        raise ContractError('explicit HPO Model4 required')
    integer(tbptt_length,1,1024)
    integer(operating['game_concurrency'],1,2**63-1)
    integer(operating['max_participant_workers'],1,2**63-1)
    integer(operating['managed_cuda_bytes'],1,2**63-1)
    for name in ('managed_ram_bytes','parent_budget_bytes','source_budget_bytes','numeric_budget_bytes',
            'save_budget_bytes','collector_budget_bytes'):
        integer(operating[name],1,2**63-1)
    learner=resource_estimate(model,inference,layout=layout,suffix_decisions=tbptt_length)['total_bytes']
    worker=resource_estimate(model,inference,layout=layout,suffix_decisions=1)['total_bytes']
    workers=2*operating['game_concurrency']
    if workers>operating['max_participant_workers']:
        raise ContractError('HPO concurrent roles exceed declared workers')
    total=workers*worker
    phases={'collection':operating['parent_budget_bytes']+learner+total+operating['collector_budget_bytes'],
        'update':operating['parent_budget_bytes']+learner+operating['source_budget_bytes']+operating['numeric_budget_bytes'],
        'save':operating['parent_budget_bytes']+learner+operating['source_budget_bytes']
            +operating['numeric_budget_bytes']+operating['save_budget_bytes']}
    peak_ram=max(phases.values())
    peak_cuda=learner+total
    return {'recipe':'hpo-joint-process-envelope-v1','workers':workers,'learner_bytes':learner,'worker_bytes':total,
        'ram_phases':phases,'peak_ram_bytes':peak_ram,'peak_cuda_bytes':peak_cuda,
        'managed_ram_bytes':operating['managed_ram_bytes'],'managed_cuda_bytes':operating['managed_cuda_bytes']}


def hpo_training_admission(model: ModelConfigV4,inference: InferenceBindingV2,tbptt_length: int,
                            operating: Json,*,layout: StructuredLayout) -> Json:
    """Original fresh-training quantities and refusal order remain authoritative."""
    base = _hpo_training_components(model,inference,tbptt_length,operating,layout=layout)
    phases, peak_ram, peak_cuda = base['ram_phases'], base['peak_ram_bytes'], base['peak_cuda_bytes']
    learner, total = base['learner_bytes'], base['worker_bytes']
    if peak_ram>operating['managed_ram_bytes']:
        phase = max(phases, key=phases.__getitem__)
        collecting = phase == 'collection'
        raise ContractError('joint HPO RAM demand exceeds managed budget: '
            f'resource=ram phase={phase} required={peak_ram} limit={operating["managed_ram_bytes"]} '
            f'learner={learner} workers={total if collecting else 0} parent={operating["parent_budget_bytes"]} '
            f'collector={operating["collector_budget_bytes"] if collecting else 0} '
            f'source={0 if collecting else operating["source_budget_bytes"]} '
            f'numeric={0 if collecting else operating["numeric_budget_bytes"]} '
            f'save={operating["save_budget_bytes"] if phase == "save" else 0}')
    if peak_cuda>operating['managed_cuda_bytes']:
        raise ContractError('joint HPO CUDA demand exceeds managed budget: '
            f'resource=cuda phase=collection required={peak_cuda} limit={operating["managed_cuda_bytes"]} '
            f'learner={learner} workers={total}')
    return base


def admit_planned_game_training_v3(actions: ActionConfigV3, model: ModelConfigV3,
                                  inference: InferenceBindingV2, learner: InferenceBindingV2, *,
                                  rules: CardInstanceRules, game_concurrency: int,
                                  global_cuda_bytes: int, max_workers: int) -> Json:
    """Pure pre-generation admission, using the exact live worker/learner formulas."""
    input_binding_game_v3(actions.action_contract, rules)
    integer(game_concurrency, 1, 2)
    integer(max_workers, 1, 4)
    integer(global_cuda_bytes, 1, 12 * 1024**3)
    if (type(model) is not ModelConfigV3 or model.max_positions > 18
            or actions.max_sequence_length > model.max_positions
            or actions.max_candidates > inference.max_candidates):
        raise ContractError('planned game model/actions capacity mismatch')
    layout = layout_for_contract(actions.action_contract)
    _, _, resident, temporary = _neural_settings(model, inference, actions, layout=layout)
    learner_bytes = learner_process_bytes(model, learner, layout=layout)
    workers = 2 * game_concurrency
    peak = learner_bytes + workers * resident + game_concurrency * temporary
    if workers > max_workers or peak > global_cuda_bytes:
        raise ContractError('planned learner and worst concurrent neural roles exceed shared budget')
    return {'recipe': PROCESS_ENVELOPE_RECIPE, 'game_concurrency': game_concurrency,
        'workers': workers, 'worker_resident_bytes': workers * resident,
        'decision_temporary_bytes': game_concurrency * temporary, 'learner_cuda_bytes': learner_bytes,
        'peak_cuda_bytes': peak, 'global_cuda_bytes': global_cuda_bytes, 'max_workers': max_workers}


def _hpo_resident_training_components(model: ModelConfigV4, inference: InferenceBindingV2,
                                     tbptt_length: int, operating: Json, runtime: Json, *,
                                     layout: StructuredLayout) -> Json:
    """Normal homogeneous actors: original host/learner, actor-specific CUDA.

    The configured cache limit also retains the uncalibrated construction/restore
    allowance per concurrent BEGIN, or for one owner in the serial SYNC loop.
    These are soft legacy reservation units, not measured physical CUDA bytes.
    """
    from gf_common.hpo_resident import resident_runtime
    limits = resident_runtime(runtime)
    from gf_models.actor_cuda import actor_cuda_estimate
    # Keep every original intrinsic tensor/managed guard, but enforce the final
    # resident device total rather than the obsolete learner-like actor gate.
    base = _hpo_training_components(model, inference, tbptt_length, operating, layout=layout)
    workers = base['workers']
    weights = 4 * sum(prod(shape) for shape in parameter_shapes(model, layout=layout).values())
    positions = 4 * model.temporal.max_decisions * model.temporal.hidden_size
    cached = min(limits['cache_entries'], limits['cache_weight_bytes'] // weights)
    if cached < 1:
        raise ContractError('resident cache cannot admit one network: resource=cache-weight '
            f'phase=resident-load required={weights} limit={limits["cache_weight_bytes"]}')
    cache = workers * (cached - 1) * weights
    position_cache = workers * cached * positions
    host_resident = base['worker_bytes'] + cache + position_cache
    actor = actor_cuda_estimate(model, inference, layout=layout)
    device_cache = workers * (cached - 1) * (weights + positions) if inference.device == 'cuda:0' else 0
    resident = workers * actor['cuda_bytes'] + device_cache
    staging = limits['cache_weight_bytes']
    begin_staging = workers * staging
    fixed = operating['parent_budget_bytes'] + base['learner_bytes']
    phases = {
        'collection': fixed + host_resident + operating['collector_budget_bytes'] + begin_staging,
        'update': fixed + host_resident + operating['source_budget_bytes'] + operating['numeric_budget_bytes'],
        'save': fixed + host_resident + operating['source_budget_bytes'] + operating['numeric_budget_bytes'] + operating['save_budget_bytes'],
        'sync': fixed + host_resident + staging,
    }
    cuda_phases = {phase: base['learner_bytes'] + resident + scratch
        for phase, scratch in (('collection', begin_staging), ('update', 0), ('save', 0), ('sync', staging))}
    cuda = max(cuda_phases.values())
    ram = max(phases.values())
    return {**base, 'recipe': 'hpo-resident-actor-envelope-v3', 'worker_bytes': resident,
        'worker_host_bytes': host_resident, 'worker_active_host_bytes': base['worker_bytes'],
        'worker_active_cuda_bytes': workers * actor['cuda_bytes'], 'worker_cache_cuda_bytes': device_cache,
        'actor_cuda_estimate': actor,
        'network_weight_bytes': weights, 'network_position_bytes': positions,
        'cached_networks_per_worker': cached, 'cache_weight_bytes': cache, 'cache_position_bytes': position_cache,
        'begin_staging_bytes': begin_staging, 'staging_bytes': staging, 'ram_phases': phases,
        'cuda_phases': cuda_phases,
        'peak_ram_bytes': ram, 'peak_cuda_bytes': cuda}


def hpo_resident_training_admission(model: ModelConfigV4, inference: InferenceBindingV2,
                                     tbptt_length: int, operating: Json, runtime: Json, *,
                                     layout: StructuredLayout, owned_cuda: bool = False) -> Json:
    """Final resident budgets; only explicitly normal owned CUDA gets a separate D."""
    result = _hpo_resident_training_components(model,inference,tbptt_length,operating,runtime,layout=layout)
    if owned_cuda and inference.device == 'cuda:0':
        from gf_models.learner_cuda import owned_learner_cuda_admission
        admission = owned_learner_cuda_admission(model,inference,layout=layout,
            logical_bytes=result['learner_bytes'],global_cuda_bytes=operating['managed_cuda_bytes'],
            pool_cuda_bytes=result['worker_bytes'],
            overlap_scratch_bytes=max(result['begin_staging_bytes'],result['staging_bytes']))
        result['owned_learner_admission'] = admission
        result['learner_device_bytes'] = admission['device_bytes']
        result['recipe'] = 'hpo-owned-device-envelope-v4'
        result['cuda_phases'] = {phase:value-result['learner_bytes']+admission['device_bytes']
            for phase,value in result['cuda_phases'].items()}
        result['peak_cuda_bytes'] = max(result['cuda_phases'].values())
    for resource, demands, limit in (('ram', result['ram_phases'], operating['managed_ram_bytes']),
                                     ('cuda', result['cuda_phases'], operating['managed_cuda_bytes'])):
        phase = max(demands,key=demands.__getitem__)
        if demands[phase] > limit:
            raise ContractError('joint resident HPO residency/transition exceeds managed budget: '
                f'resource={resource} phase={phase} required={demands[phase]} limit={limit} '
                f'legacy_learner={result["learner_bytes"]} legacy_workers={result["worker_active_host_bytes"]} '
                f'worker_host={result["worker_host_bytes"]} actor_cuda={result["worker_active_cuda_bytes"]} '
                f'extra_cache={result["cache_weight_bytes"]+result["cache_position_bytes"] if resource == "ram" else result["worker_cache_cuda_bytes"]} '
                f'begin_staging={result["begin_staging_bytes"] if phase == "collection" else 0} '
                f'sync_staging={result["staging_bytes"] if phase == "sync" else 0}')
    return result
