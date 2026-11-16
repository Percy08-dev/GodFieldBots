"""HPO adapters to the existing original Source and custody owners."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, RuntimePolicyBinding, canonical, digest, integer
from gf_common.hpo import selection_context
from gf_models.structured_binding import policy_label_game_v4, safe_options_game_v4
from gf_models.structured_config import InferenceBindingV2
from gf_training.collection import CollectionPlan
from gf_training.hpo_collection import training_groups_hpo
from gf_training.hpo_config import hpo_training_config, normalize_training_for_model
from gf_training.hpo_lineage import hpo_spec, hpo_state, validate_hpo_chain
from gf_training.lineage import checkpoint_path
from gf_training.structured_lineage import _ChainScan


def original_inference(scan: _ChainScan, state: Json, record: Json, generation: int,
                        *, live: bool) -> tuple[Json, str]:
    """Use a receipt only for a deliberately deleted historical manifest.

    The returned SHA names the original bytes; G0 supplies unchanged public
    semantics when those bytes have an independently validated deletion receipt.
    """
    from gf_training.hpo_lineage import _receipt
    from gf_training.lineage import generation_path
    registry = scan.object(scan.root / 'artifact-registry.json')
    entry = registry['generations'][str(generation)]['files']['inference.json']
    expected = record['inference_sha256']
    status = _receipt(entry, expected)
    path = checkpoint_path(scan.root, generation) / 'inference.json'
    if status == 'retained':
        return scan.object(path, expected), expected
    if status != 'deleted' or live:
        raise ContractError('HPO live/missing/uncertain inference cannot use historical deletion evidence')
    # SAME structural scan must have admitted this exact immutable generation.
    if (canonical(scan.object(generation_path(scan.root, generation))) != canonical(record)
            or record['artifact_files']['inference.json'] != expected):
        raise ContractError('HPO historical manifest hash differs from immutable generation')
    if path.exists() or path.is_symlink():
        raise ContractError('HPO deleted historical inference was replaced')
    protected = scan.object(generation_path(scan.root, 0))
    return scan.object(checkpoint_path(scan.root, 0) / 'inference.json', protected['inference_sha256']), expected


def is_hpo(state: Json) -> bool:
    return (type(state.get('schema_version')) is int and state['schema_version'] == 5
        and state.get('run_variant') in ('hpo-ppo-v1', 'hpo-ppo-resident-v1','hpo-ppo-owned-v1') and state.get('model_version') == '4')


def hpo_binding(directory: Path, state: Json, record: Json, generation: int) -> tuple[str, RuntimePolicyBinding]:
    actions, model, _ = hpo_state(state)
    execution = state['execution']['inference']
    binding = InferenceBindingV2(**execution['binding'])
    from gf_models.structured_binding import input_binding_game_v3
    safe = safe_options_game_v4({'version': '4', 'checkpoint': str(checkpoint_path(directory, generation)),
        'expected_manifest_digest': record['inference_sha256'], 'model': asdict(model),
        'inference': asdict(binding), 'cuda_visible_devices': execution['cuda_visible_devices'],
        'action_config_hash': digest(asdict(actions)),
        **input_binding_game_v3(actions.action_contract, card_instance_rules())}, actions=actions, rules=card_instance_rules())
    label = policy_label_game_v4(model, binding, record['inference_sha256'],
        action_contract=actions.action_contract, rules=card_instance_rules(), action_config_hash=digest(asdict(actions)),
        cuda_visible_devices=execution['cuda_visible_devices'])
    return label, RuntimePolicyBinding(model.model_id, safe)


def _initial_schedule(state: Json, records: list[Json]) -> None:
    integer(state['seed'], 0, 2**63 - 1)
    initial = np.random.Generator(np.random.PCG64(state['seed']))
    expected = {'schedule_opponent_rng': initial.bit_generator.state,
        'snapshot_pool': [0], 'collection_position': 0}
    if not records or canonical(records[0]['parent_controls']) != canonical(expected):
        raise ContractError('HPO original generation0 schedule RNG differs from frozen learning seed')


def recorded_hpo_plan(directory: Path, state: Json, records: list[Json], generation: int,
                      deadline: float) -> tuple[CollectionPlan, str, Json]:
    hpo_state(state)
    _initial_schedule(state, records)
    if len(records) != state['updates'] + 1:
        raise ContractError('HPO plan requires the complete committed parent generations')
    integer(generation, 0, state['updates'])
    spec = hpo_spec(state)
    recipe = hpo_training_config(spec['training_recipe'])
    label, learner = hpo_binding(directory, state, records[generation], generation)
    registry = {'random-card-instance-v2': RuntimePolicyBinding('random'),
        'heuristic-card-instance-v2': RuntimePolicyBinding('heuristic'), label: learner}
    controls = records[generation]['parent_controls']
    snapshots = []
    for number in controls['snapshot_pool']:
        name, value = hpo_binding(directory, state, records[number], number)
        registry[name] = value
        snapshots.append(name)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = controls['schedule_opponent_rng']
    from gf_common.training_opponents import refresh_opponent, training_opponents
    mixed = training_opponents(state['effective_config'])
    if mixed is not None:
        from gf_common.card_instance import action_config_v3
        from gf_training.hpo_collection import training_groups_hpo_refresh
        actions = action_config_v3(state['effective_config']['actions'])
        name, binding = refresh_opponent(actions, mixed)
        registry[name] = binding
        groups = training_groups_hpo_refresh(state['seed'], generation, label, tuple(snapshots), rng,
            collection_games=recipe.collection_games, actions=actions, training_opponents=mixed)
    else:
        groups = training_groups_hpo(state['seed'], generation, label, tuple(snapshots), rng,
            collection_games=recipe.collection_games)
    from gf_training.card_instance_source import frozen_linkage
    link = frozen_linkage(state, directory, generation, groups, registry)
    link['selection_context'] = selection_context(decision_budget=spec['decision_budget'],
        used_before=records[generation]['used_decisions'], learning_seed=state['seed'], batch=generation)
    return CollectionPlan(groups, registry, directory / 'collections' / link['batch_id'], link,
        deadline, deadline), label, dict(rng.bit_generator.state)


@contextmanager
def open_hpo_batch(*args: Any, **options: Any) -> Iterator[Any]:
    """Thin diagnostic adapter; only the existing live issuer grants an update."""
    from gf_training.card_instance_source import _open_batch
    with _open_batch(*args, **options) as source:
        if not is_hpo(source.state):
            raise ContractError('explicit HPO Source variant required')
        yield source


@contextmanager
def hpo_batch_source(plan: CollectionPlan, ledger: Any, **options: Any) -> Iterator[Any]:
    from gf_training.card_instance_source import game_batch_source
    with game_batch_source(plan, ledger, **options) as source:
        if not is_hpo(source.state):
            raise ContractError('explicit HPO live Source required')
        yield source


def require_hpo_production(state: Json) -> None:
    _, model, _ = hpo_state(state)
    spec = hpo_spec(state)
    recipe = hpo_training_config(spec['training_recipe'])
    if (state['device'] != 'cuda:0' or state['execution']['inference']['binding']['device'] != 'cuda:0'
            or canonical(asdict(normalize_training_for_model(model.model_id, recipe))) != canonical(spec['training_recipe'])):
        raise ContractError('HPO production requires actual CUDA execution and normalized training recipe')
    if spec['storage']['numeric_root']!=spec['storage']['parent_root']:
        raise ContractError('HPO production currently requires colocated numeric and parent roots')
    if state['effective_config']['run']['output_dir']!=spec['storage']['parent_root']:
        raise ContractError('HPO production output root differs from sole storage spec')
    storage=state['effective_config'].get('storage')
    if storage is None or storage['budget_mib']*1024**2>spec['operating']['collector_budget_bytes']:
        raise ContractError('HPO production requires packed originals within collector RAM envelope')


def validate_hpo_parent(directory: Path, *, require_complete: bool = True, max_owned_bytes: int,
                         deadline: float, extra_owned_bytes: int = 0,
                         _scan: _ChainScan | None = None) -> tuple[Json, list[Json]]:
    """Full originals are replayed under the SAME structural scan; no formal authority."""
    from gf_common.preprocessing import preparation_config
    from gf_training.card_instance_source import _open_batch, recheck_game_parent_snapshot
    directory = directory.resolve()
    with ExitStack() as scope:
        scan = _scan or scope.enter_context(_ChainScan(directory=directory,
            max_owned_bytes=max_owned_bytes - extra_owned_bytes, deadline=deadline))
        state, records = validate_hpo_chain(directory, require_complete=require_complete,
            max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=extra_owned_bytes, scan=scan)
        if state['run_variant']=='hpo-ppo-owned-v1':
            from gf_training.hpo_owned_restore import audit_owned_samples
            require_hpo_production(state)
            audit_owned_samples(scan,state,records,deadline=deadline)
            return state,records
        declared=Path(hpo_spec(state)['storage']['parent_root']).resolve()
        if directory==declared or not directory.is_relative_to(declared):
            raise ContractError('HPO source parent is outside its declared colocated root')
        _initial_schedule(state, records)
        require_hpo_production(state)
        scan.configure_packed(state)
        settings = preparation_config(state['effective_config']['preprocessing'])
        from gf_common.training_opponents import training_opponents
        if training_opponents(state['effective_config']) is not None:
            from gf_training.native_backend import require_native_refresh_opponents
            require_native_refresh_opponents()
        if settings.backend == 'rust':
            from gf_training.native_backend import hpo_reader_settings
            hpo_reader_settings(settings, packed='storage_binding' in state, resident=state.get('run_variant') == 'hpo-ppo-resident-v1')
        inventories = []
        for generation in range(state['updates']):
            plan, _, after = recorded_hpo_plan(directory, state, records, generation, deadline)
            if canonical(after) != canonical(records[generation + 1]['parent_controls']['schedule_opponent_rng']):
                raise ContractError('HPO original opponent RNG schedule differs')
            update = scan.object(directory / 'updates' / f'{generation + 1:06}.json', records[generation + 1]['update_sha256'])
            with _open_batch(plan, state, records, scan, settings, online=None,
                    max_owned_bytes=max_owned_bytes, deadline=deadline, extra_owned_bytes=extra_owned_bytes) as source:
                if canonical(source.proof) != canonical(update['source_proof']):
                    # Packed custody includes current process-local observations;
                    # compare physical custody through its established helper.
                    if 'storage_binding' not in state:
                        raise ContractError('HPO original Source proof differs from saved update')
                    from gf_common.group_pack_game import same_storage_custody
                    left, right = source.proof, update['source_proof']
                    if (not same_storage_custody(left['storage'], left['custody'], right['storage'], right['custody'])
                            or canonical({k: v for k, v in left.items() if k not in ('storage', 'custody')})
                            != canonical({k: v for k, v in right.items() if k not in ('storage', 'custody')})):
                        raise ContractError('HPO packed original Source proof differs')
                rollout = source.read_numeric(directory / update['numeric'], update['numeric_sha256'])
                rollout.require_learning()
                expected_context = learner_context_costs(source.selection, hpo_training_config(hpo_spec(state)['training_recipe']))
                if any(update['metrics'][key] != value for key, value in expected_context.items()):
                    raise ContractError('HPO original selected-prefix compute counts differ')
                del rollout
                if source.snapshot is not None:
                    original = source.snapshot._owner
                    for name, entry in original.files.items():
                        scan.charge(1024 + 4 * len(name))
                        path = original.root / name
                        if 'storage_binding' in state:
                            if path in scan.logical_files and scan.logical_files[path] != entry:
                                raise ContractError('HPO original logical Source changed across cohorts')
                            scan.logical_files[path] = entry
                        else:
                            value = (tuple(entry['identity']), entry['sha256'])
                            if path in scan.files and scan.files[path] != value:
                                raise ContractError('HPO original changed across cohort snapshots')
                            scan.files[path] = value
                    if 'storage_binding' not in state:
                        from gf_training.structured_rollout import _owned_bytes
                        scan.charge(_owned_bytes(original.inventories) + 512 + 4 * len(str(original.root)))
                        inventories.append((original.root, tuple(original.inventories)))
        # Preserve current actual originals and scan snapshots through caller scope.
        scan._game_source_inventories = inventories
        scan._game_source_settings = settings
        scan._game_source_extra_bytes = extra_owned_bytes
        scan._game_source_max_owned_bytes = max_owned_bytes
        recheck_game_parent_snapshot(scan)
        scan.finalize()
        return state, records


def learner_context_costs(selection: Json, training: Any) -> Json:
    from gf_common.hpo import selected_locations
    positions = [position + 1 for _, position in selected_locations(selection)]
    return {'context_decisions': training.epochs * sum(positions),
        'suffix_decisions': training.epochs * sum(min(position, training.tbptt_length) for position in positions)}
