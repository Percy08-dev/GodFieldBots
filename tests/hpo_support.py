"""Complete hand-authored originals for structural tests; no executed-game authority."""
import copy
import random
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from gf_common.diverse_actions import PublicCandidateGeneratorV3
from gf_common.domain import digest
from gf_common.schedule import serialize_schedule
from gf_common.structured import DecisionFrame, EventBank, Selection, event_delta
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.arrays import sha256
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_training.card_instance_journal import CardInstanceJournal
from gf_training.hpo_collection import hpo_collection_metadata
from gf_training.journal import atomic_json
from tests.card_instance_action_support import inputs
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES


def original_hpo_collection(plan, state, learner, *, decisions=True, logical_retirement=None):
    directory = plan.output
    directory.mkdir()
    owner = RetainedGameArtifacts.for_hpo_registry(plan.bindings, ACTIONS, pool_size=state['effective_config']['hpo_spec']['operating']['opponent_pool_size'], rules=RULES, max_owned_bytes=1024**3,
        preflight_deadline=time.monotonic() + 60, finalization_deadline=time.monotonic() + 120)
    results = []
    try:
        for group in plan.groups:
            for spec in group.episodes:
                folder = directory / spec.episode_id
                folder.mkdir()
                game = uuid.uuid4().hex
                result = {'episode_id': spec.episode_id, 'pair_id': spec.pair_id, 'seed': group.seed,
                    'policies': list(spec.policies), 'game_id': game, 'end_kind': 'natural', 'outcome': 'draw',
                    'failure': None, 'health': {}, 'simulator_abort_reason': None, 'replay_reconciled': True}
                for viewer in (1, 2):
                    label = spec.policies[viewer - 1]
                    binding = plan.bindings[label]
                    caps, costs, _, _ = own_game_worker_settings_v3(binding.model_id, ACTIONS, binding.policy_options, rules=RULES)
                    neural = None if binding.model_id in ('random', 'heuristic', 'heuristic_refresh') else owner.journal_metadata(label)
                    value = inputs([], can_pray=True)
                    if viewer == 2:
                        obs, hints = copy.deepcopy(value.observation), copy.deepcopy(value.action_options)
                        obs.update(viewer_player_id=2, current_player=2, active_player=2)
                        obs['own']['player_id'], obs['opponent']['player_id'] = 2, 1
                        hints['viewer_player_id'], hints['action_options']['actor'] = 2, 2
                        value = replace(value, observation=obs, action_options=hints)
                    bank = EventBank(viewer, spec.episode_id + f'-{viewer}', layout=LAYOUT)
                    delta = event_delta(viewer, bank.episode, 0, bank.boundary.prefix_hash,
                        [{'type': 'resource_changed', 'player': viewer, 'hp_delta': 0, 'mp_delta': 0, 'money_delta': 0}], layout=LAYOUT)
                    bank.append(delta)
                    value = replace(value, events=tuple(delta.events))
                    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
                    journal = CardInstanceJournal(folder, spec.episode_id, viewer, label, run_id='fixture-collection', game_id=game,
                        actions=ACTIONS, bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs),
                        model_id=binding.model_id, runtime_evidence=receipt(), sampling_seed=spec.sampling_seeds[viewer - 1], neural=neural, **({'baseline_options': binding.policy_options} if binding.model_id == 'heuristic_refresh' else {}))
                    accepted = []
                    if decisions and neural is not None:
                        seed = int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': spec.sampling_seeds[viewer - 1]})[:16], 16)
                        generated = PublicCandidateGeneratorV3(ACTIONS, rules=RULES, clock=lambda: 0.).generate_game(frame, bank.view(), random.Random(seed), costs)
                        tokens = tokenize_frame(frame, bank.view(), layout=LAYOUT)
                        candidates = tokenize_table(generated.table, frame, 18, layout=LAYOUT)
                        with learner.torch.no_grad():
                            logs, val, _ = learner.network((tokens,), candidates)
                        selection = Selection(generated.table.table_hash, 0, label, float(logs[0].cpu()), float(val.cpu()))
                        action = selection.validate(generated.table, frame)
                        command = {'command_id': str(uuid.uuid4()), 'expected_revision': 0, 'action': action}
                        number = journal.begin_frame(frame, delta, generated, selection, command, {})
                        journal.acknowledge(number, {'command_id': command['command_id'], 'previous_revision': 0,
                            'state_revision': 1, 'status': 'active', 'result': None})
                        accepted.append(action)
                    final = copy.deepcopy(value.observation)
                    final.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None,
                                 current_player=None, phase='terminal', result='draw')
                    journal.finish('natural', 'draw', final)
                    replay = {'game_id': game, 'viewer_player_id': viewer, 'requested_turn': None,
                        'rule_set_version': ACTIONS.action_contract.rule_set_version, 'observation_schema_version': '2',
                        'public_event_schema_version': '2', 'card_ref_scheme': ACTIONS.action_contract.card_ref_scheme,
                        'episode_end': {'type': 'terminal', 'result': 'draw'}, 'turns': []}
                    if accepted:
                        replay['turns'] = [{'turn': 1, 'active_player': viewer, 'decisions': [
                            {'actor': viewer, 'observation': value.observation, 'action_options': value.action_options['action_options'],
                             'visible_action': {'type': 'action', 'action': action}, 'visible_events': [], 'aborted': False} for action in accepted]}]
                    atomic_json(folder / f'replay-player-{viewer}.json', replay)
                atomic_json(folder / 'result.json', result)
                results.append(result)
        summary = {'schema_version': 1, 'planned_games': sum(len(g.episodes) for g in plan.groups), 'natural_denominator': sum(len(g.episodes) for g in plan.groups), 'comparison_valid': True,
            'failures': {}, 'health_failures': {}, 'group_health_failures': {}}
        atomic_json(directory / 'report.json', summary)
        inventory = [{'path': path.relative_to(directory).as_posix(), 'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'report.json']
        manifest = hpo_collection_metadata(actions=ACTIONS, config=state['effective_config'],
            runtime_evidence=[receipt() for _ in plan.groups], schedule=serialize_schedule(plan.groups, 4), inventory=inventory,
            policies=list(plan.bindings), bots_revision='fixture', policy_bindings=plan.bindings,
            neural_inputs=owner.finalize(deadline=time.monotonic() + 60), linkage=plan.linkage, logical_retirement=logical_retirement)
        manifest.update(kind='card-instance-collection', run_id='fixture-collection', status='complete',
            groups=[{'runtime_evidence': receipt()} for _ in plan.groups], episodes=results,
            inputs=state['config_inputs'], code=state['identity']['code'], report_sha256=sha256(directory / 'report.json'))
        atomic_json(directory / 'manifest.json', manifest)
        return manifest
    finally:
        owner.release()


def original_hpo_chain(directory, *, updates=3, model_id='mlp', decision_budget=None, numeric_budget_bytes=None, packed=False,
                       _unqualified_numeric_components=False):
    """Actual CPU numeric saves + complete hand-authored journals, unqualified.

    The explicit 1-byte native allocation fixture uses admitted numeric component
    APIs instead of the production Source writer. Its completed metadata is only
    structural test input; it cannot claim a valid completed 1-byte production
    parent. The ordinary fixture still exercises all Source phase guards.
    """
    if _unqualified_numeric_components and (numeric_budget_bytes != 1 or packed):
        raise ValueError('unqualified numeric components are only for the plain native 1-byte fixture')
    from gf_common.preprocessing import preparation_config
    from gf_models.sharded_arrays import read_json
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.lineage import checkpoint_path, generation_path
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import initial_hpo_chain
    if model_id == 'mlp' and decision_budget is None and numeric_budget_bytes is None and not packed:
        state, records, learner = initial_hpo_chain(directory, decision_budget=8 * updates)
    else:
        state, records, learner = variable_initial_hpo_chain(directory, games=8, model_id=model_id,
            decision_budget=8*updates if decision_budget is None else decision_budget, numeric_budget_bytes=numeric_budget_bytes, packed=packed)
    registry = read_json(directory/'artifact-registry.json')
    for generation in range(1, updates + 1):
        deadline = time.monotonic()+60
        plan, _, after = recorded_hpo_plan(directory, state, records, generation - 1, deadline)
        original_hpo_collection(plan, state, learner)
        if packed:
            from tests.test_group_pack_game_source import pack_original
            pack_original(plan.output,Path(state['effective_config']['storage']['db_root'])/state['run_id']/'collections'/plan.linkage['batch_id'])
        with _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline) as scan:
            scan.configure_packed(state)
            validate_hpo_chain(directory, max_owned_bytes=1024**3, deadline=deadline, scan=scan)
            with open_hpo_batch(plan, state, records, scan, preparation_config(state['effective_config']['preprocessing']),
                    online=None, max_owned_bytes=1024**3, deadline=deadline) as source:
                numeric = directory/'numeric'/plan.linkage['batch_id']
                if _unqualified_numeric_components:
                    manifest, rollout = _unqualified_numeric_components_for_native(source, numeric)
                    numeric_sha = sha256(numeric/'rollout.json')
                else:
                    manifest = source.write_numeric(numeric)
                    numeric_sha = sha256(numeric/'rollout.json')
                    rollout = source.read_numeric(numeric, numeric_sha)
                proof = source.proof
        metrics = learner.update(rollout)
        saved = learner.save(checkpoint_path(directory, generation), shard_bytes=524288)
        used, collected = rollout.used_decisions, rollout.collected_decisions
        update = {'schema_version': 5, 'status': 'complete', 'run_id': state['run_id'], 'generation': generation,
            'parent_sha256': state['generation_sha256'], 'batch_id': plan.linkage['batch_id'],
            'collection': str(plan.output.relative_to(directory)), 'collection_manifest_sha256': proof['manifest_sha256'],
            'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': numeric_sha,
            'accepted_decisions': used, 'collected_decisions': collected, 'used_decisions': used,
            'overshoot_decisions': collected-used, 'expected_optimizer_step_delta': metrics['optimizer_step_delta'],
            'source_proof': proof, 'metrics': metrics, 'planned_games': 8, 'actual_games': 8, 'health': {},
            'policy_bindings': {key: asdict(value) for key, value in plan.bindings.items()},
            'learner_seats': plan.linkage['learner_seats'], 'collection_seconds': .1, 'update_seconds': .2,
            'model_id': state['model_id'], 'model_version': '4', 'input_recipe': state['input_recipe'],
            'numeric_files': {'rollout.json': numeric_sha, **manifest['data']['files']},
            'hpo_spec_sha256': state['hpo_spec_sha256'], **learner.input_binding}
        if packed:
            update['storage_binding'] = state['storage_binding']
        atomic_json(directory/'updates'/f'{generation:06}.json', update)
        previous = records[-1]
        state['used_decisions'] += used
        state['accepted_decisions'] += used
        state['collected_decisions'] += collected
        pool = previous['parent_controls']['snapshot_pool']
        operating = state['effective_config']['hpo_spec']['operating']
        if state['used_decisions']//operating['pool_every_decisions'] > previous['used_decisions']//operating['pool_every_decisions']:
            pool = [*pool, generation][-operating['opponent_pool_size']:]
        record = {**copy.deepcopy(previous), 'generation': generation, 'parent_sha256': state['generation_sha256'],
            'inference_sha256': saved['inference_digest'], 'resume_sha256': saved['resume_digest'], 'artifact_files': saved['files'],
            'accepted_decisions': state['used_decisions'], 'used_decisions': state['used_decisions'],
            'collected_decisions': state['collected_decisions'], 'optimizer_steps': learner.steps,
            'update_sha256': sha256(directory/'updates'/f'{generation:06}.json'),
            'parent_controls': {'schedule_opponent_rng': after, 'snapshot_pool': pool, 'collection_position': generation}}
        atomic_json(generation_path(directory, generation), record)
        records.append(record)
        state.update(updates=generation, optimizer_steps=learner.steps,
            generation_sha256=sha256(generation_path(directory, generation)))
        registry['generations'][str(generation)] = {'files': {name: {'sha256': checksum,
            'status': 'retained', 'deletion_intent': None, 'deletion_receipt': None}
            for name, checksum in saved['files'].items()}}
        atomic_json(directory/'run.json', state)
        atomic_json(directory/'latest.json', {'generation': generation, 'sha256': state['generation_sha256']})
        atomic_json(directory/'artifact-registry.json', registry)
    return state, records, learner


def _unqualified_numeric_components_for_native(source, directory):
    """Component bytes under the SAME actual original owner, no Source authority.

    This explicit component admission is separate from the parent's deliberately
    unusable 1-byte production phase limit. Neither Source guard is patched.
    """
    from gf_training.hpo_collection import replay_selection
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_rollout import HpoRollout, read_hpo_rollout, write_hpo_rollout

    snapshot, records = source.snapshot, source.records
    assert snapshot is not None and records is not None
    recipe = hpo_training_config(source.state['effective_config']['hpo_spec']['training_recipe'])
    extra = snapshot.extra_bytes_for(records) + source._selection_owned
    rollout = HpoRollout(records, source.model, recipe, actions=source.actions, rules=RULES,
        selected_locations=replay_selection(source.selection, tuple(len(record.frames) for record in records)),
        _reserved_bytes=extra)
    manifest = write_hpo_rollout(directory, rollout, selection=source.selection,
        max_owned_bytes=snapshot.max_owned_bytes, extra_owned_bytes=extra,
        check=snapshot.check, reserve=snapshot.reserve)
    del rollout
    decoded = read_hpo_rollout(directory, sha256(directory/'rollout.json'), source.model, recipe,
        actions=source.actions, rules=RULES, expected_selection=source.selection,
        expected_records=records, numeric_budget_bytes=snapshot.max_owned_bytes,
        max_owned_bytes=snapshot.max_owned_bytes, extra_owned_bytes=extra,
        check=snapshot.check, reserve=snapshot.reserve)
    return manifest, decoded


def variable_initial_hpo_chain(directory, *, games=24, model_id=None, decision_budget=3, numeric_budget_bytes=None, packed=False, training_opponents=None, native=None, resident=False, diagnostic_storage=None):
    """Override only the controlled fixture constructor BEFORE its real save."""
    from unittest.mock import patch

    from gf_models.structured_binding import ConfiguredNeuralModelV4
    from gf_training.hpo_config import effective_settings
    from gf_training.lineage import generation_path
    from tests.hpo_lineage_support import hpo_state_fixture, initial_hpo_chain
    from tests.test_hpo_network import small_model4
    def fixture(*args, **kwargs):
        state, model, recipe, binding = hpo_state_fixture(*args, **kwargs)
        if model_id is not None:
            model = small_model4(model_id)
            state['model_id'] = model_id
            state['effective_config']['model'] = asdict(ConfiguredNeuralModelV4(
                **{name:getattr(model,name) for name in model.__dataclass_fields__}, inference=binding))
        recipe = replace(recipe, collection_games=games)
        spec = state['effective_config']['hpo_spec']
        if diagnostic_storage is not None:
            spec['storage']['hpo_diagnostics'] = diagnostic_storage
        if training_opponents is not None:
            spec.update(schema_version=2, training_opponents=copy.deepcopy(training_opponents))
        if packed:
            state['effective_config']['storage'].update(db_root=str(directory.parent/'db'),ram_root=str(directory.parent/'ram'))
        if numeric_budget_bytes is not None:
            spec['operating']['numeric_budget_bytes'] = numeric_budget_bytes
        spec['training_recipe'] = asdict(recipe)
        spec['effective_settings_sha256'] = effective_settings(model, recipe, {}, layout=LAYOUT).identity
        if native is not None:
            from gf_training.native_backend import binary_identity
            state['effective_config']['preprocessing']['backend'] = 'rust'
            state['identity']['preprocessing'] = {'backend': 'rust', 'reader_contract': 'participant-structured-v2',
                'implementation': 'rust-player-card-refs-v2', 'binary': binary_identity(native), **{key: state[key] for key in ('action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature')}}
        state['hpo_spec_sha256'] = digest(spec)
        state['config_sha256'] = digest(state['effective_config'])
        state['resume_identity'] = digest({'version': 'hpo-ppo-resume-v1', 'config_sha256': state['config_sha256'],
            'identity': state['identity'], 'execution': state['execution'],
            'classification': {key: state[key] for key in ('action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature')},
            'storage_binding': state.get('storage_binding')})
        return state, model, recipe, binding
    with patch('tests.hpo_lineage_support.hpo_state_fixture', fixture):
        state, records, learner = initial_hpo_chain(directory, decision_budget=decision_budget, packed=packed,resident=resident)
    if model_id is not None:
        records[0]['model_id'] = model_id
        atomic_json(generation_path(directory,0),records[0])
        state['generation_sha256'] = sha256(generation_path(directory,0))
        atomic_json(directory/'run.json',state)
        atomic_json(directory/'latest.json',{'generation':0,'sha256':state['generation_sha256']})
    return state,records,learner


def unadmitted_live_hpo_fixture(directory, *, model_id='mlp', decisions=True):
    """Actual bytes for Rust-only live component tests, never Source qualification.

    Keep the numeric and next checkpoint outside the parent's append namespace
    until the native test has crossed the corresponding lifecycle boundary.
    """
    from gf_common.preprocessing import preparation_config
    from gf_training.hpo_lineage import validate_hpo_chain
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.structured_lineage import _ChainScan

    parent = directory/'parent'
    state, records, learner = variable_initial_hpo_chain(parent, games=8,
        model_id=model_id, decision_budget=3)
    deadline = time.monotonic()+120
    plan, _, _ = recorded_hpo_plan(parent, state, records, 0, deadline)
    original_hpo_collection(plan, state, learner, decisions=decisions)
    with _ChainScan(directory=parent, max_owned_bytes=1024**3, deadline=deadline) as scan:
        validate_hpo_chain(parent,max_owned_bytes=1024**3,deadline=deadline,scan=scan)
        with open_hpo_batch(plan, state, records, scan, preparation_config(state['effective_config']['preprocessing']),
                online=None, max_owned_bytes=1024**3, deadline=deadline) as source:
            if not decisions:
                atomic_json(directory/'component.json', {'origin':'unqualified empty held CPU originals; cfg(test) admission only',
                    'selection':source.selection,'training_recipe':state['effective_config']['hpo_spec']['training_recipe'],
                    'expected_optimizer_step_delta':source.proof['expected_optimizer_step_delta'],
                    'correspondence':source.correspondence()})
                return parent
            numeric = directory/'staged-numeric'
            manifest = source.write_numeric(numeric)
            numeric_sha = sha256(numeric/'rollout.json')
            rollout = source.read_numeric(numeric, numeric_sha)
            correspondence = source.correspondence()
    metrics = learner.update(rollout)
    learner.save(directory/'staged-checkpoint', shard_bytes=524288)
    atomic_json(directory/'component.json', {'origin': 'unqualified held CPU originals; cfg(test) admission only',
        'model_id':model_id, 'metrics':metrics, 'correspondence':correspondence,
        'binding':{'directory':str(parent/'numeric'/'batch-000000'),
            'manifest_sha256':numeric_sha, 'files':{'rollout.json':numeric_sha, **manifest['data']['files']}}})
    return parent
