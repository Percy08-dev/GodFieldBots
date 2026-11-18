"""Complete hand-authored originals for structural tests; no executed-game authority."""
import copy
import random
import time
import uuid
from dataclasses import asdict, replace

import numpy as np

from gf_cli.config import ExperimentConfig, RunConfigV2
from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
from gf_common.diverse_actions import PublicCandidateGeneratorV3
from gf_common.domain import digest
from gf_common.preprocessing import PreparationConfig
from gf_common.schedule import serialize_schedule
from gf_common.structured import DecisionFrame, EventBank, Selection, event_delta
from gf_common.structured_tokens import tokenize_frame, tokenize_table
from gf_models.arrays import sha256
from gf_models.card_instance_artifacts import RetainedGameArtifacts
from gf_models.config import PPOTrainingConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3, input_binding_game_v3
from gf_models.structured_runtime import PROCESS_ENVELOPE_RECIPE, own_game_worker_settings_v3
from gf_models.torch_runtime import hardware
from gf_training.card_instance_collection import ppo_collection_metadata
from gf_training.card_instance_journal import CardInstanceJournal
from gf_training.journal import atomic_json
from gf_training.lineage import checkpoint_path
from gf_training.structured_lineage import commit_game_generation, resume_identity
from gf_training.structured_numeric import GameLearner
from tests.card_instance_action_support import inputs
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES, learner_for


def initial_parent(directory, *, updates=2, device='cpu', training_opponents=None, native=None):
    base = learner_for(device=device)
    settings = PreparationConfig(reader='participant-structured-v2', memory_budget_mib=1024)
    config = asdict(ExperimentConfig())
    config.update(actions=asdict(ACTIONS), run=asdict(RunConfigV2(split='train')),
        preprocessing=asdict(settings),
        model=asdict(ConfiguredNeuralModelV3(**{key: getattr(base.model, key) for key in base.model.__dataclass_fields__}, inference=base.binding, cuda_visible_devices='0' if device != 'cpu' else None)),
        training=asdict(PPOTrainingConfig(seeds=(17,), max_updates=updates, device=device)))
    if training_opponents is not None:
        config['training_opponents'] = copy.deepcopy(training_opponents)
    server_config = directory.parent / 'server.toml'
    server_config.write_text('[fixture]\n')
    config['server']['config'] = str(server_config)
    execution = {role: {'binding': asdict(base.binding), 'cuda_visible_devices': '0' if device != 'cpu' else ''} for role in ('learner', 'inference')}
    classification = input_binding_game_v3(ACTIONS.action_contract, RULES)
    identity = {'code': {'revision': None, 'dirty': True}, 'process_envelope_recipe': PROCESS_ENVELOPE_RECIPE,
        'preprocessing': {'backend': 'python', 'reader_contract': 'participant-structured-v2',
            'implementation': 'python-reference-player-card-refs-v2', **classification},
        'integration': {'binary_sha256': receipt()['binary']['sha256'], 'server_config_sha256': sha256(server_config),
            'declared_simulator_revision': ACTIONS.action_contract.simulator_revision,
            'source_revision': ACTIONS.action_contract.simulator_revision,
            'source_spec_sha256': dict(CORRECTED_REVIEWED_SPEC_HASHES)}}
    if native is not None:
        from gf_training.native_backend import binary_identity
        settings = replace(settings, backend='rust')
        config['preprocessing'] = asdict(settings)
        identity['preprocessing'] = {'backend': 'rust', 'reader_contract': 'participant-structured-v2',
            'implementation': 'rust-player-card-refs-v2', 'binary': binary_identity(native), **classification}
    resume = resume_identity(digest(config), identity, execution, classification)
    learner = GameLearner(base.model, 17, base.binding, resume, actions=ACTIONS, rules=RULES)
    now = time.time()
    state = {'schema_version': 3, 'algorithm': 'ppo', 'run_id': 'structural-game-fixture', 'status': 'running',
        'config_sha256': digest(config), 'effective_config': config, 'config_inputs': [], 'identity': identity,
        'seed': 17, 'device': device, 'created_unix': now, 'last_unix': now,
        'wall_deadline_unix': now + 7200, 'admission_deadline_unix': now + 5400, 'execution_deadline_unix': now + 7140,
        'updates': 0, 'accepted_decisions': 0, 'optimizer_steps': 0, 'generation_sha256': None,
        'collection_seconds': 0., 'update_seconds': 0., 'invocations': [], 'completed_updates': None,
        'model_id': 'mlp', 'model_version': '3', 'input_recipe': LAYOUT.input_recipe, 'resume_identity': resume,
        'execution': execution, 'hardware': hardware(device), **classification}
    for name in ('artifacts', 'generations', 'updates', 'collections', 'numeric'):
        (directory / name).mkdir()
    saved = learner.save(checkpoint_path(directory, 0))
    rng = np.random.Generator(np.random.PCG64(17))
    record = commit_game_generation(directory, state, generation=0, update=None,
        controls={'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': [0], 'collection_position': 0},
        artifact_files=saved['files'], max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
    return learner, state, [record], settings


def original_collection(plan, state, learner, *, decisions=True):
    from gf_common.card_instance import action_config_v3
    from tests.shared_inference_support import fixture_runtime_receipt
    ACTIONS=action_config_v3(state['effective_config']['actions'])
    receipt=lambda:fixture_runtime_receipt(ACTIONS.action_contract)
    directory = plan.output
    directory.mkdir()
    owner = RetainedGameArtifacts.for_registry(plan.bindings, ACTIONS, rules=RULES, max_owned_bytes=1024**3,
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
        summary = {'schema_version': 1, 'planned_games': 16, 'natural_denominator': 16, 'comparison_valid': True,
            'failures': {}, 'health_failures': {}, 'group_health_failures': {}}
        atomic_json(directory / 'report.json', summary)
        inventory = [{'path': path.relative_to(directory).as_posix(), 'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'report.json']
        manifest = ppo_collection_metadata(actions=ACTIONS, config=state['effective_config'],
            runtime_evidence=[receipt() for _ in plan.groups], schedule=serialize_schedule(plan.groups, 4), inventory=inventory,
            policies=list(plan.bindings), bots_revision='fixture', policy_bindings=plan.bindings,
            neural_inputs=owner.finalize(deadline=time.monotonic() + 60), linkage=plan.linkage)
        manifest.update(kind='card-instance-collection', run_id='fixture-collection', status='complete',
            groups=[{'runtime_evidence': receipt()} for _ in plan.groups], episodes=results,
            inputs=state['config_inputs'], code=state['identity']['code'], report_sha256=sha256(directory / 'report.json'))
        atomic_json(directory / 'manifest.json', manifest)
        return manifest
    finally:
        owner.release()
