"""Unadmitted baseline-only C5/J6 originals; NEVER a formal NN Source."""
import copy
import random
import time
import uuid
from dataclasses import asdict, replace

import numpy as np

from gf_cli.config import load_config
from gf_common.card_instance_collection import (
    FINAL_VALIDATION,
    config_projection,
    validation_invocation,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.diverse_actions import PublicCandidateGeneratorV3
from gf_common.domain import RuntimePolicyBinding, digest
from gf_common.schedule import serialize_schedule
from gf_common.structured import DecisionFrame, EventBank, Selection
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_models.arrays import sha256
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_training.card_instance_collection import collection_metadata
from gf_training.card_instance_journal import CardInstanceJournal
from gf_training.collection import training_groups, validation_groups
from gf_training.journal import atomic_json
from tests.card_instance_action_support import inputs
from tests.test_card_instance_baseline_runtime import receipt


def baseline_originals(directory, *, role=FINAL_VALIDATION, decisions=False):
    """Structural-only input: no declaration, NN artifact, PPO run or qualification."""
    from pathlib import Path
    root = Path(__file__).parents[1]
    loaded = load_config(root / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml')
    actions, rules = loaded.config.actions, card_instance_rules()
    projected = replace(loaded.config, run=replace(loaded.config.run, split=role.split))
    registry = {label + '-card-instance-v2': RuntimePolicyBinding(label) for label in ('random', 'heuristic')}
    label = 'random-card-instance-v2'
    groups = (validation_groups(17, label, actions_version=3) if role is FINAL_VALIDATION else
        training_groups(17, 0, label, (label,), np.random.Generator(np.random.PCG64(17)), actions_version=3))
    serialized = serialize_schedule(groups, 4)
    link = {'parent_run_id': 'UNADMITTED-baseline-only-not-a-parent', 'parent_run_path': str(directory.parent / 'UNADMITTED-parent'),
        'batch_id': 'validation-final' if role is FINAL_VALIDATION else 'batch-000000',
        'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes},
        'schedule_digest': digest(serialized), 'policy_registry_digest': digest({key: asdict(value) for key, value in registry.items()})}
    if role is FINAL_VALIDATION:
        link.update(final_generation=1000, final_inference_sha256='a' * 64, policy_label=label)
    else:
        link['collection_generation'] = 0
    directory.mkdir()
    episodes = []
    for group in groups:
        for spec in group.episodes:
            folder = directory / spec.episode_id
            folder.mkdir()
            game = uuid.uuid4().hex
            result = {'episode_id': spec.episode_id, 'pair_id': spec.pair_id, 'seed': group.seed,
                'policies': list(spec.policies), 'game_id': game, 'end_kind': 'natural', 'outcome': 'draw',
                'failure': None, 'health': {}, 'simulator_abort_reason': None, 'replay_reconciled': True}
            for viewer in (1, 2):
                value = inputs([], can_pray=True)
                if viewer == 2:
                    obs, hints = copy.deepcopy(value.observation), copy.deepcopy(value.action_options)
                    obs.update(viewer_player_id=2, current_player=2, active_player=2)
                    obs['own']['player_id'], obs['opponent']['player_id'] = 2, 1
                    hints['viewer_player_id'], hints['action_options']['actor'] = 2, 2
                    value = replace(value, observation=obs, action_options=hints)
                bank = EventBank(viewer, spec.episode_id + f'-{viewer}', layout=LAYOUT)
                binding = registry[spec.policies[viewer - 1]]
                caps, costs, _, _ = own_game_worker_settings_v3(binding.model_id, actions, None, rules=rules)
                journal = CardInstanceJournal(folder, spec.episode_id, viewer, spec.policies[viewer - 1], run_id='UNADMITTED-baseline-structural',
                    game_id=game, actions=actions, bank_episode=bank.episode, caps=asdict(caps), numeric_costs=asdict(costs),
                    model_id=binding.model_id, runtime_evidence=receipt(), sampling_seed=spec.sampling_seeds[viewer - 1])
                accepted = []
                if decisions and viewer == spec.learner_viewer:
                    from gf_common.structured import event_delta
                    delta = event_delta(viewer, bank.episode, 0, bank.boundary.prefix_hash, [], layout=LAYOUT)
                    frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
                    search_seed = int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': spec.sampling_seeds[viewer - 1]})[:16], 16)
                    generated = PublicCandidateGeneratorV3(actions, rules=rules, clock=lambda: 0.).generate_game(frame, bank.view(), random.Random(search_seed), costs)
                    selection = Selection(generated.table.table_hash, 0, spec.policies[viewer - 1], None, None)
                    action = selection.validate(generated.table, frame)
                    command = {'command_id': str(uuid.uuid4()), 'expected_revision': 0, 'action': action}
                    number = journal.begin_frame(frame, delta, generated, selection, command, {})
                    journal.acknowledge(number, {'command_id': command['command_id'], 'previous_revision': 0, 'state_revision': 1, 'status': 'active', 'result': None})
                    accepted.append(action)
                final = copy.deepcopy(value.observation)
                final.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None, current_player=None, phase='terminal', result='draw')
                journal.finish('natural', 'draw', final)
                replay = {'game_id': game, 'viewer_player_id': viewer, 'requested_turn': None,
                    'rule_set_version': actions.action_contract.rule_set_version, 'observation_schema_version': '2',
                    'public_event_schema_version': '2', 'card_ref_scheme': actions.action_contract.card_ref_scheme,
                    'episode_end': {'type': 'terminal', 'result': 'draw'}, 'turns': []}
                if accepted:
                    replay['turns'] = [{'turn': 1, 'active_player': viewer, 'decisions': [
                        {'actor': viewer, 'observation': value.observation, 'action_options': value.action_options['action_options'],
                            'visible_action': {'type': 'action', 'action': action}, 'visible_events': [], 'aborted': False} for action in accepted]}]
                atomic_json(folder / f'replay-player-{viewer}.json', replay)
            atomic_json(folder / 'result.json', result)
            episodes.append(result)
    report = {'schema_version': 1, 'healthy': True, 'planned_games': role.games, 'natural_denominator': role.games,
        'comparison_valid': True, 'failures': {}, 'health_failures': {}, 'group_health_failures': {}, 'wall_seconds': .1}
    atomic_json(directory / 'report.json', report)
    inventory = [{'path': path.relative_to(directory).as_posix(), 'sha256': sha256(path), 'bytes': path.stat().st_size}
        for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'report.json']
    manifest = collection_metadata(actions=actions, config=asdict(projected), runtime_evidence=[receipt() for _ in groups],
        schedule=serialized, inventory=inventory, policies=list(registry), bots_revision='UNADMITTED-baseline-structural')
    # Exercise the structural raw validator independently. These baseline-only
    # records cannot satisfy final NN producer/binding or FormalGameParent admission.
    manifest.update(model_version='3', neural_inputs={'input_contract': EXTERNAL_INPUT_CONTRACT, 'policies': {}},
        policy_bindings={key: asdict(value) for key, value in registry.items()}, collection_role=role.role, linkage=link,
        kind='card-instance-collection', run_id='UNADMITTED-baseline-structural', status='complete',
        groups=[{'runtime_evidence': receipt()} for _ in groups], episodes=episodes, inputs=list(loaded.inputs),
        code={'scope': 'UNADMITTED'}, report_sha256=sha256(directory / 'report.json'))
    if role is FINAL_VALIDATION:
        parent = {**loaded.effective, 'run': {**loaded.effective['run'], 'split': 'train'}}
        manifest.update(config_projection=config_projection(parent, list(loaded.inputs), asdict(projected)),
            validation_invocation=validation_invocation(started_unix=time.time(), started_monotonic=1000.,
                caller_deadline_monotonic=8200., run=asdict(projected.run)), elapsed_seconds=.2)
    atomic_json(directory / 'manifest.json', manifest)
    return loaded, manifest, report


def rewrite_manifest(directory, manifest):
    for entry in manifest['files']:
        path = directory / entry['path']
        if path.exists(): entry.update(sha256=sha256(path), bytes=path.stat().st_size)
    manifest['report_sha256'] = sha256(directory / 'report.json')
    atomic_json(directory / 'manifest.json', manifest)
