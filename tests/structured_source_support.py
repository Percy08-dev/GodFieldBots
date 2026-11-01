"""Hand-authored structural collection fixtures, not executed game evidence."""
import random
import uuid
from dataclasses import asdict, replace

from gf_cli.config import ExperimentConfig, RunConfigV2
from gf_common.action_contract import engineering_contract
from gf_common.diverse_actions import NumericCosts, PublicCandidateGeneratorV2
from gf_common.domain import digest
from gf_common.structured import ActionConfigV2, EventBank, Selection
from gf_models.structured_binding import input_binding, policy_label
from gf_models.structured_config import InferenceBindingV2
from gf_training.journal import atomic_json
from gf_training.structured_journal import StructuredJournal
from tests.test_structured_network import small_model
from tests.test_structured_protocol import own_frame


def collection_fixture(tmp_path, make_input, *, end_kind='natural', decisions=True, pending=False,
                       schema=3, neural=False, missing_replay=False, games=1, outcome='draw', abort_reason=None):
    directory = tmp_path / 'source'
    directory.mkdir()
    actions = ActionConfigV2(action_contract=engineering_contract(), max_sequence_length=1, max_candidates=8)
    metadata = input_binding(actions.action_contract, 'component', False)
    model, inference = small_model(), InferenceBindingV2()
    neural_label = policy_label(model, inference, 'a' * 64, cuda_visible_devices='')
    labels = (neural_label, 'random-structured-v1') if neural else ('random-structured-v1', 'heuristic-structured-v1')
    registry = {label: {'model_id': 'random' if label.startswith('random') else 'heuristic', 'policy_options': None} for label in labels}
    if neural:
        registry[neural_label] = {'model_id': 'mlp', 'policy_options': {'version': '2', 'checkpoint': 'unverified-component',
            'expected_manifest_digest': 'a' * 64, 'model': asdict(model), 'inference': asdict(inference),
            'cuda_visible_devices': '', **metadata}}
    config = asdict(ExperimentConfig())
    config['actions'], config['run'] = asdict(actions), asdict(RunConfigV2(split='train'))
    schedule, episodes = [], []
    all_games = []
    kinds = list(end_kind) if isinstance(end_kind, tuple) else [end_kind] * games
    for index in range(games):
        current_kind = kinds[index]
        has_decisions = decisions[index] if isinstance(decisions, tuple) else decisions
        reason = None if current_kind == 'natural' else 'transport_failure' if missing_replay else current_kind
        if abort_reason is not None:
            reason = abort_reason
        name, game = f'episode-{index}', f'{index + 1:026d}'
        path = directory / name
        path.mkdir()
        paired = schema == 4 and games == 16
        swap = paired and index % 2 == 1
        current_labels = tuple(reversed(labels)) if swap else labels
        spec = {'episode_id': name, 'policies': list(current_labels), 'sampling_seeds': [19, 17] if swap else [17, 19],
                'pair_id': f'pair-{index // 2 if paired else index}', 'learner_viewer': 2 if swap else 1}
        if swap:
            schedule[-1]['episodes'].append(spec)
        else:
            schedule.append({'seed': str(uuid.UUID(int=index // 2 + 1 if paired else index + 1)), 'episodes': [spec], 'ancestry': None})
        health = {'replay_or_finalization_failure': 'connection poisoned'} if missing_replay else {}
        episode = {'episode_id': name, 'pair_id': spec['pair_id'], 'seed': schedule[-1]['seed'], 'policies': list(current_labels),
            'game_id': game, 'end_kind': current_kind, 'outcome': outcome if current_kind == 'natural' else None,
            'failure': reason, 'health': health, 'simulator_abort_reason': abort_reason, 'replay_reconciled': not missing_replay,
            **metadata}
        for viewer in (1, 2):
            value = make_input(hand=('bronze-club',))
            if viewer == 2:
                raw = value.to_dict()
                obs = raw['observation']
                obs.update(viewer_player_id=2, current_player=2, active_player=2)
                obs['own']['player_id'], obs['opponent']['player_id'] = 2, 1
                raw['action_options']['viewer_player_id'] = 2
                value = replace(value, observation=obs, action_options=raw['action_options'])
            bank = EventBank(viewer, f'opaque-{index}-{viewer}')
            frame, delta = own_frame(value, bank, 0)
            costs = NumericCosts()
            rng = random.Random(int(digest({'recipe': 'structured-search-stream-v1', 'sampling_seed': spec['sampling_seeds'][viewer - 1]})[:16], 16))
            generated = PublicCandidateGeneratorV2(actions, clock=lambda: 0.).generate(frame, bank.view(), rng, costs)
            model_id = registry[current_labels[viewer - 1]]['model_id']
            nn = model_id == 'mlp'
            journal = StructuredJournal(path, name, viewer, current_labels[viewer - 1], run_id='component-run', game_id=game,
                action_config_hash=digest(asdict(actions)), bank_episode=bank.episode,
                caps={'max_interval_events': model.event.max_interval if nn else 4096,
                      'max_episode_events': model.event.max_episode if nn else 262144,
                      'max_decisions': model.temporal.max_decisions if nn else None,
                      'max_positions': 1, 'max_candidates': 8}, numeric_costs=asdict(costs), model_id=model_id,
                model_version='2' if nn else '1', sampling_seed=spec['sampling_seeds'][viewer - 1], **{**metadata, 'action_contract': actions.action_contract})
            selected = Selection(generated.table.table_hash, 0, current_labels[viewer - 1], 0. if nn else None, 0. if nn else None)
            accepted = []
            if has_decisions:
                command = {'command_id': str(uuid.uuid4()), 'expected_revision': 0, 'action': selected.validate(generated.table, frame)}
                number = journal.begin_frame(frame, delta, generated, selected, command, {})
                if not pending:
                    journal.acknowledge(number, {'command_id': command['command_id'], 'previous_revision': 0,
                        'state_revision': 1, 'status': 'active', 'result': None})
                    accepted.append(command['action'])
            final = frame.observation if current_kind == 'natural' else None
            if final is not None:
                final.update(state_revision=1, actionable=False, typed_action_schema=[], required_choice=None,
                             current_player=None, phase='terminal', result=outcome)
            journal.finish(current_kind, episode['outcome'], final, abort_reason)
            absent = missing_replay is True or type(missing_replay) is int and missing_replay == viewer
            if not absent:
                replay = {'game_id': game, 'viewer_player_id': viewer, 'observation_schema_version': '1',
                    'episode_end': {'type': 'terminal', 'result': outcome} if current_kind == 'natural' else None,
                    'turns': [{'decisions': [{'actor': viewer, 'visible_action': {'type': 'action', 'action': action}} for action in accepted]}]}
                if abort_reason is not None:
                    replay['episode_end'] = {'type': 'aborted', 'reason': abort_reason, 'player': None,
                        'decision': {'turn': 1, 'decision_index': None, 'invalid_attempt_count': 0}}
                atomic_json(path / f'replay-player-{viewer}.json', replay)
        atomic_json(path / 'result.json', episode)
        episodes.append(episode)
        all_games.append(name)
    report = {'schema_version': 1, 'planned_games': games, 'natural_denominator': sum(kind == 'natural' for kind in kinds),
        'comparison_valid': all(kind == 'natural' for kind in kinds),
        'failures': {kind: kinds.count(kind) for kind in kinds if kind != 'natural'},
        'health_failures': {}, 'group_health_failures': {}}
    atomic_json(directory / 'report.json', report)
    from gf_common.domain import (
        CATALOG_META,
        END_PROJECTION_VERSION,
        POLICY_INPUT_VERSION,
        REVIEWED_SPEC_HASHES,
        SIMULATOR_REVISION,
    )
    from gf_models.arrays import sha256
    manifest = {'schema_version': schema, 'status': 'complete' if all(kind == 'natural' for kind in kinds) else 'incomplete', 'run_id': 'component-run',
        'effective_config': config, 'config_sha256': digest(config), 'split': 'train', 'elapsed_seconds': 1.,
        'journal_schema_version': 5, 'feature_version': 'participant-structured-v1', 'input_recipe': 'participant-structured-v1',
        'reward_policy_version': 'natural-terminal-v1', 'end_projection_version': END_PROJECTION_VERSION,
        'policy_input_version': POLICY_INPUT_VERSION, 'catalog': CATALOG_META, 'simulator_contract_revision': SIMULATOR_REVISION,
        'reviewed_simulator_spec_sha256': REVIEWED_SPEC_HASHES, 'observation_schema_version': '1', 'action_schema_version': 1,
        'schedule': schedule, 'episodes': episodes, 'groups': [{'episode_ids': all_games}], 'policy_bindings': registry,
        'report_sha256': sha256(directory / 'report.json'), **metadata}
    if schema == 4:
        manifest.update(collection_kind='ppo_batch' if games == 16 else 'ppo_validation', ppo={'parent_run_id': 'unqualified-parent', 'parent_run_path': 'unverified',
            'collection_generation': 0, 'batch_id': 'component-batch',
            'learner_seats': {spec['episode_id']: spec['learner_viewer'] for group in schedule for spec in group['episodes']}})
    atomic_json(directory / 'manifest.json', manifest)
    return directory, actions, manifest
