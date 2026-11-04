"""Predeclared component originals with actual initial CPU MLP artifacts; no games."""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict, replace
from pathlib import Path

from gf_common import card_instance_fixture as fixture
from gf_common.candidate_columns import encode_candidate_columns_v2
from gf_common.card_instance import ActionConfigV3
from gf_common.domain import canonical, digest
from gf_common.structured import CandidateTable, DecisionFrame, EventBank, event_delta
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature
from gf_common.structured_wire import frame_dict
from tests.card_instance_action_support import inputs


def write(path, value):
    path.write_text(canonical(value)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_receipt(path):
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def rehash_originals(directory):
    import json
    manifest = json.loads((directory/'manifest.json').read_text())
    manifest['files'] = {file: file_receipt(directory/file) for file in manifest['files']}
    manifest['declaration']['sha256'] = manifest['files']['declaration.json']['sha256']
    manifest['content_sha256'] = digest({k: v for k, v in manifest.items() if k != 'content_sha256'})
    return write(directory/'manifest.json', manifest)


def construct(directory: Path):
    # Optional Torch is imported only by the explicit construction helper.
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.structured_artifacts import save_inference_v3
    from gf_models.structured_config import InferenceBindingV2
    from gf_models.structured_network import make_network
    from tests.model3_support import model3

    started = time.monotonic()
    directory.mkdir()
    model = model3()
    model = replace(model, temporal=replace(model.temporal, max_decisions=16))
    actions = ActionConfigV3(action_contract=LAYOUT.action_contract, max_candidates=64)
    execution = InferenceBindingV2(max_candidates=64, candidate_chunk=2)
    populations = [{'population_id': f'p{seed}', 'construction_seed': seed} for seed in fixture.SEEDS]
    cohorts = []
    for population in populations:
        for generation in range(3):
            split, count = ('validation', 64) if generation == 2 else ('train', 16)
            cid = population['population_id']+f'-g{generation}'
            episodes = []
            for i in range(count):
                viewer = 1+i % 2
                episodes.append({'episode': cid+f'-e{i:03}', 'viewer': viewer,
                    'roles': [{'participant': seat, 'role': 'declared-selected' if seat == viewer else 'declared-opponent',
                               'policy_generation': generation if seat == viewer else i % 2} for seat in (1, 2)],
                    'decision_count': 0 if i == 0 else 2 if i == 1 else 1,
                    'outcome': {'kind': 'draw'} if i % 3 == 0 else {'kind': 'win', 'winner': 1+(i//2) % 2}})
            cohorts.append({'cohort_id': cid, 'population_id': population['population_id'],
                            'split': split, 'generation': generation, 'episodes': episodes})
    frozen = fixture.freeze_declaration({'schema_version': 1, 'kind': 'card-instance-tree-fixture-declaration',
        'fixture_id': 'fixed-component-tree', 'neural_model': asdict(model), 'actions': asdict(actions),
        'execution': asdict(execution), 'input_binding': fixture.input_binding(), 'layout_signature': signature(layout=LAYOUT),
        'label_recipe': fixture.LABEL_RECIPE, 'construction_recipe': fixture.CONSTRUCTION_RECIPE,
        'populations': populations, 'cohorts': cohorts})
    # The immutable construction recipe exists before the first actual weights or originals.
    declaration_sha = write(directory/'declaration.json', frozen)
    policies, ids = [], {}
    for population in populations:
        for generation in range(3):
            artifact = directory.parent/(population['population_id']+f'-policy{generation}')
            network = make_network(model, population['construction_seed']+generation, execution, layout=LAYOUT)
            save_inference_v3(artifact, network, model, digest(asdict(actions)), layout=LAYOUT,
                action_contract=LAYOUT.action_contract, rules=card_instance_rules(), purpose='component', synthetic_capacity=False)
            inference_sha = file_receipt(artifact/'inference.json')['sha256']
            pid = fixture.policy_id(frozen['fixture_id'], population['population_id'], generation, inference_sha)
            ids[population['population_id'], generation] = pid
            policies.append({'population_id': population['population_id'], 'generation': generation,
                'directory': str(artifact.absolute()), 'inference_sha256': inference_sha, 'policy_id': pid,
                'files': {path.name: file_receipt(path) for path in artifact.iterdir()}})
            del network
    receipts = []
    for cohort in cohorts:
        episode_receipts = []
        for i, spec in enumerate(cohort['episodes']):
            raw = {'schema_version': 1, 'kind': 'card-instance-component-original', 'fixture_id': frozen['fixture_id'],
                'declaration_sha256': declaration_sha, 'population_id': cohort['population_id'],
                'cohort_id': cohort['cohort_id'], 'split': cohort['split'], 'generation': cohort['generation'],
                'episode': spec['episode'], 'viewer': spec['viewer'],
                'roles': [{**role, 'policy_id': ids[cohort['population_id'], role['policy_generation']]} for role in spec['roles']],
                'selected_policy_id': ids[cohort['population_id'], cohort['generation']], 'outcome': spec['outcome'],
                'outcome_origin': fixture.OUTCOME_ORIGIN, 'events': [], 'decisions': []}
            columns = []
            if spec['decision_count']:
                kind = LAYOUT.action_kinds[(i-1) % len(LAYOUT.action_kinds)]
                choice = 'defense' if kind == 'respond' else 'buy_accept_reject' if kind == 'buy_decision' else 'turn_action'
                count = (7, 12, 18)[i % 3]
                cards = [('fireball', 'learned')]*count if i % 4 == 0 else [
                    ('bronze-club', None), ('fireball', 'unused'), ('fireball', 'learned')]
                value = inputs(cards, choice=choice)
                obs, hints = copy.deepcopy(value.observation), copy.deepcopy(value.action_options)
                viewer = spec['viewer']
                obs.update(viewer_player_id=viewer, current_player=viewer, active_player=viewer)
                obs['own']['player_id'], obs['opponent']['player_id'] = viewer, 3-viewer
                hints['viewer_player_id'], hints['action_options']['actor'] = viewer, viewer
                if i % 5 == 0:
                    obs['own']['status'].update(fog=True, dream=True)
                    obs['opponent'] = {'player_id': 3-viewer, 'learned_miracles': ['aura', 'aura'], 'ascended': False}
                events = [{'type': 'guardian_set', 'player': viewer, 'definition_id': 'sun'},
                          {'type': 'artifact_used', 'player': 3-viewer, 'definition_id': 'angel-axe'}]
                # Use an existing public Guardian namespace, not a new catalog entry.
                from gf_common.structured_vocab import GUARDIANS
                events[0]['definition_id'] = GUARDIANS[0]
                value = replace(value, observation=obs, action_options=hints, events=tuple(events))
                bank = EventBank(viewer, spec['episode'], layout=LAYOUT)
                bank.append(event_delta(viewer, spec['episode'], 0, bank.boundary.prefix_hash, events, layout=LAYOUT))
                frame = DecisionFrame.from_input(value, bank, 0, layout=LAYOUT)
                a, b = frame.references[:2]
                extra = {
                    'play_sequence': {'ordered_card_refs': [b, a], 'target': 3-viewer},
                    'respond': {'ordered_card_refs': [b, a]}, 'discard': {'card_refs': [b, a]},
                    'sacrifice': {'card_refs': [a, b]}, 'exchange': {'card_ref': a, 'hp': -1, 'mp': 0, 'money': 1},
                    'sell': {'card_ref': a, 'sold_card_ref': b, 'target': viewer}, 'buy': {'card_ref': b, 'target': 3-viewer},
                    'buy_decision': {'accept': False}, 'invoke_phenomenon': {'card_ref': a}, 'pray': {}}
                action = {'type': kind, **extra[kind]}
                other = copy.deepcopy(action)
                if 'ordered_card_refs' in other:
                    other['ordered_card_refs'].reverse()
                elif kind == 'buy_decision':
                    other['accept'] = True
                elif 'target' in other:
                    other['target'] = viewer if other['target'] != viewer else 3-viewer
                table = CandidateTable.from_actions(frame, [action, other], action_contract=LAYOUT.action_contract,
                                                    purpose='component', synthetic_capacity=False)
                file = spec['episode']+'-d0000.bin'
                payload, descriptor = encode_candidate_columns_v2(table, file, layout=LAYOUT)
                (directory/file).write_bytes(payload)
                raw['events'] = events
                raw['decisions'].append({'frame': frame_dict(frame, layout=LAYOUT), 'columns': descriptor, 'selected_index': 1})
                columns.append(file)
                if spec['decision_count'] == 2:
                    start = bank.boundary.end
                    incoming = [{'type': 'safe', 'player': viewer}]
                    bank.append(event_delta(viewer, spec['episode'], start, bank.boundary.prefix_hash, incoming, layout=LAYOUT))
                    events.extend(incoming)
                    later = copy.deepcopy(value)
                    later.observation['state_revision'] = 1
                    later.action_options['state_revision'] = 1
                    later.action_options['action_options']['state_revision'] = 1
                    later = replace(later, events=tuple(events))
                    later_frame = DecisionFrame.from_input(later, bank, start, layout=LAYOUT)
                    later_table = CandidateTable.from_actions(later_frame, [action, other], action_contract=LAYOUT.action_contract,
                                                              purpose='component', synthetic_capacity=False)
                    file = spec['episode']+'-d0001.bin'
                    payload, descriptor = encode_candidate_columns_v2(later_table, file, layout=LAYOUT)
                    (directory/file).write_bytes(payload)
                    raw['decisions'].append({'frame': frame_dict(later_frame, layout=LAYOUT), 'columns': descriptor, 'selected_index': 0})
                    columns.append(file)
            file = spec['episode']+'.json'
            write(directory/file, raw)
            episode_receipts.append({'episode': spec['episode'], 'file': file, 'columns': columns})
        receipts.append({'cohort_id': cohort['cohort_id'], 'episodes': episode_receipts})
    manifest = {'schema_version': 1, 'kind': 'card-instance-tree-fixture-originals',
        'declaration': {'file': 'declaration.json', 'sha256': declaration_sha}, 'origin': fixture.origin(),
        'policies': policies, 'cohorts': receipts, 'files': {path.name: file_receipt(path) for path in directory.iterdir()},
        'construction_cost': {'seconds': time.monotonic()-started, 'scope': 'fixture-construction-only'}}
    manifest['content_sha256'] = digest(manifest)
    manifest_sha = write(directory/'manifest.json', manifest)
    return directory/'manifest.json', manifest_sha
