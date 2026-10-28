"""ADR0009 opt-in fixed public prefix; no snapshots, engine copy or adaptive probes."""
import asyncio
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.config import load_config
from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, PolicyDecision, digest
from gf_common.validation import decision_input, reconcile_replay
from gf_simulator_client.http import AttestedClient, seed_fingerprint
from gf_simulator_client.process import ManagedServer
from gf_training.journal import ParticipantJournal, atomic_json, read_journal
from tests.conftest import fixture

ROOT = Path(__file__).parents[1]
pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('GFB_RUN_INTEGRATION') != '1', reason='explicit binary integration opt-in')]


def test_current_catalog_and_magical_stick_target_payment_replay_matrix(tmp_path):
    """Eight prescribed fresh games: MP0/10 × omitted/null/self/opponent."""
    async def run():
        control = fixture('magical_stick_control_public.json')
        settings = ActionConfig()
        server_config = load_config(ROOT / 'configs/experiments/smoke.toml').config.server
        seed = control['game_seed']
        server = ManagedServer(server_config.binary, server_config.config, seed, 8, tmp_path / 'server')
        client = None
        completed = []
        try:
            startup = await server.start()
            assert server.expected['server_idle_timeout_secs'] == 0
            client = AttestedClient(startup['base_url'], expected_fingerprint=seed_fingerprint(seed), fixed_seed_experiment=True,
                                    expected_effective=server.expected, startup_info=startup['test_server_info'])
            await client.attest()
            definitions = await client.public_catalog()
            # Compatibility endpoint content/canonical hashes are diagnostics, never policy context.
            old = await client._request('GET', '/v1/card-catalogs/public-ja-v1')
            current = await client._request('GET', '/v1/card-catalogs/public-ja-v2')
            assert old == fixture('catalog_public.json')
            assert current == fixture('catalog_public_v2.json')
            for payload in (old, current):
                assert digest({key: value for key, value in payload.items() if key != 'content_sha256'}) == payload['content_sha256']
            atomic_json(tmp_path / 'catalog-v1.json', old)
            atomic_json(tmp_path / 'catalog-v2.json', current)
            for mp in (0, 10):
                for target_form in ('self', 'opponent', 'omitted', 'null'):
                    name = f'mp{mp}-{target_form}'
                    directory = tmp_path / name
                    directory.mkdir()
                    game = (await client.create())['game_id']
                    for _ in range(2):
                        await client.join(game, str(uuid.uuid4()))
                    run_id = str(uuid.uuid4())
                    journals = {viewer: ParticipantJournal(directory, name, viewer, 'fixed-public-contract-component-only', run_id=run_id,
                                 game_id=game, action_config_hash=digest(asdict(settings))) for viewer in (1, 2)}
                    histories, cursors, accepted = {1: [], 2: []}, {1: None, 2: None}, {1: [], 2: []}
                    trace = []
                    async def current_input(viewer, game=game, cursors=cursors, histories=histories):
                        obs = await client.observe(game, viewer)
                        assert obs['actionable'] and obs['current_player'] == viewer
                        options = await client.actor_options(game, viewer, obs)
                        events, cursors[viewer] = await client.events(game, viewer, cursors[viewer])
                        histories[viewer].extend(events)
                        return decision_input(obs, histories[viewer], cursors[viewer], definitions, options)
                    async def submit(viewer, value, action, *, generated=True, game=game, journals=journals, trace=trace, accepted=accepted):
                        command = {'command_id': str(uuid.uuid4()), 'expected_revision': value.observation['state_revision'], 'action': action}
                        index = None
                        if generated:
                            candidates = PublicCandidateGenerator(settings).generate(value)
                            selection = PolicyDecision(candidates.actions.index(action), 'fixed-public-contract-component-v1')
                            index = journals[viewer].begin(value, candidates, selection, command, 0)
                        # Omitted/null forms are published typed-contract fixtures, not gen-v3 candidates.
                        trace.append({'viewer': viewer, 'policy_input': value.to_dict(), 'command': command,
                                      'scope': 'generated-candidate-journal' if generated else 'typed-contract-diagnostic-only'})
                        ack = await client.act(game, viewer, command)
                        trace[-1]['acknowledgement'] = ack
                        if generated:
                            journals[viewer].acknowledge(index, ack)
                        accepted[viewer].append(action)
                    try:
                        count = control['zero_mp_prefix_length' if mp == 0 else 'positive_mp_prefix_length']
                        for step in control['prefix'][:count]:
                            value = await current_input(step['player_id'])
                            # Check only current public actor/revision; never repair or require old MP/events/RNG.
                            assert value.observation['state_revision'] == step['expected_revision']
                            await submit(step['player_id'], value, step['action'])
                        value = await current_input(2)
                        before = value.observation
                        assert before['own']['mp'] == mp and before['own']['status']['dream'] is False
                        assert {'instance_id': 13, 'definition_id': 'magical-stick'} in before['own']['hand']
                        assert 13 in value.action_options['action_options']['usable_hand_elements']
                        assert set(value.action_options['action_options']['valid_targets']) == {1, 2}
                        branch_revision = before['state_revision']
                        action = {'type': 'play_sequence', 'ordered_elements': [{'source': 'hand_artifact', 'instance_id': 13}]}
                        if target_form != 'omitted':
                            action['target'] = {'self': 2, 'opponent': 1, 'null': None}[target_form]
                        if mp == 0 and target_form == 'self':
                            assert action == control['original_rejected_action']['action']
                        await submit(2, value, action, generated=target_form in ('self', 'opponent'))
                        attack_events, cursors[2] = await client.events(game, 2, cursors[2])
                        histories[2].extend(attack_events)
                        trace[-1]['player_visible_events'] = attack_events
                        used = next(index for index, event in enumerate(attack_events) if event['type'] == 'artifact_used' and event['definition_id'] == 'magical-stick')
                        declared = [(index, event) for index, event in enumerate(attack_events) if event['type'] == 'attack_declared']
                        assert len(declared) == 1
                        declared_index, declaration = declared[0]
                        assert declaration == {'type': 'attack_declared', 'attacker': 2, 'target': 2 if target_form == 'self' else 1, 'attack': 2 * mp}
                        payments = [(index, event) for index, event in enumerate(attack_events[:declared_index]) if event['type'] == 'resource_changed' and event['player'] == 2]
                        assert len(payments) == int(mp > 0)
                        if mp:
                            payment_index, payment = payments[0]
                            assert used < payment_index < declared_index
                            assert payment == {'type': 'resource_changed', 'player': 2, 'hp_delta': 0, 'mp_delta': -mp, 'money_delta': 0}
                        own = await client.observe(game, 2)
                        next_player = await client.observe(game, 1)
                        assert not any(card['instance_id'] == 13 for card in own['own']['hand'])
                        if target_form == 'self':
                            before_refill = next((index for index, event in enumerate(attack_events) if event['type'] in ('artifact_drawn', 'turn_started')), len(attack_events))
                            if mp == 0:
                                assert not any(event['type'] == 'damage_applied' for event in attack_events[:before_refill])
                            assert next_player['current_player'] == 1 and next_player['required_choice'] == 'turn_action'
                            assert any(event['type'] == 'turn_started' for event in attack_events)
                        else:
                            assert next_player['current_player'] == 1 and next_player['required_choice'] == 'defense'
                            assert own['own']['mp'] == 0  # observed before response/refill, not a promise about final MP
                            assert len(own['own']['hand']) == len(before['own']['hand']) - 1
                            assert not any(event['type'] in ('artifact_drawn', 'turn_started') for event in attack_events)
                            defender = await current_input(1)
                            assert defender.action_options['action_options']['can_allow'] is True
                            await submit(1, defender, {'type': 'respond', 'ordered_defenses': []})
                            response_events, cursors[2] = await client.events(game, 2, cursors[2])
                            trace[-1]['player_visible_events'] = response_events
                            before_refill = next((index for index, event in enumerate(response_events) if event['type'] in ('artifact_drawn', 'turn_started')), len(response_events))
                            if mp == 0:
                                assert not any(event['type'] == 'damage_applied' for event in response_events[:before_refill])
                            assert any(event['type'] == 'turn_started' for event in response_events)
                        resolution_events = attack_events if target_form == 'self' else response_events
                        first_refill = next(index for index, event in enumerate(resolution_events) if event['type'] == 'artifact_drawn' and event['player'] == 2)
                        turn_start = next(index for index, event in enumerate(resolution_events) if event['type'] == 'turn_started')
                        assert first_refill < turn_start
                        damage = [(index, event) for index, event in enumerate(resolution_events[:first_refill]) if event['type'] == 'damage_applied']
                        if mp > 0:
                            assert len(damage) == 1
                            assert damage[0][1]['player'] == (2 if target_form == 'self' else 1)
                            assert damage[0][1]['amount'] == 20
                        else:
                            assert not damage
                        after_resolution = await client.observe(game, 2)
                        assert len(after_resolution['own']['hand']) == len(before['own']['hand'])
                        for viewer in (1, 2):
                            replay = await client.replay(game, viewer)
                            reconcile_replay(replay, viewer, accepted[viewer], None)
                            atomic_json(directory / f'replay-player-{viewer}.json', replay)
                            if viewer == 2:
                                branch = next(decision for turn in replay['turns'] for decision in turn['decisions']
                                              if decision['actor'] == 2 and decision['observation']['state_revision'] == branch_revision)
                                assert branch['visible_events'] == attack_events
                            final = await client.observe(game, viewer)
                            if final['result'] is None:
                                journals[viewer].finish('truncated', None)
                            else:
                                journals[viewer].finish('natural', final['result'], final)
                            records = read_journal(journals[viewer].final_path)
                            assert not records[-1]['pending_decisions']
                            if final['result'] is None:
                                assert records[-1]['terminal_reward'] is None
                        completed.append(name)
                    finally:
                        atomic_json(directory / 'contract-diagnostic-trace.json', {'purpose': 'component-only-not-training', 'journal_scope': 'generated prefix and explicit target branches only; omitted/null actor action is in this diagnostic trace and Replay', 'trace': trace})
                        for journal in journals.values():
                            journal.close_incomplete()
                    await client.delete(game)
            assert len(completed) == 8
        finally:
            if client:
                await client.close()
            await server.stop()
        assert server.process.returncode == 0 and server.health['shutdown'] == 'stdin_eof'
    asyncio.run(run())
