"""Configured refresh baseline crosses runtime/storage boundaries explicitly."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli.card_instance import _baseline_labels
from gf_cli.config import load_config
from gf_common.card_instance import card_instance_contract
from gf_common.domain import ContractError, RuntimePolicyBinding
from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label
from gf_models.refresh_heuristic import RefreshHeuristicPolicy
from gf_models.refresh_heuristic_config import RefreshHeuristicConfig
from gf_models.structured_runtime import own_game_worker_settings_v3
from gf_simulator_client.card_instance_worker import CardInstanceParticipantWorker
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_collection import collection_metadata
from gf_training.card_instance_journal import (
    CardInstanceJournal,
    _Context,
    read_card_instance_journal,
)
from tests.test_card_instance_baseline_runtime import ACTIONS, RULES, game, receipt

PRESET = Path(__file__).parents[1] / 'configs/experiments/heuristic-refresh-baseline.toml'


@pytest.mark.parametrize('seat', ['model', 'opponent'])
@pytest.mark.parametrize('threshold', [12, 7])
def test_refresh_preset_selects_each_seat_before_legacy_defaults(seat, threshold):
    loaded = load_config(PRESET, {'model': {'model_id': 'heuristic_refresh', 'attack_threshold': threshold},
        **({'opponent': {'model_id': 'heuristic_refresh', 'attack_threshold': threshold}} if seat == 'opponent' else {})})
    configured = getattr(loaded.config, seat)
    assert type(configured) is RefreshHeuristicConfig
    assert asdict(configured) == {'model_id': 'heuristic_refresh', 'version': '1', 'attack_threshold': threshold}
    bindings, labels = _baseline_labels(loaded)
    label = labels[0 if seat == 'model' else 1]
    assert bindings[label] == RuntimePolicyBinding('heuristic_refresh', {'version': '1', 'attack_threshold': threshold})
    assert label.startswith('heuristic_refresh-card-instance-v2-') and len(label.rsplit('-', 1)[1]) == 64


@pytest.mark.parametrize('options', [None, {}, {'version': '1', 'attack_threshold': True}, {'version': '1', 'attack_threshold': 0}, {'version': '1', 'attack_threshold': 12, 'checkpoint': 'x'}])
def test_refresh_settings_reject_nonclosed_options(options):
    with pytest.raises(ContractError):
        own_game_worker_settings_v3('heuristic_refresh', ACTIONS, options, rules=RULES)


def test_refresh_rejects_preceding_layout_and_keeps_baseline_costs():
    options = asdict(RefreshOptions())
    assert own_game_worker_settings_v3('heuristic_refresh', ACTIONS, options, rules=RULES) == own_game_worker_settings_v3('heuristic', ACTIONS, None, rules=RULES)
    with pytest.raises(ContractError):
        own_game_worker_settings_v3('heuristic_refresh', replace(ACTIONS, action_contract=card_instance_contract()), options, rules=RULES)


@pytest.mark.parametrize('threshold', [7, 12])
def test_real_refresh_worker_bootstrap_returns_exact_configured_identity(threshold):
    async def run():
        ledger = ReservationLedger()
        options = RefreshOptions(attack_threshold=threshold)
        worker = CardInstanceParticipantWorker('heuristic_refresh', 17, ACTIONS, ledger=ledger, policy_options=asdict(options))
        frame, bank, delta, generated = game()
        expected = RefreshHeuristicPolicy(ACTIONS, rules=RULES, options=options)
        expected.reset(17)
        try:
            actual_generated, selected = await worker.select_frame(frame, delta)
            assert selected.policy_version == refresh_policy_label(ACTIONS, RULES, options)
            assert selected.logprob is selected.value is None
            assert selected.validate(actual_generated.table, frame) == expected.select_frame(frame, bank.view(), generated).validate(generated.table, frame)
            assert worker.neural_metadata is worker.tree_metadata is None
            assert worker.baseline_options == asdict(options)
        finally:
            await worker.close()
        assert not ledger.workers and worker.process.poll() is not None
    asyncio.run(run())


def refresh_journal(tmp_path, threshold=12):
    options = RefreshOptions(attack_threshold=threshold)
    caps, costs, _, _ = own_game_worker_settings_v3('heuristic', ACTIONS, None, rules=RULES)
    return CardInstanceJournal(tmp_path, 'episode', 1, refresh_policy_label(ACTIONS, RULES, options),
        run_id='run', game_id='game', actions=ACTIONS, bank_episode='episode', caps=asdict(caps),
        numeric_costs=asdict(costs), model_id='heuristic_refresh', baseline_options=asdict(options), runtime_evidence=receipt())


def test_refresh_journal_roundtrip_has_no_learned_metadata(tmp_path):
    journal = refresh_journal(tmp_path)
    journal.finish('incomplete', None)
    data = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert data.header['baseline_options'] == {'version': '1', 'attack_threshold': 12}
    assert data.header['model_version'] == '1'
    assert 'neural' not in data.header and 'tree' not in data.header


@pytest.mark.parametrize('mutation', ['missing', 'threshold', 'label', 'model', 'caps', 'costs'])
def test_refresh_journal_rejects_conditional_key_identity_and_settings_tampering(tmp_path, mutation):
    journal = refresh_journal(tmp_path)
    journal.finish('incomplete', None)
    header = copy.deepcopy(journal.header)
    if mutation == 'missing':
        del header['baseline_options']
    elif mutation == 'threshold':
        header['baseline_options']['attack_threshold'] = 13
    elif mutation == 'label':
        header['policy'] += '0'
    elif mutation == 'model':
        header.update(model_id='heuristic', policy='heuristic-card-instance-v2')
    elif mutation == 'caps':
        header['caps']['max_episode_events'] += 1
    else:
        header['numeric_costs']['context_width'] += 1
    with pytest.raises(ContractError):
        _Context(ACTIONS, RULES).validate_header(header)


@pytest.mark.parametrize("hpo", [False, True])
def test_refresh_baseline_only_collection_registry_keeps_model_version_one(hpo):
    label = refresh_policy_label(ACTIONS, RULES, RefreshOptions())
    registry = {label: RuntimePolicyBinding('heuristic_refresh', asdict(RefreshOptions()))}
    raw = collection_metadata(actions=ACTIONS, config={}, runtime_evidence=[receipt()], schedule=[{}],
        inventory=[{}], policies=[label], bots_revision='a' * 40, policy_bindings=registry, _hpo=hpo)
    assert raw['model_version'] == '1' and raw['policy_bindings'][label]['policy_options']['attack_threshold'] == 12
    assert 'neural_inputs' not in raw and 'tree_inputs' not in raw
    bad = {label: RuntimePolicyBinding('heuristic_refresh', {'version': '1', 'attack_threshold': 13})}
    with pytest.raises(ContractError):
        collection_metadata(actions=ACTIONS, config={}, runtime_evidence=[receipt()], schedule=[{}],
            inventory=[{}], policies=[label], bots_revision='a' * 40, policy_bindings=bad)


def test_retained_neural_owner_skips_configured_baseline_artifacts(tmp_path):
    import time

    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_training.card_instance_source import recorded_plan
    from gf_training.lineage import run_lock
    from tests.card_instance_learning_support import ACTIONS as LEARNING_ACTIONS
    from tests.card_instance_learning_support import initial_parent
    with run_lock(tmp_path / 'parent', create=True):
        _, state, records, _ = initial_parent(tmp_path / 'parent')
        plan, _, _ = recorded_plan(tmp_path / 'parent', state, records, 0, time.monotonic() + 60)
        registry = dict(plan.bindings)
        label = refresh_policy_label(LEARNING_ACTIONS, RULES, RefreshOptions())
        registry[label] = RuntimePolicyBinding('heuristic_refresh', asdict(RefreshOptions()))
        owner = RetainedGameArtifacts.for_registry(registry, LEARNING_ACTIONS, rules=RULES,
            max_owned_bytes=1024**3, preflight_deadline=time.monotonic() + 60, finalization_deadline=time.monotonic() + 120)
        try:
            assert set(owner.finalize(deadline=time.monotonic() + 60)['policies']) == {key for key, value in registry.items() if value.model_id == 'mlp'}
        finally:
            owner.release()


@pytest.mark.parametrize('tamper', [False, True])
def test_original_source_neural_refresh_registry_and_header_correspondence(tmp_path, tamper):
    import json
    import time

    from gf_common.domain import canonical, digest
    from gf_models.arrays import sha256
    from gf_training.card_instance_source import recorded_plan
    from gf_training.journal import atomic_json
    from gf_training.lineage import run_lock
    from gf_training.structured_source import _inspect_component_collection, _SourceOwner
    from tests.card_instance_learning_support import ACTIONS as LEARNING_ACTIONS
    from tests.card_instance_learning_support import initial_parent, original_collection
    root = tmp_path / 'parent'
    with run_lock(root, create=True):
        learner, state, records, settings = initial_parent(root)
        plan, _, _ = recorded_plan(root, state, records, 0, time.monotonic() + 60)
        manifest = original_collection(plan, state, learner, decisions=False)
        old = 'heuristic-card-instance-v2'
        label = refresh_policy_label(LEARNING_ACTIONS, RULES, RefreshOptions())
        manifest['policies'] = [label if value == old else value for value in manifest['policies']]
        del manifest['policy_bindings'][old]
        manifest['policy_bindings'][label] = asdict(RuntimePolicyBinding('heuristic_refresh', asdict(RefreshOptions())))
        for group in manifest['schedule']:
            for spec in group['episodes']:
                spec['policies'] = [label if value == old else value for value in spec['policies']]
        for episode in manifest['episodes']:
            episode['policies'] = [label if value == old else value for value in episode['policies']]
            path = plan.output / episode['episode_id'] / 'result.json'
            atomic_json(path, episode)
        for path in plan.output.glob('*/player-*.jsonl'):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if rows[0]['policy'] == old:
                rows[0].update(policy=label, model_id='heuristic_refresh', baseline_options=asdict(RefreshOptions()))
                path.write_text(''.join(canonical(row) + '\n' for row in rows))
        manifest['linkage'].update(schedule_digest=digest(manifest['schedule']), policy_registry_digest=digest(manifest['policy_bindings']))
        manifest['files'] = [{'path': row['path'], 'bytes': (plan.output / row['path']).stat().st_size,
            'sha256': sha256(plan.output / row['path'])} for row in manifest['files']]
        if tamper:
            manifest['policy_bindings'][label]['policy_options']['attack_threshold'] = 13
            manifest['linkage']['policy_registry_digest'] = digest(manifest['policy_bindings'])
        atomic_json(plan.output / 'manifest.json', manifest)
        owner = _SourceOwner(plan.output, settings, time.monotonic() + 60, max_owned_bytes=1024**3)
        try:
            if tamper:
                with pytest.raises(ContractError):
                    _inspect_component_collection(owner, LEARNING_ACTIONS, 'train', _game=True)
            else:
                source = _inspect_component_collection(owner, LEARNING_ACTIONS, 'train', _game=True)
                assert len(source.episodes) == 32
                assert {ep.model_id for ep in source.episodes} == {'random', 'heuristic_refresh', 'mlp'}
        finally:
            owner.close()


def test_refresh_legacy_layout_config_fails_as_contract_error():
    with pytest.raises(ContractError):
        load_config(Path(__file__).parents[1] / 'configs/experiments/baseline.toml', {'model': {'model_id': 'heuristic_refresh'}})


@pytest.mark.parametrize('tamper', [False, True])
def test_environment_configured_baseline_needs_no_artifact_owner(tmp_path, monkeypatch, tamper):
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.environment import EnvironmentSettings
    binary = tmp_path / 'binary'
    runtime = CardInstanceRuntimeEvidence(binary, tmp_path)
    # Isolate runtime identity I/O; this test exercises registry admission only.
    monkeypatch.setattr(runtime, 'preflight', lambda: None)
    label = refresh_policy_label(ACTIONS, RULES, RefreshOptions())
    options = {'version': '1', 'attack_threshold': 13 if tamper else 12}
    settings = EnvironmentSettings(str(binary), 'unused', tmp_path / 'out', 'run', ACTIONS,
        policy_bindings={label: RuntimePolicyBinding('heuristic_refresh', options)})
    if tamper:
        with pytest.raises(ContractError):
            CardInstanceSimulatorEnvironment(settings, CardInstanceJournal, ledger=ReservationLedger(), runtime_evidence=runtime)
    else:
        environment = CardInstanceSimulatorEnvironment(settings, CardInstanceJournal, ledger=ReservationLedger(), runtime_evidence=runtime)
        assert environment.artifact_inputs is None


def test_refresh_journal_roundtrips_actual_selection_and_pending_command(tmp_path):
    import uuid

    from gf_common.diverse_actions import NumericCosts
    from gf_models.structured_runtime import WorkerCaps
    frame, bank, delta, generated = game(seed=17)
    policy = RefreshHeuristicPolicy(ACTIONS, rules=RULES, options=RefreshOptions())
    policy.reset(17)
    selected = policy.select_frame(frame, bank.view(), generated)
    journal = CardInstanceJournal(tmp_path, 'episode', 1, policy.version, run_id='run', game_id='game',
        actions=ACTIONS, bank_episode=bank.episode, caps=asdict(WorkerCaps(max_positions=4, max_candidates=50)),
        numeric_costs=asdict(NumericCosts()), model_id='heuristic_refresh', baseline_options=asdict(RefreshOptions()),
        runtime_evidence=receipt(), sampling_seed=17)
    command = {'command_id': str(uuid.uuid4()), 'expected_revision': frame.revision,
        'action': selected.validate(generated.table, frame)}
    journal.begin_frame(frame, delta, generated, selected, command, {})
    journal.finish('incomplete', None)
    actual = read_card_instance_journal(journal.final_path, actions=ACTIONS)
    assert actual.decisions[0].selection == selected
    assert actual.decisions[0].command == command
    assert actual.accepted == () and actual.end['pending_decisions'] == [0]


@pytest.mark.parametrize('mutation', ['options', 'caps'])
def test_refresh_child_revalidates_configured_baseline_before_ready(monkeypatch, mutation):
    async def run():
        ledger = ReservationLedger()
        worker = CardInstanceParticipantWorker('heuristic_refresh', 17, ACTIONS, ledger=ledger, policy_options=asdict(RefreshOptions()))
        original = worker._roundtrip
        async def corrupt(payload):
            payload = copy.deepcopy(payload)
            if mutation == 'options':
                payload['policy_options']['attack_threshold'] = True
            else:
                payload['caps']['max_episode_events'] += 1
            return await original(payload)
        monkeypatch.setattr(worker, '_roundtrip', corrupt)
        try:
            with pytest.raises((ContractError, RuntimeError)):
                await worker.initialize()
        finally:
            await worker.close()
        assert not ledger.workers and worker.process.poll() is not None
    asyncio.run(run())


def test_old_native_refresh_refusal_precedes_runtime_or_artifact_work(monkeypatch):
    from types import SimpleNamespace

    import gf_cli.card_instance as runtime
    import gf_training.native_backend as backend
    from gf_common.preprocessing import PreparationConfig
    from gf_models.structured_binding import ConfiguredNeuralModelV3
    loaded = load_config(PRESET)
    configured = replace(loaded.config, model=ConfiguredNeuralModelV3(checkpoint='/unused'),
        opponent=RefreshHeuristicConfig(), preprocessing=PreparationConfig(backend='rust', reader='participant-structured-v2'))
    loaded = replace(loaded, config=configured)
    monkeypatch.setattr(backend, 'load_native', lambda: SimpleNamespace(build_info=list))
    monkeypatch.setattr(runtime, 'CardInstanceRuntimeEvidence', lambda *a, **k: pytest.fail('old native reached runtime I/O'))
    with pytest.raises(ContractError, match='refresh'):
        asyncio.run(runtime.run_inference(loaded, 'smoke'))


@pytest.mark.parametrize('seat', ['model', 'opponent'])
@pytest.mark.parametrize('source', ['file', 'override'])
@pytest.mark.parametrize('threshold', [2**130, 0, -1, True, 12.0])
def test_refresh_config_threshold_domain_matches_runtime_options(tmp_path, seat, source, threshold):
    import json

    settings = {'model_id': 'heuristic_refresh', 'attack_threshold': threshold}
    if source == 'override':
        load = lambda: load_config(PRESET, {seat: settings})
    else:
        preset = PRESET.read_text()
        for relative in ('../actions/card-instance-v2-corrected.toml', '../models/heuristic-refresh.toml', '../models/heuristic.toml'):
            preset = preset.replace(json.dumps(relative), json.dumps(str((PRESET.parent / relative).resolve())))
        literal = json.dumps(threshold)
        experiment = tmp_path / 'refresh-threshold.toml'
        experiment.write_text(preset + f'\n[{seat}]\nmodel_id = "heuristic_refresh"\nattack_threshold = {literal}\n')
        load = lambda: load_config(experiment)
    if type(threshold) is int and threshold > 0:
        loaded = load()
        configured = getattr(loaded.config, seat)
        assert type(configured) is RefreshHeuristicConfig
        assert configured.attack_threshold == 2**130
        bindings, labels = _baseline_labels(loaded)
        assert bindings[labels[0 if seat == 'model' else 1]].policy_options == {
            'version': '1', 'attack_threshold': 2**130}
    else:
        with pytest.raises(ContractError):
            load()
