"""Original J5 structural components, never qualified-source/parent evidence."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_training.journal import atomic_json, export_parquet, project_journals
from gf_training.preparation import ResourceLimit
from gf_training.q_data import load_run
from gf_training.structured_source import component_episode_records, inspect_component_collection
from tests.structured_source_support import collection_fixture

SETTINGS = PreparationConfig(reader='structured-v1', memory_budget_mib=512)


def inspect(directory, actions):
    return inspect_component_collection(directory, actions=actions, settings=SETTINGS, expected_split='train')


def test_original_dual_view_snapshots_ack_endpoints_and_nonqualification(tmp_path, make_input):
    directory, actions, manifest = collection_fixture(tmp_path, make_input)
    result = inspect(directory, actions)
    assert len(result.episodes) == 2 and result.proof['games'] == 1
    assert result.proof['accepted_decisions'] == 2 and result.proof['qualified'] is False
    assert result.proof['parent_qualification'] == 'not_performed'
    assert result.proof['closure_counts'] == {'natural': 1}
    assert len(result.proof['files']) == 9  # manifest/report/result +2journals/Replays/columns.
    assert all(item.records.outcome == 'draw' and len(item.records.frames) == 1 for item in result.episodes)
    assert result.episodes[0].records.bank.actor == 1 and result.episodes[1].records.bank.actor == 2
    altered = result.manifest
    altered['status'] = 'failed'
    assert result.manifest == json.loads(json.dumps(manifest))
    assert result.episodes[0].records.tables[0].synthetic_capacity is False


@pytest.mark.parametrize('kind', ['truncated', 'cancelled', 'incomplete'])
@pytest.mark.parametrize('pending', [False, True])
def test_interruption_keeps_only_positive_ack_and_exact_reason_without_targets(tmp_path, make_input, kind, pending):
    directory, actions, _ = collection_fixture(tmp_path, make_input, end_kind=kind, pending=pending)
    result = inspect(directory, actions)
    assert result.proof['accepted_decisions'] == (0 if pending else 2)
    for item in result.episodes:
        assert item.records.end_kind == item.records.reason == kind
        assert item.records.outcome is None and len(item.records.frames) == (0 if pending else 1)
    assert result.proof['qualified'] is False


def test_zero_decision_natural_and_unknown_delivery_never_invent_endpoint_or_outcome(tmp_path, make_input):
    natural = tmp_path / 'natural'
    natural.mkdir()
    directory, actions, _ = collection_fixture(natural, make_input, decisions=False)
    result = inspect(directory, actions)
    assert all(not item.records.frames and item.records.outcome == 'draw' for item in result.episodes)
    unknown = tmp_path / 'unknown'
    unknown.mkdir()
    directory, actions, _ = collection_fixture(unknown, make_input, end_kind='incomplete', pending=True, missing_replay=True)
    result = inspect(directory, actions)
    assert result.proof['accepted_decisions'] == 0
    assert all(item.records.reason == 'transport_failure' and item.records.outcome is None for item in result.episodes)
    source = directory / 'manifest.json'
    manifest = json.loads(source.read_text())
    manifest['episodes'][0]['health'] = {}
    atomic_json(source, manifest)
    atomic_json(directory / 'episode-0/result.json', manifest['episodes'][0])
    with pytest.raises(ContractError, match='missing original.*Replay'):
        inspect(directory, actions)


@pytest.mark.parametrize('mutation', ['replay', 'report', 'sidecar', 'missing_view', 'extra_column', 'result', 'profile', 'config', 'unknown_end'])
def test_original_mutation_or_conflict_rejects(tmp_path, make_input, mutation):
    directory, actions, manifest = collection_fixture(tmp_path, make_input)
    if mutation == 'replay':
        path = directory / 'episode-0/replay-player-1.json'
        replay = json.loads(path.read_text())
        replay['turns'] = []
        atomic_json(path, replay)
    elif mutation == 'report':
        atomic_json(directory / 'report.json', {})
    elif mutation == 'sidecar':
        path = next(directory.glob('*/*.columns'))
        path.write_bytes(b'corrupt')
    elif mutation == 'missing_view':
        (directory / 'episode-0/player-2.jsonl').unlink()
    elif mutation == 'extra_column':
        (directory / 'episode-0/foreign.columns').write_bytes(b'orphan')
    elif mutation == 'result':
        path = directory / 'episode-0/result.json'
        result = json.loads(path.read_text())
        result['game_id'] = 'other'
        atomic_json(path, result)
    else:
        if mutation == 'profile':
            del manifest['action_contract']
        elif mutation == 'config':
            manifest['effective_config']['actions']['action_contract']['adopted_rules_sha256'] = '0' * 64
            manifest['config_sha256'] = digest(manifest['effective_config'])
        else:
            manifest['episodes'][0]['end_kind'] = 'unknown'
        atomic_json(directory / 'manifest.json', manifest)
    with pytest.raises(ContractError):
        inspect(directory, actions)


@pytest.mark.parametrize('change', ['bytes', 'inventory'])
def test_final_recheck_binds_read_snapshots_and_file_inventory(tmp_path, make_input, monkeypatch, change):
    import gf_training.structured_source as module
    directory, actions, _ = collection_fixture(tmp_path, make_input)
    original = module._SourceOwner.finalize
    def mutate(owner):
        if change == 'bytes':
            path = owner.root / 'episode-0/player-1.jsonl'
            path.write_bytes(path.read_bytes() + b'\n')
        else:
            (owner.root / 'episode-0/added.columns').write_bytes(b'new')
        return original(owner)
    monkeypatch.setattr(module._SourceOwner, 'finalize', mutate)
    with pytest.raises(ContractError, match='changed|mismatch'):
        inspect(directory, actions)


def test_source_resource_admission_precedes_candidate_decode(tmp_path, make_input, monkeypatch):
    import gf_training.structured_source as module
    directory, actions, _ = collection_fixture(tmp_path, make_input)
    def forbidden(*args, **kwargs):
        pytest.fail('source reservation failure reached candidate decoding')
    monkeypatch.setattr(module, 'read_structured_journal', forbidden)
    with pytest.raises(ResourceLimit):
        inspect_component_collection(directory, actions=actions, settings=replace(SETTINGS, memory_budget_mib=1, cache_budget_mib=0, chunk_target_mib=1))


def test_explicit_schema4_neural_roles_produce_component_records_without_parent_authority(tmp_path, make_input):
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True, games=16)
    result = inspect(directory, actions)
    selected = component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])
    assert len(selected) == 16 and all(record.selections[0].value == 0. for record in selected)
    assert result.proof['qualified'] is False and result.proof['parent_qualification'] == 'not_performed'
    with pytest.raises(ContractError, match='exact original learner roles'):
        component_episode_records(result, learner_seats={name: 2 for name in manifest['ppo']['learner_seats']})


def test_baseline_role_cannot_be_imputed_into_ppo_values(tmp_path, make_input):
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4)
    result = inspect(directory, actions)
    with pytest.raises(ContractError, match='neural model2'):
        component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])


@pytest.mark.parametrize('operation', ['qdata', 'project', 'export'])
def test_public_qualified_paths_reject_engineering_before_legacy_decode_or_optional_arrow(tmp_path, make_input, monkeypatch, operation):
    import gf_training.journal as module
    directory, _, _ = collection_fixture(tmp_path, make_input)
    def forbidden(*args, **kwargs):
        pytest.fail('engineering source reached legacy journal decoding')
    monkeypatch.setattr(module, 'read_journal', forbidden)
    with pytest.raises(ContractError, match='engineering_only'):
        if operation == 'qdata':
            load_run(directory)
        elif operation == 'project':
            project_journals(list(directory.glob('*/player-*.jsonl')))
        else:
            export_parquet(list(directory.glob('*/player-*.jsonl')), tmp_path / 'never.parquet')
    assert not (tmp_path / 'never.parquet').exists()


def test_component_source_inspection_does_not_import_or_require_torch(tmp_path, make_input):
    directory, _, _ = collection_fixture(tmp_path, make_input, schema=4, neural=True)
    code = '''
import importlib.abc,importlib.metadata,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name=='torch' or name.startswith('torch.'): raise RuntimeError('Source imported Torch')
sys.meta_path.insert(0,NoTorch())
original=importlib.metadata.version
def version(name):
 if name=='torch': raise RuntimeError('Source required Torch metadata')
 return original(name)
importlib.metadata.version=version
from pathlib import Path
from gf_common.structured import action_config_v2
from gf_common.preprocessing import PreparationConfig
from gf_training.structured_source import inspect_component_collection
root=Path(sys.argv[1])
actions=action_config_v2(json.loads((root/'manifest.json').read_text())['effective_config']['actions'])
result=inspect_component_collection(root,actions=actions,settings=PreparationConfig(reader='structured-v1',memory_budget_mib=512))
assert result.proof['qualified'] is False and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(directory)], capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr


def test_fakehttp_original_collection_projects_to_actor_local_components(tmp_path, monkeypatch):
    import gf_cli
    from tests.test_structured_environment import instrument, structured_config
    instrument(monkeypatch)
    loaded = structured_config(tmp_path)
    directory, _ = asyncio.run(gf_cli.run_experiment(loaded, 'collect'))
    projected = inspect_component_collection(directory, actions=loaded.config.actions, settings=SETTINGS, expected_split='validation')
    assert projected.proof['games'] == 2 and projected.proof['accepted_decisions'] == 8
    assert len(projected.episodes) == 4 and all(len(item.records.frames) == 2 for item in projected.episodes)


def test_swapped_explicit_learner_roles_preserve_natural_reward_perspective(tmp_path, make_input):
    from dataclasses import asdict

    from gf_training.structured_rollout import StructuredRollout
    from tests.test_structured_network import small_model
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True, games=16,
                                                      outcome={'win': {'winner': 1}})
    result = inspect(directory, actions)
    records = component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])
    assert [record.bank.actor for record in records] == [1, 2] * 8
    rollout = StructuredRollout(records, small_model(), digest(asdict(actions)), action_contract=actions.action_contract,
                                purpose='component', synthetic_capacity=False)
    assert [float(episode.target[0]) for episode in rollout.episodes] == [1., -1.] * 8


def test_neural_missing_value_rejects_before_source_candidate_decode(tmp_path, make_input, monkeypatch):
    import gf_training.structured_journal as module
    directory, actions, _ = collection_fixture(tmp_path, make_input, schema=4, neural=True)
    path = directory / 'episode-0/player-1.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in rows if row['kind'] == 'decision')['selection']['value'] = None
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    def forbidden(*args, **kwargs):
        pytest.fail('missing neural value reached candidate ownership')
    monkeypatch.setattr(module, 'read_candidates', forbidden)
    with pytest.raises(ContractError, match='finite logprob/value'):
        inspect(directory, actions)


@pytest.mark.parametrize('field,value', [('failures', None), ('schema_version', True), ('planned_games', True)])
def test_report_requires_original_typed_health_ledger(tmp_path, make_input, field, value):
    from gf_models.arrays import sha256
    directory, actions, manifest = collection_fixture(tmp_path, make_input)
    report = json.loads((directory / 'report.json').read_text())
    report[field] = value
    atomic_json(directory / 'report.json', report)
    manifest['report_sha256'] = sha256(directory / 'report.json')
    atomic_json(directory / 'manifest.json', manifest)
    with pytest.raises(ContractError, match='health/count ledger'):
        inspect(directory, actions)


@pytest.mark.parametrize('kind', ['cancelled', 'incomplete'])
def test_new_interrupted_closures_never_create_targets_or_bootstrap(tmp_path, make_input, kind):
    from dataclasses import asdict

    from gf_training.structured_rollout import StructuredRollout
    from tests.test_structured_network import small_model
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True, end_kind=kind)
    result = inspect(directory, actions)
    records = component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])
    rollout = StructuredRollout(records, small_model(), digest(asdict(actions)), action_contract=actions.action_contract,
                                purpose='component', synthetic_capacity=False)
    assert len(rollout) == 1 and all(item.target is item.advantage is None for item in rollout.episodes)
    with pytest.raises(ContractError, match='natural nonempty complete'):
        rollout.require_learning()


@pytest.mark.parametrize('corruption', ['symlink_column', 'missing_natural_replay', 'extra_replay'])
def test_source_rejects_foreign_paths_and_natural_replay_inventory(tmp_path, make_input, corruption):
    directory, actions, _ = collection_fixture(tmp_path, make_input)
    if corruption == 'symlink_column':
        path = next(directory.glob('*/*.columns'))
        outside = tmp_path / 'outside'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif corruption == 'missing_natural_replay':
        (directory / 'episode-0/replay-player-1.json').unlink()
    else:
        atomic_json(directory / 'episode-0/replay-player-3.json', {})
    with pytest.raises(ContractError):
        inspect(directory, actions)


@pytest.mark.parametrize('missing_viewer', [1, 2])
def test_one_missing_interrupted_replay_is_not_conflicting_with_known_surviving_abort(tmp_path, make_input, missing_viewer):
    from dataclasses import asdict

    from gf_training.structured_rollout import StructuredRollout
    from tests.test_structured_network import small_model
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True,
        end_kind='truncated', missing_replay=missing_viewer, abort_reason='turn_limit_exceeded')
    result = inspect(directory, actions)
    assert result.proof['missing_replays'] == [f'episode-0/replay-player-{missing_viewer}.json']
    assert result.proof['accepted_decisions'] == 2 and result.proof['qualified'] is False
    assert result.proof['parent_qualification'] == 'not_performed'
    assert all(item.records.reason == 'turn_limit_exceeded' and item.records.outcome is None for item in result.episodes)
    records = component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])
    rollout = StructuredRollout(records, small_model(), digest(asdict(actions)), action_contract=actions.action_contract,
                                purpose='component', synthetic_capacity=False)
    assert all(item.target is item.advantage is None for item in rollout.episodes)
    with pytest.raises(ContractError):
        rollout.require_learning()
    present_path = directory / f'episode-0/replay-player-{3 - missing_viewer}.json'
    replay = json.loads(present_path.read_text())
    replay['episode_end']['reason'] = 'server_shutdown'
    atomic_json(present_path, replay)
    with pytest.raises(ContractError, match='abort cause/Replay'):
        inspect(directory, actions)


@pytest.mark.parametrize('explicit,expected', [(float('inf'), 225.), (210., 210.)])
def test_source_deadline_uses_original_settings_before_owner_or_io(tmp_path, make_input, monkeypatch, explicit, expected):
    import gf_training.structured_source as module
    directory, actions, _ = collection_fixture(tmp_path, make_input)
    captured = []
    class StopBeforeIO(Exception):
        pass
    def owner(self, directory, settings, deadline):
        captured.append(deadline)
        raise StopBeforeIO
    monkeypatch.setattr(module.time, 'monotonic', lambda: 200.)
    monkeypatch.setattr(module._SourceOwner, '__init__', owner)
    with pytest.raises(StopBeforeIO):
        inspect_component_collection(directory, actions=actions,
            settings=replace(SETTINGS, wall_seconds=30, cleanup_seconds=5), deadline=explicit)
    assert captured == [expected]


def test_all_natural_cohort_requires_success_report_even_if_manifest_incomplete(tmp_path, make_input):
    from gf_models.arrays import sha256
    directory, actions, manifest = collection_fixture(tmp_path, make_input)
    report = json.loads((directory / 'report.json').read_text())
    report.update(comparison_valid=False, failures={'durably_invalidated': 1})
    atomic_json(directory / 'report.json', report)
    manifest.update(status='incomplete', report_sha256=sha256(directory / 'report.json'))
    atomic_json(directory / 'manifest.json', manifest)
    with pytest.raises(ContractError, match='all-natural.*successful report'):
        inspect(directory, actions)


def test_mixed_incomplete_cohort_keeps_all_roles_zero_natural_and_interrupted_without_update(tmp_path, make_input):
    from dataclasses import asdict

    from gf_training.structured_rollout import StructuredRollout
    from tests.test_structured_network import small_model
    directory, actions, manifest = collection_fixture(tmp_path, make_input, schema=4, neural=True, games=2,
        end_kind=('natural', 'cancelled'), decisions=(False, True))
    result = inspect(directory, actions)
    assert len(result.episodes) == 4 and result.proof['closure_counts'] == {'natural': 1, 'cancelled': 1}
    assert all(item.records.outcome == 'draw' and not item.records.frames for item in result.episodes[:2])
    assert all(item.records.outcome is None and len(item.records.frames) == 1 for item in result.episodes[2:])
    records = component_episode_records(result, learner_seats=manifest['ppo']['learner_seats'])
    rollout = StructuredRollout(records, small_model(), digest(asdict(actions)), action_contract=actions.action_contract,
                                purpose='component', synthetic_capacity=False)
    assert len(rollout.episodes) == 2 and all(item.target is item.advantage is None for item in rollout.episodes)
    with pytest.raises(ContractError):
        rollout.require_learning()


def test_consistently_duplicated_game_and_participants_reject_before_table_decode(tmp_path, make_input, monkeypatch):
    import gf_training.structured_source as module
    directory, actions, manifest = collection_fixture(tmp_path, make_input, games=2)
    duplicate = manifest['episodes'][0]['game_id']
    manifest['episodes'][1]['game_id'] = duplicate
    atomic_json(directory / 'manifest.json', manifest)
    result_path = directory / 'episode-1/result.json'
    result = json.loads(result_path.read_text())
    result['game_id'] = duplicate
    atomic_json(result_path, result)
    for viewer in (1, 2):
        path = directory / f'episode-1/player-{viewer}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['game_id'] = duplicate
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        path = directory / f'episode-1/replay-player-{viewer}.json'
        replay = json.loads(path.read_text())
        replay['game_id'] = duplicate
        atomic_json(path, replay)
    def forbidden(*args, **kwargs):
        pytest.fail('duplicate original game reached journal/table decode')
    monkeypatch.setattr(module, 'read_structured_journal', forbidden)
    with pytest.raises(ContractError, match='duplicate.*game/participant'):
        inspect(directory, actions)


@pytest.mark.parametrize('seeds', [[True, 19], [17], [17, 19, 23], [17., 19], [2**64, 19]])
def test_original_schedule_seed_domain_is_strict_before_journal_decode(tmp_path, make_input, monkeypatch, seeds):
    import gf_training.structured_source as module
    directory, actions, manifest = collection_fixture(tmp_path, make_input)
    manifest['schedule'][0]['episodes'][0]['sampling_seeds'] = seeds
    atomic_json(directory / 'manifest.json', manifest)
    monkeypatch.setattr(module, 'read_structured_journal', lambda *args, **kwargs: pytest.fail('bad seed reached decode'))
    with pytest.raises(ContractError):
        inspect(directory, actions)


@pytest.mark.parametrize('neural', [False, True])
@pytest.mark.parametrize('field', ['sampling_seed', 'max_interval_events', 'max_episode_events', 'max_decisions', 'max_positions', 'max_candidates'])
def test_header_seed_and_caps_match_declared_original_policy_before_table_decode(tmp_path, make_input, monkeypatch, neural, field):
    import gf_training.structured_source as module
    directory, actions, _ = collection_fixture(tmp_path, make_input, neural=neural)
    path = directory / 'episode-0/player-1.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if field == 'sampling_seed':
        rows[0][field] += 1
    else:
        current = rows[0]['caps'][field]
        rows[0]['caps'][field] = 99 if current is None else current + 1
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(module, 'read_structured_journal', lambda *args, **kwargs: pytest.fail('mismatched header reached decode'))
    with pytest.raises(ContractError, match='sampling stream|caps differ'):
        inspect(directory, actions)


@pytest.mark.parametrize('capacity', ['positions', 'candidates'])
def test_saved_neural_policy_must_cover_original_action_capacity(tmp_path, make_input, monkeypatch, capacity):
    from dataclasses import asdict

    import gf_training.structured_source as module
    directory, actions, manifest = collection_fixture(tmp_path, make_input, neural=True)
    label = manifest['schedule'][0]['episodes'][0]['policies'][0]
    options = manifest['policy_bindings'][label]['policy_options']
    if capacity == 'positions':
        actions = replace(actions, max_sequence_length=2)
        manifest['effective_config']['actions'] = asdict(actions)
        manifest['config_sha256'] = digest(manifest['effective_config'])
        options['model']['max_positions'] = 1
        for viewer in (1, 2):
            path = directory / f'episode-0/player-{viewer}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['action_config_hash'] = digest(asdict(actions))
            rows[0]['caps']['max_positions'] = 2
            for row in rows:
                if row['kind'] == 'decision':
                    row['search_report']['config_hash'] = digest(asdict(actions))
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        options['inference']['max_candidates'] = 1
    atomic_json(directory / 'manifest.json', manifest)
    monkeypatch.setattr(module, 'read_structured_journal', lambda *args, **kwargs: pytest.fail('under-capacity policy reached decode'))
    with pytest.raises(ContractError, match='capacity.*cover'):
        inspect(directory, actions)


@pytest.mark.parametrize('manifest_complete,report_success', [(True, True), (True, False), (False, True)])
def test_interrupted_cohort_cannot_claim_completion_or_success(tmp_path, make_input, manifest_complete, report_success):
    from gf_models.arrays import sha256
    directory, actions, manifest = collection_fixture(tmp_path, make_input, end_kind='truncated')
    manifest['status'] = 'complete' if manifest_complete else 'incomplete'
    report = json.loads((directory / 'report.json').read_text())
    report.update(comparison_valid=report_success, failures={})
    atomic_json(directory / 'report.json', report)
    manifest['report_sha256'] = sha256(directory / 'report.json')
    atomic_json(directory / 'manifest.json', manifest)
    with pytest.raises(ContractError, match='must be all natural'):
        inspect(directory, actions)


@pytest.mark.parametrize('late_change', ['inventory', 'earlier_file'])
def test_final_hash_sweep_cannot_hide_late_inventory_or_earlier_file_mutation(tmp_path, make_input, monkeypatch, late_change):
    import gf_training.structured_source as module
    directory, actions, _ = collection_fixture(tmp_path, make_input)
    actual = module._bounded_hash
    calls = []
    def hash_with_mutation(source, size):
        result = actual(source, size)
        calls.append(1)
        if len(calls) == 2:
            if late_change == 'inventory':
                (directory / 'episode-0/late.columns').write_bytes(b'late')
            else:
                path = directory / 'manifest.json'
                path.write_bytes(path.read_bytes() + b'\n')
        return result
    monkeypatch.setattr(module, '_bounded_hash', hash_with_mutation)
    with pytest.raises(ContractError, match='after hash sweep'):
        inspect(directory, actions)


@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('case', ['one_byte', 'expired', 'few_fit'])
def test_source_inventory_budget_and_deadline_stop_scandir_before_thousand(tmp_path, monkeypatch, final, case):
    from types import SimpleNamespace

    import gf_training.structured_source as module
    owner = module._SourceOwner(tmp_path, SETTINGS, float('inf'))
    consumed = []
    class Entries:
        def __init__(self, directory): self.directory = directory
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def __iter__(self): return self
        def __next__(self):
            if self.directory:
                if hasattr(self, 'done'): raise StopIteration
                self.done = True
                return SimpleNamespace(name='episode', path=str(tmp_path / 'episode'), is_symlink=lambda: False,
                                       is_dir=lambda **kwargs: True)
            index = len(consumed)
            if index >= 1000: raise StopIteration
            consumed.append(index)
            return SimpleNamespace(name=f'e{index:04}.columns', is_file=lambda **kwargs: True)
    monkeypatch.setattr(module.os, 'scandir', lambda path: Entries(str(path) == str(tmp_path)))
    owner.used = SETTINGS.memory_budget_mib * 1024**2 - (1 if case == 'one_byte' else 6000)
    if case == 'expired':
        owner.deadline = -1.
    with pytest.raises((ResourceLimit, TimeoutError)):
        if final:
            owner.inventories = [('*/*.columns', ())]
            owner.finalize()
        else:
            owner.inventory('*/*.columns')
    assert len(consumed) < 1000
    if case != 'few_fit':
        assert not consumed
    else:
        assert consumed
