"""Hand-authored complete originals test Source mechanics, not game acceptance."""
import copy
import json
import time
from contextlib import contextmanager

import pytest

from gf_common.domain import ContractError, canonical
from gf_models.arrays import sha256
from gf_training.card_instance_source import (
    _open_batch,
    recorded_plan,
    validate_game_parent,
    validate_owned_collection,
)
from gf_training.lineage import run_lock
from gf_training.structured_lineage import _ChainScan, _validate_component_chain
from tests.card_instance_learning_support import initial_parent, original_collection


@contextmanager
def structural_source(tmp_path, *, decisions=True):
    directory = tmp_path / 'parent'
    with run_lock(directory, create=True):
        learner, state, records, settings = initial_parent(directory)
        deadline = time.monotonic() + 60
        plan, _, _ = recorded_plan(directory, state, records, 0, deadline)
        original_collection(plan, state, learner, decisions=decisions)
        scan = _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline)
        _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=True)
        with _open_batch(plan, state, records, scan, settings, online=None, max_owned_bytes=1024**3, deadline=deadline) as source:
            yield source


def test_original16_roles_only_and_numeric_correspondence(tmp_path):
    with structural_source(tmp_path) as source:
        assert source.proof['qualified'] is False and source.proof['parent_qualification'] == 'not_performed'
        assert source.proof['games'] == 16 and len(source.records) == 16
        assert len(source.snapshot.collection.episodes) == 32
        assert {record.bank.actor for record in source.records} == {1, 2}
        assert sum(len(record.frames) for record in source.records) == 16
        assert source.proof['accepted_decisions'] > 16  # Self-play opponent rows are excluded explicitly.
        numeric = source.plan.output.parent.parent / 'numeric/batch-000000'
        source.write_numeric(numeric)
        rollout = source.read_numeric(numeric, sha256(numeric / 'rollout.json'))
        rollout.require_learning()
        assert len(rollout) == 16
        assert tuple(item.records for item in rollout.episodes) == source.records
        assert all(item.records.bank.actor == source.plan.linkage['learner_seats'][item.records.bank.episode.rsplit('-', 1)[0]] for item in rollout.episodes)
        assert all(item.target[-1] == 0 for item in rollout.episodes)
        with pytest.raises(ContractError, match='online'):
            source.issue(numeric, {'rollout.json': 'a' * 64})
        with pytest.raises(ContractError, match='complete nonzero'):
            validate_game_parent(source.controls.root, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)


def test_empty_natural16_original_episodes_do_not_invent_endpoints(tmp_path):
    with structural_source(tmp_path, decisions=False) as source:
        assert len(source.records) == 16 and all(not record.frames for record in source.records)
        numeric = source.controls.root / 'numeric/batch-000000'
        source.write_numeric(numeric)
        rollout = source.read_numeric(numeric, sha256(numeric / 'rollout.json'))
        with pytest.raises(ContractError, match='nonempty'):
            rollout.require_learning()


@pytest.mark.parametrize('mutation', ['journal', 'sidecar', 'inventory', 'control', 'policy'])
def test_original_mutation_during_projection_rejects_final_guard(tmp_path, mutation):
    with pytest.raises(ContractError), structural_source(tmp_path) as source:
        if mutation == 'journal':
            path = next(source.plan.output.glob('*/player-1.jsonl'))
        elif mutation == 'sidecar':
            path = next(source.plan.output.glob('*/*.columns'))
        elif mutation == 'inventory':
            path = source.plan.output / 'late-diagnostic.log'
        elif mutation == 'control':
            path = source.controls.root / 'run.json'
        else:
            path = source.controls.root / 'artifacts/000000/inference.json'
        path.write_bytes((path.read_bytes() if path.exists() else b'') + b'\n')


@pytest.mark.parametrize('mutation', ['role', 'missing_event', 'receipt', 'file_hash', 'cohort', 'policy'])
def test_original_cohort_mixed_role_and_missing_event_refusal(tmp_path, mutation):
    directory = tmp_path / 'parent'
    with run_lock(directory, create=True):
        learner, state, records, settings = initial_parent(directory)
        deadline = time.monotonic() + 60
        plan, _, _ = recorded_plan(directory, state, records, 0, deadline)
        manifest = original_collection(plan, state, learner)
        if mutation == 'role':
            name = next(iter(manifest['linkage']['learner_seats']))
            manifest['linkage']['learner_seats'][name] = 3 - manifest['linkage']['learner_seats'][name]
        elif mutation == 'missing_event':
            path = next(plan.output.glob('*/*.columns')).with_name('player-1.jsonl')
            if not any(json.loads(line).get('kind') == 'decision' for line in path.read_text().splitlines()):
                path = path.with_name('player-2.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(''.join(canonical(row) + '\n' for row in rows if row['kind'] != 'events'))
            for entry in manifest['files']:
                if entry['path'] == path.relative_to(plan.output).as_posix():
                    entry.update(sha256=sha256(path), bytes=path.stat().st_size)
        elif mutation == 'receipt':
            manifest['runtime_evidence'][0]['executed'] = None
        elif mutation == 'file_hash':
            manifest['files'][0]['sha256'] = 'f' * 64
        elif mutation == 'cohort':
            manifest['episodes'].pop()
        else:
            manifest['policy_bindings'][next(key for key, row in manifest['policy_bindings'].items() if row['policy_options'])]['model_id'] = 'gru'
        (plan.output / 'manifest.json').write_text(canonical(manifest))
        scan = _ChainScan(directory=directory, max_owned_bytes=1024**3, deadline=deadline)
        _validate_component_chain(directory, require_complete=False, require_retained=False, scan=scan, _game=True)
        with pytest.raises(ContractError), _open_batch(plan, state, records, scan, settings, online=None, max_owned_bytes=1024**3, deadline=deadline):
            pytest.fail('malformed original fixture passed Source inspection')


def test_offline_and_copied_metadata_cannot_supply_live_learner_authority(tmp_path):
    from gf_simulator_client.structured_resources import ReservationLedger
    with structural_source(tmp_path) as source:
        plan = copy.deepcopy(source.plan)
        with pytest.raises(ContractError, match='SAME learner'):
            validate_owned_collection(plan, ReservationLedger(), max_owned_bytes=1024**3, deadline=time.monotonic() + 30)


@pytest.mark.parametrize('mutation', [None, 'after-source', 'after-control', 'after-numeric', 'after-policy',
    'before-source', 'before-control', 'before-numeric', 'before-policy', 'before-config'])
@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_live_projection_real_wire_update_and_precommit_guard(tmp_path, mutation, device):
    import asyncio
    from dataclasses import asdict

    from gf_common.domain import digest
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.card_instance_source import _OnlineGameOwner, game_batch_source
    from gf_training.structured_learning_process import GameLearningProcess
    from gf_training.structured_lineage import commit_game_generation, validate_game_chain
    from tests.test_card_instance_learning import ACTIONS

    async def run(directory, ownership):
        learner, state, records, _ = initial_parent(directory, device=device)
        deadline = time.monotonic() + 120
        plan, _, after_rng = recorded_plan(directory, state, records, 0, deadline)
        original_collection(plan, state, learner)
        ledger = ReservationLedger()
        child = GameLearningProcess(directory, 1024**2, learner.model, learner.binding,
            digest(asdict(ACTIONS)), state['resume_identity'], actions=ACTIONS, ledger=ledger,
            cuda_visible_devices='0' if device != 'cpu' else '', cleanup_deadline=deadline)
        online = _OnlineGameOwner(ownership, state, child, ledger)
        child.online_owner = online
        await child.start()
        try:
            await child.initialize(17, directory / 'unused', restore=directory / 'artifacts/000000', deadline=deadline)
            with game_batch_source(plan, ledger, max_owned_bytes=1024**3, deadline=deadline) as source:
                numeric = directory / 'numeric/batch-000000'
                manifest = source.write_numeric(numeric)
                files = {'rollout.json': sha256(numeric / 'rollout.json'), **manifest['data']['files']}
                rollout = source.read_numeric(numeric, files['rollout.json'])
                count = len(rollout)
                del rollout
                for name, checksum in files.items():
                    with source.controls.file(numeric / name, checksum):
                        pass
                handle = source.issue(numeric, files)
                expected = handle.expectation
                expected['expected_optimizer_steps'] = 100
                assert handle.expectation['expected_optimizer_steps'] == 0
                if mutation and mutation.startswith('before-'):
                    from pathlib import Path
                    path = (next(plan.output.glob('*/player-1.jsonl')) if mutation == 'before-source' else
                        directory / 'run.json' if mutation == 'before-control' else
                        numeric / 'rollout.json' if mutation == 'before-numeric' else
                        Path(state['effective_config']['server']['config']) if mutation == 'before-config' else
                        directory / 'artifacts/000000/inference.json')
                    path.write_bytes(path.read_bytes() + b'\n')
                    with pytest.raises(ContractError):
                        await child.update(handle, directory / 'artifacts/000001', deadline=deadline)
                    assert child.steps == 0 and not (directory / 'artifacts/000001').exists()
                    return
                reply = await child.update(handle, directory / 'artifacts/000001', deadline=deadline)
                assert child.steps == 4 and handle.completed
                with pytest.raises(ContractError):
                    await child.update(handle, directory / 'forbidden', deadline=deadline)
                if mutation:
                    path = (next(plan.output.glob('*/player-1.jsonl')) if mutation == 'after-source' else
                        directory / 'run.json' if mutation == 'after-control' else
                        numeric / 'rollout.json' if mutation == 'after-numeric' else directory / 'artifacts/000000/inference.json')
                    path.write_bytes(path.read_bytes() + b'\n')
                    with pytest.raises(ContractError):
                        handle.prepare_commit()
                    assert not (directory / 'generations/000001.json').exists()
                    # The enclosing Source exit also reports the same mutation.
                    return
                state.update(accepted_decisions=count, optimizer_steps=child.steps, collection_seconds=1., update_seconds=1.)
                update = {'schema_version': 3, 'status': 'complete', 'run_id': state['run_id'], 'generation': 1,
                    'parent_sha256': state['generation_sha256'], 'batch_id': plan.linkage['batch_id'],
                    'collection': str(plan.output.relative_to(directory)), 'collection_manifest_sha256': source.proof['manifest_sha256'],
                    'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': files['rollout.json'], 'accepted_decisions': count,
                    'source_proof': source.proof, 'metrics': reply['metrics'], 'planned_games': 16, 'actual_games': 16,
                    'health': {}, 'policy_bindings': {key: asdict(value) for key, value in plan.bindings.items()},
                    'learner_seats': plan.linkage['learner_seats'], 'collection_seconds': 1., 'update_seconds': 1.,
                    'model_id': 'mlp', 'model_version': '3', 'input_recipe': state['input_recipe'], 'numeric_files': files,
                    **{key: state[key] for key in ('action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature')}}
                commit_game_generation(directory, state, generation=1, update=update,
                    controls={'schedule_opponent_rng': after_rng, 'snapshot_pool': [0], 'collection_position': 1},
                    artifact_files=reply['artifact']['files'], projection=handle, max_owned_bytes=1024**3, deadline=deadline)
                assert handle.prepared
            assert not handle.active and online.collection_controls is None
            state2, chain = validate_game_chain(directory, require_complete=False, max_owned_bytes=1024**3, deadline=deadline)
            assert state2['updates'] == 1 and len(chain) == 2
            validate_game_parent(directory, require_complete=False, max_owned_bytes=1024**3, deadline=deadline)
        finally:
            await child.close(deadline)
        assert ledger.learner is None and not ledger.workers and not ledger.updating

    directory = tmp_path / 'parent'
    with run_lock(directory, create=True) as ownership:
        if mutation:
            with pytest.raises(ContractError):
                asyncio.run(run(directory, ownership))
        else:
            asyncio.run(run(directory, ownership))


@pytest.mark.parametrize('mutation', ['lock', 'steps', 'wire', 'lease', 'config-replacement', 'last-decrease', 'last-nonfinite', 'foreign-field', 'last-progress'])
def test_live_collector_guard_rejects_copied_owner_and_control_replacement(tmp_path, mutation):
    import asyncio
    import os
    from dataclasses import asdict
    from pathlib import Path

    from gf_common.domain import digest
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.card_instance_source import _OnlineGameOwner
    from gf_training.structured_learning_process import GameLearningProcess
    from tests.test_card_instance_learning import ACTIONS

    async def run(directory, ownership):
        learner, state, records, _ = initial_parent(directory)
        deadline = time.monotonic() + 60
        plan, _, _ = recorded_plan(directory, state, records, 0, deadline)
        ledger = ReservationLedger()
        child = GameLearningProcess(directory, 1024**2, learner.model, learner.binding,
            digest(asdict(ACTIONS)), state['resume_identity'], actions=ACTIONS, ledger=ledger,
            cuda_visible_devices='', cleanup_deadline=deadline)
        online = _OnlineGameOwner(ownership, state, child, ledger)
        child.online_owner = online
        await child.start()
        try:
            await child.initialize(17, directory / 'unused', restore=directory / 'artifacts/000000', deadline=deadline)
            assert validate_owned_collection(plan, ledger, max_owned_bytes=1024**3, deadline=deadline) is online
            previous = (online.ownership, child.steps, copy.deepcopy(child.identity), child.lease._cleanup_owner)
            saved_state = copy.deepcopy(state)
            try:
                if mutation == 'lock':
                    online.ownership = type(ownership)(directory, 'copied-token')
                elif mutation == 'steps':
                    child.steps += 4
                elif mutation == 'wire':
                    child.identity['execution_digest'] = 'f' * 64
                elif mutation == 'lease':
                    child.lease._cleanup_owner = object()
                elif mutation == 'last-decrease':
                    state['last_unix'] -= 1
                elif mutation == 'last-nonfinite':
                    state['last_unix'] = float('inf')
                elif mutation == 'foreign-field':
                    state['foreign'] = True
                elif mutation == 'last-progress':
                    state['last_unix'] += .001
                else:
                    config = Path(state['effective_config']['server']['config'])
                    replacement = config.with_suffix('.replacement')
                    replacement.write_bytes(config.read_bytes())
                    os.replace(replacement, config)  # Same bytes are not the original owned file.
                if mutation == 'last-progress':
                    assert validate_owned_collection(plan, ledger, max_owned_bytes=1024**3, deadline=deadline) is online
                else:
                    with pytest.raises(ContractError):
                        validate_owned_collection(plan, ledger, max_owned_bytes=1024**3, deadline=deadline)
            finally:
                state.clear()
                state.update(saved_state)
                online.ownership, child.steps, child.identity, child.lease._cleanup_owner = previous
        finally:
            await child.close(deadline)
        assert ledger.learner is None

    directory = tmp_path / 'parent'
    with run_lock(directory, create=True) as ownership:
        asyncio.run(run(directory, ownership))
