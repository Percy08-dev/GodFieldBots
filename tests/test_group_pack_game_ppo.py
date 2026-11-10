"""Controlled original fixtures with a real wire5 process; no executed-game claim."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli.config import GroupPackExperimentConfig, GroupPackStorageConfig, LoadedConfig
from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_models.arrays import sha256
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_training.card_instance_ppo import _GameOperations, train_card_instance_ppo
from gf_training.card_instance_source import validate_game_parent
from gf_training.collection import PackedFinalizedCollection
from gf_training.group_pack_source import CollectionLocator
from gf_training.structured_numeric import GameLearner
from tests.card_instance_learning_support import original_collection
from tests.structured_ppo_support import configuration
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_learning import ACTIONS, RULES, learner_for
from tests.test_group_pack_game_source import pack_original
from tests.test_structured_numeric_worker import assert_artifacts_equal


def setup(tmp_path, monkeypatch):
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    base = configuration(tmp_path)
    learner = learner_for()
    model = ConfiguredNeuralModelV3(**{key: getattr(learner.model, key) for key in learner.model.__dataclass_fields__}, inference=learner.binding)
    config = GroupPackExperimentConfig(**{key: getattr(base.config, key) for key in base.config.__dataclass_fields__},
        storage=GroupPackStorageConfig(db_root=str(tmp_path / 'database'), ram_root=str(tmp_path / 'ram')))
    config = replace(config, actions=ACTIONS, model=model,
        preprocessing=PreparationConfig(reader='participant-structured-v2', memory_budget_mib=1024))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    monkeypatch.setattr(_GameOperations, 'preflight', lambda self, config, deadline: None)
    def integration(self, config, deadline):
        return {'binary_sha256': receipt()['binary']['sha256'], 'server_config_sha256': sha256(Path(config.server.config)),
            'declared_simulator_revision': ACTIONS.action_contract.simulator_revision,
            'source_revision': ACTIONS.action_contract.simulator_revision,
            'source_spec_sha256': dict(CORRECTED_REVIEWED_SPEC_HASHES)}
    monkeypatch.setattr(_GameOperations, 'integration_identity', integration)
    seen = []
    async def collector(plan, ledger):
        child = ledger.learner._cleanup_owner
        assert child.protocol == 5
        seen.append(child)
        state = json.loads((child.directory / 'run.json').read_bytes())
        generation = plan.linkage['collection_generation']
        frozen = GameLearner(learner.model, 17, learner.binding, state['resume_identity'], actions=ACTIONS, rules=RULES)
        frozen.restore(child.directory / 'artifacts' / f'{generation:06}')
        original_collection(plan, state, frozen)
        physical = Path(config.storage.db_root) / state['run_id'] / 'collections' / plan.linkage['batch_id']
        pack_original(plan.output, physical)
        pointer = json.loads((plan.output / 'collection.locator.json').read_bytes())
        return PackedFinalizedCollection(plan.output, {'comparison_valid': True, 'healthy': True},
            CollectionLocator('sqlite-group-pack-v1', Path(pointer['descriptor'])),
            sha256(plan.output / 'collection.locator.json'), {'publication_scope': 'published'})
    return loaded, collector, seen


def test_ordinary_packed_coordinator_real_wire5_two_updates_pause_resume(tmp_path, monkeypatch):
    loaded, collector, seen = setup(tmp_path, monkeypatch)
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    first = asyncio.run(train_card_instance_ppo(loaded, collector, continuous))
    paused = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, pause_after_update=1))
    assert first['updates'] == 2 and paused['updates'] == 1 and paused['status'] == 'paused'
    completed = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, mode='resume'))
    assert completed['updates'] == 2 and completed['status'] == 'complete'
    assert_artifacts_equal(continuous / 'artifacts/000002', resumed / 'artifacts/000002')
    for directory in (continuous, resumed):
        state, chain = validate_game_parent(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 90)
        assert state['schema_version'] == 4 and state['accepted_decisions'] == 32 and len(chain) == 3
        for number in (1, 2):
            update = json.loads((directory / 'updates' / f'{number:06}.json').read_bytes())
            assert update['schema_version'] == 4 and update['source_proof']['schema_version'] == 2
            assert update['storage_binding'] == state['storage_binding']
    assert all(child.online_owner.packed_batch is None for child in seen)
    assert all(child.ledger.learner is None for child in seen)


def test_pause_reopen_identical_original_replacement_refuses_resume(tmp_path, monkeypatch):
    loaded, collector, _ = setup(tmp_path, monkeypatch)
    parent = tmp_path / 'parent'
    asyncio.run(train_card_instance_ppo(loaded, collector, parent, pause_after_update=1))
    pointer = json.loads((parent / 'collections/batch-000000/collection.locator.json').read_bytes())
    database = Path(pointer['descriptor']).parent / 'collection.sqlite3'
    replacement = database.with_suffix('.replacement')
    replacement.write_bytes(database.read_bytes())
    replacement.replace(database)
    with pytest.raises(ContractError, match='proof|custody'):
        asyncio.run(train_card_instance_ppo(loaded, collector, parent, mode='resume'))
    assert not (parent / 'updates/000002.json').exists()


def test_cancel_before_optimizer_retires_same_packed_batch(tmp_path, monkeypatch):
    from gf_training.structured_learning_process import GameLearningProcess
    loaded, collector, seen = setup(tmp_path, monkeypatch)
    held = []
    async def cancel(self, projection, output, *, deadline):
        custody = self.online_owner.packed_batch
        assert custody is not None and custody.owner is projection.source.snapshot._owner
        held.append(custody)
        raise asyncio.CancelledError('controlled preoptimizer cancellation')
    monkeypatch.setattr(GameLearningProcess, 'update', cancel)
    parent = tmp_path / 'cancelled'
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(train_card_instance_ppo(loaded, collector, parent))
    assert held and held[0].retired and held[0].owner._closed
    assert held[0].owner.locator.fd == -1 and held[0].owner.pool.closed
    assert not (parent / 'updates/000001.json').exists()
    assert all(child.online_owner.packed_batch is None for child in seen)
    assert all(child.ledger.learner is None for child in seen)


@pytest.mark.parametrize('kind', ['cancel', 'timeout'])
def test_expired_work_deadline_before_optimizer_retires_actual_online_custody(tmp_path, monkeypatch, kind):

    from gf_training.structured_learning_process import GameLearningProcess
    from tests.test_group_pack_game_failures import assert_retired, expire
    loaded, collector, seen = setup(tmp_path, monkeypatch)
    held = []
    failure = asyncio.CancelledError('own online operation cancelled') if kind == 'cancel' else TimeoutError('own online operation expired')
    async def expire_then_fail(self, projection, output, *, deadline):
        custody = self.online_owner.packed_batch
        assert custody.owner is projection.source.snapshot._owner
        connection = custody.owner.artifacts._originals.db
        held.append((custody, connection))
        expire(custody.owner)
        for scan in custody.scans:
            scan.deadline = 0.
        raise failure
    monkeypatch.setattr(GameLearningProcess, 'update', expire_then_fail)
    parent = tmp_path / 'expired-online'
    with pytest.raises(type(failure)) as captured:
        asyncio.run(train_card_instance_ppo(loaded, collector, parent))
    assert captured.value is failure
    custody, connection = held[0]
    assert_retired(custody.owner, connection)
    assert custody.retired and custody.owner.pool.closed and custody.owner.pool.budget.used == 0
    assert all(child.online_owner.packed_batch is None and child.ledger.learner is None for child in seen)
    assert not (parent / 'updates/000001.json').exists()
