"""Private synthetic collector checks; no executed games or qualified Source claim."""
import asyncio
import time
from dataclasses import asdict, replace
from pathlib import Path

from gf_cli.config import LoadedConfig
from gf_common.domain import digest
from gf_common.preprocessing import PreparationConfig
from gf_models.structured_binding import ConfiguredNeuralModelV3
from gf_training.card_instance_ppo import _GameOperations, train_card_instance_ppo
from gf_training.card_instance_source import validate_game_parent
from gf_training.collection import FinalizedCollection
from gf_training.structured_numeric import GameLearner
from tests.card_instance_learning_support import original_collection
from tests.structured_ppo_support import configuration
from tests.test_card_instance_baseline_runtime import receipt
from tests.test_card_instance_learning import ACTIONS, RULES, learner_for
from tests.test_structured_numeric_worker import assert_artifacts_equal


def test_single_coordinator_game_continuous_two_vs_pause_resume(tmp_path, monkeypatch):
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    from gf_models.arrays import sha256
    base = configuration(tmp_path)
    learner = learner_for()
    model = ConfiguredNeuralModelV3(**{key: getattr(learner.model, key) for key in learner.model.__dataclass_fields__}, inference=learner.binding)
    config = replace(base.config, actions=ACTIONS, model=model,
        preprocessing=PreparationConfig(reader='participant-structured-v2', memory_budget_mib=1024))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    # Only external runtime I/O is doubled. Real numerical child, guards, Source,
    # numeric correspondence, chain publication and ordinary resume remain active.
    monkeypatch.setattr(_GameOperations, 'preflight', lambda self, config, deadline: None)
    def integration(self, config, deadline):
        return {'binary_sha256': receipt()['binary']['sha256'], 'server_config_sha256': sha256(Path(config.server.config)),
            'declared_simulator_revision': ACTIONS.action_contract.simulator_revision,
            'source_revision': ACTIONS.action_contract.simulator_revision,
            'source_spec_sha256': dict(CORRECTED_REVIEWED_SPEC_HASHES)}
    monkeypatch.setattr(_GameOperations, 'integration_identity', integration)
    async def collector(plan, ledger):
        child = ledger.learner._cleanup_owner
        from gf_models.sharded_arrays import read_json
        state = read_json(child.directory / 'run.json')
        generation = plan.linkage['collection_generation']
        frozen = GameLearner(learner.model, 17, learner.binding, state['resume_identity'], actions=ACTIONS, rules=RULES)
        frozen.restore(child.directory / 'artifacts' / f'{generation:06}')
        original_collection(plan, state, frozen)
        return FinalizedCollection(plan.output, {'comparison_valid': True})
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    first = asyncio.run(train_card_instance_ppo(loaded, collector, continuous))
    paused = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, pause_after_update=1))
    assert first['updates'] == 2 and paused['updates'] == 1 and paused['status'] == 'paused'
    completed = asyncio.run(train_card_instance_ppo(loaded, collector, resumed, mode='resume'))
    assert completed['updates'] == 2 and completed['status'] == 'complete'
    assert_artifacts_equal(continuous / 'artifacts/000002', resumed / 'artifacts/000002')
    for directory in (continuous, resumed):
        state, chain = validate_game_parent(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
        assert state['schema_version'] == 3 and state['accepted_decisions'] == 32 and len(chain) == 3
        assert (directory / 'artifacts/000000/resume.json').is_file()
        assert not (directory / 'artifacts/000001/resume.json').exists()
