"""Candidate-wheel Transformer integration with real coordinator and learner IPC."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace

import pytest

from gf_cli.config import LoadedConfig, load_config
from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_training.fixed_data import build_dataset
from gf_training.lineage import read_json, validate_chain
from gf_training.ppo import ingest_collection
from gf_training.ppo_run import train_ppo
from gf_training.sequence_rollout import write_rollout
from tests.test_gru_ppo import synthetic_sequence_collector
from tests.test_native_ppo import online_context


def test_native_transformer_banks_match_reference_and_source_finalizes_before_two_updates(tmp_path, make_input, monkeypatch):
    extension = pytest.importorskip('gf_native')
    if 'transformer' not in dict(extension.build_info()).get('ppo_policy_models', '').split(','):
        pytest.skip('requires the Transformer-capable candidate native wheel')
    pytest.importorskip('torch')
    import gf_training.native_ppo as native
    import gf_training.ppo_run as coordinator
    loaded = load_config('configs/experiments/transformer-ppo-cpu.toml')
    config = replace(loaded.config, training=replace(loaded.config.training, max_updates=2, sequence_length=2))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    collector = synthetic_sequence_collector(make_input, 'transformer')
    references, proofs = {}, {}
    active = set()
    original_owner = native.NativePreparation
    class Owner(original_owner):
        @classmethod
        async def create_async(cls, *args, **options):
            owner = await super().create_async(*args, **options)
            # The real native boundary publishes only the required compact
            # identity, not the full experiment configuration.
            assert owner.summary['parents'][0]['state']['model_id'] == 'transformer'
            assert 'effective_config' not in owner.summary['parents'][0]['state']
            assert owner.identity['build']['ppo_policy_models'] == 'mlp,gru,transformer'
            active.add(id(owner))
            return owner
        async def afinalize(self):
            proof = await super().afinalize()
            active.remove(id(self))
            return proof
        async def aabort(self):
            try:
                await super().aabort()
            finally:
                active.discard(id(self))
    monkeypatch.setattr(native, 'NativePreparation', Owner)
    original_request = coordinator.LearningProcess.request
    async def request(self, message, deadline):
        if message['command'] == 'update':
            assert not active
        return await original_request(self, message, deadline)
    monkeypatch.setattr(coordinator.LearningProcess, 'request', request)
    async def collected(plan):
        result = await collector(plan)
        rollout, proof = ingest_collection(plan.output, loaded.config.model, plan.linkage,
                                           deadline=plan.execution_deadline, online_context=online_context(plan))
        batch = plan.linkage['batch_id']
        references[batch] = write_rollout(tmp_path / 'reference' / batch, rollout, loaded.config.model)
        proofs[batch] = proof
        return result
    directory = tmp_path / 'parent'
    result = asyncio.run(train_ppo(loaded, collected, directory, pause_after_update=1))
    assert result['status'] == 'paused' and not active
    result = asyncio.run(train_ppo(loaded, collected, directory, mode='resume'))
    assert result['status'] == 'complete' and result['updates'] == 2 and not active
    state, _ = validate_chain(directory)
    assert state['identity']['preprocessing']['native']['build']['ppo_policy_models'] == 'mlp,gru,transformer'
    for batch, reference in references.items():
        assert read_json(directory / 'numeric' / batch / 'rollout.json') == reference
        number = int(batch[-6:]) + 1
        assert read_json(directory / 'updates' / f'{number:06}.json')['source_proof'] == proofs[batch]
    # Both production source backends keep the old fixed-data MLP-only gate,
    # even though this Transformer parent has a healthy complete generation chain.
    for backend in ('python', 'rust'):
        with pytest.raises(ContractError, match='requires MLP PPO parents'):
            build_dataset((directory, tmp_path / 'z-second', tmp_path / 'z-third'),
                          (tmp_path / 'v1', tmp_path / 'v2', tmp_path / 'v3'), tmp_path / f'fixed-{backend}.json',
                          preprocessing=PreparationConfig(backend=backend))
