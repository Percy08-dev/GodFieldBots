"""Synthetic PPO coordinator ownership, native ingestion and update ordering."""
import asyncio
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import LoadedConfig
from gf_common.domain import ContractError, digest
from gf_training.collection import OnlinePPOContext
from gf_training.lineage import _OWNED_LOCKS, checkpoint_path, read_json, validate_chain
from gf_training.ppo import ingest_collection, write_rollout
from gf_training.ppo_run import train_ppo
from tests.test_ppo_lineage import configuration, synthetic_collector


def native_configuration(tmp_path, max_updates=2):
    loaded = configuration(tmp_path, max_updates=max_updates)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing,
                     backend='rust', memory_budget_mib=128, cache_budget_mib=16, chunk_target_mib=1))
    return LoadedConfig(config, asdict(config), digest(asdict(config)), ())


def online_context(plan):
    parent = Path(plan.linkage['parent_run_path']).resolve()
    return OnlinePPOContext(parent, plan.linkage['parent_run_id'], plan.linkage['batch_id'],
                            plan.linkage['collection_generation'], _OWNED_LOCKS[parent])


def test_native_preflight_missing_package_precedes_parent_child_and_collection(tmp_path, monkeypatch):
    import gf_training.native_ppo as native
    import gf_training.ppo_run as coordinator
    loaded = native_configuration(tmp_path)
    def missing():
        raise ContractError('native preprocessing requires the optional native package')
    monkeypatch.setattr(native, 'load_native', missing)
    monkeypatch.setattr(coordinator.LearningProcess, 'start', lambda self: pytest.fail('no learner before native preflight'))
    async def forbidden(plan):
        pytest.fail('no collection before native preflight')
    path = tmp_path / 'missing-native'
    with pytest.raises(ContractError, match='optional native'):
        asyncio.run(train_ppo(loaded, forbidden, path))
    assert not path.exists()


def test_native_preflight_missing_session_api_precedes_collection(tmp_path, monkeypatch):
    import gf_training.native_ppo as native
    loaded = native_configuration(tmp_path)
    monkeypatch.setattr(native, 'load_native', lambda: SimpleNamespace(NativeSession=object))
    monkeypatch.setattr(native, 'binary_identity', lambda *args: pytest.fail('incomplete API cannot attest preprocessing'))
    async def forbidden(plan):
        pytest.fail('no collection with incomplete native implementation')
    path = tmp_path / 'incomplete-native'
    with pytest.raises(ContractError, match='incomplete'):
        asyncio.run(train_ppo(loaded, forbidden, path))
    assert not path.exists()


def test_native_pause_resume_exact_rollout_identity_and_source_close_before_update(tmp_path, make_input, monkeypatch):
    pytest.importorskip('gf_native')
    pytest.importorskip('torch')
    import gf_training.native_ppo as native
    import gf_training.ppo_run as coordinator
    loaded = native_configuration(tmp_path)
    base_collector = synthetic_collector(make_input, both_viewers=True)
    reference_manifests, reference_proofs = {}, {}
    active = set()
    events = []
    original_owner = native.NativePreparation
    class WatchedOwner(original_owner):
        @classmethod
        async def create_async(cls, *args, **kwargs):
            self = await super().create_async(*args, **kwargs)
            active.add(id(self))
            events.append('source-open')
            return self
        async def afinalize(self):
            result = await super().afinalize()
            active.remove(id(self))
            events.append('source-finalized')
            return result
        async def aabort(self):
            try:
                return await super().aabort()
            finally:
                active.discard(id(self))
                events.append('source-aborted')
    monkeypatch.setattr(native, 'NativePreparation', WatchedOwner)
    request = coordinator.LearningProcess.request
    async def guarded_request(self, message, deadline):
        if message['command'] == 'update':
            assert not active
            assert events[-1] == 'source-finalized'
            events.append('optimizer-request')
        return await request(self, message, deadline)
    monkeypatch.setattr(coordinator.LearningProcess, 'request', guarded_request)
    commit = coordinator.commit_generation
    def guarded_commit(*args, **kwargs):
        assert not active
        return commit(*args, **kwargs)
    monkeypatch.setattr(coordinator, 'commit_generation', guarded_commit)
    async def collected(plan):
        result = await base_collector(plan)
        decisions, proof = ingest_collection(plan.output, loaded.config.model, plan.linkage,
                                             deadline=plan.execution_deadline, online_context=online_context(plan))
        reference = tmp_path / 'reference-rollouts' / plan.linkage['batch_id']
        reference_manifests[plan.linkage['batch_id']] = write_rollout(reference, decisions)
        reference_proofs[plan.linkage['batch_id']] = proof
        return result
    directory = tmp_path / 'native-parent'
    result = asyncio.run(train_ppo(loaded, collected, directory, pause_after_update=1))
    assert result['status'] == 'paused' and result['updates'] == 1
    state = read_json(directory / 'run.json')
    assert state['identity']['preprocessing']['backend'] == 'rust'
    assert state['identity']['preprocessing']['native']['extension_sha256']
    assert state['identity']['preprocessing']['native']['build']['recipe_identity']
    assert not active
    original_identity = coordinator.preprocessing_identity
    def changed_binary(config, *, deadline):
        value = copy.deepcopy(original_identity(config, deadline=deadline))
        value['native']['extension_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(coordinator, 'preprocessing_identity', changed_binary)
    async def forbidden(plan):
        pytest.fail('resume identity drift must reject before collection')
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(loaded, forbidden, directory, mode='resume'))
    monkeypatch.setattr(coordinator, 'preprocessing_identity', original_identity)
    python_config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='python'))
    changed = LoadedConfig(python_config, asdict(python_config), digest(asdict(python_config)), ())
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(changed, forbidden, directory, mode='resume'))
    assert read_json(directory / 'run.json')['status'] == 'paused'
    result = asyncio.run(train_ppo(loaded, collected, directory, mode='resume'))
    assert result['status'] == 'complete' and result['updates'] == 2
    validate_chain(directory)
    assert events.count('source-finalized') == events.count('optimizer-request') == 2
    for batch, manifest in reference_manifests.items():
        assert read_json(directory / 'numeric' / batch / 'rollout.json') == manifest
        update = read_json(directory / 'updates' / f'{int(batch[-6:]) + 1:06}.json')
        assert update['source_proof'] == reference_proofs[batch]
    assert read_json(directory / 'run.json')['invocations'][-1]['child_health']['exit_code'] == 0


def test_native_online_wrong_owner_batch_and_generation_rejected(tmp_path, make_input):
    pytest.importorskip('gf_native')
    pytest.importorskip('torch')
    import gf_training.native_ppo as native
    loaded = native_configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input, both_viewers=True)
    checked = []
    async def guarded(plan):
        result = await collector(plan)
        context = online_context(plan)
        identity = read_json(context.parent_directory / 'run.json')['identity']['preprocessing']
        for changed in (replace(context, ownership_token='forged'),
                        replace(context, batch_id='wrong-batch'),
                        replace(context, generation=context.generation + 1),
                        replace(context, parent_run_id='foreign-parent')):
            with pytest.raises(ContractError, match='context'):
                await native.stage_rollout(plan.output, tmp_path / 'forbidden-numeric', loaded.config.model,
                                     plan.linkage, loaded.config.preprocessing, deadline=plan.execution_deadline,
                                     online_context=changed, expected_identity=identity)
            checked.append(changed)
        # Matching forged linkage/context cannot override the actual parent's generation.
        forged = replace(context, generation=context.generation + 1)
        link = {**plan.linkage, 'collection_generation': forged.generation}
        with pytest.raises(ContractError, match='parent|ownership|generation'):
            await native.stage_rollout(plan.output, tmp_path / 'forbidden-numeric', loaded.config.model,
                                 link, loaded.config.preprocessing, deadline=plan.execution_deadline,
                                 online_context=forged, expected_identity=identity)
        assert not (tmp_path / 'forbidden-numeric').exists()
        return result
    result = asyncio.run(train_ppo(loaded, guarded, tmp_path / 'owned-parent'))
    assert result['status'] == 'complete'
    assert len(checked) == 4


@pytest.mark.parametrize('failure', ['reward', 'finalization'])
def test_native_invalid_batch_or_late_cleanup_never_requests_update(tmp_path, make_input, monkeypatch, failure):
    pytest.importorskip('gf_native')
    pytest.importorskip('torch')
    import gf_training.native_ppo as native
    import gf_training.ppo_run as coordinator
    loaded = native_configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input, both_viewers=True)
    requests = []
    original_request = coordinator.LearningProcess.request
    async def tracked(self, request, deadline):
        requests.append(request['command'])
        return await original_request(self, request, deadline)
    monkeypatch.setattr(coordinator.LearningProcess, 'request', tracked)
    if failure == 'finalization':
        finalize = native.NativePreparation.afinalize
        async def late_failure(self):
            await finalize(self)
            raise ContractError('synthetic native finalization failure')
        monkeypatch.setattr(native.NativePreparation, 'afinalize', late_failure)
    async def broken(plan):
        result = await collector(plan)
        if failure == 'reward':
            source = next(plan.output.glob('*/player-2.jsonl'))
            records = [json.loads(line) for line in source.read_text().splitlines()]
            records[-1]['terminal_reward'] = 1 if records[-1]['terminal_reward'] != 1 else -1
            source.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
        return result
    directory = tmp_path / f'failed-{failure}'
    with pytest.raises(ContractError, match='reward|finalization'):
        asyncio.run(train_ppo(loaded, broken, directory))
    assert 'initialize' in requests and 'close' in requests and 'update' not in requests
    state = read_json(directory / 'run.json')
    assert state['status'] == 'failed' and state['updates'] == 0
    assert state['invocations'][-1]['child_health']['exit_code'] == 0
    assert not checkpoint_path(directory, 1).exists()


def test_async_native_cancellation_joins_before_parent_unlock_and_never_updates(tmp_path, make_input, monkeypatch):
    pytest.importorskip('gf_native')
    pytest.importorskip('torch')
    import threading

    import gf_training.native_ppo as native
    import gf_training.ppo_run as coordinator
    loaded = native_configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input, both_viewers=True)
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    requests = []
    original_owner = native.NativePreparation
    class DelayedOwner(original_owner):
        async def anext_batch(self, retained, layout):
            self.check()
            def blocked_native_call():
                entered.set()
                release.wait(10)
                try:
                    return self._session.next_batch(retained, layout)
                finally:
                    ended.set()
            return self._batch(await self._acall(blocked_native_call))
    monkeypatch.setattr(native, 'NativePreparation', DelayedOwner)
    original_request = coordinator.LearningProcess.request
    async def tracked(self, request, deadline):
        requests.append(request['command'])
        return await original_request(self, request, deadline)
    monkeypatch.setattr(coordinator.LearningProcess, 'request', tracked)
    directory = tmp_path / 'cancelled-native'
    async def run():
        async with asyncio.timeout(20):
            task = asyncio.create_task(train_ppo(loaded, collector, directory))
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel('original PPO cancellation')
            for _ in range(4):
                await asyncio.sleep(0)
                task.cancel('repeated PPO cancellation')
            assert not task.done()
            assert directory.resolve() in _OWNED_LOCKS
            release.set()
            with pytest.raises(asyncio.CancelledError, match='original PPO cancellation'):
                await task
        assert ended.is_set()
        assert directory.resolve() not in _OWNED_LOCKS
        assert not [thread for thread in threading.enumerate() if thread.name.startswith('gf-native-ffi')]
    asyncio.run(run())
    assert 'initialize' in requests and 'close' in requests and 'update' not in requests
    state = read_json(directory / 'run.json')
    assert state['status'] == 'failed' and state['updates'] == 0
    assert state['invocations'][-1]['child_health']['exit_code'] == 0
