"""CPU component evidence only: fake games, actual dataclasses/Pack/publication.

No Simulator, CUDA inference or Source qualification is exercised here.
"""
import asyncio
import json
import time
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, GroupResult, clone, digest
from gf_simulator_client.card_instance_environment import (
    CardInstanceEpisodeResult,
    CardInstanceSimulatorEnvironment,
)
from gf_training.group_pack_source import open_artifact_source
from tests.test_hpo_eval_collector import scenario


def actual_pack_scenario(tmp_path, monkeypatch, *, cpu_processes=False):
    from gf_cli.config import LoadedConfig
    loaded, plan, job, owner, _configured, _started = scenario(tmp_path, monkeypatch)
    config = replace(loaded.config, storage=replace(loaded.config.storage,
        db_root=str(tmp_path / 'db'), ram_root=str(tmp_path / 'ram')))
    raw = clone(asdict(config))
    loaded = LoadedConfig(config, raw, digest(raw), loaded.inputs)
    install_component_execution(monkeypatch, cpu_processes=cpu_processes)
    return loaded, plan, job, owner


def install_component_execution(monkeypatch, *, cpu_processes=True):
    from itertools import count

    import gf_cli.card_instance as core
    from tests.test_card_instance_baseline_runtime import receipt
    executions = count()
    class Runtime:
        def __init__(self, binary, source, **options):
            self.binary, self.source_root = binary, source
        def preflight(self): return {}
    async def fake_games(self, group):
        self._storage_group_owner = self
        self._managed_server = None
        if cpu_processes:
            import sys

            from gf_simulator_client.process import ManagedServer
            from gf_simulator_client.structured_worker import StructuredParticipantWorker
            worker = object.__new__(StructuredParticipantWorker)
            worker._spawn(self.ledger, 16, [sys.executable, '-I', '-c', 'pass'], {})
            lease, process = worker.lease, worker.process
            process.wait(timeout=5)
            lease.reaped(process)
            self._component_worker = worker
            server = ManagedServer(sys.executable, self.settings.server_config, group.seed, len(group.episodes),
                                   self.settings.output_dir / 'component-server', shutdown_seconds=5,
                                   cleanup_deadline=time.monotonic() + 10)
            server.process = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c',
                "print('CPU component only')", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            server.drainers = [asyncio.create_task(server.process.stdout.read())]
            self._managed_server = server
            await server.stop()
        self.artifact_sink.write(group.episodes[0].episode_id + '/component.txt', b'CPU unqualified fixture')
        episodes = [CardInstanceEpisodeResult(spec.episode_id, spec.pair_id, group.seed,
            spec.policies, game_id='component-' + spec.episode_id, outcome='draw',
            end_kind='natural', replay_reconciled=True, action_contract=self.settings.actions.action_contract)
            for spec in group.episodes]
        runtime = receipt()
        index = next(executions)
        runtime['executed'].update(pid=123 + index, start_time=1 + index)
        return GroupResult(episodes, {'runtime_evidence': runtime,
            'process': {'exit_code': 0, 'shutdown': 'stdin_eof', 'log_error': None}})
    monkeypatch.setattr(core, 'CardInstanceRuntimeEvidence', Runtime)
    monkeypatch.setattr(CardInstanceSimulatorEnvironment, 'run_group', fake_games)
    monkeypatch.setattr(core, '_validate_recorded_episodes', lambda *args, **kwargs: None)


def test_actual_eval_producer_real_pack_publication_and_locator_roundtrip(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    from gf_cli.hpo_evaluation import _HpoEvaluationCleanupObservation
    from gf_common.card_instance_collection import checked_hpo_evaluation_manifest
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch, cpu_processes=True)
    try:
        result = asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
            _cleanup_observation=_HpoEvaluationCleanupObservation(plan, job, owner)))
        source = open_artifact_source(result.locator, deadline=time.monotonic() + 30, max_owned_bytes=512 * 1024**2)
        try:
            manifest = json.loads(source.read('manifest.json'))
            checked_hpo_evaluation_manifest(manifest)
            assert len(manifest['episodes']) == 4
            assert all(row['outcome'] == 'draw' and row['end_kind'] == 'natural' for row in manifest['episodes'])
            assert all(type(row['policies']) is list for row in manifest['episodes'])
        finally:
            source.close()
    finally:
        owner.release()
        job.confirm_termination()
        job.close()


def test_after_prepare_failure_retires_same_publication_and_keeps_primary(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = ValueError('after actual prepare')
    prepared = []
    actual = GroupPackCollectionPublication.prepare
    async def fail(self, *args, **kwargs):
        await actual(self, *args, **kwargs)
        prepared.append(self)
        raise primary
    monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', fail)
    try:
        with pytest.raises(ValueError) as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job,
                artifact_inputs=owner, _cleanup_observation=observation))
        assert caught.value is primary
        assert prepared[0].cleanup_complete
        assert prepared[0].collection.budget.used == 0
        observation.require_core_retired(error=primary)
    finally:
        owner.release()
        job.confirm_termination()
        job.close()


def test_failed_operation_receipt_requires_real_retired_core_and_real_pin(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_cli.hpo_storage import CampaignStore
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_lineage_support import advanced_hpo_chain
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    parent = tmp_path / 'parent'
    _, records, _ = advanced_hpo_chain(parent, updates=1, budget=3)
    acquired = pin_hpo_evaluation(parent, job.ticket, 1, records[1]['inference_sha256'],
        max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    actual = GroupPackCollectionPublication.prepare
    primary = ValueError('data publication failure')
    async def fail(self, *args, **kwargs):
        await actual(self, *args, **kwargs)
        raise primary
    monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', fail)
    try:
        with pytest.raises(ValueError) as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
        observation.release_artifacts()
        released = release_hpo_evaluation(parent, acquired.binding, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
        with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=8 * 1024**2) as store:
            evaluation._confirm_failed_evaluation(caught.value, store, 'failed-termination.json', observation,
                job, [(parent, acquired)], {id(acquired): released})
            record, _ = store.get('failed-termination.json')
            assert record['kind'] == 'hpo-failed-evaluation-termination'
            assert record['gpu_termination']['ticket'] == job.ticket
            assert not store.exists('evaluation.json')
            assert not store.exists('objective.json')
            assert job._receipt is not None
    finally:
        owner.release()
        job.close()


@pytest.fixture
def retired_failure(tmp_path, monkeypatch, request):
    """Actual CPU worker proofs, Pack and pin transactions; no game authority."""
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_lineage_support import advanced_hpo_chain
    stage = getattr(request, 'param', 'prepare')
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch, cpu_processes=True)
    parent = tmp_path / 'parent'
    _, records, _ = advanced_hpo_chain(parent, updates=1, budget=3)
    acquired = pin_hpo_evaluation(parent, job.ticket, 1, records[1]['inference_sha256'],
        max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = ValueError('original ' + stage + ' failure')
    if stage == 'prepare':
        actual = GroupPackCollectionPublication.prepare
        async def fail(self, *args, **kwargs):
            await actual(self, *args, **kwargs)
            raise primary
        monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', fail)
    elif stage == 'commit':
        def fail(*args, **kwargs): raise primary
        monkeypatch.setattr(GroupPackCollectionPublication, '_link', fail)
    try:
        if stage in ('prepare', 'commit'):
            with pytest.raises(ValueError) as caught:
                asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                    _cleanup_observation=observation))
            assert caught.value is primary
        else:
            result = asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
            from gf_cli.hpo_config import campaign_config
            from gf_training import structured_source
            from tests.test_hpo_config import campaign_raw
            campaign = campaign_config(campaign_raw(), origin=tmp_path)
            campaign = replace(campaign, config=replace(campaign.config,
                storage=replace(campaign.config.storage, originals_root=str(tmp_path / 'db'))))
            def fail(*args, **kwargs): raise primary
            monkeypatch.setattr(structured_source, '_inspect_component_collection', fail)
            if stage == 'source-close':
                from gf_training.group_pack_game_source import PackedSourceOwner
                original_close = PackedSourceOwner.close
                def close(self):
                    original_close(self)
                    raise OSError('uncertain Source close')
                monkeypatch.setattr(PackedSourceOwner, 'close', close)
            with pytest.raises(ValueError) as caught:
                evaluation.verify_saved_evaluation(result.directory, campaign, plan.linkage,
                    deadline=time.monotonic() + 30, extra_owned_bytes=owner._scan.peak,
                    _cleanup_observation=observation)
            assert caught.value is primary
            assert observation._publication.committed  # Outer errors never revoke committed data.
        observation.release_artifacts()
        released = release_hpo_evaluation(parent, acquired.binding, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
        yield evaluation, observation, job, [(parent, acquired)], {id(acquired): released}, primary
    finally:
        owner.release()
        job.close()


def confirm_failure(tmp_path, values, monkeypatch=None, *, copy_error=False):
    from gf_cli.hpo_storage import CampaignStore
    evaluation, observation, job, pins, releases, primary = values
    with CampaignStore(tmp_path / 'failure-store', mode='fresh', max_owned_bytes=8 * 1024**2) as store:
        if copy_error:
            def fail(*args, **kwargs): raise OSError('local record failed')
            monkeypatch.setattr(store, 'put', fail)
        evaluation._confirm_failed_evaluation(primary, store, 'failed.json', observation, job, pins, releases)
        assert not store.exists('evaluation.json') and not store.exists('objective.json')
        return store.exists('failed.json')


@pytest.mark.parametrize('retired_failure', ['prepare', 'commit', 'verify'], indirect=True)
def test_real_cpu_resources_terminate_on_failed_data_only(tmp_path, retired_failure):
    _, observation, job, _, _, primary = retired_failure
    assert confirm_failure(tmp_path, retired_failure)
    assert job._receipt.resources and all(resource.outcome == 'reaped' for resource in job._receipt.resources)
    assert primary.hpo_gpu_termination_receipt is job._receipt
    assert observation._pack.budget.used == 0


@pytest.mark.parametrize('retired_failure', ['source-close'], indirect=True)
def test_source_close_error_preserves_primary_and_withholds_termination(tmp_path, retired_failure):
    _, observation, job, _, _, primary = retired_failure
    assert observation._verifier is primary.hpo_verifier_cleanup_owner
    assert not confirm_failure(tmp_path, retired_failure)
    assert job._receipt is None
    assert any('uncertain Source close' in note for note in primary.__notes__)


@pytest.mark.parametrize('uncertainty', ['worker', 'server', 'drainer', 'future', 'publication', 'artifact', 'pin', 'foreign', 'error'])
def test_actual_owner_uncertainty_never_mints_termination(tmp_path, retired_failure, uncertainty):
    import concurrent.futures

    from gf_simulator_client.structured_resources import ReservationLedger
    _, observation, job, pins, releases, primary = retired_failure
    restore = lambda: None
    if uncertainty == 'worker':
        token = job.ledger._history[0]
        proof, token._proof = token._proof, None
        restore = lambda: setattr(token, '_proof', proof)
    elif uncertainty in ('server', 'drainer'):
        runtime, server, _process, drainers = next(iter(observation._runtime_originals.values()))
        if uncertainty == 'server':
            runtime._managed_server = None
            restore = lambda: setattr(runtime, '_managed_server', server)
        else:
            server.drainers = []
            restore = lambda: setattr(server, 'drainers', list(drainers))
    elif uncertainty == 'future':
        future = concurrent.futures.Future()
        observation._pack._futures.add(future)
        restore = lambda: observation._pack._futures.discard(future)
    elif uncertainty == 'publication':
        reservation = observation._publication._reservation
        reservation.closed = False
        restore = lambda: setattr(reservation, 'closed', True)
    elif uncertainty == 'artifact':
        observation._identity[3]._released = False
        restore = lambda: setattr(observation._identity[3], '_released', True)
    elif uncertainty == 'pin':
        releases[id(pins[0][1])] = OSError('pin release uncertain')
    elif uncertainty == 'foreign':
        original, job._ledger = job._ledger, ReservationLedger()
        restore = lambda: setattr(job, '_ledger', original)
    else:
        observation._core_error = ValueError('different original error')
    try:
        assert not confirm_failure(tmp_path, retired_failure)
        assert job._receipt is None
        assert not (job._lock.root / 'jobs' / job.ticket / 'termination.json').exists()
        assert primary.hpo_termination_cleanup_error is not None
    finally:
        restore()


def test_failed_termination_global_write_error_poisoned_and_primary_preserved(tmp_path, monkeypatch, retired_failure):
    from gf_training import hpo_gpu_lease as gpu
    _, _, job, _, _, primary = retired_failure
    actual = gpu._write
    def fail(path, *args, **kwargs):
        if path.name == 'termination.json': raise OSError('global termination fsync failed')
        return actual(path, *args, **kwargs)
    monkeypatch.setattr(gpu, '_write', fail)
    assert not confirm_failure(tmp_path, retired_failure)
    assert job._poisoned and job._receipt is None
    assert not hasattr(primary, 'hpo_gpu_termination_receipt')


def test_failed_termination_local_copy_error_keeps_real_global_receipt(tmp_path, monkeypatch, retired_failure):
    _, _, job, _, _, primary = retired_failure
    assert not confirm_failure(tmp_path, retired_failure, monkeypatch, copy_error=True)
    assert job._receipt is primary.hpo_gpu_termination_receipt
    assert isinstance(primary.hpo_failed_termination_copy_error, OSError)
    assert (job._lock.root / 'jobs' / job.ticket / 'termination.json').exists()


@pytest.fixture
def outer_workflow(tmp_path, monkeypatch, request):
    """Real saved parents/pins/CPU processes; mocked Source inspection is UNQUALIFIED."""
    import gf_cli.hpo_evaluation as evaluation
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_config import campaign_config
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.structured_binding import policy_label_game_v4
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_source import hpo_binding
    from tests.hpo_lineage_support import advanced_hpo_chain
    from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES
    from tests.test_hpo_config import campaign_raw
    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path / 'gpu')
    import uuid

    from tests import hpo_lineage_support
    original_state = hpo_lineage_support.hpo_state_fixture
    def canonical_run(*args, **kwargs):
        values = original_state(*args, **kwargs)
        values[0]['run_id'] = str(uuid.uuid4())
        return values
    monkeypatch.setattr(hpo_lineage_support, 'hpo_state_fixture', canonical_run)
    parents = [tmp_path / 'left', tmp_path / 'right']
    chains = [advanced_hpo_chain(parent, updates=1, budget=3) for parent in parents]
    learner = chains[0][2]
    resident_cpu=getattr(request,'param',False)
    raw = campaign_raw()
    if resident_cpu:
        raw.update(schema_version=3,runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':134217728})
    raw.update(decision_budget=3, learning_seeds=[17], model=asdict(learner.model), training=asdict(learner.training),
        space={'parameters': [], 'coupled': []}, storage={'parent_root': str(tmp_path / 'parents'),
            'numeric_root': str(tmp_path / 'parents'), 'originals_root': str(tmp_path / 'db'),
            'optuna_path': str(tmp_path / 'unused.sqlite3')})
    loaded = campaign_config(raw, origin=tmp_path)
    baseconfig = replace(loaded.base.config, actions=ACTIONS)
    baseraw = clone(asdict(baseconfig))
    base = LoadedConfig(baseconfig, baseraw, digest(baseraw), loaded.base.inputs)
    config = replace(loaded.config, base_config_sha256=base.sha256)
    configraw = clone(asdict(config))
    loaded = replace(loaded, base=base, config=config, effective=configraw, sha256=digest(configraw))
    settings = effective_settings(learner.model, learner.training, {}, layout=LAYOUT)
    proposal = Proposal(0, 0, 0, 'a'*64, settings, '{}', '{}')
    inspections = {}
    bindings = {}
    training = []
    for parent, (state, records, _) in zip(parents, chains, strict=True):
        source = {'run_id': state['run_id'], 'run_path': str(parent), 'training_seed': 17,
            'generation': 1, 'generation_sha256': state['generation_sha256'] if resident_cpu else 'b'*64, 'inference_sha256': records[1]['inference_sha256'],
            'used_decisions': 3}
        inspections[str(parent)] = evaluation.ParentInspection(state, tuple(records), source)
        label, runtime = hpo_binding(parent, state, records[1], 1)
        options = clone(runtime.policy_options)
        if not resident_cpu: options['inference']['device'] = 'cuda:0'
        inference = learner.binding if resident_cpu else replace(learner.binding, device='cuda:0')
        label = policy_label_game_v4(learner.model, inference, records[1]['inference_sha256'],
            action_contract=ACTIONS.action_contract, rules=RULES, action_config_hash=digest(asdict(ACTIONS)),
            cuda_visible_devices=options['cuda_visible_devices'])
        reference = {key: source[key] for key in ('run_id', 'run_path', 'training_seed', 'generation',
            'generation_sha256', 'inference_sha256')}
        reference.update(source_receipt_sha256='c'*64, policy_label=label)
        bindings[state['run_id']] = (reference, RuntimePolicyBinding(runtime.model_id, options))
        training.append({**source, 'source_receipt_sha256': 'c'*64})
    monkeypatch.setattr(evaluation, 'inspect_training_parent', lambda directory, *args, **kwargs: inspections[str(directory)])
    monkeypatch.setattr(evaluation, '_final_policy_reference', lambda inspected, source_sha: bindings[inspected.state['run_id']])
    if resident_cpu:
        install_unqualified_cpu_resident_originals(monkeypatch)
    else:
        install_component_execution(monkeypatch)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=8*1024**2) as store:
        store.put('proposals/000000/seeds/17/training.json', training[0])
        yield evaluation, store, loaded, proposal, training, parents


@pytest.mark.parametrize('comparison', [False, True])
def test_actual_outer_failed_publication_records_resource_only_and_releases_all_pins(tmp_path, monkeypatch, outer_workflow, comparison):
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    evaluation, store, loaded, proposal, training, parents = outer_workflow
    primary = ValueError('outer real publication failure')
    original = GroupPackCollectionPublication.prepare
    async def fail(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        raise primary
    monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', fail)
    with pytest.raises(ValueError) as caught:
        if comparison:
            asyncio.run(evaluation.evaluate_comparison(store, loaded, loaded, proposal, proposal, *training,
                comparison_id='component', conditions_sha256='c'*64, output=tmp_path / 'evaluation'))
        else:
            asyncio.run(evaluation.evaluate_seed(store, loaded, {'campaign_id': 'component', 'conditions_sha256': 'c'*64},
                proposal, training[0], output=tmp_path / 'evaluation'))
    assert caught.value is primary
    assert primary.hpo_gpu_termination_receipt.resources
    folder = 'matchups/17-17' if comparison else 'proposals/000000/seeds/17'
    name = 'failed-gpu-termination.json' if comparison else 'evaluation-failed-gpu-termination.json'
    assert store.exists(folder + '/' + name)
    assert not store.exists(folder + '/receipt.json') and not store.exists(folder + '/evaluation.json')
    for parent in parents[:2 if comparison else 1]:
        journal = parent / 'evaluation-reference-receipts' / primary.hpo_gpu_termination_receipt.ticket
        assert (journal / 'release.receipt.json').exists()


@pytest.mark.parametrize('comparison', [False, True])
def test_outer_attempts_other_pins_when_release_uncertain_and_never_terminates(tmp_path, monkeypatch, outer_workflow, comparison):
    from gf_training import hpo_retention as retention
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    evaluation, store, loaded, proposal, training, _ = outer_workflow
    primary = ValueError('primary data error')
    actual = GroupPackCollectionPublication.prepare
    async def fail(self, *args, **kwargs):
        await actual(self, *args, **kwargs)
        raise primary
    monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', fail)
    attempted = []
    release = retention.release_hpo_evaluation
    def uncertain(directory, *args, **kwargs):
        attempted.append(directory)
        if len(attempted) == 1: raise OSError('first pin uncertain')
        return release(directory, *args, **kwargs)
    monkeypatch.setattr(retention, 'release_hpo_evaluation', uncertain)
    with pytest.raises(ValueError) as caught:
        if comparison:
            asyncio.run(evaluation.evaluate_comparison(store, loaded, loaded, proposal, proposal, *training,
                comparison_id='component', conditions_sha256='c'*64, output=tmp_path / 'evaluation'))
        else:
            asyncio.run(evaluation.evaluate_seed(store, loaded, {'campaign_id': 'component', 'conditions_sha256': 'c'*64},
                proposal, training[0], output=tmp_path / 'evaluation'))
    assert caught.value is primary
    assert len(attempted) == (2 if comparison else 1)
    assert not hasattr(primary, 'hpo_gpu_termination_receipt')
    assert primary.hpo_termination_cleanup_error is not None


@pytest.mark.parametrize('mutation', ['policies', 'game-id', 'health', 'replay'])
def test_actual_publication_keeps_strict_eval_rows(tmp_path, monkeypatch, mutation):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    actual = GroupPackCollectionPublication.prepare
    async def mutate(self, report, manifest):
        rows = manifest['episodes']
        if mutation == 'policies': rows[0]['policies'] = rows[0]['policies'][::-1]
        elif mutation == 'game-id': rows[1]['game_id'] = rows[0]['game_id']
        elif mutation == 'health': rows[0]['health']['infra'] = 'unhealthy'
        else: rows[0]['replay_reconciled'] = False
        await actual(self, report, manifest)
    monkeypatch.setattr(GroupPackCollectionPublication, 'prepare', mutate)
    try:
        with pytest.raises(ContractError, match='non-natural/unhealthy/mismatched') as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
        observation.require_core_retired(error=caught.value)
        assert observation._publication.cleanup_complete
        assert not (plan.output / 'collection.locator.json').exists()
    finally:
        owner.release()
        job.confirm_termination()
        job.close()


def test_partial_factory_transfer_keeps_original_tasks_and_refuses_authority(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch, cpu_processes=True)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = RuntimeError('factory transferred then raised')
    transferred = []
    original = asyncio.create_task
    count = 0
    def factory(coroutine, **kwargs):
        nonlocal count
        if coroutine.__name__ == 'group_run':
            count += 1
            if count == 2:
                transferred.append(original(coroutine, **kwargs))
                raise primary
        return original(coroutine, **kwargs)
    monkeypatch.setattr(asyncio, 'create_task', factory)
    async def run():
        with pytest.raises(RuntimeError) as caught:
            await core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation)
        assert caught.value is primary
        assert len(observation._tasks) == 1 and observation._tasks[0].done()
        assert primary.hpo_task_creation_coroutine is transferred[0].get_coro()
        # The factory's unknown transfer was never closed by the failed creator.
        await asyncio.gather(*transferred, return_exceptions=True)
        with pytest.raises(ContractError, match='creation not settled'):
            observation.require_core_retired(error=primary)
        await observation._pack.close()
    try:
        asyncio.run(run())
    finally:
        owner.release()
        job.close()


def test_verifier_constructor_retains_same_partial_source_and_blocks_termination(tmp_path, monkeypatch, retired_failure):
    # Use a real completed original publication, then fail its actual initializer.
    import gf_cli.hpo_evaluation as evaluation
    from gf_cli.hpo_config import campaign_config
    from gf_training.group_pack_game_source import PackedSourceOwner
    from tests.test_hpo_config import campaign_raw
    _, observation, job, _, _, primary = retired_failure
    actual = PackedSourceOwner.__init__
    def fail(self, *args, **kwargs):
        actual(self, *args, **kwargs)
        raise primary
    monkeypatch.setattr(PackedSourceOwner, '__init__', fail)
    # Constructor fails before a new original can be established; keep that
    # actual partial owner, never treat absent returned owner as no resources.
    with pytest.raises((ValueError, FileNotFoundError)) as caught:
        evaluation.verify_saved_evaluation(observation._identity[0].output, campaign_config(campaign_raw(), origin=tmp_path),
            observation._identity[0].linkage, deadline=time.monotonic() + 30, _cleanup_observation=observation)
    assert caught.value.hpo_verifier_cleanup_owner is observation._verifier
    assert not confirm_failure(tmp_path, retired_failure)
    assert job._receipt is None
    observation._verifier.close()


def test_precreation_error_has_positive_unentered_phase_not_empty_ledger_oracle(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from tests.hpo_lineage_support import advanced_hpo_chain
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    parent = tmp_path / 'parent'
    _, records, _ = advanced_hpo_chain(parent, updates=1, budget=3)
    acquired = pin_hpo_evaluation(parent, job.ticket, 1, records[1]['inference_sha256'],
        max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = ValueError('actual precreation artifact recheck failed')
    def fail(*args, **kwargs): raise primary
    monkeypatch.setattr(owner, 'finalize', fail)
    try:
        with pytest.raises(ValueError) as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
        assert caught.value is primary and observation._run_phase == 'unentered'
        observation.release_artifacts()
        released = release_hpo_evaluation(parent, acquired.binding, max_owned_bytes=1024**3, deadline=time.monotonic() + 30)
        values = evaluation, observation, job, [(parent, acquired)], {id(acquired): released}, primary
        assert confirm_failure(tmp_path, values)
        assert job._receipt.resources == ()
    finally:
        owner.release()
        job.close()


def test_pending_actual_server_drain_cannot_authorize_cleanup(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_simulator_client.process import ManagedServer
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch, cpu_processes=True)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    stop = ManagedServer.stop
    pending = []
    async def not_drained(self, *args, **kwargs):
        result = await stop(self, *args, **kwargs)
        task = asyncio.create_task(asyncio.Event().wait())
        self.drainers.append(task)
        pending.append(task)
        return result
    monkeypatch.setattr(ManagedServer, 'stop', not_drained)
    async def run():
        with pytest.raises(Exception) as caught:
            await core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation)
        assert pending and all(not task.done() for task in pending)
        with pytest.raises(ContractError): observation.require_core_retired(error=caught.value)
        for task in pending: task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    try:
        asyncio.run(run())
        assert job._receipt is None
    finally:
        owner.release()
        job.close()


def test_partial_pack_constructor_is_reachable_cleaned_and_never_proof(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    from gf_training.group_pack_collection import GroupPackCollection
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = ValueError('unknown partial Pack initialization')
    original = GroupPackCollection.__init__
    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise primary
    monkeypatch.setattr(GroupPackCollection, '__init__', fail)
    try:
        with pytest.raises(ValueError) as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
        assert caught.value is primary
        assert primary.hpo_partial_pack_owner is observation._pack
        assert observation._pack._closed
        with pytest.raises(ContractError, match='Pack retirement'):
            observation.require_core_retired(error=primary)
        assert job._receipt is None
    finally:
        owner.release()
        job.close()


def test_partial_group_environment_retained_before_initializer_returns(tmp_path, monkeypatch):
    import gf_cli.card_instance as core
    import gf_cli.hpo_evaluation as evaluation
    loaded, plan, job, owner = actual_pack_scenario(tmp_path, monkeypatch)
    observation = evaluation._HpoEvaluationCleanupObservation(plan, job, owner)
    primary = ValueError('group initialization failed')
    original = CardInstanceSimulatorEnvironment.__init__
    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if kwargs.get('artifact_sink') is not None: raise primary
    monkeypatch.setattr(CardInstanceSimulatorEnvironment, '__init__', fail)
    try:
        with pytest.raises(ValueError) as caught:
            asyncio.run(core.collect_hpo_evaluation(loaded, plan, gpu_job=job, artifact_inputs=owner,
                _cleanup_observation=observation))
        assert caught.value is primary
        environments = [value[2] for value in observation._groups.values() if value[1] == 'creating']
        assert environments and all(environment is not None for environment in environments)
        assert all(value[2] is None for value in observation._groups.values() if value[1] == 'scheduled')
        assert all(group.runtime_owner in environments for group in observation._pack._groups)
        with pytest.raises(ContractError): observation.require_core_retired(error=primary)
    finally:
        owner.release()
        job.close()


def install_unqualified_cpu_resident_originals(monkeypatch):
    """Hand-authored terminal DTOs; real workers/journals/Replay/Source/Pack gates.

    Existing parent Source-inspection and Simulator/runtime facts remain the
    explicitly UNQUALIFIED fixture boundaries. The one additional production
    bypass is the approved CUDA-only guard. Evaluation journals/Replay/Source
    parsing and ownership remain real; no Simulator/GPU or eligible study ran.
    """
    import copy
    import uuid
    from itertools import count

    import gf_cli.card_instance as core
    from gf_common.domain import GroupResult
    from tests.card_instance_action_support import inputs
    from tests.test_card_instance_baseline_runtime import receipt
    from tests.test_card_instance_learning import ACTIONS
    validate=core._validate_recorded_episodes
    install_component_execution(monkeypatch,cpu_processes=False)
    monkeypatch.setattr(core,'_validate_recorded_episodes',validate)
    def require_cpu_fixture(binding):
        assert binding.policy_options['version']=='4' and binding.policy_options['inference']['device']=='cpu'
    monkeypatch.setattr(core,'_require_hpo_evaluation_policy',require_cpu_fixture)
    monkeypatch.setattr(core.CardInstanceRuntimeEvidence,'receipt',lambda self,**kwargs:self._component_receipt,raising=False)
    executions=count()
    async def originals(self,group):
        self._storage_group_owner=self
        self._managed_server=None
        runtime=receipt()
        index=next(executions)
        runtime['executed'].update(pid=123+index,start_time=1+index)
        self.runtime_evidence._component_receipt=runtime
        episodes=[]
        for spec in group.episodes:
            game=uuid.uuid4().hex
            result=CardInstanceEpisodeResult(spec.episode_id,spec.pair_id,group.seed,spec.policies,
                game_id=game,outcome='draw',end_kind='natural',replay_reconciled=True,action_contract=ACTIONS.action_contract)
            folder=self.settings.output_dir/spec.episode_id
            journals,workers,contexts={},{},{}
            try:
                for viewer in (1,2):
                    self._create_actor(folder,spec,viewer,game,journals,workers,contexts,result)
                await self._ready_actors(workers,result)
                for viewer in (1,2):
                    value=inputs([],can_pray=True)
                    observation=copy.deepcopy(value.observation)
                    if viewer==2:
                        observation.update(viewer_player_id=2,current_player=2,active_player=2)
                        observation['own']['player_id'],observation['opponent']['player_id']=2,1
                    observation.update(state_revision=1,actionable=False,typed_action_schema=[],required_choice=None,
                        current_player=None,phase='terminal',result='draw')
                    journals[viewer].finish('natural','draw',observation)
                    replay={'game_id':game,'viewer_player_id':viewer,'requested_turn':None,
                        'rule_set_version':ACTIONS.action_contract.rule_set_version,'observation_schema_version':'2',
                        'public_event_schema_version':'2','card_ref_scheme':ACTIONS.action_contract.card_ref_scheme,
                        'episode_end':{'type':'terminal','result':'draw'},'turns':[]}
                    self._save_artifact(folder/f'replay-player-{viewer}.json',replay)
                self._before_worker_close(workers,result)
            finally:
                await asyncio.gather(*(worker.close() for worker in workers.values()))
            self._save_artifact(folder/'result.json',asdict(result))
            episodes.append(result)
        return GroupResult(episodes,{'runtime_evidence':runtime,'process':{'exit_code':0,'shutdown':'stdin_eof','log_error':None}})
    monkeypatch.setattr(CardInstanceSimulatorEnvironment,'run_group',originals)


@pytest.mark.parametrize('outer_workflow',[True],indirect=True)
@pytest.mark.parametrize('comparison',[False,True])
@pytest.mark.parametrize('publication',['termination','evaluation'])
def test_resident_local_publication_failure_keeps_exact_confirmed_global_receipt(tmp_path,monkeypatch,outer_workflow,comparison,publication):
    evaluation,store,loaded,proposal,training,_parents=outer_workflow
    primary=OSError('local publication failed after actual global confirmation')
    original=store.put
    captured=[]
    def fail(name,value,*args,**kwargs):
        target=(name.endswith('/gpu-termination.json') if comparison else name.endswith('/evaluation-gpu-termination.json')) if publication=='termination' else (name.endswith('/receipt.json') if comparison else name.endswith('/evaluation.json'))
        if target:
            from gf_training import hpo_gpu_lease as gpu
            current=gpu._read(gpu._GLOBAL_ROOT/'current.json')
            assert (gpu._GLOBAL_ROOT/'jobs'/current['ticket']/'termination.json').is_file()
            captured.append(current['ticket'])
            raise primary
        return original(name,value,*args,**kwargs)
    monkeypatch.setattr(store,'put',fail)
    with pytest.raises(OSError) as caught:
        if comparison:
            asyncio.run(evaluation.evaluate_comparison(store,loaded,loaded,proposal,proposal,*training,
                comparison_id='component',conditions_sha256='c'*64,output=tmp_path/'evaluation',evaluation_runtime=loaded.config.runtime))
        else:
            asyncio.run(evaluation.evaluate_seed(store,loaded,{'campaign_id':'component','conditions_sha256':'c'*64},
                proposal,training[0],output=tmp_path/'evaluation'))
    assert caught.value is primary and captured
    assert primary.hpo_gpu_termination_receipt.ticket==captured[0]
    assert not hasattr(primary,'hpo_termination_cleanup_error')
    folder='matchups/17-17' if comparison else 'proposals/000000/seeds/17'
    assert not store.exists(folder+('/receipt.json' if comparison else '/evaluation.json'))
    failed=store.get(folder+('/failed-gpu-termination.json' if comparison else '/evaluation-failed-gpu-termination.json'))[0]
    assert failed['evaluation_eligible'] is False and failed['gpu_termination']['ticket']==captured[0]
