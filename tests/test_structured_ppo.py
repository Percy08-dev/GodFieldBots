"""Engineering-only existing coordinator integration, no game/fit acceptance."""
from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.domain import ContractError, digest
from gf_training.journal import atomic_json
from gf_training.structured_ppo import _train_component_ppo
from gf_training.structured_ppo_source import (
    check_original_collection_parent,
    validate_component_parent,
    validate_owned_collection,
)
from tests.structured_ppo_support import (
    CompletedComponentExecution,
    clone_component_parent,
    component_http,
    component_inventory,
    configuration,
)


def test_default_mlp_cpu_original16game_two_update_common_loop(completed_execution):
    parent, result = completed_execution.parent, completed_execution.result
    scripts, workers, ledgers = completed_execution.scripts, completed_execution.workers, completed_execution.ledgers
    assert result['status'] == 'complete' and result['updates'] == 2 and result['accepted_decisions'] == 64
    state, generations = validate_component_parent(parent, max_owned_bytes=2 * 1024**3,
                                                    deadline=time.monotonic() + 60)
    assert state['schema_version'] == 2 and len(generations) == 3
    assert state['identity']['process_envelope_recipe'] == 'structured-process-envelope-v1'
    assert all(ledger is ledgers[0] for ledger in ledgers) and ledgers[0].learner is None
    assert not ledgers[0].workers and not ledgers[0].updating
    assert len(workers) == 64 and all(worker.process.poll() is not None and worker.closed for worker in workers)
    assert len(scripts) == 16 and all(script.deleted for script in scripts)
    for number in (1, 2):
        update = json.loads((parent / 'updates' / f'{number:06}.json').read_text())
        assert update['actual_games'] == update['planned_games'] == 16
        assert update['source_proof']['qualified'] is False and update['source_proof']['parent_qualification'] == 'not_performed'
        assert update['accepted_decisions'] == 32 and len(update['learner_seats']) == 16


@pytest.mark.parametrize('mutation', ['visibility', 'execution_digest', 'classification'])
def test_full_owned_wire_identity_rejects_before_any_collection(tmp_path, monkeypatch, mutation):
    loaded = configuration(tmp_path)
    called = []
    async def collector(plan, ledger):
        owner = ledger.learner._cleanup_owner
        identity, visibility = copy.deepcopy(owner.identity), owner.visibility
        try:
            if mutation == 'visibility':
                owner.visibility = '0'
            elif mutation == 'execution_digest':
                owner.identity['execution_digest'] = 'f' * 64
            else:
                owner.identity['synthetic_capacity'] = True
            with pytest.raises(ContractError, match='learner identity'):
                validate_owned_collection(plan, ledger, max_owned_bytes=2 * 1024**3, deadline=plan.execution_deadline)
            called.append(1)
        finally:
            owner.identity, owner.visibility = identity, visibility
        raise RuntimeError('stop after rejected owner probe')
    with pytest.raises(RuntimeError, match='rejected owner probe'):
        asyncio.run(_train_component_ppo(loaded, collector, tmp_path / 'parent'))
    assert called == [1] and not list((tmp_path / 'parent/collections').iterdir())
    state = json.loads((tmp_path / 'parent/run.json').read_text())
    assert state['status'] == 'failed' and state['updates'] == 0


@pytest.mark.parametrize('mutation', ['training', 'server', 'binary_sha256', 'server_config_sha256'])
def test_original_full_config_and_recorded_provenance_guard(tmp_path, mutation):
    loaded = configuration(tmp_path)
    state = {'effective_config': loaded.effective, 'config_sha256': loaded.sha256, 'config_inputs': [],
             'identity': {'code': {'revision': None, 'dirty': None, 'installed_build_revision': None},
                 'integration': {'binary_sha256': 'a' * 64, 'server_config_sha256': 'b' * 64,
                     'declared_simulator_revision': loaded.config.server.simulator_revision,
                     'source_revision': None, 'source_spec_sha256': None}}}
    manifest = {'effective_config': copy.deepcopy(loaded.effective), 'config_sha256': loaded.sha256, 'config_inputs': [],
        'code': state['identity']['code'], 'binary_sha256': 'a' * 64, 'server_config_sha256': 'b' * 64,
        'simulator_contract_revision': loaded.config.server.simulator_revision, 'simulator': {'revision': None, 'spec_sha256': None}}
    check_original_collection_parent(manifest, state)
    if mutation in ('training', 'server'):
        if mutation == 'training':
            manifest['effective_config']['training']['device'] = 'cuda:0'
        else:
            manifest['effective_config']['server']['binary'] = '/foreign/unexecuted'
        manifest['config_sha256'] = digest(manifest['effective_config'])
    else:
        manifest[mutation] = 'f' * 64
    with pytest.raises(ContractError, match='recorded parent'):
        check_original_collection_parent(manifest, state)


@pytest.fixture(scope='module')
def completed_execution(tmp_path_factory):
    directory = tmp_path_factory.mktemp('component-parent-template')
    loaded = configuration(directory)
    parent = directory / 'parent'
    with pytest.MonkeyPatch.context() as patch:
        collector, scripts, workers, ledgers = component_http(patch, loaded)
        result = asyncio.run(_train_component_ppo(loaded, collector, parent))
        assert all(worker.process.poll() is not None for worker in workers)
        assert all(ledger.learner is None and not ledger.workers for ledger in ledgers)
    execution = CompletedComponentExecution(loaded, parent, result, tuple(scripts), tuple(workers), tuple(ledgers),
                                            component_inventory(parent))
    execution.assert_default_cpu_reference(kind='mlp', learning_device='cpu', inference_device='cpu')
    validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    yield execution
    execution.assert_closed()
    execution.assert_unchanged()


@pytest.fixture(scope='module')
def completed_component(completed_execution):
    return completed_execution.parent


@pytest.mark.parametrize('mutation', ['report', 'journal', 'sidecar', 'inventory'])
def test_whole_parent_rechecks_earlier_source_during_later_batch(completed_component, tmp_path, monkeypatch, mutation):
    import gf_training.structured_ppo_source as module
    parent = clone_component_parent(completed_component, tmp_path / 'parent')
    module.validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    actual = module.component_batch_source
    touched = []
    @contextmanager
    def later(plan, **kwargs):
        with actual(plan, **kwargs) as source:
            if plan.linkage['collection_generation'] == 1:
                earlier = parent / 'collections/batch-000000'
                if mutation == 'report':
                    path = earlier / 'report.json'
                else:
                    episode = next(earlier.glob('b*-seat0'))
                    path = (episode / 'player-1.jsonl' if mutation == 'journal' else
                            next(episode.glob('*.columns')) if mutation == 'sidecar' else episode / 'late.columns')
                path.write_bytes((path.read_bytes() if path.exists() else b'') + b'late change')
                touched.append(path)
            yield source
    monkeypatch.setattr(module, 'component_batch_source', later)
    with pytest.raises(ContractError, match='Source|source|hash|snapshot|changed'):
        module.validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    assert len(touched) == 1


def test_raw_source_retires_before_next_batch_metadata_only(completed_component, tmp_path, monkeypatch):
    import weakref

    import gf_training.structured_ppo_source as module
    parent = clone_component_parent(completed_component, tmp_path / 'parent')
    actual, refs = module.component_batch_source, []
    @contextmanager
    def observed(plan, **kwargs):
        if refs:
            assert refs[0]() is None
        with actual(plan, **kwargs) as source:
            refs.append(weakref.ref(source.records[0]))
            yield source
    monkeypatch.setattr(module, 'component_batch_source', observed)
    module.validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    assert len(refs) == 2 and all(reference() is None for reference in refs)


def test_generation_zero_rng_anchor_cannot_be_rehashed_to_another_seed(completed_component, tmp_path):
    from gf_models.arrays import sha256
    parent = clone_component_parent(completed_component, tmp_path / 'parent')
    validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    rng = np.random.Generator(np.random.PCG64(42))
    previous = None
    for number in range(3):
        path = parent / 'generations' / f'{number:06}.json'
        record = json.loads(path.read_text())
        record['parent_controls']['schedule_opponent_rng'] = rng.bit_generator.state
        record['parent_sha256'] = previous
        if number:
            update_path = parent / 'updates' / f'{number:06}.json'
            update = json.loads(update_path.read_text())
            update['parent_sha256'] = previous
            atomic_json(update_path, update)
            record['update_sha256'] = sha256(update_path)
        atomic_json(path, record)
        previous = sha256(path)
        for _ in range(4):
            rng.integers(1)
    state = json.loads((parent / 'run.json').read_text())
    state['generation_sha256'] = previous
    atomic_json(parent / 'run.json', state)
    atomic_json(parent / 'latest.json', {'generation': 2, 'sha256': previous})
    with pytest.raises(ContractError, match='generation0 PCG64'):
        validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)


def test_cpu_host_budget_is_independent_of_tiny_cuda_budget(tmp_path):
    loaded = configuration(tmp_path)
    config = replace(loaded.config, run=replace(loaded.config.run, global_cuda_budget_bytes=1))
    from gf_cli.config import LoadedConfig
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    called = []
    async def collector(plan, ledger):
        assert ledger.global_cuda_bytes == 1 and ledger.cuda_bytes == 0
        validate_owned_collection(plan, ledger, max_owned_bytes=2 * 1024**3, deadline=plan.execution_deadline)
        called.append(1)
        raise RuntimeError('stop host-budget boundary probe')
    with pytest.raises(RuntimeError, match='host-budget boundary'):
        asyncio.run(_train_component_ppo(loaded, collector, tmp_path / 'parent'))
    assert called == [1]


@pytest.mark.parametrize('field', ['global_cuda_bytes', 'max_workers'])
def test_owned_ledger_limits_are_bound_to_frozen_run(tmp_path, field):
    loaded = configuration(tmp_path)
    called = []
    async def collector(plan, ledger):
        previous = getattr(ledger, field)
        setattr(ledger, field, previous - 1)
        try:
            with pytest.raises(ContractError, match='ledger limits'):
                validate_owned_collection(plan, ledger, max_owned_bytes=2 * 1024**3, deadline=plan.execution_deadline)
            called.append(1)
        finally:
            setattr(ledger, field, previous)
        raise RuntimeError('stop changed-ledger boundary probe')
    with pytest.raises(RuntimeError, match='changed-ledger boundary'):
        asyncio.run(_train_component_ppo(loaded, collector, tmp_path / 'parent'))
    assert called == [1]


def test_public_and_direct_guards_unpatched_before_any_real_activity(tmp_path, monkeypatch):
    import gf_cli
    from gf_cli.config import LoadedConfig
    from gf_common.action_contract import require_qualified_contract
    from gf_common.domain import EpisodeGroup, EpisodeSpec
    from gf_models.config import ModelConfig
    from gf_simulator_client import environment
    from gf_simulator_client.environment import EnvironmentSettings
    from gf_simulator_client.structured_environment import StructuredSimulatorEnvironment
    from gf_simulator_client.structured_resources import ReservationLedger
    loaded = configuration(tmp_path)
    assert gf_cli.require_qualified_contract is require_qualified_contract
    assert environment.require_qualified_contract is require_qualified_contract
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('real activity occurred before central guard')
    monkeypatch.setattr(environment, 'ManagedServer', forbidden)
    monkeypatch.setattr(environment, 'AttestedClient', forbidden)
    with pytest.raises(ContractError, match='engineering'):
        asyncio.run(gf_cli.run_experiment(loaded, 'collect'))
    settings = EnvironmentSettings(loaded.config.server.binary, loaded.config.server.config, tmp_path / 'env', 'run',
                                   actions=loaded.config.actions)
    with pytest.raises(ContractError, match='engineering'):
        StructuredSimulatorEnvironment(settings, forbidden, forbidden, ledger=ReservationLedger())
    direct = StructuredSimulatorEnvironment.__new__(StructuredSimulatorEnvironment)
    direct.settings = settings
    group = EpisodeGroup('00000000-0000-0000-0000-000000000001', ())
    with pytest.raises(ContractError, match='engineering'):
        asyncio.run(direct.run_group(group))
    with pytest.raises(ContractError, match='engineering'):
        asyncio.run(direct._episode(None, {}, group, EpisodeSpec('e', ('a', 'b'), (1, 2), 'p')))
    baseline = replace(loaded.config, model=ModelConfig(model_id='heuristic'),
        run=replace(loaded.config.run, output_dir=str(tmp_path / 'private-normal')))
    baseline = LoadedConfig(baseline, asdict(baseline), digest(asdict(baseline)), ())
    # Normal factory through the private shared body still reaches its guarded
    # constructor, and cannot start the real server/client/participant path.
    with pytest.raises(ContractError, match='engineering'):
        asyncio.run(gf_cli._run_experiment(baseline, 'collect'))
    assert not calls


def test_v2_control_io_is_caller_bounded_and_preserves_original_identity(tmp_path, monkeypatch):
    import gf_training.structured_ppo as module
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing,
        memory_budget_mib=1, cache_budget_mib=0, chunk_target_mib=1))
    path = tmp_path / 'run.json'
    path.write_bytes(b' ' * 2 * 1024**2 + b'{"execution_deadline_unix":9999999999}')
    import gf_models.structured_sealed as sealed
    loads = sealed.json.loads
    def forbidden(*args, **kwargs):
        raise AssertionError('unadmitted control JSON parsed')
    monkeypatch.setattr(sealed.json, 'loads', forbidden)
    with pytest.raises(ContractError, match='reservation'):
        module.read_control(path, config, deadline=time.monotonic() + 10)
    monkeypatch.setattr(sealed.json, 'loads', loads)
    atomic_json(path, {'execution_deadline_unix': 9999999999})
    _, snapshot = module.read_control(path, loaded.config, deadline=time.monotonic() + 10)
    atomic_json(path, {'execution_deadline_unix': 9999999998})
    with pytest.raises(ContractError, match='snapshot|changed|hash'):
        module.persist_control(path, {'execution_deadline_unix': 9999999997}, loaded.config,
                               deadline=time.monotonic() + 10, original=snapshot)


def test_v2_control_temp_collision_preserves_foreign_target(tmp_path):
    from gf_training.structured_ppo import persist_control
    loaded = configuration(tmp_path)
    outside = tmp_path / 'outside'
    outside.write_bytes(b'foreign evidence')
    path = tmp_path / 'run.json'
    path.with_name('run.json.tmp').symlink_to(outside)
    with pytest.raises(FileExistsError):
        persist_control(path, {'schema_version': 2}, loaded.config, deadline=time.monotonic() + 10)
    assert outside.read_bytes() == b'foreign evidence' and not path.exists()


def test_extra_parent_metadata_and_lower_source_cap_apply_before_first_json(tmp_path, make_input, monkeypatch):
    import gf_training.structured_source as module
    from gf_common.preprocessing import PreparationConfig
    from gf_training.preparation import ResourceLimit
    from tests.test_structured_ppo_source import fixture
    values = fixture(tmp_path, make_input)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Source JSON parsed outside joint caller cap')
    monkeypatch.setattr(module, '_json', forbidden)
    with pytest.raises(ResourceLimit), module._component_source_snapshot(values[0].output,
            actions=values[1], settings=PreparationConfig(reader='structured-v1', memory_budget_mib=512),
            max_owned_bytes=1024**2, extra_owned_bytes=1024**2 - 1,
            deadline=time.monotonic() + 10):
        pass
    assert calls == []


def run_continuous_and_resume(tmp_path, monkeypatch, *, kind, learning_device, inference_device,
                              completed_reference=None):
    from gf_models.arrays import sha256
    from tests.test_structured_numeric_worker import assert_artifacts_equal
    if completed_reference is None:
        loaded = configuration(tmp_path, kind=kind, learning_device=learning_device, inference_device=inference_device)
    else:
        assert isinstance(completed_reference, CompletedComponentExecution)
        completed_reference.assert_default_cpu_reference(kind=kind, learning_device=learning_device,
                                                         inference_device=inference_device)
        loaded = completed_reference.loaded
    assert loaded.config.model.hand.kind == 'attention'
    if kind == 'transformer':
        assert loaded.config.model.temporal.num_layers == 2
    assert loaded.config.run.wall_seconds == 7200 and loaded.config.run.admission_seconds == 5400
    assert loaded.config.run.execution_seconds == 7140 and loaded.config.training.max_owned_bytes == 10 * 1024**3
    assert loaded.config.training.admission_reservation_bytes == 1024**3
    collector, scripts, workers, ledgers = component_http(monkeypatch, loaded)
    continuous, resumed = tmp_path / 'continuous', tmp_path / 'resumed'
    if completed_reference is None:
        first = asyncio.run(_train_component_ppo(loaded, collector, continuous))
    else:
        continuous, first = completed_reference.parent, completed_reference.result
    paused = asyncio.run(_train_component_ppo(loaded, collector, resumed, pause_after_update=1))
    assert first['updates'] == 2 and paused['status'] == 'paused' and paused['updates'] == 1
    paused_state = json.loads((resumed / 'run.json').read_text())
    if kind == 'mlp' and learning_device == 'cpu':
        from gf_cli.config import LoadedConfig
        scripts_before = len(scripts)
        for mutation in ('window', 'resources', 'visibility', 'device'):
            config = loaded.config
            if mutation == 'window':
                config = replace(config, model=replace(config.model,
                    event=replace(config.model.event, aggregate_window=64)))
            elif mutation == 'resources':
                config = replace(config, model=replace(config.model,
                    resources=replace(config.model.resources, workspace_bytes=config.model.resources.workspace_bytes + 1)))
            elif mutation == 'visibility':
                config = replace(config, model=replace(config.model, cuda_visible_devices='0'))
            else:
                config = replace(config, training=replace(config.training, device='cuda:0'))
            changed = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
            with pytest.raises(ContractError, match='resume code/runtime/config/device'):
                asyncio.run(_train_component_ppo(changed, collector, resumed, mode='resume'))
            assert json.loads((resumed / 'run.json').read_text())['status'] == 'paused'
            assert len(scripts) == scripts_before
    completed = asyncio.run(_train_component_ppo(loaded, collector, resumed, mode='resume'))
    assert completed['status'] == 'complete' and completed['updates'] == 2
    assert_artifacts_equal(continuous / 'artifacts/000002', resumed / 'artifacts/000002')
    left = json.loads((continuous / 'artifacts/000002/resume.json').read_text())
    right = json.loads((resumed / 'artifacts/000002/resume.json').read_text())
    assert left['resume_identity'] == right['resume_identity']
    for number in (0, 1):
        a = json.loads((continuous / 'collections' / f'batch-{number:06}/manifest.json').read_text())
        b = json.loads((resumed / 'collections' / f'batch-{number:06}/manifest.json').read_text())
        assert a['schedule'] == b['schedule'] and a['ppo']['learner_seats'] == b['ppo']['learner_seats']
        assert len(a['episodes']) == len(b['episodes']) == 16
        assert a['journal_schema_version'] == b['journal_schema_version'] == 5
        for label in a['policy_bindings']:
            x, y = copy.deepcopy(a['policy_bindings'][label]), copy.deepcopy(b['policy_bindings'][label])
            if x['policy_options'] is not None:
                assert x['policy_options'].pop('checkpoint').replace(str(continuous), '') == y['policy_options'].pop('checkpoint').replace(str(resumed), '')
            assert x == y
    a, records_a = validate_component_parent(continuous, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    b, records_b = validate_component_parent(resumed, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    for key in ('created_unix', 'wall_deadline_unix', 'admission_deadline_unix', 'execution_deadline_unix', 'resume_identity'):
        assert b[key] == paused_state[key]
    assert a['accepted_decisions'] == b['accepted_decisions'] == 64
    assert a['optimizer_steps'] == b['optimizer_steps'] == 8
    assert [record['parent_controls'] for record in records_a] == [record['parent_controls'] for record in records_b]
    for parent in (continuous, resumed):
        registry = json.loads((parent / 'artifact-registry.json').read_text())
        assert registry['generations']['1']['status'] == 'deleted'
        assert {path.name for path in (parent / 'artifacts/000001').iterdir()} == {'inference.json'}
        assert registry['generations']['1']['original_files']['inference.json'] == sha256(parent / 'artifacts/000001/inference.json')
    pair_workers, pair_scripts, pair_ledgers = tuple(workers), tuple(scripts), tuple(ledgers)
    if completed_reference is not None:
        completed_reference.assert_closed()
        completed_reference.assert_unchanged()
        pair_workers += completed_reference.workers
        pair_scripts += completed_reference.scripts
        pair_ledgers += completed_reference.ledgers
    assert len(pair_workers) == 128 and len(pair_scripts) == 32
    assert all(worker.closed and worker.process.poll() is not None for worker in pair_workers)
    assert len({id(ledger) for ledger in pair_ledgers}) == 3  # one continuous +fresh paused +resume invocation.
    assert all(not ledger.workers and ledger.learner is None and not ledger.updating for ledger in pair_ledgers)


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_default_cpu_three_models_exact_continuous_vs_pause_resume(tmp_path, monkeypatch, kind, request):
    reference = request.getfixturevalue('completed_execution') if kind == 'mlp' else None
    run_continuous_and_resume(tmp_path, monkeypatch, kind=kind, learning_device='cpu', inference_device='cpu',
                              completed_reference=reference)


@pytest.mark.parametrize('mutation', ['kind', 'learning_device', 'inference_device', 'recorded_model',
                                     'recorded_device', 'recorded_config', 'recorded_result'])
def test_completed_reference_mismatch_rejects_before_any_activity(completed_execution, tmp_path, monkeypatch, mutation):
    from gf_cli.config import LoadedConfig
    from gf_models.structured_binding import model_defaults_v2

    reference = completed_execution
    options = {'kind': 'mlp', 'learning_device': 'cpu', 'inference_device': 'cpu'}
    if mutation in options:
        options[mutation] = 'gru' if mutation == 'kind' else 'cuda:0'
    elif mutation == 'recorded_result':
        reference = replace(reference, result={**reference.result, 'status': 'paused'})
    else:
        config = reference.loaded.config
        if mutation == 'recorded_model':
            config = replace(config, model=model_defaults_v2('gru'))
        elif mutation == 'recorded_device':
            config = replace(config, training=replace(config.training, device='cuda:0'))
        else:
            config = replace(config, run=replace(config.run, wall_seconds=config.run.wall_seconds - 1))
        reference = replace(reference, loaded=LoadedConfig(config, asdict(config), digest(asdict(config)), ()))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('reference mismatch reached collection or training')
    monkeypatch.setattr(sys.modules[__name__], 'component_http', forbidden)
    monkeypatch.setattr(sys.modules[__name__], '_train_component_ppo', forbidden)
    with pytest.raises(AssertionError):
        run_continuous_and_resume(tmp_path, monkeypatch, **options, completed_reference=reference)
    assert calls == [] and not list(tmp_path.iterdir())
    completed_execution.assert_closed()
    completed_execution.assert_unchanged()


def test_whole_interrupted_cohort_never_updates_or_salvages_healthy_rows(tmp_path, monkeypatch):
    loaded = configuration(tmp_path)
    collector, scripts, workers, ledgers = component_http(monkeypatch, loaded, mode='turn_limit')
    parent = tmp_path / 'parent'
    with pytest.raises(ContractError, match='original batch'):
        asyncio.run(_train_component_ppo(loaded, collector, parent))
    state = json.loads((parent / 'run.json').read_text())
    assert state['status'] == 'failed' and state['quarantine'] and state['updates'] == 0
    assert not list((parent / 'numeric').iterdir())
    manifest = json.loads((parent / 'collections/batch-000000/manifest.json').read_text())
    assert len(manifest['episodes']) == 16 and all(item['outcome'] is None for item in manifest['episodes'])
    assert all(item['end_kind'] == 'truncated' for item in manifest['episodes'])
    assert all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert all(ledger.learner is None and not ledger.workers for ledger in ledgers) and scripts
    with pytest.raises(ContractError):
        validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)


@pytest.mark.parametrize('failure', ['ack', 'commit', 'postupdate_source'])
def test_failed_update_boundary_never_commits_healthy_generation(tmp_path, monkeypatch, failure):
    import gf_training.structured_ppo as module
    from gf_training.ppo_run import LearningProcess
    from gf_training.structured_learning_process import StructuredLearningProcess
    loaded = configuration(tmp_path)
    collector, _, workers, ledgers = component_http(monkeypatch, loaded)
    parent = tmp_path / 'parent'
    if failure == 'ack':
        actual = LearningProcess.request
        async def bad(self, request, deadline):
            if request.get('command') == 'update':
                return {'protocol_version': 3, 'command': 'update', 'identity': self.identity, 'artifact': None}
            return await actual(self, request, deadline)
        monkeypatch.setattr(LearningProcess, 'request', bad)
    elif failure == 'commit':
        actual_commit = module.commit
        def bad_commit(*args, **kwargs):
            if kwargs['generation'] == 1:
                raise OSError('injected generation publication failure')
            return actual_commit(*args, **kwargs)
        monkeypatch.setattr(module, 'commit', bad_commit)
    else:
        actual_update = StructuredLearningProcess.update
        async def changed(self, *args, **kwargs):
            value = await actual_update(self, *args, **kwargs)
            report = parent / 'collections/batch-000000/report.json'
            report.write_bytes(report.read_bytes() + b' ')
            return value
        monkeypatch.setattr(StructuredLearningProcess, 'update', changed)
    with pytest.raises((ContractError, OSError)):
        asyncio.run(_train_component_ppo(loaded, collector, parent))
    state = json.loads((parent / 'run.json').read_text())
    assert state['status'] == 'failed' and state['quarantine'] and state['updates'] == 0
    assert not (parent / 'generations/000001.json').exists()
    assert json.loads((parent / 'latest.json').read_text())['generation'] == 0
    assert all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert all(ledger.learner is None and not ledger.workers and not ledger.updating for ledger in ledgers)
    with pytest.raises(ContractError):
        validate_component_parent(parent, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)


def test_whole_parent_historical_validation_needs_no_torch_current_runtime_or_hardware(completed_component):
    code = '''
import importlib.abc, importlib.metadata, sys, time
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('historical validator imported Torch')
sys.meta_path.insert(0, NoTorch())
def forbidden(*args, **kwargs):
    raise AssertionError('historical validator queried current metadata/hardware/producer')
importlib.metadata.version = forbidden
from gf_models import structured_artifacts, torch_runtime
structured_artifacts.runtime_metadata = forbidden
structured_artifacts.producer_identity = forbidden
torch_runtime.hardware = forbidden
from gf_training.structured_ppo_source import validate_component_parent
state, records = validate_component_parent(Path(sys.argv[1]), max_owned_bytes=2*1024**3, deadline=time.monotonic()+60)
assert state['purpose'] == 'component' and state['synthetic_capacity'] is False and len(records) == 3
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(completed_component)],
        capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('hand', ['direct', 'deepsets'])
def test_other_hand_encoders_use_real_component_child_and_original_full_batch(tmp_path, monkeypatch, hand):
    loaded = configuration(tmp_path, hand=hand, max_updates=1)
    collector, scripts, workers, ledgers = component_http(monkeypatch, loaded)
    result = asyncio.run(_train_component_ppo(loaded, collector, tmp_path / 'parent'))
    assert result['status'] == 'complete' and result['updates'] == 1 and result['accepted_decisions'] == 32
    state, _ = validate_component_parent(tmp_path / 'parent', max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    assert state['effective_config']['model']['hand']['kind'] == hand
    assert len(scripts) == 8 and len(workers) == 32
    assert all(worker.process.poll() is not None for worker in workers)
    assert all(ledger.learner is None and not ledger.workers for ledger in ledgers)


def test_unavailable_cuda_learner_fails_before_any_collection_without_fallback(tmp_path, monkeypatch):
    import gf_training.structured_ppo as module
    loaded = configuration(tmp_path, learning_device='cuda:0')
    config = replace(loaded.config, model=replace(loaded.config.model, cuda_visible_devices=''))
    from gf_cli.config import LoadedConfig
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    actual, children, called = module.make_child, [], []
    def created(*args, **kwargs):
        value = actual(*args, **kwargs)
        children.append(value)
        return value
    async def forbidden(*args):
        called.append(1)
        raise AssertionError('collection began before unavailable CUDA failure')
    monkeypatch.setattr(module, 'make_child', created)
    with pytest.raises(RuntimeError, match='numerical child failed'):
        asyncio.run(_train_component_ppo(loaded, forbidden, tmp_path / 'parent'))
    assert called == [] and len(children) == 1
    child, ledger = children[0]
    assert child.process.returncode is not None and ledger.learner is None
    assert not list((tmp_path / 'parent/collections').iterdir())
    assert json.loads((tmp_path / 'parent/run.json').read_text())['status'] == 'failed'


def test_missing_native_structured_capability_precedes_numeric_or_collection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import gf_training.native_backend as backend
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    from gf_cli.config import LoadedConfig
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    monkeypatch.setattr(backend, 'load_native', lambda: SimpleNamespace(build_info=dict))
    called = []
    async def forbidden(*args):
        called.append(1)
        raise AssertionError('collection began without native capability')
    with pytest.raises(ContractError, match='structured-v1'):
        asyncio.run(_train_component_ppo(loaded, forbidden, tmp_path / 'parent'))
    assert called == [] and not (tmp_path / 'parent').exists()


@pytest.fixture(scope='module')
def paused_component(tmp_path_factory):
    root = tmp_path_factory.mktemp('paused-component-template')
    loaded = configuration(root)
    with pytest.MonkeyPatch.context() as patch:
        collector, _, workers, ledgers = component_http(patch, loaded)
        result = asyncio.run(_train_component_ppo(loaded, collector, root / 'parent', pause_after_update=1))
        assert result['status'] == 'paused' and result['updates'] == 1
        assert all(worker.process.poll() is not None for worker in workers)
        assert all(ledger.learner is None for ledger in ledgers)
    return root / 'parent', loaded


@pytest.mark.parametrize('field', ['candidate_chunk', 'managed_budget_bytes'])
def test_rehashed_recorded_learner_execution_cannot_override_current_config(paused_component, tmp_path, monkeypatch, field):
    import gf_training.structured_ppo as module
    from gf_models.arrays import sha256
    from gf_models.structured_binding import input_binding
    from gf_training.structured_lineage import resume_identity
    original, loaded = paused_component
    parent = clone_component_parent(original, tmp_path / 'parent')
    state = json.loads((parent / 'run.json').read_text())
    state['execution']['learner']['binding'][field] += 1 if field == 'candidate_chunk' else -1
    state['resume_identity'] = resume_identity(state['config_sha256'], state['identity'], state['execution'],
        input_binding(loaded.config.actions.action_contract, 'component', False))
    registry = json.loads((parent / 'artifact-registry.json').read_text())
    previous = None
    for number in range(2):
        artifact = parent / 'artifacts' / f'{number:06}'
        resume = json.loads((artifact / 'resume.json').read_text())
        resume.update(binding=state['execution']['learner']['binding'], resume_identity=state['resume_identity'])
        atomic_json(artifact / 'resume.json', resume)
        path = parent / 'generations' / f'{number:06}.json'
        record = json.loads(path.read_text())
        record['parent_sha256'] = previous
        record['resume_sha256'] = record['artifact_files']['resume.json'] = sha256(artifact / 'resume.json')
        registry['generations'][str(number)]['original_files']['resume.json'] = record['resume_sha256']
        if number:
            update_path = parent / 'updates/000001.json'
            update = json.loads(update_path.read_text())
            update['parent_sha256'] = previous
            atomic_json(update_path, update)
            record['update_sha256'] = sha256(update_path)
        atomic_json(path, record)
        previous = sha256(path)
    state['generation_sha256'] = previous
    atomic_json(parent / 'run.json', state)
    atomic_json(parent / 'latest.json', {'generation': 1, 'sha256': previous})
    atomic_json(parent / 'artifact-registry.json', registry)
    # Its recorded historical declarations are coherent; ordinary active resume
    # must additionally match the binding derived from the unchanged Config.
    validate_component_parent(parent, require_complete=False, max_owned_bytes=2 * 1024**3, deadline=time.monotonic() + 60)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('child created before execution mismatch rejection')
    monkeypatch.setattr(module, 'make_child', forbidden)
    async def forbidden_collector(*args):
        raise AssertionError('collection began after execution mismatch')
    with pytest.raises(ContractError, match='execution'):
        asyncio.run(_train_component_ppo(loaded, forbidden_collector, parent, mode='resume'))
    assert calls == [] and json.loads((parent / 'run.json').read_text())['status'] == 'paused'
