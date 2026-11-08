"""Synthetic finalized collections exercise whole-run commits and failure boundaries."""
import asyncio
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from gf_cli.config import LoadedConfig
from gf_common.domain import ContractError, digest
from gf_common.schedule import serialize_schedule
from gf_training.collection import training_groups, validation_groups
from gf_training.journal import atomic_json
from gf_training.lineage import checkpoint_path, qualify_mlp, read_json, run_lock, validate_chain
from gf_training.ppo_run import train_ppo
from tests.ppo_support import configuration as configuration  # noqa: PLC0414 - compatibility re-export
from tests.ppo_support import synthetic_collector as synthetic_collector  # noqa: PLC0414 - compatibility re-export


def test_explicit_source_provenance_still_rejects_changed_normative_documents(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import gf_training.ppo_run as module
    from gf_common.domain import REVIEWED_SPEC_HASHES, SIMULATOR_REVISION
    loaded = configuration(tmp_path)
    source = tmp_path / 'explicit-source'
    for name in REVIEWED_SPEC_HASHES:
        document = source / name
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_bytes(b'deliberately changed normative document identity')
    config = replace(loaded.config, server=replace(loaded.config.server, source_dir=str(source)))
    def revision(command, **kwargs):
        assert command == ['git', '-C', str(source), 'rev-parse', 'HEAD']
        return SimpleNamespace(stdout=SIMULATOR_REVISION + '\n')
    monkeypatch.setattr(module.subprocess, 'run', revision)
    with pytest.raises(ContractError, match='normative source hashes differ'):
        module.integration_identity(config, float('inf'))
    assert all((source / name).read_bytes() == b'deliberately changed normative document identity' for name in REVIEWED_SPEC_HASHES)


def test_explicit_learner_generation_zero_selfplay_and_namespaces():
    groups = training_groups(17, 0, 'same', ('same',), np.random.default_rng(17))
    assert len(groups) == 8 and sum(len(group.episodes) for group in groups) == 16
    assert len({group.seed for group in groups}) == 8
    assert groups[-1].episodes[0].policies == ('same', 'same')
    assert [spec.learner_viewer for spec in groups[-1].episodes] == [1, 2]
    assert groups[-1].episodes[0].sampling_seeds == groups[-1].episodes[1].sampling_seeds[::-1]
    historical = serialize_schedule(groups, 3)
    assert all('learner_viewer' not in spec for group in historical for spec in group['episodes'])
    assert serialize_schedule(groups, 4)[-1]['episodes'][1]['learner_viewer'] == 2
    validation = validation_groups(17, 'same')
    assert len(validation) == 32 and validation[15].episodes[0].policies[1] == 'random'
    assert validation[16].episodes[0].policies[1] == 'heuristic'
    assert not {group.seed for group in groups} & {group.seed for group in validation}


def test_two_update_pause_resume_and_formal_qualification(tmp_path, make_input):
    pytest.importorskip('torch')
    loaded = configuration(tmp_path)
    collector = synthetic_collector(make_input)
    path = tmp_path / 'run'
    result = asyncio.run(train_ppo(loaded, collector, path, pause_after_update=1))
    assert result['status'] == 'paused' and result['updates'] == 1
    state, records = validate_chain(path, require_complete=False)
    first_record = (path / 'generations' / '000001.json').read_bytes()
    with pytest.raises(ContractError, match='completed'):
        qualify_mlp(checkpoint_path(path, 1), records[-1]['inference_sha256'])
    result = asyncio.run(train_ppo(loaded, collector, path, mode='resume'))
    assert result['status'] == 'complete' and result['updates'] == 2
    state, records = validate_chain(path)
    assert (path / 'generations' / '000001.json').read_bytes() == first_record
    assert checkpoint_path(path, 0).joinpath('weights.npz').exists()
    assert not checkpoint_path(path, 1).joinpath('weights.npz').exists()
    assert checkpoint_path(path, 1).joinpath('inference.json').exists()
    qualify_mlp(checkpoint_path(path, 2), records[-1]['inference_sha256'])
    with pytest.raises(ContractError):
        asyncio.run(train_ppo(loaded, collector, path, mode='resume'))
    with pytest.raises(ContractError, match='already owned'), run_lock(path), run_lock(path):
        pass
    state['status'] = 'failed'
    atomic_json(path / 'run.json', state)
    with pytest.raises(ContractError):
        qualify_mlp(checkpoint_path(path, 2), records[-1]['inference_sha256'])


def test_failed_later_collection_invalidates_earlier_generation(tmp_path, make_input):
    pytest.importorskip('torch')
    loaded = configuration(tmp_path)
    collector = synthetic_collector(make_input)
    async def failure(plan):
        if plan.linkage['collection_generation'] == 1:
            raise RuntimeError('synthetic collection failure')
        return await collector(plan)
    path = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='synthetic collection'):
        asyncio.run(train_ppo(loaded, failure, path))
    state = read_json(path / 'run.json')
    assert state['status'] == 'failed' and state['updates'] == 1
    with pytest.raises(ContractError):
        validate_chain(path)
    assert not checkpoint_path(path, 2).exists()


def test_resume_rejects_mutated_bytes_clock_and_config_before_collection(tmp_path, make_input):
    pytest.importorskip('torch')
    loaded = configuration(tmp_path)
    path = tmp_path / 'paused'
    asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path, pause_after_update=1))
    checkpoint_path(path, 1).joinpath('resume.npz').write_bytes(b'corrupt')
    async def forbidden(plan):
        pytest.fail('resume must fail before collection')
    with pytest.raises(ContractError):
        asyncio.run(train_ppo(loaded, forbidden, path, mode='resume'))
    assert read_json(path / 'run.json')['status'] == 'paused'


def test_capacity_rejects_symlinks(tmp_path):
    from gf_training.ppo_run import owned_bytes
    (tmp_path / 'outside').symlink_to('/tmp')
    with pytest.raises(ContractError, match='symlink'):
        owned_bytes(tmp_path)


def test_cleanup_failure_never_publishes_completed_or_paused_parent(tmp_path, make_input, monkeypatch):
    from gf_training.ppo_run import LearningProcess
    loaded = configuration(tmp_path, max_updates=1)
    original = LearningProcess.close
    async def failed_close(self, deadline):
        await original(self, deadline)
        raise RuntimeError('synthetic cleanup failure')
    monkeypatch.setattr(LearningProcess, 'close', failed_close)
    path = tmp_path / 'cleanup-failed'
    with pytest.raises(RuntimeError, match='cleanup'):
        asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path, pause_after_update=1))
    state = read_json(path / 'run.json')
    assert state['status'] == 'failed' and state['updates'] == 1
    with pytest.raises(ContractError):
        validate_chain(path)


def test_device_failure_precedes_any_collection_and_invalid_pause_precedes_output(tmp_path, monkeypatch):
    from gf_training.ppo_run import LearningProcess
    loaded = configuration(tmp_path)
    async def forbidden(plan):
        pytest.fail('unavailable learner device cannot collect')
    with pytest.raises(ContractError, match='pause'):
        asyncio.run(train_ppo(loaded, forbidden, tmp_path / 'invalid', pause_after_update=0))
    assert not (tmp_path / 'invalid').exists()
    async def unavailable(self, request, deadline):
        raise ContractError('configured learner device unavailable')
    monkeypatch.setattr(LearningProcess, 'request', unavailable)
    path = tmp_path / 'device-failed'
    with pytest.raises(ContractError, match='unavailable'):
        asyncio.run(train_ppo(loaded, forbidden, path))
    assert read_json(path / 'run.json')['status'] == 'failed'
    assert not list((path / 'collections').iterdir())


def test_remaining_capacity_stops_new_batch_after_healthy_update(tmp_path, make_input, monkeypatch):
    import gf_training.ppo_run as module
    loaded = configuration(tmp_path)
    training = replace(loaded.config.training, max_owned_bytes=100_000_000, admission_reservation_bytes=1_000_000)
    config = replace(loaded.config, training=training)
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    original = module.owned_bytes
    def capacity(directory, deadline=float("inf")):
        state = read_json(directory / 'run.json')
        return 99_500_000 if state['updates'] else original(directory)
    monkeypatch.setattr(module, 'owned_bytes', capacity)
    path = tmp_path / 'capacity'
    result = asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path))
    assert result['status'] == 'complete' and result['updates'] == 1
    validate_chain(path)


def test_resume_backend_code_and_expired_original_deadline_rejected(tmp_path, make_input, monkeypatch):
    import gf_training.ppo_run as module
    loaded = configuration(tmp_path)
    path = tmp_path / 'paused-guards'
    asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path, pause_after_update=1))
    async def forbidden(plan):
        pytest.fail('invalid resume must not collect')
    config = replace(loaded.config, training=replace(loaded.config.training, device='cuda:0'))
    changed = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(changed, forbidden, path, mode='resume'))
    original_identity = module.source_identity
    monkeypatch.setattr(module, 'source_identity', lambda deadline: {'changed': True})
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(loaded, forbidden, path, mode='resume'))
    monkeypatch.setattr(module, 'source_identity', original_identity)
    state = read_json(path / 'run.json')
    monkeypatch.setattr(module.time, 'time', lambda: state['wall_deadline_unix'] + 1)
    with pytest.raises((ContractError, TimeoutError)):
        asyncio.run(train_ppo(loaded, forbidden, path, mode='resume'))
    assert read_json(path / 'run.json')['status'] == 'paused'


def test_retention_records_intent_before_deleting_nonperiodic_predecessor(tmp_path, monkeypatch):
    from gf_training.lineage import retain
    registry = {}
    for number in (0, 10, 11):
        path = checkpoint_path(tmp_path, number)
        path.mkdir(parents=True)
        for name in ('inference.json', 'weights.npz', 'resume.npz', 'resume.json'):
            (path / name).write_text('component fixture')
        registry[str(number)] = {'status': 'retained', 'inference_sha256': 'a' * 64}
    atomic_json(tmp_path / 'artifact-registry.json', registry)
    original = Path.unlink
    def guarded_unlink(path, *args, **kwargs):
        assert read_json(tmp_path / 'artifact-registry.json')['11']['deletion_intent'] is True
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', guarded_unlink)
    retain(tmp_path, 12)
    retain(tmp_path, 11)
    retain(tmp_path, 1)
    assert not checkpoint_path(tmp_path, 11).joinpath('weights.npz').exists()
    assert checkpoint_path(tmp_path, 11).joinpath('inference.json').exists()
    assert checkpoint_path(tmp_path, 0).joinpath('weights.npz').exists()
    assert checkpoint_path(tmp_path, 10).joinpath('weights.npz').exists()


@pytest.mark.parametrize('health_key', ['cleanup_overrun', 'stdin_close_failure', 'drain_failure', 'log_limit_exceeded'])
def test_diagnostic_child_cleanup_failure_cannot_publish_success(tmp_path, make_input, monkeypatch, health_key):
    from gf_training.ppo_run import LearningProcess
    loaded = configuration(tmp_path, max_updates=1)
    original = LearningProcess.close
    async def bad_health(self, deadline):
        await original(self, deadline)
        self.health[health_key] = True
    monkeypatch.setattr(LearningProcess, 'close', bad_health)
    path = tmp_path / 'health-failed'
    with pytest.raises(RuntimeError, match='health'):
        asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path, pause_after_update=1))
    assert read_json(path / 'run.json')['status'] == 'failed'


def test_default_projection_rejects_running_parent_but_owned_ingestion_succeeds(tmp_path, make_input):
    from gf_training.collection import OnlinePPOContext
    from gf_training.journal import project_journals
    from gf_training.q_data import load_run
    loaded = configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input)
    async def guarded(plan):
        result = await collector(plan)
        journals = list(plan.output.glob('*/player-*.jsonl'))
        with pytest.raises(ContractError):
            load_run(plan.output)
        with pytest.raises(ContractError):
            project_journals(journals)
        forged = OnlinePPOContext(tmp_path / 'guarded', plan.linkage['parent_run_id'], plan.linkage['batch_id'], 0, 'not-the-owner-token')
        with pytest.raises(ContractError, match='context'):
            load_run(plan.output, online_context=forged)
        return result
    path = tmp_path / 'guarded'
    result = asyncio.run(train_ppo(loaded, guarded, path))
    assert result['status'] == 'complete'
    collection = path / 'collections' / 'batch-000000'
    bundle = load_run(collection)
    assert len(bundle.journals) == 32
    state = read_json(path / 'run.json')
    state['status'] = 'failed'
    atomic_json(path / 'run.json', state)
    with pytest.raises(ContractError):
        project_journals(list(bundle.journals))


def test_retention_crossing_execution_deadline_cannot_publish_pause(tmp_path, make_input, monkeypatch):
    import gf_training.ppo_run as module
    loaded = configuration(tmp_path, max_updates=1)
    config = replace(loaded.config, run=replace(loaded.config.run, wall_seconds=80., admission_seconds=19., execution_seconds=20.))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    original = module.retain
    def late(directory, generation, *, deadline):
        original(directory, generation, deadline=deadline)
        # Inject a monotonic clock crossing only after all real owned work completed.
        current_mono, current_unix = module.time.monotonic(), module.time.time()
        difference = deadline + .1 - current_mono
        monkeypatch.setattr(module.time, 'monotonic', lambda: deadline + .1)
        monkeypatch.setattr(module.time, 'time', lambda: current_unix + difference)
    monkeypatch.setattr(module, 'retain', late)
    path = tmp_path / 'late-retention'
    with pytest.raises(RuntimeError, match='overrun'):
        asyncio.run(train_ppo(loaded, synthetic_collector(make_input), path, pause_after_update=1))
    assert read_json(path / 'run.json')['status'] == 'failed'


@pytest.mark.parametrize('changed_file', ['server_config', 'binary'])
def test_resume_rejects_same_path_integration_bytes_change(tmp_path, make_input, changed_file):
    loaded = configuration(tmp_path)
    server_config, binary = tmp_path / 'server.toml', tmp_path / 'binary'
    server_config.write_bytes(Path(loaded.config.server.config).read_bytes())
    binary.write_bytes(b'standalone synthetic simulator identity fixture')
    config = replace(loaded.config, server=replace(loaded.config.server, config=str(server_config), binary=str(binary), source_dir=None))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    directory = tmp_path / 'integration-paused'
    asyncio.run(train_ppo(loaded, synthetic_collector(make_input), directory, pause_after_update=1))
    target = server_config if changed_file == 'server_config' else binary
    target.write_bytes(target.read_bytes() + b'changed')
    async def forbidden(plan):
        pytest.fail('integration bytes drift must reject before collection')
    with pytest.raises(ContractError, match='mismatch'):
        asyncio.run(train_ppo(loaded, forbidden, directory, mode='resume'))
    assert read_json(directory / 'run.json')['status'] == 'paused'


def test_collection_cannot_change_frozen_simulator_identity(tmp_path, make_input):
    loaded = configuration(tmp_path, max_updates=1)
    collector = synthetic_collector(make_input)
    async def changed(plan):
        result = await collector(plan)
        manifest = read_json(plan.output / 'manifest.json')
        manifest['binary_sha256'] = '0' * 64
        atomic_json(plan.output / 'manifest.json', manifest)
        return result
    path = tmp_path / 'changed-integration'
    with pytest.raises(ContractError, match='integration'):
        asyncio.run(train_ppo(loaded, changed, path))
    assert read_json(path / 'run.json')['status'] == 'failed'
    assert not checkpoint_path(path, 1).exists()
