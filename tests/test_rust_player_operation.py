"""Fixed-operation configuration and actual local owners; no training execution."""
from __future__ import annotations

import fcntl
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli import rust_player_entry as entry
from gf_cli.hpo_config import campaign_config, materialize_hpo_run
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
from gf_training.hpo_config import effective_settings
from gf_training.hpo_gpu_lease import JobContext
from tests.test_rust_player_config import ROOT, RUN_ID, campaign_raw

EXPERIMENT = '00000000-0000-0000-0000-000000000018'


def campaign(tmp_path):
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    raw['storage'].update(parent_root=str(tmp_path / 'runs'), numeric_root=str(tmp_path / 'runs'))
    return campaign_config(raw, origin=ROOT)


def materialized(tmp_path):
    loaded = campaign(tmp_path)
    selected = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    return materialize_hpo_run(loaded, selected, learning_seed=17, run_id=RUN_ID)


def context(op='1' * 32):
    return JobContext(EXPERIMENT, op, 'training', RUN_ID, None)


def assert_locked(path):
    with path.open('r+b') as competitor, pytest.raises(BlockingIOError):
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def assert_unlocked(path):
    with path.open('r+b') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_hpo_factory_owns_both_original_locks_and_fixed_records(tmp_path):
    loaded = materialized(tmp_path)
    namespace = tmp_path / 'operation-control'
    parent = tmp_path / 'runs' / RUN_ID
    with entry.prepare_hpo_operation(loaded, context(), namespace, mode='fresh', pause_after_update=1) as prepared:
        assert prepared.operation['operation_type'] == 'hpo-train'
        assert prepared.operation['run_id'] == RUN_ID
        assert prepared.operation['expected_context'] == asdict(context())
        assert prepared.operation['execution_request'] == {'mode': 'fresh', 'pause_after_update': 1}
        assert prepared.launch['python_executable'] == str(Path(sys.executable).absolute())
        assert [lock['lock_kind'] for lock in prepared.launch['locks']] == ['campaign', 'parent']
        for lock in prepared.launch['locks']:
            actual = os.fstat(lock['fd'])
            assert (actual.st_dev, actual.st_ino) == (lock['device'], lock['inode'])
        assert_locked(namespace / 'lock')
        assert_locked(parent / '.lock')
        assert not (namespace / 'campaign.json').exists()
        assert prepared.clock['operation_sha256'] == digest(prepared.operation)
        assert json.loads(prepared.provenance_path.read_bytes()) == {'inputs': list(loaded.inputs)}
        assert prepared.loaded.sha256 == loaded.sha256
        assert prepared.loaded.inputs == ({'path': prepared.operation['config_path'], 'sha256': loaded.sha256},)
        prepared.require()
    assert_unlocked(namespace / 'lock')
    assert_unlocked(parent / '.lock')
    with pytest.raises(ContractError):
        prepared.require()


def test_hpo_anchor_precedes_real_store_and_parent_ownership(tmp_path, monkeypatch):
    from gf_cli import hpo_storage
    from gf_training import lineage
    original = entry._capture_clock
    seen = []
    def capture():
        seen.append('anchor')
        return original()
    original_store = hpo_storage.CampaignStore.__init__
    def store(self, *args, **kwargs):
        assert seen == ['anchor']
        seen.append('store')
        original_store(self, *args, **kwargs)
    original_parent = lineage.run_lock
    def parent(*args, **kwargs):
        assert seen == ['anchor', 'store']
        seen.append('parent')
        return original_parent(*args, **kwargs)
    monkeypatch.setattr(entry, '_capture_clock', capture)
    monkeypatch.setattr(hpo_storage.CampaignStore, '__init__', store)
    monkeypatch.setattr(lineage, 'run_lock', parent)
    with entry.prepare_hpo_operation(materialized(tmp_path), context(), tmp_path / 'control', mode='fresh'):
        assert seen == ['anchor', 'store', 'parent']


def test_hpo_failed_exec_seals_both_actual_ofds_and_closes_resources(tmp_path, monkeypatch):
    from gf_common.rust_player import LINUX_F_GET_SEALS, LINUX_REQUIRED_LAUNCH_SEALS
    from gf_simulator_client import rust_player_companion
    descriptors = []
    def refuse(executable, argv, env):
        launch_fd = int(argv[2])
        row = json.loads(os.pread(launch_fd, 65536, 0))
        assert fcntl.fcntl(launch_fd, LINUX_F_GET_SEALS) & LINUX_REQUIRED_LAUNCH_SEALS == LINUX_REQUIRED_LAUNCH_SEALS
        assert {lock['lock_kind'] for lock in row['locks']} == {'campaign', 'parent'}
        descriptors.extend([executable, launch_fd, *(lock['fd'] for lock in row['locks'])])
        assert all(os.get_inheritable(fd) for fd in descriptors[1:])
        assert_locked(tmp_path / 'control/lock')
        assert_locked(tmp_path / 'runs' / RUN_ID / '.lock')
        raise OSError('unit-only refused exec')
    monkeypatch.setattr(rust_player_companion.os, 'execve', refuse)
    with pytest.raises(OSError, match='unit-only'), entry.prepare_hpo_operation(
            materialized(tmp_path), context(), tmp_path / 'control', mode='fresh') as prepared:
        prepared.execute()
    assert len(descriptors) == 4
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert_unlocked(tmp_path / 'control/lock')
    assert_unlocked(tmp_path / 'runs' / RUN_ID / '.lock')


@pytest.mark.parametrize('fault', ['replace', 'launch', 'companion', 'config', 'thread'])
def test_hpo_factory_facts_cannot_be_reconstructed_or_retargeted(tmp_path, fault):
    import threading
    with entry.prepare_hpo_operation(materialized(tmp_path), context(), tmp_path / 'control', mode='fresh') as prepared:
        target = prepared
        if fault == 'replace':
            target = replace(prepared)
        elif fault == 'launch':
            prepared.launch['python_executable'] = '/foreign/python'
        elif fault == 'companion':
            prepared.companion.manifest['executable_sha256'] = '0' * 64
        elif fault == 'config':
            Path(prepared.operation['config_path']).write_text('{}')
        if fault == 'thread':
            errors = []
            def foreign():
                try:
                    target.require()
                except ContractError as error:
                    errors.append(error)
            thread = threading.Thread(target=foreign)
            thread.start(); thread.join()
            assert len(errors) == 1
        else:
            with pytest.raises(ContractError):
                target.require()


def paused_hpo_data(prepared):
    # Launcher DATA only. This cannot pass the actual RustResumeAudit or admit CUDA.
    from tests.test_rust_player_public_entry import _paused_launcher_data
    state = _paused_launcher_data(prepared)
    state.update(run_variant='hpo-ppo-owned-rust-player-v1', model_version='4',
        hpo_spec_sha256=digest(prepared.loaded.effective['hpo_spec']), used_decisions=1,
        collected_decisions=1, stop_reason='pause_after_update')
    (prepared.ownership.directory / 'run.json').write_text(canonical(state))
    return state


def test_hpo_resume_reuses_namespace_origin_and_every_clock_bit(tmp_path, monkeypatch):
    loaded = materialized(tmp_path)
    namespace = tmp_path / 'control'
    with entry.prepare_hpo_operation(loaded, context(), namespace, mode='fresh', pause_after_update=1) as first:
        paused_hpo_data(first)
        clock, operation = first.clock, first.operation
        identity = os.stat(namespace / 'lock')
    def forbidden():
        raise AssertionError('resume recaptured original clock')
    monkeypatch.setattr(entry, '_capture_clock', forbidden)
    with entry.prepare_hpo_operation(loaded, context('2' * 32), namespace, mode='resume') as resumed:
        assert resumed.clock['clock_binding'] == clock['clock_binding']
        assert resumed.operation['deadlines'] == operation['deadlines']
        assert resumed.operation['operation_id'] != operation['operation_id']
        assert resumed.operation['expected_context'] == asdict(context('2' * 32))
        assert os.stat(namespace / 'lock').st_ino == identity.st_ino


def test_hpo_resume_requires_context_distinct_from_all_previous_invocations(tmp_path):
    loaded = materialized(tmp_path)
    namespace = tmp_path / 'control'
    with entry.prepare_hpo_operation(loaded, context(), namespace, mode='fresh') as first:
        state = paused_hpo_data(first)
        previous = dict(next(iter(state['invocations'].values())))
        previous.update(operation_path=str(first.operation_path.parent / f'operation-{"2" * 32}.json'),
            mode='resume', execution_request={'mode': 'resume', 'pause_after_update': None})
        state['invocations']['3' * 32] = previous
        (first.ownership.directory / 'run.json').write_text(canonical(state))
    with pytest.raises(ContractError), entry.prepare_hpo_operation(
            loaded, context('2' * 32), namespace, mode='resume'):
        raise AssertionError('historical operation context reused')


@pytest.mark.parametrize('fault', ['namespace', 'config-inputs', 'boot', 'expired', 'failed', 'run-id'])
def test_hpo_resume_refuses_changed_origin_without_new_anchor(tmp_path, monkeypatch, fault):
    loaded = materialized(tmp_path)
    namespace = tmp_path / 'control'
    with entry.prepare_hpo_operation(loaded, context(), namespace, mode='fresh') as first:
        state = paused_hpo_data(first)
        if fault == 'failed':
            state['status'] = 'failed'
            (first.ownership.directory / 'run.json').write_text(canonical(state))
    if fault == 'namespace':
        (namespace / 'operation-control.json').unlink()
    elif fault == 'config-inputs':
        loaded = replace(loaded, inputs=(*loaded.inputs, {'path': '/foreign/config', 'sha256': '0' * 64}))
    elif fault == 'boot':
        monkeypatch.setattr(entry, '_boot_id', lambda: 'foreign')
    elif fault == 'expired':
        monkeypatch.setattr(entry.time, 'monotonic', lambda: 1e100)
    actual_context = context('2' * 32)
    if fault == 'run-id':
        actual_context = replace(actual_context, run_id=EXPERIMENT)
    def forbidden():
        raise AssertionError('invalid resume recaptured clock')
    monkeypatch.setattr(entry, '_capture_clock', forbidden)
    with pytest.raises((ContractError, OSError)), entry.prepare_hpo_operation(
            loaded, actual_context, namespace, mode='resume'):
        raise AssertionError('invalid resume accepted')


def test_operation_tool_loads_real_config_and_configured_seed_inside_command(tmp_path):
    from tools import rust_player_operation as tool
    path = ROOT / 'configs/hpo/mlp-shared-inference-scaling.toml'
    loaded, run, job_context, namespace = tool.operation_inputs(path, run_id=RUN_ID,
        experiment_id=EXPERIMENT, learning_seed=17)
    assert loaded.config.schema_version == 5
    assert run.config.hpo_spec['schema_version'] == 5
    assert run.config.training.seeds == (17,)
    assert run.config.hpo_spec['training_recipe']['collection_games'] == 32
    assert run.config.training.max_updates == 2
    assert run.config.hpo_spec['decision_budget'] == 10000
    assert job_context.run_id == RUN_ID and job_context.campaign_id == EXPERIMENT
    assert namespace != Path(run.config.run.output_dir) / RUN_ID
    assert not namespace.exists()


@pytest.mark.parametrize('field,value', [('learning_seed', 18), ('run_id', 'bad'), ('experiment_id', 'bad')])
def test_operation_tool_refuses_undeclared_seed_and_noncanonical_ids(field, value):
    from tools import rust_player_operation as tool
    options = {'run_id': RUN_ID, 'experiment_id': EXPERIMENT, 'learning_seed': 17}
    options[field] = value
    with pytest.raises(ContractError):
        tool.operation_inputs(ROOT / 'configs/hpo/mlp-shared-inference.toml', **options)


def native_config_path(tmp_path, *, post_host=True):
    import re
    text = (ROOT / 'configs/hpo/mlp-shared-inference.toml').read_text()
    text = text.replace('schema_version = 5', 'schema_version = 6', 1)
    text = text.replace('"../experiments/mlp-shared-inference-base.toml"',
        json.dumps(str(ROOT / 'configs/experiments/mlp-shared-inference-base.toml')))
    before, rest = text.split('[runtime]\n', 1)
    _old, after = rest.split('[inference]\n', 1)
    text = before + '''[runtime]
mode = "rust-player-runtime-v1"
scheduler = "tokio"
game_concurrency = 3
player_worker_threads = 7
max_batch_size = 3
formation_wait_ms = 0.0
\n[inference]\n''' + after
    if post_host:
        text = text.replace('[operating]\n', '[operating]\npost_host_seconds = 30.0\n')
    for name in ('parent_root', 'numeric_root'):
        text = re.sub(rf'^{name} = .*$', f'{name} = {json.dumps(str(tmp_path / "runs"))}', text, flags=re.MULTILINE)
    path = tmp_path / 'hpo.toml'
    path.write_text(text)
    return path


def test_operation_command_loads_native_config_then_execs_same_fixed_handler(tmp_path, monkeypatch):
    from gf_simulator_client import rust_player_companion
    from tools import rust_player_operation as tool
    path = native_config_path(tmp_path)
    launches = []
    def refuse(executable, argv, env):
        launch = json.loads(os.pread(int(argv[2]), 65536, 0))
        launches.append(launch)
        operation = json.loads(Path(launch['operation_path']).read_bytes())
        assert operation['operation_type'] == 'hpo-train'
        assert operation['expected_context']['campaign_id'] == EXPERIMENT
        snapshot = json.loads(Path(operation['config_path']).read_bytes())
        assert snapshot['training']['seeds'] == [17]
        assert snapshot['hpo_spec']['schema_version'] == 6
        namespace = tmp_path / 'runs/.operation-control' / EXPERIMENT / RUN_ID
        invocation = json.loads((namespace / 'invocations' / f'{operation["operation_id"]}.json').read_bytes())
        assert invocation['context'] == operation['expected_context']
        assert invocation['launcher_sha256'] == __import__('hashlib').sha256(Path(tool.__file__).read_bytes()).hexdigest()
        raise OSError('unit-only command exec')
    monkeypatch.setattr(rust_player_companion.os, 'execve', refuse)
    with pytest.raises(OSError, match='unit-only command exec'):
        tool.main(['--config', str(path), '--run-id', RUN_ID, '--experiment-id', EXPERIMENT,
            '--learning-seed', '17', '--mode', 'fresh', '--pause-after-update', '1'])
    assert len(launches) == 1
    assert_unlocked(tmp_path / 'runs' / RUN_ID / '.lock')
    assert_unlocked(tmp_path / 'runs/.operation-control' / EXPERIMENT / RUN_ID / 'lock')


@pytest.mark.parametrize('preset', ['owned-gru-smoke.toml', 'owned-mlp-smoke.toml'])
def test_operation_tool_refuses_other_models_and_historical_schema(preset):
    from tools import rust_player_operation as tool
    with pytest.raises(ContractError, match='only shared5/native6 MLP'):
        tool.operation_inputs(ROOT / 'configs/hpo' / preset, run_id=RUN_ID,
            experiment_id=EXPERIMENT, learning_seed=17)


def test_operation_tool_never_invents_native_post_host_allocation(tmp_path):
    from tools import rust_player_operation as tool
    with pytest.raises(ContractError, match='explicit post_host_seconds'):
        tool.operation_inputs(native_config_path(tmp_path, post_host=False), run_id=RUN_ID,
            experiment_id=EXPERIMENT, learning_seed=17)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('changed', ['directory', 'lock'])
def test_hpo_resume_refuses_copied_namespace_instead_of_original_identity(tmp_path, changed):
    import shutil
    loaded = materialized(tmp_path)
    namespace = tmp_path / 'control'
    with entry.prepare_hpo_operation(loaded, context(), namespace, mode='fresh') as first:
        paused_hpo_data(first)
    if changed == 'directory':
        namespace.rename(tmp_path / 'old-control')
        shutil.copytree(tmp_path / 'old-control', namespace)
    else:
        (namespace / 'lock').rename(namespace / 'old-lock')
        (namespace / 'lock').touch(mode=0o600)
    with pytest.raises(ContractError), entry.prepare_hpo_operation(
            loaded, context('2' * 32), namespace, mode='resume'):
        raise AssertionError('substitute namespace accepted')


def test_hpo_factory_rejects_path_operation_id_before_ownership(tmp_path):
    with pytest.raises(ContractError), entry.prepare_hpo_operation(
            materialized(tmp_path), context('../foreign'), tmp_path / 'control', mode='fresh'):
        raise AssertionError('operation path component accepted')
    assert not (tmp_path / 'control').exists()


def test_hpo_factory_refuses_missing_companion_before_anchor_and_namespace(tmp_path, monkeypatch):
    from gf_simulator_client import rust_player_companion
    def missing(_identity):
        raise ContractError('unit-only missing companion')
    def forbidden():
        raise AssertionError('missing companion captured a clock')
    monkeypatch.setattr(rust_player_companion, 'discover_companion', missing)
    monkeypatch.setattr(entry, '_capture_clock', forbidden)
    with pytest.raises(ContractError, match='missing companion'), entry.prepare_hpo_operation(
            materialized(tmp_path), context(), tmp_path / 'control', mode='fresh'):
        raise AssertionError('missing companion accepted')
    assert not (tmp_path / 'control').exists()


@pytest.mark.parametrize('failure', ['none', 'complete', 'train', 'completion'])
def test_legacy_operation_uses_real_routine_boundary_and_exact_job_ledger(tmp_path, monkeypatch, failure):
    """Only admission/training/collection are unit doubles; Store remains actual."""
    import asyncio
    from contextlib import contextmanager

    from gf_cli import card_instance, hpo_config
    from gf_training import hpo_gpu_lease, hpo_ppo
    from tools import rust_player_operation as tool
    raw = campaign_raw()
    raw.update(schema_version=5,
        runtime={'mode': 'shared-mlp-v1', 'max_batch_size': 3, 'formation_wait_ms': 1.,
            'cache_entries': 6, 'cache_weight_bytes': 134217728})
    raw['storage'].update(parent_root=str(tmp_path / 'runs'), numeric_root=str(tmp_path / 'runs'))
    loaded = hpo_config.campaign_config(raw, origin=ROOT)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    namespace = tmp_path / 'control'
    ledger = object()
    events = []
    class UnitJob:
        pass
    job = UnitJob()
    job.ledger = ledger
    @contextmanager
    def admit(actual_context, *, global_cuda_bytes, max_workers):
        assert actual_context == context()
        assert global_cuda_bytes == loaded.config.operating.managed_cuda_bytes
        assert max_workers == loaded.config.operating.max_participant_workers
        assert_locked(namespace / 'lock')
        events.append('admitted')
        try:
            yield job
        finally:
            assert_locked(namespace / 'lock')
            events.append('job-closed')
    class UnitCompletion:
        def close(self):
            assert_locked(namespace / 'lock')
            events.append('completion-closed')
            if failure == 'completion':
                raise RuntimeError('unit completion close failure')
    async def collect(actual_run, plan, actual_ledger):
        assert actual_run is run and plan == 'unit-plan' and actual_ledger is ledger
        events.append('collected')
        return 'unit-collection'
    async def train(actual_run, collector, directory, *, gpu_job, mode, pause_after_update):
        assert actual_run is run and gpu_job is job
        assert directory == tmp_path / 'runs' / RUN_ID
        assert mode == 'fresh' and pause_after_update == 1
        with pytest.raises(ContractError, match='exact caller job ledger'):
            await collector('unit-plan', object())
        assert await collector('unit-plan', ledger) == 'unit-collection'
        events.append('trained')
        if failure == 'train':
            raise RuntimeError('unit training failure')
        receipt = hpo_gpu_lease.TerminationReceipt(1, 'unit-data-only', 'ticket', 'a' * 64,
            'b' * 64, 'boot', 1., ())
        complete = failure in ('complete', 'completion')
        return hpo_ppo.HpoRunResult({'status': 'complete' if complete else 'incomplete', 'updates': 2,
            'used_decisions': 10000 if complete else 123,
            'stop_reason': 'decision_budget' if complete else 'update_cap'}, receipt,
            UnitCompletion() if complete else None)
    monkeypatch.setattr(hpo_gpu_lease, 'admit_gpu_job', admit)
    monkeypatch.setattr(hpo_ppo, 'train_hpo_ppo', train)
    monkeypatch.setattr(card_instance, 'collect_hpo_ppo', collect)
    invoke = tool._legacy_operation(loaded, run, context(), namespace, mode='fresh', pause_after_update=1)
    if failure in ('train', 'completion'):
        with pytest.raises(RuntimeError, match='unit') as raised:
            asyncio.run(invoke)
        if failure == 'completion':
            assert isinstance(raised.value.hpo_completion_owner, UnitCompletion)
    else:
        result = asyncio.run(invoke)
        assert result['status'] == ('complete' if failure == 'complete' else 'incomplete')
        assert result['used_decisions'] == (10000 if failure == 'complete' else 123)
        assert 'objective' not in result
    assert events == ['admitted', 'collected', 'trained'] + (['completion-closed'] if failure in ('complete', 'completion') else []) + ['job-closed']
    assert_unlocked(namespace / 'lock')
    assert not (namespace / 'campaign.json').exists()
