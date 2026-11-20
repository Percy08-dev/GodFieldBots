"""Public factory DATA/real FD controls; execution routing remains closed."""
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.config import load_config, load_rust_player_effective
from gf_cli.hpo_config import campaign_config, materialize_hpo_run
from gf_common.domain import ContractError, canonical
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
from gf_training.hpo_config import effective_settings
from tests.test_rust_player_config import NORMAL_BASE, ROOT, RUN_ID, campaign_raw, runtime


def normal(*, p=30., **overrides):
    values = {'run': {'player_runtime': runtime(), 'post_host_seconds': p}}
    for key, value in overrides.items():
        values.setdefault(key, {}).update(value)
    return load_config(NORMAL_BASE, values)


def test_new_ordinary_timing_variant_preserves_legacy_and_old_native_shapes(tmp_path):
    old = load_config(NORMAL_BASE, {'run': {'player_runtime': runtime()}})
    legacy = load_config(NORMAL_BASE)
    loaded = normal()
    assert 'post_host_seconds' not in asdict(old.config.run)
    assert 'post_host_seconds' not in asdict(legacy.config.run)
    assert type(loaded.config.run) is not type(old.config.run)
    assert asdict(loaded.config.run) == asdict(old.config.run) | {'post_host_seconds': 30.}
    assert len(loaded.config.run.player_runtime) == 6
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {'run': {'post_host_seconds': 30.}})
    path = tmp_path / 'effective.json'
    path.write_bytes(canonical(loaded.effective).encode())
    again = load_rust_player_effective(path, expected_sha256=loaded.sha256)
    assert type(again.config.run) is type(loaded.config.run)
    assert again.sha256 == loaded.sha256 and asdict(again.config) == asdict(loaded.config)
    del loaded.effective['run']['post_host_seconds']
    path.write_bytes(canonical(loaded.effective).encode())
    with pytest.raises(ContractError):
        load_rust_player_effective(path, expected_sha256=again.sha256)


@pytest.mark.parametrize('p', [None, True, 0, -1, 60., 61., float('inf'), float('nan')])
def test_public_post_host_refuses_invalid_or_unallocated_tail(p):
    with pytest.raises(ContractError):
        normal(p=p)


@pytest.mark.parametrize('timer', ['wall_seconds', 'execution_seconds'])
@pytest.mark.parametrize('value', ['bad', {}, True, None])
def test_r1_malformed_timer_is_rejected_before_post_host_arithmetic(timer, value):
    with pytest.raises(ContractError):
        normal(run={timer: value})


def test_r1_inherited_runtime_and_wider_merged_tail_keep_existing_validation():
    from gf_cli.config import _validate_table
    loaded = normal()
    _validate_table('run', {'post_host_seconds': 20.}, context=loaded.effective)
    _validate_table('run', {'wall_seconds': 7500., 'post_host_seconds': 100.}, context=loaded.effective)
    assert normal(run={'wall_seconds': 7500., 'post_host_seconds': 100.}).config.run.post_host_seconds == 100.


@pytest.mark.parametrize('boundary', ['ordinary', 'campaign', 'clock'])
def test_unrepresentable_post_host_integer_is_a_controlled_configuration_error(boundary):
    from gf_common.rust_player_clock import clock_deadlines
    from tests.test_rust_player_clock import anchor, inputs
    huge = 10**400
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = huge
    calls = {'ordinary': lambda: normal(p=huge), 'campaign': lambda: campaign_config(raw, origin=ROOT),
        'clock': lambda: clock_deadlines(anchor(), inputs=inputs(), post_host_seconds=huge)}
    with pytest.raises(ContractError):
        calls[boundary]()


def test_hpo6_post_host_projects_exactly_and_old_versions_remain_closed(tmp_path):
    raw = campaign_raw()
    old = campaign_config(raw, origin=ROOT)
    raw['operating']['post_host_seconds'] = 30.
    loaded = campaign_config(raw, origin=ROOT)
    assert 'post_host_seconds' not in asdict(old.config.operating)
    assert type(loaded.config.operating) is not type(old.config.operating)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    trial = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert trial.config.run.post_host_seconds == trial.config.hpo_spec['operating']['post_host_seconds'] == 30.
    path = tmp_path / 'effective.json'
    path.write_bytes(canonical(trial.effective).encode())
    assert asdict(load_rust_player_effective(path, expected_sha256=trial.sha256).config) == asdict(trial.config)
    from gf_cli.hpo_campaign import canonical_control
    from gf_cli.rust_player_entry import startup_preview
    control = materialize_hpo_run(loaded, canonical_control(loaded.config), learning_seed=17, run_id=RUN_ID)
    for projected in (trial, control):
        row = startup_preview(projected)
        assert row['learner_tensor_bytes'] == row['g0_serving_tensor_bytes'] > 0
        assert row['declared_ram_bytes'] == loaded.config.operating.managed_ram_bytes
        assert row['future_phase_checks_required'] and not row['host_admission']
    raw['schema_version'] = 5
    raw['runtime'] = {'mode': 'shared-mlp-v1', 'max_batch_size': 3, 'formation_wait_ms': 0.}
    with pytest.raises(ContractError):
        campaign_config(raw, origin=ROOT)


def test_timed_native_variants_cannot_open_existing_public_execution_guards():
    from gf_common.rust_player import require_legacy_player_entry
    with pytest.raises(ContractError, match='authenticated Rust-supervised'):
        require_legacy_player_entry(normal().config)


@pytest.mark.parametrize('malformed', [None, 7])
def test_new_effective_reader_keeps_operating_type_errors_controlled(tmp_path, malformed):
    from gf_common.domain import digest
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    loaded = campaign_config(raw, origin=ROOT)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    effective = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID).effective
    effective['hpo_spec']['operating'] = malformed
    path = tmp_path / 'effective.json'
    path.write_bytes(canonical(effective).encode())
    with pytest.raises(ContractError) as failed:
        load_rust_player_effective(path, expected_sha256=digest(effective))
    assert isinstance(failed.value.__cause__, TypeError)


@pytest.mark.parametrize('base_p', [20., 30.])
def test_campaign_post_host_is_authoritative_and_base_cannot_conflict(monkeypatch, base_p):
    from gf_cli import hpo_config
    base = load_config(NORMAL_BASE,
        {'run': {'player_runtime': runtime(), 'post_host_seconds': base_p}})
    monkeypatch.setattr(hpo_config, 'load_config', lambda path: base)
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    if base_p != 30.:
        with pytest.raises(ContractError, match='post_host'):
            campaign_config(raw, origin=ROOT)
    else:
        assert campaign_config(raw, origin=ROOT).config.operating.post_host_seconds == 30.


@pytest.mark.parametrize('learner,serving', [('cpu', 'cpu'), ('cpu', 'cuda:0'), ('cuda:0', 'cpu'), ('cuda:0', 'cuda:0')])
def test_startup_preview_counts_only_actual_planned_role_networks_once(learner, serving):
    from gf_cli.rust_player_entry import startup_preview
    loaded = normal(training={'device': learner}, model={'inference': {'device': serving}})
    preview = startup_preview(loaded)
    network = preview['parameter_bytes'] + preview['position_bytes']
    assert preview['learner_tensor_bytes'] == (network if learner != 'cpu' else 0)
    assert preview['g0_serving_tensor_bytes'] == (network if serving != 'cpu' else 0)
    assert preview['startup_cuda_bytes'] == (preview['host_fixed_bytes']
        + preview['learner_tensor_bytes'] + preview['g0_serving_tensor_bytes'])
    expected_fixed = loaded.config.model.resources.cuda_context_bytes + loaded.config.model.resources.workspace_bytes
    assert preview['host_fixed_bytes'] == (expected_fixed if 'cuda:0' in (learner, serving) else 0)
    assert preview['future_phase_checks_required'] and preview['host_admission'] is False
    assert preview['declared_source_bytes'] == loaded.config.preprocessing.memory_budget_mib * 1024**2


def test_preview_has_no_torch_import_and_refuses_old_missing_p_in_fresh_interpreter():
    import subprocess
    import sys
    code = '''
import sys
from gf_cli.rust_player_entry import startup_preview
from tests.test_rust_player_public_entry import normal
from gf_cli.config import load_config
from tests.test_rust_player_config import NORMAL_BASE,runtime
from gf_common.domain import ContractError
for learner,serving in [('cpu','cpu'),('cpu','cuda:0'),('cuda:0','cpu'),('cuda:0','cuda:0')]:
    row=startup_preview(normal(training={'device':learner},model={'inference':{'device':serving}}))
    n=row['parameter_bytes']+row['position_bytes']
    assert row['learner_tensor_bytes']==(n if learner!='cpu' else 0)
    assert row['g0_serving_tensor_bytes']==(n if serving!='cpu' else 0)
    assert row['startup_cuda_bytes']==row['host_fixed_bytes']+row['learner_tensor_bytes']+row['g0_serving_tensor_bytes']
assert 'torch' not in sys.modules
try: startup_preview(load_config(NORMAL_BASE, {'run': {'player_runtime': runtime()}}))
except ContractError: pass
else: raise AssertionError('missing P silently became execution-ready')
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_ordinary_prepare_holds_real_parent_and_publishes_one_stable_snapshot(tmp_path):
    import copy
    import fcntl
    import json
    import sys

    from gf_cli.rust_player_entry import prepare_ordinary
    from gf_common.domain import digest
    parent = tmp_path / 'parent'
    loaded = normal(training={'device': 'cpu'}, model={'inference': {'device': 'cpu'}})
    with prepare_ordinary(loaded, parent, mode='fresh', pause_after_update=1) as prepared:
        # Dereferencing this symlink selects base Python rather than this venv.
        assert prepared.launch['python_executable'] == str(Path(sys.executable).absolute())
        assert prepared.loaded.inputs == ({'path': str(parent / '.rust-player/effective-config.json'), 'sha256': loaded.sha256},)
        assert prepared.loaded.sha256 == loaded.sha256
        assert prepared.operation['execution_request'] == {'mode': 'fresh', 'pause_after_update': 1}
        assert prepared.operation['config_path'] == prepared.loaded.inputs[0]['path']
        assert prepared.clock['operation_sha256'] == digest(prepared.operation)
        assert json.loads(prepared.provenance_path.read_bytes()) == {'inputs': list(loaded.inputs)}
        with (parent / '.lock').open('r+b') as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TypeError):
            copy.copy(prepared)
        prepared.require()
    with (parent / '.lock').open('r+b') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(ContractError):
        prepared.require()


def test_failed_fixed_exec_closes_sealed_launch_executable_and_actual_parent(tmp_path, monkeypatch):
    import fcntl
    import json
    import os

    from gf_cli.rust_player_entry import exec_ordinary
    from gf_common.rust_player import LINUX_F_GET_SEALS, LINUX_REQUIRED_LAUNCH_SEALS
    from gf_simulator_client import rust_player_companion
    owned_fds = []
    parent = tmp_path / 'parent'
    def refuse(executable, argv, env):
        launch_fd = int(argv[2])
        launch = json.loads(os.pread(launch_fd, 65536, 0))
        mapping = launch['locks'][0]
        assert mapping['lock_kind'] == 'parent'
        assert os.get_inheritable(mapping['fd']) and os.get_inheritable(launch_fd)
        assert fcntl.fcntl(launch_fd, LINUX_F_GET_SEALS) & LINUX_REQUIRED_LAUNCH_SEALS == LINUX_REQUIRED_LAUNCH_SEALS
        assert (os.fstat(mapping['fd']).st_dev, os.fstat(mapping['fd']).st_ino) == (mapping['device'], mapping['inode'])
        owned_fds.extend([executable, launch_fd, mapping['fd']])
        raise OSError('test-only refused fixed exec')
    monkeypatch.setattr(rust_player_companion.os, 'execve', refuse)
    with pytest.raises(OSError, match='refused fixed exec'):
        exec_ordinary(normal(), parent, mode='fresh')
    assert len(owned_fds) == 3
    for fd in owned_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    with (parent / '.lock').open('r+b') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_fixed_exec_refuses_replaced_executable_before_launch_and_closes_its_fd(tmp_path, monkeypatch):
    import os

    from gf_cli.rust_player_entry import prepare_ordinary
    from gf_simulator_client import rust_player_companion
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'not the actual installed companion bytes')
    actual_open, opened = os.open, []
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        def changed(path, *args, **kwargs):
            if path == prepared.companion.executable:
                fd = actual_open(replacement, *args, **kwargs)
                opened.append(fd)
                return fd
            return actual_open(path, *args, **kwargs)
        def forbidden(*args):
            raise AssertionError('changed executable reached sealed launch/exec')
        monkeypatch.setattr(rust_player_companion.os, 'open', changed)
        monkeypatch.setattr(rust_player_companion, 'sealed_launch_fd', forbidden)
        with pytest.raises(ContractError, match='changed before exec'):
            prepared.execute()
        assert len(opened) == 1
        with pytest.raises(OSError):
            os.fstat(opened[0])
        prepared.require()  # Existing parent remains owned until its actual context ends.


def test_missing_companion_fails_before_capture_or_parent_publication(tmp_path, monkeypatch):
    from gf_cli import rust_player_entry
    from gf_simulator_client import rust_player_companion
    def missing(*args):
        raise ContractError('missing installed companion')
    def forbidden():
        raise AssertionError('clock was captured before pure preflight finished')
    monkeypatch.setattr(rust_player_companion, 'discover_companion', missing)
    monkeypatch.setattr(rust_player_entry, '_capture_clock', forbidden)
    with pytest.raises(ContractError, match='missing installed companion'), rust_player_entry.prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh'):
        raise AssertionError('missing companion accepted')
    assert not (tmp_path / 'parent').exists()


def test_snapshot_tamper_and_foreign_thread_refuse_before_exec(tmp_path):
    import threading

    from gf_cli.rust_player_entry import prepare_ordinary
    failures = []
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        def foreign():
            try:
                prepared.require()
            except ContractError as error:
                failures.append(error)
        thread = threading.Thread(target=foreign)
        thread.start(); thread.join()
        assert len(failures) == 1
        Path(prepared.operation['config_path']).write_bytes(b'{}')
        with pytest.raises(ContractError):
            prepared.require()


@pytest.mark.parametrize('field', ['python_executable', 'operation_path', 'record_directory'])
def test_prepared_launch_cannot_be_retargeted_from_its_fixed_immutable_data(tmp_path, field):
    from gf_cli.rust_player_entry import prepare_ordinary
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        prepared.launch[field] = '/foreign/retargeted'
        with pytest.raises(ContractError):
            prepared.require()


@pytest.mark.parametrize('builder', ['replace', 'constructor'])
@pytest.mark.parametrize('retarget', ['unchanged', 'python_executable', 'record_directory', 'operation_path', 'companion'])
def test_r1_reconstructed_wrapper_cannot_issue_fixed_factory_facts(tmp_path, monkeypatch, builder, retarget):
    from dataclasses import fields, replace

    from gf_cli.rust_player_entry import prepare_ordinary
    from gf_simulator_client import rust_player_companion
    calls = []
    def intercepted(companion, launch):
        calls.append((companion, launch))
        raise RuntimeError('mock final exec was reached')
    monkeypatch.setattr(rust_player_companion, 'exec_supervisor', intercepted)
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        values = {item.name: getattr(prepared, item.name) for item in fields(prepared) if item.init}
        if retarget == 'companion':
            values['companion'] = replace(prepared.companion, executable=tmp_path / 'foreign-program')
        elif retarget != 'unchanged':
            values['launch'] = {**prepared.launch, retarget: '/foreign/retargeted'}
        rebuilt = replace(prepared, **values) if builder == 'replace' else type(prepared)(**values)
        assert rebuilt.ownership is prepared.ownership  # REAL owner, not a metadata substitute.
        with pytest.raises(ContractError):
            rebuilt.execute()
        assert not calls
        prepared.require()
        with pytest.raises(RuntimeError, match='mock final exec'):
            prepared.execute()
        assert calls == [(prepared.companion, prepared.launch)]
    with pytest.raises(ContractError):
        rebuilt.require()  # Even unchanged DATA cannot create a live owner after close.


def test_r1_original_discovered_companion_manifest_remains_bound_before_exec(tmp_path, monkeypatch):
    from gf_cli.rust_player_entry import prepare_ordinary
    from gf_simulator_client import rust_player_companion
    def forbidden(*args):
        raise AssertionError('changed discovered companion reached final exec')
    monkeypatch.setattr(rust_player_companion, 'exec_supervisor', forbidden)
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        prepared.companion.manifest['executable_sha256'] = '0' * 64
        with pytest.raises(ContractError):
            prepared.execute()


def test_prepared_original_control_root_cannot_be_redirected_after_publication(tmp_path):
    from gf_cli.rust_player_entry import prepare_ordinary
    with prepare_ordinary(normal(), tmp_path / 'parent', mode='fresh') as prepared:
        root = prepared.operation_path.parent
        moved = root.with_name('moved')
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
        with pytest.raises(ContractError):
            prepared.require()


def _paused_launcher_data(prepared):
    """Portable Run5 metadata only; cannot pass actual full resume admission audit."""
    import json

    from gf_common.domain import digest
    from gf_training.rust_player_lineage import resume_identity
    state = json.loads((ROOT / 'tests/fixtures/rust_player_run5_common_metadata.json').read_bytes())
    state.update(status='paused', run_id=prepared.operation['run_id'], config_sha256=prepared.loaded.sha256,
        effective_config=prepared.loaded.effective, config_inputs=list(prepared.loaded.inputs),
        runtime_binding_sha256=prepared.operation['runtime_binding_sha256'],
        monotonic_authority={'boot_id': prepared.clock['clock_binding']['boot_id'], 'deadlines': prepared.operation['deadlines']})
    state['identity']['player_runtime'] = prepared.runtime_binding
    state['identity']['configuration_inputs_sha256'] = digest(prepared.loaded.inputs)
    entry = next(iter(state['invocations'].values()))
    entry.update(operation_path=str(prepared.operation_path), operation_sha256=prepared.clock['operation_sha256'],
        execution_request=prepared.operation['execution_request'])
    state['invocations'] = {'1' * 32: entry}
    classification = {key: state[key] for key in ('action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature')}
    state['resume_identity'] = resume_identity(effective_config=state['effective_config'], identity=state['identity'],
        execution=state['execution'], classification=classification, storage_binding=state['storage_binding'])
    (prepared.ownership.directory / 'run.json').write_bytes(canonical(state).encode())
    return state


def test_resume_uses_actual_referenced_origin_and_never_recaptures_clock(tmp_path, monkeypatch):
    from gf_cli import rust_player_entry
    parent = tmp_path / 'parent'
    loaded = normal(training={'device': 'cpu'}, model={'inference': {'device': 'cpu'}})
    with rust_player_entry.prepare_ordinary(loaded, parent, mode='fresh', pause_after_update=1) as first:
        _paused_launcher_data(first)
        original_clock, original_operation = first.clock, first.operation
    def forbidden():
        raise AssertionError('resume recaptured original clock')
    monkeypatch.setattr(rust_player_entry, '_capture_clock', forbidden)
    with rust_player_entry.prepare_ordinary(loaded, parent, mode='resume') as resumed:
        assert resumed.clock['clock_binding'] == original_clock['clock_binding']
        assert resumed.operation['deadlines'] == original_operation['deadlines']
        assert resumed.operation['operation_id'] != original_operation['operation_id']
        assert resumed.clock['post_host_seconds'] == original_clock['post_host_seconds']
        assert resumed.operation['execution_request'] == {'mode': 'resume', 'pause_after_update': None}


@pytest.mark.parametrize('fault', ['missing-origin', 'foreign-path', 'bad-sha', 'boot', 'expired', 'config', 'not-paused'])
def test_resume_missing_foreign_changed_or_expired_data_refuses_without_new_anchor(tmp_path, monkeypatch, fault):
    import json

    from gf_cli import rust_player_entry
    parent = tmp_path / 'parent'
    loaded = normal()
    with rust_player_entry.prepare_ordinary(loaded, parent, mode='fresh') as first:
        state = _paused_launcher_data(first)
        if fault == 'missing-origin': first.operation_path.unlink()
        elif fault == 'foreign-path': next(iter(state['invocations'].values()))['operation_path'] = '/foreign/operation-fresh.json'
        elif fault == 'bad-sha': next(iter(state['invocations'].values()))['operation_sha256'] = '0' * 64
        elif fault == 'boot': monkeypatch.setattr(rust_player_entry, '_boot_id', lambda: 'foreign-boot')
        elif fault == 'expired': monkeypatch.setattr(rust_player_entry.time, 'monotonic', lambda: 1e100)
        elif fault == 'config': state['config_sha256'] = '0' * 64
        elif fault == 'not-paused': state['status'] = 'complete'
        (parent / 'run.json').write_text(canonical(state))
    def forbidden():
        raise AssertionError('failed resume attempted a fresh clock')
    monkeypatch.setattr(rust_player_entry, '_capture_clock', forbidden)
    with pytest.raises((ContractError, OSError)), rust_player_entry.prepare_ordinary(loaded, parent, mode='resume'):
        raise AssertionError('invalid paused metadata accepted')
    assert json.loads((parent / 'run.json').read_bytes()) == state


def test_ordinary_ppo_update_effective_table_roundtrip_and_legacy_shape(tmp_path):
    from gf_common.domain import digest

    legacy = normal()
    assert legacy.effective['training'] == {
        'algorithm': 'ppo', 'device': 'cuda:0', 'seeds': [17], 'max_updates': 2,
        'max_owned_bytes': 10737418240, 'admission_reservation_bytes': 1073741824,
        'cleanup_seconds': 30, 'max_log_bytes': 1048576,
    }
    loaded = normal(training={'ppo_update': {'minibatch_size': 128, 'gpu_microbatch_size': 128}})
    table = {
        'schema_version': 1, 'collection_games': 256, 'minibatch_size': 128, 'epochs': 4,
        'shuffle_strategy': 'all', 'gpu_microbatch_size': 128,
    }
    assert loaded.effective['training'] == legacy.effective['training'] | {'ppo_update': table}
    assert asdict(loaded.config.training.ppo_update) == table
    path = tmp_path / 'effective.json'
    path.write_text(canonical(loaded.effective))
    assert load_rust_player_effective(path, expected_sha256=loaded.sha256).effective == loaded.effective
    del loaded.effective['training']['ppo_update']['epochs']
    path.write_text(canonical(loaded.effective))
    with pytest.raises(ContractError):
        load_rust_player_effective(path, expected_sha256=digest(loaded.effective))


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2), ('collection_games', 15), ('collection_games', 0),
    ('minibatch_size', 0), ('minibatch_size', True), ('epochs', -1),
    ('gpu_microbatch_size', 0), ('shuffle_strategy', 'episode'), ('extra', 1),
])
def test_ordinary_ppo_update_refuses_invalid_settings(field, value):
    with pytest.raises(ContractError):
        normal(training={'ppo_update': {field: value}})


def test_ordinary_ppo_update_requires_native_model3():
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {'training': {'ppo_update': {}}})


def test_ordinary_resume_restores_recipe_omitted_from_cli(tmp_path):
    from gf_cli.rust_player_entry import prepare_ordinary
    from gf_training.rust_player_supervision import read_control_record

    parent = tmp_path / 'parent'
    initial = normal(training={'ppo_update': {'minibatch_size': 128}})
    with prepare_ordinary(initial, parent, mode='fresh', pause_after_update=1) as first:
        _paused_launcher_data(first)
        clock, operation = first.clock, first.operation
    with prepare_ordinary(normal(), parent, mode='resume') as resumed:
        assert resumed.loaded.effective == initial.effective
        assert resumed.loaded.sha256 == initial.sha256
        assert resumed.operation['deadlines'] == operation['deadlines']
        assert resumed.clock['clock_binding'] == clock['clock_binding']
        record, _ = read_control_record(resumed.provenance_path, parent / '.rust-player')
        assert record == {'inputs': list(initial.inputs)}
    with pytest.raises(ContractError), prepare_ordinary(
            normal(training={'ppo_update': {'minibatch_size': 64}}), parent, mode='resume'):
        pytest.fail('resume accepted a different explicit update recipe')




def test_ordinary_ppo_update_toml_and_dotted_cli_merge(tmp_path, capsys):
    import json

    from gf_cli import main
    from tests.test_rust_player_config import normal_file

    source = normal_file(tmp_path, runtime())
    source.write_text(source.read_text() + '\n[training.ppo_update]\ncollection_games = 24\nepochs = 2\n')
    assert main(['validate', '--config', str(source), '--set', 'training.ppo_update.minibatch_size=128',
        '--set', 'training.ppo_update.shuffle_strategy="episode_interleave"',
        '--set', 'server.config=' + json.dumps(str(ROOT / 'configs/test-server.toml'))]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['effective_config']['training']['ppo_update'] == {
        'schema_version': 1, 'collection_games': 24, 'minibatch_size': 128, 'epochs': 2,
        'shuffle_strategy': 'episode_interleave', 'gpu_microbatch_size': 16,
    }


def test_ordinary_resume_refuses_changed_raw_input_with_equal_effective_config(tmp_path):
    from gf_cli.rust_player_entry import prepare_ordinary
    from tests.test_rust_player_config import normal_file

    source = normal_file(tmp_path, runtime())
    override = {'run': {'post_host_seconds': 30.}, 'training': {'ppo_update': {'minibatch_size': 64}}}
    first_input = load_config(source, override)
    parent = tmp_path / 'parent'
    with prepare_ordinary(first_input, parent, mode='fresh', pause_after_update=1) as first:
        _paused_launcher_data(first)
    source.write_text(source.read_text() + '\n# Changed raw bytes only\n')
    changed = load_config(source, override)
    assert changed.sha256 == first_input.sha256 and changed.inputs != first_input.inputs
    with pytest.raises(ContractError, match='input-file receipts differ'), prepare_ordinary(
            changed, parent, mode='resume'):
        pytest.fail('resume accepted modified input provenance')


def test_mixed_refresh_needs_16_ordinary_rust_games(tmp_path):
    from tests.test_rust_player_config import normal_file

    path = normal_file(tmp_path, runtime())
    path.write_text(path.read_text() + (
        '\n[training_opponents]\nversion="1"\npreset="mixed-refresh-16-v1"\n'
        '[training_opponents.refresh]\nversion="1"\nattack_threshold=12\n'
    ))
    with pytest.raises(ContractError, match='mixed-refresh-16-v1'):
        load_config(path, {'training': {'ppo_update': {}}})
    loaded = load_config(path, {'training': {'ppo_update': {'collection_games': 16}}})
    assert loaded.config.training.ppo_update.collection_games == 16


def test_hpo_refuses_ordinary_rust_update_table_in_base(tmp_path):
    from tests.test_rust_player_config import normal_file

    source = normal_file(tmp_path, runtime())
    source.write_text(source.read_text() + '\n[training.ppo_update]\ncollection_games = 16\n')
    with pytest.raises(ContractError, match='HPO base must not declare'):
        campaign_config(campaign_raw() | {'base_experiment': str(source)}, origin=tmp_path)
