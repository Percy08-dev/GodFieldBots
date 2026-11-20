"""Canonical campaign CONFIG and actual Store launch controls; no live training."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.hpo_config import campaign_config
from gf_common.domain import ContractError, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
from tests.test_rust_player_config import ROOT
from tests.test_rust_player_native_planning import loaded_campaign


def test_campaign_snapshot_roundtrip_uses_existing_parsers_without_late_toml_reads(tmp_path, monkeypatch):
    from gf_cli import hpo_config
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot
    loaded = loaded_campaign(tmp_path)
    snapshot = campaign_snapshot(loaded)
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical snapshot reopened original TOML')
    monkeypatch.setattr(hpo_config, 'load_config', forbidden)
    restored = load_campaign_snapshot(snapshot)
    assert asdict(restored.config) == asdict(loaded.config)
    assert asdict(restored.base.config) == asdict(loaded.base.config)
    assert restored.sha256 == loaded.sha256 and restored.inputs == loaded.inputs
    assert snapshot['inputs'] == list(loaded.inputs)
    assert restored.base.inputs == loaded.inputs  # Combined provenance DATA, not a claimed subset.


@pytest.mark.parametrize('fault', ['unknown', 'missing-default', 'derived-space', 'base-sha', 'combined-sha'])
def test_campaign_snapshot_refuses_unknown_missing_derived_or_hash_changes(tmp_path, fault):
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot
    snapshot = campaign_snapshot(loaded_campaign(tmp_path))
    if fault == 'unknown': snapshot['campaign_effective']['extra'] = None
    elif fault == 'missing-default': del snapshot['campaign_effective']['training']['epochs']
    elif fault == 'derived-space': snapshot['campaign_effective']['space']['control_values'] = []
    elif fault == 'base-sha': snapshot['base_sha256'] = '0' * 64
    elif fault == 'combined-sha': snapshot['campaign_sha256'] = '0' * 64
    if fault in ('unknown', 'missing-default', 'derived-space'):
        snapshot['campaign_sha256'] = digest({'campaign': snapshot['campaign_effective'], 'base': snapshot['base_sha256']})
    with pytest.raises(ContractError):
        load_campaign_snapshot(snapshot)


def test_pinned_base_does_not_change_default_campaign_file_loading(tmp_path, monkeypatch):
    from gf_cli import hpo_config
    from tests.test_rust_player_config import campaign_raw
    loaded = loaded_campaign(tmp_path)
    calls = []
    def actual(path):
        calls.append(path)
        return loaded.base
    monkeypatch.setattr(hpo_config, 'load_config', actual)
    raw = campaign_raw()
    plain = campaign_config(raw, origin=ROOT)
    assert len(calls) == 1 and plain.base is loaded.base
    pinned = campaign_config(raw, origin=ROOT, _pinned_base=loaded.base)
    assert len(calls) == 1 and pinned.base is loaded.base


def test_full_normalized_search_space_roundtrips_without_dropping_derived_fields(tmp_path):
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot
    from tests.test_rust_player_config import campaign_raw
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    loaded = campaign_config(raw, origin=ROOT)
    restored = load_campaign_snapshot(campaign_snapshot(loaded))
    assert asdict(restored.config.space) == asdict(loaded.config.space)


@pytest.mark.parametrize('mutation', ['deleted', 'changed'])
def test_pinned_snapshot_survives_changed_or_deleted_original_tomls(tmp_path, monkeypatch, mutation):
    import re

    from gf_cli import hpo_config
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot
    from tests.test_rust_player_config import campaign_raw
    raw = campaign_raw()
    original = Path(raw['base_experiment'])
    model_copy = tmp_path / 'model.toml'
    def reference(match):
        path = (original.parent / match[2]).resolve()
        if match[1] == 'model':
            model_copy.write_bytes(path.read_bytes())
            path = model_copy
        return f'{match[1]} = "{path}"'
    text = re.sub(r'(?m)^(actions|model|opponent|preprocessing|training) = "([^"]+)"$', reference, original.read_text())
    base_copy = tmp_path / 'base.toml'
    base_copy.write_text(text)
    raw['base_experiment'] = str(base_copy)
    raw['operating']['post_host_seconds'] = 30.
    loaded = campaign_config(raw, origin=ROOT)
    snapshot = campaign_snapshot(loaded)
    assert {str(base_copy), str(model_copy)} <= {item['path'] for item in snapshot['inputs']}
    for path in (base_copy, model_copy):
        if mutation == 'deleted': path.unlink()
        else: path.write_text('changed invalid TOML = [')
    def forbidden(*args, **kwargs):
        raise AssertionError('snapshot attempted an original input reload')
    monkeypatch.setattr(hpo_config, 'load_config', forbidden)
    restored = load_campaign_snapshot(snapshot)
    assert restored.sha256 == loaded.sha256 and restored.inputs == loaded.inputs


def test_session_clock_and_initial_cpu_operation_use_exact_soft_bits_without_training_clock(tmp_path):
    from gf_cli.rust_player_campaign import campaign_session_clock, initial_campaign_operation
    from gf_common.rust_player import deadline_bits, deadline_seconds
    from tests.test_rust_player_clock import anchor
    loaded = loaded_campaign(tmp_path)
    session = campaign_session_clock(loaded, supervision_id='1' * 32, clock_binding=anchor())
    assert deadline_seconds(session['soft_dispatch_finish_bits']) == 100. + loaded.config.session_seconds
    assert deadline_seconds(session['hard_session_finish_bits']) == 100. + loaded.config.session_seconds + loaded.config.run_wall_seconds
    op = initial_campaign_operation(session, operation_id='2' * 32,
        bootstrap_path=tmp_path / 'bootstrap.json', bootstrap_sha256='a' * 64, runtime_binding_sha256='b' * 64)
    assert op['deadlines'] == {**dict.fromkeys(('admission_bits', 'execution_bits', 'term_bits', 'kill_bits'),
        session['soft_dispatch_finish_bits']), 'finish_bits': session['hard_session_finish_bits']}
    assert op['execution_request'] is None and op['run_id'] is None and op['parent_directory'] is None
    with pytest.raises(ContractError):
        campaign_session_clock(loaded, supervision_id='1' * 32, clock_binding=anchor(1e100))
    assert session['max_operation_wall_seconds'] == loaded.config.run_wall_seconds
    assert op['deadlines']['finish_bits'] != deadline_bits(100. + loaded.config.session_seconds
        + loaded.config.run_wall_seconds + loaded.base.config.run.finalization_seconds)


def test_seed_snapshot_reconstructs_typed_settings_and_preserves_resources_without_reading_source(tmp_path):
    from gf_cli.hpo import _settings_inputs
    from gf_cli.rust_player_campaign import load_seed_snapshot
    from gf_training.hpo_config import effective_settings
    loaded = loaded_campaign(tmp_path)
    selected = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    from gf_common.domain import canonical
    seed = tmp_path / 'seed.json'
    seed.write_text(canonical({'schema_version': 1, 'kind': 'hpo-settings-seed',
        'model': {key: value for key, value in asdict(selected.model).items() if key != 'resources'},
        'training': asdict(selected.training), 'effective_settings_sha256': selected.identity,
        'origin': {'campaign_id': 'old', 'study_identity': 'old-study', 'proposal_sha256': 'a' * 64}}) + '\n')
    _settings, sources = _settings_inputs((seed,), loaded)
    value = {'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': sources}
    seed.unlink()
    actual = load_seed_snapshot(value, loaded.config, study_identity='new-study')
    assert asdict(actual[0]) == sources[0]['settings']
    bad = deepcopy(value)
    bad['sources'][0]['settings']['model']['resources']['cuda_context_bytes'] += 1
    with pytest.raises(ContractError):
        load_seed_snapshot(bad, loaded.config, study_identity='new-study')
    with pytest.raises(ContractError):
        load_seed_snapshot(value, loaded.config, study_identity='old-study')


def _factory_loaded(tmp_path, monkeypatch):
    from gf_cli import hpo
    loaded = loaded_campaign(tmp_path)
    monkeypatch.setattr(hpo, 'check_storage', lambda *args, **kwargs: {'scope': 'isolated CONFIG storage I/O'})
    monkeypatch.setattr(hpo, '_conditions', lambda *args: {'scope': 'CONFIG-only fixture, NO scientific qualification'})
    return loaded


def test_campaign_factory_owns_actual_store_and_canonical_bootstrap_without_parent_or_admission(tmp_path, monkeypatch):
    import copy
    import fcntl
    import os
    from dataclasses import replace

    from gf_cli.rust_player_campaign import load_campaign_snapshot, prepare_campaign
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh', progress='off') as prepared:
        store = prepared.store
        root = directory / 'supervision' / prepared.launch['supervision_id']
        assert prepared.launch['record_directory'] == str(root)
        assert prepared.operation['config_path'] == str(root / 'bootstrap.json')
        assert prepared.operation['config_sha256'] == prepared.bootstrap_sha256
        assert prepared.operation['execution_request'] is None
        assert [item['lock_kind'] for item in prepared.launch['locks']] == ['campaign']
        fd = prepared.launch['locks'][0]['fd']
        assert os.fstat(fd).st_ino == prepared.launch['locks'][0]['inode']
        with (directory / 'lock').open('r+b') as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot, _ = store.get(str((root / 'loaded-campaign.json').relative_to(directory)))
        assert load_campaign_snapshot(snapshot).sha256 == loaded.sha256
        assert not list(root.glob('clock-*.json'))  # Initial CPU op has only session provenance.
        cursor, _ = store.get('supervision/current.json')
        assert cursor['phase'] == 'control' and cursor['awaiting_path'] is None
        with pytest.raises(TypeError): copy.copy(prepared)
        with pytest.raises(ContractError): replace(prepared).require()
        prepared.require()
    with pytest.raises(OSError): os.fstat(fd)
    with pytest.raises(ContractError): prepared.require()


def test_campaign_exec_failure_closes_real_sealed_launch_executable_and_store_fd(tmp_path, monkeypatch):
    import json
    import os

    from gf_cli.rust_player_campaign import exec_campaign
    from gf_simulator_client import rust_player_companion
    loaded = _factory_loaded(tmp_path, monkeypatch)
    fds = []
    def refuse(executable, argv, env):
        launch_fd = int(argv[2])
        launch = json.loads(os.pread(launch_fd, 65536, 0))
        assert [row['lock_kind'] for row in launch['locks']] == ['campaign']
        fds.extend([executable, launch_fd, launch['locks'][0]['fd']])
        raise OSError('test-only campaign exec refusal')
    monkeypatch.setattr(rust_player_companion.os, 'execve', refuse)
    with pytest.raises(OSError, match='campaign exec refusal'):
        exec_campaign(loaded, tmp_path / 'campaign', mode='fresh')
    assert len(fds) == 3
    for fd in fds:
        with pytest.raises(OSError): os.fstat(fd)


def test_campaign_resume_no_work_starts_new_session_without_mutating_old_cpu_operation(tmp_path, monkeypatch):
    from gf_cli.rust_player_campaign import prepare_campaign
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh') as first:
        original_path = first.operation_path
        original_bytes = original_path.read_bytes()
        original_finish = first.operation['deadlines']['finish_bits']
        first_session = first.launch['supervision_id']
    with prepare_campaign(loaded, directory, mode='resume') as resumed:
        assert resumed.launch['supervision_id'] != first_session
        assert original_path.read_bytes() == original_bytes
        assert resumed.operation_path != original_path
        assert resumed.session_clock['hard_session_finish_bits'] != original_finish


@pytest.mark.parametrize('history', ['awaiting', 'success'])
def test_campaign_resume_refuses_unclosed_or_success_history_without_mutating_cursor(tmp_path, monkeypatch, history):
    from gf_cli.rust_player_campaign import prepare_campaign
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh') as first:
        store = first.store
        cursor, cursor_sha = store.get('supervision/current.json')
        if history == 'success':
            store.put('proposals/000000/seeds/17/training.json', {'scope': 'unqualified successful-history DATA needs C2b'})
        else:
            root = first.operation_path.parent
            path = root / 'awaiting' / 'unclosed.json'
            awaiting = {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
                'campaign_id': cursor['campaign_id'], 'proposal_no': 0, 'learning_seed': 17,
                'run_id': 'old-run', 'run_path': str(tmp_path / 'old-run'),
                'config_path': str(tmp_path / 'old-run/config.json'), 'config_sha256': 'a' * 64,
                'runtime_binding_sha256': 'b' * 64, 'operation_path': str(root / 'operation-unclosed.json'),
                'operation_sha256': 'c' * 64, 'supervision_id': first.launch['supervision_id'],
                'session_clock_sha256': store.get(str((root / 'session-clock.json').relative_to(directory)))[1], 'created_unix': 1.}
            sha = store.put(str(path.relative_to(directory)), awaiting)
            store.update_supervision_cursor(dict(cursor, phase='awaiting-host', awaiting_path=str(path),
                awaiting_sha256=sha), expected_sha256=cursor_sha)
        before = (directory / 'supervision/current.json').read_bytes()
    with pytest.raises(ContractError), prepare_campaign(loaded, directory, mode='resume'):
        raise AssertionError('unsupported history acquired an execution continuation')
    assert (directory / 'supervision/current.json').read_bytes() == before


@pytest.mark.parametrize('count', [1, 2])
def test_known_ack_candidate_defers_full_reader_until_new_cpu_parent_custody(tmp_path, monkeypatch, count):
    """Shape/read-boundary DATA only; no actual native ACK or Source proof."""
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import _audit_terminal_history, prepare_campaign
    from gf_training.lineage import require_hpo_run_owner, run_lock
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory, parent = tmp_path / 'campaign', tmp_path / 'old-run'
    with run_lock(parent, create=True): pass
    with prepare_campaign(loaded, directory, mode='fresh') as first:
        store, root = first.store, first.operation_path.parent
        cursor, cursor_sha = store.get('supervision/current.json')
        awaiting_template = {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
            'campaign_id': cursor['campaign_id'], 'proposal_no': 0, 'learning_seed': 17,
            'run_id': 'old-run', 'run_path': str(parent), 'config_path': str(parent / 'config.json'),
            'config_sha256': 'a' * 64, 'runtime_binding_sha256': 'b' * 64,
            'operation_path': str(root / 'operation-old-op.json'), 'operation_sha256': 'c' * 64,
            'supervision_id': first.launch['supervision_id'], 'session_clock_sha256': cursor['session_clock_sha256'],
            'created_unix': 1.}
        rows = {}
        for index in range(count):
            path = root / f'awaiting/old-op-{index}.json'
            awaiting = {**awaiting_template, 'operation_path': str(root / f'operation-old-op-{index}.json')}
            sha = store.put(str(path.relative_to(directory)), awaiting)
            terminal = str((root / f'terminal/old-op-{index}.json').relative_to(directory))
            disposition = {'schema_version': 1, 'kind': 'rust-player-campaign-terminal-disposition-v1',
                **{key: awaiting[key] for key in ('campaign_id', 'proposal_no', 'learning_seed', 'supervision_id',
                    'session_clock_sha256', 'operation_path', 'operation_sha256')},
                'proposal_sha256': 'a' * 64, 'awaiting_path': str(path), 'awaiting_sha256': sha,
                'invocation_id': f'{index + 1:032x}', 'classification': 'healthy_incomplete',
                'evidence': {'completion_path': str(root / 'completion.json'), 'completion_sha256': 'a' * 64,
                    'physical_path': str(root / 'physical.json'), 'physical_sha256': 'a' * 64},
                **{key + '_path': str(directory / (key + '.json')) for key in ('proposal_failure', 'tell_intent', 'tell_receipt')},
                **dict.fromkeys((key + '_sha256' for key in ('proposal_failure', 'tell_intent', 'tell_receipt')), 'a' * 64)}
            disposition_sha = store.put(terminal, disposition)
            store.put(str((root / ('disposition-ack-' + disposition['invocation_id'] + '.json')).relative_to(directory)),
                {'schema_version': 1, 'kind': 'rust-player-campaign-disposition-ack-v1',
                    'supervision_id': awaiting['supervision_id'], 'invocation_id': disposition['invocation_id'],
                    'control_invocation_id': 'f' * 32, 'operation_sha256': awaiting['operation_sha256'],
                    'disposition_sha256': disposition_sha, 'evidence_sha256': digest(disposition['evidence'])})
            rows[terminal] = (path, sha)
        store.update_supervision_cursor(dict(cursor, phase='awaiting-host', awaiting_path=str(path), awaiting_sha256=sha),
            expected_sha256=cursor_sha)
        old_cursor = (directory / 'supervision/current.json').read_bytes()
    checked = []
    def verified_reader(store, name):
        store._require()
        owner = require_hpo_run_owner(parent)
        assert owner.directory == parent
        checked.append(name)
        path, sha = rows[name]
        return {'campaign_id': awaiting['campaign_id'], 'awaiting_path': str(path), 'awaiting_sha256': sha}, {}
    monkeypatch.setattr(CampaignStore, 'read_acknowledged_supervision_disposition', verified_reader)
    with prepare_campaign(loaded, directory, mode='resume') as resumed:
        assert checked == []  # Pre-exec only classifies; no old Source audit here.
        assert (directory / 'supervision/current.json').read_bytes() == old_cursor
        current, _ = resumed.store.get('supervision/current.json')
        assert current['phase'] == 'awaiting-host' and current['awaiting_path'] == str(path)
        for terminal in rows:
            _audit_terminal_history(resumed.store, terminal, awaiting, parent_root=parent.parent)
        assert checked == list(rows)
        with pytest.raises(ContractError): require_hpo_run_owner(parent)


def test_campaign_prepared_reconstruction_cannot_retarget_original_companion_or_interpreter(tmp_path, monkeypatch):
    from dataclasses import replace

    from gf_cli.rust_player_campaign import prepare_campaign
    from gf_simulator_client import rust_player_companion
    loaded = _factory_loaded(tmp_path, monkeypatch)
    def forbidden(*args):
        raise AssertionError('reconstructed campaign reached final exec')
    monkeypatch.setattr(rust_player_companion, 'exec_supervisor', forbidden)
    with prepare_campaign(loaded, tmp_path / 'campaign', mode='fresh') as prepared:
        for clone in (replace(prepared), replace(prepared, launch={**prepared.launch, 'python_executable': '/foreign'}),
                replace(prepared, companion=replace(prepared.companion, executable=tmp_path / 'foreign'))):
            with pytest.raises(ContractError): clone.execute()


def test_r1_real_prepare_conditions_use_legacy_fresh_eval_with_external_preflight_only_isolated(tmp_path, monkeypatch):
    from gf_cli import hpo
    from gf_cli.config import load_config
    from gf_cli.rust_player_campaign import prepare_campaign
    from gf_common import card_instance
    from gf_common.shared_inference import evaluation_runtime
    from gf_models import arrays
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_training import native_backend, ppo_run
    from tests.test_rust_player_config import campaign_raw
    raw = campaign_raw()
    raw['operating']['post_host_seconds'] = 30.
    raw['storage'].update(parent_root=str(tmp_path), numeric_root=str(tmp_path), optuna_path=str(tmp_path / 'study.sqlite3'))
    base = load_config(raw['base_experiment'], {'server': {'source_dir': str(tmp_path / 'unopened-source')}})
    loaded = campaign_config(raw, origin=ROOT, _pinned_base=base)
    checked = []
    def preflight(self):
        checked.append((self.binary, self.source_root))
        return {'schema_version': 2, 'binary': {'sha256': 'a' * 64},
            'source': {'revision': 'external-preflight-revision', 'dirty': False}}
    # Isolate only external source/native/file preflight providers. The real
    # _conditions and real prepare/publish/composition logic run unchanged.
    monkeypatch.setattr(CardInstanceRuntimeEvidence, 'preflight', preflight)
    monkeypatch.setattr(card_instance, 'reviewed_specs', lambda *args: {'spec': 'b' * 64})
    monkeypatch.setattr(arrays, 'sha256', lambda *args, **kwargs: 'c' * 64)
    monkeypatch.setattr(ppo_run, 'source_identity', lambda *args: {'scope': 'external source identity I/O isolated'})
    monkeypatch.setattr(native_backend, 'require_native_refresh_opponents', lambda: None)
    monkeypatch.setattr(hpo, 'check_storage', lambda *args, **kwargs: {'scope': 'external storage I/O isolated'})
    with pytest.raises(ContractError): evaluation_runtime(loaded.config)  # Global resolver remains strict.
    with prepare_campaign(loaded, tmp_path / 'campaign', mode='fresh') as prepared:
        manifest, _sha = prepared.store.get('campaign.json')
        conditions = manifest['conditions']
        assert conditions['evaluation_runtime'] is None
        assert conditions['integration_identity']['source_revision'] == 'external-preflight-revision'
        assert conditions['integration_identity']['source_dirty'] is False
        assert conditions['integration_identity']['binary_sha256'] == 'a' * 64
        assert manifest['conditions_sha256'] == digest(conditions)
        assert prepared.operation['runtime_binding_sha256']
    assert len(checked) == 1
