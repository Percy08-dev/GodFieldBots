"""Closed campaign control DATA and narrow storage ownership seams; no science."""
from __future__ import annotations

import copy

import pytest

from gf_common import rust_player as records
from gf_common.domain import ContractError, digest


def clock():
    bits = records.deadline_bits
    return {'schema_version': 1, 'kind': 'rust-player-campaign-clock-v1',
        'supervision_id': 'a'*32, 'boot_id': 'boot', 'monotonic_before_bits': bits(10.),
        'monotonic_after_bits': bits(11.), 'unix_anchor_bits': bits(100.),
        'soft_dispatch_finish_bits': bits(21.), 'hard_session_finish_bits': bits(141.),
        'max_operation_wall_seconds': 120., 'mapping_recipe': 'soft-dispatch-plus-max-operation-wall-v1'}


def cursor():
    return {'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
        'campaign_id': 'campaign', 'supervision_id': 'a'*32, 'session_clock_sha256': digest(clock()),
        'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None, 'phase': 'control'}


def test_clock_is_closed_and_preserves_original_hard_mapping():
    validate = getattr(records, 'validate_campaign_clock', None)
    assert callable(validate), 'campaign clock DATA validator missing'
    assert validate(clock()) == clock()
    for changes in ({'hard_session_finish_bits': records.deadline_bits(142.)},
                    {'mapping_recipe': 'renewed'}, {'extra': True},
                    {'max_operation_wall_seconds': True}):
        with pytest.raises(ContractError):
            validate(dict(clock(), **changes))


def test_cursor_has_exact_nullable_pairs_and_no_fictional_invocation():
    validate = getattr(records, 'validate_campaign_cursor', None)
    assert callable(validate), 'campaign cursor DATA validator missing'
    assert validate(cursor()) == cursor()
    for changes in ({'phase': 'unknown'}, {'awaiting_path': '/orphan'},
                    {'current_invocation_id': 'b'*32}, {'extra': 1}):
        with pytest.raises(ContractError):
            validate(dict(cursor(), **changes))


def test_cursor_cas_changes_only_supervision_current_and_put_stays_immutable(tmp_path):
    from gf_cli.hpo_storage import CampaignStore
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=1024**2) as store:
        update = getattr(store, 'update_supervision_cursor', None)
        assert callable(update), 'narrow supervision cursor CAS missing'
        first = update(cursor(), expected_sha256=None)
        assert store.get('supervision/current.json') == (cursor(), first)
        with pytest.raises(ContractError):
            update(cursor(), expected_sha256=None)
    with CampaignStore(tmp_path / 'campaign', mode='resume', max_owned_bytes=1024**2) as store:
        next_value = dict(cursor(), phase='finalized')
        second = store.update_supervision_cursor(next_value, expected_sha256=first)
        assert second != first
        with pytest.raises(ContractError):
            store.put('supervision/current.json', cursor())
    with (CampaignStore(tmp_path / 'campaign', mode='resume', max_owned_bytes=1024**2) as store,
          pytest.raises(ContractError)):
        store.update_supervision_cursor({'kind': 'arbitrary'}, expected_sha256=second)


def test_campaign_store_cannot_be_copied(tmp_path):
    from gf_cli.hpo_storage import CampaignStore
    with (CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=1024**2) as store,
          pytest.raises(TypeError)):
        copy.copy(store)


def test_borrowed_campaign_store_requires_actual_authenticated_hostcontrol():
    from gf_cli.hpo_storage import CampaignStore
    borrow = getattr(CampaignStore, 'borrow', None)
    assert callable(borrow), 'authenticated SAME-OFD campaign store constructor missing'
    with pytest.raises(ContractError):
        borrow(object(), max_owned_bytes=1024**2)


def test_new_terminal_data_helpers_and_internal_phase_are_required():
    from gf_training.rust_player_supervision import HostControl
    for name in ('training_failed', 'campaign_return', 'campaign_disposition', 'control_complete',
                 'campaign_bootstrap', 'initial_operation'):
        assert hasattr(HostControl, name), f'fixed protocol seam missing: {name}'


def test_failure_disposition_ack_is_closed_and_binds_distinct_actual_controller():
    validate = getattr(records, 'validate_campaign_disposition_ack', None)
    assert callable(validate)
    value = {'schema_version': 1, 'kind': 'rust-player-campaign-disposition-ack-v1',
        'supervision_id': 'a'*32, 'invocation_id': 'b'*32, 'control_invocation_id': 'c'*32,
        'operation_sha256': 'd'*64, 'disposition_sha256': 'e'*64, 'evidence_sha256': 'f'*64}
    assert validate(value) == value
    for changes in ({'control_invocation_id': 'b'*32}, {'extra': None}, {'evidence_sha256': None}):
        with pytest.raises(ContractError):
            validate(dict(value, **changes))


def test_internal_dispatch_uses_initial_purpose_and_never_retrains_failed_hpo(monkeypatch):
    from types import SimpleNamespace

    from gf_cli import rust_player_host
    requested = []
    def unavailable(name):
        requested.append(name)
        raise ImportError('Part2 application deliberately absent')
    monkeypatch.setattr(rust_player_host.importlib, 'import_module', unavailable)
    control = SimpleNamespace(initial_operation={'operation_type': 'campaign-continue'},
        campaign_bootstrap={'authenticated': True}, operation={'operation_type': 'hpo-train'},
        campaign_return={'read_only': True})
    with pytest.raises(ContractError):
        rust_player_host._dispatch(control, 'campaign-return')
    assert requested == ['gf_cli.rust_player_campaign']


def test_closed_terminal_disposition_has_variant_evidence_and_no_authority_fields():
    value = {'schema_version': 1, 'kind': 'rust-player-campaign-terminal-disposition-v1',
        'campaign_id': 'campaign', 'proposal_no': 0, 'proposal_sha256': 'a'*64, 'learning_seed': 17,
        'supervision_id': 'b'*32, 'session_clock_sha256': 'c'*64, 'awaiting_path': '/campaign/awaiting.json',
        'awaiting_sha256': 'd'*64, 'operation_path': '/control/operation.json', 'operation_sha256': 'e'*64,
        'invocation_id': 'f'*32, 'classification': 'training_failed', 'evidence': {
            'failure_path': '/control/failure.json', 'failure_sha256': 'a'*64,
            'exit_path': '/control/exit.json', 'exit_sha256': 'b'*64,
            'physical_path': '/namespace/termination.json', 'physical_sha256': 'c'*64},
        'proposal_failure_path': '/campaign/failure.json', 'proposal_failure_sha256': 'd'*64,
        'tell_intent_path': '/campaign/tell.intent.json', 'tell_intent_sha256': 'e'*64,
        'tell_receipt_path': '/campaign/tell.receipt.json', 'tell_receipt_sha256': 'f'*64}
    assert records.validate_campaign_disposition(value) == value
    for changes in ({'classification': 'completed'}, {'finalized': True},
                    {'evidence': dict(value['evidence'], completion_path=None)}):
        with pytest.raises(ContractError):
            records.validate_campaign_disposition(dict(value, **changes))


def test_control_exit_is_real_wait_data_without_admission_or_prepared():
    validate = getattr(records, 'validate_control_exit', None)
    assert callable(validate), 'control-only original wait DATA validator missing'
    value = {'schema_version': 1, 'kind': 'rust-player-control-exit-v1',
        'supervision_id': 'a'*32, 'operation_sha256': 'b'*64, 'control_completion_sha256': 'c'*64,
        'host_original': {'pid': 12, 'start_ticks': 100, 'pgid': 12},
        'wait': {'exit_code': 0, 'signal': None, 'core_dumped': False},
        'descendants': [], 'descendants_reaped': True, 'signals': []}
    assert validate(value) == value
    for changes in ({'prepared_sha256': None}, {'descendants_reaped': False},
                    {'wait': {'exit_code': 1, 'signal': None, 'core_dumped': False}}):
        with pytest.raises(ContractError):
            validate(dict(value, **changes))


def test_loaded_campaign_data_uses_original_input_sequence_and_combined_hash():
    base, campaign = {'base': 'data'}, {'campaign': 'data'}
    value = {'schema_version': 1, 'kind': 'rust-player-loaded-campaign-v1',
        'campaign_effective': campaign, 'campaign_sha256': digest({'campaign': campaign, 'base': digest(base)}),
        'base_effective': base, 'base_sha256': digest(base),
        'inputs': [{'path': '/original/config.toml', 'sha256': 'a'*64}]}
    assert records.validate_loaded_campaign(value) == value
    for changes in ({'base_sha256': 'b'*64}, {'campaign_sha256': 'c'*64}, {'inputs': {}}):
        with pytest.raises(ContractError):
            records.validate_loaded_campaign(dict(value, **changes))


def test_store_encoding_domains_preserve_raw_hashes_and_refuse_wrong_style(tmp_path):
    import hashlib

    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import canonical
    directory = tmp_path / 'campaign'
    with CampaignStore(directory, mode='fresh', max_owned_bytes=1024**2) as store:
        value = {'unicode': '神', 'float': 1e-7, 'negative_zero': -0.0}
        for name, ending in (('old.json', b'\n'), ('supervision/data.json', b'')):
            actual = store.put(name, value)
            data = canonical(value).encode() + ending
            assert (directory / name).read_bytes() == data
            assert actual == hashlib.sha256(data).hexdigest()
            assert store.get(name) == (value, actual)
    for name, data in (('old.json', b'{}'), ('supervision/data.json', b'{}\n')):
        (directory / name).write_bytes(data)
        with CampaignStore(directory, mode='resume', max_owned_bytes=1024**2) as store, pytest.raises(ContractError):
            store.get(name)


def test_seed_settings_snapshot_is_closed_decoded_provenance_without_late_reads():
    validate = getattr(records, 'validate_seed_settings', None)
    assert callable(validate), 'seed-settings DATA validator missing'
    value = {'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': []}
    assert validate(value) == value
    row = {'path': '/no-late-read.json', 'sha256': 'a'*64,
        'settings': {'model': {}, 'training': {}, 'identity': 'b'*64},
        'origin': {'campaign_id': 'campaign', 'study_identity': 'c'*64, 'proposal_sha256': 'd'*64}}
    assert validate(dict(value, sources=[row]))['sources'] == [row]
    for sources in ({}, [{'path': '/only-path'}], [dict(row, state={})]):
        with pytest.raises(ContractError):
            validate(dict(value, sources=sources))


def test_acknowledged_disposition_readback_requires_native_ack_not_report_shape(tmp_path):
    from gf_cli.hpo_storage import CampaignStore
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=1024**2) as store:
        reader = getattr(store, 'read_acknowledged_supervision_disposition', None)
        assert callable(reader), 'strict already-acknowledged DATA readback seam missing'
        with pytest.raises((FileNotFoundError, ContractError)):
            reader('supervision/' + 'a'*32 + '/terminal/op.json')


# Private fixed CPU-only mechanism fixture. No numerical learner, Source, Parent,
# evaluation or public campaign handler is exercised by these original processes.
def _case_loaded(directory, *, short=False):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gf_cli.hpo_config import campaign_config
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw.update(run_wall_seconds=4 if short else 90, session_seconds=2 if short else 30)
    raw['storage'] = {'parent_root': str(directory), 'numeric_root': str(directory),
        'optuna_path': str(directory / 'protocol-study.sqlite3')}
    return campaign_config(raw, origin=directory)


def _prepare_entry_case(binary, mode):
    import os
    import tempfile
    import time
    import uuid
    from pathlib import Path

    from gf_cli.hpo_campaign import CampaignStudy, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import canonical
    from gf_common.rust_player import BUILD_KEYS, deadline_bits
    from gf_simulator_client.rust_player_companion import sealed_launch_fd
    from gf_training.native_backend import binary_identity
    directory = Path(tempfile.mkdtemp(prefix='gfb-entry-' + mode + '-'))
    loaded = _case_loaded(directory, short=mode in ('forced-cleanup', 'old-deadline') or os.environ.get('GFB_ENTRY_PREPARED_STOP_CROSS') == '1')
    store = CampaignStore(directory / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes)
    manifest = initialize_campaign(store, loaded, {'scope': 'CPU protocol mechanism; no scientific parent or eligibility'})
    if mode not in ('no-work', 'control-nonzero', 'control-unannounced'):
        proposal = CampaignStudy(store, loaded, manifest).propose(0, slot=0, kind='sample')
        assert proposal is not None
    sup = uuid.uuid4().hex
    root = store.directory / 'supervision' / sup
    now = time.monotonic()
    wall = loaded.config.run_wall_seconds
    soft = now + loaded.config.session_seconds
    clock_value = {'schema_version': 1, 'kind': 'rust-player-campaign-clock-v1', 'supervision_id': sup,
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'monotonic_before_bits': deadline_bits(now), 'monotonic_after_bits': deadline_bits(now),
        'unix_anchor_bits': deadline_bits(time.time()), 'soft_dispatch_finish_bits': deadline_bits(soft),
        'hard_session_finish_bits': deadline_bits(soft + wall), 'max_operation_wall_seconds': wall,
        'mapping_recipe': 'soft-dispatch-plus-max-operation-wall-v1'}
    relative = 'supervision/' + sup + '/'
    clock_sha = store.put(relative + 'session-clock.json', clock_value)
    snapshot = {'schema_version': 1, 'kind': 'rust-player-loaded-campaign-v1',
        'campaign_effective': loaded.effective, 'campaign_sha256': loaded.sha256,
        'base_effective': loaded.base.effective, 'base_sha256': loaded.base.sha256, 'inputs': list(loaded.inputs)}
    loaded_sha = store.put(relative + 'loaded-campaign.json', snapshot)
    seed_sha = store.put(relative + 'seed-settings.json', {'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': []})
    bootstrap = {'schema_version': 1, 'kind': 'rust-player-campaign-bootstrap-v1', 'supervision_id': sup,
        'campaign_directory': str(store.directory), 'mode': 'fresh', 'loaded_campaign_path': str(root / 'loaded-campaign.json'),
        'loaded_campaign_sha256': loaded_sha, 'seed_settings_path': str(root / 'seed-settings.json'), 'seed_settings_sha256': seed_sha,
        'progress': 'off', 'session_clock_path': str(root / 'session-clock.json'), 'session_clock_sha256': clock_sha,
        'cursor_path': str(store.directory / 'supervision/current.json')}
    bootstrap_sha = store.put(relative + 'bootstrap.json', bootstrap)
    store.update_supervision_cursor({'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
        'campaign_id': manifest['campaign_id'], 'supervision_id': sup, 'session_clock_sha256': clock_sha,
        'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None, 'phase': 'control'}, expected_sha256=None)
    milestones = {name: deadline_bits(now + wall * fraction) for name, fraction in zip(
        records.DEADLINE_KEYS, (.3, .6, .8, .9, 1.), strict=True)}
    operation = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': 'initial-control', 'run_id': None,
        'mode': 'hpo', 'operation_type': 'campaign-continue', 'config_path': str(root / 'bootstrap.json'), 'config_sha256': bootstrap_sha,
        'runtime_binding_sha256': 'b'*64, 'deadlines': milestones, 'parent_directory': None, 'expected_context': None, 'execution_request': None}
    operation_sha = store.put(relative + 'initial-operation.json', operation)
    identity = binary_identity()
    lock = store._lock
    assert lock is not None
    info = os.fstat(lock.fileno())
    launch = {'schema_version': 1, 'kind': 'rust-player-launch-v1', 'supervision_id': sup,
        'python_executable': str(Path(__file__).resolve().parents[1] / '.venv/bin/python'),
        'operation_path': str(root / 'initial-operation.json'), 'operation_sha256': operation_sha,
        'record_directory': str(root), 'mode': 'hpo', 'session_deadlines': {'finish_bits': clock_value['hard_session_finish_bits']},
        'clock_binding': {key: clock_value[key] for key in ('boot_id', 'monotonic_before_bits', 'monotonic_after_bits', 'unix_anchor_bits')},
        'binding': {'python_abi': identity['python_abi'], 'extension_sha256': identity['extension_sha256'],
            'build': {key: identity['build'][key] for key in BUILD_KEYS}},
        'locks': [{'lock_kind': 'campaign', 'fd': lock.fileno(), 'device': info.st_dev, 'inode': info.st_ino}]}
    if mode == 'corrupt-bootstrap':
        (root / 'bootstrap.json').write_text(canonical(dict(bootstrap, progress='wrong')))
    launch_fd = sealed_launch_fd(launch)
    for fd in (launch_fd, lock.fileno()):
        os.set_inheritable(fd, True)
    environment = dict(os.environ, GFB_ENTRY_LAUNCH_FD=str(launch_fd), GFB_ENTRY_CASE=mode,
        CUDA_VISIBLE_DEVICES='')
    os.execve(binary, [binary, '--exact', 'entry_protocol_tests::entry_protocol_inherited_child',
        '--ignored', '--nocapture', '--test-threads=1'], environment)


def _entry_competitor(path):
    import fcntl
    import os
    descriptor = os.open(path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)


def _entry_post_terminal_race(control, point):
    import array
    import os
    import socket
    import time

    from gf_training.rust_player_supervision import read_control_record, write_control_record
    selector = os.environ.get('GFB_ENTRY_POST_TERMINAL_RACE')
    if selector not in (point, point + '-zero'):
        return
    path = control.root / 'terminal-eagain.json'
    while not path.exists():
        assert time.monotonic() < control._finish(), 'original barrier authority expired'
        time.sleep(.002)
    observed, _ = read_control_record(path, control.root)
    assert observed['invocation_id'] == control._envelope['invocation_id']
    fd = control._borrowed['campaign']['fd']
    payload = b'' if selector.endswith('-zero') else b'{'  # Deliberate malformed v1 record.
    sent = control._socket.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))], socket.MSG_NOSIGNAL)
    assert sent == len(payload)
    write_control_record(control.root / 'terminal-race-sent.json', {
        'invocation_id': control._envelope['invocation_id'], 'point': point, 'actual_send_bytes': sent, 'actual_campaign_fd': fd})
    os._exit(0)  # Native must observe the SAME original via WNOWAIT before checking queued input.


def _entry_complete(control):
    from gf_training.rust_player_supervision import write_control_record
    path = control.root / 'control-result.json'
    write_control_record(path, {'schema_version': 1, 'kind': 'rust-player-control-completion-v1',
        'supervision_id': control.launch['supervision_id'], 'operation_sha256': control.operation_sha256,
        'session_clock_sha256': control.campaign_bootstrap['session_clock_sha256'], 'status': 'no_work'})
    control.control_complete(path)
    _entry_post_terminal_race(control, 'control')


def _component_prepare(control, admission):
    from gf_training.rust_player_supervision import write_control_record
    local = admission.observation
    identity = {key: local[key] for key in ('supervision_id', 'invocation_id', 'operation_id', 'run_id', 'runtime_binding_sha256')}
    common = dict(identity, admission_sha256=digest(local))
    servers_path = control.root / 'component-servers.json'
    servers_sha = write_control_record(servers_path, dict(schema_version=1, kind='rust-player-server-retirement-v1',
        **{key: common[key] for key in ('supervision_id', 'invocation_id', 'operation_id')}, servers=[]))
    close_path = control.root / 'component-close.json'
    close_sha = write_control_record(close_path, dict(schema_version=1, kind='rust-player-host-close-ack-v1', **common,
        actors_joined=True, service_joined=True, writer_joined=True, buffers_released=True, gpu_fenced=True,
        native_retirement_sha256='b'*64, writer_retirement_sha256='b'*64, service_retirement_sha256='b'*64,
        cache_retirement_sha256='b'*64, servers_path=str(servers_path), servers_sha256=servers_sha, server_count=0))
    path = control.root / ('prepared-' + local['invocation_id'] + '.json')
    write_control_record(path, dict(schema_version=1, kind='rust-player-prepared-v1', **common,
        config_sha256=local['config_sha256'], state='incomplete', committed_update=0, generation=0,
        state_sha256='c'*64, generation_sha256='c'*64, artifacts_sha256='c'*64, source_manifest_sha256='c'*64,
        close_ack_path=str(close_path), close_ack_sha256=close_sha))
    import json
    import os
    import select
    import socket
    import time
    if os.environ.get('GFB_ENTRY_PREPARED_STOP_CROSS') == '1':
        ready, _, _ = select.select([control._socket], [], [], control._finish() - time.monotonic())
        assert ready, 'no actual native Stop before original finish'
        # PEEK leaves native Stop queued. Actual _request must first send
        # Prepared, then authenticate/consume Stop and send its pending ACK.
        stop = json.loads(control._socket.recv(65536, socket.MSG_PEEK))
        assert stop['kind'] == 'stop' and stop['seq'] == control._server_sequence
        assert all(stop[key] == value for key, value in control._envelope.items())
        write_control_record(control.root / 'stop-cross-peek.json', stop)
    control.prepare(admission, path)


def _entry_borrow_construction_faults(control, budget):
    for stage, uncertain in (('fdopen', False), ('fstat', False), ('fstat', True)):
        _entry_borrow_construction_fault(control, budget, stage, uncertain)


def _entry_borrow_construction_fault(control, budget, stage, uncertain):
    import os

    from gf_cli.hpo_storage import CampaignStore
    from gf_training import rust_player_supervision as supervision
    duplicated, wrapped, positively_closed, wrappers_closed = [], [], [], []
    actual_dup, actual_fdopen, actual_fstat, actual_close = os.dup, os.fdopen, os.fstat, os.close
    def dup(fd):
        original = actual_dup(fd)
        duplicated.append(original)
        return original
    def fdopen(fd, *args, **kwargs):
        if fd not in duplicated:
            return actual_fdopen(fd, *args, **kwargs)
        assert kwargs.get('closefd') is False, 'wrapper must not own raw FD retirement, even on failure'
        if stage == 'fdopen':
            raise OSError('injected campaign fdopen constructor failure')
        file = actual_fdopen(fd, *args, **kwargs)
        wrapped.append(file)
        close = file.close
        def close_file():
            if uncertain:
                raise OSError('injected uncertain actual wrapper close')
            close()
            wrappers_closed.append(fd)
        patch.setattr(file, 'close', close_file)
        return file
    def fstat(fd):
        if stage == 'fstat' and fd in duplicated:
            raise OSError('injected campaign fstat constructor failure')
        return actual_fstat(fd)
    def close_fd(fd):
        if fd in duplicated and uncertain:
            raise OSError('injected uncertain actual raw close')
        actual_close(fd)
        if fd in duplicated:
            positively_closed.append(fd)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(os, 'dup', dup)
            patch.setattr(os, 'fdopen', fdopen)
            patch.setattr(os, 'fstat', fstat)
            patch.setattr(os, 'close', close_fd)
            with pytest.raises(OSError) as caught:
                CampaignStore.borrow(control, max_owned_bytes=budget)
        assert len(duplicated) == 1
        # Outer process boundaries clear exception contexts; actual custody must
        # remain registered independently of those traceback references.
        caught.value.__traceback__ = None
        caught.value.__cause__ = None
        caught.value.__context__ = None
        fd = duplicated[0]
        pending = list(control._campaign_borrowers)
        proof = {'stage': stage, 'uncertain': uncertain, 'actual_duplicate_fd': fd,
            'registered_borrowers': len(pending), 'positive_constructor_closes': list(positively_closed),
            'nonowning_wrappers_closed': list(wrappers_closed)}
        if not positively_closed:
            info = actual_fstat(fd)
            proof['actual_open_inode'] = [info.st_dev, info.st_ino]
        supervision.write_control_record(control.root / f'constructor-{stage}-{uncertain}.json', proof)
        _entry_competitor(control.campaign_bootstrap['campaign_directory'] + '/lock')
        if uncertain:
            assert len(pending) == 1, 'uncertain actual constructor close lost registration'
            owner = getattr(caught.value, 'cleanup_owner', None)
            assert type(owner) is CampaignStore and pending[0].owner is owner
            with pytest.raises(ContractError):
                control.close()
            owner.close()  # Real original pending raw/wrapped owner, after fault restoration.
            assert not control._campaign_borrowers
        else:
            assert positively_closed == [fd], 'constructor FD was not positively closed'
            assert not pending
        _entry_competitor(control.campaign_bootstrap['campaign_directory'] + '/lock')
    finally:
        # Fixture-only disposal of the actual FD/wrapper captured from dup,
        # never a reconstructed capability or a production cleanup receipt.
        if not control._campaign_borrowers and duplicated and not positively_closed and not uncertain:
            if wrapped:
                wrapped[0].close()
            else:
                actual_close(duplicated[0])


def _entry_raw_close_replacement(control, budget):
    import os

    from gf_cli.hpo_storage import CampaignStore
    from gf_training.rust_player_supervision import write_control_record
    actual_dup, actual_dup2, actual_fdopen, actual_close = os.dup, os.dup2, os.fdopen, os.close
    duplicated, replacements, attempts = [], [], []
    def dup(fd):
        result = actual_dup(fd)
        duplicated.append(result)
        return result
    def fdopen(fd, *args, **kwargs):
        if fd in duplicated:
            raise OSError('injected wrapping failure before returning wrapper')
        return actual_fdopen(fd, *args, **kwargs)
    def side_effect_close(fd):
        if fd in duplicated:
            attempts.append(fd)
            actual_close(fd)  # The first close really completed before reporting failure.
            actual_dup2(control._borrowed['campaign']['fd'], fd)
            replacement = actual_fdopen(fd, 'r+b')  # DIFFERENT actual owner, SAME number/OFD.
            replacements.append(replacement)
            raise OSError('injected raw close side effect then uncertain result')
        return actual_close(fd)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(os, 'dup', dup)
            patch.setattr(os, 'fdopen', fdopen)
            patch.setattr(os, 'close', side_effect_close)
            with pytest.raises(OSError) as caught:
                CampaignStore.borrow(control, max_owned_bytes=budget)
        assert len(replacements) == len(attempts) == 1
        replacement = replacements[0]
        owner = caught.value.cleanup_owner
        token = next(iter(control._campaign_borrowers))
        assert token.owner is owner
        info = os.fstat(replacement.fileno())
        with pytest.raises(ContractError):
            owner.close()  # MUST NOT close another owner's indistinguishable replacement.
        after = os.fstat(replacement.fileno())
        assert (after.st_dev, after.st_ino) == (info.st_dev, info.st_ino)
        assert token in control._campaign_borrowers and len(attempts) == 1
        with pytest.raises(ContractError):
            control.close()
        _entry_competitor(control.campaign_bootstrap['campaign_directory'] + '/lock')
        write_control_record(control.root / 'raw-replacement-proof.json', {
            'original_close_attempts': attempts, 'replacement_fd': replacement.fileno(),
            'replacement_original_inode': [info.st_dev, info.st_ino], 'replacement_after_inode': [after.st_dev, after.st_ino],
            'retained_registration': True})
        replacement.close()  # Only the ACTUAL replacement owner closes its own file.
        with pytest.raises(ContractError):
            owner.close()  # Still no numeric retry/retirement of uncertain original custody.
        assert token in control._campaign_borrowers
        _entry_complete(control)
        os._exit(0)  # Unknown original custody remains registered UNTIL actual host exit.
    finally:
        for replacement in replacements:
            if not replacement.closed:
                try:
                    replacement.close()
                except OSError:
                    pass  # RED fixture disposal, never a production cleanup proof.


def _entry_host(mode, control_fd, launch_fd, phase):
    import hashlib
    import os
    import signal
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from gf_cli.hpo_campaign import CampaignStudy
    from gf_cli.hpo_storage import CampaignStore
    from gf_training.lineage import borrow_run_lock, run_lock, transfer_run_lock
    from gf_training.rust_player_supervision import (
        HostControl,
        runtime_producer,
        write_control_record,
    )
    control = HostControl(int(control_fd), int(launch_fd), phase=phase)
    root = control.root
    (root / ('host-started-' + str(os.getpid()))).write_text(str(os.getpid()))
    write_control_record(root / ('phase-' + phase + '.json'), {'pid': os.getpid(), 'invocation_id': control._envelope['invocation_id']})
    directory = Path(control.campaign_bootstrap['campaign_directory'])
    loaded = _case_loaded(directory.parent, short=mode in ('forced-cleanup', 'old-deadline') or os.environ.get('GFB_ENTRY_PREPARED_STOP_CROSS') == '1')
    # Same inode is insufficient: an accidentally closed/reused FD must not
    # turn a competing open description into a borrowed supervisor OFD.
    mapped_fd = control._borrowed['campaign']['fd']
    retained = os.dup(mapped_fd)
    substitute = os.open(directory / 'lock', os.O_RDWR)
    try:
        os.dup2(substitute, mapped_fd)
        with pytest.raises(ContractError):
            CampaignStore.borrow(control, max_owned_bytes=loaded.config.operating.parent_budget_bytes)
    finally:
        os.dup2(retained, mapped_fd)
        os.close(retained)
        os.close(substitute)
    if os.environ.get('GFB_ENTRY_RAW_REPLACEMENT') == '1':
        _entry_raw_close_replacement(control, loaded.config.operating.parent_budget_bytes)
    if os.environ.get('GFB_ENTRY_BORROW_CTOR_FAULTS') == '1':
        _entry_borrow_construction_faults(control, loaded.config.operating.parent_budget_bytes)
    with CampaignStore.borrow(control, max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        _entry_competitor(directory / 'lock')
        with pytest.raises(TypeError):
            copy.copy(store)
        import threading
        foreign_errors = []
        def foreign_store_read():
            try:
                store.get('campaign.json')
            except ContractError as error:
                foreign_errors.append(error)
        thread = threading.Thread(target=foreign_store_read)
        thread.start()
        thread.join()
        assert len(foreign_errors) == 1
        token = store._borrower
        original_close = token.file.close
        def failed_close():
            raise OSError('injected actual campaign borrower close failure')
        token.file.close = failed_close
        try:
            with pytest.raises(OSError):
                store.close()
            assert token in control._campaign_borrowers
            with pytest.raises(ContractError):
                control.close()
        finally:
            token.file.close = original_close
        with pytest.raises(ContractError):
            control.close()
        if phase == 'host':
            if mode in ('no-work', 'control-nonzero', 'control-unannounced'):
                if mode != 'control-unannounced':
                    _entry_complete(control)
                if mode == 'control-nonzero':
                    os._exit(7)
                if mode == 'no-work':
                    with pytest.raises(ContractError):
                        store.put('supervision/forbidden-after-terminal.json', {'forbidden': True})
                return
            loaded = _case_loaded(directory.parent, short=mode in ('forced-cleanup', 'old-deadline') or os.environ.get('GFB_ENTRY_PREPARED_STOP_CROSS') == '1')
            config_path = root / 'training-config.json'
            config_sha = write_control_record(config_path, {'scope': 'control mechanism only, no model/learning'})
            parent = directory.parent / 'parent'
            parent.mkdir(mode=0o700)
            operation = dict(control.operation, operation_id='op-1', run_id='run-1', operation_type='hpo-train',
                config_path=str(config_path), config_sha256=config_sha, parent_directory=str(parent),
                expected_context={'campaign_id': store.get('campaign.json')[0]['campaign_id'], 'operation_id': 'op-1',
                    'kind': 'training', 'run_id': 'run-1', 'evaluation_id': None},
                execution_request={'mode': 'fresh', 'pause_after_update': None})
            operation_path = root / 'operation-op-1.json'
            operation_sha = write_control_record(operation_path, operation)
            awaiting = {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
                'campaign_id': operation['expected_context']['campaign_id'], 'proposal_no': 0, 'learning_seed': 17,
                'run_id': 'run-1', 'run_path': str(parent), 'config_path': str(config_path), 'config_sha256': config_sha,
                'runtime_binding_sha256': operation['runtime_binding_sha256'], 'operation_path': str(operation_path),
                'operation_sha256': operation_sha, 'supervision_id': control.launch['supervision_id'],
                'session_clock_sha256': control.campaign_bootstrap['session_clock_sha256'], 'created_unix': time.time()}
            awaiting_sha = store.put('supervision/' + control.launch['supervision_id'] + '/awaiting/op-1.json', awaiting)
            cursor_value, cursor_sha = store.get('supervision/current.json')
            store.update_supervision_cursor(dict(cursor_value, awaiting_path=str(root / 'awaiting/op-1.json'),
                awaiting_sha256=awaiting_sha, phase='awaiting-host'), expected_sha256=cursor_sha)
            if mode == 'transfer-before-bind':
                from gf_training.lineage import _HELD_RUN_OWNERS
                with run_lock(parent):
                    fd = _HELD_RUN_OWNERS[parent].descriptor
                    info = os.fstat(fd)
                    control._send('lock', control._next_sequence,
                        {'lock_kind': 'parent', 'device': info.st_dev, 'inode': info.st_ino}, fd=fd)
                    os._exit(7)  # Deliberate invalid ordering, no ACK expectation or fabricated handoff.
            control.bind_operation(operation_path)
            with run_lock(parent) as ownership:
                transfer_run_lock(ownership, control)
                _entry_competitor(parent / '.lock')
            producer = runtime_producer(SimpleNamespace(manifest={'executable_sha256': hashlib.sha256(
                Path('/proc/' + str(os.getppid()) + '/exe').read_bytes()).hexdigest()}), sampling_runtime_sha256='a'*64)
            admission = control.admit({'managed_ram_bytes': 1024**2, 'managed_cuda_bytes': 0, 'game_concurrency': 1,
                'player_worker_threads': 1, 'max_batch_size': 1, 'storage_bytes': 1, 'source_workspace_bytes': 0}, producer)
            cursor_value, cursor_sha = store.get('supervision/current.json')
            store.update_supervision_cursor(dict(cursor_value, current_invocation_id=admission.observation['invocation_id'],
                phase='awaiting-cold'), expected_sha256=cursor_sha)
            if mode == 'healthy-incomplete':
                _component_prepare(control, admission)
                return
            if mode == 'unexpected-eof':
                os._exit(7)
            if mode == 'report-without-arm':
                write_control_record(root / ('failure-' + admission.observation['invocation_id'] + '.json'),
                    dict(schema_version=1, kind='rust-player-training-failure-v1', **control._envelope,
                        operation_sha256=control.operation_sha256, admission_sha256=digest(admission.observation),
                        failure_class='application_error', reason='declared CPU mechanism failure'))
                os._exit(7)
            if mode == 'bad-terminal':
                path = root / ('failure-' + control._envelope['invocation_id'] + '.json')
                checksum = write_control_record(path, dict(schema_version=1, kind='rust-player-training-failure-v1',
                    **control._envelope, operation_sha256=control.operation_sha256,
                    admission_sha256=digest(admission.observation), failure_class='not-application-error', reason='invalid arm'))
                control._send('training_failed', control._next_sequence,
                    {'failure_path': str(path), 'failure_sha256': checksum})
                os._exit(7)  # Rejecting a malformed request must not require a reply.
            reason = 'declared CPU mechanism failure'
            if os.environ.get('GFB_ENTRY_FAILURE_RECORD_BOUNDARY') == '1':
                from gf_common.domain import canonical
                probe = dict(schema_version=1, kind='rust-player-training-failure-v1', **control._envelope,
                    operation_sha256=control.operation_sha256, admission_sha256=digest(admission.observation),
                    failure_class='application_error', reason='')
                available = 65536 - len(canonical(probe).encode())
                reason = '神' * (available // 3) + 'a' * (available % 3)
                assert len(canonical(dict(probe, reason=reason)).encode()) == 65536
            control.training_failed(admission, reason)
            _entry_post_terminal_race(control, 'failure')
            with pytest.raises(ContractError):
                store.put('supervision/forbidden-after-terminal.json', {'forbidden': True})
            with pytest.raises(ContractError):
                admission.require()
            if mode == 'post-failure-ack':
                control._send('malformed_after_failure', control._next_sequence, {})
            if mode == 'forced-signal':
                os.kill(os.getpid(), signal.SIGTERM)
            if mode == 'forced-cleanup':
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                while True:
                    time.sleep(.05)
            if mode == 'failure-write':
                (root / ('exit-' + control._envelope['invocation_id'] + '.json')).mkdir()
            return
        assert phase in ('campaign-return', 'cold')
        if phase == 'campaign-return':
            assert control._next_sequence == 2, 'accepted return handshake consumes seq1; normal requests begin seq2'
            payload = control.campaign_return
            evidence = {key: payload[key] for key in ('failure_path', 'failure_sha256', 'exit_path', 'exit_sha256', 'physical_path', 'physical_sha256')}
            classification = 'training_failed'
        else:
            local, prepared, exit_value = control.cold.require_pair()
            assert control.cold.physical_termination.require() == exit_value
            path = root / ('completion-' + local['invocation_id'] + '.json')
            audit_path = root / 'component-only-audit.json'
            audit_sha = write_control_record(audit_path, {'scope': 'protocol mechanism only; NO Source/Raw/Replay/Parent qualification'})
            completion = dict(schema_version=1, kind='rust-player-training-completion-v1',
                **{key: local[key] for key in ('supervision_id', 'invocation_id', 'operation_id', 'run_id', 'config_sha256', 'runtime_binding_sha256')},
                operation_sha256=control.operation_sha256, prepared_sha256=digest(prepared), exit_sha256=digest(exit_value),
                state_sha256=prepared['state_sha256'], generation_sha256=prepared['generation_sha256'],
                source_manifest_sha256=prepared['source_manifest_sha256'], state='incomplete', committed_update=0,
                audit_path=str(audit_path), audit_sha256=audit_sha, eligible=False, parent_metadata_validation=None)
            completion_sha = write_control_record(path, completion)
            control.cold.acknowledge_finalizer_output(path)
            physical_path = root / 'test-namespace' / 'jobs' / local['global_ticket'] / 'termination.json'
            from gf_training.rust_player_supervision import read_control_record
            _, physical_sha = read_control_record(physical_path, root / 'test-namespace')
            payload = dict(control.cold._payload)
            evidence = {'completion_path': str(path), 'completion_sha256': completion_sha,
                'physical_path': str(physical_path), 'physical_sha256': physical_sha}
            classification = 'healthy_incomplete'

        _entry_competitor(Path(control.operation['parent_directory']) / '.lock')
        assert payload['admission']['invocation_id'] != control._envelope['invocation_id']
        if mode == 'old-deadline':
            (root / 'return-started.json').write_text(str(control._finish()))
            time.sleep(max(0., control._finish() - time.monotonic()) + .05)
            _entry_complete(control)
            raise AssertionError('failed operation deadline was renewed')
        loaded = _case_loaded(directory.parent, short=os.environ.get('GFB_ENTRY_PREPARED_STOP_CROSS') == '1')
        study = CampaignStudy(store, loaded, store.get('campaign.json')[0])
        intent, _ = study._intent(0)
        proposal = study._proposal(0, intent)
        reason = 'declared CPU mechanism failure'
        if phase == 'campaign-return':
            from gf_training.rust_player_supervision import read_control_record
            reason = read_control_record(Path(payload['failure_path']), root)[0]['reason']
        failure_sha = store.put('proposals/000000/failure.json', {'schema_version': 1, 'kind': 'hpo-proposal-failure',
            'proposal_sha256': proposal.sha256, 'reason': reason, 'boundary': 'CPU protocol mechanism only'})
        study.fail(proposal, reason)  # REAL local Optuna FAIL; never a learned parent/objective.
        if mode == 'after-fail':
            os._exit(9)
        with study.study(0) as actual_study:
            trial = actual_study.get_trials()[proposal.trial_number]
            assert trial.state.name == 'FAIL' and trial.value is None
        awaiting_path = root / 'awaiting/op-1.json'
        awaiting, awaiting_sha = store.get(str(awaiting_path.relative_to(directory)))
        disposition = {'schema_version': 1, 'kind': 'rust-player-campaign-terminal-disposition-v1',
            'campaign_id': awaiting['campaign_id'], 'proposal_no': 0, 'proposal_sha256': proposal.sha256, 'learning_seed': 17,
            'supervision_id': control.launch['supervision_id'], 'session_clock_sha256': awaiting['session_clock_sha256'],
            'awaiting_path': str(awaiting_path), 'awaiting_sha256': awaiting_sha, 'operation_path': str(control.operation_path),
            'operation_sha256': control.operation_sha256, 'invocation_id': payload['admission']['invocation_id'],
            'classification': classification, 'evidence': evidence,
            'proposal_failure_path': str(directory / 'proposals/000000/failure.json'), 'proposal_failure_sha256': failure_sha,
            'tell_intent_path': str(directory / 'proposals/000000/tell.intent.json'),
            'tell_intent_sha256': store.get('proposals/000000/tell.intent.json')[1],
            'tell_receipt_path': str(directory / 'proposals/000000/tell.receipt.json'),
            'tell_receipt_sha256': store.get('proposals/000000/tell.receipt.json')[1]}
        disposition_path = root / 'terminal/op-1.json'
        store.put(str(disposition_path.relative_to(directory)), disposition)
        if mode == 'before-disposition':
            os._exit(9)
        if mode == 'bad-disposition':
            disposition_path.write_text('{}')
        if mode == 'ack-write':
            (root / ('disposition-ack-' + disposition['invocation_id'] + '.json')).mkdir()
        with borrow_run_lock(Path(control.operation['parent_directory']), control):
            with pytest.raises(ContractError):
                control.release_parent()
            if mode == 'ack-loss':
                checksum = store.get(str(disposition_path.relative_to(directory)))[1]
                control._send('campaign_disposition', control._next_sequence,
                    {'disposition_path': str(disposition_path), 'disposition_sha256': checksum})
                ack_path = root / ('disposition-ack-' + disposition['invocation_id'] + '.json')
                while not ack_path.exists():
                    assert time.monotonic() < control._finish()
                    time.sleep(.002)
                from gf_training.rust_player_supervision import read_control_record
                ack, _ = read_control_record(ack_path, root)
                records.validate_campaign_disposition_ack(ack)
                assert ack['disposition_sha256'] == checksum
                os._exit(9)  # Native durable ACK exists; response was NEVER received.
            if mode == 'ack-write':
                checksum = store.get(str(disposition_path.relative_to(directory)))[1]
                control._send('campaign_disposition', control._next_sequence,
                    {'disposition_path': str(disposition_path), 'disposition_sha256': checksum})
                os._exit(7)  # Native publication must fail, no success response is expected.
            ack = control.campaign_disposition(disposition_path)
            assert ack['control_invocation_id'] == control._envelope['invocation_id']
            with pytest.raises(ContractError):
                control.release_parent()
        control.release_parent()
        write_control_record(root / 'parent-released.json', {'operation_sha256': control.operation_sha256})
        # Proposal ACK leaves SAME campaign control-store live for actual planning.
        store.read_acknowledged_supervision_disposition(str(disposition_path.relative_to(directory)))
        cursor_value, cursor_sha = store.get('supervision/current.json')
        cursor_sha = store.update_supervision_cursor(dict(cursor_value, awaiting_path=None, awaiting_sha256=None,
            current_invocation_id=None, phase='control'), expected_sha256=cursor_sha)
        write_control_record(root / 'cursor-advanced.json', {'cursor_sha256': cursor_sha})
        if mode == 'next-bind':
            following = dict(control.operation, operation_id='op-2', run_id='run-2',
                parent_directory=str(directory.parent / 'parent-2'),
                expected_context=dict(control.operation['expected_context'], operation_id='op-2', run_id='run-2'))
            (directory.parent / 'parent-2').mkdir(mode=0o700)
            following_path = root / 'operation-op-2.json'
            write_control_record(following_path, following)
            control.bind_operation(following_path)
            assert control._campaign_return is None and control._disposition_ack is None
            (root / 'distinct-bound.json').write_text(control.operation_sha256)
            with pytest.raises(ContractError):
                control.release_parent()
            with pytest.raises(ContractError):
                _entry_complete(control)
            return  # deliberate unannounced exit: no fake second admitted operation.
        if phase != 'cold':
            _entry_complete(control)
    control.close()




def test_seed_snapshot_preserves_existing_informational_origin_identities():
    row = {'path': '/already-validated.json', 'sha256': 'a'*64,
        'settings': {'model': {}, 'training': {}, 'identity': 'b'*64},
        'origin': {'campaign_id': 'historical campaign', 'study_identity': 'older-study', 'proposal_sha256': 'informational-origin'}}
    snapshot = {'schema_version': 1, 'kind': 'rust-player-seed-settings-v1', 'sources': [row]}
    assert records.validate_seed_settings(snapshot) == snapshot




def test_generated_component_does_not_invent_a_smaller_filename_limit():
    assert records.safe_component('a' * 200) == 'a' * 200
    for value in ('../escape', '', 'x/y'):
        with pytest.raises(ContractError):
            records.safe_component(value)


@pytest.mark.parametrize('failure', ['replace', 'directory-sync'])
def test_cursor_publication_failure_poisoned_owner_keeps_orphan_immutable(tmp_path, monkeypatch, failure):
    from gf_cli import hpo_storage
    directory = tmp_path / 'campaign'
    store = hpo_storage.CampaignStore(directory, mode='fresh', max_owned_bytes=1024**2)
    before = store.update_supervision_cursor(cursor(), expected_sha256=None)
    orphan = 'supervision/' + 'a' * 32 + '/awaiting/orphan.json'
    orphan_sha = store.put(orphan, {'scope': 'immutable orphan DATA; no dispatch authority'})
    replaced = []
    original_replace, original_sync = hpo_storage.os.replace, hpo_storage._sync
    def replace(*args):
        if failure == 'replace':
            raise OSError('injected cursor publication failure')
        original_replace(*args)
        replaced.append(True)
    def sync(path):
        if replaced:
            raise OSError('injected cursor publication durability failure')
        original_sync(path)
    with monkeypatch.context() as patch:
        patch.setattr(hpo_storage.os, 'replace', replace)
        patch.setattr(hpo_storage, '_sync', sync)
        with pytest.raises(OSError):
            store.update_supervision_cursor(dict(cursor(), phase='finalized'), expected_sha256=before)
        with pytest.raises(ContractError):
            store.get(orphan)
    store.close()
    with hpo_storage.CampaignStore(directory, mode='resume', max_owned_bytes=1024**2) as fresh:
        assert fresh.get(orphan)[1] == orphan_sha
        assert fresh.get('supervision/current.json')[0]['phase'] == ('control' if failure == 'replace' else 'finalized')


def _failure_record_data(reason):
    from tests.test_rust_player_supervisor import _new_namespace_records
    global_admission, _ = _new_namespace_records()
    admission = global_admission['host_admission']
    return dict(schema_version=1, kind='rust-player-training-failure-v1',
        **{key: admission[key] for key in ('supervision_id', 'invocation_id', 'nonce')},
        operation_sha256='d' * 64, admission_sha256=digest(admission),
        failure_class='application_error', reason=reason), admission


def test_failure_record_accepts_4097_ascii_chars_within_whole_envelope():
    from gf_common.domain import canonical
    value, admission = _failure_record_data('a' * 4097)
    assert len(canonical(value).encode()) == 4524
    assert records.validate_training_failure(value, admission, 'd' * 64) == value


@pytest.mark.parametrize('unit', ['a', '神', '\n', '\\', '"'])
def test_failure_whole_canonical_utf8_boundary_and_specific_io(tmp_path, unit):
    from gf_common.domain import canonical
    from gf_training.rust_player_supervision import read_control_record, write_control_record
    value, admission = _failure_record_data('')
    overhead = len(canonical(value).encode())
    width = len(canonical(unit).encode()) - 2
    count, remainder = divmod(65536 - overhead, width)
    value['reason'] = unit * count + 'a' * remainder
    assert len(canonical(value).encode()) == 65536
    records.validate_training_failure(value, admission, 'd' * 64)
    path = tmp_path / 'failure.json'
    expected = write_control_record(path, value)
    assert read_control_record(path, tmp_path) == (value, expected)
    overflow = dict(value, reason=value['reason'] + 'a')
    assert len(canonical(overflow).encode()) == 65537
    with pytest.raises(ContractError):
        records.validate_training_failure(overflow, admission, 'd' * 64)
    with pytest.raises(ContractError):
        write_control_record(tmp_path / 'overflow.json', overflow)
    path.write_bytes(canonical(overflow).encode())
    with pytest.raises(ContractError):
        read_control_record(path, tmp_path)


def test_failure_record_limit_does_not_cap_unrelated_control_files(tmp_path):
    from gf_training.rust_player_supervision import read_control_record, write_control_record
    value = {'schema_version': 1, 'kind': 'rust-player-loaded-campaign-v1', 'data': 'x' * 70000}
    expected = write_control_record(tmp_path / 'loaded-campaign.json', value)
    assert read_control_record(tmp_path / 'loaded-campaign.json', tmp_path) == (value, expected)


def test_nonowning_campaign_wrapper_preserves_raw_lifetime_on_failure_and_close(tmp_path):
    import os
    original = (tmp_path / 'actual-file').open('w+b')
    fd = os.dup(original.fileno())
    identity = os.fstat(fd)
    try:
        with pytest.raises(ValueError):
            os.fdopen(fd, 'r+b', encoding='utf8', closefd=False)
        assert os.fstat(fd).st_ino == identity.st_ino
        wrapper = os.fdopen(fd, 'r+b', closefd=False)
        wrapper.close()
        assert os.fstat(fd).st_ino == identity.st_ino
    finally:
        os.close(fd)  # FIRST known raw close, not a retry after uncertainty.
        original.close()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '_prepare':
        _prepare_entry_case(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == '_host':
        _entry_host(*sys.argv[2:])
    else:
        raise SystemExit('fixed private CPU protocol fixture only')
