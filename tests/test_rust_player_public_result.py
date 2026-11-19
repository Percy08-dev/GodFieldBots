"""Cold stdout DATA and cleanup-edge tests; no fake Source/Parent authority."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.rust_player import deadline_bits
from tests.test_rust_player_clock import anchor
from tests.test_rust_player_public_entry import normal


@pytest.mark.parametrize('status', ['complete', 'paused', 'incomplete'])
def test_summary_uses_only_actual_finalized_data_and_copies_before_owner_retirement(status):
    from gf_cli.rust_player_entry import training_summary_data
    state = {'status': status, 'updates': 1, 'accepted_decisions': 17,
        'generation_sha256': 'g' * 64, 'created_unix': 10., 'last_unix': 20.,
        'runtime_binding_sha256': 'r' * 64}
    completion = {'state': status, 'committed_update': 1, 'generation_sha256': 'g' * 64,
        'audit_path': '/control/audit.json', 'audit_sha256': 'a' * 64, 'eligible': status == 'complete'}
    value = training_summary_data(state, completion, directory=Path('/parent'),
        completion_path=Path('/control/completion.json'), completion_sha256='c' * 64, clock_reference={})
    output = json.loads(value)
    assert output['status'] == status and output['updates'] == 1
    assert output['rust_player']['eligible'] is (status == 'complete')
    assert output['rust_player']['completion_sha256'] == 'c' * 64
    state.clear(); completion.clear()
    assert json.loads(value)['accepted_decisions'] == 17


def test_clock_provenance_is_checked_data_after_execution_but_with_original_session_bound(tmp_path, monkeypatch):
    from gf_cli import rust_player_entry, rust_player_host
    from gf_common.rust_player_clock import clock_deadlines, operation_clock
    from gf_training.rust_player_supervision import write_control_record
    loaded = normal()
    timers = {key: getattr(loaded.config.run, key) for key in
        ('admission_seconds', 'execution_seconds', 'wall_seconds', 'finalization_seconds')}
    clock_binding = anchor()
    op = {'schema_version': 2, 'kind': 'rust-player-operation-v2', 'operation_id': 'a' * 32,
        'run_id': 'run', 'mode': 'ordinary', 'operation_type': 'ordinary-train',
        'config_path': '/parent/effective.json', 'config_sha256': loaded.sha256,
        'runtime_binding_sha256': 'b' * 64, 'parent_directory': '/parent', 'expected_context': None,
        'execution_request': {'mode': 'fresh', 'pause_after_update': None},
        'deadlines': clock_deadlines(clock_binding, inputs=timers, post_host_seconds=30.)}
    row = operation_clock(op, clock_binding=clock_binding, inputs=timers, post_host_seconds=30.)
    path = tmp_path / ('clock-' + op['operation_id'] + '.json')
    sha = write_control_record(path, row)
    control = SimpleNamespace(root=tmp_path, operation=op, operation_sha256=digest(op),
        launch={'clock_binding': clock_binding, 'session_deadlines': {'finish_bits': op['deadlines']['finish_bits']}})
    monkeypatch.setattr(rust_player_entry, '_boot_id', lambda: 'boot')
    def forbidden():
        raise AssertionError('cold provenance used a fresh/prelaunch execution clock')
    monkeypatch.setattr(rust_player_entry.time, 'monotonic', forbidden)
    assert rust_player_host._clock_reference(control, loaded) == {'operation_clock_path': str(path), 'operation_clock_sha256': sha}
    control.launch['session_deadlines']['finish_bits'] = deadline_bits(101.)
    with pytest.raises(ContractError):
        rust_player_host._clock_reference(control, loaded)


@pytest.mark.parametrize('fault', [None, 'close', 'release'])
def test_cold_dispatch_copies_only_after_positive_owner_close_and_release(tmp_path, monkeypatch, capsys, fault):
    from gf_cli import rust_player_host
    from gf_training import rust_player_learning
    calls = []
    fd = os.open(tmp_path / 'data', os.O_CREAT | os.O_RDWR, 0o600)
    # Real FD cleanup I/O receiver only; actual typed adapter is separately
    # replaced with fixed unqualified DATA. This object grants no audit/capability.
    class Receiver:
        def close(self):
            calls.append('close')
            if fault == 'close':
                raise OSError('actual FD close not confirmed')
            os.close(fd)
    receiver = Receiver()
    operation = {'operation_type': 'ordinary-train', 'config_path': '/unused', 'config_sha256': 'a' * 64}
    def release():
        calls.append('release')
        if fault == 'release': raise OSError('parent release not confirmed')
    control = SimpleNamespace(initial_operation=operation, operation=operation, cold=object(), release_parent=release)
    monkeypatch.setattr(rust_player_host, '_cold_output', lambda *args: '{"status":"paused","scope":"DATA-only"}')
    monkeypatch.setattr(rust_player_learning, 'finalize_cold', lambda *args, **kwargs: receiver)
    from gf_cli import config
    monkeypatch.setattr(config, 'load_rust_player_effective', lambda *args, **kwargs: object())
    try:
        if fault:
            with pytest.raises(OSError) as failed:
                rust_player_host._dispatch(control, 'cold')
            assert failed.value.rust_player_result_owner is receiver
            assert not capsys.readouterr().out
        else:
            output = rust_player_host._dispatch(control, 'cold')
            assert json.loads(output)['status'] == 'paused'
            assert not capsys.readouterr().out
        assert calls == (['close'] if fault == 'close' else ['close', 'release'])
    finally:
        if fault == 'close': os.close(fd)


@pytest.mark.parametrize('control_fault', [False, True])
def test_bootstrap_emits_cold_data_only_after_positive_control_close(tmp_path, monkeypatch, capsys, control_fault):
    from gf_cli import config, rust_player_host
    from gf_training import rust_player_learning
    calls = []
    class Receiver:
        def close(self): calls.append('result-close')
    receiver = Receiver()
    operation = {'operation_type': 'ordinary-train', 'config_path': '/unused', 'config_sha256': 'a' * 64}
    class Control:
        initial_operation = operation
        cold = object()
        def release_parent(self): calls.append('release')
        def close(self):
            assert not capsys.readouterr().out, 'stdout preceded control cleanup'
            calls.append('control-close')
            if control_fault: raise OSError('actual control cleanup remains uncertain')
    control = Control()
    control.operation = operation
    monkeypatch.setattr(rust_player_host, 'HostControl', lambda *args, **kwargs: control)
    monkeypatch.setattr(rust_player_host, '_cold_output', lambda *args: '{"status":"paused","scope":"DATA-only"}')
    monkeypatch.setattr(rust_player_learning, 'finalize_cold', lambda *args, **kwargs: receiver)
    monkeypatch.setattr(config, 'load_rust_player_effective', lambda *args, **kwargs: object())
    import sys
    monkeypatch.setattr(sys, 'argv', ['host', '--control-fd', '1', '--launch-fd', '2', '--phase', 'cold'])
    assert rust_player_host.main() == int(control_fault)
    output = capsys.readouterr()
    if control_fault: assert not output.out
    else: assert json.loads(output.out)['status'] == 'paused'
    assert calls == ['result-close', 'release', 'control-close']


def test_bootstrap_retains_original_result_and_inner_error_owners_through_control_cleanup(tmp_path, monkeypatch, capsys):
    import gc
    import sys
    import weakref

    from gf_cli import config, rust_player_host
    from gf_training import rust_player_learning
    references, cleanup = [], []
    class Receiver:
        def __init__(self):
            self.inner = SimpleNamespace(file=(tmp_path / 'actual-data').open('w+b'))
        def close(self):
            error = OSError('actual data close remains uncertain')
            error.cleanup_owner = self.inner
            self.error = error
            raise error
    def returned(*args, **kwargs):
        receiver = Receiver()
        references.append(weakref.ref(receiver))
        return receiver
    operation = {'operation_type': 'ordinary-train', 'config_path': '/unused', 'config_sha256': 'a' * 64}
    class Control:
        initial_operation = operation
        cold = object()
        def release_parent(self): raise AssertionError('uncertain owner was released')
        def close(self):
            gc.collect()
            receiver = references[0]()
            assert receiver is not None
            assert receiver.error.cleanup_owner is receiver.inner
            assert receiver.error.rust_player_result_owner is receiver
            os.fstat(receiver.inner.file.fileno())
            cleanup.append(receiver.inner.file)
    control = Control()
    control.operation = operation
    monkeypatch.setattr(rust_player_host, 'HostControl', lambda *args, **kwargs: control)
    monkeypatch.setattr(rust_player_host, '_cold_output', lambda *args: '{"status":"paused"}')
    monkeypatch.setattr(rust_player_learning, 'finalize_cold', returned)
    monkeypatch.setattr(config, 'load_rust_player_effective', lambda *args, **kwargs: object())
    monkeypatch.setattr(sys, 'argv', ['host', '--control-fd', '1', '--launch-fd', '2', '--phase', 'cold'])
    try:
        assert rust_player_host.main() == 1
        assert cleanup and not capsys.readouterr().out
    finally:
        for file in cleanup: file.close()


def test_cold_output_refuses_metadata_instead_of_actual_typed_returned_owner():
    from gf_cli.rust_player_host import _cold_output
    with pytest.raises(ContractError, match='actual returned'):
        _cold_output(object(), SimpleNamespace(state={'status': 'complete'}), object())
