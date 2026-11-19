import threading

import pytest

from gf_simulator_client.performance_profile import ForwardProfile, ProfileStatus


class Profiler:
    def __init__(self):
        self.events = []
    def start(self):
        self.events.append(('start', threading.get_native_id()))
    def stop(self):
        self.events.append(('stop', threading.get_native_id()))
    def export_chrome_trace(self, path):
        with open(path, 'w') as f:
            f.write('{"traceEvents":[]}')


def test_actual_worker_post1_post3_and_boundary_rows(tmp_path):
    status = ProfileStatus.local()
    status.begin({'run_id': 'r', 'operation_id': 'op', 'invocation_id': 'i', 'batch_id': 'b', 'phase': 'collection'})
    p = Profiler()
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference', profiler_factory=lambda: p, torch_version='test')
    tids = []
    def run():
        tids.append(threading.get_native_id())
        for rows in [2, 4, 3, 9]:
            callback.after_forward(scored_rows=rows, boundary='legacy-post-transport-forward')
    t = threading.Thread(target=run)
    t.start(); t.join()
    result = status.read()['result']
    assert result['complete'] and result['callbacks_seen'] == 3
    assert result['worker_tid'] == tids[0] != threading.get_native_id()
    assert result['scored_rows'] == [2, 4, 3]
    assert result['trace_bytes'] == 18 and len(result['trace_sha256']) == 64
    assert result['trace_clock'] == 'torch-profiler-native'
    assert p.events == [('start', tids[0]), ('stop', tids[0])]


def test_scope_end_is_terminal_later_callback_only_cleans(tmp_path):
    status = ProfileStatus.local()
    status.begin({'phase': 'collection', 'batch_id': 'b1'})
    p = Profiler()
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference', profiler_factory=lambda: p, torch_version='test')
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    status.end()
    assert not status.read()['result']['complete']
    status.begin({'phase': 'collection', 'batch_id': 'b2'})
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    result = status.read()['result']
    assert not result['complete'] and result['callbacks_seen'] == 1
    assert result['scored_rows'] == [1, None, None]
    assert len(p.events) == 2 and not list(tmp_path.glob('*.trace.json'))


def test_foreign_thread_cannot_stop_worker_profiler(tmp_path):
    status = ProfileStatus.local(); status.begin({'phase': 'collection'})
    p = Profiler()
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference', profiler_factory=lambda: p, torch_version='test')
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    t = threading.Thread(target=lambda: callback.after_forward(scored_rows=1, boundary='native-post-row-forward'))
    t.start(); t.join()
    assert not status.read()['result']['complete']
    assert len(p.events) == 1
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    assert len(p.events) == 2


def test_diagnostic_status_failure_cannot_escape_successful_forward(tmp_path):
    class BrokenStatus:
        def read(self):
            return {'state': 'active', 'scope': {'phase': 'collection'}, 'result': None}
        def publish(self, result, **kwargs):
            raise OSError('diagnostic shared status lost')
    callback = ForwardProfile(tmp_path, 'obs', BrokenStatus(), role='native-inference',
        profiler_factory=Profiler, torch_version='test')
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')


def test_trace_replacement_cannot_qualify_export(tmp_path):
    status = ProfileStatus.local(); status.begin({'phase': 'collection'})
    class Replacing(Profiler):
        def export_chrome_trace(self, path):
            super().export_chrome_trace(path)
            name = next(tmp_path.glob('*.trace.json'))
            name.unlink()
            name.symlink_to('/dev/null')
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference',
        profiler_factory=Replacing, torch_version='test')
    for _ in range(3):
        callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    result = status.read()['result']
    assert not result['complete'] and result['trace_sha256'] is None
    assert result['errors']


def test_scope_end_is_nonblocking_even_when_shared_status_lock_is_unavailable(tmp_path):
    status = ProfileStatus.local()
    status.begin({'phase': 'collection'})
    status.lock = threading.Lock()
    status.lock.acquire()
    try:
        status.end()
        assert status.terminal.value == 1
        assert status.read()['state'] == 'ended'
    finally:
        status.lock.release()


def test_trace_close_failure_retains_original_fd_and_incomplete_status(tmp_path, monkeypatch):
    import os

    from gf_simulator_client import performance_profile as module
    status = ProfileStatus.local(); status.begin({'phase': 'collection'})
    real_close = os.close
    failed = []
    def close(fd):
        if not failed and os.readlink(f'/proc/self/fd/{fd}').endswith('.trace.json'):
            failed.append(fd)
            raise OSError('diagnostic close fault')
        real_close(fd)
    monkeypatch.setattr(module.os, 'close', close)
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference',
        profiler_factory=Profiler, torch_version='test')
    for _ in range(3):
        callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    assert not status.read()['result']['complete']
    try:
        assert callback._trace_fd == failed[0]
        assert os.fstat(callback._trace_fd).st_size == 18
    finally:
        real_close(failed[0])


def test_no_callback_result_has_no_fabricated_tid_clock_or_trace():
    from gf_simulator_client.performance_profile import missing_profile_result
    r = missing_profile_result(observation_id='obs', owner={'pid': 1}, clock={'name': 'CLOCK_MONOTONIC'},
        scope={'phase': 'collection'}, boundary='native-post-row-forward', torch_version='test')
    assert not r['complete'] and r['callbacks_seen'] == 0
    assert r['scored_rows'] == [None, None, None]
    for name in ('worker_tid', 'start_monotonic_ns', 'end_monotonic_ns', 'trace_path', 'trace_bytes', 'trace_sha256'):
        assert r[name] is None


def test_typed_accelerator_execution_failure_preserves_existing_runtime_failure(tmp_path, monkeypatch):
    import pytest
    import torch
    status = ProfileStatus.local(); status.begin({'phase': 'collection'})
    class FailedAccelerator(Profiler):
        def start(self):
            raise torch.cuda.AcceleratorError('typed execution failure; no CUDA work performed')
    monkeypatch.setattr(torch.profiler, 'profile', lambda **kwargs: FailedAccelerator())
    callback = ForwardProfile(tmp_path, 'obs', status, role='native-inference')
    with pytest.raises(torch.cuda.AcceleratorError):
        callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    assert not status.read()['result']['complete']


@pytest.mark.parametrize('fault', ['snapshot', 'decode', 'publication'])
def test_fix1_t2_logical_profile_end_survives_native_snapshot_failure(tmp_path, monkeypatch, fault):
    import json
    from types import SimpleNamespace

    from gf_common.performance_metrics import PhaseMetrics
    from gf_simulator_client.performance_diagnostics import Diagnostics
    from gf_simulator_client.rust_player import RustPlayerRuntime
    d = Diagnostics(tmp_path, 'fix1-profile', profile_recipe='forward-window-v1')
    first = d.scope(phase='collection', batch_id='first')
    second = d.scope(phase='collection', batch_id='second')
    status = ProfileStatus.local(); status.begin(first)
    profiler = Profiler()
    callback = ForwardProfile(tmp_path, 'fix1-profile', status, role='native-inference',
        profiler_factory=lambda: profiler, torch_version='test')
    callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
    row = json.dumps(PhaseMetrics(scope=first, start_ns=0).snapshot(end_ns=1)).encode()
    def snapshot(*args):
        if fault == 'snapshot':
            raise RuntimeError('snapshot failed')
        return b'invalid JSON' if fault == 'decode' else row
    owner = RustPlayerRuntime.__new__(RustPlayerRuntime)
    owner.diagnostics, owner.service = d, None
    owner._measurement_scope, owner._measurement_stages = first, None
    owner._profile_status, owner._profile_reported = status, False
    owner._settled = threading.Condition()
    owner._active_routes, owner._holds = {}, {}
    owner._native = SimpleNamespace(measurement_end=snapshot, measurement_begin=lambda scope: None)
    if fault == 'publication':
        def broken_publication(*args, **kwargs):
            raise OSError('diagnostic publication failed')
        monkeypatch.setattr(d, 'metrics', broken_publication)
    try:
        owner.measurement_end()
        assert status.terminal.value == 1
        owner.measurement_begin(second)
        callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
        callback.after_forward(scored_rows=1, boundary='native-post-row-forward')
        result = status.read()['result']
        assert not result['complete'] and result['callbacks_seen'] == 1
        assert result['scope'] == first and result['scored_rows'] == [1, None, None]
        assert not list(tmp_path.glob('*.trace.json'))
        assert [event[0] for event in profiler.events] == ['start', 'stop']
    finally:
        d.close()
