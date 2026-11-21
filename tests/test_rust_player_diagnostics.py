import json
import os

import pytest

from gf_simulator_client.performance_diagnostics import Diagnostics, diagnostics_from_environment


def test_disabled_pair_allocates_nothing():
    assert diagnostics_from_environment({}) is None
    assert diagnostics_from_environment({'GF_BOTS_PERFORMANCE_PROFILE': 'off'}) is None


@pytest.mark.parametrize('env', [
    {'GF_BOTS_PERFORMANCE_OBSERVATION_ID': 'x'},
    {'GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR': '/tmp'},
    {'GF_BOTS_PERFORMANCE_PROFILE': 'forward-window-v1'},
    {'GF_BOTS_PERFORMANCE_PROFILE': 'unknown'},
])
def test_partial_or_invalid_setup_refuses(env):
    with pytest.raises(ValueError):
        diagnostics_from_environment(env)


def test_exclusive_closed_phase_and_metrics_files(tmp_path):
    d = Diagnostics(tmp_path, 'observation')
    scope = d.scope(run_id='r', batch_id='b', phase='collection')
    d.phase('phase_start', scope, monotonic_ns=5)
    row = json.loads(next(tmp_path.glob('*.json')).read_bytes())
    assert set(row) == {'schema_version', 'kind', 'observation_id', 'owner', 'clock', 'scope', 'monotonic_ns', 'event', 'links'}
    assert row['clock']['name'] == 'CLOCK_MONOTONIC'
    assert row['owner']['pid'] == os.getpid() and row['owner']['role'] == 'host'
    assert row['monotonic_ns'] == 5 and row['scope']['batch_id'] == 'b'
    assert set(row['links']) == {'operation_sha256', 'prepared_sha256', 'exit_sha256', 'completion_sha256'}
    assert d.errors == []
    d.close()


def test_symlink_directory_refused_and_flush_failure_is_only_diagnostic(tmp_path):
    link = tmp_path / 'link'
    actual = tmp_path / 'actual'
    actual.mkdir()
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(OSError):
        Diagnostics(link, 'observation')
    d = Diagnostics(actual, 'observation')
    d.close()
    assert d.phase('phase_end', d.scope(phase='collection')) is None
    assert d.errors


def test_native_measurement_begin_end_reset_and_all_pyo3_classes():
    import gf_native
    config = {'scheduler': 'fixed', 'player_worker_threads': 1, 'game_concurrency': 1,
        'max_batch_size': 1, 'formation_wait_ms': 0.0, 'storage_bytes': 1048576,
        'storage_queue': 1, 'runtime_identity': 'test'}
    runtime = gf_native.PlayerRuntime(json.dumps(config).encode())
    scope = Diagnostics.scope(run_id='run', batch_id='one', phase='collection')
    runtime.measurement_begin(json.dumps(scope).encode())
    with pytest.raises(ValueError):
        runtime.measurement_begin(json.dumps(scope).encode())
    account = runtime.recording_account()
    _ = account.limit
    lease = account.reserve(10)
    assert lease.size == 10
    lease.resize(11)
    lease.close()
    row = json.loads(runtime.measurement_end(json.dumps(scope).encode()))
    assert row['complete'] and row['scope']['batch_id'] == 'one'
    assert row['counters']['pyo3_other'] == 9
    runtime.measurement_begin(json.dumps(scope | {'batch_id': 'two'}).encode())
    second = json.loads(runtime.measurement_end(json.dumps(scope | {'batch_id': 'two'}).encode()))
    assert second['counters']['pyo3_other'] == 2
    runtime.join()


def test_invalid_optional_timing_never_invalidates_otherwise_valid_action():
    from gf_common.performance_metrics import PhaseMetrics
    from gf_simulator_client.performance_diagnostics import record_accepted_reply
    from gf_simulator_client.structured_worker import _timings
    for ready in (None, True, -1, 'bad', 101):
        timings = {'generation_ms': 0, 'policy_ms': 1, 'worker_decision_ms': 2, 'shared_numeric_ready_ns': ready}
        _timings(timings)
        m = PhaseMetrics(scope={}, start_ns=0)
        record_accepted_reply(m, timings, end_ns=100)
        assert not m.snapshot(end_ns=100)['complete']
        assert m.counters['inference_successes'] == 1
        assert m.histograms['inference_roundtrip'].count == 0
    m = PhaseMetrics(scope={}, start_ns=0)
    record_accepted_reply(m, {'shared_numeric_ready_ns': 20}, end_ns=100)
    assert m.counters['inference_successes'] == 1
    assert m.histograms['inference_roundtrip'].snapshot()['min_ns'] == 80


def test_logical_events_pages_include_empty_pages_and_transport_failures():
    import asyncio

    import httpx

    from gf_common.performance_metrics import PhaseMetrics
    from gf_simulator_client.http import AttestedClient, DeliveryUnknown
    attempts = 0
    def reply(request):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise httpx.ConnectError('test', request=request)
        return httpx.Response(200, json={'events': [], 'next_cursor': str(attempts), 'has_more': attempts == 1})
    async def run():
        client = AttestedClient('http://127.0.0.1:8000', expected_fingerprint='x', fixed_seed_experiment=True,
            expected_effective={}, transport=httpx.MockTransport(reply))
        client.cached = {}
        client.performance_metrics = PhaseMetrics(scope={}, start_ns=0)
        rows, cursor = await client.events('game', 1, None)
        assert rows == [] and cursor == '2'
        assert client.performance_metrics.counters['event_page_successes'] == 2
        assert client.performance_metrics.histograms['event_page'].count == 2
        with pytest.raises(DeliveryUnknown):
            await client.events('game', 1, cursor)
        assert client.performance_metrics.counters['event_page_failures'] == 1
        assert client.performance_metrics.counters['event_transport_attempts'] == 3
        await client.close()
    asyncio.run(run())


def test_standalone_dispatcher_counts_polls_and_failed_submit():
    import gf_native
    dispatcher = gf_native.PlayerInferenceDispatcher(1, 1, 0.0, 'test')
    scope = json.dumps(Diagnostics.scope(phase='collection', batch_id='test')).encode()
    dispatcher.measurement_begin(scope)
    with pytest.raises(TimeoutError):
        dispatcher.next_batch(0)
    with pytest.raises(ValueError):
        dispatcher.submit_results(0, [])
    dispatcher.checkpoint_references()
    row = json.loads(dispatcher.measurement_end(scope))
    assert row['counters']['pyo3_next_batch'] == 1
    assert row['counters']['pyo3_submit_results'] == 1
    assert row['counters']['pyo3_other'] == 3
    dispatcher.close()


def test_diagnostic_close_loss_is_contained_and_original_fd_retained(tmp_path, monkeypatch):
    from gf_simulator_client import performance_diagnostics as module
    d = Diagnostics(tmp_path, 'close-fault')
    real_close = os.close
    failed = []
    def close(fd):
        if not failed and os.readlink(f'/proc/self/fd/{fd}').endswith('.json'):
            failed.append(fd)
            raise OSError('diagnostic close loss')
        real_close(fd)
    monkeypatch.setattr(module.os, 'close', close)
    try:
        d.phase('phase_start', d.scope(phase='collection'))
        assert d.errors and failed[0] in d._retained_fds
        assert d.phase('phase_end', d.scope(phase='collection')) is None
    finally:
        if failed:
            real_close(failed[0])
        d.close()


_CPU_PROBE = pytest.mark.skipif(os.environ.get('GF_BOTS_RUN_PERFORMANCE_CPU_PROBE') != '1',
    reason='requires Root-approved bounded actual CPU diagnostic mechanism budget')


def _cpu_probe_scope(tmp_path, monkeypatch, *, legacy=False):
    from gf_simulator_client.performance_diagnostics import host_diagnostics
    folder = tmp_path / 'diagnostics'
    folder.mkdir()
    monkeypatch.setenv('GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR', str(folder))
    monkeypatch.setenv('GF_BOTS_PERFORMANCE_OBSERVATION_ID', 'bounded-cpu-probe')
    monkeypatch.setenv('GF_BOTS_PERFORMANCE_PROFILE', 'off')
    host_diagnostics.cache_clear()
    d = host_diagnostics()
    d.current_scope = d.scope(run_id='a'*32 if legacy else 'run',
        invocation_id='a'*32 if legacy else None, batch_id='trial', phase='collection')
    return d


def _cpu_probe_finish(d):
    from gf_simulator_client.performance_diagnostics import host_diagnostics
    d.close()
    host_diagnostics.cache_clear()


@_CPU_PROBE
def test_cpu_probe_legacy_private_endpoint_full_window(tmp_path, monkeypatch):
    from gf_simulator_client.shared_service import SharedInferenceService
    from tests.test_shared_inference_actor import (
        test_real_cpu_participant_uses_private_broker_scores_without_weights,
    )
    d = _cpu_probe_scope(tmp_path, monkeypatch, legacy=True)
    original_publish, original_close = SharedInferenceService.publish, SharedInferenceService.close
    def publish(service, *args, **kwargs):
        result = original_publish(service, *args, **kwargs)
        service.measurement_begin(d.current_scope)
        return result
    def close(service, *args, **kwargs):
        if not service.closed:
            service.quiesce()
            service.measurement_end()
        return original_close(service, *args, **kwargs)
    monkeypatch.setattr(SharedInferenceService, 'publish', publish)
    monkeypatch.setattr(SharedInferenceService, 'close', close)
    try:
        test_real_cpu_participant_uses_private_broker_scores_without_weights(tmp_path)
        records = [json.loads(path.read_bytes()) for path in d.directory.glob('*.json')]
        broker = next(r for r in records if r['owner']['role'] == 'legacy-shared-broker')
        participant = next(r for r in records if r['histograms']['inference_roundtrip'] is not None)
        assert broker['complete'] and broker['batch_rows']['batch_size_counts'] == {'1': 1}
        assert broker['batch_rows']['issued_requests'] == broker['batch_rows']['completed_responses'] == 1
        assert participant['complete'] and participant['histograms']['inference_roundtrip']['count'] == 1
        assert not d.errors
    finally:
        _cpu_probe_finish(d)


@_CPU_PROBE
@pytest.mark.parametrize('scheduler', ['tokio', 'fixed'])
@pytest.mark.parametrize('workers', [1, 2])
def test_cpu_probe_native_two_games_same_dispatcher(tmp_path, monkeypatch, scheduler, workers):
    from tests.test_rust_player_runtime import (
        assert_source_match,
        test_actual_mlp_same_dispatcher_retains_then_releases_ended_game,
    )
    d = _cpu_probe_scope(tmp_path, monkeypatch)
    try:
        assert_source_match()
        test_actual_mlp_same_dispatcher_retains_then_releases_ended_game(scheduler, workers, tmp_path)
        records = [json.loads(path.read_bytes()) for path in d.directory.glob('*.json')]
        native = next(r for r in records if r['owner']['role'] == 'native-runtime')
        inference = next(r for r in records if r['owner']['role'] == 'native-inference')
        assert native['complete'], native['errors']
        assert native['counters']['inference_issued'] == native['counters']['inference_successes'] == 8
        assert native['histograms']['inference_roundtrip']['count'] == 8
        assert native['histograms']['event_page']['count'] > 0
        assert native['histograms']['storage_queue_age']['count'] == native['counters']['writer_admissions'] > 0
        assert sum(int(b) * n for b, n in native['batch_rows']['batch_size_counts'].items()) == 8
        assert native['counters']['pyo3_next_batch'] > 0 and native['counters']['pyo3_submit_results'] > 0
        assert inference['complete'] and inference['stage_seconds']['forward']['seconds'] > 0
        assert not d.errors
    finally:
        _cpu_probe_finish(d)


@_CPU_PROBE
def test_cpu_probe_distinct_sqlite_thread_fifo(tmp_path, monkeypatch):
    from tests.test_rust_player_storage import (
        test_real_storage_thread_stages_partial_pair_and_seals_originals,
    )
    d = _cpu_probe_scope(tmp_path, monkeypatch)
    try:
        test_real_storage_thread_stages_partial_pair_and_seals_originals(tmp_path, monkeypatch, False)
        records = [json.loads(path.read_bytes()) for path in d.directory.glob('*.json')]
        fifo = next(r for r in records if r['queue']['model'] == 'thread-fifo')
        assert fifo['complete'], fifo['errors']
        assert fifo['counters']['writer_admissions'] > 0
        assert fifo['histograms']['storage_queue_age']['count'] == fifo['counters']['writer_admissions']
        assert fifo['queue']['queued_bytes_peak'] > 0 and fifo['queue']['in_service_bytes_peak'] > 0
        assert not d.errors
    finally:
        _cpu_probe_finish(d)


@pytest.mark.parametrize('identifier', ['計測 α', 'x' * 200, '  preserved metadata  '])
def test_observation_id_is_nonempty_utf8_metadata_without_arbitrary_syntax(tmp_path, identifier):
    d = diagnostics_from_environment({'GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR': str(tmp_path),
        'GF_BOTS_PERFORMANCE_OBSERVATION_ID': identifier})
    try:
        d.phase('phase_start', d.scope(phase='startup'))
        assert json.loads(next(tmp_path.glob('*.json')).read_text())['observation_id'] == identifier
    finally:
        d.close()


@pytest.mark.parametrize('identifier', ['', '\udcff', 'bad\0id'])
def test_invalid_utf8_empty_or_nul_observation_id_refuses_before_work(tmp_path, identifier):
    with pytest.raises(ValueError):
        diagnostics_from_environment({'GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR': str(tmp_path),
            'GF_BOTS_PERFORMANCE_OBSERVATION_ID': identifier})


def test_disabled_runtime_and_http_allocate_no_histogram_or_diagnostic_clock(monkeypatch):
    import asyncio

    from gf_common import performance_metrics
    from gf_simulator_client import performance_diagnostics
    from gf_simulator_client.http import AttestedClient
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from tests.test_rust_player_runtime import runtime_config
    for key in ('GF_BOTS_PERFORMANCE_DIAGNOSTICS_DIR', 'GF_BOTS_PERFORMANCE_OBSERVATION_ID', 'GF_BOTS_PERFORMANCE_PROFILE'):
        monkeypatch.delenv(key, raising=False)
    performance_diagnostics.host_diagnostics.cache_clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled diagnostics allocated/timestamped')
    monkeypatch.setattr(performance_metrics.Histogram, '__init__', forbidden)
    monkeypatch.setattr(performance_diagnostics.time, 'monotonic_ns', forbidden)
    owner = RustPlayerRuntime(runtime_config())
    assert owner.diagnostics is None
    owner.join()
    client = AttestedClient('http://127.0.0.1:8000', expected_fingerprint='x', fixed_seed_experiment=True, expected_effective={})
    assert client.performance_metrics is None
    asyncio.run(client.close())
    performance_diagnostics.host_diagnostics.cache_clear()


def test_collection_serialization_wait_starts_after_preflight_checks(monkeypatch):
    """Real asyncio mutex timing only; no collection/source capability is issued."""
    import asyncio

    from gf_common.performance_metrics import PhaseMetrics
    from gf_training import group_pack_collection as module
    clock = [0]
    metrics = PhaseMetrics(scope={}, start_ns=0, counters=(), histograms=('collection_storage_serialization_wait',))
    async def run():
        owner = module.GroupPackCollection.__new__(module.GroupPackCollection)
        owner._performance_metrics = metrics
        owner._operation_lock = asyncio.Lock()
        owner.deadline = asyncio.get_running_loop().time() + 5
        owner._check_submission = lambda **kwargs: clock.__setitem__(0, 1000)
        owner._metric_add = lambda *args: None
        async def operation(*args, **kwargs):
            return 'finished'
        owner._run_sync = operation
        assert await owner._serialized(lambda: None, (), {}, recovery=False) == 'finished'
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: clock[0])
    asyncio.run(run())
    h = metrics.histograms['collection_storage_serialization_wait']
    assert h.count == 1 and h.snapshot()['min_ns'] == 0


def _fix1_broker_branch(command, namespace):
    """Execute the real branch without importing the environment-clearing bootstrap."""
    import ast
    from pathlib import Path
    namespace['command'] = command
    source = namespace.get('_fix1_source', Path('src/gf_simulator_client/shared_bootstrap.py'))
    tree = ast.parse(Path(source).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    for node in ast.walk(main):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if not isinstance(node.test.left, ast.Name) or node.test.left.id != 'command':
            continue
        right = node.test.comparators[0]
        matches = isinstance(right, ast.Constant) and right.value == command
        if isinstance(right, ast.Tuple):
            matches = any(isinstance(value, ast.Constant) and value.value == command for value in right.elts)
        if matches:
            bounded = ast.For(target=ast.Name(id='_fix1_once', ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(1)], keywords=[]),
                body=node.body, orelse=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[bounded], type_ignores=[])),  # noqa: S102 - own static source, bounded control branch only.
                '<actual-broker-branch>', 'exec'), namespace)
            return
    raise AssertionError('actual broker branch missing: ' + command)


def _fix1_broker_state(diagnostics, scope):
    import math
    import threading
    import time
    from io import BytesIO
    from types import SimpleNamespace

    from gf_common.domain import ContractError, closed
    from gf_common.shared_requests import RequestAuthority
    from gf_simulator_client.framing import write_frame
    return {'diagnostics': diagnostics, 'control': {'command': 'DIAGNOSTIC_BEGIN', 'scope': scope},
        'measurement': None, 'response_window': None, 'profile_status': None, 'profile_target_scope': None,
        'publish_profile': lambda: True, 'authority': RequestAuthority('execution'),
        'active': {}, 'lock': threading.Lock(), 'server': SimpleNamespace(is_alive=True),
        'output': BytesIO(), 'closed': closed, 'write_frame': write_frame, 'time': time, 'math': math,
        'ContractError': ContractError, 'initial': {'execution': 'execution', 'cleanup_deadline': None},
        'deadlines': {}, 'counters': {'window_peak_pending': 0, 'failures': 0}}


def _fix1_reply(state):
    from gf_simulator_client.framing import read_frame
    state['output'].seek(0)
    return read_frame(state['output'])


@pytest.mark.parametrize('command,fault', [('DIAGNOSTIC_BEGIN', 'validation'),
    ('DIAGNOSTIC_BEGIN', 'counter'), ('DIAGNOSTIC_BEGIN', 'profile'),
    ('DIAGNOSTIC_END', 'snapshot'), ('DIAGNOSTIC_END', 'file'), ('DIAGNOSTIC_END', 'profile')])
def test_fix1_t1_framed_diagnostic_processing_failure_keeps_real_authority_usable(tmp_path, monkeypatch, command, fault):
    from io import BytesIO

    from gf_common import performance_metrics
    from gf_common.performance_metrics import PhaseMetrics, ResponseWindow
    from gf_simulator_client.performance_profile import ProfileStatus
    d = Diagnostics(tmp_path, 'fix1-processing')
    scope = d.scope(run_id='run', batch_id='batch', phase='collection')
    state = _fix1_broker_state(d, scope)
    status = ProfileStatus.local()
    state['profile_status'] = status
    state['control']['command'] = command
    def broken(*args, **kwargs):
        if fault == 'snapshot':
            assert status.terminal.value == 1  # Logical END precedes fallible acquisition.
        raise RuntimeError('diagnostic processing fault')
    if command == 'DIAGNOSTIC_END':
        state['measurement'] = PhaseMetrics(scope=scope, start_ns=0)
        state['response_window'] = ResponseWindow()
        state['profile_target_scope'] = scope
        status.begin(scope)
    if fault == 'validation':
        state['control']['extra'] = 'invalid diagnostic key'
    elif fault == 'counter':
        monkeypatch.setattr(performance_metrics, 'ResponseWindow', broken)
    elif fault == 'snapshot':
        monkeypatch.setattr(state['response_window'], 'snapshot', broken)
    elif fault == 'file':
        monkeypatch.setattr(d, 'metrics', lambda *args, **kwargs: None)
    elif fault == 'profile':
        if command == 'DIAGNOSTIC_BEGIN':
            original = status.begin
            def partially_started(scope):
                original(scope)
                broken()
            monkeypatch.setattr(status, 'begin', partially_started)
        else:
            state['publish_profile'] = broken
    try:
        _fix1_broker_branch(command, state)
        assert _fix1_reply(state) == {'accepted': False}
        assert state['authority'].empty and state['authority'].accepted == 0
        assert state['measurement'] is state['response_window'] is None
        expected_error = 'ContractError' if fault == 'validation' else 'OSError' if fault == 'file' else 'RuntimeError'
        assert 'diagnostic_processing:' + expected_error in d.errors
        assert status.terminal.value == 1 and state['counters']['failures'] == 0
        state['output'] = BytesIO()
        routing = {'execution': 'execution', 'game': 'game', 'participant': 1,
            'revision': 1, 'checkpoint': 'checkpoint', 'request': 1}
        state['control'] = {'command': 'AUTHORIZE', 'endpoint': 'own', 'routing': routing,
            'deadline': state['time'].monotonic() + 5}
        _fix1_broker_branch('AUTHORIZE', state)
        assert _fix1_reply(state) == {'accepted': True}
        state['authority'].match('own', routing)
        assert state['authority'].accepted == 1
    finally:
        d.close()


def test_fix1_t3_emitted_phases_have_batch_id_only_for_collection(tmp_path):
    import ast
    from pathlib import Path

    from gf_simulator_client.performance_diagnostics import finish_phase, transition_phase
    d = Diagnostics(tmp_path, 'fix1-phases')
    try:
        tree = ast.parse(Path('src/gf_training/ppo_run.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'transition_phase'
            and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)
            and n.args[1].value in ('update', 'save')]
        assert len(calls) == 2
        transition_phase(d, 'collection', batch_id='actual-batch')
        for call in calls:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[])), '<actual-ppo-phase>', 'exec'),  # noqa: S102 - own static emitted phase call only.
                {'transition_phase': transition_phase, 'performance': d, 'batch_id': 'actual-batch'})
        for phase in ('startup', 'update', 'save', 'retirement', 'cold', 'evaluation', 'idle'):
            transition_phase(d, phase, batch_id='must-not-leak')
        finish_phase(d)
        rows = [json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
        assert any(row['scope']['phase'] == 'collection' and row['scope']['batch_id'] == 'actual-batch' for row in rows)
        assert all(row['scope']['batch_id'] is None for row in rows if row['scope']['phase'] != 'collection')
    finally:
        d.close()


def test_fix1_t4_completed_retire_waits_for_actual_relay_observation(tmp_path):
    from gf_common.performance_metrics import ResponseWindow
    from gf_simulator_client.performance_diagnostics import Diagnostics
    d = Diagnostics(tmp_path, 'fix1-retire')
    state = _fix1_broker_state(d, d.scope(phase='collection', batch_id='b'))
    window = ResponseWindow()
    state['response_window'] = window
    state['authority'].begin('own', game='game', participant=1, revision=1, checkpoint='c')
    window.authorize('own')
    schedule = []
    class ReplyVisibleBeforeRelayObservation:
        def wait(self, timeout):
            assert timeout == 30.0
            assert state['authority'].empty  # Original retirement still precedes original wait.
            assert state['lock'].acquire(False)  # No lock was held across that wait.
            try:
                schedule.append('relay-observed')
                window.delivered('own', 1)
                state['active'].pop('own')
            finally:
                state['lock'].release()
            return True
    state['active']['own'] = ReplyVisibleBeforeRelayObservation()
    state['control'] = {'command': 'RETIRE', 'endpoint': 'own', 'completed': True}
    try:
        _fix1_broker_branch('RETIRE', state)
        assert schedule == ['relay-observed'] and _fix1_reply(state) == {'accepted': True}
        result = window.snapshot(quiescent_end=True)
        assert result['failures'] == 0 and result['batch_size_counts'] == {1: 1}
    finally:
        d.close()


def _fix1_parent_frames(tmp_path, monkeypatch, reply):
    import threading
    import time
    from io import BytesIO
    from types import SimpleNamespace

    from gf_common.shared_requests import RequestAuthority
    from gf_simulator_client import shared_service
    service = shared_service.SharedInferenceService.__new__(shared_service.SharedInferenceService)
    service.closed = service.failed = service.uncertain = False
    service.lock, service.deadline = threading.RLock(), time.monotonic() + 5
    service.process = SimpleNamespace(stdin=BytesIO(), stdout=reply, poll=lambda: None)
    service.diagnostics = Diagnostics(tmp_path, 'fix1-parent-frames')
    service.authority = RequestAuthority('execution')
    service._measurement_scope = None
    # Byte streams exercise the actual framing; no child process or GPU/admission exists.
    monkeypatch.setattr(shared_service, '_DeadlineReader', lambda source, deadline: source)
    return service


@pytest.mark.parametrize('method', ['begin', 'end'])
def test_fix1_t1_parent_framed_negative_reply_does_not_quarantine_ordinary_exchange(tmp_path, monkeypatch, method):
    from io import BytesIO

    from gf_simulator_client.framing import write_frame
    replies = BytesIO()
    write_frame(replies, {'accepted': False})
    write_frame(replies, {'accepted': True})
    replies.seek(0)
    service = _fix1_parent_frames(tmp_path, monkeypatch, replies)
    scope = service.diagnostics.scope(phase='collection', batch_id='batch')
    try:
        if method == 'begin':
            service.measurement_begin(scope)
        else:
            service._measurement_scope = scope
            service.measurement_end()
        assert service.diagnostics.errors and not service.uncertain and not service.failed
        routing = service.authority.begin('own', game='game', participant=1, revision=1, checkpoint='c')
        assert service._exchange({'command': 'AUTHORIZE', 'endpoint': 'own', 'routing': routing,
            'deadline': service.deadline}) == {'accepted': True}
        service.authority.match('own', routing)
        assert not service.uncertain and not service.failed
    finally:
        service.diagnostics.close()


@pytest.mark.parametrize('method', ['begin', 'end'])
@pytest.mark.parametrize('fault', ['read', 'write', 'eof', 'partial', 'malformed', 'timeout', 'health'])
def test_fix1_t1_actual_shared_transport_uncertainty_stays_quarantined(tmp_path, monkeypatch, method, fault):
    from io import BytesIO

    from gf_common.domain import ContractError
    class ReadFault(BytesIO):
        def read(self, size):
            raise TimeoutError('actual shared read deadline') if fault == 'timeout' else OSError('shared read failed')
    class WriteFault(BytesIO):
        def write(self, data):
            raise OSError('shared write failed')
    reply = ReadFault() if fault in ('read', 'timeout') else BytesIO(
        b'\0\0\0\0' if fault == 'partial' else (1).to_bytes(8, 'big') + b'[' if fault == 'malformed' else b'')
    service = _fix1_parent_frames(tmp_path, monkeypatch, reply)
    if fault == 'write':
        service.process.stdin = WriteFault()
    if fault == 'health':
        service.process.poll = lambda: 1
    scope = service.diagnostics.scope(phase='collection', batch_id='batch')
    try:
        if method == 'begin':
            service.measurement_begin(scope)
        else:
            service._measurement_scope = scope
            service.measurement_end()
        assert service.uncertain and service.diagnostics.errors
        position = reply.tell()
        with pytest.raises(ContractError, match='uncertain'):
            service._exchange({'command': 'AUTHORIZE'})
        assert service.uncertain and reply.tell() == position  # No resync or late-reply discard.
    finally:
        service.diagnostics.close()


@pytest.mark.parametrize('command', ['DIAGNOSTIC_BEGIN', 'DIAGNOSTIC_END'])
def test_fix1_t1_broker_health_and_response_write_stay_outside_processing_containment(tmp_path, command):
    from gf_common.domain import ContractError
    d = Diagnostics(tmp_path, 'fix1-channel-boundary')
    state = _fix1_broker_state(d, d.scope(phase='collection', batch_id='b'))
    state['control']['command'] = command
    try:
        state['server'].is_alive = False
        with pytest.raises(ContractError, match='health'):
            _fix1_broker_branch(command, state)
        assert state['output'].tell() == 0 and not d.errors
        state['server'].is_alive = True
        def write_failed(*args):
            raise OSError('actual response write failed')
        state['write_frame'] = write_failed
        with pytest.raises(OSError, match='response write'):
            _fix1_broker_branch(command, state)
    finally:
        d.close()


@pytest.mark.parametrize('completed,wait_succeeds', [(False, True), (True, False)])
def test_fix1_t4_cancel_and_unconfirmed_relay_retirement_never_qualify_batches(tmp_path, completed, wait_succeeds):
    from gf_common.domain import ContractError
    from gf_common.performance_metrics import ResponseWindow
    d = Diagnostics(tmp_path, 'fix1-retire-failure')
    state = _fix1_broker_state(d, d.scope(phase='collection', batch_id='b'))
    window = ResponseWindow()
    state['response_window'] = window
    state['authority'].begin('own', game='game', participant=1, revision=1, checkpoint='c')
    window.authorize('own')
    class Completion:
        def wait(self, timeout):
            assert completed
            return wait_succeeds
    state['active']['own'] = Completion()
    state['control'] = {'command': 'RETIRE', 'endpoint': 'own', 'completed': completed}
    try:
        if completed:
            with pytest.raises(ContractError, match='retirement unconfirmed'):
                _fix1_broker_branch('RETIRE', state)
            assert state['output'].tell() == 0
        else:
            _fix1_broker_branch('RETIRE', state)
            assert _fix1_reply(state) == {'accepted': True}
            assert window.failures == 1
        assert window.snapshot(quiescent_end=True)['batch_size_counts'] is None
        assert state['authority'].empty
    finally:
        d.close()


def test_fix1_t1_healthy_cycle_retains_complete_exact_window_and_real_requests(tmp_path):
    from io import BytesIO
    d = Diagnostics(tmp_path, 'fix1-healthy-cycle')
    scope = d.scope(phase='collection', batch_id='batch')
    state = _fix1_broker_state(d, scope)
    try:
        _fix1_broker_branch('DIAGNOSTIC_BEGIN', state)
        assert _fix1_reply(state) == {'accepted': True}
        state['output'] = BytesIO()
        route = {'execution': 'execution', 'game': 'game', 'participant': 1,
            'revision': 1, 'checkpoint': 'c', 'request': 1}
        state['control'] = {'command': 'AUTHORIZE', 'endpoint': 'own', 'routing': route,
            'deadline': state['time'].monotonic() + 5}
        _fix1_broker_branch('AUTHORIZE', state)
        state['authority'].match('own', route)
        state['response_window'].delivered('own', 1)
        state['output'] = BytesIO()
        state['control'] = {'command': 'RETIRE', 'endpoint': 'own', 'completed': True}
        _fix1_broker_branch('RETIRE', state)
        assert _fix1_reply(state) == {'accepted': True}
        state['output'] = BytesIO()
        state['control'] = {'command': 'DIAGNOSTIC_END', 'scope': scope}
        _fix1_broker_branch('DIAGNOSTIC_END', state)
        assert _fix1_reply(state) == {'accepted': True}
        row = json.loads(next(tmp_path.glob('*.json')).read_text())
        assert row['complete'] and row['batch_rows']['batch_size_counts'] == {'1': 1}
        assert state['authority'].accepted == state['authority'].completed == 1
        assert state['counters']['failures'] == 0 and not d.errors
    finally:
        d.close()


def test_fix1_t1_active_requests_survive_rejected_diagnostic_end(tmp_path):
    from io import BytesIO

    from gf_common.performance_metrics import PhaseMetrics, ResponseWindow
    from gf_simulator_client.performance_profile import ProfileStatus
    d = Diagnostics(tmp_path, 'fix1-active-window')
    scope = d.scope(phase='collection', batch_id='batch')
    state = _fix1_broker_state(d, scope)
    state['measurement'] = PhaseMetrics(scope=scope, start_ns=0)
    window = ResponseWindow(); window.authorize('own')
    state['response_window'] = window
    status = ProfileStatus.local(); status.begin(scope)
    state['profile_status'], state['profile_target_scope'] = status, scope
    state['authority'].begin('own', game='game', participant=1, revision=1, checkpoint='c')
    class Completion:
        def wait(self, timeout):
            with state['lock']:
                window.delivered('own', 1)
                state['active'].pop('own')
            return True
    state['active']['own'] = Completion()
    state['control'] = {'command': 'DIAGNOSTIC_END', 'scope': scope}
    try:
        _fix1_broker_branch('DIAGNOSTIC_END', state)
        assert _fix1_reply(state) == {'accepted': False}
        assert state['response_window'] is window and not state['authority'].empty
        assert state['authority'].accepted == 1 and state['authority'].completed == 0
        assert state['counters']['failures'] == window.failures == 0 and status.terminal.value == 1
        state['output'] = BytesIO()
        state['control'] = {'command': 'RETIRE', 'endpoint': 'own', 'completed': True}
        _fix1_broker_branch('RETIRE', state)
        assert _fix1_reply(state) == {'accepted': True}
        assert state['authority'].completed == 1
        assert window.snapshot(quiescent_end=True)['batch_size_counts'] is None
    finally:
        d.close()
