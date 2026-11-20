"""Post-exit diagnostic projections and audit wiring; no workload execution."""
import json

import pytest

from gf_common.domain import ContractError
from gf_common.performance_metrics import Histogram, PhaseMetrics


def test_histogram_readback_preserves_rank_bin_bounds_and_overflow():
    from tools.rust_player_verify_operation import histogram_data
    h = Histogram()
    for n in (1, 32, 33, 1000):
        h.add(n)
    value = histogram_data(h.snapshot())
    assert value['p95_ns'] == [992, 1008]
    assert value['p99_ns'] == [992, 1008]
    h.add(2**64)
    assert histogram_data(h.snapshot())['p99_ns'] == [2**64, None]
    assert histogram_data(None) is None


@pytest.mark.parametrize('fault', ['recipe', 'total', 'negative', 'bool', 'min'])
def test_histogram_invalid_evidence_refuses(fault):
    from tools.rust_player_verify_operation import histogram_data
    h = Histogram(); h.add(32)
    row = h.snapshot()
    if fault == 'recipe': row['recipe'] = 'other'
    elif fault == 'total': row['count'] = 2
    elif fault == 'negative': row['counts'][0] = -1
    elif fault == 'bool': row['count'] = True
    elif fault == 'min': row['min_ns'] = None
    with pytest.raises(ContractError):
        histogram_data(row)


def metric():
    scope = {'run_id': 'run', 'operation_id': 'op', 'invocation_id': 'inv',
        'batch_id': 'batch-000000', 'phase': 'collection'}
    m = PhaseMetrics(scope=scope, start_ns=100)
    m.increment('pyo3_next_batch', 3); m.increment('pyo3_submit_results', 2)
    m.increment('pyo3_other', 7)
    return {'schema_version': 1, 'kind': 'rust-player-phase-metrics-v1',
        'observation_id': 'obs', 'owner': {'pid': 5, 'start_ticks': 2, 'role': 'native-runtime'},
        'clock': {'name': 'CLOCK_MONOTONIC', 'boot_id': 'boot'}, **m.snapshot(end_ns=200)}


def test_metric_keeps_owner_scope_and_host_stage_semantics():
    from tools.rust_player_verify_operation import metric_data
    row = metric()
    value = metric_data(row, observation_id='obs', boot_id='boot', run_id='run', interval=(0, 300))
    assert value['pyo3_calls'] == 12
    assert value['scope']['batch_id'] == 'batch-000000'
    assert value['stage_seconds']['forward'] is None
    assert value['batch_histogram'] is None
    assert value['owner']['pid'] == 5


@pytest.mark.parametrize('fault', ['id', 'boot', 'interval', 'extra', 'negative', 'batch-scope'])
def test_metric_bad_pairing_or_closed_schema_refuses(fault):
    from tools.rust_player_verify_operation import metric_data
    row = metric()
    if fault == 'id': row['observation_id'] = 'foreign'
    elif fault == 'boot': row['clock']['boot_id'] = 'foreign'
    elif fault == 'interval': row['end_monotonic_ns'] = 301
    elif fault == 'extra': row['hidden'] = 1
    elif fault == 'negative': row['counters']['pyo3_other'] = -1
    elif fault == 'batch-scope': row['scope']['phase'] = 'update'
    with pytest.raises(ContractError):
        metric_data(row, observation_id='obs', boot_id='boot', run_id='run', interval=(0, 300))


def test_batch_inversion_requires_complete_coverage_and_zero_failures():
    from tools.rust_player_verify_operation import metric_data
    row = metric()
    row['batch_rows'] = {'method': 'complete-response-size-inversion-v1',
        'request_size_counts': {'2': 4}, 'batch_size_counts': {'2': 2},
        'issued_requests': 4, 'completed_responses': 4, 'quiescent_start': True,
        'quiescent_end': True, 'failures': 0}
    row['counters'].update(inference_issued=4, inference_successes=4)
    h = Histogram()
    for n in range(4): h.add(n)
    row['histograms']['inference_roundtrip'] = h.snapshot()
    options = {'observation_id': 'obs', 'boot_id': 'boot', 'run_id': 'run', 'interval': (0, 300)}
    assert metric_data(row, **options)['batch_histogram'] == {'2': 2}
    row['counters']['inference_failures'] = 1
    assert metric_data(row, **options)['batch_histogram'] is None


def test_journal_projection_counts_only_matched_acknowledged_decisions():
    from tools.rust_player_verify_operation import journal_data
    rows = [{'kind': 'header', 'schema_version': 6, 'viewer': 1},
        {'kind': 'decision', 'decision': 0, 'command': {'command_id': 'a'},
         'table': {'count': 3}, 'search_report': {'cutoff_reasons': ['count_time']}},
        {'kind': 'acknowledgement', 'decision': 0, 'response': {'command_id': 'a', 'status': 'active'}},
        {'kind': 'decision', 'decision': 1, 'command': {'command_id': 'b'},
         'table': {'count': 9}, 'search_report': {'cutoff_reasons': []}}]
    value = journal_data([json.dumps(row).encode() for row in rows])
    assert value == {'viewer': 1, 'decisions': 2, 'accepted_decisions': 1,
        'candidate_total': 12, 'candidate_max': 9, 'cutoff_reasons': {'count_time': 1}}
    rows[2]['response']['command_id'] = 'foreign'
    with pytest.raises(ContractError):
        journal_data([json.dumps(row).encode() for row in rows])


def test_journal_rejection_is_not_an_acknowledgement():
    from tools.rust_player_verify_operation import journal_data
    rows = [{'kind': 'header', 'schema_version': 6, 'viewer': 2},
        {'kind': 'decision', 'decision': 0, 'command': {'command_id': 'a'},
         'table': {'count': 1}, 'search_report': {'cutoff_reasons': []}},
        {'kind': 'rejection', 'decision': 0}]
    assert journal_data([json.dumps(row).encode() for row in rows])['accepted_decisions'] == 0


def test_collection_rates_use_common_persisted_wall_and_all_seats():
    from tools.rust_player_verify_operation import collection_data
    update = {'batch_id': 'batch-000000', 'actual_games': 16, 'planned_games': 16,
        'collection_seconds': 2., 'update_seconds': 4.,
        'source_proof': {'closure_counts': {'natural': 16}}}
    rows = [{'accepted_decisions': 3, 'decisions': 3}, {'accepted_decisions': 5, 'decisions': 5}]
    value = collection_data(update, rows, ordinal=1)
    assert value['natural_games'] == 16
    assert value['accepted_decisions'] == 8
    assert value['games_per_second'] == 8.
    assert value['decisions_per_second'] == 4.
    update['source_proof']['closure_counts'] = {'aborted': 16}
    assert collection_data(update, rows, ordinal=1)['natural_games'] == 0


def test_observer_requires_actual_success_and_exact_original_wait_wall(tmp_path):
    import time

    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import observer_data
    row = {'schema_version': 1, 'kind': 'rust-player-process-observation-v1',
        'command': {'argv': ['python', '--config', '/original.toml'], 'pid': 5, 'start_ticks': 2,
            'start_monotonic_ns': 100, 'finish_monotonic_ns': 300, 'wall_ns': 200},
        'clock': {'name': 'CLOCK_MONOTONIC', 'boot_id': 'boot', 'hz': 100, 'page_bytes': 4096, 'kernel': 'test'},
        'observer': {'pid': 1, 'start_monotonic_ns': 90, 'finish_monotonic_ns': 310,
            'user_ticks': 1, 'system_ticks': 2},
        'command_wait': {'exit_code': 0, 'signal': None, 'observed': True},
        'sampling': {'enabled': False, 'cpu_interval_ns': 100000000, 'pss_interval_ns': 1000000000},
        'peak_complete_pss_bytes': None,
        'coverage': {'process_samples': 0, 'pss_sweeps': 0, 'complete_pss_sweeps': 0,
            'partial_pss_sweeps': 0, 'peak_eligible_pss_sweeps': 0, 'observed_processes': 1},
        'errors': [], 'limitations': ['sampling misses short processes']}
    path = tmp_path / 'summary.json'
    (tmp_path / 'process.jsonl').write_bytes(b'')
    (tmp_path / 'pss.jsonl').write_bytes(b'')
    path.write_text(json.dumps(row))
    scan = SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10)
    result = observer_data(tmp_path, scan=scan)
    assert result['wall_seconds'] == 2e-7
    assert result['peak_complete_pss_bytes'] is None
    row['command_wait']['exit_code'] = 143
    path.write_text(json.dumps(row))
    with pytest.raises(ContractError):
        observer_data(tmp_path, scan=SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10))
    row['command_wait']['exit_code'] = False
    path.write_text(json.dumps(row))
    with pytest.raises(ContractError):
        observer_data(tmp_path, scan=SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10))


def test_config_must_be_the_actual_observed_argument(tmp_path):
    from tools.rust_player_verify_operation import check_observed_config
    with pytest.raises(ContractError, match='observed CONFIG'):
        check_observed_config(tmp_path / 'other.toml',
            argv=['python', '--config', str(tmp_path / 'original.toml')])


def test_archived_cpu_native_post_exit_verification_is_data_only(monkeypatch):
    import fcntl
    from pathlib import Path

    from tools import rust_player_verify_operation as tool
    root = Path('/home/percy/develop/GodFieldBots/runs/rust-player-runtime-v1/components/public-ordinary-cpu-55cd')
    if not root.is_dir():
        pytest.skip('authorized archived CPU case unavailable')
    project = tool._project_collections
    def under_owner(*args, **kwargs):
        from gf_cli.config import LoadedConfig
        assert type(kwargs['retained'][0]) is LoadedConfig
        with (root / 'parent/.lock').open('rb') as competing, pytest.raises(BlockingIOError):
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return project(*args, **kwargs)
    monkeypatch.setattr(tool, '_project_collections', under_owner)
    result = tool.verify_operation(config_path=root / 'cpu-smoke.toml', parent=root / 'parent',
        observer_directory=root / 'observer', telemetry_directory=root / 'diagnostics',
        observation_id='public-ordinary-cpu-55cd', lane='native-ordinary')
    assert result['verified'] is True
    assert result['audit']['saved_pair_full_content_validated'] is True
    assert result['collections'][0]['natural_games'] == 16
    assert result['collections'][0]['accepted_decisions'] == 436
    assert len(result['collections'][0]['seats']) == 32
    assert result['collections'][0]['collection_seconds'] == 15.032642317994032
    assert result['observer']['wall_seconds'] == 139.024214756
    assert result['observer']['peak_complete_pss_bytes'] == 903681024
    assert result['telemetry']['metrics']
    assert not result['source_authority']


@pytest.mark.parametrize('fault', ['clock', 'completion', 'supervisor', 'cold-saved', 'cold-numeric', 'inputs', 'phase-sha'])
def test_native_receipt_corruption_cannot_qualify_archive(monkeypatch, fault):
    from pathlib import Path

    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import verify_operation
    root = Path('/home/percy/develop/GodFieldBots/runs/rust-player-runtime-v1/components/public-ordinary-cpu-55cd')
    if not root.is_dir():
        pytest.skip('authorized archived CPU case unavailable')
    original = SealedScan.object
    def changed(self, path, expected=None):
        value = original(self, path, expected)
        if fault == 'clock' and path.name.startswith('clock-'):
            value['inputs']['wall_seconds'] += 1
        elif fault == 'completion' and path.name.startswith('completion-'):
            value['eligible'] = False
        elif fault == 'supervisor' and path.name.startswith('admission-'):
            value['supervisor']['pid'] += 1
        elif fault == 'cold-saved' and path.name.startswith('cold-audit-'):
            value['saved_pair']['full_content_validated'] = False
        elif fault == 'cold-numeric' and path.name.startswith('cold-audit-'):
            value['updates'][0]['numeric_sha256'] = '0' * 64
        elif fault == 'inputs' and path.name.startswith('inputs-'):
            value['inputs'][0]['sha256'] = '0' * 64
        elif fault == 'phase-sha' and value.get('kind') == 'rust-player-phase-event-v1':
            value['links']['operation_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(SealedScan, 'object', changed)
    with pytest.raises(ContractError):
        verify_operation(config_path=root / 'cpu-smoke.toml', parent=root / 'parent',
            observer_directory=root / 'observer', observation_id='public-ordinary-cpu-55cd', lane='native-ordinary',
            telemetry_directory=root / 'diagnostics' if fault == 'phase-sha' else None)


def test_audit_uses_effective_source_cap_and_exact_finite_wall():
    from types import SimpleNamespace

    from tools.rust_player_verify_operation import _bounds
    loaded = SimpleNamespace(config=SimpleNamespace(preprocessing=SimpleNamespace(memory_budget_mib=8),
        hpo_spec={'operating': {'source_budget_bytes': 3 * 1024**2}}, run=SimpleNamespace(wall_seconds=123.)))
    assert _bounds(loaded) == (3 * 1024**2, 123.)
    loaded.config.run.wall_seconds = float('inf')
    with pytest.raises(ContractError):
        _bounds(loaded)


def test_off_telemetry_is_explicitly_unavailable_without_invalidating_common_rates():
    from tools.rust_player_verify_operation import _telemetry
    value = _telemetry(None, observer={}, audit={}, observation_id='obs', scan=None)
    assert value['enabled'] is False
    assert value['metrics'] is None and value['phases'] is None and value['complete'] is None


def test_legacy_full_validator_seam_receives_same_actual_owned_scan(tmp_path, monkeypatch):
    import time
    from pathlib import Path

    from gf_training import hpo_source
    from gf_training.lineage import require_hpo_run_owner, run_lock
    from gf_training.structured_lineage import _ChainScan
    from tools import rust_player_verify_operation as tool
    run_id = '00000000-0000-0000-0000-000000000017'
    config_path = Path(__file__).parents[1] / 'configs/hpo/mlp-shared-inference-scaling.toml'
    loaded = tool._configuration(config_path, 'legacy-hpo', run_id=run_id, learning_seed=17)
    parent = tmp_path / 'parent'
    observer = tmp_path / 'observer'; observer.mkdir()
    state = {'status': 'incomplete', 'run_id': run_id, 'updates': 0, 'optimizer_steps': 0,
        'config_sha256': loaded.sha256, 'effective_config': loaded.effective,
        'config_inputs': list(loaded.inputs),
        'generation_sha256': '1' * 64, 'identity': {'code': {'revision': 'unit-data-only'}}}
    stdout = {'run_dir': str(parent), 'status': 'incomplete', 'updates': 0, 'generation_sha256': '1' * 64}
    (observer / 'command.stdout.log').write_text(json.dumps(stdout)+'\n')
    monkeypatch.setattr(tool, 'observer_data', lambda directory, *, scan:
        {'command': {'argv': ['python', '--config', str(config_path)]}})
    monkeypatch.setattr(tool, '_legacy_hpo_binding', lambda *args, **kwargs: {'unit_seam': True})
    scans = []
    deadline = time.monotonic() + 30
    cap = tool._bounds(loaded)[0]
    def full_validator(directory, *, require_complete, max_owned_bytes, deadline, _scan):
        assert directory == parent and require_complete is False
        assert max_owned_bytes == cap
        assert type(_scan) is _ChainScan and _scan.deadline == deadline
        require_hpo_run_owner(parent)
        assert parent / 'run.json' in _scan.files
        scans.append(_scan)
        return state, [{}]  # Validator boundary unit double; no Source/Parent issued.
    monkeypatch.setattr(hpo_source, 'validate_hpo_parent', full_validator)
    with run_lock(parent, create=True) as owner:
        (parent / 'run.json').write_text(json.dumps(state))
        result, _ = tool._audit_parent(parent, loaded, lane='legacy-hpo', observer_directory=observer,
            deadline=deadline, limit=cap, config_path=config_path, ownership=owner)
    assert result['status'] == 'incomplete'
    assert len(scans) == 1 and scans[0]._closed


def test_profile_relative_original_trace_is_hash_checked_without_gpu_time_inference(tmp_path):
    import hashlib
    import time

    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import _telemetry
    trace = tmp_path / 'forward-unit.trace.json'
    trace.write_bytes(b'{"traceEvents":[]}')
    row = metric()
    profile = {key: row[key] for key in ('schema_version', 'observation_id', 'owner', 'clock', 'scope')}
    profile.update(kind='rust-player-forward-profile-v1', recipe='forward-window-v1', enabled=True,
        boundary='native-post-row-forward', worker_tid=9, start_after_forward=1, stop_after_forward=3,
        callbacks_seen=3, scored_rows=[1, 1, 1], start_monotonic_ns=100, end_monotonic_ns=200,
        complete=True, errors=[], trace_path=trace.name, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        trace_bytes=trace.stat().st_size, trace_clock='recorded profile clock', torch_version='test')
    (tmp_path / 'profile.json').write_text(json.dumps(profile))
    audit = {'run_id': 'run', 'receipts': [], 'updates': [
        {'batch_id': 'batch-000000', 'invocation_id': 'inv', 'collection_seconds': 1.}]}
    observer = {'command': {'start_monotonic_ns': 0, 'finish_monotonic_ns': 300},
        'clock': {'boot_id': 'boot'}, 'processes': [{'pid': 5, 'start_ticks': 2}]}
    result = _telemetry(tmp_path, observer=observer, audit=audit, observation_id='obs',
        scan=SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10))
    assert result['profiles'][0]['gpu_event_metrics'] is None
    assert result['profiles'][0]['record']['trace_sha256'] == profile['trace_sha256']
    assert not result['complete']  # No collection metrics/phases were invented.


def test_historical_nonshared_hpo_config_does_not_enter_experiment_lane():
    from pathlib import Path

    from tools.rust_player_verify_operation import _configuration
    with pytest.raises(ContractError):
        _configuration(Path(__file__).parents[1] / 'configs/hpo/owned-mlp-smoke.toml',
            'legacy-hpo', run_id='00000000-0000-0000-0000-000000000017', learning_seed=17)



def phase(scope, *, event='phase_start', at=10, role='host', pid=5, ticks=2):
    return {'schema_version': 1, 'kind': 'rust-player-phase-event-v1',
        'observation_id': 'obs', 'owner': {'pid': pid, 'start_ticks': ticks, 'role': role},
        'clock': {'name': 'CLOCK_MONOTONIC', 'boot_id': 'boot'}, 'scope': scope,
        'event': event, 'monotonic_ns': at,
        'links': dict.fromkeys(('operation_sha256', 'prepared_sha256', 'exit_sha256', 'completion_sha256'))}


def telemetry_fixture(tmp_path, rows, *, lane='legacy-ordinary', updates=None, measurement_scope='fixture'):
    import time

    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import _telemetry
    for n, row in enumerate(rows): (tmp_path / f'{n}.json').write_text(json.dumps(row))
    observer = {'command': {'pid': 5, 'start_ticks': 2, 'start_monotonic_ns': 0, 'finish_monotonic_ns': 300},
        'clock': {'boot_id': 'boot'}, 'processes': [{'pid': 5, 'start_ticks': 2}, {'pid': 6, 'start_ticks': 3}]}
    audit = {'run_id': 'run', 'receipts': [], 'updates': updates or [], 'lane': lane}
    if measurement_scope == 'fixture':
        assigned = next((row['scope'] for row in rows if row['scope']['run_id'] is not None), {})
        measurement_scope = {'invocation_id': assigned.get('invocation_id'),
            'operation_id': assigned.get('operation_id'), 'start_generation': 0, 'end_generation': len(updates or [])}
        for generation, update in enumerate(audit['updates'], 1): update['generation'] = generation
    audit['measurement_scope'] = measurement_scope
    return _telemetry(tmp_path, observer=observer, audit=audit, observation_id='obs',
        scan=SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10))


def test_real_legacy_null_startup_then_known_run_binds_original_coordinator(tmp_path):
    startup = dict.fromkeys(('run_id', 'operation_id', 'invocation_id', 'batch_id')) | {'phase': 'startup'}
    known = startup | {'run_id': 'run', 'phase': 'retirement'}
    result = telemetry_fixture(tmp_path, [phase(startup), phase(startup, event='phase_end', at=20),
        phase(known, at=20), phase(known, event='phase_end', at=30)])
    assert result['legacy_observer_binding']['pid'] == 5
    assert result['phases'][0]['scope']['run_id'] is None


@pytest.mark.parametrize('fault', ['foreign-startup', 'unbound-later', 'wrong-pid', 'wrong-ticks', 'wrong-boot', 'late-startup', 'wrong-timing', 'assigned-startup'])
def test_legacy_phase_binding_refuses_foreign_or_unbound_evidence(tmp_path, fault):
    startup = dict.fromkeys(('run_id', 'operation_id', 'invocation_id', 'batch_id')) | {'phase': 'startup'}
    known = startup | {'run_id': 'run', 'phase': 'retirement'}
    rows = [phase(startup), phase(startup, event='phase_end', at=20), phase(known, at=20)]
    if fault == 'foreign-startup': rows[0]['scope'] = startup | {'run_id': 'foreign'}
    elif fault == 'unbound-later': rows[2]['scope'] = known | {'run_id': None}
    elif fault == 'wrong-pid': rows[2]['owner'].update(pid=6, start_ticks=3)
    elif fault == 'wrong-ticks': rows[2]['owner']['start_ticks'] = 3
    elif fault == 'wrong-boot': rows[0]['clock']['boot_id'] = 'other'
    elif fault == 'late-startup': rows[0]['monotonic_ns'] = 25
    elif fault == 'wrong-timing': rows[2]['monotonic_ns'] = 301
    elif fault == 'assigned-startup': rows[0]['scope'] = startup | {'operation_id': 'foreign'}
    with pytest.raises(ContractError): telemetry_fixture(tmp_path, rows)


def test_ordinary_legacy_refuses_missing_coordinator_telemetry():
    from tools.rust_player_verify_operation import _telemetry
    with pytest.raises(ContractError, match='ordinary.*diagnostics'):
        _telemetry(None, observer={}, audit={'lane': 'legacy-ordinary'}, observation_id='obs', scan=None)


def test_streaming_rows_exceed_manifest_total_without_history_reservation(tmp_path):
    import time

    from gf_models.sharded_arrays import MAX_MANIFEST_BYTES
    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import _rows
    path = tmp_path / 'large.jsonl'
    line = json.dumps({'monotonic_ns': 1, 'processes': [], 'errors': [], 'padding': 'x' * 8192}).encode() + b'\n'
    count = MAX_MANIFEST_BYTES // len(line) + 1
    with path.open('wb') as out:
        for _ in range(count): out.write(line)
    scan = SealedScan(max_owned_bytes=2 * 1024**2, deadline=time.monotonic()+20)
    assert sum(1 for row in _rows(path, scan) if row['monotonic_ns'] == 1) == count
    assert scan.retained < 4096
    scan.finalize()
    with pytest.raises(ContractError):
        SealedScan(max_owned_bytes=2 * 1024**2, deadline=time.monotonic()+20).object(path)


@pytest.mark.parametrize('fault', ['oversize', 'malformed', 'changed', 'deadline'])
def test_streaming_rows_keep_budget_original_identity_and_deadline(tmp_path, fault):
    import time

    from gf_models.structured_sealed import SealedScan
    from tools.rust_player_verify_operation import _rows
    path = tmp_path / 'rows.jsonl'
    path.write_bytes(b'{"data":1}\n')
    scan = SealedScan(max_owned_bytes=100000, deadline=time.monotonic()+10)
    if fault == 'oversize': path.write_bytes(b'x' * 10000)
    elif fault == 'malformed': path.write_bytes(b'{bad}\n')
    elif fault == 'changed':
        with scan.file(path): pass
        path.write_bytes(b'{"data":2}\n')
    else: scan.deadline = time.monotonic()-1
    with pytest.raises((ContractError, ValueError, TimeoutError)): list(_rows(path, scan))


def test_stage_only_collection_cannot_claim_mandatory_native_metrics(tmp_path):
    row = metric()
    row['owner']['role'] = 'native-inference'
    row['counters'] = dict.fromkeys(row['counters'])
    row['histograms'] = dict.fromkeys(row['histograms'])
    scope = row['scope']
    result = telemetry_fixture(tmp_path,
        [row, phase(scope, at=50), phase(scope, event='phase_end', at=250)], lane='native-ordinary',
        updates=[{'batch_id': scope['batch_id'], 'invocation_id': 'inv', 'collection_seconds': 1.}])
    assert not result['complete']
    assert result['collection_coverage'][0]['mandatory_metrics'] == {
        'event_page': False, 'inference_roundtrip': False, 'actual_batch': False, 'pyo3': False}



def hpo_binding_fixture(tmp_path, monkeypatch, *, fault=None):
    import time
    from types import SimpleNamespace

    from gf_models.structured_sealed import SealedScan
    from gf_training import hpo_gpu_lease
    from gf_training.lineage import digest
    from tools import rust_player_verify_operation as tool
    root = tmp_path / 'parents'; parent = root / 'run'
    context = {'campaign_id': 'experiment', 'operation_id': 'a' * 32,
        'kind': 'training', 'run_id': 'run', 'evaluation_id': None}
    campaign = SimpleNamespace(sha256='c' * 64, inputs=({'path': 'original', 'sha256': 'b' * 64},))
    loaded = SimpleNamespace(sha256='d' * 64, config=SimpleNamespace(run=SimpleNamespace(output_dir=str(root))))
    monkeypatch.setattr('gf_cli.hpo_config.load_hpo_config', lambda path: campaign)
    namespace = root / '.operation-control/experiment/run'
    invocation = {'schema_version': 1, 'kind': 'hpo-operation-invocation-data-v1', 'context': context,
        'campaign_config_sha256': campaign.sha256, 'config_sha256': loaded.sha256,
        'config_inputs': list(campaign.inputs), 'mode': 'fresh', 'pause_after_update': None,
        'launcher_path': '/original/launcher.py', 'launcher_sha256': 'e' * 64,
        'python_executable': '/original/python'}
    global_root = tmp_path / 'global'; monkeypatch.setattr(hpo_gpu_lease, '_GLOBAL_ROOT', global_root)
    admission = {'schema_version': 1, 'kind': 'hpo-gpu-admission', 'ticket': 'f' * 32,
        'context': context, 'boot_id': 'boot', 'owner_pid': 5, 'admitted_unix': 10.}
    termination = {'schema_version': 1, 'kind': 'hpo-gpu-termination', 'ticket': admission['ticket'],
        'admission_sha256': digest(admission), 'context_sha256': digest(context), 'boot_id': 'boot',
        'confirmed_unix': 20., 'resources': []}
    paths = {namespace / 'invocations' / (context['operation_id'] + '.json'): invocation,
        global_root / 'jobs' / admission['ticket'] / 'admission.json': admission,
        global_root / 'jobs' / admission['ticket'] / 'termination.json': termination}
    for path, row in paths.items(): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row))
    stdout = {'operation_context': dict(context), 'gpu_termination': dict(termination)}
    observer = {'command': {'pid': 5, 'start_ticks': 2, 'argv': ['/original/python', '/original/launcher.py',
        '--experiment-id', 'experiment', '--mode', 'fresh']}, 'clock': {'boot_id': 'boot'}}
    if fault == 'copied': observer['command']['pid'] = 6
    elif fault == 'boot': observer['clock']['boot_id'] = 'other'
    elif fault == 'context': stdout['operation_context']['operation_id'] = '0' * 32
    elif fault == 'invocation': invocation['config_sha256'] = '0' * 64; next(iter(paths)).write_text(json.dumps(invocation))
    elif fault == 'termination': stdout['gpu_termination']['confirmed_unix'] = 21.
    elif fault == 'mode': observer['command']['argv'][-1] = 'resume'
    scan = SealedScan(max_owned_bytes=10**7, deadline=time.monotonic()+10)
    result = tool._legacy_hpo_binding(parent, loaded, config_path=tmp_path / 'config.toml',
        stdout=stdout, observer=observer, scan=scan)
    scan.finalize()
    return result


def test_hpo_calibration_off_binds_original_physical_data_without_live_job(tmp_path, monkeypatch):
    result = hpo_binding_fixture(tmp_path, monkeypatch)
    assert result['pid'] == 5 and result['boot_id'] == 'boot'
    assert result['ticket'] == 'f' * 32


@pytest.mark.parametrize('fault', ['copied', 'boot', 'context', 'invocation', 'termination', 'mode'])
def test_hpo_copied_stdout_or_foreign_original_records_refuse(tmp_path, monkeypatch, fault):
    with pytest.raises((ContractError, FileNotFoundError)):
        hpo_binding_fixture(tmp_path, monkeypatch, fault=fault)



def test_observer_large_valid_stream_accounts_retained_projection_not_history(tmp_path):
    import time

    from gf_models.sharded_arrays import MAX_MANIFEST_BYTES
    from gf_models.structured_sealed import SealedScan
    from gf_training.structured_rollout import _owned_bytes
    from tools.rust_player_verify_operation import observer_data
    line = b'{"monotonic_ns":100,"processes":[],"errors":[]}' + b' ' * 8192 + b'\n'
    count = MAX_MANIFEST_BYTES // len(line) + 1
    with (tmp_path / 'process.jsonl').open('wb') as out:
        for _ in range(count): out.write(line)
    (tmp_path / 'pss.jsonl').write_bytes(b'')
    summary = {'schema_version': 1, 'kind': 'rust-player-process-observation-v1',
        'command': {'argv': ['python'], 'pid': 5, 'start_ticks': 2,
            'start_monotonic_ns': 100, 'finish_monotonic_ns': 300, 'wall_ns': 200},
        'clock': {'name': 'CLOCK_MONOTONIC', 'boot_id': 'boot', 'hz': 100, 'page_bytes': 4096, 'kernel': 'test'},
        'observer': {'pid': 1, 'start_monotonic_ns': 90, 'finish_monotonic_ns': 310, 'user_ticks': 1, 'system_ticks': 2},
        'command_wait': {'exit_code': 0, 'signal': None, 'observed': True},
        'sampling': {'enabled': True, 'cpu_interval_ns': 100000000, 'pss_interval_ns': 1000000000},
        'peak_complete_pss_bytes': None,
        'coverage': {'process_samples': count, 'pss_sweeps': 0, 'complete_pss_sweeps': 0,
            'partial_pss_sweeps': 0, 'peak_eligible_pss_sweeps': 0, 'observed_processes': 1},
        'errors': [], 'limitations': []}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    scan = SealedScan(max_owned_bytes=8 * 1024**2, deadline=time.monotonic()+20)
    result = observer_data(tmp_path, scan=scan)
    assert result['coverage']['process_samples'] == count
    assert _owned_bytes(result) <= scan.retained < scan.limit
    scan.finalize()
    with pytest.raises(ContractError, match='reservation'):
        observer_data(tmp_path, scan=SealedScan(max_owned_bytes=1024**2, deadline=time.monotonic()+20))



def test_legacy_mandatory_metrics_allow_unsupported_pyo3_and_broker_failure_counters(tmp_path):
    event = metric(); event['owner']['role'] = 'legacy-coordinator'
    event['counters'] = dict.fromkeys(event['counters'])
    event['counters'].update(event_page_successes=0, event_page_failures=0, event_transport_attempts=0,
        inference_successes=0)
    broker = metric(); broker['owner']['role'] = 'legacy-shared-broker'
    broker['counters'] = dict.fromkeys(broker['counters'])
    broker['counters'].update(inference_issued=0, inference_successes=0, inference_failures=0)
    broker['histograms'] = dict.fromkeys(broker['histograms'])
    broker['batch_rows'] = {'method': 'complete-response-size-inversion-v1', 'request_size_counts': {},
        'batch_size_counts': {}, 'issued_requests': 0, 'completed_responses': 0,
        'quiescent_start': True, 'quiescent_end': True, 'failures': 0}
    scope = event['scope']
    result = telemetry_fixture(tmp_path,
        [event, broker, phase(scope | {'invocation_id': None}, at=10),
            phase(scope | {'invocation_id': None}, event='phase_end', at=20),
            phase(scope, at=50), phase(scope, event='phase_end', at=250)],
        updates=[{'batch_id': scope['batch_id'], 'invocation_id': 'inv', 'collection_seconds': 1.}])
    assert result['complete']
    assert result['collection_coverage'][0]['mandatory_metrics'] == {
        'event_page': True, 'inference_roundtrip': True, 'actual_batch': True, 'pyo3': None}
    assert all(row['pyo3_calls'] is None for row in result['metrics'])
    assert result['collection_coverage'][0]['host_phase_seconds'] == [2e-7]



@pytest.mark.parametrize('fault', ['unreachable', 'min-outside-lowest', 'max-outside-highest', 'overflow-only-extrema'])
def test_histogram_occupied_bins_and_extrema_match_integer_recipe(fault):
    from tools.rust_player_verify_operation import histogram_data
    h = Histogram(); h.add(32)
    row = h.snapshot()
    if fault == 'unreachable':
        row['counts'] = [0] * 2049; row['counts'][2] = 1
        row['min_ns'] = row['max_ns'] = 1
    elif fault == 'min-outside-lowest': row['min_ns'] = 1
    elif fault == 'max-outside-highest': row['max_ns'] = 33
    else:
        row['counts'] = [0] * 2049; row['overflow_count'] = 1
    with pytest.raises(ContractError): histogram_data(row)


def test_histogram_exact_integer_and_overflow_edges_remain_valid():
    from tools.rust_player_verify_operation import histogram_data
    h = Histogram()
    for n in [0, 1, 2, 3, 31, 32, 33, 63, 64, 2**64-1, 2**64]: h.add(n)
    result = histogram_data(h.snapshot())
    assert result['count'] == 11 and result['overflow_count'] == 1
    assert result['min_ns'] == 0 and result['max_ns'] == 2**64-1
    overflow = Histogram(); overflow.add(2**64)
    assert histogram_data(overflow.snapshot())['p95_ns'] == [2**64, None]


@pytest.mark.parametrize('family,counter', [('event_page', 'event_page_successes'),
    ('inference_roundtrip', 'inference_successes')])
@pytest.mark.parametrize('role', ['native-runtime', 'legacy-coordinator'])
@pytest.mark.parametrize('shape', ['empty', 'missing'])
def test_complete_producer_requires_one_histogram_sample_per_success(family, counter, role, shape):
    from tools.rust_player_verify_operation import metric_data
    row = metric(); row['owner']['role'] = role; row['counters'][counter] = 4
    if shape == 'missing': row['histograms'][family] = None
    if family == 'inference_roundtrip' and role == 'native-runtime':
        row['counters']['inference_issued'] = 4
        row['batch_rows'] = {'method': 'native-issued-batches-v1', 'request_size_counts': None,
            'batch_size_counts': {'4': 1}, 'issued_requests': 4, 'completed_responses': 4,
            'quiescent_start': True, 'quiescent_end': True, 'failures': 0}
    with pytest.raises(ContractError, match='success.*sample|sample.*success'):
        metric_data(row, observation_id='obs', boot_id='boot', run_id='run', interval=(0, 300))
    h = Histogram()
    for n in range(4): h.add(n)
    row['histograms'][family] = h.snapshot()
    assert metric_data(row, observation_id='obs', boot_id='boot', run_id='run', interval=(0, 300))['complete']


def test_native_scope_comes_from_actual_frozen_invocation_chain():
    from tools.rust_player_verify_operation import _native_measurement_scope
    rows = [{'invocation_id': 'old', 'operation_id': 'old-op', 'current': False,
        'mode': 'fresh', 'committed_update': 1, 'invocation_history': ['old']},
        {'invocation_id': 'new', 'operation_id': 'new-op', 'current': True,
        'mode': 'resume', 'committed_update': 2, 'invocation_history': ['old', 'new']}]
    assert _native_measurement_scope(rows) == {'invocation_id': 'new', 'operation_id': 'new-op',
        'start_generation': 1, 'end_generation': 2, 'evidence': 'verified native frozen-state/completion chain'}
    rows[1]['invocation_history'] = ['new']
    with pytest.raises(ContractError): _native_measurement_scope(rows)


def test_legacy_scope_uses_finalized_counts_and_bound_actual_job():
    from tools.rust_player_verify_operation import _legacy_measurement_scope
    state = {'updates': 2, 'invocations': [
        {'mode': 'fresh', 'shared_service': {'finished': True, 'start_generation': 0,
            'end_generation': 1, 'execution': 'old', 'job_id': 'old-job'}},
        {'mode': 'resume', 'shared_service': {'finished': True, 'start_generation': 1,
            'end_generation': 2, 'execution': 'new', 'job_id': 'new-job'}}]}
    result = _legacy_measurement_scope(state, binding={'ticket': 'new-job'})
    assert result['start_generation'] == 1 and result['end_generation'] == 2
    assert result['invocation_id'] == 'new'
    with pytest.raises(ContractError): _legacy_measurement_scope(state, binding={'ticket': 'old-job'})
    assert _legacy_measurement_scope({'updates': 2, 'invocations': []}, binding=None) is None


def resume_telemetry(tmp_path, *, stale=False, unknown=False):
    row = metric(); row['scope']['batch_id'] = 'batch-000000' if stale else 'batch-000001'
    row['scope']['invocation_id'] = 'old' if stale else 'new'
    row['batch_rows'] = {'method': 'native-issued-batches-v1', 'request_size_counts': None,
        'batch_size_counts': {}, 'issued_requests': 0, 'completed_responses': 0,
        'quiescent_start': True, 'quiescent_end': True, 'failures': 0}
    scope = row['scope']
    return telemetry_fixture(tmp_path, [row, phase(scope, at=50), phase(scope, event='phase_end', at=250)],
        lane='native-ordinary', updates=[
            {'batch_id': 'batch-000000', 'invocation_id': 'old', 'generation': 1,
                'measurement_current': False, 'collection_seconds': 1.},
            {'batch_id': 'batch-000001', 'invocation_id': 'new', 'generation': 2,
                'measurement_current': True, 'collection_seconds': 1.}],
        measurement_scope=None if unknown else {'invocation_id': 'new', 'operation_id': 'op',
            'start_generation': 1, 'end_generation': 2})


def test_resume_requires_only_verified_current_batches_without_old_telemetry(tmp_path):
    result = resume_telemetry(tmp_path)
    assert result['complete']
    assert [row['batch_id'] for row in result['collection_coverage']] == ['batch-000001']


def test_resume_refuses_stale_batch_even_if_original_historical_batch_was_audited(tmp_path):
    with pytest.raises(ContractError): resume_telemetry(tmp_path, stale=True)


def test_unknown_current_invocation_scope_cannot_qualify_metrics(tmp_path):
    assert not resume_telemetry(tmp_path, unknown=True)['complete']



def test_current_rate_projection_uses_persisted_interval_and_never_file_presence(monkeypatch):
    from tools import rust_player_verify_operation as tool
    audit = {'updates': [{'generation': 1, 'batch_id': 'old'}, {'generation': 2, 'batch_id': 'current'}],
        'measurement_scope': {'start_generation': 1, 'end_generation': 2}}
    assert [row['batch_id'] for row in tool._measurement_updates(audit)] == ['current']
    audit['measurement_scope'] = None
    def unexpected_open(*args, **kwargs):
        pytest.fail('unknown current interval must never produce historical rate projections')
    monkeypatch.setattr('gf_training.group_pack_source.PackedSource', unexpected_open)
    assert tool._project_collections(audit, limit=10**7, deadline=1., retained=(), native=True) == []


def test_native_current_scope_rejects_a_stale_observed_original():
    from tools.rust_player_verify_operation import _native_measurement_scope
    with pytest.raises(ContractError):
        _native_measurement_scope([
            {'invocation_id': 'old', 'operation_id': 'old-op', 'current': True,
                'mode': 'fresh', 'committed_update': 1, 'invocation_history': ['old']},
            {'invocation_id': 'new', 'operation_id': 'new-op', 'current': False,
                'mode': 'resume', 'committed_update': 2, 'invocation_history': ['old', 'new']}])


def test_native_resume_refuses_wrong_current_invocation_even_with_current_batch(tmp_path):
    row = metric(); row['scope']['invocation_id'] = 'old'
    with pytest.raises(ContractError):
        telemetry_fixture(tmp_path, [row, phase(row['scope'], at=50), phase(row['scope'], event='phase_end', at=250)],
            lane='native-ordinary', updates=[{'batch_id': 'batch-000000', 'invocation_id': 'new',
                'generation': 2, 'collection_seconds': 1.}], measurement_scope={
                    'invocation_id': 'new', 'operation_id': 'op', 'start_generation': 1, 'end_generation': 2})
