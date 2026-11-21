"""Bounded diagnostics math; these records have no scientific authority."""
import pytest

from gf_common.performance_metrics import Histogram, PhaseMetrics, invert_response_sizes


def test_histogram_fixed_bins_bounds_rank_and_overflow():
    h = Histogram()
    for n in (0, 1, 2, 31, 32, 33, 63, 64, 2**64 - 1, 2**64):
        h.add(n)
    record = h.snapshot()
    assert len(record['counts']) == 2049
    assert sum(record['counts']) + record['overflow_count'] == record['count'] == 10
    assert h.quantile_bounds(95, 100) == (2**64, None)
    finite = Histogram()
    finite.add(2**64 - 1)
    assert finite.quantile_bounds(99, 100) == (63 * 2**58, 2**64)
    assert record['min_ns'] == 0 and record['max_ns'] == 2**64 - 1


@pytest.mark.parametrize('n', [1, 2, 3, 7, 15, 31, 32, 33, 63, 64, 999, 2**63])
def test_histogram_bounds_contain_actual_sample(n):
    h = Histogram()
    h.add(n)
    low, high = h.quantile_bounds(95, 100)
    assert low <= n < high
    if n < 32:
        assert (low, high) == (n, n + 1)


def test_merge_empty_and_invalid_intervals():
    h = Histogram()
    assert h.quantile_bounds(95, 100) is None
    h.add(0)
    other = Histogram()
    other.add(42)
    h.merge(other)
    assert h.snapshot()['count'] == 2
    with pytest.raises(ValueError):
        h.add(-1)
    with pytest.raises(ValueError):
        h.add(True)


def test_scope_partition_closed_record_and_negative_clock_is_incomplete():
    m = PhaseMetrics(scope={'run_id': 'run', 'operation_id': None, 'invocation_id': 'i', 'batch_id': 'b1', 'phase': 'collection'}, start_ns=10)
    m.interval('event_page', 10, 15)
    m.interval('inference_roundtrip', 20, 19)
    m.increment('event_page_successes')
    row = m.snapshot(end_ns=30)
    assert row['histograms']['event_page']['count'] == 1
    assert not row['complete'] and row['errors']
    assert row['histograms']['inference_roundtrip']['count'] == 0
    assert set(row['counters']) == {'event_page_successes', 'event_page_failures', 'event_transport_attempts', 'inference_issued', 'inference_successes', 'inference_cancellations', 'inference_expiries', 'inference_rejections', 'inference_failures', 'pyo3_next_batch', 'pyo3_submit_results', 'pyo3_other', 'writer_admissions', 'writer_admission_failures'}
    with pytest.raises(KeyError):
        m.increment('arbitrary')


def test_batch_inversion_requires_full_coverage_not_divisibility():
    facts = {'issued_requests': 6, 'completed_responses': 6, 'quiescent_start': True, 'quiescent_end': True, 'failures': 0, 'coverage_complete': True}
    assert invert_response_sizes({2: 2, 4: 4}, **facts) == {2: 1, 4: 1}
    for key, bad in [('coverage_complete', False), ('quiescent_start', False), ('quiescent_end', False), ('failures', 1), ('issued_requests', 7), ('completed_responses', 5)]:
        assert invert_response_sizes({2: 2, 4: 4}, **(facts | {key: bad})) is None
    assert invert_response_sizes({4: 6}, **facts) is None


def test_storage_observation_zero_failed_wait_and_separate_queue_ownership():
    from gf_common.performance_metrics import StorageMetrics
    q = StorageMetrics(scope={}, start_ns=0, model='thread-fifo')
    first = q.attempt(1)
    q.admit(first, 1, 10)
    second = q.attempt(2)
    q.admit(second, 12, 20)
    q.dispatch(first, 15)
    assert q.metrics.queue['queued_bytes_peak'] == 30
    assert q.metrics.queue['in_service_bytes_peak'] == 10
    assert q.metrics.queue['depth_peak'] == 2
    q.release(first)
    q.dispatch(second, 20)
    q.release(second)
    failed = q.attempt(21)
    q.fail(failed, 25)
    row = q.metrics.snapshot(end_ns=30)
    assert row['counters']['writer_admissions'] == 2
    assert row['counters']['writer_admission_failures'] == 1
    assert row['histograms']['storage_admission_wait']['count'] == 3
    assert row['histograms']['storage_admission_wait']['min_ns'] == 0
    assert row['histograms']['storage_queue_age']['count'] == 2
    assert row['queue']['failed_attempts'] == 1
    assert q.queued == q.in_service == q.depth == 0


def test_response_window_covers_failures_before_admission_and_all_exact_responses():
    from gf_common.performance_metrics import ResponseWindow
    w = ResponseWindow()
    w.authorize('own')
    w.delivered('own', 1)
    w.retire('own', completed=True)
    assert w.snapshot(quiescent_end=True)['batch_size_counts'] == {1: 1}
    expired = ResponseWindow()
    expired.failure()
    assert expired.snapshot(quiescent_end=True)['batch_size_counts'] is None
    assert expired.snapshot(quiescent_end=True)['failures'] == 1
    duplicate = ResponseWindow()
    duplicate.authorize('own')
    duplicate.delivered('own', 1)
    duplicate.delivered('own', 1)
    duplicate.retire('own', completed=True)
    assert duplicate.snapshot(quiescent_end=True)['batch_size_counts'] is None
    foreign = ResponseWindow()
    foreign.delivered('foreign', 1)
    assert not foreign.pending
    assert foreign.snapshot(quiescent_end=True)['batch_size_counts'] is None
    incomplete = ResponseWindow()
    incomplete.authorize('own')
    assert incomplete.snapshot(quiescent_end=False)['batch_size_counts'] is None


def test_histogram_merge_requires_matching_scope_definition():
    left = PhaseMetrics(scope={'batch_id': 'one'}, start_ns=0)
    right = PhaseMetrics(scope={'batch_id': 'two'}, start_ns=0)
    with pytest.raises(ValueError, match='scope'):
        left.histograms['event_page'].merge(right.histograms['event_page'])
