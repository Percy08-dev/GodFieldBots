"""Pure bounded telemetry values. No clocks, environment, I/O or policy imports."""
from __future__ import annotations

from typing import Any

RECIPE = 'ns-exp2-sub32-v1'
COUNTERS = ('event_page_successes', 'event_page_failures', 'event_transport_attempts',
    'inference_issued', 'inference_successes', 'inference_cancellations',
    'inference_expiries', 'inference_rejections', 'inference_failures',
    'pyo3_next_batch', 'pyo3_submit_results', 'pyo3_other',
    'writer_admissions', 'writer_admission_failures')
HISTOGRAMS = ('event_page', 'inference_roundtrip', 'storage_admission_wait',
    'storage_queue_age', 'collection_storage_serialization_wait')
STAGES = ('assembly', 'cpu_share', 'h2d', 'forward', 'sampling', 'return',
    'native_wait', 'native_submit')


class Histogram:
    """2049 bins, with explicit overflow. Quantiles are nearest-rank bin bounds."""
    def __init__(self, *, scope: dict[str, Any] | None = None) -> None:
        self.scope = None if scope is None else dict(scope)
        self.counts = [0] * 2049
        self.count = self.overflow = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.sum_seconds = 0.0

    def add(self, ns: int) -> None:
        if type(ns) is not int or ns < 0:
            raise ValueError('nonnegative integer nanoseconds required')
        self.count += 1
        self.sum_seconds += ns / 1_000_000_000
        if ns >= 2**64:
            self.overflow += 1
            return
        self.minimum = ns if self.minimum is None else min(self.minimum, ns)
        self.maximum = ns if self.maximum is None else max(self.maximum, ns)
        if ns == 0:
            index = 0
        else:
            e = ns.bit_length() - 1
            index = 1 + 32 * e + ((ns - 2**e) * 32 // 2**e)
        self.counts[index] += 1

    def merge(self, other: Histogram) -> None:
        if self.scope != other.scope:
            raise ValueError("histogram scope definitions differ")
        self.counts = [a + b for a, b in zip(self.counts, other.counts, strict=True)]
        self.count += other.count
        self.overflow += other.overflow
        self.sum_seconds += other.sum_seconds
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)  # type: ignore[type-var]

    def quantile_bounds(self, numerator: int, denominator: int) -> tuple[int, int | None] | None:
        if not 0 < numerator <= denominator:
            raise ValueError('quantile must be in (0, 1]')
        if not self.count:
            return None
        rank = (numerator * self.count + denominator - 1) // denominator
        accumulated = 0
        for index, count in enumerate(self.counts):
            accumulated += count
            if accumulated >= rank:
                if index == 0:
                    return (0, 1)
                e, s = divmod(index - 1, 32)
                base = 2**e
                return (base + (s * base + 31) // 32,
                    base + ((s + 1) * base + 31) // 32)
        return (2**64, None)

    def snapshot(self) -> dict[str, Any]:
        return {'recipe': RECIPE, 'count': self.count, 'counts': list(self.counts),
            'overflow_count': self.overflow, 'min_ns': self.minimum,
            'max_ns': self.maximum, 'sum_seconds': self.sum_seconds}


def invert_response_sizes(counts: dict[int, int], *, issued_requests: int,
        completed_responses: int, quiescent_start: bool, quiescent_end: bool,
        failures: int, coverage_complete: bool) -> dict[int, int] | None:
    """Full response-size inversion requires independently established coverage."""
    if (not coverage_complete or not quiescent_start or not quiescent_end or failures
            or issued_requests != completed_responses
            or sum(counts.values()) != completed_responses
            or any(type(b) is not int or b <= 0 or type(n) is not int or n < 0 or n % b
                for b, n in counts.items())):
        return None
    return {b: n // b for b, n in counts.items()}


class PhaseMetrics:
    """One current scope only. Missing owner capabilities remain null."""
    def __init__(self, *, scope: dict[str, Any], start_ns: int,
            counters: tuple[str, ...] = COUNTERS, histograms: tuple[str, ...] = HISTOGRAMS,
            queue_model: str | None = None) -> None:
        self.scope, self.start_ns = dict(scope), start_ns
        self.counters: dict[str, int | None] = {key: 0 if key in counters else None for key in COUNTERS}
        self.histograms = {key: Histogram(scope=self.scope) if key in histograms else None for key in HISTOGRAMS}
        self.errors: list[str] = []
        self.batch_rows: dict[str, Any] = dict.fromkeys(('method', 'request_size_counts',
            'batch_size_counts', 'issued_requests', 'completed_responses',
            'quiescent_start', 'quiescent_end', 'failures'))
        self.stages: dict[str, Any] = dict.fromkeys(STAGES)
        self.queue: dict[str, Any] = dict.fromkeys(('model', 'queued_bytes_peak',
            'in_service_bytes_peak', 'outstanding_bytes_peak', 'depth_peak', 'failed_attempts'))
        self.queue['model'] = queue_model

    def error(self, value: str) -> None:
        # Fixed diagnostic categories, not an unbounded per-request failure log.
        if value not in self.errors:
            self.errors.append(value)

    def increment(self, name: str, value: int = 1) -> None:
        current = self.counters[name]
        if current is not None:
            self.counters[name] = current + value

    def interval(self, name: str, start: int, end: int) -> None:
        h = self.histograms[name]
        if end < start:
            self.error('invalid_clock_interval:' + name)
        elif h is not None:
            h.add(end - start)

    def snapshot(self, *, end_ns: int, complete: bool = True) -> dict[str, Any]:
        if end_ns < self.start_ns:
            self.error('invalid_scope_clock_interval')
        return {'scope': dict(self.scope), 'start_monotonic_ns': self.start_ns,
            'end_monotonic_ns': end_ns, 'complete': complete and not self.errors,
            'errors': list(self.errors), 'counters': dict(self.counters),
            'histograms': {key: None if h is None else h.snapshot() for key, h in self.histograms.items()},
            'batch_rows': dict(self.batch_rows), 'stage_seconds': dict(self.stages),
            'queue': dict(self.queue)}


class StorageTicket:
    def __init__(self, start_ns: int) -> None:
        self.start_ns = start_ns
        self.admitted_ns: int | None = None
        self.size = 0
        self.state = 'waiting'


class StorageMetrics:
    """Observational queue ownership. Caller holds its existing short queue lock."""
    def __init__(self, *, scope: dict[str, Any], start_ns: int, model: str) -> None:
        self.metrics = PhaseMetrics(scope=scope, start_ns=start_ns,
            counters=('writer_admissions', 'writer_admission_failures'),
            histograms=('storage_admission_wait', 'storage_queue_age'), queue_model=model)
        self.metrics.queue.update(queued_bytes_peak=0, in_service_bytes_peak=0,
            outstanding_bytes_peak=0, depth_peak=0, failed_attempts=0)
        self.queued = self.in_service = self.depth = 0

    @staticmethod
    def attempt(now_ns: int) -> StorageTicket:
        return StorageTicket(now_ns)

    def _peaks(self) -> None:
        for key, value in (('queued_bytes_peak', self.queued),
                ('in_service_bytes_peak', self.in_service),
                ('outstanding_bytes_peak', self.queued + self.in_service), ('depth_peak', self.depth)):
            self.metrics.queue[key] = max(self.metrics.queue[key], value)

    def admit(self, ticket: StorageTicket, now_ns: int, size: int) -> None:
        ticket.admitted_ns, ticket.size, ticket.state = now_ns, size, 'queued'
        self.metrics.interval('storage_admission_wait', ticket.start_ns, now_ns)
        self.metrics.increment('writer_admissions')
        self.queued += size
        self.depth += 1
        self._peaks()

    def dispatch(self, ticket: StorageTicket, now_ns: int) -> None:
        assert ticket.admitted_ns is not None and ticket.state == 'queued'
        self.metrics.interval('storage_queue_age', ticket.admitted_ns, now_ns)
        ticket.state = 'in_service'
        self.queued -= ticket.size
        self.in_service += ticket.size
        self.depth -= 1
        self._peaks()

    def release(self, ticket: StorageTicket) -> None:
        if ticket.state == 'queued':
            self.queued -= ticket.size
            self.depth -= 1
        elif ticket.state == 'in_service':
            self.in_service -= ticket.size
        ticket.state = 'released'

    def fail(self, ticket: StorageTicket, now_ns: int) -> None:
        if ticket.state == 'waiting':
            self.metrics.interval('storage_admission_wait', ticket.start_ns, now_ns)
            self.metrics.increment('writer_admission_failures')
            self.metrics.queue['failed_attempts'] += 1
        self.release(ticket)


class ResponseWindow:
    """Full-window observational coverage; pending keys only follow actual authorizations."""
    def __init__(self) -> None:
        self.issued = self.completed = self.failures = 0
        self.coverage = True
        self.pending: dict[str, bool] = {}
        self.counts: dict[int, int] = {}

    def authorize(self, endpoint: str) -> None:
        if endpoint in self.pending:
            self.coverage = False
        self.pending[endpoint] = False
        self.issued += 1

    def delivered(self, endpoint: str, size: int) -> None:
        if self.pending.get(endpoint) is not False:
            self.coverage = False
        if endpoint in self.pending:
            self.pending[endpoint] = True
        self.completed += 1
        self.counts[size] = self.counts.get(size, 0) + 1

    def failure(self) -> None:
        self.failures += 1

    def retire(self, endpoint: str, *, completed: bool) -> None:
        seen = self.pending.pop(endpoint, None)
        if not completed or seen is not True:
            self.failures += 1
            self.coverage = False

    def snapshot(self, *, quiescent_end: bool) -> dict[str, Any]:
        inversion = invert_response_sizes(self.counts, issued_requests=self.issued,
            completed_responses=self.completed, quiescent_start=True,
            quiescent_end=quiescent_end, failures=self.failures,
            coverage_complete=self.coverage and not self.pending)
        return {'method': 'complete-response-size-inversion-v1',
            'request_size_counts': dict(self.counts), 'batch_size_counts': inversion,
            'issued_requests': self.issued, 'completed_responses': self.completed,
            'quiescent_start': True, 'quiescent_end': quiescent_end, 'failures': self.failures}
