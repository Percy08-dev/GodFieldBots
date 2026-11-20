"""Bounded post-exit experiment DATA verification; never grants Source authority."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, closed, integer
from gf_common.performance_metrics import (
    COUNTERS,
    HISTOGRAMS,
    RECIPE,
    STAGES,
    Histogram,
    invert_response_sizes,
)

_PROCESS_READ_ERROR = re.compile(r'process ([1-9][0-9]*)/(0|[1-9][0-9]*): .+', re.DOTALL)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ContractError('finite nonnegative diagnostic number required')
    return float(value)


def histogram_data(row: Json | None) -> Json | None:
    if row is None:
        return None
    closed(row, {'recipe', 'count', 'counts', 'overflow_count', 'min_ns', 'max_ns', 'sum_seconds'})
    if row['recipe'] != RECIPE or type(row['counts']) is not list or len(row['counts']) != 2049:
        raise ContractError('supported fixed histogram required')
    for value in [row['count'], row['overflow_count'], *row['counts']]:
        integer(value, 0)
    if sum(row['counts']) + row['overflow_count'] != row['count']:
        raise ContractError('histogram counts differ')
    occupied = []
    for index, count in enumerate(row['counts']):
        if not count:
            continue
        if index == 0:
            lower, upper = 0, 1
        else:
            exponent, sub = divmod(index - 1, 32)
            base = 2**exponent
            lower, upper = base + (sub * base + 31) // 32, base + ((sub + 1) * base + 31) // 32
        if lower >= upper:
            raise ContractError('occupied histogram bin has no achievable integer sample')
        occupied.append((lower, upper))
    finite = row['count'] - row['overflow_count']
    if finite:
        integer(row['min_ns'], 0, 2**64 - 1)
        integer(row['max_ns'], row['min_ns'], 2**64 - 1)
        if not (occupied[0][0] <= row['min_ns'] < occupied[0][1]
                and occupied[-1][0] <= row['max_ns'] < occupied[-1][1]):
            raise ContractError('histogram finite extrema differ from actual occupied bins')
    elif row['min_ns'] is not None or row['max_ns'] is not None:
        raise ContractError('empty finite histogram must retain null extrema')
    _number(row['sum_seconds'])
    value = Histogram()
    value.count, value.counts, value.overflow = row['count'], list(row['counts']), row['overflow_count']
    return {**row, **{f'p{p}_ns': list(bounds) if (bounds := value.quantile_bounds(p, 100)) else None
        for p in (95, 99)}}


def _scope(row: Json) -> None:
    closed(row, {'run_id', 'operation_id', 'invocation_id', 'batch_id', 'phase'})
    if row['phase'] not in ('startup', 'collection', 'update', 'save', 'retirement', 'cold', 'evaluation', 'idle'):
        raise ContractError('unknown diagnostic phase')
    if row['phase'] != 'collection' and row['batch_id'] is not None:
        raise ContractError('noncollection batch scope must be null')


def metric_data(row: Json, *, observation_id: str, boot_id: str, run_id: str,
                interval: tuple[int, int], startup_owner: tuple[int, int] | None = None) -> Json:
    closed(row, {'schema_version', 'kind', 'observation_id', 'owner', 'clock', 'scope',
        'start_monotonic_ns', 'end_monotonic_ns', 'complete', 'errors', 'counters',
        'histograms', 'batch_rows', 'stage_seconds', 'queue'})
    _pairing(row, observation_id=observation_id, boot_id=boot_id, run_id=run_id, startup_owner=startup_owner)
    if row['kind'] != 'rust-player-phase-metrics-v1':
        raise ContractError('metrics1 required')
    start, end = row['start_monotonic_ns'], row['end_monotonic_ns']
    integer(start, interval[0], interval[1]); integer(end, start, interval[1])
    if type(row['complete']) is not bool or type(row['errors']) is not list:
        raise ContractError('explicit diagnostic completeness required')
    closed(row['counters'], set(COUNTERS)); closed(row['histograms'], set(HISTOGRAMS))
    closed(row['stage_seconds'], set(STAGES))
    for value in row['counters'].values():
        if value is not None:
            integer(value, 0)
    for value in row['stage_seconds'].values():
        if value is not None:
            closed(value, {'method', 'seconds'})
            if value['method'] != 'synchronized-host-wall-v1':
                raise ContractError('unknown host stage timing method')
            _number(value['seconds'])
    closed(row['queue'], {'model', 'queued_bytes_peak', 'in_service_bytes_peak',
        'outstanding_bytes_peak', 'depth_peak', 'failed_attempts'})
    if row['queue']['model'] not in (None, 'thread-fifo', 'native-ingress', 'legacy-parent-rpc'):
        raise ContractError('unknown storage queue measurement model')
    for key, value in row['queue'].items():
        if key != 'model' and value is not None:
            integer(value, 0)
    batches = closed(row['batch_rows'], {'method', 'request_size_counts', 'batch_size_counts',
        'issued_requests', 'completed_responses', 'quiescent_start', 'quiescent_end', 'failures'})
    histogram = None
    failures = [row['counters'][key] for key in ('inference_failures', 'inference_cancellations',
        'inference_expiries', 'inference_rejections')]
    complete = row['complete'] and not row['errors']
    if batches['method'] is not None:
        for key in ('issued_requests', 'completed_responses', 'failures'):
            integer(batches[key], 0)
        for key in ('quiescent_start', 'quiescent_end'):
            if type(batches[key]) is not bool:
                raise ContractError('actual batch quiescence required')
        coverage = (complete and row['counters']['inference_failures'] == 0 and all(value in (0, None) for value in failures)
            and row['counters']['inference_issued'] == batches['issued_requests']
            and row['counters']['inference_successes'] == batches['completed_responses'])
        if batches['method'] == 'complete-response-size-inversion-v1':
            counts = _batch_map(batches['request_size_counts'])
            histogram = invert_response_sizes(counts, issued_requests=batches['issued_requests'],
                completed_responses=batches['completed_responses'], quiescent_start=batches['quiescent_start'],
                quiescent_end=batches['quiescent_end'], failures=batches['failures'], coverage_complete=coverage)
            if histogram is not None and {str(k): v for k, v in histogram.items()} != batches['batch_size_counts']:
                raise ContractError('recorded batch inversion differs')
        elif batches['method'] == 'native-issued-batches-v1':
            histogram = _batch_map(batches['batch_size_counts'])
            if sum(k * v for k, v in histogram.items()) != batches['issued_requests']:
                raise ContractError('native batch rows differ from issued requests')
            if not coverage or not batches['quiescent_start'] or not batches['quiescent_end'] or batches['failures']:
                histogram = None
        else:
            raise ContractError('unknown batch measurement method')
    histograms = {key: histogram_data(value) for key, value in row['histograms'].items()}
    if complete and row['owner']['role'] in ('native-runtime', 'legacy-coordinator'):
        for family, counter in (('event_page', 'event_page_successes'), ('inference_roundtrip', 'inference_successes')):
            samples, successes = histograms[family], row['counters'][counter]
            if successes is not None and (samples is None or samples['count'] != successes):
                raise ContractError('diagnostic success counter differs from actual histogram samples:' + family)
    pyo3 = [row['counters'][key] for key in ('pyo3_next_batch', 'pyo3_submit_results', 'pyo3_other')]
    return {'owner': row['owner'], 'scope': row['scope'], 'complete': complete,
        'start_monotonic_ns': start, 'end_monotonic_ns': end, 'errors': row['errors'],
        'counters': row['counters'], 'histograms': histograms,
        'batch_rows': batches, 'batch_histogram': None if histogram is None else {str(k): v for k, v in histogram.items()},
        'pyo3_calls': None if None in pyo3 else sum(pyo3),
        'stage_seconds': row['stage_seconds'], 'stage_scope': 'synchronized host wall; not CUDA event time',
        'queue': row['queue']}


def _batch_map(value: Any) -> dict[int, int]:
    if type(value) is not dict:
        raise ContractError('actual batch size count map required')
    result = {}
    for key, count in value.items():
        if type(key) is not str or not key.isdecimal() or str(int(key)) != key:
            raise ContractError('canonical positive batch size required')
        size = int(key); integer(size, 1); integer(count, 0)
        result[size] = count
    return result


def _pairing(row: Json, *, observation_id: str, boot_id: str, run_id: str,
             startup_owner: tuple[int, int] | None = None) -> None:
    if type(row['schema_version']) is not int or row['schema_version'] != 1 or row['observation_id'] != observation_id:
        raise ContractError('diagnostic observation identity/version differs')
    if row['clock'] != {'name': 'CLOCK_MONOTONIC', 'boot_id': boot_id}:
        raise ContractError('diagnostic boot/clock differs')
    closed(row['owner'], {'pid', 'start_ticks', 'role'})
    integer(row['owner']['pid'], 1); integer(row['owner']['start_ticks'], 0)
    if row['owner']['role'] not in ('host', 'native-runtime', 'legacy-coordinator',
            'legacy-shared-broker', 'legacy-gpu-worker', 'legacy-participant',
            'native-supervisor', 'native-inference', 'writer'):
        raise ContractError('unknown diagnostic owner role')
    _scope(row['scope'])
    # Only actual coordinator startup can precede assignment. A known foreign
    # run never becomes unassigned, and no later phase inherits this exception.
    unassigned_startup = (startup_owner == (row['owner']['pid'], row['owner']['start_ticks'])
        and row['owner']['role'] == 'host' and row['scope']['phase'] == 'startup'
        and all(row['scope'][key] is None for key in ('run_id', 'operation_id', 'invocation_id', 'batch_id')))
    if row['scope']['run_id'] != run_id and not unassigned_startup:
        raise ContractError('diagnostic run differs')


def journal_data(lines: Iterable[bytes], *, expected_identity: Json | None = None) -> Json:
    """Project acknowledged participant decisions only; no policy/validation authority."""
    iterator = iter(lines)
    first = next(iterator, None)
    if first is None:
        raise ContractError('participant Journal6 header is absent')
    header = json.loads(first)
    if header['kind'] != 'header' or type(header['schema_version']) is not int or header['schema_version'] != 6:
        raise ContractError('participant Journal6 header required')
    integer(header['viewer'], 1, 2)
    if expected_identity is not None and any(header.get(key) != value for key, value in expected_identity.items()):
        raise ContractError('participant Journal6 identity differs from authenticated collection/name')
    pending: tuple[int, str] | None = None
    decisions = accepted = total = maximum = 0
    cutoffs: dict[str, int] = {}
    for line in iterator:
        row = json.loads(line)
        if row['kind'] == 'decision':
            number = row['decision']; integer(number, 0)
            pending = number, row['command']['command_id']
            count = row['table']['count']; integer(count, 0)
            decisions += 1; total += count; maximum = max(maximum, count)
            for reason in row['search_report']['cutoff_reasons']:
                if type(reason) is not str:
                    raise ContractError('recorded cutoff name required')
                cutoffs[reason] = cutoffs.get(reason, 0) + 1
        elif row['kind'] == 'acknowledgement':
            if pending != (row['decision'], row['response']['command_id']):
                raise ContractError('acknowledgement differs from participant decision')
            accepted += 1; pending = None
        elif row['kind'] == 'rejection':
            pending = None
    return {'viewer': header['viewer'], 'decisions': decisions, 'accepted_decisions': accepted,
        'candidate_total': total, 'candidate_max': maximum, 'cutoff_reasons': cutoffs}


def collection_data(update: Json, journals: list[Json], *, ordinal: int) -> Json:
    seconds = _number(update['collection_seconds'])
    accepted = sum(row['accepted_decisions'] for row in journals)
    natural = update['source_proof']['closure_counts'].get('natural', 0)
    integer(natural, 0, update['actual_games'])
    return {'ordinal': ordinal, 'batch_id': update['batch_id'], 'actual_games': update['actual_games'],
        'planned_games': update['planned_games'], 'natural_games': natural,
        'accepted_decisions': accepted, 'journal_decisions': sum(row['decisions'] for row in journals),
        'collection_seconds': seconds, 'update_seconds': _number(update['update_seconds']),
        'rate_scope': 'persisted committed update collection_seconds; all participant acknowledgements',
        'games_per_second': natural / seconds if seconds else None,
        'decisions_per_second': accepted / seconds if seconds else None, 'seats': journals}


def check_observed_config(path: Path, *, argv: list[str]) -> None:
    if argv.count('--config') != 1 or '--set' in argv:
        raise ContractError('one explicit observed CONFIG required; override reconstruction unsupported')
    index = argv.index('--config') + 1
    if index == len(argv) or Path(argv[index]).absolute() != path.absolute():
        raise ContractError('observed CONFIG differs from requested original')


def _index_bytes(values: set[int]) -> int:
    return sys.getsizeof(values) + len(values) * sys.getsizeof(id(values))


def _rows(path: Path, scan: Any, *, retained_bytes: Callable[[], int] = lambda: 0) -> Iterable[Json]:
    from gf_models.sharded_arrays import _pairs
    with scan.file(path) as (source, _info):
        while True:
            scan.check()
            # Same JSON decode estimate as manifests, applied to the current row.
            # Bound readline BEFORE allocation; history size is not live workspace.
            maximum = max(0, (scan.limit - scan.retained - retained_bytes() - 65536) // 32)
            scan.charge(retained_bytes() + 65536, temporary=True)
            line = source.readline(maximum + 1)
            if not line:
                break
            workspace = 32 * len(line) + 65536
            scan.charge(retained_bytes() + workspace, temporary=True)
            row = json.loads(line, object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ContractError('nonfinite diagnostic JSON')))
            if type(row) is not dict:
                raise ContractError('diagnostic row object required')
            yield row
            # Consumer projections coexist with the row until next iteration.
            scan.charge(retained_bytes() + workspace, temporary=True)
            del row, line


def observer_data(directory: Path, *, scan: Any) -> Json:
    from gf_training.structured_rollout import _owned_bytes
    row = scan.object(directory / 'summary.json')
    closed(row, {'schema_version', 'kind', 'command', 'clock', 'observer', 'command_wait',
        'sampling', 'peak_complete_pss_bytes', 'coverage', 'errors', 'limitations'})
    if type(row['schema_version']) is not int or row['schema_version'] != 1 or row['kind'] != 'rust-player-process-observation-v1':
        raise ContractError('observer1 required')
    command = closed(row['command'], {'argv', 'pid', 'start_ticks', 'start_monotonic_ns', 'finish_monotonic_ns', 'wall_ns'})
    integer(command['pid'], 1); integer(command['start_ticks'], 0); integer(command['wall_ns'], 0)
    if type(command['argv']) is not list or not all(type(value) is str for value in command['argv']):
        raise ContractError('original observed command argv required')
    start, finish = command['start_monotonic_ns'], command['finish_monotonic_ns']
    integer(start, 0); integer(finish, start)
    if command['wall_ns'] != finish - start:
        raise ContractError('observer whole Child.wait wall differs')
    closed(row['command_wait'], {'exit_code', 'signal', 'observed'})
    if (type(row['command_wait']['exit_code']) is not int
            or row['command_wait'] != {'exit_code': 0, 'signal': None, 'observed': True}
            or row['command_wait']['observed'] is not True or row['errors']):
        raise ContractError('actual observed successful command and healthy observer required')
    closed(row['clock'], {'name', 'boot_id', 'hz', 'page_bytes', 'kernel'})
    if row['clock']['name'] != 'CLOCK_MONOTONIC':
        raise ContractError('observer monotonic clock required')
    integer(row['clock']['hz'], 1)
    closed(row['sampling'], {'enabled', 'cpu_interval_ns', 'pss_interval_ns'})
    if type(row['sampling']['enabled']) is not bool:
        raise ContractError('explicit observer sampling status required')
    closed(row['observer'], {'pid', 'start_monotonic_ns', 'finish_monotonic_ns', 'user_ticks', 'system_ticks'})
    coverage = closed(row['coverage'], {'process_samples', 'pss_sweeps', 'complete_pss_sweeps',
        'partial_pss_sweeps', 'peak_eligible_pss_sweeps', 'observed_processes'})
    for value in coverage.values(): integer(value, 0)
    if not row['sampling']['enabled'] and (any(coverage.values()) or row['peak_complete_pss_bytes'] is not None):
        raise ContractError('disabled observer sampling has coverage or a PSS peak')
    processes: dict[tuple[int, int], Json] = {}
    if row['sampling']['enabled']:
        processes[(command['pid'], command['start_ticks'])] = {
            'pid': command['pid'], 'start_ticks': command['start_ticks'], 'samples': 0,
            'first_monotonic_ns': None, 'last_monotonic_ns': None, 'cpu_first': None, 'cpu_last': None,
            'executable': None, 'argv': None, 'tasks': {}}
    samples = 0
    sample_counts: list[Json] = []
    sample_errors: dict[str, int] = {}
    # Discover precedes process_json: a primary read failure can be the ONLY
    # retained evidence of an owned identity. It supplies no sampled body data.
    error_only: dict[tuple[int, int], None] = {}
    projection_bytes = _owned_bytes((processes, sample_counts, sample_errors, error_only))
    sample_seen: set[int] = set()  # Only objects retained by sample_counts, never transient row IDs.
    def projection_estimate() -> int:
        # This dedup index also has an actual lifetime during streaming. It is
        # discarded with this reader, not charged as part of the returned DATA.
        return projection_bytes + _index_bytes(sample_seen)
    for sample in _rows(directory / 'process.jsonl', scan, retained_bytes=projection_estimate):
        closed(sample, {'monotonic_ns', 'processes', 'errors'})
        integer(sample['monotonic_ns'], start)
        if type(sample['errors']) is not list or any(type(error) is not str for error in sample['errors']):
            raise ContractError('sampled process errors must be a list of strings')
        samples += 1
        old_errors = _owned_bytes(sample_errors)
        old_error_identities = _owned_bytes(error_only) if sample['errors'] else None
        missing_process_body = False
        for error in sample['errors']:
            sample_errors[error] = sample_errors.get(error, 0) + 1
            if error.startswith('process ') and (match := _PROCESS_READ_ERROR.fullmatch(error)):
                identity = int(match[1]), int(match[2])
                integer(identity[0], 1, 2**32 - 1)
                integer(identity[1], 0, 2**64 - 1)
                missing_process_body = True
                if identity not in processes:
                    error_only[identity] = None
        if old_error_identities is not None:
            projection_bytes += _owned_bytes(error_only) - old_error_identities
        projection_bytes += _owned_bytes(sample_errors) - old_errors
        old_list = sys.getsizeof(sample_counts)
        sample_counts.append({'monotonic_ns': sample['monotonic_ns'],
            'processes': len(sample['processes']),
            'python_pids': [p['pid'] for p in sample['processes'] if p['executable'] is not None
                and Path(p['executable']).name.startswith('python')],
            'executable_coverage_complete': not missing_process_body
                and all(p['executable'] is not None for p in sample['processes'])})
        projection_bytes += sys.getsizeof(sample_counts) - old_list + _owned_bytes(sample_counts[-1], sample_seen)
        unresolved_bytes = None
        for proc in sample['processes']:
            closed(proc, {'pid', 'start_ticks', 'ppid', 'state', 'user_ticks', 'system_ticks',
                'rss_bytes', 'threads', 'executable', 'argv', 'tasks'})
            integer(proc['pid'], 1); integer(proc['start_ticks'], 0); integer(proc['ppid'], 0)
            for count_key in ('user_ticks', 'system_ticks', 'rss_bytes', 'threads'):
                if proc[count_key] is not None: integer(proc[count_key], 0)
            key = proc['pid'], proc['start_ticks']
            if key in error_only:
                if unresolved_bytes is None:
                    unresolved_bytes = _owned_bytes(error_only)
                del error_only[key]
            old_process = (_owned_bytes(processes[key]) if key in processes else 0) + sys.getsizeof(processes)
            if key not in processes:
                projection_bytes += _owned_bytes(key)
            value = processes.setdefault(key, {'pid': key[0], 'start_ticks': key[1], 'samples': 0,
                'first_monotonic_ns': None, 'cpu_first': None, 'tasks': {}})
            value['samples'] += 1
            value['first_monotonic_ns'] = value['first_monotonic_ns'] or sample['monotonic_ns']
            value['last_monotonic_ns'] = sample['monotonic_ns']
            ticks = None if proc['user_ticks'] is None or proc['system_ticks'] is None else proc['user_ticks'] + proc['system_ticks']
            if value['cpu_first'] is None: value['cpu_first'] = ticks
            value['cpu_last'] = ticks
            value['executable'], value['argv'] = proc['executable'], proc['argv']
            for task in proc['tasks'] or []:
                closed(task, {'tid', 'name'})
                value['tasks'][str(task['tid'])] = task['name']
            projection_bytes += _owned_bytes(value) + sys.getsizeof(processes) - old_process
        if unresolved_bytes is not None:
            projection_bytes += _owned_bytes(error_only) - unresolved_bytes
        # No historical raw row remains when the next bounded decode starts.
        if sample['processes']:
            del proc
        del sample
    complete = partial = eligible = sweeps = 0
    peak = None
    for sweep in _rows(directory / 'pss.jsonl', scan, retained_bytes=projection_estimate):
        closed(sweep, {'start_monotonic_ns', 'finish_monotonic_ns', 'complete', 'total_pss_bytes', 'processes', 'errors'})
        if type(sweep['complete']) is not bool:
            raise ContractError('explicit complete/partial PSS sweep required')
        integer(sweep['start_monotonic_ns'], 0)
        integer(sweep['finish_monotonic_ns'], sweep['start_monotonic_ns'])
        sweeps += 1
        complete += int(sweep['complete']); partial += int(not sweep['complete'])
        members = sweep['processes']
        qualifies = (sweep['complete'] and bool(members) and not sweep['errors']
            and start <= sweep['start_monotonic_ns'] <= sweep['finish_monotonic_ns'] <= finish)
        total = 0
        for member in members:
            closed(member, {'pid', 'start_ticks', 'start_monotonic_ns', 'finish_monotonic_ns', 'pss_bytes', 'error'})
            identity = member['pid'], member['start_ticks']
            if identity not in processes and identity not in error_only:
                raise ContractError('PSS member lacks a discovered process identity')
            qualifies = qualifies and member['error'] is None
            if member['pss_bytes'] is None: qualifies = False
            else:
                integer(member['pss_bytes'], 0); total += member['pss_bytes']
            qualifies = qualifies and sweep['start_monotonic_ns'] <= member['start_monotonic_ns'] <= member['finish_monotonic_ns'] <= sweep['finish_monotonic_ns']
        if sweep['complete'] and sweep['total_pss_bytes'] != total or not sweep['complete'] and sweep['total_pss_bytes'] is not None:
            raise ContractError('PSS complete/partial sum differs')
        if qualifies:
            eligible += 1; peak = total if peak is None else max(peak, total)
        if members:
            del member
        del sweep, members
    actual_coverage = {'process_samples': samples, 'pss_sweeps': sweeps,
        'complete_pss_sweeps': complete, 'partial_pss_sweeps': partial,
        'peak_eligible_pss_sweeps': eligible, 'observed_processes': len(processes) + len(error_only)}
    if coverage != actual_coverage or peak != row['peak_complete_pss_bytes']:
        raise ContractError('observer raw coverage/PSS peak differs')
    for value in processes.values():
        scan.charge(projection_estimate(), temporary=True)
        old_process = _owned_bytes(value)
        before, after = value['cpu_first'], value.get('cpu_last')
        ticks = None if before is None or after is None else after - before
        if ticks is not None: integer(ticks, 0)
        value['sampled_cpu_ticks'] = ticks
        value['sampled_cpu_seconds'] = None if ticks is None else ticks / row['clock']['hz']
        projection_bytes += _owned_bytes(value) - old_process
    result = {'command': command, 'clock': row['clock'], 'wall_seconds': command['wall_ns'] / 1e9,
        'command_wait': row['command_wait'], 'sampling': row['sampling'], 'observer_cost': row['observer'],
        'coverage': coverage, 'peak_complete_pss_bytes': peak, 'processes': list(processes.values()),
        'sample_counts': sample_counts, 'sample_errors': sample_errors,
        'limitations': row['limitations']}
    already_owned: set[int] = set()
    _owned_bytes(row, already_owned)  # Summary decode already admitted by scan.object.
    new_bytes = _owned_bytes(result, already_owned)
    scan.charge(new_bytes + _index_bytes(sample_seen) + _index_bytes(already_owned)
        + _owned_bytes(error_only), temporary=True)
    scan.charge(new_bytes)
    return result


def _configuration(path: Path, lane: str, *, run_id: str | None, learning_seed: int | None) -> Any:
    if lane.endswith('-hpo'):
        from gf_cli.hpo_config import load_hpo_config, materialize_hpo_run
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_training.hpo_config import effective_settings
        if run_id is None or learning_seed is None:
            raise ContractError('HPO requires original declared run UUID/learning seed')
        campaign = load_hpo_config(path)
        if campaign.config.schema_version != (6 if lane == 'native-hpo' else 5):
            raise ContractError('experiment requires shared HPO5 or native HPO6 CONFIG')
        selected = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
        loaded = materialize_hpo_run(campaign, selected, learning_seed=learning_seed, run_id=run_id)
    else:
        from gf_cli.config import load_config
        loaded = load_config(path)
    cfg = loaded.config
    native = hasattr(cfg.run, 'player_runtime') or getattr(cfg, 'hpo_spec', {}).get('schema_version') == 6
    if (native != lane.startswith('native-') or cfg.model.model_id != 'mlp'
            or not lane.endswith('-hpo') and cfg.model.version != '3'):
        raise ContractError('explicit lane differs from original MLP CONFIG')
    return loaded


def _bounds(loaded: Any) -> tuple[int, float]:
    cfg = loaded.config
    limit = cfg.preprocessing.memory_budget_mib * 1024**2
    spec = getattr(cfg, 'hpo_spec', {})
    if spec:
        limit = min(limit, spec['operating']['source_budget_bytes'])
    seconds = _number(cfg.run.wall_seconds)
    if not seconds:
        raise ContractError('finite positive original audit wall required')
    return limit, seconds


def _native_chain(parent: Path, loaded: Any, scan: Any, state: Json, state_sha: str,
                  observer: Json) -> tuple[list[Json], list[Json]]:
    from gf_common.rust_player import (
        validate_exit,
        validate_local_admission,
        validate_operation,
        validate_prepared,
    )
    from gf_common.rust_player_clock import operation_clock
    from gf_training.rust_player_completion import _audit_saved_data
    from gf_training.rust_player_lineage import (
        validate_completion,
        validate_record_shape,
        validate_run,
    )
    from gf_training.rust_player_retention import _validate_historical_audit
    hpo = bool(getattr(loaded.config, 'hpo_spec', {}))
    validate_run(state, hpo=hpo)
    records = []
    previous = None
    for generation in range(state['updates'] + 1):
        row = scan.object(parent / 'generations' / f'{generation:06}.json')
        validate_record_shape(row, 'generation', hpo=hpo)
        sha = scan.verified_digest(parent / 'generations' / f'{generation:06}.json')
        if row['generation'] != generation or row['parent_sha256'] != previous or row['run_id'] != state['run_id']:
            raise ContractError('native committed generation chain differs')
        records.append(row); previous = sha
    if previous != state['generation_sha256'] or scan.object(parent / 'latest.json') != {'generation': state['updates'], 'sha256': previous}:
        raise ContractError('native current generation differs')
    receipts = []
    current = None
    for invocation, entry in state['invocations'].items():
        root = Path(entry['operation_path']).parent
        original = scan.object(parent / f'state-{invocation}.json')
        original_sha = scan.verified_digest(parent / f'state-{invocation}.json')
        validate_run(original, hpo=hpo)
        operation = scan.object(Path(entry['operation_path']), entry['operation_sha256'])
        validate_operation(operation)
        if operation['schema_version'] != 2:
            raise ContractError('actual operation2 required for configured experiment')
        admission = scan.object(Path(entry['admission_path']), entry['admission_sha256'])
        validate_local_admission(admission)
        prepared_path, exit_path = root / f'prepared-{invocation}.json', root / f'exit-{invocation}.json'
        prepared, exit_row = scan.object(prepared_path), scan.object(exit_path)
        prepared_sha, exit_sha = scan.verified_digest(prepared_path), scan.verified_digest(exit_path)
        validate_prepared(prepared, admission); validate_exit(exit_row, admission)
        close = scan.object(Path(prepared['close_ack_path']), prepared['close_ack_sha256'])
        if (close['kind'] != 'rust-player-host-close-ack-v1'
                or close['admission_sha256'] != entry['admission_sha256']
                or close['invocation_id'] != invocation):
            raise ContractError('native original close acknowledgement differs')
        for name, key in (('native', 'native_retirement_sha256'), ('writers', 'writer_retirement_sha256'),
                ('service', 'service_retirement_sha256'), ('cache', 'cache_retirement_sha256')):
            scan.object(parent / 'retirement' / f'{name}-{invocation}.json', close[key])
        scan.object(Path(close['servers_path']), close['servers_sha256'])
        completion = scan.object(root / f'completion-{invocation}.json')
        validate_completion(completion)
        for key in ('run_id', 'config_sha256', 'runtime_binding_sha256', 'operation_id'):
            if operation[key] != admission[key] or completion[key] != admission[key]:
                raise ContractError('native operation/admission/completion binding differs')
        for key in ('supervision_id', 'invocation_id'):
            if completion[key] != admission[key]:
                raise ContractError('native original invocation/supervision differs')
        expected_eligible = original['status'] == 'complete' and (not hpo or original['used_decisions'] == loaded.config.hpo_spec['decision_budget'])
        if (operation['parent_directory'] != str(parent) or operation['run_id'] != state['run_id']
                or operation['config_sha256'] != loaded.sha256 or operation['deadlines'] != state['monotonic_authority']['deadlines']
                or operation['execution_request'] != entry['execution_request']
                or admission['deadlines'] != operation['deadlines']
                or admission['context'] != operation['expected_context']
                or admission['invocation_id'] != invocation
                or original['invocations'].get(invocation) != entry
                or admission['producer'] != state['identity']['player_runtime']['producer']
                or prepared['state_sha256'] != original_sha or prepared['state'] != original['status']
                or prepared['generation_sha256'] != original['generation_sha256']
                or completion['state_sha256'] != original_sha or completion['state'] != original['status']
                or completion['generation_sha256'] != original['generation_sha256']
                or completion['prepared_sha256'] != prepared_sha or completion['exit_sha256'] != exit_sha
                or completion['operation_sha256'] != entry['operation_sha256']
                or completion['committed_update'] != original['updates'] or completion['eligible'] != expected_eligible
                or exit_row['prepared_sha256'] != prepared_sha or not exit_row['cooperative']
                or exit_row['wait']['exit_code'] != 0 or not exit_row['descendants_reaped'] or exit_row['signals']):
            raise ContractError('native immutable Prepared/Exit/completion chain differs')
        clock = scan.object(root / f'clock-{operation["operation_id"]}.json')
        inputs = {key: getattr(loaded.config.run, key) for key in
            ('admission_seconds', 'execution_seconds', 'wall_seconds', 'finalization_seconds')}
        if (clock['inputs'] != inputs or clock != operation_clock(operation, clock_binding=clock['clock_binding'],
                inputs=clock['inputs'], post_host_seconds=loaded.config.run.post_host_seconds)
                or clock['clock_binding']['boot_id'] != observer['clock']['boot_id']
                or clock['clock_binding']['boot_id'] != state['monotonic_authority']['boot_id']):
            raise ContractError('native original operation clock differs')
        if scan.object(Path(operation['config_path']), operation['config_sha256']) != loaded.effective:
            raise ContractError('native original effective snapshot differs')
        if scan.object(root / f'inputs-{operation["operation_id"]}.json') != {'inputs': list(loaded.inputs)}:
            raise ContractError('native actual original configuration inputs differ')
        expected_audit = root / f'cold-audit-{invocation}.json'
        if completion['audit_path'] != str(expected_audit):
            raise ContractError('native cold audit path differs')
        audit = scan.object(expected_audit, completion['audit_sha256'])
        _validate_historical_audit(audit, state=original, prepared=prepared, completion=completion)
        saved_generation = records[original['updates']]
        if any(audit['saved_pair'][key] != saved_generation[key] for key in
                ('inference_sha256', 'resume_sha256', 'artifact_files', 'optimizer_steps')):
            raise ContractError('cold saved pair differs from actual committed generation')
        for fact in audit['updates']:
            update = scan.object(parent / 'updates' / f'{fact["generation"]:06}.json', records[fact['generation']]['update_sha256'])
            validate_record_shape(update, 'update', hpo=hpo)
            if (update['collection_manifest_sha256'] != fact['manifest_sha256']
                    or update['numeric_sha256'] != fact['numeric_sha256']
                    or update['accepted_decisions'] != fact['learner_decisions']
                    or update['source_proof']['source_set_sha256'] != fact['source_set_sha256']
                    or update['source_proof']['episode_digests'] != fact['episode_digests']):
                raise ContractError('native actual cold update receipt differs')
            numeric = Path(update['numeric'])
            if numeric.is_absolute() or '..' in numeric.parts:
                raise ContractError('native numeric data leaves original parent')
            from gf_models.structured_sealed import basename
            for name, sha in update['numeric_files'].items():
                with scan.file(parent / numeric / basename(name), sha):
                    pass
        receipt = {'invocation_id': invocation, 'operation_id': operation['operation_id'],
            'operation_sha256': entry['operation_sha256'], 'admission_sha256': entry['admission_sha256'],
            'prepared_sha256': prepared_sha, 'exit_sha256': exit_sha,
            'completion_sha256': scan.verified_digest(root / f'completion-{invocation}.json'),
            'audit_sha256': completion['audit_sha256'], 'eligible': completion['eligible'],
            'current': original_sha == state_sha, 'mode': entry['mode'],
            'committed_update': original['updates'], 'invocation_history': sorted(original['invocations']),
            'clock_sha256': scan.verified_digest(root / f'clock-{operation["operation_id"]}.json')}
        receipts.append(receipt)
        if original_sha == state_sha:
            if current is not None:
                raise ContractError('ambiguous current native immutable state')
            expected_supervisor = {'pid': observer['command']['pid'], 'start_ticks': observer['command']['start_ticks']}
            sampled = {(row['pid'], row['start_ticks']) for row in observer['processes']}
            if (admission['supervisor'] != expected_supervisor
                    or observer['sampling']['enabled'] and (admission['host']['pid'], admission['host']['start_ticks']) not in sampled):
                raise ContractError('current native original supervisor/host differs from observed process identities')
            current = original
    if current is None:
        raise ContractError('actual current cold state is absent')
    artifact = _audit_saved_data(parent, loaded, state, records[-1], scan)
    del artifact
    return records, receipts


def _legacy_hpo_binding(parent: Path, loaded: Any, *, config_path: Path,
                        stdout: Json, observer: Json, scan: Any) -> Json:
    """Original DATA readers only: no global lock, live job or admission issued."""
    from gf_cli.hpo_config import load_hpo_config
    from gf_common.rust_player import hex_identity, validate_context
    from gf_models.structured_sealed import checksum
    from gf_training.hpo_gpu_lease import _GLOBAL_ROOT, _admission, _termination
    context = validate_context(stdout['operation_context'])
    hex_identity(context['operation_id'], 32)
    if context['kind'] != 'training' or context['run_id'] != parent.name or context['evaluation_id'] is not None:
        raise ContractError('legacy HPO actual training context differs')
    argv = observer['command']['argv']
    def argument(name: str, *, optional: bool = False) -> str | None:
        if optional and name not in argv:
            return None
        if argv.count(name) != 1 or argv.index(name) + 1 == len(argv):
            raise ContractError('legacy HPO original invocation argument absent or ambiguous')
        return argv[argv.index(name) + 1]
    if argument('--experiment-id') != context['campaign_id']:
        raise ContractError('legacy HPO observed experiment differs')
    namespace = Path(loaded.config.run.output_dir) / '.operation-control' / context['campaign_id'] / context['run_id']
    invocation = scan.object(namespace / 'invocations' / (context['operation_id'] + '.json'))
    closed(invocation, {'schema_version', 'kind', 'context', 'campaign_config_sha256', 'config_sha256',
        'config_inputs', 'mode', 'pause_after_update', 'launcher_path', 'launcher_sha256', 'python_executable'})
    campaign = load_hpo_config(config_path)
    pause = argument('--pause-after-update', optional=True)
    if (type(invocation['schema_version']) is not int or invocation['schema_version'] != 1
            or invocation['kind'] != 'hpo-operation-invocation-data-v1' or invocation['context'] != context
            or invocation['config_sha256'] != loaded.sha256 or invocation['campaign_config_sha256'] != campaign.sha256
            or canonical(invocation['config_inputs']) != canonical(list(campaign.inputs))
            or invocation['mode'] not in ('fresh', 'resume') or argument('--mode') != invocation['mode']
            or pause != (None if invocation['pause_after_update'] is None else str(invocation['pause_after_update']))
            or Path(argv[0]).absolute() != Path(invocation['python_executable'])
            or len(argv) < 2 or Path(argv[1]).absolute() != Path(invocation['launcher_path'])):
        raise ContractError('legacy HPO original launch invocation differs')
    checksum(invocation['launcher_sha256'])  # Historical producer, not current source equality.
    physical = stdout['gpu_termination']
    ticket = hex_identity(physical['ticket'], 32)
    original = _GLOBAL_ROOT / 'jobs' / ticket
    admission = scan.object(original / 'admission.json')
    _admission(admission, ticket)
    termination = scan.object(original / 'termination.json')
    _termination(termination, admission)
    if (admission['schema_version'] != 1 or admission['context'] != context
            or admission['boot_id'] != observer['clock']['boot_id']
            or admission['owner_pid'] != observer['command']['pid'] or termination != physical):
        raise ContractError('legacy HPO actual original physical owner/context differs from observed root')
    return {'pid': admission['owner_pid'], 'boot_id': admission['boot_id'], 'ticket': ticket,
        'context': context, 'invocation_sha256': scan.verified_digest(namespace / 'invocations' / (context['operation_id'] + '.json')),
        'admission_sha256': scan.verified_digest(original / 'admission.json'),
        'termination_sha256': scan.verified_digest(original / 'termination.json')}


def _native_measurement_scope(receipts: list[Json]) -> Json:
    """Verified immutable states delimit updates; file presence never selects them."""
    history: set[str] = set()
    previous_end = 0
    selected = None
    for ordinal, receipt in enumerate(sorted(receipts, key=lambda row: len(row['invocation_history']))):
        invocation = receipt['invocation_id']
        if (invocation in history or set(receipt['invocation_history']) != history | {invocation}
                or receipt['mode'] != ('fresh' if ordinal == 0 else 'resume')):
            raise ContractError('native frozen invocation history is not one original chain')
        end = integer(receipt['committed_update'], previous_end)
        if receipt['current']:
            if selected is not None or ordinal != len(receipts) - 1:
                raise ContractError('observed native current state is not the final original invocation')
            selected = {'invocation_id': invocation, 'operation_id': receipt['operation_id'],
                'start_generation': previous_end, 'end_generation': end,
                'evidence': 'verified native frozen-state/completion chain'}
        history.add(invocation)
        previous_end = end
    if selected is None:
        raise ContractError('observed native current invocation interval absent')
    return selected


def _legacy_measurement_scope(state: Json, *, binding: Json | None) -> Json | None:
    invocations = state.get('invocations', [])
    if not invocations or 'shared_service' not in invocations[-1]:
        return None  # No durable interval: historical data cannot qualify current timing.
    current = invocations[-1]
    stamp = current['shared_service']
    if stamp['finished'] is not True or stamp['end_generation'] != state['updates']:
        raise ContractError('legacy shared current invocation lacks finalized committed interval')
    start = integer(stamp['start_generation'], 0, state['updates'])
    if binding is not None and stamp['job_id'] != binding['ticket']:
        raise ContractError('legacy shared current invocation differs from observed original job')
    return {'invocation_id': stamp['execution'], 'operation_id': None,
        'start_generation': start, 'end_generation': state['updates'],
        'evidence': 'verified legacy finalized shared invocation stamp'}


def _measurement_updates(audit: Json) -> list[Json]:
    scope = audit.get('measurement_scope')
    if scope is None:
        return []
    return [row for row in audit['updates']
        if scope['start_generation'] < row['generation'] <= scope['end_generation']]


def _legacy_packed_diagnostics(scan: Any, parent: Path, state: Json, update: Json, *, lane: str) -> Json:
    """Project bounded DATA from the SAME full-validator scan's actual originals."""
    from gf_common.group_pack_game import packed_state
    from gf_training.structured_rollout import _owned_bytes
    proof = update['source_proof']
    if (lane not in ('legacy-hpo', 'legacy-ordinary') or not packed_state(state)
            or lane == 'legacy-hpo' and (state['run_variant'] != 'hpo-ppo-owned-shared-v1'
                or type(proof['schema_version']) is not int or proof['schema_version'] != 6)):
        raise ContractError('diagnostic adapter requires validated original packed legacy lane')
    collection = parent / update['collection']
    owner = scan.packed_owner(collection)
    manifest = scan.object(collection / 'manifest.json', update['collection_manifest_sha256'])
    storage = owner.storage
    descriptor = owner.artifacts._originals.descriptor_file
    assert descriptor is not None
    storage = {**storage, 'descriptor_path': str(descriptor.path)}
    closures: dict[str, int] = {}
    journals: Json = {}
    for episode in manifest['episodes']:
        closure = episode['end_kind']
        closures[closure] = closures.get(closure, 0) + 1
        for viewer in (1, 2):
            name = f'{episode["episode_id"]}/player-{viewer}.jsonl'
            if name in journals:
                raise ContractError('duplicate authenticated legacy episode journal')
            journals[name] = {'run_id': manifest['run_id'], 'episode_id': episode['episode_id'],
                'game_id': episode['game_id'], 'viewer': viewer}
    if len(journals) != 2 * update['actual_games']:
        raise ContractError('authenticated legacy manifest episode count differs')
    projected = {'source_proof': {'storage': storage, 'closure_counts': closures,
        # Shared HPO proof6 contains no all-seat counter. The exact complete
        # named journals supply it later; learner-only update counts never do.
        'accepted_decisions': proof.get('accepted_decisions') if lane == 'legacy-ordinary' else None},
        'journals': journals}
    scan.charge(_owned_bytes(projected))
    return projected


def _audit_parent(parent: Path, loaded: Any, *, lane: str, observer_directory: Path,
                  deadline: float, limit: int, config_path: Path, ownership: Any) -> tuple[Json, Json]:
    from gf_models.structured_sealed import MAX_SEALED_SCAN_BYTES
    from gf_training.lineage import require_hpo_run_owner
    from gf_training.structured_lineage import _ChainScan
    from gf_training.structured_rollout import _owned_bytes
    require_hpo_run_owner(parent, ownership)
    with _ChainScan(directory=parent, max_owned_bytes=min(limit, MAX_SEALED_SCAN_BYTES), deadline=deadline) as scan:
        scan.charge(_owned_bytes(loaded))
        observer = observer_data(observer_directory, scan=scan)
        check_observed_config(config_path, argv=observer['command']['argv'])
        state = scan.object(parent / 'run.json')
        state_sha = scan.verified_digest(parent / 'run.json')
        if state['config_sha256'] != loaded.sha256 or canonical(state['effective_config']) != canonical(loaded.effective):
            raise ContractError('original configured effective state differs')
        if not lane.startswith('native-') and canonical(state['config_inputs']) != canonical(list(loaded.inputs)):
            raise ContractError('legacy actual original configuration inputs differ')
        if state['status'] not in ('complete', 'incomplete', 'paused'):
            raise ContractError('unhealthy or unfinished parent cannot qualify measurement')
        if lane.startswith('native-'):
            records, receipts = _native_chain(parent, loaded, scan, state, state_sha, observer)
        elif lane == 'legacy-hpo':
            from gf_training.hpo_source import validate_hpo_parent
            state, records = validate_hpo_parent(parent, require_complete=False, max_owned_bytes=limit,
                deadline=deadline, _scan=scan)
            receipts = []
        else:
            from gf_training.card_instance_source import validate_game_parent
            state, records = validate_game_parent(parent, require_complete=False, max_owned_bytes=limit,
                deadline=deadline, _scan=scan)
            # Its existing _validate_component_chain calls _inspect_pair for
            # every retained generation; the current pair is mandatory retained.
            # That is the full v3 sealed decoder, including Adam/RNG. Repeating
            # it here would duplicate the same whole-pair parse and allowance.
            receipts = []
        stdout: list[Json] = []
        for output_row in _rows(observer_directory / 'command.stdout.log', scan):
            if stdout:
                raise ContractError('exactly one original command result required')
            scan.charge(_owned_bytes(output_row))
            stdout.append(output_row)
        if (len(stdout) != 1 or stdout[0].get('run_dir') != str(parent)
                or stdout[0].get('status') != state['status'] or stdout[0].get('updates') != state['updates']
                or stdout[0].get('generation_sha256') != state['generation_sha256']):
            raise ContractError('actual observed command result differs from owned parent')
        legacy_binding = None
        if lane == 'legacy-hpo':
            legacy_binding = _legacy_hpo_binding(parent, loaded, config_path=config_path,
                stdout=stdout[0], observer=observer, scan=scan)
        if lane.startswith('native-'):
            current_receipt = next(row for row in receipts if row['current'])
            output = stdout[0].get('rust_player', {})
            if any(output.get(key) != current_receipt[key] for key in ('completion_sha256', 'audit_sha256', 'eligible')) or output.get('operation_clock_sha256') != current_receipt['clock_sha256']:
                raise ContractError('observed native completion/clock/audit differs from original receipts')
        measurement_scope = (_native_measurement_scope(receipts) if lane.startswith('native-')
            else _legacy_measurement_scope(state, binding=legacy_binding))
        updates = []
        for generation, record in enumerate(records[1:], 1):
            update = scan.object(parent / 'updates' / f'{generation:06}.json', record['update_sha256'])
            proof = update['source_proof']
            if lane.startswith('native-'):
                diagnostic = {'source_proof': {'storage': proof['storage'], 'closure_counts': proof['closure_counts'],
                    'accepted_decisions': proof['accepted_decisions']}}
            else:
                diagnostic = _legacy_packed_diagnostics(scan, parent, state, update, lane=lane)
            updates.append({key: update[key] for key in ('batch_id', 'actual_games', 'planned_games',
                'collection_seconds', 'update_seconds')} | diagnostic
                | {'manifest_sha256': update['collection_manifest_sha256']})
            usage = proof.get('service_usage', {})
            updates[-1]['invocation_id'] = usage.get('invocation_id', usage.get('execution'))
            updates[-1]['generation'] = generation
            updates[-1]['measurement_current'] = (measurement_scope is not None
                and measurement_scope['start_generation'] < generation <= measurement_scope['end_generation'])
            if measurement_scope is not None and updates[-1]['measurement_current'] and updates[-1]['invocation_id'] != measurement_scope['invocation_id']:
                raise ContractError('current update differs from verified original invocation interval')
        result = {'measurement_scope': measurement_scope, 'lane': lane, 'run_id': state['run_id'], 'status': state['status'], 'updates': updates,
            'generation_sha256': state['generation_sha256'], 'state_sha256': state_sha,
            'optimizer_steps': state['optimizer_steps'], 'receipts': receipts, 'legacy_observer_binding': legacy_binding,
            'config_sha256': state['config_sha256'], 'audit_peak_estimated_bytes': scan.peak,
            'recorded_code': state['identity']['code'],
            'recorded_simulator_revision': loaded.effective['actions']['action_contract']['simulator_revision'],
            'recorded_player_producer': state['identity'].get('player_runtime', {}).get('producer'),
            'saved_pair_full_content_validated': True,
            'original_replay_numeric_audit': 'existing legacy full audit' if not lane.startswith('native-') else 'hash-checked actual in-command Cold audit receipts'}
        scan.finalize()
        return result, observer


def _project_collections(audit: Json, *, limit: int, deadline: float, retained: Any,
                         native: bool) -> list[Json]:
    from gf_training.group_pack_source import PackedSource
    from gf_training.structured_rollout import _owned_bytes
    results: list[Json] = []
    for ordinal, update in enumerate(_measurement_updates(audit), 1):
        storage = update['source_proof']['storage']
        path = Path(storage['descriptor_path'])
        remaining = limit - _owned_bytes((retained, results))
        source = PackedSource(path, deadline=deadline, max_owned_bytes=remaining, _allow_pending=native)
        projected = []
        try:
            if source._originals.descriptor_file.sha256 != storage['descriptor_sha256']:
                raise ContractError('actual committed packed descriptor differs')
            if source._originals.database_file.sha256 != storage['database_sha256']:
                raise ContractError('actual committed packed database differs')
            if source._controls['manifest.json']['sha256'] != update['manifest_sha256']:
                raise ContractError('packed original is not the committed collection')
            journals = update.get('journals')
            if journals is not None:
                count = 0
                for member in source._index['members']:
                    name = member['name']
                    if member['kind'] == 'journal':
                        count += 1
                        if name not in journals:
                            raise ContractError('extra or foreign legacy participant journal')
                    elif name in journals:
                        raise ContractError('expected legacy participant path is not a typed journal')
                if count != len(journals):
                    raise ContractError('missing legacy participant journal')
            for group_id in source._groups:
                with source.borrow_group(group_id) as loan:
                    for name in loan.names():
                        if journals is None:
                            if not name.endswith('.jsonl'):
                                continue
                        elif name not in journals:
                            continue
                        raw = loan.read(name)
                        workspace = 32 * len(raw) + 65536
                        source._budget.acquire(workspace, deadline)
                        try:
                            row = (journal_data(io.BytesIO(raw)) if journals is None else
                                journal_data(io.BytesIO(raw), expected_identity=journals[name]))
                            row['journal'] = name
                            charge = _owned_bytes(row)
                            source._budget.acquire(charge, deadline)
                            projected.append(row)
                            del row, raw
                        finally:
                            source._budget.release(workspace)
                source.evict_group()  # All loan payloads have reached last use.
                del loan
            source.recheck()
            expected_accepted = update['source_proof']['accepted_decisions']
            if (journals is None or expected_accepted is not None) and sum(row['accepted_decisions'] for row in projected) != expected_accepted:
                raise ContractError('all-seat acknowledged decisions differ from verified Source')
            result = collection_data(update, projected, ordinal=ordinal)
        finally:
            source.close()
            # Source originals/groups are positively closed before these DATA
            # graphs transfer to the next interval's retained-byte subtraction.
            del source
        results.append(result)
    return results


def _legacy_broker_pretransition(row: Json, sequence: list[Json], *, lane: str,
                                 current_invocation: str | None) -> bool:
    """Exact DIAGNOSTIC_BEGIN crossing of adjacent ORIGINAL host phases only."""
    if (lane not in ('legacy-hpo', 'legacy-ordinary') or row['owner']['role'] != 'legacy-shared-broker'
            or current_invocation is None or row['scope']['invocation_id'] != current_invocation):
        return False
    for index in range(1, len(sequence)):
        earlier, later = sequence[index-1], sequence[index]
        if (later['scope'] != row['scope'] or earlier['scope']['invocation_id'] not in (None, current_invocation)
                or any(earlier['scope'][key] != row['scope'][key]
                    for key in ('run_id', 'batch_id', 'phase', 'operation_id'))):
            continue
        if (earlier['start_monotonic_ns'] <= row['start_monotonic_ns'] <= earlier['end_monotonic_ns']
                <= later['start_monotonic_ns'] <= row['end_monotonic_ns'] <= later['end_monotonic_ns']):
            return True
    return False


def _telemetry(directory: Path | None, *, observer: Json, audit: Json, observation_id: str,
               scan: Any) -> Json:
    if directory is None:
        if audit.get('lane') == 'legacy-ordinary':
            raise ContractError('ordinary legacy diagnostics required for original coordinator binding')
        return {'enabled': False, 'metrics': None, 'phases': None, 'profiles': None,
            'complete': None, 'limitations': ['application diagnostics disabled; common persisted collection rate remains available']}
    interval = observer['command']['start_monotonic_ns'], observer['command']['finish_monotonic_ns']
    options = {'observation_id': observation_id, 'boot_id': observer['clock']['boot_id'], 'run_id': audit['run_id']}
    startup_owner = None
    if audit.get('lane', '').startswith('legacy-'):
        startup_owner = observer['command']['pid'], observer['command']['start_ticks']
    options['startup_owner'] = startup_owner
    identities = {(row['pid'], row['start_ticks']) for row in observer['processes']}
    measurement_scope = audit.get('measurement_scope')
    operations = ({measurement_scope['operation_id']} if measurement_scope is not None
        and measurement_scope['operation_id'] is not None else set())
    metrics, events, profiles = [], [], []
    trace_paths = set()
    trace_files = {path for path in directory.glob('*.trace.json')}
    current_updates = _measurement_updates(audit)
    batches = {row['batch_id']: row for row in (current_updates if measurement_scope is not None else audit['updates'])}
    for path in sorted(directory.glob('*.json')):
        if path in trace_files:
            continue  # Profiler payload has its own recorded SHA, not metrics1.
        row = scan.object(path)
        kind = row.get('kind')
        if kind == 'rust-player-phase-metrics-v1':
            value = metric_data(row, interval=interval, **options)
            metrics.append(value)
        elif kind == 'rust-player-phase-event-v1':
            closed(row, {'schema_version', 'kind', 'observation_id', 'owner', 'clock', 'scope', 'event', 'monotonic_ns', 'links'})
            _pairing(row, **options)
            integer(row['monotonic_ns'], interval[0], interval[1])
            closed(row['links'], {'operation_sha256', 'prepared_sha256', 'exit_sha256', 'completion_sha256'})
            if operations:
                for key, sha in row['links'].items():
                    if sha is not None and not any(receipt['operation_id'] == row['scope']['operation_id']
                            and receipt['current'] and receipt[key] == sha for receipt in audit['receipts']):
                        raise ContractError('phase SHA link differs from actual original receipt')
            if row['event'] not in ('phase_start', 'phase_end', 'host_reaped', 'prepared_written', 'completion_written'):
                raise ContractError('unknown phase1 event')
            events.append(row)
        elif kind == 'rust-player-forward-profile-v1':
            closed(row, {'schema_version', 'kind', 'observation_id', 'owner', 'clock', 'scope', 'recipe',
                'enabled', 'boundary', 'worker_tid', 'start_after_forward', 'stop_after_forward',
                'callbacks_seen', 'scored_rows', 'start_monotonic_ns', 'end_monotonic_ns', 'complete',
                'errors', 'trace_path', 'trace_sha256', 'trace_bytes', 'trace_clock', 'torch_version'})
            _pairing(row, **options)
            if row['recipe'] != 'forward-window-v1' or row['start_after_forward'] != 1 or row['stop_after_forward'] != 3:
                raise ContractError('unknown fixed profiler window')
            if row['enabled'] is not True or type(row['complete']) is not bool:
                raise ContractError('explicit enabled profile1 status required')
            if row['complete'] and (row['callbacks_seen'] != 3 or len(row['scored_rows']) != 3
                    or any(type(value) is not int or value < 1 for value in row['scored_rows'])):
                raise ContractError('completed profile requires actual three callback row counts')
            if row['trace_path'] is not None:
                trace = Path(row['trace_path'])
                trace = trace if trace.is_absolute() else directory / trace
                if '..' in trace.parts or not trace.is_relative_to(directory):
                    raise ContractError('profile trace leaves actual diagnostics directory')
                with scan.file(trace, row['trace_sha256']) as (_source, info):
                    if info.st_size != row['trace_bytes']:
                        raise ContractError('profile trace size differs')
                trace_paths.add(trace)
            profiles.append({'record': row, 'gpu_event_metrics': None,
                'limitation': 'exact trace event parser not pinned; host sums are not GPU event durations'})
        else:
            raise ContractError('unsupported diagnostics record; no guessed interpretation')
        if (row['owner']['pid'], row['owner']['start_ticks']) not in identities:
            raise ContractError('diagnostic owner absent from sampled process identities')
        if operations and row['scope']['operation_id'] not in operations:
            raise ContractError('diagnostic operation differs from actual original receipts')
        if row['scope']['phase'] == 'collection':
            batch = batches.get(row['scope']['batch_id'])
            transitional_host = (row['kind'] == 'rust-player-phase-event-v1' and row['owner']['role'] == 'host'
                and row['scope']['invocation_id'] is None and startup_owner is not None)
            if batch is None or (batch['invocation_id'] is not None
                    and row['scope']['invocation_id'] != batch['invocation_id'] and not transitional_host):
                raise ContractError('diagnostic collection differs from actual committed source scope')
    legacy_binding = None
    if startup_owner is not None:
        bound = [row for row in events if row['owner']['role'] == 'host'
            and (row['owner']['pid'], row['owner']['start_ticks']) == startup_owner
            and row['scope']['run_id'] == audit['run_id']
            and (measurement_scope is None or row['scope']['invocation_id'] == measurement_scope['invocation_id'])]
        if not bound:
            raise ContractError('legacy actual original coordinator/run phase binding absent')
        assigned_at = min(row['monotonic_ns'] for row in bound)
        if any(row['scope']['run_id'] is None and row['monotonic_ns'] > assigned_at for row in events):
            raise ContractError('unassigned legacy startup follows known run phase')
        legacy_binding = {'pid': startup_owner[0], 'start_ticks': startup_owner[1],
            'boot_id': observer['clock']['boot_id'], 'run_id': audit['run_id']}
    phases = []
    pending: dict[str, Json] = {}
    errors = []
    if measurement_scope is None:
        errors.append('current committed invocation interval unavailable')
    if trace_files - trace_paths:
        errors.append('unreferenced profiler trace files')
    for row in sorted(events, key=lambda item: item['monotonic_ns']):
        key = canonical({'owner': row['owner'], 'scope': row['scope']})
        if row['event'] == 'phase_start':
            if key in pending:
                raise ContractError('overlapping same-owner/scope phase starts')
            pending[key] = row
        elif row['event'] == 'phase_end':
            start = pending.pop(key, None)
            if start is None:
                errors.append('phase end without original same-owner/scope start')
                continue
            phases.append({'owner': row['owner'], 'scope': row['scope'],
                'start_monotonic_ns': start['monotonic_ns'], 'end_monotonic_ns': row['monotonic_ns'],
                'seconds': (row['monotonic_ns'] - start['monotonic_ns']) / 1e9})
    if pending: errors.append('phase start without original same-owner/scope end')
    def original_host(row: Json) -> bool:
        return (startup_owner is not None and row['owner']['role'] == 'host'
            and (row['owner']['pid'], row['owner']['start_ticks']) == startup_owner)
    # Include ALL this original host's closed phases, not a collection-only
    # subset that could conceal an intervening phase or an unclosed boundary.
    host_sequence = sorted((row for row in phases if original_host(row)),
        key=lambda row: (row['start_monotonic_ns'], row['end_monotonic_ns']))
    if (sum(original_host(row) and row['event'] in ('phase_start', 'phase_end') for row in events) != 2 * len(host_sequence)
            or any(host_sequence[index-1]['end_monotonic_ns'] > host_sequence[index]['start_monotonic_ns']
                for index in range(1, len(host_sequence)))):
        host_sequence = []
    collection_coverage = []
    for batch in current_updates:
        selected = [row for row in metrics if row['scope']['phase'] == 'collection'
            and row['scope']['batch_id'] == batch['batch_id']]
        host_phases = [row for row in phases if row['scope']['phase'] == 'collection'
            and row['scope']['batch_id'] == batch['batch_id']
            and row['owner']['role'] in ('host', 'legacy-coordinator')
            and row['scope']['invocation_id'] == batch['invocation_id']]
        if not selected or not host_phases:
            errors.append('missing collection diagnostics:' + batch['batch_id'])
        for row in selected:
            if (host_phases and not any(phase['start_monotonic_ns'] <= row['start_monotonic_ns']
                    <= row['end_monotonic_ns'] <= phase['end_monotonic_ns'] for phase in host_phases)
                    and not _legacy_broker_pretransition(row, host_sequence, lane=audit.get('lane', ''),
                        current_invocation=measurement_scope['invocation_id'] if measurement_scope is not None else None)):
                errors.append('collection metric interval outside host phase:' + batch['batch_id'])
        native = audit.get('lane', 'native').startswith('native')
        runtime_role = 'native-runtime' if native else 'legacy-coordinator'
        runtime = [row for row in selected if row['owner']['role'] == runtime_role and row['complete']]
        broker = [row for row in selected if row['owner']['role'] == ('native-runtime' if native else 'legacy-shared-broker')
            and row['complete']]
        mandatory = {
            'event_page': any(row['histograms']['event_page'] is not None
                and all(row['counters'][key] is not None for key in
                    ('event_page_successes', 'event_page_failures', 'event_transport_attempts')) for row in runtime),
            'inference_roundtrip': any(row['histograms']['inference_roundtrip'] is not None
                and row['counters']['inference_successes'] is not None for row in runtime),
            'actual_batch': any(row['batch_histogram'] is not None for row in broker),
            'pyo3': any(row['pyo3_calls'] is not None for row in runtime) if native else None}
        if any(value is False for value in mandatory.values()):
            errors.append('missing mandatory collection metric families:' + batch['batch_id'])
        collection_coverage.append({'batch_id': batch['batch_id'], 'mandatory_metrics': mandatory,
            'metrics_present': bool(selected), 'host_phase_seconds': [row['seconds'] for row in host_phases] or None,
            'common_collection_seconds': batch['collection_seconds'],
            'scope': 'common persisted collector wall and wider host phase reported separately'})
    return {'enabled': True, 'metrics': metrics or None, 'phases': phases or None,
        'events': events, 'profiles': profiles or None, 'legacy_observer_binding': legacy_binding,
        'collection_coverage': collection_coverage,
        'complete': bool(metrics) and not errors and all(row['complete'] for row in metrics)
            and all(row['record']['complete'] for row in profiles),
        'errors': errors, 'limitations': ['owner/scope phases remain distinct; host stages are not CUDA event times']}


def verify_operation(*, config_path: Path, parent: Path, observer_directory: Path,
                     observation_id: str, lane: str, telemetry_directory: Path | None = None,
                     run_id: str | None = None, learning_seed: int | None = None) -> Json:
    """Read a finished original; separate external audit clocks never renew training."""
    from gf_training.lineage import run_lock
    if lane not in ('native-ordinary', 'native-hpo', 'legacy-ordinary', 'legacy-hpo'):
        raise ContractError('explicit supported operation lane required')
    started = time.monotonic()
    loaded = _configuration(config_path, lane, run_id=run_id, learning_seed=learning_seed)
    limit, seconds = _bounds(loaded)
    deadline = started + seconds
    parent = parent.absolute()
    with run_lock(parent) as ownership:
        return _verify_owned(config_path=config_path, parent=parent, observer_directory=observer_directory,
            telemetry_directory=telemetry_directory, observation_id=observation_id, lane=lane,
            run_id=run_id, learning_seed=learning_seed, loaded=loaded, limit=limit, deadline=deadline,
            started=started, seconds=seconds, ownership=ownership)


def _verify_owned(*, config_path: Path, parent: Path, observer_directory: Path,
                  telemetry_directory: Path | None, observation_id: str, lane: str,
                  run_id: str | None, learning_seed: int | None, loaded: Any,
                  limit: int, deadline: float, started: float, seconds: float, ownership: Any) -> Json:
    from gf_models.structured_sealed import MAX_SEALED_SCAN_BYTES, SealedScan
    from gf_training.lineage import require_hpo_run_owner
    from gf_training.structured_rollout import _owned_bytes
    audit, observer = _audit_parent(parent, loaded, lane=lane, observer_directory=observer_directory,
        deadline=deadline, limit=limit, config_path=config_path, ownership=ownership)
    if lane.endswith('-hpo'):
        assert run_id is not None
        argv = observer['command']['argv']
        for name, value in (('--run-id', run_id), ('--learning-seed', str(learning_seed))):
            if argv.count(name) != 1 or argv.index(name) + 1 == len(argv) or argv[argv.index(name)+1] != value:
                raise ContractError('original observed HPO materialization identity differs')
        if parent != Path(loaded.config.run.output_dir) / run_id:
            raise ContractError('HPO actual parent differs from declared run UUID')
    # The whole audit scan has positively closed and its state/checkpoint graphs
    # have left scope before this separate reader interval under the SAME cap.
    collections = _project_collections(audit, limit=limit, deadline=deadline,
        retained=(loaded, audit, observer), native=lane.startswith('native-'))
    scan = SealedScan(max_owned_bytes=min(limit, MAX_SEALED_SCAN_BYTES), deadline=deadline)
    scan.charge(_owned_bytes((loaded, audit, observer, collections)))
    telemetry = _telemetry(telemetry_directory, observer=observer, audit=audit,
        observation_id=observation_id, scan=scan)
    scan.finalize()
    require_hpo_run_owner(parent, ownership)
    return {'schema_version': 1, 'kind': 'rust-player-operation-verification-data-v1',
        'verified': True, 'source_authority': False, 'adoption': None,
        'observation_id': observation_id, 'lane': lane, 'config_path': str(config_path.absolute()),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'config_inputs': list(loaded.inputs),
        'audit': audit, 'observer': observer, 'collections': collections,
        'telemetry': telemetry, 'external_audit': {'seconds': time.monotonic()-started,
            'wall_budget_seconds': seconds, 'source_limit_bytes': limit,
            'scope': 'post-command independent DATA read; excluded from original Child.wait wall'},
        'limitations': ['diagnostic verification grants no Source/Parent/objective or adoption',
            'sampling may miss processes/peaks; profile GPU trace metrics require a separately pinned parser']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path)
    parser.add_argument('--observation-id', required=True)
    parser.add_argument('--lane', choices=('native-ordinary', 'native-hpo', 'legacy-ordinary', 'legacy-hpo'), required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--learning-seed', type=int)
    args = parser.parse_args(argv)
    try:
        value = verify_operation(config_path=args.config, parent=args.run_dir, observer_directory=args.observer,
            telemetry_directory=args.telemetry, observation_id=args.observation_id, lane=args.lane,
            run_id=args.run_id, learning_seed=args.learning_seed)
    except (ContractError, OSError, ValueError, TimeoutError, KeyError, TypeError, RuntimeError) as error:
        print(canonical({'verified': False, 'source_authority': False, 'adoption': None,
            'error': type(error).__name__, 'reason': str(error)}))
        return 2
    print(canonical(value))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
