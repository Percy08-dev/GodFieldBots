"""Read-only replay/pair aggregation; concurrent and nested scopes never add to wall."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

from common import REPO, read_json, sha256


def replay_summary(path: Path) -> dict[str, Any]:
    result = read_json(path/'result.json')
    request = read_json(path/'request.json')
    receipt = read_json(path/'receipt.json')
    if receipt['returncode'] or receipt['failure'] or result['profiled']:
        raise ValueError('only successful unprofiled replay belongs to timing population')
    loops = result['loops']
    if not loops:
        raise ValueError('nonempty replay loop population required')
    if result['compiler'].get('failures') or result['compiler'].get('fallbacks', 0):
        raise ValueError('compiler failure/fallback excludes replay timing')
    last_compile = max([0, *[i for i, row in enumerate(loops) if row['compiler_backend_graph_delta'] > 0]])
    steady = loops[last_compile+1:]
    stable = len(steady) >= 2 and all(row['compiler_backend_graph_delta'] == 0 for row in steady)
    per_decision = [row['host_wall_seconds']/row['valid_decisions'] for row in steady]
    return {'path': str(path), 'options': request['options'], 'cache_state': request['cache_state'],
        'input_sha256': request['input_manifest_sha256'], 'device': result['device'], 'code': request['code'],
        'first_call_host_seconds': result['first_call_host_seconds'],
        'installation_host_seconds': result.get('installation_host_seconds'),
        'loops': loops, 'initial_population_loops': last_compile+1, 'steady_loop_count': len(steady),
        'steady_status': 'MEASURED' if stable else 'UNAVAILABLE',
        'warm_host_seconds_per_decision': statistics.median(per_decision) if stable else None,
        'valid_decisions_per_loop': [row['valid_decisions'] for row in loops],
        'valid_candidates_per_loop': [row['valid_candidates'] for row in loops],
        'backend_graphs': result['compiler']['backend_graphs'],
        'backend_compile_seconds': result['compiler']['backend_compile_seconds']}


def amortization(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    unavailable = {'finite_decision_count': None, 'additional_steady_decisions': None,
                   'meaning': 'requires covered setup and at least two steady loops after final compilation with positive saving'}
    warm_base, warm_candidate = base['warm_host_seconds_per_decision'], candidate['warm_host_seconds_per_decision']
    setup_base, setup_candidate = base.get('installation_host_seconds'), candidate.get('installation_host_seconds')
    if any(value is None for value in (warm_base, warm_candidate, setup_base, setup_candidate)):
        return unavailable
    assert warm_base is not None and warm_candidate is not None and setup_base is not None and setup_candidate is not None
    saving = warm_base-warm_candidate
    count = candidate['initial_population_loops']
    candidate_initial, base_initial = candidate['loops'][:count], base['loops'][:count]
    if len(candidate_initial) != len(base_initial) or any(a['valid_decisions'] != b['valid_decisions']
            for a, b in zip(candidate_initial, base_initial, strict=True)):
        raise ValueError('matched initial compiling populations required')
    overhead = max(0., setup_candidate-setup_base + sum(row['host_wall_seconds'] for row in candidate_initial)
                   -sum(row['host_wall_seconds'] for row in base_initial))
    decisions = sum(row['valid_decisions'] for row in candidate_initial)
    extra = overhead/saving if saving > 0 else None
    return {'warm_saving_seconds_per_decision': saving, 'initial_population_extra_seconds': overhead,
        'installation_extra_seconds': setup_candidate-setup_base, 'initial_population_decisions': decisions,
        'additional_steady_decisions': extra, 'finite_decision_count': decisions+extra if extra is not None else None,
        'meaning': 'all loops through final shape compilation plus installer; measured matched original loops; remaining positive warm saving',
        'excluded_startup': 'interpreter/artifact load before installer and loop timing'}


def require_matched_actual(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    if base['matched_conditions'] != candidate['matched_conditions']:
        raise ValueError('actual pair requires matched model/config/source/freeze/runtime conditions')


def actual_conditions(result: dict[str, Any]) -> dict[str, Any]:
    from gf_common.domain import digest
    request, prep, after = result['request'], result['preparation'], result['after']
    run_id = request['run_id']
    materialized = prep['materialized_config']
    if (request['arm'] != 'C' or request['mode'] != 'fresh' or request['pause_after_update'] is not None
            or materialized['run']['name'] != run_id or after['run_id'] != run_id
            or after['effective_config'] != materialized or digest(materialized) != prep['materialized_sha256']
            or after['config_sha256'] != prep['materialized_sha256']
            or any(result[key] != expected or after[key] != expected for key, expected in
                   (('status','incomplete'), ('stop_reason','update_cap'), ('updates',2)))
            or after['seed'] != 17 or after['device'] != 'cuda:0'):
        raise ValueError('actual original materialization/status conditions differ')
    expected_parent = Path(materialized['hpo_spec']['storage']['parent_root'])/run_id
    if Path(request['parent']) != expected_parent or Path(result['parent']) != expected_parent:
        raise ValueError('actual parent namespace differs from exact run identity')
    normalized = copy.deepcopy(materialized)
    normalized['run']['name'] = '<matched-run-id>'  # Sole materialized difference across actual fresh runs.
    # Compare entire original loaded config/conditions and all materialized fields
    # after validating that sole run-name difference; never broad-drop model data.
    return {'materialized': normalized, 'loaded_config_sha256': prep['loaded_config_sha256'],
        'loaded_effective_config': prep['effective_config'], 'config_inputs': prep['config_inputs'],
        'effective_settings_sha256': prep['effective_settings_sha256'], 'conditions': prep['conditions'],
        'identity': request['identity'], 'code': request['code'], 'config_file_sha256': request['config_file_sha256'],
        'python': request['python'], 'python_executable': request['python_executable'],
        'gil_disabled': request['gil_disabled'], 'platform': request['platform'],
        'torch_runtime': result['torch_runtime'], 'hardware': after['hardware'], 'device': after['device'],
        'seed': after['seed'], 'boot_id': result['gpu_termination']['boot_id']}


def require_telemetry(analyze: Any, result: dict[str, Any], rows: Any, diagnostics: Any) -> None:
    measurement = result['measurement']
    if measurement['exclude_from_performance'] or measurement['dropped'] or measurement['failures']:
        raise ValueError('actual telemetry excluded, dropped or failed')
    issues = analyze.telemetry_issues(result, rows, diagnostics)
    if issues:
        raise ValueError('actual historical telemetry coverage failed: '+repr(issues))


def actual_summary(path: Path) -> dict[str, Any]:
    result = read_json(path/'result.json')
    if result['exit_code'] or not result['probe_expectation_met'] or not result['gpu_termination_context_verified']:
        raise ValueError('actual result lacks successful original update-cap/termination evidence')
    prior = REPO/'runs/hpo-producer-owned/assessment/device-io-detail-v1/analyze.py'
    if not prior.is_file() or prior.is_symlink():
        raise ValueError('required local historical disjoint-ledger analyzer missing: '+str(prior))
    frozen = {str((REPO/name).resolve()): value for name, value in result['request']['identity']['files'].items()}
    if frozen.get(str(prior.resolve())) != sha256(prior):
        raise ValueError('historical analyzer is absent from actual root freeze or changed')
    spec = importlib.util.spec_from_file_location('inference_prior_analyze', prior)
    assert spec is not None and spec.loader is not None
    analyze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze)
    rows = [json.loads(line) for line in (path/'timings.jsonl').read_text().splitlines()]
    diagnostics = [json.loads(line) for line in (path/'diagnostics.jsonl').read_text().splitlines()]
    require_telemetry(analyze, result, rows, diagnostics)
    matched = actual_conditions(result)
    phases = analyze.exclusive_ledger(result['helper_started_monotonic'], result['helper_finished_monotonic'],
                                     result['measurement']['parent_phases'], rows)
    return {'path': str(path), 'options': result['request']['options'], 'matched_conditions': matched,
        'helper_wall_seconds': result['helper_elapsed_seconds'], 'disjoint_phases': phases,
        'collected_decisions': result['collected_decisions'], 'used_decisions': result['used_decisions'],
        'nested_collection_rows': [row for row in rows if row['label'] in ('collector_callback', 'game_operations', 'archive_metrics',
            'archive_queue_and_write', 'actor_synchronization', 'owned_numeric_construction', 'owned_numeric_write')],
        'scope': 'disjoint parent phases reconcile wall; nested/concurrent rows descriptive only'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('replay', 'actual'), required=True)
    parser.add_argument('--pair', nargs=2, action='append', type=Path, required=True,
                        metavar=('ORIGINAL', 'CANDIDATE'))
    args = parser.parse_args()
    condition_keys = (('matched_conditions',) if args.kind == 'actual' else
        ('input_sha256', 'device', 'code', 'cache_state', 'valid_decisions_per_loop', 'valid_candidates_per_loop'))
    pairs: list[dict[str, Any]] = []
    for left, right in args.pair:
        if args.kind == 'replay':
            base, candidate = replay_summary(left), replay_summary(right)
            if any(base[key] != candidate[key] for key in condition_keys):
                raise ValueError('replay pair input/device/code/cache population differs')
            a, b = base['warm_host_seconds_per_decision'], candidate['warm_host_seconds_per_decision']
            delta = None if a is None or b is None else b-a
            pairs.append({'original': base, 'candidate': candidate, 'delta_seconds_per_decision': delta,
                          'amortization': amortization(base, candidate)})
        else:
            base, candidate = actual_summary(left), actual_summary(right)
            require_matched_actual(base, candidate)
            pairs.append({'original': base, 'candidate': candidate,
                'delta_helper_seconds': candidate['helper_wall_seconds']-base['helper_wall_seconds'],
                'delta_collected_decisions': candidate['collected_decisions']-base['collected_decisions'],
                'trajectory_equality': 'not assumed'})
    first = pairs[0]
    if any(any(pair[side]['options'] != first[side]['options'] for side in ('original', 'candidate'))
            or any(pair['original'][field] != first['original'][field] for field in condition_keys)
            for pair in pairs[1:]):
        raise ValueError('median population requires identical conditions and original/candidate option comparison')
    key = 'delta_seconds_per_decision' if args.kind == 'replay' else 'delta_helper_seconds'
    delta_values: list[float] = [pair[key] for pair in pairs if pair[key] is not None]
    print(json.dumps({'kind': args.kind, 'pairs': pairs,
        'median_delta': statistics.median(delta_values) if delta_values else None,
        'qualification': 'exploratory timing only; no quality/strength inference'}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
