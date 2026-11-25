"""Read-only observation of committed standalone HPO records."""

from __future__ import annotations

import os
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, digest, integer
from gf_common.hpo_progress import finite_number
from gf_common.hpo_storage import safe_storage_path
from gf_common.rust_player import validate_context
from gf_models.structured_sealed import MAX_SEALED_SCAN_BYTES, SealedScan, checksum
from gf_training.hpo_update_storage import read_hpo_update
from gf_training.lineage import generation_path

_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError)
_PPO_METRICS = ('total_loss', 'policy_loss', 'value_mse', 'entropy', 'approx_kl')
_DIAGNOSTICS = ('clip_fraction', 'explained_variance', 'pre_clip_grad_norm_mean',
                'pre_clip_grad_norm_max', 'learning_rate')
_DURATIONS = ('collection_seconds', 'update_seconds')


def collection_rate(count: object, seconds: Any) -> float | None:
    if (type(count) is not int or not 0 <= count < 2**64
            or not finite_number(seconds) or seconds <= 0):
        return None
    rate = count / seconds
    return rate if finite_number(rate) else None


def _elapsed(state: Json) -> float | None:
    elapsed = state.get('elapsed_seconds')
    if elapsed is not None and finite_number(elapsed) and elapsed >= 0:
        return float(elapsed)
    start, last = state.get('created_unix'), state.get('last_unix')
    if (start is not None and last is not None
            and finite_number(start) and finite_number(last) and last >= start):
        elapsed = last - start
        if finite_number(elapsed):
            return float(elapsed)
    return None


def _standalone_namespace(parent: Path, operation: Json) -> tuple[Path, Json]:
    parent = parent.absolute()
    context = validate_context(operation['expected_context'])
    run, experiment = context['run_id'], context['campaign_id']
    if (str(uuid.UUID(run)) != run or str(uuid.UUID(experiment)) != experiment
            or parent.name != run or operation['run_id'] != run
            or operation['parent_directory'] != str(parent)):
        raise ContractError('standalone progress requires the exact configured parent and safe IDs')
    safe_storage_path(parent)
    return parent.parent / '.operation-control' / experiment / run, context


def namespace_identity(directory: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    safe_storage_path(directory)
    info = directory.lstat()
    lock = (directory / 'lock').lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
            or not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.getuid() or lock.st_nlink != 1):
        raise ContractError('standalone progress requires the original private control namespace')
    return (info.st_dev, info.st_ino), (lock.st_dev, lock.st_ino)


def checked_namespace(parent: Path, operation: Json, scan: SealedScan) -> tuple[
        Path, tuple[int, int], tuple[int, int]]:
    directory, context = _standalone_namespace(parent, operation)
    identity, lock = namespace_identity(directory)
    record = scan.object(directory / 'operation-control.json')
    expected = {'schema_version': 1, 'kind': 'hpo-operation-control-data-v1',
        'experiment_id': context['campaign_id'], 'run_id': context['run_id'],
        'namespace_directory': str(directory), 'parent_directory': str(parent.absolute()),
        'config_sha256': operation['config_sha256'],
        'namespace_identity': {'directory_device': identity[0], 'directory_inode': identity[1],
            'lock_device': lock[0], 'lock_inode': lock[1]}}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ContractError('standalone progress control namespace record differs')
    return directory, identity, lock


def _record_scan() -> SealedScan:
    return SealedScan(max_owned_bytes=MAX_SEALED_SCAN_BYTES, deadline=time.monotonic() + 1.)


def _standalone_operation(root: Path, scan: SealedScan) -> Json | None:
    safe_storage_path(root)
    operation = scan.object(root / '.rust-player/operation-fresh.json')
    if operation.get('schema_version') == 2 and operation.get('operation_type') == 'hpo-train':
        return operation
    return None


def _standalone_state(root: Path, operation: Json, scan: SealedScan) -> Json:
    state = scan.object(root / 'run.json')
    effective = state['effective_config']
    config_sha = checksum(state['config_sha256'])
    spec = effective['hpo_spec']
    if (state.get('schema_version') != 5 or state.get('run_variant') != 'hpo-ppo-owned-rust-player-v1'
            or state.get('model_id') != 'mlp' or spec.get('schema_version') != 6
            or operation.get('run_id') != state['run_id'] or operation.get('config_sha256') != config_sha
            or digest(effective) != config_sha):
        raise ContractError('standalone progress Run5/operation identity differs')
    if state['status'] not in ('running', 'paused', 'complete', 'incomplete', 'failed'):
        raise ContractError('unrecognized standalone run status')
    integer(effective['training']['max_updates'], 1)
    integer(spec['decision_budget'], 1)
    integer(state['seed'], 0)
    return state


def _read_generation(root: Path, state: Json, number: int, sha: str, scan: SealedScan) -> Json:
    generation = scan.object(generation_path(root, number), sha)
    if (generation.get('schema_version') != 6 or generation.get('run_id') != state['run_id']
            or type(generation.get('generation')) is not int or generation['generation'] != number):
        raise ContractError('standalone progress generation identity differs')
    return generation


def _read_update(
    root: Path, state: Json, number: int, generation: Json, scan: SealedScan,
) -> Json:
    update = read_hpo_update(root, number, state, generation['update_sha256'], scan=scan)
    if (update.get('schema_version') != 8 or update.get('status') != 'complete'
            or update.get('run_id') != state['run_id']
            or type(update.get('generation')) is not int or update['generation'] != number):
        raise ContractError('standalone progress update identity differs')
    return update


@dataclass(frozen=True)
class ScalarUpdate:
    step: int
    values: dict[str, float]


def _scalars(number: int, generation: Json, update: Json, budget: int) -> ScalarUpdate:
    metrics = update['metrics']
    used = integer(generation['used_decisions'], 0, budget)
    collected = integer(generation['collected_decisions'], used)
    values: dict[str, float] = {
        'progress/used_decisions': used,
        'progress/collected_decisions': collected,
    }
    values.update({f'ppo/{name}': metrics[name] for name in _PPO_METRICS
                   if finite_number(metrics.get(name))})
    diagnostics = metrics.get('diagnostics')
    if type(diagnostics) is dict and diagnostics.get('schema_version') == 1:
        values.update({f'diagnostics/{name}': diagnostics[name] for name in _DIAGNOSTICS
                       if finite_number(diagnostics.get(name))})
    values.update({f'timing/{name}': update[name] for name in _DURATIONS
                   if finite_number(update.get(name)) and update[name] >= 0})
    return ScalarUpdate(number, values)


class StandaloneScalarHistory:
    """Incremental verified scalar history for one dashboard session."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self._head: tuple[str, str, int, str] | None = None

    def poll(self) -> list[ScalarUpdate] | None:
        """Return new verified rows, or None while a consistent head is unavailable."""
        try:
            return self._poll()
        except _READ_ERRORS:
            return None

    def _poll(self) -> list[ScalarUpdate] | None:
        scan = _record_scan()
        operation = _standalone_operation(self.root, scan)
        if operation is None:
            return None
        state = _standalone_state(self.root, operation, scan)
        if state['generation_sha256'] is None:
            return None
        effective = state['effective_config']
        number = integer(state['updates'], 0, effective['training']['max_updates'])
        sha = checksum(state['generation_sha256'])
        latest = scan.object(self.root / 'latest.json')
        if type(latest.get('generation')) is not int or latest != {'generation': number, 'sha256': sha}:
            return None
        head = (state['run_id'], state['config_sha256'], number, sha)
        prior = self._head
        if head == prior:
            return []
        if prior is not None and (head[:2] != prior[:2] or number <= prior[2]):
            return None

        rows: list[ScalarUpdate] = []
        anchor = 0 if prior is None else prior[2]
        budget = effective['hpo_spec']['decision_budget']
        for step in range(number, anchor, -1):
            record_scan = _record_scan()
            generation = _read_generation(self.root, state, step, sha, record_scan)
            sha = checksum(generation['parent_sha256'])
            update = _read_update(self.root, state, step, generation, record_scan)
            if update.get('parent_sha256') != sha:
                return None
            rows.append(_scalars(step, generation, update, budget))
            del update
        if prior is None:
            initial = _read_generation(self.root, state, 0, sha, _record_scan())
            if initial['parent_sha256'] is not None or initial['update_sha256'] is not None:
                return None
        elif sha != prior[3]:
            return None
        self._head = head
        rows.reverse()
        return rows


class StandaloneProgressRecords:
    """One watcher's last coherent diagnostic observation, without lock custody."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.live_root: Path | None = None
        self._last: Json | None = None
        self._commit: tuple[tuple[str, str, int, str], Json] | None = None

    def read(self) -> Json | None:
        self.live_root = None
        try:
            value = self._read()
        except _READ_ERRORS:
            if self._last is None:
                return None
            return {**self._last, 'availability': 'stale'}
        self._last = value
        return value

    def _read(self) -> Json | None:
        scan = _record_scan()
        root = self.root.absolute()
        # Only standalone HPO uses this initial operation record. Dashboard
        # parents and the ordinary PPO lane keep their existing watcher shape.
        operation = _standalone_operation(root, scan)
        if operation is None:
            return None
        try:
            self.live_root, _, _ = checked_namespace(root, operation, scan)
        except _READ_ERRORS:
            # The older run may have no retained operation-control namespace.
            # The parent pointer still supports the read-only durable view.
            pass
        state = _standalone_state(root, operation, scan)
        effective = state['effective_config']
        config_sha = state['config_sha256']
        status = state['status']
        target = effective['training']['max_updates']
        budget = effective['hpo_spec']['decision_budget']
        host_result = state.get('host_result')
        observed: Json = {'availability': 'available', 'status': status, 'model': 'mlp',
            'seed': state['seed'], 'decision_budget': budget, 'target_updates': target,
            'elapsed_seconds': _elapsed(state),
            'host_result': ('awaiting-supervised-finalization'
                if host_result == 'awaiting-supervised-finalization' else None),
            'committed_updates': None,
            'committed_used': None, 'committed_collected': None,
            'latest_metrics': None, 'latest_durations': None,
            'last_collection_games_per_second': None, 'last_collection_actions_per_second': None}
        if state['generation_sha256'] is None:
            return observed
        try:
            number = state['updates']
            integer(number, 0, target)
            identity = (state['run_id'], config_sha, number, checksum(state['generation_sha256']))
            if self._commit is None or self._commit[0] != identity:
                commit = self._read_commit(state, identity, budget, scan)
                self._commit = identity, commit
            return {**observed, **self._commit[1]}
        except _READ_ERRORS:
            # A counterpart may have been atomically replaced between reads.
            # Never pair a newly read counter with metrics from another commit.
            if self._commit is not None and self._commit[0][:2] == (state['run_id'], config_sha):
                return {**observed, **self._commit[1], 'availability': 'stale'}
            return {**observed, 'availability': 'partial'}

    def _read_commit(self, state: Json, identity: tuple[str, str, int, str],
                     budget: int, scan: SealedScan) -> Json:
        _, _, number, sha = identity
        generation = _read_generation(self.root, state, number, sha, scan)
        used, collected = generation['used_decisions'], generation['collected_decisions']
        integer(used, 0, budget)
        integer(collected, used)
        observed: Json = {'committed_updates': number, 'committed_used': used, 'committed_collected': collected,
            'latest_metrics': None, 'latest_durations': None,
            'last_collection_games_per_second': None, 'last_collection_actions_per_second': None}
        if number:
            update = _read_update(self.root, state, number, generation, scan)
            metrics = update.get('metrics', {})
            observed['latest_metrics'] = {name: metrics[name]
                for name in _PPO_METRICS
                if finite_number(metrics.get(name))}
            seconds = update.get('collection_seconds')
            proof = update.get('source_proof')
            actions = proof.get('accepted_decisions') if type(proof) is dict else None
            observed['last_collection_games_per_second'] = collection_rate(update.get('actual_games'), seconds)
            observed['last_collection_actions_per_second'] = collection_rate(actions, seconds)
            observed['latest_durations'] = {name: update[name]
                for name in _DURATIONS
                if finite_number(update.get(name)) and update[name] >= 0}
        return observed
