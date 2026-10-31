"""PPO coordinator lock and immutable generation/update verification."""
from __future__ import annotations

import fcntl
import json
import math
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import (
    ContractError,
    Json,
    RuntimePolicyBinding,
    canonical,
    closed,
    digest,
    integer,
)
from gf_models.arrays import sha256
from gf_models.ppo_identity import model_id
from gf_training.journal import atomic_json


def read_json(path: Path, *, session: Any = None) -> Json:
    if session is not None:
        return session.read_json(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError('object record required')
    return value


_OWNED_LOCKS: dict[Path, str] = {}


@dataclass(frozen=True)
class RunOwnership:
    directory: Path
    token: str


@contextmanager
def run_lock(directory: Path, *, create: bool = False) -> Iterator[RunOwnership]:
    directory = directory.resolve()
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError('PPO run already owned') from exc
        token = uuid.uuid4().hex
        _OWNED_LOCKS[directory] = token
        try:
            yield RunOwnership(directory, token)
        finally:
            _OWNED_LOCKS.pop(directory, None)
            fcntl.flock(lock, fcntl.LOCK_UN)


def immutable_json(path: Path, value: Any) -> str:
    with path.open('x') as output:
        output.write(canonical(value) + '\n')
    return sha256(path)


def generation_path(directory: Path, generation: int) -> Path:
    return directory / 'generations' / f'{generation:06}.json'


def checkpoint_path(directory: Path, generation: int) -> Path:
    return directory / 'artifacts' / f'{generation:06}'


def artifact_hashes(path: Path, deadline: float = float("inf"), *, session: Any = None) -> Json:
    return {name: session.hash_file(path / name) if session is not None else sha256(path / name, deadline)
            for name in ('inference.json', 'weights.npz', 'weights.json', 'resume.json', 'resume.npz')}


def validate_chain(directory: Path, *, require_complete: bool = True, require_artifacts: bool = False, deadline: float = float("inf"), session: Any = None) -> tuple[Json, list[Json]]:
    def _hash(path: Path) -> str:
        return session.hash_file(path) if session is not None else sha256(path, deadline)
    def _read(path: Path) -> Json:
        return read_json(path, session=session)
    if time.monotonic() >= deadline:
        raise TimeoutError("generation validation deadline")
    state = _read(directory / 'run.json')
    if state.get('schema_version') != 1 or state.get('algorithm') != 'ppo' or state.get('status') not in ('running', 'paused', 'complete') or require_complete and state.get('status') != 'complete':
        raise ContractError('healthy completed PPO parent required')
    if any(state.get(key) for key in ('failure', 'quarantine', 'cleanup_failure', 'budget_overrun_seconds')):
        raise ContractError('PPO parent contains failure evidence')
    config = state.get('effective_config', {})
    run = config.get('run', {})
    identity = model_id(config.get('model', {}))
    if digest(config) != state.get('config_sha256') or state.get('device') not in ('cpu', 'cuda:0'):
        raise ContractError('PPO parent config/device identity mismatch')
    times: list[Any] = [state.get(key) for key in ('created_unix', 'last_unix', 'wall_deadline_unix', 'admission_deadline_unix', 'execution_deadline_unix')]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in times):
        raise ContractError('invalid parent clock provenance')
    created, last, wall, admission, execution = times
    if not created <= last <= wall or not created < admission <= execution < wall or wall - created > 7200 or admission - created > 5400 or execution - created > 7140 or abs((wall - created) - run.get('wall_seconds', -1)) > .001:
        raise ContractError('parent original deadline/elapsed provenance invalid')
    if state['status'] in ('paused', 'complete'):
        elapsed: Any = state.get('elapsed_seconds')
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= wall - created or abs(elapsed - (last - created)) > .001:
            raise ContractError('parent completion elapsed exceeds original budget')
        for invocation in state.get('invocations', []):
            health = invocation.get('child_health', {})
            if health.get('exit_code') != 0 or set(health) - {'exit_code', 'stderr_bytes'}:
                raise ContractError('parent learner invocation unhealthy')
    latest = _read(directory / 'latest.json')
    closed(latest, {'generation', 'sha256'})
    count = integer(state.get('updates'), 0, 1000)
    if latest['generation'] != count or latest['sha256'] != _hash(generation_path(directory, count)):
        raise ContractError('incomplete latest generation commit')
    if state.get('generation_sha256') != latest['sha256']:
        raise ContractError('run/latest commit mismatch')
    expected_generations = {generation_path(directory, number).name for number in range(count + 1)}
    if {path.name for path in (directory / 'generations').glob('*.json')} != expected_generations:
        raise ContractError('missing/extra immutable generation record')
    if {path.name for path in (directory / 'updates').glob('*.json')} != {f'{number:06}.json' for number in range(1, count + 1)}:
        raise ContractError('missing/extra update commit')
    registry = _read(directory / 'artifact-registry.json')
    records = []
    previous = None
    cumulative = 0
    steps = 0
    for number in range(count + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("generation validation deadline")
        record = _read(generation_path(directory, number))
        closed(record, {'schema_version', 'run_id', 'generation', 'parent_sha256', 'inference_sha256', 'weights_sha256',
                        'artifact_hashes', 'accepted_decisions', 'optimizer_steps', 'update_sha256', 'parent_controls'})
        if record['schema_version'] != 1 or record['run_id'] != state['run_id'] or record['generation'] != number or record['parent_sha256'] != previous:
            raise ContractError('broken generation chain')
        path = checkpoint_path(directory, number)
        # Public inference manifests are never deleted.
        if _hash(path / 'inference.json') != record['inference_sha256']:
            raise ContractError('generation inference hash mismatch')
        inference = _read(path / 'inference.json')
        if model_id(inference.get('model', {})) != identity:
            raise ContractError('generation model differs from parent identity')
        if inference['weights']['sha256'] != record['weights_sha256']:
            raise ContractError('generation weights identity mismatch')
        retained = number == 0 or number % 10 == 0 or number == count
        receipt = registry.get(str(number))
        if receipt is None or receipt.get('inference_sha256') != record['inference_sha256']:
            raise ContractError('missing artifact registry identity')
        if retained or require_artifacts:
            if artifact_hashes(path, deadline, session=session) != record['artifact_hashes'] or receipt.get('status') != 'retained':
                raise ContractError('retained generation artifacts mismatch')
        elif receipt.get('status') == 'deleted':
            if receipt.get('deletion_intent') is not True or any((path / name).exists() for name in ('weights.npz', 'resume.npz', 'resume.json', 'resume.json.tmp')):
                raise ContractError('artifact deletion receipt mismatch')
        elif receipt.get('status') != 'retained' or artifact_hashes(path, deadline, session=session) != record['artifact_hashes']:
            raise ContractError('unrecognized generation artifact state')
        if number:
            update_path = directory / 'updates' / f'{number:06}.json'
            update = _read(update_path)
            if record['update_sha256'] != _hash(update_path) or update.get('run_id') != state['run_id'] or update.get('generation') != number or update.get('parent_sha256') != previous or update.get('status') != 'complete':
                raise ContractError('broken immutable update linkage')
            source = directory / update['collection']
            if not source.resolve().is_relative_to(directory.resolve()) or _hash(source / 'manifest.json') != update['collection_manifest_sha256'] or _hash(directory / update['numeric'] / 'rollout.json') != update['numeric_sha256']:
                raise ContractError('update source/hash linkage mismatch')
            cumulative += integer(update.get('accepted_decisions'), 1)
            if record['accepted_decisions'] != cumulative or integer(record['optimizer_steps'], 1) <= steps:
                raise ContractError('generation counters mismatch')
        elif record['accepted_decisions'] != 0 or record['optimizer_steps'] != 0 or record['update_sha256'] is not None:
            raise ContractError('generation zero counters mismatch')
        steps = record['optimizer_steps']
        previous = _hash(generation_path(directory, number))
        records.append(record)
    if require_complete and (not count or state.get('completed_updates') != count):
        raise ContractError('zero-update/unfinalized PPO run ineligible')
    return state, records


def qualify_ppo(checkpoint: Path, expected_digest: str) -> Json:
    directory = checkpoint.resolve().parent.parent
    with run_lock(directory):
        state, records = validate_chain(directory)
        if checkpoint.resolve() != checkpoint_path(directory, state['updates']).resolve() or records[-1]['inference_sha256'] != expected_digest:
            raise ContractError('formal PPO binding requires final committed policy')
        return state


def qualify_mlp(checkpoint: Path, expected_digest: str) -> Json:
    """Historical entrypoint; qualification is shared by the closed PPO models."""
    return qualify_ppo(checkpoint, expected_digest)


def owned_binding(directory: Path, generation: int) -> tuple[str, RuntimePolicyBinding]:
    # Caller holds the coordinator lifetime lock; do not relax general qualification.
    record = read_json(generation_path(directory, generation))
    path = checkpoint_path(directory, generation)
    if sha256(path / 'inference.json') != record['inference_sha256'] or sha256(path / 'weights.npz') != record['weights_sha256']:
        raise ContractError('owned frozen policy artifact changed')
    sha = record['inference_sha256']
    identity = model_id(read_json(path / 'inference.json')['model'])
    return identity + '-' + sha, RuntimePolicyBinding(identity, {'checkpoint': str(path.resolve()), 'expected_manifest_digest': sha})


def commit_generation(directory: Path, state: Json, *, generation: int, update: Json | None, controls: Json, deadline: float = float("inf")) -> Json:
    path = checkpoint_path(directory, generation)
    hashes = artifact_hashes(path, deadline)
    parent = state.get('generation_sha256')
    update_hash = immutable_json(directory / 'updates' / f'{generation:06}.json', update) if update is not None else None
    record = {'schema_version': 1, 'run_id': state['run_id'], 'generation': generation, 'parent_sha256': parent,
              'inference_sha256': hashes['inference.json'], 'weights_sha256': hashes['weights.npz'],
              'artifact_hashes': hashes, 'accepted_decisions': state.get('accepted_decisions', 0),
              'optimizer_steps': state.get('optimizer_steps', 0), 'update_sha256': update_hash, 'parent_controls': controls}
    if time.monotonic() >= deadline:
        raise TimeoutError('generation commit deadline')
    sha = immutable_json(generation_path(directory, generation), record)
    registry = read_json(directory / 'artifact-registry.json')
    registry[str(generation)] = {'status': 'retained', 'inference_sha256': hashes['inference.json']}
    atomic_json(directory / 'artifact-registry.json', registry)
    if time.monotonic() >= deadline:
        raise TimeoutError('generation commit deadline')
    atomic_json(directory / 'latest.json', {'generation': generation, 'sha256': sha})
    state.update(updates=generation, generation_sha256=sha)
    atomic_json(directory / 'run.json', state)
    if time.monotonic() >= deadline:
        raise TimeoutError("generation commit deadline")
    return record


def retain(directory: Path, generation: int, *, deadline: float = float("inf")) -> None:
    predecessor = generation - 1
    if predecessor <= 0 or predecessor % 10 == 0:
        return
    registry = read_json(directory / 'artifact-registry.json')
    if time.monotonic() >= deadline:
        raise TimeoutError('retention deadline')
    registry[str(predecessor)]['deletion_intent'] = True
    atomic_json(directory / 'artifact-registry.json', registry)
    path = checkpoint_path(directory, predecessor)
    for name in ('weights.npz', 'resume.npz', 'resume.json'):
        if time.monotonic() >= deadline:
            raise TimeoutError('retention deadline')
        (path / name).unlink()
    if time.monotonic() >= deadline:
        raise TimeoutError('retention deadline')
    registry[str(predecessor)]['status'] = 'deleted'
    atomic_json(directory / 'artifact-registry.json', registry)


def validate_collection_linkage(manifest: Json) -> None:
    closed(manifest.get('ppo'), {'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation', 'learner_seats'})
    if manifest.get('collection_kind') not in ('ppo_batch', 'ppo_validation'):
        raise ContractError('unsupported schema-4 collection kind')
    link = manifest['ppo']
    if not isinstance(link['parent_run_id'], str) or not isinstance(link['parent_run_path'], str) or not isinstance(link['batch_id'], str):
        raise ContractError('invalid PPO collection identity')
    integer(link['collection_generation'], 0, 1000)
    seats = {}
    for group in manifest.get('schedule', []):
        for spec in group.get('episodes', []):
            if spec.get('learner_viewer') not in (1, 2) or type(spec['learner_viewer']) is not int:
                raise ContractError('explicit schema-4 learner seat required')
            seats[spec['episode_id']] = spec['learner_viewer']
    if not seats or seats != link['learner_seats']:
        raise ContractError('PPO schedule/learner linkage mismatch')
    if manifest.get('collection_kind') == 'ppo_batch' and (len(seats) != 16 or len(manifest['schedule']) != 8):
        raise ContractError('PPO batch requires all eight swapped pairs')


def validate_collection_parent(manifest: Json, collection_directory: Path, *, online_context: Any = None,
                               parent_cache: dict[str, Any] | None = None, locks: ExitStack | None = None, deadline: float = float("inf"), session: Any = None) -> None:
    """Default is offline. Only a live coordinator token admits its current batch."""
    import numpy as np

    from gf_common.schedule import serialize_schedule
    from gf_training.collection import OnlinePPOContext, training_groups, validation_groups
    if time.monotonic() >= deadline:
        raise TimeoutError("collection parent validation deadline")
    validate_collection_linkage(manifest)
    link = manifest['ppo']
    parent = Path(link['parent_run_path']).resolve()
    if parent_cache is None or locks is None:
        with ExitStack() as owned:
            validate_collection_parent(manifest, collection_directory, online_context=online_context, parent_cache={}, locks=owned, deadline=deadline, session=session)
        return
    key = str(parent)
    if key not in parent_cache:
        if parent not in _OWNED_LOCKS:
            locks.enter_context(run_lock(parent))
        parent_cache[key] = validate_chain(parent, require_complete=online_context is None, deadline=deadline, session=session)
    state, records = parent_cache[key]
    integration = state['identity']['integration']
    if manifest.get('binary_sha256') != integration['binary_sha256'] or manifest.get('server_config_sha256') != integration['server_config_sha256'] or manifest.get('simulator', {}).get('revision') != integration['source_revision'] or manifest.get('simulator', {}).get('spec_sha256') != integration['source_spec_sha256']:
        raise ContractError('collection Simulator integration differs from frozen parent identity')
    generation = link['collection_generation']
    if link['parent_run_id'] != state['run_id'] or generation >= len(records):
        raise ContractError('collection parent/generation mismatch')
    if online_context is not None:
        if not isinstance(online_context, OnlinePPOContext) or _OWNED_LOCKS.get(parent) != online_context.ownership_token or online_context.parent_directory.resolve() != parent or online_context.parent_run_id != state['run_id'] or online_context.batch_id != link['batch_id'] or online_context.generation != generation or generation != state['updates'] or state['status'] != 'running' or manifest['collection_kind'] != 'ppo_batch':
            raise ContractError('owned online PPO context does not authorize this exact frozen batch')
    elif state['status'] != 'complete':
        raise ContractError('offline PPO data requires a healthy completed whole parent')
    identity = model_id(state['effective_config']['model'])
    label = identity + '-' + records[generation]['inference_sha256']
    config = records[generation]['parent_controls']
    registry = {'random': RuntimePolicyBinding('random'), 'heuristic': RuntimePolicyBinding('heuristic')}
    # Reconstruct bindings from immutable inference identities; tensors may have been retained/deleted.
    def binding(number: int) -> tuple[str, RuntimePolicyBinding]:
        sha = records[number]['inference_sha256']
        return identity + '-' + sha, RuntimePolicyBinding(identity, {'checkpoint': str(checkpoint_path(parent, number)), 'expected_manifest_digest': sha})
    name, value = binding(generation)
    registry[name] = value
    if manifest['collection_kind'] == 'ppo_batch':
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = config['schedule_opponent_rng']
        snapshots = []
        for number in config['snapshot_pool']:
            name, value = binding(number)
            registry[name] = value
            snapshots.append(name)
        expected = training_groups(state['seed'], generation, label, tuple(snapshots), rng)
        if online_context is None:
            if generation >= state['updates']:
                raise ContractError('offline batch has no committed update')
            update = read_json(parent / 'updates' / f'{generation + 1:06}.json', session=session)
            if (parent / update['collection']).resolve() != collection_directory.resolve() or update['batch_id'] != link['batch_id'] or (session.hash_file(collection_directory / 'manifest.json') if session is not None else sha256(collection_directory / 'manifest.json')) != update['collection_manifest_sha256']:
                raise ContractError('offline batch differs from immutable update source')
    else:
        if online_context is not None or generation != state['updates']:
            raise ContractError('validation must use the completed final generation')
        expected = validation_groups(state['seed'], label)
    if manifest['schedule'] != serialize_schedule(expected, 4) or manifest.get('policy_bindings') != {key: {'model_id': value.model_id, 'policy_options': value.policy_options} for key, value in registry.items()}:
        raise ContractError('PPO source schedule/frozen policy bindings mismatch')
