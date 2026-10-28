"""Frozen complete successful PPO sources, revalidated from original evidence."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.preprocessing import PreparationConfig
from gf_common.schedule import serialize_schedule
from gf_models.config import EncoderConfig, MLPModelConfig, PPOTrainingConfig
from gf_models.mlp.arrays import sha256
from gf_training.collection import training_groups, validation_groups
from gf_training.journal import (
    AcceptedDecision,
    _project_journals,
    atomic_json,
    projection_metadata,
)
from gf_training.lineage import generation_path, owned_binding, read_json, run_lock, validate_chain
from gf_training.preparation import PreparationSession, deep_size
from gf_training.preparation_index import IndexLedger
from gf_training.preparation_numeric import PreparedInputs, prepare_inputs
from gf_training.q_data import (
    RunBundle,
    Sample,
    check_deadline,
    load_run,
    provenance,
)

SELECTION = 'all-explicit-learner-accepted-natural-returns-v1'


def _scan_sources(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], *,
                  session: PreparationSession,
                  sink: Callable[[AcceptedDecision, RunBundle], None] | None = None) -> Json:
    deadline = session.deadline
    if len(ppo_runs) != 3 or len(validation_runs) != 3 or len(set(ppo_runs + validation_runs)) != 6:
        raise ContractError('fixed dataset requires exactly three distinct ordered PPO and validation parents')
    all_bundles: list[RunBundle] = []
    parents = []
    collections = []
    total_seconds = 0.
    expected_proofs = {}
    for directory in sorted(ppo_runs):
        session.validate_parent(directory)
    for directory, validation_directory, seed in zip(ppo_runs, validation_runs, (17, 29, 43), strict=True):
        state, records = session.validate_parent(directory)
        if state['seed'] != seed:
            raise ContractError('fixed ordered PPO learning seeds must be 17,29,43')
        registry = session.read_json(directory / 'artifact-registry.json')
        generations = []
        for number, record in enumerate(records):
            check_deadline(deadline)
            generations.append({'generation': number, 'record_sha256': session.hash_file(generation_path(directory, number)),
                                'inference_sha256': record['inference_sha256'], 'weights_sha256': record['weights_sha256']})
            if not number:
                continue
            update = session.read_json(directory / 'updates' / f'{number:06}.json')
            bundle = load_run(directory / update['collection'], 'train', deadline=deadline, session=session)
            link = bundle.manifest.get('ppo', {})
            if link.get('parent_run_id') != state['run_id'] or Path(link.get('parent_run_path', '')).resolve() != directory or link.get('collection_generation') != number - 1 or link.get('batch_id') != update['batch_id']:
                raise ContractError('offline train collection parent/generation linkage mismatch')
            # Reconstruct the exact frozen mixed-opponent schedule from committed RNG controls.
            previous_controls = records[number - 1]['parent_controls']
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = previous_controls['schedule_opponent_rng']
            learner = 'mlp-' + records[number - 1]['inference_sha256']
            pool = tuple('mlp-' + records[item]['inference_sha256'] for item in previous_controls['snapshot_pool'])
            expected = training_groups(seed, number - 1, learner, pool, rng)
            if bundle.manifest['schedule'] != serialize_schedule(expected, 4):
                raise ContractError('offline train schedule differs from immutable generation controls')
            actual_bindings = bundle.manifest.get('policy_bindings')
            if actual_bindings != update['policy_bindings']:
                raise ContractError('collection frozen binding evidence mismatch')
            expected_proofs[bundle.directory] = update['source_proof']['source_set_sha256']
            all_bundles.append(bundle)
            collections.append(provenance(bundle, deadline=deadline, session=session))
        validation = load_run(validation_directory, 'validation', deadline=deadline, session=session)
        link = validation.manifest.get('ppo', {})
        final_label = 'mlp-' + records[-1]['inference_sha256']
        expected_validation = validation_groups(seed, final_label)
        if validation.manifest.get('collection_kind') != 'ppo_validation' or link.get('parent_run_id') != state['run_id'] or Path(link.get('parent_run_path', '')).resolve() != directory or link.get('collection_generation') != state['updates'] or validation.manifest['schedule'] != serialize_schedule(expected_validation, 4):
            raise ContractError('validation requires exactly 32 prescribed final-policy swapped pairs')
        all_bundles.append(validation)
        collections.append(provenance(validation, deadline=deadline, session=session))
        total_seconds += state['elapsed_seconds'] + validation.collection_seconds
        parents.append({'directory': str(directory), 'run_id': state['run_id'], 'run_sha256': session.hash_file(directory / 'run.json'),
                        'seed': seed, 'generations': generations, 'accepted_decisions': state['accepted_decisions'],
                        'generation_cost_seconds': state['elapsed_seconds'], 'artifact_registry_sha256': digest(registry)})
    if len({bundle.manifest['run_id'] for bundle in all_bundles}) != len(all_bundles):
        raise ContractError('duplicate source run identity')
    source_metadata: list[Json] = []
    manifest_metadata: dict[str, Json] = {}
    counts = {'train': 0, 'validation': 0}
    for bundle in sorted(all_bundles, key=lambda bundle: str(bundle.directory) + '/'):
        def accepted(decision: AcceptedDecision, bundle: RunBundle = bundle) -> None:
            counts[decision.split] += 1
            if sink is not None:
                sink(decision, bundle)
        _, part = _project_journals(list(bundle.journals), deadline=deadline, session=session,
                                    decision_sink=accepted, materialize=False)
        if not all(source['eligible'] for source in part['sources']):
            raise ContractError('source bundle failed journal/report/replay/health verification')
        expected_proof = expected_proofs.get(bundle.directory)
        if expected_proof is not None and part['source_set_sha256'] != expected_proof:
            raise ContractError('offline source journal/Replay proof changed since PPO update')
        source_metadata.extend(part['sources'])
        for run_id, proof in part['manifests'].items():
            previous = manifest_metadata.get(run_id)
            if previous is not None and previous['sha256'] != proof['sha256']:
                raise ContractError('run identity refers to different manifests')
            manifest_metadata[run_id] = proof
    session.charge(len(source_metadata) * 2048)  # Global DSU/proof container headroom.
    evidence = projection_metadata(source_metadata, manifest_metadata, deadline=deadline)
    action_hashes = {source['header']['action_config_hash'] for source in evidence['sources']}
    if len(action_hashes) != 1:
        raise ContractError('frozen sources must share identical action conditions')
    if not counts['train'] or not counts['validation'] or counts['train'] != sum(parent['accepted_decisions'] for parent in parents):
        raise ContractError('fixed dataset complete learner decision count mismatch')
    manifest = {'schema_version': 1, 'selection': SELECTION, 'ppo_runs': [str(path) for path in ppo_runs],
                'validation_runs': [str(path) for path in validation_runs], 'parents': parents, 'collections': collections,
                'source_set_sha256': evidence['source_set_sha256'], 'train_decisions': counts['train'],
                'validation_decisions': counts['validation'], 'validation_games': 192,
                'shared_generation_seconds': total_seconds,
                'shared_generation_cost_key': digest({'parent_run_ids': [parent['run_id'] for parent in parents], 'validation_run_ids': [bundle.manifest['run_id'] for bundle in all_bundles if bundle.manifest['split'] == 'validation']}),
                'action_config_hash': next(iter(action_hashes)),
                'label_recipe': 'behavior-selected-natural-return-gamma1-v1',
                'split_namespaces': {'train': 'ppo-train-v1', 'validation': 'fixed-existing-32', 'test': 'reserved-not-run'}}
    manifest['content_sha256'] = digest(manifest)
    session.retain(manifest)
    return manifest



def _validated_dataset(path: Path, session: PreparationSession,
                       sink: Callable[[AcceptedDecision, RunBundle], None] | None = None) -> Json:
    frozen = session.read_json(path)
    if frozen.get('schema_version') != 1 or frozen.get('selection') != SELECTION:
        raise ContractError('unsupported fixed dataset manifest')
    actual = _scan_sources(tuple(Path(item).resolve() for item in frozen['ppo_runs']),
                           tuple(Path(item).resolve() for item in frozen['validation_runs']), session=session, sink=sink)
    if actual != frozen:
        raise ContractError('frozen dataset original evidence/content changed')
    return actual


def _validate_sources(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], *, deadline: float) -> tuple[Json, tuple[Sample, ...], tuple[Sample, ...]]:
    train: list[Sample] = []
    validation: list[Sample] = []
    with PreparationSession(deadline=deadline) as session:
        def retain(decision: AcceptedDecision, bundle: RunBundle) -> None:
            sample = Sample(decision.value, decision.action, decision.target, decision.identity)
            session.retain(sample)
            (train if decision.split == 'train' else validation).append(sample)
        manifest = _scan_sources(ppo_runs, validation_runs, session=session, sink=retain)
    return manifest, tuple(train), tuple(validation)


def _publish_owned(session: PreparationSession, outputs: tuple[tuple[Path, Path], ...]) -> None:
    """Fresh-path publication stays owned through every post-write check."""
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        session.finalize()
        check_deadline(session.wall_deadline)
        for temporary, destination in outputs:
            # Atomic no-overwrite publication also protects a target created
            # while the long source scan was running.
            stat = temporary.stat()
            os.link(temporary, destination)
            published.append((destination, (stat.st_dev, stat.st_ino)))
        for temporary, _ in outputs:
            temporary.unlink()
        check_deadline(session.wall_deadline)
    except BaseException:
        for destination, identity in reversed(published):
            try:
                stat = destination.stat()
                if (stat.st_dev, stat.st_ino) == identity:
                    destination.unlink()
            except FileNotFoundError:
                continue
            except OSError as error:
                session.diagnostics['publication_cleanup_error'] = str(error)
        raise


def build_dataset(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], output: Path, *,
                  deadline: float = float('inf'), preprocessing: PreparationConfig | None = None) -> Json:
    if output.exists(follow_symlinks=False):
        raise ContractError('frozen dataset output already exists')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with PreparationSession(preprocessing, deadline=deadline) as session:
            manifest = _scan_sources(tuple(path.resolve() for path in ppo_runs),
                                     tuple(path.resolve() for path in validation_runs), session=session)
            size = 2 * deep_size(manifest) + 4096
            session.charge(size)
            session.reserve_scratch(size)
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(temporary, manifest)
            _publish_owned(session, ((temporary, output),))
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_name(temporary.name + '.tmp').unlink(missing_ok=True)


def inspect_dataset_summary(path: Path, *, deadline: float = float('inf'),
                            preprocessing: PreparationConfig | None = None) -> Json:
    with PreparationSession(preprocessing, deadline=deadline) as session:
        actual = _validated_dataset(path, session)
    return actual


def inspect_dataset(path: Path, *, deadline: float = float('inf'),
                    preprocessing: PreparationConfig | None = None) -> tuple[Json, tuple[Sample, ...], tuple[Sample, ...]]:
    """Compatibility adapter; production summary/index/fit paths do not retain samples."""
    train: list[Sample] = []
    validation: list[Sample] = []
    with PreparationSession(preprocessing, deadline=deadline) as session:
        def retain(decision: AcceptedDecision, bundle: RunBundle) -> None:
            sample = Sample(decision.value, decision.action, decision.target, decision.identity)
            session.retain(sample)
            (train if decision.split == 'train' else validation).append(sample)
        actual = _validated_dataset(path, session, retain)
    return actual, tuple(train), tuple(validation)


def prepare_fixed_inputs(path: Path, encoder: EncoderConfig, *, preprocessing: PreparationConfig | None = None,
                         deadline: float = float('inf'), wall_deadline: float | None = None) -> PreparedInputs:
    session = PreparationSession(preprocessing, deadline=deadline, wall_deadline=wall_deadline)
    return prepare_inputs(path, encoder, session, _validated_dataset)


async def validate_policy(loaded: Any, directory: Path, output: Path, collector: Any) -> Json:
    from gf_common.domain import RuntimePolicyBinding
    from gf_training.collection import CollectionPlan
    started = time.monotonic()
    config = loaded.config
    if config.server.external_url is not None:
        raise ContractError('final validation requires owned distinct-seed Simulator groups')
    if not isinstance(config.model, MLPModelConfig) or not isinstance(config.training, PPOTrainingConfig):
        raise ContractError('MLP validation requires MLP config')
    directory = directory.resolve()
    with run_lock(directory):
        state, records = validate_chain(directory, deadline=started + config.run.execution_seconds)
        from gf_training.ppo_run import integration_identity
        if integration_identity(config, started + config.run.execution_seconds) != state['identity']['integration']:
            raise ContractError('validation Simulator integration differs from frozen parent identity')
        if digest(asdict(config.actions)) != read_json(directory / 'artifacts' / f'{state["updates"]:06}' / 'inference.json')['action_config_hash']:
            raise ContractError('validation action recipe differs from trained policy')
        label, binding = owned_binding(directory, state['updates'])
        groups = validation_groups(state['seed'], label)
        linkage = {'parent_run_id': state['run_id'], 'parent_run_path': str(directory), 'batch_id': 'final-validation',
                   'collection_generation': state['updates'], 'learner_seats': {spec.episode_id: spec.learner_viewer for group in groups for spec in group.episodes}}
        registry = {label: binding, 'random': RuntimePolicyBinding('random'), 'heuristic': RuntimePolicyBinding('heuristic')}
        result = await collector(CollectionPlan(groups, registry, output.resolve(), linkage, started + config.run.execution_seconds,
                                                started + config.run.wall_seconds, 'ppo_validation', started_monotonic=started))
        if result.report.get('comparison_valid') is not True:
            raise ContractError('final validation failed')
        return {'validation_dir': str(result.directory), 'parent_run_id': state['run_id'], 'generation': records[-1]['generation'],
                'games': 64, 'wall_seconds': time.monotonic() - started}


def index_dataset(path: Path, output: Path, *, deadline: float = float('inf'),
                  preprocessing: PreparationConfig | None = None) -> Json:
    """Rebuild into fresh output/sidecar paths; existing indexes are preserved.

    Original sources are scanned once; a bounded spill ledger sorts index rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists(follow_symlinks=False) or sidecar.exists(follow_symlinks=False):
        raise ContractError('index output and sidecar must both be fresh paths')
    session = PreparationSession(preprocessing, deadline=deadline)
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary_sidecar = temporary.with_suffix(temporary.suffix + '.json')
    writer: Any = None
    try:
        frozen = session.read_json(path)
        ledger = IndexLedger(session)
        count = 0
        def accept(decision: AcceptedDecision, bundle: RunBundle) -> None:
            nonlocal count
            manifest = bundle.manifest
            episode = next(item for item in manifest['episodes'] if item['game_id'] == decision.identity[1])
            spec = next(spec for group in manifest['schedule'] for spec in group['episodes']
                        if spec['episode_id'] == episode['episode_id'])
            viewer = decision.identity[2]
            row = {'run_id': manifest['run_id'], 'parent_run_id': manifest['ppo']['parent_run_id'],
                   'batch_id': manifest['ppo']['batch_id'], 'generation': manifest['ppo']['collection_generation'],
                   'split': manifest['split'], 'learner_viewer': viewer, 'opponent': spec['policies'][2 - viewer],
                   'source_manifest_sha256': session.hash_file(bundle.directory / 'manifest.json'),
                   'game_id': decision.identity[1], 'decision': decision.identity[3], 'return': decision.target,
                   'dataset_content_sha256': frozen['content_sha256']}
            # Reserve simultaneous final Parquet/sidecar staging conservatively.
            session.reserve_scratch(4 * len(canonical(row).encode()) + 4096)
            ledger.add(decision.identity, row)
            count += 1
        manifest = _validated_dataset(path, session, accept)
        output.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256(b'[')
        batch: list[Json] = []
        batch_charge = 0
        target = session.config.chunk_target_mib * 1024 * 1024
        def flush() -> None:
            nonlocal writer, batch_charge
            if not batch:
                return
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
            batch.clear()
            session.release(batch_charge)
            batch_charge = 0
        with closing(ledger.sorted_rows()) as sorted_rows:
            for index, encoded in enumerate(sorted_rows):
                session.check()
                if index:
                    hasher.update(b',')
                hasher.update(encoded.encode())
                charge = 128 * len(encoded.encode()) + 4096  # Python decode plus Arrow/copy headroom.
                if batch and batch_charge + charge > target:
                    flush()
                session.charge(charge)
                batch_charge += charge
                batch.append(json.loads(encoded))
        flush()
        hasher.update(b']')
        if writer is None:
            raise ContractError('fixed index requires rows')
        writer.close()
        writer = None
        session.check()
        result = {'schema_version': 1, 'authoritative': False, 'dataset_content_sha256': manifest['content_sha256'],
                  'rows': count, 'row_content_sha256': hasher.hexdigest(), 'parquet_sha256': sha256(temporary, deadline=session.deadline)}
        atomic_json(temporary_sidecar, result)
        _publish_owned(session, ((temporary, output), (temporary_sidecar, sidecar)))
        return result
    except BaseException:
        for close in (lambda: writer.close() if writer is not None else None, session.abort):
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - preserve index failure and release source locks
                session.diagnostics['cleanup_error'] = str(error)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        temporary_sidecar.with_name(temporary_sidecar.name + '.tmp').unlink(missing_ok=True)
