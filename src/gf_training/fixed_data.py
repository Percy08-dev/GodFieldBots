"""Frozen complete successful PPO sources, revalidated from original evidence."""
from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.domain import ContractError, Json, digest
from gf_common.schedule import serialize_schedule
from gf_models.config import MLPModelConfig, PPOTrainingConfig
from gf_models.mlp.arrays import sha256
from gf_training.collection import training_groups, validation_groups
from gf_training.journal import atomic_json
from gf_training.lineage import generation_path, owned_binding, read_json, run_lock, validate_chain
from gf_training.q_data import (
    RunBundle,
    Sample,
    check_deadline,
    labeled_samples,
    load_run,
    provenance,
    validate_bundles,
)

SELECTION = 'all-explicit-learner-accepted-natural-returns-v1'


def _validate_sources(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], *, deadline: float) -> tuple[Json, tuple[Sample, ...], tuple[Sample, ...]]:
    if len(ppo_runs) != 3 or len(validation_runs) != 3 or len(set(ppo_runs + validation_runs)) != 6:
        raise ContractError('fixed dataset requires exactly three distinct ordered PPO and validation parents')
    all_bundles: list[RunBundle] = []
    parents = []
    collections = []
    total_seconds = 0.
    with ExitStack() as locks:
        for directory in sorted(ppo_runs):
            locks.enter_context(run_lock(directory))
        for directory, validation_directory, seed in zip(ppo_runs, validation_runs, (17, 29, 43), strict=True):
            state, records = validate_chain(directory, deadline=deadline)
            if state['seed'] != seed:
                raise ContractError('fixed ordered PPO learning seeds must be 17,29,43')
            registry = read_json(directory / 'artifact-registry.json')
            generations = []
            for number, record in enumerate(records):
                check_deadline(deadline)
                generations.append({'generation': number, 'record_sha256': sha256(generation_path(directory, number)),
                                    'inference_sha256': record['inference_sha256'], 'weights_sha256': record['weights_sha256']})
                if not number:
                    continue
                update = read_json(directory / 'updates' / f'{number:06}.json')
                bundle = load_run(directory / update['collection'], 'train', deadline=deadline)
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
                _, source_proof = validate_bundles([bundle], deadline=deadline)
                if source_proof['source_set_sha256'] != update['source_proof']['source_set_sha256']:
                    raise ContractError('offline source journal/Replay proof changed since PPO update')
                all_bundles.append(bundle)
                collections.append(provenance(bundle, deadline=deadline))
            validation = load_run(validation_directory, 'validation', deadline=deadline)
            link = validation.manifest.get('ppo', {})
            final_label = 'mlp-' + records[-1]['inference_sha256']
            expected_validation = validation_groups(seed, final_label)
            if validation.manifest.get('collection_kind') != 'ppo_validation' or link.get('parent_run_id') != state['run_id'] or Path(link.get('parent_run_path', '')).resolve() != directory or link.get('collection_generation') != state['updates'] or validation.manifest['schedule'] != serialize_schedule(expected_validation, 4):
                raise ContractError('validation requires exactly 32 prescribed final-policy swapped pairs')
            all_bundles.append(validation)
            collections.append(provenance(validation, deadline=deadline))
            total_seconds += state['elapsed_seconds'] + validation.collection_seconds
            parents.append({'directory': str(directory), 'run_id': state['run_id'], 'run_sha256': sha256(directory / 'run.json'),
                            'seed': seed, 'generations': generations, 'accepted_decisions': state['accepted_decisions'],
                            'generation_cost_seconds': state['elapsed_seconds'], 'artifact_registry_sha256': digest(registry)})
        rows, evidence = validate_bundles(all_bundles, deadline=deadline)
        action_hashes = {source['header']['action_config_hash'] for source in evidence['sources']}
        if len(action_hashes) != 1:
            raise ContractError('frozen sources must share identical action conditions')
        learner_seats = {(bundle.manifest['run_id'], spec['episode_id']): spec['learner_viewer']
                         for bundle in all_bundles for group in bundle.manifest['schedule'] for spec in group['episodes']}
        selected = [row for row in rows if row['viewer'] == learner_seats[(row['run_id'], row['episode_id'])]]
        train = labeled_samples(selected, evidence, 'train', deadline=deadline)
        validation_samples = labeled_samples(selected, evidence, 'validation', deadline=deadline)
        if not train or not validation_samples or len(train) != sum(parent['accepted_decisions'] for parent in parents):
            raise ContractError('fixed dataset complete learner decision count mismatch')
        manifest = {'schema_version': 1, 'selection': SELECTION, 'ppo_runs': [str(path) for path in ppo_runs],
                    'validation_runs': [str(path) for path in validation_runs], 'parents': parents, 'collections': collections,
                    'source_set_sha256': evidence['source_set_sha256'], 'train_decisions': len(train),
                    'validation_decisions': len(validation_samples), 'validation_games': 192,
                    'shared_generation_seconds': total_seconds,
                    'shared_generation_cost_key': digest({'parent_run_ids': [parent['run_id'] for parent in parents], 'validation_run_ids': [bundle.manifest['run_id'] for bundle in all_bundles if bundle.manifest['split'] == 'validation']}),
                    'action_config_hash': next(iter(action_hashes)),
                    'label_recipe': 'behavior-selected-natural-return-gamma1-v1',
                    'split_namespaces': {'train': 'ppo-train-v1', 'validation': 'fixed-existing-32', 'test': 'reserved-not-run'}}
        manifest['content_sha256'] = digest(manifest)
        return manifest, train, validation_samples


def build_dataset(ppo_runs: tuple[Path, ...], validation_runs: tuple[Path, ...], output: Path, *, deadline: float = float('inf')) -> Json:
    if output.exists():
        raise ContractError('frozen dataset output already exists')
    manifest, _, _ = _validate_sources(tuple(path.resolve() for path in ppo_runs), tuple(path.resolve() for path in validation_runs), deadline=deadline)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, manifest)
    return manifest


def inspect_dataset(path: Path, *, deadline: float = float('inf')) -> tuple[Json, tuple[Sample, ...], tuple[Sample, ...]]:
    frozen = read_json(path)
    if frozen.get('schema_version') != 1 or frozen.get('selection') != SELECTION:
        raise ContractError('unsupported fixed dataset manifest')
    actual, train, validation = _validate_sources(tuple(Path(item).resolve() for item in frozen['ppo_runs']), tuple(Path(item).resolve() for item in frozen['validation_runs']), deadline=deadline)
    if actual != frozen:
        raise ContractError('frozen dataset original evidence/content changed')
    return actual, train, validation


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


def index_dataset(path: Path, output: Path, *, deadline: float = float('inf')) -> Json:
    """Rebuildable query aid: originals are always validated before writing."""
    manifest, train, validation = inspect_dataset(path, deadline=deadline)
    import pyarrow as pa
    import pyarrow.parquet as pq
    lookup = {}
    for source in manifest['collections']:
        check_deadline(deadline)
        collection = read_json(Path(source['directory']) / 'manifest.json')
        specifications = {spec['episode_id']: spec for group in collection['schedule'] for spec in group['episodes']}
        for episode in collection['episodes']:
            spec = specifications[episode['episode_id']]
            viewer = spec['learner_viewer']
            lookup[(collection['run_id'], episode['game_id'], viewer)] = {
                'run_id': collection['run_id'], 'parent_run_id': collection['ppo']['parent_run_id'],
                'batch_id': collection['ppo']['batch_id'], 'generation': collection['ppo']['collection_generation'],
                'split': collection['split'], 'learner_viewer': viewer, 'opponent': spec['policies'][2 - viewer],
                'source_manifest_sha256': source['manifest_sha256']}
    rows = []
    for sample in sorted((*train, *validation), key=lambda sample: sample.identity):
        check_deadline(deadline)
        row = lookup[sample.identity[:3]]
        rows.append({**row, 'game_id': sample.identity[1], 'decision': sample.identity[3], 'return': sample.target,
                     'dataset_content_sha256': manifest['content_sha256']})
    output.parent.mkdir(parents=True, exist_ok=True)
    check_deadline(deadline)
    pq.write_table(pa.Table.from_pylist(rows), output)
    check_deadline(deadline)
    result = {'schema_version': 1, 'authoritative': False, 'dataset_content_sha256': manifest['content_sha256'],
              'rows': len(rows), 'row_content_sha256': digest(rows), 'parquet_sha256': sha256(output)}
    atomic_json(output.with_suffix(output.suffix + '.json'), result)
    return result
