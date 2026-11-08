"""Session-private legacy PPO originals and fully validated per-test relocations.

This deliberately supports only the current three-seed, one-update test recipe.
It is not a migration tool or an authorization path for production sources.
"""
from __future__ import annotations

import asyncio
import shutil
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gf_cli.config import LoadedConfig
from gf_common.domain import ContractError, digest
from gf_common.validation import catalog
from gf_models.arrays import sha256
from gf_training.fixed_data import build_dataset, validate_policy
from gf_training.journal import atomic_json, project_journals
from gf_training.lineage import read_json, validate_chain
from gf_training.ppo_run import train_ppo
from tests.input_support import fixture, input_factory
from tests.ppo_support import configuration, synthetic_collector

SEEDS = (17, 29, 43)
BINARY = 'synthetic-simulator.identity'


def _sources(root):
    return (tuple(root / f'parent-{seed}' for seed in SEEDS),
            tuple(root / f'validation-{seed}' for seed in SEEDS))


def _files(root):
    """Reject links/special files before following any source content."""
    for tree in (*_sources(root)[0], *_sources(root)[1], root / BINARY):
        for path in (tree, *sorted(tree.rglob('*'))):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ContractError('shared PPO sources require regular files/directories')
            if path.name != '.lock':
                yield path


def _inventory(root):
    return tuple((str(path.relative_to(root)), sha256(path)) for path in _files(root))


def _layout(root):
    parents, validations = _sources(root)
    if (root / BINARY).read_bytes() != b'synthetic collector identity only; not an executable engine':
        raise ContractError('unsupported shared PPO binary identity')
    for seed, parent, validation in zip(SEEDS, parents, validations, strict=True):
        state, records = validate_chain(parent)
        update = read_json(parent / 'updates/000001.json')
        if (state['schema_version'] != 1 or state['seed'] != seed or state['updates'] != 1
                or state['effective_config']['model']['model_id'] != 'mlp'
                or state['effective_config']['server']['binary'] != str(root / BINARY)
                or len(records) != 2 or update['schema_version'] != 1
                or update['collection'] != 'collections/batch-000000'
                or update['numeric'] != 'numeric/batch-000000'
                or set(update['source_proof']) != {'source_set_sha256', 'manifest_sha256', 'decisions', 'games'}):
            raise ContractError('unsupported shared PPO parent layout/version')
        for directory, kind in ((parent / update['collection'], 'ppo_batch'), (validation, 'ppo_validation')):
            manifest = read_json(directory / 'manifest.json')
            if (manifest['schema_version'] != 4 or manifest['collection_kind'] != kind
                    or manifest['ppo']['parent_run_path'] != str(parent)):
                raise ContractError('unsupported shared PPO collection layout/version')


def _bindings(bindings, original, relocated):
    for binding in bindings.values():
        options = binding['policy_options']
        if options is None:
            continue
        checkpoint = Path(options['checkpoint'])
        if checkpoint not in (original / 'artifacts/000000', original / 'artifacts/000001'):
            raise ContractError('unsupported shared PPO checkpoint reference')
        options['checkpoint'] = str(relocated / checkpoint.relative_to(original))


def _reseal(parent, update):
    """Private construction only; normal offline readers validate every result."""
    atomic_json(parent / 'updates/000001.json', update)
    generation = read_json(parent / 'generations/000001.json')
    generation['update_sha256'] = sha256(parent / 'updates/000001.json')
    atomic_json(parent / 'generations/000001.json', generation)
    final_hash = sha256(parent / 'generations/000001.json')
    atomic_json(parent / 'latest.json', {'generation': 1, 'sha256': final_hash})
    state = read_json(parent / 'run.json')
    state['generation_sha256'] = final_hash
    atomic_json(parent / 'run.json', state)


@dataclass(frozen=True)
class SharedPPOSources:
    root: Path
    inventory: tuple[tuple[str, str], ...]

    @classmethod
    def admit(cls, root):
        root = root.resolve()
        tuple(_files(root))  # Reject unsafe files before any validator opens them.
        _layout(root)
        # Crucially validate untouched original source proofs BEFORE any resealing.
        build_dataset(*_sources(root), root / 'original-dataset.json')
        return cls(root, _inventory(root))

    def clone(self, destination):
        if _inventory(self.root) != self.inventory:
            raise ContractError('shared PPO template inventory changed')
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=False)
        for source in (*_sources(self.root)[0], *_sources(self.root)[1]):
            shutil.copytree(source, destination / source.name, ignore=shutil.ignore_patterns('.lock'))
        shutil.copyfile(self.root / BINARY, destination / BINARY)
        parents, validations = _sources(destination)
        for original, parent, validation in zip(_sources(self.root)[0], parents, validations, strict=True):
            state = read_json(parent / 'run.json')
            state['effective_config']['server']['binary'] = str(destination / BINARY)
            state['config_sha256'] = digest(state['effective_config'])
            atomic_json(parent / 'run.json', state)
            collection = parent / 'collections/batch-000000'
            for directory in (collection, validation):
                manifest = read_json(directory / 'manifest.json')
                manifest['ppo']['parent_run_path'] = str(parent)
                _bindings(manifest['policy_bindings'], original, parent)
                atomic_json(directory / 'manifest.json', manifest)
            update = read_json(parent / 'updates/000001.json')
            _bindings(update['policy_bindings'], original, parent)
            manifest_hash = sha256(collection / 'manifest.json')
            update['collection_manifest_sha256'] = manifest_hash
            update['source_proof']['manifest_sha256'] = manifest_hash
            _reseal(parent, update)
            # Stage one establishes a coherent chain so ordinary offline projection
            # can verify the copied originals without any fabricated online owner.
            _, evidence = project_journals(sorted(collection.glob('*/player-*.jsonl')))
            update['source_proof']['source_set_sha256'] = evidence['source_set_sha256']
            _reseal(parent, update)
        build_dataset(parents, validations, destination / 'clone-dataset.json')
        if _inventory(self.root) != self.inventory:
            raise ContractError('shared PPO template changed while copying')
        return parents, validations


def generate_sources(root):
    """Keep the exact legacy generation workload, including numerical learning."""
    make_input = input_factory(catalog(fixture('catalog_public_v2.json')))
    collector = synthetic_collector(make_input, both_viewers=True, outcomes={'win': {'winner': 1}})
    parents, validations = _sources(root)
    for seed, parent, validation in zip(SEEDS, parents, validations, strict=True):
        loaded = configuration(root, max_updates=1, seed=seed)
        asyncio.run(train_ppo(loaded, collector, parent))
        config = replace(loaded.config, run=replace(loaded.config.run, split='validation'))
        validation_loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
        asyncio.run(validate_policy(validation_loaded, parent, validation, collector))
    return SharedPPOSources.admit(root)
