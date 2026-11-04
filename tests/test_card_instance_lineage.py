"""Bounded scalar/array lineage fixtures; never qualified original Sources."""
import json
import time

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_models.arrays import sha256
from gf_models.sharded_arrays import write_bundle
from gf_models.structured_artifacts import semantics_v3
from gf_models.structured_sealed import SealedScan, inspect_pruned_game_inference_v3
from gf_training.card_instance_source import _binding, validate_game_parent
from gf_training.journal import atomic_json
from gf_training.lineage import checkpoint_path, generation_path, run_lock
from gf_training.structured_lineage import retain_game_generations, validate_game_chain
from tests.card_instance_learning_support import initial_parent
from tests.test_card_instance_learning import ACTIONS, LAYOUT, RULES


def append_structural(directory, learner, state, records, stop):
    """Write complete structural records without optimizer or Simulator execution."""
    classification = {key: state[key] for key in ('action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature')}
    rng = np.random.Generator(np.random.PCG64(17))
    for number in range(state['updates'] + 1, stop + 1):
        learner.steps = 4 * number  # Synthetic scalar boundary, not a training claim.
        saved = learner.save(checkpoint_path(directory, number))
        previous = records[-1]
        label, _ = _binding(directory, state, previous, number - 1)
        batch = f'batch-{number - 1:06}'
        numeric = directory / 'numeric' / batch
        numeric.mkdir()
        data = write_bundle(numeric, 'rollout', {'structural-fixture': np.array([0], np.int64)})
        episodes = [{**classification, 'actor': 1 + index % 2, 'episode': f'{batch}-{index}',
            'end_kind': 'natural', 'outcome': 'draw', 'reason': None, 'has_targets': True,
            'frames': [{**classification}], 'selections': [{}]} for index in range(16)]
        manifest = {'schema_version': 4, 'kind': 'structured-rollout', **classification,
            **semantics_v3(learner.model, layout=LAYOUT, action_contract=ACTIONS.action_contract, rules=RULES),
            'architecture_identity': previous and json.loads((checkpoint_path(directory, number) / 'inference.json').read_text())['architecture_identity'],
            'action_config_hash': learner.action_config_hash, 'policy_version': label, 'decision_count': 16,
            'episodes': episodes, 'data': data}
        atomic_json(numeric / 'rollout.json', manifest)
        numeric_hash = sha256(numeric / 'rollout.json')
        collection = directory / 'collections' / batch
        collection.mkdir()
        atomic_json(collection / 'manifest.json', {'schema_version': 5, 'journal_schema_version': 6, **classification})
        collection_hash = sha256(collection / 'manifest.json')
        update = {'schema_version': 3, 'status': 'complete', 'run_id': state['run_id'], 'generation': number,
            'parent_sha256': state['generation_sha256'], 'batch_id': batch, 'collection': str(collection.relative_to(directory)),
            'collection_manifest_sha256': collection_hash, 'numeric': str(numeric.relative_to(directory)), 'numeric_sha256': numeric_hash,
            'accepted_decisions': 16, 'source_proof': {'schema_version': 1, 'kind': 'card-instance-original-source-proof',
                'qualified': False, 'parent_qualification': 'not_performed', 'games': 16, 'manifest_sha256': collection_hash,
                'source_set_sha256': 'a' * 64, 'accepted_decisions': 16, **classification},
            'metrics': {'optimizer_steps': 4 * number}, 'planned_games': 16, 'actual_games': 16, 'health': {},
            'policy_bindings': {}, 'learner_seats': {f'episode-{index}': 1 + index % 2 for index in range(16)},
            'collection_seconds': 1., 'update_seconds': 1., 'model_id': 'mlp', 'model_version': '3',
            'input_recipe': state['input_recipe'], 'numeric_files': {'rollout.json': numeric_hash, **data['files']}, **classification}
        update_path = directory / 'updates' / f'{number:06}.json'
        atomic_json(update_path, update)
        record = {'schema_version': 3, 'run_id': state['run_id'], 'generation': number,
            'parent_sha256': state['generation_sha256'], 'model_id': 'mlp', 'model_version': '3', 'input_recipe': state['input_recipe'],
            'inference_sha256': saved['inference_digest'], 'resume_sha256': saved['resume_digest'], 'artifact_files': saved['files'],
            'accepted_decisions': number * 16, 'optimizer_steps': number * 4, 'update_sha256': sha256(update_path),
            'parent_controls': {'schedule_opponent_rng': rng.bit_generator.state,
                'snapshot_pool': [n for n in range(number + 1) if n % 10 == 0][-5:], 'collection_position': number}, **classification}
        atomic_json(generation_path(directory, number), record)
        registry = json.loads((directory / 'artifact-registry.json').read_text())
        registry['generations'][str(number)] = {'status': 'retained', 'inference_sha256': saved['inference_digest'],
            'original_files': saved['files'], 'deletion_intent': None, 'deletion_receipt': None}
        atomic_json(directory / 'artifact-registry.json', registry)
        state.update(updates=number, generation_sha256=sha256(generation_path(directory, number)),
                     accepted_decisions=number * 16, optimizer_steps=number * 4)
        atomic_json(directory / 'latest.json', {'generation': number, 'sha256': state['generation_sha256']})
        atomic_json(directory / 'run.json', state)
        records.append(record)


def test_registry3_complete_chain_crosses10_and50_60_pool_eviction(tmp_path):
    directory = tmp_path / 'parent'
    deadline = time.monotonic() + 180
    with run_lock(directory, create=True):
        learner, state, records, _ = initial_parent(directory, updates=60)
        append_structural(directory, learner, state, records, 2)
        retain_game_generations(directory, 2, max_owned_bytes=1024**3, deadline=deadline)
        append_structural(directory, learner, state, records, 11)
        retain_game_generations(directory, 11, max_owned_bytes=1024**3, deadline=deadline)
        append_structural(directory, learner, state, records, 50)
        retain_game_generations(directory, 50, max_owned_bytes=1024**3, deadline=deadline)
        assert records[-1]['parent_controls']['snapshot_pool'] == [10, 20, 30, 40, 50]
        append_structural(directory, learner, state, records, 60)
        retain_game_generations(directory, 60, max_owned_bytes=1024**3, deadline=deadline)
        observed, chain = validate_game_chain(directory, require_complete=False, max_owned_bytes=1024**3, deadline=deadline)
        assert observed['updates'] == 60 and len(chain) == 61
        assert chain[-1]['parent_controls']['snapshot_pool'] == [20, 30, 40, 50, 60]
        registry = json.loads((directory / 'artifact-registry.json').read_text())
        for number in (0, 10, 20, 30, 40, 50, 60):
            assert registry['generations'][str(number)]['status'] == 'retained'
            assert (checkpoint_path(directory, number) / 'resume.json').is_file()
        for number in (1, 49, 59):
            receipt = registry['generations'][str(number)]
            assert receipt['status'] == 'deleted'
            artifact = inspect_pruned_game_inference_v3(checkpoint_path(directory, number),
                expected_inference_digest=chain[number]['inference_sha256'], expected_resume_digest=chain[number]['resume_sha256'],
                original_files=chain[number]['artifact_files'], deletion_receipt=receipt, actions=ACTIONS, rules=RULES,
                scan=SealedScan(max_owned_bytes=1024**3, deadline=deadline))
            assert artifact.resume is None and not artifact.retained
        with pytest.raises(ContractError):
            validate_game_parent(directory, require_complete=True, max_owned_bytes=1024**3, deadline=deadline)


@pytest.mark.parametrize('mutation', ['missing', 'foreign', 'replaced'])
def test_game_retained_inventory_refusal(tmp_path, mutation):
    directory = tmp_path / 'parent'
    with run_lock(directory, create=True):
        learner, state, records, _ = initial_parent(directory, updates=2)
        append_structural(directory, learner, state, records, 2)
        path = checkpoint_path(directory, 1) / 'resume.json'
        if mutation == 'missing':
            path.unlink()
        elif mutation == 'foreign':
            (path.parent / 'foreign.json').write_text('{}')
        else:
            path.write_text('{}')
        with pytest.raises((ContractError, FileNotFoundError)):
            retain_game_generations(directory, 2, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
        assert json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']['status'] == 'retained'


def test_game_interrupted_pruning_does_not_authorize_historical_receipt(tmp_path, monkeypatch):
    import gf_training.structured_lineage as module
    directory = tmp_path / 'parent'
    with run_lock(directory, create=True):
        learner, state, records, _ = initial_parent(directory, updates=2)
        append_structural(directory, learner, state, records, 2)
        original = module.os.unlink
        def interrupted(path, *args, **kwargs):
            if kwargs.get('dir_fd') is not None:
                raise OSError('fixture interrupted private prune')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(module.os, 'unlink', interrupted)
        with pytest.raises(OSError, match='interrupted'):
            retain_game_generations(directory, 2, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
        registry = json.loads((directory / 'artifact-registry.json').read_text())
        assert registry['generations']['1']['status'] == 'deleting'
        with pytest.raises(ContractError):
            validate_game_chain(directory, require_complete=False, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
