"""Sealed historical/Generation2 structural components; no qualification claim."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError
from gf_models.arrays import atomic_json, sha256
from gf_models.sharded_arrays import validate_bundle_metadata
from gf_models.structured_sealed import (
    SealedScan,
    inspect_pruned_inference,
    inspect_retained_artifact,
)
from tests.test_structured_learning import learner_for


def inspect(path, *, limit=256 * 1024**2):
    scan = SealedScan(max_owned_bytes=limit, deadline=float('inf'))
    result = inspect_retained_artifact(path, expected_inference_digest=sha256(path / 'inference.json'),
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False, scan=scan)
    return result, scan


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_real_c1_sharded_artifact_inspection_without_active_runtime_comparison(tmp_path, kind):
    learner = learner_for(kind, synthetic_capacity=False)
    identity = learner.save(tmp_path / 'artifact', shard_bytes=400000)
    result, scan = inspect(tmp_path / 'artifact')
    assert result.files == identity['files'] and result.retained
    assert result.resume['optimizer_steps'] == 0
    assert all(state == {'present': False} for state in result.resume['parameter_states'].values())
    assert scan.peak <= scan.limit


def test_pruned_inference_checks_exact_original_map_and_receipt_without_rebuilding_arrays(tmp_path):
    learner = learner_for(synthetic_capacity=False)
    original = learner.save(tmp_path / 'artifact', shard_bytes=400000)
    deleted = {name: sha for name, sha in original['files'].items() if name != 'inference.json'}
    for name in deleted:
        (tmp_path / 'artifact' / name).unlink()
    receipt = {'status': 'deleted', 'inference_sha256': original['inference_digest'], 'original_files': original['files'],
               'deletion_intent': {'files': deleted, 'staging_basename': 'prune-' + 'a' * 32}, 'deletion_receipt': {'files': deleted}}
    result = inspect_pruned_inference(tmp_path / 'artifact', expected_inference_digest=original['inference_digest'],
        expected_resume_digest=original['resume_digest'], original_files=original['files'], deletion_receipt=receipt,
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False,
        scan=SealedScan(max_owned_bytes=256 * 1024**2, deadline=float('inf')))
    assert not result.retained and result.resume is None
    assert list((tmp_path / 'artifact').iterdir()) == [tmp_path / 'artifact/inference.json']


def test_cumulative_joint_budget_rejects_before_archive_allocation(tmp_path, monkeypatch):
    import gf_models.structured_sealed as module
    learner = learner_for(synthetic_capacity=False)
    learner.save(tmp_path / 'artifact')
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted archive allocation')
    monkeypatch.setattr(module.zipfile, 'ZipFile', forbidden)
    with pytest.raises(ContractError, match='reservation'):
        inspect(tmp_path / 'artifact', limit=65536)


def test_no_torch_or_installed_runtime_metadata_is_needed_for_sealed_inspection(tmp_path):
    learner = learner_for(synthetic_capacity=False)
    learner.save(tmp_path / 'artifact')
    for name in ('inference.json', 'resume.json'):
        path = tmp_path / 'artifact' / name
        value = json.loads(path.read_text())
        value['runtime'].update(torch='historical-build', python='historical-python')
        value['producer']['source_digest'] = '1' * 64
        if name == 'resume.json':
            value['inference_digest'] = sha256(tmp_path / 'artifact/inference.json')
        atomic_json(path, value)
    code = '''
import importlib.abc,importlib.metadata,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name=='torch' or name.startswith('torch.'): raise RuntimeError('imported Torch')
sys.meta_path.insert(0,NoTorch())
def forbidden(*args,**kwargs): raise RuntimeError('queried installed runtime')
importlib.metadata.version=forbidden
from pathlib import Path
from gf_common.action_contract import engineering_contract
from gf_models.arrays import sha256
from gf_models.structured_sealed import SealedScan,inspect_retained_artifact
root=Path(sys.argv[1])
result=inspect_retained_artifact(root,expected_inference_digest=sha256(root/'inference.json'),action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,scan=SealedScan(max_owned_bytes=256*1024**2,deadline=float('inf')))
assert result.retained and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'artifact')], capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr


def test_pure_bundle_metadata_validation_performs_no_file_reads(tmp_path, monkeypatch):
    learner = learner_for(synthetic_capacity=False)
    learner.save(tmp_path / 'artifact')
    value = json.loads((tmp_path / 'artifact/inference.json').read_text())['weights']
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('pure metadata validator read a file'))
    descriptors = validate_bundle_metadata(value)
    assert descriptors and sum(item['bytes'] for item in descriptors.values()) == value['decoded_bytes']


def test_sealed_partial_adam_presence_and_recorded_step_ownership(tmp_path, make_input):
    from tests.test_structured_learning import rollout_for
    learner = learner_for(hand='direct', synthetic_capacity=False)
    learner.update(rollout_for(make_input, learner, count=1, empty=True, events=False))
    learner.save(tmp_path / 'partial')
    result, _ = inspect(tmp_path / 'partial', limit=512 * 1024**2)
    presence = result.resume['parameter_states']
    assert any(state['present'] for state in presence.values()) and any(not state['present'] for state in presence.values())
    assert all(not state['present'] for key, state in presence.items() if key.startswith('event_encoder'))


def test_scanner_joint_retained_metadata_budget_is_not_per_document(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    atomic_json(first, {'payload': 'x' * 20000})
    atomic_json(second, {'payload': 'y' * 20000})
    scan = SealedScan(max_owned_bytes=900000, deadline=float('inf'))
    scan.object(first)
    with pytest.raises(ContractError, match='joint metadata'):
        scan.object(second)


@pytest.mark.parametrize('mutation', ['extra', 'symlink', 'paired_link', 'paired_model', 'missing_profile'])
def test_sealed_rejects_foreign_inventory_and_rehashed_paired_corruption(tmp_path, mutation):
    learner = learner_for(synthetic_capacity=False)
    learner.save(tmp_path / 'artifact')
    if mutation == 'extra':
        (tmp_path / 'artifact/foreign.json').write_text('{}')
    elif mutation == 'symlink':
        path = next((tmp_path / 'artifact').glob('weights-*.npz'))
        outside = tmp_path / 'outside'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        path = tmp_path / 'artifact/resume.json'
        value = json.loads(path.read_text())
        if mutation == 'paired_link':
            value['inference_digest'] = '0' * 64
        elif mutation == 'paired_model':
            value['model']['hand']['pooling'] = 'max'
        else:
            del value['action_contract']
        atomic_json(path, value)
    with pytest.raises((ContractError, OSError)):
        inspect(tmp_path / 'artifact')


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_generation2_real_c1_chain_and_owned_binding(tmp_path, make_input, kind):
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import owned_component_binding, validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, expected = make_chain(tmp_path, make_input, kind=kind, updates=1)
    state, records = validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert state['accepted_decisions'] == 16 and records[-1]['optimizer_steps'] == 4
    assert state['config_sha256'] == expected['config_sha256']
    with pytest.raises(ContractError, match='ownership'):
        owned_component_binding(directory, 1, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    with run_lock(directory):
        label, binding = owned_component_binding(directory, 1, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert label.startswith(kind + '-ppo-v2-') and binding.policy_options['synthetic_capacity'] is False


def test_generation2_retention_preserves_maps_and_rejects_pruned_active_binding(tmp_path, make_input):
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import (
        owned_component_binding,
        retain_component_generations,
        validate_component_chain,
    )
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    with run_lock(directory):
        retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
        state, records = validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
        assert state['updates'] == 2 and records[1]['artifact_files']['resume.json']
        assert {p.name for p in (directory / 'artifacts/000001').iterdir()} == {'inference.json'}
        with pytest.raises(ContractError):
            owned_component_binding(directory, 1, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
        owned_component_binding(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    with pytest.raises(ContractError):
        validate_component_chain(directory, require_retained=True, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


@pytest.mark.parametrize('mutation', ['counter', 'clock', 'health', 'hardware', 'classification', 'resume_identity', 'schema'])
def test_generation2_rejects_changed_run_metadata(tmp_path, make_input, mutation):
    from gf_training.structured_lineage import validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, state = make_chain(tmp_path, make_input, updates=1)
    if mutation == 'counter':
        state['accepted_decisions'] += 1
    elif mutation == 'clock':
        state['execution_deadline_unix'] += 1
    elif mutation == 'health':
        state['invocations'][0]['child_health']['exit_code'] = 1
    elif mutation == 'hardware':
        state['hardware']['device_count'] = 1
    elif mutation == 'classification':
        state['synthetic_capacity'] = True
    elif mutation == 'resume_identity':
        state['resume_identity'] = '0' * 64
    else:
        state['schema_version'] = True
    atomic_json(directory / 'run.json', state)
    with pytest.raises(ContractError):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_generation2_partial_pruning_intent_is_not_completed_or_recoverable_binding(tmp_path, make_input, monkeypatch):
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import (
        retain_component_generations,
        validate_component_chain,
    )
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    import os
    original = os.unlink
    calls = []
    def failed_second(path, *args, **kwargs):
        if kwargs.get('dir_fd') is not None:
            calls.append(path)
            if len(calls) == 2:
                raise OSError('injected bounded deletion failure')
        return original(path, *args, **kwargs)
    with run_lock(directory), monkeypatch.context() as patch:
        patch.setattr(os, 'unlink', failed_second)
        with pytest.raises(OSError, match='deletion failure'):
            retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    registry = json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']
    assert registry['status'] == 'deleting' and registry['deletion_receipt'] is None
    assert registry['deletion_intent']['files'] and (directory / 'artifacts/000001/inference.json').is_file()
    with pytest.raises(ContractError):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_retention_refuses_foreign_file_before_deletion_intent(tmp_path, make_input):
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import retain_component_generations
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    before = (directory / 'artifact-registry.json').read_bytes()
    (directory / 'artifacts/000001/foreign.txt').write_text('foreign evidence')
    with run_lock(directory), pytest.raises(ContractError, match='inventory'):
        retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (directory / 'artifact-registry.json').read_bytes() == before
    assert (directory / 'artifacts/000001/foreign.txt').read_text() == 'foreign evidence'


def test_component_chain_historical_validation_has_no_torch_version_or_hardware_query(tmp_path, make_input):
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    code = '''
import sys,importlib.abc,importlib.metadata
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name=='torch' or name.startswith('torch.'): raise RuntimeError('imported Torch')
sys.meta_path.insert(0,NoTorch())
def forbidden(*args,**kwargs):raise RuntimeError('current runtime query')
importlib.metadata.version=forbidden
from pathlib import Path
from gf_training.structured_lineage import validate_component_chain
state,records=validate_component_chain(Path(sys.argv[1]),max_owned_bytes=512*1024**2,deadline=float('inf'))
assert state['purpose']=='component' and state['action_contract']['eligibility']=='engineering_only'
assert len(records)==2 and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(directory)], capture_output=True, text=True, check=False, timeout=20)
    assert result.returncode == 0, result.stderr


def test_generation2_broken_immutable_controls_and_numeric_scalar_links_reject(tmp_path, make_input):
    from gf_training.structured_lineage import _controls, validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    path = directory / 'generations/000001.json'
    record = json.loads(path.read_text())
    with pytest.raises(ContractError):
        _controls({**record['parent_controls'], 'snapshot_pool': [False]}, 1)
    record['parent_controls']['collection_position'] = 0
    atomic_json(path, record)
    with pytest.raises(ContractError):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))




def test_resume_inference_reference_digest_is_required(tmp_path):
    learner = learner_for(synthetic_capacity=False)
    learner.save(tmp_path / 'artifact')
    path = tmp_path / 'artifact/resume.json'
    value = json.loads(path.read_text())
    value['inference_digest'] = None
    atomic_json(path, value)
    with pytest.raises(ContractError, match='SHA256'):
        inspect(tmp_path / 'artifact')


def _reseal_update(directory, update, *, accepted=None):
    path = directory / 'updates/000001.json'
    atomic_json(path, update)
    generation_path = directory / 'generations/000001.json'
    record = json.loads(generation_path.read_text())
    record['update_sha256'] = sha256(path)
    if accepted is not None:
        record['accepted_decisions'] = accepted
    atomic_json(generation_path, record)
    checksum = sha256(generation_path)
    atomic_json(directory / 'latest.json', {'generation': 1, 'sha256': checksum})
    state = json.loads((directory / 'run.json').read_text())
    state['generation_sha256'] = checksum
    if accepted is not None:
        state['accepted_decisions'] = accepted
    atomic_json(directory / 'run.json', state)


@pytest.mark.parametrize('mutation', ['both_view_count', 'numeric_files', 'qualified_source'])
def test_consistently_rehashed_update_scalar_or_map_corruption_rejects(tmp_path, make_input, mutation):
    from gf_training.structured_lineage import validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    update = json.loads((directory / 'updates/000001.json').read_text())
    accepted = None
    if mutation == 'both_view_count':
        accepted = update['source_proof']['accepted_decisions']
        update['accepted_decisions'] = accepted
    elif mutation == 'numeric_files':
        del update['numeric_files'][next(name for name in update['numeric_files'] if name.endswith('.npz'))]
    else:
        update['source_proof']['qualified'] = True
    _reseal_update(directory, update, accepted=accepted)
    with pytest.raises(ContractError):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_partial_commit_never_becomes_a_healthy_latest_generation(tmp_path, make_input, monkeypatch):
    import gf_training.structured_lineage as module
    from tests.structured_lineage_support import make_chain
    original = module._write
    def failed_latest(path, value, **kwargs):
        if path.name == 'latest.json' and value['generation'] == 1:
            raise OSError('injected latest publication failure')
        return original(path, value, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_write', failed_latest)
        with pytest.raises(OSError, match='publication failure'):
            make_chain(tmp_path, make_input, updates=1)
    assert json.loads((tmp_path / 'parent/latest.json').read_text())['generation'] == 0
    assert (tmp_path / 'parent/generations/000001.json').is_file()
    with pytest.raises(ContractError):
        module.validate_component_chain(tmp_path / 'parent', require_complete=False,
            max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_retention_cadence_and_last_five_controls_do_not_delete_protected_snapshot(tmp_path):
    import numpy as np

    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import _controls, retain_component_generations
    directory = tmp_path / 'cadence'
    with run_lock(directory, create=True):
        saved = directory / 'artifacts/000010'
        saved.mkdir(parents=True)
        (saved / 'inference.json').write_bytes(b'protected snapshot identity')
        retain_component_generations(directory, 11, max_owned_bytes=1000000, deadline=float('inf'))
        assert (saved / 'inference.json').read_bytes() == b'protected snapshot identity'
    rng = np.random.Generator(np.random.PCG64(17))
    _controls({'schedule_opponent_rng': rng.bit_generator.state, 'snapshot_pool': [10, 20, 30, 40, 50], 'collection_position': 50}, 50)


@pytest.mark.parametrize('required', ['collection', 'numeric', 'update_link'])
def test_rehashed_required_none_hashes_cannot_disable_verification(tmp_path, make_input, required):
    from gf_training.structured_lineage import validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    update = json.loads((directory / 'updates/000001.json').read_text())
    if required == 'collection':
        update['collection_manifest_sha256'] = None
        update['source_proof']['manifest_sha256'] = None
        _reseal_update(directory, update)
    elif required == 'numeric':
        update['numeric_sha256'] = None
        update['numeric_files']['rollout.json'] = None
        _reseal_update(directory, update)
    else:
        path = directory / 'generations/000001.json'
        record = json.loads(path.read_text())
        record['update_sha256'] = None
        atomic_json(path, record)
        state = json.loads((directory / 'run.json').read_text())
        state['generation_sha256'] = sha256(path)
        atomic_json(directory / 'latest.json', {'generation': 1, 'sha256': state['generation_sha256']})
        atomic_json(directory / 'run.json', state)
    with pytest.raises(ContractError, match='SHA256'):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def _reseal_artifact_generation(directory, number, mutation):
    path = directory / 'artifacts' / f'{number:06}'
    inference = json.loads((path / 'inference.json').read_text())
    resume = json.loads((path / 'resume.json').read_text())
    if mutation == 'resources':
        inference['model']['resources']['workspace_bytes'] += 1
        resume['model']['resources']['workspace_bytes'] += 1
        atomic_json(path / 'inference.json', inference)
        resume['inference_digest'] = sha256(path / 'inference.json')
    else:
        resume['binding']['candidate_chunk'] += 1
    atomic_json(path / 'resume.json', resume)
    return {'inference.json': sha256(path / 'inference.json'), 'resume.json': sha256(path / 'resume.json')}


@pytest.mark.parametrize('mutation', ['resources', 'binding'])
def test_retained_rehashed_model_resources_and_numeric_binding_must_equal_frozen_run(tmp_path, make_input, mutation):
    from gf_training.structured_lineage import validate_component_chain
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    changed = _reseal_artifact_generation(directory, 1, mutation)
    gen_path = directory / 'generations/000001.json'
    record = json.loads(gen_path.read_text())
    record['artifact_files'].update(changed)
    record.update(inference_sha256=changed['inference.json'], resume_sha256=changed['resume.json'])
    atomic_json(gen_path, record)
    registry = json.loads((directory / 'artifact-registry.json').read_text())
    registry['generations']['1']['original_files'].update(changed)
    registry['generations']['1']['inference_sha256'] = changed['inference.json']
    atomic_json(directory / 'artifact-registry.json', registry)
    state = json.loads((directory / 'run.json').read_text())
    state['generation_sha256'] = sha256(gen_path)
    atomic_json(directory / 'run.json', state)
    atomic_json(directory / 'latest.json', {'generation': 1, 'sha256': state['generation_sha256']})
    with pytest.raises(ContractError, match='model/resources/learner binding'):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_owned_binding_keeps_original_parent_snapshots_through_final_return(tmp_path, make_input, monkeypatch):
    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input, updates=1)
    original = module.policy_label
    calls = []
    def late(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:  # first is numeric-link validation, second is binding publication.
            state = json.loads((directory / 'run.json').read_text())
            state['hardware']['name'] = 'changed after chain validation'
            atomic_json(directory / 'run.json', state)
        return value
    monkeypatch.setattr(module, 'policy_label', late)
    with run_lock(directory), pytest.raises(ContractError, match='snapshot|changed|hash mismatch'):
        module.owned_component_binding(directory, 1, max_owned_bytes=512 * 1024**2, deadline=float('inf'))


def test_gen0_publication_parent_symlink_never_writes_outside(tmp_path, make_input, monkeypatch):
    import gf_training.structured_lineage as module
    from tests.structured_lineage_support import make_chain
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'evidence').write_text('keep')
    original = module._write
    def redirect(path, value, **kwargs):
        if path.parent.name == 'generations':
            path.parent.rmdir()
            path.parent.symlink_to(outside, target_is_directory=True)
        return original(path, value, **kwargs)
    monkeypatch.setattr(module, '_write', redirect)
    with pytest.raises(ContractError, match='nonsymlink'):
        make_chain(tmp_path, make_input, updates=0)
    assert {path.name for path in outside.iterdir()} == {'evidence'}
    assert (outside / 'evidence').read_text() == 'keep'


@pytest.mark.parametrize('replacement', ['different', 'identical', 'symlink', 'directory'])
def test_pruning_captures_public_replacements_without_deleting_them(tmp_path, make_input, monkeypatch, replacement):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    source = directory / 'artifacts/000001'
    original = os.rename
    seen = []
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside survives')
    def replace_before_rename(src, dst, **kwargs):
        if not seen and 'src_dir_fd' in kwargs:
            seen.append(src)
            path = source / src
            old = path.read_bytes()
            path.unlink()
            if replacement == 'symlink':
                path.symlink_to(outside)
            elif replacement == 'directory':
                path.mkdir()
                (path / 'foreign').write_bytes(b'foreign directory')
            else:
                path.write_bytes(old if replacement == 'identical' else b'foreign replacement')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(os, 'rename', replace_before_rename)
    with run_lock(directory), pytest.raises(ContractError, match='differs from verified original'):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    entry = json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']
    assert entry['status'] == 'deleting' and entry['deletion_receipt'] is None
    captured = directory / 'artifacts' / entry['deletion_intent']['staging_basename'] / seen[0]
    assert captured.exists() or captured.is_symlink()
    if replacement == 'symlink':
        assert captured.is_symlink() and outside.read_bytes() == b'outside survives'
    elif replacement == 'directory':
        assert (captured / 'foreign').read_bytes() == b'foreign directory'
    elif replacement == 'different':
        assert captured.read_bytes() == b'foreign replacement'


def test_pruning_only_unlinks_private_names_and_preserves_new_public_replacement(tmp_path, make_input, monkeypatch):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    original, inserted = os.unlink, []
    def public_replacement(name, **kwargs):
        assert kwargs.get('dir_fd') is not None
        if not inserted:
            inserted.append(name)
            (directory / 'artifacts/000001' / name).write_bytes(b'public replacement survives')
        return original(name, **kwargs)
    monkeypatch.setattr(os, 'unlink', public_replacement)
    with run_lock(directory), pytest.raises(ContractError, match='completion inventory'):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (directory / 'artifacts/000001' / inserted[0]).read_bytes() == b'public replacement survives'
    entry = json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']
    assert entry['status'] == 'deleting' and entry['deletion_receipt'] is None


@pytest.mark.parametrize('after', ['rename', 'one_deletion'])
def test_pruning_interruption_preserves_incomplete_intent_and_staging(tmp_path, make_input, monkeypatch, after):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    seen = []
    original = os.rename if after == 'rename' else os.unlink
    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        if not seen:
            seen.append(args[0])
            raise KeyboardInterrupt('bounded staged interruption')
        return result
    monkeypatch.setattr(os, 'rename' if after == 'rename' else 'unlink', interrupted)
    with run_lock(directory), pytest.raises(KeyboardInterrupt):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    entry = json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']
    staging = directory / 'artifacts' / entry['deletion_intent']['staging_basename']
    assert entry['status'] == 'deleting' and entry['deletion_receipt'] is None
    assert staging.is_dir() and (staging / 'failure.json').is_file()
    if after == 'rename':
        assert (staging / seen[0]).is_file()
    else:
        assert not (directory / 'artifacts/000001' / seen[0]).exists()


def test_pruning_stage_collision_preserves_foreign_directory_and_registry(tmp_path, make_input, monkeypatch):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    original = os.mkdir
    seen = []
    before = (directory / 'artifact-registry.json').read_bytes()
    def collision(name, mode=0o777, **kwargs):
        if kwargs.get('dir_fd') is not None and not seen:
            seen.append(name)
            original(name, mode, **kwargs)
            (directory / 'artifacts' / name / 'foreign').write_bytes(b'preserved')
        return original(name, mode, **kwargs)
    monkeypatch.setattr(os, 'mkdir', collision)
    with run_lock(directory), pytest.raises(FileExistsError):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (directory / 'artifact-registry.json').read_bytes() == before
    assert (directory / 'artifacts' / seen[0] / 'foreign').read_bytes() == b'preserved'


def test_pruning_stage_substitution_never_deletes_foreign_namespace(tmp_path, make_input, monkeypatch):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    original = os.rename
    replaced = []
    def substitute(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if not replaced and 'src_dir_fd' in kwargs:
            entry = json.loads((directory / 'artifact-registry.json').read_text())['generations']['1']
            stage = directory / 'artifacts' / entry['deletion_intent']['staging_basename']
            original(stage, directory / 'artifacts/captured-owned-stage')
            stage.mkdir()
            (stage / 'foreign').write_bytes(b'foreign namespace survives')
            replaced.append(stage)
        return result
    monkeypatch.setattr(os, 'rename', substitute)
    with run_lock(directory), pytest.raises(ContractError, match='directory entry was replaced'):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (replaced[0] / 'foreign').read_bytes() == b'foreign namespace survives'
    assert any(path.name != 'failure.json' for path in (directory / 'artifacts/captured-owned-stage').iterdir())


@pytest.mark.parametrize('mutation', ['resources', 'binding'])
def test_commit_rejects_rehashed_frozen_declaration_mismatch_before_controls(tmp_path, make_input, monkeypatch, mutation):
    from gf_training.structured_numeric import Learner
    from tests.structured_lineage_support import make_chain
    original = Learner.save
    def changed(self, path, **kwargs):
        saved = original(self, path, **kwargs)
        saved['files'].update(_reseal_artifact_generation(path.parent.parent, int(path.name), mutation))
        return saved
    monkeypatch.setattr(Learner, 'save', changed)
    with pytest.raises(ContractError, match='model/resources/learner binding'):
        make_chain(tmp_path, make_input, updates=0)
    assert not (tmp_path / 'parent/generations/000000.json').exists()
    assert not (tmp_path / 'parent/latest.json').exists()


def test_retention_keeps_parent_snapshot_from_validation_before_intent(tmp_path, make_input, monkeypatch):
    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    original = module._validate_component_chain
    registry_before = (directory / 'artifact-registry.json').read_bytes()
    def late(*args, **kwargs):
        result = original(*args, **kwargs)
        state = json.loads((directory / 'run.json').read_text())
        state['hardware']['name'] = 'changed after validation'
        atomic_json(directory / 'run.json', state)
        return result
    monkeypatch.setattr(module, '_validate_component_chain', late)
    with run_lock(directory), pytest.raises(ContractError, match='hash mismatch|snapshot|changed'):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (directory / 'artifact-registry.json').read_bytes() == registry_before


def test_completed_pruning_requires_stage_absence_and_exact_new_intent_shape(tmp_path, make_input):
    from gf_training.lineage import run_lock
    from gf_training.structured_lineage import (
        retain_component_generations,
        validate_component_chain,
    )
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    with run_lock(directory):
        retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    registry_path = directory / 'artifact-registry.json'
    registry = json.loads(registry_path.read_text())
    entry = registry['generations']['1']
    assert set(entry['deletion_intent']) == {'files', 'staging_basename'}
    stage = directory / 'artifacts' / entry['deletion_intent']['staging_basename']
    assert not stage.exists()
    stage.mkdir()
    (stage / 'foreign').write_text('preserve')
    with pytest.raises(ContractError):
        validate_component_chain(directory, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert (stage / 'foreign').read_text() == 'preserve'


@pytest.mark.parametrize('attack', ['unlink_different', 'unlink_identical', 'third_lstat'])
def test_immutable_publication_never_adopts_replacement_snapshot(tmp_path, monkeypatch, attack):
    import os

    import gf_training.structured_lineage as module
    scan = module._ChainScan(directory=tmp_path, max_owned_bytes=1024**2, deadline=float('inf'))
    path = tmp_path / 'generation.json'
    temporary = path.with_name(path.name + '.tmp')
    replacement = tmp_path / 'foreign'
    original_unlink, original_lstat = Path.unlink, Path.lstat
    calls, replaced = [], []

    def swap():
        replacement.write_bytes(path.read_bytes() if attack == 'unlink_identical' else b'foreign evidence')
        previous = original_lstat(path)
        os.replace(replacement, path)
        assert original_lstat(path).st_ino != previous.st_ino
        replaced.append(1)

    def unlink(self, *args, **kwargs):
        value = original_unlink(self, *args, **kwargs)
        if self == temporary and attack.startswith('unlink'):
            swap()
        return value

    def lstat(self, *args, **kwargs):
        if self == path and attack == 'third_lstat':
            calls.append(1)
            if len(calls) == 3:
                swap()
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', unlink)
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(ContractError, match='published lineage'):
        module._write(path, {'generation': 0}, scan=scan, immutable=True)
    assert replaced and path not in scan.files
    assert path.read_bytes() == (b'{"generation":0}\n' if attack == 'unlink_identical' else b'foreign evidence')


def test_pruning_cannot_adopt_recreated_artifacts_parent_with_same_generation_inodes(tmp_path, make_input, monkeypatch):
    import os

    import gf_training.structured_lineage as module
    from gf_training.lineage import run_lock
    from tests.structured_lineage_support import make_chain
    directory, _ = make_chain(tmp_path, make_input)
    parent = directory / 'artifacts'
    moved = directory / 'original-artifacts'
    before = (directory / 'artifact-registry.json').read_bytes()
    original = module.secrets.token_hex
    swapped = []

    def replace_parent(size):
        value = original(size)
        parent.rename(moved)
        parent.mkdir()
        for child in moved.iterdir():
            os.rename(child, parent / child.name)  # same generation-directory/file inodes.
        swapped.append(1)
        return value

    monkeypatch.setattr(module.secrets, 'token_hex', replace_parent)
    with run_lock(directory), pytest.raises(ContractError, match='original held identity'):
        module.retain_component_generations(directory, 2, max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    assert swapped and (directory / 'artifact-registry.json').read_bytes() == before
    assert sorted(child.name for child in parent.iterdir()) == ['000000', '000001', '000002']


def test_pruning_token_failure_opens_no_parent_fd(tmp_path, monkeypatch):
    import gf_training.structured_lineage as module
    scan = module._ChainScan(directory=tmp_path, max_owned_bytes=1024**2, deadline=float('inf'))
    calls = []
    original_open = module.os.open

    def opened(*args, **kwargs):
        calls.append(args[0])
        return original_open(*args, **kwargs)

    def failed(_size):
        raise MemoryError('stage token failed')

    monkeypatch.setattr(module.os, 'open', opened)
    monkeypatch.setattr(module.secrets, 'token_hex', failed)
    with pytest.raises(MemoryError, match='stage token failed'):
        module._prune_staged(tmp_path, tmp_path / 'artifacts/000001', {}, {}, {}, '0' * 64, scan)
    assert calls == []
