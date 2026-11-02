"""Controller derivation negatives; metadata smokes never qualify Source."""
import json
import shutil
import time

import pytest

from gf_models.arrays import atomic_json, sha256
from gf_models.structured_binding import model_defaults_v2
from tests.native_structured_support import metadata_parent

native = pytest.importorskip('gf_native')
pytestmark = pytest.mark.skipif(not hasattr(native, 'StructuredNativeSession'), reason='candidate Session required')
SETTINGS = b'{"memory_bytes":536870912,"scratch_bytes":1073741824,"max_source_file_bytes":67108864,"max_record_bytes":8388608,"chunk_target_bytes":16777216}'


def test_historical_uncommitted_neural_batch_is_not_downgraded_to_raw_nulls(tmp_path):
    root = tmp_path / 'parent'
    metadata_parent(root, model_defaults_v2().architecture())
    collection = root / 'collections/batch-000000'
    collection.mkdir()
    atomic_json(collection / 'manifest.json', {'schema_version': 4, 'collection_kind': 'ppo_batch'})
    plan = {'schema_version': 2, 'mode': 'historical_parent',
        'parents': [{'directory': str(root), 'require_complete': False}],
        'collections': [{'directory': str(collection), 'parent_index': 0}]}
    with pytest.raises(ValueError, match='uncommitted neural train collection'):
        native.StructuredNativeSession(json.dumps(plan).encode(), SETTINGS, time.monotonic() + 60)


def test_duplicate_committed_collection_derivation_rejects_before_raw_source_scan(tmp_path, make_input):
    from tests.structured_lineage_support import make_chain
    root, _ = make_chain(tmp_path, make_input)
    zero = root / 'artifacts/000000'
    original_inference = json.loads((zero / 'inference.json').read_text())
    same_inference = sha256(zero / 'inference.json')
    registry = json.loads((root / 'artifact-registry.json').read_text())
    previous = sha256(root / 'generations/000000.json')
    first_update = json.loads((root / 'updates/000001.json').read_text())
    for generation in (1, 2):
        artifact = root / 'artifacts' / f'{generation:06}'
        # Explicit metadata-only corruption fixture: equal inference bytes are
        # possible with zero gradients. Real NPZ files remain typed/hashed; this
        # is NOT a claim that these altered records describe genuine learning.
        for name in ('inference.json', *original_inference['weights']['files']):
            shutil.copyfile(zero / name, artifact / name)
        resume = json.loads((artifact / 'resume.json').read_text())
        resume['inference_digest'] = same_inference
        atomic_json(artifact / 'resume.json', resume)
        record_path = root / 'generations' / f'{generation:06}.json'
        record = json.loads(record_path.read_text())
        record.update(parent_sha256=previous, inference_sha256=same_inference, resume_sha256=sha256(artifact / 'resume.json'))
        record['artifact_files'].update(original_inference['weights']['files'])
        record['artifact_files']['inference.json'] = same_inference
        record['artifact_files']['resume.json'] = record['resume_sha256']
        update_path = root / 'updates' / f'{generation:06}.json'
        update = json.loads(update_path.read_text())
        update['parent_sha256'] = previous
        if generation == 2:
            for key in ('collection', 'collection_manifest_sha256', 'source_proof', 'numeric', 'numeric_sha256', 'numeric_files'):
                update[key] = first_update[key]
        atomic_json(update_path, update)
        record['update_sha256'] = sha256(update_path)
        atomic_json(record_path, record)
        registry['generations'][str(generation)].update(inference_sha256=same_inference, original_files=record['artifact_files'])
        previous = sha256(record_path)
    atomic_json(root / 'artifact-registry.json', registry)
    state = json.loads((root / 'run.json').read_text())
    state['generation_sha256'] = previous
    atomic_json(root / 'run.json', state)
    atomic_json(root / 'latest.json', {'generation': 2, 'sha256': previous})
    # C2a's metadata/maps/byte-hash layer accepts this coherent control fixture;
    # the Source composition must inspect EACH generation ownership, not find1.
    core = native.StructuredParentControls(str(root), b'{"memory_bytes":536870912,"max_json_bytes":67108864}', time.monotonic() + 60)
    core.close()
    plan = {'schema_version': 2, 'mode': 'historical_parent',
        'parents': [{'directory': str(root), 'require_complete': True}], 'collections': []}
    with pytest.raises(ValueError, match='duplicate committed collection ownership'):
        native.StructuredNativeSession(json.dumps(plan).encode(), SETTINGS, time.monotonic() + 60)
