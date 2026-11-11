"""Review regressions use actual storage or explicitly unadmitted control values."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.group_pack_game import storage_binding
from gf_common.preprocessing import PreparationConfig
from tests.test_group_pack_game_source import owner, packed


def test_fresh_dataset_training_row_emits_actual_custody2_from_custody1_update(tmp_path, monkeypatch):
    from gf_training.card_instance_game_tree_data import _GameTreeOriginals
    root, _ = packed(tmp_path)
    original = owner(root)
    try:
        old = deepcopy(original.custody)
        old['recipe'] = 'packed-original-custody-v1'
        old['fingerprint'] = digest({k: v for k, v in old.items() if k != 'fingerprint'})
        manifest = {'collection_role': 'ppo-batch', 'linkage': {'parent_run_id': 'UNADMITTED',
            'parent_run_path': str(tmp_path), 'collection_generation': 0}, 'schedule': [], 'policy_bindings': {}}
        update = {'collection': 'envelope', 'collection_manifest_sha256': 'a'*64,
            'numeric': 'numeric', 'numeric_sha256': 'b'*64, 'numeric_files': {},
            'accepted_decisions': 0, 'collection_seconds': 0,
            'source_proof': {'source_set_sha256': 'c'*64, 'storage': original.storage, 'custody': old}}
        source = object.__new__(_GameTreeOriginals)
        source._packed = True
        source._custody = SimpleNamespace(parents=(tmp_path,))
        source._states = [{'run_id': 'UNADMITTED'}]
        source._records = [[{}, {'update_sha256': 'd'*64}]]
        source._parents = [SimpleNamespace(scan=SimpleNamespace(packed_owner=lambda *_: original))]
        monkeypatch.setattr(source, '_object', lambda path, expected=None: update if path.parent.name == 'updates'
            else {'schema_version': 4, 'kind': 'structured-rollout', 'decision_count': 0})
        monkeypatch.setattr(source, '_logical_object', lambda *_: manifest)
        monkeypatch.setattr(source, '_control_digest', lambda *_: 'a'*64)
        row, _, _ = source._collection(0, 0, validation=False)
        assert row['custody'] == original.custody
        assert row['custody']['recipe'] == 'packed-original-custody-v2'
        assert update['source_proof']['custody']['recipe'] == 'packed-original-custody-v1'
    finally:
        original.close()


@pytest.mark.parametrize('marker', [None, 'wrong'])
def test_checked_rust_parent_refuses_before_original_constructor(tmp_path, monkeypatch, marker):
    from gf_training import native_backend
    from gf_training.group_pack_game_source import PackedSourceOwner
    from gf_training.structured_lineage import _ChainScan
    native = SimpleNamespace(build_info=lambda: {'card_instance_pack_checksum': marker})
    monkeypatch.setattr(native_backend, 'load_native', lambda: native)
    monkeypatch.setattr(PackedSourceOwner, '__init__', lambda *_args, **_kw: pytest.fail('packed admission before marker'))
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512*1024**2, deadline=time.monotonic()+30)
    state = {'schema_version': 4, 'storage_binding': storage_binding(), 'run_id': 'UNADMITTED',
        'effective_config': {'preprocessing': asdict(PreparationConfig(backend='rust', reader='participant-structured-v2')),
            'storage': {'budget_mib': 512, 'db_root': str(tmp_path/'db')}}}
    try:
        scan.configure_packed(state)
        with pytest.raises(ContractError, match='checksum'):
            scan.packed_owner(tmp_path/'collections/batch-000000')
        assert scan._packed_pool is None
    finally:
        scan.close(recheck=False)


def test_explicit_historical_native_candidate_also_requires_checksum_marker(tmp_path, monkeypatch):
    from gf_common.card_instance import ActionConfigV3, corrected_card_instance_contract
    from gf_training import native_backend
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.native_group_pack import OriginalArtifactProvider
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    plan = {'schema_version': 4, 'mode': 'historical_parent', 'parents': [], 'collections': [],
        'storage_binding': storage_binding(), 'artifact_registry_sha256': provider.registry_sha256}
    native = SimpleNamespace(build_info=dict)
    inputs = _NativeGameInputs(plan, PreparationConfig(backend='rust', reader='participant-structured-v2'),
        actions=ActionConfigV3(action_contract=corrected_card_instance_contract()), max_owned_bytes=1024**3,
        deadline=time.monotonic()+30, native=native, artifact_provider=provider)
    monkeypatch.setattr(native_backend, 'binary_identity', lambda *_: pytest.fail('candidate binary work before marker'))
    try:
        with pytest.raises(ContractError, match='checksum'):
            inputs.__enter__()
    finally:
        provider.close()
        original.close()


def test_saved_dataset_correspondence_uses_original_controls_and_actual_episodes(monkeypatch):
    from gf_common import card_instance_game_tree as game
    from gf_common.group_pack_game import same_packed_dataset
    from gf_models.structured_sealed import SEMANTIC_KEYS_V3
    from gf_training.card_instance_game_tree_data import _GameTreeOriginals
    from gf_training.structured_rollout import _source_correspondence
    from tests.test_group_pack_dataset_contract import packed_dto
    old = packed_dto()
    current = deepcopy(old)
    semantics = dict.fromkeys(SEMANTIC_KEYS_V3, 'UNADMITTED-comparison-only')
    fields = lambda row: {'original_control': 'exact', 'storage': row['storage'], 'custody': row['custody']}
    for left, right in zip(old['collections'], current['collections'], strict=True):
        right['custody']['recipe'] = 'packed-original-custody-v2'
        right['custody']['fingerprint'] = digest({k: v for k, v in right['custody'].items() if k != 'fingerprint'})
        if left['split'] == 'train':
            for row in (left, right):
                row['original_correspondence'] = _source_correspondence(digest(fields(row)), semantics,
                    tuple(row['original_correspondence']['episode_digests']), game=True, storage_binding=storage_binding())
    def aggregate(value):
        value['source_set_sha256'] = game.source_identity(value['parents'], value['collections'])
        value['content_sha256'] = digest({k: v for k, v in value.items() if k != 'content_sha256'})
    aggregate(old)
    aggregate(current)
    source = object.__new__(_GameTreeOriginals)
    source._collections = current['collections']
    source._states = [{'storage_binding': storage_binding()} for _ in range(3)]
    source._neural = {'semantics': semantics}
    monkeypatch.setattr(source, '_training_fingerprint_fields', lambda _i, _g, row: fields(row))
    comparison = source.historical_correspondences(old)
    assert same_packed_dataset(old, current, historical_correspondences=comparison)
    # Changed episode data cannot pass by retaining or copying the saved digest.
    current['collections'][0]['original_correspondence']['episode_digests'] = ['f'*64]*16
    relation = current['collections'][0]['original_correspondence']
    relation['binding_sha256'] = digest({k: v for k, v in relation.items() if k != 'binding_sha256'})
    aggregate(current)
    assert not same_packed_dataset(old, current,
        historical_correspondences=source.historical_correspondences(old))
    assert all(row['custody']['recipe'] == 'packed-original-custody-v2' for row in current['collections'])
