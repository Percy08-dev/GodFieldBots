"""Strict empty marker DATA validation, with no live Source authority."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError


def marker():
    root = Path('/data/run/collections/batch-000002')
    return {'batch_id': 'batch-000002', 'collection_generation': 2,
        'manifest_path': str(root / 'manifest.json'), 'manifest_sha256': 'a' * 64,
        'descriptor_path': str(root / 'collection.pending.json'), 'descriptor_sha256': 'b' * 64,
        'learner_decisions': 0}


@pytest.mark.parametrize('field,value', [('learner_decisions', True), ('learner_decisions', 1),
    ('collection_generation', True), ('batch_id', 'batch-000001'), ('manifest_path', '/output/manifest.json'),
    ('descriptor_path', '/foreign/run/collections/batch-000002/collection.pending.json'),
    ('manifest_sha256', 'x' * 64), ('extra', None)])
def test_empty_marker_has_exact_logical_original_root_and_physical_descriptor(field, value):
    from gf_training.rust_player_lineage import validate_empty_collection
    state = {'run_id': 'run', 'updates': 2, 'status': 'incomplete', 'stop_reason': 'empty_collection',
        'effective_config': {'run': {}}}
    config = SimpleNamespace(storage=SimpleNamespace(db_root=Path('/data')))
    row = marker()
    assert validate_empty_collection(row, state=state, config=config) is row
    changed = deepcopy(row)
    changed[field] = value
    with pytest.raises(ContractError):
        validate_empty_collection(changed, state=state, config=config)


def test_empty_data_owner_cannot_be_minted_from_metadata():
    from gf_training.rust_player_source import RustEmptyCollection
    with pytest.raises(ContractError, match='actual'):
        RustEmptyCollection(marker())
    assert not hasattr(RustEmptyCollection, 'persist')
    assert not hasattr(RustEmptyCollection, 'consume')


def test_empty_marker_cannot_be_added_to_ordinary_or_successful_invocations():
    import json

    from gf_training.rust_player_lineage import validate_run
    path = Path(__file__).parent / 'fixtures/rust_player_run5_common_metadata.json'
    state = json.loads(path.read_bytes())
    key = next(iter(state['invocations']))
    state['invocations'][key]['empty_collection'] = marker()
    with pytest.raises(ContractError):
        validate_run(state)


@pytest.mark.parametrize('closure,missing', [({'truncated': 2}, []), ({'aborted': 2}, []),
    ({'cancelled': 2}, []), ({'natural': 1}, []), ({'natural': 2}, ['episode'])])
def test_empty_inspection_requires_whole_natural_collection_and_preserves_outcomes(closure, missing):
    from gf_training.rust_player_source import learner_records
    before = deepcopy(closure)
    rows = [SimpleNamespace(episode_id='one', viewer=1, records=SimpleNamespace(frames=())),
        SimpleNamespace(episode_id='two', viewer=2, records=SimpleNamespace(frames=()))]
    with pytest.raises(ContractError):
        learner_records(rows, {'one': 1, 'two': 2}, closure, missing, games=2)
    assert closure == before


def test_empty_counts_come_from_selected_own_seats_without_filtering_nonempty_rows():
    from gf_training.rust_player_source import learner_records
    selected = SimpleNamespace(frames=())
    other = SimpleNamespace(frames=(object(),))
    rows = [SimpleNamespace(episode_id='one', viewer=1, records=selected),
        SimpleNamespace(episode_id='one', viewer=2, records=other)]
    assert learner_records(rows, {'one': 1}, {'natural': 1}, [], games=1) == (selected,)
    assert learner_records(rows, {'one': 2}, {'natural': 1}, [], games=1) == (other,)


@pytest.mark.parametrize('generation', [0, 2])
def test_run5_first_or_later_empty_marker_is_optional_only_for_actual_empty_reason(generation):
    """Pure configuration/record DATA; does not claim an actual empty capture."""
    import json

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.domain import digest
    from gf_common.rust_player import runtime_binding
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    from gf_training.rust_player_lineage import validate_run
    from tests.test_rust_player_config import RUN_ID, campaign_raw
    state = json.loads((Path(__file__).parent / 'fixtures/rust_player_run5_common_metadata.json').read_bytes())
    campaign = campaign_config(campaign_raw(), origin=Path(__file__).parents[1])
    settings = effective_settings(campaign.config.model, campaign.config.training, {}, layout=layout)
    loaded = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID)
    bound = runtime_binding(loaded.config.hpo_spec['runtime'], state['identity']['player_runtime']['producer'])
    state['identity']['player_runtime'] = bound
    state.update(effective_config=loaded.effective, config_sha256=loaded.sha256,
        runtime_binding_sha256=digest(bound), hpo_spec_sha256=digest(loaded.config.hpo_spec),
        run_variant='hpo-ppo-owned-rust-player-v1', model_version='4', updates=generation,
        used_decisions=0, collected_decisions=0, stop_reason='empty_collection', status='incomplete')
    key = next(iter(state['invocations']))
    root = Path(loaded.config.storage.db_root) / state['run_id'] / 'collections' / f'batch-{generation:06}'
    row = marker() | {'batch_id': f'batch-{generation:06}', 'collection_generation': generation,
        'descriptor_path': str(root / 'collection.pending.json'), 'manifest_path': str(root / 'manifest.json')}
    state['invocations'][key]['empty_collection'] = row
    assert validate_run(state, hpo=True) is state
    no_marker = deepcopy(state)
    del no_marker['invocations'][key]['empty_collection']
    with pytest.raises(ContractError, match='exactly one'):
        validate_run(no_marker, hpo=True)
    for field, changed in [('status', 'complete'), ('stop_reason', 'update_cap')]:
        with pytest.raises(ContractError):
            validate_run(state | {field: changed}, hpo=True)


def test_empty_cold_schedule_rederives_frozen_rng_pool_and_refuses_foreign_bindings():
    """Synthetic schedule DATA only, never a claim of capture or an engine result."""
    import numpy as np

    from gf_common.domain import digest
    from gf_common.schedule import serialize_schedule
    from gf_training.hpo_collection import training_groups_hpo
    from gf_training.rust_player_completion import _audit_empty_schedule
    rng = np.random.Generator(np.random.PCG64(17))
    current = {'parent_controls': {'snapshot_pool': [0], 'schedule_opponent_rng': deepcopy(rng.bit_generator.state)}}
    state = {'updates': 0, 'seed': 17}
    loaded = SimpleNamespace(effective={}, config=SimpleNamespace(hpo_spec={'training_recipe': {'collection_games': 16}}))
    parent = Path('/owned/parent')
    bindings = {'learner': {'model_id': 'mlp', 'policy_options': {'checkpoint': str(parent / 'artifacts/000000'),
        'expected_manifest_digest': 'a' * 64}}}
    manifest = {'policy_bindings': bindings, 'schedule': serialize_schedule(training_groups_hpo(17, 0,
        'learner', ('learner',), rng, collection_games=16), 4)}
    _audit_empty_schedule(manifest, state=state, current=current, loaded=loaded, parent=parent, inference_files={0: 'a' * 64})
    for changed in ('schedule', 'checkpoint', 'manifest'):
        bad = deepcopy(manifest)
        if changed == 'schedule':
            bad['schedule'][0]['game_seed'] = 'foreign'
        elif changed == 'checkpoint':
            bad['policy_bindings']['learner']['policy_options']['checkpoint'] = '/foreign/artifacts/000000'
        else:
            bad['policy_bindings']['learner']['policy_options']['expected_manifest_digest'] = digest({})
        with pytest.raises(ContractError):
            _audit_empty_schedule(bad, state=state, current=current, loaded=loaded, parent=parent, inference_files={0: 'a' * 64})


@pytest.mark.parametrize('mutation', [None, 'false-zero', 'manifest-sha', 'descriptor-sha', 'selection-context',
    'truncated', 'aborted', 'cancelled', 'missing-replay'])
def test_cold_empty_data_inspection_rechecks_counts_and_retires_actual_test_fd(tmp_path, monkeypatch, mutation):
    """Synthetic inspection outputs and manifest DATA, not actual Raw/engine evidence."""
    import os
    import time

    import numpy as np

    from gf_common.domain import digest
    from gf_common.hpo import selection_context
    from gf_common.schedule import serialize_schedule
    from gf_training import card_instance_collection_source, structured_source
    from gf_training.group_pack_source import OriginalFile
    from gf_training.hpo_collection import training_groups_hpo
    from gf_training.rust_player_completion import _audit_empty_originals
    parent, originals_root = tmp_path / 'parent', tmp_path / 'originals'
    originals_root.mkdir()
    descriptor = originals_root / 'collection.pending.json'
    descriptor.write_text('synthetic data; no pending/Cold capability')
    original = OriginalFile(descriptor, time.monotonic() + 30)
    fd = original.fd
    rng = np.random.Generator(np.random.PCG64(17))
    current = {'snapshot_identity': 'snapshot', 'parent_controls': {'snapshot_pool': [0],
        'schedule_opponent_rng': deepcopy(rng.bit_generator.state)}}
    groups = training_groups_hpo(17, 0, 'learner', ('learner',), rng, collection_games=16)
    schedule = serialize_schedule(groups, 4)
    seats = {episode.episode_id: episode.learner_viewer for group in groups for episode in group.episodes}
    admission = {'invocation_id': '2' * 32, 'producer': {}}
    state = {'run_id': 'run', 'updates': 0, 'seed': 17, 'used_decisions': 0,
        'runtime_binding_sha256': 'b' * 64, 'identity': {'code': {}}, 'config_inputs': []}
    loaded = SimpleNamespace(effective={}, sha256=digest({}), config=SimpleNamespace(actions=None,
        hpo_spec={'decision_budget': 64, 'training_recipe': {'collection_games': 16}}))
    common = {'schema_version': 1, 'run_id': 'run', 'batch_id': 'batch-000000', 'generation': 0,
        'invocation_id': admission['invocation_id'], 'admission_sha256': digest(admission), 'runtime_binding_sha256': 'b' * 64}
    settlement = {'run_id': 'run', 'batch_id': 'batch-000000', 'generation': 0,
        'invocation_id': admission['invocation_id'], 'service_id': 'service', 'gpu_fenced': True,
        'active_policy_references': 0, 'native_pending': 0, 'started_producers': 0}
    capture = common | {'kind': 'rust-player-capture-authority-v1', 'actor_protocol_version': 11,
        'encoding_contexts_sha256': digest([]), 'accepted_records': [], 'completed_reports_sha256': digest([])}
    retirement = common | {'kind': 'rust-player-collection-retirement-v1', 'service_id': 'service',
        'episode_ids': list(seats), 'completed_reports': [], 'accepted_records_sha256': digest([]),
        'service_settlement': settlement, 'service_settlement_sha256': digest(settlement), 'gpu_fenced': True}
    manifest = {'config': {}, 'code': {}, 'inputs': [], 'config_hash': loaded.sha256,
        'runtime_binding_sha256': 'b' * 64, 'player_producer': {}, 'snapshot_identity': 'snapshot',
        'capture_authority': capture, 'logical_retirement': retirement, 'schedule': schedule,
        'linkage': {'parent_run_id': 'run', 'parent_run_path': str(parent), 'batch_id': 'batch-000000',
            'collection_generation': 0, 'learner_seats': seats,
            'selection_context': selection_context(decision_budget=64, used_before=0, learning_seed=17, batch=0)},
        'policy_bindings': {'learner': {'model_id': 'mlp', 'policy_options': {'checkpoint': str(parent / 'artifacts/000000'),
            'expected_manifest_digest': 'a' * 64}}}}
    if mutation == 'selection-context':
        manifest['linkage']['selection_context']['used_before'] = 1
    marker_data = {'batch_id': 'batch-000000', 'manifest_path': str(originals_root / 'manifest.json'), 'manifest_sha256': digest(manifest),
        'descriptor_sha256': original.sha256}
    if mutation == 'manifest-sha': marker_data['manifest_sha256'] = '0' * 64
    if mutation == 'descriptor-sha': marker_data['descriptor_sha256'] = '0' * 64
    class RawData:
        root = originals_root
        _descriptor_sha256 = original.sha256
        def object(self, path):
            assert path == self.root / 'manifest.json'
            return manifest
        def close(self):
            original.close()
    raw = RawData()
    outputs = [SimpleNamespace(viewer=viewer, episode_id=episode, records=SimpleNamespace(
        frames=(object(),) if mutation == 'false-zero' else ())) for episode, viewer in seats.items()]
    closures = {mutation: 16} if mutation in ('truncated', 'aborted', 'cancelled') else {'natural': 16}
    missing = ['one'] if mutation == 'missing-replay' else []
    monkeypatch.setattr(card_instance_collection_source, 'inspect_rust_game_manifest', lambda *a, **k: ([], []))
    monkeypatch.setattr(structured_source, '_inspect_collection_content', lambda *a, **k: (outputs, 0, closures, missing))
    try:
        if mutation is None:
            _audit_empty_originals(raw, loaded=loaded, state=state, current=current, marker=marker_data,
                admission=admission, parent=parent, inference_files={0: 'a' * 64})
        else:
            with pytest.raises(ContractError) as failed:
                _audit_empty_originals(raw, loaded=loaded, state=state, current=current, marker=marker_data,
                    admission=admission, parent=parent, inference_files={0: 'a' * 64})
            os.fstat(fd)
            failed.value.cleanup_owner.close()
        with pytest.raises(OSError):
            os.fstat(fd)
        assert not (originals_root / 'manifest.json').exists()
        assert not (parent / 'numeric').exists()
    finally:
        original.close()


def test_empty_cleanup_only_partial_owner_keeps_actual_fd_and_pool_hold_until_positive_close(tmp_path):
    """Cleanup-only partial object: cannot issue capture, marker, Source or Parent."""
    import os
    import time
    import weakref

    import numpy as np

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.group_pack_source import OriginalFile
    from gf_training.rust_player_source import RustEmptyCollection
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    path = tmp_path / 'actual-original'
    path.write_bytes(b'cleanup data')
    original = OriginalFile(path, pool.deadline)
    descriptor, actual_close = original.fd, original.close
    loan = pool.budget.reserve(4096)
    partial = RustEmptyCollection.__new__(RustEmptyCollection)
    partial.state, partial._cleanup_children = 'SEALED', []
    partial._records, partial.manifest = [np.arange(64)], None
    array = weakref.ref(partial._records[0])
    class DataOriginals:
        def close(self):
            assert array() is None
            original.close()
            loan.close()
    class CleanupOnlyLease:
        closed = False
        def close(self, owner):
            assert owner is partial
            self.closed = True
    partial.collection = DataOriginals()
    partial._parent_lease = CleanupOnlyLease()
    partial._pool_hold = pool.hold_close(partial)
    partial.registry = SimpleNamespace(pending=partial)
    before = pool.budget.used
    def fail():
        raise OSError('before original terminal close')
    original.close = fail
    try:
        with pytest.raises(OSError, match='terminal close'):
            partial.close()
        os.fstat(descriptor)
        assert pool.budget.used == before and not partial._parent_lease.closed
        assert partial.registry.pending is partial and partial.state != 'CLOSED'
        original.close = actual_close
        partial.close()
        partial.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert partial.registry.pending is None and partial._parent_lease.closed
        assert partial.state == 'CLOSED' and loan.closed
    finally:
        original.close = actual_close
        partial.close()
        pool.close()
    assert pool.budget.used == 0
