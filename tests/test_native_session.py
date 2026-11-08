"""Native whole-source validation and zero-copy lifetime, using synthetic evidence."""
import gc
import time
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig
from gf_models.encoder import encode
from gf_training.fixed_data import (
    build_dataset,
    index_dataset,
    inspect_dataset,
    inspect_dataset_summary,
    prepare_fixed_inputs,
)
from gf_training.native_authorization import NativePreparation
from gf_training.native_learning import fixed_consumer_layout, fixed_targets

pytest.importorskip('gf_native')


@pytest.fixture(params=('legacy','typed-borrowed-v1'))
def reader(request):
    import gf_native

    if request.param=='typed-borrowed-v1' and dict(gf_native.build_info()).get('typed_reader_contract')!='typed-borrowed-v1':
        pytest.skip('installed historical native wheel has no typed reader capability')
    return request.param


@pytest.mark.parametrize('history', ['none', 'public_aggregate'])
def test_native_whole_source_numeric_and_manifest_parity(reader, tmp_path, frozen_sources, history):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    frozen = build_dataset(parents, validations, path)
    _, train, validation = inspect_dataset(path)
    encoder = EncoderConfig(history=history)
    settings = replace(PreparationConfig(), backend='rust', reader=reader, memory_budget_mib=128, cache_budget_mib=16,
                       chunk_target_mib=1)
    features = {0: [], 1: []}
    targets = {0: [], 1: []}
    batches = 0
    with NativePreparation({'kind': 'offline', 'encoder': asdict(encoder),
                            'dataset_path': str(path)}, settings, deadline=time.monotonic() + 60) as owner:
        assert owner._session.control_reserved_bytes() >= 32 * 1024 * 1024
        def retained_bytes():
            return sum(array.nbytes for values in (*features.values(), *targets.values()) for array in values)
        while (batch := owner.next_batch(retained_bytes(), fixed_consumer_layout(encoder))) is not None:
            batches += 1
            labels = fixed_targets(batch, global_ready=owner.proof()['structural_valid'])
            for index, split in enumerate(batch.split):
                start, stop = batch.episode_offsets[index:index + 2]
                features[int(split)].append(batch.features[start:stop].copy())
                targets[int(split)].append(labels[start:stop].copy())
            del batch
            del labels
        assert owner.proof()['dataset_manifest'] == frozen
        assert owner.proof()['reader_contract']==reader
        assert owner._session.control_reserved_bytes() >= 32 * 1024 * 1024
    assert owner._session.control_reserved_bytes() == 0
    assert owner.diagnostics['ffi_data_calls'] == batches + 1
    assert owner.diagnostics['reader_contract']==reader
    if reader=='typed-borrowed-v1':
        assert owner.diagnostics['typed_journal_files']==480
        assert owner.diagnostics['legacy_journal_files']==0
    assert owner.diagnostics['ffi_batches_returned'] == batches
    assert owner.diagnostics['ffi_eof_returns'] == 1
    copies = owner.diagnostics['payload_copy_stats']
    assert copies['verified_shared_numeric_exports'] > 0
    assert copies['verified_shared_numeric_bytes'] > 0
    assert copies['numeric_copy_events'] == copies['numeric_copy_bytes'] == 0
    assert owner.diagnostics['stage_read_bytes'] > 0 and owner.diagnostics['stage_write_bytes'] > 0
    phases = owner.diagnostics['aggregated_worker_phase_calls']
    assert all(phases[name] > 0 for name in ('source_read_hash', 'source_decode',
                                            'journal_input_validation', 'replay_validation',
                                            'feature_encoding', 'proof_staging'))
    for split, samples in ((0, train), (1, validation)):
        expected = np.stack([encode(sample.value, sample.action, encoder) for sample in samples])
        np.testing.assert_array_equal(np.concatenate(features[split]), expected)
        np.testing.assert_array_equal(np.concatenate(targets[split]), [sample.target for sample in samples])


def test_native_source_mutation_invalidates_finalization_and_slices_survive_close(reader, tmp_path, frozen_sources):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    build_dataset(parents, validations, path)
    encoder = EncoderConfig()
    settings = replace(PreparationConfig(), backend='rust', reader=reader, memory_budget_mib=128, cache_budget_mib=16, chunk_target_mib=1)
    owner = NativePreparation({'kind': 'offline', 'encoder': asdict(encoder),
                               'dataset_path': str(path)}, settings, deadline=time.monotonic() + 60)
    batch = owner.next_batch(0, fixed_consumer_layout(encoder))
    view = batch.features[::2]
    expected = view.copy()
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.setflags(write=True)
    del batch
    while owner.next_batch(0, fixed_consumer_layout(encoder)) is not None:
        pass
    source = next(parents[0].glob('collections/*/*/player-1.jsonl'))
    source.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(ContractError, match='source invalidated'):
        owner.finalize()
    np.testing.assert_array_equal(view, expected)
    assert owner.diagnostics['reader_contract']==reader
    retained = owner._session.used_bytes()
    assert retained >= view.nbytes
    del view
    gc.collect()
    assert owner._session.used_bytes() < retained


def test_native_binary_identity_is_the_loaded_extension():
    from importlib.machinery import EXTENSION_SUFFIXES

    from gf_training.native_backend import binary_identity
    identity = binary_identity()
    assert any(identity['extension_path'].endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    assert len(identity['extension_sha256']) == 64
    assert identity['build']['cargo_lock_sha256']
    assert identity['build']['native_source_sha256']


def test_native_public_summary_build_and_numeric_writer_match_legacy(reader, tmp_path, frozen_sources):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    manifest = build_dataset(parents, validations, path)
    settings = replace(PreparationConfig(), backend='rust', reader=reader, memory_budget_mib=128,
                       cache_budget_mib=16, chunk_target_mib=1)
    assert inspect_dataset_summary(path, preprocessing=settings) == manifest
    assert build_dataset(parents, validations, tmp_path / 'native-dataset.json',
                         preprocessing=settings) == manifest
    config = EncoderConfig(history='public_aggregate')
    with prepare_fixed_inputs(path, config) as reference:
        reference.write_numeric(tmp_path / 'reference.npz')
        identity = reference.diagnostics['sample_identity_sha256']
        expected = reference.x.copy()
    with prepare_fixed_inputs(path, config, preprocessing=settings) as prepared:
        prepared.write_numeric(tmp_path / 'native.npz')
        assert prepared.diagnostics['sample_identity_sha256'] == identity
        assert prepared.diagnostics['reader_contract']==reader
        np.testing.assert_array_equal(prepared.x, expected)
        view = prepared.x[::2]
        captured = view.copy()
    np.testing.assert_array_equal(view, captured)
    assert (tmp_path / 'reference.npz').read_bytes() == (tmp_path / 'native.npz').read_bytes()


def test_native_index_typed_columns_match_legacy_order_and_content(reader, tmp_path, frozen_sources):
    pq = pytest.importorskip('pyarrow.parquet')
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    build_dataset(parents, validations, path)
    settings = replace(PreparationConfig(), backend='rust', reader=reader, memory_budget_mib=128,
                       cache_budget_mib=16, chunk_target_mib=1)
    reference = index_dataset(path, tmp_path / 'reference.parquet')
    actual = index_dataset(path, tmp_path / 'native.parquet', preprocessing=settings)
    repeated = index_dataset(path, tmp_path / 'native-again.parquet', preprocessing=settings)
    assert actual == repeated
    assert actual['row_content_sha256'] == reference['row_content_sha256']
    assert actual['rows'] == reference['rows']
    assert pq.read_table(tmp_path / 'reference.parquet').equals(pq.read_table(tmp_path / 'native.parquet'))
