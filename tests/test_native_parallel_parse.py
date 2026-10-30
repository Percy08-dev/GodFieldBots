"""Public parity for historical parse windows and owned episode pipeline wheels."""
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig
from gf_models.encoder import encode
from gf_training.fixed_data import build_dataset, inspect_dataset
from gf_training.native_authorization import NativePreparation
from gf_training.native_learning import fixed_consumer_layout
from tests.test_fixed_ppo_data import frozen_sources as frozen_sources  # noqa: PLC0414

pytestmark = pytest.mark.skipif(
    os.environ.get('GFB_TEST_PARALLEL_PARSE') != '1',
    reason='requires the isolated Stage5a candidate wheel; opt in with GFB_TEST_PARALLEL_PARSE=1',
)


@pytest.mark.parametrize('workers', [1, 2, 4, 8])
@pytest.mark.parametrize('reader', ['legacy', 'typed-borrowed-v1'])
def test_scoped_parse_public_arrays_order_wall_and_retained_views(
    tmp_path, frozen_sources, workers, reader,
):
    parents, validations = frozen_sources
    path = tmp_path / 'dataset.json'
    manifest = build_dataset(parents, validations, path)
    _, train, validation = inspect_dataset(path)
    encoder = EncoderConfig(hash_buckets=2)
    settings = replace(
        PreparationConfig(), backend='rust', reader=reader, workers=workers,
        max_in_flight=workers, memory_budget_mib=256, cache_budget_mib=16,
        chunk_target_mib=1,
    )
    features = {0: [], 1: []}
    with NativePreparation(
        {'kind': 'offline', 'encoder': asdict(encoder), 'dataset_path': str(path)},
        settings, deadline=time.monotonic() + 60,
    ) as owner:
        while (batch := owner.next_batch(0, fixed_consumer_layout(encoder))) is not None:
            for index, split in enumerate(batch.split):
                start, stop = batch.episode_offsets[index:index + 2]
                # Keep native-backed slices through finalization and owner release.
                features[int(split)].append(batch.features[start:stop])
            del batch
        assert owner.proof()['dataset_manifest'] == manifest
        diagnostic = json.loads(owner._session.telemetry())
        wall = diagnostic['coordinator_scan_wall']
        partitions = wall['partition_nanoseconds']
        assert sum(partitions.values()) == wall['partition_sum_nanoseconds'] == wall['parent_nanoseconds']
        assert wall['residual_nanoseconds'] == 0
        parse = diagnostic['typed_parse_parallelism']
        decode = diagnostic['decode_parallelism']
        assert parse['active_jobs'] == decode['active_jobs'] == 0
        assert parse['configured_workers'] == decode['configured_workers'] == workers
        assert parse['job_unit'] == 'participant-journal-file'
        assert parse['job_unit_version'] == 2
        pipeline = diagnostic.get('complete_episode_pipeline')
        if pipeline is not None:
            assert pipeline['version'] == pipeline['job_unit_version'] == 1
            assert pipeline['job_unit'] == 'complete-episode'
            assert pipeline['configured_workers'] == workers
            assert pipeline['configured_max_in_flight'] == workers
            assert pipeline['admitted'] == pipeline['completed'] == pipeline['committed'] == 240
            assert pipeline['failed'] == pipeline['active_jobs'] == 0
            assert pipeline['pool_threads_started'] == pipeline['pool_threads_exited'] == (0 if workers == 1 else workers)
            assert 1 <= pipeline['active_peak'] <= workers
            assert pipeline['admitted_uncommitted_peak'] <= workers
            assert wall['version'] == 2
            assert partitions['episode_pool_wait'] > 0
            assert partitions['episode_ordered_commit'] > 0
            assert parse['scheduled_journal_tasks'] == parse['spawned_threads'] == 0
            assert decode['scheduled_episodes'] == decode['spawned_threads'] == 0
            assert diagnostic['typed_journal_files'] == (0 if reader == 'legacy' else 480)
            if reader != 'legacy':
                assert 'complete episode jobs own' in diagnostic['journal_parse_scope']
        elif reader == 'legacy':
            assert partitions['typed_parse_window_dispatch_and_join'] == 0
            assert parse['scheduled_episodes'] == parse['spawned_threads'] == parse['active_parse_peak'] == 0
            assert parse['scheduled_journal_tasks'] == 0
        else:
            assert partitions['typed_parse_window_dispatch_and_join'] > 0
            assert parse['scheduled_episodes'] * 2 == parse['scheduled_journal_tasks'] == diagnostic['typed_journal_files'] == 480
            assert sum(slot['jobs'] for slot in parse['logical_slots']) == parse['scheduled_journal_tasks']
            assert sum(slot['failed_jobs'] for slot in parse['logical_slots']) == 0
            assert 1 <= parse['active_parse_peak'] <= workers
            assert parse['spawned_threads'] == (0 if workers == 1 else decode['spawned_threads'])
            assert parse['effective_slots_peak'] == workers
            assert 'bounded scoped borrowed-raw parse windows' in diagnostic['journal_parse_scope']
    assert owner._session.control_reserved_bytes() == 0
    for split, samples in ((0, train), (1, validation)):
        assert all(not view.flags.writeable for view in features[split])
        expected = np.stack([encode(sample.value, sample.action, encoder) for sample in samples])
        np.testing.assert_array_equal(np.concatenate(features[split]), expected)
