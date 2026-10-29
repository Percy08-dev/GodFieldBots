"""Exclusive construction accounting and frozen data/final metric snapshots."""
import json
from types import SimpleNamespace

import pytest

from gf_training.native_authorization import NativePreparation
from gf_training.native_preparation import _PreparationWall


def test_preparation_wall_nested_exception_restores_and_reconciles():
    now = [0]
    wall = _PreparationWall(lambda: now[0])
    now[0] = 10
    with pytest.raises(RuntimeError), wall.phase('owner_session_setup'):
        now[0] = 20
        with wall.phase('proof_and_initialization'):
            now[0] = 35
            raise RuntimeError('nested failure')
    assert wall._current == 'other'
    now[0] = 40
    result = wall.finish(success=False)
    parts = result['partition_nanoseconds']
    assert parts['owner_session_setup'] == 10
    assert parts['proof_and_initialization'] == 15
    assert parts['other'] == 15
    assert sum(parts.values()) == result['parent_nanoseconds'] == 40
    assert result['residual_nanoseconds'] == 0
    assert result['success'] is False
    now[0] = 1000
    assert wall.finish(success=True) == result  # Later finalization cannot extend it.


def test_native_data_snapshot_stays_separate_from_later_finalization():
    payload = {'phase_seconds': {'finalize_or_abort': 0},
               'coordinator_scan_wall': {'parent_nanoseconds': 10}}
    owner = NativePreparation.__new__(NativePreparation)
    owner.check = lambda: None
    owner.started = 0
    owner.diagnostics = {}
    owner._session = SimpleNamespace(telemetry=lambda: json.dumps(payload), peak_bytes=lambda: 123)
    owner._executor = lambda: SimpleNamespace(call=lambda function: function())
    stage = owner.data_stage_snapshot()
    owner.diagnostics['native_data_stage_snapshot'] = stage
    payload['phase_seconds']['finalize_or_abort'] = 8
    owner._metrics('complete')
    assert stage['phase_seconds']['finalize_or_abort'] == 0
    assert stage['snapshot_stage'] == 'data_stage_complete_before_archive_and_finalization'
    final = owner.diagnostics['native_final_snapshot']
    assert final['phase_seconds']['finalize_or_abort'] == 8
    assert final['snapshot_stage'] == 'after_finalization_or_abort'
    assert stage['coordinator_scan_wall'] == final['coordinator_scan_wall']
