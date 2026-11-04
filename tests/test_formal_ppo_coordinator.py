"""Inert SAME-loop controls, no numerical child/game and no Source qualification."""
import asyncio
import copy
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.card_instance_formal import RUN_VARIANT
from gf_common.domain import ContractError, RuntimePolicyBinding, digest
from gf_common.schedule import serialize_schedule
from gf_models.arrays import sha256
from gf_training.card_instance_formal import FormalGameParent
from gf_training.card_instance_ppo import _GameOperations, train_formal_card_instance_ppo
from gf_training.collection import FinalizedCollection
from gf_training.journal import atomic_json
from tests.test_formal_ppo_control import base_identity, loaded


def inert_boundaries(monkeypatch, value, *, stop_elapsed=5401., cleanup_seconds=0., fail_cleanup=False, expected_formal=True, cleanup_unix_skew=0.):
    import gf_training.ppo_run as coordinator
    anchored_unix, anchored_mono = time.time(), time.monotonic()
    clocks = {'unix': anchored_unix, 'mono': anchored_mono}
    calls = []
    ledger = SimpleNamespace(workers={}, updating=False)
    fake_records = []
    monkeypatch.setattr(coordinator, 'time', SimpleNamespace(time=lambda: clocks['unix'], monotonic=lambda: clocks['mono']))
    monkeypatch.setattr(coordinator, 'owned_bytes', lambda *args: 1234)
    monkeypatch.setattr(coordinator, 'source_identity', lambda deadline: base_identity(value))
    monkeypatch.setattr(_GameOperations, 'preflight', lambda *args, **kwargs: calls.append('metadata-runtime-boundary-only'))
    monkeypatch.setattr(_GameOperations, 'integration_identity', lambda *args: base_identity(value)['integration'])

    class Child:
        def __init__(self):
            self.health = {}
        async def start(self):
            calls.append('inert-child-start-NOT-a-learner')
        async def initialize(self, seed, output, *, restore, deadline):
            calls.append(('initialize', seed, str(output), str(restore)))
            return {'hardware': {'component': 'inert-NOT-hardware'}, 'artifact': {'files': {}}}
        async def update(self, handle, output, *, deadline):
            calls.append('inert-update-NO-optimizer')
            return {'metrics': {'optimizer_steps': 4}, 'timings': {}, 'artifact': {'files': {}}}
        async def close(self, deadline):
            clocks['mono'] += cleanup_seconds
            clocks['unix'] += cleanup_seconds + cleanup_unix_skew
            self.health = {'exit_code': 0, 'stderr_bytes': 0}
            if fail_cleanup:
                raise OSError('injected inert cleanup failure')
    monkeypatch.setattr(_GameOperations, 'make_child', lambda *args, **kwargs: (Child(), ledger))
    monkeypatch.setattr(_GameOperations, 'validate_owned_collection', lambda *args, **kwargs: calls.append('inert-collection-boundary-NOT-source'))
    monkeypatch.setattr(_GameOperations, 'binding', lambda self, directory, generation, config, deadline:
        ('inert-model-' + str(generation), RuntimePolicyBinding('mlp', {'component-control-only': True})))
    def commit(self, directory, state, config, *, generation, update, controls, acknowledgement, deadline):
        assert (directory / 'formal-declaration.json').is_file() is expected_formal
        if expected_formal:
            assert state['identity']['formal_parent'] == state['formal_declaration']
        else:
            assert 'formal_parent' not in state['identity']
        state.update(updates=generation, generation_sha256=digest({'inert-generation': generation}))
        fake_records.append({'parent_controls': copy.deepcopy(controls)})
        atomic_json(directory / 'run.json', state)
        if generation:
            clocks['mono'] = anchored_mono + stop_elapsed
            clocks['unix'] = anchored_unix + stop_elapsed
        return {'component': 'not-a-generation-record'}
    monkeypatch.setattr(_GameOperations, 'commit', commit)
    monkeypatch.setattr(_GameOperations, 'retain', lambda *args: None)
    monkeypatch.setattr(_GameOperations, 'update_record', lambda self, update, state, files: update.update(schema_version=3, numeric_files=files))
    monkeypatch.setattr(_GameOperations, 'validate_parent', lambda self, directory, config, deadline, **kwargs:
        (json.loads((directory / 'run.json').read_text()), copy.deepcopy(fake_records)))
    @contextmanager
    def project(self, plan, numeric, config, learner_label, generation, deadline, **kwargs):
        numeric.mkdir()
        atomic_json(numeric / 'rollout.json', {'scope': 'inert-NOT-numeric4'})
        yield 1, {'games': 16, 'manifest_sha256': sha256(plan.output / 'manifest.json'), 'qualified': False,
            'parent_qualification': 'not_performed'}, {'rollout.json': sha256(numeric / 'rollout.json')}, object()
    monkeypatch.setattr(_GameOperations, 'project', project)
    async def collector(plan, actual_ledger):
        assert actual_ledger is ledger
        plan.output.mkdir()
        atomic_json(plan.output / 'manifest.json', {'scope': 'inert-NOT-C5/J6',
            'schedule': serialize_schedule(plan.groups, 4), 'policy_bindings': {key: asdict(v) for key, v in plan.bindings.items()}})
        return FinalizedCollection(plan.output, {'comparison_valid': True})
    return collector, calls, clocks, coordinator


@pytest.mark.parametrize('stop_elapsed', [5401., 7139.])
def test_explicit_formal_fresh_same_coordinator_captures_stop_before_cleanup_reserve(tmp_path, monkeypatch, stop_elapsed):
    optional_before = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    value = loaded(tmp_path)
    collector, calls, _clocks, _ = inert_boundaries(monkeypatch, value, stop_elapsed=stop_elapsed, cleanup_seconds=30.)
    parent = tmp_path / 'parent'
    result = asyncio.run(train_formal_card_instance_ppo(value, collector, parent))
    state = json.loads((parent / 'run.json').read_text())
    assert result['status'] == 'complete' and state['run_variant'] == RUN_VARIANT
    assert state['updates'] == 1 and state['formal_stop']['reason'] == 'admission_deadline'
    assert abs(state['formal_stop']['elapsed_seconds'] - stop_elapsed) < .001
    assert abs(state['elapsed_seconds'] - stop_elapsed - 30.) < .001
    assert state['formal_stop']['owned_bytes'] == 1234
    assert state['formal_stop']['clock']['observed_monotonic'] < state['formal_stop']['clock']['execution_deadline_monotonic']
    assert 'inert-update-NO-optimizer' in calls
    assert not (parent / 'generations/000000.json').exists()
    optional_after = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    assert optional_after.keys() == optional_before.keys()
    assert all(optional_after[name] is module for name, module in optional_before.items())
    with pytest.raises((ContractError, FileNotFoundError)), FormalGameParent(parent, max_owned_bytes=1024**3, deadline=time.monotonic() + 60):
        pytest.fail('inert controls cannot qualify as a real formal Source')


def test_formal_pause_is_not_complete_and_resume_uses_same_declaration(tmp_path, monkeypatch):
    value = loaded(tmp_path)
    collector, _, _clocks, _ = inert_boundaries(monkeypatch, value)
    parent = tmp_path / 'parent'
    paused = asyncio.run(train_formal_card_instance_ppo(value, collector, parent, pause_after_update=1))
    assert paused['status'] == 'paused'
    state = json.loads((parent / 'run.json').read_text())
    assert state['formal_stop'] is None and state['completed_updates'] is None
    declaration = (parent / 'formal-declaration.json').read_bytes()
    completed = asyncio.run(train_formal_card_instance_ppo(value, collector, parent, mode='resume'))
    assert completed['status'] == 'complete'
    assert (parent / 'formal-declaration.json').read_bytes() == declaration
    state2 = json.loads((parent / 'run.json').read_text())
    assert state2['identity']['formal_parent'] == state['identity']['formal_parent']
    assert state2['formal_stop']['generation'] == 1


def test_bad_cleanup_cannot_turn_normal_stop_into_healthy_formal_complete(tmp_path, monkeypatch):
    value = loaded(tmp_path)
    collector, _, _, _ = inert_boundaries(monkeypatch, value, fail_cleanup=True)
    parent = tmp_path / 'parent'
    with pytest.raises(OSError, match='cleanup failure'):
        asyncio.run(train_formal_card_instance_ppo(value, collector, parent))
    state = json.loads((parent / 'run.json').read_text())
    assert state['status'] == 'failed' and state['quarantine'] is True
    assert state['formal_stop']['reason'] == 'admission_deadline'


def test_outer_post_persistence_failure_invalidates_owned_complete_first(tmp_path, monkeypatch):
    import gf_training.card_instance_formal as formal
    value = loaded(tmp_path)
    collector, _, clocks, _coordinator = inert_boundaries(monkeypatch, value)
    real = formal.FormalCompletion.publish
    def late(self, directory, state, **kwargs):
        real(self, directory, state, **kwargs)
        clocks['mono'] = state['formal_stop']['clock']['wall_deadline_monotonic'] + 1
        clocks['unix'] += 2000
    monkeypatch.setattr(formal.FormalCompletion, 'publish', late)
    parent = tmp_path / 'parent'
    with pytest.raises((ContractError, RuntimeError)):
        asyncio.run(train_formal_card_instance_ppo(value, collector, parent))
    raw = (parent / 'run.json').read_bytes()
    assert raw.startswith(b'!') or json.loads(raw)['status'] == 'failed'


def test_ordinary_run3_does_not_gain_formal_fields_or_identity(tmp_path, monkeypatch):
    from gf_training.card_instance_formal import FormalControl
    from gf_training.card_instance_ppo import train_card_instance_ppo
    value = loaded(tmp_path)
    collector, _, _, _ = inert_boundaries(monkeypatch, value, expected_formal=False)
    def forbidden(*args):
        pytest.fail('ordinary Run3 entered formal preflight')
    monkeypatch.setattr(FormalControl, 'preflight', forbidden)
    parent = tmp_path / 'ordinary'
    result = asyncio.run(train_card_instance_ppo(value, collector, parent))
    state = json.loads((parent / 'run.json').read_text())
    assert result['status'] == 'complete' and state['schema_version'] == 3
    assert not ({'run_variant', 'formal_declaration', 'formal_stop'} & set(state))
    assert 'formal_parent' not in state['identity'] and not (parent / 'formal-declaration.json').exists()


@pytest.mark.parametrize('skew,accepted', [(-.05, True), (-.1, True), (-.1001, False)])
def test_inert_formal_close_keeps_original_coordinator_point_one_clock_tolerance(tmp_path, monkeypatch, skew, accepted):
    value = loaded(tmp_path)
    collector, _, _, _ = inert_boundaries(monkeypatch, value, cleanup_unix_skew=skew)
    parent = tmp_path / 'clock-control'
    if accepted:
        result = asyncio.run(train_formal_card_instance_ppo(value, collector, parent))
        state = json.loads((parent / 'run.json').read_text())
        assert result['status'] == 'complete'
        assert abs(state['formal_stop']['elapsed_seconds'] - 5401.) < .001
        assert abs(state['elapsed_seconds'] - (5401. + skew)) < .001
    else:
        with pytest.raises(ContractError, match='rollback'):
            asyncio.run(train_formal_card_instance_ppo(value, collector, parent))
        state = json.loads((parent / 'run.json').read_text())
        assert state['status'] == 'failed' and state['quarantine'] is True
