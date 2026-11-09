"""Actual wire3 numerical components; no Simulator or qualified Source."""
from __future__ import annotations

import asyncio
import copy
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError, canonical
from gf_models.arrays import sha256
from gf_models.sharded_arrays import read_bundle, read_json
from gf_models.structured_binding import model_defaults_v2
from gf_models.structured_config import InferenceBindingV2, resource_estimate
from gf_models.structured_runtime import (
    PROCESS_ENVELOPE_RECIPE,
    learner_process_bytes,
    own_worker_settings,
)
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import training_groups, validation_groups
from gf_training.structured_learning_process import StructuredLearningProcess
from gf_training.structured_numeric_worker import validate_reply
from gf_training.structured_rollout import write_rollout
from tests.test_structured_learning import learner_for, rollout_for


def process_for(tmp_path, learner, ledger, *, visibility=None):
    tmp_path.mkdir(exist_ok=True)
    return StructuredLearningProcess(tmp_path, 1024**2, learner.model, learner.binding,
        learner.action_config_hash, learner.resume_identity, action_contract=learner.action_contract,
        ledger=ledger, cuda_visible_devices=visibility, cleanup_deadline=time.monotonic() + 60)


def assert_artifacts_equal(left, right):
    for manifest_name, bundle_name in (('inference.json', 'weights'), ('resume.json', 'state')):
        a, b = read_json(left / manifest_name), read_json(right / manifest_name)
        arrays_a, arrays_b = read_bundle(left, a[bundle_name]), read_bundle(right, b[bundle_name])
        assert arrays_a.keys() == arrays_b.keys()
        for name in arrays_a:
            np.testing.assert_array_equal(arrays_a[name], arrays_b[name], err_msg=name)
        if manifest_name == 'resume.json':
            for name in ('parameter_states', 'shuffle_rng', 'optimizer_steps', 'cuda_rng_count'):
                assert canonical(a[name]) == canonical(b[name])


@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_actual_cpu_wire_initialize_two_updates_restore_and_close(tmp_path, make_input, kind, numeric_cpu_preload):
    learner = learner_for(kind, synthetic_capacity=False)
    run_lifecycle(tmp_path, make_input, learner)


def run_lifecycle(tmp_path, make_input, learner):
    ledger = ReservationLedger()
    expected_visibility = '' if learner.binding.device == 'cpu' else '0'
    expected_devices = 0 if learner.binding.device == 'cpu' else 1
    first = rollout_for(make_input, learner, count=2)
    write_rollout(tmp_path / 'batch0', first)

    async def run():
        child = process_for(tmp_path / 'logs0', learner, ledger, visibility='0')
        deadline = time.monotonic() + 90
        await child.start()
        try:
            initial = await child.initialize(17, tmp_path / 'g0', restore=None, deadline=deadline)
            assert initial['identity']['execution']['cuda_visible_devices'] == expected_visibility
            assert initial['hardware']['device_count'] == expected_devices
            assert ledger.learner is child.lease and ledger.cuda_bytes == learner_process_bytes(learner.model, learner.binding)
            assert not ledger.workers
            await child.update(tmp_path / 'batch0', sha256(tmp_path / 'batch0/rollout.json'),
                               first.policy_version, tmp_path / 'g1', deadline=deadline)
            learner.update(first)
            second = rollout_for(make_input, learner, count=2, episode='next')
            write_rollout(tmp_path / 'batch1', second)
            learner.update(second)
            learner.save(tmp_path / 'direct')
            await child.update(tmp_path / 'batch1', sha256(tmp_path / 'batch1/rollout.json'),
                               second.policy_version, tmp_path / 'g2', deadline=deadline)
            assert_artifacts_equal(tmp_path / 'direct', tmp_path / 'g2')
        finally:
            await child.close(deadline)
        assert child.process.returncode == 0 and child.lease.released and ledger.learner is None
        restored = process_for(tmp_path / 'logs1', learner, ledger, visibility='0')
        await restored.start()
        try:
            before = (tmp_path / 'g1/resume.json').read_bytes()
            reply = await restored.initialize(17, tmp_path / 'unused', restore=tmp_path / 'g1', deadline=deadline)
            assert reply['identity']['execution']['cuda_visible_devices'] == expected_visibility
            assert reply['hardware']['device_count'] == expected_devices
            assert reply['artifact']['optimizer_steps'] == 4
            assert not (tmp_path / 'unused').exists() and (tmp_path / 'g1/resume.json').read_bytes() == before
            await restored.update(tmp_path / 'batch1', sha256(tmp_path / 'batch1/rollout.json'),
                                  second.policy_version, tmp_path / 'resumed', deadline=deadline)
        finally:
            await restored.close(deadline)
        assert_artifacts_equal(tmp_path / 'g2', tmp_path / 'resumed')
        assert ledger.learner is None and not ledger.updating and ledger.cuda_bytes == 0
    asyncio.run(run())


def test_explicit_labels_preserve_original_schedule_rng_and_roles():
    old_rng, new_rng = np.random.default_rng(17), np.random.default_rng(17)
    old = training_groups(17, 3, 'learner', ('snapshot0', 'snapshot10'), old_rng)
    new = training_groups(17, 3, 'learner', ('snapshot0', 'snapshot10'), new_rng, actions_version=2)
    assert canonical(old_rng.bit_generator.state) == canonical(new_rng.bit_generator.state)
    assert len(new) == 8 and sum(len(group.episodes) for group in new) == 16
    mapping = {'random': 'random-structured-v1', 'heuristic': 'heuristic-structured-v1'}
    for before, after in zip(old, new, strict=True):
        assert before.seed == after.seed
        for a, b in zip(before.episodes, after.episodes, strict=True):
            assert asdict(a) | {'policies': tuple(mapping.get(label, label) for label in a.policies)} == asdict(b)
    assert validation_groups(17, 'learner', actions_version=2)[0].episodes[0].policies[1] == mapping['random']
    for invalid in (True, 0, 4, '2'):
        with pytest.raises(ContractError):
            training_groups(17, 0, 'x', ('x',), new_rng, actions_version=invalid)


def test_learner_reservation_pending_creation_and_confirmed_reap():
    ledger = ReservationLedger(100, 1)
    owner = SimpleNamespace(process=None)
    lease = ledger.reserve_learner(60, owner)
    lease.creation_pending = True
    assert ledger.cuda_bytes == 60 and not ledger.workers and lease._cleanup_owner is owner
    with pytest.raises(ContractError):
        lease.cancel_before_spawn()
    with pytest.raises(ContractError):
        ledger.reserve_learner(0, owner)
    worker = ledger.reserve_worker(40)
    with pytest.raises(ContractError):
        ledger.begin_update(0)
    worker.cancel_before_spawn()
    process = SimpleNamespace(pid=123, returncode=None)
    owner.process = process
    lease.spawned(process)
    with pytest.raises(ContractError):
        lease.reaped(process)
    ledger.begin_update(0)
    assert ledger.cuda_bytes == 60
    process.returncode = 0
    with pytest.raises(ContractError):
        lease.reaped(process)
    ledger.finish_update()
    lease.reaped(process)
    assert ledger.cuda_bytes == 0 and lease._cleanup_owner is None
    with pytest.raises(ContractError):
        ReservationLedger(100, learner_bytes=1).reserve_learner(1, owner)
    assert PROCESS_ENVELOPE_RECIPE == 'structured-process-envelope-v1'
    assert learner_process_bytes(learner_for(synthetic_capacity=False).model,
                                 learner_for(synthetic_capacity=False).binding) == 0


@pytest.mark.parametrize('mutation', ['version', 'identity', 'checksum', 'files', 'counter', 'hardware', 'timing'])
def test_acknowledgement_closed_and_strict(tmp_path, mutation):
    learner = learner_for(synthetic_capacity=False)
    child = process_for(tmp_path, learner, ReservationLedger())
    reply = {'protocol_version': 3, 'command': 'initialize', 'identity': child.identity,
        'artifact': {'inference_digest': 'a' * 64, 'resume_digest': 'b' * 64, 'optimizer_steps': 0,
            'files': {'inference.json': 'a' * 64, 'resume.json': 'b' * 64,
                'weights-000000.npz': 'c' * 64, 'weights-000000.json': 'd' * 64,
                'resume-000000.npz': 'e' * 64, 'resume-000000.json': 'f' * 64}},
        'hardware': {'device': 'cpu', 'name': 'cpu', 'cuda_runtime': None, 'device_count': 0,
                     'cuda_driver_api_version': None, 'exact_driver_build': 'unverified'},
        'timings': {'elapsed_seconds': 0., 'initialize_seconds': 0., 'save_seconds': 0., 'peak_cpu_memory_bytes': 1}}
    validate_reply(reply, 'initialize', child.identity)
    bad = copy.deepcopy(reply)
    if mutation == 'version':
        bad['protocol_version'] = 3.0
    elif mutation == 'identity':
        bad['identity']['execution']['binding']['candidate_chunk'] += 1
    elif mutation == 'checksum':
        bad['artifact']['inference_digest'] = None
    elif mutation == 'files':
        del bad['artifact']['files']['resume-000000.json']
    elif mutation == 'counter':
        bad['artifact']['optimizer_steps'] = True
    elif mutation == 'hardware':
        bad['hardware']['device_count'] = 1
    else:
        bad['timings']['save_seconds'] = 10**400
    with pytest.raises(ContractError):
        validate_reply(bad, 'initialize', child.identity)


def test_wire_metadata_imports_are_optional_free_and_default_process_environment_unchanged():
    code = '''
import sys
from pathlib import Path
from gf_training.structured_learning_process import StructuredLearningProcess
from gf_training.structured_numeric_worker import wire_identity
from gf_training.ppo_run import LearningProcess
assert "torch" not in sys.modules
assert "CUDA_VISIBLE_DEVICES" not in LearningProcess(Path("."), 1)._environment()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_full_lifetime_envelopes_fit_four_default_workers_and_learner():
    from gf_common.action_contract import engineering_contract
    from gf_common.structured import ActionConfigV2
    actions = ActionConfigV2(action_contract=engineering_contract())
    for kind in ('mlp', 'gru', 'transformer'):
        model = model_defaults_v2(kind).architecture()
        binding = InferenceBindingV2(device='cuda:0')
        options = {'version': '2', 'checkpoint': '/component-only', 'expected_manifest_digest': 'a' * 64,
            'model': asdict(model), 'inference': asdict(binding), 'cuda_visible_devices': None,
            'action_contract': asdict(actions.action_contract), 'purpose': 'component', 'synthetic_capacity': False}
        _, _, resident, temporary = own_worker_settings(kind, actions, options)
        assert resident == resource_estimate(model, binding, suffix_decisions=1)['total_bytes']
        assert temporary == 0
        assert learner_process_bytes(model, binding) == resource_estimate(model, binding)['total_bytes']
        assert 4 * resident + learner_process_bytes(model, binding) <= 12 * 1024**3
        if kind == 'transformer':
            assert model.temporal.num_layers == 2


def test_cancellation_during_shielded_creation_retains_owner_until_real_child_reap(tmp_path, monkeypatch):
    original = asyncio.create_subprocess_exec
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    actual = []

    async def run():
        returned, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            process = await original(*args, **kwargs)
            actual.append(process)
            returned.set()
            await release.wait()
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
        child = process_for(tmp_path, learner, ledger)
        starting = asyncio.create_task(child.start())
        try:
            await asyncio.wait_for(returned.wait(), 10)
            starting.cancel()
            await asyncio.sleep(0)
            assert child.process is None and child.lease.creation_pending
            assert ledger.learner._cleanup_owner is child
            with pytest.raises(ContractError):
                child.lease.cancel_before_spawn()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await starting
            assert actual[0].returncode is not None and ledger.learner is None
            assert child.lease.released
        finally:
            release.set()
            await asyncio.gather(starting, return_exceptions=True)
            for process in actual:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
    asyncio.run(run())


@pytest.mark.parametrize('registered', [False, True])
@pytest.mark.parametrize('failure', [MemoryError, asyncio.CancelledError])
def test_registration_failure_reaps_returned_real_child_before_release(tmp_path, monkeypatch, registered, failure):
    from gf_simulator_client.structured_resources import LearnerReservation
    original = LearnerReservation.spawned
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    child = process_for(tmp_path, learner, ledger)
    def fail(self, process):
        assert process.returncode is None and ledger.learner is self
        if registered:
            original(self, process)
        raise failure('injected registration failure')
    monkeypatch.setattr(LearnerReservation, 'spawned', fail)
    async def run():
        try:
            with pytest.raises(failure, match='injected registration failure'):
                await child.start()
            assert child.process.returncode is not None
            assert ledger.learner is None and child.lease.released
        finally:
            if child.process is not None and child.process.returncode is None:
                child.process.kill()
                await child.process.wait()
    asyncio.run(run())


@pytest.mark.parametrize(('device', 'visibility'), [('cpu', ''), ('cuda:0', None), ('cuda:0', '0')])
def test_visibility_selected_before_any_numeric_or_torch_import(tmp_path, device, visibility):
    import json
    from dataclasses import replace

    learner = learner_for(synthetic_capacity=False)
    binding = replace(learner.binding, device=device)
    child = StructuredLearningProcess(tmp_path, 1, learner.model, binding,
        learner.action_config_hash, learner.resume_identity, action_contract=learner.action_contract,
        ledger=ReservationLedger(), cuda_visible_devices=visibility, cleanup_deadline=time.monotonic() + 30)
    initial = {'protocol_version': 3, 'command': 'initialize', 'identity': child.identity,
               'model': asdict(learner.model), 'seed': 17, 'output': 'never-written', 'restore': None}
    code = '''
import builtins, json, os, sys
from gf_training.structured_numeric_worker import serve
assert 'torch' not in sys.modules
os.environ['CUDA_VISIBLE_DEVICES'] = 'must-not-inherit'
initial = json.loads(sys.argv[1])
expected = initial['identity']['execution']['cuda_visible_devices']
original = builtins.__import__
class Checked(Exception): pass
def guarded(name, *args, **kwargs):
    if name == 'gf_training.structured_numeric':
        assert 'torch' not in sys.modules
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected
        raise Checked()
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    serve(initial, ())
except Checked:
    pass
else:
    raise AssertionError('numerical import was not reached')
'''
    result = subprocess.run([sys.executable, '-c', code, json.dumps(initial)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_bad_reply_never_advances_parent_and_retains_update_scope_until_reap(tmp_path, monkeypatch):
    from gf_training.ppo_run import LearningProcess
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    original = LearningProcess.request
    child = process_for(tmp_path, learner, ledger)
    async def bad(self, request, deadline):
        if request['command'] == 'update':
            return {'protocol_version': 3.0, 'command': 'update', 'identity': child.identity}
        return await original(self, request, deadline)
    async def run():
        deadline = time.monotonic() + 30
        await child.start()
        try:
            await child.initialize(17, tmp_path / 'g0', restore=None, deadline=deadline)
            monkeypatch.setattr(LearningProcess, 'request', bad)
            with pytest.raises(ContractError):
                await child.update(tmp_path / 'batch', 'a' * 64, 'component-policy', tmp_path / 'g1', deadline=deadline)
            assert child.steps == 0 and ledger.updating and ledger.learner is child.lease
            assert not (tmp_path / 'g1').exists() and child.process.returncode is None
        finally:
            await child.close(deadline)
        assert child.process.returncode == 0 and ledger.learner is None and not ledger.updating
    asyncio.run(run())


def test_creation_failure_rolls_back_only_after_creation_is_resolved(tmp_path, monkeypatch):
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    child = process_for(tmp_path, learner, ledger)
    async def failed(*args, **kwargs):
        assert ledger.learner is child.lease and child.lease.creation_pending
        raise FileNotFoundError('injected creation failure')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', failed)
    async def run():
        with pytest.raises(FileNotFoundError, match='injected creation failure'):
            await child.start()
        assert child.creation_task.done() and child.process is None
        assert child.lease.released and ledger.learner is None
    asyncio.run(run())


def test_unavailable_configured_cuda_has_no_cpu_fallback_and_reaps(tmp_path):
    from dataclasses import replace
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    binding = replace(learner.binding, device='cuda:0')
    child = StructuredLearningProcess(tmp_path, 1024**2, learner.model, binding,
        learner.action_config_hash, learner.resume_identity, action_contract=learner.action_contract,
        ledger=ledger, cuda_visible_devices='', cleanup_deadline=time.monotonic() + 30)
    async def run():
        deadline = time.monotonic() + 30
        await child.start()
        assert ledger.cuda_bytes == learner_process_bytes(learner.model, binding) > 0
        try:
            with pytest.raises(RuntimeError, match='numerical child failed'):
                await child.initialize(17, tmp_path / 'no-output', restore=None, deadline=deadline)
            assert child.steps is None and not (tmp_path / 'no-output').exists()
        finally:
            with pytest.raises((RuntimeError, BrokenPipeError, ConnectionResetError)):
                await child.close(deadline)
        assert child.process.returncode is not None and child.process.returncode != 0
        assert ledger.learner is None and child.lease.released and ledger.cuda_bytes == 0
    asyncio.run(run())


@pytest.mark.parametrize('cancel_start', [False, True])
def test_public_close_retries_expired_pending_creation_with_singleflight_real_reap(tmp_path, monkeypatch, cancel_start):
    from gf_simulator_client.structured_resources import LearnerReservation
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    child = process_for(tmp_path, learner, ledger)
    original_create = asyncio.create_subprocess_exec
    original_release = LearnerReservation._release
    original_close = child._close
    actual, releases, attempts = [], [], []

    def counted_release(self):
        assert self._cleanup_owner.process.returncode is not None
        releases.append(self)
        original_release(self)

    async def counted_close(deadline):
        attempts.append(deadline)
        await original_close(deadline)

    monkeypatch.setattr(LearnerReservation, '_release', counted_release)
    monkeypatch.setattr(child, '_close', counted_close)

    async def run():
        created, allow_return = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            process = await original_create(*args, **kwargs)
            actual.append(process)
            created.set()
            await allow_return.wait()
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed)
        starting = asyncio.create_task(child.start())
        retries = []
        try:
            await asyncio.wait_for(created.wait(), 10)
            if cancel_start:
                starting.cancel('original startup cancellation')
                await asyncio.sleep(0)
            with pytest.raises(TimeoutError):
                await child.close(time.monotonic() + .01)
            failed_attempt = child.close_task
            assert failed_attempt.done() and isinstance(failed_attempt.exception(), TimeoutError)
            assert child.process is None and child.lease.creation_pending and not releases
            assert ledger.learner._cleanup_owner is child and actual[0].returncode is None
            with pytest.raises(ContractError):
                child.lease.cancel_before_spawn()
            deadline = time.monotonic() + 20
            retries = [asyncio.create_task(child.close(deadline)), asyncio.create_task(child.close(deadline))]
            await asyncio.sleep(0)
            retry_attempt = child.close_task
            assert retry_attempt is not failed_attempt and not retry_attempt.done()
            allow_return.set()
            # An uninitialized wire3 child cannot acknowledge close; owned
            # cleanup still must kill/reap it, retaining unsuccessful health.
            results = await asyncio.gather(*retries, return_exceptions=True)
            assert all(isinstance(result, (RuntimeError, BrokenPipeError, ConnectionResetError)) for result in results)
            if cancel_start:
                with pytest.raises(asyncio.CancelledError, match='original startup cancellation'):
                    await starting
            else:
                await starting
            assert child.close_task is retry_attempt and len(attempts) == 2
            assert actual[0].returncode is not None and len(releases) == 1
            assert child.lease.released and ledger.learner is None and ledger.cuda_bytes == 0
        finally:
            allow_return.set()
            await asyncio.gather(starting, *retries, return_exceptions=True)
            for process in actual:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
    asyncio.run(run())


def test_close_before_start_is_terminal_without_reservation_or_creation(tmp_path, monkeypatch):
    learner = learner_for(synthetic_capacity=False)
    ledger = ReservationLedger()
    child = process_for(tmp_path, learner, ledger)
    calls = []
    async def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('closed owner must not create a child')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    async def run():
        await child.close(time.monotonic() + 1)
        with pytest.raises(ContractError, match='closed'):
            await child.start()
        await child.close(time.monotonic() + 1)
        assert not calls and child.process is None and child.lease is None and ledger.learner is None
    asyncio.run(run())
