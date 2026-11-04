"""Controlled metadata/FD ownership tests; NEVER eligible original game Sources."""
import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import LoadedConfig, load_config
from gf_common.card_instance_formal import validate_formal_state
from gf_common.domain import ContractError, canonical, digest
from gf_models.arrays import sha256
from gf_models.structured_sealed import SealedScan
from gf_training.card_instance_formal import (
    FormalControl,
    FormalGameParent,
    load_recipe,
    read_formal_control,
)
from gf_training.card_instance_ppo import _GameOperations, train_formal_card_instance_ppo
from gf_training.journal import atomic_json
from gf_training.lineage import _OWNED_LOCKS, RunOwnership, run_lock
from tests.test_formal_ppo_contract import config_for

ROOT = Path(__file__).resolve().parents[1]


def loaded(tmp_path):
    expected = config_for()
    server = tmp_path / 'original-server-input.toml'
    server.write_bytes((ROOT / 'configs/test-server.toml').read_bytes())
    expected['server']['config'] = str(server)
    value = load_config(ROOT / 'configs/experiments/card-instance-mlp-ppo-acceptance.toml',
        {key: expected[key] for key in ('run', 'server', 'training')})
    return LoadedConfig(value.config, asdict(value.config), digest(asdict(value.config)), value.inputs)


def base_identity(value):
    return {'code': {'scope': 'component-controlled-metadata-NOT-executed-source'},
        'configuration_inputs_sha256': digest(value.inputs),
        'integration': {'server_config_sha256': load_recipe()['server_config_sha256']}}


def control_state(value, identity, execution):
    return {'schema_version': 3, 'algorithm': 'ppo', 'run_id': 'component-controls-only-not-source', 'status': 'running',
        'effective_config': value.effective, 'config_sha256': value.sha256, 'config_inputs': list(value.inputs),
        'identity': identity, 'execution': execution, 'seed': 17, 'device': 'cuda:0',
        'created_unix': 1_000_000., 'last_unix': 1_000_000., 'wall_deadline_unix': 1_007_200.,
        'admission_deadline_unix': 1_005_400., 'execution_deadline_unix': 1_007_140.,
        'updates': 1, 'completed_updates': None, 'optimizer_steps': 4, 'accepted_decisions': 1,
        'generation_sha256': 'b' * 64, 'elapsed_seconds': 5500.}


def prepared_control(tmp_path):
    value = loaded(tmp_path)
    control = FormalControl(value)
    control.preflight()
    directory = tmp_path / 'parent'
    return value, control, directory


def initialize(control, value, directory, ownership):
    identity = base_identity(value)
    execution = _GameOperations().execution(value.config)
    control.attach_identity(directory, identity, execution, mode='fresh', deadline=time.monotonic() + 60, ownership=ownership)
    state = control_state(value, identity, execution)
    control.initialize_state(state)
    return state


def complete_state(control, state):
    assert control.observe_stop(state, 1_000_000., 1000., 6401., 1234)
    state.update(status='complete', completed_updates=1, last_unix=state['created_unix'] + state['elapsed_seconds'])
    validate_formal_state(state)
    return state


def test_declaration_created_before_g0_and_binds_base_identity_without_cycle(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        declaration = json.loads((directory / 'formal-declaration.json').read_text())
        assert not (directory / 'artifacts').exists()
        assert set(directory.iterdir()) == {directory / '.lock', directory / 'formal-declaration.json'}
        assert declaration['identity_sha256'] == digest(base_identity(value))
        assert declaration['execution_identity_sha256'] == digest(state['execution'])
        assert 'formal_parent' not in base_identity(value)
        assert state['formal_declaration'] == state['identity']['formal_parent']
        assert state['formal_declaration']['sha256'] == sha256(directory / 'formal-declaration.json')
        scan = SealedScan(max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
        read_formal_control(directory, state, scan)
    control.finish()
    assert not control.completion.exposed


@pytest.mark.parametrize('entry', ['run.json', 'artifacts'])
def test_fresh_declaration_cannot_backfill_existing_parent_even_with_real_lock(tmp_path, entry):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        if entry == 'artifacts':
            (directory / entry).mkdir()
        else:
            (directory / entry).write_text('{}')
        with pytest.raises(ContractError, match='backfill'):
            initialize(control, value, directory, ownership)
        assert not (directory / 'formal-declaration.json').exists()


def test_copied_or_stale_lock_cannot_publish_declaration(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    directory.mkdir()
    with pytest.raises(ContractError, match='SAME'):
        initialize(control, value, directory, RunOwnership(directory, 'copied-token'))
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize('mutation', ['bytes', 'reference', 'both-references', 'base', 'execution', 'input'])
def test_original_declaration_identity_mutations_refuse(tmp_path, mutation):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        if mutation == 'bytes':
            path = directory / 'formal-declaration.json'
            path.write_bytes(path.read_bytes() + b'\n')
        elif mutation == 'reference':
            state['identity']['formal_parent']['seed_slot'] = 29
        elif mutation == 'both-references':
            state['formal_declaration']['seed_slot'] = 29
            state['identity']['formal_parent']['seed_slot'] = 29
        elif mutation == 'base':
            state['identity']['code']['changed'] = True
        elif mutation == 'execution':
            state['execution']['learner']['binding']['candidate_chunk'] += 1
        else:
            state['config_inputs'] = []
        with pytest.raises(ContractError):
            read_formal_control(directory, state, SealedScan(max_owned_bytes=1024**3, deadline=time.monotonic() + 60))


def test_resume_reuses_exact_sidecar_and_changed_config_or_base_cannot_replace_it(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        state['status'] = 'paused'
        atomic_json(directory / 'run.json', state)
    original = (directory / 'formal-declaration.json').read_bytes()
    resumed = FormalControl(value)
    resumed.preflight()
    with run_lock(directory) as ownership:
        identity = base_identity(value)
        resumed.attach_identity(directory, identity, state['execution'], mode='resume', deadline=time.monotonic() + 60, ownership=ownership)
        assert identity == state['identity']
        assert original == (directory / 'formal-declaration.json').read_bytes()
        wrong = FormalControl(value)
        wrong.preflight()
        with pytest.raises(ContractError, match='base'):
            wrong.attach_identity(directory, {**base_identity(value), 'foreign': True}, state['execution'], mode='resume', deadline=time.monotonic() + 60, ownership=ownership)


def test_complete_owned_inode_is_kept_across_lock_exit_and_invalidated_on_late_failure(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        assert control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
        assert control.completion.exposed and not control.completion.committed
        assert json.loads((directory / 'run.json').read_text())['status'] == 'complete'
    assert directory not in _OWNED_LOCKS
    initiating = OSError('injected late outer run-lock/report failure')
    control.fail(initiating)
    assert (directory / 'run.json').read_bytes().startswith(b'!')
    assert control.completion.revoked and not control.completion.committed
    assert str(initiating) == 'injected late outer run-lock/report failure'


def test_success_releases_only_after_outer_lock_exit_and_inert_control_is_not_a_source(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    control.finish()
    assert control.completion.committed and control.completion.fd is None and control.completion.directory_fd is None
    with pytest.raises((ContractError, FileNotFoundError)), FormalGameParent(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 60):
        pytest.fail('controlled metadata must not qualify without actual original Sources/artifacts')


def test_foreign_current_completion_is_preserved_while_known_original_fd_is_invalidated(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
        replacement = directory / 'foreign.json'
        replacement.write_text('foreign-current-path')
        os.replace(replacement, directory / 'run.json')
    initiating = RuntimeError('namespace substituted')
    control.fail(initiating)
    assert (directory / 'run.json').read_text() == 'foreign-current-path'
    assert control.completion.revoked


@pytest.mark.parametrize('role', ['directory', 'completion'])
def test_single_attempt_close_uncertainty_never_retries_or_inspects_stale_descriptor(tmp_path, monkeypatch, role):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    owner = control.completion
    targeted = owner.directory_fd if role == 'directory' else owner.fd
    actual_close = os.close
    events = []
    def fail_once(descriptor):
        events.append(descriptor)
        if descriptor == targeted and events.count(descriptor) == 1:
            actual_close(descriptor)
            raise OSError('injected close already took effect')
        return actual_close(descriptor)
    monkeypatch.setattr(os, 'close', fail_once)
    with pytest.raises(OSError, match='already took effect') as caught:
        control.finish()
    control.fail(caught.value)
    assert owner.revoked and owner._uncertain_releases
    assert caught.value._formal_completion_cleanup_owner is control
    assert (directory / 'run.json').read_bytes().startswith(b'!')


def test_mismatching_formal_recipe_refuses_before_any_model_runtime_lease_or_output(tmp_path, monkeypatch):
    optional_before = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    value = loaded(tmp_path)
    wrong = copy.deepcopy(value.effective)
    wrong['training']['max_updates'] = 2
    malformed = SimpleNamespace(config=value.config, effective=wrong, sha256=digest(wrong), inputs=value.inputs)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid formal configuration reached model/runtime allocation')
    monkeypatch.setattr(_GameOperations, 'validate_config', forbidden)
    target = tmp_path / 'never-output'
    with pytest.raises(ContractError, match='frozen literal'):
        asyncio.run(train_formal_card_instance_ppo(malformed, forbidden, target))
    assert not target.exists()
    optional_after = {name: module for name, module in sys.modules.items()
        if name.split('.')[0] in ('torch','lightgbm')}
    assert optional_after.keys() == optional_before.keys()
    assert all(optional_after[name] is module for name, module in optional_before.items())


def test_terminal_after_close_guard_uses_fresh_known_inode_to_revoke(tmp_path, monkeypatch):
    import gf_training.ppo_run as coordinator
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    calls = []
    def storage(*args):
        calls.append(1)
        return 0 if len(calls) == 1 else value.config.training.max_owned_bytes + 1
    monkeypatch.setattr(coordinator, 'owned_bytes', storage)
    with pytest.raises(ContractError, match='outer completion') as error:
        control.finish()
    assert control.completion.fd is None and control.completion.directory_fd is None
    control.fail(error.value)
    assert control.completion.revoked and (directory / 'run.json').read_bytes().startswith(b'!')


@pytest.mark.parametrize('role', ['directory', 'completion'])
def test_close_error_reused_foreign_fd_is_never_inspected_or_closed_again(tmp_path, monkeypatch, role):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    owner = control.completion
    target = owner.directory_fd if role == 'directory' else owner.fd
    foreign_path = tmp_path / 'foreign-fd-file'
    foreign_path.write_bytes(b'foreign')
    real_close, real_fstat = os.close, os.fstat
    foreign = []
    def close(descriptor):
        if descriptor == target and not foreign:
            real_close(descriptor)
            replacement = os.open(foreign_path, os.O_RDONLY)
            if replacement != descriptor:
                os.dup2(replacement, descriptor)
                real_close(replacement)
            foreign.append(descriptor)
            raise OSError('closed then descriptor reused by foreign object')
        if foreign and descriptor == foreign[0]:
            pytest.fail('uncertain stale descriptor retried as foreign capability')
        real_close(descriptor)
    def fstat(descriptor):
        if foreign and descriptor == foreign[0]:
            pytest.fail('uncertain stale descriptor inspected as foreign capability')
        return real_fstat(descriptor)
    monkeypatch.setattr(os, 'close', close)
    monkeypatch.setattr(os, 'fstat', fstat)
    try:
        with pytest.raises(OSError, match='reused') as error:
            control.finish()
        control.fail(error.value)
        assert owner.revoked and error.value._formal_completion_cleanup_owner is control
        assert foreign_path.read_bytes() == b'foreign'
        assert os.read(foreign[0], 7) == b'foreign'
    finally:
        if foreign:
            real_close(foreign[0])


def test_failed_invalidation_preserves_primary_and_reachable_owner(tmp_path, monkeypatch):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    original = RuntimeError('first reportable failure')
    real = os.pwrite
    def fail(*args):
        raise OSError('revocation unconfirmed')
    monkeypatch.setattr(os, 'pwrite', fail)
    control.fail(original)
    assert str(original) == 'first reportable failure'
    assert original._formal_completion_cleanup_owner is control
    assert not control.completion.committed and not control.completion.revoked
    monkeypatch.setattr(os, 'pwrite', real)
    control.completion.revoke()
    assert (directory / 'run.json').read_bytes().startswith(b'!')


def test_sidecar_input_snapshot_survives_resume_and_cannot_rebind_same_bytes(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        initialize(control, value, directory, ownership)
    path = directory / 'formal-declaration.json'
    replacement = directory / 'replacement'
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(ContractError):
        control.finish()


def test_formal_context_type_and_original_caller_cannot_be_crossed(tmp_path):
    with pytest.raises(ContractError, match='concrete'):
        _GameOperations(formal=SimpleNamespace(preflight=lambda *args: None))
    value = loaded(tmp_path)
    control = FormalControl(value)
    with pytest.raises(ContractError, match='SAME original'):
        control.preflight(copy.deepcopy(value))


def test_failure_note_allocation_cannot_replace_primary_or_hide_uncertain_owner(tmp_path, monkeypatch):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        state = initialize(control, value, directory, ownership)
        atomic_json(directory / 'run.json', state)
        complete_state(control, state)
        control.persist(directory / 'run.json', state, deadline=time.monotonic() + 60, max_owned_bytes=1024**3)
    class Primary(RuntimeError):
        def add_note(self, note):
            raise MemoryError('injected note allocation')
    primary = Primary('preserve this exact primary')
    real = os.pwrite
    monkeypatch.setattr(os, 'pwrite', lambda *args: (_ for _ in ()).throw(OSError('unconfirmed')))
    control.fail(primary)
    assert primary._formal_completion_cleanup_owner is control
    assert str(primary) == 'preserve this exact primary'
    monkeypatch.setattr(os, 'pwrite', real)
    control.completion.revoke()


def test_formal_parent_borrows_only_explicit_same_lock_and_never_releases_outer_lock(tmp_path):
    directory = tmp_path / 'ordinary'
    with run_lock(directory, create=True) as ownership:
        atomic_json(directory / 'run.json', {'scope': 'ordinary-not-formal'})
        with pytest.raises(ContractError, match='SAME'), FormalGameParent(directory, max_owned_bytes=1024**3, deadline=time.monotonic() + 60):
            pytest.fail('implicit borrowed lock accepted')
        assert _OWNED_LOCKS[directory] == ownership.token
        with pytest.raises(ContractError, match='ordinary'), FormalGameParent(directory, ownership=ownership, max_owned_bytes=1024**3, deadline=time.monotonic() + 60):
            pytest.fail('ordinary parent with real borrowed lock became formal')
        assert _OWNED_LOCKS[directory] == ownership.token


def test_whole_formal_invocation_holds_original_server_config_identity(tmp_path):
    value, control, directory = prepared_control(tmp_path)
    with run_lock(directory, create=True) as ownership:
        initialize(control, value, directory, ownership)
    path = Path(value.config.server.config)
    replacement = path.with_suffix('.same-bytes')
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(ContractError):
        control.finish()


def unadmitted_parent_lifecycle(directory, ownership, monkeypatch):
    """PRIVATE lifecycle primitive only: no Source admission/qualifier is run."""
    import gf_training.card_instance_source as module
    owner = FormalGameParent(directory, ownership=ownership, max_owned_bytes=1024**3, deadline=time.monotonic() + 60)
    state = {'scope': 'UNADMITTED-lifecycle-NOT-a-formal-Source', 'updates': 7, 'nested': {'value': 1}}
    records = [{'generation': 0}, {'generation': 7, 'inference_sha256': 'a' * 64}]
    owner._state_bytes = canonical(state).encode()
    owner._records_bytes = canonical(records).encode()
    owner._active, owner._entered, owner._token = True, True, ownership.token
    owner.scan = SimpleNamespace(scope='UNADMITTED-mocked-file-guard',retained=0,files={},inventories={})
    checked = []
    def check(scan):
        assert scan is owner.scan
        checked.append(True)
    def binding(path, actual, record, generation):
        return copy.deepcopy((actual, record, generation))
    monkeypatch.setattr(module, 'recheck_game_parent_snapshot', check)
    monkeypatch.setattr(module, '_binding', binding)
    return owner, state, records, checked


def test_unadmitted_lifecycle_public_state_records_are_clones_and_binding_uses_protected_originals(tmp_path, monkeypatch):
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, state, records, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        public_state, public_records = owner.state, owner.records
        public_state['updates'] = 999
        public_state['nested']['value'] = 999
        public_records[-1]['generation'] = 999
        public_records[-1]['inference_sha256'] = 'b' * 64
        assert owner.state == state and owner.records == records
        with pytest.raises(AttributeError):
            owner.state = public_state
        with pytest.raises(AttributeError):
            owner.records = public_records
        actual, record, generation = owner.final_binding()
        assert actual == state and record == records[-1] and generation == 7
        actual['nested']['value'] = 555
        record['generation'] = 555
        assert owner.final_binding() == (state, records[-1], 7)
        owner.__exit__(None, None, None)
        assert _OWNED_LOCKS[directory] == ownership.token
        with pytest.raises(ContractError):
            owner.final_binding()
        with pytest.raises(ContractError, match='one lifetime'):
            owner.__enter__()
        assert not (directory / 'formal-declaration.json').exists()
        assert not (directory / 'generations').exists()


def test_unadmitted_lifecycle_recheck_failure_poison_survives_repair_and_closes_before_success(tmp_path, monkeypatch):
    import gf_training.card_instance_source as module
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, _, _, checked = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        first = ContractError('first original-snapshot uncertainty')
        monkeypatch.setattr(module, 'recheck_game_parent_snapshot', lambda scan: (_ for _ in ()).throw(first))
        with pytest.raises(ContractError) as caught:
            owner.recheck()
        assert caught.value is first
        monkeypatch.setattr(module, 'recheck_game_parent_snapshot', lambda scan: checked.append('repaired-only-for-cleanup'))
        with pytest.raises(ContractError, match='poisoned'):
            owner.final_binding()
        with pytest.raises(ContractError, match='poisoned'):
            owner.recheck()
        with pytest.raises(ContractError) as closed:
            owner.__exit__(None, None, None)
        assert closed.value is first and not owner._active and owner._closed
        assert _OWNED_LOCKS[directory] == ownership.token
        with pytest.raises(ContractError, match='one lifetime'):
            owner.__enter__()


def test_unadmitted_lifecycle_binding_error_and_active_reentry_are_sticky(tmp_path, monkeypatch):
    import gf_training.card_instance_source as module
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, _, _, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        first = ValueError('binding derivation failed')
        monkeypatch.setattr(module, '_binding', lambda *args: (_ for _ in ()).throw(first))
        with pytest.raises(ValueError) as caught:
            owner.final_binding()
        assert caught.value is first
        monkeypatch.setattr(module, '_binding', lambda *args: 'repaired')
        with pytest.raises(ContractError, match='poisoned'):
            owner.final_binding()
        with pytest.raises(ValueError):
            owner.__exit__(None, None, None)
        other, _, _, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        with pytest.raises(ContractError, match='one lifetime'):
            other.__enter__()
        with pytest.raises(ContractError, match='poisoned'):
            other.final_binding()
        with pytest.raises(ContractError):
            other.__exit__(None, None, None)


def test_unadmitted_lifecycle_cleanup_uncertainty_retains_same_owner(tmp_path, monkeypatch):
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, _, _, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        first = OSError('lock close uncertain')
        monkeypatch.setattr(owner._scope, 'close', lambda: (_ for _ in ()).throw(first))
        with pytest.raises(OSError) as error:
            owner.__exit__(None, None, None)
        assert error.value is first and error.value._formal_parent_cleanup_owner is owner
        with pytest.raises(ContractError, match='poisoned'):
            owner.final_binding()
        assert _OWNED_LOCKS[directory] == ownership.token


def test_unadmitted_lifecycle_lock_mismatch_on_exit_is_sticky(tmp_path, monkeypatch):
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, _, _, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        _OWNED_LOCKS[directory] = 'foreign-token'
        try:
            with pytest.raises(ContractError, match='lock required') as error:
                owner.__exit__(None, None, None)
        finally:
            _OWNED_LOCKS[directory] = ownership.token
        assert owner._failure is error.value and not owner._active
        with pytest.raises(ContractError, match='poisoned'):
            owner.final_binding()


def test_unadmitted_lifecycle_binding_copy_allocation_error_is_sticky(tmp_path, monkeypatch):
    import gf_training.card_instance_formal as module
    directory = tmp_path / 'unadmitted-lifecycle'
    with run_lock(directory, create=True) as ownership:
        owner, _, _, _ = unadmitted_parent_lifecycle(directory, ownership, monkeypatch)
        first = MemoryError('private binding copy allocation failed')
        original = module.json
        monkeypatch.setattr(module, 'json', SimpleNamespace(loads=lambda *args: (_ for _ in ()).throw(first)))
        with pytest.raises(MemoryError) as error:
            owner.final_binding()
        monkeypatch.setattr(module, 'json', original)
        assert owner._failure is error.value
        with pytest.raises(ContractError, match='poisoned'):
            owner.final_binding()
        with pytest.raises(MemoryError) as closed:
            owner.__exit__(None, None, None)
        assert closed.value is first
