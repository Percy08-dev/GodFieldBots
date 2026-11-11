"""Controlled terminal metadata only. These files cannot qualify a formal Source."""
import os
import time

import pytest

from gf_common.domain import ContractError, canonical, digest
from gf_training.card_instance_formal import FormalCompletion
from tests.test_formal_ppo_contract import state_for
from tests.test_group_pack_formal_contract import packed_state


def controls(directory, packed=False):
    directory.mkdir()
    state = packed_state() if packed else state_for()
    state.update(run_id='UNADMITTED-control-only', identity={'scope': 'no original execution or Source'},
        generation_sha256='b'*64, accepted_decisions=1, config_sha256=digest(state['effective_config']))
    (directory/'run.json').write_text(canonical(state)+'\n')
    state['status'] = 'complete'  # No valid formal stop, artifacts or original sidecar exists.
    return state


@pytest.mark.parametrize('packed', [False, True])
def test_writer_is_actually_closed_before_replace_and_fresh_revoker_is_retained(tmp_path, monkeypatch, packed):
    directory = tmp_path/'control'
    state = controls(directory, packed)
    owner = FormalCompletion()
    replace = os.replace
    def checked_replace(source, target, **options):
        expected = os.stat(source, dir_fd=options['src_dir_fd'], follow_symlinks=False)
        for name in os.listdir('/proc/self/fd'):
            try:
                held = os.fstat(int(name))
            except OSError:
                continue
            assert (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino)
        replace(source, target, **options)
    monkeypatch.setattr(os, 'replace', checked_replace)
    owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    assert owner.fd is not None and len(owner.snapshot) == 5
    assert (os.fstat(owner.fd).st_dev, os.fstat(owner.fd).st_ino) == owner.snapshot[:2]
    owner.finish()
    assert owner.committed and owner.fd is None and owner.directory_fd is None


@pytest.mark.parametrize('after', [False, True])
def test_replace_failure_recovers_only_owned_fixed_candidate_through_actual_directory(tmp_path, monkeypatch, after):
    directory = tmp_path/'control'
    state = controls(directory)
    previous = (directory/'run.json').read_bytes()
    owner = FormalCompletion()
    replace = os.replace
    first = OSError('controlled replace failure')
    def fault(*args, **options):
        if after:
            replace(*args, **options)
        raise first
    monkeypatch.setattr(os, 'replace', fault)
    with pytest.raises(OSError) as error:
        owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    assert error.value is first
    owner.revoke()
    assert owner.revoked and not owner.committed
    if after:
        assert (directory/'run.json').read_bytes().startswith(b'!')
    else:
        assert (directory/'run.json').read_bytes() == previous
        assert (directory/'run.json.formal-complete.tmp').read_bytes().startswith(b'!')


def test_foreign_fresh_open_never_replaces_expected_original_or_gains_invalidation_right(tmp_path, monkeypatch):
    directory = tmp_path/'control'
    state = controls(directory)
    foreign = tmp_path/'foreign'
    foreign.write_bytes(b'FOREIGN preserved')
    owner = FormalCompletion()
    opening = os.open
    injected = []
    def open_file(path, flags, *args, **options):
        if path == 'run.json' and flags & os.O_RDWR and not injected:
            injected.append(True)
            return opening(foreign, flags)
        return opening(path, flags, *args, **options)
    monkeypatch.setattr(os, 'open', open_file)
    with pytest.raises(ContractError):
        owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    original = (directory/'run.json').stat()
    assert owner.snapshot[:2] == (original.st_dev, original.st_ino)
    owner.revoke()
    assert injected and foreign.read_bytes() == b'FOREIGN preserved'
    assert (directory/'run.json').read_bytes().startswith(b'!')


def test_substituted_absolute_parent_does_not_block_held_directory_original_revocation(tmp_path, monkeypatch):
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    replace = os.replace
    moved = tmp_path/'actual-owned-directory'
    def fault(*args, **options):
        replace(*args, **options)
        replace(directory, moved)
        directory.mkdir()
        (directory/'run.json').write_bytes(b'FOREIGN current namespace')
        raise OSError('parent namespace substituted after effect')
    monkeypatch.setattr(os, 'replace', fault)
    with pytest.raises(OSError):
        owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    owner.revoke()
    assert owner.revoked
    assert (moved/'run.json').read_bytes().startswith(b'!')
    assert (directory/'run.json').read_bytes() == b'FOREIGN current namespace'


def test_expired_final_guard_does_no_owned_storage_io(tmp_path, monkeypatch):
    import gf_training.ppo_run as coordinator
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    owner.deadline = 0
    monkeypatch.setattr(coordinator, 'owned_bytes', lambda *args, **kwargs: pytest.fail('storage scan after original deadline'))
    try:
        with pytest.raises(ContractError, match='outer completion'):
            owner.finish()
    finally:
        owner.revoke()


@pytest.mark.parametrize('after', [False, True])
def test_uncertain_writer_close_never_renames_or_retries_consumed_number(tmp_path, monkeypatch, after):
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    closing, opening, fstat = os.close, os.open, os.fstat
    foreign = tmp_path/'foreign'
    foreign.write_bytes(b'foreign safe')
    uncertain = []
    first = OSError('writer close uncertainty')
    def close(fd):
        if fd == owner._writer.fd and not uncertain:
            # OwnedDescriptor consumes its field BEFORE calling os.close.
            pytest.fail('writer stage was not consumed before close')
        if owner._writer.fd == -1 and owner._original is not None and not owner.exposed and not uncertain:
            uncertain.append(fd)
            if after:
                closing(fd)
                replacement = opening(foreign, os.O_RDONLY)
                if replacement != fd:
                    os.dup2(replacement, fd)
                    closing(replacement)
            raise first
        if uncertain and fd == uncertain[0]:
            pytest.fail('consumed writer descriptor was retried')
        return closing(fd)
    def inspect(fd):
        if uncertain and fd == uncertain[0]:
            pytest.fail('consumed writer descriptor was inspected')
        return fstat(fd)
    monkeypatch.setattr(os, 'close', close)
    monkeypatch.setattr(os, 'fstat', inspect)
    monkeypatch.setattr(os, 'replace', lambda *args, **kwargs: pytest.fail('rename after uncertain writer close'))
    try:
        with pytest.raises(OSError) as caught:
            owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
        assert caught.value is first
        with pytest.raises(RuntimeError):
            owner.revoke()
        assert owner.revoked and not owner.exposed and owner._writer._uncertain_releases
        assert (directory/'run.json.formal-complete.tmp').read_bytes().startswith(b'!')
        assert foreign.read_bytes() == b'foreign safe'
    finally:
        if uncertain:
            closing(uncertain[0])  # Controlled test-only disposal of the known injected uncertainty.


@pytest.mark.parametrize('phase', ['open', 'fstat', 'hash'])
def test_fresh_retained_failure_keeps_original_expected_identity_and_revokes_actual_original(tmp_path, monkeypatch, phase):
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    opening, fstat, hashing = os.open, os.fstat, owner._hash
    injected = []
    retained = []
    first = OSError('fresh retained '+phase+' failure')
    def open_file(name, flags, *args, **options):
        if name == 'run.json' and flags & os.O_RDWR and not injected:
            if phase == 'open':
                injected.append(True)
                raise first
            fd = opening(name, flags, *args, **options)
            retained.append(fd)
            return fd
        return opening(name, flags, *args, **options)
    def inspect(fd):
        if phase == 'fstat' and retained and fd == retained[0] and not injected:
            injected.append(True)
            raise first
        return fstat(fd)
    def hashed(fd):
        if phase == 'hash' and retained and fd == retained[0] and not injected:
            injected.append(True)
            raise first
        return hashing(fd)
    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, 'fstat', inspect)
    monkeypatch.setattr(owner, '_hash', hashed)
    with pytest.raises(OSError) as error:
        owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    assert error.value is first and owner.fd is None
    expected = owner.snapshot[:2]
    owner.revoke()
    actual = (directory/'run.json').stat()
    assert (actual.st_dev, actual.st_ino) == expected
    assert owner.revoked and (directory/'run.json').read_bytes().startswith(b'!')


def test_retained_nonwriting_close_allows_no_second_ctime_handoff(tmp_path, monkeypatch):
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    fd, closing = owner.fd, os.close
    injected = []
    def close(descriptor):
        result = closing(descriptor)
        if descriptor == fd and not injected:
            injected.append(True)
            path = directory/'run.json'
            path.chmod(path.stat().st_mode)
        return result
    monkeypatch.setattr(os, 'close', close)
    with pytest.raises(ContractError, match='outer completion'):
        owner.finish()
    owner.revoke()
    assert injected and owner.revoked and not owner.committed
    assert (directory/'run.json').read_bytes().startswith(b'!')


def test_writer_hash_identity_is_rechecked_before_controlled_close_window(tmp_path, monkeypatch):
    directory = tmp_path/'control'
    state = controls(directory)
    owner = FormalCompletion()
    hashing = owner._hash
    injected = []
    def hash_then_mutate(fd):
        value = hashing(fd)
        if fd == owner._writer.fd and not injected:
            injected.append(True)
            path = directory/'run.json.formal-complete.tmp'
            path.chmod(path.stat().st_mode)
        return value
    monkeypatch.setattr(owner, '_hash', hash_then_mutate)
    try:
        with pytest.raises(ContractError, match='formal'):
            owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
    finally:
        owner.revoke()
    assert injected and owner.revoked and not owner.committed


def test_late_writer_stat_cannot_overwrite_independent_expected_original(tmp_path, monkeypatch):
    directory = tmp_path/'control'
    state = controls(directory)
    foreign = tmp_path/'foreign'
    foreign.write_bytes((canonical(state)+'\n').encode())
    owner = FormalCompletion()
    inspecting = os.fstat
    foreign_info = foreign.stat()
    injected = []
    def inspect(fd):
        if fd == owner._writer.fd and owner._completion_fd_identity is not None and not injected:
            injected.append(True)
            return foreign_info
        return inspecting(fd)
    monkeypatch.setattr(os, 'fstat', inspect)
    try:
        with pytest.raises(ContractError):
            owner.publish(directory, state, deadline=time.monotonic()+30, max_owned_bytes=1024**3)
        original = (directory/'run.json.formal-complete.tmp').stat()
        assert owner.snapshot[:2] == (original.st_dev, original.st_ino)
    finally:
        owner.revoke()
    assert injected and owner.revoked and foreign.read_bytes() == (canonical(state)+'\n').encode()
