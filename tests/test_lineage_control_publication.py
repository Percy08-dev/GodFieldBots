"""Persistence-only owned lineage publication and terminal failure probes."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, canonical
from gf_training import structured_lineage as module


def setup_write(tmp_path):
    path = tmp_path / 'control.json'
    scan = module._ChainScan(directory=tmp_path, max_owned_bytes=1024**2, deadline=float('inf'))
    return path, scan


@pytest.mark.parametrize('immutable', [False, True])
def test_close_precedes_every_namespace_operation_and_bytes_stay_canonical(tmp_path, monkeypatch, immutable):
    path, scan = setup_write(tmp_path)
    operations = []
    original_replace, original_link, original_unlink = os.replace, os.link, Path.unlink

    def no_original_fd(source):
        info = source.lstat()
        for name in os.listdir('/proc/self/fd'):
            try:
                held = os.fstat(int(name))
            except OSError:
                continue
            assert (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino)

    def replace(source, destination):
        no_original_fd(source)
        operations.append('replace')
        return original_replace(source, destination)

    def link(source, destination, **kwargs):
        no_original_fd(source)
        operations.append('link')
        return original_link(source, destination, **kwargs)

    def unlink(source, **kwargs):
        no_original_fd(source)
        operations.append('unlink')
        return original_unlink(source, **kwargs)

    monkeypatch.setattr(os, 'replace', replace)
    monkeypatch.setattr(os, 'link', link)
    monkeypatch.setattr(Path, 'unlink', unlink)
    value = {'generation': 0, 'nested': {'unicode': '神'}}
    sha = module._write(path, value, scan=scan, immutable=immutable)
    raw = (canonical(value) + '\n').encode()
    assert path.read_bytes() == raw
    assert sha == hashlib.sha256(raw).hexdigest()
    assert scan.files[path] == (module._stat_identity(path.lstat()), sha)
    assert len(scan.files[path][0]) == 5
    assert operations == (['link', 'unlink'] if immutable else ['replace'])
    scan.finalize()


@pytest.mark.parametrize('immutable', [False, True])
@pytest.mark.parametrize('attack', ['inode', 'symlink', 'mode', 'size', 'mtime', 'hash'])
def test_readonly_handoff_cannot_adopt_changed_original(tmp_path, monkeypatch, immutable, attack):
    path, scan = setup_write(tmp_path)
    original_open = os.open
    attacked = []

    def opening(name, flags, *args, **kwargs):
        if Path(name) == path and not flags & os.O_RDWR:
            before = path.lstat()
            if attack in {'inode', 'symlink'}:
                other = tmp_path / 'replacement'
                other.write_bytes(path.read_bytes())
                path.unlink()
                if attack == 'inode':
                    os.replace(other, path)
                else:
                    path.symlink_to(other)
            elif attack == 'mode':
                path.chmod(stat.S_IMODE(before.st_mode) ^ stat.S_IXUSR)
            elif attack == 'size':
                path.write_bytes(b'x')
            elif attack == 'mtime':
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
            else:
                raw = path.read_bytes()
                path.write_bytes(b'!' + raw[1:])
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            attacked.append(True)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', opening)
    with pytest.raises((ContractError, OSError)) as caught:
        module._write(path, {'generation': 0}, scan=scan, immutable=immutable)
    assert attacked and path not in scan.files
    assert caught.value._lineage_publication_owner is caught.value.cleanup_owner


@pytest.mark.parametrize('phase', ['write', 'fsync', 'replace', 'replace_after', 'link', 'link_after', 'unlink', 'unlink_after', 'read_open', 'read'])
def test_fault_keeps_actual_owner_primary_and_previous_snapshot(tmp_path, monkeypatch, phase):
    path, scan = setup_write(tmp_path)
    previous = ((1, 2, 3, 4, 5), 'old')
    scan.files[path] = previous
    first = OSError('inert ' + phase)
    if phase == 'write':
        target, attribute = os, 'write'
    elif phase == 'fsync':
        target, attribute = os, 'fsync'
    elif phase.startswith('replace'):
        target, attribute = os, 'replace'
    elif phase.startswith('link'):
        target, attribute = os, 'link'
    elif phase.startswith('unlink'):
        target, attribute = Path, 'unlink'
    elif phase == 'read_open':
        target, attribute = os, 'open'
    else:
        target, attribute = os, 'pread'
    original = getattr(target, attribute)
    calls = []

    def fault(*args, **kwargs):
        if phase == 'read_open' and Path(args[0]) != path:
            return original(*args, **kwargs)
        if phase == 'read' and not path.exists():
            return original(*args, **kwargs)
        calls.append(args)
        if phase.endswith('_after'):
            original(*args, **kwargs)
        raise first

    monkeypatch.setattr(target, attribute, fault)
    with pytest.raises(OSError) as caught:
        module._write(path, {'generation': 0}, scan=scan, immutable=phase.startswith(('link', 'unlink')))
    assert caught.value is first and len(calls) == 1
    owner = first._lineage_publication_owner
    assert owner is first.cleanup_owner and owner.path == path
    assert scan.files[path] is previous
    assert owner.writer.fd == owner.reader.fd == -1


@pytest.mark.parametrize('readonly', [False, True])
def test_uncertain_actual_close_is_owned_once_and_never_closes_reused_number(tmp_path, monkeypatch, readonly):
    path, scan = setup_write(tmp_path)
    original_close = os.close
    first = OSError('inert close after effect')
    calls, reused = [], []

    def close(fd):
        assert path not in scan.files
        is_read = path.exists()
        if is_read != readonly:
            return original_close(fd)
        calls.append(fd)
        original_close(fd)
        new = os.open('/dev/null', os.O_RDONLY)
        assert new == fd
        reused.append(new)
        raise first

    monkeypatch.setattr(os, 'close', close)
    try:
        with pytest.raises(OSError) as caught:
            module._write(path, {'generation': 0}, scan=scan)
        assert caught.value is first
        owner = first._lineage_publication_owner
        assert owner.writer.fd == owner.reader.fd == -1 and (owner.writer._uncertain_releases or owner.reader._uncertain_releases)
        with pytest.raises(RuntimeError):
            owner.close()
        assert len(calls) == 1
        os.fstat(reused[0])
        assert path not in scan.files
        assert path.exists() == readonly
    finally:
        for fd in reused:
            original_close(fd)


def test_original_write_failure_survives_secondary_uncertain_close(tmp_path, monkeypatch):
    path, scan = setup_write(tmp_path)
    original_close = os.close
    first, secondary = OSError('inert write failure'), OSError('inert cleanup failure')

    def write(*args):
        raise first

    def close(fd):
        original_close(fd)
        raise secondary

    monkeypatch.setattr(os, 'write', write)
    monkeypatch.setattr(os, 'close', close)
    with pytest.raises(OSError) as caught:
        module._write(path, {}, scan=scan)
    assert caught.value is first
    assert first._lineage_publication_owner.writer._uncertain_releases[0][-1] is secondary
    assert first.cleanup_owner is first._lineage_publication_owner
    assert first.__notes__ and path not in scan.files


@pytest.mark.parametrize('immutable', [False, True])
@pytest.mark.parametrize('attack', ['inode', 'symlink', 'mode', 'size', 'mtime', 'hash'])
def test_writer_close_handoff_rechecks_temporary_before_publication(tmp_path, monkeypatch, immutable, attack):
    path, scan = setup_write(tmp_path)
    temporary = path.with_name(path.name + '.tmp')
    original_close = os.close
    attacked = []

    def close(fd):
        result = original_close(fd)
        if not attacked and temporary.exists() and not path.exists():
            info, raw = temporary.lstat(), temporary.read_bytes()
            if attack in {'inode', 'symlink'}:
                other = tmp_path / 'other'
                other.write_bytes(raw)
                temporary.unlink()
                if attack == 'inode':
                    os.replace(other, temporary)
                else:
                    temporary.symlink_to(other)
            elif attack == 'mode':
                temporary.chmod(stat.S_IMODE(info.st_mode) ^ stat.S_IXUSR)
            elif attack == 'size':
                temporary.write_bytes(b'x')
            elif attack == 'mtime':
                os.utime(temporary, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
            else:
                temporary.write_bytes(b'!' + raw[1:])
                os.utime(temporary, ns=(info.st_atime_ns, info.st_mtime_ns))
            attacked.append(True)
        return result

    monkeypatch.setattr(os, 'close', close)
    with pytest.raises(ContractError, match='published lineage'):
        module._write(path, {}, scan=scan, immutable=immutable)
    assert attacked and path not in scan.files
    # A same-metadata content mutation is detected by the final full hash.
    assert path.exists() == (attack == 'hash')


@pytest.mark.parametrize('readonly', [False, True])
def test_close_failure_before_effect_retires_number_without_retry(tmp_path, monkeypatch, readonly):
    path, scan = setup_write(tmp_path)
    original_close = os.close
    first = OSError('inert pre-effect close failure')
    uncertain = []

    def close(fd):
        if path.exists() == readonly:
            uncertain.append(fd)
            raise first
        return original_close(fd)

    monkeypatch.setattr(os, 'close', close)
    try:
        with pytest.raises(OSError) as caught:
            module._write(path, {}, scan=scan)
        assert caught.value is first
        owner = first._lineage_publication_owner
        assert owner.writer.fd == owner.reader.fd == -1
        with pytest.raises(RuntimeError):
            owner.close()
        assert len(uncertain) == 1 and path not in scan.files
        os.fstat(uncertain[0])  # The actual uncertain capability is still live.
    finally:
        for fd in uncertain:
            original_close(fd)  # Explicit test-only disposal after the assertion.


def test_short_writes_are_completed_and_zero_progress_fails(tmp_path, monkeypatch):
    path, scan = setup_write(tmp_path)
    original_write = os.write
    counts = []

    def partial(fd, data):
        counts.append(1)
        return original_write(fd, data[:2])

    monkeypatch.setattr(os, 'write', partial)
    module._write(path, {'value': 1}, scan=scan)
    assert len(counts) > 1 and path.read_bytes() == b'{"value":1}\n'
    prior = scan.files[path]
    monkeypatch.setattr(os, 'write', lambda *_: 0)
    with pytest.raises(ContractError, match='no progress'):
        module._write(path, {'value': 2}, scan=scan)
    assert scan.files[path] is prior


@pytest.mark.parametrize('readonly', [False, True])
def test_inspection_fault_after_actual_open_retains_capability_and_first_error(tmp_path, monkeypatch, readonly):
    path, scan = setup_write(tmp_path)
    original_fstat = os.fstat
    first = OSError('inert actual-open inspection failure')

    def fstat(fd):
        if path.exists() == readonly:
            raise first
        return original_fstat(fd)

    monkeypatch.setattr(os, 'fstat', fstat)
    with pytest.raises(OSError) as caught:
        module._write(path, {}, scan=scan)
    assert caught.value is first
    owner = first._lineage_publication_owner
    assert owner.writer.fd == owner.reader.fd == -1 and path not in scan.files


@pytest.mark.parametrize('field', ['st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'])
def test_every_readonly_identity_component_is_compared(tmp_path, monkeypatch, field):
    path, scan = setup_write(tmp_path)
    original_fstat = os.fstat
    injected = []

    def fstat(fd):
        result = original_fstat(fd)
        if path.exists():
            fields = {name: getattr(result, name) for name in (
                'st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
            fields[field] += 1
            injected.append(True)
            return SimpleNamespace(**fields)
        return result

    monkeypatch.setattr(os, 'fstat', fstat)
    with pytest.raises(ContractError, match='published lineage'):
        module._write(path, {}, scan=scan)
    assert injected and path not in scan.files


def test_same_bytes_parent_directory_replacement_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    root.mkdir()
    nested = root / 'nested'
    nested.mkdir()
    path, scan = setup_write(root)
    path = nested / path.name
    original_replace = os.replace

    def replace(source, destination):
        original_replace(source, destination)
        original_replace(nested, root / 'old')
        nested.mkdir()
        (nested / path.name).write_bytes((root / 'old' / path.name).read_bytes())

    monkeypatch.setattr(os, 'replace', replace)
    with pytest.raises(ContractError, match='published lineage parent'):
        module._write(path, {}, scan=scan)
    assert path not in scan.files


@pytest.mark.parametrize('phase', ['read', 'read_close', 'final_path'])
def test_ctime_drift_after_readonly_admission_is_strict(tmp_path, monkeypatch, phase):
    path, scan = setup_write(tmp_path)
    original_pread, original_close, original_lstat = os.pread, os.close, Path.lstat
    mutated = []
    read_finished = []
    readonly_closed = []

    def mutate():
        if not mutated:
            path.chmod(stat.S_IMODE(original_lstat(path).st_mode))
            mutated.append(True)

    def pread(fd, amount, offset):
        result = original_pread(fd, amount, offset)
        if path.exists():
            if phase == 'read':
                mutate()
            read_finished.append(True)
        return result

    def close(fd):
        result = original_close(fd)
        if read_finished:
            readonly_closed.append(True)
            if phase == 'read_close':
                mutate()
        return result

    def lstat(self, *args, **kwargs):
        if self == path and phase == 'final_path' and readonly_closed:
            mutate()
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(os, 'pread', pread)
    monkeypatch.setattr(os, 'close', close)
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(ContractError, match='published lineage'):
        module._write(path, {}, scan=scan)
    assert mutated and path not in scan.files


def test_immutable_collision_and_preexisting_temporary_are_preserved(tmp_path):
    path, scan = setup_write(tmp_path)
    path.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        module._write(path, {}, scan=scan, immutable=True)
    assert path.read_bytes() == b'original' and path not in scan.files
    temporary = path.with_name(path.name + '.tmp')
    owned_temporary = temporary.read_bytes()
    with pytest.raises(FileExistsError):
        module._write(path, {}, scan=scan)
    assert temporary.read_bytes() == owned_temporary


def test_parent_symlink_and_budget_deadline_fail_before_effects(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    other = tmp_path / 'other'
    other.mkdir()
    (root / 'link').symlink_to(other, target_is_directory=True)
    path, scan = setup_write(root)
    with pytest.raises(ContractError):
        module._write(root / 'link' / 'control.json', {}, scan=scan)
    assert not list(other.iterdir())
    scan.limit = 1
    with pytest.raises(ContractError):
        module._write(path, {}, scan=scan)
    scan.limit = 1024**2
    scan.deadline = 0
    with pytest.raises(TimeoutError):
        module._write(path, {}, scan=scan)
    assert not path.exists() and not path.with_name(path.name + '.tmp').exists()
