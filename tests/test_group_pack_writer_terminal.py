"""Direct Writer terminal controls retain exact original ownership and accounting."""
from __future__ import annotations

import os
import stat
import time

import pytest

from gf_common.group_pack import MIB, StorageSettings
from gf_training import group_pack_writer as module
from gf_training.group_pack_resources import StorageBudget
from gf_training.group_pack_source import PackedOriginals, open_artifact_source
from tests.test_group_pack_storage import submit


def prepared(tmp_path):
    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram")
    budget = StorageBudget(settings, time.monotonic() + 30)
    writer = module.GroupPackWriter(tmp_path / "collection", "terminal-test", settings=settings,
        deadline=budget.deadline, budget=budget)
    submit(writer, "group")
    writer.store_control("control", b"\x00exact\xff")
    return writer, budget


def touch_ctime(path):
    before = path.stat()
    path.chmod(stat.S_IMODE(before.st_mode))
    after = path.stat()
    assert before.st_mtime_ns == after.st_mtime_ns and before.st_ino == after.st_ino
    assert before.st_ctime_ns != after.st_ctime_ns


def after_write_close(monkeypatch, writer, action):
    actual = module.close_raw_descriptor
    changed = False
    def close(owner, attribute):
        nonlocal changed
        inject = owner is writer and attribute == "_publication_fd" and owner._publication_fd >= 0
        actual(owner, attribute)
        if inject and not changed:
            changed = True
            action(writer._published[0])
    monkeypatch.setattr(module, "close_raw_descriptor", close)


@pytest.mark.parametrize("publication", ["published", "pending"])
def test_direct_ctime_handoff_preserves_bytes_and_strict_source(tmp_path, monkeypatch, publication):
    writer, budget = prepared(tmp_path)
    after_write_close(monkeypatch, writer, touch_ctime)
    locator = writer.seal(publication=publication, collect_metrics=True)
    assert writer.returncode == 0 and budget.used == writer.owned_bytes == 0
    assert writer.checkpoint_metrics is not None
    if publication == "published":
        source = open_artifact_source(locator, deadline=time.monotonic() + 20, max_owned_bytes=512 * MIB)
        assert source.read("control") == b"\x00exact\xff"
        with source.borrow_group("group") as loan:
            assert loan.read("group/e0") == b"\xff\x00"
        source.close()
    originals = PackedOriginals(locator.path, deadline=time.monotonic() + 20,
        max_owned_bytes=512 * MIB, allow_pending=True)
    try:
        touch_ctime(locator.path)
        with pytest.raises(ValueError):
            originals.recheck()
    finally:
        originals.close()
    writer.close()
    assert budget.used == 0


@pytest.mark.parametrize("publication", ["published", "pending"])
@pytest.mark.parametrize("mutation", ["inode", "content", "mode", "mtime", "database-ctime"])
def test_direct_handoff_refuses_changed_original(tmp_path, monkeypatch, mutation, publication):
    writer, budget = prepared(tmp_path)
    def mutate(path):
        if mutation == "database-ctime":
            touch_ctime(path.parent / "collection.sqlite3")
            return
        before = path.stat()
        if mutation == "inode":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        elif mutation == "content":
            raw = path.read_bytes()
            path.write_bytes(b"!" + raw[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif mutation == "mode":
            path.chmod(stat.S_IMODE(before.st_mode) ^ stat.S_IXUSR)
        else:
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
    after_write_close(monkeypatch, writer, mutate)
    with pytest.raises(ValueError) as caught:
        writer.seal(publication=publication)
    assert caught.value.cleanup_owner is writer and not writer._sealed
    if mutation == "inode":
        with pytest.raises(RuntimeError):
            writer.close()
        assert budget.used > 0  # Foreign replacement keeps actual cleanup owner.
    else:
        writer.close()
        assert budget.used == 0
        assert (writer.directory / f"collection.{publication}.json").read_bytes().startswith(b"GFB-QUARANTINED")


@pytest.mark.parametrize("publication", ["published", "pending"])
@pytest.mark.parametrize("failure", ["during-read", "readonly-close-ctime", "readonly-close-content", "readonly-close-uncertain", "write-close-uncertain"])
def test_direct_readonly_custody_and_uncertain_retirement(tmp_path, monkeypatch, failure, publication):
    from gf_training.group_pack_publication import _TerminalOriginal

    writer, budget = prepared(tmp_path)
    actual_pread, actual_close = os.pread, os.close
    actual_terminal_close = _TerminalOriginal.close
    target_fd = None
    injected = False
    attempts = []
    def read(fd, size, offset):
        nonlocal injected
        raw = actual_pread(fd, size, offset)
        original = getattr(writer, "_terminal_original", None)
        if not injected and original is not None and original.fd == fd:
            injected = True
            touch_ctime(original.path)
        return raw
    def terminal_close(original):
        nonlocal target_fd
        live = original.fd >= 0
        if live:
            target_fd = original.fd
        actual_terminal_close(original)
        if live and failure == "readonly-close-ctime":
            touch_ctime(original.path)
        elif live and failure == "readonly-close-content":
            before = original.path.stat()
            raw = original.path.read_bytes()
            original.path.write_bytes(b"!" + raw[1:])
            os.utime(original.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    actual_raw_close = module.close_raw_descriptor
    def write_close(owner, attribute):
        nonlocal target_fd
        if owner is writer and attribute == "_publication_fd" and target_fd is None:
            target_fd = owner._publication_fd
        actual_raw_close(owner, attribute)
    def close(fd):
        nonlocal injected
        actual_close(fd)
        if fd == target_fd and not injected:
            attempts.append(fd)
            injected = True
            raise OSError("injected terminal post-release failure")
    if failure == "during-read":
        monkeypatch.setattr(os, "pread", read)
    else:
        monkeypatch.setattr(_TerminalOriginal, "close", terminal_close)
    if failure.endswith("uncertain"):
        monkeypatch.setattr(os, "close", close)
    if failure == "write-close-uncertain":
        monkeypatch.setattr(module, "close_raw_descriptor", write_close)
    with pytest.raises((ValueError, OSError)) as caught:
        writer.seal(publication=publication)
    assert not writer._sealed and caught.value.cleanup_owner is writer
    assert (writer.directory / f"collection.{publication}.json").read_bytes().startswith(b"GFB-QUARANTINED")
    if failure.endswith("uncertain"):
        assert str(caught.value) == "injected terminal post-release failure"
        assert writer._publication_fd == -1
        assert writer._terminal_original.fd == -1
        if failure == "readonly-close-uncertain":
            assert writer._terminal_original._uncertain_releases
            assert writer._terminal_original in writer._cleanup_children
        else:
            assert writer._uncertain_releases
        assert len(attempts) == 1 and budget.used == writer.owned_bytes > module.WRITER_BASE
        with pytest.raises(RuntimeError):
            writer.close()
        assert len(attempts) == 1 and budget.used > 0
    else:
        writer.close()
        assert budget.used == 0


def test_direct_readonly_preallocation_and_retirement_use_same_account(tmp_path, monkeypatch):
    from gf_training.group_pack_publication import _TerminalOriginal

    writer, budget = prepared(tmp_path)
    actual_open, actual_close = _TerminalOriginal.open_verified, _TerminalOriginal.close
    observed = []
    def check():
        observed.append(budget.used)
        assert budget.used == writer.owned_bytes >= module.WRITER_BASE + 65536 + 4096
    def open_verified(original):
        check()
        actual_open(original)
    def close(original):
        if original.fd >= 0:
            check()
        actual_close(original)
    monkeypatch.setattr(_TerminalOriginal, "open_verified", open_verified)
    monkeypatch.setattr(_TerminalOriginal, "close", close)
    writer.seal()
    assert len(observed) == 2 and budget.used == writer.owned_bytes == 0


def test_direct_terminal_admission_failure_precedes_readonly_open(tmp_path, monkeypatch):
    from gf_training.group_pack_publication import _TerminalOriginal

    writer, budget = prepared(tmp_path)
    actual_acquire = writer._budget.acquire
    opened = []
    def acquire(size, deadline, *, wait=False):
        if size == 65536 + 4096:
            raise ValueError("terminal admission refused")
        actual_acquire(size, deadline, wait=wait)
    monkeypatch.setattr(writer._budget, "acquire", acquire)
    monkeypatch.setattr(_TerminalOriginal, "open_verified", lambda original: opened.append(original))
    with pytest.raises(ValueError, match="terminal admission refused"):
        writer.seal()
    writer.close()
    assert not opened and budget.used == 0


@pytest.mark.parametrize("stage", ["before-open", "readonly-live"])
def test_direct_terminal_expired_deadline_still_retires_owned_resources(tmp_path, monkeypatch, stage):
    from gf_training.group_pack_publication import _TerminalOriginal

    writer, budget = prepared(tmp_path)
    original_deadline = writer.deadline
    attribute = "open_verified" if stage == "before-open" else "recheck"
    actual = getattr(_TerminalOriginal, attribute)
    active = []
    def expire(original):
        active.append(original.fd)
        writer.deadline = original.deadline = time.monotonic() - 1
        actual(original)
    monkeypatch.setattr(_TerminalOriginal, attribute, expire)
    with pytest.raises(TimeoutError) as caught:
        writer.seal()
    assert caught.value.cleanup_owner is writer
    assert writer.deadline < original_deadline
    assert active and (active[0] >= 0 if stage == "readonly-live" else active[0] == -1)
    assert writer._terminal_original.fd == -1
    writer.close()
    assert writer.returncode == 0 and budget.used == writer.owned_bytes == 0


def test_direct_standalone_account_covers_readonly_handoff(tmp_path, monkeypatch):
    from gf_training.group_pack_publication import _TerminalOriginal

    settings = StorageSettings(db_root=tmp_path / "catalog", ram_root=tmp_path / "ram")
    writer = module.GroupPackWriter(tmp_path / "collection", "standalone", settings=settings,
        deadline=time.monotonic() + 30)
    actual = _TerminalOriginal.open_verified
    observed = []
    def open_verified(original):
        observed.append(writer.owned_bytes)
        assert writer.owned_bytes >= module.WRITER_BASE + 65536 + 4096
        actual(original)
    monkeypatch.setattr(_TerminalOriginal, "open_verified", open_verified)
    writer.seal()
    assert observed and writer.owned_bytes == 0
    writer.close()


def test_retained_pending_preserves_exact_owner_without_terminal_handoff(tmp_path):
    writer, budget = prepared(tmp_path)
    writer.seal(publication="pending", retain_pending=True)
    pending = writer.take_pending_owner()
    assert writer._terminal_original is None and writer._terminal_bytes == 0
    pending.recheck()
    touch_ctime(pending.path)
    with pytest.raises(ValueError, match="retained pending descriptor changed"):
        pending.recheck()
    pending.close()
    writer.close()
    assert budget.used == 0
