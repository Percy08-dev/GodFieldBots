"""Publication close errors never authorize healthy files or stale FD operations."""

from __future__ import annotations

import os
from time import monotonic

import pytest

from gf_training.card_instance_collection import CollectionPublication


def publisher(directory):
    directory.mkdir(exist_ok=True)
    return CollectionPublication(
        directory, work_deadline=monotonic() + 10, cleanup_deadline=monotonic() + 20
    )


def target_handle(owner, target):
    if target == "directory":
        return owner._directory_handle
    return next(output.handle for output in owner.outputs if output.name == target + ".json")


@pytest.mark.parametrize("target", ["report", "manifest", "directory"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_each_initial_release_failure_recovers_both_exact_inodes(
    tmp_path, monkeypatch, target, phase
):
    owner = publisher(tmp_path)
    real_close = os.close
    boom = OSError("injected " + target + " " + phase)
    fault = {}

    def close(descriptor):
        handle = target_handle(owner, target)
        if not fault and descriptor == handle.uncertain_number:
            fault["descriptor"] = descriptor
            if phase == "after":
                real_close(descriptor)
            raise boom
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close)
    try:
        with pytest.raises(OSError) as error:
            owner.commit({"healthy": True}, {"schema_version": 5})
        assert error.value is boom
        assert error.value._card_instance_collection_cleanup_owner is owner
        assert owner.quarantined and not owner.committed and not owner.files
        assert all(output.invalidated for output in owner.outputs)
        assert all(
            (tmp_path / name).read_bytes().startswith(b"!")
            for name in ("report.json", "manifest.json")
        )
        assert target_handle(owner, target).state == "uncertain"
        assert target_handle(owner, target).descriptor is None
        assert sum(handle.state == "uncertain" for handle in owner._handles) == 1
        assert all(handle.state != "live" for handle in owner._handles)
    finally:
        if phase == "before" and fault:
            # The injector knows it did not close. Production deliberately cannot
            # use this synthetic knowledge to retry the uncertain number.
            real_close(fault["descriptor"])


@pytest.mark.parametrize("target", ["report", "manifest", "directory"])
@pytest.mark.parametrize("same_inode", [False, True])
def test_close_then_number_reuse_never_touches_or_closes_new_owner(
    tmp_path, monkeypatch, target, same_inode
):
    owner = publisher(tmp_path / "owned")
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"FOREIGN")
    real_close, real_open, real_fstat = os.close, os.open, os.fstat
    fault = {}
    boom = OSError("released and reused")

    def close(descriptor):
        handle = target_handle(owner, target)
        if not fault and descriptor == handle.uncertain_number:
            real_close(descriptor)
            path = (
                owner.directory if target == "directory" else owner.directory / (target + ".json")
            )
            if not same_inode:
                path = foreign
            flags = os.O_RDONLY | (os.O_DIRECTORY if same_inode and target == "directory" else 0)
            reopened = real_open(path, flags)
            if reopened != descriptor:
                os.dup2(reopened, descriptor)
                real_close(reopened)
                reopened = descriptor
            fault.update(number=reopened, identity=real_fstat(reopened))
            raise boom
        if descriptor == fault.get("number"):
            pytest.fail("closed reused uncertain number")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close)
    for operation in ("fstat", "pwrite", "pread", "fsync"):
        original = getattr(os, operation)

        def guarded(descriptor, *args, _operation=operation, _original=original, **kwargs):
            if descriptor == fault.get("number"):
                pytest.fail("used reused uncertain number: " + _operation)
            return _original(descriptor, *args, **kwargs)

        monkeypatch.setattr(os, operation, guarded)
    try:
        with pytest.raises(OSError) as error:
            owner.commit({"healthy": True}, {"schema_version": 5})
        assert error.value is boom and error.value._card_instance_collection_cleanup_owner is owner
        assert real_fstat(fault["number"]) == fault["identity"] or same_inode
        assert foreign.read_bytes() == b"FOREIGN"
        assert all(output.invalidated for output in owner.outputs)
        assert not owner.committed and owner.quarantined
    finally:
        if fault:
            real_close(fault["number"])


@pytest.mark.parametrize("target", ["report", "manifest"])
@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_released_output_recovery_preserves_foreign_name_and_does_not_search_renames(
    tmp_path, monkeypatch, target, replacement
):
    owner = publisher(tmp_path / "owned")
    foreign = tmp_path / "foreign"
    foreign.write_text("FOREIGN")
    saved = tmp_path / "renamed-original"
    real_close = os.close
    fault = {}

    def close(descriptor):
        handle = target_handle(owner, target)
        if not fault and descriptor == handle.uncertain_number:
            fault["closed"] = True
            real_close(descriptor)
            path = owner.directory / (target + ".json")
            path.rename(saved)
            if replacement == "file":
                path.write_text("FOREIGN")
            else:
                path.symlink_to(foreign)
            raise OSError("foreign output substitution")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError) as error:
        owner.commit({"healthy": True}, {"schema_version": 5})
    assert error.value._card_instance_collection_cleanup_owner is owner
    assert (owner.directory / (target + ".json")).read_text() == "FOREIGN"
    assert foreign.read_text() == "FOREIGN"
    assert not saved.read_bytes().startswith(b"!")  # no renamed-original search.
    other = "manifest" if target == "report" else "report"
    assert (owner.directory / (other + ".json")).read_bytes().startswith(b"!")
    assert owner.quarantined and not owner.committed


@pytest.mark.parametrize("target", ["report", "directory"])
def test_directory_substitution_uses_only_still_owned_directory_or_quarantines(
    tmp_path, monkeypatch, target
):
    path = tmp_path / "owned"
    owner = publisher(path)
    saved = tmp_path / "renamed-directory"
    real_close = os.close
    fault = {}

    def close(descriptor):
        handle = target_handle(owner, target)
        if not fault and descriptor == handle.uncertain_number:
            fault["closed"] = True
            real_close(descriptor)
            path.rename(saved)
            path.mkdir()
            (path / "report.json").write_text("FOREIGN REPORT")
            (path / "manifest.json").write_text("FOREIGN MANIFEST")
            raise OSError("foreign directory substitution")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError) as error:
        owner.commit({"healthy": True}, {"schema_version": 5})
    assert error.value._card_instance_collection_cleanup_owner is owner
    assert (path / "report.json").read_text() == "FOREIGN REPORT"
    assert (path / "manifest.json").read_text() == "FOREIGN MANIFEST"
    assert all(
        (saved / name).read_bytes().startswith(b"!") == (target == "report")
        for name in ("report.json", "manifest.json")
    )
    assert not owner.committed and owner.quarantined


@pytest.mark.parametrize("failure", ["directory_open", "output_open", "write", "fsync", "deadline"])
def test_recovery_failure_still_attempts_both_outputs_independently(tmp_path, monkeypatch, failure):
    owner = publisher(tmp_path)
    real_close, real_open, real_write, real_fsync = os.close, os.open, os.pwrite, os.fsync
    fault = {}
    opened = {}
    attempts = []
    original_invalidate = owner._invalidate_output

    def invalidate(output):
        attempts.append(output.name)
        return original_invalidate(output)

    monkeypatch.setattr(owner, "_invalidate_output", invalidate)

    def close(descriptor):
        handle = target_handle(owner, "directory")
        if not fault and descriptor == handle.uncertain_number:
            real_close(descriptor)
            fault["released"] = True
            if failure == "deadline":
                owner.cleanup_deadline = 0
            raise OSError("initiating close")
        real_close(descriptor)

    def open_(path, flags, *args, **kwargs):
        if fault and not fault.get("injected") and (
            failure == "directory_open" and path == owner.directory
            or failure == "output_open" and path == "report.json"
        ):
            fault["injected"] = True
            raise OSError("recovery open")
        descriptor = real_open(path, flags, *args, **kwargs)
        if fault and path in ("report.json", "manifest.json"):
            assert not flags & (os.O_CREAT | os.O_TRUNC)
            assert flags & os.O_NOFOLLOW
            opened[descriptor] = path
        return descriptor

    def write(descriptor, *args):
        if (
            fault
            and failure == "write"
            and opened.get(descriptor) == "report.json"
            and not fault.get("injected")
        ):
            fault["injected"] = True
            raise OSError("recovery write")
        return real_write(descriptor, *args)

    def fsync(descriptor):
        if (
            fault
            and failure == "fsync"
            and opened.get(descriptor) == "report.json"
            and not fault.get("injected")
        ):
            fault["injected"] = True
            raise OSError("recovery fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "pwrite", write)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError, match="initiating close") as error:
        owner.commit({"healthy": True}, {"schema_version": 5})
    assert error.value._card_instance_collection_cleanup_owner is owner
    assert attempts == ["report.json", "manifest.json"]
    assert not owner.committed and owner.quarantined
    if failure != "deadline":
        assert (tmp_path / "manifest.json").read_bytes().startswith(b"!")
    else:
        assert all(not output.invalidated for output in owner.outputs)


def test_normal_release_closes_every_owned_resource_before_commit(tmp_path, monkeypatch):
    owner = publisher(tmp_path)
    original = owner._release

    def release():
        assert not owner.committed
        original()
        assert not owner.files and owner.directory_fd is None
        assert all(
            handle.state == "released" and handle.descriptor is None for handle in owner._handles
        )
        monkeypatch.setattr(owner, "_check", lambda: pytest.fail("post-release fallible check"))

    monkeypatch.setattr(owner, "_release", release)
    owner.commit({"healthy": True}, {"schema_version": 5})
    assert owner.committed and not owner.quarantined
