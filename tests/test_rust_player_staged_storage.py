"""Native PPO RAM collection becomes Source only after verified HDD publication."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import sqlite3
import time
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_cli.config import load_config, load_rust_player_effective
from gf_cli.hpo_config import campaign_config, materialize_hpo_run
from gf_common.domain import ContractError, canonical
from gf_common.group_pack import FORMAT
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
from gf_training.group_pack_db import DATABASE, catalog_update
from gf_training.group_pack_source import OriginalFile, PackedOriginals
from gf_training.hpo_config import effective_settings
from tests.test_group_pack_storage import settings, submit, writer
from tests.test_rust_player_config import BASE, NORMAL_BASE, RUN_ID, campaign_raw, runtime


def _base_with_staging(tmp_path: Path, base: Path) -> Path:
    raw = base.read_text().replace('"../', '"' + str(base.parent.parent) + '/')
    if "[storage]" not in raw:
        raw += "\n[storage]\n"
    raw = raw.replace("[storage]", '[storage]\ncollection_work_root = "local-work"')
    file = tmp_path / "base.toml"
    file.write_text(raw)
    return file


def test_staging_round_trips_native_effective_and_hpo_materialization(tmp_path: Path) -> None:
    old = load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    assert "collection_work_root" not in old.effective["storage"]

    file = _base_with_staging(tmp_path, NORMAL_BASE)
    native = load_config(file, {"run": {"player_runtime": runtime()}})
    expected = str(tmp_path / "local-work")
    assert native.effective["storage"]["collection_work_root"] == expected
    out = tmp_path / "effective.json"
    out.write_text(canonical(native.effective))
    assert load_rust_player_effective(out, expected_sha256=native.sha256).effective == native.effective

    raw = campaign_raw()
    raw["base_experiment"] = str(_base_with_staging(tmp_path, BASE))
    raw["storage"]["originals_root"] = str(tmp_path / "durable")
    loaded = campaign_config(raw, origin=tmp_path)
    chosen = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
    hpo = materialize_hpo_run(loaded, chosen, learning_seed=17, run_id=RUN_ID)
    assert hpo.effective["storage"]["collection_work_root"] == expected
    assert hpo.effective["storage"]["db_root"] == str(tmp_path / "durable")
    out.write_text(canonical(hpo.effective))
    assert load_rust_player_effective(out, expected_sha256=hpo.sha256).effective == hpo.effective


def test_staged_destination_refuses_distro_ext4_and_non_tmpfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gf_common import hpo_storage

    ram, hdd = tmp_path / "ram", tmp_path / "hdd"
    ram.mkdir()
    hdd.mkdir()
    root = hpo_storage.Mount(1, "0:1", "/", "/", "ext4", "distro", ("rw",))
    tmpfs = hpo_storage.Mount(2, "0:2", "/", str(ram), "tmpfs", "tmpfs", ("rw",))
    ext4 = hpo_storage.Mount(3, "0:3", "/", str(hdd), "ext4", "hdd", ("rw",))
    monkeypatch.setattr(hpo_storage, "read_mounts", lambda: (root, tmpfs, ext4))
    assert hpo_storage.check_staged_collection_destinations(
        ram / "work", hdd / "parent", catalog_root=hdd / "originals",
        originals=hdd / "originals/run/collection",
    ) == (tmpfs, ext4)
    with pytest.raises(ContractError, match="tmpfs"):
        hpo_storage.check_staged_collection_destinations(hdd, hdd, catalog_root=hdd)
    with pytest.raises(ContractError, match="ext4"):
        hpo_storage.check_staged_collection_destinations(ram, hdd, catalog_root=tmp_path / "on-ssd")
    with pytest.raises(ContractError, match="ext4"):
        hpo_storage.check_staged_collection_destinations(ram, tmp_path / "on-ssd", catalog_root=hdd)
    assert list(hdd.iterdir()) == []


@pytest.fixture
def sealed(tmp_path: Path):
    stage = tmp_path / "ram"  # Unit fixture simulates separate namespaces.
    owner = writer(stage)
    submit(owner, "pair")
    owner.store_control("manifest.json", b'{"example":true}')
    locator = owner.seal(publication="pending")
    descriptor = json.loads(locator.path.read_bytes())
    original = OriginalFile(locator.path.parent / DATABASE, time.monotonic() + 30)
    try:
        yield original, descriptor, locator.path.read_bytes()
    finally:
        original.close()


def _promote(tmp_path: Path, sealed):
    from gf_training.rust_player_promotion import promote_collection

    source, descriptor, raw = sealed
    final = tmp_path / "hdd/catalog/run-1/collections/batch-000000"
    owner = promote_collection(
        source, directory=final, descriptor=descriptor, raw=raw,
        settings=settings(tmp_path / "hdd"), deadline=source.deadline,
        max_owned_bytes=512 * 1024**2, mount_root=tmp_path,
    )
    return final, owner


def test_returns_same_verified_hdd_files_and_final_catalog(tmp_path: Path, sealed) -> None:
    final, owner = _promote(tmp_path, sealed)
    source, descriptor, raw = sealed
    try:
        owner.recheck()
        assert owner.directory == final
        assert owner.raw == raw
        assert owner.database.path == final / DATABASE
        assert owner.database.identity.st_ino != source.identity.st_ino
        assert sha256((final / DATABASE).read_bytes()).hexdigest() == descriptor["database_sha256"]
        assert {p.name for p in final.iterdir()} == {DATABASE, owner.path.name}
        assert (owner.snapshot.st_dev, owner.snapshot.st_ino) == (
            owner.path.stat().st_dev, owner.path.stat().st_ino,
        )
        original_path = source.path
        source.close()
        original_path.unlink()
        with closing(PackedOriginals(
            owner.path, allow_pending=True, deadline=time.monotonic() + 30,
            max_owned_bytes=512 * 1024**2,
        )) as durable:
            assert durable.descriptor == descriptor
        with closing(sqlite3.connect(tmp_path / "hdd/catalog/catalog.sqlite3")) as catalog:
            assert catalog.execute("select database,state,seal_id,descriptor_sha256 from collections").fetchone() == (
                str(final / DATABASE), "sealed", descriptor["seal_id"], sha256(raw).hexdigest(),
            )
        assert descriptor["format"] == FORMAT
    finally:
        owner.close()


@pytest.mark.parametrize("number", [errno.EIO, errno.ENOSPC])
def test_failed_copy_leaves_no_completed_destination_and_can_reuse_same_seal(
    tmp_path: Path, sealed, monkeypatch: pytest.MonkeyPatch, number: int,
) -> None:
    with monkeypatch.context() as nested:
        def fail_write(_fd, _buffer):
            raise OSError(number, os.strerror(number))

        nested.setattr(os, "write", fail_write)
        with pytest.raises(OSError) as failure:
            _promote(tmp_path, sealed)
        assert failure.value.errno == number
    final = tmp_path / "hdd/catalog/run-1/collections/batch-000000"
    assert not final.exists()
    assert not (tmp_path / "hdd/catalog/catalog.sqlite3").exists()
    assert list(final.parent.iterdir()) == []
    final, recovered = _promote(tmp_path, sealed)
    assert recovered.descriptor["seal_id"] == sealed[1]["seal_id"]
    recovered.close()


def test_corrupt_hdd_bytes_are_refused_before_sync_or_publication(
    tmp_path: Path, sealed, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write = os.write
    corrupted = False

    def mutate(fd, buffer):
        nonlocal corrupted
        if not corrupted and bytes(buffer[:16]) == b"SQLite format 3\0":
            corrupted = True
            buffer = bytes([buffer[0] ^ 1]) + buffer[1:]
        return write(fd, buffer)

    monkeypatch.setattr(os, "write", mutate)
    with pytest.raises(ValueError, match="SHA-256"):
        _promote(tmp_path, sealed)
    assert corrupted
    assert not (tmp_path / "hdd/catalog/catalog.sqlite3").exists()
    assert not (tmp_path / "hdd/catalog/run-1/collections/batch-000000").exists()


def test_stale_sealed_catalog_is_never_overwritten(tmp_path: Path, sealed) -> None:
    source, descriptor, _ = sealed
    conf = settings(tmp_path / "hdd")
    final = conf.db_root / "run-1/collections/batch-000000"
    old_seal = "00000000-0000-0000-0000-000000000001"
    catalog_update(conf, final / DATABASE, "run-1", "sealed", source.deadline, descriptor | {"seal_id": old_seal})
    with pytest.raises(ValueError, match="catalog"):
        _promote(tmp_path, sealed)
    with closing(sqlite3.connect(conf.db_root / "catalog.sqlite3")) as catalog:
        assert catalog.execute("select state, seal_id from collections").fetchone() == ("sealed", old_seal)
    assert not final.exists()


def test_existing_durable_batch_is_not_replaced(tmp_path: Path, sealed) -> None:
    final = tmp_path / "hdd/catalog/run-1/collections/batch-000000"
    final.mkdir(parents=True)
    (final / "sentinel").write_bytes(b"prior")
    with pytest.raises(FileExistsError):
        _promote(tmp_path, sealed)
    assert {p.name for p in final.iterdir()} == {"sentinel"}
    assert (final / "sentinel").read_bytes() == b"prior"


def test_durable_order_verifies_staged_inodes_before_publish(
    tmp_path: Path, sealed, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gf_training import rust_player_promotion as promotion

    audit, sync, rename, catalog = promotion.validate_database, os.fsync, os.rename, promotion.catalog_update
    observed = []

    def verification(*args, **kwargs):
        observed.append("integrity")
        return audit(*args, **kwargs)

    def file_sync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if path.name in {DATABASE, "collection.pending.json", "batch-000000"}:
            observed.append("sync:" + path.name)
        return sync(fd)

    def install(source, target, **kwargs):
        observed.append("rename:" + Path(target).name)
        return rename(source, target, **kwargs)

    def catalog_seal(*args, **kwargs):
        observed.append("catalog")
        return catalog(*args, **kwargs)

    monkeypatch.setattr(promotion, "validate_database", verification)
    monkeypatch.setattr(promotion, "catalog_update", catalog_seal)
    monkeypatch.setattr(os, "fsync", file_sync)
    monkeypatch.setattr(os, "rename", install)
    _, owner = _promote(tmp_path, sealed)
    owner.close()
    assert observed == [
        "integrity", f"sync:{DATABASE}", "sync:collection.pending.json",
        f"rename:{DATABASE}", "rename:collection.pending.json", "sync:batch-000000", "catalog",
    ]


def test_catalog_commit_then_sync_failure_retains_complete_descriptor(
    tmp_path: Path, sealed, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gf_training import rust_player_promotion as promotion

    catalog, sync = promotion.catalog_update, os.fsync
    catalog_committed = False

    def catalog_seal(*args, **kwargs):
        nonlocal catalog_committed
        result = catalog(*args, **kwargs)
        catalog_committed = True
        return result

    def fail_catalog_sync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if catalog_committed and path == tmp_path / "hdd/catalog":
            raise OSError(errno.EIO, "injected catalog directory sync failure")
        return sync(fd)

    monkeypatch.setattr(promotion, "catalog_update", catalog_seal)
    monkeypatch.setattr(os, "fsync", fail_catalog_sync)
    with pytest.raises(OSError, match="catalog directory sync"):
        _promote(tmp_path, sealed)
    assert catalog_committed
    _, descriptor, raw = sealed
    final = tmp_path / "hdd/catalog/run-1/collections/batch-000000"
    assert (final / "collection.pending.json").read_bytes() == raw
    assert sha256((final / DATABASE).read_bytes()).hexdigest() == descriptor["database_sha256"]


def test_descriptor_rename_failure_cannot_publish_source(tmp_path: Path, sealed,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    rename = os.rename

    def fail_marker(source, target, **kwargs):
        if Path(target).name == "collection.pending.json":
            raise OSError(errno.EIO, "injected descriptor rename failure")
        return rename(source, target, **kwargs)

    monkeypatch.setattr(os, "rename", fail_marker)
    with pytest.raises(OSError, match="descriptor rename"):
        _promote(tmp_path, sealed)
    final = tmp_path / "hdd/catalog/run-1/collections/batch-000000"
    assert {path.name for path in final.iterdir()} == {DATABASE}
    assert not (tmp_path / "hdd/catalog/catalog.sqlite3").exists()


def test_workspace_released_only_after_original_workers_retire(tmp_path: Path) -> None:
    from gf_training.rust_player_collection import RustPlayerCollector

    work = tmp_path / "volatile"
    work.mkdir()
    (work / DATABASE).write_bytes(b"retained until join")
    info = work.stat()
    collector = object.__new__(RustPlayerCollector)
    collector._work = work, info.st_dev, info.st_ino
    collector._active_writer = SimpleNamespace(staging_input_retired=lambda: False)
    collector._matches = SimpleNamespace(_pending_http={})
    generator = SimpleNamespace(done=lambda: False)
    server = SimpleNamespace(stopped=True, drainers=[generator])
    collector._servers = {0: server}
    with pytest.raises(ContractError, match="original activity"):
        collector.retire_work()
    assert work.is_dir()

    collector._active_writer.staging_input_retired = lambda: True
    with pytest.raises(ContractError, match="original activity"):
        collector.retire_work()
    assert work.is_dir()

    generator.done = lambda: True
    collector.retire_work()
    assert not work.exists() and collector._work is None


def test_removed_tmpfs_is_refused_before_writer_or_simulator(tmp_path: Path) -> None:
    from gf_training.rust_player_collection import RustPlayerCollector

    collector = object.__new__(RustPlayerCollector)
    collector._work = None
    collector._matches = SimpleNamespace(_pending_http={})
    collector._servers = {}
    with pytest.raises(ContractError, match="admitted tmpfs"):
        collector._create_work(tmp_path, tmp_path.stat().st_dev + 1)
    assert collector._active_writer is None
    assert collector._work is not None and list(collector._work[0].iterdir()) == []
    collector.retire_work()
    assert list(tmp_path.iterdir()) == []


def test_hpo_accounting_failure_retires_unclaimed_durable_collection(tmp_path: Path, sealed) -> None:
    from gf_training.rust_player_collection import _ISSUER, RustCollectedBatch
    from gf_training.rust_player_learning import RustHpoOperations

    final, pending = _promote(tmp_path, sealed)
    closed = []
    registry = SimpleNamespace(pending=None, policies={})
    collector = SimpleNamespace(registry=registry, _inventory=None, _source_control_charge=None)
    plan = SimpleNamespace(output=tmp_path, linkage={"batch_id": "batch-000000"}, execution_deadline=pending.database.deadline)
    writer = SimpleNamespace(_workspace_lease=None)
    trajectory = SimpleNamespace(close=lambda: closed.append("trajectory"))
    collector.current = RustCollectedBatch(_ISSUER, collector, plan, writer, pending, "sha", "capture", "retirement",
                                           training=(trajectory,))

    async def collect(request, actual_registry):
        assert request is plan and actual_registry is registry
        return collector.current

    def fail_accounting(path, record, **kwargs):
        assert path.name == "accounting-batch-000000.json"
        assert record["phase"] == "sealed-before-source"
        raise OSError(errno.EIO, "injected pre-Source accounting failure")

    collector.collect = collect
    ops = object.__new__(RustHpoOperations)
    ops._ownership = SimpleNamespace(directory=tmp_path)
    ops.registry, ops.collector = registry, collector
    ops.pool = SimpleNamespace(_base_charge=0, budget=SimpleNamespace(limit=10, used=0, peak=0))
    ops.writer = SimpleNamespace(write=fail_accounting)
    with pytest.raises(OSError, match="pre-Source accounting"):
        asyncio.run(ops.collect(plan, registry))
    assert registry.pending is None
    pending.recheck()
    ops.close_batch()
    assert pending.fd < 0 and pending.database.fd < 0
    assert closed == ["trajectory"] and collector.current._closed
    assert (final / "collection.pending.json").read_bytes() == sealed[2]
