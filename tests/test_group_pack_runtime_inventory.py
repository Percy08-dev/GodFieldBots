"""Metrics are immutable original bytes declared in the complete logical inventory."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from copy import deepcopy

import pytest

from gf_common.domain import canonical
from gf_common.group_pack import StorageSettings
from gf_training.group_pack_collection import GroupPackCollection
from gf_training.group_pack_publication import GroupPackCollectionPublication
from gf_training.group_pack_source import open_artifact_source


def audit(source):
    manifest = json.loads(source.read("manifest.json"))
    assert set(source.names()) == {row["path"] for row in manifest["files"]} | {"manifest.json", "report.json"}
    assert len(manifest["files"]) == len({row["path"] for row in manifest["files"]})
    saved_inventory = source.inventory()
    assert {entry["name"] for entry in saved_inventory} == set(source.names())
    for entry in saved_inventory:  # Includes both original control files too.
        raw = source.read(entry["name"])
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    for row in manifest["files"]:
        raw = source.read(row["path"])
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    report_raw = source.read("report.json")
    assert hashlib.sha256(report_raw).hexdigest() == manifest["report_sha256"]
    report = json.loads(report_raw)
    assert report["artifact_bytes"] == sum(row["bytes"] for row in manifest["files"])
    return manifest, report


def test_real_writer_freezes_exact_declared_metrics_and_final_metrics_stay_separate(tmp_path):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        row = owner.freeze_storage_metrics()
        assert row == owner.freeze_storage_metrics()
        assert owner.inventory == [row]
        owner._metric_add("write_seconds", 17.0)
        assert owner.freeze_storage_metrics() == row  # Never regenerate from live metrics.
        report = {"healthy": True, "artifact_bytes": row["bytes"]}
        report_raw = (canonical(report) + "\n").encode()
        manifest = {"schema_version": 5, "files": deepcopy(owner.inventory),
            "report_sha256": hashlib.sha256(report_raw).hexdigest()}
        original_manifest = (canonical(manifest) + "\n").encode()
        envelope = tmp_path / "envelope"
        envelope.mkdir()
        publication = GroupPackCollectionPublication(owner, envelope)
        await publication.prepare(report, manifest)
        result = publication.commit(report, manifest)
        source = open_artifact_source(result.locator, deadline=owner.deadline, max_owned_bytes=512 * 1024**2)
        try:
            saved, saved_report = audit(source)
            assert source.read("manifest.json") == original_manifest
            assert source.read("report.json") == report_raw
            frozen = json.loads(source.read("storage-metrics.json"))
            assert frozen["write_seconds"] != result.storage_metrics["write_seconds"]
            assert frozen["checkpoint_seconds"] is None
            assert result.storage_metrics["checkpoint_seconds"] >= 0
            assert saved == manifest and saved_report == report
        finally:
            source.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "length", "length-type", "no-freeze", "control-collision"])
def test_healthy_prepare_rejects_invalid_metrics_declaration_without_rewriting(tmp_path, mutation):
    async def run():
        owner = GroupPackCollection(StorageSettings(db_root=tmp_path / "db", ram_root=tmp_path / "ram"),
            "run", deadline=time.monotonic() + 20)
        await owner.start()
        try:
            row = owner.freeze_storage_metrics() if mutation != "no-freeze" else {
                "path": "storage-metrics.json", "sha256": "a" * 64, "bytes": 1}
            files = [dict(row)]
            if mutation == "missing":
                files = []
            elif mutation == "duplicate":
                files.append(dict(row))
            elif mutation == "hash":
                files[0]["sha256"] = "b" * 64
            elif mutation == "length":
                files[0]["bytes"] += 1
            elif mutation == "length-type":
                files[0]["bytes"] = float(files[0]["bytes"])
            controls = {"report.json": b'{"healthy":true}\n',
                "manifest.json": (canonical({"schema_version": 5, "files": files}) + "\n").encode()}
            if mutation == "control-collision":
                controls["storage-metrics.json"] = b"caller replacement"
            original = dict(controls)
            with pytest.raises(ValueError, match="metrics"):
                await owner.prepare(controls)
            assert controls == original
        finally:
            await owner.close()
    asyncio.run(run())


def test_metrics_freeze_rejects_inventory_collision_or_duplicate(tmp_path):
    async def run():
        for frozen in (False, True):
            root = tmp_path / str(frozen)
            owner = GroupPackCollection(StorageSettings(db_root=root / "db", ram_root=root / "ram"),
                "run", deadline=time.monotonic() + 20)
            await owner.start()
            try:
                row = owner.freeze_storage_metrics() if frozen else {
                    "path": "storage-metrics.json", "bytes": 0, "sha256": "a" * 64}
                owner.inventory.append(dict(row))
                with pytest.raises(ValueError, match="metrics"):
                    owner.freeze_storage_metrics()
            finally:
                await owner.close()
    asyncio.run(run())
