"""Disposition of the original managed pair HTTP stream."""
from __future__ import annotations

import asyncio
import json
import time
from compression import zstd
from pathlib import Path

import httpx
import pytest

from tests.test_http import client_for, info, response


def _capture(tmp_path, *, retention="errors", compression="zstd"):
    from gf_simulator_client.http_diagnostics import HttpPairCapture
    return HttpPairCapture(tmp_path / "ram", tmp_path / "archive",
        operation_id="operation", collection_id="collection", pair_id="pair",
        retention=retention, compression=compression)


def _records(*statuses):
    records = []
    for index, status in enumerate(statuses):
        common = {"request_id": f"req-{index}", "game_id": None, "player_id": None}
        records.extend([
            {"kind": "request", "method": "GET", "path": "/v2/info", **common},
            {"kind": "response", "status": status, "duration_ms": 1, **common},
        ])
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def _settle(capture, **kwargs):
    return capture.settle(deadline=time.monotonic() + 5, **kwargs)


def test_successful_pair_cleans_tmpfs_without_opening_archive(tmp_path):
    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(_records(200, 204))
    report = _settle(capture, stopped=True, responses=2)
    assert report["status"] == "discarded"
    assert report["raw_bytes"] == len(_records(200, 204))
    assert not source.parent.exists()
    assert not (tmp_path / "archive").exists()


@pytest.mark.parametrize(("retention", "compression", "statuses", "abnormal", "transport"), [
    ("errors", "zstd", (200, 429, 200), False, 0),
    ("errors", "none", (200, 503, 204), False, 0),
    ("errors", "zstd", (200, 204), False, 1),
    ("errors", "zstd", (200, 204), True, 0),
    ("all", "zstd", (200, 204), False, 0),
])
def test_retains_original_pair_bytes(tmp_path, retention, compression, statuses, abnormal, transport):
    capture = _capture(tmp_path, retention=retention, compression=compression)
    raw = _records(*statuses)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(raw)
    report = _settle(capture, stopped=True, abnormal=abnormal, responses=len(statuses),
                     transport_errors=transport)
    archive = Path(report["archive"])
    assert report["status"] == "archived"
    assert report["raw_bytes"] == len(raw)
    assert report["archive_bytes"] == archive.stat().st_size
    assert archive.relative_to(tmp_path / "archive").parts[:2] == ("operation", "collection")
    assert archive.parent.stat().st_mode & 0o777 == 0o700
    assert archive.stat().st_mode & 0o777 == 0o600
    assert archive.suffix == (".zst" if compression == "zstd" else ".jsonl")
    got = zstd.decompress(archive.read_bytes()) if compression == "zstd" else archive.read_bytes()
    assert got == raw
    assert not source.exists()


@pytest.mark.parametrize("raw", [b"{broken", _records(200).splitlines(keepends=True)[0],
    _records(200), _records(200) + _records(204).splitlines(keepends=True)[1]])
def test_invalid_incomplete_or_underreported_raw_capture_is_archived(tmp_path, raw):
    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(raw)
    report = _settle(capture, stopped=True, responses=2, transport_errors=1)
    assert report["status"] == "archived"
    assert report["capture_error"]
    assert zstd.decompress(Path(report["archive"]).read_bytes()) == raw
    assert not source.exists()


def test_missing_raw_capture_stays_reachable(tmp_path):
    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.unlink()
    report = _settle(capture, stopped=True, responses=2)
    assert report["status"] == "capture_failed"
    assert report["residual"] == str(source)
    assert (source.parent / "diagnostic.json").exists()


def test_uncertain_reap_preserves_original_until_positive_cleanup(tmp_path):
    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(_records(200))
    report = _settle(capture, stopped=False, responses=1)
    assert report["status"] == "stop_unconfirmed"
    assert report["residual"] == str(source)
    assert source.exists()
    stopped = _settle(capture, stopped=True, abnormal=True, responses=1)
    assert stopped["status"] == "archived"
    assert not source.exists()


def test_archive_creation_failure_keeps_bytes_and_durable_locator(tmp_path):
    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(_records(500))
    (tmp_path / "archive").touch()
    report = _settle(capture, stopped=True, responses=1)
    assert report["status"] == "archive_failed"
    assert report["residual"] == str(source)
    assert report["error"]
    assert source.read_bytes() == _records(500)
    assert json.loads((source.parent / "diagnostic.json").read_bytes()) == report


@pytest.mark.parametrize("kind", ["staging_symlink", "archive_symlink", "archive_public"])
def test_pair_scope_cannot_write_in_linked_or_public_directory(tmp_path, kind):
    from gf_common.domain import ContractError

    capture = _capture(tmp_path, retention="all")
    elsewhere = tmp_path / "foreign"
    elsewhere.mkdir(mode=0o700)
    root = tmp_path / ("ram" if kind == "staging_symlink" else "archive")
    root.mkdir(mode=0o700)
    if kind.endswith("symlink"):
        (root / "operation").symlink_to(elsewhere, target_is_directory=True)
    else:
        (root / "operation").mkdir(mode=0o755)
    if kind.startswith("staging"):
        with pytest.raises(ContractError):
            capture.prepare()
    else:
        source = capture.prepare() / "http.jsonl"
        source.write_bytes(_records(200))
        report = _settle(capture, stopped=True, responses=1)
        assert report["status"] == "archive_failed"
        assert report["residual"] == str(source)
        assert source.is_file()
    assert not tuple(elsewhere.iterdir())


@pytest.mark.parametrize("drop_status", [None, 503])
def test_recovered_transport_and_received_error_status_are_sticky(drop_status):
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"incomplete":'
            raise httpx.ReadError("body disconnected after headers")

    async def run():
        value = info()
        attempts = 0

        def handler(request):
            nonlocal attempts
            if request.url.path == "/debug/v1/test-server/info":
                return response(value)
            attempts += 1
            if attempts == 1:
                if drop_status:
                    return httpx.Response(drop_status, stream=BrokenBody())
                raise httpx.ReadTimeout("first delivery lost")
            return response({"recovered": True}, status=200 if attempts == 2 else 503)

        from gf_simulator_client.http_diagnostics import HttpFacts
        client = client_for(handler, value, http_facts=HttpFacts())
        async with client.http:
            await client.attest()
            assert await client._request("POST", "/first", retry=True) == {"recovered": True}
            from gf_simulator_client.http import ApiError
            with pytest.raises(ApiError):
                await client._request("GET", "/second")
            assert client.http_facts.transport_errors == 1
            assert client.http_facts.responses == 4 + (drop_status is not None)
            assert client.http_facts.error_responses == 1 + (drop_status is not None)

    asyncio.run(run())


@pytest.mark.integration
def test_native_terminal_sticky_transport_is_outside_original_record(tmp_path):
    import os

    binary = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_SERVER")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    source = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_SOURCE")
    if not binary or not config or not source:
        pytest.skip("explicit managed Simulator binary/config/approved source required")

    from gf_common.card_instance import shared_card_instance_contract
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.process import ManagedServer
    from gf_simulator_client.rust_player import RustPlayerRuntime
    from tests.test_rust_player_runtime import drain, fault_proxy, game_spec, runtime_config

    async def run():
        capture = _capture(tmp_path)
        evidence = CardInstanceRuntimeEvidence(Path(binary), Path(source),
                                              contract=shared_card_instance_contract())
        evidence.preflight()
        server = ManagedServer(binary, config, "00000000-0000-0000-0000-000000000017", 2,
                               tmp_path / "server", http_capture=capture)
        owner = RustPlayerRuntime(runtime_config())
        try:
            startup = await server.start()
            receipt = evidence.verify_executed(server.process)
            with fault_proxy(startup, "drop_act") as (routed, state):
                specs = [game_spec(routed, receipt, episode=f"planned-{seat}") for seat in range(2)]
                for spec in specs:
                    spec["max_decisions"] = 1
                owner.start(specs)
                records = await asyncio.to_thread(drain, owner, 2, retain=True)
                assert state["dropped"]
            terminals = [row for row in records if row["kind"] == "terminal"]
            facts = [row["bundle"]._http_diagnostics() for row in terminals]
            assert sum(row[2] for row in facts) == 1
            assert all(row[3] for row in facts)
            assert all("http_diagnostics" not in row["data"] for row in records)
            assert sum(row["kind"] == "ack" for row in records) == 2
        finally:
            await asyncio.to_thread(owner.join)
            await server.stop()
        report = await server.settle_http(deadline=time.monotonic() + 5, abnormal=True,
            responses=sum(row[0] for row in facts), error_responses=sum(row[1] for row in facts),
            transport_errors=sum(row[2] for row in facts))
        assert report["status"] == "archived" and "capture_error" not in report, report
        assert report["logged_responses"] > report["observed_responses"]

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("fault", [None, "http", "early_exit"])
def test_real_managed_pair_routes_original_log_and_settles_after_stop(tmp_path, fault):
    import os

    from gf_simulator_client.process import ManagedServer

    binary = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_SERVER")
    config = os.environ.get("GFB_HTTP_DIAGNOSTICS_TEST_CONFIG")
    if not binary or not config:
        pytest.skip("explicit managed Simulator binary/config required")

    async def run():
        capture = _capture(tmp_path)
        server = ManagedServer(binary, config, "00000000-0000-0000-0000-000000000001", 2,
                               tmp_path / "server", http_capture=capture)
        try:
            startup = await server.start()
            assert capture.directory is not None
            assert Path(startup["http_log_path"]) == capture.directory / "http.jsonl"
            assert not (server.directory / "http").exists()
            live = await server.settle_http(deadline=time.monotonic() + 5, abnormal=False)
            assert live["status"] == "stop_unconfirmed"
            async with httpx.AsyncClient(base_url=startup["base_url"], trust_env=False, timeout=5) as client:
                request = {"rule_set_version": "true-match-1v1-v2",
                           "random_algorithm": "xoshiro256pp", "random_algorithm_version": "1"}
                for _ in range(2):
                    created = await client.post("/v2/games", json=request)
                    assert created.status_code == 201, created.text
                    assert (await client.delete("/v2/games/" + created.json()["game_id"])).status_code == 204
                if fault == "http":
                    exhausted = await client.post("/v2/games", json=request)
                    assert exhausted.status_code == 503, exhausted.text
            if fault == "early_exit":
                assert server.process is not None and server.process.stdin is not None
                server.process.stdin.close()
                assert await server.process.wait() == 0
        finally:
            await server.stop()
        report = await server.settle_http(deadline=time.monotonic() + 5, abnormal=False,
                                         responses=4 + (fault == "http"))
        assert report["status"] == ("archived" if fault else "discarded"), report
        assert report["logged_error_responses"] == int(fault == "http")
        assert report["abnormal"] == (fault == "early_exit")
        assert not capture.directory.exists()

    asyncio.run(run())


def test_saved_eval_accepts_diagnostic_failure_as_operational_metadata(tmp_path):
    from gf_common.card_instance_collection import hpo_evaluation_role
    from gf_common.preprocessing import PreparationConfig
    from gf_training.card_instance_collection_source import inspect_game_manifest
    from gf_training.structured_source import _SourceOwner
    from tests.test_hpo_evaluation_role import actual_metadata

    capture = _capture(tmp_path)
    source = capture.prepare() / "http.jsonl"
    source.write_bytes(_records(500))
    (tmp_path / "archive").touch()
    evidence = _settle(capture, stopped=True, responses=1)
    assert evidence["status"] == "archive_failed"

    directory, manifest, actions = actual_metadata(tmp_path)
    manifest["groups"][0]["http_diagnostics"] = evidence
    owner = _SourceOwner(directory, PreparationConfig(), time.monotonic() + 30)
    inspect_game_manifest(owner, manifest, actions, "validation", role=hpo_evaluation_role(4))


def test_archive_syncs_every_created_directory_entry_before_retiring_raw(tmp_path, monkeypatch):
    from gf_simulator_client import http_diagnostics

    capture = _capture(tmp_path, retention="all")
    capture.archive_root = tmp_path / "deep" / "first" / "second" / "archive"
    raw = capture.prepare() / "http.jsonl"
    raw.write_bytes(_records(200))
    sync_dir = http_diagnostics._sync_dir
    synchronized = set()

    def checked_sync(path):
        assert raw.is_file()
        sync_dir(path)
        synchronized.add(path)

    monkeypatch.setattr(http_diagnostics, "_sync_dir", checked_sync)
    report = _settle(capture, stopped=True, responses=1)
    assert report["status"] == "archived"
    assert {tmp_path, tmp_path / "deep", tmp_path / "deep" / "first",
            tmp_path / "deep" / "first" / "second", capture.archive_root} <= synchronized
    assert not raw.exists()


def test_cleanup_error_reports_the_surviving_directory_after_unlink(tmp_path, monkeypatch):
    capture = _capture(tmp_path, retention="all")
    stage = capture.prepare()
    raw = stage / "http.jsonl"
    raw.write_bytes(_records(200))
    rmdir = Path.rmdir

    def blocked_remove(path):
        if path == stage:
            raise OSError("fault during empty directory removal")
        return rmdir(path)

    monkeypatch.setattr(Path, "rmdir", blocked_remove)
    report = _settle(capture, stopped=True, responses=1)
    assert report["status"] == "cleanup_failed"
    assert report["residual"] == str(stage)
    assert Path(report["archive"]).is_file()
    assert not raw.exists()
    assert (stage / "diagnostic.json").is_file()


def test_ordinary_http_import_does_not_require_zstd(tmp_path):
    import subprocess
    import sys
    from textwrap import dedent

    code = dedent("""
        import sys

        class WithoutZstd:
            def find_spec(self, fullname, path, target=None):
                if fullname == 'compression.zstd':
                    raise ImportError('zstd unavailable for this check')
                return None

        sys.meta_path.insert(0, WithoutZstd())
        from pathlib import Path
        from time import monotonic
        from gf_simulator_client.http import AttestedClient
        from gf_simulator_client.http_diagnostics import HttpPairCapture
        assert AttestedClient is not None
        capture = HttpPairCapture(Path.cwd() / 'ram', Path.cwd() / 'archive',
            operation_id='operation', collection_id='collection', pair_id='pair',
            retention='all', compression='none')
        source = capture.prepare() / 'http.jsonl'
        source.write_bytes(b'capturable partial line')
        result = capture.settle(stopped=True, deadline=monotonic() + 5, abnormal=True)
        assert result['status'] == 'archived', result
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                               capture_output=True, text=True, timeout=5, check=False)
    assert completed.returncode == 0, completed.stderr


def test_managed_archive_deadline_keeps_task_and_raw_location(tmp_path, monkeypatch):
    import threading

    from gf_simulator_client.process import ManagedServer

    async def run():
        capture = _capture(tmp_path)
        stage = capture.prepare()
        raw = stage / "http.jsonl"
        raw.write_bytes(_records(200))
        config = Path(__file__).parents[1] / "configs" / "test-server.toml"
        server = ManagedServer("unused", str(config), "seed", 2, tmp_path / "server",
                               http_capture=capture)
        server.stopped = True
        release = threading.Event()
        original = capture.settle

        def blocked(**kwargs):
            release.wait(1)
            return original(**kwargs)

        monkeypatch.setattr(capture, "settle", blocked)
        deadline = time.monotonic() + 0.05
        try:
            pending = await server.settle_http(deadline=deadline, abnormal=True)
            assert pending["status"] == "settlement_unconfirmed"
            assert pending["residual"] == str(stage)
            assert raw.is_file()
        finally:
            release.set()
        assert server._http_settlement is not None
        await asyncio.shield(server._http_settlement)
        settled = await server.settle_http(deadline=deadline, abnormal=True)
        assert settled["status"] == "capture_failed"
        assert settled["residual"] == str(raw)

    asyncio.run(run())
