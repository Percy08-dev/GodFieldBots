"""Online native runtime: real scheduler/HTTP ownership, explicit writer acceptance."""

import json
import time
from pathlib import Path

import pytest

from gf_simulator_client.rust_player import RustPlayerRuntime

FIXTURE = Path(__file__).parent / "fixtures/rust_player_v5/originals.json"


def runtime_config(scheduler="tokio", workers=1):
    return {
        "scheduler": scheduler,
        "player_worker_threads": workers,
        "game_concurrency": 2,
        "max_batch_size": 2,
        "formation_wait_ms": 1.0,
        "storage_bytes": 64 * 1024 * 1024,
        "storage_queue": 2,
        "runtime_identity": "runtime-test",
    }


def test_runtime_cancel_wakes_storage_and_joins_original_threads():
    import concurrent.futures

    owner = RustPlayerRuntime(runtime_config())
    owner.start([])
    with concurrent.futures.ThreadPoolExecutor() as pool:
        waiting = pool.submit(owner.next_storage, time.monotonic() + 10)
        owner.cancel("test cancellation")
        with pytest.raises((ValueError, RuntimeError), match="test cancellation|closed"):
            waiting.result(2)
    report = owner.join()
    assert report["joined"] and report["games"] == []
    assert report["player_threads"] == 1


@pytest.mark.parametrize("kind", ["random", "heuristic", "refresh"])
def test_baseline_capacity_checked_before_native_start(kind):
    episode = json.loads(FIXTURE.read_text())["episodes"][0]
    actions = episode["actions"] | {"max_sequence_length": 19}
    owner = RustPlayerRuntime(runtime_config())
    with pytest.raises(ValueError, match="capacity"):
        owner.start(
            [
                {
                    "episode": "planned",
                    "run": "run",
                    "trial": "trial",
                    "server": {},
                    "actions": actions,
                    "rules": episode["rules"],
                    "numeric_costs": episode["numeric_costs"],
                    "policies": [{"kind": kind, "seed": 17}, {"kind": kind, "seed": 18}],
                    "max_decisions": 5,
                    "deadline": time.monotonic() + 10,
                    "cleanup_seconds": 2,
                }
            ]
        )
    owner.cancel("done")
    owner.join()


def assert_source_match():
    import hashlib

    import gf_native

    root = Path(__file__).parents[1] / "native"
    assets = [
        "structured-token-recipe-v1.json",
        "structured-token-recipe-public-exact-v2.json",
        "card-instance-token-recipe-corrected-v2.json",
        "card-instance-validation-pins-v2.json",
        "card-instance-formal-ppo-v1.json",
        "card-instance-formal-ppo-group-pack-v1.json",
        "card-instance-tree3-layout-recipe-v1.json",
    ]
    files = (
        list((root / "src").rglob("*.rs"))
        + [root / "build.rs"]
        + [root / "assets" / p for p in assets]
    )
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(str(file.relative_to(root)).encode() + b"\0" + file.read_bytes() + b"\0")
    info = dict(gf_native.build_info())
    assert info["native_source_sha256"] == digest.hexdigest()
    assert (
        info["cargo_lock_sha256"] == hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest()
    )
    assert info["profile"] == "release" and info["rustflags"] == ""
    return info


def test_loaded_native_matches_actual_sources():
    assert_source_match()


def game_spec(startup, receipt, *, episode="planned", deadline=None, kind="heuristic"):
    original = json.loads(FIXTURE.read_text())["episodes"][0]
    actions = original["actions"] | {
        "max_candidates": 64,
        "max_expansions": 512,
        "count_states": 128,
        "count_milliseconds": 1,
        "search_milliseconds": 1,
    }
    return {
        "episode": episode,
        "run": "run",
        "trial": "trial",
        "server": {
            "base_url": startup["base_url"],
            "expected_fingerprint": startup["test_server_info"]["seed_fingerprint"],
            "startup_info": startup["test_server_info"],
            "expected_effective": startup["test_server_config"],
            "request_seconds": 2.0,
        },
        "actions": actions,
        "rules": original["rules"],
        "numeric_costs": original["numeric_costs"],
        "policies": [{"kind": kind, "seed": 17}, {"kind": kind, "seed": 18}],
        "max_decisions": 8,
        "deadline": deadline or time.monotonic() + 30,
        "cleanup_seconds": 5.0,
        "runtime_evidence": receipt,
    }


def drain(owner, count, *, retain=False, initial_pending=0):
    records = []

    def accept(bundle):
        import hashlib

        view = bundle.view()
        assert bundle.transport_sha256 == hashlib.sha256(memoryview(view)).hexdigest()
        assert not view.flags.writeable
        data = json.loads(memoryview(view).tobytes())
        records.append(
            dict(
                kind=bundle.kind,
                episode=bundle.episode,
                game=bundle.game,
                data=data,
                **({"bundle": bundle, "view": view} if retain else {}),
            )
        )
        owner.storage_ack(bundle)

    # Both actual game actors must reach pending-before-Act simultaneously;
    # accepting neither first rules out a sequential replacement scheduler.
    held = [owner.next_storage(time.monotonic() + 10) for _ in range(initial_pending)]
    assert all(bundle.kind == "pending" for bundle in held)
    assert len({bundle.episode for bundle in held}) == initial_pending
    for bundle in held:
        accept(bundle)
    held.clear()
    while sum(r["kind"] == "terminal" for r in records) < count:
        accept(owner.next_storage(time.monotonic() + 30))
    owner.wait(time.monotonic() + 10)
    return records


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
@pytest.mark.parametrize("workers", [1, 2])
def test_pinned_simulator_two_games_four_participants(scheduler, workers, tmp_path):
    import asyncio

    from gf_common.card_instance import shared_card_instance_contract
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.process import ManagedServer

    simulator = Path("/tmp/gfb-shared-simulator-072dabb")
    binary = simulator / "target/release/gf-test-server"
    if not binary.exists():
        pytest.skip("approved external Simulator release unavailable")
    assert_source_match()

    async def scenario():
        evidence = CardInstanceRuntimeEvidence(
            binary, simulator, contract=shared_card_instance_contract()
        )
        evidence.preflight()
        servers = [
            ManagedServer(
                str(binary),
                str(Path(__file__).parents[1] / "configs/test-server.toml"),
                "00000000-0000-0000-0000-000000000017",
                1,
                tmp_path / str(i),
            )
            for i in range(2)
        ]
        owner = RustPlayerRuntime(runtime_config(scheduler, workers))
        try:
            starts = await asyncio.gather(*(server.start() for server in servers))
            receipts = [evidence.verify_executed(server.process) for server in servers]
            games = [
                game_spec(start, receipts[i], episode=f"planned-{i}")
                for i, start in enumerate(starts)
            ]
            owner.start(games)
            records = drain(owner, 2, initial_pending=2)
            report = owner.join()
            (tmp_path / "native-report.json").write_text(json.dumps(report, indent=2))
            (tmp_path / "native-records.json").write_text(json.dumps(records, indent=2))
            assert report["joined"] and report["player_threads"] == workers
            assert len(report["games"]) == 2
            assert all(g["game"] and g["game"] != g["episode"] for g in report["games"])
            assert all(
                g["decisions"] == 8
                and g["end_kind"] == "truncated"
                and g["error"] == "decision_limit"
                and g["cleanup_errors"] == []
                and g["storage_accepted"]
                for g in report["games"]
            ), report
            assert {
                (r["episode"], r["data"]["actor"]) for r in records if r["kind"] == "pending"
            } == {(f"planned-{i}", seat) for i in range(2) for seat in (1, 2)}
            for terminal in (r["data"] for r in records if r["kind"] == "terminal"):
                assert len(terminal["terminal_observations"]) == len(terminal["replays"]) == 2
                assert terminal["runtime_evidence"] in receipts
            return report
        finally:
            owner.cancel("test cleanup")
            owner.join()
            await asyncio.gather(*(server.stop() for server in servers))

    asyncio.run(scenario())


from contextlib import contextmanager


@contextmanager
def managed(tmp_path, *, count=1):
    import asyncio
    import threading

    from gf_common.card_instance import shared_card_instance_contract
    from gf_simulator_client.card_instance_runtime import CardInstanceRuntimeEvidence
    from gf_simulator_client.process import ManagedServer

    simulator = Path("/tmp/gfb-shared-simulator-072dabb")
    binary = simulator / "target/release/gf-test-server"
    if not binary.exists():
        pytest.skip("approved external Simulator release unavailable")
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    server = ManagedServer(
        str(binary),
        str(Path(__file__).parents[1] / "configs/test-server.toml"),
        "00000000-0000-0000-0000-000000000017",
        count,
        tmp_path / "server",
    )
    evidence = CardInstanceRuntimeEvidence(
        binary, simulator, contract=shared_card_instance_contract()
    )
    evidence.preflight()
    try:
        start = asyncio.run_coroutine_threadsafe(server.start(), loop).result(15)
        receipt = evidence.verify_executed(server.process)
        yield start, receipt
    finally:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result(35)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


@contextmanager
def fault_proxy(startup, mode):
    """Test-only HTTP fault injector; all game transitions still use the real engine."""
    import copy
    import http.client
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit

    upstream = urlsplit(startup["base_url"])
    state = {
        "calls": [],
        "dropped": False,
        "pages": {},
        "act_started": threading.Event(),
        "release": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.route()

        def do_POST(self):
            self.route()

        def do_DELETE(self):
            self.route()

        def route(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            state["calls"].append((self.command, self.path, body))
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            cursor = query.get("after", [None])[0]
            page = state["pages"].pop(cursor, None)
            if page is not None:
                status, headers, data = 200, {}, page
            else:
                connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=5)
                connection.request(
                    self.command, self.path, body=body, headers={"content-type": "application/json"}
                )
                response = connection.getresponse()
                status, headers = response.status, dict(response.getheaders())
                raw = response.read()
                connection.close()
                data = json.loads(raw) if raw else None
            if parsed.path == "/debug/v1/test-server/info" and mode == "wrong_attestation":
                data["server_instance_id"] = "00000000-0000-4000-8000-000000000099"
            if "/action-options/" in parsed.path and mode == "stale_hints":
                data["state_revision"] -= 1
            if "/events/" in parsed.path and mode == "stalled_cursor":
                data["has_more"] = True
                data["next_cursor"] = cursor
            if "/events/" in parsed.path and mode == "paginate" and page is None:
                next_cursor = f"fault-page-{len(state['calls'])}"
                rest = copy.deepcopy(data)
                rest["events"] = rest["events"][:]
                state["pages"][next_cursor] = rest
                data["events"] = []
                data["has_more"], data["next_cursor"] = True, next_cursor
            if self.command == "POST" and parsed.path.endswith("/actions"):
                state["act_started"].set()
                if mode == "drop_act" and not state["dropped"]:
                    state["dropped"] = True
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if mode == "mismatch_ack":
                    data["command_id"] = "00000000-0000-4000-8000-000000000099"
                if mode == "delay_ack":
                    state["release"].wait(5)
                if mode == "rejection":
                    status, data = 409, {"diagnostic": "scripted conflict"}
                if mode.startswith("abort_"):
                    data["status"] = "aborted"
                    data["result"] = None
                    data["action_options"] = None
                    if data["next_observation"] is not None:
                        data["next_observation"]["actionable"] = False
                        data["next_observation"]["typed_action_schema"] = []
            if "/replay/" in parsed.path and mode.startswith("abort_"):
                reason = (
                    "turn_limit_exceeded" if mode == "abort_turn_limit" else "player_action_timeout"
                )
                if mode == "abort_disagree" and parsed.path.endswith("/2"):
                    reason = "turn_limit_exceeded"
                data["episode_end"] = {
                    "type": "aborted",
                    "reason": reason,
                    "player": None,
                    "decision": {"turn": None, "decision_index": None, "invalid_attempt_count": 0},
                }
                if mode == "abort_unknown_natural":
                    data["episode_end"] = {"type": "terminal", "result": {"win": {"winner": 1}}}
            encoded = json.dumps(data).encode() if data is not None else b""
            try:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                if "cache-control" in headers:
                    self.send_header("cache-control", headers["cache-control"])
                self.end_headers()
                self.wfile.write(encoded)
            except BrokenPipeError, ConnectionResetError:
                pass

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=proxy.serve_forever)
    thread.start()
    routed = copy.deepcopy(startup)
    routed["base_url"] = f"http://127.0.0.1:{proxy.server_port}"
    try:
        yield routed, state
    finally:
        state["release"].set()
        proxy.shutdown()
        proxy.server_close()
        thread.join(5)


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
@pytest.mark.parametrize(
    "fault",
    [
        "wrong_attestation",
        "stale_hints",
        "stalled_cursor",
        "drop_act",
        "mismatch_ack",
        "rejection",
        "paginate",
    ],
)
def test_real_http_faults_stop_or_retry_same_command(scheduler, fault, tmp_path):
    assert_source_match()
    with managed(tmp_path) as (startup, receipt), fault_proxy(startup, fault) as (start, state):
        owner = RustPlayerRuntime(runtime_config(scheduler))
        try:
            spec = game_spec(start, receipt)
            spec["max_decisions"] = 1
            owner.start([spec])
            records = drain(owner, 1)
            report = owner.join()["games"][0]
            calls = [c for c in state["calls"] if c[0] == "POST" and c[1].endswith("/actions")]
            if fault == "drop_act":
                assert len(calls) == 2 and calls[0][2] == calls[1][2]
                first = state["calls"].index(calls[0])
                second = state["calls"].index(calls[1], first + 1)
                assert any(
                    c[1] == "/debug/v1/test-server/info" for c in state["calls"][first + 1 : second]
                )
                assert report["decisions"] == 1 and report["cleanup_errors"] == []
                assert sum(r["kind"] == "ack" for r in records) == 1
            elif fault == "paginate":
                assert any("after=fault-page" in c[1] for c in state["calls"])
                assert report["decisions"] == 1 and report["cleanup_errors"] == []
            else:
                assert report["decisions"] == 0 and report["outcome"] is None and report["error"]
                assert sum(r["kind"] == "ack" for r in records) == 0
                assert len(calls) == int(fault in ("mismatch_ack", "rejection"))
                if fault == "wrong_attestation":
                    assert not any(c[:2] == ("POST", "/v2/games") for c in state["calls"])
        finally:
            owner.cancel("fault test cleanup")
            owner.join()


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_cancel_delayed_ack_does_not_accept_experience(scheduler, tmp_path):
    import concurrent.futures

    with (
        managed(tmp_path) as (startup, receipt),
        fault_proxy(startup, "delay_ack") as (start, state),
    ):
        owner = RustPlayerRuntime(runtime_config(scheduler))
        owner.start([game_spec(start, receipt)])
        pending = owner.next_storage(time.monotonic() + 5)
        assert pending.kind == "pending"
        owner.storage_ack(pending)
        assert state["act_started"].wait(3)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            waiting = pool.submit(owner.next_storage, time.monotonic() + 20)
            owner.cancel("cancel delayed ACK")
            with pytest.raises(ValueError, match="cancel delayed ACK"):
                waiting.result(2)
        report = owner.join()["games"][0]
        assert report["end_kind"] == "cancelled" and report["decisions"] == 0
        assert report["outcome"] is None and not report["storage_accepted"]
        state["release"].set()


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_storage_backpressure_cancel_and_last_array_owner(scheduler, tmp_path):
    import gc

    with managed(tmp_path, count=2) as (startup, receipt):
        config = runtime_config(scheduler)
        config["storage_queue"] = 1
        owner = RustPlayerRuntime(config)
        owner.start([game_spec(startup, receipt, episode=f"planned-{i}") for i in range(2)])
        bundle = owner.next_storage(time.monotonic() + 5)
        view = bundle.view()
        retained = bundle.reserved_bytes
        deadline = time.monotonic() + 3
        while owner.storage_bytes() <= retained and time.monotonic() < deadline:
            time.sleep(0.005)
        # The second capacity-blocked producer holds its actual reservation once.
        assert owner.storage_bytes() > retained
        with pytest.raises(ValueError, match="duplicate or foreign|same immutable"):
            foreign = RustPlayerRuntime(runtime_config())
            foreign.storage_ack(bundle)
        foreign.cancel("done")
        foreign.join()
        owner.cancel("full storage cancellation")
        report = owner.join()
        assert len(report["games"]) == 2 and all(g["decisions"] == 0 for g in report["games"])
        del bundle
        gc.collect()
        assert (
            owner.storage_bytes() == retained
            and json.loads(view.tobytes())["schema"] == "rust-player-pending-v1"
        )
        with pytest.raises(ValueError):
            view.setflags(write=True)
        del view
        gc.collect()
        assert owner.storage_bytes() == 0


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_natural_outcome_and_idle_runtime_next_wave(scheduler, tmp_path):
    with managed(tmp_path, count=2) as (startup, receipt):
        owner = RustPlayerRuntime(runtime_config(scheduler, 2))
        all_records = []
        try:
            for index in range(2):
                spec = game_spec(
                    startup, receipt, episode=f"wave-{index}", deadline=time.monotonic() + 40
                )
                spec["max_decisions"] = 500
                owner.start([spec])
                records = drain(owner, 1)
                terminal = records[-1]["data"]
                assert terminal["report"]["end_kind"] == "natural", terminal["report"]
                assert terminal["report"]["outcome"] is not None
                assert terminal["report"]["cleanup_errors"] == []
                assert len(terminal["terminal_observations"]) == 2
                assert all(
                    obs["result"] == terminal["report"]["outcome"]
                    for obs in terminal["terminal_observations"]
                )
                all_records.extend(records)
            report = owner.join()
            (tmp_path / "natural-records.json").write_text(json.dumps(all_records))
            assert len(report["games"]) == 2 and all(g["storage_accepted"] for g in report["games"])
        finally:
            owner.cancel("natural test cleanup")
            owner.join()


@pytest.mark.parametrize("change", ["schema1", "binary", "revision"])
def test_actual_receipt_boundary_refuses_historical_or_changed_identity(change, tmp_path):
    import copy

    with managed(tmp_path) as (startup, receipt), fault_proxy(startup, "none") as (start, state):
        changed = copy.deepcopy(receipt)
        if change == "schema1":
            changed["schema_version"] = 1
        elif change == "binary":
            changed["binary"]["sha256"] = "0" * 64
        else:
            changed["source"]["revision"] = "0" * 40
        owner = RustPlayerRuntime(runtime_config())
        with pytest.raises(ValueError):
            owner.start([game_spec(start, changed)])
        assert state["calls"] == []
        owner.cancel("done")
        owner.join()


def actual_service(device="cpu"):
    from dataclasses import asdict

    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_models.rust_player import RustPlayerInference, serialized_torch
    from gf_models.structured_config import InferenceBindingV2, model_config_v3
    from gf_models.structured_network import make_network
    from tests.model3_support import model3

    service = RustPlayerInference(device=device, cache_entries=3, cache_weight_bytes=64 * 1024**2)
    raw = asdict(model3())
    raw["event"].update(max_interval=4096, max_episode=262144)
    with serialized_torch():
        network = make_network(
            model_config_v3(raw, layout=layout),
            17,
            InferenceBindingV2(max_candidates=64),
            layout=layout,
        )
    service.publish(network, "a" * 64)
    return service, network


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
@pytest.mark.parametrize("workers", [1, 2])
def test_actual_mlp_same_dispatcher_retains_then_releases_ended_game(scheduler, workers, tmp_path):
    service, _network = actual_service()
    original_evaluate = service.evaluate
    observed = {}
    batch_sizes = []

    def record_selected_scalars(batch):
        batch_sizes.append(len(batch["rows"]))
        answers = original_evaluate(batch)
        for answer in answers:
            identity = json.loads(answer["identity"])
            key = (identity["game"], identity["player"], identity["request"])
            observed[key] = (
                float(answer["logs"][answer["sample"]]),
                float(answer["value"]),
                answer["sample"],
            )
        return answers

    service.evaluate = record_selected_scalars
    config = runtime_config(scheduler, workers) | {
        "runtime_identity": service.runtime,
        "formation_wait_ms": 100.0,
    }
    with managed(tmp_path, count=2) as (startup, receipt):
        owner = RustPlayerRuntime(config, service=service)
        try:
            specs = [game_spec(startup, receipt, episode=f"mlp-{i}") for i in range(2)]
            for spec in specs:
                spec["policies"] = [
                    {"kind": "mlp", "seed": 17 + seat, "checkpoint": "a" * 64} for seat in (0, 1)
                ]
                spec["max_decisions"] = 4
            owner.start(specs)
            records = drain(owner, 2, initial_pending=2)
            assert (
                service.references["a" * 64] == 0
            )  # Released at actual episode handoff, before global close.
            assert service.calls["rows"] == 8
            assert 2 in batch_sizes  # Real mixed-game batch, one inference pending per game.
            for record in records:
                if record["kind"] == "pending":
                    score = record["data"]["scores"]
                    identity = score["identity"]
                    expected = observed[(identity["game"], identity["player"], identity["request"])]
                    assert score["logprob"].hex() == expected[0].hex()
                    assert score["value"].hex() == expected[1].hex()
                    assert score["sample"] == expected[2] and "logs" not in score
            assert all(
                r["data"]["scores"]["identity"]["game"] == r["game"]
                for r in records
                if r["kind"] == "pending"
            )
            assert {
                (r["episode"], r["data"]["actor"]) for r in records if r["kind"] == "pending"
            } == {(f"mlp-{i}", seat) for i in range(2) for seat in (1, 2)}
            service.evict("a" * 64, dispatcher=owner.dispatcher)
            report = owner.join()
            assert all(
                g["decisions"] == 4 and g["cleanup_errors"] == [] for g in report["games"]
            ), report
            assert report["python_service_threads"] == 1
        finally:
            owner.cancel("mlp test cleanup")
            owner.join()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cancel_actual_inflight_service_keeps_original_lease_and_retain_until_return(
    device, tmp_path
):
    import concurrent.futures
    import threading

    import torch

    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    service, _ = actual_service(device)
    entered, release = threading.Event(), threading.Event()
    original = service.evaluate

    def delayed(batch):
        answers = original(batch)
        entered.set()
        assert release.wait(10)
        return answers

    service.evaluate = delayed
    with managed(tmp_path) as (startup, receipt):
        owner = RustPlayerRuntime(
            runtime_config("fixed", 1) | {"runtime_identity": service.runtime}, service=service
        )
        spec = game_spec(startup, receipt)
        spec["policies"] = [
            {"kind": "mlp", "seed": 17 + seat, "checkpoint": "a" * 64} for seat in (0, 1)
        ]
        try:
            owner.start([spec])
            assert entered.wait(5)
            owner.cancel("cancel actual response wait")
            with pytest.raises(ValueError):
                owner.next_storage(time.monotonic() + 2)
            with concurrent.futures.ThreadPoolExecutor() as pool:
                joining = pool.submit(owner.join)
                with pytest.raises(concurrent.futures.TimeoutError):
                    joining.result(0.03)
                assert service.references["a" * 64] == 2
                with pytest.raises(ValueError, match="referenced"):
                    service.evict("a" * 64, dispatcher=owner.dispatcher)
                release.set()
                report = joining.result(10)
            assert report["games"][0]["decisions"] == 0
            assert report["games"][0]["end_kind"] == "cancelled"
            assert service.references["a" * 64] == 0
            assert owner.dispatcher.checkpoint_references() == {}
        finally:
            release.set()
            owner.cancel("done")
            owner.join()


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_original_join_does_not_block_concurrent_writer_acknowledgements(scheduler, tmp_path):
    import concurrent.futures

    with managed(tmp_path) as (startup, receipt):
        owner = RustPlayerRuntime(runtime_config(scheduler))
        spec = game_spec(startup, receipt)
        spec["max_decisions"] = 1
        owner.start([spec])
        bundle = owner.next_storage(time.monotonic() + 5)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            joining = pool.submit(owner.join)
            try:
                ack = pool.submit(owner.storage_ack, bundle)
                ack.result(2)
                while bundle.kind != "terminal":
                    bundle = owner.next_storage(time.monotonic() + 5)
                    owner.storage_ack(bundle)
                report = joining.result(5)
                assert report["games"][0]["decisions"] == 1
            finally:
                owner.cancel("concurrent join cleanup")
                joining.result(5)


def test_writer_failure_and_oversize_stop_before_any_act(tmp_path):
    service, _ = actual_service()
    with managed(tmp_path) as (startup, receipt):
        owner = RustPlayerRuntime(
            runtime_config() | {"runtime_identity": service.runtime}, service=service
        )
        spec = game_spec(startup, receipt)
        spec["policies"] = [
            {"kind": "mlp", "seed": 17 + seat, "checkpoint": "a" * 64} for seat in (0, 1)
        ]
        owner.start([spec])
        pending = owner.next_storage(time.monotonic() + 5)
        owner.storage_ack(pending, failure="disk full")
        assert service.references["a" * 64] == 2
        with pytest.raises(ValueError):
            owner.storage_ack(pending)
        report = owner.join()
        assert report["games"][0]["decisions"] == 0 and not report["games"][0]["storage_accepted"]
        assert service.references["a" * 64] == 0
        assert owner.join() == report and service.references["a" * 64] == 0
    with managed(tmp_path / "oversize") as (startup, receipt):
        owner = RustPlayerRuntime(runtime_config() | {"storage_bytes": 1})
        owner.start([game_spec(startup, receipt)])
        with pytest.raises(MemoryError, match="storage byte budget"):
            owner.next_storage(time.monotonic() + 5)
        report = owner.join()
        assert report["games"][0]["decisions"] == 0 and not report["games"][0]["storage_accepted"]
        assert owner.storage_bytes() == 0


def test_actual_model_limits_cannot_be_overridden_before_native_admission(tmp_path):
    service, _ = actual_service()
    with managed(tmp_path) as (startup, receipt), fault_proxy(startup, "none") as (start, state):
        owner = RustPlayerRuntime(
            runtime_config() | {"runtime_identity": service.runtime}, service=service
        )
        spec = game_spec(start, receipt)
        spec["policies"][0] = {
            "kind": "mlp",
            "seed": 17,
            "checkpoint": "a" * 64,
            "limits": {"max_positions": 128},
        }
        with pytest.raises(ValueError, match="actual model/binding"):
            owner.start([spec])
        assert state["calls"] == []
        owner.cancel("done")
        owner.join()
        assert service.references["a" * 64] == 0


@pytest.mark.parametrize("scheduler", ["tokio", "fixed"])
def test_quiesce_rejects_admission_then_joins_original_idle_workers(scheduler):
    owner = RustPlayerRuntime(runtime_config(scheduler, 2))
    owner.start([])
    owner.quiesce()
    with pytest.raises(ValueError, match="admission closed"):
        owner.start([])
    owner.wait(time.monotonic() + 1)
    assert owner.join()["player_threads"] == 2


@pytest.mark.parametrize(
    "fault", ["abort_turn_limit", "abort_timeout", "abort_disagree", "abort_unknown_natural"]
)
def test_scripted_paired_abort_closure_preserves_distinctions(fault, tmp_path):
    # Fault-injected response contract test; genuine natural-engine evidence is
    # separately captured by the unmodified pinned Simulator tests above.
    with managed(tmp_path) as (startup, receipt), fault_proxy(startup, fault) as (start, _):
        owner = RustPlayerRuntime(runtime_config("fixed", 1))
        owner.start([game_spec(start, receipt)])
        records = drain(owner, 1)
        report = owner.join()["games"][0]
        assert report["decisions"] == 1 and report["outcome"] is None
        if fault == "abort_turn_limit":
            assert (
                report["end_kind"] == "truncated"
                and report["simulator_abort_reason"] == "turn_limit_exceeded"
            )
            assert report["cleanup_errors"] == []
        elif fault == "abort_timeout":
            assert (
                report["end_kind"] == "aborted"
                and report["simulator_abort_reason"] == "player_action_timeout"
            )
            assert report["cleanup_errors"] == []
        else:
            assert report["end_kind"] == "aborted" and report["simulator_abort_reason"] is None
            assert any(
                "closure identities disagree" in e or "unknown natural outcome" in e
                for e in report["cleanup_errors"]
            )
        assert records[-1]["data"]["report"]["outcome"] is None


@pytest.mark.parametrize("phase", ["evaluate", "submit"])
def test_fix1_failed_service_cannot_release_terminal_holds_before_original_join(phase, tmp_path):
    import concurrent.futures
    import gc
    import threading
    import weakref
    from contextlib import suppress

    import torch

    service, _ = actual_service()
    owner = RustPlayerRuntime(
        runtime_config("fixed", 1) | {"runtime_identity": service.runtime}, service=service
    )
    reached, release = threading.Event(), threading.Event()
    tensors = []
    original_dispatcher = owner.dispatcher
    original_evaluate = service.evaluate

    class SettlementBarrier:
        def __init__(self):
            self.inner = threading.Condition()

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            result = self.inner.__exit__(*args)
            if (
                threading.current_thread().name == "gf-player-inference"
                and not owner._active_routes
            ):
                reached.set()
                assert release.wait(10)
            return result

        def __getattr__(self, name):
            return getattr(self.inner, name)

    owner._settled = SettlementBarrier()

    def reject_with_live_tensor(batch_id):
        tensor = torch.ones(4)
        tensors.append(weakref.ref(tensor))
        try:
            # Dispose the ORIGINAL exported native lease with an actually
            # rejected submission. The actor then ends and mints a real terminal
            # capability while the service's failed traceback still owns Tensor.
            original_dispatcher.submit_results(batch_id, [])
        except ValueError as exc:
            raise RuntimeError(f"backend-private-{phase}") from exc
        raise AssertionError("invalid original submission unexpectedly succeeded")

    if phase == "evaluate":

        def failing_evaluate(batch):
            answers = original_evaluate(batch)
            reject_with_live_tensor(batch["batch_id"])
            return answers

        service.evaluate = failing_evaluate
    else:

        class FailingSubmission:
            def __getattr__(self, name):
                return getattr(original_dispatcher, name)

            def submit_results(self, batch_id, _answers):
                reject_with_live_tensor(batch_id)

        owner.dispatcher = FailingSubmission()

    with managed(tmp_path) as (startup, receipt):
        spec = game_spec(startup, receipt)
        spec["policies"] = [
            {"kind": "mlp", "seed": 17 + seat, "checkpoint": "a" * 64} for seat in (0, 1)
        ]
        try:
            owner.start([spec])
            assert reached.wait(5)
            terminal = owner.next_storage(time.monotonic() + 5)
            assert terminal.kind == "terminal" and terminal.release_key is not None
            facts = json.loads(terminal.view().tobytes())
            assert facts["report"]["decisions"] == 0
            assert "backend-private" not in facts["report"]["error"]
            assert owner._active_routes == {} and tensors[0]() is not None
            owner.storage_ack(terminal)
            assert service.references["a" * 64] == 2
            assert owner._service_error is not None and tensors[0]() is not None
            with pytest.raises(ValueError, match="referenced"):
                service.evict("a" * 64, dispatcher=original_dispatcher)
            with concurrent.futures.ThreadPoolExecutor() as pool:
                receiver = pool.submit(owner.next_storage, time.monotonic() + 10)
                joining = pool.submit(owner.join)
                with pytest.raises(concurrent.futures.TimeoutError):
                    joining.result(0.03)
                assert service.references["a" * 64] == 2 and tensors[0]() is not None
                with pytest.raises((ValueError, RuntimeError)) as closed:
                    receiver.result(2)
                assert "backend-private" not in str(closed.value)
                release.set()
                with pytest.raises(RuntimeError, match=f"backend-private-{phase}"):
                    joining.result(5)
            gc.collect()
            assert tensors[0]() is None and service.references["a" * 64] == 0
            with pytest.raises(RuntimeError, match="native inference service failed"):
                owner.join()
            assert service.references["a" * 64] == 0 and service.calls["failures"] == 1
        finally:
            release.set()
            owner.cancel("failure regression cleanup")
            with suppress(RuntimeError):
                owner.join()


@pytest.mark.parametrize("threshold", [1, 12, 2**64 - 1, 2**64, 2**100])
def test_fix1_live_refresh_preserves_full_positive_integer_option(threshold, tmp_path):
    with managed(tmp_path) as (startup, receipt):
        owner = RustPlayerRuntime(runtime_config("fixed", 1))
        spec = game_spec(startup, receipt, kind="refresh")
        spec["max_decisions"] = 1
        for policy in spec["policies"]:
            policy["attack_threshold"] = threshold
        try:
            owner.start([spec])
            records = drain(owner, 1)
            pending = next(r["data"] for r in records if r["kind"] == "pending")
            assert pending["policy"]["attack_threshold"] == threshold
            from gf_common.card_instance import action_config_v3
            from gf_common.card_instance_rules import parse_card_instance_rules
            from gf_common.refresh_heuristic import RefreshOptions, refresh_policy_label

            actions = action_config_v3(spec["actions"])
            rules = parse_card_instance_rules(spec["rules"])
            captured = refresh_policy_label(
                actions,
                rules,
                RefreshOptions(attack_threshold=pending["policy"]["attack_threshold"]),
            )
            assert captured == refresh_policy_label(
                actions, rules, RefreshOptions(attack_threshold=threshold)
            )
            if threshold > 2**64 - 1:
                assert captured != refresh_policy_label(
                    actions, rules, RefreshOptions(attack_threshold=2**64 - 1)
                )
            assert all(p["attack_threshold"] == threshold for p in records[-1]["data"]["policies"])
            assert owner.join()["games"][0]["decisions"] == 1
        finally:
            owner.cancel("threshold regression cleanup")
            owner.join()


@pytest.mark.parametrize("threshold", [True, False, 1.0, 0, -1, 1e100])
def test_fix1_live_refresh_rejects_nonpositive_or_noninteger_before_http(threshold, tmp_path):
    with managed(tmp_path) as (startup, receipt), fault_proxy(startup, "none") as (start, state):
        owner = RustPlayerRuntime(runtime_config())
        spec = game_spec(start, receipt, kind="refresh")
        spec["policies"][0]["attack_threshold"] = threshold
        try:
            with pytest.raises(ValueError):
                owner.start([spec])
            assert state["calls"] == []
        finally:
            owner.cancel("invalid threshold cleanup")
            owner.join()
