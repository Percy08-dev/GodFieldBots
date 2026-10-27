import asyncio
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.domain import ActionConfig, GroupResult
from gf_models.baselines import make_policy
from gf_simulator_client.environment import reconcile_replay
from gf_simulator_client.process import ManagedServer, server_effective
from gf_simulator_client.workers import ParticipantWorker
from tests.worker_fixtures import audit_factory, slow_factory, stuck_reset_factory

ROOT = Path(__file__).parents[1]
SEED = "00000000-0000-0000-0000-000000000001"


def test_spawn_worker_local_input_and_bounded_timeout(make_input):
    async def run():
        worker = ParticipantWorker(make_policy, "heuristic", 3, ActionConfig(), 3)
        assert worker.process.pid != os.getpid()
        candidates, selected = await worker.select(make_input())
        assert candidates.actions[selected.index]["type"] == "play_sequence"
        await worker.close()
        assert not worker.process.poll() is None
        slow = ParticipantWorker(slow_factory, "slow", 0, ActionConfig(), 0.1)
        with pytest.raises(TimeoutError):
            await slow.select(make_input())
        assert not slow.process.poll() is None
    asyncio.run(run())


def test_managed_process_startup_failure_drains_and_reaps(tmp_path):
    async def run():
        script = tmp_path / "fake-server"
        script.write_text(f'#!{sys.executable}\nimport sys\nprint("not startup", flush=True)\nprint("diagnostic", file=sys.stderr, flush=True)\nsys.stdin.read()\n')
        script.chmod(0o755)
        server = ManagedServer(str(script), str(ROOT / "configs/test-server.toml"), SEED, 2,
                               tmp_path / "server", startup_seconds=0.1, shutdown_seconds=0.5)
        with pytest.raises(TimeoutError):
            await server.start()
        assert server.process.returncode == 0
        assert server.health["shutdown"] == "stdin_eof"
        assert "diagnostic" in (tmp_path / "server/stderr.log").read_text()
        assert all(t.done() for t in server.drainers)
    asyncio.run(run())


def test_expected_lifecycle_config_validated_before_process(tmp_path):
    effective = server_effective(ROOT / "configs/test-server.toml", 7)
    assert effective["seed_use_limit"] == 7 and effective["server_idle_timeout_secs"] == 0
    path = tmp_path / "bad.toml"
    path.write_text((ROOT / "configs/test-server.toml").read_text().replace("max_game_turns = 2000", "max_game_turns = true"))
    with pytest.raises(ValueError):
        server_effective(path, 1)


def test_replay_only_matches_actor_acceptances():
    action = {"type": "play_sequence", "ordered_elements": [{"source": "hand_artifact", "instance_id": 1}]}
    outcome = {"win": {"winner": 1}}
    replay = {"viewer_player_id": 1, "observation_schema_version": "1", "episode_end": {"type": "terminal", "result": outcome},
              "turns": [{"decisions": [{"actor": 2, "visible_action": {"type": "concealed"}},
                                        {"actor": 1, "visible_action": {"type": "action", "action": {**action, "target": None}}}]}]}
    reconcile_replay(replay, 1, [action], outcome)
    with pytest.raises(ValueError):
        reconcile_replay(replay, 1, [], outcome)


def test_provenance_moved_binary_without_source_and_external_no_binary_access(tmp_path, monkeypatch):
    import gf_cli
    from gf_cli.config import load_config
    from gf_common.domain import EpisodeResult

    moved = tmp_path / "moved-server"
    moved.write_bytes(b"prebuilt binary with unknown build provenance")
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {"run": {"output_dir": str(tmp_path / "runs")}, "server": {"binary": str(moved), "source_dir": None}})
    constructed = []
    class FakeEnvironment:
        def __init__(self, settings, policy_factory, journal_factory):
            constructed.append(settings)
        async def run_group(self, group):
            return GroupResult([EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                               outcome="draw", end_kind="natural", first_player=2) for spec in group.episodes])
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", FakeEnvironment)
    directory, _ = asyncio.run(gf_cli.run_experiment(loaded, "smoke"))
    import json
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["simulator"]["revision"] is None
    assert manifest["simulator"]["verification"] == "unverified"
    assert manifest["binary_sha256"]
    assert manifest["binary_build_revision"] is None
    from gf_common.domain import POLICY_INPUT_VERSION, REVIEWED_SPEC_HASHES
    assert manifest["schema_version"] == 3
    assert manifest["policy_input_version"] == POLICY_INPUT_VERSION
    assert manifest["reviewed_simulator_spec_sha256"] == REVIEWED_SPEC_HASHES
    assert manifest["simulator"]["spec_sha256"] is None
    absent = str(tmp_path / "does-not-exist")
    external = replace(loaded.config.server, binary=absent, external_url="http://127.0.0.1:1234", expected_fingerprint="fixed")
    loaded = replace(loaded, config=replace(loaded.config, server=external))
    real_hash = gf_cli._file_hash
    def guarded_hash(path, deadline=float("inf")):
        assert str(path) != absent
        return real_hash(path, deadline)
    monkeypatch.setattr(gf_cli, "_file_hash", guarded_hash)
    directory, _ = asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["binary_sha256"] is None
    assert len(constructed) == 2


def test_wrong_explicit_source_rejected_before_environment(tmp_path, monkeypatch):
    import gf_cli
    from gf_cli.config import load_config
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {"server": {"source_dir": str(tmp_path)}})
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", lambda *a: pytest.fail("must not construct environment"))
    with pytest.raises(ValueError, match="source revision"):
        asyncio.run(gf_cli.run_experiment(loaded, "smoke"))


def test_large_payload_stuck_reset_timeout_reaps_io_thread(make_input):
    async def run():
        value = make_input()
        value.observation["padding_for_transport_test"] = "x" * 2_000_000
        # The child never reaches input validation: reset intentionally stalls.
        worker = ParticipantWorker(stuck_reset_factory, "slow", 0, ActionConfig(), 0.05)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await worker.select(value)
        assert time.monotonic() - started < 2
        assert not worker.process.poll() is None
        assert not worker.io_thread.is_alive()
        await worker.close()
    asyncio.run(run())


def test_server_stop_idempotent_preserves_forced_shutdown(tmp_path):
    async def run():
        script = tmp_path / "stubborn-server"
        script.write_text(f'#!{sys.executable}\nimport signal,time\nsignal.signal(signal.SIGTERM, lambda *a: None)\nprint("not startup", flush=True)\nwhile True: time.sleep(0.1)\n')
        script.chmod(0o755)
        server = ManagedServer(str(script), str(ROOT / "configs/test-server.toml"), SEED, 1, tmp_path / "server",
                               startup_seconds=0.05, shutdown_seconds=0.05)
        with pytest.raises(TimeoutError):
            await server.start()
        first = dict(server.health)
        await server.stop()
        assert server.health == first
        assert server.health["shutdown"] == "kill"
        assert server.process.returncode is not None
    asyncio.run(run())


def test_clean_bootstrap_scrubs_argv_and_environment_before_factory_import(tmp_path, make_input, monkeypatch):
    import sys
    async def run():
        parent_argv = list(sys.argv)
        parent_env = dict(os.environ)
        monkeypatch.setattr(sys, "argv", ["parent-main", "--set", "run.seeds=secret-game-seed-sentinel", "server.external_url=secret-server-url-sentinel"])
        monkeypatch.setenv("GFB_PRIVATE_SENTINEL", "secret-game-seed-sentinel/secret-server-url-sentinel")
        worker = ParticipantWorker(audit_factory, "audit", 0, ActionConfig(), 2)
        try:
            _, selected = await worker.select(make_input())
            assert selected.policy_version == "audit-v1"
            assert os.environ["GFB_PRIVATE_SENTINEL"].startswith("secret-game")
            assert "secret-game-seed-sentinel" in sys.argv[2]
        finally:
            await worker.close()
        # Only monkeypatch teardown restores parent globals; worker never changed them.
        assert parent_argv != sys.argv and parent_env != dict(os.environ)
    asyncio.run(run())


def test_worker_framing_truncated_oversized_and_partial_writes():
    import io

    from gf_common.domain import ContractError
    from gf_simulator_client.framing import MAX_FRAME_BYTES, read_frame, write_frame
    for encoded in (b"short", (100).to_bytes(8, "big") + b"{}"):
        with pytest.raises(EOFError):
            read_frame(io.BytesIO(encoded))
    with pytest.raises(ContractError):
        read_frame(io.BytesIO((MAX_FRAME_BYTES + 1).to_bytes(8, "big")))
    class PartialWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])
    output = PartialWriter()
    write_frame(output, {"test": "value"})
    output.seek(0)
    assert read_frame(output) == {"test": "value"}


def test_factory_descriptor_rejects_unimportable_local_before_launch():
    from gf_common.domain import ContractError
    def factory(model_id):
        return None
    with pytest.raises(ContractError, match="top-level"):
        ParticipantWorker(factory, "invalid", 0, ActionConfig())


@pytest.mark.parametrize("failure", ["timeout", "cancel", "frame", "worker_error"])
def test_worker_abort_has_zero_ordinary_grace_and_resistant_child_is_reaped(make_input, failure):
    from tests.worker_fixtures import (
        error_factory,
        malformed_frame_factory,
        resistant_reset_factory,
    )
    async def run():
        factory = malformed_frame_factory if failure == "frame" else error_factory if failure == "worker_error" else resistant_reset_factory
        worker = ParticipantWorker(factory, "fixture", 0, ActionConfig(), 0.2)
        close_args = []
        real_close = worker.close
        async def observed_close(timeout=2):
            close_args.append(timeout)
            await real_close(timeout=timeout)
        worker.close = observed_close
        started = time.monotonic()
        task = asyncio.create_task(worker.select(make_input()))
        if failure == "cancel":
            await asyncio.sleep(0.15)
            task.cancel()
        with pytest.raises((TimeoutError, asyncio.CancelledError, RuntimeError, ValueError)):
            await task
        assert close_args == [0]
        assert time.monotonic() - started < 0.85
        assert worker.process.poll() is not None
        assert worker.io_thread is None or not worker.io_thread.is_alive()
    asyncio.run(run())


def test_explicit_source_records_actual_normative_snapshot_hashes(tmp_path, monkeypatch):
    import hashlib
    import json

    import gf_cli
    from gf_cli.config import load_config
    from gf_common.domain import REVIEWED_SPEC_HASHES, SIMULATOR_REVISION, EpisodeResult

    source = tmp_path / "simulator-source"
    for name in REVIEWED_SPEC_HASHES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local user edited normative snapshot: " + name)
    binary = tmp_path / "prebuilt"
    binary.write_bytes(b"unknown build revision")
    loaded = load_config(ROOT / "configs/experiments/smoke.toml", {
        "run": {"output_dir": str(tmp_path / "runs")},
        "server": {"source_dir": str(source), "binary": str(binary)}})
    original_git = gf_cli._git
    def git(path, deadline=float("inf")):
        if path == source:
            return {"revision": SIMULATOR_REVISION, "dirty": True}
        return original_git(path, deadline)
    class FakeEnvironment:
        def __init__(self, *args):
            pass
        async def run_group(self, group):
            return GroupResult([EpisodeResult(spec.episode_id, spec.pair_id, group.seed, spec.policies,
                                               outcome="draw", end_kind="natural", first_player=2) for spec in group.episodes])
    monkeypatch.setattr(gf_cli, "_git", git)
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", FakeEnvironment)
    directory, _ = asyncio.run(gf_cli.run_experiment(loaded, "smoke"))
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["reviewed_simulator_spec_sha256"] == REVIEWED_SPEC_HASHES
    assert manifest["simulator"]["spec_sha256"] == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in REVIEWED_SPEC_HASHES}
    assert manifest["simulator"]["dirty"]
    assert manifest["binary_build_revision"] is None
