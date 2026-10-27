"""Owned native work and independent durable completion/readiness regression tests."""
import asyncio
import hashlib
import json
import time
from dataclasses import replace

import pytest

from gf_cli.bindings import native_preflight, qualify_training_bundle
from gf_common.domain import ActionConfig, ContractError, RuntimePolicyBinding, canonical
from gf_models.lightgbm.model import inspect_artifact
from gf_models.lightgbm.policy import manifest_digest
from gf_simulator_client.workers import ParticipantWorker
from gf_training.lightgbm_run import run_numeric
from tests import test_lightgbm_native
from tests.worker_fixtures import stuck_reset_factory

toy_data = test_lightgbm_native.toy_data


def fit(directory, toy_data, **options):
    model, training, x, y = toy_data
    return run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training, seed=17, **options)


def test_parent_never_imports_or_constructs_native_during_fit(toy_data, tmp_path, monkeypatch):
    import gf_models.lightgbm.model as native
    monkeypatch.setattr(native, "runtime", lambda: pytest.fail("native execution belongs in child"))
    monkeypatch.setattr(native, "load_artifact", lambda *a, **k: pytest.fail("parent cannot load Booster"))
    result = asyncio.run(fit(tmp_path / "model", toy_data))
    assert result["status"] == "complete" and result["health"]["exit_code"] == 0
    assert result["child_validation"]["validated_model_sha256"]


def test_existing_worker_initialization_only_is_bounded_and_reaped():
    async def run():
        worker = ParticipantWorker(stuck_reset_factory, "slow", 17, ActionConfig(), .05)
        with pytest.raises(TimeoutError):
            await worker.initialize()
        assert worker.closed and worker.process.poll() is not None and not worker.io_thread.is_alive()
    asyncio.run(run())


def test_native_preflight_reuses_ready_handshake_without_parent_native(toy_data, tmp_path, monkeypatch):
    import gf_models.lightgbm.model as native
    directory = tmp_path / "model"
    asyncio.run(fit(directory, toy_data))
    checkpoint = directory / "checkpoint"
    monkeypatch.setattr(native, "runtime", lambda: pytest.fail("parent native runtime"))
    registry = {"toy-component-only": RuntimePolicyBinding("lightgbm", {"checkpoint": str(checkpoint),
                "expected_manifest_digest": manifest_digest(str(checkpoint)), "allow_toy": True})}
    asyncio.run(native_preflight(registry, ActionConfig(), deadline=time.monotonic() + 5, worker_seconds=2))


@pytest.mark.parametrize("name", ["request.json", "arrays.npz"])
def test_child_verifies_parent_exact_input_hash_before_fit(toy_data, tmp_path, monkeypatch, name):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / "tampered"
    original = owner.asyncio.create_subprocess_exec
    async def changed(*args, **kwargs):
        with (directory / name).open("ab") as output:
            output.write(b"\n")
        return await original(*args, **kwargs)
    monkeypatch.setattr(owner.asyncio, "create_subprocess_exec", changed)
    with pytest.raises(RuntimeError, match="native fit failed"):
        asyncio.run(fit(directory, toy_data))
    record = json.loads((directory / "training.json").read_text())
    assert record["input_hashes"]["request_sha256"]
    assert record["input_hashes"]["arrays_sha256"]
    assert "digest mismatch" in (directory / "stderr.log").read_text()
    assert not (directory / "checkpoint").exists()


def test_readiness_is_published_last_after_cleanup_and_durable_completion(toy_data, tmp_path, monkeypatch):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / "ordered"
    original_stop, original_atomic = owner._stop, owner.atomic_json
    events = []
    async def stop(*args, **kwargs):
        assert json.loads((directory / "staging" / "inference.json").read_text())["ready"] is False
        assert json.loads((directory / "training.json").read_text())["status"] == "running"
        await original_stop(*args, **kwargs)
        events.append("cleanup")
    def atomic(path, value):
        if path.name == "training.json" and value.get("status") == "complete":
            assert "cleanup" in events and value["health"]["exit_code"] == 0
            events.append("completion")
        if path.name == "inference.json" and value.get("ready"):
            assert events == ["cleanup", "completion"]
            record = json.loads((directory / "training.json").read_text())
            expected = hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()
            assert record["inference_manifest_sha256"] == expected
            events.append("ready")
        original_atomic(path, value)
    monkeypatch.setattr(owner, "_stop", stop)
    monkeypatch.setattr(owner, "atomic_json", atomic)
    result = asyncio.run(fit(directory, toy_data))
    assert result["status"] == "complete" and events == ["cleanup", "completion", "ready"]


def test_cleanup_failure_never_publishes_readiness(toy_data, tmp_path, monkeypatch):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / "cleanup-failure"
    original_stop = owner._stop
    async def stop(*args, **kwargs):
        await original_stop(*args, **kwargs)
        raise OSError("controlled cleanup failure")
    monkeypatch.setattr(owner, "_stop", stop)
    with pytest.raises(OSError, match="cleanup failure"):
        asyncio.run(fit(directory, toy_data))
    assert not (directory / "checkpoint").exists()
    assert json.loads((directory / "staging" / "inference.json").read_text())["ready"] is False


@pytest.mark.parametrize("failed_store", ["artifact", "run"])
def test_either_durable_failure_store_blocks_formal_qualification(toy_data, tmp_path, monkeypatch, failed_store):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / failed_store
    model, training, x, y = toy_data
    training = replace(training, wall_seconds=2.5, cleanup_seconds=.3)
    original = owner.atomic_json
    def atomic(path, value):
        if failed_store == "run" and path.name == "training.json" and value.get("status") == "failed":
            raise OSError("run correction unavailable")
        if failed_store == "artifact" and path.name == "inference.json" and value.get("ready") is False:
            raise OSError("artifact correction unavailable")
        original(path, value)
        if path.name == "inference.json" and value.get("ready") is True:
            time.sleep(2.6)  # detected overrun after the ready write
    monkeypatch.setattr(owner, "atomic_json", atomic)
    with pytest.raises(TimeoutError):
        asyncio.run(owner.run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training, seed=17))
    checkpoint = directory / "checkpoint"
    record = json.loads((directory / "training.json").read_text())
    artifact = json.loads((checkpoint / "inference.json").read_text())
    if failed_store == "artifact":
        assert record["status"] == "failed" and artifact["ready"] is True
        # Control-only qualification fixture: the actual model remains Toy and
        # is never used as a trained comparison artifact.
        record["purpose"] = "trained"
        (directory / "training.json").write_text(canonical(record))
        with pytest.raises(ContractError, match="completed trained run"):
            qualify_training_bundle(checkpoint, manifest_digest(str(checkpoint)))
    else:
        assert record["status"] == "complete" and artifact["ready"] is False
        with pytest.raises(ContractError, match="unready"):
            inspect_artifact(checkpoint, allow_toy=True)


def test_returned_time_includes_final_persistence_and_persisted_scope_is_explicit(toy_data, tmp_path, monkeypatch):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / "measured"
    original = owner.atomic_json
    def atomic(path, value):
        original(path, value)
        if path.name == "inference.json" and value.get("ready"):
            time.sleep(.2)
    monkeypatch.setattr(owner, "atomic_json", atomic)
    started = time.monotonic()
    result = asyncio.run(fit(directory, toy_data, charged_seconds=.4))
    observed = time.monotonic() - started
    persisted = json.loads((directory / "training.json").read_text())
    assert result["elapsed_seconds"] >= observed - .03
    assert result["elapsed_seconds"] >= persisted["elapsed_seconds"] + .2
    assert result["charged_total_seconds"] == .4 + result["elapsed_seconds"]
    assert persisted["timing_scope"].startswith("pre-write snapshot")
    assert result["timing_scope"].startswith("returned measurement")


def test_failed_completion_write_cannot_publish_ready_even_if_invalidation_fails(toy_data, tmp_path, monkeypatch):
    import gf_training.lightgbm_run as owner
    directory = tmp_path / "completion-failure"
    original = owner.atomic_json
    ready_writes = []
    def atomic(path, value):
        if path.name == "training.json" and value.get("status") == "complete":
            raise OSError("completion unavailable")
        if path.name == "inference.json":
            if value.get("ready"):
                ready_writes.append(path)
            else:
                raise OSError("invalidation unavailable")
        original(path, value)
    monkeypatch.setattr(owner, "atomic_json", atomic)
    with pytest.raises(OSError, match="completion unavailable"):
        asyncio.run(fit(directory, toy_data))
    assert not ready_writes
    assert json.loads((directory / "checkpoint" / "inference.json").read_text())["ready"] is False
    assert json.loads((directory / "training.json").read_text())["status"] == "failed"


def test_native_load_rejects_before_any_simulator_listener(toy_data, tmp_path, monkeypatch):
    import gf_cli
    import gf_cli.bindings as binding_module
    from gf_cli.config import load_config
    directory = tmp_path / "model"
    asyncio.run(fit(directory, toy_data))
    checkpoint = directory / "checkpoint"
    (checkpoint / "model.txt").write_text("not a native model\n")
    artifact = json.loads((checkpoint / "inference.json").read_text())
    artifact["model_sha256"] = hashlib.sha256((checkpoint / "model.txt").read_bytes()).hexdigest()
    (checkpoint / "inference.json").write_text(canonical(artifact))
    sha = manifest_digest(str(checkpoint))
    # Explicit low-level Toy test binding: the production qualifier is tested
    # separately, and no Toy artifact is relabeled as a comparison model.
    registry = {"test-toy": RuntimePolicyBinding("lightgbm", {"checkpoint": str(checkpoint), "expected_manifest_digest": sha, "allow_toy": True})}
    monkeypatch.setattr(binding_module, "bindings", lambda *a, **k: (registry, ("test-toy", "random")))
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", lambda *a, **k: pytest.fail("must not create Simulator listener"))
    loaded = load_config("configs/experiments/smoke.toml", {"model": {"model_id": "lightgbm"}, "run": {"output_dir": str(tmp_path / "episodes")}})
    with pytest.raises((RuntimeError, EOFError)):
        asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert not (tmp_path / "episodes").exists()


@pytest.mark.parametrize("damage", ["missing", "hash", "health", "budget", "checkpoint"])
def test_formal_completion_qualification_is_parent_only_and_fail_closed(toy_data, tmp_path, damage):
    directory = tmp_path / "bundle"
    asyncio.run(fit(directory, toy_data))
    checkpoint = directory / "checkpoint"
    sha = manifest_digest(str(checkpoint))
    record = json.loads((directory / "training.json").read_text())
    # Parent-control unit fixture only: model/inference bytes remain purpose Toy.
    record["purpose"] = "trained"
    if damage == "missing":
        (directory / "training.json").unlink()
    else:
        if damage == "hash":
            record["inference_manifest_sha256"] = "0" * 64
        elif damage == "health":
            record["health"]["unknown_failure"] = "bad"
        elif damage == "budget":
            record["charged_total_seconds"] = record["wall_seconds"] + 1
        else:
            record["checkpoint"] = "foreign"
        (directory / "training.json").write_text(canonical(record))
    with pytest.raises(ContractError):
        qualify_training_bundle(checkpoint, sha)
