import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest

from gf_cli.bindings import bindings
from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, ContractError, canonical, digest
from gf_models.baselines import make_policy
from gf_models.lightgbm.config import EncoderConfig, ModelConfig, NetworkConfig, TrainingConfig
from gf_models.lightgbm.encoder import feature_names
from gf_models.lightgbm.model import (
    fit_numeric,
    inspect_artifact,
    load_artifact,
    native_parameters,
    save_artifact,
)
from gf_models.lightgbm.policy import LightGBMPolicy, manifest_digest
from gf_simulator_client.workers import ParticipantWorker
from gf_training.lightgbm_run import run_numeric


@pytest.fixture
def toy_data():
    pytest.importorskip("lightgbm")
    model = ModelConfig(model_id="lightgbm", encoder=EncoderConfig(hash_buckets=2), network=NetworkConfig(num_leaves=5))
    training = TrainingConfig(num_boost_round=30, early_stopping_rounds=3, min_data_in_leaf=1, learning_rate=.3, wall_seconds=15, cleanup_seconds=1)
    x = np.zeros((32, len(feature_names(model.encoder))))
    x[:, 0] = np.arange(32)
    y = np.where(x[:, 0] < 16, -1., 1.)
    return model, training, x, y


@pytest.fixture
def toy_artifact(tmp_path, toy_data):
    model, training, x, y = toy_data
    booster, metrics = fit_numeric(x, y, x, y, params=native_parameters(model, training, 17), rounds=training.num_boost_round,
                                  early_stopping=training.early_stopping_rounds, names=feature_names(model.encoder))
    path = tmp_path / "toy-artifact"
    save_artifact(booster, path, model, digest(asdict(ActionConfig())), purpose="toy", ready=True)
    return path, model, booster, metrics


def test_native_tiny_fit_save_load_predictions_early_stop_and_safe_text(toy_artifact, toy_data):
    path, model, booster, metrics = toy_artifact
    _, _, x, y = toy_data
    loaded, manifest = load_artifact(path, allow_toy=True, model=model, action_config_hash=digest(asdict(ActionConfig())))
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))
    assert metrics["validation_l2"] < .05 and 1 <= metrics["best_iteration"] <= 30
    # Flat validation labels exercise actual early stopping, independently of
    # successful interpolation on the matched toy train/validation distribution.
    stopped, _ = fit_numeric(x, y, x, -y, params=native_parameters(model, TrainingConfig(min_data_in_leaf=1), 29),
                             rounds=30, early_stopping=2, names=feature_names(model.encoder))
    assert stopped.best_iteration < 30
    text = (path / "model.txt").read_text()
    safe_json = (path / "inference.json").read_text()
    for forbidden in (str(path.parent), "00000000-0000-0000-0000-000000000001", "http://", "train_runs", "pilot_run", "game_seed", "ancestry"):
        assert forbidden not in text and forbidden not in safe_json
    assert "seed" in text  # legitimate native model RNG metadata
    from gf_common.domain import SIMULATOR_REVISION
    assert manifest["schema_version"] == 2 and manifest["simulator_contract_revision"] == SIMULATOR_REVISION
    assert manifest["purpose"] == "toy"
    with pytest.raises(ContractError, match="toy"):
        inspect_artifact(path)


@pytest.mark.parametrize("tamper", ["model", "digest", "runtime", "extra", "signature", "unready", "action", "config"])
def test_safe_artifact_rejects_tampering_or_compatibility_change(toy_artifact, tamper):
    path, model, _, _ = toy_artifact
    options = {"allow_toy": True}
    manifest = json.loads((path / "inference.json").read_text())
    if tamper == "model":
        (path / "model.txt").write_text((path / "model.txt").read_text() + "\n")
    elif tamper == "digest":
        options["expected_digest"] = "0" * 64
    elif tamper == "runtime":
        manifest["runtime"]["lightgbm"] = "unknown"
    elif tamper == "extra":
        manifest["game_seed"] = "forbidden"
    elif tamper == "signature":
        manifest["encoder_signature"] = "bad"
    elif tamper == "unready":
        manifest["ready"] = False
    elif tamper == "action":
        options["action_config_hash"] = "1" * 64
    else:
        options["model"] = replace(model, encoder=replace(model.encoder, history="public_aggregate"))
    (path / "inference.json").write_text(canonical(manifest))
    with pytest.raises(ContractError):
        load_artifact(path, **options)


def test_closed_native_parameters_reject_control_metadata(toy_data):
    model, training, x, y = toy_data
    params = native_parameters(model, training, 17)
    params["data"] = "/private/source-run/numeric-file"
    with pytest.raises(ContractError):
        fit_numeric(x, y, x, y, params=params, rounds=2, early_stopping=1, names=feature_names(model.encoder))


def test_policy_exact_scores_permutation_finite_shape_and_no_fake_value(toy_artifact, make_input):
    path, _, _, _ = toy_artifact
    policy = LightGBMPolicy(str(path), manifest_digest(str(path)), allow_toy=True)
    value = make_input(("bronze-club", "god-sword"))
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    scores = policy.scores(value, candidates)
    reversed_candidates = replace(candidates, actions=tuple(reversed(candidates.actions)))
    np.testing.assert_array_equal(policy.scores(value, reversed_candidates), scores[::-1])
    decision = policy.select(value, candidates)
    assert decision.index == int(np.argmax(scores))
    assert decision.value is None and decision.logprob is None
    assert decision.policy_version.endswith(manifest_digest(str(path)))
    class BadBooster:
        def predict(self, *args, **kwargs):
            return [float("nan")] * len(candidates.actions)
    policy.booster = BadBooster()
    with pytest.raises(ContractError, match="finite"):
        policy.select(value, candidates)


def test_worker_optional_binding_toy_direct_and_same_model_checkpoint_labels(toy_artifact, make_input, tmp_path, monkeypatch):
    path, model, _, _ = toy_artifact
    async def run():
        worker = ParticipantWorker(make_policy, "lightgbm", 0, ActionConfig(), 3,
                                   policy_options={"checkpoint": str(path), "expected_manifest_digest": manifest_digest(str(path)), "allow_toy": True})
        try:
            _, selected = await worker.select(make_input())
            assert selected.policy_version == "lightgbm-q-v1:" + manifest_digest(str(path))
        finally:
            await worker.close()
        wrong = ParticipantWorker(make_policy, "lightgbm", 0, ActionConfig(), 3,
                                  policy_options={"checkpoint": str(path), "expected_manifest_digest": "0" * 64, "allow_toy": True})
        with pytest.raises(RuntimeError, match="digest"):
            await wrong.select(make_input())
        assert wrong.closed
    asyncio.run(run())
    with pytest.raises(ContractError, match="toy"):
        bindings(replace(model, checkpoint=str(path)), ModelConfig(), ActionConfig())
    # Keep both actual artifacts toy. Stub only formal validation to isolate
    # registry identity without manufacturing a trained comparison checkpoint.
    manifest = json.loads((path / "inference.json").read_text())
    (path / "inference.json").write_text(canonical(manifest))
    second = tmp_path / "other"
    second.mkdir()
    (second / "model.txt").write_bytes((path / "model.txt").read_bytes())
    manifest["initialization"] = {"recipe": "seed-derived-v1"}
    # Different native bytes preserve native format while producing distinct identity.
    data = (second / "model.txt").read_bytes() + b"\n"
    (second / "model.txt").write_bytes(data)
    manifest["model_sha256"] = hashlib.sha256(data).hexdigest()
    (second / "inference.json").write_text(canonical(manifest))
    monkeypatch.setattr("gf_models.lightgbm.model.inspect_artifact", lambda *a, **k: manifest)
    monkeypatch.setattr("gf_cli.bindings.qualify_training_bundle", lambda *a, **k: {})
    registry, labels = bindings(replace(model, checkpoint=str(path)), replace(model, checkpoint=str(second)), ActionConfig())
    assert labels[0] != labels[1] and len(registry) == 2
    assert registry[labels[0]].policy_options["checkpoint"] == str(path)
    assert registry[labels[1]].policy_options["checkpoint"] == str(second)


def test_owned_toy_child_success_safely_promotes_and_keeps_control_provenance_separate(tmp_path, toy_data):
    model, training, x, y = toy_data
    directory = tmp_path / "native-child"
    control = {"source_path": "/secret-source-run", "server_url": "http://secret-control", "game_seed": "00000000-0000-0000-0000-000000009999"}
    manifest = asyncio.run(run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training, seed=17,
                                      provenance_data=control))
    assert manifest["status"] == "complete" and manifest["health"]["exit_code"] == 0
    assert manifest["fit_code_provenance"] == {"verification": "unknown", "revision": None, "dirty": None, "installed_build_revision": None}
    artifact = inspect_artifact(directory / "checkpoint", allow_toy=True)
    assert artifact["ready"] and artifact["purpose"] == "toy"
    text = (directory / "checkpoint" / "model.txt").read_text() + (directory / "checkpoint" / "inference.json").read_text()
    assert all(value not in text for value in control.values())
    assert json.loads((directory / "training.json").read_text())["source_provenance"] == control


@pytest.mark.parametrize("mode", ["timeout", "cancel", "log_limit", "failure"])
def test_owned_native_child_failures_reap_and_never_promote(tmp_path, toy_data, mode):
    model, training, x, y = toy_data
    directory = tmp_path / mode
    if mode == "log_limit":
        code = 'import sys,time;sys.stdout.write("x"*100000);sys.stdout.flush();time.sleep(10)'
    elif mode == "failure":
        code = 'raise RuntimeError("controlled fit failure")'
    else:
        code = 'import time;time.sleep(10)'
    training = replace(training, wall_seconds=1.5, cleanup_seconds=.5, max_log_bytes=1000)
    async def run():
        task = asyncio.create_task(run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training, seed=17,
                                               command=(sys.executable, "-c", code)))
        if mode == "cancel":
            for _ in range(100):
                if (directory / "stdout.log").exists():
                    break
                await asyncio.sleep(.01)
            task.cancel()
        with pytest.raises((TimeoutError, asyncio.CancelledError, RuntimeError)):
            await task
    asyncio.run(run())
    manifest = json.loads((directory / "training.json").read_text())
    assert manifest["status"] != "complete" and manifest["health"]["exit_code"] is not None
    assert not (directory / "checkpoint").exists()
    if mode == "log_limit":
        assert (directory / "stdout.log").stat().st_size <= 1000


def test_formal_commands_reject_toy_or_unready_before_listener(toy_artifact, monkeypatch, tmp_path):
    import gf_cli
    from gf_cli.config import load_config
    path, model, _, _ = toy_artifact
    loaded = load_config("configs/experiments/smoke.toml", {"model": {**asdict(model), "checkpoint": str(path)},
                          "run": {"output_dir": str(tmp_path / "output")}})
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", lambda *a, **k: pytest.fail("must not create listener"))
    with pytest.raises(ContractError, match="toy"):
        asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert not (tmp_path / "output").exists()
    manifest = json.loads((path / "inference.json").read_text())
    manifest["ready"] = False
    (path / "inference.json").write_text(canonical(manifest))
    with pytest.raises(ContractError, match="unready"):
        asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))


def test_two_checkpoint_same_model_environment_seat_binding_uses_exact_artifacts(toy_artifact, make_input, definitions, tmp_path):
    from gf_common.domain import EpisodeGroup, EpisodeSpec, RuntimePolicyBinding
    from gf_simulator_client.environment import EnvironmentSettings, SimulatorEnvironment
    from gf_training.journal import ParticipantJournal, read_journal
    from tests.episode_support import ROOT, SEED, PublicScript
    path, _, _, _ = toy_artifact
    other = tmp_path / "second-toy"
    other.mkdir()
    native = (path / "model.txt").read_bytes() + b"\n"
    (other / "model.txt").write_bytes(native)
    manifest = json.loads((path / "inference.json").read_text())
    manifest["model_sha256"] = hashlib.sha256(native).hexdigest()
    (other / "inference.json").write_text(canonical(manifest))
    labels = ["lightgbm-" + manifest_digest(str(checkpoint)) for checkpoint in (path, other)]
    registry = {label: RuntimePolicyBinding("lightgbm", {"checkpoint": str(checkpoint), "expected_manifest_digest": manifest_digest(str(checkpoint)), "allow_toy": True})
                for label, checkpoint in zip(labels, (path, other), strict=True)}
    async def run():
        script = PublicScript(actors=(2, 1))
        output = tmp_path / "episodes"
        settings = EnvironmentSettings("unused", str(ROOT / "configs/test-server.toml"), output, "toy-test-run", ActionConfig(), policy_bindings=registry)
        env = SimulatorEnvironment(settings, make_policy, ParticipantJournal)
        specs = (EpisodeSpec("seat0", tuple(labels), (17, 29), "pair"), EpisodeSpec("seat1", tuple(reversed(labels)), (29, 17), "pair"))
        client = script.client()
        await client.attest()
        try:
            for spec in specs:
                result = await env._episode(client, definitions, EpisodeGroup(SEED, (spec,)), spec)
                assert result.end_kind == "natural" and result.failure is None
                for viewer in (1, 2):
                    records = read_journal(output / spec.episode_id / f"player-{viewer}.jsonl")
                    decision = next(record for record in records if record["kind"] == "decision")
                    assert decision["selection"]["policy_version"] == "lightgbm-q-v1:" + spec.policies[viewer - 1].removeprefix("lightgbm-")
        finally:
            await client.close()
    asyncio.run(run())


def test_native_child_delayed_final_persistence_invalidates_promoted_artifact(tmp_path, toy_data, monkeypatch):
    import time

    import gf_training.lightgbm_run as module
    model, training, x, y = toy_data
    training = replace(training, wall_seconds=1.2, cleanup_seconds=.2)
    original = module.atomic_json
    delayed = False
    def atomic(path, value):
        nonlocal delayed
        original(path, value)
        if path.name == "training.json" and value.get("status") == "complete" and not delayed:
            delayed = True
            time.sleep(1.3)
    monkeypatch.setattr(module, "atomic_json", atomic)
    directory = tmp_path / "late"
    with pytest.raises((RuntimeError, TimeoutError)):
        asyncio.run(run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training, seed=17))
    manifest = json.loads((directory / "training.json").read_text())
    assert manifest["status"] == "failed" and manifest["quarantine"]
    if (directory / "checkpoint" / "inference.json").exists():
        assert json.loads((directory / "checkpoint" / "inference.json").read_text())["ready"] is False


def test_cancellation_during_native_process_creation_retains_ownership(tmp_path, toy_data, monkeypatch):
    import gf_training.lightgbm_run as module
    model, training, x, y = toy_data
    directory = tmp_path / "launch-cancel"
    original = module.asyncio.create_subprocess_exec
    entered = asyncio.Event()
    release = asyncio.Event()
    owned = []
    async def launch(*args, **kwargs):
        child = await original(*args, **kwargs)
        owned.append(child)
        entered.set()
        await release.wait()
        return child
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", launch)
    async def run():
        task = asyncio.create_task(run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training,
                                               seed=17, command=(sys.executable, "-c", "import time;time.sleep(10)")))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owned[0].returncode is not None
    asyncio.run(run())
    assert json.loads((directory / "training.json").read_text())["status"] == "interrupted"


def test_current_fit_code_provenance_stays_parent_only_and_is_budgeted(tmp_path, toy_data):
    from gf_training.lightgbm_run import _run_numeric
    model, training, x, y = toy_data
    code = {"revision": "current-fit-revision-sentinel", "dirty": True,
            "source_location": "/current-fit-control-path-sentinel"}
    directory = tmp_path / "fit-provenance"
    manifest = asyncio.run(_run_numeric(directory, x, y, x, y, model=model, actions=ActionConfig(), training=training,
                                       seed=17, purpose="toy", fit_code_provenance=code, fit_code_measurement_seconds=.03,
                                       charged_seconds=.03, provenance_data={"collection_code": "different-source-revision"}))
    assert manifest["fit_code_provenance"] == code
    assert manifest["fit_code_measurement_seconds"] == .03
    assert manifest["charged_prior_seconds"] >= .03
    text = (directory / "request.json").read_text() + (directory / "checkpoint" / "model.txt").read_text() + (directory / "checkpoint" / "inference.json").read_text()
    assert "current-fit-revision-sentinel" not in text and "/current-fit-control-path-sentinel" not in text


@pytest.mark.parametrize("damage", ["schema1", "missing_pin", "wrong_pin"])
def test_gameplay_identity_rejects_before_runtime_native_worker_or_listener(toy_artifact, tmp_path, monkeypatch, damage):
    import gf_cli
    import gf_models.lightgbm.model as adapter
    from gf_cli.config import load_config
    path, model, _, _ = toy_artifact
    manifest = json.loads((path / "inference.json").read_text())
    if damage == "schema1":
        manifest["schema_version"] = 1
    elif damage == "missing_pin":
        manifest.pop("simulator_contract_revision")
    else:
        manifest["simulator_contract_revision"] = "44a26e3a16b808e826deb332fa72acfc5037d3ef"
    (path / "inference.json").write_text(canonical(manifest))
    monkeypatch.setattr(adapter, "runtime_metadata", lambda: pytest.fail("invalid contract must reject before runtime inspection"))
    monkeypatch.setattr(adapter, "runtime", lambda: pytest.fail("invalid contract must reject before native import"))
    monkeypatch.setattr(gf_cli, "SimulatorEnvironment", lambda *a, **k: pytest.fail("invalid checkpoint must not create listener"))
    with pytest.raises(ContractError):
        load_artifact(path, allow_toy=True)
    loaded = load_config("configs/experiments/smoke.toml", {"model": {**asdict(model), "checkpoint": str(path)},
                         "run": {"output_dir": str(tmp_path / "output")}})
    with pytest.raises(ContractError):
        asyncio.run(gf_cli.run_experiment(loaded, "evaluate"))
    assert not (tmp_path / "output").exists()
    async def child():
        worker = ParticipantWorker(make_policy, "lightgbm", 17, ActionConfig(), 3,
                                   policy_options={"checkpoint": str(path), "expected_manifest_digest": manifest_digest(str(path)), "allow_toy": True})
        with pytest.raises(RuntimeError, match="ContractError"):
            await worker.initialize()
        assert worker.closed and worker.process.poll() is not None
    asyncio.run(child())
