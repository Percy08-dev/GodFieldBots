"""Original persisted collection facts must match the live execution plan."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import gf_cli.card_instance as cli
from gf_cli.config import load_config
from gf_common.domain import ContractError, EpisodeSpec, RuntimePolicyBinding


def stub_draw_replay(monkeypatch, expected_contract):
    def reconcile(value, viewer, accepted, outcome, *, action_contract):
        assert action_contract == expected_contract
        assert value == {"viewer": viewer} and accepted == [] and outcome == "draw"

    def closure(value, game, viewer, *, action_contract):
        assert action_contract == expected_contract
        assert value == {"viewer": viewer} and game == "game"
        return {"type": "terminal", "result": "draw"}

    monkeypatch.setattr(cli, "reconcile_replay", reconcile)
    monkeypatch.setattr(cli, "replay_closure", closure)


@pytest.mark.parametrize("mutation", [None, "runtime", "sampling_seed", "policy"])
def test_persisted_journal_must_match_owned_runtime_and_planned_viewer_even_without_decisions(
    tmp_path, monkeypatch, mutation
):
    loaded = load_config(Path(__file__).parents[1] / "configs/experiments/card-instance-v2-corrected-contract.toml")
    episode = SimpleNamespace(episode_id="episode", game_id="game", policies=("heuristic-card-instance-v2", "random-card-instance-v2"),
                              end_kind="natural", outcome="draw")
    spec = EpisodeSpec("episode", episode.policies, (17, 29), "pair")
    runtime = {"receipt": "actual-owned-live-process"}
    folder = tmp_path / "episode"
    folder.mkdir()
    for viewer in (1, 2):
        (folder / f"replay-player-{viewer}.json").write_text(json.dumps({"viewer": viewer}))

    def read(path, **kwargs):
        viewer = int(path.stem[-1])
        header = {"run_id": "run", "game_id": "game", "episode_id": "episode", "viewer": viewer,
                  "policy": episode.policies[viewer - 1], "sampling_seed": spec.sampling_seeds[viewer - 1],
                  "runtime_evidence": deepcopy(runtime)}
        if viewer == 1:
            if mutation == "runtime":
                header["runtime_evidence"] = {"receipt": "different-process"}
            elif mutation == "sampling_seed":
                header["sampling_seed"] = 43
            elif mutation == "policy":
                header["policy"] = "random-card-instance-v2"
        return SimpleNamespace(header=header, end={"end_kind": "natural", "outcome": "draw"}, decisions=(), accepted=())

    monkeypatch.setattr(cli, "read_card_instance_journal", read)
    stub_draw_replay(monkeypatch, loaded.config.actions.action_contract)
    if mutation is None:
        cli._validate_recorded_episodes(tmp_path, [episode], loaded.config.actions, "run", float("inf"),
                                        {"episode": (spec, runtime)})
    else:
        with pytest.raises(ContractError, match="identity mismatch"):
            cli._validate_recorded_episodes(tmp_path, [episode], loaded.config.actions, "run", float("inf"),
                                            {"episode": (spec, runtime)})


def test_neural_config_is_rejected_before_runtime_io(tmp_path, monkeypatch):
    import asyncio

    loaded = load_config(Path(__file__).parents[1] / "configs/experiments/card-instance-mlp-contract.toml",
                         {"run": {"output_dir": str(tmp_path / "unused")}})
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: pytest.fail("runtime I/O reached"))
    with pytest.raises(ContractError, match="Random/Heuristic only"):
        asyncio.run(cli.run_baseline(loaded, "collect"))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("name", ["mlp", "gru", "transformer"])
def test_neural_inference_requires_checkpoint_before_runtime_io(tmp_path, monkeypatch, name):
    import asyncio

    loaded = load_config(Path(__file__).parents[1] / f"configs/experiments/card-instance-{name}-contract.toml",
                         {"run": {"output_dir": str(tmp_path / "unused")}})
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: pytest.fail("runtime I/O reached"))
    with pytest.raises(ContractError, match="explicit game checkpoint"):
        asyncio.run(cli.run_inference(loaded, "collect"))
    assert not (tmp_path / "unused").exists()


def test_joint_neural_admission_precedes_output_and_environment(tmp_path, monkeypatch):
    import asyncio

    import gf_models.card_instance_artifacts as artifacts
    import gf_models.structured_runtime as runtime

    loaded = load_config(Path(__file__).parents[1] / "configs/experiments/card-instance-mlp-contract.toml",
                         {"model": {"checkpoint": str(tmp_path / "inference")},
                          "run": {"output_dir": str(tmp_path / "unused")}})
    released = []
    owner = SimpleNamespace(binding=lambda model: ("neural", RuntimePolicyBinding("mlp", {})),
                            release=lambda: released.append(True))
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: SimpleNamespace(preflight=lambda: None))
    monkeypatch.setattr(artifacts, "RetainedGameArtifacts", lambda *args, **kwargs: owner)
    monkeypatch.setattr(cli, "CardInstanceSimulatorEnvironment", lambda *args, **kwargs: pytest.fail("environment allocated"))

    def refuse(actions, roles, **kwargs):
        assert [role.model_id for role in roles] == ["mlp", "random"]
        assert kwargs["game_concurrency"] == 2 and kwargs["max_workers"] == 4
        raise ContractError("combined envelope exceeded")

    monkeypatch.setattr(runtime, "admit_game_workers_v3", refuse)
    with pytest.raises(ContractError, match="combined envelope"):
        asyncio.run(cli.run_inference(loaded, "collect"))
    assert released == [True]
    assert not (tmp_path / "unused").exists()


def test_neural_constructor_failure_keeps_diagnostics_and_releases_inputs(tmp_path, monkeypatch):
    import asyncio

    import gf_models.card_instance_artifacts as artifacts
    import gf_models.structured_runtime as runtime

    loaded = load_config(Path(__file__).parents[1] / "configs/experiments/card-instance-mlp-contract.toml",
                         {"model": {"checkpoint": str(tmp_path / "inference")},
                          "run": {"output_dir": str(tmp_path / "runs")}})
    released = []
    owner = SimpleNamespace(binding=lambda model: ("neural", RuntimePolicyBinding("mlp", {})),
                            release=lambda: released.append(True))
    monkeypatch.setattr(cli, "CardInstanceRuntimeEvidence", lambda *args, **kwargs: SimpleNamespace(preflight=lambda: None))
    monkeypatch.setattr(artifacts, "RetainedGameArtifacts", lambda *args, **kwargs: owner)
    monkeypatch.setattr(runtime, "admit_game_workers_v3", lambda *args, **kwargs: {"total_peak_bytes": 0})

    def fail(*args, **kwargs):
        raise ContractError("late external input failure")

    monkeypatch.setattr(cli, "CardInstanceSimulatorEnvironment", fail)
    with pytest.raises(ContractError, match="late external input"):
        asyncio.run(cli.run_inference(loaded, "collect"))
    assert released == [True]
    diagnostics = list((tmp_path / "runs").glob("*/failure.json"))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["message"] == "late external input failure"
    assert not list((tmp_path / "runs").glob("*/manifest.json"))
    assert not list((tmp_path / "runs").glob("*/report.json"))


@pytest.mark.parametrize("changed", [False, True])
def test_neural_journal_matches_retained_input_metadata(tmp_path, monkeypatch, changed):
    loaded = load_config(Path(__file__).parents[1] / "configs/experiments/card-instance-v2-corrected-contract.toml")
    labels = ("mlp-ppo-v3-owned-artifact", "random-card-instance-v2")
    episode = SimpleNamespace(episode_id="episode", game_id="game", policies=labels,
                              end_kind="natural", outcome="draw")
    spec = EpisodeSpec("episode", labels, (17, 29), "pair")
    runtime = {"receipt": "owned-live-runtime"}
    retained = {"input_contract": "retained-external-inference-v1", "inference_manifest": {"digest": "owned"}}
    folder = tmp_path / "episode"
    folder.mkdir()
    for viewer in (1, 2):
        (folder / f"replay-player-{viewer}.json").write_text(json.dumps({"viewer": viewer}))

    def read(path, **kwargs):
        viewer = int(path.stem[-1])
        header = {"run_id": "run", "game_id": "game", "episode_id": "episode", "viewer": viewer,
                  "policy": labels[viewer - 1], "sampling_seed": spec.sampling_seeds[viewer - 1],
                  "runtime_evidence": deepcopy(runtime)}
        if viewer == 1:
            header["neural"] = deepcopy(retained)
            if changed:
                header["neural"]["inference_manifest"]["digest"] = "foreign"
        return SimpleNamespace(header=header, end={"end_kind": "natural", "outcome": "draw"}, decisions=(), accepted=())

    monkeypatch.setattr(cli, "read_card_instance_journal", read)
    stub_draw_replay(monkeypatch, loaded.config.actions.action_contract)
    args = (tmp_path, [episode], loaded.config.actions, "run", float("inf"), {"episode": (spec, runtime)})
    if changed:
        with pytest.raises(ContractError, match="retained artifact"):
            cli._validate_recorded_episodes(*args, neural_metadata={labels[0]: retained})
    else:
        cli._validate_recorded_episodes(*args, neural_metadata={labels[0]: retained})
