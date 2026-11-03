"""Original persisted collection facts must match the live execution plan."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import gf_cli.card_instance as cli
from gf_cli.config import load_config
from gf_common.domain import ContractError, EpisodeSpec


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
    monkeypatch.setattr(cli, "reconcile_replay", lambda *args: None)
    monkeypatch.setattr(cli, "replay_closure", lambda *args: {"type": "terminal", "result": "draw"})
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
