"""Scheduled checkpoint evaluation opt-in and historical config identity."""

import tomllib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from gf_cli.config import LoadedConfig, load_rust_player_effective
from gf_cli.hpo_config import LoadedHPOConfig, campaign_config, materialize_hpo_run
from gf_common.domain import ContractError, Json, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_training.hpo_config import effective_settings
from tests.test_hpo_config import campaign_raw as legacy_campaign
from tests.test_rust_player_config import PPO_SOURCE, ROOT, RUN_ID
from tests.test_rust_player_config import campaign_raw as rust_campaign

FINAL_TEST_SEED = "20000000-0000-0000-0000-000000000001"


def _campaign_file(stem: str) -> dict[str, Any]:
    raw = tomllib.loads((ROOT / "configs/hpo" / f"{stem}.toml").read_text())
    raw["base_experiment"] = str(ROOT / "configs/hpo" / raw["base_experiment"])
    return raw


def _schedule(raw: dict[str, Any], options: object, *, cap: int = 2) -> dict[str, Any]:
    return raw | {"operating": raw.get("operating", {}) | {
        "max_updates": cap, "checkpoint_evaluation": options}}


def _run(loaded: LoadedHPOConfig) -> LoadedConfig:
    cfg = loaded.config
    settings = effective_settings(cfg.model, cfg.training, {}, layout=LAYOUT)
    return materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)


@pytest.mark.parametrize("factory", [legacy_campaign, rust_campaign])
def test_explicit_opt_out_has_no_serialized_or_study_identity_footprint(
    factory: Callable[[], Json], tmp_path: Path,
) -> None:
    raw = factory()
    cap = raw.get("operating", {}).get("max_updates", 1000)
    absent = campaign_config(raw, origin=tmp_path)
    disabled = campaign_config(_schedule(raw, {"enabled": False}, cap=cap), origin=tmp_path)

    assert disabled.effective == absent.effective
    assert disabled.sha256 == absent.sha256
    assert disabled.config.study_identity == absent.config.study_identity
    assert asdict(disabled.config.operating) == asdict(absent.config.operating)
    assert "checkpoint_evaluation" not in disabled.effective["operating"]
    assert _run(disabled).effective == _run(absent).effective
    assert _run(disabled).sha256 == _run(absent).sha256


@pytest.mark.parametrize("stem", ["owned-mlp-smoke", "owned-gru-smoke", "owned-transformer-smoke",
    "mlp-shared-inference", None])
def test_supported_campaigns_clip_default_schedule(stem: str | None, tmp_path: Path) -> None:
    from gf_training.hpo_lineage import hpo_spec

    raw = rust_campaign() if stem is None else _campaign_file(stem)
    loaded = campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100}, cap=250), origin=tmp_path)

    assert loaded.effective["operating"]["checkpoint_evaluation"] == {
        "enabled": True, "wall_seconds": 100.0, "updates": [0, 100, 250], "retain_updates": [],
        "final_test_game_seeds": [],
    }
    assert canonical(asdict(loaded.config)) == canonical(loaded.effective)
    run = _run(loaded)
    spec = run.effective["hpo_spec"]
    assert spec["operating"]["checkpoint_evaluation"] == (
        loaded.effective["operating"]["checkpoint_evaluation"]
        | {"evaluation_game_seeds_sha256": digest(raw["evaluation_game_seeds"])}
    )
    if stem is not None:
        variant = "hpo-ppo-owned-shared-v1" if spec["schema_version"] == 5 else "hpo-ppo-owned-v1"
        assert hpo_spec({"effective_config": run.effective, "run_variant": variant,
            "hpo_spec_sha256": digest(spec)}) == spec


@pytest.mark.parametrize("stem", [None, "heuristic-refresh-mlp-smoke", "resident-mlp-smoke"])
def test_enabled_options_fail_on_historical_campaign_schemas(stem: str | None, tmp_path: Path) -> None:
    raw = legacy_campaign() if stem is None else _campaign_file(stem)
    with pytest.raises(ContractError, match="checkpoint_evaluation.*schema"):
        campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100}), origin=tmp_path)


@pytest.mark.parametrize(("options", "message"), [
    ([], "checkpoint_evaluation"),
    ({}, "checkpoint_evaluation"),
    ({"enabled": "false"}, "enabled"),
    ({"enabled": False, "unknown": 1}, "checkpoint_evaluation"),
    ({"enabled": False, "updates": [0]}, "checkpoint_evaluation"),
    ({"enabled": True}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": True}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": float("inf")}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": 0}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": 60}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": 7201}, "wall_seconds"),
    ({"enabled": True, "wall_seconds": 100, "updates": []}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "updates": [False]}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "updates": [-1]}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "updates": [3]}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "updates": [2, 1]}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "updates": [1, 1]}, "updates"),
    ({"enabled": True, "wall_seconds": 100, "retain_updates": [2, 1]}, "retain_updates"),
    ({"enabled": True, "wall_seconds": 100, "retain_updates": [4]}, "retain_updates"),
    ({"enabled": True, "wall_seconds": 100, "final_test_game_seeds": ["unexpected"]}, "GameSeed"),
    ({"enabled": True, "wall_seconds": 100, "final_test_game_seeds": [FINAL_TEST_SEED, FINAL_TEST_SEED]},
     "duplicate"),
    ({"enabled": True, "wall_seconds": 100,
      "evaluation_game_seeds_sha256": "0" * 64}, "checkpoint_evaluation"),
])
def test_checkpoint_options_reject_invalid_fields_and_values(options: object, message: str, tmp_path: Path) -> None:
    with pytest.raises(ContractError, match=message):
        campaign_config(_schedule(rust_campaign(), options), origin=tmp_path)


def test_checkpoint_validation_panel_must_be_disjoint_from_final_test(tmp_path: Path) -> None:
    raw = rust_campaign()
    with pytest.raises(ContractError, match="disjoint"):
        campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100,
            "final_test_game_seeds": [raw["evaluation_game_seeds"][0]]}), origin=tmp_path)


def test_enabled_rust_seed_wall_must_fit_before_session_soft_finish(tmp_path: Path) -> None:
    raw = rust_campaign()
    raw["session_seconds"] = raw["run_wall_seconds"]
    campaign_config(raw, origin=tmp_path)
    with pytest.raises(ContractError, match="session_seconds.*run_wall_seconds"):
        campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100}), origin=tmp_path)


def test_rust_campaign_snapshot_preserves_checkpoint_seed_split_and_timed_source(tmp_path: Path) -> None:
    from gf_cli.rust_player_campaign import campaign_snapshot, load_campaign_snapshot

    raw = rust_campaign() | {"encoder": "canonical-rust", "ppo_source": PPO_SOURCE}
    raw["operating"]["post_host_seconds"] = 45
    options = {"enabled": True, "wall_seconds": 100, "updates": [0, 2], "retain_updates": [1],
        "final_test_game_seeds": [FINAL_TEST_SEED]}
    active = campaign_config(_schedule(raw, options), origin=tmp_path)
    without_holdout = campaign_config(_schedule(raw, options | {"final_test_game_seeds": []}), origin=tmp_path)
    assert active.config.study_identity != without_holdout.config.study_identity
    assert active.sha256 != without_holdout.sha256
    assert _run(active).sha256 != _run(without_holdout).sha256

    restored = load_campaign_snapshot(campaign_snapshot(active))
    assert restored.config.checkpoint_evaluation == active.config.checkpoint_evaluation
    assert restored.config.study_identity == active.config.study_identity
    assert restored.sha256 == active.sha256
    run = _run(restored)
    assert run.effective == _run(active).effective
    assert run.effective["run"]["post_host_seconds"] == 45
    assert run.effective["run"]["ppo_source"] == PPO_SOURCE
    assert run.effective["hpo_spec"]["operating"]["checkpoint_evaluation"] == options | {
        "evaluation_game_seeds_sha256": digest(raw["evaluation_game_seeds"]), "wall_seconds": 100.0,
    }
    path = tmp_path / "run-config.json"
    path.write_text(canonical(run.effective))
    reloaded = load_rust_player_effective(path, expected_sha256=run.sha256)
    assert reloaded.effective == run.effective
    assert reloaded.sha256 == run.sha256

    checkpoint = run.effective["hpo_spec"]["operating"]["checkpoint_evaluation"]
    checkpoint["evaluation_game_seeds_sha256"] = "unbound"
    path.write_text(canonical(run.effective))
    with pytest.raises(ContractError, match="evaluation_game_seeds_sha256"):
        load_rust_player_effective(path, expected_sha256=digest(run.effective))


def test_persisted_checkpoint_config_requires_the_active_canonical_block(tmp_path: Path) -> None:
    from gf_common.hpo_checkpoint_evaluation import persisted_checkpoint_evaluation

    raw = rust_campaign()
    loaded = campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100}), origin=tmp_path)
    normalized = _run(loaded).effective["hpo_spec"]["operating"]["checkpoint_evaluation"]
    context: dict[str, Any] = {
        "schema_version": 6, "max_updates": 2, "run_wall_seconds": 7200.0, "finalization_seconds": 60.0}
    assert persisted_checkpoint_evaluation(normalized, **context) == loaded.config.checkpoint_evaluation
    for invalid in (
        {"enabled": False}, normalized | {"wall_seconds": 100},
        normalized | {"evaluation_game_seeds_sha256": "unknown"}, normalized | {"updates": [0, 9]},
        normalized | {"extra": None}, {key: value for key, value in normalized.items() if key != "retain_updates"},
    ):
        with pytest.raises(ContractError):
            persisted_checkpoint_evaluation(invalid, **context)


def test_scheduled_and_explicit_retention_preserve_only_inference_for_old_generations(tmp_path: Path) -> None:
    from gf_training.hpo_lineage import protected_hpo_files_for_operating, rust_player_hpo_spec
    from gf_training.rust_player_retention import protected_files

    raw = rust_campaign()
    raw["operating"].update(inference_latest=2, inference_periodic_decisions=1000)
    active = campaign_config(_schedule(raw, {"enabled": True, "wall_seconds": 100, "updates": [0, 1, 10],
        "retain_updates": [3]}, cap=10), origin=tmp_path)
    operating = rust_player_hpo_spec(_run(active).effective)["operating"]
    inference = {"inference.json", "weights-000.npz", "weights-001.npz"}
    complete = inference | {"resume.json", "state-000.npz"}
    records = [{"generation": g, "used_decisions": g * 100, "parent_controls": {"snapshot_pool": [0]},
        "artifact_files": dict.fromkeys(complete, "a" * 64)} for g in range(8)]
    expected = {0: complete, 1: inference, 2: set(), 3: inference, 4: set(), 5: set(), 6: complete, 7: complete}
    assert protected_hpo_files_for_operating(operating, records, {"evaluation_references": {}}) == expected
    assert protected_files(dict(enumerate(records)), {"service_references": {}, "evaluation_references": {}},
        generation=7, operating=operating) == expected


@pytest.mark.parametrize("case", ["scheduled", "opt_out", "later_schedule", "collected", "used", "stepped",
    "uncommitted"])
def test_owned_paused_initial_generation_requires_schedule_commit_and_zero_counts(case: str, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from gf_training.hpo_lineage import hpo_state
    from tests.hpo_lineage_support import hpo_state_fixture

    state, *_ = hpo_state_fixture(tmp_path / "parent", resident=True)
    config = state["effective_config"]
    spec = config["hpo_spec"]
    spec.update(schema_version=4, origin="producer-owned-v1", encoder="canonical-python")
    if case != "opt_out":
        spec["operating"]["checkpoint_evaluation"] = {
            "enabled": True, "wall_seconds": 100.0, "updates": [1] if case == "later_schedule" else [0],
            "retain_updates": [], "final_test_game_seeds": [],
            "evaluation_game_seeds_sha256": digest([FINAL_TEST_SEED]),
        }
    state.update(run_variant="hpo-ppo-owned-v1", status="paused", stop_reason="pause", artifact_bytes=0,
        elapsed_seconds=0.0, generation_sha256=None if case == "uncommitted" else "a" * 64,
        invocations=[{"mode": "fresh", "elapsed_seconds": 0.0, "child_health": {"exit_code": 0, "stderr_bytes": 0}}],
        hpo_spec_sha256=digest(spec), config_sha256=digest(config))
    state["resume_identity"] = digest({"version": "hpo-ppo-resume-v1", "config_sha256": state["config_sha256"],
        "identity": state["identity"], "execution": state["execution"], "storage_binding": None,
        "classification": {key: state[key] for key in (
            "action_contract", "purpose", "synthetic_capacity", "rule_signature", "layout_signature")}})
    if case in ("collected", "used"):
        state["collected_decisions"] = 1
    if case == "used":
        state.update(used_decisions=1, accepted_decisions=1)
    if case == "stepped":
        state["optimizer_steps"] = 1

    if case == "scheduled":
        assert hpo_state(state)[2].max_updates == 4
    else:
        with pytest.raises(ContractError):
            hpo_state(state)
