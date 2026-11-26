"""Pure runtime selection cannot authorize old process or live host execution."""

import asyncio
import subprocess
import sys
import tomllib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.config import load_config, load_rust_player_effective
from gf_cli.hpo_config import campaign_config, load_hpo_config, materialize_hpo_run
from gf_common.card_instance import (
    pestle_mirage_card_instance_contract,
    sacrifice_card_instance_contract,
)
from gf_common.domain import ContractError, canonical, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_training.hpo_config import effective_settings

ROOT = Path(__file__).parents[1]
BASE = ROOT / "configs/experiments/mlp-shared-inference-base.toml"
NORMAL_BASE = ROOT / "configs/experiments/mlp-shared-inference-scaling-base.toml"
RUN_ID = "00000000-0000-0000-0000-000000000017"
PPO_SOURCE = {"training_data": "rust-ppo-training-v1", "audit": "sealed-pack-sha256-v1"}


def runtime(**changes):
    return {
        "mode": "rust-player-runtime-v1",
        "scheduler": "tokio",
        "game_concurrency": 2,
        "player_worker_threads": 7,
        "max_batch_size": 3,
        "formation_wait_ms": 0,
    } | changes


def campaign_raw():
    raw = tomllib.loads((ROOT / "configs/hpo/mlp-shared-inference.toml").read_text())
    raw.update(schema_version=6, base_experiment=str(BASE), runtime=runtime())
    raw["operating"].update(game_concurrency=2)
    return raw


def test_ordinary_source_recipe_is_closed_and_round_trips(tmp_path):
    runtime_overrides = {"player_runtime": runtime()}
    legacy = load_config(NORMAL_BASE, {"run": runtime_overrides})
    loaded = load_config(NORMAL_BASE, {"run": runtime_overrides | {"ppo_source": PPO_SOURCE}})
    assert "ppo_source" not in legacy.effective["run"]
    assert loaded.effective["run"]["ppo_source"] == PPO_SOURCE
    assert loaded.sha256 != legacy.sha256
    path = tmp_path / "effective.json"
    path.write_text(canonical(loaded.effective))
    assert load_rust_player_effective(path, expected_sha256=loaded.sha256).effective == loaded.effective
    with pytest.raises(ContractError, match="source"):
        load_config(NORMAL_BASE, {"run": runtime_overrides | {"ppo_source": PPO_SOURCE | {"audit": "other"}}})
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {"run": {"ppo_source": PPO_SOURCE}})


def test_hpo_source_recipe_materializes_with_canonical_rust_encoder(tmp_path):
    raw = campaign_raw() | {"ppo_source": PPO_SOURCE, "encoder": "canonical-rust"}
    loaded = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert run.effective["run"]["ppo_source"] == PPO_SOURCE
    assert run.effective["hpo_spec"]["encoder"] == "canonical-rust"
    path = tmp_path / "effective.json"
    path.write_text(canonical(run.effective))
    assert load_rust_player_effective(path, expected_sha256=run.sha256).effective == run.effective
    with pytest.raises(ContractError, match="encoder"):
        campaign_config(raw | {"encoder": "canonical-python"}, origin=tmp_path)


def test_default_convergence_preset_uses_sacrifice_stage_and_keeps_archived_base():
    loaded = load_hpo_config(ROOT / "configs/hpo/mlp-256-convergence-1000.toml")
    archived = load_config(ROOT / "configs/experiments/hpo-rust-mlp-pestle-mirage-base.toml")
    release = sacrifice_card_instance_contract()
    assert loaded.config.stage == "mlp-256-convergence-1000-sacrifice"
    assert loaded.base.config.actions.action_contract == release
    assert loaded.base.config.server.simulator_revision == release.simulator_revision
    assert archived.config.actions.action_contract == pestle_mirage_card_instance_contract()


def test_normal_explicit_runtime_normalizes_wait_and_keeps_effective_hash():
    loaded = load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    assert loaded.effective["run"]["player_runtime"] == runtime(formation_wait_ms=0.0)
    assert type(loaded.config.run.player_runtime["formation_wait_ms"]) is float
    assert canonical(loaded.effective) == canonical(asdict(loaded.config))
    assert loaded.sha256 == digest(loaded.effective)
    assert (
        load_config(NORMAL_BASE, {"run": {"player_runtime": runtime(formation_wait_ms=0.0)}}).sha256
        == loaded.sha256
    )
    assert (
        load_config(NORMAL_BASE, {"run": {"player_runtime": runtime(scheduler="fixed")}}).sha256
        != loaded.sha256
    )


@pytest.mark.parametrize("field", list(runtime()))
def test_fixed_runtime_requires_every_field_explicitly(field):
    declaration = runtime(scheduler="fixed")
    del declaration[field]
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {"run": {"player_runtime": declaration}})


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "shared-mlp-v1"},
        {"scheduler": "other"},
        {"game_concurrency": 1},
        {"game_concurrency": True},
        {"player_worker_threads": False},
        {"player_worker_threads": 0},
        {"max_batch_size": 1.5},
        {"formation_wait_ms": True},
        {"formation_wait_ms": -1},
        {"formation_wait_ms": float("nan")},
        {"formation_wait_ms": float("inf")},
        {"producer": {}},
    ],
)
def test_normal_rejects_closed_malformed_or_mismatched_runtime(changes):
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {"run": {"player_runtime": runtime(**changes)}})


def test_normal_does_not_apply_legacy_process_concurrency_to_rust_threads():
    loaded = load_config(
        NORMAL_BASE,
        {
            "run": {
                "concurrency": 8,
                "max_participant_workers": 1,
                "player_runtime": runtime(game_concurrency=8, player_worker_threads=13),
            }
        },
    )
    assert loaded.config.run.concurrency == 8
    assert loaded.config.run.max_participant_workers == 1


def test_dual_selectors_and_incompatible_normal_models_are_refused():
    with pytest.raises(ContractError, match="mutually exclusive"):
        load_config(
            ROOT / "configs/experiments/mlp-shared-inference-smoke.toml",
            {"run": {"player_runtime": runtime()}},
        )
    for model in ("gru", "transformer"):
        with pytest.raises(ContractError, match="MLP"):
            load_config(
                NORMAL_BASE, {"run": {"player_runtime": runtime()}, "model": {"model_id": model}}
            )
    with pytest.raises(ContractError):
        load_config(
            ROOT / "configs/experiments/card-instance-mlp-ppo-acceptance.toml",
            {"run": {"player_runtime": runtime()}},
        )


def test_hpo6_composition_binds_runtime_and_does_not_call_legacy_admission(tmp_path, monkeypatch):
    import gf_cli.hpo_config as module

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy physical process admission reached")

    monkeypatch.setattr(module, "joint_resource_admission", forbidden)
    loaded = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert run.config.hpo_spec["schema_version"] == 6
    assert run.config.hpo_spec["runtime"] == runtime(formation_wait_ms=0.0)
    assert run.config.hpo_spec["origin"] == "producer-owned-v1"
    assert run.config.hpo_spec["encoder"] == "canonical-python"
    assert run.config.preprocessing.backend == "python"
    assert "player_runtime" not in run.effective["run"]
    assert "inference_service" not in run.effective["run"]
    assert canonical(run.effective) == canonical(asdict(run.config))
    assert run.sha256 == digest(run.effective)
    assert (
        materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID).sha256 == run.sha256
    )
    assert not list(tmp_path.iterdir())


def test_hpo6_study_identity_includes_runtime_science_curriculum_and_encoder(tmp_path):
    raw = campaign_raw()
    original = campaign_config(raw, origin=tmp_path)
    for key, value in runtime(
        scheduler="fixed", player_worker_threads=8, max_batch_size=4, formation_wait_ms=2
    ).items():
        if value == raw["runtime"][key]:
            continue
        changed = deepcopy(raw)
        changed["runtime"][key] = value
        assert (
            campaign_config(changed, origin=tmp_path).config.study_identity
            != original.config.study_identity
        )
    for section, changes in (
        ("operating", {"pool_every_decisions": 5000}),
        ("training", {"epochs": 3}),
    ):
        changed = deepcopy(raw)
        changed.setdefault(section, {}).update(changes)
        assert (
            campaign_config(changed, origin=tmp_path).config.study_identity
            != original.config.study_identity
        )
    changed = deepcopy(raw)
    changed["operating"]["game_concurrency"] = 3
    changed["runtime"]["game_concurrency"] = 3
    assert (
        campaign_config(changed, origin=tmp_path).config.study_identity
        != original.config.study_identity
    )
    changed = deepcopy(raw)
    changed["training_opponents"]["refresh"]["attack_threshold"] = 13
    assert (
        campaign_config(changed, origin=tmp_path).config.study_identity
        != original.config.study_identity
    )


@pytest.mark.parametrize(
    "change",
    [
        {"runtime": runtime(game_concurrency=1)},
        {"runtime": runtime(unknown=1)},
        {"runtime": {"mode": "rust-player-runtime-v1"}},
        {"model_id": "gru"},
        {"model_id": "transformer"},
        {"origin": "other"},
        {"encoder": "native"},
    ],
)
def test_hpo6_rejects_bad_declaration_before_base_io(tmp_path, change):
    raw = campaign_raw() | change
    raw["base_experiment"] = str(tmp_path / "does-not-exist.toml")
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)


def test_legacy_training_entries_refuse_new_selector_before_work(tmp_path):
    from gf_training.card_instance_ppo import (
        train_card_instance_ppo,
        train_formal_card_instance_ppo,
    )
    from gf_training.hpo_ppo import train_hpo_ppo

    normal = load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    for entry in (train_card_instance_ppo, train_formal_card_instance_ppo):
        with pytest.raises(ContractError, match="authenticated Rust-supervised"):
            asyncio.run(entry(normal, None, tmp_path / "unused"))
    campaign = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(
        campaign.config.model, campaign.config.training, {}, layout=LAYOUT
    )
    run = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID)
    with pytest.raises(ContractError, match="authenticated Rust-supervised"):
        asyncio.run(train_hpo_ppo(run, None, tmp_path / "unused", gpu_job=None))
    assert not list(tmp_path.iterdir())


def test_optional_free_config_and_old_cli_imports():
    script = """
import sys
import gf_cli
import gf_cli.config
import gf_cli.hpo_config
import gf_common.rust_player
assert not ({'torch', 'lightgbm', 'pyarrow', 'optuna', 'gf_native'} & sys.modules.keys())
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def normal_file(tmp_path, declaration):
    text = BASE.read_text()
    for directory in ("actions", "models", "preprocessing"):
        text = text.replace(f'"../{directory}/', f'"{ROOT}/configs/{directory}/')
    text += (
        "\n[storage]\n\n[run.player_runtime]\n"
        + "\n".join(
            f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
            for key, value in declaration.items()
        )
        + "\n"
    )
    path = tmp_path / "normal.toml"
    path.write_text(text)
    return path


def test_hpo6_matching_base_selector_is_removed_and_mismatch_rejected(tmp_path):
    base = normal_file(tmp_path, runtime())
    raw = campaign_raw() | {"base_experiment": str(base)}
    loaded = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert "player_runtime" not in run.effective["run"]
    assert run.config.hpo_spec["runtime"] == runtime(formation_wait_ms=0.0)
    with pytest.raises(ContractError, match="conflicts"):
        campaign_config(raw | {"runtime": runtime(scheduler="fixed")}, origin=tmp_path)
    old = tomllib.loads((ROOT / "configs/hpo/mlp-shared-inference.toml").read_text())
    old["base_experiment"] = str(base)
    with pytest.raises(ContractError, match="legacy HPO"):
        campaign_config(old, origin=tmp_path)


def test_hpo6_loader_bypasses_legacy_admission_and_retains_evaluation_capacity(
    tmp_path, monkeypatch
):
    import gf_cli.hpo_config as module

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy physical process admission reached")

    text = (ROOT / "configs/hpo/mlp-shared-inference.toml").read_text()
    text = text.replace("schema_version = 5", "schema_version = 6").replace(
        'base_experiment = "../experiments/mlp-shared-inference-base.toml"',
        f'base_experiment = "{BASE}"',
    )
    start, rest = text.split("[runtime]\n", 1)
    _, end = rest.split("[inference]", 1)
    declaration = runtime(game_concurrency=3)
    block = "\n".join(
        f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
        for key, value in declaration.items()
    )
    path = tmp_path / "campaign.toml"
    path.write_text(start + "[runtime]\n" + block + "\n[inference]" + end)
    monkeypatch.setattr(module, "joint_resource_admission", forbidden)
    loaded = module.load_hpo_config(path)
    assert loaded.effective["runtime"]["game_concurrency"] == 3
    assert loaded.config.runtime["player_worker_threads"] == 7
    assert loaded.inputs[-1]["kind"] == "ppo-hpo"
    raw = campaign_raw()
    raw["operating"].update(max_participant_workers=1)
    with pytest.raises(ContractError, match="evaluation capacity"):
        campaign_config(raw, origin=tmp_path)


def test_hpo6_cannot_issue_legacy_admission_or_execution_authority(tmp_path):
    from gf_cli.hpo_config import joint_resource_admission, require_hpo_runtime_execution

    loaded = campaign_config(campaign_raw(), origin=tmp_path)
    with pytest.raises(ContractError, match="authenticated HostControl"):
        joint_resource_admission(loaded.config, layout=LAYOUT)
    with pytest.raises(ContractError, match="authenticated Rust-supervised"):
        require_hpo_runtime_execution(loaded.config)


def test_hpo6_old_serving_selector_and_offline_native_source_are_refused(tmp_path):
    raw = campaign_raw()
    raw["base_experiment"] = str(ROOT / "configs/experiments/mlp-shared-inference-smoke.toml")
    with pytest.raises(ContractError, match="inference_service"):
        campaign_config(raw, origin=tmp_path)
    path = normal_file(tmp_path, runtime())
    path.write_text(path.read_text() + '\n[preprocessing]\nbackend="rust"\n')
    raw["base_experiment"] = str(path)
    with pytest.raises(ContractError):
        campaign_config(raw, origin=tmp_path)


def test_normal_requires_explicit_packed_storage():
    with pytest.raises(ContractError, match="packed storage"):
        load_config(BASE, {"run": {"player_runtime": runtime()}})


@pytest.mark.parametrize("command", ["train", "collect", "evaluate", "train-formal"])
def test_public_legacy_commands_reject_rust_selection_before_external_work(command):
    from gf_cli.config import require_storage_command

    loaded = load_config(NORMAL_BASE, {"run": {"player_runtime": runtime()}})
    require_storage_command(loaded.config, "validate")
    with pytest.raises(ContractError, match="authenticated Rust-supervised"):
        require_storage_command(loaded.config, command)


def test_normal_cpu_binding_and_hpo6_without_mixed_curriculum_remain_pure(tmp_path):
    normal = load_config(
        NORMAL_BASE,
        {
            "run": {"player_runtime": runtime()},
            "model": {"inference": {"device": "cpu"}},
            "training": {"device": "cpu"},
        },
    )
    assert normal.config.model.inference.device == normal.config.training.device == "cpu"
    raw = campaign_raw()
    del raw["training_opponents"]
    campaign = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(
        campaign.config.model, campaign.config.training, {}, layout=LAYOUT
    )
    run = materialize_hpo_run(campaign, settings, learning_seed=17, run_id=RUN_ID)
    assert run.config.hpo_spec["schema_version"] == 6
    assert "training_opponents" not in run.config.hpo_spec


@pytest.mark.parametrize("key", ["game_concurrency", "player_worker_threads", "max_batch_size"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "2"])
def test_runtime_atomic_counts_are_positive_integers(key, bad):
    from gf_common.rust_player import player_runtime

    with pytest.raises(ContractError):
        player_runtime(runtime(**{key: bad}), game_concurrency=2)


@pytest.mark.parametrize("kind", ["gru", "transformer", "lightgbm"])
def test_normal_rust_refuses_unsupported_opponents_and_legacy_config_still_accepts(kind):
    opponent = {"version": "3", "model_id": kind}
    old = load_config(NORMAL_BASE, {"opponent": opponent})
    assert old.config.opponent.model_id == kind
    with pytest.raises(ContractError, match="opponent"):
        load_config(NORMAL_BASE, {"opponent": opponent, "run": {"player_runtime": runtime()}})


@pytest.mark.parametrize("kind", ["gru", "transformer", "lightgbm"])
def test_hpo6_refuses_unsupported_base_opponent_even_for_fixed_curriculum(tmp_path, kind):
    path = normal_file(tmp_path, runtime())
    text = path.read_text().split("[run.player_runtime]")[0]
    path.write_text(text + f'\n[opponent]\nversion="3"\nmodel_id="{kind}"\n')
    old = load_config(path)
    assert old.config.opponent.model_id == kind
    with pytest.raises(ContractError, match="opponent"):
        campaign_config(campaign_raw() | {"base_experiment": str(path)}, origin=tmp_path)


@pytest.mark.parametrize(
    "kind,version", [("mlp", "3"), ("random", "1"), ("heuristic", "1"), ("heuristic_refresh", "1")]
)
def test_rust_supported_opponent_composition_remains_explicit(kind, version):
    loaded = load_config(
        NORMAL_BASE,
        {"opponent": {"model_id": kind, "version": version}, "run": {"player_runtime": runtime()}},
    )
    assert loaded.config.opponent.model_id == kind


@pytest.mark.parametrize("base_batched,campaign_batched", [(False, True), (True, False), (True, True)])
def test_hpo6_base_agrees_on_schedule_and_campaign_owns_inference(tmp_path, base_batched, campaign_batched):
    recipe = {"inference_recipe": "batched-torch-v1"}
    base = normal_file(tmp_path, runtime() | (recipe if base_batched else {}))
    declaration = runtime() | (recipe if campaign_batched else {})
    raw = campaign_raw() | {"base_experiment": str(base), "runtime": declaration}
    loaded = campaign_config(raw, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert run.config.hpo_spec["runtime"] == declaration
    assert "player_runtime" not in run.effective["run"]
    with pytest.raises(ContractError, match="conflicts"):
        campaign_config(raw | {"runtime": declaration | {"max_batch_size": 99}}, origin=tmp_path)


def test_tokio_graph_runtime_defaults_fresh_and_requires_materialized_worker_count(tmp_path):
    from gf_common.rust_player import player_runtime
    from gf_training.hpo_lineage import rust_player_hpo_spec

    graph = runtime(inference_recipe="batched-torch-graph-v1")
    declaration = {key: value for key, value in graph.items() if key != "player_worker_threads"}
    ordinary = load_config(NORMAL_BASE, {"run": {"player_runtime": declaration}})
    assert ordinary.effective["run"]["player_runtime"]["player_worker_threads"] == 16
    ordinary_incomplete = deepcopy(ordinary.effective)
    del ordinary_incomplete["run"]["player_runtime"]["player_worker_threads"]
    archived = tmp_path / "effective.json"
    archived.write_text(canonical(ordinary_incomplete))
    with pytest.raises(ContractError, match="normalized effective"):
        load_rust_player_effective(archived, expected_sha256=digest(ordinary_incomplete))
    loaded = campaign_config(campaign_raw() | {"runtime": declaration}, origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert rust_player_hpo_spec(run.effective)["runtime"]["player_worker_threads"] == 16
    malformed = deepcopy(run.effective)
    del malformed["hpo_spec"]["runtime"]["player_worker_threads"]
    with pytest.raises(ContractError, match="normalized.*runtime"):
        rust_player_hpo_spec(malformed)
    assert player_runtime(graph, game_concurrency=2)["player_worker_threads"] == 7
    fixed = declaration | {"scheduler": "fixed"}
    with pytest.raises(ContractError):
        player_runtime(fixed, game_concurrency=2)
    assert player_runtime(fixed | {"player_worker_threads": 3}, game_concurrency=2)["player_worker_threads"] == 3


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_hpo6_sequence_recipe_has_matching_base_and_canonical_python_source(tmp_path, kind):
    base = load_config(BASE, {"model": {"model_id": kind}})
    raw = campaign_raw() | {"model_id": kind, "runtime": runtime(inference_recipe="batched-torch-sequence-v1")}
    loaded = campaign_config(raw, origin=tmp_path, _pinned_base=base)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    assert run.config.model.model_id == kind
    assert run.config.hpo_spec["encoder"] == "canonical-python"
    path = tmp_path / "effective.json"
    path.write_text(canonical(run.effective))
    restored = load_rust_player_effective(path, expected_sha256=run.sha256)
    assert restored.effective == run.effective
    for invalid in (raw | {"runtime": runtime()},
                    raw | {"encoder": "canonical-rust", "ppo_source": PPO_SOURCE}):
        with pytest.raises(ContractError):
            campaign_config(invalid, origin=tmp_path, _pinned_base=base)
    oversized = raw | {"operating": raw["operating"] | {"numeric_budget_bytes": 1024**3}}
    with pytest.raises(ContractError, match="canonical-python numeric budget"):
        campaign_config(oversized, origin=tmp_path, _pinned_base=base)
    with pytest.raises(ContractError):
        load_config(NORMAL_BASE, {"run": {"player_runtime": raw["runtime"]}})


@pytest.mark.parametrize(
    "hpo,source,model,profile,games,opponent,expected",
    [
        (True, True, "mlp", "current", 256, "heuristic_refresh", True),
        (False, True, "mlp", "current", 256, "heuristic", False),
        (True, False, "mlp", "current", 256, "heuristic", False),
        (True, True, "gru", "current", 256, "heuristic", False),
        (True, True, "mlp", "historical", 256, "heuristic", False),
        (True, True, "mlp", "current", 257, "heuristic", False),
        (True, True, "mlp", "current", 256, "transformer", False),
    ],
)
def test_native_journal_is_default_only_for_supported_hpo_collection(
    hpo, source, model, profile, games, opponent, expected
):
    from types import SimpleNamespace

    from gf_common.card_instance import (
        sacrifice_card_instance_contract,
        shared_card_instance_contract,
    )
    from gf_training.rust_player_collection import RustPlayerCollector

    contract = (
        sacrifice_card_instance_contract() if profile == "current" else shared_card_instance_contract()
    )
    collector = SimpleNamespace(
        registry=SimpleNamespace(hpo=hpo, state={"model_id": model}),
        mode={} if source else None,
        config=SimpleNamespace(actions=SimpleNamespace(action_contract=contract)),
        declaration={"game_concurrency": 256},
    )
    plan = SimpleNamespace(
        groups=(SimpleNamespace(episodes=range(games)),),
        bindings={"learner": SimpleNamespace(model_id=model), "opponent": SimpleNamespace(model_id=opponent)},
    )
    assert RustPlayerCollector._uses_native_journal(collector, plan) is expected
