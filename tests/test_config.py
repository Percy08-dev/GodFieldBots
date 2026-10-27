from pathlib import Path

import pytest

from gf_cli.config import load_config
from gf_common.domain import ContractError

ROOT = Path(__file__).parents[1]


def test_root_preset_precedence_hash_and_origin_paths(tmp_path):
    preset = tmp_path / "model.toml"
    preset.write_text('schema_version=1\nkind="model"\n[model]\nmodel_id="heuristic"\n')
    experiment = tmp_path / "nested" / "experiment.toml"
    experiment.parent.mkdir()
    experiment.write_text('schema_version=1\nkind="experiment"\n[references]\nmodel="../model.toml"\n[server]\nfixed_seed_experiment=true\nconfig="../server.toml"\n[run]\nconcurrency=4\n')
    result = load_config(experiment, {"actions": {"beam_width": 8}, "model": {"model_id": "random"}})
    assert result.config.model.model_id == "random"
    assert result.config.actions.beam_width == 8
    assert result.config.actions.max_candidates == 512
    assert result.config.run.concurrency == 4
    assert result.config.server.config == str(tmp_path / "server.toml")
    assert len(result.inputs) == 2
    assert load_config(experiment).sha256 != result.sha256
    assert load_config(experiment).sha256 == load_config(experiment).sha256
    preset.write_text(preset.read_text() + 'version="1"\n')
    assert result.inputs[0]["sha256"] != load_config(experiment).inputs[0]["sha256"]


@pytest.mark.parametrize("extra", [
    '\n[run]\nconcurrency=true\n', '\n[actions]\nmax_expansions=0\n',
    '\n[actions]\nbeam_width=nan\n', '\n[model]\nmodel_id="ppo"\n',
    '\n[run]\nconcurrency=1\nunknown=1\n', '\n[training]\nepochs=1\n',
    '\n[references]\ntraining="unknown.toml"\n', '\n[actions]\nscorer="hidden"\n',
])
def test_invalid_config_rejected_before_execution(tmp_path, extra):
    path = tmp_path / "bad.toml"
    path.write_text('schema_version=1\nkind="experiment"\n[server]\nfixed_seed_experiment=true\n' + extra)
    with pytest.raises(ContractError):
        load_config(path)


def test_unknown_version_refs_and_model_mismatch(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('schema_version=true\nkind="experiment"\n')
    with pytest.raises(ContractError):
        load_config(path)
    path.write_text('schema_version=1\nkind="experiment"\n[references]\nmodel="bad.toml"\n')
    with pytest.raises(ContractError, match="cycle"):
        load_config(path)


def test_configured_smoke_and_closed_cli_overrides():
    result = load_config(ROOT / "configs/experiments/smoke.toml")
    assert result.config.model.model_id == "heuristic"
    with pytest.raises(ContractError):
        load_config(ROOT / "configs/experiments/smoke.toml", {"server": {"made_up": 1}})
    with pytest.raises(ContractError):
        load_config(ROOT / "configs/experiments/smoke.toml", {"run": {"execution_seconds": 1000}})


def test_cleanup_and_report_reserves_must_fit_finalization():
    with pytest.raises(ContractError, match="reserves"):
        load_config(ROOT / "configs/experiments/smoke.toml", {"run": {"report_reserve_seconds": 59}, "server": {"cleanup_reserve_seconds": 2}})
