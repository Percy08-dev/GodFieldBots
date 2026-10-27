import json
import subprocess
import sys
from pathlib import Path

import pytest

import gf_cli

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/smoke.toml"


@pytest.mark.parametrize("command", ["smoke", "collect", "evaluate", "benchmark"])
@pytest.mark.parametrize("valid,exit_code", [(True, 0), (False, 1)])
def test_main_routes_effective_explicit_overrides_and_comparison_exit_code(monkeypatch, capsys, command, valid, exit_code):
    received = []
    async def run(loaded, selected):
        received.append((loaded, selected))
        return Path("/recorded-run"), {"planned_games": 2, "natural_denominator": 2 if valid else 1,
            "wins": 1, "draws": 0, "losses": 1 if valid else 0, "comparison_valid": valid,
            "failures": {} if valid else {"rejection": 1}, "wall_seconds": 0.5}
    monkeypatch.setattr(gf_cli, "run_experiment", run)
    code = gf_cli.main([command, "--config", str(CONFIG), "--set", "run.concurrency=3",
                        "--set", 'model.model_id="random"', "--set", 'run.name="from-cli"'])
    assert code == exit_code
    output = json.loads(capsys.readouterr().out)
    assert output["comparison_valid"] is valid and output["run_dir"] == "/recorded-run"
    loaded, selected = received[0]
    assert selected == command and loaded.config.run.concurrency == 3
    assert loaded.config.model.model_id == "random" and loaded.config.run.name == "from-cli"
    assert loaded.effective["run"]["concurrency"] == 3


def test_main_validate_emits_effective_hash_without_starting_games(monkeypatch, capsys):
    monkeypatch.setattr(gf_cli, "run_experiment", lambda *args: pytest.fail("validation must not run games"))
    assert gf_cli.main(["validate", "--config", str(CONFIG), "--set", "actions.max_candidates=7"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] and len(output["config_sha256"]) == 64
    assert output["effective_config"]["actions"]["max_candidates"] == 7
    assert output["inputs"]


@pytest.mark.parametrize("assignment", ["run", "run.extra.field=1", "run.concurrency=", "run.concurrency=true", "actions.unknown=1", 'model.model_id="future-ppo"'])
def test_main_invalid_set_or_config_never_runs_and_returns_structured_error(monkeypatch, capsys, assignment):
    monkeypatch.setattr(gf_cli, "run_experiment", lambda *args: pytest.fail("invalid config must stop before I/O"))
    assert gf_cli.main(["smoke", "--config", str(CONFIG), "--set", assignment]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"] in ("ValueError", "TOMLDecodeError", "ContractError")


def test_main_export_routes_paths_and_reports_missing_optional_library(monkeypatch, capsys):
    import gf_training.journal as journals
    received = []
    def export(sources, destination):
        received.append((sources, destination))
        return {"derived_schema_version": 2, "rows": 3}
    monkeypatch.setattr(journals, "export_parquet", export)
    assert gf_cli.main(["export", "one.jsonl", "two.jsonl", "--output", "derived.parquet"]) == 0
    assert received == [([Path("one.jsonl"), Path("two.jsonl")], Path("derived.parquet"))]
    assert json.loads(capsys.readouterr().out)["rows"] == 3
    def missing(*args):
        raise ImportError("optional data extra unavailable")
    monkeypatch.setattr(journals, "export_parquet", missing)
    assert gf_cli.main(["export", "one.jsonl", "--output", "derived.parquet"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ImportError"


def test_parser_usage_error_is_exit_two_without_runtime(monkeypatch):
    monkeypatch.setattr(gf_cli, "run_experiment", lambda *args: pytest.fail("parser must stop"))
    with pytest.raises(SystemExit) as captured:
        gf_cli.main(["smoke"])
    assert captured.value.code == 2


@pytest.mark.parametrize("command", [
    [sys.executable, "-m", "gf_cli", "--help"],
    [sys.executable, "-m", "godfieldbots", "--help"],
    [str(Path(sys.executable).with_name("gf-bots")), "--help"],
    [str(Path(sys.executable).with_name("godfieldbots")), "--help"],
])
def test_real_module_and_console_entrypoints_have_working_help(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0 and "usage:" in result.stdout
    assert "validate" in result.stdout and "export" in result.stdout
    assert not result.stderr


def test_help_imports_no_optional_learning_or_data_libraries():
    script = "import sys,gf_cli\ntry: gf_cli.main(['--help'])\nexcept SystemExit: pass\nprint('OPTIONALS', [k for k in ('torch','lightgbm','pyarrow') if k in sys.modules])\n"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0 and "OPTIONALS []" in result.stdout
