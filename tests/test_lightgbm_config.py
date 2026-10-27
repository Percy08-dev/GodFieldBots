
import pytest

from gf_cli import main
from gf_cli.config import load_config
from gf_common.domain import ContractError


def write(path, text):
    path.write_text('schema_version=1\n' + text)
    return path


def test_model_training_precedence_origin_paths_and_opponent_exclusion(tmp_path):
    write(tmp_path / "common.toml", 'kind="training"\n[training]\nlearning_rate=0.1\ntrain_runs=["train"]\nvalidation_runs=["validation"]\npilot_run="pilot"\n')
    write(tmp_path / "primary.toml", 'kind="model"\n[model]\nmodel_id="lightgbm"\ncheckpoint="ready"\n[model.encoder]\nhistory="public_aggregate"\n[training]\nlearning_rate=0.2\nnum_boost_round=80\n')
    write(tmp_path / "opponent.toml", 'kind="model"\n[model]\nmodel_id="lightgbm"\ncheckpoint="other"\n[training]\nlearning_rate=0.9\n')
    path = write(tmp_path / "experiment.toml", 'kind="experiment"\n[references]\nopponent="opponent.toml"\nmodel="primary.toml"\ntraining="common.toml"\n[server]\nfixed_seed_experiment=true\n[training]\nnum_boost_round=90\n')
    loaded = load_config(path, {"training": {"early_stopping_rounds": 7}})
    assert loaded.config.training.learning_rate == .2
    assert loaded.config.training.num_boost_round == 90
    assert loaded.config.training.early_stopping_rounds == 7
    assert loaded.config.training.train_runs == (str(tmp_path / "train"),)
    assert loaded.config.training.pilot_run == str(tmp_path / "pilot")
    assert loaded.config.model.checkpoint == str(tmp_path / "ready")
    assert loaded.config.opponent.checkpoint == str(tmp_path / "other")
    assert loaded.config.model.encoder.history == "public_aggregate"


@pytest.mark.parametrize("table", [
    '[model]\nmodel_id="lightgbm"\n[model.encoder]\nmax_elements=1',
    '[model]\nmodel_id="lightgbm"\n[model.initialization]\nrecipe="unknown"',
    '[model]\nmodel_id="lightgbm"\n[model.network]\nnum_leaves=true',
    '[model]\nmodel_id="lightgbm"\n[actions]\nmax_sequence_length=0',
    '[model]\nmodel_id="lightgbm"\n[actions]\nmax_sequence_length=4',
    '[model]\nmodel_id="lightgbm"\n[training]\nlearning_rate=nan',
    '[model]\nmodel_id="lightgbm"\n[training]\naccepted_decision_target=true',
    '[model]\nmodel_id="lightgbm"\n[training]\nseeds=[1,1]',
    '[model]\nmodel_id="random"\n[training]\nlearning_rate=.2',
])
def test_invalid_model_training_settings_rejected(tmp_path, table):
    path = write(tmp_path / "bad.toml", 'kind="experiment"\n[server]\nfixed_seed_experiment=true\n' + table + '\n')
    with pytest.raises((ContractError, ValueError)):
        load_config(path)


def test_default_real_training_and_resume_fail_before_optional_fit_or_output(tmp_path, monkeypatch, capsys):
    path = write(tmp_path / "empty.toml", f'kind="experiment"\n[model]\nmodel_id="lightgbm"\n[server]\nfixed_seed_experiment=true\n[run]\noutput_dir="{tmp_path / "output"}"\n')
    import gf_training.lightgbm_run as module
    monkeypatch.setattr(module, "_run_numeric", lambda *a, **k: pytest.fail("must not fit"))
    assert main(["train", "--config", str(path)]) == 2
    assert "pilot" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
    assert main(["train", "--config", str(tmp_path / "missing.toml"), "--mode", "resume"]) == 2
    assert "resume is unsupported" in capsys.readouterr().err


def test_baseline_import_help_validate_remain_native_dependency_free():
    import subprocess
    import sys
    code = '''
import importlib.abc, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"lightgbm", "sklearn", "torch", "pyarrow"}:
            raise AssertionError("unexpected optional import: " + fullname)
sys.meta_path.insert(0, NoML())
import gf_cli
assert not any(name in sys.modules for name in ("lightgbm", "sklearn", "torch", "pyarrow"))
try:
    gf_cli.main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
assert gf_cli.main(["validate", "--config", "configs/experiments/smoke.toml"]) == 0
'''
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_cli_current_fit_code_provider_uses_current_identity_without_importing_worker(monkeypatch):
    import hashlib
    import os
    import sys
    import time
    from pathlib import Path

    import gf_cli
    monkeypatch.setattr(gf_cli, "_code_provenance", lambda deadline: {"revision": "current-parent-revision", "dirty": True, "installed_build_revision": None})
    argv, environment = list(sys.argv), dict(os.environ)
    result = gf_cli._fit_code_provenance(time.monotonic() + 10)
    assert result["revision"] == "current-parent-revision" and result["package_version"]
    assert "gf_training.lightgbm_worker" not in sys.modules
    assert result["fit_module_sha256"]["gf_models.lightgbm.encoder"]
    from gf_models.lightgbm import encoder
    assert result["fit_module_sha256"]["gf_models.lightgbm.encoder"] == hashlib.sha256(Path(encoder.__file__).read_bytes()).hexdigest()
    assert argv == sys.argv and environment == dict(os.environ)
