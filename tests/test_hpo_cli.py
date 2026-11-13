"""Optional-free HPO command parsing and honest saved-state inspection."""
import subprocess
import sys

from gf_cli.hpo_campaign import initialize_campaign
from gf_cli.hpo_storage import CampaignStore
from tests.test_hpo_optuna import loaded_for


def test_hpo_help_never_imports_torch_or_optuna():
    source = """
import sys
from gf_cli.hpo import main
try: main(['--help'])
except SystemExit as e: assert e.code==0
assert 'torch' not in sys.modules
assert 'optuna' not in sys.modules
"""
    result = subprocess.run([sys.executable,"-E","-c",source],check=False,capture_output=True,text=True,timeout=10)
    assert result.returncode == 0,result.stderr
    assert "compare" in result.stdout and "run" in result.stdout
    source = source.replace('from gf_cli.hpo import main','from gf_cli import main').replace("main(['--help'])","main(['hpo','--help'])")
    result = subprocess.run([sys.executable,"-E","-c",source],check=False,capture_output=True,text=True,timeout=10)
    assert result.returncode == 0,result.stderr


def test_saved_inspect_is_structural_and_cannot_claim_qualification(tmp_path,capsys):
    from gf_cli.hpo import main
    loaded = loaded_for(tmp_path)
    directory = tmp_path / "campaign"
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        initialize_campaign(store,loaded,{"scope":"unqualified component"})
    assert main(["inspect","--campaign-dir",str(directory)]) == 0
    import json
    result = json.loads(capsys.readouterr().out)
    assert result["inspection_scope"] == "structural-only"
    assert result["qualification"] == "not-performed"


def test_run_requires_explicit_fresh_or_resume_mode(tmp_path):
    result = subprocess.run([sys.executable,"-E","-m","gf_cli.hpo","run","--config","missing.toml","--campaign-dir",str(tmp_path)],
        check=False,capture_output=True,text=True,timeout=10)
    assert result.returncode != 0
    assert "--mode" in result.stderr
