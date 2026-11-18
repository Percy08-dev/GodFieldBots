"""Shared training selects existing fresh evaluation and its real role geometry."""
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError


def test_shared_campaign_uses_fresh_evaluation_without_relabeling_training():
    from gf_cli.hpo_config import load_hpo_config
    from gf_common.shared_inference import evaluation_runtime
    loaded=load_hpo_config(Path(__file__).parents[1]/'configs/hpo/mlp-shared-inference.toml')
    assert evaluation_runtime(loaded.config) is None
    assert loaded.config.runtime['mode']=='shared-mlp-v1'
    old=load_hpo_config(Path(__file__).parents[1]/'configs/hpo/owned-mlp-smoke.toml')
    assert evaluation_runtime(old.config)==old.config.runtime


def test_full_campaign_preflight_charges_fresh_eval_roles_before_training():
    from gf_cli.hpo_config import load_hpo_config
    from gf_cli.hpo_evaluation import prospective_evaluation_admission
    loaded=load_hpo_config(Path(__file__).parents[1]/'configs/hpo/mlp-shared-inference.toml')
    result=prospective_evaluation_admission(loaded)
    assert result['workers']==6
    assert result['worker_resident_bytes']>0
    assert result['evaluation_runtime'] is None
    bigger=replace(loaded,config=replace(loaded.config,operating=replace(loaded.config.operating,
        game_concurrency=16,max_participant_workers=32)))
    with pytest.raises(ContractError,match='budget'):
        prospective_evaluation_admission(bigger)
