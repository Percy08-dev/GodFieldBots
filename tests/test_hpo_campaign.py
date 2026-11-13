"""Actual durable controller/Optuna state with external execution I/O isolated.

These synthetic receipts cannot qualify a learning/evaluation parent; production
verification is independently exercised by the receipt-boundary negative tests.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gf_cli.hpo_config import campaign_config
from gf_cli.hpo_storage import CampaignStore
from tests.test_hpo_config import campaign_raw


def controller_fixture(tmp_path,monkeypatch):
    import gf_cli.hpo as cli
    import gf_cli.hpo_evaluation as evaluation
    raw = campaign_raw()
    raw.update(trial_count=1,session_seconds=5)
    raw["space"] = {"parameters":[{"path":"training.epochs","kind":"categorical","choices":[2,4]}],"coupled":[]}
    raw["storage"] = {"parent_root":str(tmp_path),"numeric_root":str(tmp_path),"optuna_path":str(tmp_path / "study.sqlite3")}
    loaded = campaign_config(raw,origin=tmp_path)
    clock = [0.]
    import time
    monkeypatch.setattr(cli,"time",SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    monkeypatch.setattr(cli,"check_storage",lambda *args,**kwargs: {"scope":"unqualified storage unit"})
    monkeypatch.setattr(cli,"_conditions",lambda loaded:{"scope":"unqualified controller unit"})
    calls = []
    async def train(store,loaded,campaign,proposal,seed):
        calls.append(("train",seed))
        folder = f"proposals/{proposal.number:06}/seeds/{seed}"
        store.put(folder+"/training.intent.json",{"scope":"unqualified"})
        store.put(folder+"/training.json",{"training_seed":seed,"scope":"unqualified"})
        clock[0] += 6
    async def evaluate(store,loaded,campaign,proposal,training,*,output):
        seed=training["training_seed"]
        calls.append(("evaluate",seed))
        folder=f"proposals/{proposal.number:06}/seeds/{seed}"
        store.put(folder+"/evaluation.intent.json",{"scope":"unqualified"})
        store.put(folder+"/evaluation.json",{"scope":"unqualified"})
        clock[0] += 6
    monkeypatch.setattr(cli,"_train_seed",train)
    monkeypatch.setattr(cli,"evaluate_seed",evaluate)
    monkeypatch.setattr(cli,"verify_training_receipt",lambda store,loaded,campaign,proposal,seed,**kwargs:
        store.get(f"proposals/{proposal.number:06}/seeds/{seed}/training.json")[0])
    monkeypatch.setattr(cli,"verify_seed_evaluation",lambda *args,**kwargs: None)
    def aggregate(store,loaded,campaign,proposal,**kwargs):
        return {"schema_version":1,"kind":"hpo-evaluation","proposal_sha256":proposal.sha256,
            "training_receipts":[],"evaluation_receipts":[],"value":.5,"projection_sha256":"a"*64,
            "label":"exploratory-hpo-validation"}
    monkeypatch.setattr(cli,"aggregate_trial_evaluation",aggregate)
    def verify(store,loaded,campaign,proposal,**kwargs):
        value,sha=store.get(f"proposals/{proposal.number:06}/evaluation.json")
        return value,value["value"],sha
    monkeypatch.setattr(evaluation,"verify_trial_evaluation",verify)
    return cli,loaded,clock,calls


def test_session_overrun_finishes_whole_operations_and_resume_never_retrains_closed_seeds(tmp_path,monkeypatch):
    cli,loaded,_,calls=controller_fixture(tmp_path,monkeypatch)
    directory=tmp_path / "campaign"
    first=asyncio.run(cli.run_campaign(loaded,directory,mode="fresh"))
    assert first["status"]=="paused"
    assert calls==[("train",17)]
    for _ in range(4): result=asyncio.run(cli.run_campaign(loaded,directory,mode="resume"))
    assert result["status"]=="complete"
    assert calls==[("train",17),("evaluate",17),("train",29),("evaluate",29)]
    with CampaignStore(directory,mode="resume",max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        value,_=store.get("proposals/000000/tell.receipt.json")
        assert value["state"]=="COMPLETE" and value["value"]==.5


def test_after_cleanup_before_closed_receipt_gap_fails_whole_trial_without_retry(tmp_path,monkeypatch):
    cli,loaded,_,calls=controller_fixture(tmp_path,monkeypatch)
    directory=tmp_path / "campaign"
    asyncio.run(cli.run_campaign(loaded,directory,mode="fresh"))
    # Controlled external cleanup may have finished, but an absent closed
    # receipt is deliberately NOT reconstructed from success-looking data.
    (directory / "proposals/000000/seeds/17/training.json").unlink()
    result=asyncio.run(cli.run_campaign(loaded,directory,mode="resume"))
    assert result["failed_in_session"]==1
    assert calls==[("train",17)]
    with CampaignStore(directory,mode="resume",max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        value,_=store.get("proposals/000000/tell.receipt.json")
        assert value["state"]=="FAIL" and value["value"] is None


def test_stage_settings_are_fresh_only_and_failures_override_success_looking_data(tmp_path,monkeypatch):
    cli,loaded,_,_=controller_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(cli.run_campaign(loaded,tmp_path / "campaign",mode="resume",seed_paths=(tmp_path / "seed",)))


def test_closed_prior_seed_is_revalidated_before_later_seed_executes(tmp_path,monkeypatch):
    cli,loaded,_,calls=controller_fixture(tmp_path,monkeypatch)
    directory=tmp_path / "campaign"
    asyncio.run(cli.run_campaign(loaded,directory,mode="fresh"))
    asyncio.run(cli.run_campaign(loaded,directory,mode="resume"))
    checked=[]
    monkeypatch.setattr(cli,"verify_seed_evaluation",lambda *args,**kwargs: checked.append(args[4]))
    original=cli._train_seed
    async def later_seed(*args):
        if args[4]==29:
            assert 17 in checked,"closed prior seed must be independently revalidated before reuse"
        await original(*args)
    monkeypatch.setattr(cli,"_train_seed",later_seed)
    result=asyncio.run(cli.run_campaign(loaded,directory,mode="resume"))
    assert result["status"]=="paused"
    assert ("train",29) in calls


def test_recorded_failure_wins_over_success_looking_seed_receipts(tmp_path,monkeypatch):
    cli,loaded,_,calls=controller_fixture(tmp_path,monkeypatch)
    directory=tmp_path / "campaign"
    asyncio.run(cli.run_campaign(loaded,directory,mode="fresh"))
    with CampaignStore(directory,mode="resume",max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        store.put("proposals/000000/failure.json",{"schema_version":1,"kind":"hpo-proposal-failure","reason":"recorded-failure"})
        store.put("proposals/000000/seeds/17/evaluation.json",{"scope":"success-looking unqualified"})
    result=asyncio.run(cli.run_campaign(loaded,directory,mode="resume"))
    assert result["failed_in_session"]==1
    assert calls==[("train",17)]
