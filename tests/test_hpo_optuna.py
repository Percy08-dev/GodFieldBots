"""Actual local Optuna5 storage/proposals; no qualified learning or GPU run."""
from dataclasses import replace

import pytest

from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign, proposal_seed
from gf_cli.hpo_config import campaign_config
from gf_cli.hpo_storage import CampaignStore
from gf_common.domain import ContractError
from tests.test_hpo_config import campaign_raw


def loaded_for(tmp_path):
    raw = campaign_raw()
    raw["space"] = {"parameters":[{"path":"training.epochs","kind":"categorical","choices":[2,4]}],"coupled":[]}
    raw["storage"] = {"parent_root":str(tmp_path / "parents"),"numeric_root":str(tmp_path / "parents"),
                      "optuna_path":str(tmp_path / "optuna.sqlite3")}
    return campaign_config(raw,origin=tmp_path)


def test_actual_enqueue_ask_complete_proposal_and_fail_tell(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"scope":"unqualified-component-fixture"})
        campaign = CampaignStudy(store,loaded,manifest)
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        assert proposal is not None
        assert proposal.params == {"training.epochs":1}
        assert proposal.settings.training.epochs == 4
        assert store.exists("proposals/000000/proposal.json")
        assert store.exists("proposals/000000/settings-seed.json")
        with campaign.study(0) as study:
            assert study.get_trials()[0].state.name == "RUNNING"
            assert study.get_trials()[0].params == proposal.params
        campaign.fail(proposal,"component test has no Source")
        with campaign.study(0) as study:
            trial = study.get_trials()[0]
            assert trial.state.name == "FAIL"
            assert trial.value is None
        campaign.fail(proposal,"component test has no Source")


def test_proposal_seed_is_number_derived_and_effective_duplicate_never_executes(tmp_path):
    loaded = loaded_for(tmp_path)
    assert proposal_seed(19,0) != proposal_seed(19,1)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        first = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        campaign.fail(first,"unqualified")
        assert campaign.propose(1,slot=1,kind="seed",requested=canonical_control(loaded.config)) is None
        assert store.exists("proposals/000001/duplicate.json")
        with campaign.study(1) as study:
            assert [trial.state.name for trial in study.get_trials()] == ["FAIL","FAIL"]
        with pytest.raises(ContractError): campaign.propose(2,slot=2,kind="sample")


def test_actual_study_cannot_be_scheduled_by_another_campaign_directory(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "first",mode="fresh",max_owned_bytes=8*1024**2) as first:
        manifest = initialize_campaign(first,loaded,{"fixture":True})
        campaign = CampaignStudy(first,loaded,manifest)
        with campaign.study(0): pass
    with CampaignStore(tmp_path / "other",mode="fresh",max_owned_bytes=8*1024**2) as second:
        second.put("campaign.json",manifest)
        with pytest.raises(ContractError), CampaignStudy(second,loaded,manifest).study(0): pass


def test_partial_running_ask_after_generation_crash_is_unexecuted_failure(tmp_path):
    loaded = loaded_for(tmp_path)
    directory = tmp_path / "campaign"
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"fixture":True})
        campaign = CampaignStudy(store,loaded,manifest)
        campaign.begin(0,slot=0,kind="sample")
        with campaign.study(0) as study:
            study.ask().suggest_categorical("training.epochs",[0,1])
    with CampaignStore(directory,mode="resume",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,manifest)
        assert campaign.propose(0,slot=0,kind="sample") is None
        assert store.exists("proposals/000000/generation-failure.json")
        with campaign.study(0) as study:
            assert study.get_trials()[0].state.name == "FAIL"
            assert study.get_trials()[0].value is None


def test_canonical_control_remains_global_default_despite_search_base_override(tmp_path):
    loaded = loaded_for(tmp_path)
    changed = replace(loaded.config,training=replace(loaded.config.training,epochs=2))
    assert canonical_control(changed).training.epochs == 4


def test_unexplained_study_trial_is_not_absorbed_as_another_attempt(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        with campaign.study(0) as study: study.ask()
        with pytest.raises(ContractError): campaign.propose(0,slot=0,kind="sample")
        assert not (store.directory / "proposals/000000/proposal.json").exists()


def test_partial_ask_with_unknown_distribution_is_not_hidden_as_our_failure(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        campaign.begin(0,slot=0,kind="sample")
        with campaign.study(0) as study: study.ask().suggest_float("unknown",0,1)
        with pytest.raises(ContractError): campaign.propose(0,slot=0,kind="sample")


def test_waiting_fixed_control_reconciles_exact_intent_after_enqueue_crash(tmp_path):
    loaded = loaded_for(tmp_path)
    directory = tmp_path / "campaign"
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"fixture":True})
        campaign = CampaignStudy(store,loaded,manifest)
        intent = campaign.begin(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        _,sha = store.get("proposals/000000/intent.json")
        attrs = {"hpo_intent":{"campaign_id":manifest["campaign_id"],"proposal_no":0,"intent_sha256":sha}}
        with campaign.study(0) as study:
            study.enqueue_trial(intent["fixed_params"],user_attrs=attrs,skip_if_exists=False)
    with CampaignStore(directory,mode="resume",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,manifest)
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        assert proposal is not None and proposal.trial_number == 0
        campaign.fail(proposal,"component")


def test_existing_proposal_cannot_hide_changed_optuna_binding(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        with campaign.study(0) as study:
            import optuna
            trial = optuna.Trial(study,study.get_trials()[0]._trial_id)
            trial.set_user_attr("hpo_proposal_sha256","0"*64)
        with pytest.raises(ContractError):
            campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))


def test_actual_conditional_coupled_choices_are_index_suggestions_and_replay(tmp_path):
    from tests.test_hpo_config import space_raw

    raw = campaign_raw()
    raw["space"] = space_raw()
    raw["storage"] = {"optuna_path":str(tmp_path / "conditional.sqlite3")}
    loaded = campaign_config(raw,origin=tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        proposal = campaign.propose(0,slot=0,kind="sample")
        assert proposal is not None
        from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
        from gf_training.hpo_config import effective_settings
        replay = effective_settings(loaded.config.model,loaded.config.training,loaded.config.space.resolve(proposal.params),layout=layout)
        assert replay == proposal.settings
        for parameter in loaded.config.space.parameters:
            if parameter.kind == "categorical" and parameter.path in proposal.params:
                assert type(proposal.params[parameter.path]) is int
        for coupled in loaded.config.space.coupled:
            if coupled.name in proposal.params:
                assert type(proposal.params[coupled.name]) is int
        campaign.fail(proposal,"unqualified component")


def test_settings_seed_contains_only_architecture_recipe_and_imports_fresh(tmp_path):
    from gf_cli.hpo_campaign import import_settings

    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "first",mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"fixture":True})
        campaign = CampaignStudy(store,loaded,manifest)
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        path = store.directory / "proposals/000000/settings-seed.json"
        seed,_ = store.get("proposals/000000/settings-seed.json")
        assert "resources" not in seed["model"]
        assert not ({"weights","checkpoint","optimizer_state","outcome","value"} & seed.keys())
        imported = import_settings((path,),loaded.config,study_identity="f"*64)
        assert imported[0].identity == proposal.settings.identity
        with pytest.raises(ContractError): import_settings((path,),loaded.config,study_identity=manifest["study_identity"])
        with pytest.raises(ContractError): import_settings((path,path),loaded.config,study_identity="f"*64)


def test_outside_space_canonical_control_never_enters_sampler_history(tmp_path):
    raw = campaign_raw()
    raw["space"] = {"parameters":[{"path":"training.epochs","kind":"categorical","choices":[2,3]}],"coupled":[]}
    raw["storage"] = {"optuna_path":str(tmp_path / "outside.sqlite3")}
    loaded = campaign_config(raw,origin=tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        proposal = campaign.propose(0,slot=None,kind="outside-control",requested=canonical_control(loaded.config))
        assert proposal is not None and proposal.slot is None and proposal.trial_number is None
        campaign.fail(proposal,"unqualified component")
        with campaign.study(1) as study: assert study.get_trials() == []


def test_actual_complete_tell_and_history_exact_replay_with_verification_io_separated(tmp_path,monkeypatch):
    """Synthetic score isolates SQLite behavior; this is NOT Source evidence."""
    import gf_cli.hpo_evaluation as evaluation

    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"scope":"unqualified SQLite unit"}))
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        aggregate = {"schema_version":1,"kind":"hpo-evaluation","proposal_sha256":proposal.sha256,
            "training_receipts":[],"evaluation_receipts":[],"value":.75,"projection_sha256":"a"*64,
            "label":"exploratory-hpo-validation"}
        sha = store.put("proposals/000000/evaluation.json",aggregate)
        monkeypatch.setattr(evaluation,"verify_trial_evaluation",lambda *args,**kwargs: (aggregate,.75,sha))
        first = campaign.complete(proposal,deadline=float("inf"))
        assert campaign.complete(proposal,deadline=float("inf")) == first
        with campaign.study(1) as study:
            assert study.get_trials()[0].state.name == "COMPLETE"
            assert study.get_trials()[0].value == .75
        intent = campaign.begin(1,slot=1,kind="sample")
        from gf_common.domain import digest
        assert intent["completed_history_sha256"] != digest([])


def test_json_aggregate_alone_cannot_authorize_complete_tell(tmp_path):
    loaded = loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        store.put("proposals/000000/evaluation.json",{"schema_version":1,"kind":"hpo-evaluation",
            "proposal_sha256":proposal.sha256,"training_receipts":[],"evaluation_receipts":[],"value":1.,
            "projection_sha256":"0"*64,"label":"exploratory-hpo-validation"})
        with pytest.raises((ContractError,FileNotFoundError)):
            campaign.complete(proposal,deadline=float("inf"))
        with campaign.study(0) as study:
            assert study.get_trials()[0].state.name == "RUNNING"
            assert study.get_trials()[0].value is None


def test_failed_partial_ask_reconciles_crash_before_failure_record(tmp_path,monkeypatch):
    loaded = loaded_for(tmp_path)
    directory = tmp_path / "campaign"
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"fixture":True})
        campaign = CampaignStudy(store,loaded,manifest)
        campaign.begin(0,slot=0,kind="sample")
        with campaign.study(0) as study: study.ask().suggest_categorical("training.epochs",[0,1])
        original = store.put
        def fail_record(name,value):
            if name.endswith("generation-failure.json"): raise OSError("failure record fault")
            return original(name,value)
        monkeypatch.setattr(store,"put",fail_record)
        with pytest.raises(OSError): campaign.propose(0,slot=0,kind="sample")
    with CampaignStore(directory,mode="resume",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,manifest)
        assert campaign.propose(0,slot=0,kind="sample") is None
        assert store.exists("proposals/000000/generation-failure.json")


def test_full_proposal_crash_before_settings_export_never_loses_fixed_configuration(tmp_path,monkeypatch):
    loaded = loaded_for(tmp_path)
    directory = tmp_path / "campaign"
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest = initialize_campaign(store,loaded,{"fixture":True})
        campaign = CampaignStudy(store,loaded,manifest)
        original = store.put
        def fail_export(name,value):
            if name.endswith("settings-seed.json"): raise OSError("export fault")
            return original(name,value)
        monkeypatch.setattr(store,"put",fail_export)
        with pytest.raises(OSError): campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
    with CampaignStore(directory,mode="resume",max_owned_bytes=8*1024**2) as store:
        campaign = CampaignStudy(store,loaded,manifest)
        proposal = campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        assert proposal.settings == canonical_control(loaded.config)
        assert store.exists("proposals/000000/settings-seed.json")
        campaign.fail(proposal,"component")


def test_pruned_state_is_unsupported_not_a_silent_failed_or_complete_trial(tmp_path):
    import optuna
    loaded=loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign=CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"fixture":True}))
        proposal=campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        with campaign.study(0) as study: study.tell(proposal.trial_number,state=optuna.trial.TrialState.PRUNED)
        with pytest.raises(ContractError):
            campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))


def test_resume_rejects_changed_actual_configuration_inputs(tmp_path):
    loaded=loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest=initialize_campaign(store,loaded,{"fixture":True})
        changed=replace(loaded,inputs=(*loaded.inputs,{"path":"foreign","sha256":"0"*64}))
        with pytest.raises(ContractError): CampaignStudy(store,changed,manifest)


def test_actual_completed_tell_crash_before_receipt_replays_exactly(tmp_path,monkeypatch):
    """SQLite/tell transaction only; synthetic verifier is not Source evidence."""
    import gf_cli.hpo_evaluation as evaluation
    loaded=loaded_for(tmp_path)
    directory=tmp_path / "campaign"
    def verification(store,loaded,campaign,proposal,**kwargs):
        value,sha=store.get(f"proposals/{proposal.number:06}/evaluation.json")
        return value,value["value"],sha
    monkeypatch.setattr(evaluation,"verify_trial_evaluation",verification)
    with CampaignStore(directory,mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest=initialize_campaign(store,loaded,{"scope":"unqualified SQLite fault unit"})
        campaign=CampaignStudy(store,loaded,manifest)
        proposal=campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        store.put("proposals/000000/evaluation.json",{"schema_version":1,"kind":"hpo-evaluation",
            "proposal_sha256":proposal.sha256,"value":.25})
        original=store.put
        def fail_receipt(name,value):
            if name.endswith("tell.receipt.json"): raise OSError("post-tell receipt fault")
            return original(name,value)
        monkeypatch.setattr(store,"put",fail_receipt)
        with pytest.raises(OSError): campaign.complete(proposal,deadline=float("inf"))
        with campaign.study(0) as study:
            assert study.get_trials()[0].state.name=="COMPLETE"
            assert study.get_trials()[0].value==.25
    with CampaignStore(directory,mode="resume",max_owned_bytes=8*1024**2) as store:
        campaign=CampaignStudy(store,loaded,manifest)
        proposal=campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        assert campaign.complete(proposal,deadline=float("inf"))["value"]==.25
        assert store.exists("proposals/000000/tell.receipt.json")


def test_exact_tell_conflict_never_overwrites_actual_completed_value(tmp_path,monkeypatch):
    import gf_cli.hpo_evaluation as evaluation
    loaded=loaded_for(tmp_path)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign=CampaignStudy(store,loaded,initialize_campaign(store,loaded,{"scope":"unqualified SQLite conflict unit"}))
        proposal=campaign.propose(0,slot=0,kind="control",requested=canonical_control(loaded.config))
        value={"schema_version":1,"kind":"hpo-evaluation","proposal_sha256":proposal.sha256,"value":.75}
        sha=store.put("proposals/000000/evaluation.json",value)
        monkeypatch.setattr(evaluation,"verify_trial_evaluation",lambda *args,**kwargs:(value,.75,sha))
        campaign.complete(proposal,deadline=float("inf"))
        monkeypatch.setattr(evaluation,"verify_trial_evaluation",lambda *args,**kwargs:(value,.5,sha))
        with pytest.raises(ContractError): campaign.complete(proposal,deadline=float("inf"))


def test_partial_ask_inactive_coupled_parameter_is_unexplained_not_our_generation(tmp_path):
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    raw=campaign_raw()
    raw["storage"]={"optuna_path":str(tmp_path / "study.sqlite3")}
    raw["space"]={"parameters":[{"path":"model.hand.kind","kind":"categorical","choices":["direct","attention"]}],
        "coupled":[{"name":"attention","paths":["model.card.output_size","model.hand.attention_heads"],
            "choices":[[32,2],[64,4]],"when":{"model.hand.kind":["attention"]}}]}
    loaded=campaign_config(raw,origin=tmp_path)
    direct=effective_settings(loaded.config.model,loaded.config.training,loaded.config.space.resolve({"model.hand.kind":0}),layout=layout)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        manifest=initialize_campaign(store,loaded,{"fixture":True})
        campaign=CampaignStudy(store,loaded,manifest)
        intent=campaign.begin(0,slot=0,kind="seed",requested=direct)
        _,sha=store.get("proposals/000000/intent.json")
        with campaign.study(0) as study:
            study.enqueue_trial(intent["fixed_params"],user_attrs={"hpo_intent":{"campaign_id":manifest["campaign_id"],
                "proposal_no":0,"intent_sha256":sha}},skip_if_exists=False)
            trial=study.ask()
            assert trial.suggest_categorical("model.hand.kind",[0,1])==0
            trial.suggest_categorical("attention",[0,1])
        with pytest.raises(ContractError): campaign.propose(0,slot=0,kind="seed",requested=direct)
