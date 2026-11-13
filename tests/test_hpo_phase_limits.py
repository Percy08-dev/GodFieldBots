"""Additional numeric/save envelopes do not replace learner model resources."""
from dataclasses import asdict

import pytest

pytest.importorskip('torch')

from gf_common.domain import ContractError
from gf_models.structured_config import InferenceBindingV2
from tests.test_card_instance_learning import ACTIONS, RULES
from tests.test_hpo_numeric import learner_for


def test_wire6_phase_limits_are_closed_and_config_bound():
    from gf_training.hpo_numeric_worker import wire_identity
    learner=learner_for()
    limits={'numeric_budget_bytes':1234567,'save_budget_bytes':512*1024**2}
    raw=wire_identity(learner.model,learner.training,learner.binding,learner.action_config_hash,
        learner.resume_identity,actions=ACTIONS,cuda_visible_devices='',phase_limits=limits)
    assert raw['phase_limits']==limits
    with pytest.raises(ContractError):
        wire_identity(learner.model,learner.training,learner.binding,learner.action_config_hash,
            learner.resume_identity,actions=ACTIONS,cuda_visible_devices='',phase_limits=limits|{'unknown':1})


def test_save_envelope_refuses_before_creating_artifact(tmp_path):
    from gf_training.hpo_artifacts import save_estimate
    learner=learner_for()
    assert save_estimate(learner.model)['additional_bytes'] < 512*1024**2
    with pytest.raises(ContractError): learner.save(tmp_path/'no-output',save_budget_bytes=1)
    assert not (tmp_path/'no-output').exists()
    saved=learner.save(tmp_path/'saved',save_budget_bytes=512*1024**2)
    assert saved['files']['resume.json']==saved['resume_digest']


def test_materialization_admits_defaults_and_rejects_independent_numeric_root(tmp_path):
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_training.hpo_config import effective_settings
    from tests.model3_support import LAYOUT
    from tests.test_hpo_config import campaign_raw
    raw=campaign_raw()
    campaign=campaign_config(raw,origin=tmp_path)
    settings=effective_settings(campaign.config.model,campaign.config.training,{},layout=LAYOUT)
    import uuid
    loaded=materialize_hpo_run(campaign,settings,learning_seed=17,run_id=str(uuid.uuid4()))
    assert loaded.config.storage.budget_mib*1024**2<=loaded.config.hpo_spec['operating']['collector_budget_bytes']
    assert loaded.config.hpo_spec['storage']['numeric_root']==loaded.config.hpo_spec['storage']['parent_root']
    assert loaded.config.storage.db_root=='/mnt/d/godfield'
    invalid=campaign_config(raw|{'storage':{'numeric_root':str(tmp_path/'other')}},origin=tmp_path)
    with pytest.raises(ContractError):
        materialize_hpo_run(invalid,settings,learning_seed=17,run_id=str(uuid.uuid4()))


def test_hpo_runtime_cuda_domain_does_not_widen_old_helper():
    from gf_cli.hpo_config import HPOOperatingConfig
    from gf_models.structured_runtime import admit_hpo_training, admit_planned_game_training_v3
    from tests.test_hpo_network import small_model4
    op=asdict(HPOOperatingConfig())
    op['managed_cuda_bytes']=32*1024**3
    model=small_model4()
    binding=InferenceBindingV2(device='cuda:0',max_candidates=64)
    assert admit_hpo_training(ACTIONS,model,binding,32,op,rules=RULES)['peak_cuda_bytes']<=op['managed_cuda_bytes']
    with pytest.raises(ContractError):
        admit_planned_game_training_v3(ACTIONS,model,binding,binding,rules=RULES,
            game_concurrency=2,global_cuda_bytes=op['managed_cuda_bytes'],max_workers=4)


def test_materialization_rejects_effective_settings_from_another_model_study(tmp_path):
    import uuid

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_models.hpo_config import model_defaults_v4
    from gf_training.hpo_config import effective_settings
    from tests.model3_support import LAYOUT
    from tests.test_hpo_config import campaign_raw
    campaign=campaign_config(campaign_raw(),origin=tmp_path)
    foreign=effective_settings(model_defaults_v4('gru'),campaign.config.training,{},layout=LAYOUT)
    with pytest.raises(ContractError,match='model'):
        materialize_hpo_run(campaign,foreign,learning_seed=17,run_id=str(uuid.uuid4()))


def test_joint_ram_rejects_retained_source_during_unequal_save_and_ack(tmp_path):
    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    from gf_models.structured_runtime import admit_hpo_training
    from tests.model3_support import LAYOUT
    from tests.test_hpo_config import campaign_raw
    raw=campaign_raw()
    raw['operating']={'game_concurrency':1,'source_budget_bytes':8*1024**3,
        'numeric_budget_bytes':512*1024**2,'save_budget_bytes':8*1024**3,'managed_ram_bytes':14*1024**3}
    loaded=campaign_config(raw,origin=tmp_path)
    config=loaded.config
    for check in (lambda:joint_resource_admission(config,layout=LAYOUT),
            lambda:admit_hpo_training(loaded.base.config.actions,config.model,config.inference,
                config.training.tbptt_length,asdict(config.operating),rules=RULES)):
        with pytest.raises(ContractError,match='RAM'): check()


def test_cli_and_runtime_share_full_save_ack_overlap_estimate(tmp_path):
    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    from gf_models.structured_runtime import admit_hpo_training
    from tests.model3_support import LAYOUT
    from tests.test_hpo_config import campaign_raw
    raw=campaign_raw()
    raw['operating']={'game_concurrency':1,'source_budget_bytes':8*1024**3,
        'numeric_budget_bytes':512*1024**2,'save_budget_bytes':8*1024**3,'managed_ram_bytes':32*1024**3}
    loaded=campaign_config(raw,origin=tmp_path)
    config=loaded.config
    cli=joint_resource_admission(config,layout=LAYOUT)
    runtime=admit_hpo_training(loaded.base.config.actions,config.model,config.inference,
        config.training.tbptt_length,asdict(config.operating),rules=RULES)
    assert cli==runtime
    op=config.operating
    assert cli['ram_phases']['save']==(op.parent_budget_bytes+cli['learner_bytes']
        +op.source_budget_bytes+op.numeric_budget_bytes+op.save_budget_bytes)
