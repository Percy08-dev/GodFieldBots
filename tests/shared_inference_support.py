"""Representation-only shared-profile fixtures; no executed-runtime authority."""
from dataclasses import asdict


def fixture_runtime_receipt(contract):
    from gf_common.card_instance import reviewed_specs, shared_card_instance_contract
    from tests.test_card_instance_baseline_runtime import receipt
    value=receipt()
    if contract==shared_card_instance_contract():
        value['schema_version']=2
        value['action_contract']=asdict(contract)
        value['build']['spec_inventory_sha256']=contract.spec_inventory_sha256
        value['build']['spec_files']=reviewed_specs(contract)
        value['source'].update(revision=contract.simulator_revision,spec_files=reviewed_specs(contract),dirty=False)
    return value


def shared_owned_configuration(tmp_path,monkeypatch,*,decision_budget=32):
    """Typed synthetic lifecycle configuration; external Simulator I/O is not exercised."""
    from dataclasses import replace
    from pathlib import Path

    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.card_instance import (
        ActionConfigV3,
        reviewed_specs,
        shared_card_instance_contract,
    )
    from gf_common.domain import clone, digest
    from gf_common.shared_inference import shared_runtime
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_config import HPOTrainingConfig, effective_settings
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.test_hpo_config import campaign_raw
    from tests.test_hpo_numeric import learner_for
    raw=campaign_raw()
    raw.update(schema_version=5,origin='producer-owned-v1',encoder='canonical-python',decision_budget=decision_budget,
        base_experiment=str(Path(__file__).parents[1]/'configs/experiments/mlp-shared-inference-base.toml'),
        runtime=shared_runtime({'mode':'shared-mlp-v1'},game_concurrency=1),space={'parameters':[],'coupled':[]},
        operating={'game_concurrency':1,'max_participant_workers':2,'max_updates':2,'parent_budget_bytes':1024**3,'save_budget_bytes':1024**3},
        storage={'parent_root':str(tmp_path/'parents'),'numeric_root':str(tmp_path/'parents'),'originals_root':str(tmp_path/'originals')})
    campaign=campaign_config(raw,origin=tmp_path)
    tiny=learner_for('mlp')
    actions=ActionConfigV3(action_contract=shared_card_instance_contract(),max_candidates=64)
    base_config=replace(campaign.base.config,actions=actions)
    base_raw=clone(asdict(base_config))
    base=replace(campaign.base,config=base_config,effective=base_raw,sha256=digest(base_raw))
    config=replace(campaign.config,inference=replace(tiny.binding,device='cuda:0'),base_config_sha256=base.sha256)
    effective=clone(asdict(config))
    campaign=replace(campaign,config=config,base=base,effective=effective,sha256=digest({'campaign':effective,'base':base.sha256}))
    settings=effective_settings(tiny.model,replace(HPOTrainingConfig(),epochs=1),{},layout=layout)
    run_id='00000000-0000-0000-0000-000000000017'
    runtime=materialize_hpo_run(campaign,settings,learning_seed=17,run_id=run_id)
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'global')
    monkeypatch.setattr(_OwnedHpoOperations,'preflight',lambda self,config,deadline:self.validate_config(config))
    def integration(self,config,deadline):
        from gf_common.card_instance_runtime import SHARED_APPROVED_BINARY_SHA256
        from gf_models.arrays import sha256
        return {'binary_sha256':SHARED_APPROVED_BINARY_SHA256,'server_config_sha256':sha256(Path(config.server.config)),
            'declared_simulator_revision':actions.action_contract.simulator_revision,
            'source_revision':actions.action_contract.simulator_revision,'source_spec_sha256':reviewed_specs(actions.action_contract),'source_dirty':False}
    monkeypatch.setattr(_OwnedHpoOperations,'integration_identity',integration)
    return runtime,run_id,campaign,settings
