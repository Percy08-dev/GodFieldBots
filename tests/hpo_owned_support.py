"""Unqualified fixed-input setup shared by owned lifecycle tests."""
from dataclasses import asdict, replace
from pathlib import Path

from gf_common.domain import clone, digest


def owned_runtime(tmp_path,monkeypatch,kind='mlp',decision_budget=32,*,save_budget_bytes=None,operating=None,diagnostic_storage=None):
    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_config import HPOTrainingConfig, effective_settings
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_config import campaign_raw
    from tests.test_hpo_numeric import learner_for
    raw=campaign_raw()
    raw.update(schema_version=4,model_id=kind,origin='producer-owned-v1',encoder='canonical-python',decision_budget=decision_budget,
        base_experiment=str(Path(__file__).parents[1]/f'configs/experiments/card-instance-{kind}-ppo-acceptance.toml'),
        runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':128*1024**2},
        storage={'parent_root':str(tmp_path/'parents'),'numeric_root':str(tmp_path/'parents'),'originals_root':str(tmp_path/'originals')})
    if operating is not None: raw['operating']=dict(operating)
    if save_budget_bytes is not None: raw.setdefault('operating',{})['save_budget_bytes']=save_budget_bytes
    if diagnostic_storage is not None: raw['storage']['hpo_diagnostics']=diagnostic_storage
    raw['space']={'parameters':[],'coupled':[]}
    campaign=campaign_config(raw,origin=tmp_path,_new_storage_defaults=False)
    tiny=learner_for(kind)
    settings=effective_settings(tiny.model,replace(HPOTrainingConfig(),epochs=1),{},layout=LAYOUT)
    run_id='00000000-0000-0000-0000-000000000017'
    runtime=materialize_hpo_run(campaign,settings,learning_seed=17,run_id=run_id)
    config=replace(runtime.config,actions=ACTIONS,model=replace(runtime.config.model,inference=tiny.binding,cuda_visible_devices=''),
        training=replace(runtime.config.training,device='cpu'))
    value=clone(asdict(config))
    runtime=LoadedConfig(config,value,digest(value),runtime.inputs)
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'global')
    monkeypatch.setattr(_OwnedHpoOperations,'preflight',lambda self,config,deadline:self.validate_config(config))
    monkeypatch.setattr(_OwnedHpoOperations,'integration_identity',lambda self,config,deadline:{'binary_sha256':__import__('gf_common.card_instance_runtime',fromlist=['APPROVED_BINARY_SHA256']).APPROVED_BINARY_SHA256,
        'declared_simulator_revision':ACTIONS.action_contract.simulator_revision,'source_revision':ACTIONS.action_contract.simulator_revision,
        'source_spec_sha256':dict(__import__('gf_common.card_instance',fromlist=['CORRECTED_REVIEWED_SPEC_HASHES']).CORRECTED_REVIEWED_SPEC_HASHES),
        'server_config_sha256':__import__('gf_models.arrays',fromlist=['sha256']).sha256(Path(config.server.config))})
    return runtime,run_id,campaign,settings


async def run_owned(runtime,directory,*,mode='fresh',pause_after_update=None,production=False):
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.ppo_run import _run_ppo
    from tests.test_hpo_owned_run import owned_fixture_collection
    config=runtime.config
    with gpu.admit_gpu_job(gpu.JobContext('fixture',config.run.name,'training',config.run.name,None),
            global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=4) as job:
        ops=_OwnedHpoOperations(job)
        async def collect(plan,ledger):
            if production:
                from gf_cli.card_instance import collect_hpo_ppo
                return await collect_hpo_ppo(runtime,plan,ledger)
            return await owned_fixture_collection(ops,config,plan,ledger)
        report=await _run_ppo(runtime,collect,directory,mode=mode,pause_after_update=pause_after_update,_game_ops=ops)
        return report,ops


def auditable_runtime(monkeypatch):
    """Controlled public endpoints with both correctly scoped diagnostic Replay views."""
    import copy

    from tests.test_group_pack_runtime_environment import PublicClient
    from tests.test_hpo_owned_run import install_owned_fixture_runtime
    install_owned_fixture_runtime(monkeypatch)
    replay=PublicClient.replay
    async def own_replay(self,game,viewer):
        result=copy.deepcopy(await replay(self,game,viewer))
        for turn in result['turns']:
            for decision in turn['decisions']:
                observation=decision['observation']
                observation.update(viewer_player_id=viewer,current_player=decision['actor'],active_player=decision['actor'])
                observation['own']['player_id']=viewer
                observation['opponent']['player_id']=3-viewer
                decision['action_options']['actor']=decision['actor']
        return result
    monkeypatch.setattr(PublicClient,'replay',own_replay)
    def component_gate(state): assert state['device']=='cpu'
    monkeypatch.setattr('gf_training.hpo_source.require_hpo_production',component_gate)
