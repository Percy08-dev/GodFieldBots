"""Real materialization/validation only; no network, learner, model or GPU allocation."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_common.domain import ContractError, clone, digest
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT

ROOT=Path(__file__).parents[1]
RUN_ID='00000000-0000-0000-0000-000000000017'


def materialized(kind='mlp',legacy=None):
    from gf_cli.hpo_config import campaign_config, load_hpo_config, materialize_hpo_run
    from gf_training.hpo_config import effective_settings
    if legacy==1:
        from tests.test_hpo_config import campaign_raw
        campaign=campaign_config(campaign_raw(),origin=ROOT)
    else:
        filename='heuristic-refresh-mlp-smoke.toml' if legacy==2 else f'resident-{kind}-smoke.toml'
        campaign=load_hpo_config(ROOT/'configs/hpo'/filename)
    settings=effective_settings(campaign.config.model,campaign.config.training,{'training.epochs':1},layout=LAYOUT)
    return materialize_hpo_run(campaign,settings,learning_seed=17,run_id=RUN_ID)


@pytest.fixture
def isolated_job(tmp_path,monkeypatch):
    from gf_training import hpo_gpu_lease as gpu
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'isolated-admission')
    def allocation_forbidden(*args,**kwargs): raise AssertionError('validation attempted resource/model allocation')
    monkeypatch.setattr('gf_models.structured_network.make_network',allocation_forbidden)
    monkeypatch.setattr('gf_training.hpo_ppo.HpoLearningProcess',allocation_forbidden)
    monkeypatch.setattr('gf_simulator_client.card_instance_worker.CardInstanceParticipantWorker',allocation_forbidden)
    with gpu.admit_gpu_job(gpu.JobContext('validation-only','bootstrap','training',RUN_ID,None),
            global_cuda_bytes=12*1024**3,max_workers=4) as job:
        yield job
        assert not job.ledger.workers and job.ledger.learner is None and job.ledger.cuda_bytes==0
        assert job.ledger._history==[]
        # This is our isolated test admission, with no resource creation calls.
        # It grants no recovery authority over the actual failed invocation.
        job.confirm_termination()


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_committed_resident_materialization_reaches_actual_config_validation(isolated_job,kind):
    from gf_training.hpo_ppo import _HpoOperations
    loaded=materialized(kind)
    assert loaded.config.hpo_spec['schema_version']==3
    assert digest(asdict(loaded.config))==loaded.sha256
    operations=_HpoOperations(isolated_job)
    model=operations.validate_config(loaded.config)
    assert model.model_id==kind and model.version=='4'
    assert operations.resident_runtime is None


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_materialized_resident_cpu_preflight_reaches_existing_cuda_guard(isolated_job,kind,monkeypatch):
    import time

    from gf_training.hpo_ppo import _HpoOperations
    loaded=materialized(kind)
    config=replace(loaded.config,training=replace(loaded.config.training,device='cpu'),
        model=replace(loaded.config.model,inference=replace(loaded.config.model.inference,device='cpu')))
    def forbidden(*args,**kwargs): raise AssertionError('CUDA guard reached Simulator runtime construction')
    monkeypatch.setattr('gf_simulator_client.card_instance_runtime.CardInstanceRuntimeEvidence',forbidden)
    with pytest.raises(ContractError,match='actual CUDA training/inference'):
        _HpoOperations(isolated_job).preflight(config,deadline=time.monotonic()+10)


@pytest.mark.parametrize('legacy',[1,2])
def test_genuine_legacy_materialization_keeps_existing_bootstrap_branch(isolated_job,legacy):
    from gf_training.hpo_ppo import _HpoOperations
    loaded=materialized(legacy=legacy)
    assert loaded.config.hpo_spec['schema_version']==legacy
    assert 'runtime' not in loaded.config.hpo_spec
    assert _HpoOperations(isolated_job).validate_config(loaded.config).model_id=='mlp'


@pytest.mark.parametrize('mutation',['old1-with-runtime','old2-with-runtime','resident-missing-runtime','runtime-foreign'])
def test_actual_config_validation_refuses_hybrid_runtime_controls(isolated_job,mutation):
    from gf_training.hpo_ppo import _HpoOperations
    loaded=materialized()
    spec=clone(loaded.config.hpo_spec)
    if mutation=='old1-with-runtime': spec['schema_version']=1
    elif mutation=='old2-with-runtime': spec['schema_version']=2
    elif mutation=='resident-missing-runtime': spec.pop('runtime')
    else: spec['runtime']['mode']='fresh'
    with pytest.raises(ContractError):
        _HpoOperations(isolated_job).validate_config(replace(loaded.config,hpo_spec=spec))


@pytest.mark.parametrize('variant',[None,'hpo-ppo-v1'])
def test_persisted_spec3_reader_still_requires_explicit_resident_discriminator(variant):
    from gf_training.hpo_lineage import hpo_spec
    loaded=materialized()
    state={'effective_config':asdict(loaded.config),'hpo_spec_sha256':digest(loaded.config.hpo_spec)}
    if variant is not None: state['run_variant']=variant
    with pytest.raises(ContractError): hpo_spec(state)
    state['run_variant']='hpo-ppo-resident-v1'
    assert hpo_spec(state)==loaded.config.hpo_spec


def test_resident_discriminator_cannot_reinterpret_legacy_spec():
    from gf_training.hpo_lineage import hpo_spec
    loaded=materialized(legacy=2)
    with pytest.raises(ContractError):
        hpo_spec({'effective_config':asdict(loaded.config),'hpo_spec_sha256':digest(loaded.config.hpo_spec),
            'run_variant':'hpo-ppo-resident-v1'})
