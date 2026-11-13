"""Actual receipt boundaries; controlled CPU files never qualify a PPO parent."""
import asyncio
from dataclasses import asdict
from pathlib import Path

import pytest

from gf_cli.hpo_campaign import canonical_control
from gf_cli.hpo_evaluation import inspect_training_parent, verify_gpu_termination
from gf_common.domain import ContractError, clone
from gf_training import hpo_gpu_lease as gpu
from tests.test_hpo_optuna import loaded_for


def test_only_fixed_namespace_owned_termination_matches_campaign_context(tmp_path,monkeypatch):
    monkeypatch.setattr(gpu,"_GLOBAL_ROOT",tmp_path / "global")
    with gpu.admit_gpu_job(gpu.JobContext("campaign","run","training","run",None),global_cuda_bytes=1024,max_workers=2) as job:
        value = clone(asdict(job.confirm_termination()))
    verify_gpu_termination(value,campaign_id="campaign",operation_id="run",run_id="run",evaluation_id=None)
    with pytest.raises(ContractError):
        verify_gpu_termination(value,campaign_id="other",operation_id="run",run_id="run",evaluation_id=None)
    path = gpu._GLOBAL_ROOT / "jobs" / job.ticket / "termination.json"
    path.rename(path.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError):
        verify_gpu_termination(value,campaign_id="campaign",operation_id="run",run_id="run",evaluation_id=None)


def test_actual_cpu_fixture_cannot_mint_complete_training_source_receipt(tmp_path):
    from tests.hpo_lineage_support import initial_hpo_chain

    loaded = loaded_for(tmp_path)
    parent = Path(loaded.config.storage.parent_root) / "unqualified"
    parent.parent.mkdir()
    initial_hpo_chain(parent)
    with pytest.raises(ContractError):
        inspect_training_parent(parent,loaded,canonical_control(loaded.config),training_seed=17,
            run_id="fixture",deadline=float("inf"))


def test_training_wrapper_invokes_actual_two_argument_collector_abi_and_same_ledger(tmp_path,monkeypatch):
    """Exercise the wrapper itself; the controlled trainer actually invokes it."""
    import gf_cli.card_instance as core
    import gf_cli.hpo as cli
    import gf_training.hpo_ppo as runtime
    from gf_cli.hpo_campaign import CampaignStudy, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_simulator_client.structured_resources import ReservationLedger

    loaded=loaded_for(tmp_path)
    monkeypatch.setattr(gpu,"_GLOBAL_ROOT",tmp_path / "global")
    observed=[]
    plan=object()
    result=object()
    async def actual_collect(loaded_run,received,supplied):
        assert received is plan
        observed.append(supplied)
        return result
    async def invoke_callback(loaded_run,collector,directory,*,gpu_job,mode):
        with pytest.raises(ContractError):
            await collector(plan,ReservationLedger())
        assert await collector(plan,gpu_job.ledger) is result
        assert observed == [gpu_job.ledger]
        return runtime.HpoRunResult({'status':'incomplete','used_decisions':0},gpu_job.confirm_termination())
    monkeypatch.setattr(core,"collect_hpo_ppo",actual_collect)
    monkeypatch.setattr(runtime,"train_hpo_ppo",invoke_callback)
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=8*1024**2) as store:
        campaign=initialize_campaign(store,loaded,{'scope':'unqualified callback ABI unit'})
        study=CampaignStudy(store,loaded,campaign)
        proposal=study.propose(0,slot=0,kind='control',requested=canonical_control(loaded.config))
        with pytest.raises(ContractError,match='incomplete'):
            asyncio.run(cli._train_seed(store,loaded,campaign,proposal,17))


def decoded_projection(value):
    from gf_evaluation.hpo import (
        GameIdentity,
        GameProjection,
        RunIdentity,
        SeedEvaluationProjection,
    )
    raw=value['run']
    run=RunIdentity(**{**raw,'games':tuple(GameIdentity(**row) for row in raw['games'])})
    games=tuple(GameProjection(**{**row,'identity':GameIdentity(**row['identity'])}) for row in value['games'])
    return SeedEvaluationProjection(run,value['used_decisions'],value['conditions_sha256'],games)


@pytest.mark.parametrize('repeat_game_id',[False,True])
def test_actual_evaluation_namespaces_scope_local_episode_ids_across_training_seeds(repeat_game_id):
    import copy

    from gf_cli.hpo_evaluation import _projection
    from gf_evaluation.hpo import EvaluationPlan, aggregate_evaluation
    projected=[]
    originals=[]
    for seed in (17,29):
        local_names=('eval-p000000-seat0','eval-p000000-seat1')
        evaluation_id=f'evaluation-{seed}'
        manifest={'run_id':evaluation_id,'linkage':{'evaluation_id':evaluation_id,
            'learner_seats':dict(zip(local_names,(1,2),strict=True))},
            'episodes':[{'episode_id':name,'game_id':f'game-{17 if repeat_game_id else seed}-{seat}',
                'seed':'common-game-seed','outcome':'draw','decisions':0} for name,seat in zip(local_names,(1,2),strict=True)]}
        before=copy.deepcopy(manifest)
        parent={'training_seed':seed,'run_id':f'parent-{seed}','inference_sha256':'a'*64,'used_decisions':100}
        projected.append(decoded_projection(_projection(manifest,parent,'b'*64,'c'*64)))
        assert manifest==before
        originals.append(before)
    plan=EvaluationPlan((17,29),('common-game-seed',),100,'c'*64,tuple(row.run for row in projected))
    if repeat_game_id:
        with pytest.raises(ContractError): aggregate_evaluation(plan,tuple(projected))
    else:
        assert aggregate_evaluation(plan,tuple(projected)).value==.5
        assert projected[0].games[0].identity.episode_id != projected[1].games[0].identity.episode_id
        assert [row['episode_id'] for row in originals[0]['episodes']]==['eval-p000000-seat0','eval-p000000-seat1']


@pytest.fixture(scope='module')
def actual_policy_parent(tmp_path_factory):
    """Real CPU saves/updates/labels, deliberately not production Source."""
    from tests.hpo_lineage_support import advanced_hpo_chain
    directory=tmp_path_factory.mktemp('actual-final-policy') / 'parent'
    state,records,learner=advanced_hpo_chain(directory,updates=2,budget=6)
    return directory,state,records,learner


@pytest.fixture
def final_policy_panel(tmp_path,monkeypatch,actual_policy_parent):
    import copy
    import time
    import uuid

    import gf_cli.hpo_evaluation as evaluation
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_config import campaign_config
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.card_instance_collection import (
        checked_linkage,
        hpo_evaluation_groups,
        hpo_evaluation_role,
    )
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_common.schedule import serialize_schedule
    from gf_models.arrays import sha256
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_retention import pin_hpo_evaluation, release_hpo_evaluation
    from gf_training.hpo_source import hpo_binding
    from gf_training.lineage import generation_path
    from tests.test_card_instance_learning import LAYOUT
    from tests.test_hpo_config import campaign_raw

    parent,state,records,learner=actual_policy_parent
    raw=campaign_raw()
    raw.update(decision_budget=6,learning_seeds=[17],model=asdict(learner.model),training=asdict(learner.training),
        space={'parameters':[],'coupled':[]},storage={'parent_root':str(parent.parent),'numeric_root':str(parent.parent),
            'optuna_path':str(tmp_path / 'component.sqlite3')})
    loaded=campaign_config(raw,origin=tmp_path)
    settings=effective_settings(learner.model,learner.training,{},layout=LAYOUT)
    proposal=Proposal(0,0,0,'a'*64,settings,'{}','{}')
    source={'schema_version':1,'kind':'hpo-parent-source-validation','run_id':state['run_id'],
        'run_path':str(parent),'training_seed':17,'config_sha256':state['config_sha256'],
        'state_sha256':sha256(parent / 'run.json'),'generation':2,'generation_sha256':sha256(generation_path(parent,2)),
        'generations_sha256':digest(records),'inference_sha256':records[2]['inference_sha256'],
        'used_decisions':6,'effective_settings_sha256':settings.identity}
    inspection=evaluation.ParentInspection(state,tuple(records),source)
    monkeypatch.setattr(evaluation,'inspect_training_parent',lambda *args,**kwargs:inspection)
    campaign={'campaign_id':'unqualified-binding-component','conditions_sha256':'c'*64,'conditions':{
        'producer_identity':{'code':state['identity']['code']},'integration_identity':state['identity']['integration'],
        'inference_binding':state['execution']['inference']['binding'],
        'cuda_visible_devices':state['execution']['inference']['cuda_visible_devices']}}
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path / 'global')
    with gpu.admit_gpu_job(gpu.JobContext(campaign['campaign_id'],state['run_id'],'training',state['run_id'],None),
            global_cuda_bytes=1024,max_workers=2) as job:
        training_term=clone(asdict(job.confirm_termination()))
    evaluation_id=str(uuid.uuid4())
    with CampaignStore(tmp_path / 'campaign',mode='fresh',max_owned_bytes=8*1024**2) as store:
        folder='proposals/000000/seeds/17'
        source_sha=store.put(folder+'/source-validation.json',source)
        term_sha=store.put(folder+'/training-gpu-termination.json',training_term)
        training={'schema_version':1,'kind':'hpo-seed-training','proposal_sha256':proposal.sha256,
            **{key:source[key] for key in ('training_seed','run_path','run_id','config_sha256','state_sha256',
                'generation','generation_sha256','inference_sha256','used_decisions')},
            'source_receipt_sha256':source_sha,'gpu_termination_sha256':term_sha}
        train_sha=store.put(folder+'/training.json',training)
        with gpu.admit_gpu_job(gpu.JobContext(campaign['campaign_id'],evaluation_id,'evaluation',state['run_id'],evaluation_id),
                global_cuda_bytes=1024,max_workers=2) as job:
            deadline=time.monotonic()+30
            acquired=pin_hpo_evaluation(parent,job.ticket,2,records[2]['inference_sha256'],max_owned_bytes=1024**3,deadline=deadline)
            released=release_hpo_evaluation(parent,acquired.binding,max_owned_bytes=1024**3,deadline=deadline)
            evaluation_term=clone(asdict(job.confirm_termination()))
            ticket=job.ticket
        gpu_sha=store.put(folder+'/evaluation-gpu-termination.json',evaluation_term)
        final_label,final_binding=hpo_binding(parent,state,records[2],2)
        expected={'run_id':state['run_id'],'run_path':str(parent),'training_seed':17,'generation':2,
            'generation_sha256':sha256(generation_path(parent,2)),'inference_sha256':records[2]['inference_sha256'],
            'source_receipt_sha256':source_sha,'policy_label':final_label}
        def publish(reference,policy_binding):
            registry={reference['policy_label']:asdict(policy_binding),
                'heuristic-card-instance-v2':asdict(RuntimePolicyBinding('heuristic'))}
            # For single-field SHA mutations, the structural registry echoes the
            # claimed digest; the real parent/expected label remains untouched.
            registry[reference['policy_label']]['policy_options']['expected_manifest_digest']=reference['inference_sha256']
            link={'schema_version':1,'kind':'hpo-evaluation-linkage','campaign_id':campaign['campaign_id'],
                'evaluation_id':evaluation_id,'gpu_ticket':ticket,'purpose':'objective','conditions_sha256':'c'*64,
                'left_parent':reference,'right_parent':None,'game_seeds':list(loaded.config.evaluation_game_seeds),
                'learner_seats':{},'schedule_digest':'0'*64,'policy_registry_digest':digest(registry)}
            schedule=clone(serialize_schedule(hpo_evaluation_groups(link),4))
            link.update(learner_seats={spec['episode_id']:spec['learner_viewer'] for group in schedule for spec in group['episodes']},
                schedule_digest=digest(schedule))
            checked_linkage(link,role=hpo_evaluation_role(2),schedule=schedule,registry=registry)
            manifest={'run_id':evaluation_id,'linkage':link,'config_hash':'d'*64,'policy_bindings':registry,
                'episodes':[{'episode_id':spec['episode_id'],'game_id':f'{evaluation_id}-{seat}',
                    'seed':group['seed'],'outcome':'draw','decisions':0} for group in schedule
                    for seat,spec in enumerate(group['episodes'])]}
            proof={'manifest_sha256':'e'*64,'source_set_sha256':'f'*64}
            # Isolate only the originals I/O boundary. Actual hpo_binding and
            # saved model/generation identities are not mocked or relabeled.
            monkeypatch.setattr(evaluation,'verify_saved_evaluation',lambda *args,**kwargs:(manifest,proof))
            store.put(folder+'/evaluation.intent.json',{'config_sha256':'d'*64})
            receipt={'schema_version':1,'kind':'hpo-seed-evaluation','training_receipt_sha256':train_sha,
                'linkage_sha256':digest(link),'manifest_sha256':'e'*64,'originals_digest':'f'*64,
                'projection':evaluation._projection(manifest,training,source_sha,'c'*64),'conditions_sha256':'c'*64,
                'pin_acquire_sha256':acquired.receipt_sha256,'pin_release_sha256':released.receipt_sha256,
                'gpu_termination_sha256':gpu_sha,'directory':str(tmp_path / 'originals'),'linkage':link,'config_sha256':'d'*64}
            store.put(folder+'/evaluation.json',receipt)
        yield store,loaded,campaign,proposal,copy.deepcopy(expected),final_binding,publish,actual_policy_parent


@pytest.mark.parametrize('field',['run_id','run_path','training_seed','generation','generation_sha256',
    'inference_sha256','source_receipt_sha256','policy_label','earlier-checkpoint'])
def test_seed_verifier_refuses_every_nonfinal_actual_policy_reference(final_policy_panel,field):
    from gf_cli.hpo_evaluation import verify_seed_evaluation
    from gf_models.arrays import sha256
    from gf_training.hpo_source import hpo_binding
    from gf_training.lineage import generation_path
    store,loaded,campaign,proposal,reference,binding,publish,parent_data=final_policy_panel
    parent,state,records,_=parent_data
    if field=='earlier-checkpoint':
        reference.update(generation=1,generation_sha256=sha256(generation_path(parent,1)),inference_sha256=records[1]['inference_sha256'])
        label,binding=hpo_binding(parent,state,records[1],1)
        reference['policy_label']=label
    elif field=='run_id': reference[field]='other-run'
    elif field=='run_path': reference[field]=str(parent.parent / 'other-parent')
    elif field=='training_seed': reference[field]=29
    elif field=='generation': reference[field]=1
    elif field=='policy_label': reference[field]='other-policy'
    else: reference[field]='0'*64
    publish(reference,binding)
    with pytest.raises(ContractError):
        verify_seed_evaluation(store,loaded,campaign,proposal,17,deadline=float('inf'))


def test_seed_verifier_accepts_actual_derived_final_binding_without_relabeling_cpu(final_policy_panel):
    from gf_cli.hpo_evaluation import verify_seed_evaluation
    store,loaded,campaign,proposal,reference,binding,publish,parent_data=final_policy_panel
    assert binding.policy_options['inference']['device']=='cpu'
    publish(reference,binding)
    projection,_,_=verify_seed_evaluation(store,loaded,campaign,proposal,17,deadline=float('inf'))
    assert projection.run.checkpoint_sha256==reference['inference_sha256']
    assert binding.policy_options['checkpoint']==str(parent_data[0] / 'artifacts/000002')
    assert parent_data[1]['device']=='cpu'


def test_seed_verifier_checks_actual_original_registry_against_derived_policy_options(final_policy_panel):
    from gf_cli.hpo_evaluation import verify_seed_evaluation
    from gf_common.domain import RuntimePolicyBinding
    store,loaded,campaign,proposal,reference,binding,publish,_=final_policy_panel
    options=clone(binding.policy_options)
    options['model']['resources']['workspace_bytes']+=1024
    publish(reference,RuntimePolicyBinding(binding.model_id,options))
    with pytest.raises(ContractError,match='final-D binding'):
        verify_seed_evaluation(store,loaded,campaign,proposal,17,deadline=float('inf'))
