"""Caller-owned HPO capacity; CPU ownership fixtures grant no learning authority."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, digest
from tests.hpo_resident_support import retirement_fact
from tests.test_hpo_config import campaign_raw
from tests.test_hpo_resident_custody import global_root as global_root  # noqa: PLC0414

RUNTIME = {'mode': 'resident-worker-v1', 'cache_entries': 6, 'cache_weight_bytes': 134217728}


def expanded_retirement(count=8):
    value = retirement_fact()
    template = value['owners'][0]
    value['owners'] = []
    for index in range(count):
        row = deepcopy(template)
        row.update(worker_key=index+1, pid=1000+index, viewer=1+index%2)
        for name in ('end_ack', 'park_ack'):
            row[name].update(worker_key=row['worker_key'], pid=row['pid'], viewer=row['viewer'])
            row[name+'_sha256'] = digest(row[name])
        value['owners'].append(row)
    return value


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('games,workers', [(4,8),(8,16)])
def test_expanded_materialization_lineage_and_all_managed_resource_phases(tmp_path, resident, games, workers):
    from gf_cli.hpo_config import campaign_config, joint_resource_admission, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_lineage import hpo_spec
    raw = campaign_raw()
    if resident:
        raw.update(schema_version=3, runtime=RUNTIME)
    before = campaign_config(raw, origin=tmp_path)
    after = campaign_config({**raw, 'operating': {'game_concurrency': games, 'max_participant_workers': workers,
        'managed_ram_bytes': 128*1024**3, 'managed_cuda_bytes': 128*1024**3}}, origin=tmp_path)
    small = joint_resource_admission(before.config, layout=layout)
    large = joint_resource_admission(after.config, layout=layout)
    assert large['workers'] == workers
    assert large['worker_bytes'] == (games//2)*small['worker_bytes']
    if resident:
        assert large['cache_weight_bytes'] == (games//2)*small['cache_weight_bytes']
        assert large['begin_staging_bytes'] == (games//2)*small['begin_staging_bytes']
        assert large['staging_bytes'] == small['staging_bytes']
        assert all(large['ram_phases'][phase] > small['ram_phases'][phase] for phase in small['ram_phases'])
    for resource, limit in (('managed_ram_bytes', 'peak_ram_bytes'), ('managed_cuda_bytes', 'peak_cuda_bytes')):
        refused = replace(after.config, operating=replace(after.config.operating, **{resource: large[limit]-1}))
        with pytest.raises(ContractError):
            joint_resource_admission(refused, layout=layout)
    settings = effective_settings(after.config.model, after.config.training, {}, layout=layout)
    run = materialize_hpo_run(after, settings, learning_seed=17, run_id='00000000-0000-0000-0000-000000000017')
    assert run.config.run.concurrency == games and run.config.run.max_participant_workers == workers
    state = {'effective_config': asdict(run.config), 'run_variant': 'hpo-ppo-resident-v1' if resident else 'hpo-ppo-v1'}
    state['hpo_spec_sha256'] = digest(run.config.hpo_spec)
    assert hpo_spec(state)['operating']['max_participant_workers'] == workers
    assert before.config.operating.game_concurrency == 2 and before.config.operating.max_participant_workers == 4
    assert after.config.operating.trial_concurrency == 1


@pytest.mark.parametrize('games,workers', [(0,8),(True,8),(4,7),(4,8.0),(4,True),(4,2**63),(2**62,2**63-1)])
def test_hpo_rejects_invalid_counts_before_io(tmp_path, games, workers):
    from gf_cli.hpo_config import campaign_config
    with pytest.raises(ContractError):
        campaign_config({**campaign_raw(), 'operating': {'game_concurrency': games, 'max_participant_workers': workers}}, origin=tmp_path)
    assert not list(tmp_path.iterdir())


def test_positive_signed63_capacity_domain_has_no_machine_ceiling():
    from gf_cli.hpo_config import _operating
    config = _operating({'game_concurrency': (2**63-1)//2, 'max_participant_workers': 2**63-1})
    assert config.max_participant_workers == 2**63-1


def test_retirement_requires_original_capacity_and_preserves_ack_and_viewer_checks():
    from gf_common.hpo_resident import checked_retirement
    value = expanded_retirement()
    assert checked_retirement(value, max_workers=8) == value
    with pytest.raises(ContractError): checked_retirement(value)
    for capacity in (7, True, 0, 2**63):
        with pytest.raises(ContractError): checked_retirement(value, max_workers=capacity)
    assert checked_retirement(value, max_workers=9) == value
    for mutation in ('duplicate', 'viewer', 'digest', 'epoch', 'dirty'):
        broken = deepcopy(value)
        if mutation == 'duplicate': broken['owners'][-1] = deepcopy(broken['owners'][0])
        elif mutation == 'viewer':
            row = broken['owners'][-1]
            row['viewer'] = 1
            for name in ('end_ack','park_ack'):
                row[name]['viewer'] = 1
                row[name+'_sha256'] = digest(row[name])
        else:
            row = broken['owners'][-1]
            if mutation == 'digest': row['end_ack_sha256'] = 'f'*64
            elif mutation == 'epoch': row['park_ack']['request_epoch'] = row['end_ack']['request_epoch']
            else: row['end_ack']['private_state_empty'] = False
            if mutation != 'digest':
                for name in ('end_ack','park_ack'): row[name+'_sha256'] = digest(row[name])
        with pytest.raises(ContractError): checked_retirement(broken, max_workers=8)


@pytest.mark.parametrize('capacity', [8, 9])
def test_eight_actual_resident_role_owners_end_park_and_reap(global_root, tmp_path, capacity):
    from gf_common.domain import RuntimePolicyBinding
    from gf_simulator_client.resident_pool import ResidentParticipantPool
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.collection import CollectionPlan
    from gf_training.hpo_gpu_lease import admit_gpu_job
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_hpo_gpu_lease import context
    from tests.test_resident_episode import ACTIONS
    with pytest.raises(ContractError): ReservationLedger(max_workers=capacity)
    async def scenario():
        with admit_gpu_job(context(), global_cuda_bytes=1024, max_workers=capacity) as job:
            custody = ResidentPoolCustody(job, run_id='run-a', pool_id='b'*32)
            pool = ResidentParticipantPool(custody, resident_bytes=16, deadline=1e12)
            plan = CollectionPlan((), {'random': RuntimePolicyBinding('random')}, tmp_path,
                {'parent_run_id':'run-a', 'collection_generation':0, 'batch_id':'batch-000000'},1e12,1e12)
            pool.admit(plan)
            logicals = [pool.worker(viewer, 'random', 17+index, ACTIONS, ledger=job.ledger)
                for index in range(4) for viewer in (1,2)]
            try:
                await asyncio.gather(*(worker.initialize() for worker in logicals))
                assert len({worker.process.pid for worker in logicals}) == 8
                assert len(job.ledger.workers) == len(custody._owners) == 8
                assert job.ledger.cuda_bytes == 128
                for viewer in (1,2):
                    with pytest.raises(ContractError, match='partition'):
                        pool.worker(viewer, 'random', 99, ACTIONS, ledger=job.ledger)
                for worker in logicals: worker.mark_completed()
                await asyncio.gather(*(worker.close() for worker in logicals))
                authority = pool.retire_collection()
                authority.require(job.ledger)
                assert len(authority.evidence['owners']) == 8
                assert all(row['end_ack']['command']=='END' and row['park_ack']['command']=='PARK'
                    for row in authority.evidence['owners'])
                with pytest.raises(ContractError): job.confirm_termination()
            finally:
                await pool.close()
            assert all(worker.process.poll() is not None for worker in logicals)
            assert not job.ledger.workers and job.ledger.cuda_bytes == 0
            receipt = job.confirm_termination()
            assert len(receipt.resources) == 8 and all(row.outcome=='reaped' for row in receipt.resources)
    asyncio.run(scenario())


def collection_manifest():
    from gf_common.hpo import selection_context
    from gf_training.hpo_config import HPOTrainingConfig
    schedule = [{'seed':'seed', 'episodes':[{'episode_id':f'episode-{i}', 'learner_viewer':1+i%2} for i in range(8)]}]
    context = selection_context(decision_budget=8, used_before=0, learning_seed=17, batch=0)
    config = {'run':{'max_participant_workers':8}, 'training':{'seeds':[17]},
        'hpo_spec':{'schema_version':3, 'runtime':RUNTIME, 'training_recipe':asdict(HPOTrainingConfig(collection_games=8)),
            'operating':{'max_participant_workers':8}, 'decision_budget':8, 'selection_version':context['selection_version']}}
    link = {'parent_run_id':'run-a', 'parent_run_path':'/parent', 'batch_id':'batch-000000',
        'collection_generation':0, 'selection_context':context,
        'learner_seats':{f'episode-{i}':1+i%2 for i in range(8)},
        'schedule_digest':digest(schedule), 'policy_registry_digest':digest({})}
    return {'schema_version':7, 'model_version':'4', 'collection_role':'hpo-ppo-batch',
        'config':config, 'config_hash':digest(config), 'linkage':link, 'schedule':schedule,
        'policy_bindings':{}, 'logical_retirement':expanded_retirement()}


def test_collection_reader_uses_hashed_original_worker_capacity_and_rejects_projection_mismatch():
    from gf_training.hpo_collection import check_hpo_collection
    manifest = collection_manifest()
    check_hpo_collection(manifest)
    for run, spec in ((4,8),(8,4),(True,8),(8,2**63)):
        broken = deepcopy(manifest)
        broken['config']['run']['max_participant_workers'] = run
        broken['config']['hpo_spec']['operating']['max_participant_workers'] = spec
        broken['config_hash'] = digest(broken['config'])
        with pytest.raises(ContractError): check_hpo_collection(broken)
    # Even an otherwise valid small receipt cannot hide a projection mismatch.
    manifest['logical_retirement'] = retirement_fact()
    manifest['config']['run']['max_participant_workers'] = 4
    manifest['config_hash'] = digest(manifest['config'])
    with pytest.raises(ContractError): check_hpo_collection(manifest)


def evaluation_metadata():
    from gf_common.card_instance_collection import HPO_EVALUATION_CORE_FIELDS
    from tests.test_hpo_evaluation_role import binding
    link, schedule, registry = binding()
    config = {'run':{'split':'validation','max_participant_workers':8},
        'hpo_spec':{'operating':{'max_participant_workers':8}}}
    runtime = {'schema_version':1, 'kind':'hpo-resident-evaluation-runtime', 'runtime':RUNTIME,
        'evaluation_id':link['evaluation_id'], 'job_ticket':link['gpu_ticket'],
        'linkage_sha256':digest(link), 'policy_registry_sha256':digest(registry)}
    retirement = expanded_retirement()
    retirement.update(run_id=link['evaluation_id'], collection_id=link['evaluation_id'],job_ticket=link['gpu_ticket'])
    for row in retirement['owners']:
        row['job_ticket'] = link['gpu_ticket']
        for name in ('end_ack','park_ack'):
            row[name].update(run_id=link['evaluation_id'],job_ticket=link['gpu_ticket'])
            row[name+'_sha256'] = digest(row[name])
    return {**dict.fromkeys(HPO_EVALUATION_CORE_FIELDS), 'schema_version':7, 'model_version':'4',
        'collection_role':'hpo-evaluation','source_eligibility':'not-authorized-by-collection-metadata',
        'config':config,'config_hash':digest(config),'journal_schema_version':6,'derived_schema_version':6,
        'participant_protocol_version':3,'candidate_codec':'candidate-columns-v2','linkage':link,
        'schedule':schedule,'policy_bindings':registry,'execution_runtime':runtime,'logical_retirement':retirement}


def test_evaluation_reader_uses_hashed_config_capacity_and_rejects_projection_mismatch():
    from gf_common.card_instance_collection import checked_hpo_evaluation_metadata
    value = evaluation_metadata()
    assert checked_hpo_evaluation_metadata(value) is value
    for run, spec in ((4,8),(8,4),(8,True),(8,2**63)):
        broken = deepcopy(value)
        broken['config']['run']['max_participant_workers'] = run
        broken['config']['hpo_spec']['operating']['max_participant_workers'] = spec
        broken['config_hash'] = digest(broken['config'])
        with pytest.raises(ContractError): checked_hpo_evaluation_metadata(broken)


@pytest.mark.parametrize('cancelled', [False, True])
def test_eight_charged_owners_keep_original_io_and_single_job_exclusion_after_failure(global_root, cancelled):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from gf_training.hpo_gpu_lease import admit_gpu_job
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_hpo_gpu_lease import context, spawn_worker
    from tests.test_hpo_resident_custody import ack, cleanup
    async def scenario():
        with admit_gpu_job(context(), global_cuda_bytes=1024, max_workers=8) as job:
            pool = ResidentPoolCustody(job, run_id='run-a', pool_id='b'*32)
            owned = []
            gate, started = threading.Event(), threading.Event()
            def blocking_io():
                started.set()
                gate.wait(timeout=10)
            with ThreadPoolExecutor(max_workers=1) as executor:
                task = completion = None
                try:
                    for index in range(8):
                        owner, worker, process = spawn_worker(job, script='import time;time.sleep(30)')
                        owned.append((owner,worker,process))
                        pool.register(worker,owner,viewer=1+index%2)
                        pool.begin_lease(worker,generation=0)
                        ack(pool,worker,'END'); ack(pool,worker,'PARK')
                    authority = pool.parked_authority(generation=0,collection_id='batch-000000')
                    authority.require(job.ledger)
                    completion = executor.submit(blocking_io)
                    async def join_io(): await asyncio.wrap_future(completion)
                    task = asyncio.create_task(join_io())
                    pool.track_io(owned[-1][1],task,completion)
                    while not started.is_set(): await asyncio.sleep(0)
                    if cancelled:
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError): await task
                    else:
                        with pytest.raises(RuntimeError,match='injected'):
                            pool.fail()
                            raise RuntimeError('injected failure above four owners')
                    assert len(job.ledger.workers)==8 and job.ledger.cuda_bytes==128
                    with pytest.raises(ContractError): pool.retire_io(owned[-1][1],task)
                    with pytest.raises(ContractError): pool.parked_authority(generation=0,collection_id='batch-000000')
                    with pytest.raises(ContractError): admit_gpu_job(context('other'),global_cuda_bytes=1024,max_workers=8)
                    pool.fail()
                    for _,worker,process in owned: cleanup(worker,process)
                    assert not job.ledger.workers and completion.running()
                    with pytest.raises(ContractError): job.confirm_termination()
                finally:
                    gate.set()
                    if completion is not None: completion.result(timeout=5)
                    if task is not None:
                        if not cancelled: await task
                        pool.retire_io(owned[-1][1],task)
                    for _,worker,process in owned:
                        if process.poll() is None: cleanup(worker,process)
            receipt = job.confirm_termination()
            assert len(receipt.resources)==8 and all(row.outcome=='reaped' for row in receipt.resources)
    asyncio.run(scenario())


def source_proof(tmp_path):
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_collection import selection_for_counts
    from gf_training.hpo_config import effective_settings
    campaign = campaign_config({**campaign_raw(), 'schema_version':3, 'runtime':RUNTIME,
        'operating':{'game_concurrency':4,'max_participant_workers':8,'managed_ram_bytes':128*1024**3,
            'managed_cuda_bytes':128*1024**3}}, origin=tmp_path)
    effective = effective_settings(campaign.config.model,campaign.config.training,{},layout=layout)
    run = materialize_hpo_run(campaign,effective,learning_seed=17,run_id='00000000-0000-0000-0000-000000000017')
    classification = {key:'structural-fixture' for key in ('action_contract','purpose','synthetic_capacity','rule_signature','layout_signature')}
    state = {'run_id':'run-a','run_variant':'hpo-ppo-resident-v1','seed':17,
        'effective_config':asdict(run.config),'hpo_spec_sha256':digest(run.config.hpo_spec),**classification}
    selection = selection_for_counts((1,)*16,learning_seed=17,batch=0,decision_budget=100,used_before=0)
    proof = {'schema_version':4,'kind':'card-instance-hpo-original-source-proof','qualified':False,
        'parent_qualification':'not_performed','manifest_sha256':'c'*64,'source_set_sha256':digest({}),
        'files':{},'games':16,'accepted_decisions':16,'closure_counts':{'natural':16},'missing_replays':[],
        'selection':selection,'expected_optimizer_step_delta':1,'logical_retirement':expanded_retirement(),**classification}
    return state, proof


def test_lineage_proof_uses_parent_spec_capacity_without_reinterpreting_receipt(tmp_path):
    from gf_training.hpo_lineage import _proof
    state, proof = source_proof(tmp_path)
    assert _proof(proof,state,{'used_decisions':0},0,1) == proof['selection']
    state['effective_config']['hpo_spec']['operating'].update(game_concurrency=2,max_participant_workers=4)
    state['hpo_spec_sha256'] = digest(state['effective_config']['hpo_spec'])
    with pytest.raises(ContractError): _proof(proof,state,{'used_decisions':0},0,1)


def test_native_payload_decoder_requires_explicit_expected_capacity_and_preserves_full_proof(tmp_path):
    from gf_common.domain import canonical
    from gf_training.native_structured_session import _payload_views
    _, proof = source_proof(tmp_path)
    proof['kind'] = 'card-instance-hpo-packed-original-source-proof'
    proof['storage'] = {'format':'sqlite-group-pack-v1','locator_sha256':'a'*64,'descriptor_sha256':'a'*64,
        'database_sha256':'a'*64,'seal_id':'structural-fixture','logical_sha256':'a'*64}
    custody = {'recipe':'packed-original-custody-v1',
        'files':[{'role':role,'path':'/'+role,'sha256':'a'*64,'identity':[1,2,3,4,5,6]}
            for role in ('database','descriptor','locator')],
        'directories':[{'path':'/a','device':1,'inode':2},{'path':'/b','device':1,'inode':3}]}
    proof['custody'] = {**custody,'fingerprint':digest(custody)}
    metadata = {'schema_version':3,'kind':'card-instance-hpo-recorded-cohort-payload','collection_index':0,
        'metadata':{},'arrays':[],'source_proof':proof,
        'correspondence':{'recipe':'card-instance-hpo-packed-source-correspondence-v1'}}
    payload = canonical(metadata).encode(), (), (), ()
    actual, arrays = _payload_views(payload,game=True,recorded=True,packed=True,hpo=True,max_workers=8)
    assert actual == metadata and arrays == {} and actual['source_proof'] == proof
    with pytest.raises(ContractError): _payload_views(payload,game=True,recorded=True,packed=True,hpo=True)
    with pytest.raises(ContractError): _payload_views(payload,game=True,recorded=True,packed=True,hpo=True,max_workers=4)


def test_hpo_game_admission_accepts_eight_roles_while_ordinary_interfaces_keep_their_caps():
    from pathlib import Path

    from gf_cli.config import load_config
    from gf_common.domain import RuntimePolicyBinding
    from gf_models.structured_runtime import admit_game_workers_v3, admit_hpo_game_workers
    from tests.test_resident_episode import ACTIONS, RULES
    roles = RuntimePolicyBinding('random'), RuntimePolicyBinding('heuristic')
    admission = admit_hpo_game_workers(ACTIONS,roles,rules=RULES,game_concurrency=4,
        global_cuda_bytes=1024,max_workers=8)
    assert admission['workers']==8 and admission['game_concurrency']==4 and admission['max_workers']==8
    with pytest.raises(ContractError):
        admit_hpo_game_workers(ACTIONS,roles,rules=RULES,game_concurrency=4,global_cuda_bytes=1024,max_workers=7)
    with pytest.raises(ContractError):
        admit_game_workers_v3(ACTIONS,roles,rules=RULES,game_concurrency=4,max_workers=8)
    base = Path(__file__).parents[1]/'configs/experiments/card-instance-mlp-ppo-acceptance.toml'
    for override in ({'concurrency':4},{'max_participant_workers':8}):
        with pytest.raises(ContractError): load_config(base,{'run':override})


def test_custody_independently_enforces_half_of_original_odd_worker_capacity(global_root):
    from gf_training.hpo_gpu_lease import admit_gpu_job
    from gf_training.hpo_resident import ResidentPoolCustody
    from tests.test_hpo_gpu_lease import context, spawn_worker
    from tests.test_hpo_resident_custody import cleanup
    with admit_gpu_job(context(),global_cuda_bytes=1024,max_workers=9) as job:
        custody = ResidentPoolCustody(job,run_id='run-a',pool_id='b'*32)
        owned = []
        try:
            for _ in range(5):
                owner, worker, process = spawn_worker(job,script='import time;time.sleep(30)')
                owned.append((owner,worker,process))
            for owner,worker,_ in owned[:4]: custody.register(worker,owner,viewer=1)
            with pytest.raises(ContractError,match='partition'):
                custody.register(owned[-1][1],owned[-1][0],viewer=1)
            assert len(job.ledger.workers)==5 and len(custody._owners)==4
        finally:
            for _,worker,process in owned: cleanup(worker,process)
        assert len(job.confirm_termination().resources)==5
