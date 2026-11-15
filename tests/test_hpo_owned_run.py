"""Fresh producer-owned composition, distinct from qualified Simulator runs."""
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError, digest


def test_owned_spec4_explicit_encoder_and_materialization(tmp_path):
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_training.hpo_config import effective_settings
    from gf_training.hpo_lineage import hpo_spec
    from tests.test_hpo_config import campaign_raw
    raw = campaign_raw()
    raw.update(schema_version=4, origin='producer-owned-v1', encoder='canonical-python',
        runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':128*1024**2})
    loaded = campaign_config(raw, origin=tmp_path)
    config = loaded.config
    assert config.origin == 'producer-owned-v1' and config.encoder == 'canonical-python'
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    settings = effective_settings(config.model,config.training,{},layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17,
        run_id='00000000-0000-0000-0000-000000000017')
    assert run.config.preprocessing.backend == 'python'
    spec = run.config.hpo_spec
    assert spec['schema_version'] == 4 and spec['origin']==config.origin and spec['encoder']==config.encoder
    assert hpo_spec({'effective_config':asdict(run.config),'run_variant':'hpo-ppo-owned-v1',
        'hpo_spec_sha256':digest(spec)}) == spec
    for key in ('origin','encoder','runtime'):
        with pytest.raises(ContractError):
            campaign_config({name:value for name,value in raw.items() if name != key}, origin=tmp_path)
    with pytest.raises(ContractError):
        campaign_config({**raw,'encoder':'rust'},origin=tmp_path)


def test_live_registry_requires_actual_save_and_current_lock(tmp_path):
    import asyncio
    import time

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.hpo_owned_lineage import OwnedRunRegistry
    from gf_training.hpo_owned_process import HpoOwnedLearningProcess
    from gf_training.lineage import RunOwnership, run_lock
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner, account, ledger = learner_for(), budget(), ReservationLedger()
    async def run(owner):
        state={'run_id':'fixture-unqualified','status':'running','updates':0,'generation_sha256':None,
            'used_decisions':0,'collected_decisions':0,'accepted_decisions':0,'optimizer_steps':0,
            'model_id':'mlp','effective_config':{'model':asdict(learner.model),'actions':asdict(ACTIONS)},
            'execution':{'inference':{'binding':asdict(learner.binding),'cuda_visible_devices':''}}}
        registry=OwnedRunRegistry(owner,state,budget=account,parent_budget_bytes=256*1024**2)
        child=HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+30,budget=account)
        registry.attach_process(child)
        import copy
        import pickle
        with pytest.raises(TypeError): copy.copy(registry)
        with pytest.raises(TypeError): pickle.dumps(registry)
        await child.start()
        try:
            snapshot=await child.initialize_owned(17,tmp_path/'artifacts/000000',deadline=time.monotonic()+20)
            controls={'schedule_opponent_rng':{},'snapshot_pool':[0],'collection_position':0}
            record=registry.commit_initial(snapshot,controls=controls,deadline=time.monotonic()+20)
            assert registry.current is snapshot and record['inference_sha256']==snapshot.files['inference.json']
            assert registry.binding(0)[1].policy_options['expected_manifest_digest']==record['inference_sha256']
            with pytest.raises(ContractError): registry.commit_initial(snapshot,controls=controls,deadline=time.monotonic()+20)
            with pytest.raises(ContractError):
                OwnedRunRegistry(RunOwnership(owner.directory,owner.token),state,budget=account,parent_budget_bytes=256*1024**2)
        finally:
            await child.close(time.monotonic()+5)
            registry.close()
    with run_lock(tmp_path) as owner:
        for name in ('artifacts','generations','updates'): (tmp_path/name).mkdir()
        asyncio.run(run(owner))


def test_public_finalized_flags_cannot_create_owned_archive_authority(tmp_path):
    from gf_training.collection import FinalizedCollection
    from gf_training.hpo_owned_collection import OwnedFinalizedCollection
    public=FinalizedCollection(tmp_path,{'healthy':True,'comparison_valid':True})
    with pytest.raises(ContractError):
        OwnedFinalizedCollection(object(),public,None,None,None)


def test_owned_inference_provider_rejects_public_metadata():
    from gf_models.hpo_owned_artifacts import OwnedGameArtifacts
    with pytest.raises(ContractError):
        OwnedGameArtifacts({'fake':(object(),object(),{})},budget=object())


def test_owned_archive_seal_requires_actual_capture_writer_transaction():
    from gf_training.group_pack_writer import GroupPackWriter
    writer=object.__new__(GroupPackWriter)
    with pytest.raises(ContractError): writer._seal_producer(object())


def test_capture_finalization_uses_decoded_replay_closure_once(monkeypatch):
    from gf_simulator_client.card_instance_environment import CardInstanceSimulatorEnvironment
    environment=object.__new__(CardInstanceSimulatorEnvironment)
    environment.capture=object()
    def forbidden(*args,**kwargs): raise AssertionError('historical Replay traversal')
    monkeypatch.setattr('gf_simulator_client.card_instance_environment.replay_closure',forbidden)
    monkeypatch.setattr('gf_simulator_client.card_instance_environment.reconcile_replay',forbidden)
    closure={'type':'aborted','reason':'turn_limit'}
    replay={'episode_end':closure}
    environment._reconcile_replay(replay,1,[],None)
    assert environment._replay_closure(replay,'game',1) is closure
    assert environment._replay_was_reconciled() is False


@pytest.mark.parametrize('kind,case',[('mlp','healthy'),('gru','healthy'),('transformer','healthy'),
    ('mlp','production_collector'),('mlp','empty'),('mlp','archive_failure'),('mlp','save_failure'),('mlp','cancel'),('mlp','startup_server'),('mlp','startup_registry'),
    ('mlp','startup_child'),('mlp','startup_runtime'),('mlp','startup_cancel'),('mlp','startup_clock'),('mlp','startup_admission'),('mlp','parent_overlap')])
def test_fresh_two_updates_actual_capture_writer_wire7_and_parked_pool(tmp_path,monkeypatch,kind,case):
    """Unqualified controlled frames; actual actors/FDs/writer/locks/Adam, no Simulator run."""
    import asyncio
    from dataclasses import replace

    from gf_cli.config import LoadedConfig
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.domain import clone
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.hpo_config import HPOTrainingConfig, effective_settings
    from gf_training.hpo_owned_run import _OwnedHpoOperations
    from gf_training.ppo_run import _run_ppo
    from tests.test_hpo_config import campaign_raw
    from tests.test_hpo_numeric import learner_for
    raw=campaign_raw()
    raw.update(schema_version=4,model_id=kind,origin='producer-owned-v1',encoder='canonical-python',decision_budget=32,
        base_experiment=str(__import__('pathlib').Path(__file__).parents[1]/f'configs/experiments/card-instance-{kind}-ppo-acceptance.toml'),
        runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':128*1024**2},
        storage={'parent_root':str(tmp_path/'parents'),'numeric_root':str(tmp_path/'parents'),'originals_root':str(tmp_path/'originals')})
    raw['space']={'parameters':[],'coupled':[]}
    if case=='parent_overlap': raw['operating']={'parent_budget_bytes':40_900_000}
    loaded=campaign_config(raw,origin=tmp_path)
    tiny=learner_for(kind)
    settings=effective_settings(tiny.model,replace(HPOTrainingConfig(),epochs=1),{},layout=LAYOUT)
    run_id='00000000-0000-0000-0000-000000000017'
    runtime=materialize_hpo_run(loaded,settings,learning_seed=17,run_id=run_id)
    from tests.test_card_instance_learning import ACTIONS
    config=replace(runtime.config,actions=ACTIONS,model=replace(runtime.config.model,inference=tiny.binding,cuda_visible_devices=''),
        training=replace(runtime.config.training,device='cpu'))
    value=clone(asdict(config))
    runtime=LoadedConfig(config,value,digest(value),runtime.inputs)
    monkeypatch.setattr(gpu,'_GLOBAL_ROOT',tmp_path/'global')
    monkeypatch.setattr(_OwnedHpoOperations,'preflight',lambda self,config,deadline:self.validate_config(config))
    monkeypatch.setattr(_OwnedHpoOperations,'integration_identity',lambda self,config,deadline:{'fixture_origin':'unqualified-controlled-frames',
        'server_config_sha256':__import__('gf_models.arrays',fromlist=['sha256']).sha256(__import__('pathlib').Path(config.server.config))})
    if case.startswith('startup_'):
        import contextlib
        if case=='startup_admission':
            pass  # Actual job learner admission is denied below, before creator request.
        elif case=='startup_clock':
            def failed_clock(*args,**kwargs): raise ContractError('fixture startup clock')
            monkeypatch.setattr('gf_training.ppo_run._validate_clock_anchor',failed_clock)
        elif case=='startup_server':
            from gf_models.structured_sealed import SealedScan
            @contextlib.contextmanager
            def failed_server(*args,**kwargs):
                raise ContractError('fixture startup server')
                yield
            monkeypatch.setattr(SealedScan,'file',failed_server)
        else:
            if case=='startup_registry':
                from gf_training.hpo_owned_lineage import OwnedRunRegistry as startup_class
            elif case=='startup_child':
                from gf_training.hpo_owned_process import HpoOwnedLearningProcess as startup_class
            else:
                from gf_training.hpo_resident_runtime import (
                    ResidentTrainingRuntime as startup_class,
                )
            construct=startup_class.__init__
            def failed_construct(self,*args,**kwargs):
                construct(self,*args,**kwargs)
                if case=='startup_cancel': raise asyncio.CancelledError('fixture startup cancellation')
                raise ContractError('fixture '+case)
            monkeypatch.setattr(startup_class,'__init__',failed_construct)
    if case in ('production_collector','parent_overlap'):
        install_owned_fixture_runtime(monkeypatch)
        def unqualified_gate(state):
            assert state['device']=='cpu' and state['run_id']==run_id
        monkeypatch.setattr('gf_training.hpo_source.require_hpo_production',unqualified_gate)
    if case=='archive_failure':
        from gf_training.group_pack_writer import GroupPackWriter
        submit=GroupPackWriter.submit
        def failed_archive(self,*args,**kwargs):
            submit(self,*args,**kwargs)
            raise OSError('fixture archive failure after actual submit')
        monkeypatch.setattr(GroupPackWriter,'submit',failed_archive)
    from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
    synchronize=ResidentTrainingRuntime.after_commit
    async def sync_and_evict(self,generation,record):
        await synchronize(self,generation,record)
        if case not in ('healthy','production_collector') or generation!=1: return
        previous=self.owned_registry.snapshots[0].actor_capability
        with pytest.raises(ContractError): self.pool.unbind_owned_snapshot(previous)
        for owner in self.pool.owners:
            entries=self.entries.get(owner.lease.key,{})
            for identity,(_,receipt) in tuple(entries.items()):
                if receipt is None or receipt.binding['generation']!=0: continue
                assert identity!=owner.protected_identity
                await owner.control('EVICT',checkpoint=receipt.binding['inference_sha256'],cache_identity=identity,timeout=10.)
                self.release_cache(owner,identity)
                del entries[identity]
                await owner.control('PARK',checkpoint=record['inference_sha256'],timeout=10.)
        self.pool.unbind_owned_snapshot(previous)
        assert previous.require() is self.owned_registry.snapshots[0]
        self.pool.bind_owned_snapshot(previous)
    monkeypatch.setattr(ResidentTrainingRuntime,'after_commit',sync_and_evict)
    calls=[]
    guards=[]
    guard=_OwnedHpoOperations.guard_collection
    def counted_guard(self,plan,ledger,**options):
        guards.append(plan.linkage['collection_generation'])
        return guard(self,plan,ledger,**options)
    monkeypatch.setattr(_OwnedHpoOperations,'guard_collection',counted_guard)
    update_owned=_OwnedHpoOperations.update_owned
    async def update_once(self,transaction,output,**options):
        result=await update_owned(self,transaction,output,**options)
        with pytest.raises(ContractError): await transaction.dispatch(output,**options)
        assert transaction.state=='SAVED'
        return result
    monkeypatch.setattr(_OwnedHpoOperations,'update_owned',update_once)
    def forbidden(*args,**kwargs): raise AssertionError('forbidden historical normal-route audit')
    forbidden_targets=('gf_training.hpo_lineage.validate_hpo_chain','gf_training.hpo_source.validate_hpo_parent',
        'gf_training.hpo_source.open_hpo_batch','gf_training.card_instance_source.game_batch_source',
        'gf_training.group_pack_source.inspect_packed','gf_training.group_pack_writer.inspect_packed',
        'gf_training.group_pack_publication.inspect_packed','gf_models.sharded_arrays.inspect_bundle',
        'gf_training.hpo_rollout.read_hpo_rollout','gf_models.structured_sealed.inspect_retained_game_artifact_v4',
        'gf_training.card_instance_source.validate_game_parent',
        'gf_simulator_client.card_instance_environment.reconcile_replay',
        'gf_simulator_client.card_instance_environment.replay_closure','gf_cli.card_instance._validate_recorded_episodes')
    import importlib
    for name in forbidden_targets: importlib.import_module(name.rsplit('.',1)[0])
    for name in forbidden_targets:
        monkeypatch.setattr(name,forbidden)
    async def run():
        with gpu.admit_gpu_job(gpu.JobContext('fixture','operation','training',run_id,None),
                global_cuda_bytes=config.run.global_cuda_budget_bytes,max_workers=4) as job:
            ops=_OwnedHpoOperations(job)
            if case=='startup_admission':
                def denied(*args,**kwargs): raise ContractError('fixture learner admission denial')
                monkeypatch.setattr(job.ledger,'reserve_learner',denied)
            async def collect(plan,ledger):
                calls.append(plan.linkage['collection_generation'])
                if case in ('production_collector','parent_overlap'):
                    from gf_cli.card_instance import collect_hpo_ppo
                    result=await collect_hpo_ppo(runtime,plan,ledger)
                else:
                    result=await owned_fixture_collection(ops,config,plan,ledger,empty=case=='empty')
                if case=='save_failure':
                    (ops.ownership.directory/'artifacts/000001').mkdir()
                if case=='cancel': raise asyncio.CancelledError()
                return result
            if case.startswith('startup_'):
                import json
                import os
                before=len(os.listdir('/proc/self/fd'))
                with pytest.raises(asyncio.CancelledError if case=='startup_cancel' else ContractError,match='fixture'):
                    await _run_ppo(runtime,collect,tmp_path/'parents'/run_id,_game_ops=ops)
                assert len(os.listdir('/proc/self/fd'))==before
                assert ops.budget.used==0 and ops.writer.parent_budget.used==0
                saved=json.loads((tmp_path/'parents'/run_id/'run.json').read_text())
                assert saved['status']=='failed' and saved['quarantine'] is True
                assert ops.receipt is not None  # Actual zero-owner ledger/handle cleanup, never JSON.
                return
            if case in ('archive_failure','save_failure','cancel','parent_overlap'):
                expected=asyncio.CancelledError if case=='cancel' else (OSError,RuntimeError,ValueError)
                with pytest.raises(expected):
                    await _run_ppo(runtime,collect,tmp_path/'parents'/run_id,_game_ops=ops)
                assert not (tmp_path/'parents'/run_id/'updates/000001.json').exists()
                assert ops.owned_registry.process.lease.released and ops.owned_registry.process.process.returncode is not None
                if case=='parent_overlap':
                    assert ops.writer.parent_budget.peak<=40_900_000
                    assert __import__('json').loads((tmp_path/'parents'/run_id/'run.json').read_text())['status']=='failed'
                if case=='save_failure':
                    assert ops.owned_registry.process._owned_failed
                    with pytest.raises(ContractError): ops.owned_registry.require()
                return
            report=await _run_ppo(runtime,collect,tmp_path/'parents'/run_id,_game_ops=ops)
            if case=='empty':
                assert report['status']=='incomplete' and report['stop_reason']=='empty_collection' and report['updates']==0
                assert ops.owned_registry.process.generation==0 and ops.budget.used==0
                return
            assert report['updates']==2 and report['used_decisions']==32
            assert (tmp_path/'parents'/run_id/'artifacts/000000/resume.json').is_file()
            assert ops.writer.parent_budget.peak<=config.hpo_spec['operating']['parent_budget_bytes']
            assert ops.receipt is not None and job.ledger.learner is None and not job.ledger.workers
            assert ops.budget.used==0
            assert all(row['source_proof']['replay_reconciled'] is False for row in [
                __import__('json').loads((tmp_path/'parents'/run_id/'updates'/f'{i:06}.json').read_text()) for i in (1,2)])
    asyncio.run(run())
    assert calls==([] if case.startswith('startup_') else [0,1] if case in ('healthy','production_collector') else [0])
    if case in ('healthy','production_collector'): assert guards==[0]*5+[1]*5


async def owned_fixture_collection(ops,config,plan,ledger,*,empty=False):
    import copy
    import uuid
    from dataclasses import asdict

    from gf_common.domain import EpisodeResult, GroupResult
    from gf_common.schedule import serialize_schedule
    from gf_common.structured import DecisionFrame, EventBank, event_delta
    from gf_training.card_instance_collection import CollectionPublication
    from gf_training.card_instance_source import validate_owned_collection
    from gf_training.group_pack_collection import GroupPackCollection
    from gf_training.group_pack_memory import MemoryCardInstanceJournal
    from gf_training.group_pack_publication import GroupPackCollectionPublication
    from gf_training.hpo_collection import hpo_collection_metadata
    from tests.card_instance_action_support import inputs
    from tests.test_card_instance_baseline_runtime import receipt
    from tests.test_card_instance_learning import ACTIONS, LAYOUT
    from tests.test_hpo_owned_capture import terminal
    registry=ops.owned_registry
    attachment=registry.capture
    from dataclasses import replace
    with pytest.raises(ContractError): attachment.require(registry,replace(plan))
    if plan.linkage['collection_generation']==1:
        assert registry.current.generation==1 and registry.snapshots[0].actor_capability.require() is registry.snapshots[0]
        with pytest.raises(ContractError): registry._require_snapshot(registry.snapshots[0])
    seats=plan.linkage['learner_seats']
    episode=next(iter(seats))
    original=seats[episode]
    seats[episode]=3-original
    with pytest.raises(ContractError): attachment.require(registry,plan)
    seats[episode]=original
    pool=ops.resident_runtime.pool
    plan.output.mkdir()
    artifacts=registry.artifacts(plan.bindings)
    collection=GroupPackCollection(config.storage.settings(),str(uuid.uuid4()),deadline=plan.execution_deadline,
        owned_directory=__import__('pathlib').Path(config.storage.db_root)/registry.state['run_id']/'collections'/plan.linkage['batch_id'])
    attachment.bind_collection(collection)
    try:
        # Same real guard at the collector's other four positions.
        for _ in range(3): validate_owned_collection(plan,ledger,max_owned_bytes=2*1024**3,
            deadline=plan.execution_deadline,resident_phase=pool.admission)
        await collection.start()
        all_results=[]
        for group in plan.groups:
            memory=collection.new_group(group.episodes[0].episode_id)
            results=[]
            for spec in group.episodes:
                game=str(uuid.uuid4())
                attachment.capture.open_episode(spec.episode_id,game)
                for viewer in (1,2):
                    label=spec.policies[viewer-1]
                    binding=plan.bindings[label]
                    logical=pool.worker(viewer,binding.model_id,spec.sampling_seeds[viewer-1],ACTIONS,ledger=ledger,
                        policy_options=binding.policy_options,artifact_inputs=artifacts if binding.policy_options is not None else None)
                    await logical.initialize()
                    if not empty:
                        value=inputs([],can_pray=True)
                        obs,hints=copy.deepcopy(value.observation),copy.deepcopy(value.action_options)
                        obs.update(viewer_player_id=viewer,current_player=viewer,active_player=viewer,state_revision=viewer-1)
                        obs['own']['player_id'],obs['opponent']['player_id']=viewer,3-viewer
                        hints['viewer_player_id'],hints['action_options']['actor']=viewer,viewer
                        hints['state_revision']=hints['action_options']['state_revision']=viewer-1
                        bank=EventBank(viewer,spec.episode_id+f'-{viewer}',layout=LAYOUT)
                        delta=event_delta(viewer,bank.episode,0,bank.boundary.prefix_hash,
                            [{'type':'resource_changed','player':viewer,'hp_delta':0,'mp_delta':0,'money_delta':0}],layout=LAYOUT)
                        bank.append(delta)
                        frame=DecisionFrame.from_input(replace_input(value,obs,hints,delta),bank,0,layout=LAYOUT)
                        generated,selection=await logical.select_frame(frame,delta)
                        command={'command_id':str(uuid.uuid4()),'expected_revision':frame.revision,'action':selection.validate(generated.table,frame)}
                        attachment.capture.begin_decision(spec.episode_id,viewer,frame,delta,generated.table,selection,command)
                        acknowledgement={'command_id':command['command_id'],'previous_revision':viewer-1,'state_revision':viewer,
                            'status':'terminal' if viewer==2 else 'active','result':'draw' if viewer==2 else None}
                        attachment.capture.accept_decision(spec.episode_id,viewer,command['command_id'],acknowledgement)
                        journal=MemoryCardInstanceJournal(plan.output/spec.episode_id,spec.episode_id,viewer,label,
                            run_id=collection.run_id,game_id=game,actions=ACTIONS,bank_episode=bank.episode,
                            caps=asdict(logical.caps),numeric_costs=asdict(logical.costs),model_id=binding.model_id,
                            runtime_evidence=receipt(),sampling_seed=spec.sampling_seeds[viewer-1],
                            neural=logical.neural_metadata,artifact_sink=memory.sink)
                        number=journal.begin_frame(frame,delta,generated,selection,command,{})
                        journal.acknowledge(number,acknowledgement)
                        journal.finish('natural','draw',terminal(viewer,2))
                        memory.sink.write(spec.episode_id+f'/replay-player-{viewer}.json',b'{"fixture_origin":"unqualified-decoded-diagnostic"}\n')
                    logical.mark_completed()
                    await logical.close()
                for viewer in (1,2): attachment.capture.terminal_observation(spec.episode_id,viewer,terminal(viewer,0 if empty else 2))
                result=EpisodeResult(spec.episode_id,spec.pair_id,group.seed,spec.policies,game_id=game,
                    end_kind='natural',outcome='draw',decisions=0 if empty else 2,replay_reconciled=False)
                attachment.capture.close_episode(spec.episode_id,result)
                memory.sink.write(spec.episode_id+'/result.json',( __import__('json').dumps(asdict(result))+'\n').encode())
                results.append(result)
            result=GroupResult(results,{'runtime_evidence':receipt()})
            attachment.capture.close_group(group,result)
            await collection.archive(memory,group.episodes[0].episode_id,result)
            all_results.extend(results)
        await pool.retire_unused()
        authority=pool.retire_collection()
        validate_owned_collection(plan,ledger,max_owned_bytes=2*1024**3,deadline=plan.execution_deadline,resident_phase=authority)
        collection.freeze_storage_metrics()
        manifest=hpo_collection_metadata(actions=ACTIONS,config=asdict(config),runtime_evidence=[receipt() for _ in plan.groups],
            schedule=serialize_schedule(plan.groups,4),inventory=collection.inventory,
            policies=list(plan.bindings),bots_revision='unqualified-fixture',policy_bindings=plan.bindings,
            neural_inputs=artifacts.finalize(deadline=plan.execution_deadline),linkage=plan.linkage,logical_retirement=authority.evidence)
        summary={'healthy':True,'comparison_valid':True,'fixture_origin':'unqualified-controlled-frames'}
        publication=GroupPackCollectionPublication(collection,plan.output)
        await publication.prepare(summary,manifest)
        composite=CollectionPublication(plan.output,work_deadline=plan.execution_deadline,cleanup_deadline=plan.wall_deadline,group_pack=publication)
        result=composite.commit(summary,manifest)
        return attachment.complete(publication,result)
    finally:
        artifacts.release()
        import sys
        primary=sys.exception()
        try: await collection.close()
        except BaseException as cleanup:
            # Preserve an actual collection failure; its pending diagnostics stay owned.
            if primary is None: raise
            primary.add_note('fixture collector cleanup: '+str(cleanup))


def replace_input(value,observation,hints,delta):
    from dataclasses import replace
    return replace(value,observation=observation,action_options=hints,events=tuple(delta.events))


def test_snapshot_save_exposes_actual_immutable_writer_manifest_without_readback(tmp_path,monkeypatch):
    import hashlib

    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner=learner_for()
    snapshot=learner.snapshot_owned(budget=budget())
    def forbidden(*args,**kwargs): raise AssertionError('saved artifact readback')
    monkeypatch.setattr('gf_models.sharded_arrays.inspect_bundle',forbidden)
    ack=snapshot.save(tmp_path/'saved')
    manifest=ack.inference_manifest
    from gf_common.domain import canonical
    assert hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()==ack.files['inference.json']
    manifest['weights']['files'].clear()
    assert ack.inference_manifest['weights']['files']
    assert snapshot.inference_manifest==ack.inference_manifest
    snapshot.close()


def test_owned_conditions_skip_rust_base_reader(tmp_path,monkeypatch):
    from dataclasses import replace

    from gf_cli.hpo import _conditions
    from gf_cli.hpo_config import campaign_config
    from gf_common.card_instance import CORRECTED_SIMULATOR_REVISION
    from tests.test_hpo_config import campaign_raw
    raw=campaign_raw()
    raw.update(schema_version=4,origin='producer-owned-v1',encoder='canonical-python',
        runtime={'mode':'resident-worker-v1','cache_entries':6,'cache_weight_bytes':128*1024**2})
    loaded=campaign_config(raw,origin=tmp_path)
    base=replace(loaded.base.config,preprocessing=replace(loaded.base.config.preprocessing,backend='rust'))
    loaded=replace(loaded,base=replace(loaded.base,config=base))
    def forbidden(*args,**kwargs): raise AssertionError('legacy native Source on owned route')
    monkeypatch.setattr('gf_training.native_backend.load_native',forbidden)
    monkeypatch.setattr('gf_training.native_backend.hpo_reader_settings',forbidden)
    monkeypatch.setattr('gf_training.native_backend.require_native_refresh_opponents',forbidden)
    monkeypatch.setattr('gf_training.ppo_run.source_identity',lambda deadline:{'fixture_origin':'conditions-only'})
    monkeypatch.setattr('gf_simulator_client.card_instance_runtime.CardInstanceRuntimeEvidence.preflight',lambda self:{
        'binary':{'sha256':'a'*64},'source':{'revision':CORRECTED_SIMULATOR_REVISION}})
    conditions=_conditions(loaded)
    assert conditions['origin']=='producer-owned-v1' and conditions['encoder']=='canonical-python'
    assert 'native_identity' not in conditions


def test_wire7_rejects_tampered_actual_saved_manifest_and_reaps(tmp_path,monkeypatch):
    import asyncio
    import time

    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training import hpo_owned_process as module
    from tests.test_card_instance_learning import ACTIONS
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner,account,ledger=learner_for(),budget(),ReservationLedger()
    receive=module._receive_control
    def tamper(*args,**kwargs):
        result=receive(*args,**kwargs)
        if kwargs['purpose']=='save':
            files=result.value['inference_manifest']['weights']['files']
            files[next(iter(files))]='0'*64
        return result
    monkeypatch.setattr(module,'_receive_control',tamper)
    async def run():
        child=module.HpoOwnedLearningProcess(tmp_path,1024**2,learner.model,learner.training,learner.binding,
            learner.action_config_hash,learner.resume_identity,actions=ACTIONS,ledger=ledger,
            cuda_visible_devices='',cleanup_deadline=time.monotonic()+20,budget=account)
        await child.start()
        try:
            with pytest.raises(ContractError): await child.initialize_owned(17,tmp_path/'saved',deadline=time.monotonic()+10)
            assert child.current_snapshot is None and child._owned_failed
            with pytest.raises(ContractError): await child.initialize_owned(17,tmp_path/'retry',deadline=time.monotonic()+10)
        finally:
            try: await child.close(time.monotonic()+2)
            except (RuntimeError,ContractError,TimeoutError,BrokenPipeError): pass
        assert child.process.returncode is not None and child.lease.released and account.used==0
    asyncio.run(run())


@pytest.mark.parametrize('kind,version',[('collection',8),('proof',5),('numeric',6),('update',6),('generation',5),('registry',6),('cache',2)])
def test_owned_record_versions_refuse_mixed_legacy_tuple(kind,version):
    from gf_common.hpo import require_owned_record_version
    value={'schema_version':version,'origin':'producer-owned-v1','encoder':'canonical-python'}
    require_owned_record_version(value,kind)
    for changed in ({**value,'schema_version':version-1},{**value,'schema_version':True},
                    {**value,'origin':'legacy'},{**value,'encoder':'rust'}):
        with pytest.raises(ContractError): require_owned_record_version(changed,kind)


def install_owned_fixture_runtime(monkeypatch):
    """Controlled external endpoints; real environment callbacks, actors and writers."""
    import copy
    import uuid
    from types import SimpleNamespace

    import gf_cli.card_instance as cli
    import gf_simulator_client.card_instance_environment as environment
    from tests.test_group_pack_runtime_environment import Evidence, PublicClient
    class Client(PublicClient):
        async def create(self):
            result=await super().create()
            self.game=str(uuid.uuid4())
            self.revision=0
            return {**result,'game_id':self.game}
        async def observe(self,game,viewer):
            obs=await super().observe(game,viewer)
            actor=1 if self.revision==0 else 2
            obs.update(state_revision=self.revision,active_player=actor,
                current_player=None if self.terminal else actor)
            return obs
        async def actor_options(self,*args):
            hints=copy.deepcopy(await super().actor_options(*args))
            hints['state_revision']=hints['action_options']['state_revision']=self.revision
            hints['action_options']['actor']=1 if self.revision==0 else 2
            return hints
        async def act(self,game,actor,payload):
            before=self.revision
            self.revision+=1
            self.attempts.append((actor,payload['action']))
            self.terminal=self.revision==2
            return {'command_id':payload['command_id'],'previous_revision':before,'state_revision':self.revision,
                'status':'terminal' if self.terminal else 'active','result':'draw' if self.terminal else None}
    class Server:
        def __init__(self,*args,**kwargs):
            self.root=args[4]
            self.process=SimpleNamespace(returncode=None)
            self.drainers=[]
            self.expected={}
        async def start(self):
            self.root.mkdir(parents=True)
            (self.root/'stdout.log').write_bytes(b'unqualified fixture server')
            return {'base_url':'http://controlled-fixture','test_server_info':{}}
        async def stop(self,*args):
            self.process.returncode=0
            return {'exit_code':0,'shutdown':'stdin_eof','log_error':None}
    monkeypatch.setattr(cli,'CardInstanceRuntimeEvidence',Evidence)
    monkeypatch.setattr(environment,'CardInstanceRuntimeEvidence',Evidence)
    monkeypatch.setattr(environment,'AttestedClient',lambda *args,**kwargs:Client('natural'))
    monkeypatch.setattr(environment.CardInstanceSimulatorEnvironment,'_new_server',lambda self,*args,**kwargs:Server(*args,**kwargs))


def test_saved_manifest_transfer_respects_save_phase_before_new_control_allocation(tmp_path,monkeypatch):
    from gf_training.hpo_owned_worker import _send_snapshot
    from tests.test_hpo_numeric import learner_for
    from tests.test_hpo_owned_numeric import budget
    learner,account=learner_for(),budget()
    snapshot=learner.snapshot_owned(budget=account)
    snapshot.save(tmp_path/'saved')
    snapshot._save_budget_limit=1
    def forbidden(*args,**kwargs): raise AssertionError('control allocated before save-phase admission')
    monkeypatch.setattr('gf_training.hpo_owned_worker._control_bank',forbidden)
    with pytest.raises(ContractError,match='save.*envelope'):
        _send_snapshot(None,snapshot,request={'transaction':'fixture'},budget=account,metrics={},identity={},learner=learner)
    snapshot.close()
