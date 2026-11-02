"""Tree2 own CPU child and non-PPO Journal5 component boundaries."""
import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict

import numpy as np
import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical, digest
from gf_models.structured_tree_artifacts import TreeOrigin, binding_options, fit_toy, save_artifact
from gf_models.structured_tree_config import ConfiguredTreeModelV2
from gf_models.structured_tree_encoder import PreparedTreeFrame, safe_tree_options_v2
from tests.test_structured_protocol import action_settings, own_frame
from tests.test_structured_tree_encoder import tokens


@pytest.fixture
def tree_binding(make_input,tmp_path):
    pytest.importorskip('lightgbm')
    actions=action_settings()
    model=ConfiguredTreeModelV2()
    frame,candidates=tokens(make_input)
    rows=next(PreparedTreeFrame.create(frame,model).iter_chunks(candidates))[1]
    x=np.tile(rows,(16,1)); y=np.tile(np.arange(4,dtype=np.float32),16)
    origin=TreeOrigin(digest(asdict(actions)),engineering_contract(),'component',False)
    tree=fit_toy([x],y,[x],y,model=model,origin=origin,seed=17,rounds=4,min_data_in_leaf=1)
    directory=tmp_path/'artifact'
    save_artifact(directory,tree)
    sha=hashlib.sha256((directory/'inference.json').read_bytes()).hexdigest()
    options=binding_options(directory,expected_digest=sha,model=model,action_config_hash=origin.action_config_hash,
        action_contract=origin.action_contract,purpose='component',synthetic_capacity=False)
    return actions,tree,options


def test_config_partial_joint_dispatch_and_optional_free_cost_binding(tmp_path,tree_binding):
    import subprocess
    import sys

    from gf_cli.config import load_config
    from gf_models.structured_runtime import own_worker_settings
    actions,_tree,options=tree_binding
    preset=tmp_path/'tree.toml'
    preset.write_text("schema_version=1\nkind='experiment'\n[model]\nmodel_id='lightgbm'\nversion='2'\n[actions]\nversion='2'\n[actions.action_contract]\n"+
        '\n'.join(key+'='+json.dumps(value) for key,value in asdict(engineering_contract()).items())+"\n[preprocessing]\nreader='structured-v1'\n[server]\nfixed_seed_experiment=true\n")
    loaded=load_config(preset,{'model':{'inference':{'candidate_chunk':1}}})
    assert isinstance(loaded.config.model,ConfiguredTreeModelV2)
    assert loaded.config.model.inference.candidate_chunk==1
    assert loaded.config.model.encoder.max_positions==18
    with pytest.raises(ContractError):
        load_config(preset,{'model':{'temporal':{'sequence_length':1}}})
    caps,costs,resident,temporary=own_worker_settings('lightgbm',actions,options)
    assert resident==temporary==0 and caps.max_decisions is None
    assert costs.tail_operations_per_candidate>0 and costs.actor_hidden_width==0
    encoded=tmp_path/'safe.json';encoded.write_text(canonical(options))
    script="""import json,sys
from gf_common.structured import action_config_v2
from gf_models.structured_runtime import own_worker_settings
options=json.loads(open(sys.argv[1]).read());actions=action_config_v2(json.loads(sys.argv[2]))
own_worker_settings('lightgbm',actions,options)
assert not ({'numpy','torch','lightgbm'} & sys.modules.keys())
"""
    subprocess.run([sys.executable,'-c',script,str(encoded),canonical(asdict(actions))],check=True)
    with pytest.raises(ContractError,match='feature signature'):
        safe_tree_options_v2({**options,'feature_signature':'0'*64},action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)


def test_real_cpu_tree_worker_two_frames_null_statistics_and_journal5(make_input,tmp_path,tree_binding):
    from gf_common.structured import EventBank
    from gf_models.structured_baselines import make_structured_policy
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    from gf_training.structured_journal import StructuredJournal, read_structured_journal

    actions,_tree,options=tree_binding
    async def run():
        ledger=ReservationLedger()
        worker=StructuredParticipantWorker(make_structured_policy,'lightgbm',2**64-1,actions,ledger=ledger,policy_options=options)
        journal=None
        try:
            await worker.initialize()
            bank=EventBank(1,'opaque-tree-episode')
            directory=tmp_path/'journal';directory.mkdir()
            journal=StructuredJournal(directory,'tree-episode',1,worker.expected_policy,run_id='run',game_id='game',
                action_config_hash=digest(asdict(actions)),bank_episode=bank.episode,caps=asdict(worker.caps),numeric_costs=asdict(worker.costs),
                model_id='lightgbm',model_version='2',sampling_seed=2**64-1,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
            assert journal.header['generation_schema'] is None
            for revision in (0,2):
                frame,delta=own_frame(make_input(),bank,revision,[{'type':'action_accepted','player':1}])
                generated,selected=await worker.select_frame(frame,delta)
                assert selected.logprob is selected.value is None
                command={'command_id':str(uuid.uuid4()),'expected_revision':frame.revision,'action':selected.validate(generated.table,frame)}
                index=journal.begin_frame(frame,delta,generated,selected,command,{})
                journal.acknowledge(index,{'command_id':command['command_id'],'previous_revision':revision,
                    'state_revision':revision+1,'status':'active','result':None})
            journal.finish('truncated',None)
            raw=read_structured_journal(journal.final_path,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
            assert raw.accepted==(0,1) and raw.end['outcome'] is None
            assert all(item.selection.value is item.selection.logprob is None for item in raw.decisions)
            if os.environ.get('GFB_RUN_STRUCTURED_TREE_NATIVE') == '1':
                import gf_native
                settings=canonical({'memory_bytes':512*1024**2,'max_json_bytes':64*1024**2,'max_record_bytes':8*1024**2}).encode()
                native=gf_native.StructuredJournalTokens(str(directory),journal.final_path.name,settings,time.monotonic()+30)
                try:
                    native_raw=json.loads(native.diagnostic())
                    assert native_raw['header']['model_id']=='lightgbm'
                    assert native_raw['header']['generation_schema'] is None
                    assert len(native_raw['decisions'])==2
                    assert all(item['selection']['value'] is item['selection']['logprob'] is None for item in native_raw['decisions'])
                    assert not hasattr(native,'export_cohort')
                    native.recheck()
                finally:
                    native.close()

        finally:
            await worker.close()
        assert worker.closed and worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def test_tree_two_seat_fake_http_keeps_public_guards_and_nonppo_headers(tmp_path,monkeypatch,tree_binding):
    import httpx

    from gf_common.domain import EpisodeGroup, EpisodeSpec, RuntimePolicyBinding
    from gf_models.structured_baselines import make_structured_policy
    from gf_models.structured_tree_config import TreeDescriptor, policy_label
    from gf_simulator_client.environment import EnvironmentSettings
    from gf_simulator_client.http import AttestedClient, seed_fingerprint
    from gf_simulator_client.process import server_effective
    from gf_simulator_client.structured_environment import StructuredSimulatorEnvironment
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.structured_journal import StructuredJournal, read_structured_journal
    from tests.episode_support import PublicScript
    from tests.structured_ppo_support import configuration

    actions,tree,options=tree_binding
    config=configuration(tmp_path/'config').config
    label=policy_label(tree.model,TreeDescriptor(**options['tree_descriptor']),options['expected_manifest_digest'])
    registry={label:RuntimePolicyBinding('lightgbm',options),'random-structured-v1':RuntimePolicyBinding('random')}
    scripts=[];by_url={}
    class Server:
        def __init__(self,binary,server_config,seed,count,directory,*args,**kwargs):
            self.script=PublicScript()
            self.expected=server_effective(server_config,count)
            self.script.info.update(self.expected,seed_fingerprint=seed_fingerprint(seed))
            self.url='http://tree-component'
            by_url[self.url]=self.script;scripts.append(self.script)
        async def start(self):
            return {'base_url':self.url,'test_server_info':self.script.info}
        async def stop(self,timeout):
            return {'exit_code':0,'shutdown':'stdin_eof','log_error':None}
    workers=[]
    import gf_simulator_client.structured_environment as module
    original_worker=module.StructuredParticipantWorker
    class Worker(original_worker):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);workers.append(self)
    monkeypatch.setattr(module,'StructuredParticipantWorker',Worker)
    class FakeEnvironment(StructuredSimulatorEnvironment):
        def __init__(self,settings,ledger):
            self._initialize_structured(settings,make_structured_policy,StructuredJournal,ledger=ledger)
        def _new_server(self,*args,**kwargs):
            return Server(*args,**kwargs)
        def _new_client(self,url,**kwargs):
            return AttestedClient(url,transport=httpx.MockTransport(by_url[url].handle),**kwargs)
        async def run_group(self,group):
            return await self._run_group_body(group)
        async def _episode(self,client,definitions,group,spec):
            return await self._episode_body(client,definitions,group,spec)
    settings=EnvironmentSettings(binary=config.server.binary,server_config=config.server.config,output_dir=tmp_path/'episodes',
        run_id='tree-component',actions=actions,policy_bindings=registry,fixed_seed_experiment=True)
    ledger=ReservationLedger()
    # Real normal constructor remains guarded; no global require_qualified patch.
    with pytest.raises(ContractError,match='engineering'):
        StructuredSimulatorEnvironment(settings,make_structured_policy,StructuredJournal,ledger=ledger)
    assert not scripts and not workers
    group=EpisodeGroup(config.run.seeds[0],(EpisodeSpec('seat1',(label,'random-structured-v1'),(17,19),'pair',1),
        EpisodeSpec('seat2',('random-structured-v1',label),(19,17),'pair',2)))
    result=asyncio.run(FakeEnvironment(settings,ledger).run_group(group))
    assert len(result.episodes)==2 and all(item.end_kind=='natural' and not item.health for item in result.episodes)
    assert len(workers)==4 and all(worker.closed and worker.process.poll() is not None for worker in workers)
    assert not ledger.workers
    for spec in group.episodes:
        for viewer in (1,2):
            path=settings.output_dir/spec.episode_id/f'player-{viewer}.jsonl'
            raw=read_structured_journal(path,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
            assert raw.header['generation_schema'] is None and raw.end['end_kind']=='natural'
            assert all(item.selection.logprob is item.selection.value is None for item in raw.decisions)
            assert raw.header['model_id']==('lightgbm' if spec.policies[viewer-1]==label else 'random')


@pytest.mark.parametrize('field,change',[('model_bytes',1),('model_bytes',-1),('tree_count',1),('max_depth',1)])
def test_factory_rejects_valid_shaped_wrong_dto_descriptor_before_native_load(tree_binding,monkeypatch,field,change):
    import copy

    import gf_models.structured_tree_artifacts as artifacts
    from gf_models.structured_baselines import make_structured_policy
    actions,_tree,options=tree_binding
    malformed=copy.deepcopy(options)
    malformed['tree_descriptor'][field]+=change
    monkeypatch.setattr(artifacts,'_native',lambda **kw:pytest.fail('DTO mismatch must reject before native import/Booster'))
    with pytest.raises(ContractError,match='inspected descriptor'):
        make_structured_policy('lightgbm',malformed,digest(asdict(actions)),action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
