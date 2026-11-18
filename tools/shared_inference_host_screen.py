"""Write-once synthetic component memory screen; no gameplay/Source authority."""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import resource
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def memory(pid):
    fields={}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty'):
            fields[key]=int(value.split()[0])*1024
    return fields


def identity():
    import hashlib
    import importlib.metadata
    import subprocess
    paths=[Path(__file__),*sorted(Path('src').rglob('*.py')),*sorted(Path('tests').glob('*.py')),*sorted(Path('configs').rglob('*.toml'))]
    revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    difference=subprocess.run(['git','diff','--binary','HEAD'],capture_output=True,check=False)
    return {'namespace':'host' if os.getpid()!=2 else 'sandbox-pid-namespace',
        'started_unix':time.time(),'python':sys.version,'pid':os.getpid(),
        'source_revision':revision.stdout.strip() if revision.returncode==0 else None,
        'source_diff_sha256':hashlib.sha256(difference.stdout).hexdigest() if difference.returncode==0 else None,
        'files':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','torch','torchrl','tensordict')}}


def components(output):
    from gf_cli.hpo_config import load_hpo_config
    from gf_common.domain import canonical
    from gf_common.structured import EventBank, event_delta
    from gf_models.shared_numeric import numeric_row, tensor_row
    from gf_models.structured_tree_encoder import _owned
    from gf_simulator_client.framing import frame_decode_bytes
    from tests.model3_support import LAYOUT, raw_inputs
    c=load_hpo_config('configs/hpo/mlp-shared-inference.toml').config
    result={'scope':'synthetic numeric capacity; no Source/quality authority','identity':START_IDENTITY,'points':{}}
    def point(name,graphs=()):
        gc.collect()
        before=memory(os.getpid());peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        # Observer allocations occur in an owned forked CPU-only child, never in the workload heap.
        reader,writer=os.pipe()
        child=os.fork()
        if child==0:
            os.close(reader)
            payload=json.dumps({'bytes':_owned(graphs),'observer_child_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}).encode()
            os.write(writer,payload);os.close(writer);os._exit(0)
        os.close(writer)
        with os.fdopen(reader,'rb') as source:measured=json.loads(source.read())
        waited,status=os.waitpid(child,0)
        if waited!=child or status!=0:raise RuntimeError('graph observer original child failed')
        result['points'][name]={'memory_before_observer':before,'python_graph_bytes':measured['bytes'],
            'workload_peak_rss_before_observer':peak,'observer_child_peak_rss':measured['observer_child_peak_rss']}
    point('import_baseline')
    bank=EventBank(1,'capacity-screen',c.model.event.max_episode,layout=LAYOUT)
    from gf_common.card_instance_vocab import DEFINITION_SHA256
    definition=max(DEFINITION_SHA256,key=len)
    event_options=[{'type':'resource_changed','player':2,'hp_delta':-(2**31),'mp_delta':2**31-1,'money_delta':-(2**31)},
        {'type':'buy_offered','buyer':2,'seller':1,'definition_id':definition,'price':2**32-1},
        {'type':'sale_offered','seller':2,'target':1,'definition_id':definition,'price':2**32-1},
        {'type':'artifact_transferred','from':1,'to':2,'definition_id':definition,'price':2**32-1},
        {'type':'damage_applied','player':2,'amount':2**31-1,'remaining_hp':-(2**31)},
        {'type':'miracle_used','player':2,'definition_id':definition,'mp_spent':2**32-1}]
    from gf_common.structured import _validate_events
    for candidate in event_options:_validate_events([candidate],LAYOUT)
    event=max(event_options,key=lambda row:_owned(json.loads(canonical(row))))
    result['event_domain_audit']=[{'event':row,'json_bytes':len(canonical(row).encode()),'decoded_graph_bytes':_owned(json.loads(canonical(row)))} for row in event_options]
    result['event_domain_scope']='largest decoded graph among audited closed DTO variants at integer extremes and longest catalog definition; status/result variants separately smaller by field inventory'
    for start in range(0,c.model.event.max_episode,c.model.event.max_interval):
        delta=event_delta(1,bank.episode,start,bank.boundary.prefix_hash,[event]*min(c.model.event.max_interval,c.model.event.max_episode-start),layout=LAYOUT)
        bank.append(delta)
    view=bank.view()
    point('max_retained_bank',(vars(bank),view,delta))
    decoded=view.events()
    point('full_retained_history_decoded',(vars(bank),view,decoded))
    del decoded
    frames,candidates,*_=raw_inputs(count=1)
    frame=frames[0]
    count=c.inference.max_candidates
    slots=(0,)*(count*c.model.max_positions)
    candidates=replace(candidates,kinds=(candidates.kinds[0],)*count,targets=(candidates.targets[0],)*count,
        numerics=tuple(tuple(int(str(2**32-1-i-j)) for j in range(4)) for i in range(count)),
        present=tuple((True,True,True,True) for _ in range(count)),
        offsets=tuple(range(0,len(slots)+1,c.model.max_positions)),slots=slots)
    frame=replace(frame,interval_numerics=tuple(tuple(value for value in frame.interval_numerics[0]) for _ in range(c.model.event.max_interval)),
        interval_present=tuple(tuple(value for value in frame.interval_present[0]) for _ in range(c.model.event.max_interval)),
        interval_categories=tuple(tuple(value for value in frame.interval_categories[0]) for _ in range(c.model.event.max_interval)),event_end=c.model.event.max_interval)
    point('max_candidate_tokens_with_retained_bank',(vars(bank),view,frame,candidates))
    numeric=numeric_row(frame,candidates,c.model,layout=LAYOUT)
    point('numeric_pack',(vars(bank),view,frame,candidates,numeric))
    payload={'routing':{'execution':'a'*32,'game':'screen','participant':1,'request':1,'revision':0,'checkpoint':'b'*64},'numeric':numeric}
    import io

    from gf_simulator_client.framing import read_frame, write_frame
    wire=io.BytesIO()
    write_frame(wire,payload)
    encoded=wire.getvalue()
    wire.seek(0)
    decoded=read_frame(wire)
    encoded_json=encoded[8:]
    framed_copy=encoded_json.__len__().to_bytes(8,'big')+encoded_json
    read_bytearray=bytearray(encoded_json)
    read_final_bytes=bytes(read_bytearray)
    point('framing_encode_and_decode',(vars(bank),view,frame,candidates,numeric,encoded,decoded,encoded_json,framed_copy,read_bytearray,read_final_bytes))
    result.update(wire_bytes=len(encoded),framing_decode_admission_bytes=frame_decode_bytes(len(encoded)))
    tensors=tensor_row(decoded['numeric'])
    result['raw_tensor_bytes']=sum(v.numel()*v.element_size() for v in tensors.values())
    point('cpu_tensors_with_framing',(vars(bank),view,frame,candidates,numeric,encoded,decoded,encoded_json,framed_copy,read_bytearray,read_final_bytes))
    del tensors,numeric,encoded,decoded,wire,payload,encoded_json,framed_copy,read_bytearray,read_final_bytes

    result['excluded_scope']='search/selection/reply/sampling graphs require the separately archived real-generator evidence; this tool does not bound them'
    result['root_scope']='every numeric/framing point retains vars(EventBank),view,raw tokens and specified payload roots; graph observer memory is separate'
    result['conditions']={'events':c.model.event.max_episode,'interval':c.model.event.max_interval,'candidates':count,
        'positions':c.model.max_positions,'aggregate_window':c.model.event.aggregate_window,'model':asdict(c.model)}
    output.write_text(json.dumps(result,indent=2)+'\n')


async def topology(output,count,device):
    from gf_cli.hpo_config import load_hpo_config
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.domain import digest
    from gf_common.ram_bank import ReadonlyRamBankSet
    from gf_common.shared_inference import ServingBudget, shared_runtime
    from gf_models.card_instance_artifacts import RetainedGameArtifacts
    from gf_models.structured_artifacts import save_game_inference_v4
    from gf_models.structured_binding import ConfiguredNeuralModelV4
    from gf_models.structured_network import make_network
    from gf_simulator_client.shared_service import SharedInferenceService
    from gf_simulator_client.shared_worker import SharedParticipantWorker
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.model3_support import LAYOUT
    from tests.test_card_instance_baseline_runtime import game
    class ScreenLedger(ReservationLedger):
        _MAX_WORKERS=2**63-1
    loaded=load_hpo_config('configs/hpo/mlp-shared-inference.toml')
    c,actions=loaded.config,loaded.base.config.actions
    network=make_network(c.model,17,replace(c.inference,device='cpu'),layout=LAYOUT)
    artifact=output.parent/(output.stem+'-inference')
    save_game_inference_v4(artifact,network,c.model,digest(asdict(actions)),action_contract=actions.action_contract,rules=card_instance_rules())
    configured=ConfiguredNeuralModelV4(**asdict(c.model),checkpoint=str(artifact),inference=c.inference)
    retained=RetainedGameArtifacts((configured,),actions,rules=card_instance_rules(),max_owned_bytes=1024**3,
        preflight_deadline=time.monotonic()+120,finalization_deadline=time.monotonic()+300)
    _,binding=retained.binding(configured)
    banks=ReadonlyRamBankSet.build({name:value.detach().numpy() for name,value in network.state_dict().items()},ServingBudget(128*1024**2))
    ledger=ScreenLedger(1024**3,max(2,count))
    runtime=shared_runtime({'mode':'shared-mlp-v1','max_batch_size':2},game_concurrency=max(1,count//2))
    service=SharedInferenceService(ledger,runtime,execution='a'*32,directory=output.parent/(output.stem+'-service'),
        slots=max(2,count),estimate={'cuda_bytes':0},device=device)
    result={'scope':'component topology; synthetic decision; no gameplay/Source authority','identity':START_IDENTITY,'actors':count,
        'device':device,'points':{},'registration':service.registration}
    actors=[]
    def point(name):
        pids={'controller':os.getpid(),'broker':service.process.pid,'service':service.registration['gpu_pid']}
        pids.update({f'actor{i}':worker.process.pid for i,worker in enumerate(actors)})
        result['points'][name]={role:memory(pid) for role,pid in pids.items()}
    try:
        service.publish(banks,model=asdict(c.model),binding=asdict(c.inference),checkpoint=binding.policy_options['expected_manifest_digest'],generation=0,identity='b'*64)
        point('loaded_service')
        for index in range(count):
            worker=SharedParticipantWorker('mlp',17+index,actions,ledger=ledger,policy_options=binding.policy_options,
                artifact_inputs=retained,service=service,game=f'component{index}',viewer=1)
            actors.append(worker)
        for worker in actors:await worker.initialize()
        point('all_actors_ready')
        async def decide(worker):
            frame,_bank,delta,_=game()
            return await worker.select_frame(frame,delta)
        await asyncio.gather(*(decide(worker) for worker in actors))
        point('all_actors_warm_2C_synthetic_games_stress')
        service.quiesce()
        result['stress_2C_diagnostics']=service.telemetry
        result['stress_scope']='count distinct synthetic games =2C active-game stress; not normal C-game admission'
        service._exchange({'command':'METRICS_RESET'})
        # A second own decision requires new revision/contiguous delta. Reset fresh CPU actors by closing/recreating only selected endpoints.
        for worker in actors[:max(1,count//2)]:await worker.close()
        normal=[]
        for index in range(max(1,count//2)):
            worker=SharedParticipantWorker('mlp',117+index,actions,ledger=ledger,policy_options=binding.policy_options,
                artifact_inputs=retained,service=service,game=f'normal-component{index}',viewer=1)
            normal.append(worker)
        actors=normal+actors[max(1,count//2):]
        for worker in normal:await worker.initialize()
        await asyncio.gather(*(decide(worker) for worker in normal))
        point('normal_C_requests_B2_after_all_actors_warm')
        service.quiesce()
        result['normal_C_diagnostics']=service.telemetry
        result['normal_C_scope']='C distinct synthetic games, one pending per game, B=2; C=count//2'
        result['configured_games']=max(1,count//2)
    finally:
        for worker in actors:await worker.close()
        result['service_proof']=service.close()
        banks.close();retained.release()
    output.write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--actors',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--components',action='store_true')
    args=parser.parse_args()
    global START_IDENTITY
    START_IDENTITY=identity()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise SystemExit('write-once output already exists')
    if args.components:components(args.output)
    else:asyncio.run(topology(args.output,args.actors,args.device))

if __name__=='__main__':main()
