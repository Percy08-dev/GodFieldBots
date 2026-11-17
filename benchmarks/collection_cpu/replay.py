"""Full32-game learner-only G0/G1 preparation, fresh replay and separate profiles."""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.collection_cpu.common import (
    MAX_OUTPUT,
    TRANSFERS,
    check_files,
    code_files,
    immutable_json,
    namespace,
    read_json,
    runtime_identity,
    sha256,
    tree_bytes,
)


def profile_endpoints(batches: list[Any]) -> list[tuple[int,int,int]]:
    rows = [(len(table['kinds']), batch, episode, index)
            for batch, data in enumerate(batches) for episode, sequence in enumerate(data['sequences'])
            for index, table in enumerate(sequence['candidates'])]
    ordered = sorted(rows)
    if len(ordered) < 6:
        raise ValueError('six distinct recorded profile endpoints required')
    return [(batch,ep,index) for _,batch,ep,index in [*ordered[:3],*ordered[-3:]]]


def prepare(args: Any) -> int:
    from gf_common.card_instance_rules import card_instance_rules
    from gf_models.sharded_arrays import read_json as original_json
    from gf_training import structured_source
    from gf_training.hpo_config import hpo_training_config
    from gf_training.hpo_lineage import hpo_state
    from gf_training.hpo_rollout import read_hpo_rollout
    from gf_training.hpo_source import validate_hpo_parent
    parent = args.source.resolve(strict=True)
    output = namespace(args.output)
    seeds: dict[tuple[str,int],int] = {}
    cohort_seats: dict[int,Any] = {}
    inspector = structured_source._inspect_component_collection

    def observed(*pos: Any, **kwargs: Any) -> Any:
        result = inspector(*pos, **kwargs)
        manifest = result.manifest
        schedule = {episode['episode_id']:episode for group in manifest['schedule'] for episode in group['episodes']}
        seats = {name:spec['learner_viewer'] for name,spec in schedule.items()}
        if seats != manifest['linkage']['learner_seats'] or len(seats) != 16:
            raise ValueError('original explicit learner seats differ')
        generation = manifest['linkage']['collection_generation']
        if generation not in (0,1) or generation in cohort_seats:
            raise ValueError('exact two original generation sampling cohorts required')
        cohort_seats[generation] = seats
        cohort_keys = set()
        for row in result.episodes:
            if seats[row.episode_id] != row.viewer:
                continue
            seed = schedule[row.episode_id]['sampling_seeds'][row.viewer-1]
            key = row.records.bank.episode, row.viewer
            if key in seeds and seeds[key] != seed:
                raise ValueError('original participant sampling identity differs')
            seeds[key] = seed
            cohort_keys.add(key)
        if len(cohort_keys) != 16 or len(seeds) > 32:
            raise ValueError('bounded16 learner sampling identities per batch required')
        return result

    # Original validator owns all authority; this read-only observer retains only
    # already validated scalar sampling controls, never rewards/hidden state.
    structured_source._inspect_component_collection = observed
    try:
        state, records = validate_hpo_parent(parent, require_complete=False,
            max_owned_bytes=2*1024**3, deadline=time.monotonic()+180)
    except BaseException:
        seeds.clear()
        cohort_seats.clear()
        raise
    finally:
        structured_source._inspect_component_collection = inspector
    if (state['status'] != 'incomplete' or state['stop_reason'] != 'update_cap'
            or state['updates'] != 2 or len(records) != 3):
        raise ValueError('accepted healthy two-update historical diagnostic parent required')
    actions, model, _ = hpo_state(state)
    expected = {'mlp':314,'gru':314,'transformer':310}[model.model_id]
    recipe = hpo_training_config(state['effective_config']['hpo_spec']['training_recipe'])
    batches: list[dict[str,Any]] = []
    files = {str(parent/'run.json'):sha256(parent/'run.json')}
    manifests = []
    source_selections = []
    sampling = []
    for generation in (0,1):
        update_path = parent/f'updates/{generation+1:06}.json'
        update = original_json(update_path, expected_digest=records[generation+1]['update_sha256'], max_owned_bytes=2*1024**3)
        source_selections.append({'generation':generation,'selection':update['source_proof']['selection'],
                                  'collection_manifest_sha256':update['collection_manifest_sha256']})
        rollout = read_hpo_rollout(parent/update['numeric'], update['numeric_sha256'], model, recipe,
            actions=actions,rules=card_instance_rules(),expected_selection=update['source_proof']['selection'],
            expected_snapshot_identity=records[generation]['snapshot_identity'],
            numeric_budget_bytes=512*1024**2,max_owned_bytes=2*1024**3)
        if (len(rollout.episodes) != 16 or len(update['learner_seats']) != 16
                or cohort_seats.get(generation) != update['learner_seats']):
            raise ValueError('all16 learner sequences required per original batch')
        checkpoint = parent/f'artifacts/{generation:06}'
        manifest = original_json(checkpoint/'inference.json', expected_digest=records[generation]['inference_sha256'], max_owned_bytes=2*1024**3)
        sequences, controls = [], []
        for episode in rollout.episodes:
            sequences.append({'frames':[asdict(frame) for frame in episode.frames],
                              'candidates':[asdict(table) for table in episode.tables]})
            key = episode.records.bank.episode, episode.records.bank.actor
            if key not in seeds:
                raise ValueError('original sampling identity absent from validated collection')
            controls.append({'sampling_seed':seeds[key], 'episode':key[0], 'viewer':key[1]})
        batches.append({'generation':generation,'sequences':sequences})
        sampling.append(controls)
        manifests.append(manifest)
        files[str(update_path)] = sha256(update_path)
        for name in update['numeric_files']:
            path = parent/update['numeric']/name
            files[str(path)] = sha256(path)
        for name, expected_sha in records[generation]['artifact_files'].items():
            if name == 'inference.json' or name.startswith('weights-'):
                path = checkpoint/name
                if sha256(path) != expected_sha:
                    raise ValueError('original inference bundle changed')
                files[str(path)] = expected_sha
    count = sum(len(sequence['frames']) for batch in batches for sequence in batch['sequences'])
    if count != expected:
        raise ValueError(f'original learner population differs: {count} != {expected}')
    check_files(files)
    output.mkdir(parents=True,exist_ok=False)
    for generation, manifest in enumerate(manifests):
        target = output/f'G{generation}'
        target.mkdir()
        checkpoint = parent/f'artifacts/{generation:06}'
        for name in ('inference.json',*manifest['weights']['files']):
            shutil.copyfile(checkpoint/name,target/name)
    immutable_json(output/'inputs.json',{'schema_version':1,'population':'learner-only-all32-natural-games', 'batches':batches})
    immutable_json(output/'sampling.json',{'schema_version':1,'scope':'original participant CPU sampling controls only; never tensor features','batches':sampling})
    local = {str(path.relative_to(output)):sha256(path) for path in output.rglob('*') if path.is_file()}
    from gf_common.card_instance import CORRECTED_REVIEWED_SPEC_HASHES
    server = state['effective_config']['server']
    simulator_files = {str(Path(server[key]).resolve()):sha256(Path(server[key])) for key in ('binary','config')}
    simulator_files.update({str(Path(server['source_dir'])/name):value for name,value in CORRECTED_REVIEWED_SPEC_HASHES.items()})
    check_files(simulator_files)
    immutable_json(output/'manifest.json',{'schema_version':1,'kind':'diagnostic-historical-learner-replay',
        'qualification':'no Source, ordinary resume, training, evaluation or producer authority',
        'parent':str(parent),'original_files':files,'files':local,'preparation_code':code_files(),
        'population':'learner-only-all32-natural-games','decisions':count,'games':32,
        'source_selections':source_selections,'simulator_files':simulator_files,
        'model':model.model_id,'runtime_model':asdict(model),'actions':asdict(actions),
        'inference_binding':state['effective_config']['model']['inference'],
        'action_config_hash':manifests[0]['action_config_hash'],
        'checkpoints':[{'generation':i,'inference_sha256':records[i]['inference_sha256'],
                        'snapshot_identity':records[i]['snapshot_identity'],'recorded_producer':m['producer'],
                        'schema_version':m['schema_version']} for i,m in enumerate(manifests)],
        'schema_versions':{'numeric':6,'tokens':1}, 'simulator':manifests[0]['action_contract'],
        'profile_endpoints':profile_endpoints(batches),
        'source_validation':'full original validate_hpo_parent(require_complete=False,2GiB,180s); selection and snapshot original readers'})
    if tree_bytes(output) > MAX_OUTPUT:
        raise ValueError('prepared replay512MiB output cap exceeded')
    print(json.dumps({'status':'PREPARED','output':str(output),'games':32,'learner_decisions':count}))
    return 0


def tuples(value: Any) -> Any:
    if isinstance(value,list):
        return tuple(tuples(item) for item in value)
    if isinstance(value,dict):
        return {key:tuples(item) for key,item in value.items()}
    return value


def install(network: Any, transfer: str) -> Any:
    if transfer == 'T0':
        return None
    if transfer == 'reuse':
        from gf_models.inference_experiment import ExperimentOptions
        from gf_models.inference_experiment import install as historical
        return historical(network,ExperimentOptions(packing='reuse'))
    from benchmarks.collection_cpu.transfer import install as transfer_install
    return transfer_install(network,transfer)


def trace_summary(paths: list[Path]) -> dict[str, Any]:
    copies: dict[str,Any] = {kind:{'count':0,'bytes':0,'bytes_complete':True} for kind in ('H2D','D2H','DtoD')}
    syncs: dict[str,int] = {}
    kernels: dict[str,int] = {}
    apis: dict[str,int] = {}
    for path in paths:
        for event in read_json_trace(path):
            name = event.get('name','')
            low = name.lower()
            category = event.get('cat','').lower()
            target = kernels if 'kernel' in category else apis if 'cuda_runtime' in category or 'cuda_driver' in category else None
            if target is not None:
                target[name] = target.get(name,0)+1
            if 'synchroniz' in low:
                syncs[name] = syncs.get(name,0)+1
            for kind, names in (('H2D',('htod','h2d')),('D2H',('dtoh','d2h')),('DtoD',('dtod','d2d'))):
                if 'memcpy' in low and any(label in low for label in names):
                    copies[kind]['count'] += 1
                    size = event.get('args',{}).get('bytes')
                    if type(size) is int:
                        copies[kind]['bytes'] += size
                    else:
                        copies[kind]['bytes_complete'] = False
    for row in copies.values():
        if not row['bytes_complete']:
            row['bytes'] = None
    return {'copies':copies,'synchronization_api_counts':syncs,'kernel_activities':kernels,'cuda_api_counts':apis,
            'scope':'actual profiler trace activities; missing byte metadata stays unavailable; event intervals include idle'}


def read_json_trace(path: Path) -> Any:
    if path.is_symlink() or path.stat().st_size > MAX_OUTPUT:
        raise ValueError('profile trace512MiB cap exceeded')
    return json.loads(path.read_text())['traceEvents']


def child(args: Any) -> int:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from gf_common.card_instance import action_config_v3
    from gf_common.card_instance_rules import card_instance_rules
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_common.structured_tokens import CandidateTokens, FrameTokens, validate_candidate_tokens
    from gf_models.hpo_config import model_config_v4
    from gf_models.structured_artifacts import load_game_inference_v4
    from gf_models.structured_config import InferenceBindingV2
    request = read_json(args.output/'request.json')
    if request['code'] != code_files() or request['input_manifest_sha256'] != sha256(args.input/'manifest.json'):
        raise ValueError('immutable replay launch code/input changed')
    manifest = read_json(args.input/'manifest.json')
    check_files({str(args.input/name):value for name,value in manifest['files'].items()})
    actions = action_config_v3(manifest['actions'])
    model = model_config_v4(manifest['runtime_model'],layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    import torch
    loaded = []
    load_started = time.perf_counter()
    for checkpoint in manifest['checkpoints']:
        network, _ = load_game_inference_v4(args.input/f"G{checkpoint['generation']}",
            expected_digest=checkpoint['inference_sha256'],model=model,
            binding=InferenceBindingV2(**{**manifest['inference_binding'],'device':args.device}),
            action_config_hash=manifest['action_config_hash'],action_contract=actions.action_contract,rules=card_instance_rules())
        loaded.append((network,install(network,args.transfer)))
    load_seconds = time.perf_counter()-load_started
    batches = read_json(args.input/'inputs.json')['batches']
    sampling = read_json(args.input/'sampling.json')['batches']
    tokenized = []
    for batch in batches:
        sequences = []
        for sequence in batch['sequences']:
            frames = [FrameTokens(**tuples(frame)) for frame in sequence['frames']]
            tables = [CandidateTokens(**tuples(table)) for table in sequence['candidates']]
            for frame, table in zip(frames,tables,strict=True):
                validate_candidate_tokens(table,frame,model.max_positions,layout=CORRECTED_CARD_INSTANCE_LAYOUT)
            sequences.append((frames,tables))
        tokenized.append(sequences)
    if args.device != 'cpu':
        torch.cuda.reset_peak_memory_stats()

    def sync() -> None:
        if args.device != 'cpu':
            torch.cuda.synchronize(args.device)

    def decision(network: Any, experiment: Any, frame: Any, table: Any, cache: Any, rng: Any) -> None:
        context,cards = network.step(frame,cache)
        logs,_ = network.distribution(context,cards,table)
        value = network.value(context)
        sync()  # Original final-forward consumption fence.
        if experiment is not None and hasattr(experiment,'consumed'):
            experiment.consumed()
        if not torch.isfinite(logs).all() or not torch.isfinite(value):
            raise ValueError('nonfinite replay inference')
        probabilities = logs.exp().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError('nonfinite replay original CPU sampling probabilities')
        selected = int(torch.multinomial(probabilities,1,generator=rng).item())
        # Preserve original sampled logprob/value CPU returns.
        float(logs[selected].cpu()); float(value.cpu())

    rows, traces = [], []
    first_call = None
    with torch.no_grad():
        if args.profile:
            endpoints = manifest['profile_endpoints']
            for ordinal,(batch,ep,end) in enumerate(endpoints):
                network,experiment = loaded[batch]
                frames,tables = tokenized[batch][ep]
                cache = network.new_cache()
                rng = torch.Generator(device='cpu').manual_seed(sampling[batch][ep]['sampling_seed'])
                for index in range(end):
                    decision(network,experiment,frames[index],tables[index],cache,rng)
                sync()
                activities = [torch.profiler.ProfilerActivity.CPU]
                if args.device != 'cpu':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as profiler:
                    decision(network,experiment,frames[end],tables[end],cache,rng)
                path = args.output/f'trace-{ordinal:02}.json'
                profiler.export_chrome_trace(str(path))
                traces.append(path)
                if sum(p.stat().st_size for p in traces) > MAX_OUTPUT:
                    raise ValueError('aggregate profile trace512MiB cap exceeded')
                cache.reset()
                if experiment is not None:
                    experiment.reset()
        else:
            for loop in range(args.warm_loops+1):
                sync()
                started,cpu_started,thread_started = time.perf_counter(),time.process_time(),time.thread_time()
                begin = torch.cuda.Event(enable_timing=True) if args.device != 'cpu' else None
                end_event = torch.cuda.Event(enable_timing=True) if begin is not None else None
                if begin is not None:
                    begin.record()
                count = candidates = 0
                for batch,sequences in enumerate(tokenized):
                    network,experiment = loaded[batch]
                    for ep,(frames,tables) in enumerate(sequences):
                        cache = network.new_cache()
                        rng = torch.Generator(device='cpu').manual_seed(sampling[batch][ep]['sampling_seed'])
                        for frame,table in zip(frames,tables,strict=True):
                            call = time.perf_counter()
                            decision(network,experiment,frame,table,cache,rng)
                            if first_call is None:
                                first_call = time.perf_counter()-call
                            count += 1
                            candidates += len(table.kinds)
                        cache.reset()
                        if experiment is not None:
                            experiment.reset()
                if end_event is not None:
                    end_event.record()
                sync()
                rows.append({'loop':loop,'population':'warmup-first' if loop == 0 else 'warm',
                    'host_wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu_started,
                    'thread_cpu_seconds':time.thread_time()-thread_started,'decisions':count,'candidates':candidates,
                    'cuda_stream_event_ms':None if begin is None or end_event is None else begin.elapsed_time(end_event)})
    immutable_json(args.output/'result.json',{'schema_version':1,'status':'PROFILED' if args.profile else 'REPLAYED',
        'profiled':args.profile,'loops':rows,'first_call_host_seconds':first_call,'cold_load_install_seconds':load_seconds,
        'profile':trace_summary(traces) if args.profile else {'status':'NOT_RUN'},
        'trace_bytes':sum(p.stat().st_size for p in traces),'transfer':args.transfer,'device':args.device,
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'threads':torch.get_num_threads(),
        'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'cuda_peak_allocated':None if args.device == 'cpu' else torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved':None if args.device == 'cpu' else torch.cuda.max_memory_reserved(),
        'experiments':[None if exp is None else exp.diagnostics() for _,exp in loaded],
        'cpu_scope':'direct process/thread time includes runtime and driver busy-wait; not certified useful work',
        'cuda_event_scope':'stream interval includes host idle; kernel activities reported only by separate profiles',
        'qualification':'diagnostic learner-only replay, not all neural actor decisions or learning quality'})
    return 0


def run(args: Any) -> int:
    args.input = args.input.resolve(strict=True)
    output = namespace(args.output)
    if not 1 <= args.warm_loops <= 10 or not 1 <= args.timeout <= 600:
        raise ValueError('bounded1..10warm loops and <=600s deadline required')
    manifest = read_json(args.input/'manifest.json')
    check_files({str(args.input/name):value for name,value in manifest['files'].items()})
    check_files(manifest['simulator_files'])
    runtime = runtime_identity()
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'request.json',{'schema_version':1,'input':str(args.input),
        'input_manifest_sha256':sha256(args.input/'manifest.json'),'transfer':args.transfer,'code':code_files(),
        'profile':args.profile,'device':args.device,'warm_loops':args.warm_loops,'timeout':args.timeout,
        'pair':args.pair,'position':args.position,'population':manifest['population'],'decisions':manifest['decisions'],
        'cold_state':'fresh process; original G0/G1 restored once before separately labeled warmup',
        'runtime':runtime,'simulator_files':manifest['simulator_files'],
        'output_namespace':str(output),'production_identity_scope':'base-only'})
    command = [sys.executable,str(Path(__file__).resolve()),'_child','--input',str(args.input),
               '--output',str(output),'--transfer',args.transfer,'--device',args.device,'--warm-loops',str(args.warm_loops)]
    if args.profile:
        command.append('--profile')
    start,failure = time.monotonic(),None
    with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-start > args.timeout or tree_bytes(output) > MAX_OUTPUT+64*1024**2:
                    raise TimeoutError('replay process deadline/output cap exceeded')
                try:
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:  # noqa: BLE001 - always retire and reap the owned child
            failure = {'type':type(error).__name__,'message':str(error)[:2048]}
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
            process.wait()
        code = process.wait()
    immutable_json(output/'receipt.json',{'pid':process.pid,'returncode':code,'failure':failure,
        'actual_child_waited':True,'controller_seconds':time.monotonic()-start})
    print(json.dumps({'output':str(output),'returncode':code,'failure':failure}))
    return 2 if code or failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    for name in ('run','_child'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--input',type=Path,required=True)
        cmd.add_argument('--output',type=Path,required=True)
        cmd.add_argument('--transfer',choices=TRANSFERS,required=True)
        cmd.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
        cmd.add_argument('--warm-loops',type=int,default=1)
        cmd.add_argument('--profile',action='store_true')
        if name == 'run':
            cmd.add_argument('--pair',type=int,choices=(1,2,3),required=True)
            cmd.add_argument('--position',type=int,choices=(1,2),required=True)
            cmd.add_argument('--timeout',type=int,default=600)
    args = parser.parse_args()
    return {'prepare':prepare,'run':run,'_child':child}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
