"""Clean CPU broker. Only its official TorchRL child owns CUDA model execution."""
from __future__ import annotations

import os
import sys

sys.argv[:]=['gf-shared-inference-broker']
os.environ.clear()
os.environ.update({'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONHASHSEED':'0',
    'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})

import math
import multiprocessing as mp
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from functools import partial
from typing import Any

from gf_common.domain import ContractError, closed, digest
from gf_common.ram_bank import RamBankChannel
from gf_common.shared_inference import ServingBudget, shared_dependency_versions, shared_runtime
from gf_common.shared_requests import RequestAuthority
from gf_simulator_client.framing import read_frame, read_frame_admitted, write_frame


def main() -> None:
    output,source=sys.stdout.buffer,sys.stdin.buffer
    server=process=transport=channel=None
    streams=[]
    executor=None
    diagnostics=measurement=profile_status=None
    try:
        initial=read_frame(source)
        closed(initial,{'service_id','execution','job_id','run_id','config_sha256','epoch','runtime','device','visibility','endpoints','bank_fd','bank_invocation','host_bytes','cleanup_deadline'}, {'performance_diagnostics'})
        runtime=shared_runtime(initial['runtime'],game_concurrency=1)
        channel=RamBankChannel._from_inherited(initial['bank_fd'],initial['bank_invocation'])
        budget=ServingBudget(initial['host_bytes'])
        libraries=shared_dependency_versions()
        import torch
        from tensordict import NonTensorData, TensorDict
        from torchrl.modules.inference_server import InferenceServer, MPTransport

        from gf_models.shared_numeric import tensor_row
        from gf_models.shared_telemetry import make_served_registry, read_allocator
        performance = initial.get('performance_diagnostics')
        diagnostics = profile_status = profile_callback = None
        profile_reported = False
        profile_target_scope = profile_owner = None
        measurement = None
        response_window: Any = None
        ctx=mp.get_context('spawn')
        if performance is not None:
            from pathlib import Path

            from gf_simulator_client.performance_diagnostics import Diagnostics
            diagnostics = Diagnostics(Path(performance['directory']), performance['observation_id'], role='legacy-shared-broker', profile_recipe=performance['profile_recipe'])
            if performance['profile_recipe'] == 'forward-window-v1':
                from gf_simulator_client.performance_profile import ForwardProfile, ProfileStatus
                profile_status = ProfileStatus.shared(ctx)
                profile_callback = ForwardProfile(diagnostics.directory, diagnostics.observation_id, profile_status, role='legacy-gpu-worker')
        def publish_profile() -> bool:
            nonlocal profile_reported
            if diagnostics is None or profile_status is None or profile_reported or profile_target_scope is None:
                return True
            from gf_simulator_client.performance_profile import missing_profile_result
            profile_status.end()
            result = profile_status.read()['result']
            if result is None:
                assert profile_owner is not None
                result = missing_profile_result(observation_id=diagnostics.observation_id,
                    owner=profile_owner, clock=diagnostics.clock, scope=profile_target_scope,
                    boundary='legacy-post-transport-forward', torch_version=torch.__version__)
            if diagnostics.write(result) is None:
                return False
            profile_reported = True
            return True
        diagnostic_block=ctx.Array('Q',8,lock=ctx.Lock())
        transport=MPTransport(ctx=ctx)
        clients=[transport.client() for _ in initial['endpoints']]
        server=InferenceServer(policy_factory=partial(make_served_registry,diagnostic_block=diagnostic_block,device=initial['device'],
            cache_entries=runtime['cache_entries'],cache_weight_bytes=runtime['cache_weight_bytes'],visibility=initial['visibility'], performance_callback=profile_callback),
            transport=transport,service_backend='process',service_backend_options={'mp_context':ctx,'startup_timeout':30.},
            policy_device=initial['device'],output_device='cpu',max_batch_size=runtime['max_batch_size'],
            min_batch_size=runtime['max_batch_size'],timeout=.001)
        server.start()
        process=server._process  # Retain original object before official shutdown may clear it.
        if profile_status is not None:
            from gf_simulator_client.performance_diagnostics import process_identity
            profile_owner = process_identity(process.pid, 'legacy-gpu-worker')
        registration={'schema_version':1,'kind':'shared-service-registration','service_id':initial['service_id'],
            'execution':initial['execution'],'job_id':initial['job_id'],'run_id':initial['run_id'],
            'config_sha256':initial['config_sha256'],'libraries':libraries,'runtime_sha256':digest(initial['runtime']),'epoch':initial['epoch'],
            'broker_pid':os.getpid(),'gpu_pid':process.pid,'device':initial['device']}
        authority=RequestAuthority(initial['execution'])
        lock=threading.Lock()
        stopped=threading.Event()
        counters={'publications':0,'requests':0,'failures':0,'window_peak_pending':0}
        active: dict[str,threading.Event]={}
        deadlines: dict[str,float]={}
        waits: deque[float]=deque(maxlen=4096)
        batches: deque[int]=deque(maxlen=4096)
        endpoints={str(index):fd for index,fd in enumerate(initial['endpoints'])}
        executor=ThreadPoolExecutor(max_workers=len(endpoints),thread_name_prefix='shared-relay')

        def relay(index: int,fd: int) -> None:
            endpoint=str(index)
            sock=socket.socket(fileno=fd)
            stream=sock.makefile('rwb',buffering=0)
            streams.append((sock,stream))
            event=None
            message=row=data=response=answer=future=None
            def decoding_started() -> None:
                nonlocal event
                event=threading.Event()
                with lock:
                    active[endpoint]=event
                    if endpoint not in authority.pending or time.monotonic()>=deadlines[endpoint]:
                        raise TimeoutError('shared request retired or expired')
            try:
                while not stopped.is_set():
                    try:
                        message=read_frame_admitted(stream,max_owned_bytes=initial['host_bytes'],on_frame=decoding_started)
                        closed(message,{'routing','numeric'})
                        with lock:authority.match(endpoint,message['routing'])
                        accepted=time.monotonic()
                        row=tensor_row(message['numeric'])
                        data=TensorDict({'routing':NonTensorData(message['routing']),'inputs':row},batch_size=[])
                        # Deadline expiry invalidates delivery; actual submitted work
                        # remains owned until completion or confirmed service death.
                        future=clients[index].submit(data)
                        response=future.result()
                        with lock:
                            authority.match(endpoint,response.get_non_tensor('routing'))
                            if time.monotonic()>=deadlines[endpoint]:raise TimeoutError('shared reply expired')
                            counters['requests']+=1
                            waits.append(max(0.,(response['forward_started'].item()-accepted)*1000))
                            batches.append(int(response['transport_batch_size'].item()))
                            answer={'routing':response.get_non_tensor('routing'),'logs':response['logs'].tolist(),
                                'probabilities':response['probabilities'].tolist(),'value':response['value'].item()}
                        write_frame(stream,answer)
                        if measurement is not None:
                            with lock:
                                response_window.delivered(endpoint, int(response['transport_batch_size'].item()))
                    finally:
                        message=row=data=response=answer=future=None
                        if event is not None:
                            with lock:
                                active.pop(endpoint,None)
                                event.set()
                            event=None
            except (EOFError,OSError):
                if not stopped.is_set():
                    with lock:
                        counters['failures']+=1
                        if response_window is not None: response_window.failure()
            except BaseException:  # noqa: BLE001 - redact other-row/library error details.
                if not stopped.is_set():
                    with lock:
                        counters['failures']+=1
                        if response_window is not None: response_window.failure()
                    try:write_frame(stream,{'error':'shared inference request failed'})
                    except OSError:pass
            finally:
                stream.close();sock.close()

        futures=[executor.submit(relay,index,fd) for index,fd in enumerate(initial['endpoints'])]
        write_frame(output,registration)
        while True:
            control=read_frame(source)
            command=control.get('command')
            if command=='AUTHORIZE':
                closed(control,{'command','endpoint','routing','deadline'})
                row=control['routing']
                if type(control['deadline']) not in (int,float) or not math.isfinite(control['deadline']):
                    raise ContractError('original request deadline required')
                if time.monotonic()>=control['deadline']:
                    if response_window is not None: response_window.failure()
                    write_frame(output,{'accepted':False,'reason':'expired'});continue
                with lock:
                    if row['execution']!=initial['execution']:raise ContractError('foreign serving invocation')
                    generated=authority.begin(control['endpoint'],game=row['game'],participant=row['participant'],revision=row['revision'],checkpoint=row['checkpoint'])
                    # The actual coordinator owns the monotonic request ID.
                    authority.pending[control['endpoint']]=dict(row)
                    deadlines[control['endpoint']]=control['deadline']
                    if response_window is not None:
                        response_window.authorize(control['endpoint'])
                    counters['window_peak_pending']=max(counters['window_peak_pending'],len(authority.pending))
                    del generated
                write_frame(output,{'accepted':True})
            elif command=='RETIRE':
                closed(control,{'command','endpoint','completed'})
                with lock:
                    authority.retire(control['endpoint'],completed=control['completed'])
                    completion=active.get(control['endpoint'])
                if control['completed'] and completion is not None:
                    # Result delivery has completed; buffer retirement is cleanup,
                    # not a renewed policy request deadline. Keep original wall cap.
                    remaining=30. if initial['cleanup_deadline'] is None else max(0.,initial['cleanup_deadline']-time.monotonic())
                    if not completion.wait(remaining):raise ContractError('original relay buffer retirement unconfirmed')
                if response_window is not None:
                    with lock:
                        response_window.retire(control['endpoint'], completed=control['completed'])
                write_frame(output,{'accepted':True})
            elif command=='PUBLISH':
                closed(control,{'command','metadata','bank_count','generation','identity'})
                deliveries=[]
                try:
                    for index in range(control['bank_count']):
                        delivery=channel.receive(purpose='model',generation=control['generation'],
                            identity=control['identity']+f':{index}',budget=budget,max_bytes=runtime['cache_weight_bytes'])
                        deliveries.append(delivery)
                    weights={descriptor.name:torch.from_numpy(delivery.bank.array(descriptor.name).copy())
                        for delivery in deliveries for descriptor in delivery.bank.descriptors}
                    publication=TensorDict({'publication':NonTensorData(control['metadata']),
                        'weights':TensorDict(weights,batch_size=[])},batch_size=[])
                    if server.update_model_weights(publication,timeout=30.)!={'accepted':True}:
                        raise ContractError('ambiguous shared publication acknowledgement')
                    del weights,publication
                    for delivery in deliveries:delivery.complete(lambda:None)
                    counters['publications']+=1
                    write_frame(output,{'accepted':True,'checkpoint':control['metadata']['checkpoint'],'publication':counters['publications']})
                finally:
                    channel.retire_received(lambda:None)
            elif command=='EVICT':
                closed(control,{'command','checkpoint'})
                with lock:
                    if any(row['checkpoint']==control['checkpoint'] for row in authority.pending.values()):
                        raise ContractError('shared eviction overlaps accepted checkpoint')
                server.update_model_weights(TensorDict({'publication':NonTensorData(control)},batch_size=[]),timeout=30.)
                write_frame(output,{'accepted':True,'checkpoint':control['checkpoint']})
            elif command in ('DIAGNOSTIC_BEGIN', 'DIAGNOSTIC_END'):
                # Health/framing/transport failures retain the existing infrastructure boundary.
                if not server.is_alive:
                    raise ContractError('original inference service health failed')
                try:
                    if command == 'DIAGNOSTIC_END' and profile_status is not None:
                        profile_status.end()
                    closed(control, {'command', 'scope'})
                    accepted = False
                    # This optional channel never changes scientific/service failure state.
                    with lock:
                        quiet = authority.empty and not active
                        if diagnostics is not None and quiet:
                            if command == 'DIAGNOSTIC_BEGIN' and measurement is None:
                                from gf_common.performance_metrics import (
                                    PhaseMetrics,
                                    ResponseWindow,
                                )
                                measurement = PhaseMetrics(scope=control['scope'], start_ns=time.monotonic_ns(),
                                    counters=('inference_issued', 'inference_successes', 'inference_failures'), histograms=())
                                response_window = ResponseWindow()
                                if profile_status is not None and profile_target_scope is None:
                                    profile_target_scope = dict(control['scope'])
                                    profile_status.begin(control['scope'])
                                accepted = True
                            elif command == 'DIAGNOSTIC_END' and measurement is not None and measurement.scope == control['scope']:
                                measurement.batch_rows.update(response_window.snapshot(quiescent_end=True))
                                measurement.counters.update(inference_issued=response_window.issued,
                                    inference_successes=response_window.completed, inference_failures=response_window.failures)
                                if measurement.batch_rows['batch_size_counts'] is None:
                                    measurement.error('incomplete_full_response_coverage')
                                if diagnostics.metrics(measurement) is None:
                                    raise OSError('diagnostic metrics publication failed')
                                measurement = response_window = None
                                if not publish_profile():
                                    raise OSError('diagnostic profile publication failed')
                                accepted = True
                    if not accepted:
                        raise ValueError('diagnostic scope is not current and positively quiescent')
                except Exception as error:  # noqa: BLE001 - only fully framed diagnostic processing is contained.
                    with lock:
                        if measurement is not None:
                            measurement.error('diagnostic_processing_failed')
                        if response_window is not None:
                            response_window.coverage = False  # Diagnostic loss is not an inference failure.
                        if profile_status is not None:
                            try:
                                profile_status.end()
                            except Exception:  # noqa: BLE001 - status loss is missing evidence, never authority.
                                if diagnostics is not None:
                                    diagnostics.error('diagnostic_profile_end_failed')
                        # Never leave a partially initialized idle window that a later BEGIN can qualify.
                        # Active relays keep their existing observation objects until they settle.
                        if authority.empty and not active:
                            measurement = response_window = None
                    if diagnostics is not None:
                        diagnostics.error('diagnostic_processing:' + type(error).__name__)
                    accepted = False
                write_frame(output, {'accepted': accepted})
            elif command=='METRICS_RESET':
                closed(control,{'command'})
                with lock:
                    if not authority.empty or active:raise ContractError('metrics reset overlaps accepted requests or buffers')
                    waits.clear();batches.clear();counters['window_peak_pending']=0
                write_frame(output,{'accepted':True})
            elif command in ('QUIESCE','CLOSE','ABORT_CLOSE'):
                closed(control,{'command'}|({'deadline'} if command!='QUIESCE' else set()))
                with lock:
                    proof={'accepted':authority.empty and not active and counters['failures']==0 and server.is_alive,
                        'pending':len(authority.pending),'active_relays':len(active),
                        'publications':counters['publications'],'requests':counters['requests'],
                        'cuda_allocator':read_allocator(diagnostic_block),'failures':counters['failures'],'transport_batches':list(batches),'accept_to_forward_ms':list(waits),
                        'sample_scope':'last4096requests','tensor_batch_size':1,'window_peak_pending':counters['window_peak_pending']}
                if command=='QUIESCE':
                    proof['transport_stats']=server.stats() if server.is_alive else {}
                    write_frame(output,proof);continue
                if measurement is not None and diagnostics is not None:
                    measurement.error('scope_interrupted')
                    if response_window is not None:
                        measurement.batch_rows.update(response_window.snapshot(quiescent_end=False))
                    diagnostics.metrics(measurement, complete=False)
                    measurement = response_window = None
                    if profile_status is not None: profile_status.end()
                abort=command=='ABORT_CLOSE'
                if not abort and not proof['accepted']:
                    write_frame(output,proof);continue
                deadline=control['deadline']
                if initial['cleanup_deadline'] is not None:deadline=min(deadline,initial['cleanup_deadline'])
                if abort:
                    with lock:
                        for endpoint in tuple(authority.pending):authority.retire(endpoint)
                stopped.set()
                for sock,_ in tuple(streams):
                    try:sock.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
                # Shutting down the peer first releases actually submitted futures.
                server.shutdown(timeout=max(0.,min(10.,deadline-time.monotonic())))
                process.join(max(0.,deadline-time.monotonic()))
                if process.is_alive() or process.exitcode is None:raise ContractError('original shared GPU child reap unconfirmed')
                publish_profile()
                if wait_futures(futures,timeout=max(0.,deadline-time.monotonic())).not_done:
                    raise ContractError('original relay retirement unconfirmed')
                executor.shutdown(wait=True,cancel_futures=False)
                for future in futures:future.result()
                with lock:
                    if active:raise ContractError('private relay buffers remain live')
                transport.close();channel.close()
                retirement={**proof,'accepted':True,'pending':0,'active_relays':0,
                    'schema_version':1,'kind':'shared-service-retirement','successful':not abort and process.exitcode==0,
                    'service_id':initial['service_id'],'registration_sha256':digest(registration),
                    'broker_pid':os.getpid(),'gpu_pid':process.pid,'gpu_exitcode':process.exitcode,'relays_joined':True}
                write_frame(output,retirement)
                return
            else:raise ContractError('unknown shared serving control')
    except BaseException:  # noqa: BLE001 - private control boundary; no policy input details.
        if diagnostics is not None and measurement is not None:
            measurement.error('broker_interrupted')
            diagnostics.metrics(measurement, complete=False)
            if profile_status is not None: profile_status.end()
        try:write_frame(output,{'error':'shared serving infrastructure failed'})
        except (OSError,ValueError):pass
        # No reap receipt on failed/ambiguous cleanup. Parent retains custody.
        if server is not None:server.shutdown(timeout=5.)
        if process is not None:
            process.join(5.)
            if profile_status is not None and not process.is_alive() and process.exitcode is not None:
                publish_profile()
        os._exit(1)


if __name__=='__main__':main()
