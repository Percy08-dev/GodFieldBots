"""Opt-in bounded update profiling. Imported only for explicitly profiled runs."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

MAX_OUTPUT=512*1024**2


def _trace_events(path: Path) -> Any:
    """Stream Chrome events with a2MiB per-record/preamble bound.

    A512MiB trace must not be decoded into an unbounded Python object graph.
    Torch's optional post-event metadata need not be retained to count events.
    """
    import json
    decoder=json.JSONDecoder()
    with path.open() as stream:
        buffer=''
        eof=False
        def fill() -> None:
            nonlocal buffer,eof
            chunk=stream.read(65536)
            eof=not chunk
            buffer+=chunk
            if len(buffer)>2*1024**2:
                raise ValueError('trace record/preamble exceeds2MiB counting bound')
        def peek() -> str:
            nonlocal buffer
            buffer=buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer=buffer.lstrip()
            return buffer[:1]
        def take(expected: str) -> None:
            nonlocal buffer
            if peek()!=expected:
                raise ValueError('invalid Chrome trace delimiter')
            buffer=buffer[1:]
        def value() -> Any:
            nonlocal buffer
            peek()
            while True:
                try:
                    result,end=decoder.raw_decode(buffer)
                    buffer=buffer[end:]
                    return result
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError('invalid/truncated Chrome trace') from error
                    fill()
        take('{')
        while peek()!='}':
            name=value()
            take(':')
            if name=='traceEvents':
                take('[')
                while peek()!=']':
                    event=value()
                    if type(event) is not dict:
                        raise ValueError('Chrome trace event must be an object')
                    yield event
                    if peek()==',':
                        take(',')
                    elif peek()!=']':
                        raise ValueError('invalid Chrome trace event separator')
                take(']')
                return
            value()
            if peek()==',':
                take(',')
            elif peek()!='}':
                raise ValueError('invalid Chrome trace metadata separator')
        raise ValueError('Chrome traceEvents array unavailable')


def trace_counts(path: Path) -> dict[str,Any]:
    """Observed trace counts, with missing CUDA activity explicitly unavailable."""
    if path.stat().st_size>MAX_OUTPUT:
        raise ValueError('trace exceeds512MiB counting input bound')
    categories: dict[str,int]={}
    event_count=0
    runtime_launch=runtime_copy=runtime_sync=driver_sync=0
    directions={'HtoD':0,'DtoH':0,'DtoD':0,'other':0}
    for event in _trace_events(path):
        event_count+=1
        name=event.get('name','')
        category=event.get('cat','')
        if type(name) is not str or type(category) is not str:
            raise ValueError('invalid Chrome trace category/name')
        cats=category.split(',') if category else []
        for cat in cats:
            categories[cat]=categories.get(cat,0)+1
        if 'cuda_runtime' in cats:
            runtime_launch+=name.startswith(('cudaLaunchKernel','cudaLaunchCooperativeKernel','cudaGraphLaunch'))
            runtime_copy+=name.startswith('cudaMemcpy')
            runtime_sync+=name.startswith(('cudaDeviceSynchronize','cudaStreamSynchronize','cudaEventSynchronize'))
        if 'cuda_driver' in cats:
            driver_sync+=name.startswith(('cuCtxSynchronize','cuStreamSynchronize','cuEventSynchronize'))
        if 'gpu_memcpy' in cats:
            direction=next((key for key in ('HtoD','DtoH','DtoD') if key.lower() in name.lower()),'other')
            directions[direction]+=1
    device_observed=any(categories.get(cat,0) for cat in ('kernel','gpu_memcpy','gpu_memset'))
    runtime_observed=categories.get('cuda_runtime',0)>0
    driver_observed=categories.get('cuda_driver',0)>0
    return {'status':'observed' if device_observed else 'CUDA_ACTIVITY_UNAVAILABLE',
        'event_count':event_count,
        'cuda_event_count':sum(categories.get(cat,0) for cat in ('kernel','gpu_memcpy','gpu_memset')) if device_observed else None,
        'scope':'Chrome trace categories; explicit launch/copy/*Synchronize runtime APIs, not inferred I/O waits',
        'category_event_counts':categories,'missing_cuda_activity':not device_observed,
        'kernel_count':categories.get('kernel',0) if device_observed else None,
        'device_copy_count':categories.get('gpu_memcpy',0) if device_observed else None,
        'device_memset_count':categories.get('gpu_memset',0) if device_observed else None,
        'device_copy_directions':directions if device_observed else None,
        'runtime_kernel_launch_count':runtime_launch if runtime_observed else None,
        'runtime_copy_count':runtime_copy if runtime_observed else None,
        'runtime_sync_count':runtime_sync if runtime_observed else None,
        'driver_sync_count':driver_sync if driver_observed else None}


class ProfileWindow:
    def __init__(self,learner: Any,output: Path,start: int,groups: int,shapes: bool) -> None:
        self.learner,self.output=learner,output
        self.start,self.groups,self.shapes=start,groups,shapes
        self.seen=0
        self.profile: Any=None
        self.active=False
        self.finished=False
        self.summary: dict[str,Any]={}

    def boundary(self) -> None:
        if self.active and self.seen>=self.start+self.groups:
            self.finish()
        if self.seen==self.start and not self.finished:
            torch=self.learner.torch
            activities=[torch.profiler.ProfilerActivity.CPU]
            if self.learner.binding.device!='cpu':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profile=torch.profiler.profile(activities=activities,record_shapes=self.shapes,
                profile_memory=False,with_stack=False)
            self.profile.__enter__()
            self.active=True
        self.seen+=1

    def finish(self) -> None:
        if not self.active:
            return
        self.profile.__exit__(None,None,None)
        self.active=False
        self.finished=True
        trace=self.output/'trace.json'
        # The diagnose process has RLIMIT_FSIZE before export; oversize traces
        # cannot consume unbounded disk. A partial trace remains failed evidence.
        self.profile.export_chrome_trace(str(trace))
        if trace.stat().st_size>MAX_OUTPUT:
            raise ValueError('profiler trace exceeded512MiB bound')
        with trace.open('rb') as stream:
            import hashlib
            trace_sha = hashlib.file_digest(stream,'sha256').hexdigest()
        try:
            counts=trace_counts(trace)
        except ValueError as error:
            counts={'status':'UNAVAILABLE','reason':str(error),'missing_cuda_activity':None}
        self.summary={'trace_counts':counts,'captured_groups':min(self.groups,max(0,self.seen-self.start)),
            'event_count':counts.get('event_count'),'cuda_event_count':counts.get('cuda_event_count'),
            'missing_cuda_activity':counts.get('missing_cuda_activity'),
            'event_count_scope':'streamed Chrome trace events; CUDA categories kernel/gpu_memcpy/gpu_memset',
            'trace_bytes':trace.stat().st_size,'trace_sha256':trace_sha}
        self.profile=None

    def label(self,name: str,original: Any) -> Any:
        def wrapped(*args: Any,**kwargs: Any) -> Any:
            if not self.active:
                return original(*args,**kwargs)
            with self.learner.torch.profiler.record_function(name):
                return original(*args,**kwargs)
        return wrapped

    def patches(self) -> ExitStack:
        from gf_training import hpo_numeric, hpo_update_execution
        stack=ExitStack()
        network=self.learner.network
        # Baseline keeps its original endpoint-forward boundary. Experimental
        # windows start before admission/packing/HtoD at the trainer group entry;
        # the next entry closes capture BEFORE the next group's packing.
        spec=getattr(self.learner,'_training_execution',None)
        owner,name=(network,'_forward_calculate') if spec is None or spec.variant=='baseline' else (hpo_update_execution,'group_forward')
        original_group=getattr(owner,name)
        def boundary(*args: Any,**kwargs: Any) -> Any:
            self.boundary()
            return original_group(*args,**kwargs)
        stack.enter_context(patch.object(owner,name,boundary))
        for name,label in (('_encode_frame_operands','frame encoder'),
                ('_candidate_chunk_calculate','candidate'),('value','critic')):
            stack.enter_context(patch.object(network,name,self.label(label,getattr(network,name))))
        original=network.encode_temporal
        def temporal(*args: Any,**kwargs: Any) -> Any:
            label='temporal suffix' if self.learner.torch.is_grad_enabled() else 'temporal prefix'
            return self.label(label,original)(*args,**kwargs)
        stack.enter_context(patch.object(network,'encode_temporal',temporal))
        for owner,name,label in ((hpo_update_execution,'pack_window','input packing/HtoD'),
                (hpo_numeric,'hpo_losses','loss'),
                (hpo_update_execution,'endpoint_terms','metric/checks'),
                (self.learner.torch,'isfinite','finite checks'),
                (self.learner.torch.Tensor,'backward','backward'),
                (self.learner.torch.nn.utils,'clip_grad_norm_','clip'),
                (self.learner.optimizer,'step','Adam'),(self.learner,'synchronize','final sync')):
            stack.enter_context(patch.object(owner,name,self.label(label,getattr(owner,name))))
        return stack
