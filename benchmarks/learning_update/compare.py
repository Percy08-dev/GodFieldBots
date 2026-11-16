"""84 fixed-state diagnostic updates; warmups and profiling are separate evidence."""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any, Self

from benchmarks.learning_update.common import (
    check_files,
    code_files,
    immutable_json,
    read_json,
    sha256,
)
from benchmarks.learning_update.diagnostic import DiagnosticRollout, load_input, restore_start


def process_memory() -> dict[str,int]:
    values={}
    with Path('/proc/self/smaps_rollup').open() as stream:
        for line in stream:
            name,_,number=line.partition(':')
            if name in ('Pss','Rss'):
                values[name.lower()+'_bytes']=int(number.split()[0])*1024
    return values


class MemorySampler:
    def __init__(self) -> None:
        self.stop=threading.Event()
        self.peak: dict[str,int]={}
        self.samples=0
        self.failure: str | None=None
        self.thread=threading.Thread(target=self._run,daemon=True)

    def _run(self) -> None:
        try:
            while not self.stop.is_set():
                for name,value in process_memory().items():
                    self.peak[name]=max(self.peak.get(name,0),value)
                self.samples+=1
                self.stop.wait(.05)
        except OSError as error:
            self.failure=str(error)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self,*args: object) -> None:
        self.stop.set()
        self.thread.join()


def update(learner: Any,rollout: DiagnosticRollout) -> Any:
    from functools import partial
    return learner._update_numeric(rollout,{'status':'not_run-diagnostic-only'},time.perf_counter(),
        partial(rollout.forward,learner.network))


def measure(learner: Any,rollout: DiagnosticRollout) -> dict[str,Any]:
    torch=learner.torch
    if learner.binding.device!='cuda:0':
        raise ValueError('planned measurements require CUDA; CPU checks are not substitutions')
    learner.synchronize()
    torch.cuda.reset_peak_memory_stats(learner.binding.device)
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    with MemorySampler() as sampler:
        begin.record()
        wall,cpu=time.perf_counter(),time.process_time()
        metrics=update(learner,rollout)
        end.record()
        end.synchronize()
        elapsed=time.perf_counter()-wall
        process=time.process_time()-cpu
    diagnostics=getattr(learner,'_training_update_diagnostics',None)
    return {'wall_seconds':elapsed,'process_cpu_seconds':process,
        'cuda_event_ms':begin.elapsed_time(end),'cuda_event_scope':'includes host submission gaps; not pure kernel duration',
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(learner.binding.device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(learner.binding.device),
        'sampled_process_peak':sampler.peak,'memory_samples':sampler.samples,
        'memory_sampling_failure':sampler.failure,'metrics':metrics,'diagnostics':diagnostics}


def compare(manifest_path: Path,output: Path,models: tuple[str,...]) -> None:
    from gf_models.torch_runtime import runtime_metadata
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    manifest=read_json(manifest_path)
    if manifest['kind']!='fixed-update-diagnostic-inputs':
        raise ValueError('prepared fixed-update inputs required')
    output.mkdir(parents=True,exist_ok=False)
    source=code_files()
    request={'schema_version':1,'kind':'fixed-update-core-comparison','input_sha256':sha256(manifest_path),
        'code':source,'models':models,'repeats':3,'ordering':'forward/reverse/forward',
        'warmup':'one warmup per measured variant, restored fully before and after warmup',
        'scope':'same diagnostic NumPy adapter and tensor core, train-only unqualified',
        'excluded':manifest['core_excludes'],'runtime':runtime_metadata()}
    immutable_json(output/'request.json',request)
    rows=[]
    try:
        for kind in models:
            for batch in (0,1):
                learner,rollout,start=load_input(manifest,kind,batch)
                variants=('baseline','metrics','packed','microbatch')+(() if kind=='mlp' else ('shared-prefix',))
                for repeat in range(3):
                    order=variants if repeat%2==0 else tuple(reversed(variants))
                    for variant in order:
                        learner._training_execution=TrainingExecutionSpec(variant=variant)
                        if hasattr(learner,'_training_update_diagnostics'):
                            del learner._training_update_diagnostics
                        restore_start(learner,start)
                        update(learner,rollout)
                        restore_start(learner,start)
                        row={'model':kind,'batch':batch,'repeat':repeat,'variant':variant,
                            'start_steps':learner.steps,'reservation_bytes':learner._training_reservation_bytes,
                            **measure(learner,rollout)}
                        rows.append(row)
                        immutable_json(output/f'{kind}-b{batch}-r{repeat}-{variant}.json',row)
                del learner,rollout
        expected=sum((4 if kind=='mlp' else 5)*6 for kind in models)
        if len(rows)!=expected:
            raise ValueError('fixed protocol measured update count differs')
        check_files(source)
        immutable_json(output/'result.json',{'status':'complete','measured_updates':len(rows),
            'expected_updates':expected,'quality_eligible':False})
    except BaseException as error:
        immutable_json(output/'failure.json',{'status':'failed','type':type(error).__name__,
            'message':str(error),'completed_measured_updates':len(rows)})
        raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=('mlp','gru','transformer'))
    args=parser.parse_args()
    compare(args.input,args.output,('mlp','gru','transformer') if args.model is None else (args.model,))


if __name__=='__main__':
    main()
