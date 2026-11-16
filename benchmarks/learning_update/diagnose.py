"""Separate bounded profiler/shape runs; never used to select timed candidates."""
from __future__ import annotations

import argparse
import resource
from pathlib import Path

from benchmarks.learning_update.common import MAX_OUTPUT, immutable_json, read_json, sha256
from benchmarks.learning_update.compare import update
from benchmarks.learning_update.diagnostic import load_input, restore_start
from gf_training.hpo_update_diagnostics import ProfileWindow


def main() -> None:
    from gf_training.hpo_training_execution import TrainingExecutionSpec
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',choices=('mlp','gru','transformer'),required=True)
    parser.add_argument('--batch',type=int,choices=(0,1),required=True)
    parser.add_argument('--variant',choices=('baseline','metrics','packed','microbatch','shared-prefix'),required=True)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--groups',type=int,default=1)
    parser.add_argument('--shapes',action='store_true')
    args=parser.parse_args()
    if not 0<=args.start<=1000000 or not 1<=args.groups<=16:
        raise ValueError('bounded predeclared profiler window required')
    args.output.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_OUTPUT,MAX_OUTPUT))
    manifest=read_json(args.input)
    immutable_json(args.output/'request.json',{'schema_version':1,'input_sha256':sha256(args.input),
        'model':args.model,'batch':args.batch,'variant':args.variant,'start':args.start,
        'groups':args.groups,'shapes':args.shapes,'timing_eligible':False,
        'excluded':manifest['core_excludes'],'trace_limit_bytes':MAX_OUTPUT})
    learner,rollout,start=load_input(manifest,args.model,args.batch)
    learner._training_execution=TrainingExecutionSpec(variant=args.variant)
    restore_start(learner,start)
    window=ProfileWindow(learner,args.output,args.start,args.groups,args.shapes)
    try:
        with window.patches():
            metrics=update(learner,rollout)
            window.finish()
        if not window.finished:
            raise ValueError('predeclared profiler window was not reached')
        immutable_json(args.output/'result.json',{'status':'complete',**window.summary,'metrics':metrics})
    except BaseException as error:
        window.finish()
        immutable_json(args.output/'failure.json',{'status':'failed','type':type(error).__name__,
            'message':str(error)})
        raise


if __name__=='__main__':
    main()
