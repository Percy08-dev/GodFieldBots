"""Write fresh fixed-protocol configs and exactly18 alternating actual trials."""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

from benchmarks.learning_update.common import REPO, immutable_json, read_json, sha256


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True,
        help='root-selected JSON mapping mlp/gru/transformer to one measured candidate each')
    args=parser.parse_args()
    root=REPO/'runs/learning-update-optimization-v1'
    candidates=read_json(args.candidates)
    if set(candidates)!={'mlp','gru','transformer'}:
        raise ValueError('one explicitly selected candidate per model required')
    configs=root/'configs'
    configs.mkdir(parents=True,exist_ok=False)
    trials: list[dict[str,Any]]=[]
    for kind in ('mlp','gru','transformer'):
        allowed=('metrics','packed','microbatch')+(() if kind=='mlp' else ('shared-prefix',))
        if candidates[kind] not in allowed:
            raise ValueError('candidate must be an applicable nonbaseline measured variant')
        original=REPO/f'runs/inference-comparison-v1/configs/{kind}.toml'
        path=configs/f'{kind}.toml'
        raw=original.read_text().replace('runs/inference-comparison-v1','runs/learning-update-optimization-v1')
        raw=raw.replace('stage = "inference-comparison-','stage = "update-comparison-')
        with path.open('x') as stream:
            stream.write(raw)
        for pair in range(3):
            for route in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
                run_id,context_id=str(uuid.uuid4()),str(uuid.uuid4())
                index=len(trials)+1
                trials.append({'index':index,'model':kind,'pair':pair,'route':route,
                    'variant':'baseline' if route=='baseline' else candidates[kind],
                    'config':str(path),'config_sha256':sha256(path),'run_id':run_id,'context_id':context_id,
                    'parent':str(root/'parents'/run_id),'original':str(root/'originals'/run_id),
                    'report':str(root/'actual'/f'{index:02d}-{kind}-{route}-{run_id}')})
    immutable_json(root/'actual-plan.json',{'schema_version':1,'kind':'update-actual-population',
        'trials':trials,'candidate_selection_sha256':sha256(args.candidates),
        'execution':'sequential; no concurrent tests/audits/reviews; audit only after timing',
        'expected_runs':18,'quality_eligible':False})


if __name__=='__main__':
    main()
