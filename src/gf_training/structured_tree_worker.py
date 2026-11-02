"""Clean CPU Tree2 numeric child: only owned request/input files, never Source."""
from __future__ import annotations

import os
import sys
from typing import Any

# This module's bootstrap precedes NumPy and any native-library import.
_ARGUMENTS=tuple(sys.argv[1:])
sys.argv[:]=['gf-structured-tree-fit']
os.environ.clear()
os.environ.update({'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','CUDA_VISIBLE_DEVICES':'',
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})


def main()->None:
    import math
    import time
    from dataclasses import asdict
    from pathlib import Path

    from gf_common.domain import ContractError, canonical, digest
    from gf_models.structured_sealed import SealedScan, checksum
    from gf_models.structured_tree_artifacts import (
        NATIVE_PARAMETERS,
        _native,
        inspect_artifact,
        runtime_metadata,
        save_provisional_trained,
    )
    from gf_models.structured_tree_encoder import feature_layout
    from gf_models.structured_tree_pairing import (
        fit_admission,
        inspect_fit_inputs,
        tree_fit_producer,
        validate_request,
    )
    from gf_training.structured_lineage import _ChainScan, _write
    if len(_ARGUMENTS)!=3:raise ContractError('Tree2 own numeric directory/requestSHA/deadline required')
    root=Path(_ARGUMENTS[0]);expected=checksum(_ARGUMENTS[1]);deadline=float(_ARGUMENTS[2])
    if not math.isfinite(deadline):raise ContractError('Tree2 bounded child deadline required')
    scan=SealedScan(max_owned_bytes=2*1024**3,deadline=deadline)
    request=validate_request(scan.object(root/'request.json',expected))
    if request['producer']!=tree_fit_producer(request['producer']['code_revision']):raise ContractError('Tree2 actual fit producer differs')
    if request['runtime']!=runtime_metadata(deadline=deadline):raise ContractError('Tree2 actual fit runtime differs')
    from gf_models.structured_tree_config import configured_tree_v2
    model=configured_tree_v2({**request['model'],'checkpoint':None,**request['execution']})
    scan.limit=min(scan.limit,model.resources.max_owned_bytes)
    fit_admission(model,request['fit_settings'],request['row_counts'],retained_bytes=scan.retained)
    envelope,blocks,targets=inspect_fit_inputs(root,request,scan,retain=True)
    # Source is never available here. This is consumed own numeric evidence only.
    scan.finalize()
    fit=request['fit_settings'];layout=feature_layout(model.encoder)
    import numpy as np
    fit_admission(model,fit,request['row_counts'],retained_bytes=scan.retained)
    labels={split:np.concatenate(values) for split,values in targets.items()}
    derived=int(digest({'recipe':model.initialization.recipe,'model_seed':request['seed']})[:8],16)%(2**31-1)
    params={**NATIVE_PARAMETERS,'metric':'l2','learning_rate':fit['learning_rate'],'num_leaves':model.network.num_leaves,
        'min_data_in_leaf':fit['min_data_in_leaf'],'num_threads':fit['num_threads'],'device_type':'cpu','verbosity':-1,
        'deterministic':True,'force_col_wise':True,'feature_pre_filter':False,
        'seed':derived,'data_random_seed':derived,'feature_fraction_seed':derived,'bagging_seed':derived}
    native=_native(deadline=deadline)
    train=native.Dataset(blocks['train'],label=labels['train'],feature_name=layout['feature_names'],
        categorical_feature=layout['categorical_indices'],params=params,free_raw_data=True)
    validation=native.Dataset(blocks['validation'],label=labels['validation'],reference=train,
        feature_name=layout['feature_names'],categorical_feature=layout['categorical_indices'],params=params,free_raw_data=True)
    consumed=0
    def count(environment:Any)->None:
        nonlocal consumed
        scan.check();consumed=environment.iteration+1
    count.__dict__['order']=0
    count.__dict__['before_iteration']=False
    booster=native.train(params,train,num_boost_round=fit['num_boost_round'],valid_sets=[validation],valid_names=['validation'],
        callbacks=[count,native.early_stopping(fit['early_stopping_rounds'],verbose=False)])
    scan.check()
    metrics={'configured_rounds':fit['num_boost_round'],'consumed_iterations':consumed,'best_iteration':booster.best_iteration,
        'retained_tree_count':booster.num_trees(),'validation_l2':float(booster.best_score['validation']['l2'])}
    if not math.isfinite(metrics['validation_l2']):raise ContractError('Tree2 child nonfinite validation loss')
    manifest=save_provisional_trained(root/'artifact',booster,model,request,expected,envelope['action_config_hash'],deadline=deadline)
    from gf_common.action_contract import parse_action_contract
    inspect_artifact(root/'artifact',expected_digest=digest_bytes(root/'artifact'/'inference.json',scan),model=model,
        action_config_hash=envelope['action_config_hash'],action_contract=parse_action_contract(envelope['input_binding']['action_contract']),
        purpose=envelope['input_binding']['purpose'],synthetic_capacity=envelope['input_binding']['synthetic_capacity'],deadline=deadline)
    scan.finalize()
    receipt={'schema_version':3,'kind':'structured-tree-fit-result','request_sha256':expected,'input_files':request['input_files'],
        'metrics':metrics,'inference_sha256':digest_bytes(root/'artifact'/'inference.json',scan),
        'model_sha256':manifest['model_file']['sha256']}
    publisher=_ChainScan(directory=root,max_owned_bytes=model.resources.max_owned_bytes-fit_admission(model,fit,request['row_counts'],retained_bytes=0)['raw_bytes'],deadline=deadline)
    _write(root/'child-result.json',receipt,scan=publisher,immutable=True)
    publisher.finalize();scan.finalize()
    # Keep every consumed numeric field observable, and no console payload.
    assert canonical(asdict(model.architecture()))==canonical(request['model'])
    assert time.monotonic()<deadline
    import resource
    print('structured_tree_fit_peak_rss_kib='+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),flush=True)


def digest_bytes(path:Any,scan:Any)->str:
    with scan.file(path):pass
    return scan.files[path][1]


if __name__=='__main__':main()
