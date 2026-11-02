"""Additive same-owner input-origin query; unchanged C2 closed summaries."""
import asyncio
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, canonical, digest
from gf_common.preprocessing import PreparationConfig
from gf_training.native_execution import NativeExecution
from gf_training.native_structured_session import NativeStructuredSource


def fake_owner(payload):
    owner=NativeStructuredSource({'schema_version':2,'mode':'historical_parent','parents':[],'collections':[]},
        PreparationConfig(backend='rust',reader='structured-v1'),max_owned_bytes=128*1024**2,deadline=time.monotonic()+30)
    owner.summary={'control_fingerprint':'a'*64,'parents':[{'run_sha256':'b'*64}]}
    owner._executor=NativeExecution(SimpleNamespace(cancel=lambda:None))
    owner._session=SimpleNamespace(original_config_inputs=lambda *args:canonical(payload).encode(),close=lambda:None)
    return owner


@pytest.mark.parametrize('field,value',[('schema_version',True),('kind','wrong'),('parent_index',True),
    ('parent_index',1),('control_fingerprint','0'*64),('run_sha256','0'*64),('config_inputs',None)])
def test_original_origin_wrapper_requires_exact_closed_binding(field,value):
    raw={'schema_version':2,'kind':'structured-original-config-inputs','control_fingerprint':'a'*64,
         'parent_index':0,'run_sha256':'b'*64,'config_inputs':[]}
    owner=fake_owner({**raw,field:value})
    try:
        with pytest.raises(ContractError,match='binding'):
            owner.original_config_inputs(0)
    finally:owner.close()


def source43():
    requested=os.environ.get('GFB_TREE_DATASET_EXISTING_SOURCE43')
    if not requested:pytest.skip('explicit preserved actual43 original required')
    return Path(requested)/'parent-43'


def actual_settings():
    return canonical({'memory_bytes':512*1024**2,'scratch_bytes':1024**3,'max_source_file_bytes':64*1024**2,
                      'max_record_bytes':8*1024**2,'chunk_target_bytes':16*1024**2}).encode()


def test_actual_origin_query_is_one_delivery_with_independent_until_close_lease():
    native=pytest.importorskip('gf_native')
    if not hasattr(native.StructuredNativeSession,'original_config_inputs'):pytest.skip('query candidate required')
    root=source43();state=json.loads((root/'run.json').read_bytes())
    plan=canonical({'schema_version':2,'mode':'historical_parent','parents':[{'directory':str(root),'require_complete':True}],'collections':[]}).encode()
    core=native.StructuredNativeSession(plan,actual_settings(),time.monotonic()+120)
    try:
        summary=core.control_summary();parsed=json.loads(summary)
        before=core.control_reserved_bytes();used=core.used_bytes()
        raw=json.loads(core.original_config_inputs(0))
        assert set(raw)=={'schema_version','kind','control_fingerprint','parent_index','run_sha256','config_inputs'}
        assert raw=={'schema_version':2,'kind':'structured-original-config-inputs','control_fingerprint':parsed['control_fingerprint'],
            'parent_index':0,'run_sha256':parsed['parents'][0]['run_sha256'],'config_inputs':state['config_inputs']}
        assert core.control_reserved_bytes()>before and core.used_bytes()>used
        reserved=core.control_reserved_bytes()
        with pytest.raises(ValueError,match='already delivered'):core.original_config_inputs(0)
        assert core.control_reserved_bytes()==reserved
        # Query is additive, never changes/reuses the existing control tree.
        assert json.loads(summary)==parsed
        core.recheck()
    finally:core.close()
    assert core.used_bytes()==core.control_reserved_bytes()==0
    with pytest.raises(RuntimeError):core.original_config_inputs(0)


def test_query_quota_failure_is_not_a_delivery_or_permission_to_allocate():
    native=pytest.importorskip('gf_native')
    if not hasattr(native.StructuredNativeSession,'original_config_inputs'):pytest.skip('query candidate required')
    root=source43();plan=canonical({'schema_version':2,'mode':'historical_parent','parents':[{'directory':str(root),'require_complete':True}],'collections':[]}).encode()
    core=native.StructuredNativeSession(plan,actual_settings(),time.monotonic()+120)
    try:
        core.control_summary()
        with pytest.raises(MemoryError):core.original_config_inputs(0,512*1024**2)
        assert json.loads(core.original_config_inputs(0))['parent_index']==0
    finally:core.close()


def _loaded_original(root):
    from gf_cli.config import (
        EvaluationConfig,
        ExperimentConfig,
        LoadedConfig,
        RunConfigV2,
        ServerConfig,
    )
    from gf_common.preprocessing import preparation_config
    from gf_common.structured import action_config_v2
    from gf_models.config import model_config, training_config
    state=json.loads((root/'run.json').read_bytes());raw=state['effective_config']
    run={**raw['run'],'seeds':tuple(raw['run']['seeds']),'sampling_seeds':tuple(raw['run']['sampling_seeds'])}
    cfg=ExperimentConfig(schema_version=raw['schema_version'],kind=raw['kind'],run=RunConfigV2(**run),
        server=ServerConfig(**raw['server']),actions=action_config_v2(raw['actions']),model=model_config(raw['model']),
        opponent=model_config(raw['opponent']),evaluation=EvaluationConfig(**raw['evaluation']),
        training=training_config(raw['training']),preprocessing=preparation_config(raw['preprocessing']))
    assert canonical(asdict(cfg))==canonical(raw)
    return LoadedConfig(cfg,raw,digest(raw),tuple(state['config_inputs']))


@pytest.mark.parametrize('mismatch',[False,True])
def test_actual43_actor_origin_entry_has_no_probe_or_collection(tmp_path,monkeypatch,mismatch):
    native=pytest.importorskip('gf_native')
    if not hasattr(native.StructuredNativeSession,'original_config_inputs'):pytest.skip('query candidate required')
    root=source43();loaded=_loaded_original(root)
    if mismatch:loaded=replace(loaded,inputs=({'path':'changed.toml','kind':'experiment','sha256':'e'*64},))
    import gf_cli.structured as composition
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.structured_tree_validation import FinalValidationContext
    monkeypatch.setattr(composition,'preflight_final_validation_actor',lambda *a,**k:pytest.fail('entry test never probes/collects'))
    async def run():
        ledger=ReservationLedger(loaded.config.run.global_cuda_budget_bytes,loaded.config.run.max_participant_workers)
        context=FinalValidationContext(loaded,root,tmp_path/'not-collected',ledger,
            source_settings=replace(loaded.config.preprocessing,backend='rust',reader='structured-v1'),deadline=time.monotonic()+120,native=native)
        if mismatch:
            with pytest.raises(ContractError,match='input origins'):await context.__aenter__()
            assert context.plan is None and context._closed
        else:
            await context.__aenter__()
            assert len(context.plan.groups)==32 and context.projected.inputs==loaded.inputs
            await context.aclose()
        assert not ledger.workers and ledger.learner is None
    asyncio.run(run())
    assert not (tmp_path/'not-collected').exists()
