"""Early Dataset2 I/O/admission seams; no fabricated source qualification."""
import hashlib
import json
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training.native_execution import NativeExecution
from gf_training.native_tree_dataset import NativeTreeDataset
from gf_training.structured_tree_data import build_component_dataset, inspect_component_dataset


def settings(**kwargs):
    return PreparationConfig(backend='rust',reader='structured-v1',**kwargs)


def plan(tmp_path):
    return {'schema_version':2,'kind':'structured-tree-dataset-plan',
        'parents':[{'directory':str(tmp_path/f'p{i}'),'validation_directory':str(tmp_path/f'v{i}')} for i in range(3)],
        'encoder':{'recipe':'participant-structured-tree-v1','max_positions':18,'aggregate_window':128,'aggregate_recipe':'public-exact-v2'},
        'action_contract':asdict(engineering_contract()),'purpose':'component','synthetic_capacity':False}


def test_initial_manifest_respects_smaller_consumer_cap_before_json_parse(tmp_path,monkeypatch):
    import gf_models.structured_sealed as sealed
    path=tmp_path/'manifest.json';raw=b'{}'+b' '*65536;path.write_bytes(raw)
    monkeypatch.setattr(sealed.json,'loads',lambda *a,**k:pytest.fail('configured1MiB must reject encoded expansion before parse'))
    with pytest.raises(ContractError,match='reservation'):
        inspect_component_dataset(path,expected_digest=hashlib.sha256(raw).hexdigest(),
            settings=settings(memory_budget_mib=1,cache_budget_mib=0,chunk_target_mib=1),
            max_owned_bytes=128*1024**2,deadline=time.monotonic()+60)


def test_initial_manifest_respects_consumer_work_deadline_before_parse(tmp_path,monkeypatch):
    import gf_training.structured_tree_data as data
    path=tmp_path/'manifest.json';path.write_bytes(b'{}')
    # Only the preparation boundary clock changes. The actual SealedScan clock
    # sees the resulting already-expired deadline before Source/open/JSON work.
    monkeypatch.setattr(data,'time',SimpleNamespace(monotonic=lambda:0.))
    with pytest.raises(TimeoutError):
        inspect_component_dataset(path,expected_digest=hashlib.sha256(b'{}').hexdigest(),
            settings=settings(wall_seconds=10,cleanup_seconds=1),max_owned_bytes=128*1024**2,deadline=time.monotonic()+60)


@pytest.mark.parametrize('alias',['dotdot','symlink'])
def test_output_alias_inside_parent_rejects_before_source_or_probe(tmp_path,monkeypatch,alias):
    import gf_training.structured_tree_data as data
    import gf_training.structured_tree_validation as validation
    from gf_simulator_client.structured_resources import ReservationLedger
    from tests.structured_ppo_support import configuration
    parent=tmp_path/'parent';parent.mkdir();other=tmp_path/'other';other.mkdir()
    output=other/'..'/'parent'/'validation'
    if alias=='symlink':
        link=tmp_path/'link';link.symlink_to(parent,target_is_directory=True);output=link/'validation'
    monkeypatch.setattr(data,'NativeTreeDataset',lambda *a,**k:pytest.fail('no Source constructor before output ownership'))
    monkeypatch.setattr(validation,'NativeStructuredSource',lambda *a,**k:pytest.fail('no parent/probe before output ownership'))
    loaded=configuration(tmp_path/'cfg')
    with pytest.raises(ContractError,match='outside|nonsymlink'):
        validation.FinalValidationContext(loaded,parent,output,ReservationLedger(),source_settings=replace(loaded.config.preprocessing,backend="rust",reader="structured-v1"),deadline=time.monotonic()+60)
    metadata=plan(tmp_path);metadata['parents'][0]['directory']=str(parent)
    with pytest.raises(ContractError,match='outside|nonsymlink'):
        build_component_dataset(metadata,output,settings(),max_owned_bytes=2*1024**3,deadline=time.monotonic()+60)
    assert not (parent/'validation').exists()


def test_explicit_sink_admission_precedes_allocation_and_every_resume(tmp_path):
    owner=NativeTreeDataset(plan(tmp_path),settings(),max_owned_bytes=4*1024**2,deadline=time.monotonic()+60)
    calls=[]
    class Session:
        held=0
        def used_bytes(self):return 128*1024+self.held
        def recheck(self,held):
            if 128*1024+held>owner.budget:raise ContractError('injected joint native/sink reservation')
            self.held=held;calls.append(('recheck',held))
        def retire_current(self,held):self.recheck(held);calls.append(('retire',held))
        def verify_complete(self,held):self.recheck(held);calls.append(('verify',held))
        def finalize(self,held):self.recheck(held);calls.append(('finalize',held))
        def close(self):pass
    owner._session=Session();owner._executor=NativeExecution(SimpleNamespace(cancel=lambda:None))
    try:
        owner.reserve_sink(1024*1024)
        # The caller allocates only after successful admission.
        buffer=bytearray(1024*1024)
        for method in ('retire_current','verify_complete','finalize'):owner._call(method)
        assert all(held>=len(buffer) for _,held in calls)
        with pytest.raises(ContractError,match='reservation'):
            owner.reserve_sink(owner.budget-1)
        del buffer
        owner.reserve_sink(0)
        assert calls[-1][1]<1024*1024
    finally:owner.close()


def test_existing_owned_publisher_requires_and_accepts_actual_output_root(tmp_path):
    from gf_training.structured_lineage import _ChainScan, _write
    root=tmp_path/'owned';root.mkdir()
    scan=_ChainScan(directory=root,max_owned_bytes=16*1024**2,deadline=time.monotonic()+30)
    value={'schema_version':2,'scope':'publisher-only fixture, not a Source manifest'}
    expected=_write(root/'manifest.json',value,scan=scan,immutable=True)
    scan.finalize()
    assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==expected
    assert json.loads((root/'manifest.json').read_bytes())==value


def test_actual_projected_plan_loadedconfig_and_consumer_settings_have_closed_budget(tmp_path,monkeypatch):
    import dataclasses
    import sys

    import gf_training.structured_tree_validation as validation
    from gf_common.domain import RuntimePolicyBinding, digest
    from gf_models.structured_binding import policy_label
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.collection import CollectionPlan, validation_groups
    from tests.structured_ppo_support import configuration
    loaded=configuration(tmp_path/'cfg');cfg=loaded.config
    parent=tmp_path/'parent';parent.mkdir()
    output=tmp_path/'validation'
    consumer=replace(cfg.preprocessing,backend='rust',reader='structured-v1')
    monkeypatch.setattr(validation,'NativeStructuredSource',lambda *a,**k:pytest.fail('budget test must not open Source'))
    context=validation.FinalValidationContext(loaded,parent,output,ReservationLedger(),source_settings=consumer,deadline=time.monotonic()+60)
    pure=cfg.model.architecture();binding=cfg.model.inference
    label=policy_label(pure,binding,'a'*64,cuda_visible_devices='')
    options={'version':'2','checkpoint':str(parent/'artifacts/000002'),'expected_manifest_digest':'a'*64,
        'model':asdict(pure),'inference':asdict(binding),'cuda_visible_devices':'',
        'action_contract':asdict(cfg.actions.action_contract),'purpose':'component','synthetic_capacity':False}
    groups=validation_groups(17,label,actions_version=2)
    registry={label:RuntimePolicyBinding('mlp',options),'random-structured-v1':RuntimePolicyBinding('random'),
              'heuristic-structured-v1':RuntimePolicyBinding('heuristic')}
    projected=replace(cfg,run=replace(cfg.run,split='validation'))
    context.projected=replace(loaded,config=projected,effective=asdict(projected),sha256=digest(asdict(projected)))
    context.plan=CollectionPlan(groups,registry,output,{'parent_run_id':'controlled','parent_run_path':str(parent),
        'batch_id':'validation-final','collection_generation':2,'learner_seats':{spec.episode_id:spec.learner_viewer for group in groups for spec in group.episodes}},
        time.monotonic()+7140,time.monotonic()+7200,'ppo_validation',time.monotonic())
    context._expected=context._plan_signature(context.plan)
    def deep(value,seen=None):
        seen=set() if seen is None else seen
        if id(value) in seen:return 0
        seen.add(id(value));size=sys.getsizeof(value)
        if dataclasses.is_dataclass(value):return size+sum(deep(getattr(value,f.name),seen) for f in dataclasses.fields(value))+deep(getattr(value,'__dict__',{}),seen)
        if isinstance(value,dict):return size+sum(deep(k,seen)+deep(v,seen) for k,v in value.items())
        if isinstance(value,(tuple,list)):return size+sum(deep(v,seen) for v in value)
        if hasattr(value,'parts'):return size+deep(str(value),seen)+deep(value.parts,seen)
        return size
    actual=deep((loaded,context.projected,context.plan,consumer,context._output_ancestors,context._expected))
    assert context._extra()>=actual>0
    assert context.plan.output is output and context.source_settings==consumer


def test_returned_report_postwrite_wall_is_only_permitted_difference():
    from gf_training.structured_tree_validation import _checked_returned_report
    snapshot={'schema_version':1,'comparison_valid':True,'wall_seconds':10.,'nested':{'natural':64}}
    returned={**snapshot,'wall_seconds':12.}
    _checked_returned_report(snapshot,returned,11.,7200.)
    for altered in ({**returned,'comparison_valid':False},{**returned,'extra':1},
                    {'schema_version':1,'wall_seconds':12.,'comparison_valid':True},
                    {**returned,'nested':{'natural':63}}):
        with pytest.raises(ContractError,match='report'):
            _checked_returned_report(snapshot,altered,11.,7200.)


@pytest.mark.parametrize('times',[(10.,11.,9.,7200.),(12.,11.,13.,7200.),(10.,12.,11.,7200.),
    (10.,11.,7201.,7200.),(-1.,11.,12.,7200.),(True,11.,12.,7200.),(10.,True,12.,7200.),
    (10.,11.,True,7200.),(10.,11.,12.,True),(float('nan'),11.,12.,7200.),
    (10.,float('inf'),12.,7200.),(10.,11.,float('nan'),7200.),(10.,11.,12.,float('inf')),
    (10.,11.,12.,10**400)])
def test_returned_report_timing_types_finiteness_and_order_are_strict(times):
    from gf_training.structured_tree_validation import _checked_returned_report
    with pytest.raises(ContractError,match='timing'):
        _checked_returned_report({'wall_seconds':times[0]},{'wall_seconds':times[2]},times[1],times[3])


@pytest.mark.parametrize('mismatch',[False,True])
def test_actor_entry_binds_original_input_origins_before_transport(tmp_path,monkeypatch,mismatch):
    import asyncio

    import gf_training.structured_tree_validation as validation
    from gf_common.domain import digest
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_training.lineage import run_lock
    from tests.structured_ppo_support import configuration
    loaded=configuration(tmp_path/'cfg');cfg=loaded.config
    original=({'path':'original-input.toml','kind':'experiment','sha256':'1'*64},)
    supplied=({'path':'changed-input.toml','kind':'experiment','sha256':'2'*64},) if mismatch else original
    loaded=replace(loaded,inputs=supplied)
    parent=tmp_path/'parent';parent.mkdir()
    model=cfg.model.architecture()
    expected_parent={'model':asdict(model),'config_inputs':list(original),
        'state':{'config_sha256':digest(asdict(cfg)),'run_id':'bounded-parent','seed':17},
        'execution':{'inference':{'binding':asdict(cfg.model.inference),'cuda_visible_devices':''}},
        'generations':[{'generation':2,'inference_sha256':'a'*64}]}
    class Source:
        _closed=False
        def __init__(self,*args,**kwargs):
            self.summary={'parents':[expected_parent]}
            self.models={0:model}
        def __enter__(self):return self
        def original_config_inputs(self,*args):return list(original)
        def recheck(self,*args):pass
        def close(self):self._closed=True
    monkeypatch.setattr(validation,'NativeStructuredSource',Source)
    async def run():
        context=validation.FinalValidationContext(loaded,parent,tmp_path/'validation',ReservationLedger(),
            source_settings=replace(cfg.preprocessing,backend='rust',reader='structured-v1'),deadline=time.monotonic()+60)
        if mismatch:
            with pytest.raises(ContractError,match='input origins'):
                await context.__aenter__()
            assert context.plan is None and context._closed
        else:
            result=await context.__aenter__()
            assert result is context and context.projected.inputs==original and len(context.plan.groups)==32
            await context.aclose()
    with run_lock(parent):asyncio.run(run())
    assert not (tmp_path/'validation').exists()


@pytest.mark.parametrize('cancelled',[False,True])
def test_tree_dataset_unconfirmed_cleanup_retains_same_owner_until_real_join(tmp_path,cancelled):
    import asyncio
    import gc
    import weakref
    owner=NativeTreeDataset(plan(tmp_path),settings(),max_owned_bytes=128*1024**2,deadline=time.monotonic()+60)
    owner._executor=NativeExecution(SimpleNamespace(cancel=lambda:None));failed=[True]
    def close():
        if failed[0]:raise ContractError('unconfirmed Dataset cleanup')
    owner._session=SimpleNamespace(close=close)
    released=[];owner._locks.callback(lambda:released.append('released'))
    original=asyncio.CancelledError('original cancel') if cancelled else RuntimeError('original body failure')
    reference=weakref.ref(owner)
    owner.__exit__(type(original),original,None)
    assert original._structured_source_cleanup_owner is owner and not released
    assert owner._executor._active is None and not owner._closed
    del owner;gc.collect()
    held=original._structured_source_cleanup_owner
    assert reference() is held
    failed[0]=False;held.close()
    assert held._closed and held._executor._closed and released==['released']
