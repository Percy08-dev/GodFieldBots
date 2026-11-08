"""Unadmitted index/resource primitives ONLY; no fabricated formal Source/fit."""
import asyncio
import hashlib
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_models.structured_sealed import SealedScan
from gf_models.structured_tree_config import ConfiguredTreeModelV3
from gf_models.structured_tree_pairing import TREE2_RECORDS, TREE3_GAME_RECORDS, game_fit_admission
from gf_training.card_instance_game_tree_data import (
    GameTreeBudget,
    VerifiedGameTreeDataset3,
    _GameTreeOriginals,
    _management_source_set,
    _verified_fit_rows,
)
from gf_training.card_instance_source_locks import GameSourceCustody, ValidationManifestOwnership
from gf_training.lineage import _OWNED_LOCKS
from gf_training.structured_tree_fit import _PreparedTreeFit, _TreeFitRun, fit_game_dataset


def settings():
    return PreparationConfig(reader='participant-structured-v2')


def test_index_decode_pre_admits_full_child_and_copy_before_serializer(tmp_path,monkeypatch):
    # Sparse bytes are not a dataset. Admission must fail before parsing them.
    (tmp_path/'manifest.json').write_bytes(b' '*65536)
    (tmp_path/'preparation.json').write_bytes(b'{}')
    def forbidden(*args,**kwargs):
        pytest.fail('index JSON decoder reached after joint cap exhausted')
    monkeypatch.setattr(SealedScan,'object',forbidden)
    dataset=VerifiedGameTreeDataset3(tmp_path/'manifest.json','a'*64,settings(),
        max_owned_bytes=1024**2,child_limit_bytes=512*1024,deadline=time.monotonic()+60)
    with pytest.raises(ContractError,match='reservation'):
        dataset.__enter__()
    assert dataset._source is None and not dataset._active


def test_old_reader_rejects_before_index_or_original_output(tmp_path):
    with pytest.raises(ContractError,match='corrected reader'):
        VerifiedGameTreeDataset3(tmp_path/'missing','a'*64,PreparationConfig(),deadline=time.monotonic()+60)
    assert not tuple(tmp_path.iterdir())


def test_fixture_index_cannot_enter_game_source(tmp_path):
    import json
    raw=json.dumps({'schema_version':3,'kind':'card-instance-fixture-fixed-sources'}).encode()
    (tmp_path/'manifest.json').write_bytes(raw)
    (tmp_path/'preparation.json').write_bytes(b'{}')
    dataset=VerifiedGameTreeDataset3(tmp_path/'manifest.json',hashlib.sha256(raw).hexdigest(),settings(),
        deadline=time.monotonic()+60)
    with pytest.raises(ContractError):
        dataset.__enter__()
    assert dataset._source is None


def test_child_local_formula_refuses_even_when_global_has_room():
    # Local5raw must fit SAME full configured2GiB, independently of8GiB global.
    model=ConfiguredTreeModelV3()
    fit={'num_boost_round':500,'early_stopping_rounds':50,'learning_rate':.05,'min_data_in_leaf':20,'num_threads':1}
    with pytest.raises(ContractError,match='CHILD-LOCAL'):
        game_fit_admission(model,fit,{'train':40000,'validation':1},retained_bytes=0,total_bytes=8*1024**3)


def test_preentry_other_owners_and_full_cap_charge_exactly_once():
    gib=1024**3
    budget=GameTreeBudget(8*gib,dataset_bytes=gib,child_limit_bytes=2*gib)
    budget.parent_intrinsic(0,2*gib)
    assert budget.external_base(1)==5*gib
    budget.parent_intrinsic(1,2*gib)
    assert budget.external_base(2)==7*gib
    budget.parent_intrinsic(2,gib)
    assert budget.retained_bytes==8*gib
    with pytest.raises(ContractError):budget.reserve_child(2*gib+1)
    with pytest.raises(ContractError):budget.reserve_sink(1)


def test_row_retirement_keeps_sink_until_all_actual_arrays_and_packing_aliases_die():
    # UNADMITTED lifecycle primitive: no constructor/active Source/fit authority.
    retired=[]
    dataset=object.__new__(VerifiedGameTreeDataset3)
    dataset._source=SimpleNamespace(_retire_sink=lambda:retired.append('physical rows gone'))
    prepared=object.__new__(_PreparedTreeFit)
    prepared.context=TREE3_GAME_RECORDS;prepared.dataset=dataset
    x=np.empty((2,3),dtype=np.float32);y=np.empty(2,dtype=np.float32)
    prepared.blocks={'train':[x],'validation':[]};prepared.labels={'train':[y],'validation':[]}
    prepared._packing_arrays={'x':x,'y':y};prepared._row_refs=[]
    with pytest.raises(ContractError,match='remain alive'):
        prepared.retire_rows()
    assert not retired and not prepared._packing_arrays and not any(prepared.blocks.values())
    del x,y
    prepared.retire_rows()
    assert retired==['physical rows gone']
    prepared.retire_rows()
    assert retired==['physical rows gone']
    assert not hasattr(dataset,'_active')  # this never pretended to be a qualified Source


def test_old_row_lifetime_remains_unchanged():
    prepared=object.__new__(_PreparedTreeFit)
    prepared.context=TREE2_RECORDS
    prepared.blocks={'train':[object()]}
    prepared.retire_rows()
    assert prepared.blocks['train']


def test_fake_dataset_summary_cannot_prepare_private_game_rows():
    dataset=SimpleNamespace(_active=True,source=object(),manifest={})
    with pytest.raises(ContractError,match='concrete verified Dataset'):
        _PreparedTreeFit(dataset,ConfiguredTreeModelV3(),{},_context=TREE3_GAME_RECORDS)


def test_public_fit_rejects_nonfresh_configuration_before_any_index_or_child(tmp_path):
    from gf_models.config import TrainingConfig
    model=replace(ConfiguredTreeModelV3(),checkpoint='/UNADMITTED-no-file')
    with pytest.raises(ContractError,match='fresh bounded'):
        asyncio.run(fit_game_dataset(tmp_path/'missing',expected_digest='a'*64,directory=tmp_path/'output',
            model=model,settings=settings(),training=TrainingConfig(),seed=17,deadline=time.monotonic()+60))
    assert not tuple(tmp_path.iterdir())


def inventory(tmp_path):
    # Opaque control/stat fixture. No J6/C5/runtime or Source is fabricated.
    (tmp_path/'opaque.bin').write_bytes(b'UNADMITTED opaque')
    (tmp_path/'manifest.json').write_bytes(b'{}')
    (tmp_path/'report.json').write_bytes(b'{}')
    entries=[{'path':'opaque.bin','sha256':hashlib.sha256((tmp_path/'opaque.bin').read_bytes()).hexdigest(),
        'bytes':(tmp_path/'opaque.bin').stat().st_size}]
    controls={}
    for name in ('manifest.json','report.json'):
        path=tmp_path/name;info=path.stat()
        controls[path]=((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns),hashlib.sha256(path.read_bytes()).hexdigest())
    return entries,controls


def test_management_stat_digest_matches_existing_original_file_map_recipe(tmp_path):
    entries,controls=inventory(tmp_path)
    expected={}
    for name in ('opaque.bin','manifest.json','report.json'):
        path=tmp_path/name;info=path.stat()
        expected[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':info.st_size,
            'identity':[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]}
    assert _management_source_set(tmp_path,entries,controls)==digest(expected)


@pytest.mark.parametrize('change',['escape','duplicate','control','size','symlink','hash','replacement'])
def test_management_reference_corruption_refuses_without_reading_opaque_bytes(tmp_path,change):
    entries,controls=inventory(tmp_path)
    if change=='escape':entries[0]['path']='../opaque.bin'
    elif change=='duplicate':entries.append(dict(entries[0]))
    elif change=='control':entries[0]['path']='manifest.json'
    elif change=='size':entries[0]['bytes']+=1
    elif change=='symlink':
        (tmp_path/'opaque.bin').unlink();(tmp_path/'opaque.bin').symlink_to(tmp_path/'report.json')
    elif change=='hash':entries[0]['sha256']='invalid'
    else:
        (tmp_path/'other').write_bytes((tmp_path/'report.json').read_bytes())
        (tmp_path/'other').replace(tmp_path/'report.json')
    with pytest.raises(ContractError):_management_source_set(tmp_path,entries,controls)


def resource_cleanup_owner(tmp_path):
    # UNADMITTED lifecycle primitive with real temporary flock/run locks only.
    # No FormalGameParent, Source records, runtime or fitting permission exists.
    parents=tuple(tmp_path/f'p{i}' for i in range(3))
    validations=tuple(tmp_path/f'v{i}' for i in range(3))
    for path in (*parents,*validations):path.mkdir()
    for path in validations:(path/'manifest.json').write_bytes(b'{}')
    custody=GameSourceCustody(parents,validations,deadline=time.monotonic()+60).__enter__()
    owner=object.__new__(_GameTreeOriginals)
    owner._custody=custody;owner._scan=SealedScan(max_owned_bytes=1024**2,deadline=time.monotonic()+60)
    owner._parents=[];owner._sources=[];owner._source_scopes=[];owner._loans=[]
    owner._retired_sources=set();owner._retired_parents=set()
    owner._active=True;owner._closed=False;owner._failure=None;owner._sink=0;owner._current_feature_bytes=0
    return owner,parents


def test_terminal_metadata_guard_failure_still_closes_actual_independent_custody(tmp_path):
    owner,parents=resource_cleanup_owner(tmp_path)
    path=tmp_path/'management.json';path.write_bytes(b'{}')
    owner._scan.object(path)
    path.write_bytes(b'{"mutated":true}')
    with pytest.raises(ContractError):owner.close()
    assert owner.cleanup_complete
    assert all(path not in _OWNED_LOCKS for path in parents)


def test_cleanup_formatting_failure_keeps_primary_and_releases_confirmed_resource_locks(tmp_path):
    owner,parents=resource_cleanup_owner(tmp_path)
    primary=MemoryError('UNADMITTED primary')
    class Unprintable(RuntimeError):
        def __str__(self):raise MemoryError('diagnostic formatting failed')
    class ResourceScope:
        def __exit__(self,*args):raise Unprintable()
    owner._failure=primary;owner._source_scopes=[ResourceScope()]
    with pytest.raises(MemoryError) as captured:owner.close()
    assert captured.value is primary and owner.cleanup_complete
    assert all(path not in _OWNED_LOCKS for path in parents)


def test_game_fit_cleanup_preserves_initiating_cancel_when_secondary_formatting_allocates():
    import sys
    class Unprintable(RuntimeError):
        def __str__(self):raise MemoryError('diagnostic allocation')
    async def run():
        entered=asyncio.Event();release=asyncio.Event()
        process=await asyncio.create_subprocess_exec(sys.executable,'-E','-c','pass')
        class UnadmittedCleanupPrimitive:
            def __init__(self):self.closed=False;self.health={}
            async def close(self,*args):
                entered.set();await release.wait()
                await process.wait();self.closed=True;self.health['reaped']=True
                raise Unprintable()
        owner=object.__new__(_TreeFitRun)
        owner.context=TREE3_GAME_RECORDS;owner.child=UnadmittedCleanupPrimitive();owner.deadline=time.monotonic()+60
        work=asyncio.create_task(owner._cleanup_child())
        await entered.wait()
        work.cancel('original cancellation')
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError,match='original cancellation'):
            await work
        assert process.returncode==0 and owner.child.closed and owner.child.health['reaped'] is True
    asyncio.run(run())


def test_resource_custody_closed_before_entry_cannot_allocate_a_second_lifetime(tmp_path,monkeypatch):
    import gf_training.card_instance_source_locks as leaf
    parents=tuple(tmp_path/f'p{i}' for i in range(3));validations=tuple(tmp_path/f'v{i}' for i in range(3))
    for path in (*parents,*validations):path.mkdir()
    for path in validations:(path/'manifest.json').write_bytes(b'{}')
    manifest=ValidationManifestOwnership(validations[0],deadline=time.monotonic()+60)
    manifest.close()
    def forbidden(*args,**kwargs):pytest.fail('closed resource custody attempted a new acquisition')
    monkeypatch.setattr(leaf.os,'open',forbidden)
    monkeypatch.setattr(leaf,'run_lock',forbidden)
    with pytest.raises(ContractError,match='one lifetime'):manifest.__enter__()
    assert manifest.cleanup_complete
    custody=GameSourceCustody(parents,validations,deadline=time.monotonic()+60)
    custody.close()
    with pytest.raises(ContractError,match='one lifetime'):custody.__enter__()
    assert custody.cleanup_complete and not custody._entries
    assert all(path not in _OWNED_LOCKS for path in parents)


def test_rust_original_history_preflight_refusal_precedes_index_io_without_fallback(tmp_path,monkeypatch):
    import gf_training.native_backend as backend
    seen=[]
    def refused(settings):
        seen.append(settings)
        raise ContractError('unadmitted structural history ABI refusal; no Python fallback')
    # Exact Native-owned approved public seam. No successful Source/runtime is
    # manufactured, and its concrete ABI tests remain Native's responsibility.
    monkeypatch.setattr(backend,'original_history_reader_settings',refused,raising=False)
    def forbidden(*args,**kwargs):pytest.fail('missing Rust history ABI reached index/Source read')
    monkeypatch.setattr(SealedScan,'object',forbidden)
    with pytest.raises(ContractError,match='no Python fallback'):
        VerifiedGameTreeDataset3(tmp_path/'manifest.json','a'*64,
            replace(settings(),backend='rust'),deadline=time.monotonic()+60)
    assert len(seen)==1 and not tuple(tmp_path.iterdir())


@pytest.mark.parametrize('split',['train','validation'])
def test_index_count_forgery_refuses_before_private_row_allocation(split,monkeypatch):
    # UNADMITTED pure correspondence negative. These dictionaries cannot create
    # a Parent/Source/Dataset; production obtains actual only from retained owners.
    actual={'train':96,'validation':192}
    metadata={'counts':{name:{'selected_decisions':count} for name,count in actual.items()}}
    metadata['counts'][split]['selected_decisions']-=1
    def forbidden(*args,**kwargs):pytest.fail('forged index count reached private row allocation')
    monkeypatch.setattr(np,'empty',forbidden)
    with pytest.raises(ContractError,match='before private rows'):
        _verified_fit_rows(metadata,actual)
