"""Concrete same-lifetime Source fit preparation and pre-native admission."""
from dataclasses import replace

import pytest

from gf_common.domain import ContractError
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV2,
    TreeInferenceConfigV2,
    TreeNetworkConfigV2,
    TreeResourcesV2,
)
from gf_models.structured_tree_pairing import fit_admission, model_envelope

FIT={'num_boost_round':500,'early_stopping_rounds':50,'learning_rate':.05,'min_data_in_leaf':20,'num_threads':1}


def test_exact_model_envelope_and_asymmetric_threads_admit_before_native():
    model=ConfiguredTreeModelV2();envelope=model_envelope(model.architecture(),500)
    assert envelope=={'recipe':'tree-fit-model-envelope-v1','feature_width':1647,'rounds':500,'num_leaves':31,'node_bound':30500,'text_bytes':32140800}
    first=fit_admission(model,FIT,{'train':192,'validation':320},retained_bytes=1000000)
    asymmetric=fit_admission(model,{**FIT,'num_threads':2},{'train':192,'validation':320},retained_bytes=1000000)
    assert asymmetric['fixed_bytes']-first['fixed_bytes']==64*1024**2
    inference=replace(model,inference=TreeInferenceConfigV2(num_threads=3))
    assert fit_admission(inference,{**FIT,'num_threads':2},{'train':192,'validation':320},retained_bytes=1000000)['native_threads']==3
    with pytest.raises(ContractError,match='envelope'):
        model_envelope(replace(model.architecture(),network=TreeNetworkConfigV2(num_leaves=127)),500)
    with pytest.raises(ContractError,match='fit admission'):
        fit_admission(replace(model,resources=TreeResourcesV2(max_owned_bytes=first['total_bytes'])),{**FIT,'num_threads':2},
                      {'train':192,'validation':320},retained_bytes=1000000)
    with pytest.raises(ContractError,match='fit label array'):
        fit_admission(model,FIT,{'train':100000000,'validation':1000000},retained_bytes=0)


def test_concrete_verified_owner_prepares_all_private_rows_and_keeps_same_sources():
    import hashlib
    import os
    import time
    from pathlib import Path

    import numpy as np

    from gf_common.preprocessing import PreparationConfig
    from gf_models.structured_tree_config import TreeEncoderConfigV2
    from gf_training.structured_tree_data import VerifiedTreeDataset
    from gf_training.structured_tree_fit import _PreparedTreeFit
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('preserved actual Dataset2 source fixture required')
    manifest=Path(root)/'dataset2/manifest.json'
    settings=PreparationConfig(backend='rust',reader='structured-v1',memory_budget_mib=2048,cache_budget_mib=512,chunk_target_mib=16)
    import gf_native
    with VerifiedTreeDataset(manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),settings,max_owned_bytes=2*1024**3,
                             deadline=time.monotonic()+300,native=gf_native) as dataset:
        original=dataset.source
        model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
        prepared=_PreparedTreeFit(dataset,model,FIT);prepared.prepare()
        assert dataset.source is original and dataset._active and dataset._completed
        assert prepared.counts=={'train':192,'validation':320}
        assert sum(block.nbytes for blocks in prepared.blocks.values() for block in blocks)+sum(block.nbytes for blocks in prepared.labels.values() for block in blocks)==prepared.admission['raw_bytes']
        assert all(block.dtype==np.float32 and block.flags.c_contiguous for blocks in prepared.blocks.values() for block in blocks)
        assert {float(label) for blocks in prepared.labels.values() for block in blocks for label in block}=={-1.,0.,1.}
        dataset.verify_complete()
        assert dataset.source is original and not dataset.source._closed
        with pytest.raises(ContractError):_PreparedTreeFit({'arbitrary':'rows'},model,FIT)


def _dataset_lifetime_double(monkeypatch):
    from types import SimpleNamespace

    import gf_training.structured_tree_data as module
    from gf_training.structured_tree_data import VerifiedTreeDataset
    owner=object.__new__(VerifiedTreeDataset)
    state=SimpleNamespace(changed=False,checks=0,terminal=0)
    class Index:
        def finalize(self):
            state.checks+=1
            if state.changed:raise ContractError('original Dataset2 index changed')
    class Source:
        _closed=False
        def finalize(self):state.terminal+=1
        def __exit__(self,*args):self._closed=True
    owner.index=Index();owner.source=Source();owner.manifest={'original':True}
    owner._active=True;owner._completed=True;owner._finalized=False
    monkeypatch.setattr(module,'_content',lambda source:{'original':True})
    return owner,state


def test_dataset_index_mutation_in_body_fails_successful_exit(monkeypatch):
    owner,state=_dataset_lifetime_double(monkeypatch)
    state.changed=True
    with pytest.raises(ContractError,match='original Dataset2 index'):
        owner.__exit__(None,None,None)
    assert owner.source._closed and state.terminal==0


def test_dataset_index_mutation_inside_native_terminal_sweep_rejects(monkeypatch):
    owner,state=_dataset_lifetime_double(monkeypatch)
    def terminal():
        state.terminal+=1;state.changed=True
    owner.source.finalize=terminal
    with pytest.raises(ContractError,match='original Dataset2 index'):
        owner.__exit__(None,None,None)
    assert owner.source._closed and state.terminal==1


def test_dataset_terminal_once_and_original_index_remains_held(monkeypatch):
    owner,state=_dataset_lifetime_double(monkeypatch)
    owner.finalize();owner.__exit__(None,None,None)
    assert state.terminal==1 and state.checks>=4 and not owner._active


def test_dataset_index_error_preserves_body_cancel_and_unconfirmed_owner(monkeypatch):
    import asyncio
    owner,state=_dataset_lifetime_double(monkeypatch)
    state.changed=True
    def unconfirmed(*args):raise RuntimeError('join unconfirmed')
    owner.source.__class__.__exit__=unconfirmed
    initiating=asyncio.CancelledError('original cancellation')
    owner.__exit__(type(initiating),initiating,None)
    assert str(initiating)=='original cancellation'
    assert initiating._structured_dataset_cleanup_owner is owner
    assert owner._active and any('original-index' in note for note in initiating.__notes__)


def test_actual_dataset_inputs_fresh_native_child_reports_consumed_default_rounds(tmp_path):
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    from gf_common.preprocessing import PreparationConfig
    from gf_models.structured_sealed import SealedScan
    from gf_models.structured_tree_config import TreeEncoderConfigV2
    from gf_models.structured_tree_pairing import inspect_fit_inputs, validate_input_envelope
    from gf_training.structured_tree_data import VerifiedTreeDataset
    from gf_training.structured_tree_fit import _prepare_request, _PreparedTreeFit
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('preserved actual Dataset2 source fixture required')
    manifest=Path(root)/'dataset2/manifest.json'
    settings=PreparationConfig(backend='rust',reader='structured-v1',memory_budget_mib=2048,cache_budget_mib=512,chunk_target_mib=16)
    import gf_native
    deadline=time.monotonic()+300
    with VerifiedTreeDataset(manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),settings,max_owned_bytes=2*1024**3,
                             deadline=deadline,native=gf_native) as dataset:
        original=dataset.source
        model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
        prepared=_PreparedTreeFit(dataset,model,FIT);prepared.prepare()
        request,request_sha,publisher=_prepare_request(prepared,tmp_path/'fit',seed=17,deadline=deadline)
        scan=SealedScan(max_owned_bytes=2*1024**3,deadline=deadline)
        envelope,blocks,labels=inspect_fit_inputs(tmp_path/'fit',request,scan,retain=True)
        assert sum(len(x) for x in blocks['train'])==192 and sum(len(x) for x in blocks['validation'])==320
        assert {float(y) for values in labels.values() for block in values for y in block}=={-1.,0.,1.}
        with pytest.raises(ContractError,match='common fields'):
            validate_input_envelope({**envelope,'row_counts':{'train':191,'validation':320}},request)
        del blocks,labels
        result=subprocess.run([sys.executable,'-E','-m','gf_training.structured_tree_worker',str(tmp_path/'fit'),request_sha,str(deadline)],
            env={'PATH':os.defpath,'CUDA_VISIBLE_DEVICES':''},capture_output=True,check=False,timeout=120)
        assert result.returncode==0,result.stderr.decode()[-8000:]
        proof=json.loads((tmp_path/'fit/child-result.json').read_bytes());metrics=proof['metrics']
        assert metrics['configured_rounds']==500
        assert metrics['consumed_iterations']>=metrics['best_iteration']>=metrics['retained_tree_count']>=1
        assert proof['input_files']==request['input_files']
        assert dataset.source is original and not original._closed
        publisher.finalize();dataset.verify_complete()
        # Provisional trained output is not bindable before complete/reaped
        # parent publication and its explicit paired evidence.
        from gf_common.action_contract import engineering_contract
        from gf_models.structured_tree_artifacts import load_artifact
        with pytest.raises(ContractError,match='explicit complete paired'):
            load_artifact(tmp_path/'fit/artifact',expected_digest=proof['inference_sha256'],model=model,
                action_config_hash=envelope['action_config_hash'],action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)


def _repartitioned_numeric_inputs(tmp_path):
    import hashlib
    from dataclasses import asdict

    import numpy as np

    from gf_common.action_contract import engineering_contract
    from gf_common.domain import canonical
    from gf_models.arrays import save_arrays
    from gf_models.structured_tree_encoder import feature_layout
    model=ConfiguredTreeModelV2(inference=TreeInferenceConfigV2(candidate_chunk=2))
    width=1647
    x0=np.zeros((2,width),np.float32);x0[:,0]=[10,11]
    x1=np.zeros((1,width),np.float32);x1[:,0]=[20]
    arrays={'train.x.000000':x0,'train.x.000001':x1,'train.y.000000':np.array([-1,0],np.float32),
        'train.y.000001':np.array([1],np.float32),'validation.x.000000':x1,'validation.y.000000':np.array([0],np.float32)}
    inputs=tmp_path/'inputs';inputs.mkdir()
    shards=[];files={}
    order=['train.x.000001','train.x.000000','train.y.000000','validation.y.000000','train.y.000001','validation.x.000000']
    for i,key in enumerate(order):
        name=f'features-{i:06}.npz';metadata=save_arrays(inputs/name,{key:arrays[key]})
        shards.append({'file':name,'metadata':metadata});files[name]=metadata['sha256']
        files[name[:-4]+'.json']=hashlib.sha256((inputs/name).with_suffix('.json').read_bytes()).hexdigest()
    bundle={'schema_version':1,'prefix':'features','shards':shards,'files':files,'tensor_count':len(arrays),
        'decoded_bytes':sum(a.nbytes for a in arrays.values())}
    classification={'action_contract':asdict(engineering_contract()),'purpose':'component','synthetic_capacity':False}
    envelope={'schema_version':3,'kind':'structured-tree-fit-inputs','action_config_hash':'1'*64,
        'feature_signature':feature_layout(model.encoder)['feature_signature'],'label_recipe':'behavior-selected-natural-return-gamma1-v1',
        'input_binding':classification,'row_counts':{'train':3,'validation':1},'data':bundle}
    raw=(canonical(envelope)+'\n').encode();(inputs/'manifest.json').write_bytes(raw)
    request={'schema_version':3,'kind':'structured-tree-fit-request','run_id':'a'*32,'model':asdict(model.architecture()),
        'execution':{'inference':asdict(model.inference),'resources':asdict(model.resources)},'fit_settings':FIT,'seed':17,
        **{key:envelope[key] for key in ('feature_signature','label_recipe','input_binding','row_counts')},
        'dataset_manifest_sha256':'2'*64,'dataset_content_sha256':'3'*64,'source_set_sha256':'4'*64,
        'input_files':{'inputs/'+name:sha for name,sha in files.items()},'producer':{'code_revision':None,'source_digest':'5'*64},
        'runtime':{'numpy_version':'2.5.4','lightgbm_version':'4.7.0','native_library_sha256':'6'*64},
        'model_envelope':model_envelope(model.architecture(),500)}
    request['input_files']['inputs/manifest.json']=hashlib.sha256(raw).hexdigest()
    return envelope,request


def test_numeric_feature_label_order_is_logical_not_physical_shards(tmp_path):
    import time

    from gf_models.structured_sealed import SealedScan
    from gf_models.structured_tree_pairing import inspect_fit_inputs
    _,request=_repartitioned_numeric_inputs(tmp_path)
    scan=SealedScan(max_owned_bytes=2*1024**3,deadline=time.monotonic()+10)
    _,blocks,labels=inspect_fit_inputs(tmp_path,request,scan,retain=True)
    assert [float(x) for a in blocks['train'] for x in a[:,0]]==[10,11,20]
    assert [float(x) for a in labels['train'] for x in a]==[-1,0,1]
    scan.finalize()


@pytest.mark.parametrize('mutation',['missing','duplicate','gap','row_mismatch'])
def test_numeric_logical_pairs_reject_incomplete_or_mismatched_members(tmp_path,mutation):
    import copy

    from gf_models.structured_tree_pairing import validate_input_envelope
    envelope,request=_repartitioned_numeric_inputs(tmp_path)
    envelope=copy.deepcopy(envelope);data=envelope['data']
    if mutation=='missing':
        del data['shards'][0]['metadata']['arrays']['train.x.000001']
    elif mutation=='duplicate':
        data['shards'][0]['metadata']['arrays']={'train.x.000000':data['shards'][0]['metadata']['arrays'].pop('train.x.000001')}
    elif mutation=='gap':
        data['shards'][0]['metadata']['arrays']={'train.x.000002':data['shards'][0]['metadata']['arrays'].pop('train.x.000001')}
    else:
        data['shards'][0]['metadata']['arrays']['train.x.000001']['shape'][0]=2
        data['shards'][0]['metadata']['arrays']['train.x.000001']['bytes']*=2
    with pytest.raises(ContractError):validate_input_envelope(envelope,request)


def test_complete_concrete_dataset_fit_pairs_only_after_source_retirement(tmp_path):
    import asyncio
    import hashlib
    import os
    from pathlib import Path

    from gf_common.action_contract import engineering_contract
    from gf_common.preprocessing import PreparationConfig
    from gf_models.config import TrainingConfig
    from gf_models.structured_tree_artifacts import load_artifact, save_artifact
    from gf_models.structured_tree_config import TreeEncoderConfigV2
    from gf_models.structured_tree_pairing import inspect_training
    from gf_training.structured_tree_fit import _fit_component_dataset
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('preserved actual Dataset2 source fixture required')
    manifest=Path(root)/'dataset2/manifest.json'
    settings=PreparationConfig(backend='rust',reader='structured-v1',memory_budget_mib=2048,cache_budget_mib=512,chunk_target_mib=16)
    import gf_native
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    completion=asyncio.run(_fit_component_dataset(manifest,expected_digest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        directory=tmp_path/'fit',model=model,settings=settings,training=TrainingConfig(),seed=17,native=gf_native))
    assert completion['status']=='complete' and completion['health']['exit_code']==0 and completion['health']['reaped'] is True
    pair=inspect_training(tmp_path/'fit')
    tree=load_artifact(tmp_path/'fit/artifact',expected_digest=completion['outputs']['inference_sha256'],model=model,
        action_config_hash=pair.inference['action_config_hash'],action_contract=engineering_contract(),purpose='component',
        synthetic_capacity=False,training_directory=tmp_path/'fit',training_binding=pair.binding)
    assert tree.origin.fit_origin=='trained' and tree.training_binding==pair.binding
    with pytest.raises(ContractError,match='paired-bundle copy'):save_artifact(tmp_path/'unpaired',tree)
    assert not (tmp_path/'unpaired').exists()


@pytest.mark.parametrize('outcome',['pending_timeout','creation_failure','eventual_success'])
def test_pending_fit_child_creation_preserves_original_cancellation_and_reaps(tmp_path,monkeypatch,outcome):
    import asyncio
    import sys
    import time

    from gf_training.structured_tree_fit import _OwnedTreeFitChild
    async def run():
        actual_create=asyncio.create_subprocess_exec
        entered=asyncio.Event();release=asyncio.Event();returned=[]
        async def delayed(*args,**options):
            entered.set();await release.wait()
            if outcome=='creation_failure':raise OSError('bounded spawn failure')
            process=await actual_create(sys.executable,'-c','import time;time.sleep(30)',
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            returned.append(process);return process
        monkeypatch.setattr(asyncio,'create_subprocess_exec',delayed)
        child=_OwnedTreeFitChild(tmp_path,time.monotonic()+2,1024)
        start=asyncio.create_task(child.start('a'*64))
        await entered.wait();start.cancel('original cancellation')
        await asyncio.sleep(0)
        if outcome=='pending_timeout':child.deadline=time.monotonic()+.03
        else:release.set()
        with pytest.raises(asyncio.CancelledError,match='original cancellation') as captured:await start
        original=captured.value
        assert original.args==('original cancellation',)
        if outcome=='pending_timeout':
            assert original._structured_tree_fit_child_cleanup_owner is child
            assert not child.creation.done() and not child.closed
            with pytest.raises(TimeoutError):await child.close(time.monotonic()+.01)
            assert not child.closed
            release.set()
        await child.close(time.monotonic()+3);await child.close(time.monotonic()+3)
        assert child.closed and child.creation.done()
        assert all(process.returncode is not None for process in returned)
        if returned:assert child.health['reaped'] is True
    asyncio.run(run())


def test_failed_spawn_close_is_confirmed_no_child_and_does_not_rethrow(tmp_path,monkeypatch):
    import asyncio
    import time

    from gf_training.structured_tree_fit import _OwnedTreeFitChild
    async def run():
        original=OSError('the original spawn failure')
        async def fail(*args,**kwargs):raise original
        monkeypatch.setattr(asyncio,'create_subprocess_exec',fail)
        child=_OwnedTreeFitChild(tmp_path,time.monotonic()+1,1024)
        with pytest.raises(OSError) as caught:await child.start('a'*64)
        assert caught.value is original
        await child.close(time.monotonic()+1);await child.close(time.monotonic()+1)
        assert child.closed and child.process is None
    asyncio.run(run())


def test_paired_snapshot_rechecks_original_ancestor_after_native_work(tmp_path):
    from gf_models.structured_tree_pairing import PairedTraining, TrainingBinding
    class Scan:
        def finalize(self):pass
        def check(self):pass
    ancestor=tmp_path/'original';ancestor.mkdir()
    info=ancestor.stat()
    owner=PairedTraining(TrainingBinding('/unused','a'*64,'b'*64),{},{},{},Scan(),{},
        {ancestor:(info.st_dev,info.st_ino)})
    owner.finalize()
    ancestor.rename(tmp_path/'held-original');ancestor.mkdir()
    with pytest.raises(ContractError,match='ancestor replaced'):owner.finalize()


def test_pending_spawn_failure_during_first_close_confirms_no_child(tmp_path,monkeypatch):
    import asyncio
    import time

    from gf_training.structured_tree_fit import _OwnedTreeFitChild
    async def run():
        release=asyncio.Event();started=asyncio.Event();original=OSError('resolved during cleanup')
        async def fail():
            started.set();await release.wait();raise original
        child=_OwnedTreeFitChild(tmp_path,time.monotonic()+1,1024)
        child.creation=asyncio.create_task(fail());await started.wait()
        closing=asyncio.create_task(child.close(time.monotonic()+1))
        await asyncio.sleep(0);release.set();await closing
        assert child.closed and child.creation.exception() is original and child.process is None
    asyncio.run(run())


def _actual_fit_setup(tmp_path):
    import hashlib
    import os
    from pathlib import Path

    from gf_common.preprocessing import PreparationConfig
    from gf_models.structured_tree_config import TreeEncoderConfigV2
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('preserved actual Dataset2 source fixture required')
    original=Path(root)/'dataset2/manifest.json'
    # This test-owned identical index still references every original Source;
    # mutations below affect ONLY the new index/numeric outputs, never Sources.
    manifest=tmp_path/'dataset-index'/'manifest.json';manifest.parent.mkdir();manifest.write_bytes(original.read_bytes())
    settings=PreparationConfig(backend='rust',reader='structured-v1',memory_budget_mib=2048,cache_budget_mib=512,chunk_target_mib=16)
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    return manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),settings,model


@pytest.mark.parametrize('fit_threads',[1,2])
def test_final_configured_fit_complete_default_and_asymmetric_threads(tmp_path,fit_threads):
    import asyncio
    import json
    import time

    from gf_models.config import TrainingConfig
    from gf_models.structured_tree_pairing import inspect_training
    from gf_training.structured_tree_data import VerifiedTreeDataset
    from gf_training.structured_tree_fit import _TreeFitRun
    manifest,sha,settings,model=_actual_fit_setup(tmp_path)
    import gf_native
    dataset=VerifiedTreeDataset(manifest,sha,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+7200,native=gf_native)
    owner=_TreeFitRun(dataset,tmp_path/'fit',model,TrainingConfig(num_threads=fit_threads),17)
    completion=asyncio.run(owner.execute())
    assert owner.closed and not dataset._active and dataset.source._closed
    assert owner.child.closed and owner.child.process.returncode==0 and completion['health']['reaped'] is True
    pair=inspect_training(tmp_path/'fit')
    assert pair.request['fit_settings']['num_threads']==fit_threads
    assert pair.request['execution']['inference']['num_threads']==1
    metrics=completion['metrics']
    assert metrics['configured_rounds']==500 and metrics['consumed_iterations']>=metrics['best_iteration']>=metrics['retained_tree_count']
    assert pair.request['row_counts']=={'train':192,'validation':320}
    assert owner.prepared.admission['total_bytes']<2*1024**3
    native_rss=int((tmp_path/'fit/stdout.log').read_text().strip().split('structured_tree_fit_peak_rss_kib=')[1])*1024
    assert native_rss>0
    # A standalone test measurement, never a new persisted model/fit schema or
    # claim that a conservative estimate is a physical RSS quota.
    (tmp_path/'measurement.json').write_text(json.dumps({'scope':'actual component native child only; not total concurrent RSS',
        'fit_threads':fit_threads,'inference_threads':1,'fit_admission':owner.prepared.admission,'native_child_peak_rss_bytes':native_rss,
        'metrics':metrics,'model_bytes':pair.inference['model_file']['bytes'],'elapsed_seconds':completion['costs']['elapsed_seconds']})+'\n')


@pytest.mark.parametrize('phase',['numeric_before_child','index_after_child'])
def test_actual_fit_mutated_own_numeric_or_index_never_completes(tmp_path,monkeypatch,phase):
    import asyncio
    import json
    import time

    import gf_training.structured_tree_fit as module
    from gf_models.config import TrainingConfig
    from gf_models.structured_tree_pairing import inspect_training
    from gf_training.structured_tree_data import VerifiedTreeDataset
    manifest,sha,settings,model=_actual_fit_setup(tmp_path)
    import gf_native
    dataset=VerifiedTreeDataset(manifest,sha,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+7200,native=gf_native)
    owner=module._TreeFitRun(dataset,tmp_path/'fit',model,TrainingConfig(),17)
    if phase=='numeric_before_child':
        actual=module._prepare_request
        def mutate(*args,**kw):
            request,sha,publisher=actual(*args,**kw)
            with next((tmp_path/'fit/inputs').glob('*.npz')).open('ab') as output:output.write(b'changed after publication')
            return request,sha,publisher
        monkeypatch.setattr(module,'_prepare_request',mutate)
    else:
        actual=module._OwnedTreeFitChild.wait
        async def mutate(child):
            await actual(child)
            with manifest.open('ab') as output:output.write(b' ')
        monkeypatch.setattr(module._OwnedTreeFitChild,'wait',mutate)
    with pytest.raises(ContractError):asyncio.run(owner.execute())
    assert not dataset._active and dataset.source._closed
    if owner.child is not None:assert owner.child.closed and owner.child.health['reaped'] is True
    else:assert phase=='numeric_before_child'
    assert json.loads((tmp_path/'fit/training.json').read_bytes())['status']=='failed'
    with pytest.raises((ContractError,ValueError)):inspect_training(tmp_path/'fit')


def test_actual_fit_cancel_retains_same_source_until_confirmed_child_cleanup(tmp_path,monkeypatch):
    import asyncio
    import time

    import gf_training.structured_tree_fit as module
    from gf_models.config import TrainingConfig
    from gf_models.structured_tree_pairing import inspect_training
    from gf_training.structured_tree_data import VerifiedTreeDataset
    manifest,sha,settings,model=_actual_fit_setup(tmp_path)
    import gf_native
    dataset=VerifiedTreeDataset(manifest,sha,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+7200,native=gf_native)
    owner=module._TreeFitRun(dataset,tmp_path/'fit',model,TrainingConfig(),17)
    actual_close=module._OwnedTreeFitChild.close
    async def run():
        reached=asyncio.Event();original_source=None
        async def wait(child):
            nonlocal original_source
            original_source=dataset.source;reached.set();await asyncio.Event().wait()
        async def unconfirmed(child,deadline):raise OSError('first cleanup unconfirmed')
        monkeypatch.setattr(module._OwnedTreeFitChild,'wait',wait)
        monkeypatch.setattr(module._OwnedTreeFitChild,'close',unconfirmed)
        task=asyncio.create_task(owner.execute());ready=asyncio.create_task(reached.wait())
        try:
            await asyncio.wait((task,ready),timeout=60,return_when=asyncio.FIRST_COMPLETED)
            if not reached.is_set():
                if task.done():await task
                raise TimeoutError('bounded test did not reach cancellation stage')
            task.cancel('original fit cancellation')
            with pytest.raises(asyncio.CancelledError,match='original fit cancellation') as caught:await task
            assert caught.value._structured_tree_fit_cleanup_owner is owner
            assert dataset._active and dataset.source is original_source and not dataset.source._closed
        finally:
            ready.cancel();await asyncio.gather(ready,return_exceptions=True)
            if not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            monkeypatch.setattr(module._OwnedTreeFitChild,'close',actual_close)
            await owner.close()
        assert owner.child.closed and owner.child.process.returncode is not None
        assert not dataset._active and dataset.source._closed
        with pytest.raises((ContractError,FileNotFoundError,ValueError)):inspect_training(tmp_path/'fit')
    asyncio.run(run())


def test_public_engineering_fit_guard_precedes_source_activity(monkeypatch,tmp_path):
    import asyncio

    import gf_training.structured_tree_fit as module
    from gf_common.action_contract import engineering_contract
    monkeypatch.setattr(module.VerifiedTreeDataset,'__enter__',lambda owner:pytest.fail('public guard reached Source'))
    with pytest.raises(ContractError,match='engineering'):
        asyncio.run(module.fit_dataset(tmp_path/'missing',action_contract=engineering_contract()))
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('body_error',[False,True])
def test_terminal_log_failure_confirms_reap_and_retires_source(tmp_path,monkeypatch,body_error):
    import asyncio
    import time
    from dataclasses import asdict
    from types import SimpleNamespace

    import gf_training.structured_tree_fit as module
    from gf_common.action_contract import engineering_contract
    from gf_models.config import TrainingConfig
    async def run():
        original=OSError('terminal log write failure')
        async def fail():raise original
        failed=asyncio.create_task(fail())
        await asyncio.sleep(0)
        child=module._OwnedTreeFitChild(tmp_path,time.monotonic()+10,1024)
        child.process=SimpleNamespace(returncode=0);child.drainers=[failed]
        class Dataset:
            deadline=time.monotonic()+10;settings=SimpleNamespace(cleanup_seconds=1);_active=True
            source=SimpleNamespace(_closed=False)
            def __init__(self):self.manifest={'input_binding':{'action_contract':asdict(engineering_contract()),'purpose':'component','synthetic_capacity':False}}
            def __enter__(self):return self
            def __exit__(self,kind,value,traceback):self._active=False;self.source._closed=True;self.original=value
        dataset=Dataset();owner=module._TreeFitRun(dataset,tmp_path,ConfiguredTreeModelV2(),TrainingConfig(),17);owner.child=child
        if body_error:
            initiating=asyncio.CancelledError('preserved fit body cancellation')
            def preparing(*args,**kw):raise initiating
            monkeypatch.setattr(module,'_PreparedTreeFit',preparing)
            with pytest.raises(asyncio.CancelledError) as caught:await owner.execute()
            assert caught.value is initiating and dataset.original is initiating
        else:
            with pytest.raises(OSError) as caught:await owner.close()
            assert caught.value is original and dataset.original is original
        assert child.closed and child.health['reaped'] is True and child.health['cleanup_failure']=='OSError'
        assert not dataset._active and dataset.source._closed
        await child.close(time.monotonic()+1);await owner.close()
        assert owner.closed
    asyncio.run(run())
