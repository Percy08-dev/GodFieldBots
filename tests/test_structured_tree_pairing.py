"""Actual complete Training3 bytes: pairing, root routing and immutable origin."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError, canonical
from gf_models.structured_tree_config import (
    ConfiguredTreeModelV2,
    TreeEncoderConfigV2,
    TreeInferenceConfigV2,
    policy_label,
)
from gf_models.structured_tree_pairing import inspect_training, resolve_checkpoint


@pytest.fixture
def complete_bytes(tmp_path):
    supplied=os.environ.get('GFB_TREE_TRAINED_FIXTURE')
    if not supplied:pytest.skip('explicit preserved actual Source-backed complete Training3 fixture required')
    source=Path(supplied)
    inspect_training(source)
    target=tmp_path/'training';shutil.copytree(source,target)
    return target


def _write(path,value):path.write_text(canonical(value)+'\n')


def test_actual_pair_root_routing_active_override_and_exact_bundle_copy(complete_bytes,tmp_path):
    from gf_models.structured_tree_artifacts import binding_options, load_artifact, save_artifact
    from gf_models.structured_tree_encoder import safe_tree_options_v2
    from gf_models.structured_tree_pairing import copy_training_bundle
    original=inspect_training(complete_bytes)
    artifact,sha,binding=resolve_checkpoint(complete_bytes)
    assert artifact==complete_bytes/'artifact' and binding==original.binding
    with pytest.raises(ContractError,match='bare trained'):resolve_checkpoint(artifact)
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1),
        inference=TreeInferenceConfigV2(candidate_chunk=1,num_threads=2))
    options=binding_options(complete_bytes,expected_digest=sha,model=model,action_config_hash=original.inference['action_config_hash'],
        action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
    assert len(options)==12 and options['training_binding']['directory']==str(complete_bytes)
    assert safe_tree_options_v2(options,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)==options
    tree=load_artifact(artifact,expected_digest=sha,model=model,action_config_hash=original.inference['action_config_hash'],
        action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,
        training_directory=complete_bytes,training_binding=binding)
    assert json.loads(tree.creation_execution)['inference']['num_threads']==1
    assert tree.creation_execution==canonical(original.inference['creation_execution']).encode()
    with pytest.raises(ContractError,match='paired-bundle copy'):save_artifact(tmp_path/'unpaired-resave',tree)
    assert not (tmp_path/'unpaired-resave').exists()
    assert tree.model.inference.num_threads==2 and tree.origin.provenance.to_dict()==original.inference['provenance']
    assert policy_label(model,tree.descriptor,sha,tree.training_binding)!=policy_label(model,tree.descriptor,sha)
    copied=copy_training_bundle(complete_bytes,tmp_path/'exact-copy')
    assert copied.request_sha256==binding.request_sha256 and copied.completion_sha256==binding.completion_sha256
    for relative in (*original.completion['outputs']['files'],'training.json'):
        assert (complete_bytes/relative).read_bytes()==(tmp_path/'exact-copy'/relative).read_bytes()
    assert inspect_training(tmp_path/'exact-copy').inference['provenance']==original.inference['provenance']


@pytest.mark.parametrize('case',['missing_completion','failed','request_hash','model_hash','input_hash','unknown_completion','ambiguous'])
def test_pair_rejects_incomplete_resealed_and_mixed_roots(complete_bytes,case):
    completion=json.loads((complete_bytes/'training.json').read_bytes())
    if case=='missing_completion':(complete_bytes/'training.json').unlink()
    elif case=='failed':completion['status']='failed';_write(complete_bytes/'training.json',completion)
    elif case=='request_hash':completion['request_sha256']='0'*64;_write(complete_bytes/'training.json',completion)
    elif case=='model_hash':(complete_bytes/'artifact/model.txt').write_bytes(b'wrong native model')
    elif case=='input_hash':
        path=next((complete_bytes/'inputs').glob('*.npz'));path.write_bytes(b'wrong own numeric inputs')
    elif case=='unknown_completion':completion['ready']=True;_write(complete_bytes/'training.json',completion)
    else:(complete_bytes/'inference.json').write_bytes((complete_bytes/'artifact/inference.json').read_bytes())
    with pytest.raises((ContractError,FileNotFoundError)):resolve_checkpoint(complete_bytes)


def test_wrong_training_dto_rejects_before_native_construction(complete_bytes,monkeypatch):
    from dataclasses import asdict

    import gf_models.structured_tree_artifacts as artifacts
    from gf_models.structured_baselines import make_structured_policy
    pair=inspect_training(complete_bytes)
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    options=artifacts.binding_options(complete_bytes,expected_digest=pair.completion['outputs']['inference_sha256'],model=model,
        action_config_hash=pair.inference['action_config_hash'],action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
    monkeypatch.setattr(artifacts,'_native',lambda **kw:pytest.fail('native construction preceded complete pair validation'))
    for binding in ({**options['training_binding'],'request_sha256':'0'*64},
                    {**options['training_binding'],'completion_sha256':'0'*64},
                    {**options['training_binding'],'directory':str(complete_bytes/'artifact')}):
        with pytest.raises((ContractError,FileNotFoundError)):
            make_structured_policy('lightgbm',{**options,'training_binding':binding},
                action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,
                action_config_hash=pair.inference['action_config_hash'])
    assert asdict(model.architecture())==pair.request['model']


def test_raw_toy_fit_cannot_claim_trained_origin_before_native(complete_bytes,monkeypatch):
    import numpy as np

    import gf_models.structured_tree_artifacts as artifacts
    from gf_models.structured_tree_pairing import provenance_for_request
    pair=inspect_training(complete_bytes)
    origin=artifacts.TreeOrigin(pair.inference['action_config_hash'],engineering_contract(),'component',False,'trained',
        provenance_for_request(pair.request,pair.binding.request_sha256))
    monkeypatch.setattr(artifacts,'_native',lambda **kw:pytest.fail('raw toy arrays reached native trained fit'))
    with pytest.raises(ContractError,match='toy Tree2 fit origin'):
        artifacts.fit_toy([np.zeros((1,1647),np.float32)],np.zeros(1,np.float32),[np.zeros((1,1647),np.float32)],
            np.zeros(1,np.float32),model=ConfiguredTreeModelV2(),origin=origin,seed=17)


def test_pair_native_fit_threads_bound_independently_from_active_inference(complete_bytes):
    request=json.loads((complete_bytes/'request.json').read_bytes())
    inference=json.loads((complete_bytes/'artifact/inference.json').read_bytes())
    completion=json.loads((complete_bytes/'training.json').read_bytes())
    request['fit_settings']['num_threads']=2
    _write(complete_bytes/'request.json',request)
    request_sha=hashlib.sha256((complete_bytes/'request.json').read_bytes()).hexdigest()
    inference['provenance']['request_sha256']=request_sha;_write(complete_bytes/'artifact/inference.json',inference)
    inference_sha=hashlib.sha256((complete_bytes/'artifact/inference.json').read_bytes()).hexdigest()
    completion['request_sha256']=request_sha;completion['outputs']['files']['request.json']=request_sha
    completion['outputs']['inference_sha256']=inference_sha;completion['outputs']['files']['artifact/inference.json']=inference_sha
    _write(complete_bytes/'training.json',completion)
    with pytest.raises(ContractError,match='actual native fitted parameters'):inspect_training(complete_bytes)


def test_trained_cpu_worker_ready_journal_and_raw_role_use_full_binding(make_input,tmp_path,complete_bytes):
    import asyncio
    import time
    import uuid
    from dataclasses import asdict

    from gf_common.domain import digest
    from gf_common.structured import EventBank, action_config_v2
    from gf_models.structured_baselines import make_structured_policy
    from gf_models.structured_tree_artifacts import binding_options
    from gf_simulator_client.structured_resources import ReservationLedger
    from gf_simulator_client.structured_worker import StructuredParticipantWorker
    from gf_training.structured_journal import StructuredJournal, read_structured_journal
    from gf_training.structured_source import _header_identity
    from tests.test_structured_protocol import own_frame
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('original own action recipe required')
    dataset=json.loads((Path(root)/'dataset2/manifest.json').read_bytes())
    state=json.loads((Path(dataset['parents'][0]['directory'])/'run.json').read_bytes())
    actions=action_config_v2(state['effective_config']['actions']);pair=inspect_training(complete_bytes)
    assert digest(asdict(actions))==pair.inference['action_config_hash']
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    options=binding_options(complete_bytes,expected_digest=pair.completion['outputs']['inference_sha256'],model=model,
        action_config_hash=digest(asdict(actions)),action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
    async def run():
        ledger=ReservationLedger();worker=StructuredParticipantWorker(make_structured_policy,'lightgbm',17,actions,ledger=ledger,policy_options=options)
        try:
            await worker.initialize()
            assert worker.initialized
            bank=EventBank(1,'trained-component')
            directory=tmp_path/'journal';directory.mkdir()
            journal=StructuredJournal(directory,'episode',1,worker.expected_policy,run_id='run',game_id='game',
                action_config_hash=digest(asdict(actions)),bank_episode=bank.episode,caps=asdict(worker.caps),numeric_costs=asdict(worker.costs),
                model_id='lightgbm',model_version='2',aggregate_window=1,sampling_seed=17,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
            for revision in (0,2):
                frame,delta=own_frame(make_input(),bank,revision,[{'type':'action_accepted','player':1}])
                generated,selected=await worker.select_frame(frame,delta)
                assert selected.logprob is selected.value is None and selected.policy_version==worker.expected_policy
                command={'command_id':str(uuid.uuid4()),'expected_revision':frame.revision,'action':selected.validate(generated.table,frame)}
                index=journal.begin_frame(frame,delta,generated,selected,command,{})
                journal.acknowledge(index,{'command_id':command['command_id'],'previous_revision':revision,'state_revision':revision+1,'status':'active','result':None})
            journal.finish('truncated',None)
            raw=read_structured_journal(journal.final_path,action_contract=engineering_contract(),purpose='component',synthetic_capacity=False)
            manifest={'run_id':'run','policy_bindings':{worker.expected_policy:{'model_id':'lightgbm','policy_options':options}}}
            spec={'episode_id':'episode','policies':[worker.expected_policy,'random-structured-v1'],'sampling_seeds':[17,19]}
            _header_identity(raw.header,manifest,{'game_id':'game'},spec,actions,1)
            assert raw.accepted==(0,1) and raw.header['generation_schema'] is None
            if os.environ.get('GFB_RUN_STRUCTURED_TREE_NATIVE')=='1':
                import gf_native
                native=gf_native.StructuredJournalTokens(str(directory),journal.final_path.name,
                    canonical({'memory_bytes':512*1024**2,'max_json_bytes':64*1024**2,'max_record_bytes':8*1024**2}).encode(),time.monotonic()+30)
                try:assert json.loads(native.diagnostic())['header']['policy']==worker.expected_policy
                finally:native.close()
        finally:await worker.close()
        assert worker.closed and worker.process.poll() is not None and not ledger.workers
    asyncio.run(run())


def _completion_publication(complete_bytes):
    import time

    from gf_training.structured_lineage import _ChainScan, _write
    from gf_training.structured_tree_fit import _CompletionPublication
    complete=json.loads((complete_bytes/'training.json').read_bytes())
    scan=_ChainScan(directory=complete_bytes,max_owned_bytes=2*1024**3,deadline=time.monotonic()+30)
    running={**complete,'status':'running'}
    _write(complete_bytes/'training.json',running,scan=scan)
    info=complete_bytes.stat();scan._tree_fit_directory_identity=(info.st_dev,info.st_ino);scan._tree_fit_ancestors={}
    return _CompletionPublication(complete_bytes,scan,time.monotonic()+35,2*1024**3),complete


@pytest.mark.parametrize('exception',['timeout','cancel','postcheck'])
def test_late_owned_completion_failure_revokes_even_when_failed_publisher_unavailable(complete_bytes,monkeypatch,exception):
    import asyncio
    import time

    import gf_training.structured_lineage as lineage
    publication,complete=_completion_publication(complete_bytes)
    original={'timeout':TimeoutError('post-atomic work deadline'),'cancel':asyncio.CancelledError('original late cancel'),
              'postcheck':ContractError('post-atomic checksum check')}[exception]
    postcheck=publication._postcheck
    def fail_after_healthy_atomic_commit():
        postcheck()
        publication.publisher.deadline=time.monotonic()-1
        raise original
    monkeypatch.setattr(publication,'_postcheck',fail_after_healthy_atomic_commit)
    monkeypatch.setattr(lineage,'_write',lambda *a,**kw:pytest.fail('ordinary failed rewrite must not run after healthy publication'))
    with pytest.raises(type(original)) as caught:publication.commit(complete)
    assert caught.value is original and publication.revoked and publication.fd is None and not publication.committed
    assert (complete_bytes/'training.json').read_bytes().startswith(b'!')
    with pytest.raises((ContractError,ValueError)):inspect_training(complete_bytes)


def test_unconfirmed_completion_invalidation_preserves_original_and_exact_fd_owner(complete_bytes,monkeypatch):
    import errno
    import weakref
    publication,complete=_completion_publication(complete_bytes)
    original=KeyboardInterrupt('original caller interruption')
    monkeypatch.setattr(publication,'_postcheck',lambda:(_ for _ in ()).throw(original))
    actual=os.pwrite
    def unavailable(*args):raise OSError(errno.EIO,'bounded invalidation fault')
    monkeypatch.setattr(os,'pwrite',unavailable)
    with pytest.raises(KeyboardInterrupt) as caught:publication.commit(complete)
    assert caught.value is original and original._structured_completion_cleanup_owner is publication
    assert publication.fd is not None and not publication.revoked
    assert any('completion revocation unconfirmed' in note for note in original.__notes__)
    reference=weakref.ref(publication);del publication
    assert reference() is original._structured_completion_cleanup_owner
    monkeypatch.setattr(os,'pwrite',actual)
    original._structured_completion_cleanup_owner.revoke()
    assert reference().fd is None and reference().revoked
    with pytest.raises((ContractError,ValueError)):inspect_training(complete_bytes)


@pytest.mark.parametrize('same_bytes',[False,True])
def test_completion_foreign_replacement_is_preserved_owned_inode_only_invalidated(complete_bytes,tmp_path,monkeypatch,same_bytes):
    publication,complete=_completion_publication(complete_bytes)
    original_postcheck=publication._postcheck;held=[]
    replacement=(canonical(complete)+'\n').encode() if same_bytes else b'preexisting-foreign-evidence'
    def replace_after_checks():
        original_postcheck()
        held.append(os.dup(publication.fd))
        foreign=tmp_path/'foreign';foreign.write_bytes(replacement)
        os.replace(foreign,complete_bytes/'training.json')
    monkeypatch.setattr(publication,'_postcheck',replace_after_checks)
    try:
        with pytest.raises(ContractError) as caught:publication.commit(complete)
        assert (complete_bytes/'training.json').read_bytes()==replacement
        assert publication.revoked and os.pread(held[0],1,0)==b'!'
        assert any('foreign current path preserved' in note for note in caught.value.__notes__)
        if same_bytes:
            # This independently coherent foreign object is outside the failed
            # publisher's revocation authority, exactly as the contract states.
            assert inspect_training(complete_bytes).completion['status']=='complete'
    finally:
        for descriptor in held:os.close(descriptor)


def test_owned_completion_publication_normal_success_and_exact_copy(complete_bytes,tmp_path):
    from gf_models.structured_tree_pairing import copy_training_bundle
    publication,complete=_completion_publication(complete_bytes)
    publication.commit(complete)
    assert publication.committed and publication.fd is None and not publication.revoked
    paired=inspect_training(complete_bytes)
    copied=copy_training_bundle(complete_bytes,tmp_path/'completed-copy')
    assert copied.completion_sha256==paired.binding.completion_sha256


def test_completion_invalidation_precedes_failing_namespace_diagnostics(complete_bytes,monkeypatch):
    publication,complete=_completion_publication(complete_bytes)
    original=TimeoutError('the original post-atomic failure')
    actual=Path.lstat;armed=False
    def fail_diagnostic(path,*args,**kw):
        if armed and path==complete_bytes/'training.json':raise MemoryError('diagnostic allocation fault')
        return actual(path,*args,**kw)
    def postcheck():
        nonlocal armed
        armed=True
        raise original
    monkeypatch.setattr(Path,'lstat',fail_diagnostic)
    monkeypatch.setattr(publication,'_postcheck',postcheck)
    with pytest.raises(TimeoutError) as caught:publication.commit(complete)
    assert caught.value is original and publication.revoked and publication.fd is None
    monkeypatch.setattr(Path,'lstat',actual)
    with pytest.raises((ContractError,ValueError)):inspect_training(complete_bytes)


def test_explicit_training_root_cli_metadata_without_torch_or_booster(complete_bytes,tmp_path):
    import subprocess
    import sys
    root=os.environ.get('GFB_TREE_DATASET_FROZEN_FIXTURE')
    if not root:pytest.skip('original own action recipe required')
    dataset=json.loads((Path(root)/'dataset2/manifest.json').read_bytes())
    state=json.loads((Path(dataset['parents'][0]['directory'])/'run.json').read_bytes())
    config=tmp_path/'actions.json';config.write_text(canonical(state['effective_config']['actions']))
    script='''import builtins,importlib.metadata,json,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','lightgbm'):raise AssertionError('optional native/NN import during metadata pairing')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
importlib.metadata.distribution=lambda *a,**k: (_ for _ in ()).throw(AssertionError('current runtime metadata queried'))
from gf_common.structured import action_config_v2
from gf_models.structured_tree_config import ConfiguredTreeModelV2,TreeEncoderConfigV2
from gf_cli.bindings import tree2_binding
from gf_models.structured_tree_pairing import inspect_training
model=ConfiguredTreeModelV2(checkpoint=sys.argv[1],encoder=TreeEncoderConfigV2(aggregate_window=1))
actions=action_config_v2(json.loads(open(sys.argv[2]).read()))
label,binding=tree2_binding(model,actions)
assert len(binding.policy_options)==12 and binding.policy_options['training_binding']['directory']==sys.argv[1]
assert label.startswith('lightgbm-tree-v2-')
assert not ({'torch','lightgbm'} & sys.modules.keys())
'''
    subprocess.run([sys.executable,'-c',script,str(complete_bytes),str(config)],check=True,capture_output=True)


@pytest.mark.parametrize('kind',['timeout','cancel','postcheck'])
def test_exact_copy_late_failure_revokes_owned_completion(complete_bytes,tmp_path,monkeypatch,kind):
    import asyncio
    import time

    from gf_models.structured_tree_pairing import _CopiedCompletion, copy_training_bundle
    original={'timeout':TimeoutError('copy postcheck timeout'),'cancel':asyncio.CancelledError('copy initiating cancellation'),
        'postcheck':MemoryError('copy final check fault')}[kind]
    actual=_CopiedCompletion._postcheck;owners=[]
    def fail(owner):
        actual(owner);owners.append(owner);raise original
    monkeypatch.setattr(_CopiedCompletion,'_postcheck',fail)
    with pytest.raises(type(original)) as caught:
        copy_training_bundle(complete_bytes,tmp_path/'copy',deadline=time.monotonic()+30)
    assert caught.value is original and owners[0].revoked and owners[0].fd is None
    with pytest.raises((ContractError,ValueError)):inspect_training(tmp_path/'copy')
    assert inspect_training(complete_bytes).completion['status']=='complete'


def test_exact_copy_unconfirmed_invalidation_retains_exact_owner_fd(complete_bytes,tmp_path,monkeypatch):
    import time

    from gf_models.structured_tree_pairing import _CopiedCompletion, copy_training_bundle
    original=TimeoutError('copy initiating failure');actual=os.pwrite
    monkeypatch.setattr(_CopiedCompletion,'_postcheck',lambda owner:(_ for _ in ()).throw(original))
    monkeypatch.setattr(os,'pwrite',lambda *a:(_ for _ in ()).throw(OSError('copy invalidation failed')))
    with pytest.raises(TimeoutError) as caught:copy_training_bundle(complete_bytes,tmp_path/'copy',deadline=time.monotonic()+30)
    assert caught.value is original
    owner=original._structured_copy_cleanup_owner
    assert owner.fd is not None and not owner.revoked
    assert any('completion revocation unconfirmed' in note for note in original.__notes__)
    monkeypatch.setattr(os,'pwrite',actual);owner.revoke()
    assert owner.revoked and owner.fd is None
    with pytest.raises((ContractError,ValueError)):inspect_training(tmp_path/'copy')


@pytest.mark.parametrize('same_bytes',[False,True])
def test_exact_copy_foreign_replacement_preserved_owned_fd_only(complete_bytes,tmp_path,monkeypatch,same_bytes):
    import time

    from gf_models.structured_tree_pairing import _CopiedCompletion, copy_training_bundle
    actual=_CopiedCompletion._postcheck;held=[];owners=[];replacement=[]
    def swap(owner):
        binding=actual(owner);owners.append(owner);held.append(os.dup(owner.fd))
        raw=(owner.target/'training.json').read_bytes() if same_bytes else b'foreign-copy-evidence'
        replacement.append(raw);path=tmp_path/'foreign';path.write_bytes(raw);os.replace(path,owner.target/'training.json')
        return binding
    monkeypatch.setattr(_CopiedCompletion,'_postcheck',swap)
    try:
        with pytest.raises(ContractError):copy_training_bundle(complete_bytes,tmp_path/'copy',deadline=time.monotonic()+30)
        assert (tmp_path/'copy/training.json').read_bytes()==replacement[0]
        assert owners[0].revoked and os.pread(held[0],1,0)==b'!'
        if same_bytes:assert inspect_training(tmp_path/'copy').completion['status']=='complete'
    finally:
        for fd in held:os.close(fd)


@pytest.mark.parametrize('alias',['dotdot','symlink'])
def test_exact_copy_alias_cannot_write_inside_immutable_original(complete_bytes,tmp_path,monkeypatch,alias):
    import gf_models.structured_tree_artifacts as artifacts
    from gf_models.structured_tree_pairing import copy_training_bundle
    other=tmp_path/'other';other.mkdir()
    if alias=='dotdot':target=other/'..'/complete_bytes.name/'copied'
    else:
        link=tmp_path/'alias';link.symlink_to(complete_bytes,target_is_directory=True);target=link/'copied'
    def snapshot():
        return {p.relative_to(complete_bytes).as_posix():(hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
            tuple(getattr(p.lstat(),key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')))
            for p in (complete_bytes,*complete_bytes.rglob('*'))}
    before=snapshot()
    monkeypatch.setattr(artifacts,'_publish',lambda *a,**kw:pytest.fail('alias reached output publisher'))
    monkeypatch.setattr(Path,'mkdir',lambda *a,**kw:pytest.fail('alias reached output mkdir'))
    with pytest.raises(ContractError):copy_training_bundle(complete_bytes,target)
    assert snapshot()==before


def test_trained_policy_deadline_expiry_after_load_precedes_extra_io_native(complete_bytes,monkeypatch):
    import gf_models.structured_tree_pairing as pairing
    import gf_models.structured_tree_policy as policy
    pair=inspect_training(complete_bytes);clock=[0.]
    actual_load=policy.load_artifact
    def late_load(*args,**kw):
        tree=actual_load(*args,**kw);clock[0]=11.
        monkeypatch.setattr(pairing,'inspect_training',lambda *a,**k:pytest.fail('expired policy performed extra paired I/O'))
        monkeypatch.setattr(policy,'_validate_booster',lambda *a,**k:pytest.fail('expired policy performed extra native validation'))
        return tree
    monkeypatch.setattr(policy,'load_artifact',late_load)
    monkeypatch.setattr(policy.time,'monotonic',lambda:clock[0])
    with pytest.raises(TimeoutError):policy.StructuredTreePolicy.from_artifact(complete_bytes/'artifact',
        expected_digest=pair.completion['outputs']['inference_sha256'],model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1)),
        action_config_hash=pair.inference['action_config_hash'],action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,
        training_directory=complete_bytes,training_binding=pair.binding,deadline=10.)


def test_trained_policy_checks_same_deadline_after_booster_validation(complete_bytes,monkeypatch):
    import gf_models.structured_tree_policy as policy
    pair=inspect_training(complete_bytes)
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    tree=policy.load_artifact(complete_bytes/'artifact',expected_digest=pair.completion['outputs']['inference_sha256'],model=model,
        action_config_hash=pair.inference['action_config_hash'],action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,
        training_directory=complete_bytes,training_binding=pair.binding)
    clock=[0.];actual=policy._validate_booster
    monkeypatch.setattr(policy.time,'monotonic',lambda:clock[0])
    def long_validation(*args,**kw):
        assert kw['deadline']==10.
        value=actual(*args,**kw);clock[0]=11.;return value
    monkeypatch.setattr(policy,'_validate_booster',long_validation)
    with pytest.raises(TimeoutError):policy.StructuredTreePolicy(tree,pair.completion['outputs']['inference_sha256'],deadline=10.)


@pytest.mark.parametrize('substitution',['symlink','real_directory'])
def test_copy_target_substitution_cannot_redirect_writes_outside_owned_root(complete_bytes,tmp_path,monkeypatch,substitution):
    from gf_models.structured_tree_pairing import copy_training_bundle
    target=tmp_path/'copy';outside=tmp_path/'foreign';outside.mkdir()
    (outside/'artifact').mkdir();(outside/'inputs').mkdir()
    (outside/'evidence').write_bytes(b'unchanged foreign evidence')
    before={str(p.relative_to(outside)):(p.read_bytes() if p.is_file() else None,
        tuple(getattr(p.lstat(),key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')))
        for p in (outside,*outside.rglob('*'))}
    actual=os.open;triggered=False
    def swap(path,flags,*args,**kw):
        nonlocal triggered
        if not triggered and path=='artifact' and 'dir_fd' in kw:
            triggered=True;target.rename(tmp_path/'held-owned-copy')
            if substitution=='symlink':target.symlink_to(outside,target_is_directory=True)
            else:
                target.mkdir();(target/'artifact').mkdir();(target/'inputs').mkdir()
        return actual(path,flags,*args,**kw)
    monkeypatch.setattr(os,'open',swap)
    with pytest.raises(ContractError,match='target directory replaced'):copy_training_bundle(complete_bytes,target)
    assert triggered
    after={str(p.relative_to(outside)):(p.read_bytes() if p.is_file() else None,
        tuple(getattr(p.lstat(),key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')))
        for p in (outside,*outside.rglob('*'))}
    assert after==before
    assert not (tmp_path/'held-owned-copy/artifact/model.txt').exists()
    assert inspect_training(complete_bytes).completion['status']=='complete'


def test_trained_policy_same_pair_rechecks_mutation_during_booster_validation(complete_bytes,monkeypatch):
    import gf_models.structured_tree_policy as policy
    pair=inspect_training(complete_bytes)
    model=ConfiguredTreeModelV2(encoder=TreeEncoderConfigV2(aggregate_window=1))
    tree=policy.load_artifact(complete_bytes/'artifact',expected_digest=pair.completion['outputs']['inference_sha256'],model=model,
        action_config_hash=pair.inference['action_config_hash'],action_contract=engineering_contract(),purpose='component',synthetic_capacity=False,
        training_directory=complete_bytes,training_binding=pair.binding)
    actual=policy._validate_booster
    def mutate(*args,**kw):
        value=actual(*args,**kw)
        with (complete_bytes/'training.json').open('ab') as output:output.write(b' ')
        return value
    monkeypatch.setattr(policy,'_validate_booster',mutate)
    with pytest.raises(ContractError,match='sealed'):
        policy.StructuredTreePolicy(tree,pair.completion['outputs']['inference_sha256'])
