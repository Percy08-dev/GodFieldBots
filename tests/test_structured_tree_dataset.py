"""Original3-parent/three64-game actor-only component Dataset2 acceptance.

Predetermined FakeHTTP supplies public outcomes; no game engine or real Source
qualification is claimed. Original budgets/populations and actual children run.
"""
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gf_cli.config import LoadedConfig
from gf_common.action_contract import engineering_contract
from gf_common.domain import canonical, digest
from gf_common.preprocessing import preparation_config
from gf_models.structured_tree_config import TreeEncoderConfigV2
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.structured_ppo import _train_component_ppo
from gf_training.structured_tree_data import build_component_dataset, inspect_component_dataset
from gf_training.structured_tree_validation import FinalValidationContext
from tests.structured_ppo_support import component_http, configuration

native=pytest.importorskip('gf_native')
pytestmark=pytest.mark.skipif(not hasattr(native,'TreeDatasetSession'),reason='isolated C3b1 Dataset candidate required')


def reused_source(root,seed):
    from gf_common.domain import RuntimePolicyBinding
    from gf_common.schedule import serialize_schedule
    from gf_models.structured_binding import recorded_policy_label
    from gf_models.structured_config import InferenceBindingV2, recorded_model_config_v2
    from gf_training.collection import validation_groups
    parent=root/f'parent-{seed}';validation=root/f'validation-{seed}'
    state=json.loads((parent/'run.json').read_bytes());manifest=json.loads((validation/'manifest.json').read_bytes());report=json.loads((validation/'report.json').read_bytes())
    assert state['seed']==seed
    assert state['status']=='complete' and state['updates']==state['completed_updates']==2 and not state.get('quarantine')
    assert all(item['child_health']['exit_code']==0 for item in state['invocations'])
    assert manifest['status']=='complete' and len(manifest['episodes'])==64 and report['natural_denominator']==report['planned_games']==64 and report['comparison_valid']
    assert not any(report[key] for key in ('failures','health_failures','group_health_failures'))
    projected={**state['effective_config'],'run':{**state['effective_config']['run'],'split':'validation'}}
    assert manifest['effective_config']==projected and manifest['config_sha256']==digest(projected)
    artifact=parent/'artifacts/000002';inference=json.loads((artifact/'inference.json').read_bytes());sha=hashlib.sha256((artifact/'inference.json').read_bytes()).hexdigest()
    model=recorded_model_config_v2(inference['model'],inference['token_signature']);binding=InferenceBindingV2(**state['execution']['inference']['binding']);visibility=state['execution']['inference']['cuda_visible_devices']
    label=recorded_policy_label(model,binding,sha,cuda_visible_devices=visibility)
    options={'version':'2','checkpoint':str(artifact),'expected_manifest_digest':sha,'model':asdict(model),'inference':asdict(binding),
        'cuda_visible_devices':visibility,'action_contract':state['action_contract'],'purpose':'component','synthetic_capacity':False}
    registry={label:RuntimePolicyBinding(model.model_id,options),'random-structured-v1':RuntimePolicyBinding('random'),'heuristic-structured-v1':RuntimePolicyBinding('heuristic')}
    assert canonical(manifest['schedule'])==canonical(serialize_schedule(validation_groups(seed,label,actions_version=2),4))
    assert canonical(manifest['policy_bindings'])==canonical({key:asdict(value) for key,value in registry.items()})
    assert hashlib.sha256((validation/'report.json').read_bytes()).hexdigest()==manifest['report_sha256']
    inventory={}
    for folder in (parent,validation):
        for path in folder.rglob('*'):
            assert not path.is_symlink()
            if path.is_file() and path.name!='.lock':
                info=path.stat();inventory[str(path)]=(hashlib.sha256(path.read_bytes()).hexdigest(),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    return parent,validation,inventory


def reused_source17(root):
    return reused_source(root,17)


def check_reused_inventory(inventory):
    for name,expected in inventory.items():
        path=Path(name);info=path.stat()
        assert (hashlib.sha256(path.read_bytes()).hexdigest(),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)==expected,name


@pytest.fixture(scope='module')
def tree_cpu_template_owner(tmp_path_factory):
    """Explicit lazy binding; only original source collection requests warmup."""
    from tests.cpu_preload import CURRENT_TEMPLATE, CpuTemplate, supported
    if not supported():
        yield None
        return
    owner=CpuTemplate(tmp_path_factory.mktemp('tree-dataset-cpu-template'))
    token=CURRENT_TEMPLATE.set(owner)
    try:
        yield owner
    finally:
        try:
            asyncio.run(owner.close())
            assert not owner.records
            if owner.process is not None:
                assert owner.process.returncode is not None and owner._leader_reaped
        finally:
            CURRENT_TEMPLATE.reset(token)


@pytest.fixture(scope='module')
def original_tree_sources(tmp_path_factory,tree_cpu_template_owner):
    # One stable fixture, no recollection for each corruption/reader test.
    root=tmp_path_factory.mktemp('tree-dataset-originals')
    from _pytest.monkeypatch import MonkeyPatch
    parents=[];validations=[];observed=[]
    reused={}
    existing={seed:os.environ.get(f'GFB_TREE_DATASET_EXISTING_SOURCE{seed}') for seed in (17,29)}
    preload=tree_cpu_template_owner is not None
    fresh_seeds=0
    for index,seed in enumerate((17,29,43)):
        if existing.get(seed):
            parent,validation,snapshots=reused_source(Path(existing[seed]),seed)
            reused.update(snapshots)
            parents.append(parent);validations.append(validation)
            observed.append({'scope':f'UNMODIFIED historical original source reuse; creation test-{seed} remains ERROR',
                'parent':str(parent),'validation':str(validation),'inventory_files':len(snapshots)})
            continue
        monkeypatch=MonkeyPatch()
        try:
            loaded=configuration(root/f'config-{seed}')
            cfg=replace(loaded.config,training=replace(loaded.config.training,seeds=(seed,)))
            loaded=LoadedConfig(cfg,asdict(cfg),digest(asdict(cfg)),loaded.inputs)
            outcome={'win':{'winner':1}} if seed!=29 else 'draw'
            collector,_,workers,ledgers=component_http(monkeypatch,loaded,outcome=outcome,pool_cpu=False,preload_cpu=preload)
            parent=root/f'parent-{seed}'
            result=asyncio.run(_train_component_ppo(loaded,collector,parent))
            assert result['status']=='complete' and result['updates']==2
            assert len(workers)==64 and all(worker.closed and worker.process.poll() is not None and worker.lease.released
                and (worker.io_thread is None or not worker.io_thread.is_alive()) for worker in workers)
            assert ledgers[0].learner is None and not ledgers[0].workers
            validation=root/f'validation-{seed}'
            source_settings=preparation_config(asdict(replace(cfg.preprocessing,backend='rust',reader='structured-v1')))
            ledger=ReservationLedger(cfg.run.global_cuda_budget_bytes,cfg.run.max_participant_workers)
            async def collect(loaded=loaded,parent=parent,validation=validation,ledger=ledger,source_settings=source_settings,cfg=cfg,monkeypatch=monkeypatch,outcome=outcome,seed=seed):
                async with FinalValidationContext(loaded,parent,validation,ledger,source_settings=source_settings,
                        deadline=time.monotonic()+cfg.run.wall_seconds,native=native) as context:
                    assert canonical(asdict(context.projected.config))==canonical({**asdict(cfg),'run':{**asdict(cfg.run),'split':'validation'}})
                    selected,_,actors,seen=component_http(monkeypatch,context.projected,validation_context=context,
                        outcome=outcome,actors=(2,2) if seed==43 else (2,1,2,1),pool_cpu=False,preload_cpu=preload)
                    collected=await selected(context.plan,ledger)
                    evidence=context.accept_collection(collected.directory,collected.report)
                    assert evidence['qualified'] is False and evidence['scope'].startswith('collection-metadata-only')
                    assert len(actors)==128 and all(worker.closed and worker.process.poll() is not None and worker.lease.released
                        and (worker.io_thread is None or not worker.io_thread.is_alive()) for worker in actors)
                    assert seen and all(item is ledger for item in seen) and ledger.learner is None
                    return evidence
            evidence=asyncio.run(collect())
            assert not ledger.workers and ledger.learner is None
            fresh_seeds+=1
            if preload:
                assert tree_cpu_template_owner.launches==192*fresh_seeds
                assert not tree_cpu_template_owner.records and tree_cpu_template_owner.failure is None
                for directory in (*sorted((parent/'collections').iterdir()),validation):
                    receipt=json.loads((directory/'manifest.json').read_bytes())['pytest_worker_lifecycle']
                    assert receipt['mode']=='pytest-fresh-forked-cpu-v1' and receipt['closed']
                    assert receipt['cpu_preload']['template_pid']==tree_cpu_template_owner.process.pid
                    assert receipt['cpu_preload']['healthy'] and receipt['cpu_preload']['unretired_tickets']==0
            parents.append(parent);validations.append(validation);observed.append(evidence)
        finally:monkeypatch.undo()
    (root/'fixture-evidence.json').write_text(canonical({'scope':'component originals only','parents':[str(p) for p in parents],
        'validations':[str(p) for p in validations],'evidence':observed})+'\n')
    check_reused_inventory(reused)
    return root,parents,validations,source_settings,reused


def dataset_plan(parents,validations,window=1):
    return {'schema_version':2,'kind':'structured-tree-dataset-plan',
        'parents':[{'directory':str(p),'validation_directory':str(v)} for p,v in zip(parents,validations,strict=True)],
        'encoder':asdict(TreeEncoderConfigV2(aggregate_window=window)),'action_contract':asdict(engineering_contract()),
        'purpose':'component','synthetic_capacity':False}


def test_three_original_parents_actor64_and_source_manifest_roundtrip(original_tree_sources):
    root,parents,validations,settings,reused=original_tree_sources
    plan=dataset_plan(parents,validations)
    output=root/'dataset2'
    manifest=build_component_dataset(plan,output,settings,max_owned_bytes=2*1024**3,deadline=time.monotonic()+7200,native=native)
    assert manifest['schema_version']==2 and manifest['counts']['train']['games']==96
    assert manifest['counts']['validation']['games']==192
    assert manifest['counts']['validation']['zero_endpoint_games']==32
    assert [p['seed'] for p in manifest['parents']]==[17,29,43]
    assert len(manifest['collections'])==9
    assert manifest['neural_cohort']['qualified'] is False and manifest['neural_cohort']['parent_qualification']=='not_performed'
    sha=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    checked=inspect_component_dataset(output/'manifest.json',expected_digest=sha,settings=settings,max_owned_bytes=2*1024**3,
                                      deadline=time.monotonic()+7200,native=native)
    assert canonical(checked)==canonical(manifest)
    check_reused_inventory(reused)
