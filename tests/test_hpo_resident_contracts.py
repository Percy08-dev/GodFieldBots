"""Explicit resident variants preserve fresh identities and require live custody."""
from dataclasses import asdict

import pytest

from gf_common.domain import ContractError
from tests.hpo_resident_support import retirement_fact
from tests.test_hpo_config import campaign_raw

RUNTIME = {"mode": "resident-worker-v1", "cache_entries": 6, "cache_weight_bytes": 134217728}


def test_campaign3_runtime_is_explicit_and_old_hash_and_asdict_stay_exact(tmp_path):
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.domain import digest
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.hpo_config import effective_settings
    raw = campaign_raw()
    old = campaign_config(raw, origin=tmp_path)
    old_hash = old.sha256
    assert "runtime" not in asdict(old.config)
    fresh = campaign_config(raw, origin=tmp_path)
    assert fresh.sha256 == old_hash
    resident = campaign_config({**raw, "schema_version": 3, "runtime": RUNTIME}, origin=tmp_path)
    assert asdict(resident.config)["runtime"] == RUNTIME
    assert resident.config.study_identity != old.config.study_identity
    settings = effective_settings(resident.config.model, resident.config.training, {}, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    run = materialize_hpo_run(resident, settings, learning_seed=17, run_id="00000000-0000-0000-0000-000000000017")
    assert run.config.hpo_spec["schema_version"] == 3
    assert run.config.hpo_spec["runtime"] == RUNTIME
    assert run.sha256 == digest(asdict(run.config))
    assert campaign_config(raw, origin=tmp_path).sha256 == old_hash


@pytest.mark.parametrize("runtime", [{}, {**RUNTIME, "cache_entries": 7}, {**RUNTIME, "cache_entries": True},
    {**RUNTIME, "cache_weight_bytes": 1}, {**RUNTIME, "mode": "fresh"}, {**RUNTIME, "extra": 1}])
def test_campaign3_refuses_missing_foreign_or_unfrozen_runtime(tmp_path, runtime):
    from gf_cli.hpo_config import campaign_config
    with pytest.raises(ContractError):
        campaign_config({**campaign_raw(), "schema_version": 3, "runtime": runtime}, origin=tmp_path)


def test_old_campaign_forbids_resident_runtime(tmp_path):
    from gf_cli.hpo_config import campaign_config
    with pytest.raises(ContractError):
        campaign_config({**campaign_raw(), "runtime": RUNTIME}, origin=tmp_path)


def test_resident_memory_keeps_idle_cache_staging_source_and_save_charged(tmp_path):
    from gf_cli.hpo_config import campaign_config, joint_resource_admission
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    old = campaign_config(campaign_raw(), origin=tmp_path)
    resident = campaign_config({**campaign_raw(), "schema_version": 3, "runtime": RUNTIME}, origin=tmp_path)
    before = joint_resource_admission(old.config, layout=layout)
    after = joint_resource_admission(resident.config, layout=layout)
    assert after["recipe"] == "hpo-resident-cache-envelope-v2"
    assert after["worker_bytes"] > before["worker_bytes"]
    assert after["ram_phases"]["save"] == before["ram_phases"]["save"] + after["worker_bytes"]
    assert after["ram_phases"]["sync"] >= after["worker_bytes"] + after["learner_bytes"] + after["staging_bytes"]
    assert after["peak_cuda_bytes"] >= after["worker_bytes"] + after["learner_bytes"] + after["staging_bytes"]


def test_run5_spec3_and_registry5_are_explicit_readable_variants(tmp_path):
    import time

    from gf_training.hpo_lineage import validate_hpo_chain
    from tests.hpo_lineage_support import initial_hpo_chain
    root = tmp_path / 'resident'
    state, records, _ = initial_hpo_chain(root, resident=True)
    actual, generations = validate_hpo_chain(root, max_owned_bytes=1024**3, deadline=time.monotonic()+30)
    from gf_common.domain import canonical
    assert canonical(actual) == canonical(state) and canonical(generations) == canonical(records)


def test_cache_pin_has_separate_durable_namespace_and_cannot_release_from_json(tmp_path):
    import time

    from gf_models.sharded_arrays import read_json
    from gf_training import hpo_retention as retention
    from tests.hpo_lineage_support import initial_hpo_chain
    root = tmp_path / 'resident'
    _, records, _ = initial_hpo_chain(root, resident=True)
    owner = {'job_ticket': 'a'*32, 'pool_id': 'b'*32, 'pool_epoch': 1,
        'worker_key': 1, 'pid': 123, 'viewer': 1, 'lease_epoch': 1}
    opts = {'max_owned_bytes': 1024**3, 'deadline': time.monotonic()+30}
    pin = retention.pin_hpo_cache(root, 'cache-1', 0, records[0]['inference_sha256'], owner=owner, **opts)
    registry = read_json(root/'artifact-registry.json')
    assert registry['evaluation_references'] == {}
    assert registry['cache_references']['cache-1'] == pin.binding
    assert (root/'cache-reference-receipts/cache-1/acquire.receipt.json').is_file()
    assert pin == retention.pin_hpo_cache(root, 'cache-1', 0, records[0]['inference_sha256'], owner=owner, **opts)
    with pytest.raises(ContractError):
        retention.release_hpo_cache(root, pin.binding, authority=pin.binding, **opts)


@pytest.mark.parametrize('stage', ['after_intent', 'after_registry'])
def test_cache_acquire_crash_retains_pin_and_exact_binding(tmp_path, monkeypatch, stage):
    import time

    from gf_training import hpo_retention as retention
    from gf_training.hpo_lineage import (
        cache_reference_journal,
        protected_hpo_files,
        validate_hpo_chain,
    )
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import initial_hpo_chain
    root = tmp_path/'resident'
    _, records, _ = initial_hpo_chain(root, resident=True)
    owner = {'job_ticket': 'a'*32, 'pool_id': 'b'*32, 'pool_epoch': 1,
        'worker_key': 1, 'pid': 123, 'viewer': 1, 'lease_epoch': 1}
    opts = {'max_owned_bytes':1024**3, 'deadline':time.monotonic()+30}
    original = retention._write_control
    def crash(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if stage == 'after_intent' and path.name == 'acquire.intent.json' or stage == 'after_registry' and path.name == 'artifact-registry.json':
            raise OSError('injected cache transaction interruption')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(retention, '_write_control', crash)
        with pytest.raises(OSError):
            retention.pin_hpo_cache(root, 'cache-1', 0, records[0]['inference_sha256'], owner=owner, **opts)
    with _ChainScan(directory=root, **opts) as scan:
        state, generations = validate_hpo_chain(root, scan=scan, **opts)
        registry = scan.object(root/'artifact-registry.json')
        pending = cache_reference_journal(scan, state, generations, registry)
        assert len(pending) == 1 and pending[0]['worker_key'] == 1
        assert 'inference.json' in protected_hpo_files(state, generations, registry, pending)[0]
    with pytest.raises(ContractError):
        retention.pin_hpo_cache(root, 'cache-1', 0, records[0]['inference_sha256'], owner={**owner, 'pid':124}, **opts)
    assert retention.pin_hpo_cache(root, 'cache-1', 0, records[0]['inference_sha256'], owner=owner, **opts).phase == 'acquire'


def test_collection7_requires_original_retirement_and_proof4_preserves_it(tmp_path):
    import time

    from gf_common.domain import canonical
    from gf_common.preprocessing import preparation_config
    from gf_models.arrays import atomic_json
    from gf_training.hpo_source import open_hpo_batch, recorded_hpo_plan
    from gf_training.structured_lineage import _ChainScan
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.hpo_support import original_hpo_collection
    root = tmp_path/'parent'
    state, records, learner = initial_hpo_chain(root, resident=True)
    deadline = time.monotonic()+60
    plan, _, _ = recorded_hpo_plan(root, state, records, 0, deadline)
    original_hpo_collection(plan, state, learner, logical_retirement=retirement_fact(state['run_id']))
    from gf_models.sharded_arrays import read_json
    manifest = read_json(plan.output/'manifest.json')
    from gf_training.hpo_collection import check_hpo_collection
    with pytest.raises(ContractError):
        check_hpo_collection({key:value for key,value in manifest.items() if key != 'logical_retirement'})
    retirement = retirement_fact(state['run_id'])
    manifest.update(schema_version=7, logical_retirement=retirement)
    atomic_json(plan.output/'manifest.json', manifest)
    check_hpo_collection(manifest)
    with (
        _ChainScan(directory=root, max_owned_bytes=1024**3, deadline=deadline) as scan,
        open_hpo_batch(plan, state, records, scan, preparation_config(state['effective_config']['preprocessing']),
            online=None, max_owned_bytes=1024**3, deadline=deadline) as source,
    ):
        assert source.proof['schema_version'] == 4
        assert canonical(source.proof['logical_retirement']) == canonical(retirement)
        from gf_training.hpo_lineage import _proof
        assert _proof(source.proof,state,records[0],0,6) == source.selection
        with pytest.raises(ContractError):
            _proof({key:value for key,value in source.proof.items() if key != 'logical_retirement'},state,records[0],0,6)



@pytest.mark.parametrize('entry', ['campaign','compare'])
def test_activated_resident_campaign_reaches_existing_production_validation(tmp_path, monkeypatch, entry):
    import asyncio

    import gf_cli.hpo as cli
    from gf_cli.hpo_config import campaign_config
    loaded = campaign_config({**campaign_raw(),'schema_version':3,'runtime':RUNTIME},origin=tmp_path)
    def forbidden(*args, **kwargs):
        raise RuntimeError('resident activation reached existing production validation')
    monkeypatch.setattr(cli,'validate',forbidden)
    operation = (cli.run_campaign(loaded,tmp_path/'must-not-exist',mode='fresh') if entry == 'campaign'
        else cli.compare_trials(loaded,tmp_path/'legacy-left',tmp_path/'legacy-right',tmp_path/'must-not-exist'))
    with pytest.raises(RuntimeError,match='resident activation reached'):
        asyncio.run(operation)
    assert not (tmp_path/'must-not-exist').exists()


def test_resident_contract_and_custody_import_without_optional_learning_libraries():
    import subprocess
    import sys
    result = subprocess.run([sys.executable,'-c',
        ("import sys; import gf_common.hpo_resident, gf_training.hpo_resident; assert 'numpy' not in sys.modules; "
        "import gf_cli.hpo_config; assert not {'torch','optuna','lightgbm','pyarrow'} & sys.modules.keys()")],capture_output=True,text=True,check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation', ['foreign_run','foreign_generation','unsafe_reset','duplicate_owner','wrong_digest'])
def test_retirement_and_release_facts_refuse_foreign_original_context(mutation):
    from gf_common.domain import digest
    from gf_common.hpo_resident import checked_cache_release, checked_retirement
    retirement = retirement_fact()
    row = retirement['owners'][0]
    if mutation == 'foreign_run':
        binding = {key:row[key] for key in ('job_ticket','pool_id','pool_epoch','worker_key','pid','viewer','lease_epoch')}
        binding.update(run_id='run-a',generation=0,inference_sha256='c'*64,cache_identity='d'*64)
        ack = {**row['park_ack'],'command':'EVICT','run_id':'foreign-run','cache_identity':'d'*64}
        release = {'schema_version':1,'kind':'hpo-resident-cache-eviction','binding_sha256':digest(binding),'ack':ack,'ack_sha256':digest(ack)}
        with pytest.raises(ContractError):
            checked_cache_release(binding,release)
    else:
        if mutation == 'foreign_generation':
            row['park_ack']['generation'] = 1
            row['park_ack_sha256'] = digest(row['park_ack'])
        elif mutation == 'unsafe_reset':
            row['park_ack']['private_state_empty'] = False
            row['park_ack_sha256'] = digest(row['park_ack'])
        elif mutation == 'duplicate_owner':
            retirement['owners'].append(row)
        else:
            row['end_ack_sha256'] = 'f'*64
        with pytest.raises(ContractError):
            checked_retirement(retirement)



def test_missing_resident_native_marker_refuses_before_source_or_session():
    from types import SimpleNamespace

    from gf_common.hpo import NATIVE_CAPABILITY
    from gf_common.preprocessing import PreparationConfig
    from gf_training.native_backend import hpo_reader_settings
    old_native = SimpleNamespace(build_info=lambda:[('card_instance_hpo',NATIVE_CAPABILITY)])
    with pytest.raises(ContractError,match='resident HPO'):
        hpo_reader_settings(PreparationConfig(backend='rust',reader='participant-structured-v2'),old_native,resident=True)


def test_campaign3_optional_mixed_descriptor_is_bound_and_keeps_spec3(tmp_path):
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_common.training_opponents import training_opponents
    from gf_training.hpo_config import effective_settings
    from tests.test_refresh_opponents import DESCRIPTOR
    raw = {**campaign_raw(),'schema_version':3,'runtime':RUNTIME}
    plain = campaign_config(raw,origin=tmp_path)
    loaded = campaign_config({**raw,'training_opponents':DESCRIPTOR},origin=tmp_path)
    assert plain.config.study_identity != loaded.config.study_identity
    settings = effective_settings(loaded.config.model,loaded.config.training,{},layout=layout)
    run = materialize_hpo_run(loaded,settings,learning_seed=17,run_id='00000000-0000-0000-0000-000000000017')
    assert run.config.hpo_spec['schema_version'] == 3
    assert training_opponents(asdict(run.config)) == DESCRIPTOR



def test_cache_and_pending_release_protect_otherwise_unretained_inference(tmp_path):
    from copy import deepcopy

    from gf_training.hpo_lineage import protected_hpo_files
    from tests.hpo_lineage_support import initial_hpo_chain
    state, initial, _ = initial_hpo_chain(tmp_path/'parent',resident=True)
    records = [deepcopy(initial[0]) for _ in range(5)]
    for generation,record in enumerate(records):
        record['generation'] = generation
    registry = {'evaluation_references':{},'cache_references':{}}
    assert protected_hpo_files(state,records,registry)[1] == set()
    inference = {name for name in records[1]['artifact_files'] if name == 'inference.json' or name.startswith('weights-')}
    registry['cache_references']['cache-1'] = {'generation':1}
    assert protected_hpo_files(state,records,registry)[1] == inference
    registry['cache_references'].clear()
    assert protected_hpo_files(state,records,registry,[{'generation':1}])[1] == inference


@pytest.mark.parametrize('entry', ['seed','comparison'])
def test_activated_resident_evaluation_keeps_actual_store_and_gpu_gates(tmp_path, entry):
    import asyncio

    from gf_cli.hpo_config import campaign_config
    from gf_cli.hpo_evaluation import evaluate_comparison, evaluate_seed
    loaded = campaign_config({**campaign_raw(),'schema_version':3,'runtime':RUNTIME},origin=tmp_path)
    if entry == 'seed':
        operation = evaluate_seed(None,loaded,{},None,{},output=tmp_path/'must-not-exist')
    else:
        operation = evaluate_comparison(None,loaded,loaded,None,None,{}, {},comparison_id='comparison',
            conditions_sha256='c'*64,output=tmp_path/'must-not-exist',evaluation_runtime=None)
    with pytest.raises(ContractError,match='actual campaign/proposal|actual comparison store'):
        asyncio.run(operation)
    assert not (tmp_path/'must-not-exist').exists()
