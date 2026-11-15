"""Retained complete originals; these CPU component fixtures grant no run authority."""
import copy
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError


@contextmanager
def held_parent(tmp_path, monkeypatch, *, cap=1024**3, packed=False, mixed=False):
    from gf_models.structured_binding import configured_model_v4
    from gf_training import hpo_gpu_lease as gpu
    from gf_training.card_instance_source import _OnlineGameOwner
    from gf_training.hpo_precollection import _PrecollectionValidation
    from gf_training.hpo_resident_runtime import ResidentTrainingRuntime
    from gf_training.hpo_source import recorded_hpo_plan
    from gf_training.lineage import run_lock
    from tests.hpo_lineage_support import initial_hpo_chain
    from tests.test_card_instance_learning import ACTIONS, LAYOUT
    monkeypatch.setattr(gpu, '_GLOBAL_ROOT', tmp_path/'admission')
    monkeypatch.setattr('gf_training.hpo_source.require_hpo_production', lambda state: None)
    root=tmp_path/'parent'
    if mixed:
        from dataclasses import asdict, replace

        from gf_common.domain import digest
        from gf_training.hpo_config import effective_settings
        from tests import hpo_lineage_support
        from tests.test_refresh_opponents import DESCRIPTOR
        original_fixture=hpo_lineage_support.hpo_state_fixture
        def mixed_fixture(*args,**kwargs):
            state,model,recipe,binding=original_fixture(*args,**kwargs)
            recipe=replace(recipe,collection_games=16)
            spec=state['effective_config']['hpo_spec']
            spec.update(training_opponents=copy.deepcopy(DESCRIPTOR),training_recipe=asdict(recipe),
                effective_settings_sha256=effective_settings(model,recipe,{},layout=LAYOUT).identity)
            state['hpo_spec_sha256']=digest(spec)
            state['config_sha256']=digest(state['effective_config'])
            state['resume_identity']=digest({'version':'hpo-ppo-resume-v1','config_sha256':state['config_sha256'],
                'identity':state['identity'],'execution':state['execution'],
                'classification':{key:state[key] for key in ('action_contract','purpose','synthetic_capacity','rule_signature','layout_signature')},
                'storage_binding':state.get('storage_binding')})
            return state,model,recipe,binding
        monkeypatch.setattr(hpo_lineage_support,'hpo_state_fixture',mixed_fixture)
    state, records, _ = initial_hpo_chain(root, resident=True, packed=packed)
    config=SimpleNamespace(model=configured_model_v4(state['effective_config']['model'], layout=LAYOUT),
        actions=ACTIONS, hpo_spec=state['effective_config']['hpo_spec'])
    deadline=time.monotonic()+30
    with run_lock(root) as ownership, gpu.admit_gpu_job(gpu.JobContext('component','component','training',state['run_id'],None),
            global_cuda_bytes=config.hpo_spec['operating']['managed_cuda_bytes'], max_workers=4) as job:
        runtime=ResidentTrainingRuntime(job, ownership, config, state, deadline=deadline, cleanup_deadline=deadline+1)
        online=_OnlineGameOwner(ownership, state, SimpleNamespace(), job.ledger, hpo_job=job, resident_runtime=runtime)
        online.process.online_owner=online
        plan, _, _=recorded_hpo_plan(root, state, records, 0, deadline)
        # Registration precedes the coordinator's normal admission construction.
        carrier=_PrecollectionValidation.attach(online, plan, job.ledger, max_owned_bytes=cap,
            deadline=deadline, cleanup_deadline=deadline+1)
        try:
            runtime.pool.admit(plan)
            yield online, plan, carrier
        finally:
            carrier.close()
            job.confirm_termination()


def read(online, plan, carrier):
    return carrier.history(online, plan, online.ledger, resident_phase=online.resident_runtime.pool.admission,
        max_owned_bytes=carrier.cap, deadline=plan.execution_deadline)


def test_three_reads_retain_complete_validation_and_expire_before_output(tmp_path, monkeypatch):
    from gf_training import hpo_source
    original=hpo_source.validate_hpo_parent
    calls=[]
    def observed(*args, **kwargs):
        calls.append(kwargs['_scan'])
        return original(*args, **kwargs)
    monkeypatch.setattr(hpo_source, 'validate_hpo_parent', observed)
    with held_parent(tmp_path, monkeypatch) as (online, plan, carrier):
        assert carrier.phase=='new' and carrier.admission is None
        first=read(online, plan, carrier)
        assert read(online, plan, carrier) is first
        assert read(online, plan, carrier) is first
        assert calls==[carrier.scan]
        assert carrier.admission is online.resident_runtime.pool.admission
        assert not plan.output.exists()
        carrier.finish(online, plan, online.ledger)
        assert carrier.phase=='closed' and carrier.scan._closed
        assert online.precollection_validation is None
        with pytest.raises(ContractError): read(online, plan, carrier)


@pytest.mark.parametrize('mutation', ['run', 'run_inode', 'cache_reference_namespace', 'evaluation_reference_namespace', 'inventory', 'facts', 'copied', 'plan', 'admission', 'deadline', 'output', 'thread', 'pid', 'job', 'ownership', 'generation'])
def test_retained_reads_reject_mutation_and_never_rebuild(tmp_path, monkeypatch, mutation):
    with held_parent(tmp_path, monkeypatch) as (online, plan, carrier):
        _state, records=read(online, plan, carrier)
        def forbidden(*args, **kwargs): raise AssertionError('invalid reuse fell back to full read')
        monkeypatch.setattr('gf_training.hpo_source.validate_hpo_parent', forbidden)
        if mutation=='run':
            (online.ownership.directory/'run.json').write_text('{}')
        elif mutation=='run_inode':
            from gf_models.arrays import sha256
            path=online.ownership.directory/'run.json'
            before=path.stat()
            expected=sha256(path)
            replacement=path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            assert sha256(path)==expected
            assert (path.stat().st_dev,path.stat().st_ino)!=(before.st_dev,before.st_ino)
        elif mutation in ('cache_reference_namespace','evaluation_reference_namespace'):
            cache=mutation=='cache_reference_namespace'
            snapshot=getattr(carrier.scan,'_hpo_cache_reference_snapshot' if cache else '_hpo_reference_snapshot')
            assert snapshot is not None
            directory=online.ownership.directory/('cache-reference-receipts' if cache else 'evaluation-reference-receipts')
            if snapshot[0] is None:
                assert not directory.exists()
                directory.mkdir()
            else:
                (directory/'new-reference').mkdir()
        elif mutation=='inventory':
            (online.ownership.directory/'generations/foreign.json').write_text('{}')
        elif mutation=='facts': records[0]['generation']=1
        elif mutation=='copied': carrier=copy.copy(carrier)
        elif mutation=='plan': plan=copy.copy(plan)
        elif mutation=='admission':
            online.resident_runtime.pool.admission.phase='retired'
            online.resident_runtime.pool.admit(copy.copy(plan))
        elif mutation=='deadline': carrier.scan.deadline=time.monotonic()-1
        elif mutation=='thread': carrier.thread+=1
        elif mutation=='pid': carrier.pid+=1
        elif mutation=='job': online.hpo_job=object()
        elif mutation=='ownership': online.ownership=copy.copy(online.ownership)
        elif mutation=='generation': plan.linkage['collection_generation']+=1
        else: plan.output.mkdir(parents=True)
        with pytest.raises((ContractError, TimeoutError)): read(online, plan, carrier)


def test_failed_build_stays_attached_until_actual_cleanup(tmp_path, monkeypatch):
    with held_parent(tmp_path, monkeypatch) as (online, plan, carrier):
        def broken(*args, **kwargs): raise OSError('historical failure')
        monkeypatch.setattr('gf_training.hpo_source.validate_hpo_parent', broken)
        with pytest.raises(OSError, match='historical failure'): read(online, plan, carrier)
        assert online.precollection_validation is carrier and carrier.phase=='failed'
        with pytest.raises(ContractError): read(online, plan, carrier)
        carrier.close()
        assert online.precollection_validation is None


def test_unconfirmed_close_retains_actual_owner_and_prevents_finish(tmp_path, monkeypatch):
    with held_parent(tmp_path, monkeypatch) as (online, plan, carrier):
        read(online, plan, carrier)
        original=carrier.scan.close
        def broken(**kwargs): raise OSError('unconfirmed scan close')
        with monkeypatch.context() as patch:
            patch.setattr(carrier.scan, 'close', broken)
            with pytest.raises(OSError, match='unconfirmed scan close') as failure:
                carrier.finish(online, plan, online.ledger)
            assert failure.value.cleanup_owner is carrier
            assert online.precollection_validation is carrier and carrier.phase=='failed'
        original(recheck=False)


def test_private_opt_in_defaults_false_and_finish_keyword_is_available():
    import inspect

    from gf_cli.card_instance import _validate_online_collection
    from gf_training.card_instance_source import validate_owned_collection
    from gf_training.hpo_ppo import train_hpo_ppo
    assert inspect.signature(train_hpo_ppo).parameters['_reuse_precollection'].default is False
    assert inspect.signature(validate_owned_collection).parameters['_finish_precollection'].default is False
    assert inspect.signature(_validate_online_collection).parameters['_finish_precollection'].default is False


def test_actual_native_g1_keeps_outer_originals_after_executor_retirement(tmp_path, monkeypatch):
    from gf_training import hpo_source
    from gf_training.native_backend import load_native
    from tests.test_resident_training import run_actual_source_case
    native=load_native()
    monkeypatch.setattr('gf_training.hpo_gpu_lease._GLOBAL_ROOT',tmp_path/'admission')
    full=[]
    original=hpo_source.validate_hpo_parent
    def observed(*args, **kwargs):
        full.append(kwargs.get('_scan'))
        return original(*args, **kwargs)
    monkeypatch.setattr(hpo_source,'validate_hpo_parent',observed)
    counts={}
    def probe(online,plan,carrier,generation,check):
        if check==0: counts[generation]=len(full)
        if check<3:
            assert len(full)==counts[generation]
        else:
            assert len(full)==counts[generation]+1
        assert not plan.output.exists()
        if check<2:
            assert online.precollection_validation is carrier
            assert carrier.owned_bytes<=carrier.cap
            if check==1:
                from gf_training.card_instance_source import validate_owned_collection
                for target,attribute,bad in [(online.ledger,'updating',True),
                        (online.process,'steps',online.process.steps+1), (online.process,'identity',{})]:
                    with monkeypatch.context() as patch:
                        patch.setattr(target,attribute,bad)
                        with pytest.raises(ContractError):
                            validate_owned_collection(plan,online.ledger,max_owned_bytes=carrier.cap,
                                deadline=plan.execution_deadline,resident_phase=carrier.admission)
            if generation:
                scan=carrier.scan
                assert any('numeric/batch-000000' in str(path) for path in scan.files)
                assert scan._packed_owners and scan._packed_pool._close_holds==[]
                assert carrier.owned_bytes==scan._packed_pool.budget.used+scan.retained-scan._pool_control_charge
        else:
            assert online.precollection_validation is None
            assert carrier.phase=='closed' and carrier.scan._closed
            if generation:
                assert carrier.scan._packed_pool.closed and carrier.scan._packed_pool.budget.used==0
    run_actual_source_case(None,tmp_path,monkeypatch,packed=True,native_backend=native,
        precollection_probe=probe)
    assert len(full)==6  # Per generation: retained first-three, then full fourth/fifth.
    assert sum(scan is not None for scan in full)==2


def operations_for(online):
    from gf_training.hpo_ppo import _HpoOperations
    operations=_HpoOperations(online.hpo_job,_reuse_precollection=True)
    operations.online, operations.resident_runtime=online,online.resident_runtime
    config=SimpleNamespace(training=SimpleNamespace(max_owned_bytes=1024**3),
        preprocessing=SimpleNamespace(memory_budget_mib=1024), hpo_spec=online.resident_runtime.config.hpo_spec)
    return operations,config


@pytest.mark.parametrize('failure', ['collector_pretry', 'cancel', 'first_read', 'registration'])
def test_outer_scope_retires_partial_originals_without_requiring_live_learner(tmp_path, monkeypatch, failure):
    import asyncio

    from gf_training.hpo_precollection import _PrecollectionValidation
    with held_parent(tmp_path,monkeypatch) as (online,plan,initial):
        initial.close()
        operations,config=operations_for(online)
        primary=asyncio.CancelledError('cancelled collector') if failure=='cancel' else OSError('injected collection error')
        actual=[]
        original=_PrecollectionValidation.attach.__func__
        def attach(cls,*args,**kwargs):
            value=original(cls,*args,**kwargs)
            actual.append(value)
            return value
        monkeypatch.setattr(_PrecollectionValidation,'attach',classmethod(attach))
        if failure=='registration': config.training.max_owned_bytes=1
        if failure=='first_read':
            def broken(*args,**kwargs):
                with kwargs['_scan'].file(online.ownership.directory/'run.json'): pass
                raise primary
            monkeypatch.setattr('gf_training.hpo_source.validate_hpo_parent',broken)
        with (pytest.raises((OSError,ContractError,asyncio.CancelledError)) as caught,
                operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
            current=online.precollection_validation
            if failure=='first_read': read(online,plan,current)
            elif failure!='registration':
                read(online,plan,current)
                online.process.online_owner=None  # Learner failure cannot block terminal close.
                raise primary
        if failure!='registration': assert caught.value is primary
        assert online.precollection_validation is None
        assert all(value.phase=='closed' and value.scan._closed for value in actual)


def test_scope_refuses_missing_finish_and_final_termination_refuses_unresolved_owner(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch) as (online,plan,initial):
        operations,config=operations_for(online)
        with pytest.raises(ContractError,match='precollection original custody'):
            operations.finish_cleanup(None)
        initial.close()
        with (pytest.raises(ContractError,match='did not expire'),
                operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
            read(online,plan,online.precollection_validation)
        assert online.precollection_validation is None


def test_metadata_is_admitted_before_fingerprint_allocation(tmp_path,monkeypatch):
    from gf_training.hpo_precollection import _PrecollectionValidation
    with held_parent(tmp_path,monkeypatch) as (online,plan,initial):
        initial.close()
        def forbidden(*args): raise AssertionError('fingerprint allocated before reservation')
        monkeypatch.setattr('gf_training.hpo_precollection.digest',forbidden)
        with pytest.raises(ContractError,match='reservation'):
            _PrecollectionValidation.attach(online,plan,online.ledger,max_owned_bytes=16384,
                deadline=plan.execution_deadline,cleanup_deadline=plan.wall_deadline)
        actual=online.precollection_validation
        assert actual is not None and actual.phase=='failed'
        actual.close()
        assert online.precollection_validation is None


@pytest.mark.parametrize('mutation',['numeric','packed_locator','native_identity'])
def test_native_g1_mutation_rejects_after_sessions_retire_and_then_closes(tmp_path,monkeypatch,mutation):
    from gf_training.native_backend import load_native
    from tests.test_resident_training import run_actual_source_case
    native=load_native()
    monkeypatch.setattr('gf_training.hpo_gpu_lease._GLOBAL_ROOT',tmp_path/'admission')
    retained=[]
    class ObservedMutation(Exception): pass
    def probe(online,plan,carrier,generation,check):
        if generation!=1 or check!=0: return
        retained.append(carrier)
        assert carrier.scan._packed_pool._close_holds==[]
        if mutation=='numeric':
            path=next(path for path in carrier.scan.files if 'numeric/batch-000000' in str(path) and path.name!='rollout.json')
            raw=bytearray(path.read_bytes())
            raw[-1]^=1
            path.write_bytes(raw)
        elif mutation=='packed_locator':
            original=next(iter(carrier.scan._packed_owners.values()))
            (original.root/'collection.locator.json').write_text('{}')
        else:
            from gf_training import native_backend
            identity=native_backend.binary_identity(native)
            assert carrier.scan.files[Path(identity['extension_path'])][1]==identity['extension_sha256']
            monkeypatch.setattr(native_backend,'binary_identity',lambda *args:{**identity,'extension_sha256':'f'*64})
        with pytest.raises((ContractError,ValueError)): read(online,plan,carrier)
        assert carrier.phase=='failed' and online.precollection_validation is carrier
        raise ObservedMutation
    with pytest.raises((ObservedMutation,ContractError,ValueError)):
        run_actual_source_case(None,tmp_path,monkeypatch,packed=True,native_backend=native,precollection_probe=probe)
    assert retained and all(value.phase=='closed' and value.scan._packed_pool.closed
        and value.scan._packed_pool.budget.used==0 for value in retained)


def test_close_attempts_actual_retirement_even_after_work_deadline_and_failed_learner(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch) as (online,plan,carrier):
        read(online,plan,carrier)
        carrier.scan.deadline=time.monotonic()-1
        online.process.online_owner=None
        carrier.close()
        assert carrier.scan._closed and online.precollection_validation is None


@pytest.mark.parametrize('enabled,resident',[(False,True),(False,False),(True,False)])
def test_default_custom_and_fresh_scopes_preserve_one_shot_path(tmp_path,monkeypatch,enabled,resident):
    from gf_training.hpo_precollection import _PrecollectionValidation
    with held_parent(tmp_path,monkeypatch) as (online,plan,initial):
        initial.close()
        operations,config=operations_for(online)
        operations._reuse_precollection=enabled
        if not resident: operations.resident_runtime=None
        def forbidden(*args,**kwargs): raise AssertionError('compatibility path allocated retained evidence')
        monkeypatch.setattr(_PrecollectionValidation,'attach',forbidden)
        with operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline):
            assert online.precollection_validation is None


@pytest.mark.parametrize('mutation',[None,'inputs'])
def test_cli_finish_checks_loaded_caller_before_requesting_actual_close(monkeypatch,mutation):
    from copy import deepcopy
    from dataclasses import replace

    from gf_cli import card_instance as cli
    from gf_cli.config import load_config
    from gf_training.card_instance_source import _OnlineGameOwner
    loaded=load_config(Path(__file__).parents[1]/'configs/experiments/card-instance-mlp-ppo-acceptance.toml')
    online=_OnlineGameOwner(None,{'config_sha256':loaded.sha256,'effective_config':deepcopy(loaded.effective),
        'config_inputs':list(loaded.inputs)},None,None)
    ledger=SimpleNamespace(learner=SimpleNamespace(_cleanup_owner=SimpleNamespace(online_owner=online)))
    calls=[]
    def validator(*args,**kwargs):
        calls.append(kwargs)
        return online
    monkeypatch.setattr('gf_training.card_instance_source.validate_owned_collection',validator)
    if mutation:
        with pytest.raises(ContractError,match='Config/input'):
            cli._validate_online_collection(replace(loaded,inputs=()),object(),ledger,time.monotonic()+10,
                _finish_precollection=True)
        assert calls==[]
    else:
        assert cli._validate_online_collection(loaded,object(),ledger,time.monotonic()+10,_finish_precollection=True) is online
        assert len(calls)==1 and calls[0]['_finish_precollection'] is True


@pytest.mark.parametrize('after_effect',[False,True])
def test_packed_refund_failure_stays_reachable_and_blocks_termination(tmp_path,monkeypatch,after_effect):
    with held_parent(tmp_path,monkeypatch,packed=True) as (online,plan,carrier):
        read(online,plan,carrier)
        pool=carrier.scan.ensure_packed_pool()
        used=pool.budget.used
        assert used>0 and carrier.owned_bytes==used
        with monkeypatch.context() as patch:
            original_release=pool.budget.release
            def broken(amount):
                if after_effect: original_release(amount)
                raise OSError('refund unconfirmed')
            patch.setattr(pool.budget,'release',broken)
            with pytest.raises(OSError,match='refund unconfirmed') as caught:
                carrier.close()
            assert caught.value.cleanup_owner is carrier
            assert online.precollection_validation is carrier
            assert pool.budget.used<used if after_effect else pool.budget.used==used
            operations,_=operations_for(online)
            with pytest.raises(ContractError,match='precollection original custody'):
                operations.finish_cleanup(caught.value)
        carrier.close()
        assert pool.closed and pool.budget.used==0 and online.precollection_validation is None


def test_nested_scope_cannot_replace_or_close_an_existing_carrier(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch) as (online,plan,original):
        operations,config=operations_for(online)
        with (pytest.raises(ContractError,match='already active'),
                operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
            raise AssertionError('nested scope entered')
        assert online.precollection_validation is original and original.phase=='new'
        original.close()
        with operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline):
            actual=online.precollection_validation
            read(online,plan,actual)
            actual.finish(online,plan,online.ledger)
            with (pytest.raises(ContractError,match='already active'),
                    operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
                raise AssertionError('expired scope restarted')
            assert online.precollection_validation is None and actual.phase=='closed'


def test_scope_preserves_primary_error_and_actual_owner_when_cleanup_also_fails(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch) as (online,plan,initial):
        initial.close()
        operations,config=operations_for(online)
        primary=OSError('primary collection failure')
        with monkeypatch.context() as patch:
            with (pytest.raises(OSError,match='primary collection failure') as caught,
                    operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
                carrier=online.precollection_validation
                read(online,plan,carrier)
                def broken(**kwargs): raise RuntimeError('actual scan close unconfirmed')
                patch.setattr(carrier.scan,'close',broken)
                raise primary
            assert caught.value is primary and caught.value.cleanup_owner is carrier
            assert any('actual scan close unconfirmed' in note for note in primary.__notes__)
            assert online.precollection_validation is carrier
        operations.close_precollection()
        assert online.precollection_validation is None


def test_zero_pool_account_is_required_before_detach(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch,packed=True) as (online,plan,carrier):
        read(online,plan,carrier)
        pool=carrier.scan.ensure_packed_pool()
        pool.budget.acquire(1024,plan.execution_deadline)
        with pytest.raises(ContractError,match='account retirement'):
            carrier.close()
        assert pool.closed and pool.budget.used==1024
        assert online.precollection_validation is carrier and carrier.facts is not None
        # Retire only this test's known actual outstanding reservation.
        pool.budget.release(1024)
        carrier.close()
        assert online.precollection_validation is None and carrier.facts is None


@pytest.mark.parametrize('after_effect',[False,True])
def test_recheck_refund_error_preserves_primary_and_never_detaches_a_live_account(tmp_path,monkeypatch,after_effect):
    with held_parent(tmp_path,monkeypatch,packed=True) as (online,plan,initial):
        initial.close()
        operations,config=operations_for(online)
        primary=OSError('original recheck refund failure')
        with monkeypatch.context() as patch:
            with (pytest.raises(OSError,match='original recheck refund failure') as caught,
                    operations.precollection_scope(plan,online.ledger,config=config,deadline=plan.execution_deadline)):
                carrier=online.precollection_validation
                read(online,plan,carrier)
                pool=carrier.scan.ensure_packed_pool()
                before=pool.budget.used
                release=pool.budget.release
                attempts=[]
                def broken(amount):
                    attempts.append(amount)
                    if after_effect: release(amount)
                    raise primary
                patch.setattr(pool.budget,'release',broken)
                read(online,plan,carrier)
            assert caught.value is primary and caught.value.cleanup_owner is carrier
            assert online.precollection_validation is carrier and carrier.phase=='failed'
        if not after_effect:
            # The original shared scan's first temporary refund failed before
            # its effect. Actual retirement can close originals but MUST still
            # refuse detach while that precise account reservation survives.
            assert pool.budget.used==before+attempts[0]
            with pytest.raises(ContractError,match='account retirement'): carrier.close()
            assert pool.closed and pool.budget.used==attempts[0]
            pool.budget.release(attempts[0])  # Actual test-observed acquisition.
        carrier.close()
        assert pool.closed and pool.budget.used==0 and online.precollection_validation is None


@pytest.mark.parametrize('boundary',['second_read','third_read','finish'])
def test_python_mixed_reuse_rechecks_current_native_opponent_capability(tmp_path,monkeypatch,boundary):
    from gf_common.training_opponents import training_opponents
    from gf_training.hpo_source import validate_hpo_parent
    with held_parent(tmp_path,monkeypatch,mixed=True) as (online,plan,carrier):
        assert online.state['effective_config']['preprocessing']['backend']=='python'
        assert training_opponents(online.state['effective_config']) is not None
        read(online,plan,carrier)  # Complete originals and current capability initially pass.
        if boundary in ('third_read','finish'): read(online,plan,carrier)
        if boundary=='finish': read(online,plan,carrier)
        def missing(*args): raise ContractError('mixed opponent capability no longer available')
        monkeypatch.setattr('gf_training.native_backend.require_native_refresh_opponents',missing)
        with pytest.raises(ContractError,match='mixed opponent capability'):
            validate_hpo_parent(online.ownership.directory,require_complete=False,
                max_owned_bytes=carrier.cap,deadline=plan.execution_deadline)
        with pytest.raises(ContractError,match='mixed opponent capability'):
            if boundary=='finish': carrier.finish(online,plan,online.ledger)
            else: read(online,plan,carrier)
        assert carrier.phase=='failed' and online.precollection_validation is carrier


def test_plain_python_reuse_does_not_require_optional_native_capabilities(tmp_path,monkeypatch):
    with held_parent(tmp_path,monkeypatch) as (online,plan,carrier):
        assert online.state['effective_config']['preprocessing']['backend']=='python'
        def forbidden(*args): raise AssertionError('plain Python reuse required optional native support')
        monkeypatch.setattr('gf_training.native_backend.load_native',forbidden)
        monkeypatch.setattr('gf_training.native_backend.require_native_refresh_opponents',forbidden)
        first=read(online,plan,carrier)
        assert read(online,plan,carrier) is first
        assert read(online,plan,carrier) is first
        carrier.finish(online,plan,online.ledger)
        assert carrier.phase=='closed' and online.precollection_validation is None
