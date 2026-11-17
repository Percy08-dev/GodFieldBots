"""UNADMITTED lifecycle primitives; never genuine formal Source positives."""
import asyncio
import copy
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, RuntimePolicyBinding, canonical, digest
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training import card_instance_final_validation as module
from gf_training.card_instance_collection import CollectionPublication
from gf_training.card_instance_formal import FormalGameParent
from gf_training.lineage import _OWNED_LOCKS, run_lock
from tests.formal_control_support import unadmitted_current_synthetic_formal_shape
from tests.test_formal_ppo_control import loaded


def unadmitted_parent(value, directory, ownership, monkeypatch):
    """Actual temporary lock + mocked parent guard/binding, NO parent admission."""
    owner = FormalGameParent(directory, ownership=ownership, max_owned_bytes=1024**3, deadline=time.monotonic() + 7200)
    owner._entered, owner._active, owner._token = True, True, ownership.token
    owner._state_bytes = canonical({'run_id': 'UNADMITTED-lifecycle-NOT-formal-Source', 'status': 'complete', 'updates': 1000,
        'seed': 17, 'effective_config': value.effective, 'config_sha256': value.sha256, 'config_inputs': list(value.inputs),
        'identity': {'code': {'scope': 'UNADMITTED-NO-runtime-Source'}, 'integration': {'binary_sha256': 'a' * 64}}}).encode()
    owner._records_bytes = canonical([{'generation': 1000, 'scope': 'UNADMITTED'}]).encode()
    def recheck(self):
        self.require_active()
    def binding(self):
        self.require_active()
        return 'UNADMITTED-final-binding-only', RuntimePolicyBinding('mlp', {'expected_manifest_digest': 'a' * 64})
    monkeypatch.setattr(FormalGameParent, 'recheck', recheck)
    monkeypatch.setattr(FormalGameParent, 'final_binding', binding)
    # Pending independently owned parent resource seam; this mock grants no admission.
    monkeypatch.setattr(FormalGameParent, 'resident_bytes', property(lambda self: 1024), raising=False)
    monkeypatch.setattr(FormalGameParent, 'admit_external', lambda self, amount: None if type(amount) is int and amount == 0 else self.require_active(), raising=False)
    return owner


def actor_context(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'formal_preflight', unadmitted_current_synthetic_formal_shape)
    value = loaded(tmp_path)
    directory = tmp_path / 'UNADMITTED-parent'
    ownership_scope = run_lock(directory, create=True)
    ownership = ownership_scope.__enter__()
    parent = unadmitted_parent(value, directory, ownership, monkeypatch)
    ledger = ReservationLedger(value.config.run.global_cuda_budget_bytes, value.config.run.max_participant_workers)
    context = module.GameFinalValidationContext(value, parent, tmp_path / 'validation', ledger, deadline=time.monotonic() + 7200)
    return value, parent, ledger, context, ownership_scope


def test_unadmitted_plan_projection_generation1000_and_no_training_clock_reuse(tmp_path, monkeypatch):
    value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        before = parent.state
        async def run():
            await context.__aenter__()
            assert context.plan.linkage['final_generation'] == 1000
            assert context.plan.linkage['final_inference_sha256'] == 'a' * 64
            assert len(context.plan.groups) == 32
            assert context.projected.config.run.split == 'validation'
            assert context.projected.inputs == value.inputs
            assert context.plan.admission_deadline < context.plan.execution_deadline < context.plan.wall_deadline
            assert parent.state == before
            assert ledger.learner is None and ledger.cuda_bytes == 0 and not ledger.updating
            assert context.artifact_budget > 0
            output = context.plan.output
            await context.aclose()
            assert not output.exists()
            with pytest.raises(ContractError, match='one lifetime'):
                await context.__aenter__()
        asyncio.run(run())
        assert _OWNED_LOCKS[parent.directory] == parent.borrowed.token
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('mutation', ['foreign-plan', 'plan-link', 'plan-registry', 'ledger', 'config', 'inputs', 'effective', 'sha', 'workers', 'learner', 'updating', 'ancestor', 'lock'])
def test_unadmitted_live_guard_mutation_is_sticky_after_repair(tmp_path, monkeypatch, mutation):
    _value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        async def run():
            await context.__aenter__()
            plan, actual_ledger, projected = context.plan, ledger, context.projected
            saved_link, saved_bindings = copy.deepcopy(plan.linkage), copy.deepcopy(plan.bindings)
            saved = ledger.workers, ledger.learner_bytes, ledger.updating
            if mutation == 'foreign-plan': plan = replace(plan)
            elif mutation == 'plan-link': plan.linkage['final_generation'] = 999
            elif mutation == 'plan-registry': plan.bindings.pop('random-card-instance-v2')
            elif mutation == 'ledger': actual_ledger = ReservationLedger()
            elif mutation == 'config': projected = replace(projected, config=replace(projected.config, run=replace(projected.config.run, concurrency=1)))
            elif mutation == 'inputs': projected = replace(projected, inputs=())
            elif mutation == 'effective': projected.effective['run']['concurrency'] = 1
            elif mutation == 'sha': projected = replace(projected, sha256='b' * 64)
            elif mutation == 'workers': ledger.workers = {1: object()}
            elif mutation == 'learner': ledger.learner_bytes = 1
            elif mutation == 'updating': ledger.updating = True
            elif mutation == 'ancestor': context._ancestors[tmp_path] = (-1, -1)
            else: _OWNED_LOCKS[parent.directory] = 'foreign'
            with pytest.raises(ContractError):
                context.validate_collection(plan, actual_ledger, projected, deadline=context.plan.execution_deadline)
            plan.linkage.clear(); plan.linkage.update(saved_link)
            plan.bindings.clear(); plan.bindings.update(saved_bindings)
            ledger.workers, ledger.learner_bytes, ledger.updating = saved
            _OWNED_LOCKS[parent.directory] = parent.borrowed.token
            with pytest.raises(ContractError, match='poisoned'):
                context.validate_collection(context.plan, ledger, context.projected, deadline=context.plan.wall_deadline)
            with pytest.raises(ContractError):
                await context.aclose()
            assert context._closed
        asyncio.run(run())
    finally:
        _OWNED_LOCKS[parent.directory] = parent.borrowed.token
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('mutation', ['foreign-config', 'inputs', 'short-parent', 'no-parent', 'short-caller', 'existing-output', 'nested-output'])
def test_context_refuses_before_output_or_actor_allocations(tmp_path, monkeypatch, mutation):
    value, parent, ledger, _context, scope = actor_context(tmp_path, monkeypatch)
    asyncio.run(_context.aclose())
    try:
        directory, deadline = tmp_path / 'new-output', time.monotonic() + 7200
        if mutation == 'foreign-config':
            cfg = replace(value.config, run=replace(value.config.run, concurrency=1))
            value = replace(value, config=cfg, effective=asdict(cfg), sha256=digest(asdict(cfg)))
        elif mutation == 'inputs': value = replace(value, inputs=())
        elif mutation == 'short-parent':
            cfg = replace(value.config, training=replace(value.config.training, max_updates=2))
            value = replace(value, config=cfg, effective=asdict(cfg), sha256=digest(asdict(cfg)))
        elif mutation == 'no-parent': parent = SimpleNamespace()
        elif mutation == 'short-caller': deadline = time.monotonic() + 60
        elif mutation == 'existing-output': directory.mkdir()
        else: directory = parent.directory / 'validation'
        with pytest.raises(ContractError):
            module.GameFinalValidationContext(value, parent, directory, ledger, deadline=deadline)
        assert ledger.serial == 0 and ledger.learner is None and not ledger.workers
    finally:
        scope.__exit__(None, None, None)


def test_accept_and_readonly_open_share_original_validator_without_qualification(tmp_path, monkeypatch):
    value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    calls = []
    # Only the orchestration seam is controlled here; NO C5/J6/Source admission.
    def open_original(*args, **kwargs):
        calls.append(kwargs)
        snapshot = SimpleNamespace(recheck=lambda: None, reserve=lambda amount: None)
        return module.GameFinalValidationSource(parent, snapshot, (), {'scope': 'UNADMITTED-not-a-Source', 'qualified': False})
    monkeypatch.setattr(module, '_open_original', open_original)
    try:
        async def run():
            await context.__aenter__()
            context.plan.output.mkdir()
            publication = CollectionPublication(context.plan.output, work_deadline=context.plan.wall_deadline - 1, cleanup_deadline=context.plan.wall_deadline)
            result = context.begin_collection(publication, {'healthy': True, 'wall_seconds': 0.},
                {'schema_version': 5, 'collection_role': 'ppo-final-validation', 'validation_invocation': context.plan.validation_invocation})
            assert context.accept_collection(result).facts['qualified'] is False
            assert calls[0]['expected_plan'] is context.plan and calls[0]['returned'] == result.report
            output = context.plan.output
            with pytest.raises(ContractError, match='resource borrow'), module.open_game_final_validation(value, parent, output,
                    deadline=time.monotonic() + 60, max_owned_bytes=1024**3):
                pytest.fail('overlapping readonly borrow accepted')
            with pytest.raises(ContractError):
                context.accept_collection(result)
            with pytest.raises(ContractError):
                await context.aclose()
            with module.open_game_final_validation(value, parent, output, deadline=time.monotonic() + 60, max_owned_bytes=1024**3) as original:
                assert original.facts['qualified'] is False
            opens = [call for call in calls if isinstance(call, dict)]
            assert 'expected_plan' not in opens[1] and 'returned' not in opens[1]
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)


def test_source_original_failure_stays_poisoned_after_repair(tmp_path, monkeypatch):
    _value, parent, _ledger, _context, scope = actor_context(tmp_path, monkeypatch)
    fail = [True]
    class UnadmittedSnapshot:
        def recheck(self):
            if fail[0]: raise ContractError('UNADMITTED original mutation')
    source = module.GameFinalValidationSource(parent, UnadmittedSnapshot(), (), {'qualified': False})
    try:
        with pytest.raises(ContractError, match='mutation'):
            source.recheck()
        fail[0] = False
        with pytest.raises(ContractError, match='poisoned'):
            source.recheck()
        with pytest.raises(ContractError, match='mutation'):
            source.close()
        assert source._closed
    finally:
        scope.__exit__(None, None, None)



@pytest.mark.parametrize('duration', [0, 60, float('nan')])
def test_context_caller_reserve_rejects_before_borrow_and_parent_guard(tmp_path, monkeypatch, duration):
    value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    asyncio.run(context.aclose())
    monkeypatch.setattr(parent, 'recheck', lambda: pytest.fail('parent guard before caller reserve rejection'))
    try:
        with pytest.raises(ContractError):
            module.GameFinalValidationContext(value, parent, tmp_path / 'new-output', ledger,
                deadline=time.monotonic() + duration)
        assert parent not in module._BORROWS and not (tmp_path / 'new-output').exists()
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('entrance', ['constructor', 'readonly', 'live-common'])
def test_near_cap_parent_refuses_before_control_clones_plan_or_serializers(tmp_path, monkeypatch, entrance):
    value, parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    base = 0
    if entrance == 'live-common':
        asyncio.run(context.__aenter__())
        base = context._reserved_bytes
    else:
        asyncio.run(context.aclose())
    admitted = []
    intrinsic = parent.limit - base - 65536 - 1
    def admit(amount):
        admitted.append(amount)
        if amount and intrinsic + amount > parent.limit:
            raise ContractError('logical near-cap parent refuses complete controls')
    def forbidden(*args, **kwargs):
        pytest.fail('control materializer/serializer ran before complete parent admission')
    monkeypatch.setattr(parent, 'admit_external', admit)
    monkeypatch.setattr(module, '_loaded', forbidden)
    for name in ('_facts', '_project', '_plan', '_signature', 'canonical', 'asdict'):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(module, 'json', SimpleNamespace(loads=forbidden))
    try:
        with pytest.raises(ContractError, match='near-cap'):
            if entrance == 'constructor':
                module.GameFinalValidationContext(value, parent, tmp_path / 'new-output', ledger, deadline=time.monotonic() + 7200)
            elif entrance == 'readonly':
                with module.open_game_final_validation(value, parent, tmp_path / 'absent-original',
                        deadline=time.monotonic() + 60, max_owned_bytes=parent.limit):
                    pytest.fail('near-cap readonly admission succeeded')
            else:
                module._open_original(value, parent, tmp_path / 'absent-live-original', deadline=time.monotonic() + 60,
                    max_owned_bytes=parent.limit, extra_owned_bytes=base)
        assert admitted[0] == base + 65536
        assert admitted[1] > admitted[0]
        assert not (tmp_path / 'new-output').exists()
    finally:
        if entrance == 'live-common':
            asyncio.run(context.aclose())
        scope.__exit__(None, None, None)


def test_protected_parent_size_accessor_never_parses_or_materializes(tmp_path, monkeypatch):
    import gf_training.card_instance_formal as formal
    _value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    expected = len(parent._state_bytes), len(parent._records_bytes)
    def forbidden(*args, **kwargs): pytest.fail('size accessor parsed protected originals')
    monkeypatch.setattr(formal, 'json', SimpleNamespace(loads=forbidden))
    try:
        assert parent.canonical_control_sizes == expected
        with pytest.raises(AttributeError): parent.canonical_control_sizes = (0, 0)
        asyncio.run(context.aclose())
        parent._active = False
        with pytest.raises(ContractError): _ = parent.canonical_control_sizes
    finally:
        scope.__exit__(None, None, None)


def test_control_size_bound_counts_json_escaping_and_shared_occurrences_without_serializing(tmp_path, monkeypatch):
    value, parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        shared = {'escaped': '\\n\\\\\"界' * 50}
        values = [shared, shared, {'integer': 2**256, 'finite_float': -1.7976931348623157e308}]
        actual = len(canonical(values).encode())
        assert module._wire_upper(values) >= actual
        assert module._wire_upper([shared, shared]) > module._wire_upper([shared])
        asyncio.run(context.__aenter__())
        assert context._reserved_bytes >= context._control_bytes()
        plan_json = {**asdict(context.plan), 'output': str(context.plan.output)}
        assert module._wire_upper(plan_json) >= len(module._signature(context.plan).encode())
        assert context._reserved_bytes >= module._control_bound(value, parent, context.plan.output)
        asyncio.run(context.aclose())
    finally:
        scope.__exit__(None, None, None)


def test_later_projected_clone_is_admitted_before_json_decoder(tmp_path, monkeypatch):
    _value, _parent, _ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        asyncio.run(context.__aenter__())
        calls = []
        admit = context._parent.admit_external
        def checked(amount):
            calls.append(('admit', amount))
            return admit(amount)
        loads = module.json.loads
        def decode(raw):
            assert calls and calls[-1] == ('admit', context._reserved_bytes)
            calls.append(('decode', len(raw)))
            return loads(raw)
        monkeypatch.setattr(context._parent, 'admit_external', checked)
        monkeypatch.setattr(module, 'json', SimpleNamespace(loads=decode))
        assert context.projected.config.run.split == 'validation'
        asyncio.run(context.aclose())
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize('entrance', ['plan-validate', 'plan-projected', 'original-loaded', 'foreign-loaded'])
def test_after_entry_control_growth_refuses_before_any_serializer_or_parent_copy(tmp_path, monkeypatch, entrance):
    value, _parent, ledger, context, scope = actor_context(tmp_path, monkeypatch)
    try:
        async def run():
            await context.__aenter__()
            projected = context.projected  # Obtain the admitted original BEFORE corruption/spies.
            supplied = projected
            reserved = context._reserved_bytes
            payload = 'x' * (34 * 1024**2)
            if entrance.startswith('plan-'):
                context.plan.linkage['UNADMITTED-after-entry-growth'] = payload
            elif entrance == 'original-loaded':
                value.effective['UNADMITTED-after-entry-growth'] = payload
            else:
                supplied = replace(projected, effective={**projected.effective, 'UNADMITTED-after-entry-growth': payload})
            def forbidden(*args, **kwargs):
                pytest.fail('growth reached serializer/decoder/parent copy before immutable-size refusal')
            for name in ('_signature', 'asdict', 'canonical', '_facts'):
                monkeypatch.setattr(module, name, forbidden)
            monkeypatch.setattr(module, 'json', SimpleNamespace(loads=forbidden))
            with pytest.raises(ContractError):
                if entrance in ('plan-projected', 'original-loaded'):
                    _ = context.projected
                else:
                    context.validate_collection(context.plan, ledger, supplied, deadline=context.plan.wall_deadline)
            assert context._reserved_bytes == reserved  # No increased claim or authority for mutable growth.
            with pytest.raises(ContractError, match='poisoned'):
                context.validate_collection(context.plan, ledger, supplied, deadline=context.plan.wall_deadline)
            with pytest.raises(ContractError): await context.aclose()
            assert context.cleanup_complete
        asyncio.run(run())
    finally:
        scope.__exit__(None, None, None)
