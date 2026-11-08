"""UNADMITTED forwarding/resource primitives; no formal parent/Source positives."""
from __future__ import annotations

import fcntl
import json
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, digest
from gf_common.preprocessing import PreparationConfig
from gf_training.card_instance_final_validation import (
    GameFinalValidationContext,
    GameFinalValidationSource,
    _acquire_borrow,
    _release_borrow,
    _require_borrow,
)
from gf_training.card_instance_formal import FormalGameParent
from gf_training.lineage import run_lock
from gf_training.native_card_instance import _NativeFinalValidationInputs, _NativeGameInputs
from tests.test_native_card_instance_final_validation import originals as reference_originals
from tests.test_native_card_instance_final_validation import packet_from_reference


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    return reference_originals.__wrapped__(tmp_path_factory)


class DirectUnadmittedExecutor:
    def call(self, function, *args):
        return function(*args)


def unadmitted_loan(tmp_path, monkeypatch, *, actions=None):
    directory = tmp_path/'UNADMITTED-parent-NOT-Run3'
    scope = run_lock(directory, create=True)
    ownership = scope.__enter__()
    parent = FormalGameParent(directory, ownership=ownership, max_owned_bytes=1024**3,
        deadline=time.monotonic()+60)
    parent._entered = parent._active = True
    parent._token = ownership.token
    parent.recheck = parent.require_active
    reservations = []
    def admit(self, amount):
        assert type(amount) is int and amount >= 0
        if amount:
            self.require_active()
            if amount+1024 > self.limit:
                raise ContractError('UNADMITTED logical joint reservation exhausted')
        self._external_bytes = amount
        reservations.append(amount)
    monkeypatch.setattr(FormalGameParent, 'admit_external', admit)
    monkeypatch.setattr(FormalGameParent, 'resident_bytes', property(lambda self: 1024))
    owner = object.__new__(_NativeGameInputs)
    owner.deadline = parent.deadline
    owner._closed = False
    owner._executor = DirectUnadmittedExecutor()
    owner._held = lambda additional=0: parent._external_bytes+additional
    owner.settings = PreparationConfig(backend='rust', reader='participant-structured-v2')
    owner.actions = actions
    owner.summary = {'parent': {'run_sha256': '0'*64, 'scope':'UNADMITTED-not-controls'}}
    owner.models = {0:SimpleNamespace(event=SimpleNamespace(aggregate_window=2))}
    session = SimpleNamespace(recheck_final_validation=lambda _: None,
        export_final_validation=lambda *_: (_ for _ in ()).throw(AssertionError('no packet installed')))
    owner._session = session
    parent._native_inputs = owner
    token = _acquire_borrow(parent)
    loan = _NativeFinalValidationInputs(owner, parent, token, deadline=parent.deadline)
    parent._native_final_inputs = loan
    assert not (directory/'run.json').exists()  # Never a public/admitted formal parent.
    return parent, owner, loan, session, token, reservations, scope


@pytest.mark.parametrize('note_failure', [False, True])
def test_partial_native_source_retains_actual_loan_primary_lock_and_borrow_until_confirmed(tmp_path, monkeypatch, note_failure):
    parent, owner, loan, session, token, reservations, scope = unadmitted_loan(tmp_path, monkeypatch)
    broken = [True]
    closes = []
    def close():
        closes.append('slot')
        if broken[0]:
            raise OSError('UNADMITTED unconfirmed slot cleanup')
    session.close_final_validation = close
    loan._started = True  # Private lifecycle primitive, NOT an inspect/admission bypass.
    source = GameFinalValidationSource(parent, None, None, {'scope':'UNADMITTED-no-Source-permission'}, native=loan)
    loan.bind_source(source)
    source._borrow_token = token
    class Primary(RuntimeError):
        def add_note(self, note):
            if note_failure:
                raise MemoryError('controlled diagnostic allocation failure')
            super().add_note(note)
    primary = Primary('initiating owner failure')
    source._failure = primary
    try:
        with pytest.raises(Primary) as failure:
            source.close()
        assert failure.value is primary and parent._failure is primary
        assert primary._game_final_validation_cleanup_owner is source
        assert primary._game_final_validation_parent_owner is parent
        assert parent._native_final_inputs is loan and loan._owner is owner
        assert not source.cleanup_complete and not loan.cleanup_complete
        assert _require_borrow(parent, token) is token
        with (parent.directory/'.lock').open('a') as independent, pytest.raises(BlockingIOError):
            fcntl.flock(independent, fcntl.LOCK_EX|fcntl.LOCK_NB)
        before = len(closes)
        with pytest.raises(ContractError):
            _ = source.selected_records
        assert len(closes) == before
        broken[0] = False
        with pytest.raises(Primary) as failure:
            source.close()
        assert failure.value is primary and source.cleanup_complete and loan.cleanup_complete
        assert reservations[-1] == 0
        with pytest.raises(ContractError):
            _require_borrow(parent, token)
        before = len(closes)
        source.close()
        assert len(closes) == before
        with pytest.raises(ContractError):
            _ = source.selected_records
    finally:
        scope.__exit__(None, None, None)


def test_parent_never_consumes_original_input_stack_while_child_cleanup_is_uncertain(tmp_path, monkeypatch):
    parent, _, loan, session, _, _, scope = unadmitted_loan(tmp_path, monkeypatch)
    events = []
    parent._inputs_scope = ExitStack()
    parent._inputs_scope.callback(lambda: events.append('original inputs retired'))
    loan._started = True
    session.close_final_validation = lambda: (_ for _ in ()).throw(OSError('UNADMITTED keep child owner'))
    try:
        primary = RuntimeError('initiating parent retirement')
        with pytest.raises(RuntimeError) as error:
            parent._retire_inputs(primary)
        assert error.value is primary
        assert not events and not parent._inputs_retired and not parent.cleanup_complete
        assert parent._native_final_inputs is loan and not loan.cleanup_complete
    finally:
        session.close_final_validation = lambda: None
        with pytest.raises(RuntimeError):
            loan.close()
        parent._inputs_scope.close()
        _release_borrow(parent, loan._token)
        scope.__exit__(None, None, None)


def test_confirmed_slot_alone_cannot_release_parent_while_source_exclusion_is_uncertain(tmp_path, monkeypatch):
    import gf_training.card_instance_final_validation as actor
    parent, _, loan, _, token, _, scope = unadmitted_loan(tmp_path, monkeypatch)
    source = GameFinalValidationSource(parent, None, None, {'scope':'UNADMITTED'}, native=loan)
    loan.bind_source(source)
    source._borrow_token = token
    actual_release = actor._release_borrow
    broken = [True]
    def release(owner, value):
        if broken[0]:
            raise OSError('UNADMITTED exclusion release fault')
        actual_release(owner, value)
    monkeypatch.setattr(actor, '_release_borrow', release)
    events = []
    parent._inputs_scope = ExitStack()
    parent._inputs_scope.callback(lambda: events.append('original input close'))
    try:
        with pytest.raises(OSError):
            source.close()
        assert loan.cleanup_complete and not loan.parent_cleanup_complete and not source.cleanup_complete
        with pytest.raises(OSError):
            parent._retire_inputs()
        assert not events and not parent._inputs_retired
        broken[0] = False
        with pytest.raises(OSError):
            source.close()
        assert source.cleanup_complete and loan.parent_cleanup_complete
        parent._inputs_scope.close()
    finally:
        if not source.cleanup_complete:
            broken[0] = False
            try:
                source.close()
            except OSError:
                pass
        scope.__exit__(None, None, None)


def test_private_packet_bridge_reserves_whole_growth_before_decoder_and_drops_failed_views(originals, tmp_path, monkeypatch):
    import gf_training.native_card_instance_final_validation as decoder
    parent, _, loan, session, token, _reservations, scope = unadmitted_loan(tmp_path, monkeypatch, actions=originals[3])
    packet, expected = packet_from_reference(originals, 1)
    session.export_final_validation = lambda *_: packet
    loan._summary = originals[-1]
    called = []
    actual = decoder._decode_final_validation_packet
    def probe(*args, **kwargs):
        called.append(parent._external_bytes)
        return actual(*args, **kwargs)
    monkeypatch.setattr(decoder, '_decode_final_validation_packet', probe)
    try:
        result = loan._next(1, base_external=65536, max_owned_bytes=parent.limit, aggregate_window=2)
        assert result == expected and called and called[0] > 65536
        assert loan._packet is None
        parent.limit = 1024+65536
        called.clear()
        with pytest.raises(ContractError, match='reservation'):
            loan._next(1, base_external=65536, max_owned_bytes=parent.limit, aggregate_window=2)
        assert not called and loan._packet is None
        assert _require_borrow(parent, token) is token  # Failure cannot silently release exclusion.
    finally:
        loan._started = False
        try:
            loan.close()
        except ContractError:
            pass
        _release_borrow(parent, token)
        scope.__exit__(None, None, None)


def test_private_all64_dto_materialization_preserves_original_order_zero_endpoints_and_cumulative_charge(originals, tmp_path, monkeypatch):
    parent, _, loan, session, token, reservations, scope = unadmitted_loan(tmp_path, monkeypatch, actions=originals[3])
    indices = []
    def export(index, held):
        indices.append(index)
        return packet_from_reference(originals, index)[0]
    session.export_final_validation = export
    loan._summary = originals[-1]  # Structural decoder reference; not public Source admission.
    try:
        records = loan.records(base_external=65536, max_owned_bytes=parent.limit, aggregate_window=2)
        assert type(records) is tuple and len(records) == 64 and indices == list(range(64))
        assert sum(not row.frames for row in records) == 16
        assert {row.bank.actor for row in records} == {1, 2}
        assert all(row.end_kind == 'natural' for row in records)
        assert reservations[-1] > 65536 and loan._packet is None
        with pytest.raises(ContractError, match='one materialization'):
            loan.records(base_external=65536, max_owned_bytes=parent.limit, aggregate_window=2)
        del records
        loan.close()
        assert loan.cleanup_complete
    finally:
        parent.admit_external(0)
        _release_borrow(parent, token)
        scope.__exit__(None, None, None)


def test_default_capability_refuses_before_parent_borrow_plan_or_actor_allocation(tmp_path, monkeypatch):
    import gf_training.card_instance_final_validation as actor
    from tests.test_formal_ppo_control import loaded
    value = loaded(tmp_path)
    config = replace(value.config, preprocessing=replace(value.config.preprocessing, backend='rust'))
    value = replace(value, config=config, effective=asdict(config), sha256=digest(asdict(config)))
    touched = []
    monkeypatch.setattr(actor, '_acquire_borrow', lambda *_: touched.append('borrow'))
    monkeypatch.setattr(actor, '_reserve_controls', lambda *_: touched.append('controls'))
    native = pytest.importorskip('gf_native')
    if 'card_instance_final_validation' in dict(native.build_info()):
        pytest.skip('capability activation is a separate Root step')
    with pytest.raises(ContractError, match='capability|native|card-instance'):
        GameFinalValidationContext(value, object(), tmp_path/'never-created', object(), deadline=time.monotonic()+7200)
    assert not touched and not (tmp_path/'never-created').exists()


def test_management_stat_sized_joint_reservation_refuses_before_object_decoder(originals, tmp_path, monkeypatch):
    import gf_training.card_instance_final_validation as actor
    from gf_common.domain import RuntimePolicyBinding
    parent, _, loan, _, token, _, scope = unadmitted_loan(tmp_path, monkeypatch, actions=originals[3])
    root = originals[0]
    control = 65536
    growth = 32*(root/'manifest.json').stat().st_size+65536
    parent.limit = 1024+control+growth-1
    parent.admit_external(control)
    parent._borrow_native_final_inputs = lambda **_: loan
    decoded = []
    monkeypatch.setattr(actor._SourceOwner, 'object', lambda *args: decoded.append(True))
    loaded = SimpleNamespace(config=SimpleNamespace(preprocessing=PreparationConfig(backend='rust',
        reader='participant-structured-v2', memory_budget_mib=1024)))
    try:
        with pytest.raises(ContractError, match='reservation'):
            actor._open_native_original(loaded, parent, root, deadline=parent.deadline,
                max_owned_bytes=parent.limit, expected_plan=None, returned=None, pending=None,
                control_bytes=control, state={}, binding=RuntimePolicyBinding('mlp', {'expected_manifest_digest':'a'*64}),
                projected=None, ancestors={})
        assert not decoded and loan.cleanup_complete
        assert _require_borrow(parent, token) is token
    finally:
        parent.admit_external(0)
        _release_borrow(parent, token)
        scope.__exit__(None, None, None)


def test_native_facts_respects_smaller_caller_allowance_before_json_copy(tmp_path, monkeypatch):
    import gf_training.card_instance_final_validation as actor
    parent, _, loan, _, token, _, scope = unadmitted_loan(tmp_path, monkeypatch)
    source = GameFinalValidationSource(parent, None, None, {'scope':'UNADMITTED'}, native=loan, reservation=65536)
    loan.bind_source(source)
    # Only the allocation boundary is controlled; this remains an incomplete,
    # unadmitted Source with no records, originals, inspection or permission.
    source.recheck = parent.require_active
    source._management = SimpleNamespace(limit=1024+65536+32*len(source._facts)+65536-1)
    decoded = []
    monkeypatch.setattr(actor.json, 'loads', lambda *_: decoded.append(True))
    try:
        with pytest.raises(ContractError, match='before decode'):
            _ = source.facts
        assert not decoded and parent._failure is source._failure
    finally:
        source._management = None
        try:
            source.close()
        except ContractError:
            pass
        _release_borrow(parent, token)
        scope.__exit__(None, None, None)


def test_management_control_handoff_reaches_native_rejection_without_python_journal_fallback(originals, tmp_path, monkeypatch):
    import gf_training.card_instance_final_validation as actor
    from gf_common.domain import RuntimePolicyBinding
    from gf_training.collection import validation_groups
    parent, _, loan, _, token, _, scope = unadmitted_loan(tmp_path, monkeypatch, actions=originals[3])
    root, fixture, manifest, _, _, _ = originals
    parent._borrow_native_final_inputs = lambda **_: loan
    registry = {key: RuntimePolicyBinding(**value) for key, value in manifest['policy_bindings'].items()}
    plan = SimpleNamespace(linkage=manifest['linkage'], config_projection=manifest['config_projection'],
        groups=validation_groups(17, manifest['linkage']['policy_label'], actions_version=3), bindings=registry)
    monkeypatch.setattr(actor, '_plan', lambda *args: plan)
    python_reads = []
    def forbidden(*args, **kwargs):
        python_reads.append(True)
        raise AssertionError('Python participant reader must not run')
    monkeypatch.setattr(actor, '_inspect_component_collection', forbidden)
    import gf_training.card_instance_journal as journal
    monkeypatch.setattr(journal, 'read_card_instance_journal', forbidden)
    seen = []
    primary = RuntimeError('UNADMITTED controlled native structural rejection')
    def inspect(encoded):
        seen.append(json.loads(encoded))
        raise primary
    loan.inspect = inspect
    loaded = SimpleNamespace(config=SimpleNamespace(preprocessing=PreparationConfig(backend='rust',
        reader='participant-structured-v2', memory_budget_mib=1024)), effective=fixture['state']['effective_config'], inputs=())
    projected = SimpleNamespace(effective=manifest['config'], sha256=manifest['config_hash'])
    state = fixture['state']
    try:
        with pytest.raises(RuntimeError) as error:
            actor._open_native_original(loaded, parent, root, deadline=parent.deadline, max_owned_bytes=parent.limit,
                expected_plan=None, returned=None, pending=None, control_bytes=4*1024**2, state=state,
                binding=registry[manifest['linkage']['policy_label']], projected=projected, ancestors={})
        assert error.value is primary and len(seen) == 1 and not python_reads
        assert seen[0]['manifest'] == {'kind':'canonical'}
        assert seen[0]['expected_plan']['validation_invocation'] == manifest['validation_invocation']
        assert seen[0]['returned_report'] is None
        assert loan.cleanup_complete and parent._failure is primary
    finally:
        _release_borrow(parent, token)
        scope.__exit__(None, None, None)
