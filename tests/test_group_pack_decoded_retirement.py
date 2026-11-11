"""Private lifecycle/accounting tests; inert actors grant no learning authority."""
from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError, canonical
from gf_common.group_pack_game import storage_binding
from gf_common.preprocessing import PreparationConfig
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.card_instance_source import (
    IssuedGameProjection,
    _GameBatchSource,
    _OnlineGameOwner,
)
from gf_training.group_pack_game_source import PackedBatchCustody, PackedSourceOwner
from gf_training.lineage import _OWNED_LOCKS, RunOwnership
from gf_training.structured_lineage import _ChainScan
from gf_training.structured_source import ComponentCollection, ComponentEpisode, _SourceSnapshot
from tests.test_group_pack_game_source import packed


@dataclass
class _DecodedRecord:
    frames: tuple[int, ...]


@pytest.fixture
def issued(tmp_path):
    root, _ = packed(tmp_path)
    original = PackedSourceOwner(root, PreparationConfig(memory_budget_mib=2048), time.monotonic()+30,
        storage_binding=storage_binding(), max_owned_bytes=2*1024**3)
    original.read_raw(root/'first/e0')  # Keep actual cached Pack/SQLite ownership.
    custody = PackedBatchCustody(original)
    parent = tmp_path
    artifacts = parent/'artifacts'
    (artifacts/'000000').mkdir(parents=True)
    (artifacts/'000001').mkdir()
    token = 'UNADMITTED-lifecycle-test'
    _OWNED_LOCKS[parent] = token
    hardware = {'scope': 'UNADMITTED'}
    process = SimpleNamespace(directory=parent, process=SimpleNamespace(pid=123, returncode=None),
        steps=8, hardware=hardware, _issued_handle=None)
    ledger = ReservationLedger()
    lease = ledger.reserve_learner(0, process)
    lease.pid = 123
    process.lease, process.ledger = lease, ledger
    state = {'schema_version': 4, 'updates': 0, 'optimizer_steps': 0, 'hardware': hardware}
    controls = _ChainScan(directory=parent, max_owned_bytes=2*1024**3, deadline=original.deadline)
    custody.borrow(controls)
    info = artifacts.stat()
    controls.artifact_snapshot = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    online = _OnlineGameOwner(RunOwnership(parent, token), state, process, ledger,
        collection_controls=SimpleNamespace(finalize=lambda: None), packed_batch=custody)
    record = _DecodedRecord(tuple(range(236)))
    reference = weakref.ref(record)
    proof = {'storage': original.storage, 'custody': original.custody, 'scope': 'UNADMITTED'}
    collection = ComponentCollection(root, (ComponentEpisode('UNADMITTED', 1, 'mlp', '3', record),),
        b'{"scope":"UNADMITTED"}', canonical(proof).encode())
    snapshot = _SourceSnapshot(collection, original, 4096)
    source = _GameBatchSource(snapshot, controls, state, None, None, None, {}, 'UNADMITTED', online, (record,))
    source._numeric = (parent/'numeric', 'a'*64, b'{}')
    handle = IssuedGameProjection(source, process, canonical({'expected_optimizer_steps': 0,
        'collection_generation': 0}).encode(), parent/'numeric', {})
    source.issued = online.issued = handle
    handle.completed = True
    del record, collection
    try:
        yield handle, original, reference
    finally:
        try:
            custody.close()
        except BaseException as error:  # noqa: BLE001 -- verify actual cleanup preserves the injected failure.
            assert error is handle._decoded_retirement_error
            assert custody.retired and original._closed
        controls.close(recheck=False)
        _OWNED_LOCKS.pop(parent, None)


def test_consumed_graph_retires_before_temporary_artifact_charge_with_same_originals(issued, monkeypatch):
    handle, original, reference = issued
    original.charge(1_760_000_000)
    budget = original.pool.budget
    temporary = 322_850_000
    with pytest.raises(ValueError, match='budget'):
        budget.acquire(temporary, original.deadline)
    database = original.artifacts._originals.database_file
    connection = original.artifacts._originals.db
    cache = original.artifacts._cache_charge
    release = budget.release
    def refund(size):
        if size <= 1024**2:
            release(size)
            return
        assert handle.source.records is None
        assert handle.source.snapshot.collection.episodes == ()
        assert reference() is None
        release(size)
    monkeypatch.setattr(budget, 'release', refund)
    handle.retire_packed_decoded()
    monkeypatch.setattr(budget, 'release', release)
    assert original.artifacts._originals.database_file is database and database.fd >= 0
    assert original.artifacts._originals.db is connection
    assert original.artifacts._cache_charge == cache and cache > 0
    budget.acquire(temporary, original.deadline)
    budget.release(temporary)
    handle.require(handle.process, after=True)
    with pytest.raises(AttributeError):
        handle._verified_decisions = 1
    handle.process.steps = 4
    with pytest.raises(ContractError):
        handle.require(handle.process, after=True)
    handle.process.steps = 8
    with pytest.raises(ContractError):
        handle.require(handle.process)
    with pytest.raises(ContractError):
        handle.retire_packed_decoded()
    handle.prepare_commit()
    assert handle.prepared and original._closed
    assert original._retained_metadata_bytes == original._control_object_charge == 0


@pytest.mark.parametrize('fault', ['incomplete', 'process', 'steps', 'generation', 'active-request', 'updating', 'worker', 'prepared', 'count'])
def test_retirement_refuses_wrong_or_active_update_before_clearing_graph(issued, fault):
    handle, original, reference = issued
    source = handle.source
    if fault == 'incomplete': handle.completed = False
    elif fault == 'process': handle.process = SimpleNamespace()
    elif fault == 'steps': handle.process.steps = 7
    elif fault == 'generation': source.online.state['updates'] = 1
    elif fault == 'active-request': handle.process._issued_handle = handle
    elif fault == 'updating': source.online.ledger.updating = True
    elif fault == 'worker': source.online.ledger.workers[1] = object()
    elif fault == 'prepared': handle.prepared = True
    elif fault == 'count': source.records = (_DecodedRecord((1,)),)
    before = original.pool.budget.used
    with pytest.raises(ContractError):
        handle.retire_packed_decoded()
    assert source.records is not None and source.snapshot.collection.episodes
    assert reference() is not None and original.pool.budget.used == before


def test_active_group_and_wrong_custody_owner_refuse_retirement(issued):
    handle, original, reference = issued
    with original.borrow_group('first'), pytest.raises(ContractError):
        handle.retire_packed_decoded()
    custody = handle.source.online.packed_batch
    custody.owner = object()
    try:
        with pytest.raises(ContractError):
            handle.retire_packed_decoded()
        assert reference() is not None
    finally:
        custody.owner = original


@pytest.mark.parametrize('bound', [-1, True, 2**40])
def test_invalid_remaining_bound_never_refunds_live_account(issued, monkeypatch, bound):
    _handle, original, _reference = issued
    before = original.pool.budget.used
    monkeypatch.setattr(original.pool.budget, 'release', lambda *_: pytest.fail('invalid bound released bytes'))
    with pytest.raises((ValueError, ContractError)):
        original.retire_decoded(retained_bytes=bound)
    assert original.pool.budget.used == before
    monkeypatch.undo()


def test_exported_control_graphs_and_remaining_metadata_keep_their_existing_charge(issued):
    handle, original, reference = issued
    root = original.root
    values = [original.object(root/'manifest.json'), original.object(root/'manifest.json')]
    control = original._entries['manifest.json']
    exported = 2*(32*control['bytes']+65536)
    original.charge(2*1024**2)
    handle.source.records = None
    handle.source.snapshot.collection = replace(handle.source.snapshot.collection, episodes=())
    assert reference() is None
    other = original.pool.budget.used-original._charge_bytes
    original.retire_decoded(retained_bytes=65536)
    assert original._charge_bytes >= exported+65536
    assert original.pool.budget.used-other == original._charge_bytes
    assert values == [{'hello': 'original'}, {'hello': 'original'}]
    before = original._charge_bytes
    original.object(root/'manifest.json')
    assert original._charge_bytes > before


def test_default_retirement_cannot_refund_live_online_source_metadata(issued):
    handle, original, reference = issued
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    assert reference() is None and handle.source.snapshot.collection._proof
    proof = handle.source.proof
    before = original.pool.budget.used
    charge = original._charge_bytes
    original.retire_decoded()  # A later historical cleanup uses the same owner.
    assert original.pool.budget.used == before and original._charge_bytes == charge
    assert handle.source.proof == proof
    assert original.artifacts._originals.database_file.fd >= 0
    control = original._entries['manifest.json']
    original.object(original.root/'manifest.json')
    original.retire_decoded()
    assert original._charge_bytes >= charge+32*control['bytes']+65536


@pytest.mark.parametrize('bound', [-1, True, 2**40])
def test_invalid_bound_does_not_change_existing_owner_pin(issued, bound):
    handle, original, _reference = issued
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    before = original.pool.budget.used
    pinned = original._retained_metadata_bytes
    assert pinned > 0
    original.retire_decoded()
    with pytest.raises((ValueError, ContractError)):
        original.retire_decoded(retained_bytes=bound)
    assert original.pool.budget.used == before
    assert original._retained_metadata_bytes == pinned
    original.retire_decoded()
    assert original.pool.budget.used == before


@pytest.mark.parametrize('after_effect', [False, True])
def test_owner_pin_survives_uncertain_close_until_actual_owner_retires(issued, monkeypatch, after_effect):
    handle, original, _reference = issued
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    pinned, charge = original._retained_metadata_bytes, original._charge_bytes
    custody = handle.source.online.packed_batch
    close = original.artifacts.close
    def uncertain():
        if after_effect:
            close()
        raise OSError('actual artifact source close uncertain')
    monkeypatch.setattr(original.artifacts, 'close', uncertain)
    with pytest.raises(OSError, match='uncertain'):
        custody.close()
    assert not original._closed and original._retained_metadata_bytes == pinned
    assert original._charge_bytes == charge
    monkeypatch.setattr(original.artifacts, 'close', close)
    # Independent locator close already retired that FD; its normal recheck
    # failure remains reportable even though subsequent actual cleanup succeeds.
    with pytest.raises(RuntimeError, match='closed'):
        custody.close()
    assert custody.retired and original._closed
    assert original._retained_metadata_bytes == original._control_object_charge == 0


@pytest.mark.parametrize('after_effect', [False, True])
def test_owned_pool_close_failure_keeps_wrapper_pin_funded_until_later_confirmed_close(issued, monkeypatch, after_effect):
    handle, original, _reference = issued
    original.object(original.root/'manifest.json')
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    pinned, wrapper = original._retained_metadata_bytes, original._charge_bytes
    subtotal = original._control_object_charge
    pool = original.pool
    close = pool.close
    base_before = pool._base_charge
    calls = []
    failure = OSError('actual owned pool close fault')
    def fault():
        calls.append(pool)
        if after_effect:
            close()
        raise failure
    monkeypatch.setattr(pool, 'close', fault)
    try:
        with pytest.raises(OSError) as caught:
            original.close()
        assert caught.value is failure and calls == [pool]
        assert vars(failure).get('cleanup_owner') is original
        assert not original._closed
        assert original._retained_metadata_bytes == pinned
        assert original._control_object_charge == subtotal
        assert original._charge_bytes == wrapper
        assert pool.budget.used >= wrapper
        assert pool._base_charge == (0 if after_effect else base_before)
    finally:
        monkeypatch.setattr(pool, 'close', close)
        # A close error never becomes a successful Source/commit. Subsequent
        # actual cleanup is allowed to retire the same idempotent pool/owner.
        try:
            handle.source.online.packed_batch.close()
        except (OSError, RuntimeError):
            assert original._closed
    assert original._closed and original._charge_bytes == 0
    assert original._retained_metadata_bytes == original._control_object_charge == 0


@pytest.mark.parametrize('after_effect', [False, True])
def test_owned_pool_base_refund_fault_is_settled_once_before_wrapper_refund(issued, monkeypatch, after_effect):
    handle, original, _reference = issued
    original.object(original.root/'manifest.json')
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    pool = original.pool
    base, wrapper, pinned = pool._base_charge, original._charge_bytes, original._retained_metadata_bytes
    release = pool.budget.release
    effects = []
    failure = OSError('owned pool base refund fault')
    def fault(size):
        if size == base:
            if after_effect:
                release(size)
                effects.append(('base', size))
            raise failure
        release(size)
        if size == wrapper:
            effects.append(('wrapper', size))
    monkeypatch.setattr(pool.budget, 'release', fault)
    try:
        with pytest.raises(OSError) as caught:
            original.close()
        assert caught.value is failure and vars(failure).get('cleanup_owner') is original
        assert original._charge_bytes == wrapper and pool.budget.used >= wrapper
        assert original._retained_metadata_bytes == pinned and not original._closed
        assert [kind for kind, _ in effects] == (['base'] if after_effect else [])
    finally:
        def healthy(size):
            release(size)
            if size == base:
                effects.append(('base', size))
            elif size == wrapper:
                effects.append(('wrapper', size))
        monkeypatch.setattr(pool.budget, 'release', healthy)
        try:
            handle.source.online.packed_batch.close()
        except (OSError, RuntimeError):
            assert original._closed
    assert [kind for kind, _ in effects] == ['base', 'wrapper']
    assert original._charge_bytes == original._retained_metadata_bytes == 0


def test_uncertain_wrapper_account_blocks_owned_pool_close_before_base_refund(issued, monkeypatch):
    handle, original, _reference = issued
    original.charge(64*1024**2)
    handle.retire_packed_decoded()
    base, wrapper, pinned = original.pool._base_charge, original._charge_bytes, original._retained_metadata_bytes
    with monkeypatch.context() as patch:
        # An inert fault flag tests the guard; no real account effect is guessed.
        patch.setattr(original, '_charge_uncertain', True)
        patch.setattr(original.pool, 'close', lambda: pytest.fail('uncertain account closed owned pool'))
        with pytest.raises(ContractError) as failure:
            original.close()
        assert vars(failure.value).get('cleanup_owner') is original
        assert original.pool._base_charge == base and original._charge_bytes == wrapper
        assert original._retained_metadata_bytes == pinned
    # Restore only the injected flag; the failure remains reportable after real cleanup.
    with pytest.raises(ContractError):
        handle.source.online.packed_batch.close()
    assert original._closed and original._charge_bytes == 0


@pytest.mark.parametrize('after_effect', [False, True])
def test_refund_fault_keeps_actual_owner_and_blocks_all_commit_retries(issued, monkeypatch, after_effect):
    handle, original, reference = issued
    original.charge(64*1024**2)
    release = original.pool.budget.release
    failure = OSError('decoded refund uncertain')
    def refund(size):
        if size <= 1024**2:
            release(size)
            return
        assert reference() is None
        if after_effect:
            release(size)
        raise failure
    monkeypatch.setattr(original.pool.budget, 'release', refund)
    with pytest.raises(OSError) as caught:
        handle.retire_packed_decoded()
    assert caught.value is failure and not handle.prepared
    assert original.artifacts._originals.database_file.fd >= 0
    with pytest.raises((ContractError, OSError)):
        handle.prepare_commit()
    assert not handle.prepared
    monkeypatch.setattr(original.pool.budget, 'release', release)
