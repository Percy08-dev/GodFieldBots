"""Faults use actual retained originals/SQLite; cleanup is never inferred from GC."""
from __future__ import annotations

import ast
import sqlite3
import time
from pathlib import Path

import pytest

from gf_common.domain import ContractError
from gf_common.group_pack import MIB
from gf_training.group_pack_game_source import PackedBatchCustody
from gf_training.group_pack_pool import PackedArtifactPool
from gf_training.structured_lineage import _ChainScan
from tests.test_group_pack_game_source import owner, packed


def expire(original):
    original.deadline = original.pool.deadline = original.artifacts._deadline = 0.
    originals = original.artifacts._originals
    originals.deadline = 0.
    for file in (original.locator, originals.database_file, originals.descriptor_file, originals.predecessor_file):
        if file is not None:
            file.deadline = 0.


def assert_retired(original, connection=None):
    assert original._closed and original.locator.fd == -1
    assert original.artifacts._originals.database_file.fd == -1
    assert original.artifacts._originals.descriptor_file.fd == -1
    assert original.artifacts._originals.db is None
    if connection is not None:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')


@pytest.mark.parametrize('active', [False, True])
def test_expired_custody_retires_actual_connection_fds_and_account(tmp_path, active):
    root, _ = packed(tmp_path)
    original = owner(root)
    if active:
        original.read(root / 'first/e0')
    connection = original.artifacts._originals.db
    custody = PackedBatchCustody(original)
    expire(original)
    try:
        with pytest.raises(TimeoutError):
            custody.close()
        assert custody.retired and original.pool.closed and original.pool.budget.used == 0
        assert_retired(original, connection)
        with pytest.raises((TimeoutError, RuntimeError)):
            original.read(root / 'first/e0')
    finally:
        # Lab cleanup permits reproducing RED without leaving leaked actual FDs.
        original.pool.deadline = original.artifacts._deadline = time.monotonic() + 30
        original.close()


def test_empty_pool_cleanup_needs_no_new_deadline_admission():
    pool = PackedArtifactPool(max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    pool.deadline = 0.
    try:
        pool.close()
        assert pool.closed and pool.budget.used == 0
    finally:
        pool.deadline = time.monotonic() + 30
        pool.close()


def test_scan_recheck_expiry_still_retires_owned_originals(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    original.read(root / 'first/e0')
    connection = original.artifacts._originals.db
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners[root] = original
    scan._packed_pool = original.pool
    expire(original)
    scan.deadline = 0.
    try:
        with pytest.raises(TimeoutError):
            scan.close()
        assert scan._closed and original.pool.closed and original.pool.budget.used == 0
        assert_retired(original, connection)
    finally:
        original.pool.deadline = original.artifacts._deadline = time.monotonic() + 30
        original.close()


def test_unrelated_handled_error_cannot_swallow_scan_close_failure(tmp_path, monkeypatch):
    root, _ = packed(tmp_path)
    original = owner(root)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners[root] = original
    actual_close = original.close
    failure = OSError('actual Source close fault')
    def fail():
        raise failure
    monkeypatch.setattr(original, 'close', fail)
    unrelated = ValueError('unrelated caller exception')
    try:
        try:
            raise unrelated
        except ValueError:
            with pytest.raises(OSError) as captured:
                scan.close(recheck=False)
        assert captured.value is failure
        assert not hasattr(unrelated, 'cleanup_owner')
        assert not scan._closed and original.locator.fd >= 0
    finally:
        monkeypatch.setattr(original, 'close', actual_close)
        scan.close(recheck=False)


def test_explicit_scan_body_error_survives_cleanup_fault(tmp_path, monkeypatch):
    root, _ = packed(tmp_path)
    original = owner(root)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners[root] = original
    actual_close = original.close
    def fail():
        raise OSError('actual retained owner close fault')
    monkeypatch.setattr(original, 'close', fail)
    primary = ContractError('own scan body failed')
    try:
        with pytest.raises(ContractError) as captured, scan:
            raise primary
        assert captured.value is primary and primary.cleanup_owner is scan
        assert not scan._closed and original.locator.fd >= 0
        assert 'actual retained owner close fault' in '\n'.join(primary.__notes__)
    finally:
        monkeypatch.setattr(original, 'close', actual_close)
        scan.close(recheck=False)


@pytest.mark.parametrize('stage', ['names', 'index', 'register', 'register-after'])
@pytest.mark.parametrize('close_failure', [False, True])
def test_partial_constructor_keeps_actual_original_owner(tmp_path, monkeypatch, stage, close_failure):
    from gf_training import group_pack_source as source_module
    _root, locator = packed(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    held = []
    original_constructor = source_module.PackedOriginals
    close_methods = []
    class FailSecondIteration(list):
        count = 0
        def __iter__(self):
            self.count += 1
            if self.count == 2:
                raise MemoryError('injected index allocation')
            return super().__iter__()
    def capture(*args, **kwargs):
        originals = original_constructor(*args, **kwargs)
        held.append(originals)
        if stage == 'index':
            originals.validated['index']['members'] = FailSecondIteration(originals.validated['index']['members'])
        if close_failure:
            file = originals.database_file
            close_methods.append(file.close)
            def fail_close():
                raise OSError('actual partial original close fault')
            monkeypatch.setattr(file, 'close', fail_close)
        return originals
    monkeypatch.setattr(source_module, 'PackedOriginals', capture)
    if stage == 'names':
        def fail_names(*args, **kwargs):
            raise MemoryError('injected name allocation')
        monkeypatch.setattr(source_module, 'sorted', fail_names, raising=False)
    if stage in ('register', 'register-after'):
        register = pool.register
        def fail_register(source):
            if stage == 'register-after':
                register(source)
            raise MemoryError('injected registration allocation')
        monkeypatch.setattr(pool, 'register', fail_register)
    before = len(tuple(Path('/proc/self/fd').iterdir()))
    try:
        with pytest.raises(MemoryError) as captured:
            source_module.PackedSource(locator.path, deadline=time.monotonic() + 30,
                max_owned_bytes=512 * MIB, pool=pool)
        partial = captured.value.cleanup_owner
        assert type(partial) is source_module.PackedSource
        assert partial._originals is held[0]
        if close_failure:
            assert not partial._closed and held[0].database_file.fd >= 0
            assert pool.budget.used > 64 * MIB
            assert partial in [*pool.sources, *pool._cleanup_children]
            assert 'close fault' in '\n'.join(captured.value.__notes__)
        else:
            assert partial._closed and held[0].database_file.fd == -1 and held[0].db is None
            assert pool.budget.used == 64 * MIB
            assert len(tuple(Path('/proc/self/fd').iterdir())) == before
    finally:
        if held:
            if close_methods:
                monkeypatch.setattr(held[0].database_file, 'close', close_methods[0])
            held[0].close()
        pool.close()
    assert pool.budget.used == 0


def test_training_has_no_cli_import_dependency():
    import gf_training.card_instance_ppo as training
    tree = ast.parse(Path(training.__file__).read_bytes())
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('gf_cli')
        for node in ast.walk(tree))


def test_scan_attempts_other_originals_when_one_close_remains_uncertain(tmp_path, monkeypatch):
    pool = PackedArtifactPool(max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    a, _ = packed(tmp_path / 'a')
    b, _ = packed(tmp_path / 'b')
    first, second = owner(a, pool), owner(b, pool)
    second.read(b / 'first/e0')
    connection = second.artifacts._originals.db
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners = {a: first, b: second}
    scan._packed_pool = pool
    file = first.artifacts._originals.database_file
    close = file.close
    def fail():
        raise OSError('first original FD close remains uncertain')
    monkeypatch.setattr(file, 'close', fail)
    pool.deadline = first.artifacts._deadline = second.artifacts._deadline = 0.
    try:
        with pytest.raises(OSError) as captured:
            scan.close(recheck=False)
        assert captured.value.cleanup_owner is scan
        assert not scan._closed and not pool.closed and file.fd >= 0
        assert_retired(second, connection)
        assert first.locator.fd == -1 and first.artifacts._originals.descriptor_file.fd == -1
        assert pool.budget.used > 64 * MIB
    finally:
        monkeypatch.setattr(file, 'close', close)
        scan.close(recheck=False)
    assert pool.closed and pool.budget.used == 0


def test_expired_live_loan_is_still_guarded_until_explicit_return(tmp_path):
    root, _ = packed(tmp_path)
    original = owner(root)
    with original.borrow_group('first') as loan:
        connection = original.artifacts._originals.db
        expire(original)
        with pytest.raises(RuntimeError, match='loan'):
            original.close()
        assert original.locator.fd >= 0 and original.artifacts._originals.database_file.fd >= 0
        with pytest.raises((TimeoutError, RuntimeError)):
            loan.read('first/e0')
    original.close()
    assert_retired(original, connection)
    assert original.pool.closed and original.pool.budget.used == 0


@pytest.mark.parametrize('close_failure', [False, True])
def test_outer_constructor_keeps_actual_partial_owner_until_confirmed_close(tmp_path, monkeypatch, close_failure):
    from gf_training import group_pack_source as source_module
    root, _ = packed(tmp_path)
    create = source_module.PackedOriginals
    originals_seen = []
    actual_close = []
    def capture(*args, **kwargs):
        originals = create(*args, **kwargs)
        originals_seen.append(originals)
        if close_failure:
            actual_close.append(originals.database_file.close)
            def fail():
                raise OSError('actual constructor original could not close')
            monkeypatch.setattr(originals.database_file, 'close', fail)
        return originals
    def fail_names(*args, **kwargs):
        raise MemoryError('outer owner index allocation')
    monkeypatch.setattr(source_module, 'PackedOriginals', capture)
    monkeypatch.setattr(source_module, 'sorted', fail_names, raising=False)
    with pytest.raises(MemoryError) as captured:
        owner(root)
    outer = captured.value.cleanup_owner
    originals = originals_seen[0]
    if close_failure:
        assert not outer._closed and not outer.pool.closed
        assert originals.database_file.fd >= 0 and outer.pool.budget.used > 64 * MIB
        assert any(partial._originals is originals for partial in outer.pool._cleanup_children)
        monkeypatch.setattr(originals.database_file, 'close', actual_close[0])
        outer.close()
    else:
        assert outer._closed and originals.database_file.fd == -1
    assert outer.pool.closed and outer.pool.budget.used == 0


def test_one_shot_success_inside_unrelated_except_propagates_real_close_fault(tmp_path, monkeypatch):
    from gf_training import structured_lineage as lineage
    root, _ = packed(tmp_path)
    original = owner(root)
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners[root] = original
    monkeypatch.setattr(lineage, '_ChainScan', lambda **kwargs: scan)
    monkeypatch.setattr(lineage, '_validate_component_chain', lambda *args, **kwargs: ({}, []))
    close = original.close
    failure = OSError('one-shot real original close failure')
    def fail():
        raise failure
    monkeypatch.setattr(original, 'close', fail)
    unrelated = ValueError('already handled unrelated caller failure')
    try:
        try:
            raise unrelated
        except ValueError:
            with pytest.raises(OSError) as captured:
                lineage.validate_game_chain(tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
        assert captured.value is failure and not hasattr(unrelated, 'cleanup_owner')
        assert not scan._closed and original.locator.fd >= 0
    finally:
        monkeypatch.setattr(original, 'close', close)
        scan.close(recheck=False)


def test_one_shot_own_timeout_retires_expired_actual_source(tmp_path, monkeypatch):
    from gf_training import structured_lineage as lineage
    root, _ = packed(tmp_path)
    original = owner(root)
    original.read(root / 'first/e0')
    connection = original.artifacts._originals.db
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    scan._packed_owners[root] = original
    scan._packed_pool = original.pool
    monkeypatch.setattr(lineage, '_ChainScan', lambda **kwargs: scan)
    primary = TimeoutError('own one-shot validation timed out')
    def timeout(*args, **kwargs):
        expire(original)
        scan.deadline = 0.
        raise primary
    monkeypatch.setattr(lineage, '_validate_component_chain', timeout)
    with pytest.raises(TimeoutError) as captured:
        lineage.validate_game_chain(tmp_path, max_owned_bytes=512 * MIB, deadline=time.monotonic() + 30)
    assert captured.value is primary
    assert scan._closed and original.pool.closed and original.pool.budget.used == 0
    assert_retired(original, connection)


def test_packed_training_dispatch_needs_no_cli_import(tmp_path, monkeypatch):
    import asyncio
    import builtins

    from gf_training import ppo_run
    from gf_training.card_instance_ppo import train_card_instance_ppo
    from tests.test_group_pack_game_ppo import setup
    loaded, collector, _ = setup(tmp_path, monkeypatch)
    previous = loaded.sha256, loaded.effective.copy()
    async def shared(loaded, collector, directory, **kwargs):
        return {'shared_coordinator': True}
    monkeypatch.setattr(ppo_run, '_run_ppo', shared)
    ordinary_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name.startswith('gf_cli'):
            pytest.fail('training depended on CLI configuration policy')
        return ordinary_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    assert asyncio.run(train_card_instance_ppo(loaded, collector, tmp_path / 'unused')) == {'shared_coordinator': True}
    assert (loaded.sha256, loaded.effective) == previous
    assert not (tmp_path / 'unused').exists()
