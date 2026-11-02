"""Concrete failed-cleanup reachability; no Source-qualification stand-in."""
import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_training.native_execution import NativeExecution
from gf_training.native_structured_session import NativeStructuredSource


@pytest.mark.parametrize('cancelled', [False, True])
def test_exit_keeps_exact_cleanup_owner_on_initiating_error_until_confirmed(tmp_path, cancelled):
    locks = []
    owner = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': [{'directory': str(tmp_path), 'parent_index': None}]},
        PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    owner._locks.callback(lambda: locks.append('released'))
    owner._executor = NativeExecution(SimpleNamespace(cancel=lambda: None))
    failed = [True]
    def close():
        if failed[0]:
            raise ContractError('injected unconfirmed cleanup')
    owner._session = SimpleNamespace(close=close)
    original = asyncio.CancelledError('original cancellation') if cancelled else RuntimeError('original caller error')
    reference = weakref.ref(owner)
    owner.__exit__(type(original), original, None)
    assert not owner._closed and not locks
    assert owner._executor._active is None  # every failing FFI operation actually joined
    assert original._structured_source_cleanup_owner is owner
    del owner
    gc.collect()
    assert reference() is original._structured_source_cleanup_owner
    retained = original._structured_source_cleanup_owner
    failed[0] = False
    retained.close()
    assert retained._closed and retained._executor._closed and locks == ['released']
    delattr(original, '_structured_source_cleanup_owner')
    del retained
    gc.collect()
    assert reference() is None


def test_enter_preserves_same_owner_when_initiating_failure_cleanup_twice_fails(tmp_path, monkeypatch):
    import gf_training.native_backend as backend
    monkeypatch.setattr(backend, 'binary_identity', lambda _: {'component': 'injected closed-lifecycle fixture'})
    failed = [True]
    initiating = RuntimeError('original constructor completion failure')
    def summary(*args):
        raise initiating
    def close():
        if failed[0]:
            raise ContractError('unconfirmed close')
    session = SimpleNamespace(control_summary=summary, close=close)
    fake = SimpleNamespace(StructuredNativeSession=lambda *args: session,
                           CancelHandle=lambda *args: SimpleNamespace(cancel=lambda: None))
    owner = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': [{'directory': str(tmp_path), 'parent_index': None}]},
        PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2,
        deadline=float('inf'), native=fake)
    reference = weakref.ref(owner)
    with pytest.raises(RuntimeError, match='original constructor completion failure') as caught:
        owner.__enter__()
    assert caught.value is initiating and initiating._structured_source_cleanup_owner is owner
    assert owner._executor._active is None
    del owner
    gc.collect()
    retained = initiating._structured_source_cleanup_owner
    assert reference() is retained
    failed[0] = False
    retained.close()
    assert retained._closed and retained._executor._closed


@pytest.mark.parametrize('entry', ['direct', 'body', 'cancel', 'enter'])
def test_executor_join_cancellation_retires_source_without_closed_executor_ffi(tmp_path, monkeypatch, entry):
    import gf_training.native_backend as backend

    monkeypatch.setattr(backend, 'binary_identity', lambda _: {'component': 'joined shutdown fixture'})
    initiating = asyncio.CancelledError('body cancellation') if entry == 'cancel' else RuntimeError('initiating body/entry error')
    session_calls = []
    def session_close():
        session_calls.append('closed')
    def summary():
        raise initiating
    session = SimpleNamespace(close=session_close, control_summary=summary)
    fake = SimpleNamespace(StructuredNativeSession=lambda *args: session,
        CancelHandle=lambda *args: SimpleNamespace(cancel=lambda: None))
    owner = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': [{'directory': str(tmp_path), 'parent_index': None}]},
        PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2,
        deadline=float('inf'), native=fake)
    released = []
    def release():
        assert owner._executor._closed and owner._executor._active is None
        released.append('after actual join')
    owner._locks.callback(release)
    original_shutdown = NativeExecution._shutdown
    def interrupted_after_join(executor):
        # Real NativeExecution shutdown joins/marks closed, then preserves a
        # cancellation. No fake closed flag or backup cleanup proves success.
        original_shutdown(executor)
        raise KeyboardInterrupt('interrupt after confirmed executor join')
    monkeypatch.setattr(NativeExecution, '_shutdown', interrupted_after_join)
    if entry == 'enter':
        with pytest.raises(RuntimeError) as caught:
            owner.__enter__()
        assert caught.value is initiating
    else:
        owner._executor = NativeExecution(fake.CancelHandle())
        owner._session = session
        owner._executor.call(lambda: None)  # create a real worker to join
        if entry == 'direct':
            with pytest.raises(KeyboardInterrupt, match='confirmed executor join'):
                owner.close()
        else:
            owner.__exit__(type(initiating), initiating, None)
    assert owner._closed and owner._session_closed
    assert session_calls == ['closed'] and released == ['after actual join']
    owner.close()  # never submits again to the closed executor
    assert session_calls == ['closed']


def test_unconfirmed_executor_shutdown_retains_owner_and_retries_without_native_close(tmp_path, monkeypatch):
    owner = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': [{'directory': str(tmp_path), 'parent_index': None}]},
        PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2, deadline=float('inf'))
    calls = []
    owner._session = SimpleNamespace(close=lambda: calls.append('native close'))
    owner._executor = NativeExecution(SimpleNamespace(cancel=lambda: None))
    owner._executor.call(lambda: None)
    released = []
    owner._locks.callback(lambda: released.append('released'))
    shutdown = owner._executor._shutdown
    failed = [True]
    def fail_before_join():
        if failed[0]:
            raise RuntimeError('shutdown still unconfirmed')
        shutdown()
    monkeypatch.setattr(owner._executor, '_shutdown', fail_before_join)
    try:
        with pytest.raises(RuntimeError, match='still unconfirmed') as caught:
            owner.close()
        assert caught.value._structured_source_cleanup_owner is owner
        assert owner._session_closed and not owner._closed and not released
        assert not owner._executor._closed
        failed[0] = False
        owner.close()
        assert owner._closed and owner._executor._closed and released == ['released']
        assert calls == ['native close']
    finally:
        failed[0] = False
        owner.close()


@pytest.mark.parametrize('allowance,expected', [(2 * 1024**2, 2 * 1024**2), (9 * 1024**3, 8 * 1024**3)])
def test_source_spool_uses_configured_scratch_bounded_by_actual_storage_allowance(tmp_path, allowance, expected):
    owner = NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
        'collections': [{'directory': str(tmp_path), 'parent_index': None}]},
        PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2,
        deadline=float('inf'), max_scratch_bytes=allowance)
    assert owner.scratch_limit == expected
    owner.close()


def test_exhausted_owned_storage_refuses_spool_before_constructor_or_source_io(tmp_path):
    with pytest.raises(ContractError, match='scratch exhausted before I/O'):
        NativeStructuredSource({'schema_version': 2, 'mode': 'historical_parent', 'parents': [],
            'collections': [{'directory': str(tmp_path / 'absent'), 'parent_index': None}]},
            PreparationConfig(backend='rust', reader='structured-v1'), max_owned_bytes=512 * 1024**2,
            deadline=float('inf'), max_scratch_bytes=0,
            native=SimpleNamespace(StructuredNativeSession=lambda *args: pytest.fail('no reservation left for native I/O')))
    assert not list(tmp_path.iterdir())


def test_native_projection_passes_exact_coordinator_allowance_before_source_creation(tmp_path, monkeypatch):
    import gf_training.native_backend as backend
    import gf_training.native_structured_session as module
    from gf_training.structured_ppo import _NativeComponentOps
    from tests.structured_ppo_support import configuration

    loaded = configuration(tmp_path)
    config = loaded.config
    monkeypatch.setattr(backend, 'binary_identity', lambda _: {'component': 'allowance propagation fixture'})
    operations = _NativeComponentOps(SimpleNamespace(StructuredNativeSession=lambda *args: None))
    # The native entry's separate config validation is covered by its ABI tests;
    # this fixture isolates transfer of the actual measured disk allowance.
    monkeypatch.setattr(operations, 'validate_config', lambda config: None)
    observed = []
    class Source:
        def __init__(self, plan, settings, **kwargs):
            self.summary = {'collections': [{'manifest_sha256': 'a' * 64, 'learner_label': 'learner'}]}
            observed.append(kwargs['max_scratch_bytes'])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write_numeric(self, directory):
            return 32, {}, {}
        def begin_update(self):
            pass
    monkeypatch.setattr(module, 'NativeStructuredSource', Source)
    plan = SimpleNamespace(linkage={'parent_run_path': str(tmp_path)}, output=tmp_path / 'collection')
    with operations.project(plan, tmp_path / 'numeric', config, 'learner', 0, float('inf'),
            expected_manifest_sha256='a' * 64, scratch_allowance_bytes=12345) as value:
        assert value[0] == 32
    assert observed == [12345]
