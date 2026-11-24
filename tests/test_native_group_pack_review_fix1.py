from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gf_training.native_group_pack import OriginalArtifactProvider, _PackedNativeExecution
from tests.test_group_pack_game_source import owner, packed


def test_empty_provider_requires_actual_owned_gen0_scan_pool_and_binding(tmp_path):
    import json
    import time

    from gf_common.domain import ContractError, digest
    from gf_common.group_pack_game import storage_binding
    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.structured_lineage import _ChainScan
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=time.monotonic()+30)
    # Inert binding test only; ordinary validation is exercised separately below.
    scan.configure_packed({'schema_version': 4, 'storage_binding': storage_binding(), 'updates': 0,
        'generation_sha256': 'a'*64, 'effective_config': {'storage': {'budget_mib': 512}}})
    scan.files[tmp_path/'run.json'] = ((1, 2, 3, 4, 5), 'b'*64)
    scan.files[tmp_path/'generations/000000.json'] = ((1, 2, 3, 4, 5), 'a'*64)
    scan.charge(4096)
    pool = scan.ensure_packed_pool()
    with pytest.raises(ContractError):
        OriginalArtifactProvider((), pool=pool)
    with OriginalArtifactProvider((), pool=pool, shared_controls=scan) as provider:
        expected = {'format': 'python-original-bytes-v1', 'roots': []}
        assert json.loads(provider.registry()) == expected and provider.registry_sha256 == digest(expected)
        for callback in (lambda: provider.describe('/unknown'), lambda: provider.borrow('/unknown', 'x'), lambda: provider.recheck('/unknown')):
            with pytest.raises(ContractError):
                callback()
    assert type(pool) is PackedArtifactPool
    scan.close(recheck=False)
    assert pool.closed and pool.budget.used == 0


@pytest.mark.parametrize('stage', ['metadata', 'query'])
@pytest.mark.parametrize('after', [False, True])
def test_provider_acquisition_fault_has_actual_owner_and_balanced_later_close(tmp_path, monkeypatch, stage, after):
    root, _ = packed(tmp_path)
    original = owner(root)
    budget = original.pool.budget
    before = budget.used
    acquire = budget.acquire
    query_size = 65536 + 1024 * len(original.artifacts._groups)
    injected = []
    failure = MemoryError(f'{stage} admission fault')
    def fault(amount, deadline, **kwargs):
        target = amount == query_size if stage == 'query' else not injected
        if target and not injected:
            injected.append(amount)
            if after:
                acquire(amount, deadline, **kwargs)
            raise failure
        return acquire(amount, deadline, **kwargs)
    monkeypatch.setattr(budget, 'acquire', fault)
    with pytest.raises(MemoryError) as caught:
        OriginalArtifactProvider((original,))
    assert caught.value is failure and injected
    actual = caught.value._native_original_provider_cleanup_owner
    assert type(actual) is OriginalArtifactProvider
    assert budget.used == before and original.pool.active is None
    actual.close()
    assert budget.used == before
    original.close()
    assert original.pool.closed and budget.used == 0


@pytest.mark.parametrize('phase', ['before', 'partial', 'complete'])
def test_actual_executor_initializer_no_submission_states_retire_without_api_on_partial(tmp_path, monkeypatch, phase):
    root, _ = packed(tmp_path)
    original = owner(root)
    provider = OriginalArtifactProvider((original,))
    before = original.pool.budget.used
    initialize = ThreadPoolExecutor.__init__
    actual_objects = []
    failure = OSError('initializer ' + phase)
    def initialize_fault(actual, *args, **kwargs):
        actual_objects.append(actual)
        if phase == 'partial':
            actual._max_workers = 1
        elif phase == 'complete':
            initialize(actual, *args, **kwargs)
        raise failure
    monkeypatch.setattr(ThreadPoolExecutor, '__init__', initialize_fault)
    submit = ThreadPoolExecutor.submit
    def no_partial_submit(actual, *args, **kwargs):
        if actual in actual_objects:
            pytest.fail('no-effect constructor scheduled cleanup work')
        return submit(actual, *args, **kwargs)
    monkeypatch.setattr(ThreadPoolExecutor, 'submit', no_partial_submit)
    try:
        with pytest.raises(OSError) as caught:
            _PackedNativeExecution(SimpleNamespace(cancel=lambda: None), provider)
        assert caught.value is failure
        executor = provider._executor_owner
        assert executor._closed and executor._worker is None and executor._hold is None
        assert not original.pool._close_holds and original.pool.budget.used == before
    finally:
        # RED-only lab cleanup of known no-submission objects, never a production
        # repair or proof: restore stdlib init on the SAME actual object.
        executor = provider._executor_owner
        if not executor._closed:
            monkeypatch.setattr(ThreadPoolExecutor, 'submit', submit)
            initialize(executor._executor, max_workers=1)
            with pytest.raises(OSError):
                executor.close()
        if executor._closed:
            provider.close()
            original.close()


def test_ordinary_fresh_native_parent_validates_actual_gen0_before_first_collection(tmp_path, monkeypatch):
    import asyncio
    import json
    from dataclasses import asdict, replace

    native = pytest.importorskip('gf_native')
    if not hasattr(native.CardInstanceNativeSession, 'from_artifacts'):
        pytest.skip('private source-matched Native3B candidate required')
    from gf_cli.config import LoadedConfig
    from gf_common.domain import digest
    from gf_training import native_backend
    from gf_training.card_instance_ppo import train_card_instance_ppo
    from gf_training.native_card_instance import _NativeGameInputs
    from tests.test_group_pack_game_ppo import setup
    loaded, _, _ = setup(tmp_path, monkeypatch)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    loaded = LoadedConfig(config, asdict(config), digest(asdict(config)), ())
    # Private component gate only. No marker is published and no collector,
    # Simulator or PPO optimizer update is executed by this fresh-control test.
    monkeypatch.setattr(native_backend, 'load_native', lambda: native)
    monkeypatch.setattr(native_backend, 'group_pack_reader_settings', lambda config, native=None, **_: asdict(config))
    construct = _NativeGameInputs._construct_session
    providers = []
    def capture(actual, session_class, plan, settings, cancel):
        value = json.loads(plan)
        provider = actual._artifact_provider
        assert value['schema_version'] == 4 and value['mode'] == 'historical_parent'
        assert value['collections'] == [] and json.loads(provider.registry())['roots'] == []
        assert provider._empty_controls is actual._shared_controls
        providers.append(provider)
        return construct(actual, session_class, plan, settings, cancel)
    monkeypatch.setattr(_NativeGameInputs, '_construct_session', capture)
    class BeforeCollection(RuntimeError):
        pass
    children = []
    async def stop_before_collection(plan, ledger):
        child = ledger.learner._cleanup_owner
        assert child.steps == 0 and child.protocol == 5
        children.append(child)
        assert providers and all(p._closed and p.pool.closed and p.pool.budget.used == 0 for p in providers)
        assert not plan.output.exists()
        raise BeforeCollection('actual generation0 validation reached collector boundary')
    parent = tmp_path / 'fresh-native-gen0'
    with pytest.raises(BeforeCollection):
        asyncio.run(train_card_instance_ppo(loaded, stop_before_collection, parent))
    assert children and all(child.process.returncode is not None and child.lease.released for child in children)
    assert not list((parent/'updates').glob('*.json'))
    assert all(p._executor_owner._closed and not p.pool._close_holds for p in providers)


def test_ordinary_paused_nonempty_factory_retains_actual_scan_pool_and_original_custody(monkeypatch):
    import json
    import os
    import time
    from dataclasses import replace
    from pathlib import Path

    native = pytest.importorskip('gf_native')
    path = os.environ.get('GFB_NATIVE_PACK_PARENT')
    if path is None or not hasattr(native.CardInstanceNativeSession, 'from_artifacts'):
        pytest.skip('preserved controlled one-update fixture/private candidate required')
    from gf_training import card_instance_source as source
    from gf_training import native_backend
    from gf_training.native_card_instance import _NativeGameInputs
    from gf_training.structured_lineage import _ChainScan
    parent = Path(path)
    original_state = (parent/'run.json').read_bytes()
    recorded = json.loads(original_state)
    assert recorded['schema_version'] == 4 and recorded['status'] == 'paused' and recorded['updates'] == 1
    scan = _ChainScan(directory=parent, max_owned_bytes=1024**3, deadline=time.monotonic()+90)
    prepare = source.preparation_config
    initialize = _NativeGameInputs.__init__
    # Historical diagnostic selection only; actual run/identity/backend bytes
    # stay unchanged. All scan/chain/proof/numeric/native checks remain real.
    monkeypatch.setattr(source, 'preparation_config', lambda value: replace(prepare(value), backend='rust'))
    monkeypatch.setattr(native_backend, 'group_pack_reader_settings', lambda config, native=None, **_: {})
    def diagnostic(actual, *args, **kwargs):
        kwargs['native'] = native
        initialize(actual, *args, **kwargs)
    monkeypatch.setattr(_NativeGameInputs, '__init__', diagnostic)
    with scan, source._open_game_parent_inputs(parent, require_complete=False,
            max_owned_bytes=1024**3, deadline=scan.deadline, _scan=scan) as (state, records, inputs):
        pool = scan._packed_pool
        provider = inputs._artifact_provider
        assert provider.pool is pool and inputs._shared_controls is scan
        assert provider._empty_controls is None and len(records) == 2
        assert set(provider._owners) == {str(parent/'collections/batch-000000')}
        assert inputs.plan['schema_version'] == 4 and inputs.summary['qualified'] is False
        assert state['identity'] == recorded['identity'] and state['status'] == 'paused'
        assert inputs.proof['custody'] == provider._owners[str(parent/'collections/batch-000000')].custody
    assert provider._closed and pool.closed and pool.budget.used == 0
    assert (parent/'run.json').read_bytes() == original_state


@pytest.mark.parametrize('target', ['base', 'controls'])
@pytest.mark.parametrize('after', [False, True])
def test_gen0_pool_and_control_admission_failure_retains_actual_scan_and_balances(tmp_path, monkeypatch, target, after):
    import time

    from gf_common.group_pack import SOURCE_BASE
    from gf_common.group_pack_game import storage_binding
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.structured_lineage import _ChainScan
    scan = _ChainScan(directory=tmp_path, max_owned_bytes=512 * 1024**2, deadline=time.monotonic()+30)
    scan.configure_packed({'schema_version': 4, 'storage_binding': storage_binding(),
        'effective_config': {'storage': {'budget_mib': 512}}})
    scan.charge(4096)
    acquire = StorageBudget.acquire
    failure = MemoryError(target + ' after=' + str(after))
    injected = []
    def fault(actual, amount, deadline, **kwargs):
        if amount == (SOURCE_BASE if target == 'base' else 4096) and not injected:
            injected.append(True)
            if after:
                acquire(actual, amount, deadline, **kwargs)
            raise failure
        return acquire(actual, amount, deadline, **kwargs)
    monkeypatch.setattr(StorageBudget, 'acquire', fault)
    with pytest.raises(MemoryError) as caught:
        scan.ensure_packed_pool()
    assert caught.value is failure and caught.value.cleanup_owner is scan
    assert scan._packed_pool is not None
    scan.close(recheck=False)
    assert scan._closed and scan._packed_pool.closed and scan._packed_pool.budget.used == 0


@pytest.mark.parametrize('stage', ['canonical', 'query'])
def test_provider_nonadmission_constructor_error_retires_all_recorded_stages(tmp_path, monkeypatch, stage):
    import gf_training.native_group_pack as module
    root, _ = packed(tmp_path)
    original = owner(root)
    before = original.pool.budget.used
    failure = RuntimeError(stage + ' constructor fault')
    if stage == 'canonical':
        monkeypatch.setattr(module, 'canonical_json', lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    else:
        activate = original.pool.activate
        class Connection:
            def __init__(self, actual):
                self.actual = actual
            def execute(self, sql, *args):
                if sql.startswith('SELECT group_id,length(pack)'):
                    raise failure
                return self.actual.execute(sql, *args)
            def close(self):
                self.actual.close()
        def fault(source):
            activate(source)
            source._originals.db = Connection(source._originals.db)
        monkeypatch.setattr(original.pool, 'activate', fault)
    with pytest.raises(RuntimeError) as caught:
        module.OriginalArtifactProvider((original,))
    assert caught.value is failure
    provider = caught.value._native_original_provider_cleanup_owner
    provider.close()
    assert original.pool.active is None and original.pool.budget.used == before
    original.close()
    assert original.pool.budget.used == 0


@pytest.mark.parametrize('refund_after_effect', [False, True])
def test_query_acquire_primary_retains_actual_provider_when_release_also_fails(tmp_path, monkeypatch, refund_after_effect):
    root, _ = packed(tmp_path)
    original = owner(root)
    budget = original.pool.budget
    before = budget.used
    acquire, release = budget.acquire, budget.release
    amount = 65536 + 1024 * len(original.artifacts._groups)
    primary = MemoryError('query acquire after effect')
    secondary = OSError('query release fault')
    completed_refunds = []
    def acquire_fault(size, deadline, **kwargs):
        acquire(size, deadline, **kwargs)
        if size == amount:
            raise primary
    def release_fault(size):
        if size == amount:
            if refund_after_effect:
                release(size)
                completed_refunds.append(size)
            raise secondary
        release(size)
    monkeypatch.setattr(budget, 'acquire', acquire_fault)
    monkeypatch.setattr(budget, 'release', release_fault)
    try:
        with pytest.raises(MemoryError) as caught:
            OriginalArtifactProvider((original,))
        assert caught.value is primary
        actual = caught.value._native_original_provider_cleanup_owner
        assert actual._failure is primary
        assert secondary._native_original_provider_cleanup_owner is actual
        assert budget.used == before if refund_after_effect else budget.used > before
        def supported_release(size):
            release(size)
            if size == amount:
                completed_refunds.append(size)
        monkeypatch.setattr(budget, 'release', supported_release)
        actual.close()
        assert budget.used == before and completed_refunds == [amount]
    finally:
        # Disposable RED cleanup through the actual provider reached by the
        # secondary error; this fallback is explicitly NOT production evidence.
        monkeypatch.setattr(budget, 'release', release)
        actual = getattr(primary, '_native_original_provider_cleanup_owner', None)
        if actual is None:
            actual = getattr(secondary, '_native_original_provider_cleanup_owner', None)
        if actual is not None:
            actual.close()
        original.close()
