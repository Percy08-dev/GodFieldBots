"""Campaign training/terminal unit boundaries; no live cap or actual training."""
import fcntl
import os
from contextlib import ExitStack
from pathlib import Path

import pytest


def test_namespace_check_uses_actual_exclusive_lock_and_creates_no_dummy_job(tmp_path, monkeypatch):
    from gf_training import hpo_gpu_lease as namespace
    monkeypatch.setattr(namespace, '_GLOBAL_ROOT', tmp_path / 'namespace')
    namespace.require_resolved_namespace()
    assert list((tmp_path / 'namespace/jobs').iterdir()) == []
    assert not (tmp_path / 'namespace/current.json').exists()


@pytest.mark.parametrize('resolved', [False, True])
def test_namespace_check_preserves_existing_scan_resolution_without_admission(tmp_path, monkeypatch, resolved):
    from gf_training import hpo_gpu_lease as namespace
    root = tmp_path / 'namespace'
    monkeypatch.setattr(namespace, '_GLOBAL_ROOT', root)
    seen = []
    def scan(lock):
        assert lock.root == root
        with (root / 'lock').open('r+b') as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen.append(True)
        return {'existing-ticket': ({'scope': 'scan boundary DATA, not a job/cap'}, resolved)}
    monkeypatch.setattr(namespace, '_scan', scan)
    if resolved:
        namespace.require_resolved_namespace()
    else:
        with pytest.raises(namespace.RecoveryRequired): namespace.require_resolved_namespace()
    assert seen == [True] and list((root / 'jobs').iterdir()) == []


def test_consumed_campaign_scope_retires_actual_registered_parent_before_prepared_edge(tmp_path):
    from gf_training.lineage import _HELD_RUN_OWNERS, require_hpo_run_owner, run_lock
    from gf_training.rust_player_learning import _consumed_campaign_parent
    scope = ExitStack()
    owner = scope.enter_context(run_lock(tmp_path / 'parent', create=True))
    fd = _HELD_RUN_OWNERS[owner.directory].descriptor
    with _consumed_campaign_parent(scope, owner, owner.directory) as actual:
        assert actual is owner and require_hpo_run_owner(owner.directory, actual) is owner
        with (owner.directory / '.lock').open('r+b') as competitor, pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # This is the real physical scope-close edge before any Prepared call.
    with pytest.raises(OSError): os.fstat(fd)
    with pytest.raises(ValueError): require_hpo_run_owner(owner.directory, owner)
    with (owner.directory / '.lock').open('r+b') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scope.close()  # Idempotent cleanup, never a second owner/context.


def test_consumed_campaign_scope_refuses_equal_field_owner_copy_and_closes_real_scope(tmp_path):
    from dataclasses import replace

    from gf_training.lineage import _HELD_RUN_OWNERS, run_lock
    from gf_training.rust_player_learning import _consumed_campaign_parent
    scope = ExitStack()
    owner = scope.enter_context(run_lock(tmp_path / 'parent', create=True))
    fd = _HELD_RUN_OWNERS[owner.directory].descriptor
    with pytest.raises(ValueError), _consumed_campaign_parent(scope, replace(owner), owner.directory):
        raise AssertionError('copied metadata acquired parent authority')
    with pytest.raises(OSError): os.fstat(fd)


@pytest.mark.parametrize('fault', ['io', 'timeout', 'memory', 'buffer', 'writer', 'writer-close',
    'resource-owner', 'nested-io', 'interruption', 'incomplete-local'])
def test_application_failure_classification_refuses_known_uncertainty_without_authority(fault):
    from types import SimpleNamespace

    from gf_training.rust_player_learning import _clean_application_error
    writer = SimpleNamespace(failed=False, _uncertain_closes=[])
    health = dict.fromkeys(('actors_joined', 'service_joined', 'writers_joined', 'servers_joined',
        'buffers_released', 'gpu_fenced'), True)
    error = ValueError('application-only diagnostic')
    assert _clean_application_error(error, writer, health)
    if fault == 'io': error = OSError('I/O uncertain')
    elif fault == 'timeout': error = TimeoutError('expired')
    elif fault == 'memory': error = MemoryError('resource failure')
    elif fault == 'buffer': error = BufferError('live data alias')
    elif fault == 'writer': writer.failed = True
    elif fault == 'writer-close': writer._uncertain_closes = [object()]
    elif fault == 'resource-owner': error.cleanup_owner = object()
    elif fault == 'nested-io': error.__cause__ = OSError('original uncertain I/O')
    elif fault == 'interruption': error = KeyboardInterrupt()
    elif fault == 'incomplete-local': health['buffers_released'] = False
    assert not _clean_application_error(error, writer, health)


@pytest.mark.parametrize('kind', ['export-reservation', 'resource-limit', 'accelerator', 'out-of-memory'])
@pytest.mark.parametrize('nested', [False, True])
def test_r1_actual_resource_exception_types_are_fail_stop_without_device_work(kind, nested):
    from types import SimpleNamespace

    from gf_training.rust_player_learning import _clean_application_error
    writer = SimpleNamespace(failed=False, _uncertain_closes=[])
    health = dict.fromkeys(('actors_joined', 'service_joined', 'writers_joined', 'servers_joined',
        'buffers_released', 'gpu_fenced'), True)
    assert _clean_application_error(RuntimeError('ordinary clean application failure'), writer, health)
    if kind == 'export-reservation':
        from gf_models.rust_player_device import RustPlayerDeviceAccount
        from tests.test_rust_player_device import checkpoint_profile
        profile = checkpoint_profile()
        account = RustPlayerDeviceAccount(global_bytes=profile.fixed_bytes, learner_device='cuda:0',
            serving_device='cpu', learner_profile=profile, admitted_profiles=(profile,))
        with pytest.raises(ValueError, match='export phase exceeds allowance') as refused:
            account.begin_export()
        error = refused.value
        assert account._active is None and not hasattr(error, 'cleanup_owner')
    elif kind == 'resource-limit':
        from gf_training.preparation import ResourceLimit
        error = ResourceLimit('known refused resource')
    else:
        import torch
        # Pinned C++ exception classes only; constructing them performs NO CUDA
        # availability query, initialization, allocation or operation.
        cls = torch._C.AcceleratorError if kind == 'accelerator' else torch._C.OutOfMemoryError
        error = cls('typed resource failure; no device work')
    if nested:
        wrapper = RuntimeError('ordinary-looking apply error')
        wrapper.__context__ = error
        error = wrapper
    assert not _clean_application_error(error, writer, health)


def test_r1_failure_classification_does_not_import_torch_for_clean_metadata_case():
    import subprocess
    import sys
    code = '''
import sys
from types import SimpleNamespace
from gf_training.rust_player_learning import _clean_application_error
w=SimpleNamespace(failed=False,_uncertain_closes=[])
h=dict.fromkeys(('actors_joined','service_joined','writers_joined','servers_joined','buffers_released','gpu_fenced'),True)
assert _clean_application_error(RuntimeError('ordinary clean app error'),w,h)
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('kinds', [('saved',), ('restored',), ('saved', 'restored')])
def test_r1_collect_accounting_counts_only_dedicated_saved_charges_and_keeps_shared_restored_funding(tmp_path, kinds):
    """Inert exact-type representations used ONLY by diagnostic projection.

    No issuer/require/save/restore/qualification method is called on them.
    """
    import asyncio
    import time
    from types import SimpleNamespace

    from gf_training.group_pack_pool import PackedArtifactPool
    from gf_training.rust_player_learning import RustGameOperations
    from gf_training.rust_player_registry import RustSavedPolicy
    from gf_training.rust_player_restore import RustRestoredPolicy
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    dedicated, shared = [], []
    policies = {}
    for index, kind in enumerate(kinds):
        cls = RustSavedPolicy if kind == 'saved' else RustRestoredPolicy
        policy = cls.__new__(cls)
        policy.inference = SimpleNamespace(size=100 + index)
        if kind == 'saved':
            policy._charges = [pool.budget.reserve(64), pool.budget.reserve(96)]
            dedicated.extend(policy._charges)
        else:
            shared.append(pool.budget.reserve(128))  # Existing audit/registry shared metadata funding.
            assert not hasattr(policy, '_charges')
        policies[index] = policy
    registry = SimpleNamespace(policies=policies)
    collection = SimpleNamespace(writer=SimpleNamespace(_workspace_lease=None))
    class Collector:
        _inventory = _source_control_charge = None
        async def collect(self, plan, actual):
            assert actual is registry
            return collection
    written = []
    ops = RustGameOperations.__new__(RustGameOperations)
    ops.registry, ops.collector, ops.pool = registry, Collector(), pool
    ops._ownership = SimpleNamespace(directory=tmp_path)
    ops.writer = SimpleNamespace(write=lambda path, value, **kwargs: written.append(value))
    plan = SimpleNamespace(linkage={'batch_id': 'batch'}, execution_deadline=pool.deadline)
    used_before = pool.budget.used
    try:
        assert asyncio.run(ops.collect(plan, registry)) is collection
        accounting = written[0]
        assert accounting['saved_metadata_bytes'] == sum(charge.size for charge in dedicated)
        assert accounting['source_used'] == used_before == pool.budget.used
        assert accounting['inference_bank_data_bytes'] == sum(100 + index for index in range(len(kinds)))
        assert all(not charge.closed for charge in (*dedicated, *shared))
    finally:
        for charge in (*dedicated, *shared): charge.close()
        pool.close()


def test_campaign_training_scope_requires_actual_control_not_data_or_callback_authority(tmp_path):
    from gf_training.lineage import run_lock
    from gf_training.rust_player_learning import run_hpo
    scope = ExitStack()
    owner = scope.enter_context(run_lock(tmp_path / 'parent', create=True))
    try:
        with pytest.raises(ValueError):
            run_hpo(None, {}, loaded=None, _campaign_parent_scope=scope, _campaign_owner=owner,
                _campaign_admitted=lambda *args: (_ for _ in ()).throw(AssertionError('no actual admission')))
    finally:
        scope.close()


def test_source_actual_admission_notification_precedes_learner_and_consumed_scope_precedes_prepared():
    """Source constraint plus real scope behavior above; not actual Native proof."""
    import ast
    import inspect

    from gf_training import rust_player_learning as module
    source = inspect.getsource(module._run_local)
    tree = ast.parse(source)
    arguments = [arg.arg for arg in tree.body[0].args.kwonlyargs]
    assert {'_campaign_parent_scope', '_campaign_owner', '_campaign_admitted'} <= set(arguments)
    assert source.index('control.admit(') < source.index('_campaign_admitted(admission)') < source.index('ops = RustHpoOperations(')
    assert source.index('_consumed_campaign_parent(') < source.index('with scope as outgoing:') < source.index('prepared = ops.prepare()')


def test_training_publication_binds_actual_config_clock_awaiting_and_cursor_under_one_owner(tmp_path):
    import time

    from gf_cli.hpo_campaign import canonical_control, initialize_campaign
    from gf_cli.hpo_config import materialize_hpo_run
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import _publish_campaign_training, campaign_session_clock
    from gf_common.rust_player import deadline_bits
    from gf_training.lineage import run_lock
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    binding = {'boot_id': 'boot', 'monotonic_before_bits': deadline_bits(time.monotonic()),
        'monotonic_after_bits': deadline_bits(time.monotonic()), 'unix_anchor_bits': deadline_bits(time.time())}
    session = campaign_session_clock(loaded, supervision_id='1' * 32, clock_binding=binding)
    run_id = '00000000-0000-0000-0000-000000000017'
    materialized = materialize_hpo_run(loaded, canonical_control(loaded.config), learning_seed=17, run_id=run_id)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        campaign = initialize_campaign(store, loaded, {'scope': 'unqualified CONFIG metadata'})
        clock_sha = store.put('supervision/' + '1' * 32 + '/session-clock.json', session)
        cursor = {'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
            'campaign_id': campaign['campaign_id'], 'supervision_id': '1' * 32, 'session_clock_sha256': clock_sha,
            'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None, 'phase': 'control'}
        store.update_supervision_cursor(cursor, expected_sha256=None)
        context = {'campaign_id': campaign['campaign_id'], 'operation_id': '2' * 32, 'kind': 'training',
            'run_id': run_id, 'evaluation_id': None}
        with run_lock(tmp_path / run_id, create=True) as owner:
            packet = _publish_campaign_training(store, owner, materialized=materialized,
                context=context, proposal_no=0, learning_seed=17, session=session,
                clock_binding=binding, runtime_binding_sha256='a' * 64)
            op = packet['operation']
            assert op['config_sha256'] == materialized.sha256
            assert packet['loaded'].inputs == ({'path': op['config_path'], 'sha256': materialized.sha256},)
            assert Path(op['config_path']).read_bytes()
            actual, _sha = store.get('supervision/current.json')
            assert actual['phase'] == 'awaiting-host' and actual['current_invocation_id'] is None
            assert actual['awaiting_path'] == str(packet['awaiting_path'])
            assert packet['awaiting']['operation_sha256'] == packet['operation_sha256']
            assert packet['clock']['operation_sha256'] == packet['operation_sha256']
            assert packet['clock']['deadlines'] == op['deadlines']


def test_terminal_records_use_actual_optuna_fail_and_immutable_existing_shapes_under_parent_custody(tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    from gf_cli.hpo_campaign import CampaignStudy, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import _terminal_records
    from gf_common.rust_player import validate_campaign_disposition
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as layout
    from gf_training.hpo_config import effective_settings
    from gf_training.lineage import run_lock
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        campaign = initialize_campaign(store, loaded, {'scope': 'unqualified CONFIG/terminal DATA unit'})
        study = CampaignStudy(store, loaded, campaign)
        requested = effective_settings(loaded.config.model, loaded.config.training, {}, layout=layout)
        proposal = study.propose(0, slot=0, kind='control', requested=requested)
        root = store.directory / 'supervision' / ('1' * 32)
        # Evidence fields are DATA only; no Native ACK, cap or qualification is fabricated.
        awaiting = {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
            'campaign_id': campaign['campaign_id'], 'proposal_no': 0, 'learning_seed': 17,
            'supervision_id': '1' * 32, 'session_clock_sha256': 'a' * 64,
            'operation_path': str(root / 'operation-op.json'), 'operation_sha256': 'b' * 64,
            'run_id': 'run', 'run_path': str(tmp_path / 'parent'), 'config_path': str(tmp_path / 'parent/config.json'),
            'config_sha256': 'c' * 64, 'runtime_binding_sha256': 'd' * 64, 'created_unix': 1.}
        awaiting_path = root / 'awaiting/op.json'
        awaiting_sha = store.put(str(awaiting_path.relative_to(store.directory)), awaiting)
        evidence = {key + suffix: str(root / (key + '.json')) if suffix == '_path' else 'c' * 64
            for key in ('failure', 'exit', 'physical') for suffix in ('_path', '_sha256')}
        events, actual_put = [], store.put
        def put(name, value):
            events.append(name)
            return actual_put(name, value)
        monkeypatch.setattr(store, 'put', put)
        with run_lock(tmp_path / 'parent', create=True) as owner:
            path = _terminal_records(store, study, proposal, owner=owner, awaiting=awaiting,
                awaiting_path=awaiting_path, awaiting_sha256=awaiting_sha, operation_id='op',
                invocation_id='2' * 32, classification='training_failed', evidence=evidence,
                reason='Non-private unit application failure')
            row, _sha = store.get(str(path.relative_to(store.directory)))
            assert validate_campaign_disposition(row) is row
            assert row['proposal_sha256'] == proposal.sha256
            intent, _ = store.get('proposals/000000/tell.intent.json')
            receipt, receipt_sha = store.get('proposals/000000/tell.receipt.json')
            assert intent == receipt and receipt['state'] == 'FAIL' and receipt['value'] is None
            assert row['tell_receipt_sha256'] == receipt_sha
        assert events[0] == 'proposals/000000/failure.json'
        assert events[-1].endswith('/terminal/op.json')
        with study.study(0) as actual_study:
            assert actual_study.get_trials()[proposal.trial_number].state.name == 'FAIL'


@pytest.mark.parametrize('fault', [None, 'ack', 'close', 'release', 'cursor'])
def test_terminal_ack_close_release_cursor_order_and_failure_retention(tmp_path, monkeypatch, fault):
    """Native RPC is an explicit I/O boundary double, never a live HostControl."""
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import _finish_terminal
    events = []
    file = (tmp_path / 'actual-parent-data').open('w+b')
    class Retire:
        def close(self):
            events.append('borrower-close')
            if fault == 'close': raise OSError('actual data close uncertain')
            file.close()
    retire = Retire()
    class NativeBoundary:
        def campaign_disposition(self, path):
            os.fstat(file.fileno())  # Last owner must survive through actual ACK boundary.
            events.append('ack')
            if fault == 'ack': raise OSError('ACK unknown')
            return {'scope': 'unit boundary DATA, NOT a genuine Native ACK'}
        def release_parent(self):
            assert file.closed
            events.append('release')
            if fault == 'release': raise OSError('release unknown')
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**24) as store:
        cursor = {'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1',
            'campaign_id': 'campaign', 'supervision_id': '1' * 32, 'session_clock_sha256': 'a' * 64,
            'awaiting_path': str(store.directory / 'supervision/awaiting.json'), 'awaiting_sha256': 'b' * 64,
            'current_invocation_id': '2' * 32, 'phase': 'awaiting-cold'}
        sha = store.update_supervision_cursor(cursor, expected_sha256=None)
        actual_update = store.update_supervision_cursor
        def update(value, **kwargs):
            events.append('cursor')
            if fault == 'cursor': raise OSError('cursor durability unknown')
            return actual_update(value, **kwargs)
        monkeypatch.setattr(store, 'update_supervision_cursor', update)
        try:
            if fault:
                with pytest.raises(OSError) as failed:
                    _finish_terminal(NativeBoundary(), store, disposition_path=tmp_path / 'unused-disposition',
                        retire_owner=retire, cursor=cursor, cursor_sha256=sha)
                assert failed.value.rust_player_terminal_owner is retire
            else:
                _finish_terminal(NativeBoundary(), store, disposition_path=tmp_path / 'unused-disposition',
                    retire_owner=retire, cursor=cursor, cursor_sha256=sha)
                current, _ = store.get('supervision/current.json')
                assert current['phase'] == 'control' and current['awaiting_path'] is None
        finally:
            file.close()
    expected = ['ack', 'borrower-close', 'release', 'cursor']
    assert events == expected[:{'ack': 1, 'close': 2, 'release': 3}.get(fault, 4)]


@pytest.mark.parametrize('entry', ['dispatch', 'failed', 'incomplete', 'complete'])
def test_campaign_mechanics_refuse_metadata_instead_of_actual_authority(entry):
    from gf_cli import rust_player_campaign as module
    if entry == 'dispatch':
        call = lambda: module.dispatch_campaign_training(None, None, None, None, learning_seed=17, materialized=None)
    elif entry == 'failed': call = lambda: module.settle_failed_campaign_training(None, None, None)
    elif entry == 'incomplete': call = lambda: module.settle_incomplete_campaign_training(None, None, None, {})
    else: call = lambda: module.complete_campaign_control(None, None, None, status='no_work')
    with pytest.raises(ValueError, match='actual campaign'): call()
