"""Controller CONFIG/real-Store/I/O boundary evidence, never actual Source caps."""
from dataclasses import replace

import pytest


def test_shared_request_sequence_preserves_canonical_control_and_seed_slot_order(tmp_path):
    from gf_cli.hpo_campaign import campaign_requests, canonical_control, encode_settings
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    config = replace(loaded.config, trial_count=3)
    control = canonical_control(config)
    seeded = object()  # Scheduling DATA only, never a model/Source/owner.
    rows = campaign_requests(config, (seeded,))
    inside = encode_settings(config, control) is not None
    if inside:
        assert rows == ((0, 'control', control), (1, 'seed', seeded), (2, 'sample', None))
    else:
        assert rows == ((None, 'outside-control', control), (0, 'seed', seeded), (1, 'sample', None), (2, 'sample', None))
    with pytest.raises(ValueError): campaign_requests(config, (seeded,) * 4)


def test_native_study_cannot_verify_or_tell_complete_without_actual_current_control(tmp_path, monkeypatch):
    import time

    from gf_cli import hpo_evaluation
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=loaded.config.operating.parent_budget_bytes) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'CONFIG-only, NOT a verified parent'})
        study = CampaignStudy(store, loaded, manifest)
        proposal = study.propose(0, slot=None, kind='outside-control', requested=canonical_control(loaded.config))
        def forbidden(*args, **kwargs):
            raise AssertionError('native study trusted detached/cached score without actual context')
        monkeypatch.setattr(hpo_evaluation, 'verify_trial_evaluation', forbidden)
        with pytest.raises(ValueError, match='actual current'):
            study.complete(proposal, deadline=time.monotonic() + 10.)
        assert not store.exists('proposals/000000/tell.receipt.json')


def test_resume_factory_preserves_exact_old_cursor_and_sha_through_exec_preparation(tmp_path, monkeypatch):
    from gf_cli.rust_player_campaign import prepare_campaign
    from tests.test_rust_player_campaign_factory import _factory_loaded
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh') as first:
        old, old_sha = first.store.get('supervision/current.json')
        old_bytes = (directory / 'supervision/current.json').read_bytes()
    with prepare_campaign(loaded, directory, mode='resume') as resumed:
        current, sha = resumed.store.get('supervision/current.json')
        assert current == old and sha == old_sha
        assert (directory / 'supervision/current.json').read_bytes() == old_bytes
        assert resumed.launch['supervision_id'] != old['supervision_id']


def test_resume_factory_refuses_changed_old_cursor_before_mocked_exec(tmp_path, monkeypatch):
    from gf_cli.rust_player_campaign import prepare_campaign
    from gf_simulator_client import rust_player_companion
    from tests.test_rust_player_campaign_factory import _factory_loaded
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh'): pass
    with prepare_campaign(loaded, directory, mode='resume') as resumed:
        old, sha = resumed.store.get('supervision/current.json')
        resumed.store.update_supervision_cursor(dict(old, session_clock_sha256='f' * 64), expected_sha256=sha)
        monkeypatch.setattr(rust_player_companion, 'exec_supervisor', lambda *args: pytest.fail('exec after old cursor changed'))
        with pytest.raises(ValueError, match='cursor changed'): resumed.execute()


def test_controller_refuses_detached_control_before_store_or_history(tmp_path, monkeypatch):
    from gf_cli import rust_player_campaign as campaign
    from gf_cli.hpo_storage import CampaignStore
    monkeypatch.setattr(CampaignStore, 'borrow', lambda *args, **kwargs: pytest.fail('borrow before actual control'))
    with pytest.raises(ValueError, match='actual authenticated'):
        campaign.run_campaign(object(), {})


def test_hpo_public_native_route_execs_fixed_factory_without_legacy_async_dispatch(tmp_path, monkeypatch):
    from gf_cli import hpo, rust_player_campaign
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    monkeypatch.setattr(hpo, 'load_hpo_config', lambda path: loaded)
    def legacy(*args, **kwargs): pytest.fail('native6 entered legacy handler')
    monkeypatch.setattr(hpo, 'run_campaign', legacy)
    class Executed(BaseException): pass
    def execute(actual, directory, **kwargs):
        assert actual is loaded and directory == (tmp_path / 'campaign').resolve()
        assert kwargs == {'mode': 'fresh', 'seed_paths': (), 'progress': 'off'}
        raise Executed
    monkeypatch.setattr(rust_player_campaign, 'exec_campaign', execute)
    with pytest.raises(Executed):
        hpo.main(['run', '--config', 'unused.toml', '--campaign-dir', str(tmp_path / 'campaign'), '--mode', 'fresh', '--progress', 'off'])


def test_acknowledged_history_reader_holds_real_new_parent_and_preserves_cursor_on_failure(tmp_path, monkeypatch):
    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_storage import CampaignStore
    from gf_training.lineage import require_hpo_run_owner, run_lock
    parent = tmp_path / 'parents' / 'old'
    with run_lock(parent, create=True): pass
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        awaiting = {'run_path': str(parent), 'run_id': 'old'}  # Owned-reader boundary DATA only.
        store.put('sentinel.json', {'unchanged': True})
        before = store.get('sentinel.json')
        def reader(actual, name):
            assert actual is store and require_hpo_run_owner(parent).directory == parent
            raise ValueError('full historical read failed')
        monkeypatch.setattr(CampaignStore, 'read_acknowledged_supervision_disposition', reader)
        with pytest.raises(ValueError, match='historical read failed'):
            module._audit_terminal_history(store, 'terminal.json', awaiting, parent_root=parent.parent)
        assert store.get('sentinel.json') == before
        with pytest.raises(ValueError): require_hpo_run_owner(parent)


def test_schedule_uses_actual_optuna_request_and_only_dispatches_one_training(tmp_path, monkeypatch):
    import time

    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_campaign import CampaignStudy, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'scheduling DATA, never an actual control or Parent'})
        study = CampaignStudy(store, loaded, manifest)
        called = []
        def dispatch(control, actual_store, actual_study, proposal, *, learning_seed, materialized):
            # This unit boundary stops before all actual control/admit/Source APIs.
            assert control is None and actual_store is store and actual_study is study
            assert materialized.config.training.seeds == (learning_seed,)
            called.append((proposal.number, learning_seed))
        monkeypatch.setattr(module, 'dispatch_campaign_training', dispatch)
        assert module._drive_campaign(None, store, study, (), soft_finish=time.monotonic()+100.,
            hard_finish=time.monotonic()+1000.) is None
        assert called == [(0, loaded.config.learning_seeds[0])]
        assert store.exists('proposals/000000/proposal.json')
        assert not store.exists('proposals/000001/intent.json')


def test_campaign_host_propagates_only_independent_handler_output(monkeypatch):
    from types import SimpleNamespace

    from gf_cli import rust_player_campaign, rust_player_host
    control = SimpleNamespace(initial_operation={'operation_type': 'campaign-continue'},
        operation={'scope': 'dispatch boundary DATA'}, campaign_bootstrap={})
    monkeypatch.setattr(rust_player_campaign, 'run_campaign', lambda actual, operation: 'independent result text')
    assert rust_player_host._dispatch(control, 'host') == 'independent result text'


def test_soft_boundary_stops_before_any_proposal_or_training(tmp_path, monkeypatch):
    import time

    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_campaign import CampaignStudy, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'unit scheduling DATA only'})
        study = CampaignStudy(store, loaded, manifest)
        monkeypatch.setattr(module, '_finish_campaign_result', lambda *a, status, **k: status)
        assert module._drive_campaign(None, store, study, (), soft_finish=time.monotonic(), hard_finish=time.monotonic()+100.) == 'session_boundary'
        assert not any(name.startswith('proposals/') for name in store.records())


def test_closed_seed_candidate_is_data_bound_to_exact_proposal_and_awaiting(tmp_path):
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import _seed_candidate
    from gf_training.rust_player_evaluation import _SEED_FIELDS
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'candidate DATA, no actual Source'})
        study = CampaignStudy(store, loaded, manifest)
        proposal = study.propose(0, slot=None, kind='outside-control', requested=canonical_control(loaded.config))
        row = dict.fromkeys(_SEED_FIELDS, 'a' * 64)
        row.update(schema_version=2, kind='hpo-rust-seed-training', proposal_sha256=proposal.sha256,
            training_seed=17, generation=1, used_decisions=64, run_id='run', run_path=str(tmp_path / 'run'))
        store.put('proposals/000000/seeds/17/training.json', row)
        assert _seed_candidate(store, manifest, 0, 17) == row
        awaiting = {key: row[key] for key in ('run_id', 'run_path', 'config_sha256', 'runtime_binding_sha256')}
        awaiting['runtime_binding_sha256'] = 'b' * 64
        with pytest.raises(ValueError, match='candidate differs'):
            _seed_candidate(store, manifest, 0, 17, awaiting=awaiting)


@pytest.mark.parametrize('fault', ['audit', 'stale', 'none'])
def test_cursor_adoption_audits_before_exact_actual_store_cas(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace

    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_storage import CampaignStore
    cursor = {'schema_version': 1, 'kind': 'rust-player-campaign-supervision-v1', 'campaign_id': 'campaign',
        'supervision_id': 'a' * 32, 'session_clock_sha256': 'a' * 64, 'phase': 'control',
        'awaiting_path': None, 'awaiting_sha256': None, 'current_invocation_id': None}
    # Audit is the isolated external boundary; this issues no actual history or
    # control capability. CAS itself uses the genuine exclusive Store and bytes.
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        sha = store.update_supervision_cursor(cursor, expected_sha256=None)
        before = (store.directory / 'supervision/current.json').read_bytes()
        events = []
        def audit(control, actual, study, *, hard_finish):
            assert control is None and actual is store and store.get('supervision/current.json')[1] == sha
            events.append('full-audit-boundary')
            if fault == 'audit': raise ValueError('original full audit refused')
            if fault == 'stale':
                store.update_supervision_cursor(dict(cursor, session_clock_sha256='c' * 64), expected_sha256=sha)
        monkeypatch.setattr(module, '_verify_campaign_history', audit)
        kwargs = {'session': {'supervision_id': 'b' * 32}, 'bootstrap': {'session_clock_sha256': 'b' * 64}, 'hard_finish': 123.}
        study = SimpleNamespace(manifest={'campaign_id': 'campaign'})
        if fault == 'none':
            module._adopt_campaign_cursor(None, store, study, **kwargs)
            current, _ = store.get('supervision/current.json')
            assert current['supervision_id'] == 'b' * 32
        else:
            with pytest.raises(ValueError): module._adopt_campaign_cursor(None, store, study, **kwargs)
            if fault == 'audit': assert (store.directory / 'supervision/current.json').read_bytes() == before
            else: assert b'cccccccc' in (store.directory / 'supervision/current.json').read_bytes()
        assert events == ['full-audit-boundary']


@pytest.mark.parametrize('phase', ['training', 'evaluation'])
def test_resume_rejects_interrupted_seed_before_exec_and_preserves_old_cursor(tmp_path, monkeypatch, phase):
    from gf_cli.rust_player_campaign import prepare_campaign
    from tests.test_rust_player_campaign_factory import _factory_loaded
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    with prepare_campaign(loaded, directory, mode='fresh') as prepared:
        before = (directory / 'supervision/current.json').read_bytes()
        prepared.store.put(f'proposals/000000/seeds/17/{phase}.intent.json', {'scope': 'unclosed intent DATA'})
    with pytest.raises(ValueError, match='interrupted seed intent'), prepare_campaign(loaded, directory, mode='resume'):
        pytest.fail('unclosed historical intent entered prepared exec')
    assert (directory / 'supervision/current.json').read_bytes() == before


@pytest.mark.parametrize('evaluation_fails', [False, True])
def test_scheduler_closed_seed_audit_then_finished_eval_loop_before_next_training(tmp_path, monkeypatch, evaluation_fails):
    """Actual Store/Optuna orchestration; independent execution/audit are DATA doubles."""
    import asyncio
    import time
    from dataclasses import asdict

    from gf_cli import rust_player_campaign as module
    from gf_cli import rust_player_evaluation as evaluation
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_common.domain import digest
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    cfg = replace(loaded.config, learning_seeds=(17, 29))
    effective = asdict(cfg)
    loaded = replace(loaded, config=cfg, effective=effective, sha256=digest({'campaign': effective, 'base': loaded.base.sha256}))
    first, second = loaded.config.learning_seeds[:2]
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        manifest = initialize_campaign(store, loaded, {'scope': 'unqualified scheduling DATA'})
        study = CampaignStudy(store, loaded, manifest)
        proposal = study.propose(0, slot=None, kind='outside-control', requested=canonical_control(loaded.config))
        folder = f'proposals/000000/seeds/{first}'
        data = {'training_seed': first, 'scope': 'DATA double, not a closed Source or Parent'}
        store.put(folder + '/training.json', data)
        events = []
        def audit(*args, **kwargs):
            assert args[3] == proposal and args[4] == first
            events.append('full-audit-boundary')
            return data
        original = ValueError('uncertain evaluator retains actual nested owner')
        original.cleanup_owner = object()
        async def evaluate(*args, **kwargs):
            assert asyncio.get_running_loop().is_running()
            assert kwargs.get('completion') is None
            events.append('evaluation-boundary')
            if evaluation_fails: raise original
            store.put(folder + '/evaluation.json', {'scope': 'unqualified scheduling DATA'})
        def dispatch(*args, learning_seed, **kwargs):
            with pytest.raises(RuntimeError): asyncio.get_running_loop()
            assert learning_seed == second
            events.append('next-training-boundary')
        monkeypatch.setattr(evaluation, 'verify_rust_training_history', audit)
        monkeypatch.setattr(evaluation, 'evaluate_rust_seed', evaluate)
        monkeypatch.setattr(module, 'dispatch_campaign_training', dispatch)
        if evaluation_fails:
            with pytest.raises(ValueError) as caught:
                module._drive_campaign(None, store, study, (), soft_finish=time.monotonic()+100., hard_finish=time.monotonic()+1000.)
            assert caught.value is original and caught.value.cleanup_owner is original.cleanup_owner
            assert events == ['full-audit-boundary', 'evaluation-boundary']
            assert not store.exists('proposals/000000/tell.receipt.json')
            assert not store.exists(f'proposals/000000/seeds/{second}/training.json')
        else:
            assert module._drive_campaign(None, store, study, (), soft_finish=time.monotonic()+100., hard_finish=time.monotonic()+1000.) is None
            assert events == ['full-audit-boundary', 'evaluation-boundary', 'next-training-boundary']


def test_audit_deadline_never_renews_original_hard_finish(tmp_path, monkeypatch):
    from gf_cli import rust_player_campaign as module
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 100.)
    assert module._audit_deadline(loaded, 101.) == 101.
    assert module._audit_deadline(loaded, 99.) == 99.


@pytest.mark.parametrize('fault', ['score', 'partial', 'receipt-only', 'unacknowledged-fail'])
def test_readonly_pre_cursor_tell_verification_refuses_cached_or_partial_objective(tmp_path, monkeypatch, fault):
    from gf_cli import hpo_evaluation
    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        campaign = initialize_campaign(store, loaded, {'scope': 'tell DATA only, never qualification'})
        study = CampaignStudy(store, loaded, campaign)
        proposal = study.propose(0, slot=None, kind='outside-control', requested=canonical_control(loaded.config))
        row = {'schema_version': 1, 'kind': 'hpo-tell', 'proposal_sha256': proposal.sha256,
            'trial_number': proposal.trial_number, 'params': proposal.params, 'distributions': proposal.distributions,
            'state': 'COMPLETE', 'value': .9, 'evaluation_receipt_sha256': 'a' * 64, 'reason': None}
        if fault == 'unacknowledged-fail':
            row.update(state='FAIL', value=None, evaluation_receipt_sha256=None, reason='unacknowledged diagnostic')
        if fault != 'receipt-only': store.put('proposals/000000/tell.intent.json', row)
        if fault != 'partial': store.put('proposals/000000/tell.receipt.json', row)
        monkeypatch.setattr(hpo_evaluation, 'verify_trial_evaluation', lambda *a, **k: ({}, .4, 'a' * 64))
        before = store.records()
        with pytest.raises(ValueError):
            module._verify_closed_tells(None, store, study, failed_proposals=set(), hard_finish=100.)
        assert store.records() == before


def test_precursor_preparation_interruption_keeps_absence_through_resume_exec_boundary(tmp_path, monkeypatch):
    from gf_cli.hpo_storage import CampaignStore
    from gf_cli.rust_player_campaign import prepare_campaign
    from gf_simulator_client import rust_player_companion
    from tests.test_rust_player_campaign_factory import _factory_loaded
    loaded = _factory_loaded(tmp_path, monkeypatch)
    directory = tmp_path / 'campaign'
    actual_update = CampaignStore.update_supervision_cursor
    class InterruptedBeforeCursor(Exception): pass
    def interrupt(*args, **kwargs): raise InterruptedBeforeCursor('after actual campaign/config publication')
    monkeypatch.setattr(CampaignStore, 'update_supervision_cursor', interrupt)
    with pytest.raises(InterruptedBeforeCursor), prepare_campaign(loaded, directory, mode='fresh'): pass
    assert (directory / 'campaign.json').is_file() and not (directory / 'supervision/current.json').exists()
    monkeypatch.setattr(CampaignStore, 'update_supervision_cursor', actual_update)
    def final_exec(companion, launch):
        assert not (directory / 'supervision/current.json').exists()
        raise InterruptedBeforeCursor('mocked final exec, no actual process')
    monkeypatch.setattr(rust_player_companion, 'exec_supervisor', final_exec)
    with prepare_campaign(loaded, directory, mode='resume') as prepared:
        assert prepared._resume_cursor_sha is None and not prepared.store.exists('supervision/current.json')
        with pytest.raises(InterruptedBeforeCursor, match='mocked final exec'): prepared.execute()
    assert not (directory / 'supervision/current.json').exists()


@pytest.mark.parametrize('history', ['empty', 'unresolved-awaiting', 'foreign-awaiting', 'interrupted-seed', 'namespace-unresolved', 'stale'])
def test_absent_cursor_full_history_and_namespace_precede_real_expected_none_cas(tmp_path, monkeypatch, history):
    """Actual Store/history scan/namespace lock; authenticated context is DATA-isolated."""
    from gf_cli import rust_player_campaign as module
    from gf_cli.hpo_campaign import CampaignStudy, canonical_control, initialize_campaign
    from gf_cli.hpo_storage import CampaignStore
    from gf_training import hpo_gpu_lease as namespace
    from tests.test_rust_player_native_planning import loaded_campaign
    loaded = loaded_campaign(tmp_path)
    monkeypatch.setattr(namespace, '_GLOBAL_ROOT', tmp_path / 'namespace')
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=2**28) as store:
        campaign = initialize_campaign(store, loaded, {'scope': 'history DATA, no HostControl/Source authority'})
        study = CampaignStudy(store, loaded, campaign)
        session, bootstrap = {'supervision_id': 'b' * 32}, {'session_clock_sha256': 'b' * 64}
        events = []
        def context(control, actual_store, actual_study):
            assert control is None and actual_store is store and actual_study is study
            events.append('authenticated-context-boundary')
            return bootstrap, session, campaign
        monkeypatch.setattr(module, '_campaign_context', context)
        if history in ('unresolved-awaiting', 'foreign-awaiting'):
            proposal = study.propose(0, slot=None, kind='outside-control', requested=canonical_control(loaded.config))
            assert proposal is not None
            store.put('supervision/' + 'a' * 32 + '/awaiting/old.json',
                {'schema_version': 1, 'kind': 'rust-player-training-awaiting-v1',
                    'campaign_id': 'foreign' if history == 'foreign-awaiting' else campaign['campaign_id'],
                    'proposal_no': 0, 'learning_seed': loaded.config.learning_seeds[0],
                    'run_id': 'old-run', 'run_path': str(tmp_path / 'old-run'),
                    'config_path': str(tmp_path / 'old-run/config.json'), 'config_sha256': 'a' * 64,
                    'runtime_binding_sha256': 'a' * 64, 'operation_path': str(tmp_path / 'operation.json'),
                    'operation_sha256': 'a' * 64, 'supervision_id': 'a' * 32,
                    'session_clock_sha256': 'a' * 64, 'created_unix': 1.})
        elif history == 'interrupted-seed':
            store.put('proposals/000000/seeds/17/training.intent.json', {'scope': 'unclosed DATA'})
        actual_namespace = namespace.require_resolved_namespace
        if history == 'namespace-unresolved':
            monkeypatch.setattr(namespace, '_scan', lambda lock: {'existing': ({'scope': 'scan DATA'}, False)})
        def namespace_check():
            assert not store.exists('supervision/current.json')
            events.append('actual-namespace-check')
            actual_namespace()
            if history == 'stale':
                store.update_supervision_cursor(module._control_cursor(store, campaign,
                    {'supervision_id': 'c' * 32}, bootstrap), expected_sha256=None)
        monkeypatch.setattr(namespace, 'require_resolved_namespace', namespace_check)
        if history == 'empty':
            module._adopt_campaign_cursor(None, store, study, session=session, bootstrap=bootstrap, hard_finish=100.)
            current, _ = store.get('supervision/current.json')
            assert current['supervision_id'] == 'b' * 32
            assert events == ['authenticated-context-boundary', 'actual-namespace-check']
            assert list((namespace._GLOBAL_ROOT / 'jobs').iterdir()) == []
        else:
            match = {'unresolved-awaiting': 'complete closed seed candidate', 'foreign-awaiting': 'another campaign',
                'interrupted-seed': 'interrupted historical', 'namespace-unresolved': 'RECOVERY_REQUIRED', 'stale': 'compare-and-swap'}[history]
            with pytest.raises((ValueError, namespace.RecoveryRequired), match=match):
                module._adopt_campaign_cursor(None, store, study, session=session, bootstrap=bootstrap, hard_finish=100.)
            if history == 'stale':
                assert store.get('supervision/current.json')[0]['supervision_id'] == 'c' * 32
            else:
                assert not store.exists('supervision/current.json')
            assert events[0] == 'authenticated-context-boundary'
