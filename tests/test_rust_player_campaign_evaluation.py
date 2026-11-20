"""Fresh schema6 composition DATA tests; no jobs, gameplay or native authority."""
import asyncio
from dataclasses import asdict, replace

import pytest

from gf_common.domain import ContractError, RuntimePolicyBinding, digest


def campaign_inputs(tmp_path):
    from gf_cli.hpo_config import campaign_config
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.hpo_config import effective_settings
    from tests.test_rust_player_config import RUN_ID, campaign_raw
    loaded = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    training = {'run_id': RUN_ID, 'training_seed': 17, 'run_path': '/parents/' + RUN_ID,
        'generation': 2, 'generation_sha256': 'a' * 64, 'inference_sha256': 'b' * 64,
        'source_receipt_sha256': 'c' * 64}
    parent = {**training, 'policy_label': 'final-policy'}
    binding = RuntimePolicyBinding('mlp', {'version': '4', 'expected_manifest_digest': 'b' * 64})
    return loaded, settings, training, parent, binding


def test_schema6_fresh_plan_uses_original_schedule_and_unextended_durations(tmp_path):
    from gf_cli.hpo_evaluation import HpoEvaluationPlan
    from gf_cli.rust_player_evaluation import _fresh_evaluation_request
    loaded, settings, training, parent, binding = campaign_inputs(tmp_path)
    projected, plan = _fresh_evaluation_request(loaded, settings, {'campaign_id': 'campaign', 'conditions_sha256': 'd' * 64},
        training, parent, binding, output=tmp_path / 'eval', evaluation_id='evaluation', ticket='e' * 32,
        started=100., hard_finish=100. + loaded.config.run_wall_seconds)
    assert type(plan) is HpoEvaluationPlan
    assert projected.config.hpo_spec['schema_version'] == 6
    assert plan.wall_deadline == 100. + loaded.config.run_wall_seconds
    assert plan.work_deadline == plan.wall_deadline - projected.config.run.report_reserve_seconds
    assert plan.execution_deadline == 100. + projected.config.run.execution_seconds
    assert plan.admission_deadline == 100. + projected.config.run.admission_seconds
    assert projected.config.run.split == 'validation'
    assert tuple(plan.linkage['game_seeds']) == loaded.config.evaluation_game_seeds
    assert plan.linkage['left_parent'] == parent and plan.linkage['gpu_ticket'] == 'e' * 32
    assert len(plan.groups) == len(loaded.config.evaluation_game_seeds)
    assert plan.bindings[parent['policy_label']] == binding
    assert projected.sha256 == digest(asdict(projected.config))


@pytest.mark.parametrize('finish', [99., 101., float('nan'), float('inf')])
def test_schema6_fresh_plan_refuses_truncated_or_nonfinite_hard_finish(tmp_path, finish):
    from gf_cli.rust_player_evaluation import _fresh_evaluation_request
    loaded, settings, training, parent, binding = campaign_inputs(tmp_path)
    with pytest.raises((ContractError, TimeoutError)):
        _fresh_evaluation_request(loaded, settings, {'campaign_id': 'campaign', 'conditions_sha256': 'd' * 64},
            training, parent, binding, output=tmp_path / 'eval', evaluation_id='evaluation', ticket='e' * 32,
            started=100., hard_finish=finish)


def test_native_storage_stays_closed_without_exact_live_evaluation_data(tmp_path):
    from gf_cli.card_instance import _require_rust_evaluation_storage
    from gf_cli.config import require_storage_command
    from gf_cli.hpo_config import materialize_hpo_run
    loaded, settings, training, _, _ = campaign_inputs(tmp_path)
    projected = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=training['run_id'])
    config = replace(projected.config, run=replace(projected.config.run, split='validation'))
    with pytest.raises(ContractError, match='Rust'):
        require_storage_command(config, 'evaluate', standalone=False)
    with pytest.raises(ContractError, match='actual'):
        _require_rust_evaluation_storage(projected, {}, {}, {}, {})


def test_native_evaluation_rejects_json_control_before_job_or_storage(tmp_path, monkeypatch):
    import gf_training.hpo_gpu_lease as gpu
    from gf_cli.rust_player_evaluation import evaluate_rust_seed
    loaded, _, training, _, _ = campaign_inputs(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('untrusted input reached job admission')
    monkeypatch.setattr(gpu, 'admit_gpu_job', forbidden)
    with pytest.raises(ContractError, match='actual'):
        asyncio.run(evaluate_rust_seed({}, loaded, {}, {}, training, control={}, output=tmp_path / 'eval',
            soft_finish=10., hard_finish=20.))


@pytest.mark.parametrize('failure', [None, 'core', 'confirm', 'provider', 'data', 'outer', 'pool'])
def test_native_retirement_uses_core_then_same_job_then_provider_data_pool(failure, monkeypatch):
    """Order-only spies at original I/O boundaries; no spy grants job authority."""
    from types import SimpleNamespace

    from gf_cli.rust_player_evaluation import _RustEvaluationAttempt
    events = []
    marker = ValueError('original collector failure')
    def step(name):
        events.append(name)
        if name == failure: raise OSError(name)
    receipt = object()
    job = SimpleNamespace(confirm_termination=lambda: (step('confirm'), receipt)[1],
        require_confirmed_termination=lambda actual: actual, _receipt=None)
    observation = SimpleNamespace(require_core_retired=lambda **kw: (step('core'), kw['error'] is marker),
        release_artifacts=lambda: step('provider'))
    released = object()
    data = SimpleNamespace(reference=object(), release_reference=None, directory='/actual-boundary-spy')
    def close_data():
        step('data')
        data.release_reference = released
    data.close = close_data
    import gf_cli.hpo_evaluation as evaluation
    def outer(actual_error, actual_observation, actual_job, pins, releases):
        assert actual_error is marker and actual_observation is observation and actual_job is job
        assert releases[id(data.reference)] is released
        step('outer')
        return receipt
    monkeypatch.setattr(evaluation, '_confirmed_evaluation_cleanup', outer)
    attempt = _RustEvaluationAttempt()
    attempt.job, attempt.observation, attempt.data = job, observation, data
    attempt.pool = SimpleNamespace(close=lambda: step('pool'))
    attempt.collector_requested = True
    try:
        if failure is None:
            assert attempt.retire_inputs(marker) is receipt
            assert attempt.released is released
        else:
            with pytest.raises(OSError, match=failure): attempt.retire_inputs(marker)
        expected = ['core', 'confirm', 'provider', 'data', 'outer', 'pool']
        assert events == expected if failure is None else events == expected[:expected.index(failure) + 1]
    finally:
        # No real resources were created by these sequencing spies.
        attempt.parent_scope.close()


@pytest.mark.parametrize('stage', ['attempted', 'live-history', 'pending-inputs'])
def test_empty_job_cleanup_refuses_uncertain_stage_or_real_history(stage):
    from types import SimpleNamespace

    from gf_cli.rust_player_evaluation import _RustEvaluationAttempt
    events = []
    attempt = _RustEvaluationAttempt()
    attempt.admission_requested = True
    attempt.collector_requested = stage == 'attempted'
    attempt.inputs_pending = stage == 'pending-inputs'
    attempt.job = SimpleNamespace(_receipt=None, ledger=SimpleNamespace(_history=[object()] if stage == 'live-history' else []),
        confirm_termination=lambda: events.append('forbidden confirm'))
    with pytest.raises(ContractError): attempt.retire_inputs(ValueError('uncertain original'))
    assert not events


@pytest.mark.parametrize('reader', ['verify_rust_training_history', 'verify_rust_evaluation_history'])
def test_native_history_refuses_detached_control_before_parent_account(tmp_path, monkeypatch, reader):
    import gf_cli.rust_player_evaluation as module
    import gf_training.group_pack_pool as pools
    loaded, _, _, _, _ = campaign_inputs(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('detached history allocated a Source account')
    monkeypatch.setattr(pools, 'PackedArtifactPool', forbidden)
    with pytest.raises(ContractError, match='actual'):
        getattr(module, reader)({}, loaded, {}, {}, 17, control={}, deadline=1.)


def test_native_aggregate_forwards_same_current_control_one_seed_at_a_time(tmp_path, monkeypatch):
    """Isolated history-reader boundary; returned DATA is no audit authority."""
    import gf_cli.rust_player_evaluation as native
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_evaluation import aggregate_trial_evaluation
    from gf_cli.hpo_storage import CampaignStore
    from gf_evaluation.hpo import (
        GameIdentity,
        GameProjection,
        RunIdentity,
        SeedEvaluationProjection,
    )
    loaded, settings, _, _, _ = campaign_inputs(tmp_path)
    proposal = Proposal(0, 0, None, 'a' * 64, settings, '{}', '{}')
    control = object()  # Forwarded solely to the explicitly mocked reader boundary.
    calls = []
    def read(store, actual_loaded, campaign, actual_proposal, seed, *, control, deadline):
        assert actual_loaded is loaded and actual_proposal is proposal
        calls.append((seed, control, deadline))
        identities = tuple(GameIdentity(game, seat, f'episode-{seed}-{game}-{seat}', f'game-{seed}-{game}-{seat}')
            for game in loaded.config.evaluation_game_seeds for seat in (1, 2))
        run = RunIdentity(seed, f'run-{seed}', 'b' * 64, 'c' * 64, identities)
        projected = SeedEvaluationProjection(run, loaded.config.decision_budget, 'd' * 64,
            tuple(GameProjection.from_result(identity, result='draw') for identity in identities))
        return projected, 'e' * 64, 'f' * 64
    monkeypatch.setattr(native, 'verify_rust_evaluation_history', read, raising=False)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=8 * 1024**2) as store:
        result = aggregate_trial_evaluation(store, loaded, {'conditions_sha256': 'd' * 64}, proposal,
            deadline=123., control=control)
    assert calls == [(seed, control, 123.) for seed in loaded.config.learning_seeds]
    assert result['value'] == .5 and result['training_receipts'] == ['e' * 64] * len(calls)


@pytest.mark.parametrize('collector_fails', [False, True])
@pytest.mark.parametrize('close_fails', [False, True])
def test_native_eligible_publication_follows_physical_copy_and_positive_job_close(tmp_path, monkeypatch, close_fails, collector_fails):
    """Public ordering with explicit admission/Source boundary spies, not authority."""
    from types import SimpleNamespace

    import gf_cli.hpo_evaluation as legacy
    import gf_cli.rust_player_evaluation as native
    from gf_common.card_instance_collection import hpo_evaluation_groups
    from gf_common.rust_player import deadline_bits
    from gf_training.hpo_gpu_lease import TerminationReceipt
    from gf_training.hpo_retention import HpoReferenceReceipt
    from tests.test_rust_player_evaluation import metadata_data, seed_data
    loaded, _, _, _, _ = campaign_inputs(tmp_path)
    training, source = seed_data(), metadata_data()
    proposal = SimpleNamespace(number=0, sha256=training['proposal_sha256'])
    campaign = {'campaign_id': 'campaign', 'conditions_sha256': 'd' * 64}
    events, records = [], {}
    primary = ValueError('original collector failed')
    parent_names = ('run_id', 'run_path', 'training_seed', 'generation', 'generation_sha256',
        'inference_sha256', 'source_receipt_sha256', 'policy_label')
    parent = {**{name: training[name] for name in parent_names[:-1]}, 'policy_label': 'final-policy'}
    # Establish the ordinary deterministic schedule using pure request DATA.
    from gf_common.schedule import serialize_schedule
    link = {'schema_version': 1, 'kind': 'hpo-evaluation-linkage', 'campaign_id': 'campaign',
        'evaluation_id': 'evaluation', 'gpu_ticket': 'e' * 32, 'purpose': 'objective',
        'conditions_sha256': 'd' * 64, 'left_parent': parent, 'right_parent': None,
        'game_seeds': list(loaded.config.evaluation_game_seeds), 'learner_seats': {},
        'schedule_digest': '0' * 64, 'policy_registry_digest': 'f' * 64}
    groups = hpo_evaluation_groups(link)
    schedule_sha = digest(serialize_schedule(groups, 4))
    facts = ('campaign', 'evaluation', 'e' * 32, 'd' * 64,
        tuple(parent[name] for name in parent_names), 'f' * 64, schedule_sha)
    termination = TerminationReceipt(1, 'hpo-gpu-termination', 'e' * 32, 'a' * 64, 'b' * 64, 'boot', 1., ())
    def get(name):
        if name.endswith('/training.json'): return training, '1' * 64
        if name.endswith('/source-validation.json'): return source, 'c' * 64
        if name.endswith('/training-gpu-termination.json'): return {}, 'd' * 64
        if name.endswith('/evaluation.intent.json'):
            return {'config_sha256': '2' * 64, 'training_receipt_sha256': '1' * 64,
                'evaluation_id': 'evaluation', 'directory': str(tmp_path / 'eval')}, '3' * 64
        raise AssertionError(name)
    def put(name, value):
        events.append('physical' if name.endswith('/evaluation-gpu-termination.json') else
            'eligible' if name.endswith('/evaluation.json') else 'failed-physical')
        records[name] = value
        return '4' * 64
    store = SimpleNamespace(get=get, put=put)
    def proof(receipt):
        events.append('proof')
        assert receipt is termination
        return receipt
    def close():
        events.append('job-close')
        if close_fails: raise OSError('actual job close uncertainty')
    job = SimpleNamespace(require_confirmed_termination=proof, close=close,
        _admission={'context': {'evaluation_id': 'evaluation'}})
    async def run(scope, *args, **kwargs):
        scope.job = job
        scope.pool = SimpleNamespace(deadline=1000.)
        scope.acquired = HpoReferenceReceipt({}, '5' * 64, 'acquire')
        scope.released = HpoReferenceReceipt({}, '6' * 64, 'release')
        return (None, primary) if collector_fails else ((facts, str(tmp_path / 'eval'), '2' * 64), None)
    def retired(scope, error):
        events.append('retired-inputs')
        scope.termination = termination
        return termination
    def saved(directory, actual_loaded, actual_link, **options):
        events.append('saved-audit')
        assert '_cleanup_observation' not in options and options['extra_owned_bytes'] > 0
        episodes = [{'seed': group.seed, 'episode_id': spec.episode_id, 'game_id': spec.episode_id,
            'outcome': 'draw', 'decisions': 0} for group in groups for spec in group.episodes]
        return {'run_id': 'evaluation', 'config_hash': '2' * 64, 'linkage': actual_link, 'episodes': episodes}, {
            'manifest_sha256': '7' * 64, 'source_set_sha256': '8' * 64}
    monkeypatch.setattr(native, '_current_context', lambda *args: {})
    monkeypatch.setattr(native, '_actual_campaign_clock', lambda control: {
        'soft_dispatch_finish_bits': deadline_bits(100.), 'hard_session_finish_bits': deadline_bits(1000.)})
    monkeypatch.setattr(native, '_run_rust_evaluation', run)
    monkeypatch.setattr(native._RustEvaluationAttempt, 'retire_inputs', retired)
    monkeypatch.setattr(legacy, 'verify_saved_evaluation', saved)
    if collector_fails:
        with pytest.raises(ValueError) as failed:
            asyncio.run(native.evaluate_rust_seed(store, loaded, campaign, proposal, training, control=object(),
                output=tmp_path / 'eval', soft_finish=100., hard_finish=1000.))
        assert failed.value is primary and 'eligible' not in events and 'saved-audit' not in events
        assert primary.hpo_gpu_termination_receipt is termination
    elif close_fails:
        with pytest.raises(OSError, match='job close uncertainty'):
            asyncio.run(native.evaluate_rust_seed(store, loaded, campaign, proposal, training, control=object(),
                output=tmp_path / 'eval', soft_finish=100., hard_finish=1000.))
        assert 'eligible' not in events
    else:
        receipt = asyncio.run(native.evaluate_rust_seed(store, loaded, campaign, proposal, training, control=object(),
            output=tmp_path / 'eval', soft_finish=100., hard_finish=1000.))
        assert receipt['schema_version'] == 1 and receipt['pin_release_sha256'] == '6' * 64
        assert events.index('physical') < events.index('job-close') < events.index('eligible')


def test_receiving_plan_dies_before_its_actual_source_phase_coupon_refund(tmp_path, monkeypatch):
    """Real pool/coupon/plan lifetime; only original job-proof boundary is a spy."""
    import time
    import weakref
    from types import SimpleNamespace

    import gf_cli.hpo_evaluation as legacy
    from gf_cli.rust_player_evaluation import _fresh_evaluation_request, _RustEvaluationAttempt
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.domain import canonical
    from gf_training.group_pack_pool import PackedArtifactPool
    loaded, settings, training, parent, binding = campaign_inputs(tmp_path)
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 8 * 1024**2)
    baseline = pool.budget.used
    def setup():
        projected, plan = _fresh_evaluation_request(loaded, settings,
            {'campaign_id': 'campaign', 'conditions_sha256': 'd' * 64}, training, parent, binding,
            output=tmp_path / 'eval', evaluation_id='evaluation', ticket='e' * 32,
            started=100., hard_finish=100. + loaded.config.run_wall_seconds)
        quota = 32 * (len(canonical(projected.effective).encode())
            + sum(len(canonical(asdict(value)).encode()) for value in plan.bindings.values())) + 131072
        attempt = _RustEvaluationAttempt()
        attempt.pool, attempt.parent_budget = pool, phase
        attempt.reserve_config_metadata(projected)
        attempt.reserve_binding_metadata(binding)
        attempt.job = SimpleNamespace(_receipt=None, confirm_termination=lambda: 'existing-proof')
        attempt.observation = SimpleNamespace(original_plan=plan, projected=projected,
            require_core_retired=lambda **kw: None, release_artifacts=lambda: None)
        attempt.data = SimpleNamespace(reference=object(), release_reference=object(), close=lambda: None,
            directory=tmp_path / 'parent')
        attempt.collector_requested = True
        return attempt, weakref.ref(plan), weakref.ref(projected), quota
    attempt, plan_ref, config_ref, quota = setup()
    monkeypatch.setattr(legacy, '_confirmed_evaluation_cleanup', lambda *args: 'existing-proof')
    original = attempt.metadata_charge.close
    binding_original = attempt.binding_charge.close
    config_quota = attempt.metadata_charge.size
    def binding_refund():
        assert plan_ref() is None and config_ref() is None
        assert phase.used == quota and pool.budget.used == baseline + quota
        binding_original()
    def refund():
        assert plan_ref() is None and config_ref() is None
        assert phase.used == config_quota and pool.budget.used == baseline + config_quota
        original()
    attempt.binding_charge.close = binding_refund
    attempt.metadata_charge.close = refund
    try:
        attempt.retire_inputs(ValueError('stopped test boundary'))
        assert phase.used == 0 and pool.closed and pool.budget.used == 0
    finally:
        attempt.observation = None
        if attempt.binding_charge is not None: binding_original()
        if attempt.metadata_charge is not None: original()
        pool.close()


def test_native_training_history_read_error_keeps_original_audit_before_cleanup(tmp_path, monkeypatch):
    """Actual Store/Proposal/config; closed-audit factory is an explicit spy."""
    from types import SimpleNamespace

    import gf_cli.rust_player_evaluation as native
    from gf_cli.hpo_campaign import Proposal
    from gf_cli.hpo_evaluation import verify_training_receipt
    from gf_cli.hpo_storage import CampaignStore
    loaded, settings, _, _, _ = campaign_inputs(tmp_path)
    proposal = Proposal(0, 0, None, 'a' * 64, settings, '{}', '{}')
    primary = ContractError('original immutable training receipt read failed')
    closes = []
    def close():
        closes.append('close')
        raise OSError('unknown actual audit close')
    audit = SimpleNamespace(close=close, _cleanup_children=[])
    monkeypatch.setattr(native, 'open_closed_rust_parent', lambda *args, **kwargs: audit)
    with CampaignStore(tmp_path / 'campaign', mode='fresh', max_owned_bytes=8 * 1024**2) as store:
        def fail(name): raise primary
        monkeypatch.setattr(store, 'get', fail)
        with pytest.raises(ContractError) as failed:
            verify_training_receipt(store, loaded, {}, proposal, 17, deadline=1., control=None)
        assert failed.value is primary and primary.cleanup_owner is audit and not closes


def test_factored_model4_storage_preserves_legacy_validation_and_role_refusal(tmp_path):
    from gf_cli.config import require_storage_command
    from gf_cli.hpo_config import campaign_config, materialize_hpo_run
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
    from gf_training.hpo_config import effective_settings
    from tests.test_hpo_config import campaign_raw
    from tests.test_rust_player_config import RUN_ID
    loaded = campaign_config(campaign_raw(), origin=tmp_path)
    settings = effective_settings(loaded.config.model, loaded.config.training, {}, layout=CORRECTED_CARD_INSTANCE_LAYOUT)
    run = materialize_hpo_run(loaded, settings, learning_seed=17, run_id=RUN_ID)
    validated = replace(run.config, run=replace(run.config.run, split='validation'))
    require_storage_command(validated, 'evaluate', standalone=False)
    with pytest.raises(ContractError, match='role'):
        require_storage_command(run.config, 'evaluate', standalone=False)
    with pytest.raises(ContractError, match='owned collector'):
        require_storage_command(validated, 'evaluate', standalone=True)



def test_known_empty_job_retires_inputs_without_fabricating_collector_proof(monkeypatch):
    """Positive order-only I/O spies; no actual GpuJob or Source is issued."""
    from types import SimpleNamespace

    import gf_cli.hpo_evaluation as legacy
    from gf_cli.rust_player_evaluation import _RustEvaluationAttempt
    events = []
    def forbidden(*args): raise AssertionError('no collector existed to issue core proof')
    monkeypatch.setattr(legacy, '_confirmed_evaluation_cleanup', forbidden)
    attempt = _RustEvaluationAttempt()
    attempt.admission_requested = True
    receipt = object()
    attempt.job = SimpleNamespace(_receipt=None, ledger=SimpleNamespace(_history=[]),
        confirm_termination=lambda: (events.append('empty-job-proof'), receipt)[1])
    attempt.observation = SimpleNamespace(_core_phase='unentered', release_artifacts=lambda: events.append('provider'))
    attempt.data = SimpleNamespace(reference=object(), release_reference=object(), close=lambda: events.append('data'),
        directory='explicit-order-spy')
    attempt.pool = SimpleNamespace(close=lambda: events.append('pool'))
    assert attempt.retire_inputs(ValueError('before collector call')) is receipt
    assert events == ['empty-job-proof', 'provider', 'data', 'pool'] and attempt.observation is None


def test_closed_import_sees_actual_config_metadata_charge_before_audit(tmp_path, monkeypatch):
    """Real pool/phase/run_lock; admission/import are explicit stopping boundaries."""
    import time
    import weakref
    from types import SimpleNamespace

    import gf_cli.rust_player_evaluation as native
    import gf_training.hpo_gpu_lease as gpu
    from gf_common.domain import canonical
    from gf_training.lineage import require_hpo_run_owner
    from gf_training.rust_player_completion import _clear_failure_frames
    loaded, settings, training, _, _ = campaign_inputs(tmp_path)
    directory = tmp_path / 'parent'
    directory.mkdir()
    training['run_path'] = str(directory)
    expected = ContractError('bounded stop at closed DATA import before qualification')
    seen = {}
    original_project = native._project_fresh_config
    def project(*args, **kwargs):
        value = original_project(*args, **kwargs)
        seen['quota'] = 32 * len(canonical(value.effective).encode()) + 131072
        seen['projected_ref'] = weakref.ref(value)
        return value
    def imported(*args, ownership, pool, parent_budget, **kwargs):
        assert require_hpo_run_owner(directory, ownership) is ownership
        seen['phase_used'], seen['source_used'], seen['source_base'] = parent_budget.used, pool.budget.used, pool._base_charge
        assert parent_budget.source is pool.budget
        raise expected
    monkeypatch.setattr(native, '_project_fresh_config', project)
    monkeypatch.setattr(native, 'open_closed_rust_parent', imported)
    monkeypatch.setattr(gpu, 'admit_gpu_job', lambda *args, **kwargs: SimpleNamespace(ticket='e' * 32))
    store = SimpleNamespace(exists=lambda name: False, get=lambda name: (training, 'f' * 64),
        put=lambda name, value: 'f' * 64)
    scope = native._RustEvaluationAttempt()
    started = time.monotonic()
    try:
        outcome, error = asyncio.run(native._run_rust_evaluation(scope, store, loaded, {'campaign_id': 'campaign'},
            SimpleNamespace(number=0, settings=settings), training, output=tmp_path / 'eval',
            soft_finish=started + 60, hard_finish=started + loaded.config.run_wall_seconds + 60, progress=None))
        assert outcome is None and error is expected
        assert seen['phase_used'] == seen['quota']
        assert seen['source_used'] == seen['source_base'] + seen['quota']
        assert seen['projected_ref']() is not None  # stopped import failure still owns its funded caller
    finally:
        _clear_failure_frames(scope)
        assert seen['projected_ref']() is None
        for name in ('binding_charge', 'metadata_charge'):
            coupon = getattr(scope, name, None)
            if coupon is not None: coupon.close()
        if scope.pool is not None: scope.pool.close()
        scope.parent_scope.close()


def test_config_then_binding_coupons_have_exact_original_total(tmp_path):
    import time

    from gf_cli.rust_player_evaluation import _project_fresh_config, _RustEvaluationAttempt
    from gf_common.artifacts import ArtifactPhaseAdmission
    from gf_common.domain import canonical
    from gf_training.group_pack_pool import PackedArtifactPool
    loaded, settings, training, _, binding = campaign_inputs(tmp_path)
    projected = _project_fresh_config(loaded, settings, training, output=tmp_path / 'eval',
        evaluation_id='evaluation', started=100., hard_finish=100. + loaded.config.run_wall_seconds)
    config_quota = 32 * len(canonical(projected.effective).encode()) + 131072
    binding_quota = 32 * sum(len(canonical(asdict(value)).encode()) for value in
        (binding, RuntimePolicyBinding('heuristic')))
    pool = PackedArtifactPool(max_owned_bytes=128 * 1024**2, deadline=time.monotonic() + 30)
    phase = ArtifactPhaseAdmission(pool.budget, 8 * 1024**2)
    baseline = pool.budget.used
    scope = _RustEvaluationAttempt()
    scope.pool, scope.parent_budget = pool, phase
    try:
        scope.reserve_config_metadata(projected)
        assert phase.used == config_quota and pool.budget.used == baseline + config_quota
        scope.reserve_binding_metadata(binding)
        assert phase.used == config_quota + binding_quota
        assert pool.budget.used == baseline + config_quota + binding_quota
        assert scope.metadata_charge.size == config_quota and scope.binding_charge.size == binding_quota
    finally:
        del projected, binding
        for name in ('binding_charge', 'metadata_charge'):
            coupon = getattr(scope, name, None)
            if coupon is not None: coupon.close()
        assert phase.used == 0
        pool.close()
