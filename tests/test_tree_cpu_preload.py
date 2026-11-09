"""Lightweight dispatch/ordering doubles only; original Tree tests own Source proof."""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import pytest

import gf_cli
from gf_common.domain import ContractError, GroupResult, RuntimePolicyBinding
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan, validation_groups
from gf_training.structured_tree_validation import FinalValidationContext
from tests.cpu_preload import CURRENT_TEMPLATE, CpuTemplate
from tests.structured_ppo_support import component_http, configuration


@pytest.fixture
def envelope(tmp_path):
    loaded = configuration(tmp_path)
    config = replace(loaded.config, run=replace(loaded.config.run, split='validation'))
    loaded = replace(loaded, config=config)
    ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
    plan = CollectionPlan(validation_groups(17, 'dispatch-model', actions_version=2)[:1],
        {'dispatch-model': RuntimePolicyBinding('mlp'), 'random-structured-v1': RuntimePolicyBinding('random')},
        tmp_path / 'dispatch-only', {}, time.monotonic() + 30, time.monotonic() + 60,
        'ppo_validation', time.monotonic())
    # Deliberately NOT a live Source. Tests either use the real rejection or
    # replace validation to inspect dispatch; none call accept_collection.
    context = object.__new__(FinalValidationContext)
    context.projected, context.plan, context.ledger = loaded, plan, ledger
    context._closed, context._source = True, None
    return loaded, plan, ledger, context


@pytest.fixture
def bound_owner(tmp_path, monkeypatch):
    import tests.cpu_preload
    monkeypatch.setattr(tests.cpu_preload, 'supported', lambda: True)
    owner = CpuTemplate(tmp_path / 'unused-template')
    token = CURRENT_TEMPLATE.set(owner)
    try:
        yield owner
        assert owner.process is None
    finally:
        CURRENT_TEMPLATE.reset(token)
        asyncio.run(owner.close())


@pytest.mark.parametrize('mutation', ['loaded', 'arbitrary', 'pool', 'native', 'cuda', 'reader', 'split'])
def test_validation_helper_rejects_ineligible_envelope_without_warmup(envelope, bound_owner, monkeypatch, mutation):
    loaded, _, _, context = envelope
    options = {'validation_context': context, 'preload_cpu': True}
    if mutation == 'loaded':
        loaded = replace(loaded)
    elif mutation == 'arbitrary':
        options['validation_context'] = object()
    elif mutation == 'pool':
        options['pool_cpu'] = True
    elif mutation == 'native':
        options['source_ops'] = object()
    else:
        config = loaded.config
        if mutation == 'cuda':
            config = replace(config, model=replace(config.model, inference=replace(config.model.inference, device='cuda:0')))
        elif mutation == 'reader':
            config = replace(config, preprocessing=replace(config.preprocessing, backend='rust'))
        else:
            config = replace(config, run=replace(config.run, split='train'))
        loaded = context.projected = replace(loaded, config=config)
    with pytest.raises(ValueError, match='CPU preload'):
        component_http(monkeypatch, loaded, **options)


@pytest.mark.parametrize('mismatch', ['stale', 'plan', 'ledger', 'loaded'])
def test_validation_helper_checks_actual_envelope_before_warmup(envelope, bound_owner, monkeypatch, mismatch):
    loaded, plan, ledger, context = envelope
    collector, *_ = component_http(monkeypatch, loaded, validation_context=context, preload_cpu=True)
    if mismatch == 'plan':
        plan = replace(plan)
    elif mismatch == 'ledger':
        ledger = ReservationLedger()
    elif mismatch == 'loaded':
        context.projected = replace(loaded)
    with pytest.raises((ValueError, ContractError), match='exact final validation|live actor-only'):
        asyncio.run(collector(plan, ledger))


@pytest.mark.parametrize('failure', ['none', 'before_warmup', 'health', 'after_callback'])
def test_helper_validation_and_health_ordering(envelope, bound_owner, monkeypatch, failure):
    loaded, plan, ledger, context = envelope
    calls = []
    original = RuntimeError('injected dispatch failure')
    def validate(self, selected, same, config, *, deadline):
        assert self is context and selected is plan and same is ledger and config is loaded.config
        assert deadline == plan.wall_deadline - loaded.config.run.report_reserve_seconds
        calls.append('validate')
        if failure == 'before_warmup' or (failure == 'after_callback' and calls.count('validate') == 2):
            raise original
    monkeypatch.setattr(FinalValidationContext, 'validate_collection', validate)
    monkeypatch.setattr(bound_owner, 'start', lambda deadline: calls.append('start'))
    def health(deadline):
        calls.append('health')
        if failure == 'health':
            raise original
        return {'healthy': True, 'unretired_tickets': 0}
    monkeypatch.setattr(bound_owner, 'health', health)
    async def experiment(*args, _pytest_worker_finalizer, **kwargs):
        calls.append('experiment')
        receipt = await _pytest_worker_finalizer()
        assert receipt['mode'] == 'pytest-fresh-forked-cpu-v1' and receipt['closed']
        calls.append('returned')
        return plan.output, {'comparison_valid': False}
    monkeypatch.setattr(gf_cli, '_run_experiment', experiment)
    collector, *_ = component_http(monkeypatch, loaded, validation_context=context, preload_cpu=True)
    if failure == 'none':
        asyncio.run(collector(plan, ledger))
        assert calls == ['validate', 'start', 'experiment', 'health', 'validate', 'returned']
    else:
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(collector(plan, ledger))
        assert raised.value is original and 'returned' not in calls
        if failure == 'before_warmup':
            assert calls == ['validate']


@pytest.mark.parametrize('mutation', ['loaded', 'plan', 'ledger', 'arbitrary', 'pool_route', 'native', 'cuda', 'reader', 'command'])
def test_cli_validation_finalizer_requires_exact_composition(envelope, monkeypatch, mutation):
    loaded, plan, ledger, context = envelope
    options = {'_structured_factory': object(), '_structured_validation_context': context}
    command = 'collect'
    if mutation == 'loaded':
        loaded = replace(loaded)
    elif mutation == 'plan':
        plan = replace(plan)
    elif mutation == 'ledger':
        ledger = ReservationLedger()
    elif mutation == 'arbitrary':
        options['_structured_validation_context'] = object()
    elif mutation == 'pool_route':
        # A generic validation collection has no live context authority.
        options['_structured_validation_context'] = None
    elif mutation == 'native':
        options['_structured_source_ops'] = object()
    elif mutation == 'command':
        command = 'evaluate'
    else:
        config = loaded.config
        if mutation == 'cuda':
            config = replace(config, training=replace(config.training, device='cuda:0'))
        else:
            config = replace(config, preprocessing=replace(config.preprocessing, backend='rust'))
        loaded = context.projected = replace(loaded, config=config)
    async def finish():
        pytest.fail('ineligible envelope reached finalizer')
    with pytest.raises(ContractError, match='pytest finalizer'):
        asyncio.run(gf_cli._run_experiment(loaded, command, collection=plan, ledger=ledger,
            _pytest_worker_finalizer=finish, **options))
    assert not plan.output.exists()


def test_cli_stale_context_rejected_before_preflight(envelope, monkeypatch):
    loaded, plan, ledger, context = envelope
    async def finish():
        pytest.fail('closed context reached callback')
    with pytest.raises(ContractError, match='live actor-only'):
        asyncio.run(gf_cli._run_experiment(loaded, 'collect', collection=plan, ledger=ledger,
            _structured_factory=object(), _structured_validation_context=context, _pytest_worker_finalizer=finish))
    assert not plan.output.exists()


@pytest.mark.parametrize('failure', ['none', 'callback', 'closed', 'pooled', 'health', 'tickets', 'lease', 'learner', 'updating', 'context', 'loaded', 'recheck'])
def test_cli_validation_rechecks_after_finalizer_before_report(envelope, monkeypatch, failure):
    """Stub groups/validator cannot confer Source authority; stop at report dispatch."""
    import gf_cli.structured
    loaded, plan, ledger, context = envelope
    calls = []
    def validate(self, selected, same, config, *, deadline):
        assert self is context and selected is plan and same is ledger and config is loaded.config
        calls.append('validate')
        if failure == 'recheck' and 'finalizer' in calls:
            raise RuntimeError('injected parent recheck failure')
    monkeypatch.setattr(FinalValidationContext, 'validate_collection', validate)
    async def preflight(*args, **kwargs):
        calls.append('cold-preflight')
    monkeypatch.setattr(gf_cli.structured, 'preflight_final_validation_actor', preflight)
    monkeypatch.setattr(gf_cli, '_code_provenance', lambda deadline: {})
    class Environment:
        def __init__(self, settings, same):
            assert same is ledger
        async def run_group(self, group):
            calls.append('group-drained')
            return GroupResult([], {})
    async def finish():
        calls.append('finalizer')
        receipt = {'closed': True, 'mode': 'pytest-fresh-forked-cpu-v1',
            'cpu_preload': {'healthy': True, 'unretired_tickets': 0}}
        if failure == 'callback':
            raise RuntimeError('injected callback failure')
        if failure == 'closed':
            receipt['closed'] = False
        elif failure == 'pooled':
            receipt['mode'] = 'pytest-cpu-pool-v1'
        elif failure == 'health':
            receipt['cpu_preload']['healthy'] = False
        elif failure == 'tickets':
            receipt['cpu_preload']['unretired_tickets'] = 1
        elif failure == 'lease':
            ledger.reserve_worker(0)
        elif failure == 'learner':
            ledger.learner_bytes = 1
        elif failure == 'updating':
            ledger.updating = True
        elif failure == 'context':
            context.plan = replace(plan)
        elif failure == 'loaded':
            context.projected = replace(loaded)
        return receipt
    def report(*args, **kwargs):
        assert failure == 'none'
        calls.append('report-dispatch')
        raise RuntimeError('stop before publication; dispatch-only evidence')
    monkeypatch.setattr(gf_cli, 'report', report)
    with pytest.raises((RuntimeError, ContractError)):
        asyncio.run(gf_cli._run_experiment(loaded, 'collect', collection=plan, ledger=ledger,
            _structured_factory=Environment, _structured_validation_context=context, _pytest_worker_finalizer=finish))
    assert calls[:4] == ['validate', 'validate', 'cold-preflight', 'group-drained']
    assert calls[4] == 'finalizer'
    if failure == 'none':
        assert calls[5:] == ['validate', 'report-dispatch']
    else:
        assert 'report-dispatch' not in calls
    assert not (plan.output / 'report.json').exists()
