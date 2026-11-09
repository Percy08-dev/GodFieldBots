"""Cheap native preload routing doubles; actual session tests own Source proof."""
from __future__ import annotations

import asyncio
import io
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

import gf_cli
from gf_common.domain import ContractError, GroupResult, RuntimePolicyBinding, digest
from gf_simulator_client.structured_resources import ReservationLedger
from gf_training.collection import CollectionPlan, validation_groups
from gf_training.structured_ppo import _NativeComponentOps
from tests.cpu_preload import CURRENT_TEMPLATE, CpuTemplate
from tests.structured_ppo_support import component_http, configuration


@pytest.fixture
def envelope(tmp_path):
    loaded = configuration(tmp_path)
    config = replace(loaded.config, preprocessing=replace(loaded.config.preprocessing, backend='rust'))
    loaded = replace(loaded, config=config, effective=asdict(config), sha256=digest(asdict(config)))
    ledger = ReservationLedger(config.run.global_cuda_budget_bytes, config.run.max_participant_workers)
    owner = SimpleNamespace(process=SimpleNamespace(returncode=None))
    ledger.reserve_learner(0, owner)
    plan = CollectionPlan(validation_groups(17, 'dispatch-model', actions_version=2)[:1],
        {'dispatch-model': RuntimePolicyBinding('mlp'), 'random-structured-v1': RuntimePolicyBinding('random')},
        tmp_path / 'dispatch-only', {'parent_run_path': str(tmp_path / 'unopened-parent')},
        time.monotonic() + 30, time.monotonic() + 60, 'ppo_batch', time.monotonic())
    # These objects prove admission routing only. They are never Source owners,
    # and the real _validate_owned_metadata rejects this synthetic learner.
    operations = object.__new__(_NativeComponentOps)
    operations._ledger, operations._learner_owner = ledger, owner
    operations.binary, operations._implementation = {}, 'dispatch-only'
    return loaded, plan, ledger, operations


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


@pytest.mark.parametrize('mutation', ['arbitrary', 'subclass', 'pool', 'mixed', 'cuda_training', 'cuda_inference', 'backend', 'reader', 'split'])
def test_native_helper_rejects_ineligible_routes_without_warmup(envelope, bound_owner, monkeypatch, mutation):
    loaded, _, _, operations = envelope
    options = {'source_ops': operations, 'preload_cpu': True}
    if mutation == 'arbitrary':
        options['source_ops'] = object()
    elif mutation == 'subclass':
        class NativeLike(_NativeComponentOps):
            pass
        options['source_ops'] = object.__new__(NativeLike)
    elif mutation == 'pool':
        options['pool_cpu'] = True
    elif mutation == 'mixed':
        options['validation_context'] = object()
    else:
        config = loaded.config
        if mutation == 'cuda_training':
            config = replace(config, training=replace(config.training, device='cuda:0'))
        elif mutation == 'cuda_inference':
            config = replace(config, model=replace(config.model, inference=replace(config.model.inference, device='cuda:0')))
        elif mutation in ('reader', 'backend'):
            config = replace(config, preprocessing=replace(config.preprocessing,
                **({'reader': 'other'} if mutation == 'reader' else {'backend': 'python'})))
        else:
            config = replace(config, run=replace(config.run, split='validation'))
        loaded = replace(loaded, config=config)
    with pytest.raises(ValueError, match='CPU preload'):
        component_http(monkeypatch, loaded, **options)


def test_generic_native_helper_stays_cold_by_default(envelope, bound_owner, monkeypatch):
    loaded, _, _, operations = envelope
    component_http(monkeypatch, loaded, source_ops=operations)
    assert bound_owner.state == 'UNSTARTED'


@pytest.mark.parametrize('failure', ['none', 'owner', 'before_warmup', 'health', 'after_callback', 'worker', 'stream', 'thread', 'lease'])
def test_native_helper_actual_envelope_checks_and_retirement_order(envelope, bound_owner, monkeypatch, failure):
    loaded, plan, ledger, operations = envelope
    calls = []
    original = RuntimeError('injected native dispatch failure')
    def validate(self, selected, same, *, config, deadline, loaded):
        assert self is operations and selected is plan and same is ledger and config is loaded.config
        assert loaded is envelope[0]
        calls.append('validate')
        if failure == 'before_warmup' or (failure == 'after_callback' and calls.count('validate') == 2):
            raise original
    monkeypatch.setattr(_NativeComponentOps, 'validate_owned_collection', validate)
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
        assert ledger.learner is not None  # Actor completion must retain the learner.
        calls.append('returned')
        return plan.output, {'comparison_valid': False}
    monkeypatch.setattr(gf_cli, '_run_experiment', experiment)
    collector, _, workers, _ = component_http(monkeypatch, loaded, source_ops=operations, preload_cpu=True)
    if failure == 'owner':
        ledger = ReservationLedger()
    if failure in ('worker', 'stream', 'thread', 'lease'):
        stream = io.BytesIO()
        if failure != 'stream':
            stream.close()
        workers.append(SimpleNamespace(closed=failure != 'worker',
            lease=SimpleNamespace(ledger=ledger, released=failure != 'lease'),
            process=SimpleNamespace(poll=lambda: 0, stdin=stream, stdout=None),
            io_thread=SimpleNamespace(is_alive=lambda: failure == 'thread')))
    if failure == 'none':
        asyncio.run(collector(plan, ledger))
        assert calls == ['validate', 'start', 'experiment', 'health', 'validate', 'returned']
    else:
        with pytest.raises((RuntimeError, ContractError)):
            asyncio.run(collector(plan, ledger))
        assert 'returned' not in calls
        if failure == 'owner':
            assert calls == []
        elif failure == 'before_warmup':
            assert calls == ['validate']


@pytest.mark.parametrize('mutation', ['none', 'default', 'effective', 'hash', 'config', 'inputs', 'parent', 'ledger', 'plan'])
def test_optional_loaded_validator_uses_one_source_and_original_inputs(envelope, monkeypatch, mutation):
    import gf_training.native_structured_session as session
    loaded, plan, ledger, operations = envelope
    calls = []
    source = SimpleNamespace(summary={'parents': [{'state': {'config_sha256': loaded.sha256}}]})
    source.original_config_inputs = lambda index, held: calls.append(('inputs', index, held)) or []
    source.recheck = lambda *, additional: calls.append(('recheck', additional))
    @contextmanager
    def same_source(directory, config, deadline, *, extra_owned_bytes=0):
        calls.append(('source', extra_owned_bytes))
        yield source
        calls.append(('closed',))
    def metadata(selected, same, parent, *, deadline):
        assert selected is plan and same is ledger and parent is source.summary['parents'][0]
        calls.append(('metadata',))
        if mutation == 'plan':
            raise ContractError('injected frozen plan mismatch')
    monkeypatch.setattr(operations, '_source', same_source)
    monkeypatch.setattr(session, '_validate_owned_metadata', metadata)
    if mutation == 'effective':
        loaded = replace(loaded, effective={})
    elif mutation == 'hash':
        loaded = replace(loaded, sha256='0' * 64)
    elif mutation == 'config':
        loaded = replace(loaded, config=replace(loaded.config))
    elif mutation == 'inputs':
        loaded = replace(loaded, inputs=({'path': 'mismatched-origin'},))
    elif mutation == 'parent':
        source.summary['parents'][0]['state']['config_sha256'] = '0' * 64
    elif mutation == 'ledger':
        ledger = ReservationLedger()
    options = {} if mutation == 'default' else {'loaded': loaded}
    if mutation in ('none', 'default'):
        operations.validate_owned_collection(plan, ledger, config=envelope[0].config, deadline=plan.execution_deadline, **options)
        assert [call[0] for call in calls] == (['source', 'metadata', 'closed'] if mutation == 'default'
            else ['source', 'inputs', 'recheck', 'metadata', 'closed'])
        assert calls[0][1] == 0 if mutation == 'default' else calls[0][1] > 65536
        if mutation == 'none':
            assert calls[1][2] == calls[0][1] and calls[2][1] > 0
    else:
        with pytest.raises(ContractError):
            operations.validate_owned_collection(plan, ledger, config=envelope[0].config,
                deadline=plan.execution_deadline, **options)
        if mutation in ('effective', 'hash', 'config', 'ledger'):
            assert calls == []
        elif mutation != 'plan':
            assert ('metadata',) not in calls


@pytest.mark.parametrize('mutation', ['arbitrary', 'subclass', 'factory', 'mixed', 'command', 'cuda', 'reader', 'split', 'kind', 'ledger'])
def test_cli_native_finalizer_requires_exact_composition(envelope, mutation):
    loaded, plan, ledger, operations = envelope
    options = {'_structured_factory': object(), '_structured_source_ops': operations}
    command = 'collect'
    if mutation == 'arbitrary':
        options['_structured_source_ops'] = object()
    elif mutation == 'subclass':
        class NativeLike(_NativeComponentOps):
            pass
        options['_structured_source_ops'] = object.__new__(NativeLike)
    elif mutation == 'factory':
        options['_structured_factory'] = None
    elif mutation == 'mixed':
        options['_structured_validation_context'] = object()
    elif mutation == 'command':
        command = 'evaluate'
    elif mutation == 'ledger':
        ledger = ReservationLedger()
    elif mutation == 'kind':
        plan = replace(plan, kind='ppo_validation')
    else:
        config = loaded.config
        if mutation == 'cuda':
            config = replace(config, training=replace(config.training, device='cuda:0'))
        elif mutation == 'reader':
            config = replace(config, preprocessing=replace(config.preprocessing, reader='other'))
        else:
            config = replace(config, run=replace(config.run, split='validation'))
        loaded = replace(loaded, config=config)
    async def finish():
        pytest.fail('ineligible native composition reached finalizer')
    with pytest.raises(ContractError, match='pytest finalizer|SAME concrete'):
        asyncio.run(gf_cli._run_experiment(loaded, command, collection=plan, ledger=ledger,
            _pytest_worker_finalizer=finish, **options))
    assert not plan.output.exists()


@pytest.mark.parametrize('failure', ['none', 'before_groups', 'callback', 'closed', 'pooled', 'health', 'tickets', 'lease', 'owner', 'envelope', 'recheck'])
def test_cli_native_rechecks_live_learner_before_publication(envelope, monkeypatch, failure):
    loaded, plan, ledger, operations = envelope
    calls = []
    def validate(self, selected, same, *, config, deadline, loaded):
        assert self is operations and selected is plan and same is ledger and loaded is envelope[0]
        calls.append('validate')
        if failure == 'before_groups' or (failure == 'recheck' and 'finalizer' in calls):
            raise RuntimeError('injected original Source recheck failure')
        if loaded.effective != asdict(config):
            raise ContractError('injected actual envelope mismatch')
    monkeypatch.setattr(_NativeComponentOps, 'validate_owned_collection', validate)
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
        elif failure == 'owner':
            operations._learner_owner = object()
        elif failure == 'envelope':
            loaded.effective.clear()
        return receipt
    def report(*args, **kwargs):
        assert failure == 'none' and ledger.learner is not None and not ledger.workers
        calls.append('report-dispatch')
        raise RuntimeError('stop before publication; dispatch-only evidence')
    monkeypatch.setattr(gf_cli, 'report', report)
    with pytest.raises((RuntimeError, ContractError)):
        asyncio.run(gf_cli._run_experiment(loaded, 'collect', collection=plan, ledger=ledger,
            _structured_factory=Environment, _structured_source_ops=operations, _pytest_worker_finalizer=finish))
    if failure == 'before_groups':
        assert calls == ['validate'] and not plan.output.exists()
    else:
        assert calls[:3] == ['validate', 'group-drained', 'finalizer']
        if failure == 'none':
            assert calls[3:] == ['validate', 'report-dispatch']
        else:
            assert 'report-dispatch' not in calls
    assert not (plan.output / 'report.json').exists()
