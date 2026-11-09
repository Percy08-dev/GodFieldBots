"""Pytest-only readiness ownership, ordering and unchanged default routing."""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from gf_common.action_contract import engineering_contract
from gf_models.structured_baselines import make_structured_policy
from gf_simulator_client.structured_environment import StructuredEpisodeResult
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from tests.structured_ppo_support import _parallel_ready_actors, component_http
from tests.test_structured_protocol import action_settings


class ControlledWorker(StructuredParticipantWorker):
    """Initializer with explicit success and abort barriers, without external I/O."""

    def __init__(self, elapsed, *, error=None):
        self.elapsed, self.error = elapsed, error
        self.startup_elapsed_ms = None
        self.started, self.release, self.done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.abort_started, self.abort_release = asyncio.Event(), asyncio.Event()
        self.task = None
        self.cancellations = []
        self.calls = 0

    async def initialize(self):
        self.calls += 1
        self.task = asyncio.current_task()
        self.started.set()
        try:
            await self.release.wait()
            if self.error is not None:
                raise self.error
            self.startup_elapsed_ms = self.elapsed
        except asyncio.CancelledError:
            self.cancellations.append(self.task.cancelling())
            self.abort_started.set()
            try:
                await self.abort_release.wait()
            except asyncio.CancelledError:
                self.cancellations.append(self.task.cancelling())
                raise
            raise
        finally:
            self.done.set()


def episode_result():
    return StructuredEpisodeResult('episode', 'pair', 'seed', ('random', 'random'),
        action_contract=engineering_contract(), purpose='component', synthetic_capacity=False)


def loaded_config(*, learning='cpu', inference='cpu'):
    return SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(device=learning),
        model=SimpleNamespace(inference=SimpleNamespace(device=inference))))


def run(coroutine):
    async def bounded():
        async with asyncio.timeout(5):
            await coroutine
    asyncio.run(bounded())


def test_both_initializers_start_and_both_must_succeed_with_explicit_viewers():
    async def check():
        first, second = ControlledWorker(11), ControlledWorker(22)
        workers, result = {2: first, 1: second}, episode_result()
        ready = asyncio.create_task(_parallel_ready_actors(workers, result))
        await first.started.wait()
        await second.started.wait()
        # Mutation after scheduling cannot change ownership or result identities.
        workers.clear()
        second.release.set()
        await second.done.wait()
        assert not ready.done() and not first.done.is_set()
        first.release.set()
        await ready
        assert result.model_startup_ms == {2: 11, 1: 22}
        assert first.calls == second.calls == 1
        assert first.task.done() and second.task.done()
    run(check())


@pytest.mark.parametrize('parallel', [False, True])
def test_component_route_keeps_default_sequential_and_selection_after_ready(tmp_path, monkeypatch, parallel):
    import gf_cli
    from gf_simulator_client import structured_environment

    async def check():
        result, selected = episode_result(), []
        original = structured_environment.StructuredSimulatorEnvironment._ready_actors
        delegated = []
        async def default(self, workers, episode):
            delegated.append(True)
            return await original(self, workers, episode)
        monkeypatch.setattr(structured_environment.StructuredSimulatorEnvironment, '_ready_actors', default)
        async def experiment(*args, _structured_factory, **kwargs):
            environment = object.__new__(_structured_factory)
            await environment._ready_actors({2: first, 1: second}, result)
            selected.append(True)
            return tmp_path, {}
        monkeypatch.setattr(gf_cli, '_run_experiment', experiment)
        kwargs = {'parallel_ready': True} if parallel else {}
        collector, scripts, workers, ledgers = component_http(monkeypatch, loaded_config(), **kwargs)
        # Retain the original default method's check against the registered class.
        class ComponentWorker(ControlledWorker, structured_environment.StructuredParticipantWorker):
            pass
        first, second = ComponentWorker(11), ComponentWorker(22)
        ledger = SimpleNamespace(workers={}, learner=object())
        task = asyncio.create_task(collector(None, ledger))
        await first.started.wait()
        if parallel:
            await second.started.wait()
        else:
            assert not second.started.is_set() and delegated == [True]
        assert not selected
        first.release.set()
        await second.started.wait()
        assert not selected
        second.release.set()
        await task
        assert selected == [True] and delegated == ([] if parallel else [True])
        assert result.model_startup_ms == {2: 11, 1: 22}
        assert scripts == workers == ledgers == []
    run(check())


@pytest.mark.parametrize('forbidden', ['learning_cuda', 'inference_cuda', 'both_cuda', 'source_ops', 'validation_context'])
def test_explicit_parallel_rejected_before_worker_or_server_activity(monkeypatch, forbidden):
    from gf_simulator_client import structured_environment

    config = loaded_config(learning='cuda:0' if forbidden in ('learning_cuda', 'both_cuda') else 'cpu',
        inference='cuda:0' if forbidden in ('inference_cuda', 'both_cuda') else 'cpu')
    original = structured_environment.StructuredParticipantWorker
    kwargs = {forbidden: object()} if forbidden in ('source_ops', 'validation_context') else {}
    with pytest.raises(ValueError, match='parallel readiness requires ordinary CPU'):
        component_http(monkeypatch, config, parallel_ready=True, **kwargs)
    # Registration occurs before any collector can start external activity.
    assert structured_environment.StructuredParticipantWorker is original


@pytest.mark.parametrize('cpu,kwargs,expected', [
    (True, {}, (True, True)), (True, {'pool_cpu': False}, (False, True)),
    (True, {'parallel_ready': False}, (True, False)),
    (False, {}, (False, False)), (True, {'source_ops': object()}, (False, False)),
    (True, {'validation_context': object()}, (False, False)),
])
def test_cpu_module_opt_in_is_independent_of_physical_pool(monkeypatch, cpu, kwargs, expected):
    from tests import test_structured_ppo as module

    calls = []
    def capture(*args, **options):
        calls.append((options['pool_cpu'], options['parallel_ready']))
        return 'collector'
    monkeypatch.setattr(module, 'fresh_component_http', capture)
    assert module.component_http(monkeypatch, loaded_config(learning='cpu' if cpu else 'cuda:0'), **kwargs) == 'collector'
    assert calls == [expected]


@pytest.mark.parametrize('sibling_fails', [False, True])
def test_ordinary_failure_drains_delayed_sibling_and_raises_stable_concrete_error(sibling_fails):
    class InitializerFailure(BaseException):
        pass

    async def check():
        first_error, second_error = InitializerFailure('viewer two'), ValueError('viewer one')
        first = ControlledWorker(11, error=first_error)
        second = ControlledWorker(22, error=second_error if sibling_fails else None)
        result = episode_result()
        ready = asyncio.create_task(_parallel_ready_actors({2: first, 1: second}, result))
        await first.started.wait()
        await second.started.wait()
        # Finish viewer one first; stable mapping order still selects viewer two.
        (second if sibling_fails else first).release.set()
        await (second if sibling_fails else first).done.wait()
        assert not ready.done()
        (first if sibling_fails else second).release.set()
        with pytest.raises(InitializerFailure) as raised:
            await ready
        assert raised.value is first_error
        assert first.done.is_set() and second.done.is_set()
        assert not first.cancellations and not second.cancellations
        assert result.model_startup_ms == ({} if sibling_fails else {1: 22})
    run(check())


@pytest.mark.parametrize('outer', ['cancel', 'timeout', 'repeated_cancel'])
def test_outer_failure_waits_for_abort_and_cancels_each_child_once(outer):
    async def check():
        first, second = ControlledWorker(11), ControlledWorker(22)
        timeout = asyncio.timeout(None)
        async def initialize():
            async with timeout:
                await _parallel_ready_actors({2: first, 1: second}, episode_result())
        ready = asyncio.create_task(initialize())
        await first.started.wait()
        await second.started.wait()
        if outer == 'timeout':
            timeout.reschedule(asyncio.get_running_loop().time())
        else:
            ready.cancel('original outer cancellation')
        await first.abort_started.wait()
        await second.abort_started.wait()
        assert not ready.done()
        if outer == 'repeated_cancel':
            for _ in range(3):
                ready.cancel('later outer cancellation')
                await asyncio.sleep(0)
                assert first.task.cancelling() == second.task.cancelling() == 1
                assert not ready.done()
        first.abort_release.set()
        await first.done.wait()
        assert not ready.done()
        second.abort_release.set()
        with pytest.raises(TimeoutError if outer == 'timeout' else asyncio.CancelledError) as raised:
            await ready
        if outer != 'timeout':
            assert raised.value.args == ('original outer cancellation',)
        assert first.cancellations == second.cancellations == [1]
        assert first.task.done() and second.task.done()
    run(check())


def test_partial_task_creation_closes_unscheduled_coroutine_and_drains_prior_task(monkeypatch):
    async def check():
        first, second = ControlledWorker(11), ControlledWorker(22)
        original_create = asyncio.create_task
        submitted, rejected = [], []
        creation_error = RuntimeError('cannot schedule second initializer')
        def create(coroutine):
            if submitted:
                rejected.append(coroutine)
                raise creation_error
            task = original_create(coroutine, eager_start=True)
            submitted.append(task)
            return task
        monkeypatch.setattr(asyncio, 'create_task', create)
        ready = original_create(_parallel_ready_actors({2: first, 1: second}, episode_result()))
        await first.abort_started.wait()
        assert not ready.done() and first.started.is_set()
        assert len(rejected) == 1 and inspect.getcoroutinestate(rejected[0]) == inspect.CORO_CLOSED
        assert not second.started.is_set() and second.calls == 0
        ready.cancel('later cancellation must preserve creation failure')
        await asyncio.sleep(0)
        assert first.task.cancelling() == 1 and not ready.done()
        first.abort_release.set()
        with pytest.raises(RuntimeError) as raised:
            await ready
        assert raised.value is creation_error
        assert submitted[0].done() and first.cancellations == [1]
    run(check())


def test_parallel_readiness_retains_actual_fresh_bootstrap_and_reap():
    async def check():
        ledger, result = ReservationLedger(), episode_result()
        workers = {viewer: StructuredParticipantWorker(make_structured_policy, 'random', viewer,
                    action_settings(), ledger=ledger) for viewer in (2, 1)}
        try:
            assert len(ledger.workers) == 2
            await _parallel_ready_actors(workers, result)
            assert all(worker.initialized and worker.process.poll() is None for worker in workers.values())
            assert result.model_startup_ms == {viewer: worker.startup_elapsed_ms for viewer, worker in workers.items()}
            assert len({worker.process.pid for worker in workers.values()}) == 2
        finally:
            for worker in workers.values():
                await worker.close()
        assert not ledger.workers
        assert all(worker.closed and worker.process.poll() is not None and worker.lease.released for worker in workers.values())
    run(check())
