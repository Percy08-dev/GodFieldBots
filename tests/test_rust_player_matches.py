"""Managed pump ordering/ownership fixtures; no game or Source authority."""
import asyncio
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gf_common.domain import EpisodeGroup, EpisodeSpec, digest
from gf_training.rust_player_storage import PolicyJournalContext


@dataclass(frozen=True)
class Controls:
    value: int


def pump_fixture(tmp_path, monkeypatch, *, failure=None, concurrency=2, http_diagnostics=None):
    from gf_training import rust_player_matches as module
    events, servers = [], []
    execution, cleanup = time.monotonic()+30, time.monotonic()+40
    groups = tuple(EpisodeGroup(f'seed-{i}', tuple(EpisodeSpec(f'e{i}{j}', ('p1', 'p2'), (1, 2),
        f'pair-{i}', 1) for j in range(2))) for i in range(3))
    policies = (PolicyJournalContext({'kind': 'random'}, {'policy': 'p1'}),
        PolicyJournalContext({'kind': 'heuristic'}, {'policy': 'p2'}))
    actions, rules = Controls(1), Controls(2)
    contexts = {ep.episode_id: {'run_id': 'run', 'trial': 'batch', 'episode_id': ep.episode_id,
        'actions': actions, 'rules': rules, 'numeric_costs': {'cost': 3}, 'policies': policies,
        'deadline': execution, 'clock': None} for group in groups for ep in group.episodes}
    arrivals = [('journal', 'e10'), ('terminal', 'e10'), ('terminal', 'e00'),
        ('terminal', 'e11'), ('terminal', 'e20'), ('terminal', 'e01'), ('terminal', 'e21')]
    class Runtime:
        def __init__(self):
            self.bundles = [SimpleNamespace(id=i, kind=kind, episode=ep, game='game-'+ep,
                transport_sha256=str(i), reserved_bytes=i+1, written=False) for i, (kind, ep) in enumerate(arrivals)]
            for bundle in self.bundles:
                bundle._http_diagnostics = lambda: (1, 0, 1, False)
            self.started = []
            self.start_batches = []
            self.condition = threading.Condition()
            self.cancelled = False
        def start(self, specs):
            with self.condition:
                self.started.extend(specs)
                self.start_batches.append(tuple(s['episode'] for s in specs))
                events.append(('start', specs[0]['episode']))
                self.condition.notify_all()
        def next_storage(self, deadline):
            assert deadline == execution
            with self.condition:
                while not self.cancelled:
                    if self.bundles and any(s['episode'] == self.bundles[0].episode for s in self.started):
                        return self.bundles.pop(0)
                    self.condition.wait(deadline - time.monotonic())
                    if time.monotonic() >= deadline:
                        raise TimeoutError('original storage deadline')
                raise OSError('original storage cancelled')
        def storage_ack(self, bundle):
            assert bundle.written
            events.append(('ack', bundle.id))
        def cancel(self, reason):
            with self.condition:
                self.cancelled = True
                events.append(('runtime-cancel', reason))
                self.condition.notify_all()
        def wait(self, deadline):
            assert deadline == execution and not self.bundles
            events.append(('wait',))
        def completed_reports(self, keys):
            events.append(('reports', tuple(keys)))
            return [{'run': run, 'trial': trial, 'episode': ep} for run, trial, ep in keys]
    runtime = Runtime()
    pump = module.RustMatchPump(runtime)
    class Server:
        def __init__(self, binary, config, seed, count, directory, **kwargs):
            self.index, self.stopped = len(servers), False
            self.process, self.expected = object(), {'seed': seed}
            self.health = {'exit_code': 0}
            self.http_capture = kwargs.pop('http_capture', None)
            assert binary == 'binary' and config == 'config' and count == 2
            assert kwargs == {'startup_seconds': 5., 'shutdown_seconds': 7.,
                'cleanup_deadline': cleanup, 'strict_cleanup_deadline': True}
            servers.append(self); events.append(('acquire', self.index))
        async def start(self):
            assert self in pump.servers.values()
            assert pump.server_history[-1] == {'server': self, 'runtime_evidence': None}
            if failure == 'start': raise OSError('original server start failure')
            return {'base_url': 'http://actual', 'test_server_info': {'seed_fingerprint': 'fingerprint'},
                'test_server_config': self.expected}
        async def stop(self):
            self.stopped = True; events.append(('stop', self.index))
            return self.health
        async def settle_http(self, **facts):
            assert self.stopped and facts == {'deadline': cleanup, 'abnormal': False,
                'responses': 2, 'error_responses': 0, 'transport_errors': 2}
            events.append(('http-settle', self.index))
            return {'status': 'archived'}
    class Evidence:
        def preflight(self): raise AssertionError('preflight belongs to caller before try')
        def verify_executed(self, process):
            server = next(server for server in servers if server.process is process)
            events.append(('receipt', server.index))
            if failure == 'receipt': raise OSError('original receipt failure')
            return {'executed': {'pid': 100+server.index}}
    class Writer:
        def __init__(self): self.contexts = []
        def bind_contexts(self, actual):
            self.contexts.append(actual); events.append(('bind', actual[0].episode_id))
            assert pump.server_history[-1]['runtime_evidence'] is actual[0].runtime_evidence
            if failure == 'bind': raise OSError('original bind failure')
        def accept(self, bundle):
            events.append(('write', bundle.id))
            if failure == 'write': raise OSError('original writer failure')
            bundle.written = True
        def cancel(self, reason):
            events.append(('writer-cancel', reason))
    writer = Writer()
    monkeypatch.setattr(module, 'ManagedServer', Server)
    server_config = SimpleNamespace(binary='binary', config='config', startup_seconds=5., shutdown_seconds=7., request_seconds=9.)
    def execute(*, batch_id='batch', output=tmp_path):
        return pump.execute(groups, contexts, writer, server=server_config, actions=actions,
            max_decisions=11, output=output, execution_deadline=execution, cleanup_deadline=cleanup,
            game_concurrency=concurrency, run_id='run', batch_id=batch_id, evidence=Evidence(),
            **({} if http_diagnostics is None else {'http_diagnostics': http_diagnostics}))
    return pump, runtime, writer, contexts, servers, events, execute


def test_match_pump_preserves_slots_write_ack_arrival_and_plan_report_order(tmp_path, monkeypatch):
    pump, runtime, writer, contexts, servers, events, execute = pump_fixture(tmp_path, monkeypatch)
    result = asyncio.run(execute())
    assert [row['episode'] for row in result.accepted] == ['e10', 'e10', 'e00', 'e11', 'e20', 'e01', 'e21']
    assert [row['episode'] for row in result.reports] == ['e00', 'e01', 'e10', 'e11', 'e20', 'e21']
    assert [spec['episode'] for spec in runtime.started] == ['e00', 'e10', 'e11', 'e01', 'e20', 'e21']
    assert events.index(('stop', 1)) < events.index(('acquire', 2))
    assert result.total == 6 and all(server.stopped for server in servers)
    assert len(pump.server_history) == len(servers) == 3
    for row in result.accepted:
        assert events.index(('write', row['id'])) < events.index(('ack', row['id']))
    assert events[-2] == ('wait',) and events[-1][0] == 'reports'
    for index, actual in enumerate(writer.contexts):
        from dataclasses import asdict
        assert result.bound_contexts[index] == digest([asdict(context) for context in actual])
        assert result.receipts[index] is actual[0].runtime_evidence
        assert result.completed_groups[index]['runtime_evidence'] is result.receipts[index]
    spec = runtime.started[0]
    assert spec['numeric_costs'] is contexts['e00']['numeric_costs']
    assert spec['policies'][0] is contexts['e00']['policies'][0].native


def test_http_pair_facts_settle_after_terminal_ack_without_changing_capture_receipts(tmp_path, monkeypatch):
    from gf_common.hpo_storage import HPODiagnosticsConfig

    config = HPODiagnosticsConfig(str(tmp_path / 'ram'), str(tmp_path / 'archive'))
    pump, _, _, _, servers, events, execute = pump_fixture(tmp_path, monkeypatch, http_diagnostics=config)
    result = asyncio.run(execute())
    assert all(server.http_capture is not None for server in servers)
    assert events.index(('ack', 3)) < events.index(('stop', 1)) < events.index(('http-settle', 1))
    assert events.index(('http-settle', 1)) < events.index(('acquire', 2))
    assert all(group['http_diagnostics'] == {'status': 'archived'} for group in result.completed_groups.values())
    assert all(item['http_diagnostics'] == {'status': 'archived'} for item in pump.server_history)
    assert all(set(row) == {'id', 'kind', 'episode', 'game', 'transport_sha256', 'bytes'}
               for row in result.accepted)


@pytest.mark.parametrize('concurrency', [6, 8])
def test_match_pump_admits_fitting_wave_once_after_all_contexts_are_bound(tmp_path, monkeypatch, concurrency):
    _, runtime, writer, _, servers, events, execute = pump_fixture(tmp_path, monkeypatch, concurrency=concurrency)
    # Odd seats finish first. Each group must remain alive until both of its
    # original terminal records have passed writer acceptance and ACK.
    runtime.bundles[:] = [runtime.bundles[i] for i in (3, 5, 6, 1, 2, 4)]
    first_start = runtime.start

    def start(specs):
        assert len(writer.contexts) == 3
        first_start(specs)

    monkeypatch.setattr(runtime, 'start', start)
    result = asyncio.run(execute())

    planned = ('e00', 'e01', 'e10', 'e11', 'e20', 'e21')
    assert runtime.start_batches == [planned]
    assert tuple(row['episode'] for row in result.reports) == planned
    assert [row['episode'] for row in result.accepted] == ['e11', 'e01', 'e21', 'e10', 'e00', 'e20']
    for index, (pair, server) in enumerate(zip(writer.contexts, servers, strict=True)):
        first, second = runtime.started[index * 2:index * 2 + 2]
        assert server.expected == {'seed': f'seed-{index}'} and server.stopped
        assert first['runtime_evidence'] is second['runtime_evidence'] is result.receipts[index]
        assert first['runtime_evidence'] is pair[0].runtime_evidence is pair[1].runtime_evidence
        for row in result.accepted:
            if row['episode'] in planned[index * 2:index * 2 + 2]:
                assert events.index(('ack', row['id'])) < events.index(('stop', index))


def test_match_pump_retains_same_runtime_handle_map_and_original_history_across_executes(tmp_path, monkeypatch):
    pump, runtime, _writer, contexts, servers, _events, execute = pump_fixture(tmp_path, monkeypatch)
    handles, history = pump.servers, pump.server_history
    asyncio.run(execute())
    original = tuple(history)
    for context in contexts.values(): context['trial'] = 'next-batch'
    arrivals = [('journal', 'e10'), ('terminal', 'e10'), ('terminal', 'e00'),
        ('terminal', 'e11'), ('terminal', 'e20'), ('terminal', 'e01'), ('terminal', 'e21')]
    runtime.bundles = [SimpleNamespace(id=100+i, kind=kind, episode=ep, game='game-'+ep,
        transport_sha256=str(i), reserved_bytes=i+1, written=False) for i, (kind, ep) in enumerate(arrivals)]
    result = asyncio.run(execute(batch_id='next-batch', output=tmp_path/'next'))
    assert pump.runtime is runtime and pump.servers is handles and pump.server_history is history
    assert all(now is before for now, before in zip(history[:3], original, strict=True))
    assert len(history) == len(servers) == 6 and all(server.stopped for server in servers)
    assert [handles[i] for i in range(3)] == servers[3:]
    assert all(key[1] == 'next-batch' for key in result.keys)
    assert not pump._pending_http


def test_match_pump_keeps_draining_originals_during_server_stop(tmp_path, monkeypatch):
    from gf_training import rust_player_matches as module

    _, runtime, _, _, _, events, execute = pump_fixture(tmp_path, monkeypatch)
    stopping_group = 1
    concurrent_terminal = runtime.bundles.pop(5)
    runtime.bundles.insert(4, concurrent_terminal)
    original_stop = module.ManagedServer.stop
    original_ack = runtime.storage_ack

    async def run():
        loop = asyncio.get_running_loop()
        released = asyncio.Event()

        def ack(bundle):
            original_ack(bundle)
            if bundle is concurrent_terminal:
                loop.call_soon_threadsafe(released.set)

        async def stop(server):
            if server.index == stopping_group:
                await asyncio.wait_for(released.wait(), 2)
            return await original_stop(server)

        monkeypatch.setattr(runtime, 'storage_ack', ack)
        monkeypatch.setattr(module.ManagedServer, 'stop', stop)
        return await execute()

    result = asyncio.run(run())
    assert events.index(('ack', concurrent_terminal.id)) < events.index(('stop', stopping_group))
    assert [r['id'] for r in result.accepted] == [0, 1, 2, 3, 5, 4, 6]


def test_match_pump_worker_failure_wakes_terminal_wait_and_cancels_both_owners(tmp_path, monkeypatch):
    _, runtime, _, _, _, events, execute = pump_fixture(tmp_path, monkeypatch, failure='write')

    async def run():
        with pytest.raises(OSError, match='original writer failure'):
            await asyncio.wait_for(execute(), 2)

    asyncio.run(run())
    assert runtime.cancelled and any(event[0] == 'writer-cancel' for event in events)
    assert not any(event[0] in ('ack', 'stop', 'wait', 'reports') for event in events)


def test_match_pump_preserves_reader_failure_when_native_rolling_start_races_it(tmp_path, monkeypatch):
    _, runtime, _, _, _, events, execute = pump_fixture(tmp_path, monkeypatch)
    rolling_start = threading.Event()
    native_cancelled = threading.Event()
    original_start, original_cancel = runtime.start, runtime.cancel
    original_error = OSError('original reader failure')
    original_next = runtime.next_storage

    def receive(deadline):
        if runtime.bundles[0].id == 2:
            assert rolling_start.wait(2)
            raise original_error
        return original_next(deadline)

    def start(specs):
        if specs[0]['episode'] == 'e11':
            rolling_start.set()
            assert native_cancelled.wait(2)
            raise RuntimeError('native scheduler closed')
        original_start(specs)

    def cancel(reason):
        original_cancel(reason)
        native_cancelled.set()

    monkeypatch.setattr(runtime, 'next_storage', receive)
    monkeypatch.setattr(runtime, 'start', start)
    monkeypatch.setattr(runtime, 'cancel', cancel)

    with pytest.raises(OSError, match='original reader failure') as caught:
        asyncio.run(execute())
    assert caught.value is original_error
    assert runtime.start_batches == [('e00',), ('e10',)]
    assert ('ack', 1) in events and ('ack', 2) not in events


def test_match_pump_repeated_cancellation_joins_original_drain(tmp_path, monkeypatch):
    _, runtime, _, _, _, events, execute = pump_fixture(tmp_path, monkeypatch)
    release = threading.Event()

    async def run():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        original_cancel = runtime.cancel

        def receive(deadline):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(4)
            events.append(('drain-exited',))
            raise OSError('original storage cancelled')

        def cancel(reason):
            original_cancel(reason)
            cancelled.set()

        monkeypatch.setattr(runtime, 'next_storage', receive)
        monkeypatch.setattr(runtime, 'cancel', cancel)
        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel('primary cancellation')
            await asyncio.wait_for(cancelled.wait(), 2)
            task.cancel('repeated cancellation')
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError, match='primary cancellation'):
            await task

    asyncio.run(run())
    assert ('drain-exited',) in events
    assert any(event[0] == 'writer-cancel' for event in events)
    assert not any(event[0] in ('ack', 'wait', 'reports') for event in events)


@pytest.mark.parametrize('failure', ['start', 'receipt', 'bind', 'write'])
def test_match_pump_failure_keeps_original_handles_and_error_for_caller_cleanup(tmp_path, monkeypatch, failure):
    pump, _runtime, _writer, _contexts, servers, events, execute = pump_fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises(OSError, match='original'):
        asyncio.run(execute())
    assert servers and pump.server_history[0]['server'] is servers[0]
    assert pump.servers[0] is servers[0] and not servers[0].stopped
    assert not any(event[0] in ('ack', 'stop', 'wait', 'reports') for event in events)
    if failure in ('start', 'receipt'):
        assert pump.server_history[0]['runtime_evidence'] is None
    assert not pump._pending_http


@pytest.mark.parametrize('cleanup_failure', [False, True])
@pytest.mark.parametrize('failure_stage', ['preflight', 'pump'])
def test_collector_keeps_original_preflight_try_cleanup_order_aliases_and_loans(tmp_path, monkeypatch, cleanup_failure, failure_stage):
    from gf_common.group_pack import StorageSettings
    from gf_common.rust_player import deadline_bits
    from gf_training import rust_player_collection as module
    from gf_training.group_pack_resources import StorageBudget
    from gf_training.rust_player_matches import RustMatchPump
    events = []
    deadline = time.monotonic()+30
    budget = StorageBudget(StorageSettings(), deadline)
    primary = OSError('original '+failure_stage+' failure')
    class Runtime:
        diagnostics = None
        def recording_account(self): return object()
        def cancel(self, reason): events.append('runtime-cancel')
    class Writer:
        def __init__(self, *args, **kwargs): events.append('writer-acquire')
        def cancel(self, reason): events.append('writer-cancel')
        def close(self):
            events.append('writer-close')
            if cleanup_failure: raise OSError('uncertain writer')
    class Evidence:
        def __init__(self, *args, **kwargs): events.append('evidence-acquire')
        def preflight(self):
            events.append('preflight')
            if failure_stage == 'preflight': raise primary
    class Server:
        stopped = False
        async def stop(self):
            events.append('server-stop')
            if cleanup_failure: raise OSError('uncertain server')
            self.stopped = True
    runtime = Runtime()
    pump = RustMatchPump(runtime)
    async def fail(*args, **kwargs):
        assert events[-1] == 'preflight'
        assert kwargs['evidence'] is not None
        events.append('pump')
        pump.servers[0] = Server()
        pump.server_history.append({'server': pump.servers[0], 'runtime_evidence': None})
        raise primary
    monkeypatch.setattr(pump, 'execute', fail)
    monkeypatch.setattr(module, 'RustPlayerStorage', Writer)
    monkeypatch.setattr(module, 'CardInstanceRuntimeEvidence', Evidence)
    monkeypatch.setattr(module, 'RecordingBudget', lambda *args: object())
    monkeypatch.setattr(module, 'RecordingArtifacts', lambda *args: object())
    async def inline(function, *args): return function(*args)
    monkeypatch.setattr(module.asyncio, 'to_thread', inline)
    # Failure-only receiver: cannot issue collection/Source/admission authority.
    collector = module.RustPlayerCollector.__new__(module.RustPlayerCollector)
    collector.runtime, collector._matches = runtime, pump
    collector._servers, collector.server_history = pump.servers, pump.server_history
    collector.writer_history, collector.current = [], None
    collector.mode = None
    collector.settings, collector.cleanup_deadline = StorageSettings(), deadline+10
    collector.declaration = {'game_concurrency': 1, 'max_batch_size': 1}
    collector.config = SimpleNamespace(storage=SimpleNamespace(db_root=str(tmp_path/'db')),
        server=SimpleNamespace(binary='binary', source_dir='source'), actions=SimpleNamespace(action_contract='contract'),
        run=SimpleNamespace(max_decisions=10))
    collector.registry = SimpleNamespace(require=lambda: None, budget=budget, state={'updates': 0, 'run_id': 'run'},
        current=SimpleNamespace(files={'inference.json': 'current'}),
        service=SimpleNamespace(bind_collection=lambda *args, **kwargs: None),
        admission=SimpleNamespace(observation={'deadlines': {'execution_bits': deadline_bits(deadline)}}))
    monkeypatch.setattr(collector, '_contexts', lambda plan: {})
    monkeypatch.setattr(collector, '_counters', dict)
    plan = SimpleNamespace(groups=(), bindings={}, linkage={'collection_generation': 0, 'batch_id': 'batch'},
        output=tmp_path/'output', execution_deadline=deadline)
    try:
        with pytest.raises(OSError, match='original') as failed:
            asyncio.run(collector._collect_owned(plan))
        assert failed.value is primary
        if failure_stage == 'pump':
            assert failed.value.cleanup_owner is collector
            assert events == ['writer-acquire', 'evidence-acquire', 'preflight', 'pump',
                'runtime-cancel', 'writer-cancel', 'server-stop', 'writer-close']
            assert collector.server_history[0]['server'] is collector._servers[0]
            if cleanup_failure: assert len(primary.__notes__) == 2
        else:
            assert events == ['writer-acquire', 'evidence-acquire', 'preflight']
            assert not hasattr(primary, 'cleanup_owner') and not collector._servers
        assert collector._servers is pump.servers and collector.server_history is pump.server_history
        assert not collector._source_control_charge.closed and budget.used > 0
    finally:
        collector._source_control_charge.close()
    assert budget.used == 0
