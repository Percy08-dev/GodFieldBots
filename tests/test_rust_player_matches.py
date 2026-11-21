"""Managed pump ordering/ownership fixtures; no game or Source authority."""
import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gf_common.domain import EpisodeGroup, EpisodeSpec, digest
from gf_training.rust_player_storage import PolicyJournalContext


@dataclass(frozen=True)
class Controls:
    value: int


def pump_fixture(tmp_path, monkeypatch, *, failure=None):
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
            self.started = []
        def start(self, specs):
            self.started.extend(specs); events.append(('start', specs[0]['episode']))
        def next_storage(self, deadline):
            assert deadline == execution
            return self.bundles.pop(0)
        def storage_ack(self, bundle):
            assert bundle.written
            events.append(('ack', bundle.id))
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
    writer = Writer()
    monkeypatch.setattr(module, 'ManagedServer', Server)
    async def inline(function, *args): return function(*args)
    monkeypatch.setattr(module.asyncio, 'to_thread', inline)
    server_config = SimpleNamespace(binary='binary', config='config', startup_seconds=5., shutdown_seconds=7., request_seconds=9.)
    def execute(*, batch_id='batch', output=tmp_path):
        return pump.execute(groups, contexts, writer, server=server_config, actions=actions,
            max_decisions=11, output=output, execution_deadline=execution, cleanup_deadline=cleanup,
            game_concurrency=2, run_id='run', batch_id=batch_id, evidence=Evidence())
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
    assert set(vars(pump)) == {'runtime', 'servers', 'server_history'}


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
    assert set(vars(pump)) == {'runtime', 'servers', 'server_history'}


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
    collector.declaration = {'game_concurrency': 1}
    collector.config = SimpleNamespace(storage=SimpleNamespace(db_root=str(tmp_path/'db')),
        server=SimpleNamespace(binary='binary', source_dir='source'), actions=SimpleNamespace(action_contract='contract'),
        run=SimpleNamespace(max_decisions=10))
    collector.registry = SimpleNamespace(require=lambda: None, budget=budget, state={'updates': 0, 'run_id': 'run'},
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
