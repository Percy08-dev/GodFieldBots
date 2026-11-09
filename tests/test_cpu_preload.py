"""Owned real CPU forks, status loss, admission and unchanged cold defaults."""
from __future__ import annotations

import array
import asyncio
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gf_common.action_contract import engineering_contract
from gf_common.domain import ContractError
from gf_common.structured import ActionConfigV2, EventBank
from gf_models.structured_baselines import make_structured_policy
from gf_simulator_client.framing import read_frame, write_frame
from gf_simulator_client.structured_resources import ReservationLedger
from gf_simulator_client.structured_worker import StructuredParticipantWorker
from tests.cpu_preload import (
    CURRENT_LAUNCH,
    CURRENT_TEMPLATE,
    CpuPreloadMixin,
    CpuTemplate,
    LaunchContext,
    supported,
)
from tests.cpu_preload_actor import AdmissionInput
from tests.cpu_preload_protocol import FACTORY, MAX_PACKET, receive
from tests.test_structured_protocol import action_settings, neural_component, own_frame

requires_preload = pytest.mark.skipif(not supported() or importlib.util.find_spec('torch') is None,
                                    reason='Linux and optional pinned Torch required')
CONTROL_FAILURES = (EOFError, ConnectionError, RuntimeError, TimeoutError)


class PreloadedWorker(CpuPreloadMixin, StructuredParticipantWorker):
    pass


def worker(service, ledger, *, seed=17, model='random', options=None, timeout=2):
    token = CURRENT_LAUNCH.set(LaunchContext(service, time.monotonic() + 20, time.monotonic() + 30, 1))
    try:
        return PreloadedWorker(make_structured_policy, model, seed, action_settings(),
            ledger=ledger, policy_options=options, timeout=timeout)
    finally:
        CURRENT_LAUNCH.reset(token)


def configuration_stub():
    return SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(device='cpu'),
        model=SimpleNamespace(model_id='mlp', inference=SimpleNamespace(device='cpu')),
        actions=ActionConfigV2(action_contract=engineering_contract()),
        preprocessing=SimpleNamespace(backend='python', reader='structured-v1'),
        run=SimpleNamespace(report_reserve_seconds=0)))


def run(coroutine):
    async def bounded():
        async with asyncio.timeout(30):
            return await coroutine
    return asyncio.run(bounded())


@pytest.fixture
def service(tmp_path):
    owner = CpuTemplate(tmp_path / 'template')
    owner.start(time.monotonic() + 15)
    try:
        yield owner
    finally:
        run(owner.close())


def test_pending_worker_reservation_cannot_cancel_or_reap():
    ledger = ReservationLedger()
    lease = ledger.reserve_worker(0)
    lease.creation_pending = True
    with pytest.raises(ContractError):
        lease.cancel_before_spawn()
    with pytest.raises(ContractError):
        lease.reaped(SimpleNamespace(pid=123, poll=lambda: pytest.fail('pending guard must run first')))
    with pytest.raises(ContractError):
        ledger.begin_update(0)
    lease.spawned(SimpleNamespace(pid=123))
    assert not lease.creation_pending and lease.pid == 123


def test_unstarted_owner_import_and_binding_never_import_torch(tmp_path):
    script = """
import asyncio
import sys
from tests.cpu_preload import CURRENT_TEMPLATE, CpuTemplate
assert 'torch' not in sys.modules
owner = CpuTemplate(sys.argv[1])
token = CURRENT_TEMPLATE.set(owner)
assert owner.process is None and 'torch' not in sys.modules
asyncio.run(owner.close())
CURRENT_TEMPLATE.reset(token)
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-E', '-c', script, str(tmp_path)],
        capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_constructing_binding_and_rejecting_helper_routes_never_warms(tmp_path, monkeypatch):
    from tests.structured_ppo_support import component_http
    from tests.test_structured_ppo import component_http as module_helper
    owner = CpuTemplate(tmp_path)
    cpu = configuration_stub()
    with pytest.raises(ValueError, match='supported bound owner'):
        component_http(monkeypatch, cpu, preload_cpu=True)
    token = CURRENT_TEMPLATE.set(owner)
    try:
        for flags in ({'source_ops': object()}, {'validation_context': object()}):
            with pytest.raises(ValueError, match='ordinary CPU'):
                component_http(monkeypatch, cpu, preload_cpu=True, **flags)
        for side in ('training', 'inference'):
            target = cpu.config.training if side == 'training' else cpu.config.model.inference
            target.device = 'cuda:0'
            with pytest.raises(ValueError, match='ordinary CPU'):
                component_http(monkeypatch, cpu, preload_cpu=True)
            target.device = 'cpu'
        module_helper(monkeypatch, cpu)
        assert owner.process is None and owner.state == 'UNSTARTED'
    finally:
        CURRENT_TEMPLATE.reset(token)
        run(owner.close())


@pytest.mark.parametrize('mutation', ['factory', 'cuda', 'model'])
def test_stream_guards_every_later_begin_before_factory_import(mutation):
    initial = {'factory': FACTORY, 'model_id': 'random', 'policy_options': None}
    changed = {**initial}
    if mutation == 'factory':
        changed['factory'] = {'module': 'unimported.private.factory', 'name': 'bad'}
    elif mutation == 'cuda':
        changed['policy_options'] = {'inference': {'device': 'cuda:0'}, 'cuda_visible_devices': '0'}
    else:
        changed['model_id'] = 'lightgbm'
    source = io.BytesIO()
    write_frame(source, {'operation': 'begin', 'payload': initial})
    write_frame(source, {'operation': 'end'})
    write_frame(source, {'operation': 'begin', 'payload': changed})
    source.seek(0)
    guarded = AdmissionInput(source, 'pooled')
    assert read_frame(guarded)['payload'] == initial
    assert read_frame(guarded) == {'operation': 'end'}
    with pytest.raises(ValueError, match='CPU|factory/model'):
        read_frame(guarded)


@pytest.mark.parametrize('packet', [b'{"op":1,"op":2}', b'{' , b'x' * (MAX_PACKET + 1)])
def test_bad_control_closes_all_received_rights(packet):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    reader, writer = os.pipe()
    try:
        before = set(os.listdir('/proc/self/fd'))
        child.sendmsg([packet], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [reader, writer]))])
        with pytest.raises((ValueError, UnicodeError)):
            receive(parent)
        assert set(os.listdir('/proc/self/fd')) == before
    finally:
        parent.close()
        child.close()
        os.close(reader)
        os.close(writer)


@requires_preload
def test_four_actual_reservations_fresh_pids_fd_eof_and_real_reap(service):
    async def check():
        ledger = ReservationLedger()
        actors = [worker(service, ledger, seed=number) for number in range(4)]
        try:
            assert len(ledger.workers) == len(service.records) == 4
            assert len({actor.process.pid for actor in actors}) == 4
            assert all(actor.process.pid != service.process.pid for actor in actors)
            with pytest.raises(ContractError, match='budget'):
                worker(service, ledger)
            await asyncio.gather(*(actor.initialize() for actor in actors))
            assert all(actor.initialized for actor in actors)
            # Other actors/template must not retain a sibling's stdin writer.
            actors[0].process.stdin.close()
            deadline = time.monotonic() + 2
            while actors[0].process.poll() is None:
                assert time.monotonic() < deadline
                await asyncio.sleep(.005)
            assert actors[0].process.returncode == 0
            assert all(actor.process.poll() is None for actor in actors[1:])
        finally:
            for actor in actors:
                await actor.close()
        assert not ledger.workers and not service.records
        assert all(actor.lease.released and actor.process.poll() is not None for actor in actors)
        assert service.health(time.monotonic() + 2)['healthy']
        entries = [json.loads(line) for line in (service.directory / 'audit.jsonl').read_text().splitlines()]
        entries = [row for row in entries if row['event'] == 'actor_entry']
        assert len(entries) == 4
        assert all(row['open_fds'] == [0, 1, 2] and row['inherited_torch']
            and row['native_threads'] == row['python_threads'] == 1 and row['cuda_initialized'] is False
            and row['template_pid'] == row['pgid'] == row['sid'] == service.process.pid for row in entries)
    run(check())


@requires_preload
@pytest.mark.parametrize('fail_collection', [False, True])
def test_collector_starts_before_activity_and_checks_publication_or_joins_poison(tmp_path, monkeypatch, fail_collection):
    import gf_cli
    from tests.structured_ppo_support import component_http
    owner = CpuTemplate(tmp_path / 'template')
    token = CURRENT_TEMPLATE.set(owner)
    original_error = LookupError('original collection failure')
    observed = []
    async def experiment(*args, _pytest_worker_finalizer, **kwargs):
        assert owner.state == 'RUNNING' and owner.process is not None
        observed.append(owner.process.pid)
        if fail_collection:
            owner.poison(RuntimeError('isolated collection control loss'))
            raise original_error
        receipt = await _pytest_worker_finalizer()
        assert receipt['cpu_preload']['healthy'] and receipt['cpu_preload']['unretired_tickets'] == 0
        assert await _pytest_worker_finalizer() is receipt
        return tmp_path, {'comparison_valid': False}
    monkeypatch.setattr(gf_cli, '_run_experiment', experiment)
    config = configuration_stub()
    try:
        collector, scripts, workers, ledgers = component_http(monkeypatch, config, preload_cpu=True)
        assert owner.process is None
        plan = SimpleNamespace(execution_deadline=time.monotonic() + 15, wall_deadline=time.monotonic() + 20)
        ledger = SimpleNamespace(workers={}, learner=object())
        if fail_collection:
            with pytest.raises(LookupError) as raised:
                run(collector(plan, ledger))
            assert raised.value is original_error
            assert owner.containment['physical_group_gone']
        else:
            run(collector(plan, ledger))
            run(collector(plan, ledger))
            assert observed[0] == observed[1] and owner.state == 'RUNNING'
        assert not scripts and not workers and not ledgers
    finally:
        CURRENT_TEMPLATE.reset(token)
        if fail_collection:
            with pytest.raises(RuntimeError, match='isolated collection control loss'):
                run(owner.close())
        else:
            run(owner.close())


@requires_preload
def test_authoritative_no_fork_releases_only_precreation_lease(tmp_path):
    service = CpuTemplate(tmp_path, fault='no_fork')
    service.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    try:
        with pytest.raises(RuntimeError, match='authoritatively refused') as raised:
            worker(service, ledger)
        assert not hasattr(raised.value, '_structured_startup_owner')
        assert not ledger.workers and not service.records and service.launches == 1
        ledger.begin_update(0)
        ledger.finish_update()
    finally:
        run(service.close())


@requires_preload
@pytest.mark.parametrize('phase', ['before_fork', 'after_fork_before_registration',
    'after_registration_before_ack', 'after_waitpid_before_status'])
def test_status_loss_holds_same_lease_and_quarantines_even_after_containment(tmp_path, phase):
    service = CpuTemplate(tmp_path, fault=phase)
    service.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    try:
        owner = worker(service, ledger, timeout=.5)
    except CONTROL_FAILURES as original:
        owner = original._structured_startup_owner
    else:
        assert phase == 'after_waitpid_before_status'
        with pytest.raises(CONTROL_FAILURES):
            run(owner.initialize())
    lease = owner.lease
    assert lease._cleanup_owner is owner and ledger.workers[lease.key] is lease
    with pytest.raises(CONTROL_FAILURES):
        run(owner.close())
    assert service.failure is not None and service.containment['physical_group_gone']
    assert service.containment['events'] == ['stop_requested', 'stop_or_death_confirmed_wnowait',
        'group_killed_while_leader_held', 'template_reaped_after_actors_gone']
    assert not lease.released and ledger.workers[lease.key] is lease
    with pytest.raises(ContractError):
        ledger.begin_update(0)
    with pytest.raises(CONTROL_FAILURES):
        service.health(time.monotonic() + 1)
    with pytest.raises(CONTROL_FAILURES):
        service.start(time.monotonic() + 15)
    deadline = service._close_deadline
    with pytest.raises(CONTROL_FAILURES):
        run(service.close(deadline=time.monotonic() + 100))
    assert service._close_deadline == deadline


@requires_preload
@pytest.mark.parametrize('fault', ['bad_control', 'truncated_control', 'stale_control'])
def test_actual_invalid_control_poison_has_no_restart_or_healthy_receipt(tmp_path, fault):
    owner = CpuTemplate(tmp_path, fault=fault)
    try:
        try:
            owner.start(time.monotonic() + 15)
        except (ValueError, EOFError, RuntimeError):
            pass
        else:
            with pytest.raises((ValueError, EOFError, RuntimeError)):
                owner.health(time.monotonic() + 2)
        assert owner.failure is not None
    finally:
        with pytest.raises((ValueError, EOFError, RuntimeError)):
            run(owner.close())
    assert owner.containment['physical_group_gone']


@requires_preload
def test_before_external_creation_local_failure_cancels_only_no_child(service, monkeypatch):
    def failure(*args):
        raise OSError('local pipe creation refused before send')
    monkeypatch.setattr(service, 'prepare', failure)
    ledger = ReservationLedger()
    with pytest.raises(OSError, match='local pipe creation'):
        worker(service, ledger)
    assert not ledger.workers and not service.records and service.launches == 0


@requires_preload
def test_partial_fd_wrapper_failure_preserves_error_and_closes_each_owned_fd_once(service, monkeypatch):
    before = set(os.listdir('/proc/self/fd'))
    actual_open, actual_close, actual_pipe = os.fdopen, os.close, os.pipe
    descriptors, raw_closes, wrapped = [], [], []
    failure = OSError('second FD wrapper failed before any launch')
    def pipe():
        result = actual_pipe()
        descriptors.extend(result)
        return result
    def fdopen(fd, mode, **kwargs):
        if mode == 'rb':
            raise failure
        result = actual_open(fd, mode, **kwargs)
        wrapped.append(result)
        return result
    def close(fd):
        raw_closes.append(fd)
        return actual_close(fd)
    ledger = ReservationLedger()
    with monkeypatch.context() as patch:
        patch.setattr(os, 'pipe', pipe)
        patch.setattr(os, 'fdopen', fdopen)
        patch.setattr(os, 'close', close)
        with pytest.raises(OSError) as raised:
            worker(service, ledger)
    assert raised.value is failure and len(descriptors) == 4
    assert len(wrapped) == 1 and wrapped[0].closed
    # FileIO owns one descriptor; every other descriptor receives one raw close.
    assert sorted(raw_closes) == sorted([descriptors[0], descriptors[2], descriptors[3]])
    assert not ledger.workers and not service.records and service.launches == 0
    assert set(os.listdir('/proc/self/fd')) == before


@requires_preload
@pytest.mark.parametrize('fault', ['', 'after_fork_before_registration'], ids=['publication', 'uncertain_close'])
def test_fresh_forked_16game_one_update_actual_train_publication(tmp_path, monkeypatch, fault):
    from gf_training.structured_ppo import _train_component_ppo
    from gf_training.structured_ppo_source import validate_component_parent
    from tests.structured_ppo_support import component_http, configuration
    loaded = configuration(tmp_path, max_updates=1)
    service = CpuTemplate(tmp_path / 'template', fault=fault)
    token = CURRENT_TEMPLATE.set(service)
    parent = tmp_path / 'parent'
    try:
        collector, scripts, workers, ledgers = component_http(monkeypatch, loaded,
            pool_cpu=False, parallel_ready=True, preload_cpu=True)
        if fault:
            with pytest.raises((ContractError, RuntimeError, OSError, EOFError)):
                run(_train_component_ppo(loaded, collector, parent))
            state = json.loads((parent / 'run.json').read_text())
            assert state['status'] == 'failed' and state['updates'] == 0
            assert service.failure is not None and service.containment['physical_group_gone']
            assert ledgers and any(ledger.workers for ledger in ledgers)
            for manifest in parent.glob('collections/**/manifest.json'):
                value = json.loads(manifest.read_text())
                assert value.get('pytest_worker_lifecycle', {}).get('closed') is not True
                report = manifest.parent / 'report.json'
                if report.exists():
                    assert json.loads(report.read_text()).get('comparison_valid') is not True
        else:
            result = run(_train_component_ppo(loaded, collector, parent))
            assert result['status'] == 'complete' and result['updates'] == 1
            assert result['accepted_decisions'] == 32 and len(workers) == 32 and len(scripts) == 8
            state, generations = validate_component_parent(parent, max_owned_bytes=2 * 1024**3,
                deadline=time.monotonic() + 20)
            assert state['status'] == 'complete' and len(generations) == 2
            manifest = json.loads((parent / 'collections/batch-000000/manifest.json').read_text())
            receipt = manifest['pytest_worker_lifecycle']
            assert receipt['mode'] == 'pytest-fresh-forked-cpu-v1' and receipt['closed'] is True
            assert receipt['cpu_preload']['healthy'] and receipt['cpu_preload']['unretired_tickets'] == 0
            assert receipt['cpu_preload']['launches'] == 32
            assert len({actor.process.pid for actor in workers}) == 32
            assert all(actor.closed and actor.lease.released and actor.process.poll() is not None
                and (actor.io_thread is None or not actor.io_thread.is_alive()) for actor in workers)
            assert all(not ledger.workers and ledger.learner is None for ledger in ledgers)
    finally:
        CURRENT_TEMPLATE.reset(token)
        if fault:
            with pytest.raises(CONTROL_FAILURES):
                run(service.close())
        else:
            run(service.close())


@requires_preload
@pytest.mark.parametrize('mutation', ['factory', 'cuda'])
def test_real_pool_later_begin_rejects_before_original_import(tmp_path, mutation):
    from tests.pooled_structured_worker import (
        CURRENT_POOL,
        CURRENT_VIEWER,
        CpuParticipantPool,
        PooledStructuredParticipantWorker,
    )
    service = CpuTemplate(tmp_path / 'template')
    service.start(time.monotonic() + 15)
    async def check():
        ledger = ReservationLedger()
        pool = CpuParticipantPool(ledger)
        def logical():
            launch = CURRENT_LAUNCH.set(LaunchContext(service, time.monotonic() + 10, time.monotonic() + 20, 1))
            pooled, viewer = CURRENT_POOL.set(pool), CURRENT_VIEWER.set(1)
            try:
                return PooledStructuredParticipantWorker(make_structured_policy, 'random', 17,
                    action_settings(), ledger=ledger)
            finally:
                CURRENT_VIEWER.reset(viewer)
                CURRENT_POOL.reset(pooled)
                CURRENT_LAUNCH.reset(launch)
        first = logical()
        await first.initialize()
        await first.close()
        second = logical()
        assert second.process is first.process
        if mutation == 'factory':
            second.initial['factory'] = {'module': 'unimported.private.factory', 'name': 'bad'}
        else:
            second.initial['policy_options'] = {'inference': {'device': 'cuda:0'}, 'cuda_visible_devices': '0'}
        with pytest.raises((EOFError, ContractError, RuntimeError)):
            await second.initialize()
        assert second.process.poll() == 1 and second.lease.released and not ledger.workers
        with pytest.raises((RuntimeError, TimeoutError)):
            await pool.close()
        # Actor admission failure must not cause an arbitrary factory import in its stderr.
        text = (service.directory / 'template.stderr.log').read_text()
        assert 'preloaded actor requires' in text and 'ModuleNotFoundError' not in text
        assert service.failure is not None and service.containment['physical_group_gone']
    try:
        run(check())
    finally:
        with pytest.raises(TimeoutError):
            run(service.close())


@requires_preload
def test_lost_pid_ack_resolves_actual_child_before_original_cleanup(tmp_path):
    service = CpuTemplate(tmp_path, fault='lost_ack')
    service.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    try:
        with pytest.raises(TimeoutError) as raised:
            worker(service, ledger, timeout=.05)
        owner = raised.value._structured_startup_owner
        assert owner.process is None and owner.lease.creation_pending
        assert owner._preload_record.state == 'SENT_PENDING'
        run(owner.close())
        assert owner.process.pid > 0 and owner.process.poll() == -15
        assert not owner.lease.creation_pending and owner.lease.released
        assert not ledger.workers and not service.records
        assert service.failure is None
    finally:
        run(service.close())


@requires_preload
def test_actor_before_ready_death_uses_actual_status_and_retires_ticket(tmp_path):
    service = CpuTemplate(tmp_path, fault='actor_pre_ready_death')
    service.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    actor = worker(service, ledger)
    try:
        with pytest.raises((EOFError, BrokenPipeError)):
            run(actor.initialize())
        assert actor.process.poll() == 7
        assert actor.closed and actor.lease.released and not ledger.workers
        assert service.failure is not None and service.containment['physical_group_gone']
        with pytest.raises(TimeoutError):
            service.health(time.monotonic() + 1)
    finally:
        with pytest.raises(TimeoutError):
            run(actor.close())
        with pytest.raises(TimeoutError):
            run(service.close())


@requires_preload
def test_close_repeated_cancel_joins_one_owner_and_keeps_first_deadline(tmp_path, monkeypatch):
    service = CpuTemplate(tmp_path, fault='before_fork')
    service.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    with pytest.raises(CONTROL_FAILURES):
        worker(service, ledger)
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        actual = service._contain
        calls = []
        async def delayed(deadline):
            calls.append(deadline)
            entered.set()
            await release.wait()
            return await actual(deadline)
        monkeypatch.setattr(service, '_contain', delayed)
        first_deadline = time.monotonic() + 1
        closing = asyncio.create_task(service.close(deadline=first_deadline))
        await entered.wait()
        closing.cancel('first cancellation')
        await asyncio.sleep(0)
        for _ in range(3):
            closing.cancel('later cancellation')
            await asyncio.sleep(0)
        assert not closing.done() and service._close_task.cancelling() == 0
        release.set()
        with pytest.raises(asyncio.CancelledError, match='first cancellation'):
            await closing
        assert calls == [first_deadline] and service._close_deadline == first_deadline
        assert service.containment['physical_group_gone']
    run(check())


@requires_preload
@pytest.mark.parametrize('kind', ['mlp', 'gru', 'transformer'])
def test_actual_normal_fork_and_pool_seed_ABA_checkpoint_wire_identity(service, tmp_path, make_input, kind):
    from tests.pooled_structured_worker import (
        CURRENT_POOL,
        CURRENT_VIEWER,
        CpuParticipantPool,
        PooledStructuredParticipantWorker,
    )
    from tests.test_pooled_structured_worker import semantic_wire
    learner, options = neural_component(tmp_path, kind)
    with learner.torch.no_grad():
        next(learner.network.parameters()).add_(.125)
    identity = learner.save(tmp_path / 'other-checkpoint')
    other = {**options, 'checkpoint': str(tmp_path / 'other-checkpoint'), 'expected_manifest_digest': identity['inference_digest']}
    async def check():
        ledger = ReservationLedger()
        pool = CpuParticipantPool(ledger)
        pids, answers = [], []
        try:
            for seed, binding in ((17, options), (29, options), (17, options), (31, other), (17, options)):
                cold = StructuredParticipantWorker(make_structured_policy, kind, seed, action_settings(), ledger=ledger, policy_options=binding)
                forked = worker(service, ledger, seed=seed, model=kind, options=binding)
                launch = CURRENT_LAUNCH.set(LaunchContext(service, time.monotonic() + 20, time.monotonic() + 30, 1))
                pooled, viewer = CURRENT_POOL.set(pool), CURRENT_VIEWER.set(1)
                try:
                    loan = PooledStructuredParticipantWorker(make_structured_policy, kind, seed, action_settings(), ledger=ledger, policy_options=binding)
                finally:
                    CURRENT_VIEWER.reset(viewer)
                    CURRENT_POOL.reset(pooled)
                    CURRENT_LAUNCH.reset(launch)
                actors = (cold, forked, loan)
                wires = [[], [], []]
                for actor, wire in zip(actors, wires, strict=True):
                    original = actor._roundtrip
                    async def tapped(payload, original=original, wire=wire):
                        reply = await original(payload)
                        wire.append((semantic_wire(payload), semantic_wire(reply)))
                        return reply
                    actor._roundtrip = tapped
                bank, choices = EventBank(1, 'same-preload-equivalence'), []
                try:
                    for revision in (0, 2):
                        frame, delta = own_frame(make_input(hand=('bronze-club', 'steel-armor'), learned=('fireball', 'aura')),
                            bank, revision, [{'type': 'action_accepted', 'player': 1}])
                        results = [await actor.select_frame(frame, delta) for actor in actors]
                        assert results[0][0].table == results[1][0].table == results[2][0].table
                        assert results[0][1] == results[1][1] == results[2][1]
                        choices.append(asdict(results[1][1]))
                    assert wires[0] == wires[1] == wires[2]
                    pids.append(forked.process.pid)
                    answers.append(choices)
                finally:
                    for actor in actors:
                        await actor.close()
            assert len(set(pids)) == 5 and answers[0] == answers[2] == answers[4]
        finally:
            receipt = await pool.close()
        assert receipt['physical_workers'] == 1 and receipt['resets'] == 5
        assert not ledger.workers and not service.records
    run(check())
