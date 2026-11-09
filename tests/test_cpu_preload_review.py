"""Concrete independent-review regression cases for pytest CPU launch ownership."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sysconfig
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gf_common.card_instance import ActionConfigV3, card_instance_contract
from gf_common.domain import ContractError
from gf_simulator_client.structured_resources import ReservationLedger, WorkerReservation
from tests.cpu_preload import CURRENT_TEMPLATE, CpuTemplate, eligible_configuration
from tests.test_cpu_preload import configuration_stub, requires_preload, run, worker
from tests.test_cpu_preload import service as service  # noqa: PLC0414 - reuse isolated fixture.


@requires_preload
@pytest.mark.parametrize('fault,status', [('closed_exit7', 7), ('closed_sigkill', -signal.SIGKILL)])
def test_actual_bad_exit_after_closed_ack_stays_unhealthy_and_never_signals_reaped_group(tmp_path, monkeypatch, fault, status):
    owner = CpuTemplate(tmp_path, fault=fault)
    owner.start(time.monotonic() + 15)
    signalled = []
    def forbidden(*args):
        signalled.append(args)
        pytest.fail('cannot signal group after actual template reap')
    monkeypatch.setattr(os, 'killpg', forbidden)
    with pytest.raises(RuntimeError, match='actual exit') as raised:
        run(owner.close())
    assert owner.failure is raised.value and owner._leader_reaped and owner.process.returncode == status
    assert not Path(f'/proc/{owner.process.pid}').exists() and not signalled
    deadline = owner._close_deadline
    with pytest.raises(RuntimeError) as repeated:
        run(owner.close(deadline=time.monotonic() + 100))
    assert repeated.value is raised.value and owner._close_deadline == deadline and not signalled


@pytest.mark.parametrize('failure_point', ['channel_setup', 'stderr_open'])
def test_start_acquisition_failure_closes_fds_and_caches_original_without_retry(tmp_path, monkeypatch, failure_point):
    owner = CpuTemplate(tmp_path)
    failure = OSError('owned startup acquisition failed')
    before = set(os.listdir('/proc/self/fd'))
    actual_pair, actual_open = socket.socketpair, Path.open
    attempts = []
    def pair(*args):
        attempts.append(True)
        parent, child = actual_pair(*args)
        if failure_point == 'channel_setup':
            class PartialChannel:
                def setblocking(self, value):
                    raise failure
                def close(self):
                    parent.close()
                def fileno(self):
                    return parent.fileno()
            return PartialChannel(), child
        return parent, child
    def open_log(path, *args, **kwargs):
        if path.name == 'template.stderr.log':
            raise failure
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(socket, 'socketpair', pair)
    monkeypatch.setattr(Path, 'open', open_log)
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('setup failure must precede process creation'))
    with pytest.raises(OSError) as raised:
        owner.start(time.monotonic() + 10)
    assert raised.value is failure and owner.failure is failure
    assert owner.process is None and owner.state == 'POISONED' and owner.channel.fileno() == -1
    assert set(os.listdir('/proc/self/fd')) == before
    with pytest.raises(OSError) as repeated:
        owner.start(time.monotonic() + 100)
    assert repeated.value is failure and attempts == [True]
    # Restore socketpair before asyncio creates its own self-pipe.
    monkeypatch.undo()
    with pytest.raises(OSError) as closing:
        run(owner.close())
    assert closing.value is failure and set(os.listdir('/proc/self/fd')) == before


@requires_preload
def test_attestation_past_launch_deadline_authoritatively_does_not_fork(tmp_path):
    owner = CpuTemplate(tmp_path, fault='delayed_attest')
    owner.start(time.monotonic() + 15)
    ledger = ReservationLedger()
    try:
        with pytest.raises(TimeoutError) as raised:
            worker(owner, ledger, timeout=.1)
        startup = getattr(raised.value, '_structured_startup_owner', None)
        if startup is not None:
            assert startup.process is None
            run(startup.close())
            assert startup.process is None and startup.lease.released
        assert not ledger.workers and not owner.records
        assert owner.health(time.monotonic() + 2)['healthy']
        audit = (tmp_path / 'audit.jsonl').read_text()
        assert 'actor_entry' not in audit and owner.launches == 1
    finally:
        run(owner.close())


@requires_preload
@pytest.mark.parametrize('operation', ['launch', 'health', 'retire', 'close'])
def test_late_real_control_reply_cannot_become_success_and_keeps_actual_ownership(tmp_path, monkeypatch, operation):
    owner = CpuTemplate(tmp_path)
    owner.start(time.monotonic() + 15)
    actual_dispatch, actual_send, actual_retire = owner._dispatch, owner._send, owner.retire
    bounds, replies = {}, []
    target = {'launch': 'launched', 'health': 'health', 'retire': 'retired', 'close': 'closed'}[operation]
    def dispatch(value, descriptors):
        result = actual_dispatch(value, descriptors)
        if value['op'] == target:
            replies.append(value)
            time.sleep(max(0, bounds['deadline'] - time.monotonic()) + .02)
        return result
    def send(op, **fields):
        if op == 'launch':
            bounds['deadline'] = fields['deadline']
        return actual_send(op, **fields)
    def retire(record, deadline):
        bounds['deadline'] = deadline
        return actual_retire(record, deadline)
    monkeypatch.setattr(owner, '_dispatch', dispatch)
    monkeypatch.setattr(owner, '_send', send)
    monkeypatch.setattr(owner, 'retire', retire)
    ledger = ReservationLedger()
    actor = None
    try:
        if operation == 'launch':
            with pytest.raises(TimeoutError) as raised:
                worker(owner, ledger, timeout=.4)
            actor = raised.value._structured_startup_owner
            assert actor.process.pid == replies[0]['pid'] and not actor.lease.creation_pending
            assert not actor.lease.released and owner.failure is None
            monkeypatch.undo()
            run(actor.close())
            assert actor.lease.released and actor.process.poll() == -signal.SIGTERM
        elif operation == 'health':
            bounds['deadline'] = time.monotonic() + .4
            with pytest.raises(TimeoutError) as raised:
                owner.health(bounds['deadline'])
            assert owner.failure is raised.value
        elif operation == 'retire':
            # The launch delay must not apply during setup for this case.
            actor = worker(owner, ledger)
            with pytest.raises(TimeoutError) as raised:
                run(actor.close(timeout=.4))
            assert actor.lease.released and actor.process.poll() == -signal.SIGTERM
            assert owner.failure is raised.value and owner.containment['physical_group_gone']
        else:
            bounds['deadline'] = time.monotonic() + .4
            with pytest.raises(TimeoutError) as raised:
                run(owner.close(deadline=bounds['deadline']))
            assert owner.failure is raised.value and owner._leader_reaped
        assert len(replies) == 1 and not ledger.workers
    finally:
        monkeypatch.undo()
        if owner.failure is None:
            run(owner.close())
        else:
            first = owner.failure
            with pytest.raises(type(first)) as closing:
                run(owner.close())
            assert closing.value is first


@requires_preload
def test_late_actual_template_wait_caches_failure_without_group_signal(tmp_path, monkeypatch):
    owner = CpuTemplate(tmp_path)
    owner.start(time.monotonic() + 15)
    actual = owner.process.wait
    deadline = time.monotonic() + .4
    def delayed_wait(timeout):
        result = actual(timeout=timeout)
        time.sleep(max(0, deadline - time.monotonic()) + .02)
        return result
    monkeypatch.setattr(owner.process, 'wait', delayed_wait)
    monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('late successful wait cannot permit another group kill'))
    with pytest.raises(TimeoutError) as raised:
        run(owner.close(deadline=deadline))
    assert owner.failure is raised.value and owner._leader_reaped and owner.process.returncode == 0
    assert not Path(f'/proc/{owner.process.pid}').exists()
    with pytest.raises(TimeoutError) as repeated:
        run(owner.close())
    assert repeated.value is raised.value


@requires_preload
@pytest.mark.parametrize('phase', ['normal_close', 'containment'])
def test_actual_wait_then_raises_remembers_reap_and_forbids_later_signalling(tmp_path, monkeypatch, phase):
    owner = CpuTemplate(tmp_path)
    owner.start(time.monotonic() + 15)
    actual_wait = owner.process.wait
    actual_kill, actual_killpg = os.kill, os.killpg
    wait_error = OSError('actual wait reaped then raised')
    original = RuntimeError('original control loss before containment') if phase == 'containment' else wait_error
    signals, late_signals = [], []
    if phase == 'containment':
        owner.poison(original)
    def wait(timeout):
        result = actual_wait(timeout=timeout)
        assert result == owner.process.returncode
        raise wait_error
    def kill(pid, number):
        if owner.process.returncode is not None:
            late_signals.append(('pid', pid, number))
            pytest.fail('cannot signal template PID after actual reap')
        signals.append(('pid', pid, number))
        return actual_kill(pid, number)
    def killpg(pgid, number):
        if owner.process.returncode is not None:
            late_signals.append(('group', pgid, number))
            pytest.fail('cannot signal template group after actual reap')
        signals.append(('group', pgid, number))
        return actual_killpg(pgid, number)
    monkeypatch.setattr(owner.process, 'wait', wait)
    monkeypatch.setattr(os, 'kill', kill)
    monkeypatch.setattr(os, 'killpg', killpg)
    with pytest.raises(type(original)) as raised:
        run(owner.close())
    assert raised.value is original and owner.failure is original and owner._leader_reaped
    assert owner.process.returncode == (-signal.SIGKILL if phase == 'containment' else 0)
    assert not Path(f'/proc/{owner.process.pid}').exists() and not late_signals
    assert signals == ([('pid', owner.process.pid, signal.SIGSTOP),
        ('group', owner.process.pid, signal.SIGKILL)] if phase == 'containment' else [])
    deadline = owner._close_deadline
    with pytest.raises(type(original)) as repeated:
        run(owner.close(deadline=time.monotonic() + 100))
    assert repeated.value is original and owner._close_deadline == deadline and not late_signals


@requires_preload
@pytest.mark.parametrize('phase', ['prepare_failure', 'before_send', 'no_fork'])
def test_no_child_cancel_failure_keeps_original_startup_owner_and_finishes_once(tmp_path, monkeypatch, phase):
    service = CpuTemplate(tmp_path, fault='no_fork' if phase == 'no_fork' else '')
    service.start(time.monotonic() + 15)
    original = OSError('original local creation failure')
    cleanup = OSError('first lease cancellation failed')
    lease_cancel = WorkerReservation.cancel_before_spawn
    actual_launch = service.launch
    errors, attempts = [], []
    def cancel(lease):
        attempts.append(lease)
        if len(attempts) == 1:
            raise cleanup
        return lease_cancel(lease)
    def launch(*args):
        try:
            if phase == 'before_send':
                raise original
            return actual_launch(*args)
        except BaseException as error:
            errors.append(error)
            raise
    if phase == 'prepare_failure':
        def prepare(*args):
            errors.append(original)
            raise original
        monkeypatch.setattr(service, 'prepare', prepare)
    else:
        monkeypatch.setattr(service, 'launch', launch)
    monkeypatch.setattr(WorkerReservation, 'cancel_before_spawn', cancel)
    ledger = ReservationLedger()
    try:
        with pytest.raises((OSError, RuntimeError, TimeoutError)) as raised:
            worker(service, ledger)
        actor = raised.value._structured_startup_owner
        assert errors[0] is raised.value and cleanup is not raised.value
        assert any('first lease cancellation failed' in note for note in raised.value.__notes__)
        assert ledger.workers[actor.lease.key] is actor.lease and actor.lease._cleanup_owner is actor
        with pytest.raises(ContractError):
            ledger.begin_update(0)
        run(actor.close())
        assert len(attempts) == 2 and attempts[0] is attempts[1]
        assert actor.lease.released and not ledger.workers and not service.records
        assert service.health(time.monotonic() + 2)['healthy']
    finally:
        monkeypatch.undo()
        run(service.close())


@requires_preload
def test_stalled_health_is_terminal_and_collector_finally_contains_it(tmp_path, monkeypatch):
    import gf_cli
    from tests.structured_ppo_support import component_http
    owner = CpuTemplate(tmp_path, fault='stalled_health')
    token = CURRENT_TEMPLATE.set(owner)
    original = []
    async def experiment(*args, _pytest_worker_finalizer, **kwargs):
        try:
            await _pytest_worker_finalizer()
        except TimeoutError as error:
            original.append(error)
            raise
        pytest.fail('stalled health cannot publish')
    monkeypatch.setattr(gf_cli, '_run_experiment', experiment)
    collector, *_ = component_http(monkeypatch, configuration_stub(), preload_cpu=True)
    # Warmup has its execution budget; only health/final cleanup use this small wall.
    plan = type('Plan', (), {'execution_deadline': time.monotonic() + 15,
        'wall_deadline': time.monotonic() + 1.5})()
    ledger = type('Ledger', (), {'workers': {}, 'learner': object()})()
    try:
        with pytest.raises(TimeoutError) as raised:
            run(collector(plan, ledger))
        assert original == [raised.value] and owner.failure is raised.value
        assert owner.containment['physical_group_gone'] and owner._leader_reaped
        with pytest.raises(TimeoutError) as restarting:
            owner.start(time.monotonic() + 100)
        assert restarting.value is raised.value
    finally:
        CURRENT_TEMPLATE.reset(token)


@requires_preload
def test_actual_template_paths_exclude_checkout_until_actor_fork(service):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(path).resolve() for path in service.ready['search_paths']]
    assert not {root, root / 'src', root / 'tests'} & set(paths)
    assert Path(sysconfig.get_path('purelib')).resolve() in paths
    assert service.ready['native_threads'] == service.ready['python_threads'] == 1
    assert service.ready['cuda_initialized'] is False


@pytest.mark.parametrize('mutation', ['contract', 'actions3', 'rust', 'reader'])
def test_ineligible_config_refuses_before_warmup_and_wrapper_does_not_opt_in(tmp_path, monkeypatch, mutation):
    from tests import test_structured_ppo as module
    from tests.structured_ppo_support import component_http
    owner = CpuTemplate(tmp_path)
    config = configuration_stub()
    if mutation == 'contract':
        object.__setattr__(config.config.actions, 'action_contract', card_instance_contract())
    elif mutation == 'actions3':
        config.config.actions = ActionConfigV3(action_contract=card_instance_contract())
    elif mutation == 'rust':
        config.config.preprocessing.backend = 'rust'
    else:
        config.config.preprocessing.reader = 'card-instance-v1'
    token = CURRENT_TEMPLATE.set(owner)
    try:
        with pytest.raises(ValueError, match='ordinary CPU'):
            component_http(monkeypatch, config, preload_cpu=True)
        options = []
        monkeypatch.setattr(module, 'fresh_component_http', lambda *args, **kwargs: options.append(kwargs))
        module.component_http(monkeypatch, config)
        assert options[0]['preload_cpu'] is False and owner.process is None
    finally:
        CURRENT_TEMPLATE.reset(token)
        run(owner.close())


def test_valid_custom_action_caps_keep_engineering_preload_eligibility(tmp_path, monkeypatch):
    from tests.structured_ppo_support import component_http
    config = configuration_stub()
    config.config.actions = replace(config.config.actions, max_expansions=131072, max_candidates=16384)
    assert eligible_configuration(config)
    owner = CpuTemplate(tmp_path)
    token = CURRENT_TEMPLATE.set(owner)
    try:
        component_http(monkeypatch, config, preload_cpu=True)
        assert owner.process is None
    finally:
        CURRENT_TEMPLATE.reset(token)
        run(owner.close())
