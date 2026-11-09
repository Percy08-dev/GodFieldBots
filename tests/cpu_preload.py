"""Explicit pytest-only CPU template ownership, actual actor receipts and quarantine."""
from __future__ import annotations

import array
import asyncio
import os
import select
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from tests.cpu_preload_protocol import (
    RSS_LIMIT_KIB,
    VERSION,
    closed,
    encode,
    integer,
    memory,
    receive,
    validate_initial,
)

CURRENT_TEMPLATE = ContextVar('pytest_cpu_template', default=None)
CURRENT_LAUNCH = ContextVar('pytest_cpu_launch', default=None)


def supported():
    return sys.platform == 'linux' and hasattr(socket, 'SOCK_SEQPACKET') and hasattr(os, 'WNOWAIT')


def eligible_configuration(loaded):
    from gf_common.action_contract import engineering_contract
    from gf_common.structured import ActionConfigV2
    actions = getattr(loaded.config, 'actions', None)
    preparation = getattr(loaded.config, 'preprocessing', None)
    return (type(actions) is ActionConfigV2 and actions.action_contract == engineering_contract()
        and getattr(preparation, 'backend', None) == 'python'
        and getattr(preparation, 'reader', None) == 'structured-v1')


@dataclass(frozen=True)
class LaunchContext:
    service: object
    execution_deadline: float
    cleanup_deadline: float
    viewer: int


class LaunchRecord:
    def __init__(self, service, token, worker, lease, reader, writer, sending):
        self.service, self.token, self.worker, self.lease = service, token, worker, lease
        self.state, self.pid, self.status = 'PREPARED', None, None
        self.facade = ActorProcess(self, writer, reader)
        self.sending = sending
        self.launch_seq = None
        self.retire_seq = None
        self.retired = False

    def close_sending(self):
        for fd in self.sending:
            os.close(fd)
        self.sending = ()

    def register(self, pid):
        integer(pid, 1)
        if self.pid is not None and self.pid != pid:
            raise RuntimeError('pytest CPU actor PID changed')
        if self.pid is None:
            self.pid = pid
            self.worker.process = self.facade
            self.lease.spawned(self.facade)
        if self.status is None:
            self.state = 'KNOWN_RUNNING'


class ActorProcess:
    def __init__(self, record, destination, source):
        self.record = record
        self.stdin, self.stdout = destination, source
        self.stderr = None

    @property
    def pid(self):
        if self.record.pid is None:
            raise RuntimeError('pytest CPU actor PID is not acknowledged')
        return self.record.pid

    @property
    def returncode(self):
        return self.poll()

    def poll(self):
        if self.record.status is not None:
            return self.record.status
        try:
            self.record.service.pump()
        except BaseException:
            if self.record.status is None:
                raise
        return self.record.status

    def terminate(self):
        self.record.service.signal(self.record, signal.SIGTERM)

    def kill(self):
        self.record.service.signal(self.record, signal.SIGKILL)

    def wait(self, timeout=None):
        deadline = time.monotonic() + (2 if timeout is None else max(0, timeout))
        while self.poll() is None:
            self.record.service.wait_readable(deadline)
        return self.record.status


class CpuTemplate:
    control_version = VERSION
    server_entry = 'cpu_preload_server.py'

    def __init__(self, directory, *, fault=''):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fault = fault
        self.process = self.channel = self.log = None
        self.state, self.failure = 'UNSTARTED', None
        self.session = uuid.uuid4().hex
        self.sequence = self.serial = self.launches = 0
        self.records, self.expected, self.answers = {}, {}, {}
        self._close_task = None
        self._close_deadline = None
        self.ready = None
        self.peak_rss_kib = self.peak_pss_kib = 0
        self.peak_actor_rss_kib = self.peak_actor_pss_kib = 0
        self.start_seconds = 0
        self.containment = None
        self._leader_reaped = False

    def poison(self, error):
        if self.failure is None:
            self.failure = error
        self.state = 'POISONED'
        for record in self.records.values():
            if record.status is None and record.state != 'CONFIRMED_NO_CHILD':
                record.state = 'STATUS_LOST'
        return self.failure

    def require_healthy(self):
        if self.failure is not None:
            raise self.failure
        if self.state != 'RUNNING':
            raise RuntimeError('pytest CPU template is not running')
        # Preserve the leader PID until its only final wait, including on healthy paths.
        state = os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if state is not None:
            raise self.poison(RuntimeError('pytest CPU template died; actor status may be lost'))

    def _usage(self, value):
        for key in ('rss_kib', 'pss_kib'):
            integer(value[key])
        if value['rss_kib'] > RSS_LIMIT_KIB:
            raise RuntimeError('pytest CPU template exceeds1GiB RSS soft ceiling')
        self.peak_rss_kib = max(self.peak_rss_kib, value['rss_kib'])
        self.peak_pss_kib = max(self.peak_pss_kib, value['pss_kib'])

    def start(self, deadline):
        if self.state == 'RUNNING':
            self.require_healthy()
            self._check_deadline(deadline)
            return
        if self.state != 'UNSTARTED' or not supported():
            raise self.failure or RuntimeError('unsupported/terminal pytest CPU template')
        if time.monotonic() >= deadline:
            raise TimeoutError('pytest CPU warmup execution deadline expired')
        started = time.monotonic()
        parent = child = None
        start_error = None
        try:
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.channel = parent
            self.channel.setblocking(False)
            environment = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
                'PYTHONHASHSEED': '0', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            root = Path(__file__).resolve().parents[1]
            self.log = (self.directory / 'template.stderr.log').open('wb')
            self.process = subprocess.Popen([sys.executable, '-E', '-S', str(root / 'tests' / self.server_entry),
                str(child.fileno()), sysconfig.get_path('purelib'), str(root), self.session, self.fault,
                str(self.directory / 'audit.jsonl')], env=environment, pass_fds=(child.fileno(),),
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.log)
            self.state = 'STARTING'
            while self.ready is None:
                self.wait_readable(deadline, starting=True)
            self.state = 'RUNNING'
            self.start_seconds = time.monotonic() - started
            self.require_healthy()
            self._check_deadline(deadline)
        except BaseException as error:
            start_error = error
            self.poison(error)
            if self.process is None:
                for handle in (parent, self.log):
                    if handle is not None:
                        try:
                            handle.close()
                        except BaseException as cleanup:  # noqa: BLE001 - startup error remains primary.
                            error.add_note(f'pytest CPU startup descriptor cleanup: {cleanup}')
            raise
        finally:
            if child is not None:
                try:
                    child.close()
                except BaseException as error:  # noqa: BLE001 - retain startup failure and descriptor ownership.
                    if start_error is None:
                        raise self.poison(error)
                    start_error.add_note(f'pytest CPU startup child descriptor cleanup: {error}')

    def _send(self, op, *, token=None, descriptors=(), **fields):
        self.require_healthy()
        self.sequence += 1
        seq = self.sequence
        value = {'version': self.control_version, 'session': self.session, 'seq': seq, 'op': op, **fields}
        if token is not None:
            value['token'] = token
        if op != 'signal':
            self.expected[seq] = (op, token)
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', descriptors))] if descriptors else []
        try:
            packet = encode(value)
            if self.channel.sendmsg([packet], ancillary) != len(packet):
                raise RuntimeError('partial pytest CPU control send')
        except BaseException as error:
            self.poison(error)
            raise
        return seq

    def _dispatch(self, value, descriptors):
        if descriptors:
            for fd in descriptors:
                os.close(fd)
            raise ValueError('unexpected pytest CPU response descriptors')
        if (type(value) is not dict or value.get('version') != self.control_version or value.get('session') != self.session
                or type(value.get('seq')) is not int):
            raise ValueError('pytest CPU response version/session/sequence mismatch')
        op, seq = value.get('op'), value['seq']
        common = {'version', 'session', 'op', 'seq'}
        if op == 'ready':
            closed(value, common | {'pid', 'torch_version', 'native_threads', 'python_threads',
                'cuda_initialized', 'preload_seconds', 'rss_kib', 'pss_kib', 'search_paths'})
            if (seq != 0 or self.state != 'STARTING' or self.ready is not None
                    or value['pid'] != self.process.pid or value['torch_version'] not in ('2.14.1+cpu', '2.14.1+cu132')
                    or value['native_threads'] != 1 or value['python_threads'] != 1 or value['cuda_initialized'] is not False):
                raise ValueError('invalid pytest CPU template readiness')
            root = Path(__file__).resolve().parents[1]
            forbidden = {root, root / 'src', root / 'tests'}
            if (type(value['search_paths']) is not list or not all(type(path) is str for path in value['search_paths'])
                    or any(Path(path).resolve() in forbidden for path in value['search_paths'])
                    or sysconfig.get_path('purelib') not in value['search_paths']):
                raise ValueError('pytest CPU template checkout search path contamination')
            self._usage(value)
            self.ready = value
            return
        if op == 'reaped':
            closed(value, common | {'token', 'pid', 'status'})
            if seq != 0 or type(value['status']) is not int:
                raise ValueError('invalid pytest CPU actual wait receipt')
            record = self.records[integer(value['token'], 1)]
            record.register(value['pid'])
            if record.status is not None and record.status != value['status']:
                raise ValueError('pytest CPU actual status changed')
            record.status, record.state = value['status'], 'KNOWN_REAPED'
            return
        if seq not in self.expected:
            raise ValueError('unmatched pytest CPU response sequence')
        requested, token = self.expected.pop(seq)
        if op in ('launched', 'refused') and requested in ('launch', 'resolve'):
            closed(value, common | ({'token', 'pid'} if op == 'launched' else {'token', 'reason'}))
            if value['token'] != token:
                raise ValueError('pytest CPU launch token mismatch')
            record = self.records[token]
            if op == 'launched':
                record.register(value['pid'])
            else:
                if record.pid is not None:
                    raise ValueError('no-child response after real PID')
                record.state = 'CONFIRMED_NO_CHILD'
                record.lease.creation_pending = False
            # Resolved old launch ACKs may no longer be needed; FIFO prevents a later valid ACK.
            if requested == 'resolve' and record.launch_seq is not None:
                self.expected.pop(record.launch_seq, None)
            return
        fields = {'retired': {'token'}, 'health': {'inventory', 'rss_kib', 'pss_kib'},
                  'closed': {'rss_kib', 'pss_kib'}}
        if op not in fields or requested != {'retired': 'retire', 'health': 'health', 'closed': 'close'}[op]:
            raise ValueError('unexpected pytest CPU response')
        closed(value, common | fields[op])
        if op == 'retired' and value['token'] != token:
            raise ValueError('pytest CPU retirement token mismatch')
        if op in ('health', 'closed'):
            self._usage(value)
        self.answers[seq] = value
        if op == 'closed':
            self.state = 'CLOSING_ACK'

    def pump(self, *, starting=False):
        if self.channel is None or self.channel.fileno() < 0:
            raise self.failure or RuntimeError('pytest CPU control is closed')
        try:
            while True:
                try:
                    value, descriptors = receive(self.channel)
                except BlockingIOError:
                    break
                self._dispatch(value, descriptors)
                if self.state == 'CLOSING_ACK':
                    break
            if not starting and self.state != 'CLOSING_ACK':
                self.require_healthy()
        except BaseException as error:  # noqa: BLE001 - poison retains and re-raises first error.
            raise self.poison(error)

    def wait_readable(self, deadline, *, starting=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([self.channel], [], [], remaining)[0]:
            raise TimeoutError('pytest CPU control deadline expired')
        self.pump(starting=starting)
        self._check_deadline(deadline)

    @staticmethod
    def _check_deadline(deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('pytest CPU absolute control deadline expired')

    def prepare(self, worker, lease):
        self.require_healthy()
        if len(self.records) >= 4:
            raise RuntimeError('pytest CPU template has four pending/live/unretired tickets')
        self.serial += 1
        incoming = outgoing = None
        source = destination = None
        try:
            incoming = list(os.pipe())
            outgoing = list(os.pipe())
            destination = os.fdopen(incoming[1], 'wb', buffering=0)
            incoming[1] = None
            source = os.fdopen(outgoing[0], 'rb', buffering=0)
            outgoing[0] = None
            record = LaunchRecord(self, self.serial, worker, lease, source, destination, (incoming[0], outgoing[1]))
            self.records[record.token] = record
            return record
        except BaseException:
            for stream in (source, destination):
                if stream is not None:
                    stream.close()
            for pipes in (incoming, outgoing):
                if pipes is not None:
                    for fd in pipes:
                        if fd is not None:
                            os.close(fd)
            raise

    def launch(self, record, bootstrap, viewer, deadline):
        record.lease.creation_pending = True
        record.state = 'SENT_PENDING'
        try:
            record.launch_seq = self._send('launch', token=record.token, descriptors=record.sending,
                bootstrap=bootstrap, viewer=viewer, deadline=deadline)
        finally:
            record.close_sending()
        self.launches += 1
        while record.state == 'SENT_PENDING':
            self.wait_readable(deadline)
        self._check_deadline(deadline)
        if record.state == 'CONFIRMED_NO_CHILD':
            raise RuntimeError('pytest CPU template authoritatively refused fork')
        if record.state == 'STATUS_LOST':
            raise self.failure
        return record.facade

    def resolve(self, record, deadline):
        self.pump()
        if record.state == 'SENT_PENDING':
            seq = self._send('resolve', token=record.token)
            while seq in self.expected:
                self.wait_readable(deadline)
        self._check_deadline(deadline)

    def signal(self, record, number):
        if record.status is None:
            self._send('signal', token=record.token, signal=number)

    def discard_no_child(self, record):
        if record.state not in ('PREPARED', 'CONFIRMED_NO_CHILD'):
            raise RuntimeError('cannot discard uncertain pytest CPU creation')
        record.close_sending()
        for stream in (record.facade.stdin, record.facade.stdout):
            stream.close()
        self.records.pop(record.token, None)

    def retire(self, record, deadline):
        if record.retired:
            return
        if not record.lease.released or record.status is None:
            raise RuntimeError('pytest CPU retirement requires actual reap and original lease cleanup')
        if record.retire_seq is None:
            record.retire_seq = self._send('retire', token=record.token)
        seq = record.retire_seq
        while seq not in self.answers:
            self.wait_readable(deadline)
        self.answers.pop(seq)
        self.records.pop(record.token)
        record.retired = True
        self._check_deadline(deadline)

    def observe_actor(self, record):
        if record.pid is not None and record.status is None:
            values = memory(record.pid)
            self.peak_actor_rss_kib = max(self.peak_actor_rss_kib, values['rss_kib'])
            self.peak_actor_pss_kib = max(self.peak_actor_pss_kib, values['pss_kib'])

    def health(self, deadline):
        try:
            self.pump()
            seq = self._send('health')
            while seq not in self.answers:
                self.wait_readable(deadline)
            value = self.answers.pop(seq)
            self._check_deadline(deadline)
            if self.records or value['inventory'] != 0:
                raise RuntimeError('pytest CPU template has unretired participant tickets')
            return {'mode': VERSION, 'template_pid': self.process.pid, 'launches': self.launches,
                'setup_seconds': self.start_seconds, 'peak_template_rss_kib': self.peak_rss_kib,
                'peak_template_pss_kib': self.peak_pss_kib, 'peak_actor_rss_kib': self.peak_actor_rss_kib,
                'peak_actor_pss_kib': self.peak_actor_pss_kib, 'rss_soft_limit_kib': RSS_LIMIT_KIB,
                'unretired_tickets': 0, 'healthy': True}
        except BaseException as error:  # noqa: BLE001 - publication health failure is terminal.
            raise self.poison(error)

    async def close(self, *, deadline=float('inf')):
        if self._close_task is None:
            self._close_deadline = min(deadline, time.monotonic() + 2)
            self._close_task = asyncio.create_task(self._close_owned(self._close_deadline))
        first_cancel = None
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError as error:
                first_cancel = first_cancel or error
            except BaseException:  # noqa: BLE001 - shared task result below preserves failure.
                break
        try:
            result = self._close_task.result()
        except BaseException as error:
            if first_cancel is not None:
                raise first_cancel from error
            raise
        if first_cancel is not None:
            raise first_cancel
        return result

    def _wait_template(self, timeout):
        """Remember actual reap even when Popen.wait raises after recording it."""
        try:
            return self.process.wait(timeout=timeout)
        finally:
            # Inspect the stored actual status only; poll would itself consume wait ownership.
            if self.process.returncode is not None:
                self._leader_reaped = True

    async def _close_owned(self, deadline):
        if self.process is None:
            if self.channel is not None:
                self.channel.close()
            if self.log is not None:
                self.log.close()
            self.state = 'CLOSED'
            if self.failure is not None:
                raise self.failure
            return {'closed': True, 'started': False}
        healthy = self.failure is None and not self.records and time.monotonic() < deadline
        if healthy:
            try:
                seq = self._send('close')
                while seq not in self.answers:
                    self.wait_readable(deadline)
                self.answers.pop(seq)
                # A validated close ACK proves no actors/future forks; group signalling is finished.
                returncode = self._wait_template(timeout=max(0, deadline - time.monotonic()))
                if returncode != 0:
                    raise RuntimeError(f'pytest CPU template closed ACK followed by actual exit{returncode}')
                self._check_deadline(deadline)
                self.state = 'CLOSED'
                self.channel.close()
                self.log.close()
                return {'closed': True, 'started': True, 'template_pid': self.process.pid}
            except BaseException as error:  # noqa: BLE001 - retain first error before containment.
                self.poison(error)
        else:
            self.poison(self.failure or RuntimeError('pytest CPU template close with unresolved tickets/deadline'))
        try:
            if not self._leader_reaped:
                await self._contain(deadline)
        finally:
            self.channel.close()
            self.log.close()
        raise self.failure

    async def _contain(self, deadline):
        if self._leader_reaped:
            raise self.failure or RuntimeError('pytest CPU template already reaped; group signalling forbidden')
        leader = self.process.pid
        # Expired caller walls permit only the existing unhealthy .5+.1 emergency allowance.
        emergency = time.monotonic() >= deadline
        if emergency:
            deadline = time.monotonic() + .5
        events = []
        try:
            os.kill(leader, signal.SIGSTOP)
            events.append('stop_requested')
            while True:
                state = os.waitid(os.P_PID, leader, os.WEXITED | os.WSTOPPED | os.WNOHANG | os.WNOWAIT)  # noqa: ASYNC222 - WNOHANG never blocks.
                if state is not None and state.si_code in (os.CLD_STOPPED, os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('pytest CPU spawner stop unconfirmed')
                await asyncio.sleep(.005)
            events.append('stop_or_death_confirmed_wnowait')
            os.killpg(leader, signal.SIGKILL)
            events.append('group_killed_while_leader_held')
            while True:
                state = os.waitid(os.P_PID, leader, os.WEXITED | os.WNOHANG | os.WNOWAIT)  # noqa: ASYNC222 - WNOHANG never blocks.
                actors = self._group_members(leader, exclude_leader=True)
                if state is not None and not actors:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'pytest CPU containment unconfirmed, actor members={actors}')
                await asyncio.sleep(.005)
            self._wait_template(timeout=max(0, deadline - time.monotonic()))
            events.append('template_reaped_after_actors_gone')
            if self._group_members(leader):
                raise RuntimeError('pytest CPU group remains after template reap')
            self.containment = {'physical_group_gone': True, 'events': events, 'emergency': emergency,
                'status_loss_remains_terminal': True}
        except BaseException as error:
            self.containment = {'physical_group_gone': False, 'events': events, 'error': str(error)}
            self.failure.add_note(f'pytest CPU containment: {error}')
            # Even unconfirmed actors retain their owners/leases; never invent a wait receipt.
            raise self.failure from error

    @staticmethod
    def _group_members(leader, *, exclude_leader=False):
        members = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == leader and not (exclude_leader and int(entry.name) == leader):
                    members.append((int(entry.name), fields[0]))
            except (FileNotFoundError, ProcessLookupError):
                pass
        return members


class CpuPreloadMixin:
    """Intercept only test-owned spawn/pending cleanup; known cleanup stays original."""

    def _spawn(self, ledger, residency, arguments, environment):
        context = CURRENT_LAUNCH.get()
        if context is None:
            return super()._spawn(ledger, residency, arguments, environment)
        validate_initial(self.initial)
        bootstrap = 'pooled' if arguments[-2:-1] == ['tests.pooled_structured_bootstrap'] else 'structured'
        if arguments[-1] != 'gf_simulator_client.structured_bootstrap' and bootstrap != 'pooled':
            raise ValueError('unsupported pytest preloaded bootstrap')
        self.process = None
        self._preload_context = context
        self.lease = ledger.reserve_worker(residency)
        self.lease._cleanup_owner = self
        record = None
        try:
            record = context.service.prepare(self, self.lease)
            self._preload_record = record
            deadline = min(context.execution_deadline, time.monotonic() + self.timeout)
            context.service.launch(record, bootstrap, context.viewer, deadline)
        except BaseException as error:
            try:
                if record is None or record.state in ('PREPARED', 'CONFIRMED_NO_CHILD'):
                    if record is not None:
                        context.service.discard_no_child(record)
                    self.lease.cancel_before_spawn()
            except BaseException as cleanup:  # noqa: BLE001 - preserve initiating creation error.
                if cleanup is not error:
                    error.add_note(f'pytest CPU no-child cleanup: {type(cleanup).__name__}: {cleanup}')
            finally:
                if not self.lease.released:
                    vars(error)['_structured_startup_owner'] = self
            raise

    async def initialize(self):
        await super().initialize()
        record = getattr(self, '_preload_record', None)
        if record is not None:
            record.service.observe_actor(record)

    async def _close_owned(self, timeout, started):
        record = getattr(self, '_preload_record', None)
        if record is None:
            return await super()._close_owned(timeout, started)
        previous = getattr(self, '_structured_cleanup_failure', None)
        if previous is not None:
            raise previous
        deadline = min(self._preload_context.cleanup_deadline, started + max(0, timeout))
        try:
            if record.state == 'SENT_PENDING':
                record.service.resolve(record, deadline)
            if record.state in ('PREPARED', 'CONFIRMED_NO_CHILD'):
                record.service.discard_no_child(record)
                if not self.lease.released:
                    self.lease.cancel_before_spawn()
                return
            if record.state == 'STATUS_LOST':
                raise record.service.failure or RuntimeError('pytest CPU actor status lost')
            await super()._close_owned(timeout, started)
            record.service.retire(record, deadline)
        except BaseException as error:
            self._structured_cleanup_failure = error
            vars(error)['_structured_cleanup_owner'] = self
            record.service.poison(error)
            if record.service.failure is not None:
                try:
                    await record.service.close(deadline=deadline)
                except BaseException as cleanup:  # noqa: BLE001 - original cleanup failure stays primary.
                    if cleanup is not error:
                        error.add_note(f'pytest CPU terminal cleanup: {cleanup}')
                for stream in (record.facade.stdin, record.facade.stdout):
                    stream.close()
                if self.io_thread is not None:
                    self.io_thread.join(timeout=.1)
                    if self.io_thread.is_alive():
                        error.add_note('pytest CPU I/O retirement unconfirmed')
            raise
