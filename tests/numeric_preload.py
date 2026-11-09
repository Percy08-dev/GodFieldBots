"""Explicit CPU pytest numeric launch; canonical learners retain all authority."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import signal
import sys
import sysconfig
import time
from collections import deque
from contextlib import contextmanager

import pytest

from tests.cpu_preload import CpuTemplate, supported
from tests.cpu_preload_protocol import NUMERIC_VERSION, integer, memory, receive

START_ALLOWANCE = 10.
CONTROL_GRACE = .05


def supported_numeric():
    if (not supported() or sys.implementation.name != 'cpython'
            or sys.version_info[:3] != (3, 14, 10)
            or sysconfig.get_config_var('Py_GIL_DISABLED')
            or not sys._is_gil_enabled()):
        return False
    try:
        if importlib.metadata.version('torch') not in ('2.14.1+cpu', '2.14.1+cu132'):
            return False
    except importlib.metadata.PackageNotFoundError:
        return False
    loop = asyncio.new_event_loop()
    try:
        return isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


class _WriteProtocol(asyncio.streams.FlowControlMixin):
    """Pinned CPython3.14.10 StreamWriter adapter, including close notification."""
    def __init__(self, record, loop):
        super().__init__(loop=loop)
        self.record, self.closed = record, loop.create_future()
        self.transport = self.pipe = None

    def connection_made(self, transport):
        self.transport = transport
        self.record.transports.append(transport)

    def connection_lost(self, exc):
        super().connection_lost(exc)
        if not self.closed.done():
            self.closed.set_result(None)

    def _get_close_waiter(self, stream):
        return self.closed


class _ReadProtocol(asyncio.StreamReaderProtocol):
    def __init__(self, record, reader, loop):
        super().__init__(reader, loop=loop)
        self.record, self.closed = record, loop.create_future()
        self.transport = self.pipe = None

    def connection_made(self, transport):
        self.transport = transport
        self.record.transports.append(transport)
        super().connection_made(transport)

    def connection_lost(self, exc):
        super().connection_lost(exc)
        if not self.closed.done():
            self.closed.set_result(None)


class NumericProcess:
    def __init__(self, record):
        self.record = record
        self.stdin = self.stdout = self.stderr = None

    @property
    def pid(self):
        if self.record.pid is None:
            raise RuntimeError('numerical PID has not been acknowledged')
        return self.record.pid

    @property
    def returncode(self):
        if self.record.status is not None:
            return self.record.status
        # Original request/Source guards use None to mean live. Terminal status
        # loss must raise here, before those guards permit another write/update.
        self.record.service.require_healthy()
        return None

    async def wait(self):
        while True:
            status = self.returncode
            if status is not None:
                return status
            await self.record.service.changed()

    def terminate(self):
        self.record.service.signal(self.record, signal.SIGTERM)

    def kill(self):
        self.record.service.signal(self.record, signal.SIGKILL)


class _Record:
    def __init__(self, service, token, owner, start_deadline):
        self.service, self.token, self.owner, self.lease = service, token, owner, owner.lease
        self.start_deadline = start_deadline
        self.state, self.pid, self.status = 'PREPARED', None, None
        self.launch_seq = self.resolve_seq = self.retire_seq = None
        self.retired = self.local_retired = False
        self.raw_fds, self.files, self.transports, self.protocols = set(), [], [], []
        self.sending = ()
        self.facade = NumericProcess(self)
        self.close_task = self.close_deadline = None

    def register(self, pid):
        integer(pid, 1)
        if self.pid is not None and self.pid != pid:
            raise RuntimeError('numerical PID changed')
        self.pid = pid
        if self.status is None:
            self.state = 'KNOWN_RUNNING'
        # Receipt only: canonical _adopt_process exclusively assigns/registers.

    def close_sending(self):
        for fd in self.sending:
            os.close(fd)
            self.raw_fds.remove(fd)
        self.sending = ()

    def file(self, fd, mode):
        stream = os.fdopen(fd, mode, buffering=0)
        self.files.append(stream)
        self.raw_fds.remove(fd)
        return stream


class NumericTemplate(CpuTemplate):
    control_version = NUMERIC_VERSION
    server_entry = 'numeric_preload_server.py'

    def __init__(self, directory, *, fault='', start_allowance=START_ALLOWANCE):
        super().__init__(directory, fault=fault)
        self.start_allowance = start_allowance
        self.loop = self.event = None
        self.retirement = None
        self.cleanup_deadline = None
        self.receipts = deque(maxlen=64)

    def start(self, deadline):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('numeric template warmup must precede the test loop')
        if self.state == 'UNSTARTED':
            self.cleanup_deadline = deadline
        return super().start(deadline)

    def poison(self, error):
        result = super().poison(error)
        self._notify()
        return result

    def _notify(self):
        if self.event is not None:
            event, self.event = self.event, asyncio.Event()
            event.set()

    def _detach(self):
        if self.loop is not None and not self.loop.is_closed():
            self.loop.remove_reader(self.channel.fileno())
        self.loop = self.event = None

    def attach(self):
        loop = asyncio.get_running_loop()
        if not isinstance(loop, asyncio.SelectorEventLoop):
            raise TypeError('numeric template requires the pinned selector loop')
        if self.loop is loop:
            return
        if self.loop is not None and (self.records or self.expected or self.answers):
            raise RuntimeError('numeric loop rebinding requires complete retirement')
        self._detach()
        self.loop, self.event = loop, asyncio.Event()
        loop.add_reader(self.channel.fileno(), self._readable)

    def _readable(self):
        try:
            while True:
                try:
                    value, descriptors = receive(self.channel)
                except BlockingIOError:
                    break
                self._dispatch(value, descriptors)
                if self.state == 'CLOSING_ACK':
                    self.loop.remove_reader(self.channel.fileno())
                    break
        except BaseException as error:  # noqa: BLE001 - retain/wake on protocol or status loss
            self.poison(error)
            self.loop.remove_reader(self.channel.fileno())
        self._notify()

    async def changed(self):
        if self.failure is not None:
            raise self.failure
        await self.event.wait()

    async def _answer(self, seq, deadline):
        while seq not in self.answers:
            async with asyncio.timeout_at(deadline):
                await self.changed()
        self._check_deadline(deadline)
        return self.answers.pop(seq)

    async def _prepare(self, record):
        pipes = []
        for _ in range(3):
            pair = os.pipe()
            record.raw_fds.update(pair)
            pipes.append(pair)
        record.sending = (pipes[0][0], pipes[1][1], pipes[2][1])
        loop = self.loop
        protocol = _WriteProtocol(record, loop)
        record.protocols.append(protocol)
        protocol.pipe = record.file(pipes[0][1], 'wb')
        transport, _ = await loop.connect_write_pipe(lambda: protocol, protocol.pipe)
        record.facade.stdin = asyncio.StreamWriter(transport, protocol, None, loop)
        for attribute, fd in (('stdout', pipes[1][0]), ('stderr', pipes[2][0])):
            reader = asyncio.StreamReader(limit=1024 * 1024, loop=loop)
            protocol = _ReadProtocol(record, reader, loop)
            record.protocols.append(protocol)
            protocol.pipe = record.file(fd, 'rb')
            await loop.connect_read_pipe(lambda protocol=protocol: protocol, protocol.pipe)
            setattr(record.facade, attribute, reader)

    async def _local_close(self, record, deadline):
        for transport in record.transports:
            transport.close()
        # Files not successfully transferred to a transport still have owners.
        for stream in record.files:
            if not any(protocol.pipe is stream and protocol.transport is not None for protocol in record.protocols):
                stream.close()
        for fd in tuple(record.raw_fds):
            os.close(fd)
            record.raw_fds.remove(fd)
        record.sending = ()
        for protocol in record.protocols:
            # A protocol without connection_made has no close callback to join.
            if protocol.transport is not None:
                async with asyncio.timeout_at(deadline):
                    await asyncio.shield(protocol.closed)
        if record.owner.drainer is not None:
            async with asyncio.timeout_at(deadline):
                await asyncio.shield(record.owner.drainer)
        for stream in record.files:
            stream.close()
        record.local_retired = True

    async def create(self, owner):
        try:
            if self._close_task is not None:
                raise RuntimeError('numeric auxiliary shutdown has begun')
            self.attach()
            self.require_healthy()
            if (owner.protocol not in (3, 4) or owner.binding.device != 'cpu' or owner.visibility != ''
                    or self.records or (self.retirement is not None and not self.retirement.done())):
                raise RuntimeError('numerical launch is ineligible or prior retirement is incomplete')
        except BaseException:
            owner._creation_no_child = True
            raise
        self.cleanup_deadline = owner.cleanup_deadline
        self.serial += 1
        deadline = min(owner.cleanup_deadline, time.monotonic() + self.start_allowance)
        record = _Record(self, self.serial, owner, deadline)
        self.records[record.token] = record
        owner._numeric_launch = record
        try:
            async with asyncio.timeout_at(deadline):
                await self._prepare(record)
            if self._close_task is not None:
                raise RuntimeError('numeric auxiliary shutdown has begun')
        except BaseException as error:
            try:
                await self._local_close(record, owner.cleanup_deadline)
                self.records.pop(record.token)
                record.state = 'CONFIRMED_NO_CHILD'
                owner._creation_no_child = True
                self._detach()
            except BaseException as cleanup:  # noqa: BLE001 - preparation error remains primary
                error.add_note('numeric preparation cleanup: ' + type(cleanup).__name__)
                self.poison(error)
            raise
        record.state = 'SENT_PENDING'
        try:
            try:
                record.launch_seq = self._send('launch', token=record.token, descriptors=record.sending,
                    bootstrap='numeric', viewer=0, deadline=deadline)
                self.launches += 1
            finally:
                record.close_sending()
            try:
                async with asyncio.timeout_at(min(deadline, time.monotonic() + CONTROL_GRACE)):
                    while record.pid is None and record.state != 'CONFIRMED_NO_CHILD':
                        await self.changed()
            except TimeoutError:
                self._check_deadline(deadline)
                record.resolve_seq = self._send('resolve', token=record.token)
            async with asyncio.timeout_at(deadline):
                while ((record.pid is None and record.state != 'CONFIRMED_NO_CHILD')
                        or record.resolve_seq in self.expected):
                    await self.changed()
            self.require_healthy()
            self._check_deadline(deadline)
            if record.state == 'CONFIRMED_NO_CHILD':
                raise RuntimeError('numeric template authoritatively refused fork')
            usage = memory(record.pid)
            self.require_healthy()
            self._check_deadline(deadline)
            self.receipts.append({'event': 'pid', 'pid': record.pid, 'token': record.token,
                'template_pid': self.process.pid, **usage})
            return record.facade
        except BaseException as error:
            if record.state != 'CONFIRMED_NO_CHILD':
                self.poison(error)
            raise

    async def settle(self, owner, deadline):
        record = owner._numeric_launch
        if not owner._startup_boundary.is_set():
            async with asyncio.timeout_at(deadline):
                await owner._startup_boundary.wait()
        if record.pid is not None:
            owner._adopt_process(record.facade)
            return
        if record.state == 'CONFIRMED_NO_CHILD':
            if not record.local_retired:
                await self._local_close(record, deadline)
            owner._creation_no_child = True
            owner.lease.creation_pending = False
            return
        raise self.failure or RuntimeError('numerical creation remains unresolved')

    async def _retire_ticket(self, record, deadline):
        if record.retired:
            return
        if record.state == 'CONFIRMED_NO_CHILD':
            self.records.pop(record.token, None)
            record.retired = True
            self._detach()
            return
        if record.status is None or not record.lease.released or not record.local_retired:
            raise RuntimeError('numeric retirement requires real wait and complete original/local cleanup')
        record.retire_seq = self._send('retire', token=record.token)
        try:
            await self._answer(record.retire_seq, min(deadline, time.monotonic() + CONTROL_GRACE))
        except TimeoutError:
            self._check_deadline(deadline)
            previous = record.retire_seq
            record.retire_seq = self._send('retire', token=record.token)
            await self._answer(record.retire_seq, deadline)
            self.expected.pop(previous, None)
            self.answers.pop(previous, None)
        self.receipts.append({'event': 'retired', 'pid': record.pid, 'status': record.status,
            'token': record.token, 'lease_released': record.lease.released})
        self.records.pop(record.token)
        record.retired = True
        if self.expected or self.answers:
            raise RuntimeError('numeric control replies remain unretired')
        self._detach()

    async def close_owner(self, owner, canonical_close, deadline):
        record = owner._numeric_launch
        if record.close_task is None:
            record.close_deadline = self.cleanup_deadline = deadline
            coroutine = self._finish_owner(owner, canonical_close, deadline)
            try:
                record.close_task = self.retirement = asyncio.create_task(coroutine)
            except BaseException:
                coroutine.close()
                raise
        cancellation = None
        while not record.close_task.done():
            try:
                await asyncio.shield(record.close_task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException:  # noqa: BLE001 - preserve retained attempt's exact failure
                break
        record.close_task.result()
        if cancellation:
            raise cancellation

    async def _finish_owner(self, owner, canonical_close, deadline):
        record = owner._numeric_launch
        error = None
        try:
            await canonical_close(owner, deadline)
        except BaseException as caught:  # noqa: BLE001 - canonical error stays primary
            error = caught
        try:
            await self._local_close(record, max(deadline, time.monotonic() + .1) if error else deadline)
            if record.state == 'CONFIRMED_NO_CHILD' or (record.status is not None and self.failure is None):
                await self._retire_ticket(record, deadline)
            else:
                error = error or self.failure or RuntimeError('numeric status/retirement lost')
        except BaseException as cleanup:  # noqa: BLE001 - preserve original cancellation/start/request failure
            if error is None:
                error = cleanup
            elif cleanup is not error:
                error.add_note('numeric extra retirement: ' + type(cleanup).__name__)
            self.poison(error)
        if error is not None:
            await self._terminal_close(error, record.close_deadline)
            raise error

    async def _terminal_close(self, error, deadline):
        self.poison(error)
        try:
            await self.close(deadline=deadline)
        except BaseException as cleanup:  # noqa: BLE001 - first operation error remains primary
            if cleanup is not error:
                error.add_note('numeric terminal containment: ' + type(cleanup).__name__)

    def quiescent(self):
        return (not self.records and not self.expected and not self.answers and self.loop is None
            and (self.retirement is None or self.retirement.done()))

    async def health_async(self, deadline):
        try:
            self.require_healthy()
            self.attach()
            if self.records:
                raise RuntimeError('numeric template still owns a learner ticket')
            seq = self._send('health')
            value = await self._answer(seq, deadline)
            if value['inventory'] != 0:
                raise RuntimeError('numeric broker inventory is not empty')
            self._detach()
            return value
        except BaseException as error:
            await self._terminal_close(error, deadline)
            raise

    async def _close_owned(self, deadline):
        if self.process is None:
            return await super()._close_owned(deadline)
        # A failed loop may be gone while unresolved ownership remains. Terminal
        # containment does not reuse its transports/reader or admit another child.
        if self.failure is not None:
            self._detach()
        else:
            self.attach()
        containment_attempted = False
        try:
            if self.failure is None and not self.records:
                seq = self._send('close')
                await self._answer(seq, deadline)
                while os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:  # noqa: ASYNC222 - WNOHANG never blocks.
                    self._check_deadline(deadline)
                    await asyncio.sleep(.001)
                if self._wait_template(timeout=0) != 0:
                    raise RuntimeError('numeric template actual exit was unsuccessful')
                self._check_deadline(deadline)
                self.state = 'CLOSED'
                return {'closed': True, 'template_pid': self.process.pid}
            self.poison(self.failure or RuntimeError('numeric close has unresolved tickets'))
            containment_attempted = True
            await self._contain(deadline)
            raise self.failure
        except BaseException as error:  # noqa: BLE001 - failed auxiliary shutdown stays terminal
            self.poison(error)
            if not self._leader_reaped and not containment_attempted:
                await self._contain(deadline)
            raise self.failure
        finally:
            self._detach()
            self.channel.close()
            self.log.close()


@contextmanager
def activate(service):
    from gf_training.ppo_run import LearningProcess
    from gf_training.structured_learning_process import StructuredLearningProcess
    original_create = LearningProcess._create_process
    original_settle = StructuredLearningProcess._settle_creation
    original_close = StructuredLearningProcess.close

    async def create(owner):
        if not isinstance(owner, StructuredLearningProcess):
            return await original_create(owner)
        return await service.create(owner)

    async def settle(owner, deadline):
        if not hasattr(owner, '_numeric_launch'):
            return await original_settle(owner, deadline)
        return await service.settle(owner, deadline)

    async def close(owner, deadline):
        if not hasattr(owner, '_numeric_launch'):
            return await original_close(owner, deadline)
        return await service.close_owner(owner, original_close, deadline)

    LearningProcess._create_process = create
    StructuredLearningProcess._settle_creation = settle
    StructuredLearningProcess.close = close
    try:
        yield
    finally:
        LearningProcess._create_process = original_create
        StructuredLearningProcess._settle_creation = original_settle
        StructuredLearningProcess.close = original_close


@pytest.fixture(scope='module')
def _numeric_cpu_template(tmp_path_factory):
    service = NumericTemplate(tmp_path_factory.mktemp('numeric-cpu-template'))
    yield service
    try:
        deadline = (service._close_deadline if service._close_task is not None else
            service.cleanup_deadline if service.failure is not None else time.monotonic() + 2)
        asyncio.run(service.close(deadline=deadline))
    finally:
        (service.directory / 'numeric-lifecycle.json').write_text(json.dumps({
            'mode': service.control_version, 'ready': service.ready,
            'launches': service.launches, 'receipts': list(service.receipts),
            'unretired_tickets': len(service.records), 'state': service.state,
            'template_exit_code': None if service.process is None else service.process.returncode,
            'peak_template_rss_kib': service.peak_rss_kib,
            'peak_template_pss_kib': service.peak_pss_kib,
            'containment': service.containment,
            'failure': None if service.failure is None else str(service.failure),
        }, indent=2))


@pytest.fixture
def numeric_cpu_preload(request, _numeric_cpu_template):
    selected = {
        'test_structured_numeric_worker.py': {'test_actual_cpu_wire_initialize_two_updates_restore_and_close'},
        'test_card_instance_source.py': {'test_live_projection_real_wire_update_and_precommit_guard',
            'test_live_collector_guard_rejects_copied_owner_and_control_replacement'},
    }
    if (request.node.originalname not in selected.get(request.path.name, set())
            or getattr(request.node, 'callspec', None) is not None
            and request.node.callspec.params.get('device', 'cpu') != 'cpu'
            or not supported_numeric()):
        yield None
        return
    service = _numeric_cpu_template
    service.start(time.monotonic() + START_ALLOWANCE)
    try:
        with activate(service):
            yield service
        if not service.quiescent():
            raise service.poison(RuntimeError('numeric case did not completely retire ownership'))
    finally:
        if service.failure is not None:
            deadline = service._close_deadline if service._close_task is not None else service.cleanup_deadline
            asyncio.run(service.close(deadline=deadline))
