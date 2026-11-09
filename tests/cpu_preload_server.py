"""Clean-exec sole fork/wait owner. Never imports an application in the template."""
from __future__ import annotations

import io
import json
import os
import select
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from cpu_preload_protocol import (
    RSS_LIMIT_KIB,
    VERSION,
    closed,
    deadline,
    encode,
    integer,
    memory,
    receive,
)


def run():
    control_fd, purelib, root, session, fault, audit_path = sys.argv[1:]
    assert sys.flags.no_site and sys.flags.ignore_environment
    assert 'sitecustomize' not in sys.modules and 'torch' not in sys.modules
    checkout = Path(root).resolve()
    sys.path[:] = [path for path in sys.path if not Path(path or os.getcwd()).resolve().is_relative_to(checkout)]
    sys.path.insert(0, purelib)
    started = time.monotonic()
    import torch
    channel = socket.socket(fileno=int(control_fd))
    channel.setblocking(False)
    inventory = {}
    handled = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    for number in handled:
        signal.signal(number, lambda number, frame: sys.exit(128 + number))

    def audit(event, **details):
        with open(audit_path, 'a') as destination:
            destination.write(json.dumps({'event': event, 'pid': os.getpid(), **details}) + '\n')

    def attest(launch_deadline=None):
        if (torch.__version__ not in ('2.14.1+cpu', '2.14.1+cu132') or torch.cuda.is_initialized()
                or threading.active_count() != 1 or len(list(Path('/proc/self/task').iterdir())) != 1
                or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
                or any(name.startswith(('gf_', 'tests.')) for name in sys.modules)):
            raise RuntimeError('pytest CPU template purity/thread/runtime attestation failed')
        usage = memory(os.getpid())
        if fault == 'delayed_attest' and launch_deadline is not None:
            time.sleep(max(0, launch_deadline - time.monotonic()) + .02)
        if usage['rss_kib'] > RSS_LIMIT_KIB:
            raise RuntimeError('pytest CPU template exceeds1GiB RSS soft ceiling')
        if any(Path(path).resolve() in (checkout, checkout / 'src', checkout / 'tests') for path in sys.path):
            raise RuntimeError('pytest CPU template contains checkout import path')
        return usage

    def send(op, seq=0, **fields):
        payload = encode({'version': VERSION, 'session': session, 'op': op, 'seq': seq, **fields})
        # A stalled control consumer is terminal, not an unbounded server queue.
        if not select.select([], [channel], [], .1)[1]:
            raise TimeoutError('pytest CPU control consumer stalled')
        if channel.send(payload) != len(payload):
            raise RuntimeError('partial pytest CPU control send')

    def crash_at(phase, token=None, actor_pid=None):
        if fault == phase:
            audit(phase, token=token, actor_pid=actor_pid)
            os.kill(os.getpid(), signal.SIGKILL)

    ready = {'pid': os.getpid(), 'torch_version': torch.__version__, 'native_threads': 1,
        'python_threads': 1, 'cuda_initialized': False, 'preload_seconds': time.monotonic() - started,
        'search_paths': list(sys.path), **attest()}
    audit('template_ready', receipt=ready)
    send('ready', **ready)
    if fault == 'bad_control':
        channel.send(b'{invalid')
    if fault == 'truncated_control':
        channel.send(b'x' * 9000)
    if fault == 'stale_control':
        send('health', 999, inventory=0, **attest())
    last_seq = 0
    while True:
        for token, record in tuple(inventory.items()):
            if record['status'] is None:
                pid, status = os.waitpid(record['pid'], os.WNOHANG)
                if pid:
                    record['status'] = os.waitstatus_to_exitcode(status)
                    crash_at('after_waitpid_before_status', token, pid)
                    send('reaped', token=token, pid=pid, status=record['status'])
        if not select.select([channel], [], [], .005)[0]:
            continue
        value, descriptors = receive(channel)
        try:
            common = {'version', 'session', 'op', 'seq'}
            if (type(value) is not dict or value.get('version') != VERSION or value.get('session') != session
                    or integer(value.get('seq'), 1) <= last_seq):
                raise ValueError('pytest CPU session/sequence mismatch')
            last_seq = value['seq']
            op, seq = value['op'], value['seq']
            fields = {'launch': {'token', 'bootstrap', 'viewer', 'deadline'},
                'resolve': {'token'}, 'signal': {'token', 'signal'}, 'retire': {'token'},
                'health': set(), 'close': set()}
            if op not in fields:
                raise ValueError('unknown pytest CPU operation')
            closed(value, common | fields[op])
            if op != 'launch' and descriptors:
                raise ValueError('unexpected pytest CPU descriptors')
            if op == 'launch':
                token = integer(value['token'], 1)
                if (len(descriptors) != 2 or value['bootstrap'] not in ('structured', 'pooled')
                        or type(value['viewer']) is not int or value['viewer'] not in (1, 2)
                        or token in inventory):
                    raise ValueError('invalid pytest CPU launch descriptor')
                deadline(value['deadline'])
                if len(inventory) >= 4 or time.monotonic() >= value['deadline'] or fault == 'no_fork':
                    send('refused', seq, token=token, reason='capacity/deadline/no-fork')
                    continue
                attest(value['deadline'])
                if time.monotonic() >= value['deadline']:
                    send('refused', seq, token=token, reason='deadline-after-attestation')
                    continue
                crash_at('before_fork', token)
                template_pid = os.getpid()
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
                try:
                    if time.monotonic() >= value['deadline']:
                        send('refused', seq, token=token, reason='deadline-before-fork')
                        continue
                    pid = os.fork()
                    if pid == 0:
                        # Only stdin/stdout and public imports survive into this fresh actor.
                        try:
                            os.dup2(descriptors[0], 0)
                            os.dup2(descriptors[1], 1)
                            for fd in [int(entry.name) for entry in Path('/proc/self/fd').iterdir()]:
                                if fd > 2:
                                    try:
                                        os.close(fd)
                                    except OSError:
                                        pass
                            for number in handled:
                                signal.signal(number, signal.SIG_DFL)
                            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                            assert (not torch.cuda.is_initialized() and threading.active_count() == 1
                                and len(list(Path('/proc/self/task').iterdir())) == 1
                                and os.getppid() == os.getpgrp() == os.getsid(0) == template_pid)
                            assert not any(name.startswith(('gf_', 'tests.')) for name in sys.modules)
                            open_fds = []
                            for fd in [int(entry.name) for entry in Path('/proc/self/fd').iterdir()]:
                                try:
                                    os.fstat(fd)
                                except OSError:
                                    continue
                                open_fds.append(fd)
                            assert sorted(open_fds) == [0, 1, 2]
                            audit('actor_entry', actor_pid=os.getpid(), template_pid=template_pid,
                                pgid=os.getpgrp(), sid=os.getsid(0), native_threads=1, python_threads=1,
                                cuda_initialized=False, inherited_torch=True, open_fds=sorted(open_fds))
                            if fault in ('actor_pre_ready_death', 'after_waitpid_before_status'):
                                os._exit(7)
                            sys.stdin = io.TextIOWrapper(os.fdopen(0, 'rb', buffering=0, closefd=False), encoding='utf-8')
                            sys.stdout = io.TextIOWrapper(os.fdopen(1, 'wb', buffering=0, closefd=False), encoding='utf-8', write_through=True)
                            sys.path[:0] = [root, str(Path(root) / 'src')]
                            from tests.cpu_preload_actor import main
                            main(value['bootstrap'], value['viewer'])
                            os._exit(0)
                        except BaseException:  # noqa: BLE001 - exact child process failure exits nonzero.
                            traceback.print_exc()
                            os._exit(1)
                    crash_at('after_fork_before_registration', token, pid)
                    inventory[token] = {'pid': pid, 'status': None}
                    crash_at('after_registration_before_ack', token, pid)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                if fault != 'lost_ack':
                    send('launched', seq, token=token, pid=pid)
                else:
                    audit('lost_ack', token=token, actor_pid=pid)
            elif op == 'resolve':
                token = integer(value['token'], 1)
                record = inventory.get(token)
                if record is None:
                    send('refused', seq, token=token, reason='authoritative-no-child')
                else:
                    send('launched', seq, token=token, pid=record['pid'])
                    if record['status'] is not None:
                        send('reaped', token=token, pid=record['pid'], status=record['status'])
            elif op == 'signal':
                record = inventory[integer(value['token'], 1)]
                if type(value['signal']) is not int or value['signal'] not in (signal.SIGTERM, signal.SIGKILL):
                    raise ValueError('unsupported pytest CPU actor signal')
                if record['status'] is None:
                    os.kill(record['pid'], value['signal'])
            elif op == 'retire':
                token = integer(value['token'], 1)
                if inventory[token]['status'] is None:
                    raise ValueError('cannot retire actor without actual wait receipt')
                del inventory[token]
                send('retired', seq, token=token)
            elif op == 'health':
                if fault == 'stalled_health':
                    while True:
                        signal.pause()
                send('health', seq, inventory=len(inventory), **attest())
            else:
                if inventory:
                    raise ValueError('template cannot close with unretired actor tickets')
                send('closed', seq, **attest())
                if fault == 'closed_exit7':
                    os._exit(7)
                if fault == 'closed_sigkill':
                    os.kill(os.getpid(), signal.SIGKILL)
                return
        finally:
            for fd in descriptors:
                os.close(fd)


if __name__ == '__main__':
    run()
