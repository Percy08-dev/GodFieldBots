"""Bounded standard-library-only control packets for the pytest CPU template."""
from __future__ import annotations

import array
import json
import math
import os
import socket
from pathlib import Path

VERSION = 'pytest-cpu-preload-v1'
MAX_PACKET = 8192
RSS_LIMIT_KIB = 1024 * 1024
MODELS = ('random', 'heuristic', 'mlp', 'gru', 'transformer')
FACTORY = {'module': 'gf_models.structured_baselines', 'name': 'make_structured_policy'}


def closed(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('malformed pytest CPU control fields')
    return value


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid pytest CPU control integer')
    return value


def encode(value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_PACKET:
        raise ValueError('oversized pytest CPU control packet')
    return data


def receive(channel):
    data, ancillary, flags, _ = channel.recvmsg(MAX_PACKET,
        socket.CMSG_SPACE(4 * array.array('i').itemsize), getattr(socket, 'MSG_CMSG_CLOEXEC', 0))
    descriptors = []
    try:
        invalid = False
        for level, kind, raw in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                invalid = True
                continue
            numbers = array.array('i')
            numbers.frombytes(raw[:len(raw) - len(raw) % numbers.itemsize])
            descriptors.extend(numbers)
        for fd in descriptors:
            os.set_inheritable(fd, False)
        if invalid:
            raise ValueError('unexpected pytest CPU ancillary data')
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError('truncated pytest CPU control packet')
        if not data:
            raise EOFError('pytest CPU control EOF; actor status unavailable')
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('duplicate pytest CPU JSON key')
                result[key] = value
            return result
        value = json.loads(data, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return value, descriptors
    except BaseException:
        for fd in descriptors:
            os.close(fd)
        raise


def memory(pid):
    result = {}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        if line.startswith(('Rss:', 'Pss:')):
            name, number, _ = line.split()
            result[name[:-1].lower() + '_kib'] = int(number)
    if set(result) != {'rss_kib', 'pss_kib'}:
        raise RuntimeError('pytest CPU memory observation unavailable')
    return result


def validate_initial(initial):
    if type(initial) is not dict or initial.get('factory') != FACTORY or initial.get('model_id') not in MODELS:
        raise ValueError('preloaded actor requires fixed structured CPU factory/model')
    options = initial.get('policy_options')
    if options is not None and (type(options) is not dict or options.get('inference', {}).get('device') != 'cpu'
            or options.get('cuda_visible_devices') != ''):
        raise ValueError('preloaded actor requires CPU and empty CUDA visibility')


def deadline(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite pytest CPU launch deadline required')
    return value
