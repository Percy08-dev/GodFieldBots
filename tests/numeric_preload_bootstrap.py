"""CPU-only first-packet admission; original numerical main/serve own all math."""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path


def _cpu_origin(initial, binding, visibility, template_pid):
    torch = sys.modules.get('torch')
    if (type(initial.get('protocol_version')) is not int or initial['protocol_version'] not in (3, 4)
            or binding.device != 'cpu' or visibility != ''
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or torch is None or torch.__version__ not in ('2.14.1+cpu', '2.14.1+cu132')
            or torch.cuda.is_initialized() or threading.active_count() != 1
            or len(list(Path('/proc/self/task').iterdir())) != 1
            or os.getppid() != template_pid or os.getpgrp() != template_pid or os.getsid(0) != template_pid):
        raise ValueError('preloaded numerical child requires clean pinned CPU template origin')


class AdmissionInput:
    def __init__(self, source):
        self.source, self.first = source, True

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.source)
        if self.first:
            self.first = False
            initial = json.loads(line)
            execution = initial.get('identity', {}).get('execution', {})
            if (type(initial.get('protocol_version')) is not int or initial['protocol_version'] not in (3, 4)
                    or execution.get('binding', {}).get('device') != 'cpu'
                    or execution.get('cuda_visible_devices') != ''):
                raise ValueError('preloaded numerical child requires explicit CPU wire3/4')
        return line


def main(template_pid, fault=''):
    if fault == 'numeric_io_witness':
        # Explicit transport-only safety witness, never numerical evidence.
        import time
        payload = b'x' * 65536
        for _ in range(32):
            remaining = payload
            while remaining:
                remaining = remaining[os.write(2, remaining):]
        time.sleep(.15)
        size = 8 * 1024**2
        while size:
            data = os.read(0, min(size, 65536))
            if not data:
                raise EOFError('numeric transport witness stdin closed')
            size -= len(data)
        print('{"transport_witness":true}', flush=True)
        return
    from gf_training import structured_numeric_worker
    from gf_training.ppo_worker import main as original_main
    structured_numeric_worker._select_visibility = lambda initial, binding, visibility: _cpu_origin(initial, binding, visibility, template_pid)
    sys.stdin = AdmissionInput(sys.stdin)
    original_main()
