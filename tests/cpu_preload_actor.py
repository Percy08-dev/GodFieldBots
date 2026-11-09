"""Admission stream only; all policy/frame/epoch semantics stay in original mains."""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace

from gf_simulator_client.framing import read_frame, write_frame
from tests.cpu_preload_protocol import validate_initial


class AdmissionInput:
    def __init__(self, source, bootstrap):
        self.source, self.bootstrap = source, bootstrap
        self.first = True
        self.pending = io.BytesIO()

    def read(self, size):
        if self.pending.tell() == len(self.pending.getbuffer()):
            value = read_frame(self.source)
            if self.bootstrap == 'structured' and self.first:
                validate_initial(value)
            elif self.bootstrap == 'pooled' and value.get('operation') == 'begin':
                validate_initial(value.get('payload'))
            self.first = False
            self.pending = io.BytesIO()
            write_frame(self.pending, value)
            self.pending.seek(0)
        return self.pending.read(size)


def main(bootstrap, viewer):
    # Inspect the first initial/begin BEFORE importing either original bootstrap.
    old_stdin = sys.stdin
    admitted = AdmissionInput(old_stdin.buffer, bootstrap)
    initial = read_frame(admitted)
    framed = io.BytesIO()
    write_frame(framed, initial)
    framed.seek(0)
    class Replay:
        def read(self, size):
            data = framed.read(size)
            return data if data else admitted.read(size)
    sys.stdin = SimpleNamespace(buffer=Replay())
    if bootstrap == 'structured':
        from gf_simulator_client.structured_bootstrap import main as original
    else:
        sys.argv[:] = ['pytest-preloaded-pool', str(viewer)]
        from tests.pooled_structured_bootstrap import main as original
    try:
        original()
    finally:
        sys.stdin = old_stdin
