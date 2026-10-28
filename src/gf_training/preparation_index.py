"""Compact deterministic index sorting with bounded spill and merge ownership."""
from __future__ import annotations

import heapq
import json
import os
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import Json, canonical
from gf_training.preparation import MIB, PreparationSession, ResourceLimit, deep_size


@dataclass(frozen=True)
class _Run:
    path: Path
    size: int
    max_line: int


class IndexLedger:
    """Whole row order is independent of source order and configured chunk sizes."""

    def __init__(self, session: PreparationSession) -> None:
        self.session = session
        self.target = session.config.chunk_target_mib * MIB
        self.entries: list[tuple[tuple[str, str, int, int], str]] = []
        self.charge = 0
        self.runs: list[_Run] = []
        self.counter = 0
        self.rows = 0
        self.session.diagnostics['index_spill_runs'] = 0

    def add(self, identity: tuple[str, str, int, int], row: Json) -> None:
        charge = 2 * deep_size(row) + 4096
        if self.entries and self.charge + charge > self.target:
            self.flush()
        self.session.charge(charge)
        self.charge += charge
        self.entries.append((identity, canonical(row)))
        self.rows += 1

    def _new_path(self) -> Path:
        self.counter += 1
        path = self.session.scratch() / f'sorted-{self.counter:06}.jsonl'
        self.session.retain(path)
        return path

    def flush(self) -> None:
        if not self.entries:
            return
        self.entries.sort(key=lambda entry: entry[0])
        path = self._new_path()
        size = max_line = 0
        with path.open('xb') as stream:
            for _, encoded in self.entries:
                block = (encoded + '\n').encode()
                self.session.reserve_scratch(len(block))
                stream.write(block)
                size += len(block)
                max_line = max(max_line, len(block))
            stream.flush()
            os.fsync(stream.fileno())
        self.runs.append(_Run(path, size, max_line))
        self.entries.clear()
        self.session.release(self.charge)
        self.charge = 0
        self.session.diagnostics['index_spill_runs'] += 1

    def _merge(self, runs: list[_Run]) -> Generator[str]:
        # Each active head owns its encoded text, decoded key and decoder headroom.
        charge = sum(64 * run.max_line + 8192 for run in runs)
        self.session.charge(charge)
        try:
            with ExitStack() as streams:
                sources = [streams.enter_context(run.path.open('rb')) for run in runs]
                heap: list[Any] = []
                def advance(index: int) -> None:
                    self.session.check()
                    line = sources[index].readline(runs[index].max_line + 1)
                    if not line:
                        return
                    if len(line) > runs[index].max_line or not line.endswith(b'\n'):
                        raise ResourceLimit('sorted index record exceeds owned bound')
                    encoded = line[:-1].decode()
                    row = json.loads(encoded)
                    identity = (row['run_id'], row['game_id'], row['learner_viewer'], row['decision'])
                    heapq.heappush(heap, (identity, index, encoded))
                for index in range(len(sources)):
                    advance(index)
                while heap:
                    _, index, encoded = heapq.heappop(heap)
                    yield encoded
                    advance(index)
        finally:
            self.session.release(charge)

    def sorted_rows(self) -> Generator[str]:
        self.flush()
        # Fan-in is bounded; all intermediate files remain under the same scratch owner.
        while len(self.runs) > 16:
            selected = self.runs[:16]
            path = self._new_path()
            size = max_line = 0
            with path.open('xb') as stream:
                for encoded in self._merge(selected):
                    block = (encoded + '\n').encode()
                    self.session.reserve_scratch(len(block))
                    stream.write(block)
                    size += len(block)
                    max_line = max(max_line, len(block))
                stream.flush()
                os.fsync(stream.fileno())
            for run in selected:
                run.path.unlink()
                self.session.release_scratch(run.size)
            self.runs[:16] = [_Run(path, size, max_line)]
        yield from self._merge(self.runs)
