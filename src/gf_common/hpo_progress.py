"""Optional diagnostic values only: no policy, Source, or execution authority."""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, TextIO

from gf_common.domain import Json

TAG = 'GFB_HPO_DIAGNOSTIC_V1 '
LINE_BYTES = 4096
UPDATE_EXECUTIONS=('baseline','metrics','packed','microbatch','shared-prefix')
PHASES = ('startup', 'collect', 'source', 'update', 'save', 'sync', 'evaluate', 'cleanup')
ProgressCallback = Callable[[Json], object]


def observe(callback: ProgressCallback | None, value: Json) -> None:
    """A failed display must never alter the original operation or its cancellation."""
    if callback is not None:
        try:
            callback(value)
        except Exception:  # noqa: BLE001, S110 - optional observation only
            pass


class ActiveRate:
    """Invocation-local committed intervals; no resumed wall-clock gap."""
    def __init__(self) -> None:
        self.previous: tuple[int, float] | None = None
        self.intervals: deque[tuple[int, float]] = deque(maxlen=3)

    def commit(self, used: int, *, now: float, budget: int) -> Json:
        if self.previous is not None:
            count, started = self.previous
            if used > count and now > started:
                self.intervals.append((used-count, now-started))
        self.previous = used, now
        rate = (sum(item[0] for item in self.intervals)/sum(item[1] for item in self.intervals)
            if len(self.intervals) >= 2 else None)
        return {'decisions_per_second': rate, 'eta_seconds': max(0,budget-used)/rate if rate else None}


class DiagnosticEmitter:
    """Explicitly enabled child diagnostic channel on its existing stderr."""
    def __init__(self, invocation: str, max_log_bytes: int, output: TextIO) -> None:
        self.invocation, self.output = invocation, output
        self.remaining = min(max_log_bytes//4, 262144)
        self.sequence = 0
        self.disabled = False

    def emit(self, update: int, facts: Json) -> None:
        if self.disabled:
            return
        try:
            value = {'schema_version':1, 'invocation':self.invocation, 'update':update,
                'sequence':self.sequence, 'facts':facts}
            line = TAG + json.dumps(value, separators=(',',':'), allow_nan=False) + '\n'
            size = len(line.encode())
            if size > LINE_BYTES or size > self.remaining:
                self.disabled = True
                return
            self.remaining -= size
            self.sequence += 1
            self.output.write(line)
            self.output.flush()
        except Exception:  # noqa: BLE001 - emission is never numerical execution
            self.disabled = True


def optional_diagnostic_emitter(environment: Mapping[str,str],output: TextIO) -> DiagnosticEmitter | None:
    """Optional setup only; never enclose learner/Source/cleanup operations."""
    try:
        invocation=environment.get('GFB_HPO_DIAGNOSTIC_INVOCATION')
        if not invocation:
            return None
        maximum=int(environment['GFB_HPO_DIAGNOSTIC_LOG_BYTES'])
        if maximum<=0:
            return None
        return DiagnosticEmitter(invocation,maximum,output)
    except Exception:  # noqa: BLE001 - ordinary display setup failure disables only diagnostics
        return None


class DiagnosticParser:
    """Bounded reassembly alongside raw log accounting; no ACK validation role."""
    def __init__(self, invocation: str, callback: ProgressCallback | None) -> None:
        self.invocation, self.callback = invocation, callback
        self.buffer = bytearray()
        self.discard = False
        self.sequence = -1
        self.update = 0

    def feed(self, chunk: bytes) -> None:
        # Do not split an arbitrarily large chunk into an unbounded line list.
        for byte in chunk:
            if byte != 10:
                if not self.discard:
                    self.buffer.append(byte)
                    if len(self.buffer) >= LINE_BYTES:
                        self.buffer.clear()
                        self.discard = True
                continue
            if not self.discard:
                self._line(bytes(self.buffer))
            self.buffer.clear()
            self.discard = False

    def _line(self, line: bytes) -> None:
        if not line.startswith(TAG.encode()):
            return
        try:
            value = json.loads(line[len(TAG):])
            if (type(value) is not dict or value.get('schema_version') != 1
                    or type(value.get('schema_version')) is not int or value.get('invocation') != self.invocation or type(value.get('sequence')) is not int
                    or value['sequence'] <= self.sequence or type(value.get('update')) is not int or value.get('update') != self.update
                    or not learner_facts(value.get('facts'))):
                return
            self.sequence = value['sequence']
            observe(self.callback, {**value['facts'], 'learner_diagnostic_status':'observed', 'diagnostic_invocation':self.invocation,
                'diagnostic_update':value['update'], 'diagnostic_sequence':value['sequence']})
        except Exception:  # noqa: BLE001, S110 - invalid log lines have no authority
            pass


def allocator_facts(learner: Any) -> Json:
    """Read at an EXISTING sync boundary. Never initialize/synchronize CUDA here."""
    if learner.binding.device == 'cpu':
        return {'cuda_allocator':None}
    cuda, device = learner.torch.cuda, learner.network.device
    return {'cuda_allocator': {'scope':'process-lifetime', 'reset':'never',
        'allocated_bytes':cuda.memory_allocated(device), 'reserved_bytes':cuda.memory_reserved(device),
        'max_allocated_bytes':cuda.max_memory_allocated(device), 'max_reserved_bytes':cuda.max_memory_reserved(device)}}


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        return False


class CollectionProgress:
    """Whole completed games only; total moves never imply learner-seat counts."""
    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback
        self.games = self.actions = 0

    def completed(self, *, games: int, total_game_actions: int) -> None:
        self.games += games
        self.actions += total_game_actions
        observe(self.callback, {'transient_games':self.games,'transient_total_game_actions':self.actions,
            'transient_learner_decisions':None})


def pool_diagnostic(value: Any) -> Json | None:
    """Drop only malformed optional facts; management validation stays separate."""
    try:
        if (type(value) is not dict or len(json.dumps(value,allow_nan=False).encode())>LINE_BYTES
                or set(value)-{'host_memory'}!={'schema_version','loads','cache_reuses','cpu_weight_restore_bytes',
                    'cuda_weight_restore_bytes','cuda_allocator'} or type(value['schema_version']) is not int
                or value['schema_version']!=1):
            return None
        for key in ('loads','cache_reuses','cpu_weight_restore_bytes','cuda_weight_restore_bytes'):
            if type(value[key]) is not int or not 0<=value[key]<=2**63-1:
                return None
        host=value.get('host_memory')
        if host is not None:
            if (type(host) is not dict or set(host)!={'scope','boundary','pss_bytes','private_bytes','shared_clean_bytes','shared_dirty_bytes','max_rss_bytes'}
                    or host['scope']!='process-current-after-command' or host['boundary'] not in ('BEGIN','END','PARK','SYNC','EVICT','CLOSE')):
                return None
            if any(type(host[key]) is not int or host[key]<0 for key in ('pss_bytes','private_bytes','shared_clean_bytes','shared_dirty_bytes','max_rss_bytes')):
                return None
        allocator = value['cuda_allocator']
        if allocator is not None:
            if (type(allocator) is not dict or set(allocator)!={'scope','reset','allocated_bytes','reserved_bytes',
                    'max_allocated_bytes','max_reserved_bytes'} or allocator['scope']!='process-lifetime'
                    or allocator['reset']!='never'):
                return None
            for key in ('allocated_bytes','reserved_bytes','max_allocated_bytes','max_reserved_bytes'):
                if type(allocator[key]) is not int or not 0<=allocator[key]<=2**63-1:
                    return None
        return value
    except Exception:  # noqa: BLE001 - malformed diagnostic only
        return None


def learner_facts(value: Any) -> bool:
    if type(value) is not dict or not value or set(value)-{'phase','cuda_allocator','diagnostic_boundary','save_seconds','parity_seconds','update_execution'}:
        return False
    if 'update_execution' in value and (type(value['update_execution']) is not str or value['update_execution'] not in UPDATE_EXECUTIONS):
        return False
    if 'phase' in value and value['phase']!='save':
        return False
    if 'diagnostic_boundary' in value and value['diagnostic_boundary'] not in ('update-synchronized','save-complete'):
        return False
    if 'save_seconds' in value and (not finite_number(value['save_seconds']) or value['save_seconds']<0):
        return False
    if 'parity_seconds' in value and (not finite_number(value['parity_seconds']) or value['parity_seconds']<0):
        return False
    if 'cuda_allocator' in value:
        # Reuse the same closed diagnostic-only allocator validation.
        return pool_diagnostic({'schema_version':1,'loads':0,'cache_reuses':0,
            'cpu_weight_restore_bytes':0,'cuda_weight_restore_bytes':0,'cuda_allocator':value['cuda_allocator']}) is not None
    return True


def update_execution_facts(learner: Any) -> Json:
    """Only a successfully resolved branch, never a configured/guessed route."""
    try:
        value=getattr(learner,'_last_update_execution',None)
        return {'update_execution':value} if type(value) is str and value in UPDATE_EXECUTIONS else {}
    except Exception:  # noqa: BLE001 - optional fact cannot acquire numerical authority
        return {}
