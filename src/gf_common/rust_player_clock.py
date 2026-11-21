"""Pure original-clock DATA, never supervision or admission authority."""
from __future__ import annotations

import math
from typing import Any

from gf_common.domain import ContractError, Json, closed, digest
from gf_common.rust_player import (
    deadline_bits,
    deadline_seconds,
    record,
    text,
    validate_deadlines,
    validate_operation,
)

_INPUTS = {'admission_seconds', 'execution_seconds', 'wall_seconds', 'finalization_seconds'}
_BINDING = {'boot_id', 'monotonic_before_bits', 'monotonic_after_bits', 'unix_anchor_bits'}
_RECIPE = 'same-total-explicit-post-host-v1'


def _positive(value: Any) -> float:
    if type(value) not in (int, float):
        raise ContractError('explicit finite positive clock allocation required')
    try:
        number = float(value)
    except OverflowError as error:
        raise ContractError('clock allocation is not representable') from error
    if not math.isfinite(number) or number <= 0:
        raise ContractError('explicit finite positive clock allocation required')
    return number


def _binding(value: Any) -> Json:
    raw = closed(value, _BINDING)
    text(raw['boot_id'])
    before, after, _unix = (deadline_seconds(raw[key]) for key in
        ('monotonic_before_bits', 'monotonic_after_bits', 'unix_anchor_bits'))
    if before > after:
        raise ContractError('original checked clock anchor is reversed')
    return raw


def clock_deadlines(clock_binding: Json, *, inputs: Json, post_host_seconds: Any) -> Json:
    """Use the caller's original anchor and explicit total; perform no I/O."""
    clock = _binding(clock_binding)
    raw = closed(inputs, _INPUTS)
    a, e, w, f = (_positive(raw[key]) for key in
        ('admission_seconds', 'execution_seconds', 'wall_seconds', 'finalization_seconds'))
    p = _positive(post_host_seconds)
    if a > e or e + f > w or not p < w - e:
        raise ContractError('original clock allocations are inconsistent')
    t = deadline_seconds(clock['monotonic_after_bits'])
    admission, execution, finish = t + a, t + e, t + w
    term = finish - p
    if not execution < term < finish:
        raise ContractError('original deadline ordering is not representable')
    result = {key: deadline_bits(value) for key, value in zip(
        ('admission_bits', 'execution_bits', 'term_bits', 'kill_bits', 'finish_bits'),
        (admission, execution, term, term, finish), strict=True)}
    return validate_deadlines(result)


def operation_clock(operation: Json, *, clock_binding: Json, inputs: Json,
                    post_host_seconds: Any) -> Json:
    """Bind the already-built immutable operation without a hash cycle."""
    validate_operation(operation)
    deadlines = clock_deadlines(clock_binding, inputs=inputs, post_host_seconds=post_host_seconds)
    if operation['schema_version'] != 2 or operation['deadlines'] != deadlines:
        raise ContractError('clock recipe differs from actual operation2 bits')
    return {'schema_version': 1, 'kind': 'rust-player-operation-clock-v1',
        'operation_id': operation['operation_id'], 'operation_sha256': digest(operation),
        'run_id': operation['run_id'], 'config_sha256': operation['config_sha256'],
        'clock_binding': dict(clock_binding), 'inputs': dict(inputs),
        'post_host_seconds': post_host_seconds, 'recipe': _RECIPE, 'deadlines': dict(deadlines)}


def validate_operation_clock(value: Any, *, operation: Json, session_clock_binding: Json,
                             session_finish: str, boot_id: str, now: float) -> Json:
    """Validate saved DATA against current original inputs; never recapture time."""
    raw = record(value, 'rust-player-operation-clock-v1', {'operation_id', 'operation_sha256',
        'run_id', 'config_sha256', 'clock_binding', 'inputs', 'post_host_seconds', 'recipe', 'deadlines'})
    session = _binding(session_clock_binding)
    expected = operation_clock(operation, clock_binding=raw['clock_binding'],
        inputs=raw['inputs'], post_host_seconds=raw['post_host_seconds'])
    if raw != expected or raw['clock_binding']['boot_id'] != session['boot_id'] or boot_id != session['boot_id']:
        raise ContractError('original operation clock/config/SHA/boot differs')
    validate_deadlines(raw['deadlines'], session_finish=session_finish)
    if _positive(now) >= deadline_seconds(raw['deadlines']['execution_bits']):
        raise ContractError('original execution clock expired')
    return raw
