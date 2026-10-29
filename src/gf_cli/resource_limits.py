"""Optional address-space fuse for standalone dataset commands only."""
from __future__ import annotations

import resource
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gf_common.domain import ContractError, integer


def virtual_bytes() -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmSize:'):
            return int(line.split()[1]) * 1024
    raise ContractError('address-space limit requires Linux VmSize telemetry')


@contextmanager
def dataset_address_limit(limit_mib: int | None) -> Iterator[None]:
    """Scope the soft limit; never change the hard limit or a training process."""
    if limit_mib is None:
        yield
        return
    integer(limit_mib, 1, (2**63 - 1) // (1024 * 1024))
    if sys.platform != 'linux':
        raise ContractError('dataset address-space limit is supported on Linux only')
    requested = limit_mib * 1024 * 1024
    original_soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and requested > hard:
        raise ContractError('requested address-space limit exceeds existing hard limit')
    # An operator request must not lift a stricter inherited limit.
    effective = requested if original_soft == resource.RLIM_INFINITY else min(requested, original_soft)
    if virtual_bytes() >= effective:
        raise ContractError('current virtual address space already reaches requested limit')
    resource.setrlimit(resource.RLIMIT_AS, (effective, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (original_soft, hard))
