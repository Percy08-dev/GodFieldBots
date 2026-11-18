"""Optional serving observation outside the unchanged numerical registry."""
from __future__ import annotations

import time
from typing import Any

DIAGNOSTIC_PAYLOAD_BYTES=8*8


def read_allocator(block: Any) -> dict[str,Any] | None:
    """One bounded coherent attempt; a dead writer cannot block serving control."""
    try:
        lock=block.get_lock()
        if not lock.acquire(False):return None
        try:values=tuple(block.get_obj())
        finally:lock.release()
        if len(values)!=8 or values[0]==0 or values[1] not in (1,2):return None
        return {'scope':'process-lifetime','reset':'never','boundary':'post-forward-host' if values[1]==1 else 'post-publication-host',
            'synchronization':'none-added','sample_sequence':values[0],'sample_monotonic_ns':values[6],
            'allocated_bytes':values[2],'reserved_bytes':values[3],'max_allocated_bytes':values[4],'max_reserved_bytes':values[5],
            'shared_payload_bytes':DIAGNOSTIC_PAYLOAD_BYTES}
    except Exception:return None  # noqa: BLE001 - unavailable optional observation, never an inference error.


def make_served_registry(*,device: str,cache_entries: int,cache_weight_bytes: int,
                         diagnostic_block: Any,visibility: str | None=None) -> Any:
    from gf_models.shared_registry import make_registry
    registry=make_registry(device=device,cache_entries=cache_entries,cache_weight_bytes=cache_weight_bytes,visibility=visibility)
    import torch

    class ServingPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.registry=registry
            self._diagnostic=diagnostic_block  # CPU diagnostics only; neither parameters nor buffers.

        def _sample(self,boundary: int) -> None:
            if device=='cpu':return
            try:
                lock=self._diagnostic.get_lock()
                if not lock.acquire(False):return
                try:
                    raw=self._diagnostic.get_obj()
                    stats=torch.cuda.memory_stats(device)  # Exactly one read, no added synchronization/reset.
                    sequence=raw[0]+1
                    raw[0]=0
                    raw[1]=boundary
                    for index,key in enumerate(('allocated_bytes.all.current','reserved_bytes.all.current',
                        'allocated_bytes.all.peak','reserved_bytes.all.peak'),start=2):raw[index]=stats[key]
                    raw[6]=time.monotonic_ns()
                    raw[7]=1
                    raw[0]=sequence
                finally:lock.release()
            except Exception:return  # noqa: BLE001 - diagnostic failure must preserve original math/exception.

        def load_policy_weights(self,weights: Any) -> None:
            self.registry.load_policy_weights(weights)
            self._sample(2)

        def forward(self,batch: Any) -> Any:
            result=self.registry(batch)
            self._sample(1)
            return result

    return ServingPolicy()
