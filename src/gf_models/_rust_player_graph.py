"""One measured fixed-shape CUDA forward owned by the Rust collection service."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, cast

import numpy as np

from gf_common.domain import ContractError
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT
from gf_models.rust_player_batch import (
    BatchEstimate,
    BatchShape,
    PackedRows,
    pack_native_rows,
    score_numeric_batch,
)
from gf_models.rust_player_device import CudaCheckpointProfile
from gf_models.structured_config import ModelConfigV3, ModelConfigV4

COLLECTION_GRAPH_SHAPE = BatchShape(32, 64, 16)
# Measured private allocator reservations for the pinned kernels and geometry.
_EXPECTED_POOL_BYTES = {"3": 44 * 1024**2, "4": 54 * 1024**2}

_MODEL3 = ModelConfigV3()
_MODEL4_BASE = ModelConfigV4()
_MODEL4 = replace(_MODEL4_BASE, state=replace(_MODEL4_BASE.state, hidden_sizes=(192,)),
    hand=replace(_MODEL4_BASE.hand, hidden_sizes=(192,), latent_size=192),
    temporal=replace(_MODEL4_BASE.temporal, hidden_size=192, num_layers=2),
    heads=replace(_MODEL4_BASE.heads, actor_hidden_sizes=(192,), critic_hidden_sizes=(192,)))


def _graph_runtime_supported(torch: Any, device: str) -> bool:
    return (torch.__version__ == "2.14.1+cu132" and torch.version.cuda == "13.2"
            and torch.cuda.get_allocator_backend() == "native"
            and torch.cuda.get_device_capability(device) == (12, 0))


def expected_collection_pool(profile: CudaCheckpointProfile, *, torch: Any, device: str) -> int | None:
    model = profile.model
    if profile.layout is not CORRECTED_CARD_INSTANCE_LAYOUT or profile.binding.candidate_chunk < COLLECTION_GRAPH_SHAPE.candidates:
        return None
    measured: ModelConfigV3 | ModelConfigV4
    if type(model) is ModelConfigV3:
        measured = _MODEL3
    elif type(model) is ModelConfigV4:
        measured = _MODEL4
    else:
        return None
    # Limits change the admitted envelope; these caps do not change the fixed forward.
    if replace(model, resources=measured.resources, event=replace(model.event,
            max_interval=measured.event.max_interval, max_episode=measured.event.max_episode)) != measured:
        return None
    return _EXPECTED_POOL_BYTES[model.version] if _graph_runtime_supported(torch, device) else None


@dataclass
class CollectionCudaGraph:
    # Logs, probabilities and values are already in the private pool.
    replay_bytes: ClassVar[int] = 4 * COLLECTION_GRAPH_SHAPE.rows * (2 * COLLECTION_GRAPH_SHAPE.candidates + 1)

    checkpoint: str
    estimate: BatchEstimate
    device: str
    roots: dict[np.dtype[Any], Any] = field(default_factory=dict)
    inputs: dict[str, Any] | None = None
    outputs: tuple[Any, Any, Any] | None = None
    pool: tuple[int, int] | None = None
    stream: Any = None
    graph: Any = None

    def capture(self, torch: Any, network: Any) -> int:
        """Called under the host Torch lock and its actual device operation."""
        torch.cuda.synchronize(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        # Keep the pool identity even if capture_end reports an error.
        self.pool = torch.cuda.graph_pool_handle()
        with torch.no_grad(), torch.cuda.stream(self.stream):
            dummy = pack_native_rows([], self.estimate.inputs)
            self.inputs = dummy.transfer(device=self.device, roots=self.roots)
            for _ in range(3):
                score_numeric_batch(network, self.inputs)
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.capture_begin(pool=self.pool)
            try:
                self.outputs = score_numeric_batch(network, self.inputs)
            finally:
                self.graph.capture_end()
            self.stream.synchronize()
        private = self._pool_bytes(torch)
        if not private:
            raise ContractError("captured CUDA Graph has no observable private pool")
        return private + sum(root.untyped_storage().nbytes() for root in self.roots.values())

    def transfer(self, torch: Any, packed: PackedRows) -> None:
        for dtype, bank in packed.banks.items():
            self.roots[dtype].copy_(torch.from_numpy(bank), non_blocking=False)

    def replay(self) -> tuple[Any, Any, Any]:
        # PyTorch launches into the caller's current stream. Its blocking D2H
        # fences this replay before the same buffers can be reused.
        self.graph.replay()
        return cast(tuple[Any, Any, Any], self.outputs)

    def retire(self, torch: Any) -> None:
        """Called after the actual CUDA fence and after temporary aliases retire."""
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self.outputs = self.inputs = None
        self.roots.clear()
        self.stream = None
        if self._pool_bytes(torch):
            # Native allocator may still cache segments after Graph.reset().
            torch.cuda.empty_cache()
            if self._pool_bytes(torch):
                raise ContractError("retired CUDA Graph still has private pool reservations")

    def _pool_bytes(self, torch: Any) -> int:
        if self.pool is None:
            return 0
        return sum(segment["total_size"] for segment in
                   torch.cuda.memory_snapshot(mempool_id=self.pool, include_traces=False))
