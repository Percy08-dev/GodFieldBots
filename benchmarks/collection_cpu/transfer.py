"""Transfer-only packing at existing tensor math helper seams."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from gf_common.domain import ContractError
from gf_models.inference_experiment import NETWORK_STORAGE_BYTES, TensorPacker


class TransferPacker:
    """Owned CPU+device capacities, including growth overlap and 8-byte padding.

    A phase cannot be overwritten until the original final-forward synchronization
    has confirmed consumption. Reset synchronizes before scrubbing; a failed
    synchronization deliberately leaves storage retained for process retirement.
    """
    def __init__(self, device: Any, mode: str, *, max_bytes: int, max_tensor_bytes: int) -> None:
        import torch
        if mode not in ('T1', 'T2', 'T3'):
            raise ContractError('closed transfer mode required')
        self.torch, self.device, self.mode = torch, torch.device(device), mode
        self.max_bytes, self.max_tensor_bytes = max_bytes, max_tensor_bytes
        self.buffers: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.inflight: set[str] = set()
        self.pack_counts: dict[str, int] = {}
        self.transfers = self.payload_bytes = 0
        self.high_water_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return sum(cpu.numel()*cpu.element_size() + (0 if cpu is dev else dev.numel()*dev.element_size())
                   for cpu, dev in self.buffers.values())

    def pack(self, values: dict[str, Any], dtypes: dict[str, Any], *, group: str) -> dict[str, Any]:
        if group in self.inflight:
            raise ContractError('experimental phase still in-flight')
        specs = []
        offsets: dict[str, int] = {}
        sizes: dict[str, int] = {}
        for name, value in values.items():
            shape = TensorPacker._shape(value)
            count = int(np.prod(shape))
            kind = np.dtype(dtypes[name])
            if kind not in (np.dtype('float32'), np.dtype('int64'), np.dtype('bool')):
                raise ContractError('unsupported transfer dtype')
            key = kind.str if self.mode == 'T1' else 'arena'
            start = sizes.get(key, 0)
            if self.mode != 'T1':
                start = (start+7)//8*8
            offsets[name] = start
            sizes[key] = start+count*kind.itemsize
            specs.append((name, shape, kind, count, key))
        multiplier = 1 if self.device.type == 'cpu' else 2
        growth = 0
        for key, size in sizes.items():
            if size > self.max_tensor_bytes:
                raise ContractError('experimental tensor cap exceeded before allocation')
            old = self.buffers.get((group, key))
            if old is None or old[0].numel()*old[0].element_size() < size:
                growth += size*multiplier
        peak = self.retained_bytes+growth
        if peak > self.max_bytes:
            raise ContractError('experimental CPU+GPU capacity cap exceeded before allocation')
        self.high_water_bytes = max(self.high_water_bytes, peak)
        for key, size in sizes.items():
            old = self.buffers.get((group, key))
            if old is None or old[0].numel()*old[0].element_size() < size:
                kind = np.dtype(key) if key != 'arena' else np.dtype('uint8')
                dtype = self.torch.from_numpy(np.empty(0, kind)).dtype
                # CPU mechanism tests exercise the arena without requiring a CUDA
                # runtime. Actual T3 always uses the pinned CUDA-owned allocation.
                cpu = self.torch.empty(size//kind.itemsize, dtype=dtype,
                    pin_memory=self.mode == 'T3' and self.device.type == 'cuda')
                dev = cpu if self.device.type == 'cpu' else self.torch.empty_like(cpu, device=self.device)
                cpu.zero_()
                if old is not None:
                    old[0].zero_()
                    if old[1] is not old[0]:
                        old[1].zero_()
                self.buffers[group, key] = cpu, dev
        result = {}
        # Mark BEFORE copies: any exception retains immutable in-flight buffers.
        self.inflight.add(group)
        for name, shape, kind, count, key in specs:
            cpu, dev = self.buffers[group, key]
            start, stop = offsets[name], offsets[name]+count*kind.itemsize
            cpu_view = cpu.view(self.torch.uint8)[start:stop].view(self.torch.from_numpy(np.empty(0, kind)).dtype).view(shape)
            cpu_view.numpy()[...] = values[name]
            result[name] = dev.view(self.torch.uint8)[start:stop].view(cpu_view.dtype).view(shape)
        for key, size in sizes.items():
            cpu, dev = self.buffers[group, key]
            if dev is not cpu:
                dev.view(self.torch.uint8)[:size].copy_(cpu.view(self.torch.uint8)[:size], non_blocking=self.mode == 'T3')
                self.transfers += 1
            self.payload_bytes += size
        self.pack_counts[group] = self.pack_counts.get(group, 0)+1
        return result

    def consumed(self) -> None:
        """Caller has completed the ORIGINAL final-forward synchronization."""
        self.inflight.clear()

    def reset(self) -> None:
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        for cpu, dev in self.buffers.values():
            cpu.zero_()
            if dev is not cpu:
                dev.zero_()
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        self.inflight.clear()
        self.buffers.clear()

    def private_empty(self) -> bool:
        return not self.inflight and all(not bool(cpu.count_nonzero()) and not bool(dev.count_nonzero())
                                        for cpu, dev in self.buffers.values())


class TransferExperiment:
    def __init__(self, network: Any, mode: str) -> None:
        import torch
        self.network, self.mode, self.torch = network, mode, torch
        self.packer = TransferPacker(network.device, mode, max_bytes=NETWORK_STORAGE_BYTES,
                                    max_tensor_bytes=network.config.resources.max_tensor_bytes)
        self.original_factorized = network.factorized_logits
        self.original_frame = network._encode_frame_calculate
        self.refs: Any = None
        network._encode_frame_calculate = self.frame
        network._candidate_features = self.features
        network._references = self.references
        network.factorized_logits = self.factorized

    def require_inference(self) -> None:
        if self.torch.is_grad_enabled():
            raise ContractError('transfer experiment requires no-grad inference')

    def pack(self, values: dict[str, Any], kinds: dict[str, Any], group: str) -> dict[str, Any]:
        self.require_inference()
        started = time.perf_counter()
        try:
            return self.packer.pack(values, kinds, group=group)
        finally:
            if self.network._measure_input_packing:
                self.network._input_packing_seconds += time.perf_counter()-started

    def frame(self, tokens: Any) -> Any:
        names = ['state_numerics', 'state_present', 'state_categories', 'state_category_present',
                 'opponent_learned', 'opponent_mask', 'card_categories', 'card_numerics', 'card_present',
                 'card_category_suffix', 'card_mask', 'card_order', 'history']
        if len(tokens.interval_numerics):
            names.extend(('interval_numerics', 'interval_present', 'interval_categories'))
        integer_fields = {'state_categories','opponent_learned','card_categories','card_category_suffix','interval_categories'}
        raw = self.pack({name: self.network._frame_operand_values(tokens, name) for name in names},
            {name: np.int64 if name in integer_fields else np.bool_ if name == 'card_mask' else np.float32 for name in names}, 'frame')
        return self.network._encode_frame_operands(tokens, raw.__getitem__)

    def features(self, tokens: Any) -> Any:
        rows = [(*tokens.slots[tokens.offsets[i]:tokens.offsets[i+1]],
                 *([18]*(self.network.config.max_positions-tokens.offsets[i+1]+tokens.offsets[i])))
                for i in range(len(tokens.kinds))]
        raw = self.pack({**{name: getattr(tokens,name) for name in ('kinds','targets','numerics','present')}, 'refs': rows},
            {'kinds':np.int64,'targets':np.int64,'numerics':np.float32,'present':np.float32,'refs':np.int64}, 'candidates')
        self.refs = raw['refs']
        return self.network._candidate_feature_operands(raw.__getitem__)

    def references(self, tokens: Any, start: int, end: int) -> Any:
        if self.refs is None:
            raise ContractError('candidate refs pack absent')
        return self.refs[start:end]

    def factorized(self, context: Any, cards: Any, tokens: Any, chunk_size: int | None = None,
                   checkpoint: bool = True, *, _operands: Any = None) -> Any:
        self.require_inference()
        if _operands is not None or chunk_size not in (None,256):
            raise ContractError('transfer inference requires original atomic inputs/chunk256')
        try:
            return self.original_factorized(context, cards, tokens, 256, checkpoint=False)
        finally:
            self.refs = None

    def consumed(self) -> None:
        self.packer.consumed()

    def reset(self) -> None:
        self.packer.reset()
        self.refs = None

    def private_empty(self) -> bool:
        return self.refs is None and self.packer.private_empty()

    def diagnostics(self) -> dict[str, Any]:
        return {'mode': self.mode, 'retained_bytes': self.packer.retained_bytes,
                'high_water_bytes': self.packer.high_water_bytes, 'packing_transfers': self.packer.transfers,
                'packing_payload_bytes': self.packer.payload_bytes, 'packs': self.packer.pack_counts,
                'network_storage_cap': NETWORK_STORAGE_BYTES, 'inflight': sorted(self.packer.inflight)}


def install(network: Any, mode: str) -> TransferExperiment:
    if mode not in ('T1','T2','T3') or hasattr(network, '_inference_experiment'):
        raise ContractError('one closed transfer installation per inference network')
    experiment = TransferExperiment(network, mode)
    network._inference_experiment = experiment
    return experiment
