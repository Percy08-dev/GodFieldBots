"""Explicit, participant-only inference experiments; ordinary construction stays eager.

The compiler region is only the shared candidate scorer. Frame/event/temporal
encoding, validation, cache control, normalization and sampling remain eager.
No training, file, Simulator or management modules are imported here.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from gf_common.domain import ContractError
from gf_models.structured_config import resource_estimate

# Seven simultaneous envelopes cover six cached identities and one transition.
# 7 * 12MiB = 84MiB < 96MiB. The remaining 12MiB covers packing metadata,
# temporary aligned scorer inputs and outputs. Compiler workspace is measured,
# not certified by this managed tensor envelope.
NETWORK_STORAGE_BYTES = 12 * 1024**2
WORKER_STORAGE_BYTES = 96 * 1024**2


@dataclass(frozen=True)
class ExperimentOptions:
    packing: str = 'original'
    chunk: int = 256
    alignment: int = 0
    feature_alignment: int = 0
    compile: str = 'none'
    precision: str = 'ieee'

    def __post_init__(self) -> None:
        if (self.packing not in ('original', 'bulk', 'reuse')
                or type(self.chunk) is not int or self.chunk not in (256, 512, 1024, 2048)
                or type(self.alignment) is not int or self.alignment not in (0, 8, 16, 32, 64, 128, 256)
                or type(self.feature_alignment) is not int or self.feature_alignment not in (0, 8, 16, 32, 64, 128, 256)
                or self.compile not in ('none', 'default', 'reduce-overhead')
                or self.precision not in ('ieee', 'tf32')):
            raise ContractError('unsupported closed inference experiment options')

    def identity(self) -> dict[str, Any]:
        return asdict(self)


class TensorPacker:
    """Three contiguous typed CPU arrays and at most three device copies per pack.

    Each group owns bounded storage. Growing storage includes both old and new
    capacities in its preallocation check. Every live element is overwritten;
    reset clears entire capacities, including previously larger input tails.
    """
    def __init__(self, device: Any, *, reuse: bool, max_bytes: int,
                 max_tensor_bytes: int) -> None:
        import torch
        self.torch, self.device, self.reuse = torch, torch.device(device), reuse
        self.max_bytes, self.max_tensor_bytes = max_bytes, max_tensor_bytes
        self.buffers: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.transfers = self.payload_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return sum(cpu.numel() * cpu.element_size() + (0 if dev is cpu else dev.numel() * dev.element_size())
                   for cpu, dev in self.buffers.values())

    def pack(self, values: dict[str, Any], dtypes: dict[str, Any], *, group: str = 'default') -> dict[str, Any]:
        specs: dict[str, list[tuple[str, tuple[int, ...], int, Any]]] = {}
        # Shape discovery avoids an unbounded preliminary NumPy allocation.
        for name, value in values.items():
            shape = self._shape(value)
            count = int(np.prod(shape))
            dtype = np.dtype(dtypes[name])
            if dtype not in (np.dtype('float32'), np.dtype('int64'), np.dtype('bool')):
                raise ContractError('experimental typed storage dtype')
            specs.setdefault(dtype.str, []).append((name, shape, count, value))
        required = {kind: sum(row[2] for row in rows) for kind, rows in specs.items()}
        multiplier = 1 if self.device.type == 'cpu' else 2
        growth = 0
        for kind, count in required.items():
            size = count * np.dtype(kind).itemsize
            if size > self.max_tensor_bytes:
                raise ContractError('experimental storage tensor cap exceeded before allocation')
            old = self.buffers.get((group, kind))
            if old is None or old[0].numel() < count or not self.reuse:
                growth += size * multiplier
        if self.retained_bytes + growth > self.max_bytes:
            raise ContractError('experimental storage cap exceeded before allocation')
        result = {}
        for kind, rows in specs.items():
            count = required[kind]
            key = (group, kind)
            old = self.buffers.get(key)
            if old is None or old[0].numel() < count or not self.reuse:
                cpu = self.torch.empty(count, dtype=self.torch.from_numpy(np.empty(0, dtype=kind)).dtype)
                device = cpu if self.device.type == 'cpu' else self.torch.empty_like(cpu, device=self.device)
                if old is not None:
                    old[0].zero_()
                    if old[1] is not old[0]:
                        old[1].zero_()
                self.buffers[key] = (cpu, device)
            cpu, device = self.buffers[key]
            array = cpu.numpy()
            offset = 0
            for name, shape, size, value in rows:
                # NumPy assignment converts directly into the owned typed array.
                array[offset:offset + size].reshape(shape)[...] = value
                result[name] = device[offset:offset + size].view(shape)
                offset += size
            if device is not cpu:
                device[:count].copy_(cpu[:count])
                self.transfers += 1
            self.payload_bytes += count * np.dtype(kind).itemsize
        return result

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...]:
        if hasattr(value, 'shape'):
            return tuple(value.shape)
        if isinstance(value, (tuple, list)):
            return (len(value), *TensorPacker._shape(value[0])) if value else (0,)
        return ()

    def reset(self) -> None:
        for cpu, device in self.buffers.values():
            cpu.zero_()
            if device is not cpu:
                device.zero_()
        # Dropping also prevents stale views from being reused across episodes.
        self.buffers.clear()

    def private_empty(self) -> bool:
        return all(not bool(cpu.count_nonzero().item()) and not bool(device.count_nonzero().item())
                   for cpu, device in self.buffers.values())


class InferenceExperiment:
    def __init__(self, network: Any, options: ExperimentOptions, *, resident: bool) -> None:
        import torch
        self.network, self.options, self.torch = network, options, torch
        if resident and options.compile == 'reduce-overhead':
            raise ContractError('reduce-overhead compiler private captured storage cannot be scrubbed for residency')
        self.packer = TensorPacker(network.device, reuse=options.packing == 'reuse',
                                  max_bytes=NETWORK_STORAGE_BYTES,
                                  max_tensor_bytes=network.config.resources.max_tensor_bytes)
        self.stats: dict[str, Any] = {'backend_graphs': 0, 'backend_compile_seconds': 0., 'calls': 0,
            'scorer_host_seconds': 0., 'valid_candidates': 0, 'padded_rows': 0, 'failures': [], 'fallbacks': 0,
            'compiled_regions': [] if options.compile == 'none' else ['candidate_chunk'],
            'eager_regions': ['frame', 'event_gru', 'temporal', 'validation', 'normalization', 'sampling'],
            'graph_break_policy': 'fullgraph scorer; graph breaks fail',
            'resident_available': options.compile != 'reduce-overhead',
            'padding_kind': 'candidate feature K zero-extension' if options.feature_alignment else
                            'candidate-row-bucket-only; no inner dimension alignment',
            'unaligned_regions': ['card projection einsum', 'frame encoder', 'temporal encoder']}
        self.original_frame = network._encode_frame_calculate
        self.original_features = network._candidate_features
        self.original_refs = network._references
        self.original_factorized = network.factorized_logits
        self.original_chunk = network._candidate_chunk_calculate
        self.compiled_chunk = None
        self._admit()
        self.apply_precision()
        if options.compile != 'none':
            # Pinned default has cudagraphs=False; make that explicit. The other
            # mode intentionally enables captured compiler-owned storage (replay).
            def backend(graph: Any, example_inputs: Any, **kwargs: Any) -> Any:
                from torch._dynamo.backends.registry import lookup_backend
                started = time.perf_counter()
                self.stats['backend_graphs'] += 1
                try:
                    patches = kwargs.pop('options', {})
                    inductor_backend: Any = lookup_backend('inductor')
                    return inductor_backend(graph, example_inputs, config_patches=patches, **kwargs)
                except Exception as error:
                    self.stats['failures'].append(type(error).__name__ + ': ' + str(error)[:2048])
                    raise
                finally:
                    self.stats['backend_compile_seconds'] += time.perf_counter() - started
            self.compiled_chunk = torch.compile(self.original_chunk, backend=backend, fullgraph=True,
                options={'triton.cudagraphs': options.compile == 'reduce-overhead'})
        network._encode_frame_calculate = self.frame_inputs
        network._candidate_features = self.candidate_features
        network._references = self.references
        network._candidate_chunk_calculate = self.chunk
        network.factorized_logits = self.factorized

    def _require_inference(self) -> None:
        if self.torch.is_grad_enabled():
            raise ContractError('experimental helpers require no-grad participant inference')

    def apply_precision(self) -> None:
        self.torch.backends.cuda.matmul.fp32_precision = self.options.precision

    def _admit(self, *, candidates: int | None = None) -> None:
        padded = self.options.chunk + max(0, self.options.alignment - 1)
        estimate = resource_estimate(self.network.config, replace(self.network.binding, candidate_chunk=padded),
            layout=self.network.layout, candidates=candidates)
        if estimate['total_bytes'] + 2*NETWORK_STORAGE_BYTES > min(
                self.network.binding.managed_budget_bytes, self.network.config.resources.max_owned_bytes):
            raise ContractError('experimental storage exceeds existing managed cap')

    def _pack(self, values: dict[str, Any], dtypes: dict[str, Any], *, group: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return self.packer.pack(values, dtypes, group=group)
        finally:
            if self.network._measure_input_packing:
                self.network._input_packing_seconds += time.perf_counter()-started

    def frame_inputs(self, tokens: Any) -> Any:
        self._require_inference()
        if self.options.packing == 'original':
            return self.original_frame(tokens)
        names = ['state_numerics', 'state_present', 'state_categories', 'state_category_present',
                 'opponent_learned', 'opponent_mask', 'card_categories', 'card_numerics', 'card_present',
                 'card_category_suffix', 'card_mask', 'card_order', 'history']
        if len(tokens.interval_numerics):
            names.extend(('interval_numerics', 'interval_present', 'interval_categories'))
        kinds = {'state_categories', 'opponent_learned', 'card_categories', 'card_category_suffix', 'interval_categories'}
        raw = self._pack({name: self.network._frame_operand_values(tokens, name) for name in names},
            {name: np.int64 if name in kinds else np.bool_ if name == 'card_mask' else np.float32 for name in names},
            group='frame')
        return self.network._encode_frame_operands(tokens, raw.__getitem__)

    def candidate_features(self, tokens: Any) -> Any:
        self._require_inference()
        if self.options.packing == 'original':
            return self.original_features(tokens)
        raw = self._pack({name: getattr(tokens, name) for name in ('kinds', 'targets', 'numerics', 'present')},
            {'kinds': np.int64, 'targets': np.int64, 'numerics': np.float32, 'present': np.float32}, group='candidates')
        return self.network._candidate_feature_operands(raw.__getitem__)

    def references(self, tokens: Any, start: int, end: int) -> Any:
        if self.options.packing == 'original':
            return self.original_refs(tokens, start, end)
        rows = [tuple(tokens.slots[tokens.offsets[i]:tokens.offsets[i+1]]) for i in range(start, end)]
        rows = [(*row, *([18] * (self.network.config.max_positions-len(row)))) for row in rows]
        return self._pack({'refs': rows}, {'refs': np.int64}, group='references')['refs']

    def factorized(self, context: Any, cards: Any, tokens: Any, chunk_size: int | None = None,
                   checkpoint: bool = True, *, _operands: Any = None) -> Any:
        self._require_inference()
        if _operands is not None:
            raise ContractError("experimental inference does not accept learner/owned operands")
        if chunk_size is not None and chunk_size != self.options.chunk:
            raise ContractError('explicit chunk differs from inference experiment')
        # Installation admits the configured maximum population/chunk/padding
        # jointly. The original scorer retains its per-call admission below.
        self.stats['valid_candidates'] += len(tokens.kinds)
        return self.original_factorized(context, cards, tokens, self.options.chunk, checkpoint=False)

    def chunk(self, shared: Any, projection: Any, features: Any, refs: Any, weight: Any, positions: Any) -> Any:
        self._require_inference()
        count = features.shape[0]
        alignment = self.options.alignment
        padded = count if not alignment else ((count + alignment - 1) // alignment) * alignment
        tail = padded - count
        feature_alignment = self.options.feature_alignment
        feature_tail = (0 if not feature_alignment else
                        (-features.shape[1]) % feature_alignment)
        # Temporary padding includes inputs, weight copy, and output. The
        # all-worker envelope reserves an additional full 12MiB for the one
        # active identity's temporary storage, on top of seven 12MiB owners.
        temporary_sizes = (padded*(features.shape[1]+feature_tail)*features.element_size(),
            padded*refs.shape[1]*refs.element_size(),
            weight.shape[0]*(weight.shape[1]+feature_tail)*weight.element_size(), padded*4)
        # Captured scorer outputs must be cloned outside compile before another
        # invocation can reuse them. Charge the whole decision's retained clones,
        # bounded by the existing maximum population, alongside current padding.
        captured = self.options.compile == 'reduce-overhead'
        owned_output_bytes = self.network.binding.max_candidates*4 if captured else 0
        # Feature-K then row padding can briefly retain both intermediate and
        # final copies. Count both before either pad or clone allocation.
        if (sum(temporary_sizes) + temporary_sizes[0] + owned_output_bytes > NETWORK_STORAGE_BYTES-65536 or
                max(temporary_sizes) > self.network.config.resources.max_tensor_bytes):
            raise ContractError('experimental temporary storage cap exceeded before allocation')
        if feature_tail:
            features = self.torch.nn.functional.pad(features, (0, feature_tail))
            weight = self.torch.nn.functional.pad(weight, (0, feature_tail))
        if tail:
            functional = self.torch.nn.functional
            features = functional.pad(features, (0, 0, 0, tail))
            refs = functional.pad(refs, (0, 0, 0, tail), value=18)
        self.stats['padded_rows'] += tail
        self.stats['calls'] += 1
        started = time.perf_counter()
        try:
            if captured:
                # This isolated scorer call is one capture generation. Its only
                # cross-call outputs are copied to eager-owned storage below.
                self.torch.compiler.cudagraph_mark_step_begin()
            result = (self.original_chunk if self.compiled_chunk is None else self.compiled_chunk)(
                shared, projection, features, refs, weight, positions)
            # Slice before the unchanged global finite checks/normalization.
            real = result[:count]
            return real.clone() if captured else real
        except Exception as error:
            self.stats['failures'].append(type(error).__name__ + ': ' + str(error)[:2048])
            raise
        finally:
            self.stats['scorer_host_seconds'] += time.perf_counter() - started

    def reset(self) -> None:
        self.packer.reset()

    def private_empty(self) -> bool:
        return self.options.compile != 'reduce-overhead' and self.packer.private_empty()

    def diagnostics(self) -> dict[str, Any]:
        return {**self.stats, 'packing_transfers': self.packer.transfers,
                'packing_payload_bytes': self.packer.payload_bytes, 'retained_bytes': self.packer.retained_bytes,
                'network_storage_cap': NETWORK_STORAGE_BYTES, 'worker_storage_envelope': WORKER_STORAGE_BYTES}


def install(network: Any, options: ExperimentOptions, *, resident: bool = False) -> InferenceExperiment:
    if type(options) is not ExperimentOptions or hasattr(network, '_inference_experiment'):
        raise ContractError('one closed inference experiment installation per network')
    experiment = InferenceExperiment(network, options, resident=resident)
    network._inference_experiment = experiment
    return experiment
