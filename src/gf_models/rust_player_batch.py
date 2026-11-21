"""Fixed native MLP serving geometry, input banks and no-grad scoring."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

import numpy as np

from gf_common.domain import integer
from gf_common.structured_layout import StructuredLayout
from gf_models.rust_player_cuda import NativeMlpRowCudaProfile
from gf_models.structured_config import InferenceBindingV2, ModelConfigV3, ModelConfigV4, dimensions

_FLOAT = np.dtype(np.float32)
_LONG = np.dtype(np.int64)
_BOOL = np.dtype(np.bool_)
_CANDIDATES = frozenset(("kinds", "targets", "numerics", "present", "references"))


def _bucket(length: int, cap: int) -> int:
    return min(1 << (length - 1).bit_length(), cap) if length else 0


def _buckets(cap: int) -> tuple[int, ...]:
    return (*tuple(1 << power for power in range((cap - 1).bit_length())), cap)


@dataclass(frozen=True)
class BatchShape:
    rows: int
    candidates: int
    events: int


@dataclass(frozen=True)
class BatchGeometry:
    rows: int
    max_candidates: int
    max_events: int

    def shape(self, candidates: int, events: int) -> BatchShape:
        integer(candidates, 1, self.max_candidates)
        integer(events, 0, self.max_events)
        return BatchShape(self.rows, _bucket(candidates, self.max_candidates), _bucket(events, self.max_events))

    def shapes(self) -> tuple[BatchShape, ...]:
        return tuple(BatchShape(self.rows, candidates, events) for candidates, events in
                     product(_buckets(self.max_candidates), (0, *_buckets(self.max_events))))


@dataclass(frozen=True)
class InputField:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    start: int
    stop: int


@dataclass(frozen=True)
class BatchInputs:
    fields: tuple[InputField, ...]
    # Offsets measure elements within each final-dtype H2D allocation.
    sizes: dict[np.dtype[Any], int]

    @property
    def bank_bytes(self) -> tuple[int, ...]:
        return tuple(size * dtype.itemsize for dtype, size in self.sizes.items())

    @classmethod
    def compile(cls, shape: BatchShape, *, positions: int, layout: StructuredLayout) -> BatchInputs:
        r, c, e = shape.rows, shape.candidates, shape.events
        state_n, state_c = len(layout.state_numeric_fields), len(layout.state_categories)
        card_n, card_c = len(layout.card_numeric_fields), len(layout.card_categories)
        event_n, event_c = len(layout.event_numeric_fields), len(layout.event_categories)
        # Same fields and final dtypes as native/src/player_runtime/numeric.rs.
        schema = (
            ("state_numerics", (state_n,), _FLOAT),
            ("state_present", (state_n,), _FLOAT),
            ("state_categories", (state_c,), _LONG),
            ("state_category_present", (state_c,), _FLOAT),
            ("opponent_learned", (18,), _LONG),
            ("opponent_mask", (18,), _FLOAT),
            ("card_categories", (18, card_c), _LONG),
            ("card_category_suffix", (18, card_c - 2), _LONG),
            ("card_numerics", (18, card_n), _FLOAT),
            ("card_present", (18, card_n), _FLOAT),
            ("card_mask", (18,), _BOOL),
            ("card_order", (18,), _FLOAT),
            ("history", (layout.history_width,), _FLOAT),
            ("interval_numerics", (e, event_n), _FLOAT),
            ("interval_present", (e, event_n), _FLOAT),
            ("interval_categories", (e, event_c), _LONG),
            ("kinds", (c,), _LONG),
            ("targets", (c,), _LONG),
            ("numerics", (c, 4), _FLOAT),
            ("present", (c, 4), _FLOAT),
            ("references", (c, positions), _LONG),
            ("mask", (c,), _BOOL),
            ("last_event", (), _LONG),
        )
        fields = []
        sizes: dict[np.dtype[Any], int] = {}
        for name, tail, dtype in schema:
            start = sizes.get(dtype, 0)
            sizes[dtype] = stop = start + r * prod(tail)
            fields.append(InputField(name, (r, *tail), dtype, start, stop))
        return cls(tuple(fields), sizes)


@dataclass(frozen=True)
class PackedRows:
    inputs: BatchInputs
    banks: dict[np.dtype[Any], np.ndarray]

    def transfer(self, *, device: str, roots: dict[np.dtype[Any], Any]) -> dict[str, Any]:
        import torch

        for dtype, bank in self.banks.items():
            roots[dtype] = torch.from_numpy(bank).to(device, non_blocking=False)
        return {field.name: roots[field.dtype][field.start:field.stop].view(field.shape)
                for field in self.inputs.fields}


def pack_native_rows(rows: list[dict[str, Any]], inputs: BatchInputs) -> PackedRows:
    """Directly copy native final-dtype operands into three contiguous banks."""
    banks = {dtype: np.zeros(size, dtype=dtype) for dtype, size in inputs.sizes.items()}
    for field in inputs.fields:
        name = field.name
        out = banks[field.dtype][field.start:field.stop].reshape(field.shape)
        if name == "mask":
            for index, row in enumerate(rows):
                out[index, :len(row["inputs"]["kinds"])] = True
            out[len(rows):, 0] = True  # A dummy still has a finite normalization.
        elif name == "last_event":
            if rows[0]["inputs"]["interval_numerics"].shape[0]:
                for index, row in enumerate(rows):
                    out[index] = row["inputs"]["interval_numerics"].shape[0] - 1
        elif name in _CANDIDATES or name.startswith("interval_"):
            if name == "references":
                out.fill(18)
            for index, row in enumerate(rows):
                value = row["inputs"][name]
                out[index, :len(value)] = value
        else:
            for index, row in enumerate(rows):
                out[index] = row["inputs"][name]
    return PackedRows(inputs, banks)


@dataclass(frozen=True)
class BatchEstimate:
    inputs: BatchInputs
    peak_bytes: int
    largest_tensor_bytes: int

    @property
    def input_bytes(self) -> int:
        return sum(self.inputs.bank_bytes)


def estimate_batch(model: ModelConfigV3 | ModelConfigV4, binding: InferenceBindingV2, *,
                   layout: StructuredLayout, shape: BatchShape,
                   row_profile: NativeMlpRowCudaProfile | None = None) -> BatchEstimate:
    """Peak overlapping live CUDA storage and largest individual tensor root."""
    r, c, e = shape.rows, shape.candidates, shape.events
    inputs = BatchInputs.compile(shape, positions=model.max_positions, layout=layout)
    row_profile = row_profile or NativeMlpRowCudaProfile(model, binding, layout=layout)
    row = row_profile.estimate(interval_events=e, candidates=c, input_leaf_bytes=0)
    d = dimensions(model, layout=layout)
    pos, first, event = model.max_positions, model.heads.actor_hidden_sizes[0], model.event.hidden_size
    chunk = min(binding.candidate_chunk, c)
    # Frame and per-candidate execution retain the scalar lifetimes at fixed
    # width. Dense event gather additionally holds its own event vector.
    phases = {name: r * size for name, size in row.additional_phases.items()}
    phases["frame_state_opponent_mask"] = r * (4 * layout.state_width + 4 * 18 * (len(layout.card_categories[0]) + 2))
    if e:
        # Dense no-cuDNN GRU returns sequence, full hidden stack and last-step
        # gather. Dense CUDA cells consume one step at a time.
        phases["frame_event_gru"] += 4 * r * event
        for name in ("frame_history", "frame_concat", "frame_projection", "frame_temporal_dense"):
            if name in phases:
                phases[name] += 4 * r * event
    # All rows use the same position weights and arange. The einsum can
    # materialize the noncontiguous weight operand once for the whole group.
    weight_copy = 4 * pos * first * d["card_actor"]
    projection_row = 4 * pos * first
    projection_peak = max(weight_copy + r * 18 * projection_row, r * 38 * projection_row)
    phases["candidate_projection"] = (
        r * (4 * (model.temporal.hidden_size + 18 * d["card_actor"] + first)
             + 4 * c * layout.candidate_width) + projection_peak)
    score_base = (4 * r * (model.temporal.hidden_size + 18 * d["card_actor"] + first
                           + c * layout.candidate_width)
                  + r * 19 * projection_row + 8 * pos)
    full, partial = divmod(c, binding.candidate_chunk)
    largest_index = 0
    for name, width, previous in (
        ("candidate_chunk_full", binding.candidate_chunk if full else 0, (full - 1) * binding.candidate_chunk),
        ("candidate_chunk_partial", partial, full * binding.candidate_chunk),
    ):
        if not width:
            continue
        # The existing geometric phase describes reduction and head work,
        # with one shared position arange in the batched case.
        phases[name] -= 8 * (r - 1) * pos
        # CUDA advanced indexing materializes noncontiguous broadcast indexes
        # together. The refs slice also copies when a chunk spans only part of
        # each row's candidate bank. These roots retire before the reduction.
        copies = sum((width * pos > 1, r * width > 1, r > 1 and width < c))
        index_bytes = 8 * r * width * pos
        largest_index = max(largest_index, index_bytes if copies else 0)
        gathered_and_x = 4 * r * width * first * (pos + 1)
        phases[f"{name}_index"] = (score_base + 4 * r * previous
                                  + 8 * r + copies * index_bytes + gathered_and_x)
    phases["output_mask"] = phases["output_logsoftmax"] + r * c
    phases["output_critic"] += 4 * r * c  # Probabilities already exist when the critic runs.
    for name in ("output_finite", "output_probability_cpu_transfer",
                 "output_logs_cpu_return", "output_value_cpu_return"):
        del phases[name]
    # Logs, probabilities and values coexist with the combined D2H buffer.
    phases["output_return"] = 8 * r * (2 * c + 1)

    # A packed bank is one allocation. Views of its leaves require no more
    # capacity; the remaining maxima include final-dtype and int64 one-hot
    # roots, expanded projection/gather, dense GRU and combined result.
    largest = max(*inputs.bank_bytes,
        4 * r * max(layout.state_width, d["frame"], model.temporal.hidden_size, *model.state.hidden_sizes),
        8 * r * 18 * (len(layout.card_categories[0]) + 2),
        8 * r * max(len(v) + 2 for v in layout.state_categories),
        8 * r * 18 * max(len(v) + 2 for v in layout.card_categories),
        4 * r * 18 * max(d["card_input"], d["card_actor"], layout.card_direct_width,
                         model.hand.latent_size, *model.hand.hidden_sizes),
        4 * r * 18 * 18 * model.hand.attention_heads if model.hand.kind == "attention" else 0,
        4 * r * 18 * model.hand.attention_feedforward if model.hand.kind == "attention" else 0,
        8 * r * e * max(len(v) + 2 for v in layout.event_categories),
        4 * r * e * max(layout.event_width, event),
        4 * r * max(3 * event, model.event.num_layers * event) if e else 0,
        8 * r * c * max(len(layout.action_kinds) + 2, len(layout.targets) + 2),
        4 * r * c * layout.candidate_width,
        weight_copy,
        4 * r * pos * 19 * first,
        4 * r * chunk * pos * first,
        largest_index,
        4 * r * chunk * max(model.heads.actor_hidden_sizes),
        4 * r * max(model.heads.critic_hidden_sizes, default=1),
        4 * r * (2 * c + 1))
    return BatchEstimate(inputs, sum(inputs.bank_bytes) + max(phases.values()), largest)


def score_numeric_batch(network: Any, raw: dict[str, Any]) -> tuple[Any, Any, Any]:
    """All participating rows use the same fixed geometry, including dummies."""
    context, cards = network._encode_frame_operands(None, raw.__getitem__,
                                                   fixed_event_indices=raw["last_event"])
    scores = network._batched_logits(raw, context, cards).masked_fill(~raw["mask"], -float("inf"))
    logs = scores.log_softmax(-1)
    return logs, logs.exp(), network.value(context)
