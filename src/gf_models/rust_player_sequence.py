"""Service-thread custody for provisional own-decision temporal state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gf_common.domain import ContractError

GameKey = tuple[str, str, str]
SeatKey = tuple[str, str, str, int, int, str]


@dataclass
class SeatState:
    position: int = 0
    revision: int = -1
    event_end: int = 0
    hidden: Any = None
    keys: Any = None
    values: Any = None


@dataclass
class PendingStep:
    key: SeatKey
    state: SeatState
    revision: int
    event_end: int
    hidden: Any = None

    def scalar(self, network: Any, encoded: Any) -> Any:
        state = self.state
        if network.config.model_id == "transformer" and state.keys is None:
            state.keys, state.values = network._new_temporal_storage()
        context, self.hidden = network._temporal_step(
            encoded, state.position, state.hidden, state.keys, state.values)
        return context


class SequenceState:
    """Only an ordered native acknowledgement advances each participant cache."""

    def __init__(self) -> None:
        self.seats: dict[SeatKey, SeatState] = {}

    def stage(self, row: dict[str, Any], identity: dict[str, Any], network: Any) -> PendingStep:
        key: SeatKey = (identity["run"], identity["trial"], identity["game"],
                        identity["player"], identity["epoch"], identity["checkpoint"])
        state = self.seats.get(key)
        if state is None:
            state = SeatState()
        start, end = row["event_start"], row["event_end"]
        model = network.config
        if (identity["request"] != state.position or identity["revision"] <= state.revision
                or start != state.event_end or not 0 <= end - start <= model.event.max_interval
                or end > model.event.max_episode or row["event_length"] > end - start
                or state.position >= model.temporal.max_decisions):
            raise ContractError("native cached own-decision continuity or model limit mismatch")
        return PendingStep(key, state, identity["revision"], end)

    def accept(self, pending: list[PendingStep | None], accepted: list[bool]) -> None:
        for step, deliver in zip(pending, accepted, strict=True):
            if step is None or not deliver:
                continue
            state = step.state
            state.hidden = step.hidden
            state.position += 1
            state.revision, state.event_end = step.revision, step.event_end
            self.seats[step.key] = state

    def retire_game(self, key: GameKey) -> None:
        for seat in [seat for seat in self.seats if seat[:3] == key]:
            del self.seats[seat]


def cache_bytes(model: Any) -> int:
    temporal = model.temporal
    hidden = 4 * temporal.num_layers * temporal.hidden_size
    if model.model_id == "gru":
        return hidden
    if model.model_id == "transformer":
        return 2 * temporal.max_decisions * hidden
    return 0


def temporal_scratch(model: Any, *, rows: int, position: int, grouped: bool) -> tuple[int, int]:
    """Estimated extra peak and largest tensor beyond the frame/head lifetimes.

    Committed banks and independently retained GRU outputs are reserved by the
    collection. Group scratch includes only gathered inputs and batched outputs.
    """
    from gf_models.actor_cuda import _attention_scratch

    t = model.temporal
    vector = 4 * rows * t.hidden_size
    stack = vector * t.num_layers
    if model.model_id == "gru":
        # Gathered old hidden and new batched hidden are live together. Each
        # no-cuDNN GRU cell holds input/hidden gates and their nonlinear output.
        geometry = 2 * stack if grouped else 0
        return geometry + 10 * vector, max(stack, 3 * vector)
    if model.model_id == "transformer":
        prefix = stack * (position + 1) if grouped else 0
        # Forced MATH SDPA materializes a scaled transposed K prefix for the
        # current layer while the old K/V and the score matrix remain live.
        scaled_key = vector * (position + 1)
        layer = rows * _attention_scratch(t.hidden_size, t.num_heads, t.feedforward_size,
            1, position + 1, scaled_key_bytes=scaled_key // rows)
        largest = max(prefix, scaled_key, 4 * rows * t.feedforward_size,
                      4 * rows * t.num_heads * (position + 1))
        return 2 * prefix + layer, largest
    return 0, 0


def grouped_step(network: Any, encoded: Any, steps: list[PendingStep]) -> Any:
    import torch

    if network.config.model_id == "gru":
        t = network.config.temporal
        past = encoded.new_zeros((t.num_layers, len(encoded), t.hidden_size))
        for i, step in enumerate(steps):
            if step.state.hidden is not None:
                past[:, i:i + 1].copy_(step.state.hidden)
        context, hidden = network._temporal_step(encoded, 0, past)
        for i, step in enumerate(steps):
            step.hidden = hidden[:, i:i + 1].clone(memory_format=torch.contiguous_format)
        return context

    position = steps[0].state.position
    t = network.config.temporal
    shape = (t.num_layers, len(encoded), t.num_heads, position + 1, t.hidden_size // t.num_heads)
    keys, values = encoded.new_zeros(shape), encoded.new_zeros(shape)
    for i, step in enumerate(steps):
        if step.state.keys is None:
            step.state.keys, step.state.values = network._new_temporal_storage()
        if position:
            keys[:, i:i + 1, :, :position].copy_(step.state.keys[..., :position, :])
            values[:, i:i + 1, :, :position].copy_(step.state.values[..., :position, :])
    context, _ = network._temporal_step(encoded, position, keys=keys, values=values)
    for i, step in enumerate(steps):
        step.state.keys[..., position, :].copy_(keys[:, i:i + 1, :, position])
        step.state.values[..., position, :].copy_(values[:, i:i + 1, :, position])
    return context
