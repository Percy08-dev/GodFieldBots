"""Lazy structured actor/critic. Pure token APIs; no trainer, files or Simulator.

Current-weight prefix recomputation is detached; all suffix encoders remain in
one autograd graph. Inference caches are ephemeral, parameter-version checked,
and never form an authoritative learning input or a persisted latent format.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any

from gf_common.domain import ContractError, integer
from gf_common.structured_tokens import (
    ACTION_KINDS,
    CARD_CATEGORIES,
    EVENT_CATEGORIES,
    SOURCES,
    STATE_CATEGORIES,
    STATE_WIDTH,
    TARGETS,
    CandidateTokens,
    FrameTokens,
    validate_candidate_tokens,
    validate_frame_tokens,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    dimensions,
    parameter_shapes,
    recorded_token_signature,
    resource_estimate,
    validate_config,
)


@dataclass
class TemporalCache:
    model_identity: int
    parameter_versions: tuple[int, ...]
    position: int = 0
    past: Any = field(default=None, repr=False)
    owner: tuple[int, str] | None = None
    revision: int = -1
    event_end: int = 0
    keys: Any = field(default=None, repr=False)
    values: Any = field(default=None, repr=False)

    def reset(self) -> None:
        """Start a fresh participant episode without reallocating owned storage."""
        for buffer in (self.keys, self.values, self.past):
            if buffer is not None:
                buffer.zero_()
        self.position, self.owner, self.revision, self.event_end = 0, None, -1, 0


def make_network(model: ModelConfigV2, seed: int,
                 binding: InferenceBindingV2 | None = None) -> Any:
    validate_config(model)
    runtime_binding = InferenceBindingV2() if binding is None else binding
    resource_estimate(model, runtime_binding)  # BEFORE Torch import / parameter allocation.
    integer(seed, 0, 2**63 - 1)
    from gf_models.torch_runtime import torch_runtime
    torch = torch_runtime(runtime_binding.device)
    nn, functional = torch.nn, torch.nn.functional
    base: Any = nn.Module
    d = dimensions(model)

    def mlp(widths: tuple[int, ...], final_relu: bool = False) -> Any:
        layers = []
        for index, (source, target) in enumerate(pairwise(widths)):
            layers.append(nn.Linear(source, target))
            if index < len(widths) - 2 or final_relu:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @contextmanager
    def no_cudnn() -> Iterator[None]:
        # Avoid cudnn.flags: Torch2.14 mixes its legacy TF32 getter with
        # the required new IEEE precision API even for CPU execution.
        enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = False
            yield
        finally:
            torch.backends.cudnn.enabled = enabled

    class AttentionLayer(base):
        def __init__(self, width: int, heads: int, feedforward: int) -> None:
            super().__init__()
            self.width, self.heads = width, heads
            self.query, self.key, self.value, self.output = (nn.Linear(width, width) for _ in range(4))
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.feedforward1, self.feedforward2 = nn.Linear(width, feedforward), nn.Linear(feedforward, width)

        def projections(self, x: Any) -> tuple[Any, Any, Any]:
            normalized = self.norm1(x)
            return tuple(layer(normalized).reshape(x.shape[0], x.shape[1], self.heads,
                         self.width // self.heads).transpose(1, 2)
                         for layer in (self.query, self.key, self.value))

        def attend(self, query: Any, key: Any, value: Any, allowed: Any = None) -> Any:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                attended = functional.scaled_dot_product_attention(query, key, value, attn_mask=allowed,
                                                                   dropout_p=0., is_causal=False)
            return attended.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[2], self.width)

        def finish(self, x: Any, attended: Any) -> Any:
            x = x + self.output(attended)
            return x + self.feedforward2(self.feedforward1(self.norm2(x)).relu())

        def forward(self, x: Any, valid: Any, past: Any = None,
                    offset: int = 0, causal: bool = False) -> tuple[Any, Any]:
            query, key, value = self.projections(x)
            if past is not None:
                key, value = torch.cat((past[0], key), 2), torch.cat((past[1], value), 2)
            allowed = valid[:, None, None, :].expand(-1, 1, x.shape[1], -1)
            if causal:
                query_positions = offset + torch.arange(x.shape[1], device=x.device)
                key_positions = torch.arange(key.shape[2], device=x.device)
                allowed = allowed & (key_positions[None, :] <= query_positions[:, None])[None, None]
            x = self.finish(x, self.attend(query, key, value, allowed))
            return x.masked_fill(~valid[:, offset:offset + x.shape[1], None], 0), (key, value)

        def step(self, x: Any, keys: Any, values: Any, position: int) -> Any:
            query, key, value = self.projections(x)
            keys[:, :, position:position + 1].copy_(key)
            values[:, :, position:position + 1].copy_(value)
            # One query attends exactly the written past/current prefix. There
            # is no future storage in these views and no offset-zero causal mask.
            return self.finish(x, self.attend(query, keys[:, :, :position + 1],
                                               values[:, :, :position + 1]))

    class ActorCritic(base):
        def __init__(self) -> None:
            super().__init__()
            self.config, self.binding = model, runtime_binding
            self._measure_input_packing = False
            self._input_packing_seconds = 0.
            self.state_encoder = (mlp((STATE_WIDTH,
                                       *model.state.hidden_sizes, model.state.output_size), True)
                                  if model.state.kind == "mlp" else nn.Identity())
            self.definition_embedding = nn.Embedding(len(CARD_CATEGORIES[0]) + 2, model.card.definition_embedding, padding_idx=0)
            self.ability_embedding = nn.Embedding(len(CARD_CATEGORIES[1]) + 2, model.card.ability_embedding, padding_idx=0)
            self.card_encoder = nn.Linear(d["card_input"], model.card.output_size)
            if model.hand.kind != "direct":
                self.hand_encoder = mlp((model.card.output_size, *model.hand.hidden_sizes, model.hand.latent_size), True)
            if model.hand.kind == "attention":
                self.hand_attention = nn.ModuleList([AttentionLayer(model.card.output_size, model.hand.attention_heads,
                                                                    model.hand.attention_feedforward)
                                                    for _ in range(model.hand.attention_layers)])
            from gf_common.structured_tokens import EVENT_WIDTH
            self.event_encoder = nn.GRU(EVENT_WIDTH, model.event.hidden_size, model.event.num_layers, batch_first=True)
            self.frame_projection = nn.Linear(d["frame"], model.temporal.hidden_size)
            if model.model_id == "gru":
                self.temporal = nn.GRU(model.temporal.hidden_size, model.temporal.hidden_size,
                                       model.temporal.num_layers, batch_first=True)
            elif model.model_id == "transformer":
                self.temporal = nn.ModuleList([AttentionLayer(model.temporal.hidden_size, model.temporal.num_heads,
                                                             model.temporal.feedforward_size)
                                              for _ in range(model.temporal.num_layers)])
            self.actor = mlp((d["actor"], *model.heads.actor_hidden_sizes, 1))
            self.critic = mlp((model.temporal.hidden_size, *model.heads.critic_hidden_sizes, 1))
            positions = torch.arange(model.temporal.max_decisions, dtype=torch.float32)[:, None]
            frequencies = torch.exp(torch.arange(0, model.temporal.hidden_size, 2, dtype=torch.float32)
                                    * (-math.log(10000.) / model.temporal.hidden_size))
            angles = positions * frequencies
            pe = torch.empty((model.temporal.max_decisions, model.temporal.hidden_size))
            pe[:, 0::2] = angles.sin()
            pe[:, 1::2] = angles[:, :model.temporal.hidden_size // 2].cos()
            self.register_buffer("positions", pe, persistent=False)

        @property
        def device(self) -> Any:
            return self.frame_projection.weight.device

        def _begin_packing_timing(self) -> None:
            if torch.is_grad_enabled() or self._measure_input_packing:
                raise ContractError("packing timing is participant-inference only")
            self._input_packing_seconds = 0.
            self._measure_input_packing = True

        def _end_packing_timing(self) -> float:
            self._measure_input_packing = False
            return self._input_packing_seconds * 1000

        def _tensor(self, values: Any, *, dtype: Any = None, device: Any = None) -> Any:
            if not self._measure_input_packing:
                return torch.tensor(values, dtype=dtype, device=self.device if device is None else device)
            started = time.perf_counter()
            try:
                return torch.tensor(values, dtype=dtype, device=self.device if device is None else device)
            finally:
                # Host wall time for raw construction/copy submission. CUDA
                # allocation/transfers may wait prior stream work; this is not
                # a measurement of isolated H2D or GPU kernel durations.
                self._input_packing_seconds += time.perf_counter() - started

        def _float(self, values: Any) -> Any:
            raw = self._tensor(values, dtype=torch.float32, device=self.device)
            return raw.sign() * raw.abs().log1p()

        def _onehot(self, values: Any, categories: tuple[tuple[str, ...], ...]) -> Any:
            raw = self._tensor(values, dtype=torch.long, device=self.device)
            return torch.cat([functional.one_hot(raw[..., index], len(vocabulary) + 2).float()
                              for index, vocabulary in enumerate(categories)], -1)

        def encode_frame(self, tokens: FrameTokens) -> tuple[Any, Any]:
            validate_frame_tokens(tokens)
            if tokens.aggregate_window != model.event.aggregate_window or tokens.recipe_signature != recorded_token_signature(model):
                raise ContractError("aggregate window recipe mismatch")
            if len(tokens.interval_numerics) > model.event.max_interval:
                raise ContractError("configured interval event limit exceeded")
            state_input = torch.cat((self._float(tokens.state_numerics),
                                     self._tensor(tokens.state_present, dtype=torch.float32, device=self.device),
                                     self._onehot(tokens.state_categories, STATE_CATEGORIES),
                                     self._tensor(tokens.state_category_present, dtype=torch.float32, device=self.device),
                                     functional.one_hot(self._tensor(tokens.opponent_learned, dtype=torch.long, device=self.device),
                                                        len(CARD_CATEGORIES[0]) + 2).float().flatten()))
            # PAD opponent slots have no representation; category0 is not a fact.
            opponent_width = len(CARD_CATEGORIES[0]) + 2
            state_input[-18 * opponent_width:] *= self._tensor(
                [value != 0 for value in tokens.opponent_learned], dtype=torch.float32,
                device=self.device).repeat_interleave(opponent_width)
            state = self.state_encoder(state_input)
            cats = self._tensor(tokens.card_categories, dtype=torch.long, device=self.device)
            numbers = torch.cat((self._float(tokens.card_numerics),
                                 self._tensor(tokens.card_present, dtype=torch.float32, device=self.device)), -1)
            card_input = torch.cat((self.definition_embedding(cats[:, 0]), self.ability_embedding(cats[:, 1]),
                                    self._onehot(tuple(row[2:] for row in tokens.card_categories), CARD_CATEGORIES[2:]), numbers), -1)
            mask = self._tensor(tokens.card_mask, dtype=torch.bool, device=self.device)
            cards = self.card_encoder(card_input).relu().masked_fill(~mask[:, None], 0)
            source = functional.one_hot(cats[:, -1], len(SOURCES) + 2).float().masked_fill(~mask[:, None], 0)
            order = self._float(tuple(row[8] for row in tokens.card_numerics))[:, None]
            actor_cards = torch.cat((cards, source, mask.float()[:, None], order), -1)
            if model.hand.kind == "direct":
                hand = torch.cat((self._onehot(tokens.card_categories, CARD_CATEGORIES), numbers), -1)
                hand = hand.masked_fill(~mask[:, None], 0).flatten()
            else:
                hand_cards = cards[None]
                if model.hand.kind == "attention":
                    for layer in self.hand_attention:
                        hand_cards, _ = layer(hand_cards, mask[None])
                latent = self.hand_encoder(hand_cards[0]).masked_fill(~mask[:, None], 0)
                if model.hand.pooling == "sum":
                    hand = latent.sum(0)
                elif model.hand.pooling == "mean":
                    hand = latent.sum(0) / mask.sum().clamp_min(1)
                else:
                    hand = latent.masked_fill(~mask[:, None], -torch.inf).max(0).values
                    hand = torch.where(mask.any(), hand, torch.zeros_like(hand))
            if tokens.interval_numerics:
                event_input = torch.cat((self._float(tokens.interval_numerics),
                                         self._tensor(tokens.interval_present, dtype=torch.float32, device=self.device),
                                         self._onehot(tokens.interval_categories, EVENT_CATEGORIES)), -1)
                with no_cudnn():
                    _, hidden = self.event_encoder(event_input[None])
                event = hidden[-1, 0]
            else:
                event = torch.zeros(model.event.hidden_size, device=self.device)
            frame = torch.cat((state, hand, event, self._float(tokens.history)))
            return self.frame_projection(frame).relu(), actor_cards

        def encode_temporal(self, frames: Any, past: Any = None, offset: int = 0) -> tuple[Any, Any]:
            length = frames.shape[0]
            integer(offset, 0, model.temporal.max_decisions)
            if not 0 < length or offset + length > model.temporal.max_decisions:
                raise ContractError("own-decision history limit exceeded")
            if model.model_id == "mlp":
                return frames, None
            if model.model_id == "gru":
                with no_cudnn():
                    output, hidden = self.temporal(frames[None], past)
                return output[0], hidden
            x = frames[None] + self.positions[offset:offset + length][None]
            valid = torch.ones((1, offset + length), dtype=torch.bool, device=self.device)
            caches = []
            for index, layer in enumerate(self.temporal):
                x, cache = layer(x, valid, None if past is None else past[index], offset, True)
                caches.append(cache)
            return x[0], tuple(caches)

        def encode_history(self, frames: tuple[FrameTokens, ...] | list[FrameTokens],
                           suffix_length: int | None = None) -> tuple[Any, Any]:
            integer(len(frames), 1, model.temporal.max_decisions)
            suffix_length = len(frames) if suffix_length is None else suffix_length
            integer(suffix_length, 1, len(frames))
            for index, frame in enumerate(frames):
                validate_frame_tokens(frame)
                if (frame.aggregate_window != model.event.aggregate_window or frame.recipe_signature != recorded_token_signature(model)
                        or len(frame.interval_numerics) > model.event.max_interval):
                    raise ContractError("historical frame window/interval cap mismatch")
                if frame.event_end > model.event.max_episode:
                    raise ContractError("configured episode event cap exceeded")
                if index and ((frame.actor, frame.episode) != (frames[0].actor, frames[0].episode)
                        or frame.revision <= frames[index - 1].revision
                        or frame.event_start != frames[index - 1].event_end):
                    raise ContractError("own-decision history continuity mismatch")
            if frames[0].event_start != 0:
                raise ContractError("full own-decision history must start at first event interval")
            resource_estimate(model, runtime_binding, decisions=len(frames),
                              interval_events=max(len(frame.interval_numerics) for frame in frames),
                              candidates=1, suffix_decisions=suffix_length if torch.is_grad_enabled() else 1)
            if model.model_id == "mlp":
                # Historical frames are validated above, but MLP consumes only
                # the current detailed state/hand/interval/aggregate observation.
                return self.encode_frame(frames[-1])
            prefix = len(frames) - suffix_length
            past = None
            if prefix and model.model_id != "mlp":
                with torch.no_grad():
                    prefix_frames = torch.stack([self.encode_frame(frame)[0] for frame in frames[:prefix]])
                    _, past = self.encode_temporal(prefix_frames)
            # Encode all suffix frames with gradients, including the endpoint's
            # shared cards used by the actor; no detached saved latent shortcut.
            encoded = [self.encode_frame(frame) for frame in frames[prefix:]]
            output, _ = self.encode_temporal(torch.stack([value[0] for value in encoded]), past, prefix)
            return output[-1], encoded[-1][1]

        def new_cache(self) -> TemporalCache:
            # Reserve the full configured capacity, not only the first observed
            # prefix. Admission happens before allocating either bounded buffer.
            resource_estimate(model, runtime_binding)
            cache = TemporalCache(id(self), tuple(parameter._version for parameter in self.parameters()))
            if model.model_id == "transformer":
                shape = (model.temporal.num_layers, 1, model.temporal.num_heads,
                         model.temporal.max_decisions,
                         model.temporal.hidden_size // model.temporal.num_heads)
                cache.keys = torch.zeros(shape, dtype=torch.float32, device=self.device)
                cache.values = torch.zeros(shape, dtype=torch.float32, device=self.device)
            return cache

        def step(self, frame: FrameTokens, cache: TemporalCache) -> tuple[Any, Any]:
            if frame.recipe_signature != recorded_token_signature(model):
                raise ContractError('cache frame aggregate recipe mismatch')
            if (cache.model_identity != id(self)
                    or cache.parameter_versions != tuple(parameter._version for parameter in self.parameters())):
                raise ContractError("stale/foreign inference cache")
            if torch.is_grad_enabled():
                raise ContractError("inference cache requires no_grad")
            integer(cache.position, 0, model.temporal.max_decisions - 1)
            if frame.event_end > model.event.max_episode:
                raise ContractError("configured episode event cap exceeded")
            if cache.position == 0 and frame.event_start != 0:
                raise ContractError("cache must start at first own-decision interval")
            if (cache.owner is not None and ((frame.actor, frame.episode) != cache.owner
                    or frame.revision <= cache.revision or frame.event_start != cache.event_end)):
                raise ContractError("cached own-decision continuity mismatch")
            if model.model_id == "transformer" and (cache.keys is None or cache.values is None
                    or cache.keys.device != self.device or cache.values.device != self.device):
                raise ContractError("missing/foreign Transformer cache storage")
            encoded, cards = self.encode_frame(frame)
            if model.model_id == "transformer":
                x = encoded[None, None] + self.positions[cache.position:cache.position + 1][None]
                for index, layer in enumerate(self.temporal):
                    x = layer.step(x, cache.keys[index], cache.values[index], cache.position)
                output = x[0]
            else:
                output, cache.past = self.encode_temporal(encoded[None], cache.past, cache.position)
            cache.position += 1
            cache.owner, cache.revision, cache.event_end = (frame.actor, frame.episode), frame.revision, frame.event_end
            return output[0], cards

        def _candidate_features(self, tokens: CandidateTokens) -> Any:
            kinds = functional.one_hot(self._tensor(tokens.kinds, device=self.device), len(ACTION_KINDS) + 2).float()
            targets = functional.one_hot(self._tensor(tokens.targets, device=self.device), len(TARGETS) + 2).float()
            return torch.cat((kinds, targets, self._float(tokens.numerics),
                              self._tensor(tokens.present, dtype=torch.float32, device=self.device)), -1)

        def _references(self, tokens: CandidateTokens, start: int, end: int) -> Any:
            rows = [list(tokens.slots[tokens.offsets[index]:tokens.offsets[index + 1]]) for index in range(start, end)]
            return self._tensor([row + [18] * (model.max_positions - len(row)) for row in rows],
                                dtype=torch.long, device=self.device)

        def dense_logits(self, context: Any, cards: Any, tokens: CandidateTokens) -> Any:
            if tokens.recipe_signature != recorded_token_signature(model):
                raise ContractError('actor candidate aggregate recipe mismatch')
            # The reference materializes all rows. Its admission uses that full
            # width, never the production factorized chunk reservation.
            resource_estimate(model, replace(runtime_binding, candidate_chunk=len(tokens.kinds)),
                              decisions=1, interval_events=0, candidates=len(tokens.kinds))
            refs = self._references(tokens, 0, len(tokens.kinds))
            padded = torch.cat((cards, torch.zeros_like(cards[:1])), 0)
            dense = torch.cat((context[None].expand(len(tokens.kinds), -1), self._candidate_features(tokens),
                               padded[refs].flatten(1)), -1)
            return self.actor(dense).squeeze(-1)

        def factorized_logits(self, context: Any, cards: Any, tokens: CandidateTokens,
                              chunk_size: int | None = None, checkpoint: bool = True) -> Any:
            if tokens.recipe_signature != recorded_token_signature(model):
                raise ContractError('actor candidate aggregate recipe mismatch')
            chunk_size = runtime_binding.candidate_chunk if chunk_size is None else chunk_size
            integer(chunk_size, 1, 65536)
            resource_estimate(model, replace(runtime_binding, candidate_chunk=chunk_size),
                              decisions=1, interval_events=0, candidates=len(tokens.kinds))
            first = self.actor[0]
            width = model.temporal.hidden_size
            shared = functional.linear(context, first.weight[:, :width], first.bias)
            features = self._candidate_features(tokens)
            candidate_width = features.shape[1]
            card_weights = first.weight[:, width + candidate_width:].reshape(first.out_features, model.max_positions, d["card_actor"]).permute(1, 0, 2)
            projection = torch.einsum("phd,sd->psh", card_weights, cards)
            projection = torch.cat((projection, torch.zeros_like(projection[:, :1])), 1)
            positions = torch.arange(model.max_positions, device=self.device)[None]
            numeric_weights = first.weight[:, width:width + candidate_width]

            def chunk_forward(shared_arg: Any, projection_arg: Any, feature_arg: Any,
                              refs_arg: Any, weight_arg: Any) -> Any:
                x = shared_arg[None] + functional.linear(feature_arg, weight_arg)
                x = x + projection_arg[positions, refs_arg].sum(1)
                return self.actor[1:](x).squeeze(-1)

            logits = []
            for start in range(0, len(tokens.kinds), chunk_size):
                end = min(len(tokens.kinds), start + chunk_size)
                refs = self._references(tokens, start, end)
                args = (shared, projection, features[start:end], refs, numeric_weights)
                if checkpoint and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint as recompute
                    logits.append(recompute(chunk_forward, *args, use_reentrant=False))
                else:
                    logits.append(chunk_forward(*args))
            return torch.cat(logits)

        def distribution(self, context: Any, cards: Any, tokens: CandidateTokens,
                         mask: Any = None, *, dense: bool = False, chunk_size: int | None = None) -> tuple[Any, Any]:
            logits = (self.dense_logits(context, cards, tokens) if dense else
                      self.factorized_logits(context, cards, tokens, chunk_size))
            if mask is not None:
                if mask.dtype != torch.bool or tuple(mask.shape) != tuple(logits.shape) or not mask.any():
                    raise ContractError("invalid/all-masked candidate mask")
                logits = logits.masked_fill(~mask, -torch.inf)
            if not torch.isfinite(logits).any() or torch.isnan(logits).any() or torch.isposinf(logits).any():
                raise ContractError("nonfinite actor output")
            # Exactly ONE normalization across every chunk's candidate logits.
            logprob = logits.log_softmax(-1)
            entropy = -(logprob.exp() * logprob.masked_fill(~torch.isfinite(logprob), 0)).sum()
            return logprob, entropy

        def value(self, context: Any) -> Any:
            return self.critic(context).squeeze(-1)

        def forward(self, frames: tuple[FrameTokens, ...] | list[FrameTokens], tokens: CandidateTokens,
                    *, suffix_length: int | None = None, dense: bool = False,
                    chunk_size: int | None = None, mask: Any = None) -> tuple[Any, Any, Any]:
            if not frames:
                raise ContractError("nonempty frame history required")
            validate_candidate_tokens(tokens, frames[-1], model.max_positions)
            if len(tokens.kinds) > runtime_binding.max_candidates:
                raise ContractError("candidate admission limit exceeded")
            # Admit the effective scoring allocation and the full endpoint as
            # one envelope before encoding. Separate history/chunk checks can
            # each fit while their combined retained allocations exceed budget.
            effective_chunk = (len(tokens.kinds) if dense else runtime_binding.candidate_chunk
                               if chunk_size is None else chunk_size)
            effective_binding = replace(runtime_binding, candidate_chunk=effective_chunk)
            resource_estimate(model, effective_binding, decisions=len(frames),
                              interval_events=max(len(frame.interval_numerics) for frame in frames),
                              candidates=len(tokens.kinds),
                              suffix_decisions=(len(frames) if suffix_length is None else suffix_length)
                              if torch.is_grad_enabled() else 1)
            context, cards = self.encode_history(frames, suffix_length)
            logprob, entropy = self.distribution(context, cards, tokens, mask, dense=dense, chunk_size=chunk_size)
            value = self.value(context)
            if not torch.isfinite(value):
                raise ContractError("nonfinite critic output")
            return logprob, value, entropy

    original_rng = torch.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        network = ActorCritic()
        for name, module in network.named_modules():
            if isinstance(module, nn.GRU):
                for parameter_name, parameter in module.named_parameters():
                    if parameter_name.startswith("weight"):
                        for gate in parameter.chunk(3, 0):
                            nn.init.orthogonal_(gate, 1.)
                    else:
                        nn.init.zeros_(parameter)
            elif isinstance(module, nn.Embedding):
                # PAD is zero and carries no category fact. Orthogonalize the
                # trainable rows themselves, so removing PAD does not change gain.
                nn.init.orthogonal_(module.weight[1:], 1.)
                with torch.no_grad():
                    module.weight[0].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if module is network.actor[-1]:
                    gain = .01
                elif module is network.critic[-1] or name.rsplit(".", 1)[-1] in (
                        "query", "key", "value", "output", "feedforward2"):
                    gain = 1.
                else:
                    gain = math.sqrt(2)
                nn.init.orthogonal_(module.weight, gain)
                nn.init.zeros_(module.bias)
        actual_shapes = {name: tuple(parameter.shape) for name, parameter in network.named_parameters()}
        if actual_shapes != parameter_shapes(model):
            raise ContractError("preallocation parameter shape proof mismatch")
        return network.to(runtime_binding.device)
    finally:
        torch.set_rng_state(original_rng)
