"""Closed stdlib-only v2 model architecture and resource admission.

Runtime device/chunk binding is deliberately outside architecture identity.
Shape admission runs before optional Torch import or parameter allocation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import prod
from typing import Any

from gf_common import structured_layout as legacy_layout_constants
from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured_history import ACTIVE_AGGREGATE, HISTORICAL_AGGREGATE, history_width
from gf_common.structured_layout import (
    CARD_INSTANCE_AGGREGATE,
    CORRECTED_CARD_INSTANCE_LAYOUT,
    LEGACY_LAYOUT,
    StructuredLayout,
    is_card_instance_layout,
    require_layout,
)
from gf_common.structured_tokens import (
    signature,
)

# Historical exported aliases retain their exact legacy meaning.
CANDIDATE_WIDTH = legacy_layout_constants.CANDIDATE_WIDTH
CARD_CATEGORIES = legacy_layout_constants.CARD_CATEGORIES
CARD_DIRECT_WIDTH = legacy_layout_constants.CARD_DIRECT_WIDTH
EVENT_WIDTH = legacy_layout_constants.EVENT_WIDTH
SOURCES = legacy_layout_constants.SOURCES
STATE_WIDTH = legacy_layout_constants.STATE_WIDTH

MAX_TENSOR_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class StateConfigV2:
    kind: str = "mlp"
    hidden_sizes: tuple[int, ...] = (128,)
    output_size: int = 64
    activation: str = "relu"


@dataclass(frozen=True)
class CardConfigV2:
    definition_embedding: int = 32
    ability_embedding: int = 16
    output_size: int = 64
    activation: str = "relu"


@dataclass(frozen=True)
class HandConfigV2:
    kind: str = "attention"
    hidden_sizes: tuple[int, ...] = (128,)
    latent_size: int = 128
    pooling: str = "mean"
    attention_layers: int = 2
    attention_heads: int = 4
    attention_feedforward: int = 128
    activation: str = "relu"


@dataclass(frozen=True)
class EventConfigV2:
    aggregate_recipe: str = ACTIVE_AGGREGATE
    hidden_size: int = 32
    num_layers: int = 1
    aggregate_window: int = 128
    max_interval: int = 4096
    max_episode: int = 262144


@dataclass(frozen=True)
class EventConfigV3(EventConfigV2):
    aggregate_recipe: str = CARD_INSTANCE_AGGREGATE


@dataclass(frozen=True)
class TemporalConfigV2:
    kind: str = "mlp"
    hidden_size: int = 128
    num_layers: int = 1
    num_heads: int = 4
    feedforward_size: int = 256
    max_decisions: int = 1024
    sequence_length: int = 32
    activation: str = "relu"


@dataclass(frozen=True)
class HeadConfigV2:
    actor_hidden_sizes: tuple[int, ...] = (128,)
    critic_hidden_sizes: tuple[int, ...] = (128,)
    activation: str = "relu"


@dataclass(frozen=True)
class ResourceConfigV2:
    max_parameters: int = 16_000_000
    max_owned_bytes: int = 12 * 1024**3
    max_tensor_bytes: int = MAX_TENSOR_BYTES
    cuda_context_bytes: int = 512 * 1024**2
    workspace_bytes: int = 256 * 1024**2


@dataclass(frozen=True)
class ModelConfigV2:
    model_id: str = "mlp"
    version: str = "2"
    input_recipe: str = "participant-structured-v1"
    max_positions: int = 18
    state: StateConfigV2 = field(default_factory=StateConfigV2)
    card: CardConfigV2 = field(default_factory=CardConfigV2)
    hand: HandConfigV2 = field(default_factory=HandConfigV2)
    event: EventConfigV2 = field(default_factory=EventConfigV2)
    temporal: TemporalConfigV2 = field(default_factory=TemporalConfigV2)
    heads: HeadConfigV2 = field(default_factory=HeadConfigV2)
    resources: ResourceConfigV2 = field(default_factory=ResourceConfigV2)
    initialization: str = "orthogonal-structured-v1"
    dropout: int = 0
    precision: str = "ieee-fp32"


@dataclass(frozen=True)
class ModelConfigV3(ModelConfigV2):
    version: str = "3"
    input_recipe: str = "participant-structured-v2"
    event: EventConfigV2 = field(default_factory=EventConfigV3)


@dataclass(frozen=True)
class TemporalConfigV4:
    """Inference context only; backward extent belongs to the HPO recipe."""
    kind: str = "mlp"
    hidden_size: int = 128
    num_layers: int = 1
    num_heads: int = 4
    feedforward_size: int = 256
    max_decisions: int = 1024
    activation: str = "relu"


@dataclass(frozen=True)
class ModelConfigV4:
    model_id: str = "mlp"
    version: str = "4"
    input_recipe: str = "participant-structured-v2"
    max_positions: int = 18
    state: StateConfigV2 = field(default_factory=StateConfigV2)
    card: CardConfigV2 = field(default_factory=CardConfigV2)
    hand: HandConfigV2 = field(default_factory=HandConfigV2)
    event: EventConfigV3 = field(default_factory=EventConfigV3)
    temporal: TemporalConfigV4 = field(default_factory=TemporalConfigV4)
    heads: HeadConfigV2 = field(default_factory=HeadConfigV2)
    resources: ResourceConfigV2 = field(default_factory=ResourceConfigV2)
    initialization: str = "orthogonal-activation-aware-v1"
    dropout: int = 0
    precision: str = "ieee-fp32"


@dataclass(frozen=True)
class HistoricalEventConfigV2:
    """Exact original serialization; never an accepted active configuration."""
    hidden_size: int = 32
    num_layers: int = 1
    aggregate_window: int = 128
    max_interval: int = 4096
    max_episode: int = 262144


@dataclass(frozen=True)
class HistoricalModelConfigV2:
    model_id: str = "mlp"
    version: str = "2"
    input_recipe: str = "participant-structured-v1"
    max_positions: int = 18
    state: StateConfigV2 = field(default_factory=StateConfigV2)
    card: CardConfigV2 = field(default_factory=CardConfigV2)
    hand: HandConfigV2 = field(default_factory=HandConfigV2)
    event: HistoricalEventConfigV2 = field(default_factory=HistoricalEventConfigV2)
    temporal: TemporalConfigV2 = field(default_factory=TemporalConfigV2)
    heads: HeadConfigV2 = field(default_factory=HeadConfigV2)
    resources: ResourceConfigV2 = field(default_factory=ResourceConfigV2)
    initialization: str = "orthogonal-structured-v1"
    dropout: int = 0
    precision: str = "ieee-fp32"


type RecordedModelConfigV2 = ModelConfigV2 | HistoricalModelConfigV2
type StructuredModel = RecordedModelConfigV2 | ModelConfigV3 | ModelConfigV4


def aggregate_recipe(model: StructuredModel) -> str:
    if type(model) is HistoricalModelConfigV2:
        return HISTORICAL_AGGREGATE
    if type(model.event) is not (EventConfigV3 if type(model) in (ModelConfigV3, ModelConfigV4) else EventConfigV2):
        raise ContractError('closed active event descriptor required')
    return model.event.aggregate_recipe


def recorded_token_signature(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> str:
    _check_model_layout(model, layout)
    return signature(model.event.aggregate_window, aggregate_recipe=aggregate_recipe(model), layout=layout)


@dataclass(frozen=True)
class InferenceBindingV2:
    device: str = "cpu"
    candidate_chunk: int = 256
    max_candidates: int = 32768
    managed_budget_bytes: int = 12 * 1024**3

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda:0"):
            raise ContractError("unsupported v2 inference device")
        integer(self.candidate_chunk, 1, 65536)
        integer(self.max_candidates, 1, 65536)
        integer(self.managed_budget_bytes, 1, 12 * 1024**3)


def _widths(value: Any) -> None:
    if not isinstance(value, tuple) or not 1 <= len(value) <= 4:
        raise ContractError("one to four hidden widths required")
    for width in value:
        integer(width, 1, 1024)


def _check_model_layout(model: StructuredModel, layout: StructuredLayout) -> None:
    require_layout(layout)
    if layout is LEGACY_LAYOUT:
        valid = type(model) in (ModelConfigV2, HistoricalModelConfigV2)
    else:
        valid = (type(model) is ModelConfigV3 and is_card_instance_layout(layout)
                 or type(model) is ModelConfigV4 and layout is CORRECTED_CARD_INSTANCE_LAYOUT)
    if not valid or model.input_recipe != layout.input_recipe:
        raise ContractError("closed structured model/exact layout mismatch")


def validate_config(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> None:
    _validate_descriptor(model, historical=False, layout=layout)


def _validate_descriptor(model: StructuredModel, *, historical: bool, layout: StructuredLayout = LEGACY_LAYOUT) -> None:
    if type(model) is ModelConfigV4:
        from gf_models.hpo_config import validate_model_v4
        validate_model_v4(model, layout=layout)
        return
    _check_model_layout(model, layout)
    modern = layout is not LEGACY_LAYOUT
    if (type(model) is not (ModelConfigV3 if modern else HistoricalModelConfigV2 if historical else ModelConfigV2) or model.version != ("3" if modern else "2")
            or model.model_id not in ("mlp", "gru", "transformer")
            or model.input_recipe != layout.input_recipe
            or model.initialization != "orthogonal-structured-v1"
            or type(model.dropout) is not int or model.dropout != 0 or model.precision != "ieee-fp32"):
        raise ContractError("unsupported v2 model recipe")
    for name, expected in (("state", StateConfigV2), ("card", CardConfigV2), ("hand", HandConfigV2),
                           ("event", EventConfigV3 if modern else HistoricalEventConfigV2 if historical else EventConfigV2), ("temporal", TemporalConfigV2),
                           ("heads", HeadConfigV2), ("resources", ResourceConfigV2)):
        if type(getattr(model, name)) is not expected:
            raise ContractError("closed typed v2 component required")
    if not historical and aggregate_recipe(model) != layout.aggregate_recipe:
        raise ContractError('only corrected public-exact-v2 history is active')
    integer(model.max_positions, 1, 128)
    for component in (model.state, model.card, model.hand, model.temporal, model.heads):
        if component.activation != "relu":
            raise ContractError("only ReLU is supported")
    if model.state.kind not in ("identity", "mlp") or model.hand.kind not in ("direct", "deepsets", "attention"):
        raise ContractError("unsupported state/hand encoder")
    _widths(model.state.hidden_sizes)
    _widths(model.hand.hidden_sizes)
    _widths(model.heads.actor_hidden_sizes)
    _widths(model.heads.critic_hidden_sizes)
    for value in (model.state.output_size, model.card.definition_embedding, model.card.ability_embedding,
                  model.card.output_size, model.hand.latent_size, model.hand.attention_feedforward,
                  model.event.hidden_size, model.temporal.hidden_size, model.temporal.feedforward_size):
        integer(value, 1, 1024)
    integer(model.hand.attention_layers, 1, 4)
    integer(model.hand.attention_heads, 1, 16)
    integer(model.event.num_layers, 1, 4)
    integer(model.event.aggregate_window, 1, min(65536, model.event.max_episode))
    integer(model.event.max_interval, 1, 65536)
    integer(model.event.max_episode, model.event.max_interval, 2**31 - 1)
    integer(model.temporal.num_layers, 1, 4)
    integer(model.temporal.num_heads, 1, 16)
    integer(model.temporal.max_decisions, 1, 1024)
    integer(model.temporal.sequence_length, 1, model.temporal.max_decisions)
    if model.hand.pooling not in ("sum", "mean", "max"):
        raise ContractError("unsupported hand pooling")
    if model.hand.kind == "attention" and model.card.output_size % model.hand.attention_heads:
        raise ContractError("hand attention width must divide heads")
    if model.temporal.kind == "transformer" and model.temporal.hidden_size % model.temporal.num_heads:
        raise ContractError("temporal attention width must divide heads")
    if model.state.kind == "identity" and model.state != StateConfigV2(kind="identity"):
        raise ContractError("noncanonical inactive identity settings")
    if model.hand.kind != "attention" and (model.hand.attention_layers, model.hand.attention_heads,
            model.hand.attention_feedforward) != (2, 4, 128):
        raise ContractError("noncanonical inactive hand attention settings")
    if model.hand.kind == "direct" and model.hand != HandConfigV2(kind="direct"):
        raise ContractError("noncanonical inactive direct hand settings")
    if model.temporal.kind != model.model_id:
        raise ContractError("temporal/model kind mismatch")
    if model.temporal.kind != "transformer" and (model.temporal.num_heads, model.temporal.feedforward_size) != (4, 256):
        raise ContractError("noncanonical inactive temporal attention settings")
    if model.temporal.kind == "mlp" and model.temporal.num_layers != 1:
        raise ContractError("noncanonical inactive MLP layer setting")
    for value, maximum in ((model.resources.max_parameters, 100_000_000),
                           (model.resources.max_owned_bytes, 12 * 1024**3),
                           (model.resources.max_tensor_bytes, MAX_TENSOR_BYTES),
                           (model.resources.cuda_context_bytes, 4 * 1024**3),
                           (model.resources.workspace_bytes, 4 * 1024**3)):
        integer(value, 1, maximum)
    shapes = parameter_shapes(model, layout=layout)
    if sum(prod(shape) for shape in shapes.values()) > model.resources.max_parameters:
        raise ContractError("structured parameter limit exceeded")
    if any(prod(shape) * 4 > model.resources.max_tensor_bytes for shape in shapes.values()):
        raise ContractError("structured parameter tensor exceeds 64MiB/configured limit")


def model_config_v2(value: Json) -> ModelConfigV2:
    result = _construct_descriptor(value, historical=False)
    assert type(result) is ModelConfigV2
    return result


def _construct_descriptor(value: Json, *, historical: bool, layout: StructuredLayout = LEGACY_LAYOUT) -> StructuredModel:
    require_layout(layout)
    modern = layout is not LEGACY_LAYOUT
    closed(value, set(asdict(ModelConfigV2())))
    converted = dict(value)
    for name, constructor in (("state", StateConfigV2), ("card", CardConfigV2), ("hand", HandConfigV2),
                              ("event", EventConfigV3 if modern else HistoricalEventConfigV2 if historical else EventConfigV2), ("temporal", TemporalConfigV2),
                              ("heads", HeadConfigV2), ("resources", ResourceConfigV2)):
        raw = closed(value[name], set(asdict(constructor())))
        fields = dict(raw)
        for key in ("hidden_sizes", "actor_hidden_sizes", "critic_hidden_sizes"):
            if key in fields:
                if not isinstance(fields[key], (list, tuple)):
                    raise ContractError("hidden width array required")
                fields[key] = tuple(fields[key])
        converted[name] = constructor(**fields)
    model = (ModelConfigV3 if modern else HistoricalModelConfigV2 if historical else ModelConfigV2)(**converted)
    _validate_descriptor(model, historical=historical, layout=layout)
    return model


def model_config_v3(value: Json, *, layout: StructuredLayout) -> ModelConfigV3:
    if not is_card_instance_layout(layout):
        raise ContractError("model3 requires explicit card-instance layout")
    model = _construct_descriptor(value, historical=False, layout=layout)
    assert type(model) is ModelConfigV3
    return model


def recorded_model_config_v2(value: Json, recorded_signature: str) -> RecordedModelConfigV2:
    """Exactly two closed descriptors, selected only with a matching signature."""
    raw_event = value.get('event')
    if not isinstance(raw_event, dict):
        raise ContractError('recorded event descriptor required')
    historical = 'aggregate_recipe' not in raw_event
    model = _construct_descriptor(value, historical=historical)
    if recorded_signature != recorded_token_signature(model):
        raise ContractError('recorded closed model/signature descriptor mismatch')
    assert isinstance(model, (ModelConfigV2, HistoricalModelConfigV2))
    return model


def architecture_identity(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> str:
    validate_config(model, layout=layout)
    return recorded_architecture_identity(model, layout=layout)


def recorded_architecture_identity(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> str:
    _validate_descriptor(model, historical=type(model) is HistoricalModelConfigV2, layout=layout)
    # Resource admission and runtime bindings are not network semantics.
    fields = asdict(model)
    del fields["resources"]
    return digest({"model": fields, "token_signature": recorded_token_signature(model, layout=layout)})


def dimensions(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> dict[str, int]:
    _check_model_layout(model, layout)
    state = layout.state_width if model.state.kind == "identity" else model.state.output_size
    hand = 18 * layout.card_direct_width if model.hand.kind == "direct" else model.hand.latent_size
    card_input = (model.card.definition_embedding + model.card.ability_embedding
                  + sum(len(values) + 2 for values in layout.card_categories[2:])
                  + 2 * len(layout.card_numeric_fields))
    card_actor = model.card.output_size + len(layout.card_categories[layout.membership_index]) + 2 + 2
    frame = state + hand + model.event.hidden_size + (history_width(aggregate_recipe(model)) if layout is LEGACY_LAYOUT else layout.history_width)
    return {"state": state, "hand": hand, "card_input": card_input, "card_actor": card_actor,
            "frame": frame, "actor": model.temporal.hidden_size + layout.candidate_width + model.max_positions * card_actor}


def parameter_shapes(model: StructuredModel, *, layout: StructuredLayout = LEGACY_LAYOUT) -> dict[str, tuple[int, ...]]:
    """Exact named parameter shapes; keep differential tests against state_dict."""
    shapes: dict[str, tuple[int, ...]] = {}
    d = dimensions(model, layout=layout)

    def linear(name: str, source: int, target: int) -> None:
        shapes[name + ".weight"] = (target, source)
        shapes[name + ".bias"] = (target,)

    def mlp(name: str, widths: tuple[int, ...]) -> None:
        for index, (source, target) in enumerate(pairwise(widths)):
            linear(f"{name}.{index * 2}", source, target)

    def gru(name: str, source: int, width: int, layers: int) -> None:
        for index in range(layers):
            shapes[f"{name}.weight_ih_l{index}"] = (3 * width, source if index == 0 else width)
            shapes[f"{name}.weight_hh_l{index}"] = (3 * width, width)
            shapes[f"{name}.bias_ih_l{index}"] = (3 * width,)
            shapes[f"{name}.bias_hh_l{index}"] = (3 * width,)

    def attention(name: str, width: int, feedforward: int) -> None:
        for part in ("query", "key", "value", "output"):
            linear(f"{name}.{part}", width, width)
        linear(f"{name}.feedforward1", width, feedforward)
        linear(f"{name}.feedforward2", feedforward, width)
        for part in ("norm1", "norm2"):
            shapes[f"{name}.{part}.weight"] = (width,)
            shapes[f"{name}.{part}.bias"] = (width,)

    if model.state.kind == "mlp":
        mlp("state_encoder", (layout.state_width, *model.state.hidden_sizes, model.state.output_size))
    shapes["definition_embedding.weight"] = (len(layout.card_categories[0]) + 2, model.card.definition_embedding)
    shapes["ability_embedding.weight"] = (len(layout.card_categories[1]) + 2, model.card.ability_embedding)
    linear("card_encoder", d["card_input"], model.card.output_size)
    if model.hand.kind != "direct":
        mlp("hand_encoder", (model.card.output_size, *model.hand.hidden_sizes, model.hand.latent_size))
    if model.hand.kind == "attention":
        for index in range(model.hand.attention_layers):
            attention(f"hand_attention.{index}", model.card.output_size, model.hand.attention_feedforward)
    gru("event_encoder", layout.event_width, model.event.hidden_size, model.event.num_layers)
    linear("frame_projection", d["frame"], model.temporal.hidden_size)
    if type(model) is ModelConfigV4 and model.model_id == "mlp":
        for index in range(model.temporal.num_layers - 1):
            linear(f"temporal_dense.{index * 2}", model.temporal.hidden_size, model.temporal.hidden_size)
    if model.model_id == "gru":
        gru("temporal", model.temporal.hidden_size, model.temporal.hidden_size, model.temporal.num_layers)
    elif model.model_id == "transformer":
        for index in range(model.temporal.num_layers):
            attention(f"temporal.{index}", model.temporal.hidden_size, model.temporal.feedforward_size)
    mlp("actor", (d["actor"], *model.heads.actor_hidden_sizes, 1))
    mlp("critic", (model.temporal.hidden_size, *model.heads.critic_hidden_sizes, 1))
    return shapes


def resource_estimate(model: StructuredModel, binding: InferenceBindingV2, *,
                      layout: StructuredLayout = LEGACY_LAYOUT,
                      decisions: int | None = None, interval_events: int | None = None,
                      candidates: int | None = None,
                      suffix_decisions: int | None = None) -> Json:
    """Conservative managed bytes, separate from measured RSS/device reservations.

    Covers one endpoint with full current-weight prefix and a differentiable
    suffix, weights/gradients/Adam/steps and CPU references. Chunking does not
    remove the global candidate-normalization allocation.
    """
    validate_config(model, layout=layout)
    decisions = model.temporal.max_decisions if decisions is None else decisions
    interval_events = model.event.max_interval if interval_events is None else interval_events
    candidates = binding.max_candidates if candidates is None else candidates
    integer(decisions, 1, model.temporal.max_decisions)
    integer(interval_events, 0, model.event.max_interval)
    integer(candidates, 1, binding.max_candidates)
    suffix_decisions = ((1 if isinstance(model.temporal, TemporalConfigV4) else min(decisions, model.temporal.sequence_length))
                        if suffix_decisions is None else suffix_decisions)
    integer(suffix_decisions, 1, decisions)
    d = dimensions(model, layout=layout)
    parameter_count = sum(prod(shape) for shape in parameter_shapes(model, layout=layout).values())
    chunk = min(binding.candidate_chunk, candidates)
    tensors = {
        "raw_state": decisions * layout.state_width * 4,
        "raw_cards": decisions * 18 * layout.card_direct_width * 4,
        "interval_tokens": interval_events * layout.event_width * 4,
        "interval_gru": interval_events * model.event.hidden_size * model.event.num_layers * 4 * 12,
        "hand_attention": suffix_decisions * model.hand.attention_layers * model.hand.attention_heads * 18 * 18 * 4 * 8,
        "frame_history": decisions * d["frame"] * 4,
        "temporal_attention": model.temporal.num_layers * model.temporal.num_heads * decisions * decisions * 4,
        "temporal_cache": 2 * model.temporal.num_layers * decisions * model.temporal.hidden_size * 4,
        "candidate_references": candidates * model.max_positions * 8,
        "candidate_columns": candidates * (layout.candidate_width + 2) * 4,
        "chunk_dense_reference": chunk * d["actor"] * 4,
        "chunk_position_gather": chunk * model.max_positions * model.heads.actor_hidden_sizes[0] * 4,
        "chunk_checkpoint_replay": chunk * sum(model.heads.actor_hidden_sizes) * 4 * 8,
        "position_card_projection": model.max_positions * 19 * model.heads.actor_hidden_sizes[0] * 4,
        "position_card_projection_unpadded": model.max_positions * 18 * model.heads.actor_hidden_sizes[0] * 4,
        "position_card_projection_padding": model.max_positions * model.heads.actor_hidden_sizes[0] * 4,
        "global_logits_probability_gradient": candidates * 4 * 5,
    }
    if any(size > model.resources.max_tensor_bytes for size in tensors.values()):
        raise ContractError("runtime tensor exceeds 64MiB/configured limit")
    suffix_intervals = suffix_decisions
    parts = {"parameter_optimizer_cpu_reference_bytes": parameter_count * 4 * 6
             + len(parameter_shapes(model, layout=layout)) * 8,
             "context_workspace_bytes": model.resources.workspace_bytes
             + (model.resources.cuda_context_bytes if binding.device == "cuda:0" else 0),
             # Prefix is detached, suffix/replay intermediate copies are bounded
             # conservatively by four complete raw/activation envelopes.
             "activation_and_copy_bytes": 4 * (sum(tensors.values())
                 + (suffix_intervals - 1) * (tensors["interval_tokens"] + tensors["interval_gru"]))}
    total = sum(parts.values())
    if total > min(model.resources.max_owned_bytes, binding.managed_budget_bytes):
        raise ContractError("structured managed memory reservation exceeded")
    return {**parts, "total_bytes": total, "parameters": parameter_count, "tensor_bytes": tensors}
