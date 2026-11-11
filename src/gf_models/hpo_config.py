"""Pure model4 construction, canonicalization and activation semantics."""
from __future__ import annotations

from dataclasses import asdict, replace
from math import prod, sqrt
from typing import Any

from gf_common.domain import ContractError, Json, closed, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, StructuredLayout
from gf_models.structured_config import (
    CardConfigV2,
    EventConfigV3,
    HandConfigV2,
    HeadConfigV2,
    ModelConfigV3,
    ModelConfigV4,
    ResourceConfigV2,
    StateConfigV2,
    TemporalConfigV2,
    TemporalConfigV4,
    _widths,
    parameter_shapes,
    validate_config,
)


def activation_gain(activation: str) -> float:
    if type(activation) is not str or activation not in ("relu", "tanh"):
        raise ContractError("model4 activation must be relu or tanh")
    return sqrt(2) if activation == "relu" else 5 / 3


def model_defaults_v4(model_id: str = "mlp") -> ModelConfigV4:
    if type(model_id) is not str or model_id not in ("mlp", "gru", "transformer"):
        raise ContractError("model4 requires a neural model kind")
    return ModelConfigV4(model_id=model_id, temporal=TemporalConfigV4(
        kind=model_id, num_layers=2 if model_id == "transformer" else 1))


def normalize_model_v4(model: ModelConfigV4) -> ModelConfigV4:
    """Discard inactive settings only after their closed types/ranges were checked."""
    state = StateConfigV2(kind="identity") if model.state.kind == "identity" else model.state
    hand = model.hand
    if hand.kind == "direct":
        hand = HandConfigV2(kind="direct")
    elif hand.kind == "deepsets":
        hand = replace(hand, attention_layers=2, attention_heads=4, attention_feedforward=128)
    temporal = model.temporal
    if temporal.kind != "transformer":
        temporal = replace(temporal, num_heads=4, feedforward_size=256)
    return replace(model, state=state, hand=hand, temporal=temporal)


def _check_fields(model: ModelConfigV4, layout: StructuredLayout) -> None:
    if (type(model) is not ModelConfigV4 or layout is not CORRECTED_CARD_INSTANCE_LAYOUT
            or model.version != "4" or model.input_recipe != layout.input_recipe
            or model.initialization != "orthogonal-activation-aware-v1"
            or type(model.dropout) is not int or model.dropout != 0 or model.precision != "ieee-fp32"):
        raise ContractError("explicit corrected model4 recipe/layout required")
    for name, cls in (("state", StateConfigV2), ("card", CardConfigV2), ("hand", HandConfigV2),
                      ("event", EventConfigV3), ("temporal", TemporalConfigV4),
                      ("heads", HeadConfigV2), ("resources", ResourceConfigV2)):
        if type(getattr(model, name)) is not cls:
            raise ContractError("closed typed model4 component required")
    for component in (model.state, model.card, model.hand, model.temporal, model.heads):
        activation_gain(component.activation)
    for widths in (model.state.hidden_sizes, model.hand.hidden_sizes,
                   model.heads.actor_hidden_sizes, model.heads.critic_hidden_sizes):
        _widths(widths)
    for width in (model.state.output_size, model.card.definition_embedding, model.card.ability_embedding,
                  model.card.output_size, model.hand.latent_size, model.hand.attention_feedforward,
                  model.temporal.hidden_size, model.temporal.feedforward_size):
        integer(width, 1, 1024)
    integer(model.hand.attention_layers, 1, 4)
    integer(model.hand.attention_heads, 1, 16)
    integer(model.temporal.num_heads, 1, 16)
    integer(model.temporal.num_layers, 1, 4)
    integer(model.temporal.max_decisions, 1024, 1024)
    integer(model.event.max_episode, 1, 2**31 - 1)
    if model.hand.pooling not in ("sum", "mean", "max"):
        raise ContractError("unsupported hand pooling")


def _legacy_validation_proxy(model: ModelConfigV4) -> ModelConfigV3:
    fields: dict[str, Any] = {name: getattr(model, name) for name in ModelConfigV3.__dataclass_fields__}
    fields.update(version="3", initialization="orthogonal-structured-v1")
    for name in ("state", "card", "hand", "heads"):
        fields[name] = replace(fields[name], activation="relu")
    temporal = asdict(model.temporal)
    temporal["activation"] = "relu"
    if model.temporal.kind == "mlp":
        temporal["num_layers"] = 1
    fields["temporal"] = TemporalConfigV2(**temporal)
    return ModelConfigV3(**fields)


def validate_model_v4(model: ModelConfigV4, *, layout: StructuredLayout) -> None:
    _check_fields(model, layout)
    if normalize_model_v4(model) != model:
        raise ContractError("model4 requires normalized inactive fields")
    # Reuse all unchanged component/layout guards without widening Model3.
    validate_config(_legacy_validation_proxy(model), layout=layout)
    shapes = parameter_shapes(model, layout=layout)
    if sum(prod(shape) for shape in shapes.values()) > model.resources.max_parameters:
        raise ContractError("model4 parameter limit exceeded")
    if any(prod(shape) * 4 > model.resources.max_tensor_bytes for shape in shapes.values()):
        raise ContractError("model4 parameter tensor exceeds configured limit")


def model_config_v4(value: Json, *, layout: StructuredLayout) -> ModelConfigV4:
    closed(value, set(asdict(ModelConfigV4())))
    converted = dict(value)
    for name, cls in (("state", StateConfigV2), ("card", CardConfigV2), ("hand", HandConfigV2),
                      ("event", EventConfigV3), ("temporal", TemporalConfigV4),
                      ("heads", HeadConfigV2), ("resources", ResourceConfigV2)):
        fields = dict(closed(value[name], set(asdict(cls()))))
        for key in ("hidden_sizes", "actor_hidden_sizes", "critic_hidden_sizes"):
            if key in fields:
                if type(fields[key]) not in (list, tuple):
                    raise ContractError("model4 hidden widths must be an array")
                fields[key] = tuple(fields[key])
        converted[name] = cls(**fields)
    model = ModelConfigV4(**converted)
    _check_fields(model, layout)
    normalized = normalize_model_v4(model)
    validate_model_v4(normalized, layout=layout)
    return normalized
