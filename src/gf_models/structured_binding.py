"""Stdlib-only application model2 settings and exact execution identities."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.action_contract import ActionContract, parse_action_contract, validate_record_purpose
from gf_common.card_instance import CardInstanceContract, parse_card_instance_contract
from gf_common.card_instance_rules import CardInstanceRules, rule_signature
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest
from gf_common.structured_layout import StructuredLayout, layout_for_contract
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    RecordedModelConfigV2,
    TemporalConfigV2,
    model_config_v2,
    model_config_v3,
    recorded_model_config_v2,
    resource_estimate,
)


@dataclass(frozen=True)
class ConfiguredNeuralModelV2(ModelConfigV2):
    checkpoint: str | None = None
    inference: InferenceBindingV2 = field(default_factory=InferenceBindingV2)
    cuda_visible_devices: str | None = None

    def architecture(self) -> ModelConfigV2:
        fields = asdict(self)
        del fields["checkpoint"]
        del fields["inference"]
        del fields["cuda_visible_devices"]
        return model_config_v2(fields)


@dataclass(frozen=True)
class ConfiguredNeuralModelV3(ModelConfigV3):
    checkpoint: str | None = None
    inference: InferenceBindingV2 = field(default_factory=InferenceBindingV2)
    cuda_visible_devices: str | None = None

    def architecture(self, *, layout: StructuredLayout) -> ModelConfigV3:
        fields = asdict(self)
        for key in ("checkpoint", "inference", "cuda_visible_devices"):
            del fields[key]
        return model_config_v3(fields, layout=layout)


def model_defaults_v3(model_id: str = "mlp") -> ConfiguredNeuralModelV3:
    if model_id not in ("mlp", "gru", "transformer"):
        raise ContractError("model3 currently supports neural models only")
    return ConfiguredNeuralModelV3(model_id=model_id,
        temporal=TemporalConfigV2(kind=model_id, num_layers=2 if model_id == "transformer" else 1))


def configured_model_v3(value: Json, *, layout: StructuredLayout) -> ConfiguredNeuralModelV3:
    closed(value, set(asdict(ConfiguredNeuralModelV3())))
    pure = model_config_v3({key: item for key, item in value.items() if key not in ("checkpoint", "inference", "cuda_visible_devices")}, layout=layout)
    inference = InferenceBindingV2(**closed(value["inference"], set(asdict(InferenceBindingV2()))))
    checkpoint = value["checkpoint"]
    if checkpoint is not None and (type(checkpoint) is not str or not checkpoint):
        raise ContractError("invalid model3 checkpoint path")
    validate_cuda_visibility(value["cuda_visible_devices"])
    resource_estimate(pure, inference, layout=layout)
    return ConfiguredNeuralModelV3(**{name: getattr(pure, name) for name in pure.__dataclass_fields__},
        checkpoint=checkpoint, inference=inference, cuda_visible_devices=value["cuda_visible_devices"])


def model_defaults_v2(model_id: str = "mlp") -> ConfiguredNeuralModelV2:
    if model_id not in ("mlp", "gru", "transformer"):
        raise ContractError("model2 currently supports neural models only; LightGBM2 awaits implementation")
    return ConfiguredNeuralModelV2(model_id=model_id,
        temporal=TemporalConfigV2(kind=model_id, num_layers=2 if model_id == "transformer" else 1))


def configured_model_v2(value: Json) -> ConfiguredNeuralModelV2:
    closed(value, set(asdict(ConfiguredNeuralModelV2())))
    pure = model_config_v2({key: item for key, item in value.items() if key not in ("checkpoint", "inference", "cuda_visible_devices")})
    raw_binding = closed(value["inference"], set(asdict(InferenceBindingV2())))
    inference = InferenceBindingV2(**raw_binding)
    checkpoint = value["checkpoint"]
    if checkpoint is not None and (type(checkpoint) is not str or not checkpoint):
        raise ContractError("invalid model2 checkpoint path")
    validate_cuda_visibility(value["cuda_visible_devices"])
    resource_estimate(pure, inference)
    return ConfiguredNeuralModelV2(**{name: getattr(pure, name) for name in pure.__dataclass_fields__},
                                    checkpoint=checkpoint, inference=inference, cuda_visible_devices=value["cuda_visible_devices"])


def validate_cuda_visibility(value: Any) -> str | None:
    if value is None or value == "" and type(value) is str:
        return value
    if type(value) is not str or len(value) > 1024:
        raise ContractError("invalid safe CUDA visibility")
    items = value.split(",")
    uuid = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    if (len(items) > 32 or len(set(items)) != len(items)
            or any(re.fullmatch(r"(?:0|[1-9][0-9]*|" + uuid + r")", item) is None for item in items)):
        raise ContractError("invalid safe CUDA visibility")
    return value


def child_cuda_visibility(device: str, configured: str | None) -> str | None:
    validate_cuda_visibility(configured)
    if device not in ("cpu", "cuda:0"):
        raise ContractError("unsupported execution device")
    return "" if device == "cpu" else configured


def execution_binding(model: ModelConfigV2, inference: InferenceBindingV2, *,
                      cuda_visible_devices: str | None = None) -> Json:
    if type(model) is not ModelConfigV2 or type(inference) is not InferenceBindingV2:
        raise ContractError("pure model/execution binding required")
    # Identity computation is not active runtime admission. Historical source
    # qualification must not depend on a later estimator/runtime installation.
    visibility = validate_cuda_visibility(cuda_visible_devices)
    return {"inference": asdict(inference), "resources": asdict(model.resources), "cuda_visible_devices": visibility}


def execution_digest(model: ModelConfigV2, inference: InferenceBindingV2, *,
                     cuda_visible_devices: str | None = None) -> str:
    return digest(execution_binding(model, inference, cuda_visible_devices=cuda_visible_devices))


def policy_label(model: ModelConfigV2, inference: InferenceBindingV2, manifest_digest: str, *,
                 cuda_visible_devices: str | None = None) -> str:
    if type(manifest_digest) is not str or re.fullmatch(r"[0-9a-f]{64}", manifest_digest) is None:
        raise ContractError("full model2 inference digest required")
    return f"{model.model_id}-ppo-v2-{manifest_digest}-{execution_digest(model, inference, cuda_visible_devices=cuda_visible_devices)}"


def recorded_policy_label(model: RecordedModelConfigV2, inference: InferenceBindingV2, manifest_digest: str, *,
                           cuda_visible_devices: str | None) -> str:
    """Metadata-only reproduction, never permission to run a historical model."""
    if type(manifest_digest) is not str or re.fullmatch(r'[0-9a-f]{64}', manifest_digest) is None:
        raise ContractError('recorded full inference digest required')
    binding = {'inference': asdict(inference), 'resources': asdict(model.resources),
               'cuda_visible_devices': validate_cuda_visibility(cuda_visible_devices)}
    return f'{model.model_id}-ppo-v2-{manifest_digest}-{digest(binding)}'


def recorded_options_v2(value: Any, *, action_contract: ActionContract, purpose: str,
                         synthetic_capacity: bool, token_signature: str) -> Json:
    raw = closed(value, {'version', 'checkpoint', 'expected_manifest_digest', 'model', 'inference', 'cuda_visible_devices',
                        'action_contract', 'purpose', 'synthetic_capacity'}, {'allow_toy'})
    check_input_binding({key: raw[key] for key in ('action_contract', 'purpose', 'synthetic_capacity')},
        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if raw['version'] != '2' or type(raw['checkpoint']) is not str or not raw['checkpoint']:
        raise ContractError('recorded model2 checkpoint binding required')
    model = recorded_model_config_v2(raw['model'], token_signature)
    inference = InferenceBindingV2(**closed(raw['inference'], set(asdict(InferenceBindingV2()))))
    recorded_policy_label(model, inference, raw['expected_manifest_digest'], cuda_visible_devices=raw['cuda_visible_devices'])
    if 'allow_toy' in raw and type(raw['allow_toy']) is not bool:
        raise ContractError('recorded toy setting must be boolean')
    return clone(raw)  # no field insertion or historical serialization rewrite.


def input_binding(action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Json:
    """Closed compatibility/provenance fields, independent of tensor features."""
    validate_record_purpose(purpose, action_contract)
    if type(synthetic_capacity) is not bool or synthetic_capacity and purpose != "component":
        raise ContractError("explicit boolean synthetic capacity/component classification required")
    return {"action_contract": asdict(action_contract), "purpose": purpose, "synthetic_capacity": synthetic_capacity}


def check_input_binding(raw: Json, *, action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> None:
    expected = input_binding(action_contract, purpose, synthetic_capacity)
    closed(raw, set(expected))
    parse_action_contract(raw["action_contract"], expected=action_contract)
    if type(raw["synthetic_capacity"]) is not bool or canonical(raw) != canonical(expected):
        raise ContractError("input contract/purpose/synthetic classification differs from authoritative Config")


@dataclass(frozen=True, slots=True)
class _NetworkInputBinding:
    action_contract: ActionContract
    purpose: str
    synthetic_capacity: bool


def _bind_network_input(network: Any, *, action_contract: ActionContract, purpose: str,
                        synthetic_capacity: bool, fresh: bool = False) -> None:
    """Bind fresh component weights once; existing provenance cannot be relabeled.

    This immutable runtime value is not a parameter/buffer or a feature. Artifact
    loaders reconstruct it from validated metadata; it grants no Source authority.
    """
    input_binding(action_contract, purpose, synthetic_capacity)
    expected = _NetworkInputBinding(action_contract, purpose, synthetic_capacity)
    if hasattr(network, "_structured_input_binding"):
        actual = network._structured_input_binding
        if type(actual) is not _NetworkInputBinding or actual != expected:
            raise ContractError("network input contract/purpose/synthetic classification mismatch")
        return
    if not fresh or purpose != "component":
        raise ContractError("network requires an explicit fresh component input binding")
    network._structured_input_binding = expected


def safe_options_v2(value: Any, *, action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Json:
    raw = closed(value, {"version", "checkpoint", "expected_manifest_digest", "model", "inference", "cuda_visible_devices",
                         "action_contract", "purpose", "synthetic_capacity"}, {"allow_toy"})
    check_input_binding({key: raw[key] for key in ("action_contract", "purpose", "synthetic_capacity")},
                        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if raw["version"] != "2" or type(raw["checkpoint"]) is not str or not raw["checkpoint"]:
        raise ContractError("invalid safe model2 checkpoint binding")
    model = model_config_v2(raw["model"])
    inference = InferenceBindingV2(**closed(raw["inference"], set(asdict(InferenceBindingV2()))))
    policy_label(model, inference, raw["expected_manifest_digest"], cuda_visible_devices=raw["cuda_visible_devices"])
    if "allow_toy" in raw and type(raw["allow_toy"]) is not bool:
        raise ContractError("toy binding permission must be boolean")
    return clone({**raw, "model": asdict(model), "inference": asdict(inference)})


def input_binding_v3(action_contract: CardInstanceContract, rules: CardInstanceRules,
                     purpose: str, synthetic_capacity: bool) -> Json:
    signature = rule_signature(action_contract, rules)
    if purpose != "component" or type(purpose) is not str or type(synthetic_capacity) is not bool:
        raise ContractError("model3 permits explicit component inference binding only")
    return {"action_contract": asdict(action_contract), "rule_signature": signature,
            "purpose": purpose, "synthetic_capacity": synthetic_capacity}


def check_input_binding_v3(raw: Json, *, action_contract: CardInstanceContract, rules: CardInstanceRules,
                           purpose: str, synthetic_capacity: bool) -> None:
    expected = input_binding_v3(action_contract, rules, purpose, synthetic_capacity)
    closed(raw, set(expected))
    parse_card_instance_contract(raw["action_contract"])
    if type(raw["synthetic_capacity"]) is not bool or canonical(raw) != canonical(expected):
        raise ContractError("model3 input/rules/purpose binding mismatch")


def _action_hash(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("explicit full action-config hash required")
    return value


@dataclass(frozen=True, slots=True)
class _NetworkInputBindingV3:
    action_contract: CardInstanceContract
    rule_signature: str
    purpose: str
    synthetic_capacity: bool
    action_config_hash: str


def _bind_network_input_v3(network: Any, *, action_contract: CardInstanceContract,
                            rules: CardInstanceRules, purpose: str, synthetic_capacity: bool,
                            action_config_hash: str, fresh: bool = False) -> None:
    context = input_binding_v3(action_contract, rules, purpose, synthetic_capacity)
    _action_hash(action_config_hash)
    if (type(network.config) is not ModelConfigV3 or network.layout is not layout_for_contract(action_contract)
            or hasattr(network, "_structured_input_binding")):
        raise ContractError("network3 model/exact layout mismatch")
    expected = _NetworkInputBindingV3(action_contract, context["rule_signature"], purpose, synthetic_capacity, action_config_hash)
    if hasattr(network, "_structured_input_binding_v3"):
        if type(network._structured_input_binding_v3) is not _NetworkInputBindingV3 or network._structured_input_binding_v3 != expected:
            raise ContractError("immutable network3 input/action-config binding mismatch")
        return
    if not fresh:
        raise ContractError("fresh component network3 binding required")
    network._structured_input_binding_v3 = expected


def execution_binding_v3(model: ModelConfigV3, inference: InferenceBindingV2, *, layout: StructuredLayout,
                          cuda_visible_devices: str | None = None) -> Json:
    model_config_v3(asdict(model), layout=layout)
    if type(inference) is not InferenceBindingV2:
        raise ContractError("closed model3 inference binding required")
    return {"inference": asdict(inference), "resources": asdict(model.resources),
            "cuda_visible_devices": validate_cuda_visibility(cuda_visible_devices)}


def policy_label_v3(model: ModelConfigV3, inference: InferenceBindingV2, manifest_digest: str, *,
                    layout: StructuredLayout, action_contract: CardInstanceContract, rules: CardInstanceRules,
                    purpose: str, synthetic_capacity: bool, action_config_hash: str,
                    cuda_visible_devices: str | None = None) -> str:
    _action_hash(manifest_digest)
    context = input_binding_v3(action_contract, rules, purpose, synthetic_capacity)
    if layout is not layout_for_contract(action_contract):
        raise ContractError("policy3 profile/layout mismatch")
    identity = {"execution": execution_binding_v3(model, inference, layout=layout, cuda_visible_devices=cuda_visible_devices),
                **context, "action_config_hash": _action_hash(action_config_hash)}
    return f"{model.model_id}-ppo-v3-{manifest_digest}-{digest(identity)}"


def safe_options_v3(value: Any, *, action_contract: CardInstanceContract, rules: CardInstanceRules,
                     purpose: str, synthetic_capacity: bool, action_config_hash: str) -> Json:
    raw = closed(value, {"version", "checkpoint", "expected_manifest_digest", "model", "inference", "cuda_visible_devices",
        "action_contract", "rule_signature", "purpose", "synthetic_capacity", "action_config_hash"})
    check_input_binding_v3({key: raw[key] for key in ("action_contract", "rule_signature", "purpose", "synthetic_capacity")},
        action_contract=action_contract, rules=rules, purpose=purpose, synthetic_capacity=synthetic_capacity)
    if raw["version"] != "3" or type(raw["checkpoint"]) is not str or not raw["checkpoint"] or raw["action_config_hash"] != _action_hash(action_config_hash):
        raise ContractError("invalid safe model3/action-config binding")
    layout = layout_for_contract(action_contract)
    model = model_config_v3(raw["model"], layout=layout)
    inference = InferenceBindingV2(**closed(raw["inference"], set(asdict(InferenceBindingV2()))))
    policy_label_v3(model, inference, raw["expected_manifest_digest"], layout=layout, action_contract=action_contract,
        rules=rules, purpose=purpose, synthetic_capacity=synthetic_capacity, action_config_hash=action_config_hash,
        cuda_visible_devices=raw["cuda_visible_devices"])
    return clone(raw)
