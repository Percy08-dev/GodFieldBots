"""Structured inference/resume3: typed shards, exact bindings, named Adam state.

These files establish numerical compatibility only. They never authorize a
formal generation, source, training parent or CLI invocation on their own.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gf_common.action_contract import ActionContract
from gf_common.card_instance import CardInstanceContract
from gf_common.card_instance_rules import CardInstanceRules
from gf_common.domain import (
    CATALOG_META,
    END_PROJECTION_VERSION,
    POLICY_INPUT_VERSION,
    REVIEWED_SPEC_HASHES,
    SIMULATOR_REVISION,
    ContractError,
    Json,
    canonical,
    closed,
    integer,
)
from gf_common.structured import GENERATOR_V2, INPUT_RECIPE
from gf_common.structured_layout import LEGACY_LAYOUT, StructuredLayout, layout_for_contract
from gf_common.structured_tokens import signature
from gf_models.arrays import MAX_ARRAY_BYTES, sha256
from gf_models.sharded_arrays import (
    create_output_directory,
    inspect_bundle,
    preflight_outputs,
    read_bundle,
    read_json,
    write_bundle,
    write_owned_json,
)
from gf_models.structured_binding import (
    _bind_network_input,
    _bind_network_input_v3,
    _bind_network_input_v4,
    check_input_binding,
    check_input_binding_game_v3,
    check_input_binding_v3,
    inference_model,
    input_binding,
    input_binding_game_v3,
    input_binding_v3,
    recorded_inference_model_v4,
)
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    ModelConfigV3,
    ModelConfigV4,
    architecture_identity,
    model_config_v2,
    model_config_v3,
    parameter_shapes,
    resource_estimate,
)
from gf_models.torch_runtime import NUMERICS, restore_weights, runtime_metadata

TRANSFORM = "signed-log1p-fp32-v1"
REWARD_POLICY = "natural-terminal-v1"
ADAM_RECIPE = {"name": "Adam", "lr": .0003, "betas": [.9, .999], "eps": 1e-8,
               "weight_decay": 0, "amsgrad": False, "maximize": False, "foreach": None,
               "capturable": False, "differentiable": False, "fused": None,
               "decoupled_weight_decay": False}


def _hash(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("SHA256 binding required")


def producer_identity(code_revision: str | None = None) -> Json:
    if code_revision is not None and (not isinstance(code_revision, str) or re.fullmatch(r"[0-9a-f]{40}", code_revision) is None):
        raise ContractError("invalid producer revision")
    root = Path(__file__).resolve().parent.parent
    source = hashlib.sha256()
    for package in ("gf_common", "gf_models", "gf_training"):
        for path in sorted((root / package).rglob("*.py")):
            source.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return {"code_revision": code_revision, "source_digest": source.hexdigest()}


def semantics(model: ModelConfigV2) -> Json:
    if type(model) in (ModelConfigV3, ModelConfigV4):
        raise ContractError("legacy semantics requires its original model identity")
    return {"input_recipe": INPUT_RECIPE, "generator_version": GENERATOR_V2,
            "policy_input_version": POLICY_INPUT_VERSION, "end_projection_version": END_PROJECTION_VERSION,
            "reward_policy_version": REWARD_POLICY, "reviewed_spec_hashes": REVIEWED_SPEC_HASHES,
            "simulator_revision": SIMULATOR_REVISION, "catalog": CATALOG_META,
            "token_signature": signature(model.event.aggregate_window), "numeric_transform": TRANSFORM,
            "runtime": runtime_metadata(), "numerics": NUMERICS}


@dataclass(frozen=True)
class _ArtifactContext:
    action_contract: ActionContract | CardInstanceContract
    purpose: str
    synthetic_capacity: bool
    layout: StructuredLayout = LEGACY_LAYOUT
    rules: CardInstanceRules | None = None
    action_config_hash: str | None = None
    _game: bool = False

    @property
    def schema(self) -> int:
        return 3 if self.layout is LEGACY_LAYOUT else 4

    @property
    def options(self) -> dict[str, Any]:
        return {} if self.schema == 3 else {"layout": self.layout}

    def classification(self) -> Json:
        if self.schema == 3:
            if type(self.action_contract) is not ActionContract:
                raise ContractError("format3 requires exact legacy action contract")
            return input_binding(self.action_contract, self.purpose, self.synthetic_capacity)
        if type(self.action_contract) is not CardInstanceContract or self.rules is None:
            raise ContractError("format4 requires explicit contract/rules")
        if self.layout is not layout_for_contract(self.action_contract):
            raise ContractError("format4 exact profile/layout mismatch")
        _hash(self.action_config_hash)
        return (input_binding_game_v3(self.action_contract, self.rules) if self._game else
            input_binding_v3(self.action_contract, self.rules, self.purpose, self.synthetic_capacity))

    def check(self, raw: Json) -> None:
        expected = self.classification()
        if self.schema == 3:
            assert type(self.action_contract) is ActionContract
            check_input_binding(raw, action_contract=self.action_contract, purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        else:
            assert type(self.action_contract) is CardInstanceContract and self.rules is not None
            if self._game:
                check_input_binding_game_v3(raw, action_contract=self.action_contract, rules=self.rules)
            else:
                check_input_binding_v3(raw, action_contract=self.action_contract, rules=self.rules,
                    purpose=self.purpose, synthetic_capacity=self.synthetic_capacity)
        if set(raw) != set(expected):
            raise ContractError("artifact classification mismatch")

    def bind(self, network: Any, *, fresh: bool = False) -> None:
        self.classification()
        if self.schema == 3:
            assert type(self.action_contract) is ActionContract
            _bind_network_input(network, action_contract=self.action_contract, purpose=self.purpose,
                synthetic_capacity=self.synthetic_capacity, fresh=fresh)
        else:
            assert type(self.action_contract) is CardInstanceContract and self.rules is not None and self.action_config_hash is not None
            _bind_network_input_v3(network, action_contract=self.action_contract, rules=self.rules,
                purpose=self.purpose, synthetic_capacity=self.synthetic_capacity,
                action_config_hash=self.action_config_hash, fresh=fresh, _game=self._game)

    def model(self, raw: Json) -> Any:
        model = model_config_v2(raw) if self.schema == 3 else model_config_v3(raw, layout=self.layout)
        if self._game and model.max_positions > 18:
            raise ContractError("game artifact capacity exceeds real18")
        return model

    def semantics(self, model: Any) -> Json:
        if self.schema == 3:
            return semantics(model)
        assert type(self.action_contract) is CardInstanceContract and self.rules is not None
        return semantics_v3(model, layout=self.layout, action_contract=self.action_contract, rules=self.rules)

    def default_model(self) -> Any:
        return ModelConfigV2() if self.schema == 3 else ModelConfigV3()


@dataclass(frozen=True)
class _ArtifactContextV4(_ArtifactContext):
    @property
    def schema(self) -> int:
        return 5

    def model(self, raw: Json) -> ModelConfigV4:
        model = recorded_inference_model_v4(raw, layout=self.layout)
        if model.max_positions > 18:
            raise ContractError("inference5 capacity exceeds real18")
        return model

    def default_model(self) -> ModelConfigV4:
        from gf_models.hpo_config import model_defaults_v4
        return model_defaults_v4()

    def bind(self, network: Any, *, fresh: bool = False) -> None:
        self.classification()
        assert type(self.action_contract) is CardInstanceContract and self.rules is not None and self.action_config_hash is not None
        _bind_network_input_v4(network, action_contract=self.action_contract, rules=self.rules,
            purpose=self.purpose, synthetic_capacity=self.synthetic_capacity,
            action_config_hash=self.action_config_hash, fresh=fresh, _game=self._game)

    def semantics(self, model: Any) -> Json:
        from gf_models.structured_sealed import static_semantics_v4
        assert type(self.action_contract) is CardInstanceContract and self.rules is not None
        return {**static_semantics_v4(model, layout=self.layout, action_contract=self.action_contract,
            rules=self.rules), "runtime": runtime_metadata()}


def _game_context_v4(action_contract: CardInstanceContract, rules: CardInstanceRules,
                     action_config_hash: str) -> _ArtifactContextV4:
    return _ArtifactContextV4(action_contract, "qualified_game", False,
        layout_for_contract(action_contract), rules, action_config_hash, True)


def semantics_v3(model: ModelConfigV3, *, layout: StructuredLayout, action_contract: CardInstanceContract,
                 rules: CardInstanceRules) -> Json:
    from gf_models.structured_sealed import static_semantics_v3
    return {**static_semantics_v3(model, layout=layout, action_contract=action_contract, rules=rules),
            "runtime": runtime_metadata()}


def _cpu_array(tensor: Any) -> Any:
    if tensor.numel() * tensor.element_size() > MAX_ARRAY_BYTES:
        raise ContractError("single tensor exceeds64MiB before CPU copy")
    return tensor.detach().cpu().numpy().copy()


def adam_recipe(learning_rate: float, beta1: float, beta2: float, epsilon: float) -> Json:
    """Closed configurable Adam descriptor, independent of training modules."""
    import math
    for value, high, strict_high in ((learning_rate, 1, False), (beta1, 1, True),
                                     (beta2, 1, True), (epsilon, 1, False)):
        if (type(value) is not float or not math.isfinite(value) or value < 0 or value > high
                or strict_high and value == high):
            raise ContractError("invalid configurable Adam scalar")
    if learning_rate == 0 or epsilon == 0:
        raise ContractError("positive Adam learning rate/epsilon required")
    return {**ADAM_RECIPE, "lr": learning_rate, "betas": [beta1, beta2], "eps": epsilon}


def _optimizer_descriptor(model: Any, value: Json | None) -> Json:
    if value is None:
        return ADAM_RECIPE
    if type(model) is not ModelConfigV4:
        raise ContractError("custom optimizer descriptor requires model4")
    raw = closed(value, set(ADAM_RECIPE))
    betas = raw["betas"]
    if type(betas) is not list or len(betas) != 2:
        raise ContractError("closed Adam betas required")
    expected = adam_recipe(raw["lr"], betas[0], betas[1], raw["eps"])
    if canonical(raw) != canonical(expected):
        raise ContractError("noncanonical configurable Adam execution options")
    return expected


def _resume_fields(context: _ArtifactContext, fields: Json | None) -> Json:
    if context.schema != 5:
        if fields is not None:
            raise ContractError("resume metadata is exclusive to format5")
        return {}
    raw = closed(fields, {"training_recipe"})
    if type(raw["training_recipe"]) is not dict:
        raise ContractError("resume5 requires a closed caller-validated training recipe")
    return raw


def validate_save_context(network: Any, model: ModelConfigV2 | ModelConfigV3 | ModelConfigV4,
                          binding: InferenceBindingV2 | None = None, optimizer: Any = None, *,
                          layout: StructuredLayout = LEGACY_LAYOUT,
                          _optimizer_recipe: Json | None = None) -> Json | None:
    """Validate actual owned execution settings before copies or file creation."""
    expected_type = ModelConfigV2 if layout is LEGACY_LAYOUT else ModelConfigV4 if type(model) is ModelConfigV4 else ModelConfigV3
    if (type(model) is not expected_type or type(network.config) is not expected_type
            or layout is not LEGACY_LAYOUT and network.layout is not layout
            or canonical(asdict(network.config)) != canonical(asdict(model))):
        raise ContractError("save network/full model config mismatch")
    actual_binding = network.binding
    if type(actual_binding) is not InferenceBindingV2 or str(network.device) != actual_binding.device:
        raise ContractError("save network/runtime binding mismatch")
    if binding is not None and (type(binding) is not InferenceBindingV2
            or canonical(asdict(actual_binding)) != canonical(asdict(binding))):
        raise ContractError("save supplied/network runtime binding mismatch")
    resource_estimate(model, actual_binding, layout=layout)
    expected = parameter_shapes(model, layout=layout)
    if {name: tuple(value.shape) for name, value in network.state_dict().items()} != expected:
        raise ContractError("network/config tensor identity mismatch")
    if any(str(value.dtype) != "torch.float32" or value.numel() * value.element_size() > MAX_ARRAY_BYTES
           for value in network.state_dict().values()):
        raise ContractError("save requires bounded FP32 network tensors")
    if optimizer is None:
        return None
    import torch
    expected_recipe = _optimizer_descriptor(model, _optimizer_recipe)
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ContractError("canonical single-group standard Adam required")
    group = optimizer.param_groups[0]
    option_keys = set(ADAM_RECIPE) - {"name"}
    if set(group) != option_keys | {"params"} or set(optimizer.defaults) != option_keys:
        raise ContractError("noncanonical Adam option fields")
    options = {"name": "Adam", **{key: group[key] for key in option_keys}}
    defaults = {"name": "Adam", **optimizer.defaults}
    for source in (group, optimizer.defaults):
        for key in option_keys:
            value, expected_value = source[key], expected_recipe[key]
            if key == "betas":
                valid_type = type(value) is tuple and len(value) == 2 and all(type(number) is float for number in value)
            else:
                valid_type = type(value) is type(expected_value)
            if not valid_type:
                raise ContractError("noncanonical Adam option types")
    try:
        matches = canonical(options) == canonical(expected_recipe) and canonical(defaults) == canonical(expected_recipe)
    except (TypeError, ValueError) as exc:
        raise ContractError("noncanonical Adam option types") from exc
    if not matches:
        raise ContractError("noncanonical Adam algorithm/execution recipe")
    parameters = list(network.parameters())
    actual_parameters = group["params"]
    if len(actual_parameters) != len(parameters) or any(actual is not expected for actual, expected in zip(actual_parameters, parameters, strict=True)):
        raise ContractError("Adam parameter ownership/order mismatch")
    owned_ids = {id(parameter) for parameter in parameters}
    if any(id(parameter) not in owned_ids for parameter in optimizer.state):
        raise ContractError("foreign Adam parameter state")
    for parameter, state in optimizer.state.items():
        if not state:
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ContractError("noncanonical Adam state fields")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.shape != ()
                or step.device.type != "cpu" or not torch.isfinite(step)
                or float(step) != int(float(step)) or float(step) < 1):
            raise ContractError("invalid canonical Adam step")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.shape != parameter.shape or value.device != parameter.device
                    or not torch.isfinite(value).all()):
                raise ContractError("invalid canonical Adam moment")
        if torch.any(state["exp_avg_sq"] < 0):
            raise ContractError("negative Adam variance")
    return options


def _save_inference(directory: Path, network: Any, model: Any, action_config_hash: str, *,
                   code_revision: str | None = None, shard_bytes: int = MAX_ARRAY_BYTES,
                   action_contract: Any, purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None) -> Json:
    context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
    classification = context.classification()
    _hash(action_config_hash)
    validate_save_context(network, model, **context.options)
    context.bind(network, fresh=True)
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    producer = producer_identity(code_revision)
    create_output_directory(directory, "inference.json")
    if network.device.type == "cuda":
        import torch
        torch.cuda.synchronize(network.device)
    expected = parameter_shapes(model, **context.options)
    if {name: tuple(value.shape) for name, value in network.state_dict().items()} != expected:
        raise ContractError("network/config tensor identity mismatch")
    arrays = {name: _cpu_array(tensor) for name, tensor in network.state_dict().items()}
    manifest = {"schema_version": context.schema, "kind": "structured-inference", **classification, **context.semantics(model),
                "model": inference_model(model), "architecture_identity": architecture_identity(model, **context.options),
                "action_config_hash": action_config_hash, "producer": producer,
                "weights": write_bundle(directory, "weights", arrays, shard_bytes=shard_bytes)}
    write_owned_json(directory / "inference.json", manifest)
    return manifest


def _inspect_inference(directory: Path, *, expected_digest: str | None = None,
                      model: Any | None = None,
                      binding: InferenceBindingV2 | None = None,
                      action_config_hash: str | None = None, action_contract: Any,
                      purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None) -> tuple[Json, Any]:
    context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
    classification = context.classification()
    manifest = read_json(directory / "inference.json", expected_digest=expected_digest,
                         max_owned_bytes=min(model.resources.max_owned_bytes if model is not None else 12 * 1024**3,
                                             binding.managed_budget_bytes if binding is not None else 12 * 1024**3))
    required = {"schema_version", "kind", "model", "architecture_identity", "action_config_hash", "producer", "weights"} | set(classification) | set(context.semantics(context.default_model()))
    closed(manifest, required)
    context.check({key: manifest[key] for key in classification})
    saved = context.model(manifest["model"])
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != context.schema or manifest["kind"] != "structured-inference":
        raise ContractError("unsupported structured inference schema")
    if any(canonical(manifest[key]) != canonical(value) for key, value in context.semantics(saved).items()):
        raise ContractError("structured inference semantic/runtime mismatch")
    _hash(manifest["action_config_hash"])
    _hash(manifest["architecture_identity"])
    if manifest["architecture_identity"] != architecture_identity(saved, **context.options):
        raise ContractError("saved architecture identity mismatch")
    producer = closed(manifest["producer"], {"code_revision", "source_digest"})
    _hash(producer["source_digest"])
    revision = producer["code_revision"]
    if revision is not None and (not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ContractError("invalid saved producer revision")
    active = saved if model is None else model
    if context.schema == 5 and model is None:
        raise ContractError("inference5 requires caller-owned active model/resource admission")
    if architecture_identity(active, **context.options) != manifest["architecture_identity"]:
        raise ContractError("active network architecture mismatch")
    expected_action_hash = context.action_config_hash if context.schema >= 4 else action_config_hash
    if expected_action_hash is not None and expected_action_hash != manifest["action_config_hash"]:
        raise ContractError("action recipe binding mismatch")
    if binding is not None and type(binding) is not InferenceBindingV2:
        raise ContractError("closed active inference binding required")
    active_binding = InferenceBindingV2() if binding is None else binding
    resource_estimate(active, active_binding, **context.options)  # before archive decoding/network allocation.
    metadata = inspect_bundle(directory, manifest["weights"], max_decoded_bytes=active.resources.max_owned_bytes)
    expected = parameter_shapes(active, **context.options)
    if set(metadata) != set(expected) or any(metadata[name]["dtype"] != "float32" or metadata[name]["shape"] != list(shape)
                                             for name, shape in expected.items()):
        raise ContractError("weight keys/shape/dtype mismatch before construction")
    return manifest, active


def _load_inference(directory: Path, *, _context: _ArtifactContext | None = None, **options: Any) -> tuple[Any, Json]:
    context = _context or _ArtifactContext(options["action_contract"], options["purpose"], options["synthetic_capacity"])
    manifest, model = _inspect_inference(directory, **options, _context=context)
    binding = options.get("binding") or InferenceBindingV2()
    arrays = read_bundle(directory, manifest["weights"], max_decoded_bytes=model.resources.max_owned_bytes)
    from gf_models.structured_network import make_network
    network = make_network(model, 0, binding, **context.options)
    context.bind(network, fresh=True)
    restore_weights(network, arrays)
    network.eval()
    return network, manifest


def _validate_shuffle_owner(shuffle: Any) -> None:
    if type(shuffle) is not np.random.Generator or type(shuffle.bit_generator) is not np.random.PCG64:
        raise ContractError("format4 requires the supplied PCG64 shuffle owner")


def _save_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int,
                model: Any, binding: InferenceBindingV2, resume_identity: str, *,
                shard_bytes: int = MAX_ARRAY_BYTES, action_contract: Any,
                purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None,
                _optimizer_recipe: Json | None = None, _resume_metadata: Json | None = None) -> Json:
    context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
    classification = context.classification()
    metadata = _resume_fields(context, _resume_metadata)
    if context.schema >= 4:
        _validate_shuffle_owner(shuffle)
    context.bind(network)
    import torch
    _hash(resume_identity)
    integer(steps, 0, 2**31 - 1)
    recipe = validate_save_context(network, model, binding, optimizer, _optimizer_recipe=_optimizer_recipe, **context.options)
    _inspect_inference(directory, model=model, binding=binding, action_contract=action_contract,
                      purpose=purpose, synthetic_capacity=synthetic_capacity, _context=context)
    preflight_outputs(directory, ["resume.json", "resume.json.tmp"])
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    if binding.device != "cpu":
        torch.cuda.synchronize(binding.device)
    arrays = {"rng.torch_cpu": _cpu_array(torch.get_rng_state())}
    cuda_states = torch.cuda.get_rng_state_all() if binding.device != "cpu" else []
    for index, value in enumerate(cuda_states):
        arrays[f"rng.torch_cuda.{index}"] = _cpu_array(value)
    presence: Json = {}
    for name, parameter in network.named_parameters():
        state = optimizer.state.get(parameter)
        present = bool(state)
        presence[name] = {"present": present}
        if present:
            integer(int(state["step"].item()), 1, steps)
            arrays[f"adam.{name}.step"] = np.asarray(int(state["step"].item()), dtype=np.int64)
            for key in ("exp_avg", "exp_avg_sq"):
                arrays[f"adam.{name}.{key}"] = _cpu_array(state[key])
    manifest = {"schema_version": context.schema, "kind": "structured-resume",
                **({"action_config_hash": context.action_config_hash} if context.schema >= 4 else {}), **classification, **context.semantics(model), **metadata,
                "inference_digest": sha256(directory / "inference.json"), "model": asdict(model),
                "binding": asdict(binding), "resume_identity": resume_identity,
                "producer": producer_identity(), "optimizer_steps": steps,
                "optimizer_recipe": recipe,
                "parameter_states": presence, "cuda_rng_count": len(cuda_states),
                "shuffle_rng": shuffle.bit_generator.state,
                "state": write_bundle(directory, "resume", arrays, shard_bytes=shard_bytes)}
    write_owned_json(directory / "resume.json", manifest)
    return manifest


def _read_resume(directory: Path, model: Any, binding: InferenceBindingV2,
                resume_identity: str, *, action_contract: Any,
                purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None,
                _optimizer_recipe: Json | None = None, _resume_metadata: Json | None = None) -> tuple[Json, dict[str, Any]]:
    context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
    classification = context.classification()
    metadata = _resume_fields(context, _resume_metadata)
    optimizer_recipe = _optimizer_descriptor(model, _optimizer_recipe)
    manifest = read_json(directory / "resume.json",
                         max_owned_bytes=min(model.resources.max_owned_bytes, binding.managed_budget_bytes))
    closed(manifest.get("producer"), {"code_revision", "source_digest"})
    _hash(resume_identity)
    _hash(manifest["producer"]["source_digest"])
    producer_revision = manifest["producer"]["code_revision"]
    if producer_revision is not None and (not isinstance(producer_revision, str) or re.fullmatch(r"[0-9a-f]{40}", producer_revision) is None):
        raise ContractError("invalid resume producer revision")
    keys = {"schema_version", "kind", "inference_digest", "model", "binding", "resume_identity", "producer",
            "optimizer_steps", "optimizer_recipe", "parameter_states", "cuda_rng_count", "shuffle_rng", "state"} | ({"action_config_hash"} if context.schema >= 4 else set()) | set(classification) | set(context.semantics(model)) | set(metadata)
    closed(manifest, keys)
    if context.schema >= 4 and manifest["action_config_hash"] != context.action_config_hash:
        raise ContractError("resume4 action-config binding mismatch")
    context.check({key: manifest[key] for key in classification})
    _inspect_inference(directory, expected_digest=manifest["inference_digest"], model=model, binding=binding,
                      action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity, _context=context)
    import torch
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != context.schema
            or manifest["kind"] != "structured-resume"
            or manifest["inference_digest"] != sha256(directory / "inference.json")
            or canonical(manifest["model"]) != canonical(asdict(model))
            or canonical(manifest["binding"]) != canonical(asdict(binding))
            or manifest["resume_identity"] != resume_identity
            or manifest["producer"]["source_digest"] != producer_identity()["source_digest"]
            or any(canonical(manifest[key]) != canonical(value) for key, value in context.semantics(model).items())
            or canonical(manifest["optimizer_recipe"]) != canonical(optimizer_recipe)
            or any(canonical(manifest[key]) != canonical(value) for key, value in metadata.items())):
        raise ContractError("ordinary resume model/binding/runtime/source identity mismatch")
    integer(manifest["optimizer_steps"], 0, 2**31 - 1)
    integer(manifest["cuda_rng_count"], 0, 1024)
    count = torch.cuda.device_count() if binding.device != "cpu" else 0
    if manifest["cuda_rng_count"] != count:
        raise ContractError("resume CUDA RNG ownership mismatch")
    resource_estimate(model, binding, **context.options)
    descriptors = inspect_bundle(directory, manifest["state"], max_decoded_bytes=model.resources.max_owned_bytes)
    expected: Json = {"rng.torch_cpu": ("uint8", None)}
    expected.update({f"rng.torch_cuda.{index}": ("uint8", None) for index in range(count)})
    parameters = parameter_shapes(model, **context.options)
    if not isinstance(manifest["parameter_states"], dict) or set(manifest["parameter_states"]) != set(parameters):
        raise ContractError("resume stable parameter ownership mismatch")
    for name, parameter in parameters.items():
        state = closed(manifest["parameter_states"][name], {"present"})
        if type(state["present"]) is not bool:
            raise ContractError("optimizer presence must be boolean")
        if state["present"]:
            expected[f"adam.{name}.step"] = ("int64", [])
            for key in ("exp_avg", "exp_avg_sq"):
                expected[f"adam.{name}.{key}"] = ("float32", list(parameter))
    if set(expected) != set(descriptors):
        raise ContractError("resume tensor presence mismatch")
    for key, (dtype, shape) in expected.items():
        if descriptors[key]["dtype"] != dtype or (shape is not None and descriptors[key]["shape"] != shape):
            raise ContractError("resume tensor dtype/shape mismatch")
    arrays = read_bundle(directory, manifest["state"], max_decoded_bytes=model.resources.max_owned_bytes)
    # Every second moment is a variance, not merely a finite-shaped tensor.
    # Reject rehashed negative values before model staging/RNG publication.
    for name in parameters:
        if manifest["parameter_states"][name]["present"] and np.any(arrays[f"adam.{name}.exp_avg_sq"] < 0):
            raise ContractError("negative Adam variance in resume state")
    # Validate all RNG/Adam data into temporary owners before publishing state.
    probe = torch.Generator(device="cpu")
    try:
        probe.set_state(torch.from_numpy(arrays["rng.torch_cpu"].copy()))
    except RuntimeError as exc:
        raise ContractError("invalid CPU RNG payload") from exc
    cuda_rng = []
    for index in range(count):
        cuda_state = torch.from_numpy(arrays[f"rng.torch_cuda.{index}"].copy())
        torch.Generator(device=f"cuda:{index}").set_state(cuda_state)
        cuda_rng.append(cuda_state)
    shuffle_state = closed(manifest["shuffle_rng"], {"bit_generator", "state", "has_uint32", "uinteger"})
    nested_state = closed(shuffle_state["state"], {"state", "inc"})
    if shuffle_state["bit_generator"] != "PCG64":
        raise ContractError("unsupported shuffle RNG")
    integer(nested_state["state"], 0, 2**128 - 1)
    integer(nested_state["inc"], 1, 2**128 - 1)
    integer(shuffle_state["has_uint32"], 0, 1)
    integer(shuffle_state["uinteger"], 0, 2**32 - 1)
    if nested_state["inc"] % 2 != 1:
        raise ContractError("invalid PCG64 stream increment")
    probe_shuffle = np.random.Generator(np.random.PCG64(0))
    probe_shuffle.bit_generator.state = shuffle_state
    for name in parameters:
        if manifest["parameter_states"][name]["present"]:
            integer(int(arrays[f"adam.{name}.step"]), 1, manifest["optimizer_steps"])
    return manifest, arrays


def _restore_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any,
                   model: Any, binding: InferenceBindingV2, resume_identity: str, *,
                   action_contract: Any, purpose: str, synthetic_capacity: bool, _context: _ArtifactContext | None = None,
                   _optimizer_recipe: Json | None = None, _resume_metadata: Json | None = None) -> int:
    context = _context or _ArtifactContext(action_contract, purpose, synthetic_capacity)
    context.classification()
    if context.schema >= 4:
        _validate_shuffle_owner(shuffle)
    import torch
    if any(hasattr(network, name) for name in ("_structured_input_binding", "_structured_input_binding_v3", "_structured_input_binding_v4")):
        context.bind(network)
    validate_save_context(network, model, binding, optimizer, _optimizer_recipe=_optimizer_recipe, **context.options)
    manifest, arrays = _read_resume(directory, model, binding, resume_identity, action_contract=action_contract,
                                   purpose=purpose, synthetic_capacity=synthetic_capacity, _context=context,
                                   _optimizer_recipe=_optimizer_recipe, _resume_metadata=_resume_metadata)
    context.bind(network, fresh=True)
    parameters = dict(network.named_parameters())
    count = manifest["cuda_rng_count"]
    states = {}
    for name, parameter in parameters.items():
        if manifest["parameter_states"][name]["present"]:
            step = int(arrays[f"adam.{name}.step"])
            integer(step, 1, manifest["optimizer_steps"])
            states[parameter] = {"step": torch.tensor(float(step), dtype=torch.float32),
                                 **{key: torch.from_numpy(arrays[f"adam.{name}.{key}"].copy()).to(binding.device)
                                    for key in ("exp_avg", "exp_avg_sq")}}
    optimizer.state.clear()
    optimizer.state.update(states)
    torch.set_rng_state(torch.from_numpy(arrays["rng.torch_cpu"].copy()))
    if count:
        torch.cuda.set_rng_state_all([torch.from_numpy(arrays[f"rng.torch_cuda.{index}"].copy()) for index in range(count)])
    shuffle.bit_generator.state = manifest["shuffle_rng"]
    return int(manifest["optimizer_steps"])


# Legacy entry points retain their exact signatures and closed schema3 context.
def save_inference(directory: Path, network: Any, model: ModelConfigV2, action_config_hash: str, *,
                   code_revision: str | None = None, shard_bytes: int = MAX_ARRAY_BYTES,
                   action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> Json:
    return _save_inference(directory, network, model, action_config_hash, code_revision=code_revision,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)


def inspect_inference(directory: Path, *, expected_digest: str | None = None,
                      model: ModelConfigV2 | None = None, binding: InferenceBindingV2 | None = None,
                      action_config_hash: str | None = None, action_contract: ActionContract,
                      purpose: str, synthetic_capacity: bool) -> tuple[Json, ModelConfigV2]:
    manifest, active = _inspect_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)
    assert type(active) is ModelConfigV2
    return manifest, active


def load_inference(directory: Path, **options: Any) -> tuple[Any, Json]:
    if "_context" in options:
        raise ContractError("private format context cannot enter the legacy loader")
    return _load_inference(directory, **options)


def save_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int,
                model: ModelConfigV2, binding: InferenceBindingV2, resume_identity: str, *,
                shard_bytes: int = MAX_ARRAY_BYTES, action_contract: ActionContract,
                purpose: str, synthetic_capacity: bool) -> Json:
    return _save_resume(directory, network, optimizer, shuffle, steps, model, binding, resume_identity,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)


def read_resume(directory: Path, model: ModelConfigV2, binding: InferenceBindingV2,
                resume_identity: str, *, action_contract: ActionContract,
                purpose: str, synthetic_capacity: bool) -> tuple[Json, dict[str, Any]]:
    return _read_resume(directory, model, binding, resume_identity, action_contract=action_contract,
        purpose=purpose, synthetic_capacity=synthetic_capacity)


def restore_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any,
                   model: ModelConfigV2, binding: InferenceBindingV2, resume_identity: str, *,
                   action_contract: ActionContract, purpose: str, synthetic_capacity: bool) -> int:
    return _restore_resume(directory, network, optimizer, shuffle, model, binding, resume_identity,
        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity)


def _context_v3(layout: StructuredLayout, action_contract: CardInstanceContract, rules: CardInstanceRules,
                purpose: str, synthetic_capacity: bool, action_config_hash: str) -> _ArtifactContext:
    context = _ArtifactContext(action_contract, purpose, synthetic_capacity, layout, rules, action_config_hash)
    context.classification()
    if context.schema != 4:
        raise ContractError("format4 requires explicit card layout")
    return context


def save_inference_v3(directory: Path, network: Any, model: ModelConfigV3, action_config_hash: str, *,
                       layout: StructuredLayout, action_contract: CardInstanceContract, rules: CardInstanceRules,
                       purpose: str, synthetic_capacity: bool, code_revision: str | None = None,
                       shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    return _save_inference(directory, network, model, action_config_hash, code_revision=code_revision,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, _context=context)


def inspect_inference_v3(directory: Path, *, expected_digest: str, model: ModelConfigV3, binding: InferenceBindingV2,
                          action_config_hash: str, layout: StructuredLayout, action_contract: CardInstanceContract,
                          rules: CardInstanceRules, purpose: str, synthetic_capacity: bool) -> tuple[Json, ModelConfigV3]:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    manifest, active = _inspect_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, _context=context)
    assert type(active) is ModelConfigV3
    return manifest, active


def load_inference_v3(directory: Path, *, expected_digest: str, model: ModelConfigV3, binding: InferenceBindingV2,
                       action_config_hash: str, layout: StructuredLayout, action_contract: CardInstanceContract,
                       rules: CardInstanceRules, purpose: str, synthetic_capacity: bool) -> tuple[Any, Json]:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    return _load_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, _context=context)


def save_resume_v3(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int,
                    model: ModelConfigV3, binding: InferenceBindingV2, resume_identity: str, *,
                    action_config_hash: str, layout: StructuredLayout, action_contract: CardInstanceContract,
                    rules: CardInstanceRules, purpose: str, synthetic_capacity: bool,
                    shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    return _save_resume(directory, network, optimizer, shuffle, steps, model, binding, resume_identity,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, _context=context)


def read_resume_v3(directory: Path, model: ModelConfigV3, binding: InferenceBindingV2, resume_identity: str, *,
                    action_config_hash: str, layout: StructuredLayout, action_contract: CardInstanceContract,
                    rules: CardInstanceRules, purpose: str, synthetic_capacity: bool) -> tuple[Json, dict[str, Any]]:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    return _read_resume(directory, model, binding, resume_identity, action_contract=action_contract, purpose=purpose,
        synthetic_capacity=synthetic_capacity, _context=context)


def restore_resume_v3(directory: Path, network: Any, optimizer: Any, shuffle: Any, model: ModelConfigV3,
                       binding: InferenceBindingV2, resume_identity: str, *, action_config_hash: str,
                       layout: StructuredLayout, action_contract: CardInstanceContract, rules: CardInstanceRules,
                       purpose: str, synthetic_capacity: bool) -> int:
    context = _context_v3(layout, action_contract, rules, purpose, synthetic_capacity, action_config_hash)
    return _restore_resume(directory, network, optimizer, shuffle, model, binding, resume_identity,
        action_contract=action_contract, purpose=purpose, synthetic_capacity=synthetic_capacity, _context=context)


def _game_context_v3(action_contract: CardInstanceContract, rules: CardInstanceRules,
                     action_config_hash: str) -> _ArtifactContext:
    context = _ArtifactContext(action_contract, "qualified_game", False,
        layout_for_contract(action_contract), rules, action_config_hash, True)
    context.classification()
    return context


def save_game_inference_v3(directory: Path, network: Any, model: ModelConfigV3, action_config_hash: str, *,
                            action_contract: CardInstanceContract, rules: CardInstanceRules,
                            code_revision: str | None = None, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    return _save_inference(directory, network, model, action_config_hash, code_revision=code_revision,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v3(action_contract, rules, action_config_hash))


def inspect_game_inference_v3(directory: Path, *, expected_digest: str, model: ModelConfigV3,
                               binding: InferenceBindingV2, action_config_hash: str,
                               action_contract: CardInstanceContract, rules: CardInstanceRules) -> tuple[Json, ModelConfigV3]:
    manifest, active = _inspect_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose="qualified_game",
        synthetic_capacity=False, _context=_game_context_v3(action_contract, rules, action_config_hash))
    assert type(active) is ModelConfigV3
    return manifest, active


def load_game_inference_v3(directory: Path, *, expected_digest: str, model: ModelConfigV3,
                            binding: InferenceBindingV2, action_config_hash: str,
                            action_contract: CardInstanceContract, rules: CardInstanceRules) -> tuple[Any, Json]:
    return _load_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose="qualified_game",
        synthetic_capacity=False, _context=_game_context_v3(action_contract, rules, action_config_hash))


def save_game_resume_v3(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int,
                         model: ModelConfigV3, binding: InferenceBindingV2, resume_identity: str, *,
                         action_config_hash: str, action_contract: CardInstanceContract, rules: CardInstanceRules,
                         shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    return _save_resume(directory, network, optimizer, shuffle, steps, model, binding, resume_identity,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v3(action_contract, rules, action_config_hash))


def read_game_resume_v3(directory: Path, model: ModelConfigV3, binding: InferenceBindingV2,
                         resume_identity: str, *, action_config_hash: str,
                         action_contract: CardInstanceContract, rules: CardInstanceRules) -> tuple[Json, dict[str, Any]]:
    return _read_resume(directory, model, binding, resume_identity, action_contract=action_contract,
        purpose="qualified_game", synthetic_capacity=False, _context=_game_context_v3(action_contract, rules, action_config_hash))


def restore_game_resume_v3(directory: Path, network: Any, optimizer: Any, shuffle: Any, model: ModelConfigV3,
                            binding: InferenceBindingV2, resume_identity: str, *, action_config_hash: str,
                            action_contract: CardInstanceContract, rules: CardInstanceRules) -> int:
    return _restore_resume(directory, network, optimizer, shuffle, model, binding, resume_identity,
        action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v3(action_contract, rules, action_config_hash))


def save_game_inference_v4(directory: Path, network: Any, model: ModelConfigV4, action_config_hash: str, *,
                            action_contract: CardInstanceContract, rules: CardInstanceRules,
                            code_revision: str | None = None, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    return _save_inference(directory, network, model, action_config_hash, code_revision=code_revision,
        shard_bytes=shard_bytes, action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v4(action_contract, rules, action_config_hash))


def inspect_game_inference_v4(directory: Path, *, expected_digest: str, model: ModelConfigV4,
                               binding: InferenceBindingV2, action_config_hash: str,
                               action_contract: CardInstanceContract, rules: CardInstanceRules) -> tuple[Json, Any]:
    _hash(expected_digest)
    return _inspect_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v4(action_contract, rules, action_config_hash))


def load_game_inference_v4(directory: Path, *, expected_digest: str, model: ModelConfigV4,
                            binding: InferenceBindingV2, action_config_hash: str,
                            action_contract: CardInstanceContract, rules: CardInstanceRules) -> tuple[Any, Json]:
    _hash(expected_digest)
    return _load_inference(directory, expected_digest=expected_digest, model=model, binding=binding,
        action_config_hash=action_config_hash, action_contract=action_contract, purpose="qualified_game", synthetic_capacity=False,
        _context=_game_context_v4(action_contract, rules, action_config_hash))
