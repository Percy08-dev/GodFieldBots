"""Structured inference/resume3: typed shards, exact bindings, named Adam state.

These files establish numerical compatibility only. They never authorize a
formal generation, source, training parent or CLI invocation on their own.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

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
from gf_models.structured_config import (
    InferenceBindingV2,
    ModelConfigV2,
    architecture_identity,
    model_config_v2,
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
    return {"input_recipe": INPUT_RECIPE, "generator_version": GENERATOR_V2,
            "policy_input_version": POLICY_INPUT_VERSION, "end_projection_version": END_PROJECTION_VERSION,
            "reward_policy_version": REWARD_POLICY, "reviewed_spec_hashes": REVIEWED_SPEC_HASHES,
            "simulator_revision": SIMULATOR_REVISION, "catalog": CATALOG_META,
            "token_signature": signature(model.event.aggregate_window), "numeric_transform": TRANSFORM,
            "runtime": runtime_metadata(), "numerics": NUMERICS}


def _cpu_array(tensor: Any) -> Any:
    if tensor.numel() * tensor.element_size() > MAX_ARRAY_BYTES:
        raise ContractError("single tensor exceeds64MiB before CPU copy")
    return tensor.detach().cpu().numpy().copy()


def validate_save_context(network: Any, model: ModelConfigV2,
                          binding: InferenceBindingV2 | None = None, optimizer: Any = None) -> Json | None:
    """Validate actual owned execution settings before copies or file creation."""
    if (type(model) is not ModelConfigV2 or type(network.config) is not ModelConfigV2
            or canonical(asdict(network.config)) != canonical(asdict(model))):
        raise ContractError("save network/full model config mismatch")
    actual_binding = network.binding
    if type(actual_binding) is not InferenceBindingV2 or str(network.device) != actual_binding.device:
        raise ContractError("save network/runtime binding mismatch")
    if binding is not None and (type(binding) is not InferenceBindingV2
            or canonical(asdict(actual_binding)) != canonical(asdict(binding))):
        raise ContractError("save supplied/network runtime binding mismatch")
    resource_estimate(model, actual_binding)
    expected = parameter_shapes(model)
    if {name: tuple(value.shape) for name, value in network.state_dict().items()} != expected:
        raise ContractError("network/config tensor identity mismatch")
    if any(str(value.dtype) != "torch.float32" or value.numel() * value.element_size() > MAX_ARRAY_BYTES
           for value in network.state_dict().values()):
        raise ContractError("save requires bounded FP32 network tensors")
    if optimizer is None:
        return None
    import torch
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
            value, expected_value = source[key], ADAM_RECIPE[key]
            if key == "betas":
                valid_type = type(value) is tuple and len(value) == 2 and all(type(number) is float for number in value)
            else:
                valid_type = type(value) is type(expected_value)
            if not valid_type:
                raise ContractError("noncanonical Adam option types")
    try:
        matches = canonical(options) == canonical(ADAM_RECIPE) and canonical(defaults) == canonical(ADAM_RECIPE)
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


def save_inference(directory: Path, network: Any, model: ModelConfigV2, action_config_hash: str, *,
                   code_revision: str | None = None, shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    _hash(action_config_hash)
    validate_save_context(network, model)
    integer(shard_bytes, 1, MAX_ARRAY_BYTES)
    producer = producer_identity(code_revision)
    create_output_directory(directory, "inference.json")
    if network.device.type == "cuda":
        import torch
        torch.cuda.synchronize(network.device)
    expected = parameter_shapes(model)
    if {name: tuple(value.shape) for name, value in network.state_dict().items()} != expected:
        raise ContractError("network/config tensor identity mismatch")
    arrays = {name: _cpu_array(tensor) for name, tensor in network.state_dict().items()}
    manifest = {"schema_version": 3, "kind": "structured-inference", **semantics(model),
                "model": asdict(model), "architecture_identity": architecture_identity(model),
                "action_config_hash": action_config_hash, "producer": producer,
                "weights": write_bundle(directory, "weights", arrays, shard_bytes=shard_bytes)}
    write_owned_json(directory / "inference.json", manifest)
    return manifest


def inspect_inference(directory: Path, *, expected_digest: str | None = None,
                      model: ModelConfigV2 | None = None,
                      binding: InferenceBindingV2 | None = None,
                      action_config_hash: str | None = None) -> tuple[Json, ModelConfigV2]:
    manifest = read_json(directory / "inference.json", expected_digest=expected_digest,
                         max_owned_bytes=min(model.resources.max_owned_bytes if model is not None else 12 * 1024**3,
                                             binding.managed_budget_bytes if binding is not None else 12 * 1024**3))
    required = {"schema_version", "kind", "model", "architecture_identity", "action_config_hash", "producer", "weights"} | set(semantics(ModelConfigV2()))
    closed(manifest, required)
    saved = model_config_v2(manifest["model"])
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 3 or manifest["kind"] != "structured-inference":
        raise ContractError("unsupported structured inference schema")
    if any(canonical(manifest[key]) != canonical(value) for key, value in semantics(saved).items()):
        raise ContractError("structured inference semantic/runtime mismatch")
    _hash(manifest["action_config_hash"])
    _hash(manifest["architecture_identity"])
    if manifest["architecture_identity"] != architecture_identity(saved):
        raise ContractError("saved architecture identity mismatch")
    producer = closed(manifest["producer"], {"code_revision", "source_digest"})
    _hash(producer["source_digest"])
    revision = producer["code_revision"]
    if revision is not None and (not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ContractError("invalid saved producer revision")
    active = saved if model is None else model
    if architecture_identity(active) != manifest["architecture_identity"]:
        raise ContractError("active network architecture mismatch")
    if action_config_hash is not None and action_config_hash != manifest["action_config_hash"]:
        raise ContractError("action recipe binding mismatch")
    if binding is not None and type(binding) is not InferenceBindingV2:
        raise ContractError("closed active inference binding required")
    active_binding = InferenceBindingV2() if binding is None else binding
    resource_estimate(active, active_binding)  # before archive decoding/network allocation.
    metadata = inspect_bundle(directory, manifest["weights"], max_decoded_bytes=active.resources.max_owned_bytes)
    expected = parameter_shapes(active)
    if set(metadata) != set(expected) or any(metadata[name]["dtype"] != "float32" or metadata[name]["shape"] != list(shape)
                                             for name, shape in expected.items()):
        raise ContractError("weight keys/shape/dtype mismatch before construction")
    return manifest, active


def load_inference(directory: Path, **options: Any) -> tuple[Any, Json]:
    manifest, model = inspect_inference(directory, **options)
    binding = options.get("binding") or InferenceBindingV2()
    # All shards and NPY headers are validated before a model is constructed.
    arrays = read_bundle(directory, manifest["weights"], max_decoded_bytes=model.resources.max_owned_bytes)
    from gf_models.structured_network import make_network
    network = make_network(model, 0, binding)
    restore_weights(network, arrays)
    network.eval()
    return network, manifest


def save_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any, steps: int,
                model: ModelConfigV2, binding: InferenceBindingV2, resume_identity: str, *,
                shard_bytes: int = MAX_ARRAY_BYTES) -> Json:
    import torch
    _hash(resume_identity)
    integer(steps, 0, 2**31 - 1)
    recipe = validate_save_context(network, model, binding, optimizer)
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
    manifest = {"schema_version": 3, "kind": "structured-resume", **semantics(model),
                "inference_digest": sha256(directory / "inference.json"), "model": asdict(model),
                "binding": asdict(binding), "resume_identity": resume_identity,
                "producer": producer_identity(), "optimizer_steps": steps,
                "optimizer_recipe": recipe,
                "parameter_states": presence, "cuda_rng_count": len(cuda_states),
                "shuffle_rng": shuffle.bit_generator.state,
                "state": write_bundle(directory, "resume", arrays, shard_bytes=shard_bytes)}
    write_owned_json(directory / "resume.json", manifest)
    return manifest


def read_resume(directory: Path, model: ModelConfigV2, binding: InferenceBindingV2,
                resume_identity: str) -> tuple[Json, dict[str, Any]]:
    import torch
    manifest = read_json(directory / "resume.json",
                         max_owned_bytes=min(model.resources.max_owned_bytes, binding.managed_budget_bytes))
    closed(manifest.get("producer"), {"code_revision", "source_digest"})
    _hash(resume_identity)
    _hash(manifest["producer"]["source_digest"])
    producer_revision = manifest["producer"]["code_revision"]
    if producer_revision is not None and (not isinstance(producer_revision, str) or re.fullmatch(r"[0-9a-f]{40}", producer_revision) is None):
        raise ContractError("invalid resume producer revision")
    keys = {"schema_version", "kind", "inference_digest", "model", "binding", "resume_identity", "producer",
            "optimizer_steps", "optimizer_recipe", "parameter_states", "cuda_rng_count", "shuffle_rng", "state"} | set(semantics(model))
    closed(manifest, keys)
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 3
            or manifest["kind"] != "structured-resume"
            or manifest["inference_digest"] != sha256(directory / "inference.json")
            or canonical(manifest["model"]) != canonical(asdict(model))
            or canonical(manifest["binding"]) != canonical(asdict(binding))
            or manifest["resume_identity"] != resume_identity
            or manifest["producer"]["source_digest"] != producer_identity()["source_digest"]
            or any(canonical(manifest[key]) != canonical(value) for key, value in semantics(model).items())
            or canonical(manifest["optimizer_recipe"]) != canonical(ADAM_RECIPE)):
        raise ContractError("ordinary resume model/binding/runtime/source identity mismatch")
    integer(manifest["optimizer_steps"], 0, 2**31 - 1)
    integer(manifest["cuda_rng_count"], 0, 1024)
    count = torch.cuda.device_count() if binding.device != "cpu" else 0
    if manifest["cuda_rng_count"] != count:
        raise ContractError("resume CUDA RNG ownership mismatch")
    resource_estimate(model, binding)
    descriptors = inspect_bundle(directory, manifest["state"], max_decoded_bytes=model.resources.max_owned_bytes)
    expected: Json = {"rng.torch_cpu": ("uint8", None)}
    expected.update({f"rng.torch_cuda.{index}": ("uint8", None) for index in range(count)})
    parameters = parameter_shapes(model)
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


def restore_resume(directory: Path, network: Any, optimizer: Any, shuffle: Any,
                   model: ModelConfigV2, binding: InferenceBindingV2, resume_identity: str) -> int:
    import torch
    validate_save_context(network, model, binding, optimizer)
    manifest, arrays = read_resume(directory, model, binding, resume_identity)
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
