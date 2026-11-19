"""Inference-only sealed model import. No training or resume dependency."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from gf_common.domain import ContractError, canonical, closed, integer
from gf_common.ram_bank import ReadonlyRamBankSet
from gf_models.structured_artifacts import _game_context_v4
from gf_models.structured_binding import inference_model
from gf_models.structured_config import architecture_identity, parameter_shapes
from gf_models.structured_network import make_network
from gf_models.torch_runtime import restore_weights


def _validate_owned_inference(banks: ReadonlyRamBankSet, metadata: Any, *, model: Any,
                         binding: Any, actions: Any, rules: Any) -> Any:
    closed(metadata, {'protocol_version', 'command', 'generation', 'snapshot_identity',
        'inference_digest', 'inference', 'arrays'})
    if metadata['protocol_version'] != 9 or type(metadata['protocol_version']) is not int or metadata['command'] != 'SYNC_OWNED':
        raise ContractError('actor9 SYNC_OWNED required')
    integer(metadata['generation'], 0, 2**63-1)
    for key in ('snapshot_identity', 'inference_digest'):
        if type(metadata[key]) is not str or len(metadata[key]) != 64 or set(metadata[key])-set('0123456789abcdef'):
            raise ContractError('actor9 snapshot/inference identity required')
    from gf_common.domain import digest
    context = _game_context_v4(actions.action_contract, rules, digest(asdict(actions)))
    expected = parameter_shapes(model, **context.options)
    descriptors = [asdict(descriptor) for bank in banks.banks for descriptor in bank.descriptors]
    if (type(banks) is not ReadonlyRamBankSet or set(banks.names) != set(expected)
            or canonical(metadata['arrays']) != canonical(descriptors)):
        raise ContractError('actor9 requires exact inference-only backing/layout')
    return _validate_inference_content(banks, metadata['inference'], model=model,
        binding=binding, actions=actions, rules=rules)


def _validate_inference_content(banks: ReadonlyRamBankSet, inference: Any, *, model: Any,
                                binding: Any, actions: Any, rules: Any) -> Any:
    """Unchanged checkpoint/tensor validation, independent of online capture authority."""
    from gf_common.domain import digest
    context = _game_context_v4(actions.action_contract, rules, digest(asdict(actions)))
    expected = parameter_shapes(model, **context.options)
    if type(banks) is not ReadonlyRamBankSet or set(banks.names) != set(expected):
        raise ContractError('exact inference-only backing/layout required')
    expected_keys = {'schema_version', 'kind', 'model', 'architecture_identity', 'producer', 'action_config_hash'} | set(context.classification()) | set(context.semantics(model))
    closed(inference, expected_keys)
    if (inference['schema_version'] != context.schema or inference['kind'] != 'structured-inference'
            or canonical(inference['model']) != canonical(inference_model(model))
            or inference['architecture_identity'] != architecture_identity(model, **context.options)
            or inference['action_config_hash'] != context.action_config_hash
            or any(canonical(inference.get(key)) != canonical(value)
                for key, value in (context.classification() | context.semantics(model)).items())):
        raise ContractError('actor9 model/input recipe mismatch')
    arrays = {name: banks.array(name) for name in banks.names}
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected[name] or not np.isfinite(array).all():
            raise ContractError('actor9 invalid inference tensor')
    return context, arrays


def load_owned_inference(banks: ReadonlyRamBankSet, metadata: Any, *, model: Any,
                         binding: Any, actions: Any, rules: Any) -> Any:
    context, arrays = _validate_owned_inference(banks, metadata, model=model,
        binding=binding, actions=actions, rules=rules)
    return _load_validated_inference(context, arrays, model=model, binding=binding)


def load_owned_inference_content(banks: ReadonlyRamBankSet, inference: Any, *, model: Any,
                                 binding: Any, actions: Any, rules: Any) -> Any:
    """Data-only local import; this grants no actor9 or collection authority."""
    context, arrays = _validate_inference_content(banks, inference, model=model,
        binding=binding, actions=actions, rules=rules)
    return _load_validated_inference(context, arrays, model=model, binding=binding)


def _load_validated_inference(context: Any, arrays: Any, *, model: Any, binding: Any) -> Any:
    network = make_network(model, 0, binding, **context.options)
    context.bind(network, fresh=True)
    restore_weights(network, arrays)
    network.eval()
    if binding.device != 'cpu':
        import torch
        torch.cuda.synchronize(binding.device)
    return network
