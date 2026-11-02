"""Closed stdlib Tree2 configuration and resource identities, separate from NN2."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from gf_common.domain import ContractError, Json, closed, digest, integer
from gf_common.structured_history import ACTIVE_AGGREGATE, HISTORY_WIDTH
from gf_common.structured_tokens import (
    CARD_CATEGORIES,
    CARD_NUMERIC_FIELDS,
    STATE_CATEGORIES,
    STATE_NUMERIC_FIELDS,
    signature,
)

TREE_RECIPE = 'participant-structured-tree-v1'
TREE_NUMERICS = 'numpy-fp32-signed-log1p-v1'
MAX_ARRAY_BYTES = 64 * 1024**2
CARD_WIDTH = 2 * len(CARD_NUMERIC_FIELDS) + len(CARD_CATEGORIES) + 1
CONTEXT_WIDTH = 2 * len(STATE_NUMERIC_FIELDS) + 2 * len(STATE_CATEGORIES) + 18 * CARD_WIDTH + 18 + HISTORY_WIDTH
CANDIDATE_FIXED_WIDTH = 2 + 8 + 1 + 18


def feature_width(positions: int) -> int:
    integer(positions, 1, 128)
    return CONTEXT_WIDTH + CANDIDATE_FIXED_WIDTH + positions * CARD_WIDTH


@dataclass(frozen=True)
class TreeEncoderConfigV2:
    recipe: str = TREE_RECIPE
    max_positions: int = 18
    aggregate_window: int = 128
    aggregate_recipe: str = ACTIVE_AGGREGATE

    def __post_init__(self) -> None:
        if self.recipe != TREE_RECIPE or self.aggregate_recipe != ACTIVE_AGGREGATE:
            raise ContractError('unsupported active Tree2 feature/history recipe')
        integer(self.max_positions, 1, 128)
        integer(self.aggregate_window, 1, 65536)


@dataclass(frozen=True)
class TreeNetworkConfigV2:
    num_leaves: int = 31

    def __post_init__(self) -> None:
        integer(self.num_leaves, 2, 65536)


@dataclass(frozen=True)
class TreeInitializationConfigV2:
    recipe: str = 'seed-derived-v1'

    def __post_init__(self) -> None:
        if self.recipe != 'seed-derived-v1':
            raise ContractError('unsupported Tree2 initialization')


@dataclass(frozen=True)
class TreeInferenceConfigV2:
    device: str = 'cpu'
    candidate_chunk: int = 256
    max_candidates: int = 32768
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.device != 'cpu':
            raise ContractError('Tree2 inference requires CPU')
        integer(self.candidate_chunk, 1, 65536)
        integer(self.max_candidates, 1, 65536)
        integer(self.num_threads, 1, 64)


@dataclass(frozen=True)
class TreeResourcesV2:
    max_owned_bytes: int = 2 * 1024**3
    max_array_bytes: int = MAX_ARRAY_BYTES

    def __post_init__(self) -> None:
        integer(self.max_owned_bytes, 1, 2**63 - 1)
        integer(self.max_array_bytes, 1, MAX_ARRAY_BYTES)


@dataclass(frozen=True)
class TreeArchitectureV2:
    model_id: str = 'lightgbm'
    version: str = '2'
    encoder: TreeEncoderConfigV2 = field(default_factory=TreeEncoderConfigV2)
    network: TreeNetworkConfigV2 = field(default_factory=TreeNetworkConfigV2)
    initialization: TreeInitializationConfigV2 = field(default_factory=TreeInitializationConfigV2)

    def __post_init__(self) -> None:
        if self.model_id != 'lightgbm' or self.version != '2':
            raise ContractError('Tree2 requires lightgbm/version2')
        for value, expected in ((self.encoder, TreeEncoderConfigV2), (self.network, TreeNetworkConfigV2),
                                (self.initialization, TreeInitializationConfigV2)):
            if type(value) is not expected:
                raise ContractError('closed typed Tree2 architecture required')


@dataclass(frozen=True)
class ConfiguredTreeModelV2(TreeArchitectureV2):
    checkpoint: str | None = None
    inference: TreeInferenceConfigV2 = field(default_factory=TreeInferenceConfigV2)
    resources: TreeResourcesV2 = field(default_factory=TreeResourcesV2)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.checkpoint is not None and (type(self.checkpoint) is not str or not self.checkpoint):
            raise ContractError('invalid Tree2 checkpoint')
        if type(self.inference) is not TreeInferenceConfigV2 or type(self.resources) is not TreeResourcesV2:
            raise ContractError('closed Tree2 inference/resources required')
        resource_estimate(self)

    def architecture(self) -> TreeArchitectureV2:
        return TreeArchitectureV2(self.model_id, self.version, self.encoder, self.network, self.initialization)


@dataclass(frozen=True)
class TreeDescriptor:
    tree_count: int
    max_depth: int
    model_sha256: str
    model_bytes: int

    def __post_init__(self) -> None:
        integer(self.tree_count, 1, 2**31 - 1)
        integer(self.max_depth, 0, 65535)
        integer(self.model_bytes, 1, MAX_ARRAY_BYTES)
        if type(self.model_sha256) is not str or re.fullmatch('[0-9a-f]{64}', self.model_sha256) is None:
            raise ContractError('Tree2 full model digest required')


def architecture_config(value: Any) -> TreeArchitectureV2:
    raw = closed(value, set(TreeArchitectureV2.__dataclass_fields__))
    return TreeArchitectureV2(raw['model_id'], raw['version'],
        TreeEncoderConfigV2(**closed(raw['encoder'], set(TreeEncoderConfigV2.__dataclass_fields__))),
        TreeNetworkConfigV2(**closed(raw['network'], {'num_leaves'})),
        TreeInitializationConfigV2(**closed(raw['initialization'], {'recipe'})))


def configured_tree_v2(value: Any) -> ConfiguredTreeModelV2:
    raw = closed(value, set(ConfiguredTreeModelV2.__dataclass_fields__))
    architecture = architecture_config({key: raw[key] for key in TreeArchitectureV2.__dataclass_fields__})
    return ConfiguredTreeModelV2(architecture.model_id, architecture.version, architecture.encoder,
        architecture.network, architecture.initialization, raw['checkpoint'],
        TreeInferenceConfigV2(**closed(raw['inference'], set(TreeInferenceConfigV2.__dataclass_fields__))),
        TreeResourcesV2(**closed(raw['resources'], set(TreeResourcesV2.__dataclass_fields__))))


def architecture_identity(model: TreeArchitectureV2) -> str:
    if type(model) is not TreeArchitectureV2:
        raise ContractError('pure Tree2 architecture required')
    return digest({'model': asdict(model), 'numeric_transform': TREE_NUMERICS,
                   'token_signature': signature(model.encoder.aggregate_window)})


def execution_binding(model: ConfiguredTreeModelV2, descriptor: TreeDescriptor, training_binding: Any=None) -> Json:
    value={'inference': asdict(model.inference), 'resources': asdict(model.resources), 'tree_descriptor': asdict(descriptor)}
    if training_binding is not None:
        from gf_models.structured_tree_pairing import TrainingBinding
        if type(training_binding) is not TrainingBinding:raise ContractError('closed trained Tree2 execution binding required')
        value['training_binding']=asdict(training_binding)
    return value


def resource_estimate(model: ConfiguredTreeModelV2, *, model_bytes: int = 0, extra_bytes: int = 0) -> Json:
    integer(model_bytes, 0, MAX_ARRAY_BYTES)
    integer(extra_bytes, 0, 2**63 - 1)
    width, chunk = feature_width(model.encoder.max_positions), model.inference.candidate_chunk
    array_bytes = chunk * width * 4
    fixed = 256 * 1024**2 + model.inference.num_threads * 64 * 1024**2 + 16 * model_bytes
    working = chunk * (12 * width + 32)
    context = (CONTEXT_WIDTH + 18 * CARD_WIDTH) * 4 + 65536
    total = fixed + working + context + extra_bytes
    if array_bytes > model.resources.max_array_bytes or total > model.resources.max_owned_bytes:
        raise ContractError('Tree2 array/joint owned resource admission exceeded')
    return {'feature_width': width, 'array_bytes': array_bytes, 'fixed_bytes': fixed,
            'context_bytes': context, 'working_bytes': working, 'total_bytes': total}


def policy_label(model: ConfiguredTreeModelV2, descriptor: TreeDescriptor, manifest_digest: str, training_binding: Any=None) -> str:
    if type(manifest_digest) is not str or re.fullmatch('[0-9a-f]{64}', manifest_digest) is None:
        raise ContractError('Tree2 full manifest identity required')
    return 'lightgbm-tree-v2-' + manifest_digest + '-' + digest(execution_binding(model, descriptor, training_binding))


def numeric_costs(model: ConfiguredTreeModelV2, descriptor: TreeDescriptor) -> Any:
    from gf_common.diverse_actions import NumericCosts
    estimate = resource_estimate(model, model_bytes=descriptor.model_bytes)
    width = estimate['feature_width']
    return NumericCosts(position_capacity=model.encoder.max_positions, candidate_chunk=model.inference.candidate_chunk,
        fixed_bytes=estimate['fixed_bytes'] + estimate['context_bytes'],
        tail_operations_per_candidate=10 * width + 2 * descriptor.tree_count * (descriptor.max_depth + 1),
        tail_working_bytes_per_candidate=12 * width + 32, tail_tensor_bytes_per_candidate=4 * width,
        max_preparation_bytes=model.resources.max_owned_bytes, max_tensor_bytes=model.resources.max_array_bytes)
