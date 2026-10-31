"""Closed PPO model identities; safe for parent/source validation without Torch."""
from __future__ import annotations

from gf_common.domain import ContractError, Json

PPO_MODELS = ('mlp', 'gru', 'transformer')


def model_id(value: Json) -> str:
    identity = value.get('model_id')
    if identity not in PPO_MODELS:
        raise ContractError('unsupported PPO model identity')
    return str(identity)


def selection_version(label: str) -> str:
    for identity in PPO_MODELS:
        prefix = identity + '-'
        if label.startswith(prefix):
            checksum = label[len(prefix):]
            # Historical fixtures use abbreviated identities; binding digests are
            # independently checked against immutable generation manifests.
            if checksum:
                return identity + '-ppo-v1-' + checksum
    raise ContractError('unsupported frozen PPO policy label')
