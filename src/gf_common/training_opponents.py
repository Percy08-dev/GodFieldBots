"""Closed opt-in training curriculum identity, never an inference architecture."""
from __future__ import annotations

from dataclasses import asdict

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_rules import card_instance_rules
from gf_common.domain import ContractError, Json, RuntimePolicyBinding, closed
from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label


def parse_training_opponents(value: Json) -> Json:
    raw = closed(value, {'version', 'preset', 'refresh'})
    if raw['version'] != '1' or raw['preset'] != 'mixed-refresh-16-v1':
        raise ContractError('unsupported training_opponents descriptor')
    return {'version': '1', 'preset': 'mixed-refresh-16-v1',
        'refresh': asdict(parse_refresh_options(raw['refresh']))}


def training_opponents(config: Json) -> Json | None:
    """Read the explicit ordinary variant or HPO spec2, without inserting defaults."""
    if 'hpo_spec' in config:
        if 'training_opponents' in config:
            raise ContractError('HPO training_opponents belongs exclusively to spec2')
        spec = config['hpo_spec']
        if type(spec) is not dict or type(spec.get('schema_version')) is not int:
            raise ContractError('explicit HPO spec required')
        if spec['schema_version'] == 1:
            if 'training_opponents' in spec:
                raise ContractError('HPO spec1 forbids training_opponents')
            return None
        if spec['schema_version'] != 2 or 'training_opponents' not in spec:
            raise ContractError('HPO spec2 requires training_opponents')
        result = parse_training_opponents(spec['training_opponents'])
        if spec.get('training_recipe', {}).get('collection_games') != 16:
            raise ContractError('mixed-refresh-16-v1 requires collection_games=16')
        return result
    return parse_training_opponents(config['training_opponents']) if 'training_opponents' in config else None


def refresh_opponent(actions: ActionConfigV3, descriptor: Json) -> tuple[str, RuntimePolicyBinding]:
    options = parse_refresh_options(parse_training_opponents(descriptor)['refresh'])
    label = refresh_policy_label(actions, card_instance_rules(), options)
    return label, RuntimePolicyBinding('heuristic_refresh', asdict(options))
