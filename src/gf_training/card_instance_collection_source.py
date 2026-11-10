"""Shared closed-role Collection5 original validator; structural evidence only."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import (
    PPO_BATCH,
    PROJECTION_KEYS,
    PROJECTION_RECIPE,
    GameCollectionRole,
    checked_invocation,
    checked_linkage,
    checked_role,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_runtime import validate_runtime_receipt
from gf_common.domain import ContractError, Json, canonical, closed, digest, integer
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT, validate_neural_metadata
from gf_models.structured_binding import input_binding_game_v3
from gf_models.structured_sealed import checksum
from gf_training.structured_source import _name, _SourceOwner

LINKAGE_KEYS = set(PPO_BATCH.linkage_keys)


def inspect_game_manifest(owner: _SourceOwner, manifest: Json, actions: ActionConfigV3,
                   expected_split: str | None, *, role: GameCollectionRole = PPO_BATCH,
                   _pending: object = None) -> tuple[Json, list[tuple[Json, Json]]]:
    role = checked_role(role)
    control_name = 'manifest.json'
    if _pending is not None:
        from gf_common.card_instance_collection import FINAL_VALIDATION
        from gf_training.card_instance_final_validation import _IssuedFinalValidationCollection
        if role is not FINAL_VALIDATION or type(_pending) is not _IssuedFinalValidationCollection:
            raise ContractError('actual final-only pending collection owner required')
        control_name = _pending.manifest_name()
    expected = input_binding_game_v3(actions.action_contract, card_instance_rules())
    if (manifest.get('schema_version') != 5 or type(manifest['schema_version']) is not int
            or manifest.get('status') != 'complete' or manifest.get('collection_role') != role.role
            or manifest.get('model_version') != '3' or manifest.get('journal_schema_version') != 6
            or manifest.get('derived_schema_version') != 6 or manifest.get('participant_protocol_version') != 3
            or manifest.get('candidate_codec') != 'candidate-columns-v2'
            or manifest.get('input_recipe') != LAYOUT.input_recipe
            or manifest.get('token_signature') != signature(128, layout=LAYOUT)
            or manifest.get('action_config_hash') != digest(asdict(actions))
            or any(canonical(manifest.get(key)) != canonical(value) for key, value in expected.items())):
        raise ContractError('complete corrected PPO collection5 context required')
    config = manifest.get('config')
    if (type(config) is not dict or digest(config) != manifest.get('config_hash')
            or canonical(config.get('actions')) != canonical(asdict(actions))
            or config.get('run', {}).get('split') != expected_split):
        raise ContractError('original collection5 frozen Config/split mismatch')
    if role is PPO_BATCH:
        link = closed(manifest.get('linkage'), LINKAGE_KEYS)
        integer(link['collection_generation'], 0, 999)
        if link['batch_id'] != f"batch-{link['collection_generation']:06}" or not Path(link['parent_run_path']).is_absolute():
            raise ContractError('original PPO batch linkage mismatch')
    else:
        link = checked_linkage(manifest.get('linkage'), role=role,
            schedule=manifest.get('schedule', []), registry=manifest.get('policy_bindings'))
        checked_invocation(manifest.get('validation_invocation'), run=config['run'])
        projection = closed(manifest.get('config_projection'), PROJECTION_KEYS)
        if (projection['recipe'] != PROJECTION_RECIPE or projection['projected_config_sha256'] != manifest['config_hash']
                or type(manifest.get('elapsed_seconds')) not in (float, int)
                or not math.isfinite(manifest['elapsed_seconds']) or manifest['elapsed_seconds'] < 0):
            raise ContractError('original final validation projection/cost required')
    specs = [(group, spec) for group in manifest.get('schedule', []) for spec in group.get('episodes', [])]
    names = [_name(spec.get('episode_id')) for _, spec in specs]
    episodes = manifest.get('episodes')
    if (len(specs) != role.games or len(set(names)) != role.games or type(episodes) is not list or len(episodes) != role.games
            or {row.get('episode_id') for row in episodes} != set(names)
            or len({row.get('game_id') for row in episodes}) != role.games
            or len(manifest.get('groups', [])) != len(manifest['schedule'])
            or len(manifest.get('runtime_evidence', [])) != len(manifest['schedule'])
            or link['learner_seats'] != {spec['episode_id']: integer(spec.get('learner_viewer'), 1, 2) for _, spec in specs}
            or link['schedule_digest'] != digest(manifest['schedule'])
            or link['policy_registry_digest'] != digest(manifest.get('policy_bindings'))):
        raise ContractError('whole16 original collection roles/schedule/registry required')
    registry = manifest.get('policy_bindings')
    neural = closed(manifest.get('neural_inputs'), {'input_contract', 'policies'})
    if (type(registry) is not dict or set(manifest.get('policies', [])) != set(registry)
            or neural['input_contract'] != EXTERNAL_INPUT_CONTRACT or type(neural['policies']) is not dict):
        raise ContractError('complete original neural registry metadata required')
    expected_neural = set()
    for label, binding in registry.items():
        closed(binding, {'model_id', 'policy_options'})
        if binding['policy_options'] is None:
            if binding['model_id'] not in ('random', 'heuristic') or label != binding['model_id'] + '-card-instance-v2':
                raise ContractError('original baseline registry binding mismatch')
        else:
            expected_neural.add(label)
            raw = neural['policies'].get(label)
            if not isinstance(raw, dict):
                raise ContractError('missing original neural registry metadata')
            safe, expected_label = validate_neural_metadata(raw, actions=actions, rules=card_instance_rules())
            if expected_label != label or canonical(safe) != canonical(binding['policy_options']) or safe['model']['model_id'] != binding['model_id']:
                raise ContractError('original neural registry binding mismatch')
    if set(neural['policies']) != expected_neural:
        raise ContractError('extra/missing original neural registry metadata')
    for group, receipt in zip(manifest['groups'], manifest['runtime_evidence'], strict=True):
        validate_runtime_receipt(receipt, require_executed=True)
        if group.get('runtime_evidence_failure') or canonical(group.get('runtime_evidence')) != canonical(receipt):
            raise ContractError('original group/runtime evidence mismatch')
    for _, spec in specs:
        streams = spec.get('sampling_seeds')
        if type(streams) is not list or len(streams) != 2:
            raise ContractError('original participant sampling streams required')
        for seed in streams:
            integer(seed, 0, 2**64 - 1)
    files = manifest.get('files')
    if type(files) is not list:
        raise ContractError('original complete inventory required')
    from gf_training.group_pack_game_source import PackedSourceOwner
    expected_names = set()
    for entry in files:
        closed(entry, {'path', 'sha256', 'bytes'})
        path = Path(entry['path'])
        if path.is_absolute() or any(part in ('.', '..') for part in path.parts) or not path.parts:
            raise ContractError('foreign original inventory path')
        name = path.as_posix()
        if name in expected_names or name in ('manifest.json', 'report.json', control_name):
            raise ContractError('duplicate/foreign original inventory entry')
        expected_names.add(name)
        # Constructor validation has already verified every packed original
        # byte/member. Opaque diagnostics need admitted evidence, not JSON decode.
        if type(owner) is PackedSourceOwner:
            owner.remember_inventory_entry(name, entry['bytes'], checksum(entry['sha256']))
        else:
            owner.read(owner.root / path)
            if owner.files[name]['sha256'] != checksum(entry['sha256']) or owner.files[name]['bytes'] != entry['bytes']:
                raise ContractError('original collection inventory hash/size mismatch')
    if type(owner) is PackedSourceOwner:
        if config.get('storage', {}).get('format') != 'sqlite-group-pack-v1':
            raise ContractError('packed original requires explicit original storage Config')
        owner.validate_index(manifest)
    elif 'storage' in config:
        raise ContractError('directory original cannot reinterpret packed Config')
    actual = owner.inventory('**/*')
    if set(actual) != expected_names | {control_name, 'report.json'}:
        raise ContractError('complete original collection inventory changed')
    return {row['episode_id']: row for row in episodes}, specs

