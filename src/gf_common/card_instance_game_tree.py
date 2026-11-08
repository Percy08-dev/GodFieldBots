"""Closed real-game Dataset3/fit records; serialized facts grant no permission."""
from __future__ import annotations

import math
import re
from pathlib import PurePosixPath
from typing import Any

from gf_common.card_instance import corrected_card_instance_contract
from gf_common.card_instance_rules import card_instance_rules, rule_signature
from gf_common.card_instance_runtime import game_binding
from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer

SOURCE_KIND = 'card-instance-formal-ppo-originals'
KIND = 'card-instance-game-fixed-sources'
SELECTION = 'all-explicit-learner-accepted-natural-return-v3'
LABEL = 'behavior-selected-natural-return-gamma1-v1'
SHARED_SCOPE = 'whole PPO parents plus final validation once; excludes per-fit preparation/fitting'
MAX_OWNED_BYTES = 8 * 1024**3
SEEDS = (17, 29, 43)
FIELDS = {'schema_version', 'kind', 'selection', 'label_recipe', 'source_kind', 'source_origin',
    'input_binding', 'input_recipe', 'tree_projection', 'neural_cohort', 'parents', 'collections',
    'counts', 'source_set_sha256', 'shared_generation', 'content_sha256'}
PARENT_FIELDS = {'parent_index', 'directory', 'run_id', 'seed', 'run_sha256', 'formal_declaration_sha256',
    'config_sha256', 'config_inputs_sha256', 'resume_identity', 'registry_sha256', 'final_generation',
    'final_generation_sha256', 'final_inference_sha256', 'generations', 'elapsed_seconds'}
GENERATION_FIELDS = {'generation', 'generation_sha256', 'update_sha256', 'inference_sha256',
    'resume_sha256', 'artifact_files_sha256'}
COLLECTION_FIELDS = {'parent_index', 'directory', 'collection_role', 'split', 'generation',
    'manifest_sha256', 'source_set_sha256', 'schedule_sha256', 'bindings_sha256',
    'original_correspondence', 'numeric', 'selected_roles', 'counts', 'elapsed_seconds'}
COUNT_FIELDS = {'games', 'participants', 'selected_decisions', 'zero_endpoint_games'}
MEMORY_FIELDS = {'recipe', 'total_bytes', 'controller_reserved_bytes', 'child_limit_bytes', 'reserved_peak_bytes'}


def sha256(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError('game Tree exact SHA256 required')
    return value


def origin() -> Json:
    return {'purpose': 'qualified_game', 'synthetic_capacity': False,
        'outcome_origin': 'observed-natural-simulator',
        'qualification_recipe': 'three-formal-parents-final64-v1'}


def input_binding() -> Json:
    contract = corrected_card_instance_contract()
    return game_binding(contract, rule_signature(contract, card_instance_rules()))


def checked_source(raw: Json) -> None:
    if raw.get('source_kind') != SOURCE_KIND or canonical(raw.get('source_origin')) != canonical(origin()):
        raise ContractError('exact observed formal game Tree source origin required')


def seconds(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ContractError('finite original game Tree cost required')
    return float(value)


def counts(value: Any, *, games: int | None = None) -> Json:
    raw = closed(value, COUNT_FIELDS)
    for number in raw.values():
        integer(number, 0, 2**63 - 1)
    if (games is not None and raw['games'] != games or raw['participants'] != 2 * raw['games']
            or raw['zero_endpoint_games'] > raw['games']):
        raise ContractError('complete game Tree population counts differ')
    return raw


def final_correspondence(manifest_sha256: str, episode_digests: tuple[str, ...] | list[str]) -> Json:
    sha256(manifest_sha256)
    if len(episode_digests) != 64:
        raise ContractError('final game Tree correspondence requires64 selected episodes')
    for episode_digest in episode_digests:
        sha256(episode_digest)
    value = {'recipe': 'card-instance-final-validation-correspondence-v1', 'input_binding': input_binding(),
        'manifest_sha256': manifest_sha256, 'episode_count': 64, 'episode_digests': list(episode_digests)}
    return {**value, 'binding_sha256': digest(value)}


def checked_final_correspondence(value: Any, *, manifest_sha256: str) -> Json:
    raw = closed(value, {'recipe', 'input_binding', 'manifest_sha256', 'episode_count', 'episode_digests', 'binding_sha256'})
    if type(raw['episode_count']) is not int or raw['episode_count'] != 64 or type(raw['episode_digests']) is not list:
        raise ContractError('closed final64 original correspondence required')
    expected = final_correspondence(manifest_sha256, raw['episode_digests'])
    if canonical(raw) != canonical(expected):
        raise ContractError('final64 original correspondence binding differs')
    return raw


def managed_memory(total_bytes: int, controller_reserved_bytes: int, child_limit_bytes: int) -> Json:
    integer(total_bytes, 1, MAX_OWNED_BYTES)
    integer(controller_reserved_bytes, 0, total_bytes)
    integer(child_limit_bytes, 1, total_bytes)
    peak = controller_reserved_bytes + child_limit_bytes
    if peak > total_bytes:
        raise ContractError('full configured child and ALL controller owners exceed joint8GiB')
    return {'recipe': 'joint-game-tree-fit-v1', 'total_bytes': total_bytes,
        'controller_reserved_bytes': controller_reserved_bytes, 'child_limit_bytes': child_limit_bytes,
        'reserved_peak_bytes': peak}


def checked_memory(value: Any, *, child_limit_bytes: int | None = None) -> Json:
    raw = closed(value, MEMORY_FIELDS)
    expected = managed_memory(raw['total_bytes'], raw['controller_reserved_bytes'], raw['child_limit_bytes'])
    if canonical(raw) != canonical(expected) or child_limit_bytes is not None and raw['child_limit_bytes'] != child_limit_bytes:
        raise ContractError('game Tree immutable joint/child resource envelope differs')
    return raw


def shared_generation(parents: list[Json], collections: list[Json]) -> Json:
    validations = [row for row in collections if row['collection_role'] == 'ppo-final-validation']
    if len(parents) != 3 or len(validations) != 3:
        raise ContractError('three original parents/final validations required for shared cost')
    total = sum(seconds(row['elapsed_seconds']) for row in parents) + sum(seconds(row['elapsed_seconds']) for row in validations)
    key = digest({'parents': [{name: row[name] for name in ('run_id', 'run_sha256', 'final_generation_sha256')} for row in parents],
        'validation': [{name: row[name] for name in ('parent_index', 'manifest_sha256', 'source_set_sha256')} for row in validations]})
    return {'seconds': total, 'cost_key': key, 'scope': SHARED_SCOPE}


def checked_shared(value: Any) -> Json:
    raw = closed(value, {'seconds', 'cost_key', 'scope'})
    seconds(raw['seconds']); sha256(raw['cost_key'])
    if raw['scope'] != SHARED_SCOPE:
        raise ContractError('game Tree shared generation scope differs')
    return raw


def normalized_formal_config(value: Json) -> Json:
    """Comparison only; never a materialized runnable Config or Source proof."""
    result = clone(value)
    for table, key in (('run', 'name'), ('run', 'output_dir'), ('server', 'source_dir'),
            ('server', 'binary'), ('server', 'config'), ('training', 'seeds')):
        if table not in result or key not in result[table]:
            raise ContractError('full original formal Config required for homogeneity comparison')
        result[table][key] = None
    return result


def checked_training_correspondence(value: Any) -> Json:
    raw = closed(value, {'recipe', 'control_fingerprint', 'semantics_sha256', 'episode_count',
        'episode_digests', 'binding_sha256'})
    if (raw['recipe'] != 'structured-source-cohort-correspondence-v2'
            or type(raw['episode_count']) is not int or raw['episode_count'] != 16
            or type(raw['episode_digests']) not in (list, tuple) or len(raw['episode_digests']) != 16):
        raise ContractError('complete16 original training correspondence required')
    for name in ('control_fingerprint', 'semantics_sha256', 'binding_sha256'):
        sha256(raw[name])
    for item in raw['episode_digests']:
        sha256(item)
    if digest({key: item for key, item in raw.items() if key != 'binding_sha256'}) != raw['binding_sha256']:
        raise ContractError('original training correspondence binding differs')
    return raw


def _directory(value: Any) -> None:
    if type(value) is not str or not value or not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
        raise ContractError('explicit absolute original game directory required')


def source_identity(parents: list[Json], collections: list[Json]) -> str:
    return digest({'recipe': 'card-instance-game-dataset-originals-v1', 'parents': parents, 'collections': collections})


def manifest(value: Any) -> Json:
    """Validate recipe/index structure only; this cannot qualify any originals."""
    raw = closed(value, FIELDS)
    checked_source(raw)
    if (type(raw['schema_version']) is not int or raw['schema_version'] != 3 or raw['kind'] != KIND
            or raw['selection'] != SELECTION or raw['label_recipe'] != LABEL
            or canonical(raw['input_binding']) != canonical(input_binding())
            or raw['input_recipe'] != corrected_card_instance_contract().input_recipe):
        raise ContractError('closed observed game Dataset3 required')
    closed(raw['tree_projection'], {'encoder', 'feature_layout', 'feature_signature', 'token_signature'})
    for name in ('feature_signature', 'token_signature'):
        sha256(raw['tree_projection'][name])
    closed(raw['neural_cohort'], {'model', 'actions_sha256', 'execution', 'semantics', 'recipe_id',
        'recipe_sha256', 'algorithm_recipe_sha256', 'code'})
    for name in ('actions_sha256', 'recipe_sha256', 'algorithm_recipe_sha256'):
        sha256(raw['neural_cohort'][name])
    parents, collections = raw['parents'], raw['collections']
    if type(parents) is not list or len(parents) != 3 or type(collections) is not list:
        raise ContractError('all three ordered formal parent originals required')
    identities = set()
    for index, (parent, seed) in enumerate(zip(parents, SEEDS, strict=True)):
        closed(parent, PARENT_FIELDS)
        _directory(parent['directory'])
        if type(parent['parent_index']) is not int or parent['parent_index'] != index or parent['seed'] != seed:
            raise ContractError('game Dataset3 exact ordered17/29/43 slots required')
        for name in ('run_sha256', 'formal_declaration_sha256', 'config_sha256', 'config_inputs_sha256',
                'resume_identity', 'registry_sha256', 'final_generation_sha256', 'final_inference_sha256'):
            sha256(parent[name])
        identity = parent['run_id'], parent['run_sha256']
        if identity in identities or type(parent['run_id']) is not str or not parent['run_id']:
            raise ContractError('distinct original formal parents required')
        identities.add(identity)
        end = integer(parent['final_generation'], 1, 1000)
        if type(parent['generations']) is not list or len(parent['generations']) != end + 1:
            raise ContractError('all original generations includingG0 required')
        for generation, record in enumerate(parent['generations']):
            closed(record, GENERATION_FIELDS)
            if type(record['generation']) is not int or record['generation'] != generation:
                raise ContractError('original game generation order differs')
            for name in GENERATION_FIELDS - {'generation', 'update_sha256'}:
                sha256(record[name])
            if generation == 0:
                if record['update_sha256'] is not None:
                    raise ContractError('originalG0 has no update')
            else:
                sha256(record['update_sha256'])
        seconds(parent['elapsed_seconds'])
    # Work order is each parent's complete training then its final validation.
    expected: list[tuple[int, str, int]] = []
    for i, parent in enumerate(parents):
        expected.extend((i, 'train', generation) for generation in range(parent['final_generation']))
        expected.append((i, 'validation', parent['final_generation']))
    if len(collections) != len(expected):
        raise ContractError('all committed training and three final64 memberships required')
    totals = {split: {name: 0 for name in COUNT_FIELDS} for split in ('train', 'validation')}
    for row, (index, split, generation) in zip(collections, expected, strict=True):
        closed(row, COLLECTION_FIELDS)
        _directory(row['directory'])
        integer(row['parent_index'],0,2); integer(row['generation'],0,1000)
        if (row['parent_index'] != index or row['split'] != split or row['generation'] != generation
                or row['collection_role'] != ('ppo-batch' if split == 'train' else 'ppo-final-validation')):
            raise ContractError('game Dataset3 original cohort order/roles differ')
        for name in ('manifest_sha256', 'source_set_sha256', 'schedule_sha256', 'bindings_sha256'):
            sha256(row[name])
        count = counts(row['counts'], games=16 if split == 'train' else 64)
        roles = row['selected_roles']
        if type(roles) is not list or len(roles) != count['games']:
            raise ContractError('explicit selected original role for every natural game required')
        endpoints = zero = 0
        names = set()
        for role in roles:
            closed(role, {'episode_id', 'viewer', 'policy_label', 'end_kind', 'outcome', 'accepted_count'})
            integer(role['viewer'], 1, 2); integer(role['accepted_count'], 0, 2**63-1)
            from gf_common.validation import result as validate_result
            validate_result(role['outcome'])
            if (type(role['episode_id']) is not str or not role['episode_id']
                    or type(role['policy_label']) is not str or not role['policy_label']
                    or role['end_kind'] != 'natural' or role['outcome'] is None or role['episode_id'] in names):
                raise ContractError('complete distinct observed natural roles required')
            names.add(role['episode_id']); endpoints += role['accepted_count']; zero += role['accepted_count'] == 0
        if endpoints != count['selected_decisions'] or zero != count['zero_endpoint_games']:
            raise ContractError('game Dataset3 endpoint/zero ledger differs')
        if split == 'validation':
            if row['numeric'] is not None:
                raise ContractError('final64 cannot fabricate a numeric update')
            checked_final_correspondence(row['original_correspondence'], manifest_sha256=row['manifest_sha256'])
        else:
            checked_training_correspondence(row['original_correspondence'])
            closed(row['numeric'], {'directory', 'sha256', 'files'}); sha256(row['numeric']['sha256'])
            _directory(row['numeric']['directory'])
        seconds(row['elapsed_seconds'])
        for name in COUNT_FIELDS:
            totals[split][name] += count[name]
    if canonical(raw['counts']) != canonical(totals) or totals['validation']['games'] != 192:
        raise ContractError('complete game Dataset3 split populations differ')
    if raw['source_set_sha256'] != source_identity(parents, collections):
        raise ContractError('game Dataset3 original source membership digest differs')
    if canonical(raw['shared_generation']) != canonical(shared_generation(parents, collections)):
        raise ContractError('game Dataset3 original once-only generation cost differs')
    if raw['content_sha256'] != digest({name: item for name, item in raw.items() if name != 'content_sha256'}):
        raise ContractError('game Dataset3 content digest differs')
    return clone(raw)
