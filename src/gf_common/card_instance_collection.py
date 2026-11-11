"""Closed corrected collection roles and final-validation records; no authority."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import ContractError, Json, canonical, clone, closed, digest, integer


@dataclass(frozen=True)
class GameCollectionRole:
    role: str
    split: str
    games: int
    linkage_keys: frozenset[str]


PPO_BATCH = GameCollectionRole('ppo-batch', 'train', 16, frozenset({
    'parent_run_id', 'parent_run_path', 'batch_id', 'collection_generation',
    'learner_seats', 'schedule_digest', 'policy_registry_digest'}))
FINAL_VALIDATION = GameCollectionRole('ppo-final-validation', 'validation', 64, frozenset({
    'parent_run_id', 'parent_run_path', 'batch_id', 'final_generation', 'final_inference_sha256',
    'policy_label', 'learner_seats', 'schedule_digest', 'policy_registry_digest'}))


@dataclass(frozen=True)
class HpoCollectionRole(GameCollectionRole):
    """Explicit variable-size exploratory role; no formal/final authority."""


def hpo_batch_role(games: int) -> HpoCollectionRole:
    integer(games, 8, 65536)
    if games % 8:
        raise ContractError('HPO collection games must preserve four opponent pairs')
    return HpoCollectionRole('hpo-ppo-batch', 'train', games,
        PPO_BATCH.linkage_keys | {'selection_context'})
PROJECTION_KEYS = {'recipe', 'parent_config_sha256', 'parent_config_inputs_sha256', 'projected_config_sha256'}
INVOCATION_KEYS = {'started_unix', 'started_monotonic', 'caller_deadline_monotonic',
    'admission_deadline_monotonic', 'execution_deadline_monotonic', 'wall_deadline_monotonic'}
PROJECTION_RECIPE = 'run-split-train-to-validation-v1'
NATIVE_FINAL_VALIDATION_MARKER = 'collection5-final64;journal6;codec2'
NATIVE_PACKED_FINAL_VALIDATION_MARKER = 'final64-packed;provider1;request2;summary2;lease1'
FINAL_FACTS_FIELDS = {'directory', 'manifest_sha256', 'parent_run_id', 'final_generation',
    'final_inference_sha256', 'config_projection', 'validation_invocation', 'elapsed_seconds',
    'games', 'selected_decisions', 'zero_endpoint_games', 'qualified', 'parent_qualification'}
PACKED_FINAL_FACTS_FIELDS = FINAL_FACTS_FIELDS | {'schema_version', 'kind', 'storage_binding',
    'storage', 'custody', 'source_set_sha256'}


def packed_final_facts(value: Json, *, storage: Json, custody: Json, source_set_sha256: str) -> Json:
    """Construct closed values only; original owners retain all admission rights."""
    from gf_common.group_pack_game import checked_storage_custody, storage_binding
    raw = closed(value, FINAL_FACTS_FIELDS)
    checked_storage_custody(storage, custody)
    _sha(source_set_sha256)
    if raw['qualified'] is not False or raw['parent_qualification'] != 'not_performed':
        raise ContractError('packed final facts cannot grant qualification')
    if type(raw['games']) is not int or raw['games'] != 64:
        raise ContractError('all64 original final endpoints required')
    integer(raw['selected_decisions'], 0)
    integer(raw['zero_endpoint_games'], 0, 64)
    return clone({**raw, 'schema_version': 2, 'kind': 'card-instance-packed-final-validation-facts',
        'storage_binding': storage_binding(), 'storage': storage, 'custody': custody,
        'source_set_sha256': source_set_sha256})



def checked_role(role: GameCollectionRole) -> GameCollectionRole:
    if type(role) is HpoCollectionRole:
        if role != hpo_batch_role(role.games):
            raise ContractError('closed HPO collection role required')
        return role
    if role is not PPO_BATCH and role is not FINAL_VALIDATION:
        raise ContractError('one explicit closed corrected collection role required')
    return role


def _sha(value: Any) -> str:
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError('original collection SHA256 required')
    return value


def checked_linkage(value: Any, *, role: GameCollectionRole, schedule: list[Json], registry: Any) -> Json:
    role = checked_role(role)
    raw = closed(value, set(role.linkage_keys))
    for key in ('parent_run_id', 'parent_run_path', 'batch_id'):
        if type(raw[key]) is not str or not raw[key]:
            raise ContractError('original collection requires explicit parent/batch linkage')
    if not Path(raw['parent_run_path']).is_absolute():
        raise ContractError('original collection parent path must be absolute')
    if role is PPO_BATCH or type(role) is HpoCollectionRole:
        generation = integer(raw['collection_generation'], 0, 2**63 - 1 if type(role) is HpoCollectionRole else 999)
        if raw['batch_id'] != f'batch-{generation:06}':
            raise ContractError('original PPO batch generation mismatch')
        if type(role) is HpoCollectionRole:
            from gf_common.hpo import checked_context
            if checked_context(raw['selection_context'])['batch'] != generation:
                raise ContractError('HPO selection batch differs from original linkage')
    else:
        integer(raw['final_generation'], 0, 1000)
        _sha(raw['final_inference_sha256'])
        if raw['batch_id'] != 'validation-final' or type(raw['policy_label']) is not str or not raw['policy_label']:
            raise ContractError('original final validation policy/batch mismatch')
    seats = {}
    for group in schedule:
        for spec in group['episodes']:
            name = spec['episode_id']
            if name in seats:
                raise ContractError('duplicate original collection episode')
            seats[name] = integer(spec.get('learner_viewer'), 1, 2)
    if (len(seats) != role.games or type(raw['learner_seats']) is not dict
            or canonical(raw['learner_seats']) != canonical(seats)
            or raw['schedule_digest'] != digest(schedule) or raw['policy_registry_digest'] != digest(registry)):
        raise ContractError('whole original collection roles/schedule/registry required')
    return raw


def config_projection(parent: Json, inputs: Any, projected: Json) -> Json:
    expected = {**parent, 'run': {**parent['run'], 'split': 'validation'}}
    if parent['run']['split'] != 'train' or canonical(projected) != canonical(expected):
        raise ContractError('validation Config may project ONLY original run.split train to validation')
    return {'recipe': PROJECTION_RECIPE, 'parent_config_sha256': digest(parent),
        'parent_config_inputs_sha256': digest(inputs), 'projected_config_sha256': digest(projected)}


def checked_projection(value: Any, *, parent: Json, inputs: Any, projected: Json) -> Json:
    raw = closed(value, PROJECTION_KEYS)
    if canonical(raw) != canonical(config_projection(parent, inputs, projected)):
        raise ContractError('validation effective Config projection/original byte origins differ')
    return raw


def validation_invocation(*, started_unix: float, started_monotonic: float,
                          caller_deadline_monotonic: float, run: Json) -> Json:
    values = (started_unix, started_monotonic, caller_deadline_monotonic)
    if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in values):
        raise ContractError('finite final validation invocation anchors required')
    if tuple(run[key] for key in ('wall_seconds', 'admission_seconds', 'execution_seconds', 'finalization_seconds')) != (7200, 5400, 7140, 60):
        raise ContractError('final validation requires fresh literal7200/5400/7140/60 budgets')
    wall = min(caller_deadline_monotonic, started_monotonic + run['wall_seconds'])
    execution = min(started_monotonic + run['execution_seconds'], wall - run['finalization_seconds'])
    if execution <= started_monotonic:
        raise ContractError('caller final validation duration cannot preserve finalization reserve')
    return {'started_unix': started_unix, 'started_monotonic': started_monotonic,
        'caller_deadline_monotonic': caller_deadline_monotonic,
        'admission_deadline_monotonic': min(started_monotonic + run['admission_seconds'], execution),
        'execution_deadline_monotonic': execution, 'wall_deadline_monotonic': wall}


def checked_invocation(value: Any, *, run: Json) -> Json:
    raw = closed(value, INVOCATION_KEYS)
    expected = validation_invocation(started_unix=raw['started_unix'], started_monotonic=raw['started_monotonic'],
        caller_deadline_monotonic=raw['caller_deadline_monotonic'], run=run)
    if canonical(raw) != canonical(expected):
        raise ContractError('final validation invocation deadlines differ from original fresh capped budgets')
    return raw
