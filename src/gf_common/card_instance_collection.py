"""Closed corrected collection roles and final-validation records; no authority."""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gf_common.domain import (
    ContractError,
    EpisodeGroup,
    EpisodeSpec,
    Json,
    canonical,
    clone,
    closed,
    digest,
    integer,
)


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


@dataclass(frozen=True)
class HpoEvaluationRole(GameCollectionRole):
    """Closed exploratory matchup role, never a learning or final64 role."""


HPO_EVALUATION_STREAM_VERSION = 'hpo-evaluation-policy-stream-v1'
HPO_EVALUATION_LINKAGE_KEYS = frozenset({'schema_version', 'kind', 'campaign_id', 'evaluation_id',
    'gpu_ticket', 'purpose', 'conditions_sha256', 'left_parent', 'right_parent', 'game_seeds',
    'learner_seats', 'schedule_digest', 'policy_registry_digest'})
HPO_EVALUATION_PARENT_KEYS = {'run_id', 'run_path', 'training_seed', 'generation',
    'generation_sha256', 'inference_sha256', 'source_receipt_sha256', 'policy_label'}
HPO_EVALUATION_CORE_FIELDS = frozenset({'schema_version', 'action_contract', 'rule_signature',
    'layout_signature', 'purpose', 'synthetic_capacity', 'config', 'config_hash', 'action_config_hash',
    'bots_revision', 'simulator_revision', 'runtime_evidence', 'schedule', 'files', 'policies',
    'journal_schema_version', 'derived_schema_version', 'participant_protocol_version',
    'candidate_codec', 'model_version', 'input_recipe', 'token_signature', 'source_eligibility',
    'policy_bindings', 'neural_inputs', 'collection_role', 'linkage'})
HPO_EVALUATION_PUBLISHED_FIELDS = frozenset({'kind', 'run_id', 'command', 'status',
    'schedule_schema_version', 'code', 'inputs', 'groups', 'episodes', 'resource_admission', 'report_sha256'})


def hpo_evaluation_role(games: int) -> HpoEvaluationRole:
    integer(games, 2, 2**31 - 1)
    if games % 2:
        raise ContractError('HPO evaluation requires whole two-seat pairs')
    return HpoEvaluationRole('hpo-evaluation', 'validation', games, HPO_EVALUATION_LINKAGE_KEYS)


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
    if type(role) is HpoEvaluationRole:
        if role != hpo_evaluation_role(role.games):
            raise ContractError('closed HPO evaluation role required')
        return role
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
    if type(role) is HpoEvaluationRole:
        return checked_hpo_evaluation_linkage(value, role=role, schedule=schedule, registry=registry)
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


def _evaluation_text(value: Any) -> str:
    if type(value) is not str or not value or '\x00' in value:
        raise ContractError('nonempty evaluation identity required')
    return value


def checked_hpo_evaluation_parent(value: Any) -> Json:
    raw = closed(value, HPO_EVALUATION_PARENT_KEYS)
    for key in ('run_id', 'run_path', 'policy_label'):
        _evaluation_text(raw[key])
    if not Path(raw['run_path']).is_absolute() or '..' in Path(raw['run_path']).parts:
        raise ContractError('absolute original evaluation parent path required')
    integer(raw['training_seed'], 0, 2**63 - 1)
    integer(raw['generation'], 0, 2**63 - 1)
    for key in ('generation_sha256', 'inference_sha256', 'source_receipt_sha256'):
        _sha(raw[key])
    return raw


def _evaluation_context(value: Any) -> Json:
    raw = closed(value, set(HPO_EVALUATION_LINKAGE_KEYS))
    integer(raw['schema_version'], 1, 1)
    if raw['kind'] != 'hpo-evaluation-linkage' or raw['purpose'] not in ('objective', 'comparison'):
        raise ContractError('explicit exploratory evaluation linkage required')
    for key in ('campaign_id', 'evaluation_id'):
        _evaluation_text(raw[key])
    if type(raw['gpu_ticket']) is not str or re.fullmatch('[0-9a-f]{32}', raw['gpu_ticket']) is None:
        raise ContractError('original GPU admission ticket required')
    for key in ('conditions_sha256', 'schedule_digest', 'policy_registry_digest'):
        _sha(raw[key])
    checked_hpo_evaluation_parent(raw['left_parent'])
    if raw['purpose'] == 'objective':
        if raw['right_parent'] is not None:
            raise ContractError('primary objective requires only the fixed heuristic opponent')
    else:
        checked_hpo_evaluation_parent(raw['right_parent'])
    seeds = raw['game_seeds']
    if type(seeds) is not list or not seeds:
        raise ContractError('explicit nonempty evaluation GameSeed panel required')
    for seed in seeds:
        try:
            if type(seed) is not str or str(uuid.UUID(seed)) != seed:
                raise ValueError
        except ValueError as error:
            raise ContractError('canonical evaluation GameSeed UUID required') from error
    if len(set(seeds)) != len(seeds):
        raise ContractError('duplicate evaluation GameSeed')
    return raw


def hpo_evaluation_groups(linkage: Json) -> tuple[EpisodeGroup, ...]:
    """GameSeeds go only to Simulator groups; policy streams use separate inputs."""
    raw = _evaluation_context(linkage)
    left, right = raw['left_parent'], raw['right_parent']
    opponent = 'heuristic-card-instance-v2' if right is None else right['policy_label']
    groups = []
    for index, seed in enumerate(raw['game_seeds']):
        pair = f'eval-p{index:06}'
        streams = tuple(int(digest({'recipe': HPO_EVALUATION_STREAM_VERSION,
            'training_seed': (left if side == 'left' or right is None else right)['training_seed'],
            'panel_index': index, 'side': side})[:16], 16) for side in ('left', 'right'))
        paired = (streams[0], streams[1])
        groups.append(EpisodeGroup(seed, (
            EpisodeSpec(pair + '-seat0', (left['policy_label'], opponent), paired, pair, 1),
            EpisodeSpec(pair + '-seat1', (opponent, left['policy_label']), paired[::-1], pair, 2))))
    return tuple(groups)


def checked_hpo_evaluation_linkage(value: Any, *, role: HpoEvaluationRole, schedule: list[Json], registry: Any) -> Json:
    from gf_common.schedule import serialize_schedule
    raw = _evaluation_context(value)
    if checked_role(role).games != 2 * len(raw['game_seeds']):
        raise ContractError('evaluation role differs from complete two-seat panel')
    expected = clone(serialize_schedule(hpo_evaluation_groups(raw), 4))
    if (canonical(schedule) != canonical(expected) or raw['schedule_digest'] != digest(schedule)
            or raw['policy_registry_digest'] != digest(registry)):
        raise ContractError('evaluation original schedule/streams/registry differ')
    seats = {spec['episode_id']: spec['learner_viewer'] for group in expected for spec in group['episodes']}
    if type(raw['learner_seats']) is not dict or canonical(raw['learner_seats']) != canonical(seats):
        raise ContractError('evaluation LEFT learner seats differ from the complete panel')
    parents = [raw['left_parent']] + ([] if raw['right_parent'] is None else [raw['right_parent']])
    labels = {parent['policy_label'] for parent in parents}
    if raw['right_parent'] is None:
        labels.add('heuristic-card-instance-v2')
    if type(registry) is not dict or set(registry) != labels:
        raise ContractError('evaluation requires the exact frozen policy registry')
    for parent in parents:
        binding = closed(registry[parent['policy_label']], {'model_id', 'policy_options'})
        options = binding['policy_options']
        if (binding['model_id'] not in ('mlp', 'gru', 'transformer') or type(options) is not dict
                or options.get('version') != '4' or options.get('expected_manifest_digest') != parent['inference_sha256']):
            raise ContractError('evaluation parent differs from explicit model4 inference identity')
    if raw['right_parent'] is None and registry['heuristic-card-instance-v2'] != {'model_id': 'heuristic', 'policy_options': None}:
        raise ContractError('evaluation requires the canonical fixed heuristic')
    return raw


def _checked_evaluation_metadata(value: Json) -> Json:
    integer(value['schema_version'], 6, 6)
    if (value['collection_role'] != 'hpo-evaluation' or value['model_version'] != '4'
            or value['source_eligibility'] != 'not-authorized-by-collection-metadata'
            or value['config_hash'] != digest(value['config']) or value['config'].get('run', {}).get('split') != 'validation'
            or value['journal_schema_version'] != 6 or value['derived_schema_version'] != 6
            or value['participant_protocol_version'] != 3 or value['candidate_codec'] != 'candidate-columns-v2'):
        raise ContractError('explicit non-authorizing Eval6/model4 metadata required')
    raw = _evaluation_context(value['linkage'])
    checked_linkage(raw, role=hpo_evaluation_role(2 * len(raw['game_seeds'])),
        schedule=value['schedule'], registry=value['policy_bindings'])
    return value


def checked_hpo_evaluation_metadata(value: Json) -> Json:
    return _checked_evaluation_metadata(closed(value, set(HPO_EVALUATION_CORE_FIELDS)))


def checked_hpo_evaluation_manifest(value: Json) -> Json:
    from gf_common.validation import result
    value = _checked_evaluation_metadata(closed(value, set(HPO_EVALUATION_CORE_FIELDS | HPO_EVALUATION_PUBLISHED_FIELDS)))
    _sha(value['report_sha256'])
    if (value['kind'] != 'card-instance-collection' or value['command'] != 'evaluate' or value['status'] != 'complete'
            or value['run_id'] != value['linkage']['evaluation_id'] or value['schedule_schema_version'] != 4):
        raise ContractError('complete published Eval6 identity required')
    expected = {spec['episode_id']: (group['seed'], spec) for group in value['schedule'] for spec in group['episodes']}
    rows = value['episodes']
    if type(rows) is not list or len(rows) != len(expected):
        raise ContractError('complete actual evaluation game rows required')
    seen: set[str] = set()
    game_ids: set[str] = set()
    for row in rows:
        if type(row) is not dict or row.get('episode_id') not in expected or row['episode_id'] in seen:
            raise ContractError('extra/duplicate evaluation episode')
        seed, spec = expected[row['episode_id']]
        game = _evaluation_text(row.get('game_id'))
        if (game in game_ids or row.get('seed') != seed or row.get('pair_id') != spec['pair_id']
                or row.get('policies') != spec['policies'] or row.get('end_kind') != 'natural'
                or row.get('failure') is not None or type(row.get('health')) is not dict
                or any(row['health'].values()) or row.get('replay_reconciled') is not True):
            raise ContractError('non-natural/unhealthy/mismatched evaluation original')
        result(row.get('outcome'))
        if row.get('outcome') is None:
            raise ContractError('natural evaluation terminal result required')
        integer(row.get('decisions'), 0)
        seen.add(row['episode_id'])
        game_ids.add(game)
    return value


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
