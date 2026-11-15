"""Shared closed-role Collection5 original validator; structural evidence only."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from gf_common.card_instance import ActionConfigV3
from gf_common.card_instance_collection import (
    PPO_BATCH,
    PROJECTION_KEYS,
    PROJECTION_RECIPE,
    GameCollectionRole,
    HpoCollectionRole,
    HpoEvaluationRole,
    checked_hpo_evaluation_manifest,
    checked_invocation,
    checked_linkage,
    checked_role,
)
from gf_common.card_instance_rules import card_instance_rules
from gf_common.card_instance_runtime import validate_runtime_receipt
from gf_common.domain import (
    ContractError,
    Json,
    canonical,
    closed,
    digest,
    group_health_failures,
    integer,
)
from gf_common.refresh_heuristic import parse_refresh_options, refresh_policy_label
from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT as LAYOUT
from gf_common.structured_tokens import signature
from gf_models.card_instance_artifacts import EXTERNAL_INPUT_CONTRACT, validate_neural_metadata
from gf_models.structured_binding import input_binding_game_v3
from gf_models.structured_sealed import checksum
from gf_training.structured_source import _json, _name, _SourceOwner

LINKAGE_KEYS = set(PPO_BATCH.linkage_keys)

_GROUP_ERRORS = {'failure', 'client_close_failure', 'finalization_cancelled', 'cleanup_failure',
    'runtime_evidence_failure', 'storage_validation_failure'}
_EPISODE_ERRORS = {'save_failure', 'finalization_deadline', 'budget_overrun', 'replay_or_finalization_failure',
    'journal_close_failure', 'worker_cleanup_failure', 'delete_failure', 'process_failure',
    'runtime_evidence_failure', 'finalization_cancelled', 'cleanup_failure'}


def _evaluation_raw_health(manifest: Json) -> None:
    """Read actual saved evidence; a success summary cannot override it."""
    groups, receipts = manifest.get('groups'), manifest.get('runtime_evidence')
    if (type(groups) is not list or type(receipts) is not list
            or len(groups) != len(manifest['schedule']) or len(receipts) != len(groups)):
        raise ContractError('complete evaluation group/runtime health correspondence required')
    seen_receipts = set()
    for group, receipt in zip(groups, receipts, strict=True):
        group = closed(group, {'runtime_evidence', 'process'}, _GROUP_ERRORS | {'startup', 'expected_effective',
            'message', 'budget_overrun_seconds'})
        original = validate_runtime_receipt(receipt, require_executed=True)
        validated = validate_runtime_receipt(group['runtime_evidence'], require_executed=True)
        identity = canonical(validated)
        if identity != canonical(original) or identity in seen_receipts:
            raise ContractError('evaluation group differs from its distinct original runtime receipt')
        seen_receipts.add(identity)
        process = closed(group['process'], {'exit_code', 'shutdown', 'log_error'},
            {'reap_failure', 'budget_overrun_seconds'})
        if type(process['exit_code']) is not int or type(process['shutdown']) is not str:
            raise ContractError('actual closed managed process health required')
        for key in ('log_error', 'reap_failure'):
            if process.get(key) is not None:
                raise ContractError('evaluation managed process has log/reap failure evidence')
        for health in (process, group):
            if 'budget_overrun_seconds' in health:
                value = health['budget_overrun_seconds']
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ContractError('finite actual cleanup budget observation required')
        if group_health_failures(group) or any(group.get(key) is not None for key in _GROUP_ERRORS):
            raise ContractError('evaluation has actual group/process failure evidence')
        for key in ('startup', 'expected_effective'):
            if key in group and type(group[key]) is not dict:
                raise ContractError('closed original managed group metadata required')
        if 'message' in group and type(group['message']) is not str:
            raise ContractError('original group diagnostic must be text')
    for episode in manifest['episodes']:
        health = closed(episode.get('health'), set(), _EPISODE_ERRORS)
        if episode.get('failure') is not None or any(value is not None for value in health.values()):
            raise ContractError('natural evaluation episode retains actual failure evidence')


def _evaluation_report_health(manifest: Json, report: Json, games: int) -> None:
    """Cross-check the report ledger against the already checked raw facts."""
    # JSON parse_constant rejects NaN/Infinity tokens, but ordinary exponent
    # parsing can still produce +/-inf. Check every parsed diagnostic value too.
    pending: list[Iterator[object]] = [iter((report,))]
    while pending:
        try:
            value = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if type(value) is float and not math.isfinite(value):
            raise ContractError('nonfinite parsed HPO evaluation report value')
        if isinstance(value, dict):
            pending.append(iter(value.values()))
        elif isinstance(value, list):
            pending.append(iter(value))
    if (type(report) is not dict or type(report.get('schema_version')) is not int or report['schema_version'] != 1
            or report.get('healthy') is not True or report.get('comparison_valid') is not True
            or report.get('report_status') == 'minimal'):
        raise ContractError('complete original evaluation report health ledger required')
    counts = {'planned_games': games, 'recorded_games': len(manifest['episodes']), 'missing_games': 0,
        'natural_denominator': games, 'planned_pairs': len(manifest['schedule']), 'complete_pairs': len(manifest['schedule'])}
    wins = draws = losses = 0
    seats = manifest['linkage']['learner_seats']
    for row in manifest['episodes']:
        if row['outcome'] == 'draw':
            draws += 1
        elif row['outcome']['win']['winner'] == seats[row['episode_id']]:
            wins += 1
        else:
            losses += 1
    counts.update(wins=wins, draws=draws, losses=losses)
    for key, expected in counts.items():
        if integer(report.get(key), 0) != expected:
            raise ContractError('original evaluation report count differs from actual complete panel')
    if (canonical(report.get('end_counts')) != canonical({'natural': games})
            or canonical(report.get('episodes')) != canonical(manifest['episodes'])):
        raise ContractError('original evaluation report episode/outcome ledger differs')
    for key in ('failures', 'health_failures', 'group_health_failures'):
        if type(report.get(key)) is not dict or report[key]:
            raise ContractError('original evaluation report health summary differs from actual healthy evidence')
    score = report.get('paired_score')
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ContractError('finite original evaluation score required')
    if not math.isfinite(score) or score != (wins + .5 * draws) / games:
        raise ContractError('original evaluation score differs from actual learner-seat terminal outcomes')


def inspect_game_manifest(owner: _SourceOwner, manifest: Json, actions: ActionConfigV3,
                   expected_split: str | None, *, role: GameCollectionRole = PPO_BATCH,
                   _pending: object = None) -> tuple[Json, list[tuple[Json, Json]]]:
    role = checked_role(role)
    evaluation = type(role) is HpoEvaluationRole
    hpo = type(role) in (HpoCollectionRole, HpoEvaluationRole)
    if evaluation:
        checked_hpo_evaluation_manifest(manifest)
        _evaluation_raw_health(manifest)
    control_name = 'manifest.json'
    if _pending is not None:
        from gf_common.card_instance_collection import FINAL_VALIDATION
        from gf_training.card_instance_final_validation import _IssuedFinalValidationCollection
        if role is not FINAL_VALIDATION or type(_pending) is not _IssuedFinalValidationCollection:
            raise ContractError('actual final-only pending collection owner required')
        control_name = _pending.manifest_name()
    expected = input_binding_game_v3(actions.action_contract, card_instance_rules())
    resident = (evaluation and 'execution_runtime' in manifest or
        hpo and not evaluation and manifest.get('config', {}).get('hpo_spec', {}).get('schema_version') == 3)
    if resident and not evaluation:
        from gf_training.hpo_collection import check_hpo_collection
        check_hpo_collection(manifest)
    if (manifest.get('schema_version') != (7 if resident else 6 if hpo else 5) or type(manifest['schema_version']) is not int
            or manifest.get('status') != 'complete' or manifest.get('collection_role') != role.role
            or manifest.get('model_version') != ('4' if hpo else '3') or manifest.get('journal_schema_version') != 6
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
    if hpo:
        link = checked_linkage(manifest.get('linkage'), role=role,
            schedule=manifest.get('schedule', []), registry=manifest.get('policy_bindings'))
    elif role is PPO_BATCH:
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
        if binding['model_id'] == 'heuristic_refresh':
            if label != refresh_policy_label(actions, card_instance_rules(), parse_refresh_options(binding['policy_options'])):
                raise ContractError('original configured baseline registry mismatch')
        elif binding['policy_options'] is None:
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
    if evaluation:
        raw_report = owner.read(owner.root / 'report.json')
        if hashlib.sha256(raw_report).hexdigest() != checksum(manifest['report_sha256']):
            raise ContractError('original HPO evaluation report hash differs')
        report = _json(raw_report)
        _evaluation_report_health(manifest, report, role.games)
    return {row['episode_id']: row for row in episodes}, specs

