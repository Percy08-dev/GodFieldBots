"""Closed Rust-player lineage data and actual local publication transactions.

Metadata functions below grant no capture, admission, update or parent authority.
Historical record kinds/readers remain separate.
"""

from __future__ import annotations

from typing import Any

from gf_common.domain import ContractError, Json, closed, digest
from gf_common.rust_player import validate_player_model, validate_runtime_binding
from gf_common.rust_ppo_source import source_encoder, source_mode, validate_source_mode

_RUN = frozenset(['schema_version', 'algorithm', 'run_id', 'status', 'config_sha256',
    'effective_config', 'config_inputs', 'identity', 'seed', 'device', 'created_unix', 'last_unix',
    'wall_deadline_unix', 'admission_deadline_unix', 'execution_deadline_unix', 'updates',
    'accepted_decisions', 'optimizer_steps', 'generation_sha256', 'collection_seconds', 'update_seconds',
    'invocations', 'completed_updates', 'run_variant', 'model_id', 'model_version', 'input_recipe',
    'execution', 'action_contract', 'purpose', 'synthetic_capacity', 'rule_signature', 'layout_signature',
    'storage_binding', 'runtime_binding_sha256', 'monotonic_authority', 'host_result', 'resume_identity'])
_RUN_OBSERVATIONS = frozenset(['hardware', 'elapsed_seconds', 'artifact_bytes',
    'peak_coordinator_memory_bytes', 'coordinator_memory_scope', 'timing_scope',
    'failure', 'quarantine', 'cleanup_failure'])


def validate_run(value: Json, *, hpo: bool = False) -> Json:
    """Actual cold Run5 input boundary; metadata alone grants no authority."""
    from gf_common.domain import canonical, integer
    extra = {'hpo_spec_sha256', 'used_decisions', 'collected_decisions', 'stop_reason'} if hpo else set()
    raw = closed(value, set(_RUN) | extra, set(_RUN_OBSERVATIONS))
    if (type(raw['schema_version']) is not int or raw['schema_version'] != 5
            or raw['run_variant'] != ('hpo-ppo-owned-rust-player-v1' if hpo else 'ppo-rust-player-v1') or raw['algorithm'] != 'ppo'
            or raw['model_version'] != ('4' if hpo else '3')
            or raw['status'] not in ('running', 'paused', 'complete', 'incomplete', 'failed')
            or raw['host_result'] not in (None, 'awaiting-supervised-finalization')
            or type(raw['invocations']) is not dict):
        raise ContractError('explicit ordinary Rust Run5 required')
    bound = validate_runtime_binding(raw['identity']['player_runtime'])
    if (digest(bound) != raw['runtime_binding_sha256']
            or digest(raw['effective_config']) != raw['config_sha256']
            or canonical(bound['runtime']) != canonical(raw['effective_config']['hpo_spec']['runtime'] if hpo else raw['effective_config']['run']['player_runtime'])):
        raise ContractError('Run5 actual config/runtime binding differs')
    effective = raw['effective_config']
    if raw['model_id'] != effective['model']['model_id']:
        raise ContractError('Run5 model identity differs from its effective config')
    validate_player_model(bound['runtime'], raw['model_id'], hpo=hpo, encoder=source_encoder(effective))
    for name in ('updates', 'accepted_decisions', 'optimizer_steps'):
        integer(raw[name], 0)
    if hpo:
        spec = raw['effective_config']['hpo_spec']
        if (spec['schema_version'] != 6 or digest(spec) != raw['hpo_spec_sha256']
                or spec['encoder'] != source_encoder(raw['effective_config'])):
            raise ContractError('explicit new HPO spec6 required')
        integer(raw['used_decisions'], 0, spec['decision_budget'])
        integer(raw['collected_decisions'], raw['used_decisions'])
    closed(raw['monotonic_authority'], {'boot_id', 'deadlines'})
    for invocation, entry in raw['invocations'].items():
        from gf_common.rust_player import hex_identity
        hex_identity(invocation, 32)
        closed(entry, {'operation_path', 'operation_sha256', 'execution_request',
            'admission_path', 'admission_sha256', 'mode', 'elapsed_seconds', 'host_health'}, {'empty_collection'} if hpo else set())
        hex_identity(entry['operation_sha256'])
        hex_identity(entry['admission_sha256'])
        if entry['mode'] not in ('fresh', 'resume') or entry['execution_request']['mode'] != entry['mode']:
            raise ContractError('Run5 invocation/request differs')
        if 'empty_collection' in entry:
            from types import SimpleNamespace
            storage = SimpleNamespace(**raw['effective_config']['storage'])
            validate_empty_collection(entry['empty_collection'], state=raw, config=SimpleNamespace(storage=storage))
    markers = [entry for entry in raw['invocations'].values() if 'empty_collection' in entry]
    if hpo and ((raw['stop_reason'] == 'empty_collection') != (len(markers) == 1)):
        raise ContractError('empty stop requires exactly one actual uncommitted collection marker')
    return raw

_CLASSIFICATION = frozenset(["action_contract", "rule_signature", "layout_signature", "purpose", "synthetic_capacity"])
_COLLECTION = _CLASSIFICATION | frozenset(["schema_version", "kind", "command", "status", "run_id", "config", "config_hash", "inputs", "code", "action_config_hash", "bots_revision", "simulator_revision", "runtime_evidence", "schedule", "schedule_schema_version", "files", "policies", "journal_schema_version", "derived_schema_version", "participant_protocol_version", "candidate_codec", "model_version", "input_recipe", "token_signature", "source_eligibility", "policy_bindings", "neural_inputs", "collection_role", "linkage", "episodes", "groups", "report_sha256", "runtime_binding_sha256", "player_producer", "actor_protocol_version", "capture_authority", "logical_retirement", "service_usage"])
_PROOF = _CLASSIFICATION | frozenset(["schema_version", "kind", "qualified", "parent_qualification", "run_id", "generation", "batch_id", "manifest_sha256", "source_set_sha256", "files", "games", "accepted_decisions", "closure_counts", "missing_replays", "runtime_binding_sha256", "player_producer", "capture_authority_sha256", "collection_retirement_sha256", "writer_retirement", "storage", "snapshot_identity", "episode_digests", "source_correspondence", "service_usage"])
_GENERATION = _CLASSIFICATION | frozenset(["schema_version", "run_id", "generation", "parent_sha256", "model_id", "model_version", "input_recipe", "inference_sha256", "resume_sha256", "artifact_files", "accepted_decisions", "optimizer_steps", "update_sha256", "parent_controls", "runtime_binding_sha256", "snapshot_identity"])
_HPO_GENERATION = frozenset(["schema_version", "origin", "encoder", "run_id", "generation", "parent_sha256", "snapshot_identity", "inference_sha256", "resume_sha256", "artifact_files", "update_sha256", "parent_controls", "optimizer_steps", "used_decisions", "collected_decisions", "accepted_decisions", "runtime_binding_sha256"])
_UPDATE = _CLASSIFICATION | frozenset(["schema_version", "status", "run_id", "generation", "parent_sha256", "batch_id", "collection", "collection_manifest_sha256", "numeric", "numeric_sha256", "accepted_decisions", "source_proof", "metrics", "planned_games", "actual_games", "health", "policy_bindings", "learner_seats", "collection_seconds", "update_seconds", "model_id", "model_version", "input_recipe", "numeric_files", "runtime_binding_sha256", "collection_retirement_sha256", "service_usage", "parent_snapshot_identity", "result_snapshot_identity"])
_REGISTRY = frozenset(["schema_version", "run_id", "runtime_binding_sha256", "generations", "evaluation_references", "service_references"])
_ORDINARY_FIELDS = {"collection": _COLLECTION, "proof": _PROOF,
    "generation": _GENERATION, "update": _UPDATE, "registry": _REGISTRY}
_HPO_EXTRA = {"collection": frozenset(["origin", "encoder", "snapshot_identity", "replay_reconciled"]),
    "proof": frozenset(["origin", "encoder", "selection", "selection_sha256", "replay_reconciled", "expected_optimizer_step_delta", "context_costs"]),
    "update": frozenset(["origin", "encoder", "hpo_spec_sha256", "collected_decisions", "used_decisions", "overshoot_decisions", "expected_optimizer_step_delta"]),
    "registry": frozenset(("origin", "encoder"))}
_VERSIONS = {"collection": (10, 11), "proof": (4, 7), "generation": (4, 6),
    "update": (6, 8), "registry": (5, 8)}
_NATIVE_SOURCE_VERSIONS = {'collection': (12, 13), 'proof': (5, 8)}
_NATIVE_PROOF_FIELDS = {'ppo_source', 'pack_verification', 'training_episodes'}


def record_fields(record_type: str, *, hpo: bool = False) -> frozenset[str]:
    """Exact new outer keys. This is not a live capability constructor."""
    if type(hpo) is not bool or record_type not in _ORDINARY_FIELDS:
        raise ContractError("unknown Rust-player record family")
    if hpo and record_type == "generation":
        return _HPO_GENERATION
    return _ORDINARY_FIELDS[record_type] | (_HPO_EXTRA[record_type] if hpo else frozenset())


def validate_record_shape(value: Json, record_type: str, *, hpo: bool = False) -> Json:
    """Select exact new kind/version/keys only; callers must validate CONTENT/OWNERS."""
    allowed = set(record_fields(record_type, hpo=hpo))
    if record_type == 'update' and type(value) is dict and 'device_plan' in value:
        allowed.add('device_plan')
    mode = None
    if type(value) is dict:
        if record_type == 'collection' and isinstance(value.get('config'), dict) and 'run' in value['config']:
            mode = source_mode(value['config'])
            if mode is not None:
                allowed.add('training_episodes')
        elif record_type == 'proof' and 'ppo_source' in value:
            mode = validate_source_mode(value['ppo_source'])
            allowed.update(_NATIVE_PROOF_FIELDS)
    raw = closed(value, allowed)
    versions = _NATIVE_SOURCE_VERSIONS if mode is not None else _VERSIONS
    if type(raw['schema_version']) is not int or raw['schema_version'] != versions[record_type][int(hpo)]:
        raise ContractError("unknown Rust-player record version")
    if record_type == "collection" and (raw["kind"] != "card-instance-collection" or raw["command"] != "collect"):
        raise ContractError("unknown Rust-player collection kind")
    if record_type == "proof":
        expected = "rust-player-hpo-source-proof-v1" if hpo else "rust-player-source-proof-v1"
        if raw["kind"] != expected or raw["qualified"] is not False or raw["parent_qualification"] != "not_performed":
            raise ContractError("Source metadata cannot grant parent/update authority")
        if mode is not None:
            from gf_training.rust_ppo_pack import validate_pack_verification
            validate_pack_verification(raw['pack_verification'], run_id=raw['run_id'])
    if mode is not None:
        validate_training_episodes(raw['training_episodes'])
    if (hpo and record_type in ('collection', 'proof')
            and raw['encoder'] != ('canonical-rust' if mode is not None else 'canonical-python')):
        raise ContractError('Rust-player encoder differs from the Source recipe')
    if record_type == 'update' and 'device_plan' in raw:
        from gf_models.rust_player_device import validate_device_plan
        validate_device_plan(raw['device_plan'], generation=raw['generation'])
    return raw


def validate_training_episodes(value: Any) -> list[Json]:
    from gf_common.domain import integer
    from gf_common.rust_player import hex_identity, text

    if type(value) is not list:
        raise ContractError('ordered Rust PPO episode evidence required')
    episodes, games = set(), set()
    for item in value:
        row = closed(item, {'episode_id', 'game_id', 'viewer', 'checkpoint', 'epoch',
            'decisions', 'episode_digest'})
        episode, game = text(row['episode_id']), text(row['game_id'])
        if episode in episodes or game in games:
            raise ContractError('duplicate Rust PPO episode or game')
        episodes.add(episode)
        games.add(game)
        integer(row['viewer'], 1, 2)
        integer(row['epoch'], 0)
        integer(row['decisions'], 0)
        hex_identity(row['checkpoint'])
        hex_identity(row['episode_digest'])
    return value


def resume_identity(*, effective_config: Json, identity: Json, execution: Json,
                    classification: Json, storage_binding: Json, hpo: bool = False) -> str:
    if type(hpo) is not bool or type(identity) is not dict or "player_runtime" not in identity:
        raise ContractError("explicit new-mode identity/runtime binding required")
    validate_runtime_binding(identity["player_runtime"])
    return digest({"recipe": "rust-player-hpo-resume-v1" if hpo else "rust-player-ppo-resume-v1",
        "effective_config": effective_config, "identity": identity, "execution": execution,
        "classification": classification, "storage_binding": storage_binding})


def validate_empty_collection(value: Json, *, state: Json, config: Any) -> Json:
    """Closed marker data; physical descriptor and logical SQLite member differ."""
    from pathlib import Path

    from gf_common.domain import integer
    from gf_common.rust_player import hex_identity
    mode = source_mode(state['effective_config'])
    fields = {'batch_id', 'collection_generation', 'manifest_path', 'manifest_sha256',
        'descriptor_path', 'descriptor_sha256', 'learner_decisions'}
    if mode is not None:
        fields.update(_NATIVE_PROOF_FIELDS | {'snapshot_identity', 'policy_bindings', 'schedule', 'service_usage'})
    row = closed(value, fields)
    generation = integer(row['collection_generation'], 0)
    integer(row['learner_decisions'], 0, 0)
    root = Path(config.storage.db_root) / state['run_id'] / 'collections' / f'batch-{generation:06}'
    if (state['status'] != 'incomplete' or state.get('stop_reason') != 'empty_collection'
            or generation != state['updates'] or row['batch_id'] != f'batch-{generation:06}'
            or row['descriptor_path'] != str(root / 'collection.pending.json')
            or row['manifest_path'] != str(root / 'manifest.json')):
        raise ContractError('empty collection exact current logical/physical binding differs')
    hex_identity(row['manifest_sha256'])
    hex_identity(row['descriptor_sha256'])
    if mode is not None:
        from gf_training.rust_ppo_pack import validate_pack_verification
        if (validate_source_mode(row['ppo_source']) != mode
                or validate_pack_verification(row['pack_verification'], run_id=state['run_id'])['descriptor_sha256'] != row['descriptor_sha256']):
            raise ContractError('empty collection saved descriptor proof differs')
        if any(item['decisions'] for item in validate_training_episodes(row['training_episodes'])):
            raise ContractError('empty collection contains a learner decision')
    return row


def validate_completion(value: Json) -> Json:
    """Closed existing completion1 DATA; parsing grants no finalizer authority."""
    from pathlib import Path

    from gf_common.domain import integer
    from gf_common.rust_player import hex_identity, text
    row = closed(value, {'schema_version', 'kind', 'supervision_id', 'invocation_id', 'operation_id',
        'run_id', 'config_sha256', 'runtime_binding_sha256', 'operation_sha256', 'prepared_sha256',
        'exit_sha256', 'state_sha256', 'generation_sha256', 'source_manifest_sha256', 'state',
        'committed_update', 'audit_path', 'audit_sha256', 'eligible', 'parent_metadata_validation'})
    initial = row['kind'] == 'rust-player-initial-pause-completion-v2'
    if (type(row['schema_version']) is not int or row['schema_version'] != (2 if initial else 1)
            or not initial and row['kind'] != 'rust-player-training-completion-v1'):
        raise ContractError('recognized Rust completion record required')
    for key in ('supervision_id', 'invocation_id'):
        hex_identity(row[key], 32)
    for key in ('operation_id', 'run_id', 'audit_path'):
        text(row[key])
    for key in ('config_sha256', 'runtime_binding_sha256', 'operation_sha256', 'prepared_sha256',
            'exit_sha256', 'state_sha256', 'generation_sha256', 'audit_sha256'):
        hex_identity(row[key])
    integer(row['committed_update'], 0)
    if (row['state'] not in ('complete', 'paused', 'incomplete') or type(row['eligible']) is not bool
            or row['parent_metadata_validation'] is not None or not Path(row['audit_path']).is_absolute()):
        raise ContractError('completion1 result/audit fields differ')
    if initial:
        if (row['state'] != 'paused' or row['committed_update'] != 0
                or row['source_manifest_sha256'] is not None or row['eligible']):
            raise ContractError('initial HPO completion requires a paused G0 with no rollout Source')
    else:
        hex_identity(row['source_manifest_sha256'])
    return row
